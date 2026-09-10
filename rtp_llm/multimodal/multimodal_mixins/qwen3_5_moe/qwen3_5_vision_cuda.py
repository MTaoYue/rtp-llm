"""Optional CUDA inference acceleration for Qwen3.5 vision.

Metadata preparation follows vLLM qwen3_vl.py (a97dacb710); interpolation
keeps RTP's BF16 rounding at every product/addition. No vLLM dependency.
"""

import logging
import threading
import time
import weakref
from collections import OrderedDict

import torch
import triton
import triton.language as tl

LOG = logging.getLogger(__name__)
MODES = {"baseline", "optimized", "compile", "cudagraph", "compile_cudagraph"}


@triton.jit
def _rope_kernel(
    Q,
    K,
    C,
    S,
    OQ,
    OK,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    CS: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    # Promote before multiplication: packed QKV strides may overflow even when
    # the contiguous Q/K outputs still contain fewer than 2**31 elements.
    if N * H * D >= 2**31 or N * QS >= 2**31 or N * KS >= 2**31 or N * CS >= 2**31:
        pid = pid.to(tl.int64)
    i = pid * B + tl.arange(0, B)
    mask = i < N * H * D
    token = i // (H * D)
    head = (i // D) % H
    dim = i % D
    other = (dim + D // 2) % D
    sign = tl.where(dim < D // 2, -1.0, 1.0)
    c = tl.load(C + token * CS + dim, mask, 0).to(tl.float32)
    s = tl.load(S + token * CS + dim, mask, 0).to(tl.float32)
    q = tl.load(Q + token * QS + head * QH + dim, mask, 0).to(tl.float32)
    qr = tl.load(Q + token * QS + head * QH + other, mask, 0).to(tl.float32) * sign
    k = tl.load(K + token * KS + head * KH + dim, mask, 0).to(tl.float32)
    kr = tl.load(K + token * KS + head * KH + other, mask, 0).to(tl.float32) * sign
    tl.store(OQ + i, q * c + qr * s, mask)
    tl.store(OK + i, k * c + kr * s, mask)


def fused_rope(q, k, cos, sin):
    # Separate outputs also support q/k views into the same packed QKV tensor.
    assert q.shape == k.shape and q.ndim == 3 and q.shape[-1] % 2 == 0
    assert (
        q.stride(-1) == k.stride(-1) == 1
        and cos.is_contiguous()
        and sin.is_contiguous()
    )
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    n, h, d = q.shape
    _rope_kernel[(triton.cdiv(n * h * d, 256),)](
        q,
        k,
        cos,
        sin,
        oq,
        ok,
        n,
        h,
        d,
        q.stride(0),
        k.stride(0),
        q.stride(1),
        k.stride(1),
        cos.stride(0),
        256,
        enable_fp_fusion=False,
    )
    return oq, ok


@triton.jit
def _pos_kernel(P, I, W, O, N: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    pid = tl.program_id(0)
    if N * D >= 2**31:
        pid = pid.to(tl.int64)
    x = pid * B + tl.arange(0, B)
    mask = x < N * D
    token = x // D
    dim = x % D
    dtype = O.dtype.element_ty
    acc = tl.full((B,), 0, tl.float32)
    for j in tl.static_range(4):
        idx = tl.load(I + j * N + token, mask, 0)
        w = tl.load(W + j * N + token, mask, 0).to(tl.float32)
        p = tl.load(P + idx * D + dim, mask, 0).to(tl.float32)
        product = (p * w).to(dtype).to(tl.float32)
        acc = (acc + product).to(dtype).to(tl.float32)
    tl.store(O + x, acc, mask)


def interpolated_positions(model, grids):
    indices, weights = [], []
    size = model.num_grid_per_side
    merge = model.spatial_merge_size
    for t, h, w in grids:
        # CPU linspace matches the reference's coordinate rounding exactly.
        hs = torch.linspace(0, size - 1, h)
        ws = torch.linspace(0, size - 1, w)
        loh, low = hs.int(), ws.int()
        hih, hiw = (loh + 1).clamp(max=size - 1), (low + 1).clamp(max=size - 1)
        dh, dw = hs - loh, ws - low
        ix = torch.stack(
            [
                (loh[:, None] * size + low).flatten(),
                (loh[:, None] * size + hiw).flatten(),
                (hih[:, None] * size + low).flatten(),
                (hih[:, None] * size + hiw).flatten(),
            ]
        )
        wt = torch.stack(
            [
                ((1 - dh[:, None]) * (1 - dw)).flatten(),
                ((1 - dh[:, None]) * dw).flatten(),
                (dh[:, None] * (1 - dw)).flatten(),
                (dh[:, None] * dw).flatten(),
            ]
        )
        order = (
            torch.arange(h * w)
            .reshape(h // merge, merge, w // merge, merge)
            .permute(0, 2, 1, 3)
            .reshape(-1)
            .repeat(t)
        )
        indices.append(ix[:, order])
        weights.append(wt[:, order])
    p = model.pos_embed.weight
    ix = torch.cat(indices, dim=1).to(device=p.device, dtype=torch.int64)
    wt = torch.cat(weights, dim=1).to(device=p.device, dtype=p.dtype)
    n = ix.shape[1]
    out = torch.empty((n, p.shape[1]), device=p.device, dtype=p.dtype)
    _pos_kernel[(triton.cdiv(out.numel(), 256),)](
        p, ix, wt, out, n, p.shape[1], 256, enable_fp_fusion=False
    )
    return out


# Port the useful Inductor bias/position fusion to inference eager execution.
# Keep convolution output and bias-add rounding identical to native Conv3d.
@triton.jit
def _bias_pos_kernel(
    X, Bias, Pos, Y, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if N >= 2**31:
        pid = pid.to(tl.int64)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    x = tl.load(X + i, mask, 0).to(tl.float32)
    b = tl.load(Bias + i % C, mask, 0).to(tl.float32)
    p = tl.load(Pos + i, mask, 0).to(tl.float32)
    # Conv3d bias is rounded to the activation dtype before adding position.
    biased = (x + b).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + i, biased + p, mask)


def fused_bias_pos(x, bias, pos):
    if not (
        x.is_cuda
        and x.ndim == 2
        and pos.shape == x.shape
        and bias.ndim == 1
        and bias.numel() == x.shape[-1]
        and x.device == pos.device == bias.device
        and x.dtype == pos.dtype == bias.dtype
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.is_contiguous()
        and pos.is_contiguous()
        and bias.is_contiguous()
    ):
        return (x + bias) + pos
    out = torch.empty_like(x)
    _bias_pos_kernel[(triton.cdiv(x.numel(), 1024),)](
        x, bias, pos, out, x.numel(), x.shape[-1], 1024, enable_fp_fusion=False
    )
    return out


def fused_patch_embed(model, x, pos):
    patch = model.patch_embed
    proj = patch.proj
    if not (
        x.is_cuda
        and torch.backends.cudnn.enabled
        and proj.weight.dtype in (torch.bfloat16, torch.float16)
        and proj.bias is not None
        and proj.padding_mode == "zeros"
    ):
        return patch(x) + pos
    shaped = x.view(
        -1,
        patch.in_channels,
        patch.temporal_patch_size,
        patch.patch_size,
        patch.patch_size,
    ).to(proj.weight.dtype)
    conv = torch.nn.functional.conv3d(
        shaped, proj.weight, None, proj.stride, proj.padding, proj.dilation, proj.groups
    ).view(-1, patch.embed_dim)
    return fused_bias_pos(conv, proj.bias, pos)


@torch.library.custom_op("rtp_qwen35_vision::native_segmented_sdpa", mutates_args=())
def native_segmented_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: list[int],
    scale: float,
) -> torch.Tensor:
    # Keep the existing SDPA backend and per-segment contiguous output layout.
    # Large heterogeneous batches otherwise unroll tens of thousands of nodes.
    splits = [torch.split(tensor, lengths, dim=2) for tensor in (q, k, v)]
    outputs = [
        torch.nn.functional.scaled_dot_product_attention(
            qi,
            ki,
            vi,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        .transpose(1, 2)
        .contiguous()
        for qi, ki, vi in zip(*splits)
    ]
    return torch.cat(outputs, dim=1)


@native_segmented_sdpa.register_fake
def _native_segmented_sdpa_fake(q, k, v, lengths, scale):
    return q.new_empty((q.shape[0], q.shape[2], q.shape[1], q.shape[3]))


# Keep the native reduction and activation rounding at compile boundaries.
# Real-weight diagnostics found tiny LayerNorm drift and FC1/GELU fusion drift
# accumulate across 27 blocks beyond the encoder's numerical contract.
@torch.library.custom_op("rtp_qwen35_vision::native_norm", mutates_args=())
def native_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    # The native CUDA kernel in some PyTorch builds wraps flattened row
    # offsets at 2**32 elements. Keep each independent row reduction intact.
    if x.is_cuda and x.numel() >= 2**31:
        width = x.shape[-1]
        flat = x.reshape(-1, width)
        rows = max(1, (2**31 - 1) // width)
        return torch.cat(
            [
                torch.nn.functional.layer_norm(part, (width,), weight, bias, eps)
                for part in flat.split(rows)
            ],
            dim=0,
        ).reshape(x.shape)
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


@native_norm.register_fake
def _native_norm_fake(x, weight, bias, eps):
    return torch.empty_like(x)


@torch.library.custom_op("rtp_qwen35_vision::native_gelu", mutates_args=())
def native_gelu(x: torch.Tensor, approximate: str) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate=approximate)


@native_gelu.register_fake
def _native_gelu_fake(x, approximate):
    return torch.empty_like(x)


class VisionRuntime:
    def __init__(
        self,
        model,
        mode,
        fused_rope=True,
        fused_pos=True,
        cache_metadata=True,
        graph_capacity=8,
        fused_patch=True,
        batch_attention=True,
    ):
        if mode not in MODES:
            raise ValueError(f"Unknown vision mode: {mode}")
        if graph_capacity < 0:
            raise ValueError("graph_capacity must be non-negative")
        self.model_ref = weakref.ref(model)
        self.mode = mode
        self.fused_rope = fused_rope
        self.fused_pos = fused_pos
        self.fused_patch = fused_patch
        self.batch_attention = batch_attention
        self.cache_metadata = cache_metadata
        self.graph_capacity = graph_capacity
        self.metadata = OrderedDict()
        self.graphs = {}
        self.failed_graphs = set()
        self.compiled = None
        self.compile_failed = False
        self.lock = threading.RLock()
        self.signature = None
        self.stats = dict(
            metadata_hits=0,
            metadata_misses=0,
            graph_hits=0,
            graph_captures=0,
            graph_capacity_fallbacks=0,
            compile_graphs=0,
            compile_failures=0,
            graph_failures=0,
        )

    def _prepare(self, grids, device):
        model = self.model_ref()
        grid = torch.tensor(grids, dtype=torch.int64)
        pos = (
            interpolated_positions(model, grids)
            if self.fused_pos
            else model.fast_pos_embed_interpolate(grid)
        )
        rope = model.rot_pos_emb(grid)
        emb = torch.cat((rope, rope), dim=-1)
        lengths = tuple(h * w for t, h, w in grids for _ in range(t))
        cu = torch.tensor((0,) + lengths, device=device, dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        return pos, emb.cos(), emb.sin(), cu, max(lengths), lengths

    def _core(self, x, pos, cos, sin, cu, maxlen, lengths):
        model = self.model_ref()
        hidden = (
            fused_patch_embed(model, x, pos)
            if self.fused_patch and not torch.compiler.is_compiling()
            else model.patch_embed(x) + pos
        )
        # Metadata is shared by all blocks. Equal-length segments remain independent.
        batch_size = (
            len(lengths)
            if self.batch_attention and len(lengths) > 1 and len(set(lengths)) == 1
            else 1
        )
        for block in model.blocks:
            if (
                torch.compiler.is_compiling() or hidden.numel() >= 2**31
            ) and model.config.hidden_act == "gelu_pytorch_tanh":
                norm = native_norm(
                    hidden, block.norm1.weight, block.norm1.bias, block.norm1.eps
                )
                hidden = hidden + block.attn(
                    norm,
                    cu_seqlens=cu,
                    position_embeddings=(cos, sin),
                    vision_max_seqlen=maxlen,
                    vision_lengths=lengths,
                    vision_fused_rope=self.fused_rope,
                    vision_batch_size=batch_size,
                )
                norm = native_norm(
                    hidden, block.norm2.weight, block.norm2.bias, block.norm2.eps
                )
                act = native_gelu(block.mlp.linear_fc1(norm), "tanh")
                hidden = hidden + block.mlp.linear_fc2(act)
            else:
                hidden = block(
                    hidden,
                    cu_seqlens=cu,
                    position_embeddings=(cos, sin),
                    vision_max_seqlen=maxlen,
                    vision_lengths=lengths,
                    vision_fused_rope=self.fused_rope,
                    vision_batch_size=batch_size,
                )
        if torch.compiler.is_compiling() or hidden.numel() >= 2**31:
            merger = model.merger
            merged = (
                hidden.view(-1, merger.hidden_size)
                if merger.use_postshuffle_norm
                else hidden
            )
            merged = native_norm(
                merged, merger.norm.weight, merger.norm.bias, merger.norm.eps
            ).view(-1, merger.hidden_size)
            pooled = merger.linear_fc2(native_gelu(merger.linear_fc1(merged), "none"))
        else:
            pooled = model.merger(hidden)
        return hidden, pooled

    def _compile_backend(self, gm, example_inputs):
        from torch._dynamo.backends.registry import lookup_backend

        self.stats["compile_graphs"] += 1
        from torch._inductor import config

        start = time.perf_counter()
        with config.patch(
            {"triton.cudagraphs": False, "emulate_precision_casts": True}
        ):
            result = lookup_backend("inductor")(gm, example_inputs)
        self.stats["compile_seconds"] = (
            self.stats.get("compile_seconds", 0.0) + time.perf_counter() - start
        )
        return result

    def _execute(self, x, meta):
        if "compile" not in self.mode or self.compile_failed:
            return self._core(x, *meta)
        try:
            if self.compiled is None:
                # No implicit graph capture: the explicit graph mode owns buffers.
                from torch._inductor import config

                with config.patch({"triton.cudagraphs": False}):
                    self.compiled = torch.compile(
                        self._core,
                        backend=self._compile_backend,
                        fullgraph=True,
                        dynamic=False,
                    )
            return self.compiled(x, *meta)
        except Exception as exc:
            self.compile_failed = True
            self.stats["compile_failures"] += 1
            LOG.warning("Qwen3.5 vision compile failed; using optimized eager: %s", exc)
            return self._core(x, *meta)

    def clear(self):
        """Release owned buffers even when Dynamo retains this runtime object."""
        with self.lock:
            completion = getattr(self, "completion", None)
            if completion is not None:
                completion.synchronize()
            self.graphs.clear()
            self.metadata.clear()
            self.failed_graphs.clear()
            self.compiled = None
            self.completion = None
            self.signature = None

    def __call__(self, x, grid_thw):
        with self.lock:
            current = torch.cuda.current_stream(x.device)
            previous = getattr(self, "completion", None)
            if previous is not None:
                current.wait_event(previous)
            try:
                return self._call_locked(x, grid_thw)
            finally:
                self.completion = torch.cuda.Event()
                self.completion.record(current)

    def _call_locked(self, x, grid_thw):
        with self.lock:
            model = self.model_ref()
            # Include parameter mutation versions to invalidate cached positions and
            # captured parameter pointers after loading/changing model weights.
            signature = tuple(
                (
                    p.data_ptr(),
                    None if torch.is_inference(p) else p._version,
                    p.device,
                    p.dtype,
                )
                for p in model.parameters()
            )
            if signature != self.signature:
                self.metadata.clear()
                self.graphs.clear()
                self.failed_graphs.clear()
                self.compiled = None
                self.signature = signature
            grids = tuple(
                tuple(int(v) for v in row) for row in grid_thw.detach().cpu().tolist()
            )
            merge = model.spatial_merge_size
            if not grids or any(
                t <= 0 or h <= 0 or w <= 0 or h % merge or w % merge
                for t, h, w in grids
            ):
                raise ValueError("grid_thw must contain positive, merge-aligned grids")
            if x.shape[0] != sum(t * h * w for t, h, w in grids):
                raise ValueError("pixel token count does not match grid_thw")
            key = (
                grids,
                x.device,
                x.dtype,
                tuple(x.shape),
                tuple(x.stride()),
                model.config._attn_implementation,
            )
            if self.cache_metadata and key in self.metadata:
                self.stats["metadata_hits"] += 1
                meta = self.metadata[key]
                self.metadata.move_to_end(key)
            else:
                self.stats["metadata_misses"] += 1
                meta = self._prepare(grids, x.device)
                if self.cache_metadata:
                    self.metadata[key] = meta
                    if len(self.metadata) > 32:
                        self.metadata.popitem(last=False)
            if "cudagraph" not in self.mode:
                return self._execute(x, meta)
            if key in self.graphs:
                static, out, graph = self.graphs[key]
                try:
                    static.copy_(x)
                    graph.replay()
                    result = tuple(t.clone() for t in out)
                    # Lock lifetime covers submission; synchronize cross-stream access.
                    self.last_stream = torch.cuda.current_stream(x.device)
                    self.stats["graph_hits"] += 1
                    return result
                except torch.OutOfMemoryError as exc:
                    self.failed_graphs.add(key)
                    self.stats["graph_failures"] += 1
                    LOG.warning(
                        "Qwen3.5 vision graph replay output allocation failed; using uncaptured core: %s",
                        exc,
                    )
                    self.graphs.pop(key, None)
                    static = out = graph = None
                # Leave the exception scope before releasing the graph pool.
                torch.cuda.empty_cache()
                return self._execute(x, meta)
            if key in self.failed_graphs:
                return self._execute(x, meta)
            if len(self.graphs) >= self.graph_capacity:
                self.stats["graph_capacity_fallbacks"] += 1
                return self._execute(x, meta)
            try:
                static = x.clone()
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._execute(static, meta)
                torch.cuda.current_stream(x.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                # Compile completes in warmup; capture uses the selected tensor core.
                core = (
                    self.compiled
                    if self.compiled is not None and not self.compile_failed
                    else self._core
                )
                with torch.cuda.graph(graph, stream=stream):
                    out = core(static, *meta)
                self.graphs[key] = (static, out, graph)
                self.stats["graph_captures"] += 1
                graph.replay()
                return tuple(t.clone() for t in out)
            except Exception as exc:
                self.failed_graphs.add(key)
                self.stats["graph_failures"] += 1
                LOG.warning(
                    "Qwen3.5 vision graph capture failed; using uncaptured core: %s",
                    exc,
                )
                self.graphs.pop(key, None)
                static = out = graph = None
            torch.cuda.empty_cache()
            return self._execute(x, meta)
