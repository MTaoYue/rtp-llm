"""CUDA regressions; run directly with python (no server/bazel artifacts required)."""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "benchmark"))
from qwen35_vision import load_module


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class VisionCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(7)
        cls.module = load_module()
        cls.cuda = importlib.import_module(cls.module.__name__ + ".qwen3_5_vision_cuda")
        cfg = cls.module.Qwen3_5MoeVisionConfig(
            depth=2,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            out_hidden_size=32,
            num_position_embeddings=16,
        )
        cfg._attn_implementation = "sdpa"
        cls.model = cls.module.Qwen3_5MoeVisionModel(cfg).eval().cuda().bfloat16()

    def setUp(self):
        self.model.configure_vision("baseline")

    @torch.inference_mode()
    def test_reconfigure_releases_retained_runtime_graphs(self):
        x, g = self.inputs([(1, 4, 4)])
        reference = self.model(x, g)
        retained = []
        for _ in range(2):
            self.model.configure_vision("compile_cudagraph")
            output = self.model(x, g)
            runtime = self.model._vision_runtime
            self.assertEqual(runtime.stats["graph_captures"], 1)
            self.assertTrue(runtime.graphs)
            retained.append(runtime)
            self.model.configure_vision("baseline")
            self.assertFalse(runtime.graphs)
            self.assertFalse(runtime.metadata)
            self.assertIsNone(runtime.compiled)
            self.same(output, reference)
        self.assertTrue(all(not runtime.graphs for runtime in retained))

    @torch.inference_mode()
    def test_graph_replay_output_oom_falls_back(self):
        x, g = self.inputs([(1, 4, 4)])
        reference = self.model(x, g)
        self.model.configure_vision("cudagraph")
        previous = self.model(x, g)
        runtime = self.model._vision_runtime
        key = next(iter(runtime.graphs))
        static, outputs, graph = runtime.graphs[key]

        class FailedOutput:
            def clone(self):
                raise torch.OutOfMemoryError("injected output allocation failure")

        runtime.graphs[key] = (static, (FailedOutput(),), graph)
        del static, outputs, graph
        self.same(self.model(x, g), reference)
        self.same(previous, reference)
        self.assertFalse(runtime.graphs)
        self.assertIn(key, runtime.failed_graphs)
        self.assertEqual(runtime.stats["graph_failures"], 1)
        captures = runtime.stats["graph_captures"]
        self.same(self.model(x, g), reference)
        self.assertEqual(runtime.stats["graph_captures"], captures)

    @torch.inference_mode()
    def test_compile_large_heterogeneous_segment_list(self):
        x, g = self.inputs([(1, 2, 4), (1, 4, 4)] * 65)
        reference = self.model(x, g)
        self.model.configure_vision("compile")
        actual = self.model(x, g)
        self.same(actual, reference)
        stats = self.model._vision_runtime.stats
        self.assertGreater(stats["compile_graphs"], 0)
        self.assertEqual(stats["compile_failures"], 0)

    def inputs(self, grids):
        return torch.randn(
            sum(t * h * w for t, h, w in grids),
            1536,
            device="cuda",
            dtype=torch.bfloat16,
        ), torch.tensor(grids, device="cuda")

    def same(self, a, b):
        torch.testing.assert_close(
            a.last_hidden_state, b.last_hidden_state, rtol=0, atol=0
        )
        torch.testing.assert_close(a.pooler_output, b.pooler_output, rtol=0, atol=0)

    @torch.inference_mode()
    def test_bias_position_rounding_alias_and_fallback(self):
        for device in ("cpu", "cuda"):
            for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
                for strided in (False, True):
                    x = torch.randn(37, 144, device=device, dtype=dtype)
                    pos = torch.randn_like(x)
                    if strided:
                        x, pos = x[:, ::2], pos[:, ::2]
                    bias = torch.randn(x.shape[-1], device=device, dtype=dtype)
                    before = x.clone()
                    ref = (x + bias) + pos
                    out = self.cuda.fused_bias_pos(x, bias, pos)
                    torch.testing.assert_close(out, ref, rtol=0, atol=0)
                    torch.testing.assert_close(x, before, rtol=0, atol=0)
                    self.assertNotEqual(out.data_ptr(), x.data_ptr())
        # A single final cast gives a different BF16 result; retain the first cast.
        x = torch.tensor([[1.0]], device="cuda", dtype=torch.bfloat16)
        bias = torch.tensor([2**-8], device="cuda", dtype=torch.bfloat16)
        pos = bias.reshape(1, 1)
        out = self.cuda.fused_bias_pos(x, bias, pos)
        self.assertEqual(out.item(), 1.0)
        self.assertNotEqual(
            out.item(), (x.float() + bias.float() + pos.float()).bfloat16().item()
        )

    @torch.inference_mode()
    def test_batch_attention_boundaries_and_mixed_fallback(self):
        for grids in (((1, 4, 8), (1, 8, 4)), ((2, 4, 4),), ((1, 2, 4), (1, 4, 4))):
            x, g = self.inputs(grids)
            self.model.configure_vision("baseline")
            ref = self.model(x, g)
            lengths = [h * w for t, h, w in grids for _ in range(t)]
            for enabled in (False, True):
                self.model.configure_vision("optimized", batch_attention=enabled)
                with patch(
                    "torch.nn.functional.scaled_dot_product_attention",
                    wraps=torch.nn.functional.scaled_dot_product_attention,
                ) as sdpa:
                    out = self.model(x, g)
                grouped = enabled and len(set(lengths)) == 1
                expected_calls = len(self.model.blocks) * (
                    1 if grouped else len(lengths)
                )
                self.assertEqual(sdpa.call_count, expected_calls)
                self.assertTrue(
                    all(
                        c.args[0].shape[0] == (len(lengths) if grouped else 1)
                        for c in sdpa.call_args_list
                    )
                )
                self.same(out, ref)

    @torch.inference_mode()
    def test_batched_compile_graph_content_and_output_lifetime(self):
        x, g = self.inputs(((1, 4, 8), (1, 8, 4)))
        other = x + torch.tensor(0.25, device=x.device, dtype=x.dtype)
        self.model.configure_vision("baseline")
        refs = [self.model(z, g) for z in (x, other)]
        self.model.configure_vision("compile_cudagraph")
        first = self.model(x, g)
        second = self.model(other, g)
        self.same(first, refs[0])
        self.same(second, refs[1])
        stats = self.model.vision_stats()
        self.assertEqual(stats["compile_failures"], 0)
        self.assertEqual(stats["graph_failures"], 0)
        self.assertEqual(stats["graph_captures"], 1)
        self.assertGreaterEqual(stats["graph_hits"], 1)
        self.assertGreaterEqual(stats["compile_graphs"], 1)

    @torch.inference_mode()
    def test_rope_strided_qkv_and_input_unchanged(self):
        qkv = torch.randn(37, 3, 4, 16, device="cuda", dtype=torch.bfloat16)
        q, k, v = qkv.unbind(1)
        before = qkv.clone()
        angles = torch.randn(37, 16, device="cuda")
        c, s = angles.cos(), angles.sin()
        ref = self.module.apply_rotary_pos_emb_vision(q, k, c, s)
        out = self.cuda.fused_rope(q, k, c, s)
        for a, b in zip(out, ref):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(qkv, before, rtol=0, atol=0)

    @torch.inference_mode()
    def test_interpolation_rounding_and_order(self):
        for grids in (((1, 2, 2),), ((1, 14, 28),), ((2, 28, 14), (1, 4, 8))):
            ref = self.model.fast_pos_embed_interpolate(torch.tensor(grids))
            out = self.cuda.interpolated_positions(self.model, grids)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)

    @torch.inference_mode()
    def test_same_token_different_grid_content_and_output_lifetime(self):
        self.model.configure_vision("cudagraph")
        previous = []
        for grids in (
            ((1, 4, 8),),
            ((1, 8, 4),),
            ((2, 4, 4),),
            ((1, 4, 4), (1, 4, 4)),
            ((1, 4, 8),),
        ):
            x, g = self.inputs(grids)
            ref = self.model._forward_reference(x, g)
            out = self.model(x, g)
            self.same(out, ref)
            previous.append(
                (out, out.last_hidden_state.clone(), out.pooler_output.clone())
            )
        for out, h, p in previous:
            torch.testing.assert_close(out.last_hidden_state, h, rtol=0, atol=0)
            torch.testing.assert_close(out.pooler_output, p, rtol=0, atol=0)
        stats = self.model.vision_stats()
        self.assertEqual(stats["graph_captures"], 4)
        self.assertEqual(stats["graph_hits"], 1)

    @torch.inference_mode()
    def test_graph_capacity_and_metadata_reuse(self):
        self.model.configure_vision("cudagraph", graph_capacity=1)
        for grids in (((1, 4, 4),), ((1, 4, 8),), ((1, 4, 8),), ((1, 4, 4),)):
            x, g = self.inputs(grids)
            self.same(self.model(x, g), self.model._forward_reference(x, g))
        stats = self.model.vision_stats()
        self.assertEqual(stats["graph_captures"], 1)
        self.assertEqual(stats["graph_capacity_fallbacks"], 2)
        self.assertEqual(stats["metadata_hits"], 2)

    @torch.inference_mode()
    def test_weight_change_invalidates_positions_and_graph(self):
        self.model.configure_vision("cudagraph")
        x, g = self.inputs(((1, 4, 4),))
        self.model(x, g)
        before = self.model.pos_embed.weight.clone()
        try:
            self.model.pos_embed.weight.add_(0.125)
            self.same(self.model(x, g), self.model._forward_reference(x, g))
            self.assertEqual(self.model.vision_stats()["graph_captures"], 2)
        finally:
            self.model.pos_embed.weight.copy_(before)

    @torch.inference_mode()
    def test_cross_stream_replay(self):
        self.model.configure_vision("cudagraph")
        x, g = self.inputs(((1, 4, 4),))
        first = self.model(x, g)
        ref1 = self.model._forward_reference(x, g)
        other = torch.cuda.Stream()
        other.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(other):
            second = self.model(x * 2, g)
            ref2 = self.model._forward_reference(x * 2, g)
        torch.cuda.current_stream().wait_stream(other)
        self.same(first, ref1)
        self.same(second, ref2)

    @torch.inference_mode()
    def test_compile_and_capture_failures_fallback_once(self):
        x, g = self.inputs(((1, 4, 4),))
        ref = self.model._forward_reference(x, g)
        self.model.configure_vision("compile")
        with patch.object(
            torch, "compile", side_effect=RuntimeError("injected compile failure")
        ):
            self.same(self.model(x, g), ref)
            self.same(self.model(x, g), ref)
        self.assertEqual(self.model.vision_stats()["compile_failures"], 1)
        self.model.configure_vision("cudagraph")
        with patch.object(
            torch.cuda,
            "CUDAGraph",
            side_effect=RuntimeError("injected capture failure"),
        ):
            self.same(self.model(x, g), ref)
            self.same(self.model(x, g), ref)
        self.assertEqual(self.model.vision_stats()["graph_failures"], 1)

    @torch.inference_mode()
    def test_ablations_and_cpu_grid(self):
        x, g = self.inputs(((1, 4, 8),))
        ref = self.model._forward_reference(x, g)
        for options in (
            dict(fused_rope=False),
            dict(fused_pos=False),
            dict(cache_metadata=False),
        ):
            self.model.configure_vision("optimized", **options)
            self.same(self.model(x, g.cpu()), ref)
            self.same(self.model(x, g.cpu()), ref)

    def test_mode_and_grid_validation(self):
        with self.assertRaises(ValueError):
            self.model.configure_vision("invalid")
        with torch.inference_mode():
            self.model.configure_vision("optimized")
            x, g = self.inputs(((1, 4, 4),))
            with self.assertRaises(ValueError):
                self.model(x[:-1], g)
            with self.assertRaises(ValueError):
                self.model(x, torch.tensor([[1, 3, 4]]))

    @torch.inference_mode()
    def test_compiled_modes_are_active_and_exact(self):
        x, g = self.inputs(((1, 4, 8),))
        ref = self.model._forward_reference(x, g)
        for mode in ("compile", "compile_cudagraph"):
            self.model.configure_vision(mode)
            self.same(self.model(x, g), ref)
            self.same(self.model(x, g), ref)
            stats = self.model.vision_stats()
            self.assertGreater(stats["compile_graphs"], 0)
            self.assertEqual(stats["compile_failures"], 0)
            if mode == "compile_cudagraph":
                self.assertEqual(stats["graph_captures"], 1)
                self.assertGreater(stats["graph_hits"], 0)

    @torch.inference_mode()
    def test_reload_and_device_conversion_release_runtime(self):
        self.model.configure_vision("optimized")
        x, g = self.inputs(((1, 4, 4),))
        self.model(x, g)
        self.assertIsNotNone(self.model._vision_runtime)
        self.model.load_state_dict(self.model.state_dict())
        self.assertIsNone(self.model._vision_runtime)
        self.model(x, g)
        self.model.to(dtype=torch.bfloat16)
        self.assertIsNone(self.model._vision_runtime)

    @torch.inference_mode()
    def test_inference_parameters_and_cpu_fallback(self):
        cfg = self.module.Qwen3_5MoeVisionConfig(
            depth=1,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            out_hidden_size=32,
            num_position_embeddings=16,
        )
        cfg._attn_implementation = "sdpa"
        small = self.module.Qwen3_5MoeVisionModel(cfg).eval()
        small.configure_vision("optimized")
        x, g = self.inputs(((1, 4, 4),))
        cpu = small(x.float().cpu(), g.cpu())
        self.assertIsNone(small._vision_runtime)
        self.assertTrue(torch.isfinite(cpu.pooler_output).all())
        small.cuda().bfloat16()
        self.same(small(x, g), small._forward_reference(x, g))
        small.clear_vision_cache()
        self.assertIsNone(small._vision_runtime)

    @torch.inference_mode()
    def test_existing_internal_patch_owns_execution(self):
        x, g = self.inputs(((1, 4, 4),))
        original = self.module.apply_rotary_pos_emb_vision

        def external_rope(*args, **kwargs):
            return original(*args, **kwargs)

        self.model.configure_vision("optimized")
        with patch.object(self.module, "apply_rotary_pos_emb_vision", external_rope):
            self.same(self.model(x, g), self.model._forward_reference(x, g))
        self.assertIsNone(self.model._vision_runtime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
