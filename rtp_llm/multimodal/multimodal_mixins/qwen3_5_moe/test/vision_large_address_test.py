"""Opt-in CUDA tests for addresses above the signed 32-bit element limit."""

import importlib
import os
import sys
import unittest
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "benchmark"))
from qwen35_vision import load_module


@unittest.skipUnless(
    torch.cuda.is_available() and os.getenv("RUN_VISION_LARGE_TESTS") == "1",
    "Set RUN_VISION_LARGE_TESTS=1 on a GPU with at least 40 GiB free",
)
class VisionLargeAddressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.mem_get_info()[0] < 40 * 2**30:
            raise unittest.SkipTest("40 GiB free GPU memory required")
        module = load_module()
        cls.cuda = importlib.import_module(module.__name__ + ".qwen3_5_vision_cuda")

    def tearDown(self):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def assert_constant(self, tensor, value):
        low, high = torch.aminmax(tensor)
        self.assertEqual(low.item(), value)
        self.assertEqual(high.item(), value)

    def test_bias_position_output_above_int32(self):
        rows, channels = 2**31 // 1152 + 1, 1152
        x = torch.ones((rows, channels), device="cuda", dtype=torch.bfloat16)
        pos = torch.full_like(x, -0.25)
        bias = torch.full((channels,), 0.5, device="cuda", dtype=x.dtype)
        out = self.cuda.fused_bias_pos(x, bias, pos)
        self.assertGreater(out.numel(), 2**31)
        self.assert_constant(out, 1.25)
        self.assert_constant(x, 1.0)
        self.assert_constant(pos, -0.25)

    def test_interpolation_output_above_int32(self):
        rows, channels = 2**31 // 1152 + 1, 1152
        table = torch.ones((1, channels), device="cuda", dtype=torch.bfloat16)
        indices = torch.zeros((4, rows), device="cuda", dtype=torch.int32)
        weights = torch.zeros((4, rows), device="cuda", dtype=torch.float32)
        weights[0].fill_(1)
        output = torch.empty((rows, channels), device="cuda", dtype=table.dtype)
        self.cuda._pos_kernel[(triton.cdiv(output.numel(), 1024),)](
            table,
            indices,
            weights,
            output,
            rows,
            channels,
            1024,
            enable_fp_fusion=False,
        )
        self.assert_constant(output, 1.0)

    def test_native_norm_above_uint32(self):
        width = 1152
        rows = 2**32 // width + 1024
        x = torch.empty((rows, width), device="cuda", dtype=torch.bfloat16).normal_()
        weight = torch.randn(width, device="cuda", dtype=x.dtype)
        bias = torch.randn_like(weight)
        # Sampling on both sides also detects wrapped writes to early rows.
        indices = [0, 1, rows // 2, 2**32 // width - 1, rows - 1]
        expected = torch.nn.functional.layer_norm(
            x[indices], (width,), weight, bias, 1e-6
        )
        actual = self.cuda.native_norm(x, weight, bias, 1e-6)
        self.assertTrue(torch.equal(actual[indices], expected))

    def test_rope_packed_stride_above_int32(self):
        stride = 2**30 + 16
        storage = torch.empty(2 * stride + 16, device="cuda", dtype=torch.bfloat16)
        q = storage.as_strided((3, 1, 8), (stride, 8, 1), 0)
        k = storage.as_strided((3, 1, 8), (stride, 8, 1), 8)
        q.copy_(torch.arange(24, device="cuda").reshape(3, 1, 8))
        k.copy_(-torch.arange(24, device="cuda").reshape(3, 1, 8))
        reference_q, reference_k = q.clone(), k.clone()
        cos = torch.ones((3, 8), device="cuda", dtype=q.dtype)
        sin = torch.zeros_like(cos)
        actual_q, actual_k = self.cuda.fused_rope(q, k, cos, sin)
        self.assertTrue(torch.equal(actual_q, reference_q))
        self.assertTrue(torch.equal(actual_k, reference_k))
        self.assertTrue(torch.equal(q, reference_q))
        self.assertTrue(torch.equal(k, reference_k))


if __name__ == "__main__":
    unittest.main(verbosity=2)
