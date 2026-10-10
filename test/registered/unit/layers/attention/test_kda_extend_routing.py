"""FlashInfer KDA prefill routing: a packed batch of one-token sequences must not
reach FlashInfer's grouped recurrent kernel (sgl-project/sglang#43615)."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.linear.kda_backend import KDAKernelDispatcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _dispatcher(flashinfer=True):
    d = object.__new__(KDAKernelDispatcher)
    d.prefill_backend = SimpleNamespace(is_flashinfer=lambda: flashinfer)
    d.extend_kernel = SimpleNamespace(name="flashinfer", supports_safe_gate=True)
    d.triton_kernel = SimpleNamespace(name="triton", supports_safe_gate=True)
    return d


class TestKDAExtendRouting(CustomTestCase):
    def test_all_single_token_packed_batch_takes_triton(self):
        d = _dispatcher()
        # two sequences, one token each: total tokens == sequences
        self.assertEqual(d.effective_extend_kernel(None, 2, 2).name, "triton")
        self.assertEqual(d.effective_extend_kernel(None, 1, 1).name, "triton")

    def test_multi_token_batch_keeps_flashinfer(self):
        d = _dispatcher()
        self.assertEqual(d.effective_extend_kernel(None, 3, 2).name, "flashinfer")
        self.assertEqual(d.effective_extend_kernel(None, 16384, 1).name, "flashinfer")

    def test_non_flashinfer_backend_is_untouched(self):
        d = _dispatcher(flashinfer=False)
        self.assertEqual(d.effective_extend_kernel(None, 2, 2).name, "flashinfer")


if __name__ == "__main__":
    unittest.main()
