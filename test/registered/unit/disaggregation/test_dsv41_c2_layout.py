"""Reject incompatible peer strides before a C2 state transfer is issued."""

import unittest

from sglang.srt.disaggregation.utils import (
    get_dsv4_request_state_indices,
    validate_dsv41_c2_state_layout,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestC2Layout(unittest.TestCase):
    def test_matching_layers(self):
        validate_dsv41_c2_state_layout([32768, 32768, 32768], [32768, 32768, 32768])

    def test_different_ring_strides(self):
        with self.assertRaisesRegex(ValueError, "matching C2 state layouts"):
            validate_dsv41_c2_state_layout([8192] * 3, [32768] * 3)

    def test_c2_indices_with_absent_c128_pool(self):
        # Index selection needs only the pool registry, not GPU buffers.
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.kv_pools = {
            4: None,
            128: None,
            2: DeepSeekV4SingleKVPool.__new__(DeepSeekV4SingleKVPool),
        }
        for length in (127, 129, 1047999):
            self.assertEqual(get_dsv4_request_state_indices(pool, 7, length).tolist(), [7])
        for length in (126, 128, 1048000):
            self.assertEqual(get_dsv4_request_state_indices(pool, 7, length).tolist(), [])


if __name__ == "__main__":
    unittest.main()
