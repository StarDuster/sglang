import unittest

import torch

from sglang.srt.layers.attention.dsa.transform_index import (
    transform_index_page_table_prefill,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSATransformIndex(CustomTestCase):
    def test_prefill_transform_pads_output_from_raw_or_padded_topk(self):
        page_table = torch.tensor(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
            ],
            dtype=torch.int64,
        )
        real_topk_indices = torch.tensor(
            [
                [0, 2, -1],
                [1, 3, 0],
                [2, -1, 1],
                [3, 2, 1],
            ],
            dtype=torch.int64,
        )
        padded_topk_indices = torch.cat(
            [
                real_topk_indices,
                torch.tensor(
                    [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
                    dtype=torch.int64,
                ),
            ]
        )
        expected = torch.tensor(
            [
                [10, 12, -1],
                [11, 13, 10],
                [22, -1, 21],
                [23, 22, 21],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
            ],
            dtype=torch.int32,
        )

        for topk_indices in (real_topk_indices, padded_topk_indices):
            with self.subTest(topk_rows=topk_indices.shape[0]):
                result = transform_index_page_table_prefill(
                    page_table=page_table,
                    topk_indices=topk_indices,
                    extend_lens_cpu=[2, 2],
                    output_num_tokens=8,
                )
                torch.testing.assert_close(result, expected)

    def test_prefill_transform_rejects_insufficient_rows(self):
        page_table = torch.arange(4).reshape(1, 4)
        cases = (
            (torch.zeros((2, 1), dtype=torch.int64), 1),
            (torch.zeros((1, 1), dtype=torch.int64), 2),
        )

        for topk_indices, output_num_tokens in cases:
            with self.subTest(
                topk_rows=topk_indices.shape[0],
                output_num_tokens=output_num_tokens,
            ):
                with self.assertRaises(AssertionError):
                    transform_index_page_table_prefill(
                        page_table=page_table,
                        topk_indices=topk_indices,
                        extend_lens_cpu=[2],
                        output_num_tokens=output_num_tokens,
                    )


if __name__ == "__main__":
    unittest.main()
