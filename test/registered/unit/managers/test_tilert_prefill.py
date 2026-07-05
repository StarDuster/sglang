import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.tilert_prefill import (
    TileRTPrefillWorker,
    _build_prefill_engine_kwargs,
    _external_prefill_warmup_len,
    export_prefill_kv,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTileRTPrefill(unittest.TestCase):
    def test_nested_prefill_engine_uses_fast_fp8_dsa_configuration(self):
        server_args = SimpleNamespace(
            trust_remote_code=True,
            tilert_prefill_tp_size=8,
            tilert_prefill_mem_fraction=0.85,
            tilert_prefill_max_total_tokens=65536,
            context_length=131072,
        )

        kwargs = _build_prefill_engine_kwargs(server_args, "/models/glm")

        self.assertEqual(kwargs["model_path"], "/models/glm")
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8_e4m3")
        self.assertEqual(kwargs["attention_backend"], "dsa")
        self.assertEqual(kwargs["dsa_prefill_backend"], "trtllm")
        self.assertEqual(kwargs["dsa_decode_backend"], "trtllm")
        self.assertTrue(kwargs["enable_dsa_prefill_context_parallel"])
        self.assertEqual(kwargs["dsa_prefill_cp_mode"], "round-robin-split")
        self.assertEqual(kwargs["attn_cp_size"], 8)
        self.assertTrue(kwargs["disable_cuda_graph"])
        self.assertEqual(kwargs["max_total_tokens"], 65536)
        self.assertEqual(kwargs["max_prefill_tokens"], 64512)
        self.assertEqual(kwargs["chunked_prefill_size"], 16384)

    def test_nested_prefill_engine_token_budget_is_consistent_with_small_cap(self):
        server_args = SimpleNamespace(
            trust_remote_code=True,
            tilert_prefill_tp_size=8,
            tilert_prefill_mem_fraction=0.92,
            tilert_prefill_max_total_tokens=16384,
            context_length=202752,
        )

        kwargs = _build_prefill_engine_kwargs(server_args, "/models/glm")

        self.assertEqual(kwargs["max_total_tokens"], 16384)
        self.assertEqual(kwargs["max_prefill_tokens"], 15360)
        self.assertEqual(kwargs["chunked_prefill_size"], 15360)

    def test_external_prefill_warmup_len_is_page_aligned_and_capped(self):
        self.assertEqual(_external_prefill_warmup_len(1024, 115994), 32768)
        self.assertEqual(_external_prefill_warmup_len(1024, 20000), 19968)
        self.assertEqual(_external_prefill_warmup_len(65536, 115994), 65536)
        self.assertEqual(_external_prefill_warmup_len(1024, 63), 0)

    def test_export_prefill_kv_returns_captured_last_hidden_state(self):
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(tp_rank=0),
            _tilert_prefill_last_hidden_states={
                "tilert-prefill-test": torch.ones(8, dtype=torch.bfloat16)
            },
        )
        layer_caches = [(torch.empty(1), torch.empty(1), torch.empty(1))]

        with patch(
            "sglang.srt.managers.tilert_prefill._collect_prefill_kv",
            return_value=(4, layer_caches, 0.25),
        ):
            cached_len, caches, last_hidden_state, gather_elapsed = export_prefill_kv(
                scheduler, [1, 2, 3, 4], rid="tilert-prefill-test"
            )

        self.assertEqual(cached_len, 4)
        self.assertIs(caches, layer_caches)
        self.assertEqual(gather_elapsed, 0.25)
        self.assertTrue(
            torch.equal(last_hidden_state, torch.ones(8, dtype=torch.bfloat16))
        )
        self.assertEqual(scheduler._tilert_prefill_last_hidden_states, {})

    def test_required_external_prefill_requires_external_prefill(self):
        server_args = ServerArgs(
            model_path="dummy",
            model_impl="tilert",
            tilert_require_external_prefill=True,
        )

        with self.assertRaisesRegex(ValueError, "requires --tilert-external-prefill"):
            server_args._check_tilert_server_args()

    def test_required_external_prefill_rejects_internal_prefill_paths(self):
        for kwargs, message in (
            ({"tilert_prefill_tail_tokens": 1}, "--tilert-prefill-tail-tokens"),
        ):
            server_args = ServerArgs(
                model_path="dummy",
                model_impl="tilert",
                tilert_external_prefill=True,
                tilert_require_external_prefill=True,
                **kwargs,
            )

            with self.assertRaisesRegex(ValueError, message):
                server_args._check_tilert_server_args()

    def test_required_external_prefill_capacity_check_rejects_long_prompt(self):
        worker = TileRTPrefillWorker.__new__(TileRTPrefillWorker)
        worker.max_prefill_tokens = 5504
        worker._ensure_ready = lambda cancel=None: None

        with self.assertRaisesRegex(RuntimeError, "5504 tokens"):
            worker.ensure_full_prompt_capacity(57005)

    def test_required_external_prefill_capacity_check_allows_full_prompt(self):
        worker = TileRTPrefillWorker.__new__(TileRTPrefillWorker)
        worker.max_prefill_tokens = 5504
        worker._ensure_ready = lambda cancel=None: None

        worker.ensure_full_prompt_capacity(5504)


if __name__ == "__main__":
    unittest.main()
