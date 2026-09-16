"""Validate heterogeneous TP peers through the real HTTP bootstrap server."""

import asyncio
import copy
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.utils import get_dsv41_spec_layout
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_layout():
    args = SimpleNamespace(
        mla_compression_ratios=[0, 2, 1],
        kv_layer_ids=[1, 2],
        kv_item_lens=[512, 1024],
        state_types=[StateType.SWA, StateType.DSV4_REQUEST_STATE, StateType.SWA],
        state_item_lens=[[512], [32768], [512]],
    )
    with get_context().override_server_args(
        speculative_algorithm="DSPARK", speculative_num_draft_tokens=6
    ):
        return get_dsv41_spec_layout(args)


class TestDSV41DSparkPD(CustomTestCase):
    def test_heterogeneous_tp_and_layout_rejection(self):
        layout = make_layout()
        for prefill_tp in (2, 8):
            with self.subTest(prefill_tp=prefill_tp):
                server = CommonKVBootstrapServer("127.0.0.1", 0)
                try:
                    deadline = time.monotonic() + 10
                    while (
                        not getattr(server, "_runner", None)
                        or not server._runner.addresses
                    ):
                        if time.monotonic() > deadline:
                            self.fail("Bootstrap server failed to bind")
                        time.sleep(0.01)
                    address = f"127.0.0.1:{server._runner.addresses[0][1]}"
                    payload = dict(
                        attn_tp_size=prefill_tp,
                        attn_tp_rank=0,
                        attn_cp_size=1,
                        attn_cp_rank=0,
                        attn_dp_size=1,
                        attn_dp_rank=0,
                        pp_size=1,
                        pp_rank=0,
                        system_dp_size=1,
                        system_dp_rank=0,
                        rank_ip="127.0.0.1",
                        rank_port=1234,
                        page_size=256,
                        kv_cache_dtype="fp8_e4m3",
                        dsv41_spec_layout=layout,
                    )
                    for rank in range(prefill_tp):
                        payload["attn_tp_rank"] = rank
                        response = requests.put(
                            f"http://{address}/route", json=payload, timeout=5
                        )
                        self.assertEqual(response.status_code, 200, response.text)
                    for rank in range(4):
                        manager = CommonKVManager.__new__(CommonKVManager)
                        manager.prefill_info_table = {}
                        manager.kv_args = SimpleNamespace(
                            page_size=256, engine_rank=rank
                        )
                        manager.kv_cache_dtype_str = "fp8_e4m3"
                        manager.dsv41_spec_layout = layout
                        manager.attn_tp_size = 4
                        manager.attn_cp_size = 1
                        manager.attn_cp_rank = 0
                        manager.dcp_size = 1
                        manager.pp_size = 1
                        manager.pp_rank = 0
                        manager.is_mla_backend = True
                        manager.is_hybrid_mla_backend = True
                        self.assertTrue(manager.try_ensure_parallel_info(address))
                        info = manager.prefill_info_table[address]
                        expected = (
                            [rank // 2] if prefill_tp == 2 else [2 * rank, 2 * rank + 1]
                        )
                        self.assertEqual(info.target_tp_ranks, expected)
                        self.assertTrue(manager.try_ensure_parallel_info(address))
                        self.assertIs(info, manager.prefill_info_table[address])

                    for field, bad_value in (
                        ("num_draft_tokens", 5),
                        ("kv_layer_ids", [2, 1]),
                        ("kv_item_lens", [256, 1024]),
                        ("state_types", ["swa"]),
                        ("state_item_lens", [[512], [8192], [512]]),
                    ):
                        with self.subTest(field=field):
                            manager.prefill_info_table.clear()
                            different = copy.deepcopy(layout)
                            different[field] = bad_value
                            manager.dsv41_spec_layout = different
                            with self.assertRaisesRegex(
                                RuntimeError, "PD layout mismatch"
                            ):
                                manager.try_ensure_parallel_info(address)
                            self.assertFalse(manager.prefill_info_table)
                    payload["dsv41_spec_layout"] = None
                    response = requests.put(
                        f"http://{address}/route", json=payload, timeout=5
                    )
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(server.dsv41_spec_layout, layout)
                finally:

                    async def cancel_cleanup():
                        tasks = list(server._background_tasks)
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)

                    try:
                        asyncio.run_coroutine_threadsafe(
                            cancel_cleanup(), server._loop
                        ).result(timeout=5)
                    finally:
                        server.close()


if __name__ == "__main__":
    unittest.main()
