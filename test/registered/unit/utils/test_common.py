import asyncio
import os
import threading
import unittest
from array import array
from unittest.mock import patch

import torch
from starlette.applications import Starlette

from sglang.srt.utils import common
from sglang.srt.utils.common import flatten_arrays_to_int64_tensor
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req get_fill_ids() slices. Tests the
    full matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


class TestPrometheusMiddleware(unittest.TestCase):
    def test_metrics_endpoint_serves_cached_snapshot_to_concurrent_scrapes(self):
        call_count = 0
        refreshed = threading.Event()
        call_count_lock = threading.Lock()

        def generate_metrics(
            prometheus_multiproc_dir_name: str, timeout_seconds: float
        ):
            nonlocal call_count
            self.assertEqual(prometheus_multiproc_dir_name, "/tmp/sgl-test-prom")
            self.assertEqual(timeout_seconds, 5.0)
            with call_count_lock:
                call_count += 1
            refreshed.set()
            return b"sglang_test_metric 1\n", "text/plain; version=0.0.4"

        async def run_check():
            app = Starlette()
            with patch.dict(
                os.environ,
                {
                    "PROMETHEUS_MULTIPROC_DIR": "/tmp/sgl-test-prom",
                    "SGLANG_PROMETHEUS_SNAPSHOT_REFRESH_INTERVAL_SECONDS": "999",
                    "SGLANG_PROMETHEUS_SNAPSHOT_MAX_STALENESS_SECONDS": "30",
                    "SGLANG_PROMETHEUS_SNAPSHOT_GENERATION_TIMEOUT_SECONDS": "5",
                },
            ), patch.object(
                common,
                "_generate_prometheus_latest_in_subprocess",
                generate_metrics,
            ):
                common.add_prometheus_middleware(app)
                try:
                    metrics_endpoint = next(
                        route.endpoint
                        for route in app.routes
                        if route.path == "/metrics"
                    )

                    self.assertTrue(
                        await asyncio.wait_for(
                            asyncio.to_thread(refreshed.wait), timeout=5
                        )
                    )
                    for _ in range(100):
                        response = await metrics_endpoint(None)
                        if response.status_code == 200:
                            break
                        await asyncio.sleep(0.01)
                    self.assertEqual(response.status_code, 200)

                    responses = await asyncio.gather(
                        *(metrics_endpoint(None) for _ in range(16))
                    )
                    self.assertTrue(
                        all(response.status_code == 200 for response in responses)
                    )
                    self.assertTrue(
                        all(
                            response.body == b"sglang_test_metric 1\n"
                            for response in responses
                        )
                    )
                    with call_count_lock:
                        self.assertEqual(call_count, 1)
                finally:
                    app.state.prometheus_metrics_snapshotter.close()

        asyncio.run(run_check())

    def test_metrics_endpoint_returns_503_until_first_snapshot_is_ready(self):
        started = threading.Event()
        release = threading.Event()

        def generate_metrics(
            prometheus_multiproc_dir_name: str, timeout_seconds: float
        ):
            self.assertEqual(prometheus_multiproc_dir_name, "/tmp/sgl-test-prom")
            self.assertEqual(timeout_seconds, 5.0)
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return b"sglang_test_metric 1\n", "text/plain; version=0.0.4"

        async def run_check():
            app = Starlette()
            with patch.dict(
                os.environ,
                {
                    "PROMETHEUS_MULTIPROC_DIR": "/tmp/sgl-test-prom",
                    "SGLANG_PROMETHEUS_SNAPSHOT_REFRESH_INTERVAL_SECONDS": "999",
                    "SGLANG_PROMETHEUS_SNAPSHOT_MAX_STALENESS_SECONDS": "30",
                    "SGLANG_PROMETHEUS_SNAPSHOT_GENERATION_TIMEOUT_SECONDS": "5",
                },
            ), patch.object(
                common,
                "_generate_prometheus_latest_in_subprocess",
                generate_metrics,
            ):
                common.add_prometheus_middleware(app)
                try:
                    metrics_endpoint = next(
                        route.endpoint
                        for route in app.routes
                        if route.path == "/metrics"
                    )

                    self.assertTrue(
                        await asyncio.wait_for(
                            asyncio.to_thread(started.wait), timeout=5
                        )
                    )

                    unavailable_response = await metrics_endpoint(None)
                    self.assertEqual(unavailable_response.status_code, 503)

                    release.set()
                    for _ in range(100):
                        response = await metrics_endpoint(None)
                        if response.status_code == 200:
                            break
                        await asyncio.sleep(0.01)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.body, b"sglang_test_metric 1\n")
                finally:
                    release.set()
                    app.state.prometheus_metrics_snapshotter.close()

        asyncio.run(run_check())

    def test_metrics_endpoint_rejects_stale_snapshot(self):
        def generate_metrics(
            prometheus_multiproc_dir_name: str, timeout_seconds: float
        ):
            self.assertEqual(prometheus_multiproc_dir_name, "/tmp/sgl-test-prom")
            self.assertEqual(timeout_seconds, 5.0)
            return b"sglang_test_metric 1\n", "text/plain; version=0.0.4"

        async def run_check():
            app = Starlette()
            with patch.dict(
                os.environ,
                {
                    "PROMETHEUS_MULTIPROC_DIR": "/tmp/sgl-test-prom",
                    "SGLANG_PROMETHEUS_SNAPSHOT_REFRESH_INTERVAL_SECONDS": "999",
                    "SGLANG_PROMETHEUS_SNAPSHOT_MAX_STALENESS_SECONDS": "0.001",
                    "SGLANG_PROMETHEUS_SNAPSHOT_GENERATION_TIMEOUT_SECONDS": "5",
                },
            ), patch.object(
                common,
                "_generate_prometheus_latest_in_subprocess",
                generate_metrics,
            ):
                common.add_prometheus_middleware(app)
                try:
                    metrics_endpoint = next(
                        route.endpoint
                        for route in app.routes
                        if route.path == "/metrics"
                    )

                    for _ in range(100):
                        response = await metrics_endpoint(None)
                        if response.status_code == 200:
                            break
                        await asyncio.sleep(0.01)
                    self.assertEqual(response.status_code, 200)

                    await asyncio.sleep(0.01)
                    stale_response = await metrics_endpoint(None)
                    self.assertEqual(stale_response.status_code, 503)
                    self.assertIn(b"metrics snapshot is stale", stale_response.body)
                finally:
                    app.state.prometheus_metrics_snapshotter.close()

        asyncio.run(run_check())


if __name__ == "__main__":
    unittest.main()
