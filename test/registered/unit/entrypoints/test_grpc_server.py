import asyncio
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.entrypoints import grpc_server
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGrpcServerMultinodeWorker(unittest.TestCase):
    def test_multinode_worker_detection(self):
        self.assertFalse(
            grpc_server._is_multinode_grpc_worker(
                SimpleNamespace(nnodes=1, node_rank=1)
            )
        )
        self.assertFalse(
            grpc_server._is_multinode_grpc_worker(
                SimpleNamespace(nnodes=4, node_rank=0)
            )
        )
        self.assertTrue(
            grpc_server._is_multinode_grpc_worker(
                SimpleNamespace(nnodes=4, node_rank=1)
            )
        )

    def test_non_rank0_uses_worker_scheduler_path(self):
        async def third_party_serve_grpc(*args, **kwargs):
            raise AssertionError("rank0 gRPC server path must not run on workers")

        server_module = types.ModuleType("smg_grpc_servicer.sglang.server")
        server_module.serve_grpc = third_party_serve_grpc

        server_args = SimpleNamespace(
            disaggregation_mode="decode",
            enable_metrics=False,
            grpc_http_sidecar_port=None,
            host="0.0.0.0",
            nnodes=4,
            node_rank=2,
            port=30000,
        )

        worker_path = AsyncMock()
        with patch.dict(
            sys.modules,
            {"smg_grpc_servicer.sglang.server": server_module},
        ), patch.object(grpc_server, "_serve_grpc_worker_node", worker_path):
            asyncio.run(grpc_server.serve_grpc(server_args))

        worker_path.assert_awaited_once()
        call_args = worker_path.await_args.args
        self.assertIs(call_args[0], server_args)
        self.assertEqual(call_args[2], 30001)

    def test_worker_node_launches_schedulers_without_frontend(self):
        proc = Mock()
        launcher = Mock(return_value=({"status": "ready"}, object(), [proc]))

        launcher_module = types.ModuleType(
            "smg_grpc_servicer.sglang.scheduler_launcher"
        )
        launcher_module.launch_scheduler_process_only = launcher

        server_args = SimpleNamespace(
            disaggregation_mode="decode",
            host="0.0.0.0",
            node_rank=3,
            port=30000,
        )
        wait_for_schedulers = AsyncMock()

        with patch.dict(
            sys.modules,
            {"smg_grpc_servicer.sglang.scheduler_launcher": launcher_module},
        ), patch.object(
            grpc_server,
            "_wait_for_scheduler_processes",
            wait_for_schedulers,
        ):
            app = grpc_server.web.Application()
            asyncio.run(grpc_server._serve_grpc_worker_node(server_args, app, 30001))

        launcher.assert_called_once_with(server_args=server_args)
        wait_for_schedulers.assert_awaited_once_with([proc])


if __name__ == "__main__":
    unittest.main()
