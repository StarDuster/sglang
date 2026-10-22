"""Retained dynamic storage cleanup must stay visible through the admin lifecycle."""

import logging
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Reuse the isolated source loader; each test restores sys.modules on exit.
from test_dynamic_storage_dcp_contract import (
    SOURCE,
    _load_module,
    _package,
    _runtime,
    _source_definition,
)

with patch.dict(sys.modules):
    _package("sglang", SOURCE)
    _package("sglang.test", SOURCE / "test")
    _package("sglang.test.ci", SOURCE / "test/ci")
    from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def _lifecycle_runtime():
    with _runtime() as runtime:
        logger = logging.getLogger(__name__)
        scope = {
            "logger": logger,
            "get_parallel": lambda: SimpleNamespace(attn_dcp_size=1),
            "PoolName": runtime.contract.PoolName,
        }
        methods = {
            name: _source_definition(
                SOURCE / "srt/managers/cache_controller.py",
                "HiCacheController",
                name,
                **scope,
            )
            for name in (
                "attach_storage_backend",
                "detach_storage_backend",
                "_storage_host_pools",
            )
        }
        controller_class = type("HiCacheController", (), methods)
        for name, attrs in (
            (
                "sglang.srt.mem_cache.buffer_mode.pipeline",
                {"validate_buffer_only_stack": Mock()},
            ),
            (
                "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
                {
                    "HybridCacheController": SimpleNamespace(
                        parse_storage_backend_extra_config=Mock(
                            return_value=({}, 1, 1, 1, False)
                        )
                    )
                },
            ),
            (
                "sglang.srt.mem_cache.unified_cache.component_type",
                {"ComponentType": SimpleNamespace()},
            ),
            (
                "sglang.srt.observability.metrics_collector",
                {
                    "STAT_LOGGER_ROLE_STORAGE": "storage",
                    "StorageMetricsCollector": object,
                    "resolve_collector_class": Mock(),
                },
            ),
            ("sglang.srt.mem_cache.utils", {"get_hash_str": Mock()}),
        ):
            module = types.ModuleType(name)
            module.__dict__.update(attrs)
            sys.modules[name] = module
        attachment = _load_module(
            "storage_attachment_cleanup_under_test",
            SOURCE / "srt/mem_cache/unified_cache/storage_attachment.py",
        ).StorageAttachment
        scheduler_detach = _source_definition(
            SOURCE / "srt/managers/scheduler.py",
            "Scheduler",
            "detach_hicache_storage_wrapped",
            logger=logger,
            DetachHiCacheStorageReqOutput=SimpleNamespace,
        )
        yield SimpleNamespace(
            Controller=controller_class,
            Attachment=attachment,
            scheduler_detach=scheduler_detach,
            factory=runtime.factory,
        )


class _RetryCloseBackend:
    requires_explicit_memory_cleanup = True

    def __init__(self):
        self.fail_close = True
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("toy memory unregister failed")

    def register_mem_pool_host(self, pool):
        self.pool = pool


def _fixture(runtime):
    backend = _RetryCloseBackend()
    controller = runtime.Controller()
    controller.storage_backend = backend
    controller.storage_backend_type = "dynamic"
    controller.enable_storage = True
    controller.prefetch_hits_sync_groups = []
    controller.prefetch_completion_sync_groups = []
    controller._stop_storage_threads = Mock()
    controller._destroy_sync_groups = Mock()
    controller._create_sync_groups = Mock(return_value=[])
    controller._start_storage_threads = Mock()
    controller.storage_stop_event = threading.Event()
    controller.storage_host_pool = SimpleNamespace(size=16, entries=[])
    controller.mem_pool_host = controller.storage_host_pool
    controller.host_memory_mode = "cache"
    controller.page_size = 1
    controller.write_policy = "write_through"
    controller._generate_storage_config = Mock(
        return_value=SimpleNamespace(extra_config={})
    )
    cache = SimpleNamespace(
        cache_controller=controller,
        enable_storage=True,
        enable_storage_metrics=True,
        drain_storage_control_queues_local=Mock(),
        is_write_back=False,
        host_memory_mode="cache",
        prefetch_stop_policy="best_effort",
        write_through_threshold=1,
        _enable_metrics_flag=False,
        extra_metric_labels=None,
        page_size=1,
        tree_core=SimpleNamespace(backfill_missing_hash_values=Mock(return_value=0)),
    )
    base_attach = controller.attach_storage_backend
    controller.attach_storage_backend = lambda **kwargs: base_attach(
        **{key: value for key, value in kwargs.items() if key != "host_pools"}
    )
    controller._generic_page_get = Mock()
    controller._generic_page_set = Mock()
    attachment = runtime.Attachment(cache)
    attachment._release_pending_storage_ops = Mock()
    attachment.resolve_prefetch_threshold = Mock(return_value=1)
    return attachment, cache, controller, backend


class TestStorageCleanupRetry(unittest.TestCase):
    def setUp(self):
        context = _lifecycle_runtime()
        self.runtime = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def test_close_failure_rejects_reattach_until_successful_retry(self):
        attachment, cache, controller, backend = _fixture(self.runtime)
        ok, _ = attachment.detach()
        self.assertFalse(ok)
        self.assertFalse(cache.enable_storage)
        self.assertFalse(controller.enable_storage)
        self.assertFalse(cache.enable_storage_metrics)
        self.assertIs(controller.storage_backend, backend)
        attachment._release_pending_storage_ops.assert_not_called()
        ok, message = attachment.attach("dynamic")
        self.assertFalse(ok)
        self.assertIn("retry detach", message)
        controller._start_storage_threads.assert_not_called()

        backend.fail_close = False
        self.assertTrue(attachment.detach()[0])
        self.assertIsNone(controller.storage_backend)
        attachment._release_pending_storage_ops.assert_called_once()
        replacement = _RetryCloseBackend()
        with patch.object(
            self.runtime.factory, "create_backend", return_value=replacement
        ):
            ok, message = attachment.attach("dynamic")
        self.assertTrue(ok, message)
        self.assertTrue(cache.enable_storage)
        self.assertTrue(controller.enable_storage)
        self.assertIs(controller.storage_backend, replacement)
        controller._start_storage_threads.assert_called_once()

    def test_shutdown_retries_retained_backend_while_disabled(self):
        attachment, cache, controller, backend = _fixture(self.runtime)
        cache.enable_storage = controller.enable_storage = False
        backend.fail_close = False
        attachment.shutdown()
        self.assertEqual(backend.close_calls, 1)
        self.assertIsNone(controller.storage_backend)

    def test_retained_backend_blocks_fresh_attachment_object(self):
        attachment, cache, controller, _ = _fixture(self.runtime)
        cache.enable_storage = controller.enable_storage = False
        self.assertFalse(attachment.attach("dynamic")[0])

    def test_thread_stop_failure_preserves_active_state(self):
        attachment, cache, controller, backend = _fixture(self.runtime)
        controller._stop_storage_threads.side_effect = RuntimeError("worker alive")
        self.assertFalse(attachment.detach()[0])
        self.assertTrue(cache.enable_storage)
        self.assertTrue(controller.enable_storage)
        self.assertEqual(backend.close_calls, 0)
        attachment._release_pending_storage_ops.assert_not_called()
        self.assertFalse(attachment.attach("dynamic")[0])

    def test_post_cleanup_failure_stays_pending_for_shutdown(self):
        attachment, cache, controller, backend = _fixture(self.runtime)
        backend.fail_close = False
        attachment._release_pending_storage_ops.side_effect = [
            RuntimeError("pending bookkeeping"),
            None,
        ]
        self.assertFalse(attachment.detach()[0])
        self.assertIsNone(controller.storage_backend)
        self.assertFalse(cache.enable_storage)
        self.assertFalse(attachment.attach("dynamic")[0])
        attachment.shutdown()
        self.assertEqual(attachment._release_pending_storage_ops.call_count, 2)
        self.assertFalse(attachment._cleanup_pending)

    def test_failed_backup_unlock_keeps_record_for_retry(self):
        attachment, cache, _, _ = _fixture(self.runtime)
        cache.ongoing_prefetch = {}
        cache.ongoing_backup = {7: (11, "lock")}
        cache.dec_host_lock_ref = Mock(side_effect=[RuntimeError("unlock"), None])
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {}
        cache.storage_prefetch_retries = Mock()
        with self.assertRaises(RuntimeError):
            self.runtime.Attachment._release_pending_storage_ops(attachment)
        self.assertEqual(cache.ongoing_backup, {7: (11, "lock")})
        self.runtime.Attachment._release_pending_storage_ops(attachment)
        self.assertEqual(cache.ongoing_backup, {})

    def test_disabled_scheduler_does_not_mask_cleanup_failure(self):
        attachment, _, _, _ = _fixture(self.runtime)
        scheduler = SimpleNamespace(
            enable_hierarchical_cache=True,
            enable_hicache_storage=False,
            is_fully_idle=lambda: True,
            tree_cache=SimpleNamespace(detach_storage_backend=attachment.detach),
        )
        result = self.runtime.scheduler_detach(scheduler, None)
        self.assertFalse(result.success)
        self.assertIn("cleanup failed", result.message)


if __name__ == "__main__":
    unittest.main()
