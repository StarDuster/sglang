"""CPU tests for dynamic DCP storage admission and cleanup.

The complete lightweight storage contract/factory modules are loaded from this
checkout. Pool class definitions and controller methods are replayed unchanged
from source, with GPU allocator bases omitted and no constructors/kernels run.
All import substitutions are scoped and restored; no installed storage plugin,
server, native client or distributed process group is used.
"""

import ast
import importlib.util
import logging
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from queue import Queue
from unittest import mock

import torch

SOURCE = Path(__file__).resolve().parents[4] / "python" / "sglang"


def _package(name, path=None):
    module = types.ModuleType(name)
    module.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = module
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Use the repository's actual CI marker without importing the public runtime API.
with mock.patch.dict(sys.modules):
    _package("sglang", SOURCE)
    _package("sglang.test", SOURCE / "test")
    _package("sglang.test.ci", SOURCE / "test/ci")
    from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _source_definition(path, class_name, method=None, **scope):
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    node = (
        cls
        if method is None
        else next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == method
        )
    )
    unit = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(unit)
    exec(compile(unit, str(path), "exec"), scope)
    return scope[class_name if method is None else method]


@contextmanager
def _runtime():
    # patch.dict restores existing module objects and removes newly loaded ones.
    with mock.patch.dict(sys.modules):
        for name, path in (
            ("sglang", SOURCE),
            ("sglang.srt", SOURCE / "srt"),
            ("sglang.srt.mem_cache", SOURCE / "srt/mem_cache"),
        ):
            _package(name, path)
        _load_module("sglang.srt.environ", SOURCE / "srt/environ.py")
        contract = _load_module(
            "sglang.srt.mem_cache.hicache_storage",
            SOURCE / "srt/mem_cache/hicache_storage.py",
        )
        allocator_base = type("AllocatorBaseNotConstructed", (), {})
        mla = _source_definition(
            SOURCE / "srt/mem_cache/pool_host/mla.py",
            "MLATokenToKVPoolHost",
            HostKVCache=allocator_base,
            HiSparseHostPoolMixin=type("HiSparseAllocatorNotConstructed", (), {}),
            torch=torch,
        )
        mamba = _source_definition(
            SOURCE / "srt/mem_cache/pool_host/mamba.py",
            "MambaPoolHost",
            HostKVCache=allocator_base,
            synchronized=lambda fn: fn,
            torch=torch,
        )
        _package("sglang.srt.mem_cache.pool_host")
        for name, cls in (("mla", mla), ("mamba", mamba)):
            module = types.ModuleType("sglang.srt.mem_cache.pool_host." + name)
            setattr(module, cls.__name__, cls)
            sys.modules[module.__name__] = module
        storage = _package(
            "sglang.srt.mem_cache.storage", SOURCE / "srt/mem_cache/storage"
        )
        factory = _load_module(
            "sglang.srt.mem_cache.storage.backend_factory",
            SOURCE / "srt/mem_cache/storage/backend_factory.py",
        ).StorageBackendFactory
        storage.StorageBackendFactory = factory
        yield types.SimpleNamespace(
            contract=contract, mla=mla, mamba=mamba, factory=factory
        )


class TestDynamicStorageDCPContract(unittest.TestCase):
    def setUp(self):
        context = _runtime()
        self.runtime = context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)
        self.events = []
        self.names = self.runtime.contract.PoolName
        events = self.events
        base = self.runtime.contract.HiCacheStorage

        class ToyStorage(base):
            dcp_storage_capabilities = frozenset(
                {
                    "mla_sharded",
                    "logical_page_indices",
                    "mamba_tp_sharded",
                    "exact_restorable_prefix_pages",
                    "draft_mla_sharded",
                }
            )

            @classmethod
            def validate_dcp_config(cls, cfg, pools):
                events.append(("validate", tuple(pools)))
                if cfg.extra_config.get("reject_validation"):
                    raise ValueError("Unsupported toy storage format")

            def __init__(self, cfg, extra_config=None):
                events.append(("construct",))
                self.cfg = cfg

            def get(self, *args, **kwargs):
                raise NotImplementedError

            def set(self, *args, **kwargs):
                raise NotImplementedError

            def batch_get(self, *args, **kwargs):
                raise NotImplementedError

            def batch_set(self, *args, **kwargs):
                raise NotImplementedError

            def exists(self, *args, **kwargs):
                raise NotImplementedError

        self.backend = ToyStorage
        plugin = types.ModuleType("test_dynamic_storage_plugin")
        plugin.ToyStorage = ToyStorage
        sys.modules[plugin.__name__] = plugin

    def config(self, dcp=8, **overrides):
        values = dict(
            tp_rank=0,
            tp_size=8,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=True,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name="toy",
            dcp_size=dcp,
            dcp_rank=0,
            logical_page_size=128,
            extra_config={
                "backend_name": "toy",
                "module_path": "test_dynamic_storage_plugin",
                "class_name": "ToyStorage",
            },
        )
        values.update(overrides)
        return self.runtime.contract.HiCacheStorageConfig(**values)

    def pool(self, **overrides):
        pool = object.__new__(self.runtime.mla)
        pool.__dict__.update(
            kv_buffer=torch.empty(1),
            layout="page_first_direct",
            device_pool=types.SimpleNamespace(kv_scale_buffer=None),
            dcp_size=8,
            dcp_rank=0,
            logical_page_size=128,
            page_size=16,
        )
        pool.__dict__.update(overrides)
        return pool

    def create(self, cfg=None, pools=None):
        cfg = cfg or self.config()
        pools = pools or {self.names.KV: self.pool()}
        return self.runtime.factory.create_backend(
            "dynamic", cfg, pools[self.names.KV], storage_host_pools=pools
        )

    def assert_no_constructor(self):
        self.assertNotIn(("construct",), self.events)

    def test_validates_before_constructor_with_actual_pool_definitions(self):
        pools = {
            self.names.KV: self.pool(),
            self.names.MAMBA: object.__new__(self.runtime.mamba),
            self.names.DRAFT: self.pool(),
        }
        result = self.create(pools=pools)
        self.assertIsInstance(result, self.backend)
        self.assertEqual(self.events, [("validate", tuple(pools)), ("construct",)])

    def test_base_defaults_deny_dcp_and_do_not_require_special_cleanup(self):
        base = self.runtime.contract.HiCacheStorage
        self.assertEqual(base.dcp_storage_capabilities, frozenset())
        self.assertFalse(base.requires_explicit_memory_cleanup)
        with self.assertRaises(NotImplementedError):
            base.validate_dcp_config(self.config(), {})

    def test_missing_capabilities_cannot_be_granted_by_configuration(self):
        self.backend.dcp_storage_capabilities = frozenset()
        cfg = self.config()
        cfg.extra_config.update(
            supports_dcp=True,
            dcp_storage_capabilities=[
                "mla_sharded",
                "logical_page_indices",
                "mamba_tp_sharded",
                "exact_restorable_prefix_pages",
                "draft_mla_sharded",
            ],
            mla_sharded=True,
            logical_page_indices=True,
        )
        with self.assertRaises(NotImplementedError):
            self.create(cfg)
        self.assertEqual(self.events, [])

    def test_malformed_capability_declarations_are_rejected(self):
        valid = self.backend.dcp_storage_capabilities
        for capabilities in (
            True,
            set(valid),
            list(valid),
            "mla_sharded",
            valid | {123},
            valid | {""},
        ):
            with self.subTest(capabilities=capabilities):
                self.backend.dcp_storage_capabilities = capabilities
                with self.assertRaises(TypeError):
                    self.create()
                self.assert_no_constructor()

    def test_capability_declaration_alone_does_not_replace_validation(self):
        self.backend.validate_dcp_config = classmethod(
            self.runtime.contract.HiCacheStorage.validate_dcp_config.__func__
        )
        with self.assertRaises(NotImplementedError):
            self.create()
        self.assert_no_constructor()

    def test_plugin_validation_error_precedes_constructor(self):
        cfg = self.config()
        cfg.extra_config["reject_validation"] = True
        with self.assertRaises(ValueError):
            self.create(cfg)
        self.assertEqual(self.events, [("validate", (self.names.KV,))])

    def test_dcp1_legacy_plugin_does_not_need_new_capabilities(self):
        self.backend.dcp_storage_capabilities = frozenset()
        self.backend.validate_dcp_config = classmethod(
            self.runtime.contract.HiCacheStorage.validate_dcp_config.__func__
        )
        self.create(self.config(dcp=1))
        self.assertEqual(self.events, [("construct",)])

    def test_dcp1_opted_in_plugin_still_validates_its_contract(self):
        cfg = self.config(dcp=1)
        cfg.extra_config["reject_validation"] = True
        with self.assertRaises(ValueError):
            self.create(cfg)
        self.assert_no_constructor()

    def test_same_shape_non_mla_object_does_not_pass_type_check(self):
        impostor = types.SimpleNamespace(**self.pool().__dict__)
        with self.assertRaises(NotImplementedError):
            self.create(pools={self.names.KV: impostor})
        self.assert_no_constructor()

    def test_actual_geometry_and_materialization_must_match(self):
        changes = [
            {"kv_buffer": None},
            {"layout": "page_first_kv_split"},
            {"dcp_rank": 1},
            {"dcp_size": 4},
            {"logical_page_size": 64},
            {"page_size": 32},
            {"device_pool": types.SimpleNamespace(kv_scale_buffer=torch.empty(1))},
        ]
        for values in changes:
            with self.subTest(values=values):
                with self.assertRaises(NotImplementedError):
                    self.create(pools={self.names.KV: self.pool(**values)})
        self.assert_no_constructor()

    def test_split_heads_and_mismatched_draft_layout_rejected(self):
        cfg = self.config()
        cfg.should_split_heads = True
        with self.assertRaises(NotImplementedError):
            self.create(cfg)
        with self.assertRaises(NotImplementedError):
            self.create(
                pools={
                    self.names.KV: self.pool(),
                    self.names.DRAFT: self.pool(layout="page_first"),
                }
            )
        self.assert_no_constructor()

    def test_each_sidecar_requires_its_own_capability(self):
        capabilities = self.backend.dcp_storage_capabilities
        for name, pool, required in (
            (self.names.MAMBA, object.__new__(self.runtime.mamba), "mamba_tp_sharded"),
            (
                self.names.MAMBA,
                object.__new__(self.runtime.mamba),
                "exact_restorable_prefix_pages",
            ),
            (self.names.DRAFT, self.pool(), "draft_mla_sharded"),
        ):
            with self.subTest(required=required):
                self.backend.dcp_storage_capabilities = capabilities - {required}
                with self.assertRaises(NotImplementedError):
                    self.create(pools={self.names.KV: self.pool(), name: pool})
        self.assert_no_constructor()

    def test_wrong_mamba_type_and_unknown_sidecars_rejected(self):
        for name, pool in (
            (self.names.MAMBA, self.pool()),
            (self.names.SWA, self.pool()),
        ):
            with self.subTest(name=name):
                with self.assertRaises(NotImplementedError):
                    self.create(pools={self.names.KV: self.pool(), name: pool})
        self.assert_no_constructor()

    def test_class_must_inherit_storage_contract(self):
        sys.modules["test_dynamic_storage_plugin"].ToyStorage = type(
            "NotAStorage", (), {}
        )
        with self.assertRaises(TypeError):
            self.create()

    def test_late_registration_publishes_only_after_validation_and_registration(self):
        events = self.events

        class Group:
            def __init__(self):
                self.entry_map = {
                    self_names.KV: types.SimpleNamespace(host_pool=anchor)
                }

            def add_entry(self, entry):
                events.append(("publish",))
                self.entry_map[entry.name] = entry

        self_names = self.names
        anchor = self.pool()
        group = Group()
        cfg = self.config()
        backend = self.backend(cfg)
        self.events.clear()

        def register(pool, name):
            events.append(("register",))

        backend.register_mem_host_pool_v2 = register
        controller = types.SimpleNamespace(
            mem_pool_host=group,
            enable_storage=True,
            storage_backend=backend,
            storage_backend_type="dynamic",
            storage_config=cfg,
            extra_host_mem_release_queues={},
            _storage_host_pools=lambda: {
                name: entry.host_pool for name, entry in group.entry_map.items()
            },
        )
        method = _source_definition(
            SOURCE / "srt/mem_cache/hybrid_cache/hybrid_cache_controller.py",
            "HybridCacheController",
            "register_host_pool_entry",
            HostPoolGroup=Group,
            Queue=Queue,
        )
        entry = types.SimpleNamespace(
            name=self.names.DRAFT, host_pool=self.pool(), is_primary_index_anchor=False
        )
        cfg.extra_config["reject_validation"] = True
        with self.assertRaises(ValueError):
            method(controller, entry)
        self.assertNotIn(self.names.DRAFT, group.entry_map)
        self.assertEqual(controller.extra_host_mem_release_queues, {})
        cfg.extra_config.pop("reject_validation")
        self.events.clear()
        backend.register_mem_host_pool_v2 = mock.Mock(
            side_effect=RuntimeError("toy registration failed")
        )
        with self.assertRaises(RuntimeError):
            method(controller, entry)
        self.assertNotIn(self.names.DRAFT, group.entry_map)
        self.assertEqual(controller.extra_host_mem_release_queues, {})
        backend.register_mem_host_pool_v2 = register
        self.events.clear()
        method(controller, entry)
        self.assertEqual(
            [event[0] for event in self.events], ["validate", "register", "publish"]
        )
        self.assertIn(self.names.DRAFT, controller.extra_host_mem_release_queues)
        self.events.clear()
        with self.assertRaises(ValueError):
            method(controller, entry)
        self.assertEqual(self.events, [])

    def test_detach_retains_explicit_cleanup_backend_until_successful_retry(self):
        method = _source_definition(
            SOURCE / "srt/managers/cache_controller.py",
            "HiCacheController",
            "detach_storage_backend",
            logger=logging.getLogger(__name__),
        )
        backend = types.SimpleNamespace(
            requires_explicit_memory_cleanup=True,
            close=mock.Mock(side_effect=RuntimeError("toy native cleanup failed")),
        )
        stop = threading.Event()
        stop.set()
        controller = types.SimpleNamespace(
            storage_backend=backend,
            enable_storage=True,
            storage_backend_type="dynamic",
            _stop_storage_threads=mock.Mock(),
            _destroy_sync_groups=mock.Mock(),
            prefetch_hits_sync_groups=[],
            prefetch_completion_sync_groups=[],
            _generic_page_get=object(),
            _generic_page_set=object(),
            storage_stop_event=stop,
        )
        with self.assertRaises(RuntimeError):
            method(controller)
        self.assertIs(controller.storage_backend, backend)
        self.assertFalse(controller.enable_storage)
        self.assertTrue(stop.is_set())
        backend.close.side_effect = None
        method(controller)
        self.assertIsNone(controller.storage_backend)
        self.assertFalse(stop.is_set())
        self.assertEqual(backend.close.call_count, 2)

    def test_attach_refuses_backend_retained_after_cleanup_failure(self):
        method = _source_definition(
            SOURCE / "srt/managers/cache_controller.py",
            "HiCacheController",
            "attach_storage_backend",
        )
        controller = types.SimpleNamespace(
            storage_backend=types.SimpleNamespace(
                requires_explicit_memory_cleanup=True
            ),
            enable_storage=False,
        )
        with self.assertRaisesRegex(RuntimeError, "retained"):
            method(controller, "dynamic")

    def test_isolated_runtime_restores_existing_module_objects(self):
        before = dict(sys.modules)
        with _runtime():
            self.assertIsNot(
                sys.modules["sglang.srt.mem_cache.hicache_storage"],
                before["sglang.srt.mem_cache.hicache_storage"],
            )
        self.assertEqual(set(sys.modules), set(before))
        self.assertTrue(
            all(sys.modules[name] is module for name, module in before.items())
        )


if __name__ == "__main__":
    unittest.main()
