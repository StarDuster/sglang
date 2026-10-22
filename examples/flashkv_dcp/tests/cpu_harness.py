"""CPU harness: the vendor FlashKV Python integration and the adapter run
unmodified; the native client, CUDA device selection and GPU-allocating host
pool classes are test doubles."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def resolve_sglang_source():
    override = os.environ.get("SGLANG_SOURCE")
    roots = (
        [Path(override).expanduser()]
        if override
        else [Path.cwd(), *Path(__file__).resolve().parents]
    )
    for root in roots:
        for candidate in (root / "python/sglang", root / "sglang", root):
            if (candidate / "srt/mem_cache/hicache_storage.py").is_file():
                return candidate.resolve()
    if override:
        raise RuntimeError("SGLANG_SOURCE does not point to an SGLang source checkout")
    pytest.skip("set SGLANG_SOURCE to an SGLang checkout", allow_module_level=True)


def resolve_vendor_source():
    override = os.environ.get("FLASHKV_PYTHON_SOURCE")
    if override:
        root = Path(override).expanduser()
        candidates = [root / "flashkv", root]
    else:
        spec = importlib.util.find_spec("flashkv")
        candidates = [
            Path(p) for p in (spec.submodule_search_locations if spec else [])
        ]
    for candidate in candidates:
        if (candidate / "integration/sglang/flashkv_storage.py").is_file():
            return candidate.resolve()
    pytest.skip(
        "set FLASHKV_PYTHON_SOURCE to the PAI flashkv package (the PyPI 'flashkv' is unrelated)",
        allow_module_level=True,
    )


SG = resolve_sglang_source()
VENDOR = resolve_vendor_source()


def package(name, path=None):
    m = types.ModuleType(name)
    m.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = m
    return m


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


for name, path in [
    ("sglang", SG),
    ("sglang.srt", SG / "srt"),
    ("sglang.srt.mem_cache", SG / "srt/mem_cache"),
]:
    package(name, path)
load("sglang.srt.environ", SG / "srt/environ.py")
contract = load(
    "sglang.srt.mem_cache.hicache_storage", SG / "srt/mem_cache/hicache_storage.py"
)
PoolName = contract.PoolName
PoolHitPolicy = contract.PoolHitPolicy
PoolTransfer = contract.PoolTransfer


class HostKVCache:
    pass


class HostPoolGroup:
    pass


class LogicalHostPool:
    pass


class MLATokenToKVPoolHost(HostKVCache):
    pass


class MambaPoolHost(HostKVCache):
    pass


p = package("sglang.srt.mem_cache.pool_host")
p.HostKVCache = HostKVCache
p.HostPoolGroup = HostPoolGroup
for suffix, cls in [("mla", MLATokenToKVPoolHost), ("mamba", MambaPoolHost)]:
    m = types.ModuleType(p.__name__ + "." + suffix)
    setattr(m, cls.__name__, cls)
    sys.modules[m.__name__] = m
m = types.ModuleType("sglang.srt.mem_cache.memory_pool_host")
m.LogicalHostPool = LogicalHostPool
sys.modules[m.__name__] = m


class Status:
    CODE_OK = 0
    CODE_PARTIAL_FAILED = 1

    def __init__(self, code=0):
        self.code = code


class MemBuf:
    def __init__(self, ptr, size):
        self.ptr = ptr
        self.size = size


class Client:
    """Byte-exact in-memory store keyed by (namespace, key); copies through pointers."""

    directory = {}
    instances = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def start(self, options):
        return Status()

    def register_memory(self, buffers):
        self.calls.append(("register", [(b.ptr, b.size) for b in buffers]))
        return Status()

    def unregister_memory(self, buffers):
        self.calls.append(("unregister", [(b.ptr, b.size) for b in buffers]))
        return Status()

    def _result(self, mask):
        return Status(0 if all(mask) else 1), [
            types.SimpleNamespace(status=Status(0 if v else 2)) for v in mask
        ]

    def batch_query(self, keys, opts):
        return self._result([(opts.key_ns, k) in self.directory for k in keys])

    def batch_put(self, keys, buffers, opts):
        for k, b in zip(keys, buffers):
            self.directory[opts.key_ns, k] = ctypes.string_at(b.ptr, b.size)
        return self._result([True] * len(keys))

    def batch_get(self, keys, buffers, opts):
        mask = []
        for k, b in zip(keys, buffers):
            value = self.directory.get((opts.key_ns, k))
            ok = value is not None and len(value) == b.size
            mask.append(ok)
            if ok:
                ctypes.memmove(b.ptr, value, b.size)
        return self._result(mask)

    def remove_all_objects(self, namespace):
        for key in [k for k in self.directory if k[0] == namespace]:
            del self.directory[key]
        return Status()


native = package("flashkv", VENDOR)
for name, value in dict(
    Client=Client,
    ClientOptions=types.SimpleNamespace,
    Status=Status,
    StringVector=list,
    MemBufVector=list,
    MemBuf=MemBuf,
    QueryOptions=types.SimpleNamespace,
    GetOptions=types.SimpleNamespace,
    PutOptions=types.SimpleNamespace,
).items():
    setattr(native, name, value)
package("flashkv.integration", VENDOR / "integration")
package("flashkv.integration.sglang", VENDOR / "integration/sglang")
load(
    "flashkv.integration.sglang.packing_utils",
    VENDOR / "integration/sglang/packing_utils.py",
)
vendor = load(
    "flashkv.integration.sglang.flashkv_storage",
    VENDOR / "integration/sglang/flashkv_storage.py",
)
Factory = load(
    "sglang.srt.mem_cache.storage.backend_factory",
    SG / "srt/mem_cache/storage/backend_factory.py",
).StorageBackendFactory
sys.path.insert(0, str(ROOT))

LOGICAL_PAGE = 128


def config(rank=0, tp=8, dcp=8, **overrides):
    kwargs = dict(
        tp_rank=rank,
        tp_size=tp,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="K3",
        dcp_size=dcp,
        dcp_rank=rank % dcp,
        logical_page_size=LOGICAL_PAGE,
        extra_config={
            "backend_name": "flashkv",
            "module_path": "apodex_flashkv.storage",
            "class_name": "FlashKVDCPStorage",
            "interface_v1": 1,
            "flashkv_dcp": {
                "model_revision": "target-sha-a",
                "draft_revision": "draft-sha-a",
                "namespace": "unit-cpu-isolated",
            },
        },
    )
    kwargs.update(overrides)
    return contract.HiCacheStorageConfig(**kwargs)


def pool(cfg, kind="mla", pages=8, **overrides):
    """Metadata-only host pool with real CPU tensors; one row per page."""
    if kind == "mamba":
        p = MambaPoolHost()
        p.page_size = 1
        p.num_mamba_layers = 2
        p.temporal_state_shape = (2, 4)
        p.conv_state_shapes = [(3, 4)]
        p.temporal_dtype = torch.float32
        p.conv_dtype = torch.bfloat16
        p.temporal_state_elem_size = 8
        p.slot_state_device_tensors = []
        p.temporal_buffer = torch.arange(pages * 8, dtype=torch.float32).reshape(
            pages, 8
        )
        p.conv_buffer = [
            torch.arange(pages * 12, dtype=torch.bfloat16).reshape(pages, 12)
        ]
        p._buffers = [p.temporal_buffer] + p.conv_buffer
        p.get_hybrid_pool_buffer = lambda: p._buffers
    else:
        p = MLATokenToKVPoolHost()
        p.logical_page_size = LOGICAL_PAGE
        p.page_size = LOGICAL_PAGE // cfg.dcp_size
        p.dcp_size = cfg.dcp_size
        p.dcp_rank = cfg.dcp_rank
        p.dtype = torch.bfloat16
        p.layer_num = 2
        p.kv_cache_dim = 8
        p.device_pool = types.SimpleNamespace(kv_scale_buffer=None)
        p.kv_buffer = torch.arange(pages * 16, dtype=torch.bfloat16).reshape(pages, 16)
        p._buffers = [p.kv_buffer]
    p.layout = "page_first_direct"
    p.__dict__.update(overrides)

    def get_meta(indices):
        logical = p.logical_page_size if kind == "mla" else 1
        rows = [
            b[int(indices[i]) // logical]
            for i in range(0, len(indices), logical)
            for b in p._buffers
        ]
        return [r.data_ptr() for r in rows], [
            r.numel() * r.element_size() for r in rows
        ]

    p.get_page_buffer_meta = get_meta
    return p


def pools(cfg, **overrides):
    result = {
        PoolName.KV: pool(cfg),
        PoolName.MAMBA: pool(cfg, "mamba"),
        PoolName.DRAFT: pool(cfg),
    }
    result.update(overrides)
    return result


def store(cfg=None, **pool_overrides):
    """Create through the factory exactly as the controller does, then register pools."""
    cfg = cfg or config()
    host_pools = pools(cfg, **pool_overrides)
    with patch("torch.cuda.current_device", return_value=0):
        obj = Factory.create_backend(
            "dynamic", cfg, host_pools[PoolName.KV], storage_host_pools=host_pools
        )
    obj.register_mem_pool_host(host_pools[PoolName.KV])
    for name in (PoolName.MAMBA, PoolName.DRAFT):
        obj.register_mem_host_pool_v2(host_pools[name], name)
    return obj, host_pools


def object_keys(s, name, keys):
    if name == PoolName.KV:
        return s._get_page_keys(keys)[0]
    return s._get_hybrid_page_component_keys(keys, PoolTransfer(name=name))[0]


def identity(s, name):
    return s.key_namespace, tuple(object_keys(s, name, ["same-page"]))


def present(s, name, page_keys):
    for key in object_keys(s, name, page_keys):
        Client.directory[s.key_namespace, key] = b"x"
