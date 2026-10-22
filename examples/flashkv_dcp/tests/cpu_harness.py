"""Full Python module harness; real CPU Torch, fake native FlashKV transport.

GPU allocator host classes are metadata stand-ins with real CPU tensor buffers.
The complete vendor Python module and adapter are imported. Native transport,
CUDA-device selection and allocator host classes are explicit test doubles.
Separate test modules replay exact checkout methods with real CPU tensors.
"""

from __future__ import annotations
import ctypes
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch
import pytest

torch = pytest.importorskip("torch", reason="CPU contract tests require real PyTorch")

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
    pytest.skip(
        "SGLang source checkout unavailable; set SGLANG_SOURCE to the checkout root",
        allow_module_level=True,
    )


def resolve_vendor_source():
    override = os.environ.get("FLASHKV_PYTHON_SOURCE")
    candidates = []
    if override:
        root = Path(override).expanduser()
        candidates = [root / "flashkv", root]
    else:
        try:
            spec = importlib.util.find_spec("flashkv")
        except (ImportError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            candidates = [Path(path) for path in spec.submodule_search_locations]
    for candidate in candidates:
        integration = candidate / "integration/sglang"
        if (integration / "flashkv_storage.py").is_file() and (
            integration / "packing_utils.py"
        ).is_file():
            return candidate.resolve()
    pytest.skip(
        "PAI FlashKV Python integration source unavailable. Set FLASHKV_PYTHON_SOURCE "
        "to an extracted package/wheel root or install the PAI client package; "
        "the vendor Python implementation is required and is never substituted.",
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


class MHATokenToKVPoolHost(HostKVCache):
    pass


class MambaPoolHost(HostKVCache):
    pass


p = package("sglang.srt.mem_cache.pool_host")
p.HostKVCache = HostKVCache
p.HostPoolGroup = HostPoolGroup
for suffix, cls in [
    ("mla", MLATokenToKVPoolHost),
    ("mha", MHATokenToKVPoolHost),
    ("mamba", MambaPoolHost),
]:
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
    directory = {}
    instances = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def start(self, options):
        self.calls.append(("start",))
        return Status()

    def register_memory(self, buffers):
        self.calls.append(("register", len(buffers)))
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
        self.calls.append(("put", len(keys)))
        for k, b in zip(keys, buffers):
            self.directory[opts.key_ns, k] = ctypes.string_at(b.ptr, b.size)
        return self._result([True] * len(keys))

    def batch_get(self, keys, buffers, opts):
        self.calls.append(("get", len(keys)))
        mask = []
        for k, b in zip(keys, buffers):
            value = self.directory.get((opts.key_ns, k))
            ok = value is not None and len(value) == b.size
            mask.append(ok)
            if ok:
                ctypes.memmove(b.ptr, value, b.size)
        return self._result(mask)

    def remove_all_objects(self, namespace):
        self.calls.append(("remove_all", namespace))
        raise AssertionError("Never clear native namespace")


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
sys.path.insert(0, str(ROOT))
from apodex_flashkv.storage import FlashKVDCPStorage


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
        logical_page_size=128,
        extra_config={
            "interface_v1": 1,
            "flashkv_dcp": {
                "schema_version": 1,
                "model_revision": "target-sha-a",
                "draft_revision": "draft-sha-a",
                "namespace": "unit-cpu-isolated",
            },
        },
    )
    kwargs.update(overrides)
    return contract.HiCacheStorageConfig(**kwargs)


def pool(cfg, kind="mla", pages=8, **overrides):
    if kind == "mamba":
        p = MambaPoolHost()
        p.page_size = 1
        p.num_mamba_layers = 2
        p.temporal_state_shape = (2, 4)
        p.conv_state_shapes = [(3, 4)]
        p.temporal_dtype = torch.float32
        p.conv_dtype = torch.bfloat16
        p.temporal_state_elem_size = 8
        p.temporal_buffer = torch.arange(pages * 8, dtype=torch.float32).reshape(
            pages, 8
        )
        p.conv_buffer = [
            torch.arange(pages * 12, dtype=torch.bfloat16).reshape(pages, 12)
        ]
        p._buffers = [p.temporal_buffer] + p.conv_buffer
    else:
        p = MLATokenToKVPoolHost() if kind == "mla" else MHATokenToKVPoolHost()
        p.logical_page_size = 128
        p.page_size = 128 // cfg.dcp_size
        p.dcp_size = cfg.dcp_size
        p.dcp_rank = cfg.dcp_rank
        p.dtype = torch.bfloat16
        p.layer_num = 2
        p.kv_cache_dim = 8
        p.head_num = 2
        p.head_dim = 4
        p.v_head_dim = 4
        p.kv_buffer = torch.arange(pages * 16, dtype=torch.bfloat16).reshape(pages, 16)
        p._buffers = [p.kv_buffer]
        if kind == "mha":
            p.v_buffer = p.kv_buffer.clone() + 3
            p._buffers.append(p.v_buffer)
    p.layout = "page_first_direct"
    p.device_pool = None
    p.__dict__.update(overrides)
    p.get_hybrid_pool_buffer = lambda: p._buffers

    def get_meta(indices):
        logical = getattr(p, "logical_page_size", p.page_size)
        ids = [int(indices[i]) // logical for i in range(0, len(indices), logical)]
        rows = [b[i] for i in ids for b in p._buffers]
        return [r.data_ptr() for r in rows], [
            r.numel() * r.element_size() for r in rows
        ]

    p.get_page_buffer_meta = get_meta
    return p


def store(cfg=None, *, cls=FlashKVDCPStorage, **pool_overrides):
    cfg = cfg or config()
    pools = {
        PoolName.KV: pool(cfg),
        PoolName.MAMBA: pool(cfg, "mamba"),
        PoolName.DRAFT: pool(cfg),
    }
    pools.update(pool_overrides)
    cls.validate_dcp_config(cfg, pools)
    with patch("torch.cuda.current_device", return_value=0):
        obj = cls(cfg)
    obj.register_mem_pool_host(pools[PoolName.KV])
    for name in (PoolName.MAMBA, PoolName.DRAFT):
        obj.register_mem_host_pool_v2(pools[name], name)
    return obj, pools


def object_keys(s, name, keys):
    if name == PoolName.KV:
        return s._get_page_keys(keys)[0]
    return s._get_hybrid_page_component_keys(keys, PoolTransfer(name=name))[0]


def identity(s, name):
    return s.key_namespace, tuple(object_keys(s, name, ["same-page"]))


def present(s, name, page_keys):
    for key in object_keys(s, name, page_keys):
        Client.directory[s.key_namespace, key] = b"x"
