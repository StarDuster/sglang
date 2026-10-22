"""DCP/TP-aware object identity and host-memory cleanup over the PAI flashkv
SGLang integration (flashkv 1.0.0+gdf62dbb).

The vendor keys MLA objects per PP rank only, so under DCP every rank's 1/dcp
KV shard would collide, and it keys KDA state without the TP rank although that
state is TP-sharded. This adapter scopes a namespace by deployment identity and
prefixes every object with a per-pool descriptor digest that carries the shard
rank. Native memory registration and transport stay in the vendor client.
"""

from __future__ import annotations

import hashlib
import json

import flashkv
from flashkv.integration.sglang.flashkv_storage import FlashKVStorage

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

KEY_SCHEMA = "apodex-flashkv-v1"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _options(cfg):
    extra = cfg.extra_config or {}
    if extra.get("interface_v1") != 1:
        raise ValueError("FlashKV requires interface_v1=1 for zero-copy page I/O")
    options = extra.get("flashkv_dcp")
    if not isinstance(options, dict) or not options.get("model_revision"):
        raise ValueError(
            "flashkv_dcp.model_revision must identify the immutable weights"
        )
    return options


def _pool_descriptor(cfg, name, pool):
    """Everything that decides whether two ranks' objects are interchangeable."""
    if name == PoolName.MAMBA:
        # KDA state is TP-sharded even though MLA KV is replicated across TP.
        return {
            "pool": "mamba",
            "rank": cfg.tp_rank,
            "layers": pool.num_mamba_layers,
            "temporal_shape": list(pool.temporal_state_shape),
            "conv_shapes": [list(shape) for shape in pool.conv_state_shapes],
            "temporal_dtype": str(pool.temporal_dtype),
            "conv_dtype": str(pool.conv_dtype),
        }
    return {
        "pool": str(name),
        "rank": cfg.dcp_rank,
        "layout": pool.layout,
        "logical_page_size": pool.logical_page_size,
        "page_size": pool.page_size,
        "dtype": str(pool.dtype),
        "layers": pool.layer_num,
        "kv_cache_dim": pool.kv_cache_dim,
    }


class FlashKVDCPStorage(FlashKVStorage):
    requires_explicit_memory_cleanup = True
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
    def validate_dcp_config(cls, storage_config, host_pools):
        options = _options(storage_config)
        if not storage_config.is_mla_model:
            raise ValueError(
                "FlashKV DCP keys MLA KV by DCP shard; MHA targets are unsupported"
            )
        if storage_config.attn_cp_size != 1:
            raise ValueError(
                "Attention CP shards are not part of the FlashKV key identity"
            )
        for name, pool in host_pools.items():
            if name == PoolName.MAMBA:
                if pool.slot_state_device_tensors:
                    raise ValueError("Mamba slot side states have no FlashKV object")
                continue
            if name not in (PoolName.KV, PoolName.DRAFT) or not isinstance(
                pool, MLATokenToKVPoolHost
            ):
                raise ValueError(f"Unsupported FlashKV pool: {name}")
            if pool.layout not in ("page_first", "page_first_direct"):
                raise ValueError(
                    f"FlashKV zero-copy needs a page-first layout: {pool.layout}"
                )
        if PoolName.DRAFT in host_pools and not options.get("draft_revision"):
            raise ValueError("flashkv_dcp.draft_revision is required with a draft pool")

    def __init__(self, hicache_config, extra_config=None):
        self._cfg = hicache_config
        options = _options(hicache_config)
        self._identity = {
            "schema": KEY_SCHEMA,
            "model_name": hicache_config.model_name,
            "model_revision": options["model_revision"],
            "draft_revision": options.get("draft_revision"),
            "namespace": options.get("namespace", ""),
            "tp_size": hicache_config.tp_size,
            "pp_size": hicache_config.pp_size,
            "pp_rank": hicache_config.pp_rank,
            "dcp_size": hicache_config.dcp_size,
        }
        self._pool_prefixes = {}
        self._registered_spans = []
        super().__init__(hicache_config, extra_config)

    def _warmup(self):
        # The vendor takes its namespace from FLASHKV_KEY_NAMESPACE; this
        # deployment's identity is the namespace, so clear() stays scoped to it.
        self.key_namespace = KEY_SCHEMA + "-" + _digest(self._identity)
        super()._warmup()

    # ---- host memory ownership ----

    def register_mem_pool_host(self, mem_pool_host):
        self._pool_prefixes[PoolName.KV] = _digest(
            _pool_descriptor(self._cfg, PoolName.KV, mem_pool_host)
        )
        super().register_mem_pool_host(mem_pool_host)

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        if host_pool_name != PoolName.KV:
            self._pool_prefixes[host_pool_name] = _digest(
                _pool_descriptor(self._cfg, host_pool_name, host_pool)
            )
        super().register_mem_host_pool_v2(host_pool, host_pool_name)

    def _register_buffer(self, buffer):
        super()._register_buffer(buffer)
        self._registered_spans.append(
            (buffer.data_ptr(), buffer.numel() * buffer.element_size())
        )

    def close(self):
        """Unregister host memory; stops at the first failure so a retry resumes there."""
        while self._registered_spans:
            buffers = flashkv.MemBufVector()
            buffers.append(flashkv.MemBuf(*self._registered_spans[-1]))
            status = self.flashkv_client.unregister_memory(buffers)
            if status.code != flashkv.Status.CODE_OK:
                raise RuntimeError(f"[FlashKV] unregister_memory failed: {status}")
            self._registered_spans.pop()

    # ---- object keys ----

    def _component_keys(self, name, components, page_keys):
        prefix = self._pool_prefixes[name]
        keys = [f"{prefix}:{part}:{key}" for key in page_keys for part in components]
        return keys, len(components)

    def _get_page_keys(self, keys):
        return self._component_keys(PoolName.KV, ("kv",), keys)

    def _get_hybrid_page_component_keys(self, page_keys, transfer):
        if transfer.name == PoolName.MAMBA:
            pool = self.registered_pools[PoolName.MAMBA]
            # Same order as MambaPoolHost.get_page_buffer_meta: temporal, then conv.
            components = ["temporal"] if pool.temporal_state_elem_size > 0 else []
            components += [f"conv_{i}" for i in range(len(pool.conv_buffer))]
        else:
            components = ("kv",)
        return self._component_keys(transfer.name, components, page_keys)

    # ---- hybrid transfers ----

    def _batch_io_v2(self, transfers, is_set):
        # DCP host indices span logical pages, not the rank's physical page size.
        # Validate before the vendor's native I/O zips keys, pointers and sizes.
        io = self._batch_put_impl if is_set else self._batch_get_impl
        results = {}
        for transfer in transfers:
            pool = self.registered_pools[transfer.name]
            page_size = getattr(pool, "logical_page_size", pool.page_size)
            if len(transfer.host_indices) != len(transfer.keys) * page_size:
                raise ValueError("Transfer keys must match the logical host page count")
            component_keys, multiplier = self._get_hybrid_page_component_keys(
                transfer.keys, transfer
            )
            ptrs, sizes = pool.get_page_buffer_meta(transfer.host_indices)
            if not (len(component_keys) == len(ptrs) == len(sizes)):
                raise ValueError("Object keys, pointers and byte sizes must match")
            results[transfer.name] = self._batch_io_postprocess(
                transfer.keys, io(component_keys, ptrs, sizes), multiplier
            )
        return results

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        kv_pages = self._query_target_prefix(keys)
        if kv_pages == 0:
            return PoolTransferResult(0, {}, [])
        hits = {PoolName.KV: kv_pages}
        # Every prefix length is restorable until a sidecar rules it out; the
        # controller intersects these sets across ranks, so a trailing-window
        # pool must report each checkpoint rather than its longest one.
        restorable = set(range(1, kv_pages + 1))
        for transfer in pool_transfers or []:
            component_keys, multiplier = self._get_hybrid_page_component_keys(
                keys[:kv_pages], transfer
            )
            mask = self._batch_query_impl(component_keys)
            exists = [
                self._page_components_exist(mask, i, multiplier)
                for i in range(kv_pages)
            ]
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = exists.index(False) if False in exists else kv_pages
                allowed = set(range(1, boundary + 1))
            else:
                window = len(transfer.keys) if transfer.keys else 1
                allowed = {
                    p
                    for p in range(1, kv_pages + 1)
                    if all(exists[max(0, p - window) : p])
                }
            if allowed:
                hits[transfer.name] = max(allowed)
            restorable &= allowed
        ordered = sorted(restorable)
        return PoolTransferResult(ordered[-1] if ordered else 0, hits, ordered)
