"""DCP/TP-safe Python adapter over PAI flashkv 1.0.0+gdf62dbb.

Requires SGLang PR #41187 plus the dynamic capability and EAGLE host-pool
patches. Native memory registration and transport remain the vendor's code.
The capability declaration describes this implementation's contract; it is
not a claim of native ARM/RDMA or inference validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager

import flashkv
from flashkv.integration.sglang.flashkv_storage import FlashKVStorage
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _options(cfg, extra=None):
    # The dynamic factory supplies an empty kwargs dict as the second argument.
    merged = dict(getattr(cfg, "extra_config", None) or {})
    merged.update(extra or {})
    if merged.get("interface_v1") != 1:
        raise ValueError("FlashKV requires interface_v1=1 for zero-copy page I/O")
    options = merged.get("flashkv_dcp")
    if not isinstance(options, dict) or options.get("schema_version") != 1:
        raise ValueError("flashkv_dcp.schema_version=1 is required")
    for field in ("model_revision",):
        if not isinstance(options.get(field), str) or not options[field].strip():
            raise ValueError(
                f"flashkv_dcp.{field} must identify the immutable weights/config"
            )
    if not isinstance(options.get("namespace", ""), str):
        # Framework startup reports invalid configuration through ValueError.
        raise ValueError("flashkv_dcp.namespace must be a string")  # noqa: TRY004
    return dict(options)


def _logical_page_size(pool):
    return getattr(pool, "logical_page_size", None) or pool.page_size


class FlashKVDCPStorage(FlashKVStorage):
    # Controllers must retain this backend on a failed unregister, for retry.
    requires_explicit_memory_cleanup = True

    dcp_storage_capabilities = frozenset(
        {
            "mla_sharded",
            "mamba_tp_sharded",
            "draft_mla_sharded",
            "exact_restorable_prefix_pages",
            "logical_page_indices",
        }
    )

    @classmethod
    def validate_dcp_config(cls, storage_config, host_pools):
        options = _options(storage_config)
        cls._validate_topology(storage_config)
        if os.getenv("FLASHKV_ENABLE_PACKED", "0") != "0":
            raise ValueError("Packed FlashKV objects are not supported by this adapter")
        if PoolName.KV not in host_pools:
            raise ValueError("A materialized KV anchor is required")
        for name, pool in host_pools.items():
            cls._pool_descriptor(storage_config, options, name, pool)
        if PoolName.DRAFT in host_pools and (
            _logical_page_size(host_pools[PoolName.DRAFT])
            != _logical_page_size(host_pools[PoolName.KV])
        ):
            raise ValueError("Draft and target must use the same logical page size")

    @staticmethod
    def _validate_topology(cfg):
        for size, rank in (
            (cfg.tp_size, cfg.tp_rank),
            (cfg.pp_size, cfg.pp_rank),
            (cfg.dcp_size, cfg.dcp_rank),
        ):
            if not isinstance(size, int) or size < 1 or not 0 <= rank < size:
                raise ValueError("Invalid storage parallel topology")
        if cfg.tp_size % cfg.dcp_size or cfg.tp_rank % cfg.dcp_size != cfg.dcp_rank:
            raise ValueError("Only contiguous DCP groups within TP are supported")
        if getattr(cfg, "attn_cp_size", 1) != 1 or getattr(cfg, "attn_cp_rank", 0) != 0:
            raise ValueError(
                "Attention context parallelism is outside the adapter contract"
            )
        if cfg.should_split_heads:
            raise ValueError("Split-head storage is unsupported")
        if cfg.dcp_size > 1 and not cfg.is_mla_model:
            raise ValueError("DCP storage requires an MLA target")

    @classmethod
    def _pool_descriptor(cls, cfg, options, name, pool):
        if name not in (PoolName.KV, PoolName.MAMBA, PoolName.DRAFT):
            raise ValueError(f"Unsupported pool: {name}")
        if pool.layout not in ("page_first", "page_first_direct"):
            raise ValueError(f"Unsupported FlashKV layout: {pool.layout}")
        device_pool = getattr(pool, "device_pool", None)
        if any(
            getattr(obj, attr, None) is not None
            for obj in (pool, device_pool)
            for attr in ("kv_scale_buffer", "k_scale_buffer", "v_scale_buffer")
        ):
            raise ValueError("Separate scale buffers are unsupported")
        if getattr(pool, "mtp_draft_device_pools", ()) or getattr(
            pool, "slot_state_device_tensors", ()
        ):
            raise ValueError("Packed draft or extra slot-state buffers are unsupported")
        logical = _logical_page_size(pool)
        physical = pool.page_size
        if (
            not isinstance(logical, int)
            or logical <= 0
            or not isinstance(physical, int)
            or physical <= 0
        ):
            raise ValueError("Invalid host page geometry")
        descriptor = {
            "pool": str(name),
            "layout": pool.layout,
            "logical_page_size": logical,
            "physical_page_size": physical,
        }
        if name == PoolName.MAMBA:
            if not isinstance(pool, MambaPoolHost) or logical != 1:
                raise ValueError("MAMBA requires a one-slot MambaPoolHost")
            descriptor.update(
                kind="mamba",
                rank=cfg.tp_rank,
                layers=int(pool.num_mamba_layers),
                temporal_shape=list(pool.temporal_state_shape),
                conv_shapes=[list(shape) for shape in pool.conv_state_shapes],
                temporal_dtype=str(pool.temporal_dtype),
                conv_dtype=str(pool.conv_dtype),
            )
            if len(pool.conv_buffer) != len(pool.conv_state_shapes):
                raise ValueError("MAMBA convolution component metadata mismatch")
            if pool.temporal_state_elem_size <= 0 and not pool.conv_buffer:
                raise ValueError("MAMBA pool has no physical state")
            if any(
                dtype not in ("torch.bfloat16", "torch.float16", "torch.float32")
                for dtype in (str(pool.temporal_dtype), str(pool.conv_dtype))
            ):
                raise ValueError("Unsupported MAMBA state dtype")
            return descriptor
        if name == PoolName.DRAFT and (
            not isinstance(options.get("draft_revision"), str)
            or not options["draft_revision"].strip()
        ):
            raise ValueError("DRAFT requires flashkv_dcp.draft_revision")
        if getattr(pool, "kv_buffer", None) is None:
            raise ValueError("A materialized KV tensor is required")
        dtype = str(pool.dtype)
        if dtype not in ("torch.bfloat16", "torch.float16"):
            raise ValueError("Only BF16/FP16 KV pages are supported")
        descriptor.update(dtype=dtype, layers=int(pool.layer_num))
        if getattr(pool, "target_layer_num", pool.layer_num) != pool.layer_num:
            raise ValueError(
                "Packed target/draft layers require a separate adapter schema"
            )
        if isinstance(pool, MLATokenToKVPoolHost):
            if name == PoolName.KV and not cfg.is_mla_model:
                raise ValueError("Target configuration disagrees with MLA pool")
            if logical != physical * cfg.dcp_size:
                raise ValueError("MLA logical/physical page size disagrees with DCP")
            if (
                getattr(pool, "dcp_size", 1) != cfg.dcp_size
                or getattr(pool, "dcp_rank", 0) != cfg.dcp_rank
            ):
                raise ValueError("MLA host-pool DCP topology disagrees with storage")
            if cfg.dcp_size > 1 and logical != cfg.logical_page_size:
                raise ValueError("MLA logical page size disagrees with storage")
            descriptor.update(
                kind="mla", rank=cfg.dcp_rank, kv_cache_dim=int(pool.kv_cache_dim)
            )
        elif isinstance(pool, MHATokenToKVPoolHost):
            if name == PoolName.DRAFT:
                raise ValueError(
                    "MHA draft requires an additional per-TP writer-ownership contract"
                )
            if cfg.dcp_size != 1 or logical != physical:
                raise ValueError("MHA draft/target currently requires DCP1")
            if name == PoolName.KV and cfg.is_mla_model:
                raise ValueError("Target configuration disagrees with MHA pool")
            if getattr(pool, "v_head_dim", pool.head_dim) != pool.head_dim:
                raise ValueError(
                    "Asymmetric MHA K/V buffers are outside this adapter contract"
                )
            descriptor.update(
                kind="mha",
                rank=cfg.tp_rank,
                head_num=int(pool.head_num),
                head_dim=int(pool.head_dim),
                v_head_dim=int(getattr(pool, "v_head_dim", pool.head_dim)),
            )
        else:
            raise ValueError(f"Unsupported KV host pool: {type(pool).__name__}")  # noqa: TRY004
        return descriptor

    def __init__(self, hicache_config, extra_config=None):
        self._lifecycle_condition = threading.Condition(threading.RLock())
        self._registration_lock = threading.RLock()
        self._lifecycle_state = "open"
        self._active_operations = 0
        # Strong references keep successfully registered storage alive until
        # native unregister confirms completion. Key is the exact registered span.
        self._registered_buffers = {}
        self._cfg = hicache_config
        self._options = _options(hicache_config, extra_config)
        self._validate_topology(hicache_config)
        if os.getenv("FLASHKV_ENABLE_PACKED", "0") != "0":
            raise ValueError("Packed FlashKV objects are unsupported")
        self._pool_prefixes = {}
        self._identity = {
            "schema": "apodex-flashkv-dcp-v1",
            "model_revision": self._options["model_revision"],
            "draft_revision": self._options.get("draft_revision"),
            "model_name": hicache_config.model_name,
            "namespace": self._options.get("namespace", ""),
            "tp_size": hicache_config.tp_size,
            "pp_size": hicache_config.pp_size,
            "pp_rank": hicache_config.pp_rank,
            "dcp_size": hicache_config.dcp_size,
        }
        try:
            super().__init__(hicache_config, extra_config)
        except Exception:
            self.close()
            raise

    def _ensure_open(self):
        if self._lifecycle_state != "open":
            raise RuntimeError("FlashKV backend is closing or closed")

    @contextmanager
    def _operation(self):
        # Allow concurrent native I/O, but prevent unregister while a native
        # operation still uses a registered pointer. Normal detach stops the
        # controller's worker threads before calling close.
        with self._lifecycle_condition:
            self._ensure_open()
            self._active_operations += 1
        try:
            yield
        finally:
            with self._lifecycle_condition:
                self._active_operations -= 1
                self._lifecycle_condition.notify_all()

    def _register_buffer(self, buffer):
        with self._registration_lock, self._operation():
            identity = (buffer.data_ptr(), buffer.numel() * buffer.element_size())
            if identity in self._registered_buffers:
                return
            if identity[0] <= 0 or identity[1] <= 0:
                raise ValueError("FlashKV cannot register an empty buffer")
            # The vendor registers exactly one buffer and raises on non-OK.
            # Failed/ambiguous native allocations are outside this registry.
            super()._register_buffer(buffer)
            self._registered_buffers[identity] = buffer

    def _unregister_buffers(self, identities):
        failures = []
        for identity in reversed(list(identities)):
            if identity not in self._registered_buffers:
                continue
            buffers = flashkv.MemBufVector()
            buffers.append(flashkv.MemBuf(*identity))
            try:
                status = self.flashkv_client.unregister_memory(buffers)
                if status.code != flashkv.Status.CODE_OK:
                    failures.append(identity)
                    continue
            except Exception:  # noqa: BLE001 - retain failed native spans for explicit retry
                failures.append(identity)
                continue
            del self._registered_buffers[identity]
        return failures

    def _rollback_registration(self, previous):
        failures = self._unregister_buffers(
            [key for key in self._registered_buffers if key not in previous]
        )
        if failures:
            self._lifecycle_state = "closing"
            raise RuntimeError(
                f"FlashKV registration rollback left {len(failures)} buffers; retry close()"
            )

    def close(self):
        """One bounded cleanup pass; retry only buffers not yet acknowledged.

        This unregisters memory, not the native client's transport threads.
        The supplied client has no documented stop/close API. Failed native
        registration allocations and client shutdown remain process-exit scope.
        """
        with self._lifecycle_condition:
            if self._lifecycle_state == "closed":
                return
            self._lifecycle_state = "closing"
            if not self._lifecycle_condition.wait_for(
                lambda: self._active_operations == 0, timeout=10
            ):
                raise RuntimeError("FlashKV close timed out waiting for native I/O")
            failures = self._unregister_buffers(self._registered_buffers)
            if failures:
                raise RuntimeError(
                    f"FlashKV close left {len(failures)} registered buffers; retry close()"
                )
            self._lifecycle_state = "closed"
            self.mem_pool_host = None
            self.registered_pools = {}

    def _batch_query_impl(self, keys):
        with self._operation():
            return super()._batch_query_impl(keys)

    def _batch_put_impl(self, keys, buffer_ptrs, buffer_sizes):
        with self._operation():
            return super()._batch_put_impl(keys, buffer_ptrs, buffer_sizes)

    def _batch_get_impl(self, keys, buffer_ptrs, buffer_sizes):
        with self._operation():
            return super()._batch_get_impl(keys, buffer_ptrs, buffer_sizes)

    def _batch_put_parts(self, keys, parts_list):
        with self._operation():
            return super()._batch_put_parts(keys, parts_list)

    def _batch_get_parts(self, keys, parts_list):
        with self._operation():
            return super()._batch_get_parts(keys, parts_list)

    def batch_exists(self, keys, extra_info=None):
        self._ensure_open()
        return super().batch_exists(keys, extra_info)

    def _init_key_prefix(self, cfg):
        self.key_prefix = "apodex-flashkv-v1-" + _digest(self._identity)

    def _warmup(self):
        # Override the vendor's unversioned namespace before its first query.
        self.key_namespace = "apodex-flashkv-v1-" + _digest(self._identity)
        super()._warmup()

    def _remember_pool(self, name, pool):
        descriptor = self._pool_descriptor(self._cfg, self._options, name, pool)
        prefix = self.key_prefix + ":" + _digest(descriptor)
        if name in self._pool_prefixes and self._pool_prefixes[name] != prefix:
            raise ValueError("A registered pool cannot change storage format in place")
        self._pool_prefixes[name] = prefix

    def register_mem_pool_host(self, mem_pool_host):
        with self._registration_lock, self._operation():
            previous_prefixes = dict(self._pool_prefixes)
            previous_pool = self.mem_pool_host
            previous_buffers = set(self._registered_buffers)
            self._remember_pool(PoolName.KV, mem_pool_host)
            try:
                super().register_mem_pool_host(mem_pool_host)
            except Exception:
                self._pool_prefixes = previous_prefixes
                self.mem_pool_host = previous_pool
                self._rollback_registration(previous_buffers)
                raise

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        with self._registration_lock, self._operation():
            if host_pool_name == PoolName.KV and host_pool is not self.mem_pool_host:
                raise ValueError(
                    "KV sidecar registration must refer to the registered anchor"
                )
            if host_pool_name == PoolName.DRAFT and (
                self.mem_pool_host is None
                or _logical_page_size(host_pool)
                != _logical_page_size(self.mem_pool_host)
            ):
                raise ValueError(
                    "Draft requires a registered target with matching logical page size"
                )
            previous_prefixes = dict(self._pool_prefixes)
            previous_pools = dict(self.registered_pools)
            previous_buffers = set(self._registered_buffers)
            self._remember_pool(host_pool_name, host_pool)
            try:
                super().register_mem_host_pool_v2(host_pool, host_pool_name)
            except Exception:
                self._pool_prefixes = previous_prefixes
                self.registered_pools = previous_pools
                self._rollback_registration(previous_buffers)
                raise

    def clear(self):
        raise NotImplementedError(
            "Bulk removal is disabled; manage isolated namespaces explicitly"
        )

    def set_draft(self, enabled=True):
        if enabled:
            raise NotImplementedError(
                "Draft restore requires unified ALL_PAGES sidecar transfers"
            )
        self.has_draft = False

    def _component_keys(self, keys, name, components):
        self._ensure_open()
        if name not in self._pool_prefixes:
            raise ValueError(
                f"Pool must be validated and registered before use: {name}"
            )
        if any(not isinstance(k, str) or not k for k in keys):
            raise ValueError("Page keys must be nonempty strings")
        prefix = self._pool_prefixes[name]
        return [
            f"{prefix}:{component}:{len(key)}:{key}"
            for key in keys
            for component in components
        ], len(components)

    def _get_page_keys(self, keys):
        return self._component_keys(
            keys, PoolName.KV, ("kv",) if self.is_mla_model else ("k", "v")
        )

    def _get_hybrid_page_component_keys(self, page_keys, transfer):
        name = transfer.name
        pool = self.registered_pools.get(name)
        if pool is None:
            raise ValueError(f"Unregistered FlashKV pool: {name}")
        if name == PoolName.MAMBA:
            components = ["temporal"] if pool.temporal_state_elem_size > 0 else []
            components += [f"conv_{i}" for i in range(len(pool.conv_buffer))]
        elif name == PoolName.DRAFT:
            components = (
                ["kv"] if isinstance(pool, MLATokenToKVPoolHost) else ["k", "v"]
            )
        else:
            raise ValueError(f"Unsupported FlashKV sidecar: {name}")
        return self._component_keys(page_keys, name, components)

    def _io(self, keys, indices, pool, component_keys, multiplier, is_set):
        self._ensure_open()
        page_size = _logical_page_size(pool)
        if indices is None or len(indices) != len(keys) * page_size:
            raise ValueError("Keys must match the exact logical-page host index count")
        if not keys:
            return []
        ptrs, sizes = pool.get_page_buffer_meta(indices)
        if len(component_keys) != len(ptrs) or len(ptrs) != len(sizes):
            raise ValueError(
                "FlashKV object keys, pointers and byte sizes must match exactly"
            )
        if any(int(ptr) <= 0 or int(size) <= 0 for ptr, size in zip(ptrs, sizes)):
            raise ValueError("FlashKV requires positive pointers and byte sizes")
        fn = self._batch_put_impl if is_set else self._batch_get_impl
        result = fn(component_keys, ptrs, sizes)
        if len(result) != len(component_keys):
            return [False] * len(keys)
        return self._batch_io_postprocess(keys, result, multiplier)

    def _batch_io_v1(self, keys, host_indices, is_set):
        component_keys, multiplier = self._get_page_keys(keys)
        return self._io(
            keys, host_indices, self.mem_pool_host, component_keys, multiplier, is_set
        )

    def _batch_io_v2(self, transfers, is_set):
        self._ensure_open()
        names = [transfer.name for transfer in transfers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate pool transfer")
        results = {}
        for transfer in transfers:
            keys = transfer.keys
            if keys is None:
                raise ValueError("Transfer keys are required")
            component_keys, multiplier = self._get_hybrid_page_component_keys(
                keys, transfer
            )
            results[transfer.name] = self._io(
                keys,
                transfer.host_indices,
                self.registered_pools[transfer.name],
                component_keys,
                multiplier,
                is_set,
            )
        return results

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        self._ensure_open()
        kv_pages = self._query_target_prefix(keys)
        self._query_draft_pages(keys)
        hits = {PoolName.KV: kv_pages} if kv_pages else {}
        restorable = set(range(1, kv_pages + 1))
        seen = set()
        for transfer in pool_transfers or []:
            if transfer.name in seen:
                raise ValueError("Duplicate pool query")
            seen.add(transfer.name)
            if (
                transfer.name == PoolName.DRAFT
                and transfer.hit_policy != PoolHitPolicy.ALL_PAGES
            ):
                raise ValueError("Draft restore requires ALL_PAGES hit policy")
            component_keys, multiplier = self._get_hybrid_page_component_keys(
                keys[:kv_pages], transfer
            )
            mask = self._batch_query_impl(component_keys) if component_keys else []
            exists = [
                self._page_components_exist(mask, i, multiplier)
                for i in range(kv_pages)
            ]
            if transfer.hit_policy == PoolHitPolicy.ALL_PAGES:
                boundary = exists.index(False) if False in exists else kv_pages
                allowed = set(range(1, boundary + 1))
            elif transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                trailing = max(1, len(transfer.keys) if transfer.keys else 1)
                allowed = {
                    p
                    for p in range(1, kv_pages + 1)
                    if all(exists[max(0, p - trailing) : p])
                }
            else:
                raise ValueError(f"Unsupported hit policy: {transfer.hit_policy}")
            if allowed:
                hits[transfer.name] = max(allowed)
            restorable.intersection_update(allowed)
        ordered = sorted(restorable)
        return PoolTransferResult(ordered[-1] if ordered else 0, hits, ordered)
