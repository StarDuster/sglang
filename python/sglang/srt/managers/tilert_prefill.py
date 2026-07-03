"""SGLang-prefill -> TileRT-decode bridge for GLM-5 (DSA/MLA models).

TileRT's megakernel processes prompts in 4-token decode-style steps, so its
input throughput is a few hundred tokens/s. This module lets a nested SGLang
engine run the prefill at normal speed, then hands the per-layer (ki, kv, pe)
caches to TileRT via ``GLM5Generator.start_sequence_from_cache``.

Two halves:

* ``stage_prefill_kv`` runs inside the prefill engine's rank-0 scheduler
  process (invoked through the generic scheduler RPC). It locates the prompt's
  KV slots in the radix cache, gathers per-layer tensors from the
  ``DSATokenToKVPool`` (dequantizing the FP8 indexer keys), and stages them to
  a file the TileRT process can read.

* ``TileRTPrefillWorker`` runs inside the TileRT scheduler process. It owns a
  nested ``sglang.Engine`` (prefill-only: radix cache on, CUDA graphs off) on
  a dedicated thread, and exposes a blocking ``prefill()`` that returns
  ``(cached_len, layer_caches)`` ready for injection.

The hand-off returns the full main-model prompt KV when it is present in
SGLang's radix cache. TileRT's MTP draft-layer cache is not transferred, so
early draft acceptance can be lower, but main-model verification still has the
complete prompt context.
"""

from __future__ import annotations

import logging
import os
import queue
import tempfile
import threading
import uuid
from array import array
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

LayerCaches = List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def stage_prefill_kv(scheduler: Scheduler, token_ids: List[int], out_path: str) -> None:
    """Gather (ki, kv, pe) for the cached prefix of ``token_ids`` and stage to disk.

    Runs inside the prefill engine's scheduler process (all ranks receive the
    RPC; only attention-TP rank 0 does the work since the MLA latent cache is
    replicated across ranks).

    The staged file contains::

        {"cached_len": int,
         "ki": bf16 [n_layers, cached_len, 128],
         "kv": bf16 [n_layers, cached_len, 512],
         "pe": bf16 [n_layers, cached_len, 64]}
    """
    if scheduler.ps.tp_rank != 0:
        return

    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.mem_cache.radix_cache import RadixKey

    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if not isinstance(pool, DSATokenToKVPool):
        raise RuntimeError(
            f"TileRT prefill staging requires a DSATokenToKVPool, got {type(pool).__name__}"
        )
    if pool.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            "TileRT prefill staging requires a bf16 KV cache in the prefill "
            f"engine (got {pool.dtype}); do not pass an fp8 --kv-cache-dtype"
        )

    match_result = scheduler.tree_cache.match_prefix(
        MatchPrefixParams(key=RadixKey(token_ids=array("q", token_ids)))
    )
    slots = match_result.device_indices
    cached_len = int(slots.numel())
    if cached_len == 0:
        raise RuntimeError(
            "Prompt not found in the prefill engine's radix cache; "
            "was the prefill request evicted before staging?"
        )

    page_size = pool.page_size
    slots = slots.to(torch.int64)
    page_starts = slots[::page_size]
    if not bool((page_starts % page_size == 0).all()):
        raise RuntimeError("Matched KV slots are not page-aligned")
    page_indices = (page_starts // page_size).to(torch.int32)

    n_layers = pool.layer_num
    ki_all = torch.empty(n_layers, cached_len, 128, dtype=torch.bfloat16)
    kv_all = torch.empty(n_layers, cached_len, pool.kv_lora_rank, dtype=torch.bfloat16)
    pe_all = torch.empty(
        n_layers, cached_len, pool.qk_rope_head_dim, dtype=torch.bfloat16
    )

    for i, layer_id in enumerate(range(pool.start_layer, pool.start_layer + n_layers)):
        nope, rope = pool.get_mla_kv_buffer(
            SimpleNamespace(layer_id=layer_id), slots, dst_dtype=torch.bfloat16
        )
        kv_all[i].copy_(nope.squeeze(1))
        pe_all[i].copy_(rope.squeeze(1))

        k_fp8 = pool.get_index_k_continuous(layer_id, cached_len, page_indices)
        k_scale = pool.get_index_k_scale_continuous(layer_id, cached_len, page_indices)
        ki = k_fp8.view(torch.float8_e4m3fn).to(
            torch.float32
        ) * k_scale.contiguous().view(torch.float32)
        ki_all[i].copy_(ki.to(torch.bfloat16))

    tmp_path = f"{out_path}.tmp"
    torch.save(
        {"cached_len": cached_len, "ki": ki_all, "kv": kv_all, "pe": pe_all},
        tmp_path,
    )
    os.rename(tmp_path, out_path)
    logger.info(
        "Staged TileRT prefill KV: %s tokens x %s layers -> %s",
        cached_len,
        n_layers,
        out_path,
    )


class TileRTPrefillWorker:
    """Owns a nested SGLang prefill engine on a dedicated thread.

    All Engine interactions happen on one long-lived thread because the
    Engine's internal asyncio loop is bound to the thread that created it,
    while the TileRT scheduler calls prefill() from per-request worker
    threads.
    """

    def __init__(self, server_args: ServerArgs) -> None:
        self.server_args = server_args
        staging_dir = server_args.tilert_prefill_staging_dir
        if staging_dir is None:
            staging_dir = (
                "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
            )
        self.staging_dir = staging_dir

        self._jobs: queue.Queue[Optional[tuple]] = queue.Queue()
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._engine_thread, name="tilert-prefill-engine", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._init_error is not None:
            raise RuntimeError(
                "Failed to launch TileRT prefill engine"
            ) from self._init_error

    def _engine_thread(self) -> None:
        try:
            # Deferred import: engine -> scheduler -> this module would cycle
            # at import time.
            from sglang.srt.entrypoints.engine import Engine

            prefill_model_path = (
                self.server_args.tilert_prefill_model_path
                or self.server_args.model_path
            )
            self._engine = Engine(
                model_path=prefill_model_path,
                trust_remote_code=self.server_args.trust_remote_code,
                tp_size=self.server_args.tilert_prefill_tp_size,
                mem_fraction_static=self.server_args.tilert_prefill_mem_fraction,
                context_length=self.server_args.context_length,
                kv_cache_dtype="bfloat16",
                disable_cuda_graph=True,
                log_level="warning",
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced to caller
            self._init_error = exc
            self._ready.set()
            return

        self._ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                self._engine.shutdown()
                return
            token_ids, result_box, done = job
            try:
                result_box["result"] = self._prefill_on_thread(token_ids)
            except BaseException as exc:  # noqa: BLE001 - surfaced to caller
                result_box["error"] = exc
            finally:
                done.set()

    def _prefill_on_thread(self, token_ids: List[int]) -> Tuple[int, LayerCaches]:
        self._engine.generate(
            input_ids=token_ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )

        out_path = os.path.join(self.staging_dir, f"tilert_kv_{uuid.uuid4().hex}.pt")
        try:
            self._engine.collective_rpc(
                "tilert_stage_prefill_kv", token_ids=token_ids, out_path=out_path
            )
            data = torch.load(out_path, weights_only=True)
        finally:
            for path in (out_path, f"{out_path}.tmp"):
                if os.path.exists(path):
                    os.unlink(path)

        cached_len = int(data["cached_len"])
        layer_caches = [
            (data["ki"][i], data["kv"][i], data["pe"][i])
            for i in range(data["ki"].shape[0])
        ]
        return cached_len, layer_caches

    def prefill(self, token_ids: List[int]) -> Tuple[int, LayerCaches]:
        """Run SGLang prefill for ``token_ids`` and return (cached_len, layer_caches).

        ``cached_len`` is capped at the prompt length because
        GLM5Generator.start_sequence_from_cache treats it as the number of
        main-model KV rows already populated, not as the runtime cur_pos.
        """
        result_box: dict[str, Any] = {}
        done = threading.Event()
        self._jobs.put((token_ids, result_box, done))
        done.wait()
        if "error" in result_box:
            raise result_box["error"]
        cached_len, layer_caches = result_box["result"]
        return min(cached_len, len(token_ids)), layer_caches

    def shutdown(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=30.0)
