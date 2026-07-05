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
  nested ``sglang.Engine`` (prefill-only: radix cache on) on
  a dedicated thread, and exposes a blocking ``prefill()`` that returns
  ``(cached_len, layer_caches, last_hidden_state)`` ready for injection.

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
import time
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
PrefillResult = Tuple[int, LayerCaches, Optional[torch.Tensor]]
_PREFILL_WARMUP_TOKENS = 32768


def _page_align_tokens(num_tokens: int) -> int:
    return num_tokens // 64 * 64


def _external_prefill_warmup_len(min_tokens: int, max_prefill_tokens: int) -> int:
    cap = _page_align_tokens(max_prefill_tokens)
    if cap <= 0:
        return 0
    return min(_page_align_tokens(max(min_tokens, _PREFILL_WARMUP_TOKENS)), cap)


def _build_prefill_engine_kwargs(
    server_args: ServerArgs, prefill_model_path: str
) -> dict:
    return {
        "model_path": prefill_model_path,
        "trust_remote_code": server_args.trust_remote_code,
        "tp_size": server_args.tilert_prefill_tp_size,
        "mem_fraction_static": server_args.tilert_prefill_mem_fraction,
        "context_length": server_args.context_length,
        "quantization": "fp8",
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "dsa",
        "dsa_prefill_backend": "trtllm",
        "dsa_decode_backend": "trtllm",
        "enable_dsa_prefill_context_parallel": True,
        "dsa_prefill_cp_mode": "round-robin-split",
        "attn_cp_size": server_args.tilert_prefill_tp_size,
        "skip_server_warmup": True,
        "disable_cuda_graph": True,
        "log_level": "warning",
    }


def _collect_prefill_kv(
    scheduler: Scheduler, token_ids: List[int], *, device: str | torch.device
) -> Tuple[int, LayerCaches, float]:
    """Gather (ki, kv, pe) for the cached prefix of ``token_ids``.

    Runs inside the prefill engine's rank-0 scheduler process. ``device="cpu"``
    is the compatibility file-staging path; ``device="cuda"`` is the hot path
    used with PyTorch CUDA IPC.
    """
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.mem_cache.radix_cache import RadixKey

    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if not isinstance(pool, DSATokenToKVPool):
        raise RuntimeError(
            f"TileRT prefill staging requires a DSATokenToKVPool, got {type(pool).__name__}"
        )
    if pool.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        raise RuntimeError(
            "TileRT prefill staging requires a bf16/fp16/fp8 KV cache in the prefill "
            f"engine (got {pool.dtype})"
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
    layer_caches: LayerCaches = []

    gather_start = time.monotonic()
    for layer_id in range(pool.start_layer, pool.start_layer + n_layers):
        nope, rope = pool.get_mla_kv_buffer(
            SimpleNamespace(layer_id=layer_id), slots, dst_dtype=torch.bfloat16
        )
        kv = nope.squeeze(1).to(device=device, non_blocking=True)
        pe = rope.squeeze(1).to(device=device, non_blocking=True)

        k_fp8 = pool.get_index_k_continuous(layer_id, cached_len, page_indices)
        k_scale = pool.get_index_k_scale_continuous(layer_id, cached_len, page_indices)
        ki = k_fp8.view(torch.float8_e4m3fn).to(
            torch.float32
        ) * k_scale.contiguous().view(torch.float32)
        ki = ki.to(torch.bfloat16)
        # TileRT's ki cache holds Hadamard-rotated indexer keys. SGLang's
        # fused indexer path stores them unrotated (the rotation is
        # logit-preserving and dropped inside the fused kernel), so rotate
        # here to match; the non-fused path already stores rotated keys.
        if dsa_indexer._use_dsa_indexer_fusion:
            ki = dsa_indexer.rotate_activation(ki)
        ki = ki.to(device=device, non_blocking=True)
        layer_caches.append((ki, kv, pe))

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    gather_elapsed = time.monotonic() - gather_start
    return cached_len, layer_caches, gather_elapsed


def export_prefill_kv(
    scheduler: Scheduler, token_ids: List[int], rid: str | None = None
):
    """Return CUDA IPC-serializable prefill KV tensors for TileRT injection."""
    if scheduler.ps.tp_rank != 0:
        return None

    cached_len, layer_caches, gather_elapsed = _collect_prefill_kv(
        scheduler, token_ids, device="cuda"
    )
    last_hidden_state = None
    if rid is not None:
        last_hidden_state = getattr(
            scheduler, "_tilert_prefill_last_hidden_states", {}
        ).pop(rid, None)
    logger.warning(
        "Exported TileRT prefill KV: tokens=%s layers=%s gather=%.3fs transport=cuda_ipc last_hidden=%s",
        cached_len,
        len(layer_caches),
        gather_elapsed,
        last_hidden_state is not None,
    )
    return cached_len, layer_caches, last_hidden_state, gather_elapsed


def stage_prefill_kv(scheduler: Scheduler, token_ids: List[int], out_path: str) -> None:
    """Gather (ki, kv, pe) for the cached prefix of ``token_ids`` and stage to disk.

    The staged file contains::

        {"cached_len": int,
         "ki": bf16 [n_layers, cached_len, 128],
         "kv": bf16 [n_layers, cached_len, 512],
         "pe": bf16 [n_layers, cached_len, 64]}
    """
    if scheduler.ps.tp_rank != 0:
        deadline = time.monotonic() + 600
        while not os.path.exists(out_path):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for TileRT prefill KV staging at {out_path}"
                )
            time.sleep(0.01)
        return

    cached_len, layer_caches, gather_elapsed = _collect_prefill_kv(
        scheduler, token_ids, device="cpu"
    )
    ki_all = torch.stack([ki for ki, _, _ in layer_caches])
    kv_all = torch.stack([kv for _, kv, _ in layer_caches])
    pe_all = torch.stack([pe for _, _, pe in layer_caches])

    tmp_path = f"{out_path}.tmp"
    save_start = time.monotonic()
    torch.save(
        {"cached_len": cached_len, "ki": ki_all, "kv": kv_all, "pe": pe_all},
        tmp_path,
    )
    os.rename(tmp_path, out_path)
    save_elapsed = time.monotonic() - save_start
    logger.warning(
        "Staged TileRT prefill KV: tokens=%s layers=%s gather=%.3fs save=%.3fs path=%s",
        cached_len,
        len(layer_caches),
        gather_elapsed,
        save_elapsed,
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
        self._broken = False
        self._init_error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._engine_thread, name="tilert-prefill-engine", daemon=True
        )
        self._thread.start()

    def _ensure_ready(self, cancel: Optional[threading.Event] = None) -> None:
        while not self._ready.wait(timeout=1.0):
            if cancel is not None and cancel.is_set():
                raise RuntimeError("TileRT prefill engine initialization cancelled")
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
            old_memory_check = os.environ.get("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK")
            os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = "false"
            try:
                self._engine = Engine(
                    **_build_prefill_engine_kwargs(self.server_args, prefill_model_path)
                )
            finally:
                if old_memory_check is None:
                    os.environ.pop("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", None)
                else:
                    os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"] = (
                        old_memory_check
                    )
        except BaseException as exc:  # noqa: BLE001 - surfaced to caller
            self._init_error = exc
            self._ready.set()
            return

        # The nested Engine downgrades this process's sglang logging to its
        # own log_level (warning); restore INFO so the TileRT scheduler's
        # request-lifecycle logs stay visible.
        logging.getLogger("sglang").setLevel(logging.INFO)

        # KV-pool ceiling of the nested engine. max_req_input_len alone is
        # not schedulable headroom: a request also needs reserved decode
        # slots, and one that never fits waits in the queue forever. Prompts
        # beyond the cap are externally prefilled up to it and finished by
        # TileRT's internal prefill (partial injection).
        scheduler_info = self._engine._scheduler_init_result.scheduler_infos[0]
        max_total = int(scheduler_info.get("max_total_num_tokens", 0) or 0)
        max_input = int(
            getattr(self._engine.tokenizer_manager, "max_req_input_len", 0) or 0
        )
        self.max_prefill_tokens = max(0, min(max_input, max_total - 1024))
        logger.warning(
            "TileRT prefill engine ready. max_total_num_tokens=%s "
            "max_req_input_len=%s -> external prefill cap=%s tokens "
            "(longer prompts are partially injected)",
            max_total,
            max_input,
            self.max_prefill_tokens,
        )
        self._ready.set()
        while True:
            job = self._jobs.get()
            if job is None:
                self._engine.shutdown()
                return
            token_ids, rid, result_box, done = job
            try:
                result_box["result"] = self._prefill_on_thread(token_ids, rid)
            except BaseException as exc:  # noqa: BLE001 - surfaced to caller
                result_box["error"] = exc
            finally:
                done.set()

    def _warmup_external_prefill(self) -> None:
        warmup_len = _external_prefill_warmup_len(
            self.server_args.tilert_prefill_min_tokens, self.max_prefill_tokens
        )
        if warmup_len <= 0:
            return

        warmup_start = time.monotonic()
        cached_len, layer_caches, _ = self._prefill_on_thread(
            [1] * warmup_len, "__tilert_external_prefill_warmup__"
        )
        del layer_caches
        self._engine.flush_cache()
        logger.warning(
            "TileRT external prefill warmup done. tokens=%s cached=%s elapsed=%.2fs",
            warmup_len,
            cached_len,
            time.monotonic() - warmup_start,
        )

    def _prefill_on_thread(self, token_ids: List[int], rid: str) -> PrefillResult:
        if 0 < self.max_prefill_tokens < len(token_ids):
            # Page-align so the radix match covers the whole capped prefix.
            cap = self.max_prefill_tokens // 64 * 64
            logger.warning(
                "Prompt (%s tokens) exceeds prefill engine capacity (%s); "
                "externally prefilling first %s tokens",
                len(token_ids),
                self.max_prefill_tokens,
                cap,
            )
            token_ids = token_ids[:cap]
        generate_start = time.monotonic()
        self._engine.generate(
            input_ids=token_ids,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            rid=rid,
        )
        generate_elapsed = time.monotonic() - generate_start

        try:
            export_start = time.monotonic()
            result = self._engine.collective_rpc_with_result(
                "tilert_export_prefill_kv", token_ids=token_ids, rid=rid
            )
            export_elapsed = time.monotonic() - export_start
            if result is None:
                raise RuntimeError("TileRT prefill KV export returned no payload")
            cached_len, layer_caches, last_hidden_state, gather_elapsed = result
            logger.warning(
                "TileRT external prefill timings: tokens=%s cached=%s generate=%.3fs gather=%.3fs ipc=%.3fs load=0.000s last_hidden=%s",
                len(token_ids),
                cached_len,
                generate_elapsed,
                gather_elapsed,
                export_elapsed,
                last_hidden_state is not None,
            )
            return int(cached_len), layer_caches, last_hidden_state
        except Exception:
            logger.exception(
                "CUDA IPC TileRT prefill export failed; falling back to file staging"
            )

        out_path = os.path.join(self.staging_dir, f"tilert_kv_{uuid.uuid4().hex}.pt")
        try:
            stage_start = time.monotonic()
            self._engine.collective_rpc(
                "tilert_stage_prefill_kv", token_ids=token_ids, out_path=out_path
            )
            stage_elapsed = time.monotonic() - stage_start
            load_start = time.monotonic()
            data = torch.load(out_path, map_location="cpu", weights_only=True)
            load_elapsed = time.monotonic() - load_start
        finally:
            for path in (out_path, f"{out_path}.tmp"):
                if os.path.exists(path):
                    os.unlink(path)

        cached_len = int(data["cached_len"])
        layer_caches = [
            (data["ki"][i], data["kv"][i], data["pe"][i])
            for i in range(data["ki"].shape[0])
        ]
        logger.warning(
            "TileRT external prefill timings: tokens=%s cached=%s generate=%.3fs stage_rpc=%.3fs load=%.3fs",
            len(token_ids),
            cached_len,
            generate_elapsed,
            stage_elapsed,
            load_elapsed,
        )
        return cached_len, layer_caches, None

    def prefill(
        self,
        token_ids: List[int],
        cancel: Optional[threading.Event] = None,
    ) -> PrefillResult:
        """Run SGLang prefill for ``token_ids`` and return cache tensors.

        Bounded: on timeout or ``cancel``, the nested request is aborted and
        this raises so the caller falls back to internal prefill. A nested
        engine that fails to honor the abort marks the worker broken (all
        later calls fail fast) instead of wedging the single-flight
        scheduler forever.

        ``cached_len`` is capped at the prompt length because
        GLM5Generator.start_sequence_from_cache treats it as the number of
        main-model KV rows already populated, not as the runtime cur_pos.
        """
        if self._broken:
            raise RuntimeError(
                "TileRT prefill engine is marked broken after an unrecoverable "
                "timeout; restart the server to re-enable external prefill"
            )
        self._ensure_ready(cancel)

        rid = f"tilert-prefill-{uuid.uuid4().hex}"
        result_box: dict[str, Any] = {}
        done = threading.Event()
        self._jobs.put((token_ids, rid, result_box, done))

        # Generous ceiling: staging is O(seconds) and SGLang prefill runs at
        # thousands of tokens/s, so 5ms/token + fixed slack is far above any
        # healthy run (a 115k prompt gets ~13 minutes).
        deadline = time.monotonic() + 180.0 + 0.005 * len(token_ids)
        aborted = False
        while not done.wait(timeout=1.0):
            cancelled = cancel is not None and cancel.is_set()
            if not aborted and (cancelled or time.monotonic() > deadline):
                reason = "cancelled" if cancelled else "timed out"
                logger.warning("External prefill %s; aborting rid=%s", reason, rid)
                self._engine.tokenizer_manager.abort_request(rid=rid)
                aborted = True
                abort_deadline = time.monotonic() + 30.0
            if aborted and time.monotonic() > abort_deadline:
                self._broken = True
                raise RuntimeError(
                    "TileRT prefill engine did not honor abort; worker disabled"
                )

        if aborted and "error" not in result_box and "result" not in result_box:
            raise RuntimeError("External prefill aborted")
        if "error" in result_box:
            raise result_box["error"]
        cached_len, layer_caches, last_hidden_state = result_box["result"]
        return min(cached_len, len(token_ids)), layer_caches, last_hidden_state

    def shutdown(self) -> None:
        self._jobs.put(None)
        self._thread.join(timeout=30.0)
