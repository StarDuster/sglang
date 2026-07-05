from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from array import array
from collections import deque
from http import HTTPStatus
from typing import Any, Deque, Optional

import zmq

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchTokenizedGenerateReqInput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    HealthCheckOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ShutdownReq,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    sock_recv,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.tilert_utils import (
    cached_len_after_external_prefill_tail,
    required_external_prefill_error,
)
from sglang.srt.managers.utils import is_health_check_generate_req
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.hf_transformers import get_tokenizer

logger = logging.getLogger(__name__)


class _NoopPrefixCache:
    cache_controller = None


class TileRTScheduler:
    """A thin SGLang scheduler facade around TileRT's native generator."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ) -> None:
        self.server_args = server_args
        self.port_args = port_args
        self.gpu_id = gpu_id
        self.gracefully_exit = False
        self.disaggregation_mode = DisaggregationMode.NULL
        self.spec_algorithm = SpeculativeAlgorithm.NONE

        self._validate_rank_args(
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=pp_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
        )

        self.ps = ParallelState(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            dp_rank=0,
            dp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            attn_dp_rank=0,
            attn_dp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            moe_dp_rank=0,
            moe_dp_size=1,
            gpu_id=gpu_id,
        )

        self.model_config = server_args.get_model_config()
        self.max_total_num_tokens = self._context_len()
        self.max_req_input_len = self.max_total_num_tokens - 1
        self.vocab_size = self.model_config.vocab_size

        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            tokenizer_backend=server_args.tokenizer_backend,
        )
        if self.vocab_size is None:
            self.vocab_size = len(self.tokenizer)

        self.ipc_channels = SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=True,
            skip_tokenizer_init=server_args.skip_tokenizer_init,
            metrics_enabled=False,
            enable_scripted_runtime=getattr(
                server_args, "enable_scripted_runtime", False
            ),
        )
        self.output_streamer = SchedulerOutputStreamer(
            send_to_detokenizer=self.ipc_channels.send_to_detokenizer,
            tree_cache=_NoopPrefixCache(),
            ps=self.ps,
            server_args=server_args,
            is_generation=True,
            spec_algorithm=self.spec_algorithm,
            disaggregation_mode=self.disaggregation_mode,
            enable_hicache_storage=lambda: False,
        )

        self.generator = self._load_generator()
        self.prefill_worker = None
        if server_args.tilert_external_prefill:
            from sglang.srt.managers.tilert_prefill import TileRTPrefillWorker

            self.prefill_worker = TileRTPrefillWorker(server_args)
            logger.info(
                "TileRT external prefill engine launching. tp=%s mem_fraction=%s "
                "max_total_tokens=%s",
                server_args.tilert_prefill_tp_size,
                server_args.tilert_prefill_mem_fraction,
                server_args.tilert_prefill_max_total_tokens,
            )
        self.stop_token_ids = set(getattr(self.generator, "stop_token_ids", set()))
        self.waiting_queue: Deque[Req] = deque()
        self.active_req: Optional[Req] = None
        self._active_cancel: Optional[threading.Event] = None
        self._active_final_sent = False
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_events: queue.Queue[tuple[str, str, Any]] = queue.Queue()

    def _validate_rank_args(
        self,
        *,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ) -> None:
        if any((tp_rank, moe_ep_rank, pp_rank, attn_cp_rank, moe_dp_rank)):
            raise ValueError("TileRT scheduler supports only rank 0 in every dimension")
        if dp_rank not in (None, 0):
            raise ValueError("TileRT scheduler supports only dp rank 0")

    def _context_len(self) -> int:
        configured = self.server_args.context_length
        if configured is not None:
            return configured
        return int(getattr(self.model_config, "context_len", 0) or 0)

    def _load_generator(self) -> Any:
        tilert_repo_path = self.server_args.tilert_repo_path
        if tilert_repo_path is not None:
            repo_path = os.path.abspath(tilert_repo_path)
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        import tilert
        from tilert.models.glm_5.generator import GLM5Generator
        from tilert.models.glm_5.model_args import ModelArgsGLM5

        model_weights_dir = (
            self.server_args.tilert_model_weights_dir or self.server_args.model_path
        )
        tilert.load_backend(self.server_args.tilert_model_type)
        model_args = ModelArgsGLM5()
        model_args.index_topk = int(self.server_args.tilert_index_topk)
        if self.max_total_num_tokens > 0:
            model_args.max_seq_len = min(
                int(model_args.max_seq_len), int(self.max_total_num_tokens)
            )
        generator = GLM5Generator(
            model_args=model_args,
            max_new_tokens=1,
            temperature=1.0,
            model_weights_dir=model_weights_dir,
            with_mtp=not self.server_args.tilert_disable_mtp,
            top_p=1.0,
            top_k=1,
            enable_thinking=self.server_args.tilert_enable_thinking,
            sampling_seed=self.server_args.random_seed,
            mtp_cache_mode=self.server_args.tilert_mtp_cache_mode,
        )
        generator.from_pretrained()
        tilert_max_seq_len = int(generator.config.max_seq_len)
        if self.max_total_num_tokens <= 0:
            self.max_total_num_tokens = tilert_max_seq_len
        else:
            self.max_total_num_tokens = min(
                self.max_total_num_tokens, tilert_max_seq_len
            )
        self.max_req_input_len = self.max_total_num_tokens - 1
        return generator

    def get_init_info(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
        }

    def release_host_resources(self) -> None:
        if self._active_cancel is not None:
            self._active_cancel.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
        self.active_req = None
        if self.prefill_worker is not None:
            self.prefill_worker.shutdown()
        cleanup = getattr(self.generator, "cleanup", None)
        if cleanup is not None:
            cleanup()

    def run_event_loop(self) -> None:
        while not self.gracefully_exit:
            made_progress = self._drain_inputs()
            made_progress = self._drain_worker_events() or made_progress
            if self.active_req is None:
                made_progress = self._start_next_request() or made_progress

            if not made_progress:
                time.sleep(0.001)

    def _drain_inputs(self) -> bool:
        made_progress = False
        while True:
            try:
                recv_req = sock_recv(
                    self.ipc_channels.recv_from_tokenizer, flags=zmq.NOBLOCK
                )
            except zmq.Again:
                break

            made_progress = True
            output = self._dispatch_request(recv_req)
            if output is not None:
                self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)

        return made_progress

    def _dispatch_request(self, recv_req: Any) -> Any:
        if isinstance(recv_req, TokenizedGenerateReqInput):
            self._handle_generate_request(recv_req)
            return None
        if isinstance(recv_req, BatchTokenizedGenerateReqInput):
            for req in recv_req:
                self._handle_generate_request(req)
            return None
        if isinstance(recv_req, AbortReq):
            self._abort_request(recv_req)
            return None
        if isinstance(recv_req, FlushCacheReqInput):
            return self._flush_cache(recv_req)
        if isinstance(recv_req, OpenSessionReqInput):
            return OpenSessionReqOutput(
                session_id=None,
                success=False,
            )
        if isinstance(recv_req, CloseSessionReqInput):
            return None
        if isinstance(recv_req, ProfileReq):
            return ProfileReqOutput(
                success=False,
                message="TileRT scheduler does not support SGLang profiling.",
            )
        if isinstance(recv_req, UpdateWeightFromDiskReqInput):
            return UpdateWeightFromDiskReqOutput(
                success=False,
                message="TileRT scheduler does not support runtime weight updates.",
            )
        if isinstance(recv_req, ConfigureLoggingReq):
            self._configure_logging(recv_req)
            self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)
            return None
        if isinstance(recv_req, ShutdownReq):
            self.gracefully_exit = True
            return None

        return AbortReq(
            rid=getattr(recv_req, "rid", ""),
            abort_message=f"TileRT scheduler does not support {type(recv_req).__name__}.",
        )

    def _handle_generate_request(self, recv_req: TokenizedGenerateReqInput) -> None:
        if is_health_check_generate_req(recv_req) and (
            self.active_req is not None or self.waiting_queue
        ):
            self.ipc_channels.send_to_tokenizer.send_output(
                HealthCheckOutput(), recv_req
            )
            return

        req = self._make_req(recv_req)
        error_msg = self._validate_generate_request(recv_req, req)
        if error_msg is not None:
            self._finish_with_abort(req, error_msg, HTTPStatus.BAD_REQUEST)
            return

        self._cap_request_output(req)
        if len(req.origin_input_ids) > self.max_req_input_len:
            self._finish_with_abort(
                req,
                f"Input length {len(req.origin_input_ids)} exceeds TileRT max input length {self.max_req_input_len}.",
                HTTPStatus.BAD_REQUEST,
            )
            return

        if req.sampling_params.max_new_tokens == 0:
            req.finished_reason = FINISH_LENGTH(length=0)
            req.finished_len = 0
            self.output_streamer.stream_output([req], return_logprob=False)
            return

        logger.info(
            "Queue TileRT request. rid=%s input_tokens=%s max_new_tokens=%s stream=%s",
            req.rid,
            len(req.origin_input_ids),
            req.sampling_params.max_new_tokens,
            req.stream,
        )
        self.waiting_queue.append(req)

    def _make_req(self, recv_req: TokenizedGenerateReqInput) -> Req:
        input_ids = recv_req.input_ids
        if input_ids is None:
            input_ids = array("q", [0])

        req = Req(
            recv_req.rid,
            recv_req.input_text,
            input_ids,
            recv_req.sampling_params,
            return_logprob=False,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=recv_req.stream,
            eos_token_ids=self.stop_token_ids,
            vocab_size=self.vocab_size,
            priority=recv_req.priority,
            routing_key=recv_req.routing_key,
            extra_key=recv_req.extra_key,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
        )
        req.tokenizer = self.tokenizer
        return req

    def _validate_generate_request(
        self, recv_req: TokenizedGenerateReqInput, req: Req
    ) -> Optional[str]:
        if recv_req.input_ids is None or len(recv_req.input_ids) == 0:
            return "TileRT scheduler requires non-empty input_ids."
        if recv_req.input_embeds is not None:
            return "TileRT scheduler does not support input_embeds."
        if recv_req.mm_inputs is not None:
            return "TileRT scheduler does not support multimodal inputs."
        if recv_req.session_id is not None or recv_req.session_params is not None:
            return "TileRT scheduler does not support sessions."
        if recv_req.lora_id is not None:
            return "TileRT scheduler does not support LoRA."
        if recv_req.custom_logit_processor is not None:
            return "TileRT scheduler does not support custom logit processors."
        if recv_req.positional_embed_overrides is not None:
            return "TileRT scheduler does not support positional embed overrides."
        if recv_req.return_logprob:
            return "TileRT scheduler does not support logprobs."
        if (
            recv_req.return_hidden_states
            or recv_req.return_routed_experts
            or recv_req.return_indexer_topk
        ):
            return "TileRT scheduler does not support hidden-state or expert outputs."

        params = req.sampling_params
        if params.n != 1:
            return "TileRT scheduler supports only n=1."
        if params.min_new_tokens:
            return "TileRT scheduler does not support min_new_tokens."
        if params.ignore_eos:
            return "TileRT scheduler does not support ignore_eos."
        if params.frequency_penalty or params.presence_penalty:
            return "TileRT scheduler does not support frequency/presence penalties."
        if params.repetition_penalty != 1.0:
            return "TileRT scheduler does not support repetition_penalty."
        if params.min_p:
            return "TileRT scheduler does not support min_p."
        if params.logit_bias is not None:
            return "TileRT scheduler does not support logit_bias."
        if (
            params.json_schema is not None
            or params.regex is not None
            or params.ebnf is not None
            or params.structural_tag is not None
        ):
            logger.warning(
                "TileRT scheduler ignores structured generation constraints. "
                "Tool calls are parsed from model-native output only."
            )
        if params.custom_params is not None:
            return "TileRT scheduler does not support custom sampling params."

        return None

    def _cap_request_output(self, req: Req) -> None:
        input_len = len(req.origin_input_ids)
        requested = req.sampling_params.max_new_tokens
        if requested is None:
            requested = 1 << 30
        req.sampling_params.max_new_tokens = max(
            0, min(int(requested), self.max_total_num_tokens - input_len)
        )

    def _start_next_request(self) -> bool:
        if not self.waiting_queue:
            return False

        req = self.waiting_queue.popleft()
        cancel = threading.Event()
        self.active_req = req
        self._active_cancel = cancel
        self._active_final_sent = False
        self._worker_thread = threading.Thread(
            target=self._run_worker_request,
            args=(
                req.rid,
                list(req.origin_input_ids),
                req.sampling_params,
                cancel,
            ),
            name=f"tilert-worker-{req.rid}",
            daemon=True,
        )
        self._worker_thread.start()
        return True

    def _start_generator_sequence(
        self,
        rid: str,
        prompt_tokens: list[int],
        sampling_params: Any,
        cancel: threading.Event | None = None,
    ) -> None:
        """Start the TileRT sequence, preferring SGLang external prefill."""
        with_mtp = not self.server_args.tilert_disable_mtp
        require_external_prefill = self.server_args.tilert_require_external_prefill
        if (
            self.prefill_worker is not None
            and len(prompt_tokens) >= self.server_args.tilert_prefill_min_tokens
        ):
            try:
                if require_external_prefill:
                    self.prefill_worker.ensure_full_prompt_capacity(
                        len(prompt_tokens), cancel=cancel
                    )
                prefill_start = time.monotonic()
                cached_len, layer_caches, last_hidden_state = (
                    self.prefill_worker.prefill(prompt_tokens, cancel=cancel)
                )
                if cached_len >= 1:
                    external_cached_len = cached_len
                    cached_len = self._apply_external_prefill_tail(
                        cached_len, len(prompt_tokens), with_mtp=with_mtp
                    )
                    if require_external_prefill:
                        error = required_external_prefill_error(
                            cached_len, len(prompt_tokens)
                        )
                        if error is not None:
                            raise RuntimeError(error)
                    if cached_len != external_cached_len or cached_len != len(
                        prompt_tokens
                    ):
                        last_hidden_state = None
                    self.generator.start_sequence_from_cache(
                        prompt_tokens,
                        layer_caches,
                        last_hidden_state=last_hidden_state,
                        cached_len=cached_len,
                        sampling_params=sampling_params,
                        with_mtp=with_mtp,
                    )
                    elapsed = time.monotonic() - prefill_start
                    logger.info(
                        "External prefill injected. rid=%s cached_tokens=%s/%s "
                        "external_cached_tokens=%s tail_prefill_tokens=%s "
                        "elapsed=%.2fs input_tps=%.2f",
                        rid,
                        cached_len,
                        len(prompt_tokens),
                        external_cached_len,
                        max(len(prompt_tokens) - cached_len, 0),
                        elapsed,
                        cached_len / max(elapsed, 1e-9),
                    )
                    return
                if require_external_prefill:
                    raise RuntimeError(
                        "External prefill cached no prompt tokens; TileRT internal "
                        "prefill is disabled."
                    )
                logger.warning(
                    "External prefill cached nothing; falling back. rid=%s", rid
                )
            except Exception:
                if require_external_prefill:
                    raise
                logger.exception(
                    "External prefill failed; falling back to TileRT internal "
                    "prefill. rid=%s",
                    rid,
                )
        elif require_external_prefill:
            raise RuntimeError(
                "External prefill is required for TileRT requests, but this "
                "prompt was not routed to the external prefill engine."
            )
        self.generator.start_sequence(
            prompt_tokens,
            sampling_params=sampling_params,
            with_mtp=with_mtp,
        )

    def _apply_external_prefill_tail(
        self, cached_len: int, prompt_len: int, *, with_mtp: bool
    ) -> int:
        return cached_len_after_external_prefill_tail(
            cached_len,
            prompt_len,
            self.server_args.tilert_prefill_tail_tokens,
            with_mtp=with_mtp,
            mtp_seq_len=int(getattr(self.generator, "mtp_seq_len", 1)),
        )

    def _run_worker_request(
        self,
        rid: str,
        prompt_tokens: list[int],
        sampling_params: Any,
        cancel: threading.Event,
    ) -> None:
        request_start_time = time.monotonic()
        last_prefill_log_time = request_start_time
        last_prefill_log_tokens = 0
        prefill_done_logged = False
        decode_start_time = 0.0
        last_decode_log_time = request_start_time
        last_decode_log_tokens = 0
        last_decode_log_steps = 0

        try:
            logger.info(
                "Start TileRT request. rid=%s input_tokens=%s max_new_tokens=%s",
                rid,
                len(prompt_tokens),
                sampling_params.max_new_tokens,
            )
            self._start_generator_sequence(rid, prompt_tokens, sampling_params, cancel)
            while not cancel.is_set() and not self.generator.is_sequence_finished():
                token_ids = self.generator.next_tokens()
                (
                    prompt_len,
                    prefilled_tokens,
                    output_tokens,
                    prefill_done,
                ) = self.generator.sequence_progress()
                now = time.monotonic()
                if not prefill_done:
                    if now - last_prefill_log_time >= 5.0:
                        elapsed = max(now - request_start_time, 1e-9)
                        interval = max(now - last_prefill_log_time, 1e-9)
                        interval_tokens = prefilled_tokens - last_prefill_log_tokens
                        logger.info(
                            "TileRT prefill progress. rid=%s input_tokens=%s prefilled_tokens=%s remaining_tokens=%s input_tps=%.2f avg_input_tps=%.2f elapsed=%.2fs",
                            rid,
                            prompt_len,
                            prefilled_tokens,
                            max(prompt_len - prefilled_tokens, 0),
                            interval_tokens / interval,
                            prefilled_tokens / elapsed,
                            elapsed,
                        )
                        last_prefill_log_time = now
                        last_prefill_log_tokens = prefilled_tokens
                elif not prefill_done_logged:
                    elapsed = max(now - request_start_time, 1e-9)
                    logger.info(
                        "TileRT prefill done. rid=%s input_tokens=%s elapsed=%.2fs avg_input_tps=%.2f",
                        rid,
                        prompt_len,
                        elapsed,
                        prompt_len / elapsed,
                    )
                    prefill_done_logged = True
                    decode_start_time = now
                    last_decode_log_time = now
                    last_decode_log_tokens = output_tokens

                if token_ids:
                    self._worker_events.put(("tokens", rid, token_ids))
                    if now - last_decode_log_time >= 5.0:
                        decode_elapsed = max(now - decode_start_time, 1e-9)
                        interval = max(now - last_decode_log_time, 1e-9)
                        interval_tokens = output_tokens - last_decode_log_tokens
                        decode_stats = self.generator.sequence_decode_stats(
                            since_step=last_decode_log_steps
                        )
                        logger.info(
                            "TileRT decode progress. rid=%s output_tokens=%s "
                            "output_tps=%.2f avg_output_tps=%.2f elapsed=%.2fs "
                            "with_mtp=%s decode_steps=%s avg_accepted=%.2f "
                            "min_accepted=%s max_accepted=%s recent_steps=%s "
                            "recent_avg_accepted=%.2f forward_elapsed=%.2fs "
                            "post_elapsed=%.2fs",
                            rid,
                            output_tokens,
                            interval_tokens / interval,
                            output_tokens / decode_elapsed,
                            decode_elapsed,
                            decode_stats["with_mtp"],
                            decode_stats["decode_steps"],
                            decode_stats["accepted_avg"],
                            decode_stats["accepted_min"],
                            decode_stats["accepted_max"],
                            decode_stats["recent_steps"],
                            decode_stats["recent_accepted_avg"],
                            decode_stats["forward_seconds"],
                            decode_stats["post_seconds"],
                        )
                        last_decode_log_time = now
                        last_decode_log_tokens = output_tokens
                        last_decode_log_steps = int(decode_stats["decode_steps"])

            if cancel.is_set():
                decode_stats = self.generator.sequence_decode_stats(
                    since_step=last_decode_log_steps
                )
                self.generator.finish_sequence()
                elapsed = time.monotonic() - request_start_time
                logger.info(
                    "Abort TileRT request. rid=%s elapsed=%.2fs with_mtp=%s "
                    "decode_steps=%s avg_accepted=%.2f min_accepted=%s "
                    "max_accepted=%s recent_steps=%s recent_avg_accepted=%.2f "
                    "forward_elapsed=%.2fs post_elapsed=%.2fs",
                    rid,
                    elapsed,
                    decode_stats["with_mtp"],
                    decode_stats["decode_steps"],
                    decode_stats["accepted_avg"],
                    decode_stats["accepted_min"],
                    decode_stats["accepted_max"],
                    decode_stats["recent_steps"],
                    decode_stats["recent_accepted_avg"],
                    decode_stats["forward_seconds"],
                    decode_stats["post_seconds"],
                )
                self._worker_events.put(("aborted", rid, None))
            else:
                decode_stats = self.generator.sequence_decode_stats(
                    since_step=last_decode_log_steps
                )
                self.generator.finish_sequence()
                elapsed = time.monotonic() - request_start_time
                decode_elapsed = max(time.monotonic() - decode_start_time, 1e-9)
                logger.info(
                    "Finish TileRT request. rid=%s elapsed=%.2fs "
                    "decode_elapsed=%.2fs output_tokens=%s avg_output_tps=%.2f "
                    "with_mtp=%s decode_steps=%s avg_accepted=%.2f "
                    "min_accepted=%s max_accepted=%s recent_steps=%s "
                    "recent_avg_accepted=%.2f forward_elapsed=%.2fs "
                    "post_elapsed=%.2fs",
                    rid,
                    elapsed,
                    decode_elapsed,
                    decode_stats["output_tokens"],
                    decode_stats["output_tokens"] / decode_elapsed,
                    decode_stats["with_mtp"],
                    decode_stats["decode_steps"],
                    decode_stats["accepted_avg"],
                    decode_stats["accepted_min"],
                    decode_stats["accepted_max"],
                    decode_stats["recent_steps"],
                    decode_stats["recent_accepted_avg"],
                    decode_stats["forward_seconds"],
                    decode_stats["post_seconds"],
                )
                self._worker_events.put(("done", rid, None))
        except Exception as exc:
            try:
                self.generator.finish_sequence()
            except Exception:
                logger.exception(
                    "TileRT failed to clean up failed request. rid=%s", rid
                )
            self._worker_events.put(("error", rid, exc))

    def _drain_worker_events(self) -> bool:
        made_progress = False
        while True:
            try:
                event_type, rid, payload = self._worker_events.get_nowait()
            except queue.Empty:
                break

            made_progress = True
            self._handle_worker_event(event_type, rid, payload)
        return made_progress

    def _handle_worker_event(self, event_type: str, rid: str, payload: Any) -> None:
        req = self.active_req
        if req is None or req.rid != rid:
            return

        if event_type == "tokens":
            if self._active_final_sent:
                return
            token_ids = payload
            if token_ids:
                req.output_ids.extend(token_ids)
                req.update_finish_state(new_accepted_len=len(token_ids))
                self.output_streamer.stream_output([req], return_logprob=False)

            if req.finished() and not self._active_final_sent:
                self._active_final_sent = True
                if self._active_cancel is not None:
                    self._active_cancel.set()
            return

        if event_type == "done":
            if not self._active_final_sent:
                if not req.finished():
                    req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
                    req.finished_len = len(req.output_ids)
                self.output_streamer.stream_output([req], return_logprob=False)
            self._clear_active_request()
            return

        if event_type == "aborted":
            if not self._active_final_sent:
                req.return_logprob = False
                req.finished_reason = FINISH_ABORT()
                req.finished_len = len(req.output_ids)
                self.output_streamer.stream_output([req], return_logprob=False)
            self._clear_active_request()
            return

        if event_type == "error":
            self._finish_with_abort(
                req,
                f"TileRT request failed: {payload}",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            self._clear_active_request()

    def _clear_active_request(self) -> None:
        self.active_req = None
        self._active_cancel = None
        self._active_final_sent = False
        self._worker_thread = None

    def _finish_with_abort(
        self, req: Req, message: str, status_code: HTTPStatus
    ) -> None:
        logger.error("%s, rid=%s", message, req.rid)
        req.return_logprob = False
        err_type = (
            "BadRequestError"
            if status_code < HTTPStatus.INTERNAL_SERVER_ERROR
            else "InternalServerError"
        )
        req.finished_reason = FINISH_ABORT(message, status_code, err_type)
        req.finished_len = len(req.output_ids)
        self.output_streamer.stream_output([req], return_logprob=False)

    def _abort_request(self, recv_req: AbortReq) -> None:
        remaining: Deque[Req] = deque()
        for req in self.waiting_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                logger.info("Abort queued TileRT request. rid=%s", req.rid)
                self.ipc_channels.send_to_tokenizer.send_output(
                    AbortReq(rid=req.rid), req
                )
            else:
                remaining.append(req)
        self.waiting_queue = remaining

        req = self.active_req
        if req is None:
            return
        if recv_req.abort_all or req.rid.startswith(recv_req.rid):
            logger.info("Cancel active TileRT request. rid=%s", req.rid)
            if self._active_cancel is not None:
                self._active_cancel.set()

    def _flush_cache(self, recv_req: FlushCacheReqInput) -> FlushCacheReqOutput:
        if self.active_req is not None or self.waiting_queue:
            return FlushCacheReqOutput(
                success=False,
                message="TileRT scheduler can flush only when idle.",
            )
        return FlushCacheReqOutput(success=True)

    def _configure_logging(self, recv_req: ConfigureLoggingReq) -> None:
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())
