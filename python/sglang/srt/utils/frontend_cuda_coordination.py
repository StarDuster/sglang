"""Coordinate the K3 frontend's preprocessing and VMM producer CUDA work.

This is a scoped concurrency mitigation, not a native CUDA deadlock diagnosis.
The lock is process-local and never shared with Scheduler processes.  It must
always be acquired before the VMM metadata lock; cancellation uses metadata
only and must not wait for CUDA.

Every section is also tracked, including the time spent waiting for the lock.
When ``SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS`` is set, an independent process
observes a small local shared-state file and collects diagnostics even if a
native call holds this process's GIL; see ``frontend_cuda_stall_observer``.
"""

import itertools
import logging
import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_frontend_cuda_lock = threading.RLock()

_sections_lock = threading.Lock()
_section_ids = itertools.count()
_sections: dict[int, "FrontendCudaSection"] = {}
_watchdog_started = False
_observer_state = None
_observer_process = None
_observer_failed = False


@dataclass
class FrontendCudaSection:
    section_id: int
    name: str
    thread_name: str
    native_thread_id: int
    started: float
    # "waiting" for the coordination lock, then "running" while holding it.
    phase: str = "waiting"
    running_started: float | None = None


def in_flight_sections() -> list[FrontendCudaSection]:
    """Snapshot of the in-flight sections, oldest first."""
    with _sections_lock:
        sections = [replace(section) for section in _sections.values()]
    return sorted(sections, key=lambda section: section.started)


@contextmanager
def _tracked_section(name: str):
    _ensure_stall_watchdog()
    if _observer_state is None:
        with _frontend_cuda_lock:
            yield
        return
    thread = threading.current_thread()
    section = FrontendCudaSection(
        section_id=next(_section_ids),
        name=name,
        thread_name=thread.name,
        native_thread_id=threading.get_native_id(),
        started=time.monotonic(),
    )
    with _sections_lock:
        _sections[section.section_id] = section
        _publish_sections()
    try:
        with _frontend_cuda_lock:
            with _sections_lock:
                section.phase = "running"
                section.running_started = time.monotonic()
                _publish_sections()
            yield
    finally:
        with _sections_lock:
            _sections.pop(section.section_id, None)
            _publish_sections()


def _publish_sections():
    """Caller holds _sections_lock. Diagnostic failure does not abort inference."""
    global _observer_failed
    if _observer_failed:
        return
    try:
        if _observer_process.poll() is not None:
            raise RuntimeError("frontend observer exited")
        _observer_state.publish(
            [
                {
                    "section_id": section.section_id,
                    "name": section.name,
                    "native_thread_id": section.native_thread_id,
                    "started": section.started,
                    "phase": section.phase,
                    "running_started": section.running_started,
                }
                for section in _sections.values()
            ]
        )
    except Exception as error:
        _observer_failed = True
        if _observer_process.poll() is None:
            try:
                _observer_process.terminate()  # Never capture from stale shared state.
            except ProcessLookupError:
                pass
        logger.error(
            "Frontend CUDA diagnostic state publication failed: %s",
            type(error).__name__,
        )


@contextmanager
def frontend_cuda_section(name: str = "frontend_cuda"):
    """Serialize the known frontend CUDA submission/synchronization domains."""
    with _tracked_section(name):
        yield


@contextmanager
def frontend_cuda_preprocess():
    """Finish preprocessing's current stream before another worker publishes.

    Preserve the existing stream and all image operations.  The explicit
    completion boundary also covers the exception path: a failed preprocessing
    call must not leave already-enqueued CUDA work racing the next producer.
    A native CUDA call can still hang here; the existing tracked worker and
    health timeout, and the external collector, remain responsible for that
    failure.  No timeout pretends to cancel a running native call.
    """
    import torch

    with _tracked_section("k3_gpu_preprocess"):
        stream = torch.cuda.current_stream()
        try:
            yield
        except BaseException as error:
            try:
                stream.synchronize()
            except BaseException as sync_error:
                error.add_note("K3 frontend preprocessing stream cleanup failed")
                raise error from sync_error
            raise
        else:
            stream.synchronize()


def _ensure_stall_watchdog() -> None:
    """Arm the observer at most once, off the request path.

    Sections keep using the plain lock until the observer is ready, and any
    configuration or startup error only disables diagnostics: it must never
    raise into preprocessing, publishing or recycling.
    """
    global _watchdog_started
    if _watchdog_started:
        return
    with _sections_lock:
        if _watchdog_started:
            return
        _watchdog_started = True
    try:
        threshold = envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS.get()
        if threshold is None or threshold <= 0:
            return
        if not math.isfinite(threshold) or threshold > 3600:
            raise ValueError(f"invalid threshold {threshold}")
        destination = (
            envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_DIR.get()
            or "/tmp/sglang_frontend_cuda_stall"
        )
        threading.Thread(
            target=_start_observer,
            args=(float(threshold), destination),
            name="frontend-cuda-observer-start",
            daemon=True,
        ).start()
    except Exception as error:
        logger.error("Frontend CUDA stall diagnostics disabled: %r", error)


def _start_observer(threshold: float, destination: str) -> None:
    global _observer_state, _observer_process, _observer_failed
    try:
        from sglang.srt.utils.frontend_cuda_stall_observer import start_observer

        state, process = start_observer(threshold, destination)
    except Exception as error:
        _observer_failed = True
        logger.error("Frontend CUDA observer startup failed: %r", error)
        return
    with _sections_lock:
        # The process first: _publish_sections polls it once state is visible.
        _observer_process = process
        _observer_state = state
    logger.info(
        "Frontend CUDA observer started pid=%s threshold=%.1fs",
        process.pid,
        threshold,
    )
