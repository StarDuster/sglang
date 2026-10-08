"""Coordinate the K3 frontend's preprocessing and VMM producer CUDA work.

This is a scoped concurrency mitigation, not a native CUDA deadlock diagnosis.
The lock is process-local and never shared with Scheduler processes.  It must
always be acquired before the VMM metadata lock; cancellation uses metadata
only and must not wait for CUDA.

Every section is also tracked, including the time spent waiting for the lock.
When ``SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS`` is set, a watchdog thread
collects native diagnostics once a section stays in flight that long; see
``frontend_cuda_stall_diagnostics``.
"""

import itertools
import logging
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


@dataclass
class FrontendCudaSection:
    section_id: int
    name: str
    thread_name: str
    native_thread_id: int
    started: float
    # "waiting" for the coordination lock, then "running" while holding it.
    phase: str = "waiting"


def in_flight_sections() -> list[FrontendCudaSection]:
    """Snapshot of the in-flight sections, oldest first."""
    with _sections_lock:
        sections = [replace(section) for section in _sections.values()]
    return sorted(sections, key=lambda section: section.started)


@contextmanager
def _tracked_section(name: str):
    _ensure_stall_watchdog()
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
    try:
        with _frontend_cuda_lock:
            with _sections_lock:
                section.phase = "running"
            yield
    finally:
        with _sections_lock:
            _sections.pop(section.section_id, None)


@contextmanager
def frontend_cuda_section(name: str):
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
    global _watchdog_started
    if _watchdog_started:
        return
    threshold = envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS.get()
    with _sections_lock:
        if _watchdog_started:
            return
        _watchdog_started = True
    if threshold is None or threshold <= 0:
        return
    threading.Thread(
        target=_stall_watchdog_loop,
        args=(float(threshold),),
        name="frontend-cuda-stall-watchdog",
        daemon=True,
    ).start()
    logger.info("Frontend CUDA stall watchdog started (threshold=%.1fs)", threshold)


# Each stalled section is dumped once; the cap bounds disk usage when a hang
# keeps queueing new sections behind it.
_MAX_STALL_DUMPS = 3


def _stall_watchdog_loop(threshold: float) -> None:
    from sglang.srt.utils.frontend_cuda_stall_diagnostics import (
        collect_frontend_cuda_stall_diagnostics,
    )

    dumped_section_ids: set[int] = set()
    num_dumps = 0
    poll_interval = min(1.0, threshold / 4)
    while num_dumps < _MAX_STALL_DUMPS:
        time.sleep(poll_interval)
        sections = in_flight_sections()
        now = time.monotonic()
        stalled = [
            section
            for section in sections
            if now - section.started >= threshold
            and section.section_id not in dumped_section_ids
        ]
        if not stalled:
            continue
        # Sections stuck behind the same hang share one dump.
        dumped_section_ids.update(section.section_id for section in sections)
        num_dumps += 1
        try:
            collect_frontend_cuda_stall_diagnostics(sections, now)
        except Exception:
            logger.exception("Frontend CUDA stall diagnostics failed")
    logger.error(
        "Frontend CUDA stall watchdog stopped after %d dumps", _MAX_STALL_DUMPS
    )
