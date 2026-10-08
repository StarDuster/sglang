"""Collect native diagnostics for a stalled frontend CUDA section.

A hung CUDA call does not raise, so CUDA's exception coredump never fires and
Python frames stop at the call site.  This captures what can tell the hang
apart -- allocator mutex, driver wait, copy engine, fabric -- while the process
is still stuck: native thread stacks, per-thread kernel wait state, driver
(Xid / NVLink) messages and GPU state.  A CUDA user-triggered coredump of this
process is requested last, only when ``CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1``
was set before CUDA initialization, because it pauses the GPU and may not
return while a driver lock is held.

Every step is time-bounded and writes its own file as soon as it finishes, so
a step that hangs never loses the earlier output.
"""

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_DEFAULT_DUMP_DIR = "/tmp/sglang_frontend_cuda_stall"


def get_stall_dump_dir() -> Path:
    return Path(envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_DIR.get() or _DEFAULT_DUMP_DIR)


def collect_frontend_cuda_stall_diagnostics(sections, now: float) -> Path:
    pid = os.getpid()
    out_dir = get_stall_dump_dir() / (
        f"{socket.gethostname()}-pid{pid}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.error(
        "Frontend CUDA section stalled; collecting diagnostics into %s: %s",
        out_dir,
        _describe_sections(sections, now),
    )

    _write(
        out_dir / "sections.json",
        json.dumps(
            [
                {
                    "name": section.name,
                    "phase": section.phase,
                    "seconds": round(now - section.started, 3),
                    "thread_name": section.thread_name,
                    "native_thread_id": section.native_thread_id,
                }
                for section in sections
            ],
            indent=2,
        ),
    )
    _write(out_dir / "python_stacks.txt", _python_stacks())
    _write(out_dir / "proc_tasks.txt", _proc_tasks(pid))
    _write(out_dir / "py_spy_native.txt", _py_spy_dump(pid))
    _write(out_dir / "dmesg.txt", _run_bounded(["dmesg", "-T"], 10, tail_lines=500)[0])
    _write(out_dir / "nvidia_smi.txt", _run_bounded(["nvidia-smi", "-q"], 30)[0])

    if os.environ.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") == "1":
        from sglang.srt.utils.cudacore_pyspy_dump_utils import (
            trigger_cuda_user_coredump,
        )

        trigger_cuda_user_coredump()
        _write(
            out_dir / "cuda_coredump.txt",
            "Requested CUDA user coredump at "
            f"{datetime.now().isoformat()}; "
            f"CUDA_COREDUMP_FILE={os.environ.get('CUDA_COREDUMP_FILE')}\n",
        )

    logger.error("Frontend CUDA stall diagnostics written to %s", out_dir)
    return out_dir


def _describe_sections(sections, now: float) -> str:
    return "; ".join(
        f"{section.name}[{section.phase}] {now - section.started:.1f}s "
        f"thread={section.thread_name}/{section.native_thread_id}"
        for section in sections
    )


def _write(path: Path, content: str) -> None:
    try:
        path.write_text(content, errors="replace")
    except OSError:
        logger.exception("Failed to write %s", path)


def _python_stacks() -> str:
    threads = {thread.ident: thread for thread in threading.enumerate()}
    chunks = []
    for ident, frame in sys._current_frames().items():
        thread = threads.get(ident)
        name = thread.name if thread is not None else "<unknown>"
        native_id = getattr(thread, "native_id", None)
        chunks.append(
            f"Thread {name} ident={ident} native_id={native_id}\n"
            + "".join(traceback.format_stack(frame))
        )
    return "\n".join(chunks)


def _proc_tasks(pid: int) -> str:
    """Per-thread kernel wait state: futex means a user-space lock, an ioctl or
    driver wait channel means the thread is inside the NVIDIA driver."""
    task_root = Path(f"/proc/{pid}/task")
    if not task_root.is_dir():
        return f"<{task_root} not available on this platform>\n"
    chunks = []
    for task in sorted(task_root.iterdir(), key=lambda path: int(path.name)):
        fields = [f"tid={task.name}"]
        # syscall: number and arguments of the blocking call; stack needs root.
        for entry in ("comm", "wchan", "syscall", "stack"):
            try:
                value = (task / entry).read_text(errors="replace").strip()
            except OSError as error:
                value = f"<{error.strerror}>"
            fields.append(f"{entry}: {value}")
        try:
            state = next(
                line
                for line in (task / "status").read_text().splitlines()
                if line.startswith("State:")
            )
        except (OSError, StopIteration):
            state = "State: <unavailable>"
        fields.append(state)
        chunks.append("\n  ".join(fields))
    return "\n".join(chunks) + "\n"


def _py_spy_dump(pid: int) -> str:
    native, ok = _run_bounded(["py-spy", "dump", "--native", "--pid", str(pid)], 60)
    if ok:
        return native
    plain, _ = _run_bounded(["py-spy", "dump", "--pid", str(pid)], 60)
    return f"{native}\n--- retry without --native ---\n{plain}"


def _run_bounded(
    cmd: list[str], timeout: float, tail_lines: int = 0
) -> tuple[str, bool]:
    """Run ``cmd`` without ever blocking on it for long; return (output, ok).

    ``subprocess.run`` waits for the killed child after a timeout; a child stuck
    in uninterruptible driver sleep would then block this thread forever.
    """
    header = f"$ {' '.join(cmd)}  # {datetime.now().isoformat()}\n"
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
    except OSError as error:
        return f"{header}<failed to start: {error}>\n", False
    try:
        output, _ = proc.communicate(timeout=timeout)
        ok = proc.returncode == 0
        status = "" if ok else f"<exit {proc.returncode}>\n"
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            output, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            output = ""
        ok = False
        status = f"<timed out after {time.monotonic() - started:.1f}s>\n"
    if tail_lines:
        output = "\n".join(output.splitlines()[-tail_lines:]) + "\n"
    return header + output + status, ok
