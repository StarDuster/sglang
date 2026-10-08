"""Independent frontend observer; stdlib only, local spool, isolated NAS copy.

Native py-spy can briefly suspend its target. A separate process enforces its
deadline. This observer never requests GPU coredumps or signals model PIDs.
"""

import argparse
import hashlib
import json
import mmap
import os
import re
import selectors
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

STATE_SIZE = 262144
MAX_OUTPUT = 1048576
MAX_SECTIONS = 128
EVENT_BUDGET = 28.0
MAX_STALL_DUMPS = 3


def process_identity(pid):
    try:
        fields = (
            (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        )
        return {"pid": pid, "birth": fields[19], "state": fields[0]}
    except (OSError, IndexError, ValueError):
        return None


def same_process(pid, birth):
    current = process_identity(pid)
    return bool(
        current and current["birth"] == birth and current["state"] not in ("Z", "X")
    )


def atomic_json(path, value):
    path = Path(path)
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    tmp = path.with_name("." + path.name + "-" + uuid.uuid4().hex)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


class StateWriter:
    """Local shared snapshot; caller serializes writes. No per-call disk flush."""

    def __init__(self, path):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.ftruncate(fd, STATE_SIZE)
            self.mapping = mmap.mmap(fd, STATE_SIZE)
        finally:
            os.close(fd)
        self.generation = 0

    def publish(self, sections):
        if len(sections) > MAX_SECTIONS:
            raise ValueError("diagnostic section capacity exceeded")
        raw = json.dumps(sections, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > STATE_SIZE - 16:
            raise ValueError("diagnostic snapshot too large")
        self.generation += 2
        struct.pack_into("<Q", self.mapping, 0, self.generation - 1)
        struct.pack_into("<Q", self.mapping, 8, len(raw))
        self.mapping[16 : 16 + len(raw)] = raw
        struct.pack_into("<Q", self.mapping, 0, self.generation)

    def close(self):
        self.mapping.close()


def read_snapshot(mapping):
    for _ in range(4):
        first, length = struct.unpack_from("<QQ", mapping, 0)
        if first == 0 or first % 2 or length > STATE_SIZE - 16:
            continue
        raw = bytes(mapping[16 : 16 + length])
        last = struct.unpack_from("<Q", mapping, 0)[0]
        if first == last:
            try:
                data = json.loads(raw)
            except (ValueError, UnicodeError):
                continue
            if isinstance(data, list) and len(data) <= MAX_SECTIONS:
                return data
    return None


def run_bounded(command, timeout, limit=MAX_OUTPUT):
    """Bound time/output; kill only the owned diagnostic process group."""
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as error:
        return {"ok": False, "error": type(error).__name__, "output": b"", "seconds": 0}
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    chunks = bytearray()
    error = None
    eof = False
    try:
        while not eof:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                error = "timeout"
                break
            for key, _ in selector.select(min(0.1, remaining)):
                part = os.read(
                    key.fileobj.fileno(), min(65536, limit + 1 - len(chunks))
                )
                if not part:
                    eof = True
                    break
                chunks.extend(part)
                if len(chunks) > limit:
                    error = "output_limit"
                    break
            if error:
                break
        if not error:
            try:
                proc.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                error = "timeout"
    finally:
        if error or proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                error = "unreaped_diagnostic_child"
        selector.close()
        proc.stdout.close()
    return {
        "ok": error is None and proc.returncode == 0,
        "error": error or (None if proc.returncode == 0 else "nonzero_exit"),
        "exit_code": proc.returncode,
        "output": bytes(chunks[:limit]),
        "seconds": round(time.monotonic() - started, 3),
    }


def proc_tasks(pid):
    result = []
    root = Path("/proc") / str(pid) / "task"
    for task in sorted(root.iterdir(), key=lambda p: int(p.name))[:512]:
        row = {"tid": int(task.name)}
        for name in ("wchan", "syscall", "stack"):
            try:
                row[name] = (task / name).read_text(errors="replace")[:8192].strip()
            except OSError as error:
                row[name] = {"error": type(error).__name__}
        try:
            status = (task / "status").read_text()
            row["state"] = next(
                line.split(":", 1)[1].strip()
                for line in status.splitlines()
                if line.startswith("State:")
            )
        except (OSError, StopIteration) as error:
            row["state"] = {"error": type(error).__name__}
        result.append(row)
    return result


def clean_pyspy(raw):
    traces = json.loads(raw)
    if not isinstance(traces, list) or not traces:
        raise ValueError("unexpected py-spy JSON")
    return [
        {
            "thread_id": t.get("thread_id"),
            "os_thread_id": t.get("os_thread_id"),
            "active": t.get("active"),
            "owns_gil": t.get("owns_gil"),
            "frames": [
                {k: f.get(k) for k in ("name", "filename", "line")}
                for f in t.get("frames", [])[:512]
            ],
        }
        for t in traces[:512]
    ]


def gpu_metadata(raw):
    root = ET.fromstring(raw)
    allowed = {
        "product_name",
        "uuid",
        "pci",
        "ecc_errors",
        "retired_pages",
        "row_remapper",
        "clocks_event_reasons",
        "clocks_throttle_reasons",
        "temperature",
        "fb_memory_usage",
        "utilization",
        "fabric",
    }

    def tree(element):
        if not len(element):
            return (element.text or "").strip()[:512]
        return [{child.tag: tree(child)} for child in element][:128]

    return {
        "driver_version": root.findtext("driver_version"),
        "gpus": [
            {child.tag: tree(child) for child in gpu if child.tag in allowed}
            for gpu in root.findall("gpu")[:16]
        ],
    }


def collect(pid, birth, sections, spool, native_timeout=8.0):
    if not same_process(pid, birth):
        return None
    folder = Path(spool) / ("event-" + uuid.uuid4().hex)
    folder.mkdir(mode=0o700, parents=True)
    started = time.monotonic()
    manifest = {
        "pid": pid,
        "birth": birth,
        "event_id": folder.name,
        "host": socket.gethostname(),
        "files": [],
        "capture_complete": False,
        "GPU_coredump_requested": False,
    }

    def record(name, value):
        path = folder / name
        atomic_json(path, value)
        manifest["files"].append(
            {"file": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        atomic_json(folder / "manifest.json", manifest)

    record("sections.json", sections)
    attach_unsafe = False

    def command_result(name, command, timeout, decoder):
        nonlocal attach_unsafe
        if not same_process(pid, birth):
            record(name, {"ok": False, "error": "process_gone_or_reused"})
            return False
        budget = min(timeout, EVENT_BUDGET - (time.monotonic() - started))
        if budget <= 0:
            record(name, {"ok": False, "error": "event_budget_exhausted"})
            return False
        result = run_bounded(command, budget)
        raw = result.pop("output")
        if result.get("error") == "unreaped_diagnostic_child":
            attach_unsafe = True
        result["birth_matches_after"] = same_process(pid, birth)
        if result["ok"] and result["birth_matches_after"]:
            try:
                result["data"] = decoder(raw)
            except (ValueError, UnicodeError, TypeError, ET.ParseError):
                result.update(ok=False, error="decode_failed")
        else:
            result["ok"] = False
        record(name, result)
        return result["ok"]

    command_result(
        "proc_tasks.json",
        [sys.executable, __file__, "--proc-tasks", str(pid)],
        2,
        json.loads,
    )
    native_ok = command_result(
        "py_spy_native.json",
        ["py-spy", "dump", "--json", "--native", "--pid", str(pid)],
        native_timeout,
        clean_pyspy,
    )
    if not native_ok and not attach_unsafe:
        command_result(
            "py_spy_python.json",
            ["py-spy", "dump", "--json", "--nonblocking", "--pid", str(pid)],
            3,
            clean_pyspy,
        )
    command_result(
        "kernel_errors.json",
        ["dmesg", "-T"],
        3,
        lambda raw: [
            line[:1024]
            for line in raw.decode(errors="replace").splitlines()
            if re.search(r"Xid|NVRM|NVLink|nvidia|fabric", line, re.I)
        ][-100:],
    )
    command_result("gpu_metadata.json", ["nvidia-smi", "-q", "-x"], 5, gpu_metadata)
    manifest.update(
        capture_complete=True,
        native_capture_passed=native_ok,
        attach_unsafe=attach_unsafe,
        seconds=round(time.monotonic() - started, 3),
        birth_matches_after=same_process(pid, birth),
    )
    atomic_json(folder / "manifest.json", manifest)
    return folder


def copy_event(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / "manifest.json").read_text())
    host = re.sub(r"[^A-Za-z0-9_.-]", "_", manifest["host"])[:100]
    target = destination / host / str(manifest["pid"]) / manifest["birth"] / source.name
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    for entry in manifest["files"]:
        name = entry["file"]
        if Path(name).name != name:
            raise ValueError("unsafe manifest path")
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("source hash mismatch")
        atomic_json(target / name, json.loads(data))
        if hashlib.sha256((target / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("NAS hash mismatch")
    atomic_json(target / "manifest.json", manifest)
    manifest_sha = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    if (
        hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest()
        != manifest_sha
    ):
        raise ValueError("NAS manifest hash mismatch")
    atomic_json(
        target / "COMPLETE.json",
        {
            "event_id": source.name,
            "verified": True,
            "manifest_sha256": manifest_sha,
            "native_capture_passed": manifest["native_capture_passed"],
        },
    )
    print(
        "K3_NATIVE_STALL_PERSIST "
        + json.dumps(
            {
                "event": source.name,
                "pid": manifest["pid"],
                "birth": manifest["birth"],
                "verified": True,
                "native_capture_passed": manifest["native_capture_passed"],
            }
        ),
        flush=True,
    )


def select_stalled(sections, now, threshold, captured):
    running = [s for s in sections if s["phase"] == "running"]
    candidates = running or sections
    if not candidates:
        return None
    oldest = min(candidates, key=lambda s: s.get("running_started") or s["started"])
    since = oldest.get("running_started") or oldest["started"]
    if now - since >= threshold and oldest["section_id"] not in captured:
        return oldest
    return None


def watch(state, pid, birth, threshold, destination, spool, ready_fd=None):
    os.nice(10)
    pending = []
    copier = None
    copy_retry_at = 0
    attach_disabled = False
    target_gone_at = None
    captured = set()
    num_dumps = 0
    fd = os.open(state, os.O_RDONLY)
    mapping = mmap.mmap(fd, STATE_SIZE, access=mmap.ACCESS_READ)
    os.close(fd)
    print(
        "K3_NATIVE_STALL_ARMED "
        + json.dumps(
            {
                "pid": pid,
                "birth": birth,
                "threshold_seconds": threshold,
                "observer_pid": os.getpid(),
                "GPU_coredump_enabled": False,
            }
        ),
        flush=True,
    )
    if ready_fd is not None:
        os.write(ready_fd, b"R")
        os.close(ready_fd)
    while True:
        alive = same_process(pid, birth)
        if not alive:
            if target_gone_at is None:
                target_gone_at = time.monotonic()
            if (
                not pending and copier is None
            ) or time.monotonic() - target_gone_at > 10:
                break
        if copier is not None:
            if copier.poll() is not None:
                if copier.returncode == 0:
                    pending.pop(0)
                else:
                    print(
                        "K3_NATIVE_STALL_COPY_ERROR "
                        + json.dumps(
                            {"event": pending[0].name, "returncode": copier.returncode}
                        ),
                        flush=True,
                    )
                copy_retry_at = (
                    float("inf")
                    if copier.returncode == 75
                    else time.monotonic() + (30 if copier.returncode else 0)
                )
                copier = None
        if pending and copier is None and time.monotonic() >= copy_retry_at:
            copier = subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--bounded-copy",
                    str(pending[0]),
                    "--destination",
                    destination,
                ],
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        sections = read_snapshot(mapping) if alive else None
        if sections is not None:
            active = {s["section_id"] for s in sections}
            captured.intersection_update(active)
            oldest = select_stalled(sections, time.monotonic(), threshold, captured)
            if oldest and num_dumps < MAX_STALL_DUMPS and not attach_disabled:
                captured.add(oldest["section_id"])
                num_dumps += 1
                try:
                    event = collect(pid, birth, sections, spool)
                    if event is not None:
                        attach_disabled = json.loads(
                            (event / "manifest.json").read_text()
                        ).get("attach_unsafe", False)
                        pending.append(event)
                        print(
                            "K3_NATIVE_STALL_CAPTURE "
                            + json.dumps(
                                {
                                    "event": event.name,
                                    "pid": pid,
                                    "birth": birth,
                                    "local_complete": True,
                                }
                            ),
                            flush=True,
                        )
                except Exception as error:
                    print(
                        "K3_NATIVE_STALL_CAPTURE_ERROR "
                        + json.dumps({"pid": pid, "error_type": type(error).__name__}),
                        flush=True,
                    )
        time.sleep(min(1.0, threshold / 4))
    mapping.close()
    if copier is not None and copier.poll() is None:
        copier.terminate()


def start_observer(threshold, destination):
    pid = os.getpid()
    identity = process_identity(pid)
    if identity is None:
        raise RuntimeError("frontend diagnostics require Linux procfs")
    root = Path(tempfile.mkdtemp(prefix="sglang-frontend-diag-"))
    state = root / "state.bin"
    writer = StateWriter(state)
    writer.publish([])
    read_fd, write_fd = os.pipe()
    child = subprocess.Popen(
        [
            sys.executable,
            __file__,
            "--watch",
            "--state",
            str(state),
            "--pid",
            str(pid),
            "--birth",
            identity["birth"],
            "--threshold",
            str(threshold),
            "--destination",
            destination,
            "--spool",
            str(root / "events"),
            "--ready-fd",
            str(write_fd),
        ],
        stdin=subprocess.DEVNULL,
        close_fds=True,
        pass_fds=(write_fd,),
        start_new_session=True,
    )
    os.close(write_fd)
    selector = selectors.DefaultSelector()
    selector.register(read_fd, selectors.EVENT_READ)
    try:
        if not selector.select(3) or os.read(read_fd, 1) != b"R":
            child.kill()
            child.wait(timeout=1)
            writer.close()
            raise RuntimeError("frontend observer readiness handshake failed")
    finally:
        selector.close()
        os.close(read_fd)
    return writer, child


def stop_observer(sig, frame):
    # Unwind run_bounded so its finally kills/reaps only our diagnostic child.
    raise SystemExit(128 + sig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--state")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--birth")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--spool")
    parser.add_argument("--destination")
    parser.add_argument("--proc-tasks", type=int)
    parser.add_argument("--copy")
    parser.add_argument("--bounded-copy")
    parser.add_argument("--ready-fd", type=int)
    args = parser.parse_args()
    if args.proc_tasks:
        print(json.dumps(proc_tasks(args.proc_tasks)))
    elif args.bounded_copy:
        result = run_bounded(
            [
                sys.executable,
                __file__,
                "--copy",
                args.bounded_copy,
                "--destination",
                args.destination,
            ],
            8,
        )
        if result["ok"]:
            print(result["output"].decode(), end="", flush=True)
        else:
            print(
                "K3_NATIVE_STALL_COPY_ERROR "
                + json.dumps(
                    {"event": Path(args.bounded_copy).name, "error": result["error"]}
                ),
                flush=True,
            )
        if result["error"] == "unreaped_diagnostic_child":
            sys.exit(75)
        sys.exit(0 if result["ok"] else 1)
    elif args.copy:
        copy_event(args.copy, args.destination)
    elif args.watch:
        signal.signal(signal.SIGTERM, stop_observer)
        signal.signal(signal.SIGINT, stop_observer)
        if not 0 < args.threshold <= 3600:
            raise ValueError("invalid threshold")
        watch(
            args.state,
            args.pid,
            args.birth,
            args.threshold,
            args.destination,
            args.spool,
            args.ready_fd,
        )
    else:
        parser.error("operation required")


if __name__ == "__main__":
    main()
