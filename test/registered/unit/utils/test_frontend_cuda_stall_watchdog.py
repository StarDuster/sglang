"""Stdlib tests for the independent observer; no production/GPU access."""

import importlib.util
import json
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

SOURCE = (
    Path(__file__).parents[4]
    / "python/sglang/srt/utils/frontend_cuda_stall_observer.py"
)
spec = importlib.util.spec_from_file_location("frontend_observer_tested", SOURCE)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)

# Load the stdlib CI registry without importing the GPU runtime package.
ci_spec = importlib.util.spec_from_file_location(
    "ci_register", Path(__file__).parents[4] / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(ci_spec)
sys.modules[ci_spec.name] = ci
ci_spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(5.0, "base-a-test-cpu")


class TestObserver(unittest.TestCase):
    def test_shared_state_rejects_incomplete_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = observer.StateWriter(Path(tmp) / "state")
            value = [{"section_id": 1, "phase": "running", "started": 1.0}]
            writer.publish(value)
            self.assertEqual(observer.read_snapshot(writer.mapping), value)
            struct.pack_into("<Q", writer.mapping, 0, 3)
            self.assertIsNone(observer.read_snapshot(writer.mapping))
            writer.close()

    def test_waiters_do_not_repeat_same_hang(self):
        holder = dict(section_id=1, phase="running", started=1.0, running_started=2.0)
        waiter = dict(section_id=2, phase="waiting", started=3.0, running_started=None)
        self.assertEqual(
            observer.select_stalled([holder, waiter], 100, 45, set()), holder
        )
        self.assertIsNone(observer.select_stalled([holder, waiter], 100, 45, {1}))
        waiter.update(phase="running", running_started=99.0)
        self.assertIsNone(observer.select_stalled([waiter], 100, 45, set()))
        self.assertEqual(observer.select_stalled([waiter], 150, 45, set()), waiter)

    def test_new_waiter_does_not_exhaust_capture_budget(self):
        holder = dict(section_id=1, phase="running", started=1.0, running_started=1.0)
        for i in range(2, 20):
            waiter = dict(
                section_id=i, phase="waiting", started=2.0, running_started=None
            )
            self.assertIsNone(observer.select_stalled([holder, waiter], 100, 45, {1}))

    def test_source_locals_and_commands_are_removed(self):
        raw = json.dumps(
            [
                {
                    "command_line": "PRIVATE_COMMAND",
                    "locals": "PRIVATE_LOCAL",
                    "thread_id": 1,
                    "frames": [
                        {
                            "name": "function",
                            "filename": "file.py",
                            "line": 3,
                            "locals": "PRIVATE_LOCAL",
                            "source": "PRIVATE_SOURCE",
                        }
                    ],
                }
            ]
        ).encode()
        cleaned = observer.clean_pyspy(raw)
        self.assertEqual(
            cleaned[0]["frames"],
            [{"name": "function", "filename": "file.py", "line": 3}],
        )
        self.assertNotIn("PRIVATE", json.dumps(cleaned))
        with self.assertRaises(ValueError):
            observer.clean_pyspy(b"[]")

    def test_run_deadline_keeps_partial_output(self):
        started = time.monotonic()
        result = observer.run_bounded(
            [
                sys.executable,
                "-c",
                "import time;print('PREFIX',flush=True);time.sleep(20)",
            ],
            0.2,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "timeout")
        self.assertIn(b"PREFIX", result["output"])
        self.assertLess(time.monotonic() - started, 1.5)

    def test_output_is_bounded(self):
        result = observer.run_bounded(
            [sys.executable, "-c", "print('x'*1000000)"], 3, limit=1024
        )
        self.assertEqual(result["error"], "output_limit")
        self.assertEqual(len(result["output"]), 1024)

    def test_missing_tool_retained(self):
        result = observer.run_bounded(["/nonexistent/diagnostic-tool"], 0.1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "FileNotFoundError")

    def test_pid_reuse_prevents_attach(self):
        with (
            mock.patch.object(observer, "same_process", return_value=False),
            mock.patch.object(observer, "run_bounded") as run,
        ):
            self.assertIsNone(observer.collect(123, "old_birth", [], "/unused"))
            run.assert_not_called()

    def test_local_capture_does_not_access_NAS(self):
        with tempfile.TemporaryDirectory() as tmp:

            def run(command, timeout, **kwargs):
                if "--proc-tasks" in command:
                    data = b"[]"
                elif "py-spy" == command[0]:
                    data = b'[{"thread_id":1,"frames":[{"name":"wait","filename":"native.so","line":0}]}]'
                elif "dmesg" == command[0]:
                    data = b"[0] NVRM: Xid 1\n"
                else:
                    data = b"<nvidia_smi_log><driver_version>test</driver_version></nvidia_smi_log>"
                return {"ok": True, "output": data}

            with (
                mock.patch.object(observer, "same_process", return_value=True),
                mock.patch.object(observer, "run_bounded", side_effect=run),
            ):
                event = observer.collect(123, "99", [], tmp)
            manifest = json.loads((event / "manifest.json").read_text())
            self.assertTrue(manifest["capture_complete"])
            self.assertTrue(manifest["native_capture_passed"])
            self.assertFalse(manifest["GPU_coredump_requested"])
            bad_destination = Path(tmp) / "not_a_directory"
            bad_destination.write_text("unavailable NAS")
            with self.assertRaises(OSError):
                observer.copy_event(event, bad_destination)
            self.assertTrue((event / "py_spy_native.json").exists())
            destination = Path(tmp) / "readback"
            observer.copy_event(event, destination)
            complete = list(destination.rglob("COMPLETE.json"))
            self.assertEqual(len(complete), 1)
            self.assertTrue(json.loads(complete[0].read_text())["verified"])
            (event / "sections.json").write_text("corrupted")
            with self.assertRaises(ValueError):
                observer.copy_event(event, Path(tmp) / "corrupt-readback")

    def test_unreaped_native_child_disables_further_attach(self):
        with tempfile.TemporaryDirectory() as tmp:

            def run(command, timeout, **kwargs):
                if command[0] == "py-spy":
                    self.assertNotIn("--nonblocking", command)
                    return {
                        "ok": False,
                        "error": "unreaped_diagnostic_child",
                        "output": b"",
                    }
                if "--proc-tasks" in command:
                    raw = b"[]"
                elif command[0] == "dmesg":
                    raw = b""
                else:
                    raw = b"<nvidia_smi_log/>"
                return {"ok": True, "output": raw}

            with (
                mock.patch.object(observer, "same_process", return_value=True),
                mock.patch.object(observer, "run_bounded", side_effect=run),
            ):
                event = observer.collect(123, "99", [], tmp)
            manifest = json.loads((event / "manifest.json").read_text())
            self.assertTrue(manifest["attach_unsafe"])
            self.assertFalse(manifest["native_capture_passed"])
            self.assertFalse((event / "py_spy_python.json").exists())

    def test_gpu_process_metadata_omits_process_commands(self):
        raw = b"<nvidia_smi_log><driver_version>1</driver_version><gpu><uuid>id</uuid><processes>PRIVATE_COMMAND</processes></gpu></nvidia_smi_log>"
        result = observer.gpu_metadata(raw)
        self.assertNotIn("PRIVATE_COMMAND", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
