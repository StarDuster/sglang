"""Unit tests for the frontend CUDA section tracking and stall diagnostics."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.utils import frontend_cuda_coordination as coordination
from sglang.srt.utils import frontend_cuda_stall_diagnostics as diagnostics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(5.0, "base-a-test-cpu")


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestSectionTracking(CustomTestCase):
    def test_section_is_tracked_while_running_and_cleared_after(self):
        with coordination.frontend_cuda_section("vmm_recycle"):
            (section,) = coordination.in_flight_sections()
            self.assertEqual(section.name, "vmm_recycle")
            self.assertEqual(section.phase, "running")
            self.assertEqual(section.native_thread_id, threading.get_native_id())
        self.assertEqual(coordination.in_flight_sections(), [])

    def test_section_waiting_for_lock_is_reported_as_waiting(self):
        release = threading.Event()
        holder_running = threading.Event()

        def hold():
            with coordination.frontend_cuda_section("k3_gpu_preprocess"):
                holder_running.set()
                release.wait()

        def wait_for_lock():
            with coordination.frontend_cuda_section("vmm_wrap_tensor"):
                pass

        holder = threading.Thread(target=hold)
        holder.start()
        self.assertTrue(holder_running.wait(5))
        waiter = threading.Thread(target=wait_for_lock)
        waiter.start()
        try:
            self.assertTrue(
                _wait_until(lambda: len(coordination.in_flight_sections()) == 2)
            )
            phases = {
                section.name: section.phase
                for section in coordination.in_flight_sections()
            }
            self.assertEqual(
                phases, {"k3_gpu_preprocess": "running", "vmm_wrap_tensor": "waiting"}
            )
        finally:
            release.set()
            holder.join(5)
            waiter.join(5)
        self.assertEqual(coordination.in_flight_sections(), [])

    def test_section_is_cleared_when_body_raises(self):
        with self.assertRaises(ValueError):
            with coordination.frontend_cuda_section("vmm_wrap_tensors"):
                raise ValueError("boom")
        self.assertEqual(coordination.in_flight_sections(), [])


class TestStallWatchdog(CustomTestCase):
    def test_stalled_sections_share_one_dump_and_loop_stops_at_cap(self):
        release = threading.Event()
        holder_running = threading.Event()
        dumps = []

        def stuck():
            with coordination.frontend_cuda_section("k3_gpu_preprocess"):
                holder_running.set()
                release.wait()

        def blocked():
            with coordination.frontend_cuda_section("vmm_recycle"):
                pass

        holder = threading.Thread(target=stuck)
        waiter = threading.Thread(target=blocked)
        with (
            mock.patch.object(coordination, "_MAX_STALL_DUMPS", 1),
            mock.patch.object(
                diagnostics,
                "collect_frontend_cuda_stall_diagnostics",
                side_effect=lambda sections, now: dumps.append(sections),
            ),
        ):
            holder.start()
            self.assertTrue(holder_running.wait(5))
            waiter.start()
            self.assertTrue(
                _wait_until(lambda: len(coordination.in_flight_sections()) == 2)
            )
            watchdog = threading.Thread(
                target=coordination._stall_watchdog_loop, args=(0.2,)
            )
            watchdog.start()
            watchdog.join(5)
            release.set()
            holder.join(5)
            waiter.join(5)

        self.assertFalse(watchdog.is_alive())
        self.assertEqual(len(dumps), 1)
        self.assertEqual(
            [(section.name, section.phase) for section in dumps[0]],
            [("k3_gpu_preprocess", "running"), ("vmm_recycle", "waiting")],
        )

    def test_short_sections_are_not_dumped(self):
        dumps = []
        with (
            mock.patch.object(coordination, "_MAX_STALL_DUMPS", 1),
            mock.patch.object(
                diagnostics,
                "collect_frontend_cuda_stall_diagnostics",
                side_effect=lambda sections, now: dumps.append(sections),
            ),
        ):
            watchdog = threading.Thread(
                target=coordination._stall_watchdog_loop, args=(0.2,), daemon=True
            )
            watchdog.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                with coordination.frontend_cuda_section("vmm_recycle"):
                    time.sleep(0.01)
        self.assertEqual(dumps, [])

    def test_watchdog_is_not_started_without_threshold(self):
        with (
            envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS.override(None),
            mock.patch.object(coordination, "_watchdog_started", False),
            mock.patch.object(coordination.threading, "Thread") as thread_cls,
        ):
            coordination._ensure_stall_watchdog()
        thread_cls.assert_not_called()

    def test_watchdog_is_started_once_with_threshold(self):
        with (
            envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS.override(45.0),
            mock.patch.object(coordination, "_watchdog_started", False),
            mock.patch.object(coordination.threading, "Thread") as thread_cls,
        ):
            coordination._ensure_stall_watchdog()
            coordination._ensure_stall_watchdog()
        thread_cls.assert_called_once()
        self.assertEqual(thread_cls.call_args.kwargs["args"], (45.0,))


class TestStallDiagnostics(CustomTestCase):
    def _collect(self, tmp_dir, environ=None):
        section = coordination.FrontendCudaSection(
            section_id=0,
            name="k3_gpu_preprocess",
            thread_name="mm-worker",
            native_thread_id=1234,
            started=time.monotonic() - 50,
            phase="running",
        )
        with (
            envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_DIR.override(tmp_dir),
            mock.patch.object(
                diagnostics, "_run_bounded", return_value=("canned\n", True)
            ) as run_bounded,
            mock.patch.dict(diagnostics.os.environ, environ or {}),
            mock.patch(
                "sglang.srt.utils.cudacore_pyspy_dump_utils.trigger_cuda_user_coredump"
            ) as trigger,
        ):
            out_dir = diagnostics.collect_frontend_cuda_stall_diagnostics(
                [section], time.monotonic()
            )
        return out_dir, run_bounded, trigger

    def test_writes_every_artifact_without_coredump_by_default(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.dict(diagnostics.os.environ):
                diagnostics.os.environ.pop("CUDA_ENABLE_USER_TRIGGERED_COREDUMP", None)
                out_dir, run_bounded, trigger = self._collect(tmp_dir)
            self.assertEqual(out_dir.parent, Path(tmp_dir))
            names = {path.name for path in out_dir.iterdir()}
            self.assertEqual(
                names,
                {
                    "sections.json",
                    "python_stacks.txt",
                    "proc_tasks.txt",
                    "py_spy_native.txt",
                    "dmesg.txt",
                    "nvidia_smi.txt",
                },
            )
            (record,) = json.loads((out_dir / "sections.json").read_text())
            self.assertEqual(record["name"], "k3_gpu_preprocess")
            self.assertGreaterEqual(record["seconds"], 50)
            self.assertIn(
                "test_writes_every_artifact",
                (out_dir / "python_stacks.txt").read_text(),
            )
            commands = [call.args[0][:2] for call in run_bounded.call_args_list]
            self.assertIn(["py-spy", "dump"], commands)
            self.assertIn(["nvidia-smi", "-q"], commands)
            trigger.assert_not_called()

    def test_requests_coredump_last_when_user_trigger_enabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir, _, trigger = self._collect(
                tmp_dir, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"}
            )
            trigger.assert_called_once_with()
            self.assertTrue((out_dir / "cuda_coredump.txt").exists())

    def test_py_spy_falls_back_without_native(self):
        results = iter([("native failed\n", False), ("python only\n", True)])
        with mock.patch.object(
            diagnostics, "_run_bounded", side_effect=lambda *a, **k: next(results)
        ) as run_bounded:
            output = diagnostics._py_spy_dump(42)
        self.assertIn("--native", run_bounded.call_args_list[0].args[0])
        self.assertNotIn("--native", run_bounded.call_args_list[1].args[0])
        self.assertIn("native failed", output)
        self.assertIn("python only", output)


class TestRunBounded(CustomTestCase):
    def test_success_and_failure(self):
        output, ok = diagnostics._run_bounded(["sh", "-c", "echo hi"], 5)
        self.assertTrue(ok)
        self.assertIn("hi", output)
        output, ok = diagnostics._run_bounded(["sh", "-c", "exit 3"], 5)
        self.assertFalse(ok)
        self.assertIn("<exit 3>", output)

    def test_timeout_returns_promptly(self):
        started = time.monotonic()
        output, ok = diagnostics._run_bounded(["sleep", "30"], 0.3)
        self.assertLess(time.monotonic() - started, 10)
        self.assertFalse(ok)
        self.assertIn("timed out", output)

    def test_missing_binary(self):
        output, ok = diagnostics._run_bounded(["sglang-no-such-binary"], 5)
        self.assertFalse(ok)
        self.assertIn("failed to start", output)

    def test_tail_lines(self):
        output, ok = diagnostics._run_bounded(
            ["sh", "-c", "for i in 1 2 3 4 5; do echo line$i; done"], 5, tail_lines=2
        )
        self.assertTrue(ok)
        self.assertNotIn("line3", output)
        self.assertIn("line4", output)
        self.assertIn("line5", output)


if __name__ == "__main__":
    unittest.main()
