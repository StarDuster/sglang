"""Frontend CUDA sections must never be blocked or failed by stall diagnostics."""

import threading
import time
import unittest
from contextlib import ExitStack
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.utils import frontend_cuda_coordination as coordination
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


class _FakeProcess:
    pid = 4242

    def __init__(self):
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True


class _FakeState:
    def __init__(self, fail=False):
        self.fail = fail
        self.snapshots = []

    def publish(self, sections):
        if self.fail:
            raise OSError("state write failed")
        self.snapshots.append(sections)


class TestCoordinationNeverBlocksRequests(CustomTestCase):
    def setUp(self):
        self._stack = ExitStack()
        for name, value in (
            ("_watchdog_started", False),
            ("_observer_state", None),
            ("_observer_process", None),
            ("_observer_failed", False),
        ):
            self._stack.enter_context(mock.patch.object(coordination, name, value))
        self._stack.enter_context(mock.patch.dict(coordination._sections, clear=True))

    def tearDown(self):
        self._stack.close()

    def _run_section(self):
        started = time.monotonic()
        with coordination.frontend_cuda_section("vmm_recycle"):
            pass
        return time.monotonic() - started

    def _patch_start_observer(self, side_effect):
        return mock.patch(
            "sglang.srt.utils.frontend_cuda_stall_observer.start_observer",
            side_effect=side_effect,
        )

    def test_disabled_by_default_uses_plain_lock(self):
        with (
            envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS.override(None),
            self._patch_start_observer(AssertionError("must not start")) as start,
        ):
            self._run_section()
        start.assert_not_called()
        self.assertIsNone(coordination._observer_state)

    def test_invalid_threshold_never_raises_into_sections(self):
        for value in ("not-a-number", "1e9", "nan", "inf"):
            with self.subTest(value=value):
                coordination._watchdog_started = False
                with (
                    envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS.override(value),
                    self._patch_start_observer(AssertionError("must not start")),
                ):
                    for _ in range(3):
                        self._run_section()
                self.assertIsNone(coordination._observer_state)

    def test_slow_observer_startup_does_not_delay_sections(self):
        release = threading.Event()

        def slow_start(threshold, destination):
            release.wait(5)
            return _FakeState(), _FakeProcess()

        with (
            envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS.override(45),
            self._patch_start_observer(slow_start),
        ):
            durations = [self._run_section() for _ in range(20)]
            self.assertLess(max(durations), 0.1)
            self.assertIsNone(coordination._observer_state)
            release.set()
            self.assertTrue(_wait_until(lambda: coordination._observer_state))
            self._run_section()
        self.assertEqual(coordination._observer_state.snapshots[-1], [])

    def test_observer_startup_failure_keeps_sections_working(self):
        with (
            envs.SGLANG_DEBUG_FRONTEND_CUDA_STALL_SECS.override(45),
            self._patch_start_observer(RuntimeError("no procfs")),
        ):
            self._run_section()
            self.assertTrue(_wait_until(lambda: coordination._observer_failed))
            self._run_section()
        self.assertIsNone(coordination._observer_state)

    def test_publish_failure_disables_diagnostics_not_sections(self):
        process = _FakeProcess()
        coordination._watchdog_started = True
        coordination._observer_process = process
        coordination._observer_state = _FakeState(fail=True)
        for _ in range(3):
            self._run_section()
        self.assertTrue(coordination._observer_failed)
        self.assertTrue(process.terminated)
        self.assertEqual(coordination.in_flight_sections(), [])

    def test_section_body_exception_still_clears_state(self):
        state = _FakeState()
        coordination._watchdog_started = True
        coordination._observer_process = _FakeProcess()
        coordination._observer_state = state
        with self.assertRaises(ValueError):
            with coordination.frontend_cuda_section("vmm_wrap_tensor"):
                raise ValueError("boom")
        self.assertEqual(coordination.in_flight_sections(), [])
        self.assertEqual(state.snapshots[-1], [])


if __name__ == "__main__":
    unittest.main()
