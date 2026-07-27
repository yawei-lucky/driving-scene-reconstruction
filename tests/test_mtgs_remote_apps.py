"""Dependency-light behavior tests for the two MTGS remote applications."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).parents[1]


def _load(name: str, relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SERVER = _load("mtgs_remote_simulator_test", "apps/mtgs_remote_simulator.py")
DRIVER = _load("mtgs_remote_driver_test", "apps/mtgs_remote_driver.py")


class RemoteAuthorityTests(unittest.TestCase):
    def test_remote_mode_fails_stopped_then_accepts_fresh_control(self) -> None:
        authority = SERVER.RemoteAuthority("auto")
        auto = SERVER.HumanControl(throttle=0.4)

        control, mode, _, stale, _ = authority.applied_control(auto)
        self.assertEqual(mode, "auto")
        self.assertEqual(control, auto)
        self.assertFalse(stale)

        authority.accept(
            SERVER.RemoteCommandPacket(
                sequence=1,
                command="mode",
                value="remote",
            )
        )
        control, mode, age_ms, stale, _ = authority.applied_control(auto)
        self.assertEqual(mode, "safe-stop")
        self.assertEqual(control, SERVER.HumanControl(brake=1.0))
        self.assertIsNone(age_ms)
        self.assertTrue(stale)

        requested = SERVER.HumanControl(steer=0.2, throttle=0.7)
        authority.accept(
            SERVER.RemoteControlPacket(
                sequence=1,
                client_time_ns=1,
                control=requested,
            )
        )
        control, mode, age_ms, stale, _ = authority.applied_control(auto)
        self.assertEqual(mode, "remote")
        self.assertEqual(control, requested)
        self.assertIsNotNone(age_ms)
        self.assertFalse(stale)

    def test_estop_latches_until_reset(self) -> None:
        authority = SERVER.RemoteAuthority("auto")
        auto = SERVER.HumanControl(throttle=1.0)
        authority.accept(
            SERVER.RemoteCommandPacket(sequence=1, command="estop")
        )

        control, mode, _, stale, _ = authority.applied_control(auto)
        self.assertEqual(mode, "estop")
        self.assertEqual(control, SERVER.HumanControl(brake=1.0))
        self.assertTrue(stale)

        authority.accept(
            SERVER.RemoteCommandPacket(sequence=2, command="reset")
        )
        self.assertTrue(authority.consume_reset())
        self.assertFalse(authority.consume_reset())
        control, mode, _, stale, _ = authority.applied_control(auto)
        self.assertEqual(mode, "auto")
        self.assertEqual(control, auto)
        self.assertFalse(stale)


class DriverInputTests(unittest.TestCase):
    def test_default_network_topology_is_driver_initiated(self) -> None:
        with mock.patch.object(DRIVER.sys, "argv", ["mtgs_remote_driver.py"]):
            args = DRIVER.parse_args()

        self.assertIn("mode=caller", args.video_source)
        self.assertIn("mode=listener", SERVER.DEFAULT_VIDEO_DESTINATION)

    def test_old_video_listen_option_remains_a_compatibility_alias(self) -> None:
        source = (
            "srt://0.0.0.0:19001?"
            "mode=listener&latency=80&transtype=live"
        )
        with mock.patch.object(
            DRIVER.sys,
            "argv",
            ["mtgs_remote_driver.py", "--video-listen", source],
        ):
            args = DRIVER.parse_args()

        self.assertEqual(args.video_source, source)

    def test_keyboard_input_is_rate_limited_and_left_positive(self) -> None:
        with mock.patch.object(
            DRIVER.time,
            "monotonic",
            side_effect=(10.0, 10.1, 10.2),
        ):
            driver_input = DRIVER.DriverInput()
            driver_input.press("w")
            driver_input.press("a")
            first = driver_input.control()
            driver_input.release("w")
            driver_input.release("a")
            second = driver_input.control()

        self.assertAlmostEqual(first.steer, 0.24)
        self.assertAlmostEqual(first.throttle, 0.82)
        self.assertEqual(first.brake, 0.0)
        self.assertAlmostEqual(second.steer, 0.0)
        self.assertEqual(second.throttle, 0.0)


if __name__ == "__main__":
    unittest.main()
