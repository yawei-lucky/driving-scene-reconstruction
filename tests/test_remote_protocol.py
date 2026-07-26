"""Tests for the dependency-free remote driving wire boundary."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from driving_scene_reconstruction.sim.control import HumanControl
from driving_scene_reconstruction.sim.remote_protocol import (
    REMOTE_PROTOCOL_VERSION,
    RemoteCommandPacket,
    RemoteControlPacket,
    RemoteControlWatchdog,
    encode_remote_message,
    parse_driver_message,
)


class RemoteProtocolTests(unittest.TestCase):
    def test_control_round_trip(self) -> None:
        packet = RemoteControlPacket(
            sequence=8,
            client_time_ns=1234,
            control=HumanControl(steer=-0.25, throttle=0.7, brake=0.0),
        )

        parsed = parse_driver_message(encode_remote_message(packet.message()))

        self.assertEqual(parsed, packet)

    def test_rejects_invalid_or_non_finite_control(self) -> None:
        base = {
            "type": "control",
            "version": REMOTE_PROTOCOL_VERSION,
            "sequence": 1,
            "client_time_ns": 1,
            "steer": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
        }
        for name, value in (
            ("steer", 1.01),
            ("throttle", -0.01),
            ("brake", float("nan")),
        ):
            message = dict(base)
            message[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_driver_message(json.dumps(message))

    def test_watchdog_ignores_reordering_and_brakes_when_stale(self) -> None:
        watchdog = RemoteControlWatchdog(timeout_seconds=0.25)
        newer = RemoteControlPacket(
            sequence=2,
            client_time_ns=2,
            control=HumanControl(steer=0.2, throttle=0.6),
        )
        older = RemoteControlPacket(
            sequence=1,
            client_time_ns=1,
            control=HumanControl(steer=-1.0, throttle=1.0),
        )

        self.assertTrue(watchdog.accept(newer, arrival_ns=1_000_000_000))
        self.assertFalse(watchdog.accept(older, arrival_ns=1_100_000_000))
        control, age_ms, stale = watchdog.control(now_ns=1_200_000_000)
        self.assertFalse(stale)
        self.assertAlmostEqual(age_ms or 0.0, 200.0)
        self.assertEqual(control, newer.control)

        control, age_ms, stale = watchdog.control(now_ns=1_251_000_000)
        self.assertTrue(stale)
        self.assertAlmostEqual(age_ms or 0.0, 251.0)
        self.assertEqual(control, HumanControl(brake=1.0))

    def test_command_validation(self) -> None:
        packet = RemoteCommandPacket(
            sequence=4,
            command="mode",
            value="remote",
        )
        self.assertEqual(
            parse_driver_message(encode_remote_message(packet.message())),
            packet,
        )
        message = packet.message()
        message["value"] = "unsafe"
        with self.assertRaises(ValueError):
            parse_driver_message(encode_remote_message(message))


if __name__ == "__main__":
    unittest.main()
