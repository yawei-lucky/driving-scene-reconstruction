"""Small, dependency-free wire protocol for the two-machine driving demo."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any, Mapping

from .control import HumanControl


REMOTE_PROTOCOL_VERSION = 1
REMOTE_CONTROL_TIMEOUT_SECONDS = 0.25
REMOTE_SRT_LATENCY_MICROSECONDS = 300_000
REMOTE_SRT_PACKET_SIZE_BYTES = 1316
REMOTE_MODES = frozenset(("auto", "remote"))
REMOTE_COMMANDS = frozenset(("mode", "reset", "estop"))


def _finite_number(
    message: Mapping[str, Any],
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    value = message.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return result


def _non_negative_integer(message: Mapping[str, Any], name: str) -> int:
    value = message.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def encode_remote_message(message: Mapping[str, Any]) -> str:
    """Serialize one compact protocol message."""

    return json.dumps(
        dict(message),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def decode_remote_message(payload: str | bytes) -> dict[str, Any]:
    """Decode a JSON object and enforce the shared protocol version."""

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if not isinstance(payload, str):
        raise ValueError("remote message must be text")
    try:
        message = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("remote message is not valid JSON") from error
    if not isinstance(message, dict):
        raise ValueError("remote message must be a JSON object")
    if message.get("version") != REMOTE_PROTOCOL_VERSION:
        raise ValueError(
            f"remote protocol version must be {REMOTE_PROTOCOL_VERSION}"
        )
    message_type = message.get("type")
    if not isinstance(message_type, str):
        raise ValueError("remote message type must be text")
    return message


@dataclass(frozen=True)
class RemoteControlPacket:
    """One latest-value steering packet from the driver computer."""

    sequence: int
    client_time_ns: int
    control: HumanControl

    @classmethod
    def from_message(
        cls,
        message: Mapping[str, Any],
    ) -> "RemoteControlPacket":
        if message.get("type") != "control":
            raise ValueError("expected a control message")
        sequence = _non_negative_integer(message, "sequence")
        client_time_ns = _non_negative_integer(message, "client_time_ns")
        control = HumanControl(
            steer=_finite_number(message, "steer", -1.0, 1.0),
            throttle=_finite_number(message, "throttle", 0.0, 1.0),
            brake=_finite_number(message, "brake", 0.0, 1.0),
        )
        return cls(
            sequence=sequence,
            client_time_ns=client_time_ns,
            control=control,
        )

    def message(self) -> dict[str, Any]:
        return {
            "type": "control",
            "version": REMOTE_PROTOCOL_VERSION,
            "sequence": self.sequence,
            "client_time_ns": self.client_time_ns,
            "steer": self.control.steer,
            "throttle": self.control.throttle,
            "brake": self.control.brake,
        }


@dataclass(frozen=True)
class RemoteCommandPacket:
    """A reliable, discrete mode/reset/emergency-stop request."""

    sequence: int
    command: str
    value: str | None = None

    @classmethod
    def from_message(
        cls,
        message: Mapping[str, Any],
    ) -> "RemoteCommandPacket":
        if message.get("type") != "command":
            raise ValueError("expected a command message")
        sequence = _non_negative_integer(message, "sequence")
        command = message.get("command")
        if command not in REMOTE_COMMANDS:
            raise ValueError(f"unsupported remote command {command!r}")
        value = message.get("value")
        if command == "mode":
            if value not in REMOTE_MODES:
                raise ValueError(f"mode must be one of {sorted(REMOTE_MODES)}")
        elif value is not None:
            raise ValueError(f"{command} does not accept a value")
        return cls(sequence=sequence, command=command, value=value)

    def message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "type": "command",
            "version": REMOTE_PROTOCOL_VERSION,
            "sequence": self.sequence,
            "command": self.command,
        }
        if self.value is not None:
            message["value"] = self.value
        return message


def parse_driver_message(
    payload: str | bytes,
) -> RemoteControlPacket | RemoteCommandPacket | dict[str, Any]:
    """Parse one driver-to-simulator message."""

    message = decode_remote_message(payload)
    if message["type"] == "control":
        return RemoteControlPacket.from_message(message)
    if message["type"] == "command":
        return RemoteCommandPacket.from_message(message)
    if message["type"] == "hello":
        role = message.get("role")
        if role != "driver":
            raise ValueError("hello role must be 'driver'")
        return message
    raise ValueError(f"unsupported driver message type {message['type']!r}")


@dataclass
class RemoteControlWatchdog:
    """Accept monotonic controls and stop using them when they become stale."""

    timeout_seconds: float = REMOTE_CONTROL_TIMEOUT_SECONDS
    latest_packet: RemoteControlPacket | None = None
    latest_arrival_ns: int | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0.0
        ):
            raise ValueError("control timeout must be finite and positive")

    def accept(
        self,
        packet: RemoteControlPacket,
        *,
        arrival_ns: int | None = None,
    ) -> bool:
        """Accept a newer packet; duplicated or reordered packets are ignored."""

        if (
            self.latest_packet is not None
            and packet.sequence <= self.latest_packet.sequence
        ):
            return False
        self.latest_packet = packet
        self.latest_arrival_ns = (
            time.monotonic_ns() if arrival_ns is None else arrival_ns
        )
        return True

    def control(
        self,
        *,
        now_ns: int | None = None,
    ) -> tuple[HumanControl, float | None, bool]:
        """Return control, arrival age in milliseconds, and stale status."""

        if self.latest_packet is None or self.latest_arrival_ns is None:
            return HumanControl(brake=1.0), None, True
        current_ns = time.monotonic_ns() if now_ns is None else now_ns
        age_seconds = max(
            0.0,
            (current_ns - self.latest_arrival_ns) / 1_000_000_000.0,
        )
        stale = age_seconds > self.timeout_seconds
        if stale:
            return HumanControl(brake=1.0), age_seconds * 1000.0, True
        return self.latest_packet.control, age_seconds * 1000.0, False


def hello_message(
    *,
    role: str,
    token: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "type": "hello",
        "version": REMOTE_PROTOCOL_VERSION,
        "role": role,
    }
    if token:
        message["token"] = token
    return message


def error_message(detail: str) -> dict[str, Any]:
    return {
        "type": "error",
        "version": REMOTE_PROTOCOL_VERSION,
        "detail": detail,
    }
