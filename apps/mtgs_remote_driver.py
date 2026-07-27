#!/usr/bin/env python3
"""Native display/control app for the remote MTGS simulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import HumanControl  # noqa: E402
from driving_scene_reconstruction.sim.remote_protocol import (  # noqa: E402
    REMOTE_PROTOCOL_VERSION,
    REMOTE_SRT_LATENCY_MICROSECONDS,
    REMOTE_SRT_PACKET_SIZE_BYTES,
    RemoteCommandPacket,
    RemoteControlPacket,
    decode_remote_message,
    encode_remote_message,
    hello_message,
)


DEFAULT_SERVER = "ws://127.0.0.1:18765"
DEFAULT_VIDEO_SOURCE = (
    "srt://127.0.0.1:19001?mode=caller"
    f"&latency={REMOTE_SRT_LATENCY_MICROSECONDS}"
    f"&pkt_size={REMOTE_SRT_PACKET_SIZE_BYTES}&transtype=live"
)
DEFAULT_VIDEO_WIDTH = 1280
DEFAULT_VIDEO_HEIGHT = 544
DRIVE_KEYS = frozenset(("w", "a", "s", "d", "up", "left", "down", "right"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help="Shidi simulator WebSocket URL, for example ws://192.168.1.20:18765",
    )
    parser.add_argument(
        "--video-source",
        "--video-listen",
        dest="video_source",
        default=DEFAULT_VIDEO_SOURCE,
        help=(
            "SRT input URL. Use mode=caller with the Shidi host so this "
            "computer initiates the video connection. --video-listen remains "
            "as a compatibility alias."
        ),
    )
    parser.add_argument("--token", default=None)
    parser.add_argument("--video-width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument("--display-scale", type=float, default=0.8)
    parser.add_argument(
        "--fullscreen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start in true fullscreen; press F11 to toggle.",
    )
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--headless-seconds",
        type=float,
        default=0.0,
        help="Run a non-GUI receive/control smoke for the requested duration.",
    )
    args = parser.parse_args()
    if args.video_width <= 0 or args.video_height <= 0:
        raise ValueError("video dimensions must be positive")
    if not 0.25 <= args.display_scale <= 1.5:
        raise ValueError("display scale must be within [0.25, 1.5]")
    if not 5.0 <= args.control_hz <= 60.0:
        raise ValueError("control rate must be within [5, 60] Hz")
    if args.headless_seconds < 0.0:
        raise ValueError("headless duration cannot be negative")
    return args


def fit_video_size(
    source_width: int,
    source_height: int,
    available_width: int,
    available_height: int,
) -> tuple[int, int]:
    """Fit one video frame inside the display without changing its aspect."""

    values = (
        source_width,
        source_height,
        available_width,
        available_height,
    )
    if any(value <= 0 for value in values):
        raise ValueError("video and display dimensions must be positive")
    scale = min(
        available_width / source_width,
        available_height / source_height,
    )
    return (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale)),
    )


class LatestValue:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Any | None = None
        self._sequence = 0

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._sequence += 1

    def get(self) -> tuple[int, Any | None]:
        with self._lock:
            return self._sequence, self._value


class DriverInput:
    """Keyboard state shaped into bounded, human-like control changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._steer = 0.0
        self._last_update = time.monotonic()

    def press(self, key: str) -> bool:
        with self._lock:
            already_pressed = key in self._pressed
            self._pressed.add(key)
            return not already_pressed

    def release(self, key: str) -> None:
        with self._lock:
            self._pressed.discard(key)

    def clear(self) -> None:
        with self._lock:
            self._pressed.clear()
            self._steer = 0.0
            self._last_update = time.monotonic()

    def control(self) -> HumanControl:
        with self._lock:
            now = time.monotonic()
            dt = min(0.1, max(0.0, now - self._last_update))
            self._last_update = now
            left = "a" in self._pressed or "left" in self._pressed
            right = "d" in self._pressed or "right" in self._pressed
            if left == right:
                target_steer = 0.0
            else:
                target_steer = 1.0 if left else -1.0
            maximum_change = 2.4 * dt
            self._steer += max(
                -maximum_change,
                min(maximum_change, target_steer - self._steer),
            )
            throttle_pressed = (
                "w" in self._pressed or "up" in self._pressed
            )
            brake_pressed = (
                "s" in self._pressed or "down" in self._pressed
            )
            throttle = 0.82 if throttle_pressed and not brake_pressed else 0.0
            brake = 1.0 if brake_pressed else 0.0
            return HumanControl(
                steer=self._steer,
                throttle=throttle,
                brake=brake,
            )


class VideoReceiver:
    """Decode the latest SRT H.264 frame through the local FFmpeg binary."""

    def __init__(
        self,
        *,
        ffmpeg: str,
        source: str,
        width: int,
        height: int,
    ) -> None:
        self.ffmpeg = ffmpeg
        self.source = source
        self.width = width
        self.height = height
        self.latest_frame = LatestValue()
        self.status = LatestValue()
        self._stop = threading.Event()
        self._process: subprocess.Popen[bytes] | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="mtgs-video-receiver",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _read_exact(self, stream: Any, length: int) -> bytes | None:
        chunks = bytearray()
        while len(chunks) < length and not self._stop.is_set():
            chunk = stream.read(length - len(chunks))
            if not chunk:
                return None
            chunks.extend(chunk)
        return bytes(chunks) if len(chunks) == length else None

    def _run(self) -> None:
        frame_bytes = self.width * self.height * 3
        while not self._stop.is_set():
            command = [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-fflags",
                "+discardcorrupt",
                "-flags",
                "low_delay",
                "-analyzeduration",
                "500000",
                "-probesize",
                "2000000",
                "-i",
                self.source,
                "-an",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
            try:
                self.status.set("waiting for SRT video")
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    bufsize=0,
                )
                if self._process.stdout is None:
                    raise RuntimeError("FFmpeg did not open decoded video output")
                while not self._stop.is_set():
                    frame = self._read_exact(self._process.stdout, frame_bytes)
                    if frame is None:
                        break
                    self.latest_frame.set(frame)
                    self.status.set("video connected")
                if not self._stop.is_set():
                    self.status.set(
                        f"video disconnected (ffmpeg {self._process.poll()})"
                    )
            except FileNotFoundError:
                self.status.set(f"FFmpeg not found: {self.ffmpeg}")
                return
            except BaseException as error:
                self.status.set(f"video error: {error}")
            finally:
                if self._process is not None:
                    if self._process.poll() is None:
                        self._process.terminate()
                    try:
                        self._process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                    self._process = None
            if not self._stop.wait(1.0):
                continue

    def close(self) -> None:
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        self._thread.join(timeout=3.0)


@dataclass(frozen=True)
class QueuedCommand:
    command: str
    value: str | None


class ControlConnection:
    """Reconnectable WebSocket control and telemetry client."""

    def __init__(
        self,
        *,
        server: str,
        token: str | None,
        control_hz: float,
        driver_input: DriverInput,
    ) -> None:
        self.server = server
        self.token = token
        self.period = 1.0 / control_hz
        self.driver_input = driver_input
        self.telemetry = LatestValue()
        self.session = LatestValue()
        self.status = LatestValue()
        self._commands: queue.Queue[QueuedCommand] = queue.Queue()
        self._sequence = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="mtgs-control-connection",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def command(self, command: str, value: str | None = None) -> None:
        self._commands.put(QueuedCommand(command, value))

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def _send_control(self, connection: Any) -> None:
        packet = RemoteControlPacket(
            sequence=self._next_sequence(),
            client_time_ns=time.monotonic_ns(),
            control=self.driver_input.control(),
        )
        connection.send(encode_remote_message(packet.message()))

    def _send_commands(self, connection: Any) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            packet = RemoteCommandPacket(
                sequence=self._next_sequence(),
                command=command.command,
                value=command.value,
            )
            connection.send(encode_remote_message(packet.message()))

    def _run_connection(self) -> None:
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect

        with connect(
            self.server,
            compression=None,
            proxy=None,
            open_timeout=5.0,
        ) as connection:
            connection.send(
                encode_remote_message(
                    hello_message(role="driver", token=self.token)
                )
            )
            self.status.set("control connected")
            next_control = time.monotonic()
            while not self._stop.is_set():
                self._send_commands(connection)
                now = time.monotonic()
                if now >= next_control:
                    self._send_control(connection)
                    next_control = now + self.period
                timeout = max(0.001, min(0.02, next_control - now))
                try:
                    payload = connection.recv(timeout=timeout)
                except TimeoutError:
                    continue
                except ConnectionClosed:
                    return
                message = decode_remote_message(payload)
                if message["type"] == "session":
                    self.session.set(message)
                elif message["type"] == "telemetry":
                    self.telemetry.set(message)
                elif message["type"] == "error":
                    self.status.set(f"server error: {message.get('detail')}")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.status.set(f"connecting control to {self.server}")
                self._run_connection()
            except BaseException as error:
                self.status.set(f"control disconnected: {error}")
            if not self._stop.wait(1.0):
                continue

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)


class DriverApp:
    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageTk

        self.tk = tk
        self.Image = Image
        self.ImageDraw = ImageDraw
        self.ImageTk = ImageTk
        self.args = args
        self.display_width = round(args.video_width * args.display_scale)
        self.display_height = round(args.video_height * args.display_scale)
        self.root = tk.Tk()
        self.root.title("MTGS Remote Driver")
        self.root.configure(bg="#05080b")
        self._fullscreen = args.fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)
        if not self._fullscreen:
            self.root.geometry(
                f"{self.display_width}x{self.display_height + 48}"
            )
        self.image_label = tk.Label(
            self.root,
            bg="#05080b",
            anchor="center",
        )
        self.image_label.pack(fill="both", expand=True)
        self.help_text = tk.StringVar(
            value=(
                "W/S/A/D 或方向键驾驶 · P 自动 · M 人工 · R 重置 · "
                "Space 急停 · F11 全屏 · Esc 退出"
            )
        )
        self.help_label = tk.Label(
            self.root,
            textvariable=self.help_text,
            bg="#070b0f",
            fg="#dce8ec",
            anchor="w",
            padx=10,
            pady=6,
        )
        self.help_label.pack(fill="x")
        self.driver_input = DriverInput()
        self.video = VideoReceiver(
            ffmpeg=args.ffmpeg,
            source=args.video_source,
            width=args.video_width,
            height=args.video_height,
        )
        self.control = ControlConnection(
            server=args.server,
            token=args.token,
            control_hz=args.control_hz,
            driver_input=self.driver_input,
        )
        self._last_frame_sequence = -1
        self._photo: Any | None = None
        self._discrete_pressed: set[str] = set()
        self._requested_mode = "auto"
        self._closing = False
        self.root.bind_all("<KeyPress>", self._key_press)
        self.root.bind_all("<KeyRelease>", self._key_release)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _normal_key(self, event: Any) -> str:
        key = str(event.keysym).lower()
        aliases = {
            "uparrow": "up",
            "downarrow": "down",
            "leftarrow": "left",
            "rightarrow": "right",
        }
        return aliases.get(key, key)

    def _key_press(self, event: Any) -> None:
        key = self._normal_key(event)
        first_press = key not in self._discrete_pressed
        self._discrete_pressed.add(key)
        if key in DRIVE_KEYS:
            self.driver_input.press(key)
            if self._requested_mode != "remote":
                self.control.command("mode", "remote")
                self._requested_mode = "remote"
        if not first_press:
            return
        if key == "p":
            self.control.command("mode", "auto")
            self._requested_mode = "auto"
            self.driver_input.clear()
        elif key == "m":
            self.control.command("mode", "remote")
            self._requested_mode = "remote"
        elif key == "r":
            self.control.command("reset")
            self.driver_input.clear()
        elif key == "space":
            self.control.command("estop")
            self.driver_input.clear()
        elif key == "tab":
            target = "remote" if self._requested_mode == "auto" else "auto"
            self.control.command("mode", target)
            self._requested_mode = target
            if target == "auto":
                self.driver_input.clear()
        elif key == "f11":
            self._toggle_fullscreen()
        elif key == "escape":
            self.close()

    def _key_release(self, event: Any) -> None:
        key = self._normal_key(event)
        self._discrete_pressed.discard(key)
        if key in DRIVE_KEYS:
            self.driver_input.release(key)

    def _placeholder(self, detail: str) -> Any:
        image = self.Image.new(
            "RGB",
            (self.args.video_width, self.args.video_height),
            (5, 9, 12),
        )
        draw = self.ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.args.video_width - 1, self.args.video_height - 1), outline=(65, 83, 94), width=2)
        draw.text((40, 42), "MTGS REMOTE DRIVER", fill=(220, 235, 240))
        draw.text((40, 78), detail, fill=(255, 216, 77))
        draw.text(
            (40, 112),
            "This app initiates both control and video connections to the Shidi simulator.",
            fill=(145, 166, 178),
        )
        return image

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)
        if not self._fullscreen:
            self.root.geometry(
                f"{self.display_width}x{self.display_height + 48}"
            )

    def _fit_for_display(self, image: Any) -> Any:
        available_width = self.root.winfo_width()
        available_height = (
            self.root.winfo_height() - self.help_label.winfo_height()
        )
        if available_width <= 1:
            available_width = self.display_width
        if available_height <= 1:
            available_height = self.display_height
        target = fit_video_size(
            self.args.video_width,
            self.args.video_height,
            available_width,
            available_height,
        )
        if image.size != target:
            image = image.resize(target, self.Image.Resampling.BILINEAR)
        return image

    def _update(self) -> None:
        if self._closing:
            return
        sequence, frame = self.video.latest_frame.get()
        video_status = self.video.status.get()[1] or "video starting"
        control_status = self.control.status.get()[1] or "control starting"
        if frame is not None and sequence != self._last_frame_sequence:
            image = self.Image.frombytes(
                "RGB",
                (self.args.video_width, self.args.video_height),
                frame,
            )
            self._last_frame_sequence = sequence
        elif self._photo is None:
            image = self._placeholder(str(video_status))
        else:
            image = None
        if image is not None:
            image = self._fit_for_display(image)
            self._photo = self.ImageTk.PhotoImage(image)
            self.image_label.configure(image=self._photo)

        telemetry = self.control.telemetry.get()[1]
        if telemetry is not None:
            mode = telemetry.get("effective_mode", "-")
            speed = telemetry.get("state", {}).get("speed_mps", 0.0)
            offset = telemetry.get("route", {}).get("lateral_meters", 0.0)
            self.root.title(
                f"MTGS Remote Driver · {mode.upper()} · "
                f"{speed:.1f} m/s · {offset:+.2f} m"
            )
        self.help_text.set(
            f"{control_status} · {video_status}  |  "
            "W/S/A/D 驾驶 · P 自动 · M 人工 · R 重置 · "
            "Space 急停 · F11 全屏"
        )
        self.root.after(16, self._update)

    def run(self) -> None:
        self.video.start()
        self.control.start()
        self.root.after(16, self._update)
        self.root.mainloop()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.driver_input.clear()
        self.control.command("estop")
        self.control.close()
        self.video.close()
        self.root.destroy()


def run_headless(args: argparse.Namespace) -> None:
    """Exercise the same SRT/WebSocket clients without opening a window."""

    driver_input = DriverInput()
    video = VideoReceiver(
        ffmpeg=args.ffmpeg,
        source=args.video_source,
        width=args.video_width,
        height=args.video_height,
    )
    control = ControlConnection(
        server=args.server,
        token=args.token,
        control_hz=args.control_hz,
        driver_input=driver_input,
    )
    video.start()
    control.start()
    wait_started = time.monotonic()
    while (
        control.telemetry.get()[0] == 0
        and time.monotonic() - wait_started < 60.0
    ):
        time.sleep(0.05)
    if control.telemetry.get()[0] == 0:
        control.close()
        video.close()
        raise RuntimeError("simulator sent no telemetry within 60 seconds")
    started = time.monotonic()
    takeover_started = False
    takeover_finished = False
    observed_modes: set[str] = set()
    last_telemetry_sequence = -1
    maximum_abs_applied_steer = 0.0
    try:
        while time.monotonic() - started < args.headless_seconds:
            elapsed = time.monotonic() - started
            telemetry_sequence, telemetry = control.telemetry.get()
            if (
                telemetry is not None
                and telemetry_sequence != last_telemetry_sequence
            ):
                last_telemetry_sequence = telemetry_sequence
                mode = telemetry.get("effective_mode")
                if isinstance(mode, str):
                    observed_modes.add(mode)
                maximum_abs_applied_steer = max(
                    maximum_abs_applied_steer,
                    abs(
                        float(
                            telemetry.get("applied_control", {}).get(
                                "steer",
                                0.0,
                            )
                        )
                    ),
                )
            if elapsed >= args.headless_seconds * 0.55 and not takeover_started:
                control.command("mode", "remote")
                driver_input.press("w")
                driver_input.press("a")
                takeover_started = True
            if elapsed >= args.headless_seconds * 0.70 and not takeover_finished:
                driver_input.clear()
                control.command("mode", "auto")
                takeover_finished = True
            time.sleep(0.02)
    finally:
        driver_input.clear()
        control.command("estop")
        time.sleep(0.1)
        control.close()
        video.close()
    video_frames = video.latest_frame.get()[0]
    telemetry_frames = control.telemetry.get()[0]
    print(f"video frames received: {video_frames}")
    print(f"telemetry messages received: {telemetry_frames}")
    print(f"observed modes: {sorted(observed_modes)}")
    print(f"maximum absolute applied steer: {maximum_abs_applied_steer:.3f}")
    print(f"video status: {video.status.get()[1]}")
    print(f"control status: {control.status.get()[1]}")
    if video_frames == 0 or telemetry_frames == 0:
        raise RuntimeError("headless remote-app smoke received no complete stream")
    if not {"auto", "remote"}.issubset(observed_modes):
        raise RuntimeError("headless remote-app smoke missed AUTO/REMOTE takeover")
    if maximum_abs_applied_steer <= 0.0:
        raise RuntimeError("headless remote-app smoke applied no steering")


def main() -> None:
    args = parse_args()
    if args.headless_seconds > 0.0:
        run_headless(args)
        return
    app = DriverApp(args)
    app.run()


if __name__ == "__main__":
    main()
