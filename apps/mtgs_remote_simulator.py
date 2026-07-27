#!/usr/bin/env python3
"""Authoritative MTGS simulator/video server for the two-computer demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hmac
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from driving_scene_reconstruction.sim import HumanControl  # noqa: E402
from driving_scene_reconstruction.sim.remote_protocol import (  # noqa: E402
    REMOTE_CONTROL_TIMEOUT_SECONDS,
    REMOTE_PROTOCOL_VERSION,
    RemoteCommandPacket,
    RemoteControlPacket,
    RemoteControlWatchdog,
    encode_remote_message,
    error_message,
    parse_driver_message,
)
from run_stage_h3_mtgs_autodrive import (  # noqa: E402
    camera_right_to_route_left_alignment,
)
from run_stage_h3_mtgs_continuous_drive import (  # noqa: E402
    SOURCE_COMMIT,
    configure_for_inference,
    cumulative_route_distances,
    sample_route_pose,
)
from stage_h3_mtgs_autodriver import (  # noqa: E402
    MtgsAutodriver,
    MtgsAutodriverConfig,
)
from stage_h3_mtgs_driving_adapter import (  # noqa: E402
    MTGS_CORRIDOR_HALF_WIDTH_METERS,
    make_mtgs_driving_adapter,
)
from stage_h3_tbv_driving_adapter import (  # noqa: E402
    COCKPIT_HORIZONTAL_FOV_DEGREES,
    COCKPIT_TOP_ANGLE_DEGREES,
    CameraProjection,
    CylindricalCockpitComposer,
)


TRAVEL_ID = 3
DEFAULT_FPS = 20
DEFAULT_CRUISE_SPEED_MPS = 12.0
DEFAULT_LATERAL_AMPLITUDE_METERS = 3.0
DEFAULT_CONTROL_PORT = 18765
DEFAULT_VIDEO_DESTINATION = (
    "srt://0.0.0.0:19001?mode=listener&latency=80&transtype=live"
)
REMOTE_COCKPIT_WIDTH = 1280
REMOTE_COCKPIT_VIEW_HEIGHT = 496
REMOTE_COCKPIT_STATUS_HEIGHT = 48
MTGS_COCKPIT_BOTTOM_ANGLE_DEGREES = -19.0
INSET_SIZE = 240
CAMERA_ALIASES = (
    ("ring_front_left", "CAM_L0"),
    ("ring_front_center", "CAM_F0"),
    ("ring_front_right", "CAM_R0"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--road-block-config", type=Path, required=True)
    parser.add_argument("--control-host", default="0.0.0.0")
    parser.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_PORT)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--video-destination",
        default=None,
        help=(
            "FFmpeg output URL; when the simulator cannot reach the driver "
            "computer, listen on "
            "srt://0.0.0.0:19001?mode=listener&latency=80&transtype=live"
        ),
    )
    destination.add_argument(
        "--record",
        type=Path,
        help="Record a local MP4 instead of opening a live video connection.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--cruise-speed-mps",
        type=float,
        default=DEFAULT_CRUISE_SPEED_MPS,
    )
    parser.add_argument(
        "--lateral-amplitude-meters",
        type=float,
        default=DEFAULT_LATERAL_AMPLITUDE_METERS,
    )
    parser.add_argument("--start-mode", choices=("auto", "remote"), default="auto")
    parser.add_argument(
        "--token",
        default=None,
        help="Optional shared control token; use only on a trusted LAN or VPN.",
    )
    parser.add_argument(
        "--encoder",
        choices=("h264_nvenc", "libx264"),
        default="h264_nvenc",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames; zero keeps the server running.",
    )
    parser.add_argument(
        "--no-control-server",
        action="store_true",
        help="Offline evidence mode: stay in AUTO and do not open WebSocket.",
    )
    args = parser.parse_args()
    if args.video_destination is None and args.record is None:
        args.video_destination = DEFAULT_VIDEO_DESTINATION
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for path in (args.config, args.checkpoint, args.road_block_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.control_port <= 0 or args.control_port > 65535:
        raise ValueError("control port must be within [1, 65535]")
    if args.max_frames < 0:
        raise ValueError("max frames cannot be negative")
    if (
        not math.isfinite(args.cruise_speed_mps)
        or not 0.0 < args.cruise_speed_mps <= 15.0
    ):
        raise ValueError("cruise speed must be within (0, 15] m/s")
    if (
        not math.isfinite(args.lateral_amplitude_meters)
        or not 0.0 < args.lateral_amplitude_meters < 5.0
    ):
        raise ValueError("lateral amplitude must remain inside +/-5 m support")
    if args.no_control_server and args.start_mode != "auto":
        raise ValueError("offline mode requires --start-mode auto")
    if args.record is not None:
        args.record = args.record.expanduser().resolve()
        if args.record.suffix.lower() != ".mp4":
            raise ValueError("record path must end in .mp4")
        if args.record.exists():
            raise RuntimeError(f"refusing to overwrite existing video: {args.record}")


class RemoteAuthority:
    """Thread-safe owner of mode, latest control, and safety state."""

    def __init__(self, start_mode: str) -> None:
        self._lock = threading.Lock()
        self._watchdog = RemoteControlWatchdog()
        self._requested_mode = start_mode
        self._estop_latched = False
        self._reset_requested = False
        self._last_command_sequence = -1
        self._client_connected = False

    def set_client_connected(self, connected: bool) -> None:
        with self._lock:
            self._client_connected = connected

    def accept(
        self,
        packet: RemoteControlPacket | RemoteCommandPacket,
    ) -> bool:
        with self._lock:
            if isinstance(packet, RemoteControlPacket):
                return self._watchdog.accept(packet)
            if packet.sequence <= self._last_command_sequence:
                return False
            self._last_command_sequence = packet.sequence
            if packet.command == "mode":
                assert packet.value is not None
                self._requested_mode = packet.value
            elif packet.command == "reset":
                self._reset_requested = True
                self._estop_latched = False
                self._watchdog.latest_packet = None
                self._watchdog.latest_arrival_ns = None
            elif packet.command == "estop":
                self._estop_latched = True
            return True

    def consume_reset(self) -> bool:
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested

    def applied_control(
        self,
        auto_control: HumanControl,
    ) -> tuple[HumanControl, str, float | None, bool, bool]:
        with self._lock:
            client_connected = self._client_connected
            if self._estop_latched:
                return (
                    HumanControl(brake=1.0),
                    "estop",
                    None,
                    True,
                    client_connected,
                )
            if self._requested_mode == "auto":
                return auto_control, "auto", None, False, client_connected
            control, age_ms, stale = self._watchdog.control()
            effective_mode = "safe-stop" if stale else "remote"
            return control, effective_mode, age_ms, stale, client_connected

    @property
    def requested_mode(self) -> str:
        with self._lock:
            return self._requested_mode


class ControlGateway:
    """Single-driver WebSocket endpoint with telemetry return."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        token: str | None,
        authority: RemoteAuthority,
        session_message: dict[str, Any],
    ) -> None:
        self.host = host
        self.port = port
        self.token = token
        self.authority = authority
        self.session_message = session_message
        self._telemetry_lock = threading.Lock()
        self._telemetry: dict[str, Any] | None = None
        self._telemetry_sequence = -1
        self._client_lock = threading.Lock()
        self._client_active = False
        self._server: Any | None = None
        self._thread: threading.Thread | None = None

    def publish(self, telemetry: dict[str, Any]) -> None:
        with self._telemetry_lock:
            self._telemetry = telemetry
            self._telemetry_sequence = int(telemetry["frame_sequence"])

    def _latest_telemetry(
        self,
    ) -> tuple[int, dict[str, Any] | None]:
        with self._telemetry_lock:
            return self._telemetry_sequence, self._telemetry

    def _handler(self, connection: Any) -> None:
        from websockets.exceptions import ConnectionClosed

        with self._client_lock:
            if self._client_active:
                connection.send(
                    encode_remote_message(error_message("driver already connected"))
                )
                connection.close()
                return
            self._client_active = True
        self.authority.set_client_connected(True)
        try:
            try:
                first = parse_driver_message(connection.recv(timeout=5.0))
            except (TimeoutError, ValueError) as error:
                connection.send(encode_remote_message(error_message(str(error))))
                return
            if not isinstance(first, dict) or first.get("type") != "hello":
                connection.send(
                    encode_remote_message(error_message("hello must be first"))
                )
                return
            supplied_token = first.get("token")
            if self.token is not None and (
                not isinstance(supplied_token, str)
                or not hmac.compare_digest(supplied_token, self.token)
            ):
                connection.send(
                    encode_remote_message(error_message("invalid control token"))
                )
                return
            connection.send(encode_remote_message(self.session_message))
            last_sent_telemetry = -1
            while True:
                sequence, telemetry = self._latest_telemetry()
                if telemetry is not None and sequence != last_sent_telemetry:
                    connection.send(encode_remote_message(telemetry))
                    last_sent_telemetry = sequence
                try:
                    payload = connection.recv(timeout=0.02)
                except TimeoutError:
                    continue
                try:
                    packet = parse_driver_message(payload)
                    if isinstance(packet, dict):
                        raise ValueError("hello was already accepted")
                    accepted = self.authority.accept(packet)
                    if isinstance(packet, RemoteCommandPacket):
                        connection.send(
                            encode_remote_message(
                                {
                                    "type": "ack",
                                    "version": REMOTE_PROTOCOL_VERSION,
                                    "sequence": packet.sequence,
                                    "accepted": accepted,
                                }
                            )
                        )
                except ValueError as error:
                    connection.send(
                        encode_remote_message(error_message(str(error)))
                    )
        except ConnectionClosed:
            pass
        finally:
            self.authority.set_client_connected(False)
            with self._client_lock:
                self._client_active = False

    def start(self) -> None:
        from websockets.sync.server import serve

        ready = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            try:
                with serve(
                    self._handler,
                    self.host,
                    self.port,
                    compression=None,
                ) as server:
                    self._server = server
                    ready.set()
                    server.serve_forever()
            except BaseException as error:
                failure.append(error)
                ready.set()

        self._thread = threading.Thread(
            target=run,
            name="mtgs-control-gateway",
            daemon=True,
        )
        self._thread.start()
        if not ready.wait(timeout=5.0):
            raise RuntimeError("timed out opening the control endpoint")
        if failure:
            raise RuntimeError("failed to open the control endpoint") from failure[0]

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class FfmpegVideoSink:
    """Encode RGB cockpit frames to an MP4 or a low-latency SRT stream."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        encoder: str,
        destination: str | None,
        record: Path | None,
    ) -> None:
        if (destination is None) == (record is None):
            raise ValueError("exactly one video destination is required")
        self.width = width
        self.height = height
        self.fps = fps
        self.is_live = destination is not None
        self.restart_count = 0
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            encoder,
        ]
        if encoder == "h264_nvenc":
            command.extend(
                [
                    "-preset",
                    "p1",
                    "-tune",
                    "ull",
                    "-rc",
                    "cbr",
                    "-b:v",
                    "12M",
                    "-maxrate",
                    "12M",
                    "-bufsize",
                    "2M",
                    "-zerolatency",
                    "1",
                ]
            )
        else:
            command.extend(
                ["-preset", "ultrafast", "-tune", "zerolatency", "-crf", "18"]
            )
        command.extend(
            [
                "-g",
                str(fps),
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
            ]
        )
        if record is not None:
            record.parent.mkdir(parents=True, exist_ok=True)
            command.extend(["-movflags", "+faststart", str(record)])
            self.output = str(record)
        else:
            assert destination is not None
            command.extend(
                [
                    "-muxdelay",
                    "0",
                    "-muxpreload",
                    "0",
                    "-f",
                    "mpegts",
                    destination,
                ]
            )
            self.output = destination
        self._command = command
        self.process: subprocess.Popen[bytes] | None = None
        self._start_process()

    def _start_process(self) -> None:
        self.process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            bufsize=0,
        )

    def _discard_process(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2.0)

    def write(self, frame: Any) -> None:
        if tuple(frame.shape) != (self.height, self.width, 3):
            raise RuntimeError(
                f"video frame is {tuple(frame.shape)}, expected "
                f"{(self.height, self.width, 3)}"
            )
        payload = frame.tobytes()
        for attempt in range(2):
            assert self.process is not None
            if self.process.poll() is not None:
                if not self.is_live or attempt > 0:
                    raise RuntimeError(
                        "video encoder exited unexpectedly "
                        f"({self.process.returncode})"
                    )
                self.restart_count += 1
                self._discard_process()
                self._start_process()
            assert self.process is not None
            if self.process.stdin is None:
                raise RuntimeError("video encoder input is closed")
            try:
                self.process.stdin.write(payload)
                return
            except BrokenPipeError as error:
                return_code = self.process.poll()
                if not self.is_live or attempt > 0:
                    raise RuntimeError(
                        f"video encoder exited unexpectedly ({return_code})"
                    ) from error
                self.restart_count += 1
                self._discard_process()
                self._start_process()
        raise RuntimeError("video encoder reconnect failed")

    def close(self) -> None:
        assert self.process is not None
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            return_code = self.process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            if not self.is_live:
                raise
            self.process.terminate()
            try:
                return_code = self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                return_code = self.process.wait(timeout=2.0)
        if return_code != 0 and not self.is_live:
            raise RuntimeError(f"video encoder exited with {return_code}")


def _tensor_scalar(value: Any) -> float:
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _homogeneous(pose: Any, np: Any) -> Any:
    result = np.eye(4, dtype=np.float32)
    result[:3, :4] = pose
    return result


@dataclass
class RenderResult:
    panorama: Any
    render_seconds: float
    compose_seconds: float


class MtgsRuntime:
    """Loaded MTGS model, route support, and calibrated three-camera rig."""

    def __init__(self, args: argparse.Namespace) -> None:
        import numpy as np
        import torch
        import yaml
        from nerfstudio.utils.eval_utils import eval_load_checkpoint

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible in the actual MTGS environment")
        self.np = np
        self.torch = torch
        self.device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        setup_started = time.perf_counter()
        config = yaml.load(args.config.read_text(), Loader=yaml.Loader)
        configure_for_inference(config, args)
        self.pipeline = config.pipeline.setup(device=self.device, test_mode="test")
        self.pipeline.eval()
        checkpoint_path, checkpoint_step = eval_load_checkpoint(
            config,
            self.pipeline,
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_step = int(checkpoint_step)
        self.setup_seconds = time.perf_counter() - setup_started

        outputs = self.pipeline.datamanager.eval_dataparser_outputs
        self.outputs = outputs
        front_indices = [
            index
            for index, (travel_id, image_path) in enumerate(
                zip(outputs.travel_ids, outputs.image_filenames)
            )
            if travel_id == TRAVEL_ID and image_path.parent.name == "CAM_F0"
        ]
        if len(front_indices) < 2:
            raise RuntimeError("selected MTGS traversal has no usable front route")
        self.route_poses = (
            outputs.cameras.camera_to_worlds[front_indices]
            .detach()
            .cpu()
            .numpy()
        )
        self.route_distances = cumulative_route_distances(self.route_poses, np)
        self.adapter = make_mtgs_driving_adapter(
            self.route_poses,
            self.route_distances,
            spawn_progress_meters=5.0,
        )
        self.driver = MtgsAutodriver(
            self.adapter.controller.vehicle_model,
            MtgsAutodriverConfig(
                cruise_speed_mps=args.cruise_speed_mps,
                lateral_amplitude_meters=args.lateral_amplitude_meters,
                spawn_progress_meters=5.0,
                endpoint_margin_meters=self.adapter.endpoint_margin_meters,
            ),
        )
        self.route_scale = (
            self.route_distances[-1] / self.adapter.controller.corridor.length
        )

        centre_index = front_indices[0]
        centre_time = _tensor_scalar(outputs.cameras.times[centre_index])
        rig_indices: dict[str, int] = {}
        time_gaps: dict[str, float] = {}
        for alias, camera_name in CAMERA_ALIASES:
            candidates = [
                index
                for index, (travel_id, image_path) in enumerate(
                    zip(outputs.travel_ids, outputs.image_filenames)
                )
                if travel_id == TRAVEL_ID
                and image_path.parent.name == camera_name
            ]
            index = min(
                candidates,
                key=lambda candidate: abs(
                    _tensor_scalar(outputs.cameras.times[candidate]) - centre_time
                ),
            )
            rig_indices[alias] = index
            time_gaps[alias] = abs(
                _tensor_scalar(outputs.cameras.times[index]) - centre_time
            )
        self.rig_time_gaps = time_gaps
        self.anchor_poses = {
            alias: _homogeneous(
                outputs.cameras.camera_to_worlds[index]
                .detach()
                .cpu()
                .numpy(),
                np,
            )
            for alias, index in rig_indices.items()
        }
        self.anchor_centre_inverse = np.linalg.inv(
            self.anchor_poses["ring_front_center"]
        ).astype(np.float32)

        self.cameras: dict[str, Any] = {}
        for alias, index in rig_indices.items():
            _, camera = self.pipeline.datamanager.cached_eval(index)
            camera = camera.to(self.device)
            if camera.times is not None:
                camera.times = torch.full_like(camera.times, centre_time)
            self.cameras[alias] = camera

        centre_pose = self.anchor_poses["ring_front_center"]
        centre_rotation = centre_pose[:3, :3]
        local_to_world = np.column_stack(
            (
                -centre_rotation[:, 2],
                -centre_rotation[:, 0],
                centre_rotation[:, 1],
            )
        )
        centre_origin = centre_pose[:3, 3]
        projections = []
        for alias, _ in CAMERA_ALIASES:
            camera = self.cameras[alias]
            pose = self.anchor_poses[alias]
            projections.append(
                CameraProjection(
                    name=alias,
                    width=int(round(_tensor_scalar(camera.width))),
                    height=int(round(_tensor_scalar(camera.height))),
                    fx=_tensor_scalar(camera.fx),
                    fy=_tensor_scalar(camera.fy),
                    cx=_tensor_scalar(camera.cx),
                    cy=_tensor_scalar(camera.cy),
                    local_to_camera=tuple(
                        tuple(float(value) for value in row)
                        for row in pose[:3, :3].T @ local_to_world
                    ),
                    camera_center_local=tuple(
                        float(value)
                        for value in local_to_world.T
                        @ (pose[:3, 3] - centre_origin)
                    ),
                )
            )
        self.composer = CylindricalCockpitComposer(
            {"mtgs": tuple(projections)},
            width=REMOTE_COCKPIT_WIDTH,
            height=REMOTE_COCKPIT_VIEW_HEIGHT,
            horizontal_fov_degrees=COCKPIT_HORIZONTAL_FOV_DEGREES,
            top_angle_degrees=COCKPIT_TOP_ANGLE_DEGREES,
            bottom_angle_degrees=MTGS_COCKPIT_BOTTOM_ANGLE_DEGREES,
        )
        self.coverage_fraction = self.composer.coverage_fraction["mtgs"]

    def reset(self) -> Any:
        self.driver.reset()
        return self.adapter.reset()

    def auto_decision(self, state: Any, dt: float) -> Any:
        return self.driver.decide(
            state,
            self.adapter.controller.corridor,
            dt,
        )

    def render(self, query: Any) -> RenderResult:
        np = self.np
        torch = self.torch
        render_progress = min(
            self.route_distances[-1],
            float(query.progress_meters) * self.route_scale,
        )
        alignment = camera_right_to_route_left_alignment(
            self.route_poses,
            self.route_distances,
            render_progress,
            np,
        )
        camera_lateral = float(query.lateral_meters) / alignment
        camera_slope = math.tan(
            float(query.heading_error_radians)
        ) / alignment
        centre_pose, _ = sample_route_pose(
            self.route_poses,
            self.route_distances,
            render_progress,
            camera_lateral,
            camera_slope,
            np,
        )
        requested_centre = _homogeneous(centre_pose, np)
        delta = requested_centre @ self.anchor_centre_inverse
        frames: dict[str, Any] = {}
        torch.cuda.synchronize()
        render_started = time.perf_counter()
        with torch.inference_mode():
            for alias, _ in CAMERA_ALIASES:
                requested_pose = delta @ self.anchor_poses[alias]
                camera = self.cameras[alias]
                camera.camera_to_worlds = torch.from_numpy(
                    requested_pose[:3, :4].astype(np.float32)
                )[None].to(self.device)
                outputs = self.pipeline.model.get_outputs_for_camera(camera=camera)
                rgb = outputs["rgb"].detach().float().cpu().numpy()
                frames[alias] = np.clip(
                    rgb * 255.0,
                    0.0,
                    255.0,
                ).astype(np.uint8)
        torch.cuda.synchronize()
        render_seconds = time.perf_counter() - render_started
        compose_started = time.perf_counter()
        panorama = self.composer.compose("mtgs", frames)
        compose_seconds = time.perf_counter() - compose_started
        return RenderResult(
            panorama=panorama,
            render_seconds=render_seconds,
            compose_seconds=compose_seconds,
        )


def _project_route_point(
    state: Any,
    x: float,
    y: float,
    z: float = 0.0,
) -> tuple[int, int]:
    cosine = math.cos(state.yaw)
    sine = math.sin(state.yaw)
    dx, dy = x - state.x, y - state.y
    forward = dx * cosine + dy * sine
    left = -dx * sine + dy * cosine
    screen_x = INSET_SIZE * 0.43 - left * 10.0 + forward * 1.65
    screen_y = INSET_SIZE - 48.0 - forward * 4.25 - left * 1.15 - z * 12.0
    return int(round(screen_x)), int(round(screen_y))


def make_route_support_inset(
    image_module: Any,
    draw_module: Any,
    runtime: MtgsRuntime,
    state: Any,
    query: Any,
    *,
    effective_mode: str,
) -> Any:
    """Draw a truthful oblique route/corridor view and the vehicle model."""

    canvas = image_module.new("RGB", (INSET_SIZE, INSET_SIZE), (11, 17, 22))
    draw = draw_module.Draw(canvas)
    corridor = runtime.adapter.controller.corridor

    for forward in (0.0, 10.0, 20.0, 30.0, 40.0):
        cosine = math.cos(state.yaw)
        sine = math.sin(state.yaw)
        points = []
        for left in (-8.0, 8.0):
            x = state.x + cosine * forward - sine * left
            y = state.y + sine * forward + cosine * left
            points.append(_project_route_point(state, x, y))
        draw.line(points, fill=(31, 43, 50), width=1)

    left_edge = []
    right_edge = []
    centre = []
    for sample in corridor.samples:
        dx, dy = sample.x - state.x, sample.y - state.y
        forward = dx * math.cos(state.yaw) + dy * math.sin(state.yaw)
        if not -10.0 <= forward <= 52.0:
            continue
        left_x = sample.x - math.sin(sample.yaw) * corridor.half_width
        left_y = sample.y + math.cos(sample.yaw) * corridor.half_width
        right_x = sample.x + math.sin(sample.yaw) * corridor.half_width
        right_y = sample.y - math.cos(sample.yaw) * corridor.half_width
        left_edge.append(_project_route_point(state, left_x, left_y))
        right_edge.append(_project_route_point(state, right_x, right_y))
        centre.append(_project_route_point(state, sample.x, sample.y))
    if len(left_edge) >= 2:
        draw.polygon(
            tuple(left_edge + list(reversed(right_edge))),
            fill=(46, 57, 63),
        )
        draw.line(left_edge, fill=(106, 188, 248), width=2, joint="curve")
        draw.line(right_edge, fill=(106, 188, 248), width=2, joint="curve")
        draw.line(centre, fill=(236, 225, 115), width=2, joint="curve")

    def local_point(
        forward: float,
        left: float,
        up: float,
    ) -> tuple[int, int]:
        cosine = math.cos(state.yaw)
        sine = math.sin(state.yaw)
        x = state.x + cosine * forward - sine * left
        y = state.y + sine * forward + cosine * left
        return _project_route_point(state, x, y, up)

    body_bottom = tuple(
        local_point(forward, left, 0.12)
        for forward, left in (
            (2.35, 1.0),
            (2.35, -1.0),
            (-2.35, -1.0),
            (-2.35, 1.0),
        )
    )
    body_roof = tuple(
        local_point(forward, left, 1.35)
        for forward, left in (
            (1.45, 0.82),
            (1.45, -0.82),
            (-1.35, -0.82),
            (-1.35, 0.82),
        )
    )
    for index, colour in enumerate(
        ((31, 38, 45), (25, 31, 38), (38, 46, 53), (28, 35, 42))
    ):
        next_index = (index + 1) % 4
        draw.polygon(
            (
                body_bottom[index],
                body_bottom[next_index],
                body_roof[next_index],
                body_roof[index],
            ),
            fill=colour,
            outline=(170, 185, 194),
        )
    draw.polygon(
        body_roof,
        fill=(48, 58, 66),
        outline=(232, 237, 238),
    )
    margin = float(query.support_margin_meters)
    model_colour = (255, 79, 70) if margin < 0.5 else (94, 235, 149)
    draw.line(
        (local_point(0.0, 0.0, 1.40), local_point(2.4, 0.0, 1.40)),
        fill=model_colour,
        width=4,
    )

    draw.rectangle((0, 0, INSET_SIZE - 1, INSET_SIZE - 1), outline=(108, 124, 134), width=2)
    draw.rectangle((1, 1, INSET_SIZE - 2, 45), fill=(7, 11, 15))
    draw.text((10, 8), "3D DRIVE SUPPORT  +/-5 m", fill=(238, 242, 230))
    draw.text(
        (10, 26),
        "route geometry · kinematic car",
        fill=(150, 172, 182),
    )
    draw.rectangle(
        (1, INSET_SIZE - 31, INSET_SIZE - 2, INSET_SIZE - 2),
        fill=(7, 11, 15),
    )
    draw.text(
        (10, INSET_SIZE - 24),
        (
            f"{effective_mode.upper()}  offset {query.lateral_meters:+.2f} m  "
            f"margin {margin:.2f} m"
        ),
        fill=model_colour,
    )
    return canvas


def make_cockpit_frame(
    image_module: Any,
    draw_module: Any,
    runtime: MtgsRuntime,
    render: RenderResult,
    state: Any,
    query: Any,
    *,
    effective_mode: str,
    requested_mode: str,
    applied_control: HumanControl,
    control_age_ms: float | None,
    client_connected: bool,
    boundary_hit: bool,
) -> Any:
    panorama = image_module.fromarray(render.panorama).convert("RGB")
    canvas = image_module.new(
        "RGB",
        (
            REMOTE_COCKPIT_WIDTH,
            REMOTE_COCKPIT_VIEW_HEIGHT + REMOTE_COCKPIT_STATUS_HEIGHT,
        ),
        (4, 6, 8),
    )
    canvas.paste(panorama, (0, 0))
    inset = make_route_support_inset(
        image_module,
        draw_module,
        runtime,
        state,
        query,
        effective_mode=effective_mode,
    )
    canvas.paste(inset, (REMOTE_COCKPIT_WIDTH - INSET_SIZE - 14, 14))
    draw = draw_module.Draw(canvas)
    draw.rectangle((8, 8, 385, 60), fill=(5, 9, 12))
    draw.text(
        (18, 16),
        f"FORWARD SURROUND  {COCKPIT_HORIZONTAL_FOV_DEGREES:.0f} DEG",
        fill=(220, 235, 240),
    )
    draw.text(
        (18, 38),
        "MTGS 3-camera calibrated rig · fixed scene time",
        fill=(143, 166, 179),
    )
    age_label = "-" if control_age_ms is None else f"{control_age_ms:.0f}ms"
    link_label = "LINK" if client_connected else "LOCAL"
    warning = " BOUNDARY" if boundary_hit else ""
    status = (
        f"MTGS remote | {link_label} | mode={effective_mode}"
        f"(requested {requested_mode}){warning} | speed={state.speed:.1f}m/s | "
        f"progress={query.progress_meters:.1f}m | offset={query.lateral_meters:+.2f}m | "
        f"steer={applied_control.steer:+.2f} throttle={applied_control.throttle:.2f} "
        f"brake={applied_control.brake:.2f} | control-age={age_label} | "
        f"render={render.render_seconds * 1000.0:.1f}ms"
    )
    status_colour = (
        (255, 92, 78)
        if effective_mode in {"safe-stop", "estop"} or boundary_hit
        else (255, 216, 77)
    )
    draw.text(
        (12, REMOTE_COCKPIT_VIEW_HEIGHT + 15),
        status,
        fill=status_colour,
    )
    return canvas


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    args = parse_args()
    _validate_args(args)
    os.environ.setdefault(
        "NERFSTUDIO_DATAPARSER_CONFIGS",
        "nuplan=mtgs.config.nuplan_dataparser:nuplan_dataparser",
    )
    os.environ.setdefault(
        "NERFSTUDIO_METHOD_CONFIGS",
        "mtgs=mtgs.config.MTGS:method",
    )

    import numpy as np
    from PIL import Image, ImageDraw

    runtime = MtgsRuntime(args)
    authority = RemoteAuthority(args.start_mode)
    session_message = {
        "type": "session",
        "version": REMOTE_PROTOCOL_VERSION,
        "scene": "MTGS road_block-365000_144000_365100_144080 travel 3",
        "fps": args.fps,
        "video_width": REMOTE_COCKPIT_WIDTH,
        "video_height": (
            REMOTE_COCKPIT_VIEW_HEIGHT + REMOTE_COCKPIT_STATUS_HEIGHT
        ),
        "forward_fov_degrees": COCKPIT_HORIZONTAL_FOV_DEGREES,
        "corridor_half_width_meters": MTGS_CORRIDOR_HALF_WIDTH_METERS,
        "route_length_meters": runtime.adapter.controller.corridor.length,
        "control_timeout_ms": REMOTE_CONTROL_TIMEOUT_SECONDS * 1000.0,
        "controls": {
            "drive": "W/S/A/D or arrow keys",
            "auto": "P",
            "remote": "M or any drive key",
            "reset": "R",
            "estop": "Space",
        },
    }
    gateway = None
    if not args.no_control_server:
        gateway = ControlGateway(
            host=args.control_host,
            port=args.control_port,
            token=args.token,
            authority=authority,
            session_message=session_message,
        )
        gateway.start()
        print(
            f"control: ws://{args.control_host}:{args.control_port} "
            f"(protocol v{REMOTE_PROTOCOL_VERSION})",
            flush=True,
        )

    sink = FfmpegVideoSink(
        width=REMOTE_COCKPIT_WIDTH,
        height=REMOTE_COCKPIT_VIEW_HEIGHT + REMOTE_COCKPIT_STATUS_HEIGHT,
        fps=args.fps,
        encoder=args.encoder,
        destination=args.video_destination,
        record=args.record,
    )
    print(f"video: {sink.output}", flush=True)
    print(
        f"checkpoint: step {runtime.checkpoint_step} | "
        f"front coverage {runtime.coverage_fraction:.1%} | "
        f"GPU {runtime.torch.cuda.get_device_name(runtime.device)}",
        flush=True,
    )

    dt = 1.0 / args.fps
    state = runtime.reset()
    render_times: list[float] = []
    compose_times: list[float] = []
    frame_records: list[dict[str, Any]] = []
    frame_sequence = 0
    auto_endpoint_hold = 0
    deadline = time.perf_counter()
    failure: BaseException | None = None
    try:
        while args.max_frames == 0 or frame_sequence < args.max_frames:
            if authority.consume_reset():
                state = runtime.reset()
                auto_endpoint_hold = 0
            auto_decision = runtime.auto_decision(state, dt)
            (
                applied_control,
                effective_mode,
                control_age_ms,
                controls_stale,
                client_connected,
            ) = authority.applied_control(auto_decision.control)
            update = runtime.adapter.step(state, applied_control, dt)
            state = update.state
            query = update.render_query
            render = runtime.render(query)
            cockpit = make_cockpit_frame(
                Image,
                ImageDraw,
                runtime,
                render,
                state,
                query,
                effective_mode=effective_mode,
                requested_mode=authority.requested_mode,
                applied_control=applied_control,
                control_age_ms=control_age_ms,
                client_connected=client_connected,
                boundary_hit=update.boundary_hit,
            )
            frame_array = np.asarray(cockpit, dtype=np.uint8)
            sink.write(frame_array)
            render_times.append(render.render_seconds)
            compose_times.append(render.compose_seconds)
            telemetry = {
                "type": "telemetry",
                "version": REMOTE_PROTOCOL_VERSION,
                "frame_sequence": frame_sequence,
                "server_time_ns": time.monotonic_ns(),
                "requested_mode": authority.requested_mode,
                "effective_mode": effective_mode,
                "client_connected": client_connected,
                "controls_stale": controls_stale,
                "control_age_ms": control_age_ms,
                "state": {
                    "x": state.x,
                    "y": state.y,
                    "yaw_radians": state.yaw,
                    "speed_mps": state.speed,
                    "simulation_time_seconds": state.time,
                },
                "route": {
                    "progress_meters": query.progress_meters,
                    "lateral_meters": query.lateral_meters,
                    "heading_error_radians": query.heading_error_radians,
                    "support_margin_meters": query.support_margin_meters,
                    "endpoint_reached": update.endpoint_reached,
                    "boundary_hit": update.boundary_hit,
                    "boundary_reason": update.boundary_reason,
                },
                "applied_control": {
                    "steer": applied_control.steer,
                    "throttle": applied_control.throttle,
                    "brake": applied_control.brake,
                },
                "render": {
                    "three_camera_seconds": render.render_seconds,
                    "panorama_compose_seconds": render.compose_seconds,
                },
                "video_encoder_restarts": sink.restart_count,
            }
            if gateway is not None:
                gateway.publish(telemetry)
            if args.record is not None:
                frame_records.append(telemetry)

            frame_sequence += 1
            if update.endpoint_reached and effective_mode == "auto":
                auto_endpoint_hold += 1
                if auto_endpoint_hold >= args.fps:
                    state = runtime.reset()
                    auto_endpoint_hold = 0
            else:
                auto_endpoint_hold = 0
            deadline += dt
            remaining = deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            elif remaining < -dt:
                deadline = time.perf_counter()
    except KeyboardInterrupt:
        print("stopping on keyboard interrupt", flush=True)
    except BaseException as error:
        failure = error
    finally:
        try:
            sink.close()
        finally:
            if gateway is not None:
                gateway.close()

    if args.record is not None and render_times:
        report_path = args.record.with_suffix(".json")
        report = {
            "format": "driving_scene_reconstruction.mtgs_remote_demo.v0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "source_commit": SOURCE_COMMIT,
            "checkpoint_path": str(runtime.checkpoint_path),
            "checkpoint_step": runtime.checkpoint_step,
            "video_path": str(args.record),
            "frame_count": frame_sequence,
            "fps": args.fps,
            "encoded_duration_seconds": frame_sequence / args.fps,
            "setup_seconds": runtime.setup_seconds,
            "device": runtime.torch.cuda.get_device_name(runtime.device),
            "forward_camera_names": [
                camera_name for _, camera_name in CAMERA_ALIASES
            ],
            "forward_fov_degrees": COCKPIT_HORIZONTAL_FOV_DEGREES,
            "front_projection_coverage_fraction": runtime.coverage_fraction,
            "rig_normalized_time_gaps": runtime.rig_time_gaps,
            "fixed_scene_time": True,
            "vehicle_model": "kinematic_bicycle",
            "wheelbase_meters": (
                runtime.adapter.controller.vehicle_model.wheelbase
            ),
            "corridor_half_width_meters": MTGS_CORRIDOR_HALF_WIDTH_METERS,
            "render_seconds_p50": statistics.median(render_times),
            "render_seconds_p95": _percentile(render_times, 0.95),
            "compose_seconds_p50": statistics.median(compose_times),
            "compose_seconds_p95": _percentile(compose_times, 0.95),
            "peak_reserved_gib": (
                runtime.torch.cuda.max_memory_reserved(runtime.device) / 1024**3
            ),
            "video_encoder_restarts": sink.restart_count,
            "frames": frame_records,
            "limitations": [
                "scene time is fixed",
                "right inset is route geometry, not reconstructed overhead RGB",
                "only the released approximately 84 m support corridor is accepted",
                "two-machine latency has not yet been measured",
            ],
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"report: {report_path}", flush=True)
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
