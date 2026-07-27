#!/usr/bin/env python3
"""Build a PPT-ready MTGS video driven through the remote-App control contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from apps.mtgs_remote_simulator import (  # noqa: E402
    CAMERA_ALIASES,
    COCKPIT_HORIZONTAL_FOV_DEGREES,
    FfmpegVideoSink,
    MTGS_CORRIDOR_HALF_WIDTH_METERS,
    REMOTE_COCKPIT_STATUS_HEIGHT,
    REMOTE_COCKPIT_VIEW_HEIGHT,
    REMOTE_COCKPIT_WIDTH,
    REMOTE_CONTROL_TIMEOUT_SECONDS,
    REMOTE_PROTOCOL_VERSION,
    SOURCE_COMMIT,
    MtgsRuntime,
    RemoteAuthority,
    RemoteCommandPacket,
    RemoteControlPacket,
    make_cockpit_frame,
)
from driving_scene_reconstruction.sim import HumanControl  # noqa: E402


PRESENTATION_WIDTH = 1280
PRESENTATION_HEIGHT = 720
CONTROL_PANEL_TOP = (
    REMOTE_COCKPIT_VIEW_HEIGHT + REMOTE_COCKPIT_STATUS_HEIGHT
)
DEFAULT_FPS = 20
AUTO_END_FRAME = 30
REMOTE_END_FRAME = 130
ESTOP_END_FRAME = 165
RESET_END_FRAME = 175
TOTAL_FRAMES = 220
PREVIEW_FRAMES = (15, 45, 77, 145, 167, 200)
PRIMARY_PREVIEW_FRAME = 77


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--road-block-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--cruise-speed-mps", type=float, default=12.0)
    parser.add_argument("--lateral-amplitude-meters", type=float, default=3.0)
    parser.add_argument(
        "--encoder",
        choices=("h264_nvenc", "libx264"),
        default="h264_nvenc",
    )
    return parser.parse_args()


def demo_phase(frame_index: int) -> str:
    """Return the presentation phase for one encoded frame."""

    if not 0 <= frame_index < TOTAL_FRAMES:
        raise ValueError("frame index lies outside the demo schedule")
    if frame_index < AUTO_END_FRAME:
        return "auto"
    if frame_index < REMOTE_END_FRAME:
        return "remote"
    if frame_index < ESTOP_END_FRAME:
        return "estop"
    if frame_index < RESET_END_FRAME:
        return "reset"
    return "auto_restarted"


def active_control_keys(
    control: HumanControl,
    effective_mode: str,
) -> frozenset[str]:
    """Map the actually applied remote control to visible App keys."""

    if effective_mode != "remote":
        return frozenset()
    keys: set[str] = set()
    if control.throttle > 0.05:
        keys.add("W")
    if control.brake > 0.05:
        keys.add("S")
    if control.steer > 0.04:
        keys.add("A")
    elif control.steer < -0.04:
        keys.add("D")
    return frozenset(keys)


def _load_fonts(image_font_module: Any) -> dict[str, Any]:
    regular_path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    bold_path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    )
    mono_path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
    )
    if not all(path.is_file() for path in (regular_path, bold_path, mono_path)):
        default = image_font_module.load_default()
        return {
            "small": default,
            "body": default,
            "label": default,
            "title": default,
            "number": default,
            "key": default,
        }
    return {
        "small": image_font_module.truetype(str(regular_path), 12),
        "body": image_font_module.truetype(str(regular_path), 15),
        "label": image_font_module.truetype(str(bold_path), 13),
        "title": image_font_module.truetype(str(bold_path), 21),
        "number": image_font_module.truetype(str(mono_path), 22),
        "key": image_font_module.truetype(str(bold_path), 23),
    }


def _draw_key(
    draw: Any,
    box: tuple[int, int, int, int],
    label: str,
    *,
    active: bool,
    font: Any,
) -> None:
    fill = (31, 151, 171) if active else (15, 24, 31)
    outline = (111, 238, 244) if active else (69, 88, 101)
    draw.rounded_rectangle(box, radius=7, fill=fill, outline=outline, width=2)
    draw.text(
        ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
        label,
        fill=(250, 254, 255) if active else (149, 166, 175),
        font=font,
        anchor="mm",
    )


def _phase_presentation(
    phase: str,
) -> tuple[str, str, tuple[int, int, int]]:
    if phase == "auto":
        return "AUTO", "simulator owns control", (241, 186, 56)
    if phase == "remote":
        return "REMOTE", "App packets applied", (63, 211, 157)
    if phase == "estop":
        return "E-STOP LATCHED", "vehicle model braking", (246, 83, 79)
    if phase == "reset":
        return "RESET  →  AUTO", "world state respawned", (177, 127, 244)
    return "AUTO RESTARTED", "simulator owns control", (66, 171, 236)


def make_presentation_frame(
    image_module: Any,
    draw_module: Any,
    cockpit: Any,
    fonts: dict[str, Any],
    *,
    frame_index: int,
    fps: int,
    phase: str,
    effective_mode: str,
    control: HumanControl,
    state: Any,
    query: Any,
    control_age_ms: float | None,
    control_sequence: int,
) -> Any:
    """Place the real cockpit above an inspectable simulated driver App."""

    canvas = image_module.new(
        "RGB",
        (PRESENTATION_WIDTH, PRESENTATION_HEIGHT),
        (5, 8, 11),
    )
    canvas.paste(cockpit, (0, 0))
    draw = draw_module.Draw(canvas)
    draw.rectangle(
        (0, CONTROL_PANEL_TOP, PRESENTATION_WIDTH, PRESENTATION_HEIGHT),
        fill=(7, 12, 17),
    )
    draw.line(
        (0, CONTROL_PANEL_TOP, PRESENTATION_WIDTH, CONTROL_PANEL_TOP),
        fill=(61, 202, 220),
        width=3,
    )

    draw.text(
        (18, CONTROL_PANEL_TOP + 13),
        "LOCAL DRIVER APP",
        fill=(236, 246, 249),
        font=fonts["title"],
    )
    draw.text(
        (19, CONTROL_PANEL_TOP + 43),
        "SIMULATED OPERATOR · CONTROL PROTOCOL v1",
        fill=(131, 156, 169),
        font=fonts["small"],
    )
    draw.rounded_rectangle(
        (18, CONTROL_PANEL_TOP + 65, 222, CONTROL_PANEL_TOP + 92),
        radius=7,
        fill=(16, 63, 54),
        outline=(63, 211, 157),
        width=1,
    )
    draw.text(
        (120, CONTROL_PANEL_TOP + 78),
        "APP LINK (SIM)  ·  CONNECTED",
        fill=(121, 239, 194),
        font=fonts["label"],
        anchor="mm",
    )
    draw.text(
        (19, CONTROL_PANEL_TOP + 103),
        f"TX {fps} Hz  ·  watchdog "
        f"{REMOTE_CONTROL_TIMEOUT_SECONDS * 1000:.0f} ms",
        fill=(113, 135, 147),
        font=fonts["small"],
    )

    keys = active_control_keys(control, effective_mode)
    _draw_key(
        draw,
        (315, CONTROL_PANEL_TOP + 12, 368, CONTROL_PANEL_TOP + 56),
        "W",
        active="W" in keys,
        font=fonts["key"],
    )
    for box, label in (
        ((255, CONTROL_PANEL_TOP + 63, 308, CONTROL_PANEL_TOP + 107), "A"),
        ((315, CONTROL_PANEL_TOP + 63, 368, CONTROL_PANEL_TOP + 107), "S"),
        ((375, CONTROL_PANEL_TOP + 63, 428, CONTROL_PANEL_TOP + 107), "D"),
    ):
        _draw_key(
            draw,
            box,
            label,
            active=label in keys,
            font=fonts["key"],
        )

    estop_active = phase == "estop"
    reset_active = phase == "reset"
    for box, label, active, active_colour in (
        (
            (451, CONTROL_PANEL_TOP + 12, 606, CONTROL_PANEL_TOP + 55),
            "SPACE  E-STOP",
            estop_active,
            (145, 35, 37),
        ),
        (
            (451, CONTROL_PANEL_TOP + 63, 606, CONTROL_PANEL_TOP + 106),
            "R  RESET",
            reset_active,
            (91, 57, 139),
        ),
    ):
        draw.rounded_rectangle(
            box,
            radius=7,
            fill=active_colour if active else (15, 24, 31),
            outline=(
                (255, 105, 98)
                if estop_active and active
                else (205, 159, 255)
                if reset_active and active
                else (69, 88, 101)
            ),
            width=2,
        )
        draw.text(
            ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2),
            label,
            fill=(252, 250, 255) if active else (149, 166, 175),
            font=fonts["label"],
            anchor="mm",
        )

    mode_title, mode_detail, mode_colour = _phase_presentation(phase)
    draw.text(
        (638, CONTROL_PANEL_TOP + 12),
        "CONTROL AUTHORITY",
        fill=(129, 151, 163),
        font=fonts["label"],
    )
    draw.rounded_rectangle(
        (638, CONTROL_PANEL_TOP + 34, 868, CONTROL_PANEL_TOP + 76),
        radius=8,
        fill=tuple(max(0, component // 5) for component in mode_colour),
        outline=mode_colour,
        width=2,
    )
    draw.text(
        (753, CONTROL_PANEL_TOP + 55),
        mode_title,
        fill=mode_colour,
        font=fonts["title"],
        anchor="mm",
    )
    age_label = "-" if control_age_ms is None else f"{control_age_ms:.0f} ms"
    detail_label = mode_detail
    if effective_mode == "remote":
        detail_label = f"{mode_detail} · age {age_label}"
    draw.text(
        (638, CONTROL_PANEL_TOP + 83),
        detail_label,
        fill=(183, 199, 207),
        font=fonts["body"],
    )
    draw.text(
        (638, CONTROL_PANEL_TOP + 104),
        (
            f"steer {control.steer:+.2f}  ·  gas {control.throttle:.2f}  ·  "
            f"brake {control.brake:.2f}"
        ),
        fill=(111, 136, 149),
        font=fonts["small"],
    )

    telemetry = (
        ("SPEED", f"{state.speed:4.1f}", "m/s"),
        ("OFFSET", f"{query.lateral_meters:+4.1f}", "m"),
        ("MARGIN", f"{query.support_margin_meters:4.1f}", "m"),
    )
    for index, (label, value, unit) in enumerate(telemetry):
        left = 900 + index * 120
        draw.text(
            (left, CONTROL_PANEL_TOP + 13),
            label,
            fill=(126, 150, 163),
            font=fonts["label"],
        )
        draw.text(
            (left, CONTROL_PANEL_TOP + 37),
            value,
            fill=(238, 246, 249),
            font=fonts["number"],
        )
        draw.text(
            (left + 69, CONTROL_PANEL_TOP + 43),
            unit,
            fill=(131, 153, 164),
            font=fonts["small"],
        )
    draw.text(
        (900, CONTROL_PANEL_TOP + 82),
        f"frame {frame_index + 1:03d}/{TOTAL_FRAMES}  ·  packet {control_sequence:03d}",
        fill=(157, 176, 185),
        font=fonts["body"],
    )
    draw.text(
        (900, CONTROL_PANEL_TOP + 104),
        f"support ±{MTGS_CORRIDOR_HALF_WIDTH_METERS:.0f} m  ·  "
        f"vehicle: kinematic bicycle",
        fill=(107, 132, 145),
        font=fonts["small"],
    )

    timeline_left = 18
    timeline_right = PRESENTATION_WIDTH - 18
    timeline_top = PRESENTATION_HEIGHT - 18
    segment_frames = (
        (0, AUTO_END_FRAME, (241, 186, 56)),
        (AUTO_END_FRAME, REMOTE_END_FRAME, (63, 211, 157)),
        (REMOTE_END_FRAME, ESTOP_END_FRAME, (246, 83, 79)),
        (ESTOP_END_FRAME, RESET_END_FRAME, (177, 127, 244)),
        (RESET_END_FRAME, TOTAL_FRAMES, (66, 171, 236)),
    )
    timeline_width = timeline_right - timeline_left
    for start, end, colour in segment_frames:
        x0 = timeline_left + round(timeline_width * start / TOTAL_FRAMES)
        x1 = timeline_left + round(timeline_width * end / TOTAL_FRAMES)
        draw.rectangle((x0, timeline_top, x1, timeline_top + 7), fill=colour)
    marker_x = timeline_left + round(
        timeline_width * frame_index / (TOTAL_FRAMES - 1)
    )
    draw.polygon(
        (
            (marker_x, timeline_top - 5),
            (marker_x - 5, timeline_top - 11),
            (marker_x + 5, timeline_top - 11),
        ),
        fill=(255, 255, 255),
    )
    draw.text(
        (timeline_right, timeline_top - 12),
        f"{frame_index / fps:04.1f}s",
        fill=(203, 214, 219),
        font=fonts["small"],
        anchor="ra",
    )
    return canvas


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_contact_sheet(
    image_module: Any,
    preview_images: list[tuple[int, Any]],
    output_path: Path,
) -> None:
    resampling = getattr(image_module, "Resampling", image_module).LANCZOS
    sheet = image_module.new("RGB", (1920, 720), (0, 0, 0))
    for position, (_, image) in enumerate(preview_images):
        thumbnail = image.resize((640, 360), resampling)
        sheet.paste(
            thumbnail,
            ((position % 3) * 640, (position // 3) * 360),
        )
    sheet.save(output_path, format="JPEG", quality=92)


def main() -> None:
    args = parse_args()
    for path in (args.config, args.checkpoint, args.road_block_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.fps != DEFAULT_FPS:
        raise ValueError("the reviewed presentation schedule requires 20 FPS")
    if (
        not math.isfinite(args.cruise_speed_mps)
        or not 0.0 < args.cruise_speed_mps <= 15.0
    ):
        raise ValueError("cruise speed must be within (0, 15] m/s")
    if (
        not math.isfinite(args.lateral_amplitude_meters)
        or not 0.0 < args.lateral_amplitude_meters < 5.0
    ):
        raise ValueError("lateral amplitude must remain within +/-5 m support")

    output_dir = args.output_dir.expanduser().resolve()
    video_path = output_dir / "mtgs_app_control_ppt_demo.mp4"
    report_path = output_dir / "mtgs_app_control_ppt_demo.json"
    preview_path = output_dir / "mtgs_app_control_ppt_preview.jpg"
    contact_path = output_dir / "mtgs_app_control_ppt_contact_sheet.jpg"
    outputs = (video_path, report_path, preview_path, contact_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise RuntimeError(
            "refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    runtime = MtgsRuntime(args)
    fonts = _load_fonts(ImageFont)
    authority = RemoteAuthority("auto")
    authority.set_client_connected(True)
    sink = FfmpegVideoSink(
        width=PRESENTATION_WIDTH,
        height=PRESENTATION_HEIGHT,
        fps=args.fps,
        encoder=args.encoder,
        destination=None,
        record=video_path,
    )

    dt = 1.0 / args.fps
    state = runtime.reset()
    command_sequence = 0
    control_sequence = 0
    command_events: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    render_times: list[float] = []
    compose_times: list[float] = []
    preview_images: list[tuple[int, Any]] = []
    key_counts = {"W": 0, "A": 0, "S": 0, "D": 0}
    failure: BaseException | None = None
    started = time.perf_counter()
    try:
        for frame_index in range(TOTAL_FRAMES):
            phase = demo_phase(frame_index)
            if frame_index == AUTO_END_FRAME:
                accepted = authority.accept(
                    RemoteCommandPacket(
                        sequence=command_sequence,
                        command="mode",
                        value="remote",
                    )
                )
                command_events.append(
                    {
                        "frame": frame_index,
                        "command": "mode",
                        "value": "remote",
                        "accepted": accepted,
                    }
                )
                command_sequence += 1
            elif frame_index == REMOTE_END_FRAME:
                accepted = authority.accept(
                    RemoteCommandPacket(
                        sequence=command_sequence,
                        command="estop",
                    )
                )
                command_events.append(
                    {
                        "frame": frame_index,
                        "command": "estop",
                        "accepted": accepted,
                    }
                )
                command_sequence += 1
            elif frame_index == ESTOP_END_FRAME:
                reset_accepted = authority.accept(
                    RemoteCommandPacket(
                        sequence=command_sequence,
                        command="reset",
                    )
                )
                command_sequence += 1
                if not authority.consume_reset():
                    raise RuntimeError("accepted reset was not observable")
                state = runtime.reset()
                auto_accepted = authority.accept(
                    RemoteCommandPacket(
                        sequence=command_sequence,
                        command="mode",
                        value="auto",
                    )
                )
                command_sequence += 1
                command_events.extend(
                    (
                        {
                            "frame": frame_index,
                            "command": "reset",
                            "accepted": reset_accepted,
                        },
                        {
                            "frame": frame_index,
                            "command": "mode",
                            "value": "auto",
                            "accepted": auto_accepted,
                        },
                    )
                )

            auto_decision = runtime.auto_decision(state, dt)
            if phase == "remote":
                accepted = authority.accept(
                    RemoteControlPacket(
                        sequence=control_sequence,
                        client_time_ns=time.monotonic_ns(),
                        control=auto_decision.control,
                    )
                )
                if not accepted:
                    raise RuntimeError("monotonic simulated App packet rejected")
                control_sequence += 1

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
            presentation = make_presentation_frame(
                Image,
                ImageDraw,
                cockpit,
                fonts,
                frame_index=frame_index,
                fps=args.fps,
                phase=phase,
                effective_mode=effective_mode,
                control=applied_control,
                state=state,
                query=query,
                control_age_ms=control_age_ms,
                control_sequence=control_sequence,
            )
            sink.write(np.asarray(presentation, dtype=np.uint8))
            if frame_index in PREVIEW_FRAMES:
                preview_images.append((frame_index, presentation.copy()))
                if frame_index == PRIMARY_PREVIEW_FRAME:
                    presentation.save(preview_path, format="JPEG", quality=94)

            active_keys = active_control_keys(
                applied_control,
                effective_mode,
            )
            for key in active_keys:
                key_counts[key] += 1
            render_times.append(render.render_seconds)
            compose_times.append(render.compose_seconds)
            frame_records.append(
                {
                    "frame": frame_index,
                    "phase": phase,
                    "effective_mode": effective_mode,
                    "requested_mode": authority.requested_mode,
                    "controls_stale": controls_stale,
                    "control_age_ms": control_age_ms,
                    "active_keys": sorted(active_keys),
                    "state": {
                        "speed_mps": state.speed,
                        "simulation_time_seconds": state.time,
                    },
                    "route": {
                        "progress_meters": query.progress_meters,
                        "lateral_meters": query.lateral_meters,
                        "support_margin_meters": query.support_margin_meters,
                        "boundary_hit": update.boundary_hit,
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
                }
            )
    except BaseException as error:
        failure = error
    finally:
        try:
            sink.close()
        except BaseException as close_error:
            if failure is None:
                failure = close_error
    if failure is not None:
        raise failure

    if len(preview_images) != len(PREVIEW_FRAMES):
        raise RuntimeError("presentation previews are incomplete")
    _make_contact_sheet(Image, preview_images, contact_path)

    phases = {record["phase"] for record in frame_records}
    modes = {record["effective_mode"] for record in frame_records}
    boundary_hits = sum(
        bool(record["route"]["boundary_hit"]) for record in frame_records
    )
    estop_frames = [
        record
        for record in frame_records
        if record["effective_mode"] == "estop"
    ]
    pass_gates = {
        "all_scheduled_phases_observed": phases
        == {"auto", "remote", "estop", "reset", "auto_restarted"},
        "auto_remote_estop_modes_observed": {"auto", "remote", "estop"}
        <= modes,
        "remote_packets_applied": control_sequence
        == REMOTE_END_FRAME - AUTO_END_FRAME,
        "w_a_d_visible": all(key_counts[key] > 0 for key in ("W", "A", "D")),
        "estop_applied_brake": bool(estop_frames)
        and all(
            record["applied_control"]["brake"] == 1.0
            for record in estop_frames
        ),
        "zero_support_boundary_hits": boundary_hits == 0,
        "complete_previews": len(preview_images) == len(PREVIEW_FRAMES),
    }
    report = {
        "format": "driving_scene_reconstruction.mtgs_app_control_ppt_demo.v0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "mtgs_source_commit": SOURCE_COMMIT,
        "checkpoint_path": str(runtime.checkpoint_path),
        "checkpoint_step": runtime.checkpoint_step,
        "video_path": str(video_path),
        "video_sha256": _sha256(video_path),
        "preview_path": str(preview_path),
        "contact_sheet_path": str(contact_path),
        "frame_count": TOTAL_FRAMES,
        "fps": args.fps,
        "encoded_duration_seconds": TOTAL_FRAMES / args.fps,
        "video_width": PRESENTATION_WIDTH,
        "video_height": PRESENTATION_HEIGHT,
        "setup_seconds": runtime.setup_seconds,
        "generation_wall_seconds": time.perf_counter() - started,
        "device": runtime.torch.cuda.get_device_name(runtime.device),
        "operator": {
            "simulated": True,
            "physical_keyboard_or_remote_machine": False,
            "control_source": (
                "MTGS simulated-human decision emitted as RemoteControlPacket"
            ),
            "transport_scope": (
                "in-process RemoteAuthority protocol exercise; no socket or LAN"
            ),
        },
        "control_contract": {
            "protocol_version": REMOTE_PROTOCOL_VERSION,
            "control_timeout_ms": REMOTE_CONTROL_TIMEOUT_SECONDS * 1000.0,
            "remote_packet_count": control_sequence,
            "command_events": command_events,
            "key_visible_frame_counts": key_counts,
            "schedule": [
                {"phase": "auto", "frames": [0, AUTO_END_FRAME - 1]},
                {
                    "phase": "remote",
                    "frames": [AUTO_END_FRAME, REMOTE_END_FRAME - 1],
                },
                {
                    "phase": "estop",
                    "frames": [REMOTE_END_FRAME, ESTOP_END_FRAME - 1],
                },
                {
                    "phase": "reset",
                    "frames": [ESTOP_END_FRAME, RESET_END_FRAME - 1],
                },
                {
                    "phase": "auto_restarted",
                    "frames": [RESET_END_FRAME, TOTAL_FRAMES - 1],
                },
            ],
        },
        "renderer": {
            "forward_camera_names": [
                camera_name for _, camera_name in CAMERA_ALIASES
            ],
            "forward_fov_degrees": COCKPIT_HORIZONTAL_FOV_DEGREES,
            "front_projection_coverage_fraction": runtime.coverage_fraction,
            "fixed_scene_time": True,
            "three_camera_seconds_p50": statistics.median(render_times),
            "three_camera_seconds_p95": _percentile(render_times, 0.95),
            "panorama_compose_seconds_p50": statistics.median(compose_times),
            "panorama_compose_seconds_p95": _percentile(compose_times, 0.95),
            "peak_reserved_gib": (
                runtime.torch.cuda.max_memory_reserved(runtime.device) / 1024**3
            ),
        },
        "drive": {
            "vehicle_model": "kinematic_bicycle",
            "wheelbase_meters": (
                runtime.adapter.controller.vehicle_model.wheelbase
            ),
            "corridor_half_width_meters": MTGS_CORRIDOR_HALF_WIDTH_METERS,
            "maximum_speed_mps": max(
                record["state"]["speed_mps"] for record in frame_records
            ),
            "maximum_absolute_lateral_meters": max(
                abs(record["route"]["lateral_meters"])
                for record in frame_records
            ),
            "minimum_support_margin_meters": min(
                record["route"]["support_margin_meters"]
                for record in frame_records
            ),
            "boundary_hit_count": boundary_hits,
        },
        "pass_gates": pass_gates,
        "overall_pass": all(pass_gates.values()),
        "frames": frame_records,
        "limitations": [
            "the operator is scripted, not a physical remote human",
            "the App protocol is exercised in-process, not over a two-computer LAN",
            "scene time is fixed and reconstructed traffic is not interactive",
            "the right inset is route geometry, not reconstructed overhead RGB",
            "the released approximately 84 m support corridor is not a long route",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not report["overall_pass"]:
        raise RuntimeError(f"PPT demo failed gates: {pass_gates}")
    print(f"video: {video_path}")
    print(f"preview: {preview_path}")
    print(f"contact sheet: {contact_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
