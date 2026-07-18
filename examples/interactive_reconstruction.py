#!/usr/bin/env python3
"""Keyboard-drive a nearby pose and display reconstructed Wayve camera views."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    CameraRig,
    CameraSpec,
    EgoState,
    HumanControl,
    NerfstudioRenderer,
    SimpleVehicleModel,
)

DEFAULT_CONFIG = Path(
    "/home/yawei/stage1_external/outputs/wayvescenes101_h1/"
    "scene_094_h1_big/splatfacto/run_v2/config.yml"
)
DEFAULT_CAMERAS = (
    "front-forward",
    "left-forward",
    "right-forward",
    "left-backward",
    "right-backward",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-frame", type=int, default=100)
    parser.add_argument("--output-scale", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="Run scripted controls without opening a window",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/dsr_stage_h2_interactive"),
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve a browser UI instead of opening a local Tk window",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def make_mosaic(
    image_module: Any,
    image_draw_module: Any,
    frames: dict[str, object],
    state: EgoState,
    fps: float,
) -> object:
    images: list[object] = []
    for name, rgb in frames.items():
        image = image_module.fromarray(rgb).convert("RGB")
        draw = image_draw_module.Draw(image)
        draw.text((10, 8), name, fill=(0, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
        images.append(image)
    camera_count = len(images)
    if camera_count == 1:
        columns = 1
    elif camera_count in (2, 4):
        columns = 2
    else:
        columns = 3
    width, height = images[0].size
    blank = image_module.new("RGB", (width, height), color=(0, 0, 0))
    while len(images) % columns:
        images.append(blank)
    rows = len(images) // columns
    status_height = 28
    mosaic = image_module.new(
        "RGB",
        (columns * width, rows * height + status_height),
        color=(0, 0, 0),
    )
    for index, image in enumerate(images):
        mosaic.paste(image, ((index % columns) * width, (index // columns) * height))
    status = (
        f"x={state.x:+.2f}m y={state.y:+.2f}m "
        f"yaw={math.degrees(state.yaw):+.1f}deg speed={state.speed:.2f}m/s "
        f"render={fps:.2f} views/s"
    )
    image_draw_module.Draw(mosaic).text(
        (10, rows * height + 7),
        status,
        fill=(255, 255, 0),
    )
    return mosaic


def control_for_key(key: str) -> HumanControl:
    if key.lower() == "w":
        return HumanControl(throttle=0.35)
    if key.lower() == "s":
        return HumanControl(brake=0.45)
    if key.lower() == "a":
        return HumanControl(steer=0.35, throttle=0.15)
    if key.lower() == "d":
        return HumanControl(steer=-0.35, throttle=0.15)
    return HumanControl(brake=0.15)


WEB_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Driving Scene Reconstruction</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font: 16px sans-serif; text-align: center; }
    h1 { font-size: 20px; margin: 12px; }
    img { max-width: 98vw; max-height: 78vh; border: 1px solid #444; }
    button { min-width: 64px; margin: 4px; padding: 12px; font-size: 18px; }
    #status { min-height: 24px; color: #ffd84d; }
  </style>
</head>
<body>
  <h1>Driving Scene Reconstruction</h1>
  <img id="view" src="/frame.jpg" alt="reconstructed camera rig">
  <div id="status">W/S/A/D 控制，R 重置</div>
  <div><button data-key="w">W 油门</button></div>
  <div>
    <button data-key="a">A 左转</button>
    <button data-key="s">S 刹车</button>
    <button data-key="d">D 右转</button>
  </div>
  <div>
    <button data-key="r">R 重置</button>
    <button id="fullscreen">全屏画面</button>
  </div>
  <script>
    const view = document.getElementById("view");
    const status = document.getElementById("status");
    let busy = false;
    async function control(key) {
      if (busy) return;
      busy = true;
      status.textContent = "正在渲染…";
      try {
        const response = await fetch("/control?key=" + encodeURIComponent(key), {method: "POST"});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "request failed");
        view.src = "/frame.jpg?t=" + Date.now();
        status.textContent =
          `x=${result.x.toFixed(2)}m y=${result.y.toFixed(2)}m ` +
          `yaw=${result.yaw_degrees.toFixed(1)}° render=${result.render_seconds.toFixed(3)}s`;
      } catch (error) {
        status.textContent = error.toString();
      } finally {
        busy = false;
      }
    }
    document.addEventListener("keydown", event => {
      const key = event.key.toLowerCase();
      if (["w", "s", "a", "d", "r"].includes(key)) {
        event.preventDefault();
        control(key);
      }
    });
    document.querySelectorAll("button").forEach(
      button => {
        if (button.dataset.key) {
          button.addEventListener("click", () => control(button.dataset.key));
        }
      }
    );
    document.getElementById("fullscreen").addEventListener(
      "click",
      () => view.requestFullscreen()
    );
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if args.headless_steps < 0:
        raise ValueError("--headless-steps cannot be negative")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.web and args.headless_steps:
        raise ValueError("--web and --headless-steps cannot be used together")
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("Pillow is required for the interactive display") from error

    renderer = NerfstudioRenderer(
        args.config,
        reference_frame_index=args.reference_frame,
        output_scale=args.output_scale,
    )
    rig = CameraRig(tuple(CameraSpec(name) for name in args.cameras))
    model = SimpleVehicleModel(max_acceleration=1.0, max_braking=2.0)
    state = EgoState()
    output_dir = args.output_dir.expanduser().resolve()
    if args.headless_steps:
        output_dir.mkdir(parents=True, exist_ok=True)

    def advance(current: EgoState, key: str) -> EgoState:
        candidate = model.step(current, control_for_key(key), args.dt)
        try:
            renderer.limits.validate(candidate)
        except ValueError as error:
            print(f"pose limit reached: {error}; press R to reset")
            return EgoState(
                x=current.x,
                y=current.y,
                yaw=current.yaw,
                speed=0.0,
                time=candidate.time,
            )
        return candidate

    if args.headless_steps:
        scripted_keys = ["w", "a", "w", "d", "s"]
        for step in range(args.headless_steps + 1):
            observation = renderer.render("scene_094", state, rig)
            fps = float(observation.metadata.get("fps_equivalent") or 0.0)
            mosaic = make_mosaic(
                Image,
                ImageDraw,
                dict(observation.frames),
                state,
                fps,
            )
            mosaic.save(output_dir / f"step_{step:03d}.jpg", quality=90)
            if step < args.headless_steps:
                state = advance(state, scripted_keys[step % len(scripted_keys)])
        return

    if args.web:
        runtime: dict[str, object] = {
            "state": state,
            "jpeg": b"",
            "metadata": {},
        }

        def render_web_frame() -> None:
            current = runtime["state"]
            assert isinstance(current, EgoState)
            observation = renderer.render("scene_094", current, rig)
            fps = float(observation.metadata.get("fps_equivalent") or 0.0)
            mosaic = make_mosaic(
                Image,
                ImageDraw,
                dict(observation.frames),
                current,
                fps,
            )
            buffer = io.BytesIO()
            mosaic.save(buffer, format="JPEG", quality=90)
            runtime["jpeg"] = buffer.getvalue()
            runtime["metadata"] = dict(observation.metadata)

        class WebHandler(BaseHTTPRequestHandler):
            def _send_bytes(
                self,
                status: int,
                content_type: str,
                payload: bytes,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/":
                    self._send_bytes(
                        200,
                        "text/html; charset=utf-8",
                        WEB_PAGE.encode("utf-8"),
                    )
                    return
                if path == "/frame.jpg":
                    self._send_bytes(
                        200,
                        "image/jpeg",
                        runtime["jpeg"],  # type: ignore[arg-type]
                    )
                    return
                self._send_bytes(404, "text/plain", b"not found")

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != "/control":
                    self._send_bytes(404, "text/plain", b"not found")
                    return
                key = parse_qs(parsed.query).get("key", [""])[0].lower()
                try:
                    current = runtime["state"]
                    assert isinstance(current, EgoState)
                    if key == "r":
                        runtime["state"] = EgoState()
                    elif key in {"w", "s", "a", "d"}:
                        runtime["state"] = advance(current, key)
                    else:
                        raise ValueError("key must be one of W/S/A/D/R")
                    render_web_frame()
                    current = runtime["state"]
                    assert isinstance(current, EgoState)
                    metadata = runtime["metadata"]
                    assert isinstance(metadata, dict)
                    payload = {
                        "x": current.x,
                        "y": current.y,
                        "yaw_degrees": math.degrees(current.yaw),
                        "speed": current.speed,
                        "render_seconds": float(metadata.get("render_seconds") or 0.0),
                    }
                    self._send_bytes(
                        200,
                        "application/json",
                        json.dumps(payload).encode("utf-8"),
                    )
                except Exception as error:
                    self._send_bytes(
                        400,
                        "application/json",
                        json.dumps({"error": str(error)}).encode("utf-8"),
                    )

            def log_message(self, format: str, *values: object) -> None:
                print(f"web: {format % values}")

        print("loading initial browser frame...")
        render_web_frame()
        server = HTTPServer((args.host, args.port), WebHandler)
        print(f"browser_url=http://{args.host}:{args.port}")
        print("Press Ctrl-C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("stopping browser viewer")
        finally:
            server.server_close()
        return

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError(
            "No graphical display is available. Run with --web and open the "
            "printed URL, or use --headless-steps to write image files."
        )

    try:
        import tkinter as tk
        from PIL import ImageTk
    except ImportError as error:
        raise RuntimeError("Tkinter and Pillow ImageTk are required for the window") from error

    root = tk.Tk()
    root.title("Driving Scene Reconstruction - W/S/A/D, R reset, Q quit")
    label = tk.Label(root)
    label.pack()
    photo: object | None = None

    def render_current() -> None:
        nonlocal photo
        observation = renderer.render("scene_094", state, rig)
        fps = float(observation.metadata.get("fps_equivalent") or 0.0)
        mosaic = make_mosaic(
            Image,
            ImageDraw,
            dict(observation.frames),
            state,
            fps,
        )
        photo = ImageTk.PhotoImage(mosaic)
        label.configure(image=photo)

    def on_key(event: object) -> None:
        nonlocal state
        key = str(event.keysym)
        if key.lower() == "q" or key == "Escape":
            root.destroy()
            return
        if key.lower() == "r":
            state = EgoState()
        elif key.lower() in {"w", "s", "a", "d"}:
            state = advance(state, key)
        else:
            return
        try:
            render_current()
        except Exception as error:
            root.destroy()
            raise error

    root.bind("<KeyPress>", on_key)
    render_current()
    root.mainloop()


if __name__ == "__main__":
    main()
