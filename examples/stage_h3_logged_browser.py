#!/usr/bin/env python3
"""Browser driving loop for the accepted PandaSet static-8k reconstruction."""

from __future__ import annotations

import argparse
import io
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    CameraRig,
    CameraSpec,
    EgoState,
    HumanControl,
    LoggedEgoOffsetController,
    SplatADLoggedRenderer,
)


DEFAULT_CONFIG = Path(
    "/home/yawei/stage3_external/outputs/pandaset_h3/"
    "scene_040_splatad_static_8000/splatad/"
    "2026-07-19_resume_2k_to_8k/config.yml"
)
CAMERAS = ("front_left", "front", "front_right", "left", "back", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scale", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


def control_for_keys(keys: set[str]) -> HumanControl:
    steer = float("a" in keys) - float("d" in keys)
    return HumanControl(
        steer=steer,
        throttle=float("w" in keys),
        brake=float("s" in keys),
    )


def make_mosaic(
    image_module: Any,
    image_draw_module: Any,
    frames: dict[str, object],
    state: EgoState,
) -> object:
    cell = (480, 270)
    status_height = 34
    canvas = image_module.new(
        "RGB",
        (cell[0] * 3, cell[1] * 2 + status_height),
        "black",
    )
    for index, name in enumerate(CAMERAS):
        tile = image_module.fromarray(frames[name]).convert("RGB")
        tile.thumbnail(cell, image_module.Resampling.LANCZOS)
        x = (index % 3) * cell[0] + (cell[0] - tile.width) // 2
        y = (index // 3) * cell[1] + (cell[1] - tile.height) // 2
        image_draw_module.Draw(tile).text(
            (8, 7),
            name,
            fill=(0, 255, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        canvas.paste(tile, (x, y))
    status = (
        f"log={state.time:05.2f}s  offset forward={state.x:+.2f}m "
        f"left={state.y:+.2f}m  yaw={math.degrees(state.yaw):+.1f}deg"
    )
    image_draw_module.Draw(canvas).text(
        (10, cell[1] * 2 + 10),
        status,
        fill=(255, 216, 77),
    )
    return canvas


WEB_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PandaSet 可驾驶重建</title>
  <style>
    body { margin: 0; background: #0d0d0d; color: #eee; font: 16px sans-serif; text-align: center; }
    h1 { font-size: 20px; margin: 8px; }
    #view { max-width: 100vw; max-height: calc(100vh - 150px); border: 1px solid #444; }
    #status { min-height: 22px; color: #ffd84d; margin: 4px; }
    button { min-width: 76px; margin: 3px; padding: 10px; font-size: 17px; }
  </style>
</head>
<body>
  <h1>PandaSet 场景 040 · 六相机驾驶</h1>
  <img id="view" src="/frame.jpg" alt="six reconstructed driving cameras">
  <div>轨迹自动前进；W/S 调整前向微偏移，A/D 小幅转向，R 重置</div>
  <div id="status">单驾驶客户端 · 请勿同时打开多个标签页</div>
  <div><button data-key="w">W 增加前向微调</button></div>
  <div>
    <button data-key="a">A 左转</button>
    <button data-key="s">S 降低微调速度</button>
    <button data-key="d">D 右转</button>
  </div>
  <div><button data-key="r">R 重置</button><button id="fullscreen">全屏</button></div>
  <script>
    const view = document.getElementById("view");
    const status = document.getElementById("status");
    const held = new Set();
    let running = true;
    let sequence = 0;
    let generation = 0;
    const tickPeriodMs = __TICK_PERIOD_MS__;
    const drivingKeys = new Set(["w", "s", "a", "d"]);

    function setHeld(key, active) {
      if (active) held.add(key); else held.delete(key);
    }
    document.addEventListener("keydown", event => {
      const key = event.key.toLowerCase();
      if (drivingKeys.has(key)) { event.preventDefault(); setHeld(key, true); }
      if (key === "r" && !event.repeat) { event.preventDefault(); reset(); }
    });
    document.addEventListener("keyup", event => {
      const key = event.key.toLowerCase();
      if (drivingKeys.has(key)) { event.preventDefault(); setHeld(key, false); }
    });
    window.addEventListener("blur", () => held.clear());
    document.querySelectorAll("button[data-key]").forEach(button => {
      const key = button.dataset.key;
      if (key === "r") { button.addEventListener("click", reset); return; }
      for (const start of ["mousedown", "touchstart"]) {
        button.addEventListener(start, event => { event.preventDefault(); setHeld(key, true); });
      }
      for (const end of ["mouseup", "mouseleave", "touchend", "touchcancel"]) {
        button.addEventListener(end, event => { event.preventDefault(); setHeld(key, false); });
      }
    });
    document.getElementById("fullscreen").addEventListener("click", () => view.requestFullscreen());

    async function reset() {
      held.clear();
      const token = ++generation;
      running = false;
      await fetch("/reset", {method: "POST"});
      await new Promise((resolve, reject) => {
        view.onload = resolve;
        view.onerror = reject;
        view.src = "/frame.jpg?n=" + (++sequence);
      });
      running = true;
      tick(token);
    }
    async function tick(token) {
      if (token !== generation) return;
      const started = performance.now();
      try {
        const query = encodeURIComponent([...held].join(""));
        const response = await fetch("/tick?keys=" + query, {method: "POST"});
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "request failed");
        await new Promise((resolve, reject) => {
          view.onload = resolve;
          view.onerror = reject;
          view.src = "/frame.jpg?n=" + (++sequence);
        });
        const eventToScreen = performance.now() - started;
        status.textContent =
          `log ${result.time.toFixed(2)}s / ${result.duration.toFixed(2)}s · ` +
          `请求→图像加载 ${eventToScreen.toFixed(0)}ms`;
        if (result.time >= result.duration) {
          running = false;
          status.textContent += " · 已到终点，按 R 重置";
        }
      } catch (error) {
        status.textContent = error.toString();
        running = false;
      }
      if (running && token === generation) {
        setTimeout(
          () => tick(token),
          Math.max(0, tickPeriodMs - (performance.now() - started))
        );
      }
    }
    tick(generation);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("--dt must be finite and positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    from PIL import Image, ImageDraw

    renderer = SplatADLoggedRenderer(
        args.config,
        output_scale=args.output_scale,
    )
    renderer.load()
    if renderer.checkpoint_step != 7999:
        raise RuntimeError(
            f"expected accepted static step 7999, got {renderer.checkpoint_step}"
        )
    controller = LoggedEgoOffsetController(renderer.logged_duration)
    rig = CameraRig(tuple(CameraSpec(name) for name in CAMERAS))
    runtime: dict[str, object] = {
        "state": controller.reset(),
        "jpeg": b"",
        "metadata": {},
    }

    def render_frame() -> None:
        state = runtime["state"]
        assert isinstance(state, EgoState)
        observation = renderer.render("040", state, rig)
        mosaic = make_mosaic(Image, ImageDraw, dict(observation.frames), state)
        buffer = io.BytesIO()
        mosaic.save(buffer, format="JPEG", quality=88)
        runtime["jpeg"] = buffer.getvalue()
        runtime["metadata"] = dict(observation.metadata)

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                page = WEB_PAGE.replace(
                    "__TICK_PERIOD_MS__",
                    f"{args.dt * 1000.0:.6f}",
                )
                self.send_bytes(200, "text/html; charset=utf-8", page.encode())
            elif path == "/frame.jpg":
                self.send_bytes(200, "image/jpeg", runtime["jpeg"])  # type: ignore[arg-type]
            else:
                self.send_bytes(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            started = time.perf_counter()
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/reset":
                    runtime["state"] = controller.reset()
                elif parsed.path == "/tick":
                    keys = set(parse_qs(parsed.query).get("keys", [""])[0])
                    if not keys <= {"w", "s", "a", "d"}:
                        raise ValueError("keys must contain only W/S/A/D")
                    state = runtime["state"]
                    assert isinstance(state, EgoState)
                    runtime["state"] = controller.step(
                        state,
                        control_for_keys(keys),
                        args.dt,
                    )
                else:
                    self.send_bytes(404, "text/plain", b"not found")
                    return
                render_frame()
                state = runtime["state"]
                metadata = runtime["metadata"]
                assert isinstance(state, EgoState)
                assert isinstance(metadata, dict)
                payload = {
                    "time": state.time,
                    "duration": renderer.logged_duration,
                    "x": state.x,
                    "y": state.y,
                    "yaw_degrees": math.degrees(state.yaw),
                    "renderer_ms": float(metadata["render_seconds"]) * 1000.0,
                    "server_control_to_jpeg_ms": (
                        time.perf_counter() - started
                    ) * 1000.0,
                }
                self.send_bytes(200, "application/json", json.dumps(payload).encode())
            except Exception as error:
                self.send_bytes(
                    400,
                    "application/json",
                    json.dumps({"error": str(error)}).encode(),
                )

        def log_message(self, format: str, *values: object) -> None:
            print(f"web: {format % values}")

    print("loading accepted static-8k and initial six-camera frame...")
    render_frame()
    server = HTTPServer((args.host, args.port), Handler)
    print(f"browser_url=http://{args.host}:{args.port}")
    print("controls=W/S/A/D, reset=R")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping browser driving loop")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
