#!/usr/bin/env python3
"""Restricted browser driving loop for the PandaSet world-pose renderer."""

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
    LoggedCenterlineCorridor,
    NearbyPoseLimits,
    SplatADWorldRenderer,
    WorldDrivingController,
)


DEFAULT_CONFIG = Path(
    "/home/yawei/stage3_external/outputs/pandaset_h3/"
    "scene_040_splatad_static_8000/splatad/"
    "2026-07-19_resume_2k_to_8k/config.yml"
)
CAMERAS = ("front_left", "front", "front_right", "left", "back", "right")
CORRIDOR_HALF_WIDTH_METERS = 1.0
CORRIDOR_MAX_HEADING_ERROR_DEGREES = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scale", type=float, default=0.25)
    parser.add_argument("--anchor-log-time", type=float, default=4.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    return parser.parse_args()


def control_for_keys(keys: set[str]) -> HumanControl:
    """Map keyboard state to throttle/brake/left-positive steering."""

    braking = "s" in keys
    return HumanControl(
        steer=float("a" in keys) - float("d" in keys),
        throttle=float("w" in keys and not braking),
        brake=float(braking),
    )


def should_step_world(keys: set[str], current_speed: float) -> bool:
    """Keep ticking while accelerating or while the vehicle is still moving."""

    return "w" in keys or current_speed > 1e-6


def boundary_help_text() -> str:
    return (
        f"当前沿原始真实轨迹建立实验管道：距中心线最多 "
        f"{CORRIDOR_HALF_WIDTH_METERS:g}m、与道路方向最多 "
        f"{CORRIDOR_MAX_HEADING_ERROR_DEGREES:g}°。它只表示数据覆盖参考，"
        "不是已认证道路；触边后请按 R。"
    )


def make_mosaic(
    image_module: Any,
    image_draw_module: Any,
    frames: dict[str, object],
    state: EgoState,
) -> object:
    """Preserve the Renderer output resolution in a native six-view mosaic."""

    sample = frames[CAMERAS[0]]
    cell = (int(sample.shape[1]), int(sample.shape[0]))
    status_height = max(42, int(cell[1] * 0.12))
    canvas = image_module.new(
        "RGB",
        (cell[0] * 3, cell[1] * 2 + status_height),
        "black",
    )
    for index, name in enumerate(CAMERAS):
        tile = image_module.fromarray(frames[name]).convert("RGB")
        image_draw_module.Draw(tile).text(
            (8, 7),
            name,
            fill=(0, 255, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        canvas.paste(tile, ((index % 3) * cell[0], (index // 3) * cell[1]))
    status = (
        f"WORLD PROVISIONAL  t={state.time:05.2f}s  x={state.x:+.2f}m  "
        f"y(left)={state.y:+.2f}m  yaw={math.degrees(state.yaw):+.1f}deg  "
        f"speed={state.speed:.2f}m/s"
    )
    image_draw_module.Draw(canvas).text(
        (10, cell[1] * 2 + 12),
        status,
        fill=(255, 216, 77),
    )
    return canvas


WEB_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PandaSet 世界坐标自由驾驶</title>
  <style>
    body { margin: 0; overflow: hidden; background: #0d0d0d; color: #eee; font: 16px sans-serif; text-align: center; }
    h1 { font-size: 18px; margin: 3px; }
    #view { display: block; width: calc(100vw - 8px); height: auto; max-height: calc(100vh - 82px); margin: 0 auto; object-fit: contain; border: 1px solid #444; }
    #status { min-height: 20px; color: #ffd84d; margin: 2px 4px; }
    #toolbar { min-height: 34px; }
    #toolbar button { min-width: 76px; margin: 1px 2px; padding: 6px 9px; font-size: 14px; }
    #controls { position: fixed; left: 50%; bottom: 48px; transform: translateX(-50%); min-width: 480px; padding: 8px; background: rgba(23,23,23,.96); border: 1px solid #555; }
    #controls button { min-width: 110px; margin: 3px; padding: 10px; font-size: 17px; }
    #warning { color: #ff9a72; margin: 4px; }
    [hidden] { display: none !important; }
  </style>
</head>
<body>
  <h1>PandaSet 场景 040 · 世界坐标六相机驾驶（实验边界）</h1>
  <img id="view" src="/frame.jpg" alt="six reconstructed driving cameras">
  <div id="status">默认静止 · ↑/W 油门 · ↓/S 刹车 · ←/A 左转 · →/D 右转 · R 重置</div>
  <div id="toolbar">
    <button data-key="r">R 重置</button>
    <button id="fullscreen">全屏</button>
    <button id="toggle-controls" aria-expanded="false">显示快捷键</button>
  </div>
  <section id="controls" hidden>
    <div id="warning">__BOUNDARY_HELP__</div>
    <div><button data-key="w">↑ / W 油门</button></div>
    <div>
      <button data-key="a">← / A 左转</button>
      <button data-key="s">↓ / S 刹车</button>
      <button data-key="d">→ / D 右转</button>
    </div>
  </section>
  <script>
    const view = document.getElementById("view");
    const status = document.getElementById("status");
    const held = new Set();
    const drivingKeys = new Set(["w", "s", "a", "d"]);
    const keyAliases = new Map([
      ["arrowup", "w"], ["arrowdown", "s"],
      ["arrowleft", "a"], ["arrowright", "d"]
    ]);
    const tickPeriodMs = __TICK_PERIOD_MS__;
    let vehicleSpeed = 0;
    let inFlight = false;
    let nextTimer = null;
    let sequence = 0;
    let generation = 0;
    let boundaryBlocked = false;
    let pendingInputStartedAt = null;

    function normalizedKey(key) {
      const lowered = key.toLowerCase();
      return keyAliases.get(lowered) || lowered;
    }
    function requestTickNow() {
      if (inFlight || boundaryBlocked) return;
      if (nextTimer !== null) { clearTimeout(nextTimer); nextTimer = null; }
      tick(generation);
    }
    function setHeld(key, active) {
      const changed = held.has(key) !== active;
      if (active) held.add(key); else held.delete(key);
      if (changed) { pendingInputStartedAt = performance.now(); requestTickNow(); }
    }
    function scheduleNext(token, started) {
      if (boundaryBlocked || token !== generation) return;
      if (!held.has("w") && vehicleSpeed <= 1e-6) return;
      nextTimer = setTimeout(
        () => tick(token),
        Math.max(0, tickPeriodMs - (performance.now() - started))
      );
    }
    document.addEventListener("keydown", event => {
      const key = normalizedKey(event.key);
      if (drivingKeys.has(key)) { event.preventDefault(); setHeld(key, true); }
      if (key === "r" && !event.repeat) { event.preventDefault(); reset(); }
    });
    document.addEventListener("keyup", event => {
      const key = normalizedKey(event.key);
      if (drivingKeys.has(key)) { event.preventDefault(); setHeld(key, false); }
    });
    window.addEventListener("blur", () => {
      if (held.size) { held.clear(); pendingInputStartedAt = performance.now(); requestTickNow(); }
    });
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
    document.getElementById("toggle-controls").addEventListener("click", event => {
      const panel = document.getElementById("controls");
      panel.hidden = !panel.hidden;
      event.currentTarget.textContent = panel.hidden ? "显示快捷键" : "隐藏快捷键";
      event.currentTarget.setAttribute("aria-expanded", String(!panel.hidden));
    });

    async function loadLatestFrame() {
      await new Promise((resolve, reject) => {
        view.onload = resolve;
        view.onerror = reject;
        view.src = "/frame.jpg?n=" + (++sequence);
      });
    }
    function stateText(result, eventToScreen, inputToScreen) {
      let text =
        `t ${result.simulation_time.toFixed(2)}s · x ${result.x.toFixed(2)}m · ` +
        `左 ${result.y.toFixed(2)}m · yaw ${result.yaw_degrees.toFixed(1)}° · ` +
        `速度 ${result.speed.toFixed(2)}m/s · 请求→画面 ${eventToScreen.toFixed(0)}ms`;
      if (result.corridor) {
        text += ` · 道路进度 ${result.corridor.progress_meters.toFixed(1)}m` +
          ` · 偏离 ${result.corridor.distance_meters.toFixed(2)}m`;
      }
      if (inputToScreen !== null) text += ` · 输入→画面 ${inputToScreen.toFixed(0)}ms`;
      if (result.boundary_hit) text += ` · 已触及实验边界：${result.boundary_reason}；按 R 重置`;
      return text;
    }
    async function reset() {
      held.clear();
      const token = ++generation;
      boundaryBlocked = false;
      if (nextTimer !== null) { clearTimeout(nextTimer); nextTimer = null; }
      const started = performance.now();
      const response = await fetch("/reset", {method: "POST"});
      const result = await response.json();
      if (!response.ok) { status.textContent = result.error || "重置失败"; return; }
      await loadLatestFrame();
      vehicleSpeed = result.speed;
      pendingInputStartedAt = null;
      status.textContent = `已重置 · ${stateText(result, performance.now() - started, null)}`;
      if (token !== generation) return;
    }
    async function tick(token) {
      if (token !== generation || inFlight || boundaryBlocked) return;
      inFlight = true;
      if (nextTimer !== null) { clearTimeout(nextTimer); nextTimer = null; }
      const started = performance.now();
      const inputStarted = pendingInputStartedAt;
      try {
        const keys = [...held].sort().join("");
        const response = await fetch(
          "/tick?keys=" + encodeURIComponent(keys), {method: "POST"}
        );
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "控制请求失败");
        vehicleSpeed = result.speed;
        await loadLatestFrame();
        const loaded = performance.now();
        const inputToScreen = inputStarted === null ? null : loaded - inputStarted;
        if (pendingInputStartedAt === inputStarted) pendingInputStartedAt = null;
        status.textContent = stateText(result, loaded - started, inputToScreen);
        boundaryBlocked = result.boundary_hit;
      } catch (error) {
        status.textContent = error.toString();
      } finally {
        inFlight = false;
      }
      scheduleNext(token, started);
    }
  </script>
</body>
</html>
"""


def render_web_page(dt: float) -> str:
    return WEB_PAGE.replace(
        "__TICK_PERIOD_MS__", f"{dt * 1000.0:.6f}"
    ).replace(
        "__BOUNDARY_HELP__", boundary_help_text()
    )


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("--dt must be finite and positive")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")

    controller: WorldDrivingController
    rig = CameraRig(tuple(CameraSpec(name) for name in CAMERAS))
    runtime: dict[str, object] = {
        "state": EgoState(),
        "jpeg": b"",
        "metadata": {},
        "boundary_hit": False,
        "boundary_reason": None,
    }
    renderer: SplatADWorldRenderer

    def state_payload(started: float | None = None) -> dict[str, object]:
        state = runtime["state"]
        metadata = runtime["metadata"]
        assert isinstance(state, EgoState)
        assert isinstance(metadata, dict)
        measurement = controller.corridor_measurement(state)
        payload: dict[str, object] = {
            "backend": "h3_splatad_world",
            "simulation_time": state.time,
            "x": state.x,
            "y": state.y,
            "yaw_degrees": math.degrees(state.yaw),
            "speed": state.speed,
            "scene_time": metadata.get("scene_time_seconds"),
            "renderer_ms": float(metadata.get("render_seconds", 0.0)) * 1000.0,
            "boundary_hit": runtime["boundary_hit"],
            "boundary_reason": runtime["boundary_reason"],
            "corridor": None
            if measurement is None
            else {
                "progress_meters": measurement.progress,
                "lateral_offset_meters": measurement.lateral_offset,
                "distance_meters": measurement.distance,
                "heading_error_degrees": math.degrees(
                    measurement.heading_error
                ),
                "length_meters": controller.corridor.length,
                "half_width_meters": controller.corridor.half_width,
            },
            "certified_drivable_corridor": False,
            "output_scale": args.output_scale,
        }
        if started is not None:
            payload["server_control_to_jpeg_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
        return payload

    def render_candidate(state: EgoState) -> tuple[bytes, dict[str, object]]:
        from PIL import Image, ImageDraw

        observation = renderer.render("040", state, rig)
        mosaic = make_mosaic(
            Image,
            ImageDraw,
            dict(observation.frames),
            state,
        )
        buffer = io.BytesIO()
        mosaic.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue(), dict(observation.metadata)

    def commit_state(
        state: EgoState,
        jpeg: bytes,
        metadata: dict[str, object],
        *,
        boundary_hit: bool,
        boundary_reason: str | None,
    ) -> None:
        runtime.update(
            state=state,
            jpeg=jpeg,
            metadata=metadata,
            boundary_hit=boundary_hit,
            boundary_reason=boundary_reason,
        )

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
                page = render_web_page(args.dt)
                self.send_bytes(200, "text/html; charset=utf-8", page.encode())
            elif path == "/frame.jpg":
                self.send_bytes(200, "image/jpeg", runtime["jpeg"])  # type: ignore[arg-type]
            elif path == "/state.json":
                self.send_bytes(
                    200,
                    "application/json",
                    json.dumps(state_payload()).encode(),
                )
            else:
                self.send_bytes(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            started = time.perf_counter()
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/reset":
                    candidate = controller.reset()
                    jpeg, metadata = render_candidate(candidate)
                    commit_state(
                        candidate,
                        jpeg,
                        metadata,
                        boundary_hit=False,
                        boundary_reason=None,
                    )
                elif parsed.path == "/tick":
                    keys = set(parse_qs(parsed.query).get("keys", [""])[0])
                    if not keys <= {"w", "s", "a", "d"}:
                        raise ValueError("keys must contain only W/S/A/D")
                    current = runtime["state"]
                    assert isinstance(current, EgoState)
                    if should_step_world(keys, current.speed):
                        update = controller.step(
                            current,
                            control_for_keys(keys),
                            args.dt,
                        )
                        # Render first and commit second. A render error leaves
                        # the last valid state and JPEG intact.
                        jpeg, metadata = render_candidate(update.state)
                        commit_state(
                            update.state,
                            jpeg,
                            metadata,
                            boundary_hit=update.boundary_hit,
                            boundary_reason=update.boundary_reason,
                        )
                else:
                    self.send_bytes(404, "text/plain", b"not found")
                    return
                self.send_bytes(
                    200,
                    "application/json",
                    json.dumps(state_payload(started)).encode(),
                )
            except Exception as error:
                self.send_bytes(
                    400,
                    "application/json",
                    json.dumps({"error": str(error)}).encode(),
                )

        def log_message(self, format: str, *values: object) -> None:
            print(f"web: {format % values}")

    # Bind before loading the 8k checkpoint so an old process on the port fails
    # immediately instead of wasting GPU/model startup time. Unknown processes
    # are never killed automatically.
    server = HTTPServer((args.host, args.port), Handler)
    try:
        print("loading accepted static-8k and initial world-space frame...")
        renderer = SplatADWorldRenderer(
            args.config,
            output_scale=args.output_scale,
            anchor_log_time=args.anchor_log_time,
            # The data-derived corridor below is the runtime boundary. These
            # broad limits only protect against accidental unbounded queries.
            limits=NearbyPoseLimits(
                max_abs_forward_meters=100.0,
                max_abs_left_meters=100.0,
                max_abs_yaw_radians=math.pi,
            ),
        )
        renderer.load()
        if renderer.checkpoint_step != 7999:
            raise RuntimeError(
                f"expected accepted static step 7999, got {renderer.checkpoint_step}"
            )
        corridor = LoggedCenterlineCorridor(
            renderer.logged_centerline,
            half_width=CORRIDOR_HALF_WIDTH_METERS,
            max_heading_error=math.radians(
                CORRIDOR_MAX_HEADING_ERROR_DEGREES
            ),
        )
        controller = WorldDrivingController(
            corridor=corridor,
            spawn_state=corridor.pose_at_progress(0.0),
        )
        initial = controller.reset()
        jpeg, metadata = render_candidate(initial)
        commit_state(
            initial,
            jpeg,
            metadata,
            boundary_hit=False,
            boundary_reason=None,
        )
        print(f"browser_url=http://{args.host}:{args.port}")
        print(
            "provisional_logged_corridor="
            f"{corridor.length:.1f}m long, +/-{corridor.half_width:.1f}m, "
            f"heading error +/-{math.degrees(corridor.max_heading_error):.1f}deg"
        )
        print("manual_controls=W throttle, S brake, A/D steer, R reset")
        print("warning=trusted Tailnet/localhost only; one driving tab")
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping world-space browser driving loop")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
