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
    BrowserTrialRecorder,
    CameraRig,
    CameraSpec,
    EgoState,
    HumanControl,
    LoggedEgoOffsetController,
    SplatADLoggedRenderer,
    logged_movement_profile,
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
    parser.add_argument("--trial-output", type=Path, default=None)
    parser.add_argument(
        "--movement-profile",
        choices=("safe", "visible"),
        default="visible",
    )
    parser.add_argument(
        "--time-mode",
        choices=("manual", "auto"),
        default="manual",
        help=(
            "manual holds the log still until W/S/A/D is pressed; auto advances "
            "the logged trajectory at the fixed tick rate"
        ),
    )
    return parser.parse_args()


def control_for_keys(keys: set[str]) -> HumanControl:
    steer = float("a" in keys) - float("d" in keys)
    return HumanControl(
        steer=steer,
        throttle=float("w" in keys),
        brake=float("s" in keys),
    )


def should_advance_log_time(
    time_mode: str,
    keys: set[str],
    autoplay: bool = False,
    current_speed: float = 0.0,
) -> bool:
    if time_mode == "auto" or autoplay:
        return True
    if time_mode == "manual":
        return "w" in keys or current_speed > 1e-6
    raise ValueError(f"unknown time mode: {time_mode}")


def mode_help_text(time_mode: str) -> str:
    if time_mode == "manual":
        return (
            "默认人工控制：↑/W 加速，↓/S 减速，←/A 左转，"
            "→/D 右转；也可点击自动播放，R 重置。"
        )
    if time_mode == "auto":
        return (
            "自动前进：轨迹按固定节奏播放；W/S 调整反事实前向偏移，"
            "A/D 调整反事实朝向，R 重置。"
        )
    raise ValueError(f"unknown time mode: {time_mode}")


def make_mosaic(
    image_module: Any,
    image_draw_module: Any,
    frames: dict[str, object],
    state: EgoState,
    movement_profile: str,
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
        f"{movement_profile}  log={state.time:05.2f}s  "
        f"offset forward={state.x:+.2f}m left={state.y:+.2f}m  "
        f"yaw={math.degrees(state.yaw):+.1f}deg"
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
    body { margin: 0; overflow-x: hidden; background: #0d0d0d; color: #eee; font: 16px sans-serif; text-align: center; }
    h1 { font-size: 18px; margin: 4px; }
    #view { display: block; width: calc(100vw - 8px); height: auto; max-height: calc(100vh - 92px); margin: 0 auto; object-fit: contain; border: 1px solid #444; }
    #status { min-height: 20px; color: #ffd84d; margin: 2px 4px; }
    #toolbar { min-height: 34px; }
    #toolbar button { min-width: 68px; margin: 1px 2px; padding: 6px 9px; font-size: 14px; }
    #controls { max-width: 760px; margin: 6px auto; padding: 8px; background: #171717; border: 1px solid #333; }
    #mode-help { margin-bottom: 5px; color: #bbb; }
    #review { max-width: 1120px; margin: 8px auto 18px; padding: 10px; background: #171717; border: 1px solid #333; text-align: left; }
    #review h2 { margin: 0 0 6px; font-size: 17px; }
    #review p { margin: 4px 0 8px; color: #bbb; }
    .review-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 8px; }
    .review-grid label { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    select, textarea { background: #f3f3f3; color: #111; border: 1px solid #aaa; border-radius: 4px; }
    select { padding: 5px; }
    textarea { width: 100%; min-height: 48px; margin-top: 8px; box-sizing: border-box; }
    #review-status { min-height: 20px; color: #93c5fd; margin-top: 5px; }
    button { min-width: 76px; margin: 3px; padding: 10px; font-size: 17px; }
    a { color: #93c5fd; }
    [hidden] { display: none !important; }
  </style>
</head>
<body>
  <h1>PandaSet 场景 040 · 六相机驾驶</h1>
  <img id="view" src="/frame.jpg" alt="six reconstructed driving cameras">
  <div id="status">单驾驶客户端 · 请勿同时打开多个标签页</div>
  <div id="toolbar">
    <button id="autoplay">自动播放</button>
    <button data-key="r">R 重置</button>
    <button id="fullscreen">全屏</button>
    <button id="toggle-controls" aria-expanded="false">显示快捷键</button>
    <button id="toggle-review" aria-expanded="false">显示验收</button>
  </div>
  <section id="controls" hidden>
    <div id="mode-help">__MODE_HELP__</div>
    <div><button data-key="w">↑ / W 加速</button></div>
    <div>
      <button data-key="a">← / A 左转</button>
      <button data-key="s">↓ / S 减速</button>
      <button data-key="d">→ / D 右转</button>
    </div>
  </section>
  <section id="review" hidden>
    <h2>人工试驾验收</h2>
    <p>自动预检不等于最终验收。开完整段后，把“能不能开”的判断保存到同一个 trial JSON。</p>
    <div><a href="/trial.json" target="_blank">试驾记录 JSON</a></div>
    <div class="review-grid">
      <label>道路/车道/路缘
        <select data-review-gate="road_lane_curb_continuity">
          <option value="unsure">不确定</option>
          <option value="pass">通过</option>
          <option value="fail">失败</option>
        </select>
      </label>
      <label>转向画面方向
        <select data-review-gate="steering_response_direction">
          <option value="unsure">不确定</option>
          <option value="pass">通过</option>
          <option value="fail">失败</option>
        </select>
      </label>
      <label>近姿态破洞/撕裂
        <select data-review-gate="nearby_pose_artifact_impact">
          <option value="unsure">不确定</option>
          <option value="pass">通过</option>
          <option value="fail">失败</option>
        </select>
      </label>
      <label>物理输入延迟
        <select data-review-gate="physical_input_display_latency">
          <option value="unsure">不确定</option>
          <option value="pass">通过</option>
          <option value="fail">失败</option>
        </select>
      </label>
      <label>动态残影影响驾驶
        <select data-review-gate="dynamic_traffic_decision_impact">
          <option value="unsure">不确定</option>
          <option value="pass">通过</option>
          <option value="fail">失败</option>
        </select>
      </label>
    </div>
    <textarea id="review-notes" maxlength="2048" placeholder="可选备注：比如哪一帧车道断了、哪个方向的近车残影像假障碍物。"></textarea>
    <button id="save-review">保存人工验收</button>
    <div id="review-status">尚未保存人工验收。</div>
  </section>
  <script>
    const view = document.getElementById("view");
    const status = document.getElementById("status");
    const reviewStatus = document.getElementById("review-status");
    const autoplayButton = document.getElementById("autoplay");
    const controlsPanel = document.getElementById("controls");
    const reviewPanel = document.getElementById("review");
    const held = new Set();
    let running = true;
    let inFlight = false;
    let nextTimer = null;
    let sequence = 0;
    let generation = 0;
    let pendingInputStartedAt = null;
    let trialSamples = 0;
    let vehicleSpeed = 0;
    const tickPeriodMs = __TICK_PERIOD_MS__;
    const speedEpsilon = 1e-6;
    const timeMode = "__TIME_MODE__";
    const drivingKeys = new Set(["w", "s", "a", "d"]);
    const keyAliases = new Map([
      ["arrowup", "w"],
      ["arrowdown", "s"],
      ["arrowleft", "a"],
      ["arrowright", "d"]
    ]);
    let autoplay = timeMode === "auto";

    function updateAutoplayButton() {
      autoplayButton.textContent = autoplay ? "暂停自动播放" : "自动播放";
    }
    function setAutoplay(enabled) {
      autoplay = enabled;
      updateAutoplayButton();
      if (!autoplay && nextTimer !== null) {
        clearTimeout(nextTimer);
        nextTimer = null;
      }
      if (autoplay) requestTickNow();
    }
    function normalizedDrivingKey(key) {
      const lowered = key.toLowerCase();
      return keyAliases.get(lowered) || lowered;
    }
    function bindPanelToggle(buttonId, panel, showText, hideText) {
      const button = document.getElementById(buttonId);
      button.addEventListener("click", () => {
        panel.hidden = !panel.hidden;
        button.textContent = panel.hidden ? showText : hideText;
        button.setAttribute("aria-expanded", String(!panel.hidden));
      });
    }

    function markInputEdge() {
      pendingInputStartedAt = performance.now();
      requestTickNow();
    }
    function setHeld(key, active) {
      const wasHeld = held.has(key);
      if (active && !wasHeld && autoplay) setAutoplay(false);
      if (active) held.add(key); else held.delete(key);
      if (wasHeld !== active) markInputEdge();
    }
    function scheduleNextTick(token, started) {
      if (!running || token !== generation) return;
      if (!autoplay && !held.has("w") && vehicleSpeed <= speedEpsilon) return;
      if (nextTimer !== null) clearTimeout(nextTimer);
      nextTimer = setTimeout(
        () => tick(token),
        Math.max(0, tickPeriodMs - (performance.now() - started))
      );
    }
    function requestTickNow() {
      if (!running || inFlight) return;
      if (nextTimer !== null) {
        clearTimeout(nextTimer);
        nextTimer = null;
      }
      tick(generation);
    }
    async function recordSample(sample) {
      try {
        const response = await fetch("/trial-sample", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(sample)
        });
        const result = await response.json();
        if (response.ok && result.summary) {
          trialSamples = result.summary.sample_count;
        }
      } catch (_) {
        /* Keep driving even if recording fails. */
      }
    }
    async function saveManualReview() {
      const gates = {};
      document.querySelectorAll("[data-review-gate]").forEach(select => {
        gates[select.dataset.reviewGate] = select.value;
      });
      try {
        const response = await fetch("/trial-review", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            client_unix_ms: Date.now(),
            reviewer: "browser_operator",
            gates,
            notes: document.getElementById("review-notes").value
          })
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "保存失败");
        const summary = result.summary;
        const verdict = summary.manual_review_all_passed ? "全部通过" : "仍有未通过/不确定项";
        reviewStatus.textContent =
          `已保存第 ${summary.manual_review_count} 次人工验收：${verdict}`;
      } catch (error) {
        reviewStatus.textContent = error.toString();
      }
    }
    document.addEventListener("keydown", event => {
      const key = normalizedDrivingKey(event.key);
      if (drivingKeys.has(key)) { event.preventDefault(); setHeld(key, true); }
      if (key === "r" && !event.repeat) { event.preventDefault(); reset(); }
    });
    document.addEventListener("keyup", event => {
      const key = normalizedDrivingKey(event.key);
      if (drivingKeys.has(key)) { event.preventDefault(); setHeld(key, false); }
    });
    window.addEventListener("blur", () => {
      if (held.size) {
        held.clear();
        markInputEdge();
      }
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
    autoplayButton.addEventListener("click", () => setAutoplay(!autoplay));
    bindPanelToggle("toggle-controls", controlsPanel, "显示快捷键", "隐藏快捷键");
    bindPanelToggle("toggle-review", reviewPanel, "显示验收", "隐藏验收");
    document.getElementById("save-review").addEventListener("click", saveManualReview);

    async function reset() {
      held.clear();
      const token = ++generation;
      running = false;
      const resetStarted = performance.now();
      if (nextTimer !== null) {
        clearTimeout(nextTimer);
        nextTimer = null;
      }
      await fetch("/reset", {method: "POST"});
      await new Promise((resolve, reject) => {
        view.onload = resolve;
        view.onerror = reject;
        view.src = "/frame.jpg?n=" + (++sequence);
      });
      const resetToScreen = performance.now() - resetStarted;
      pendingInputStartedAt = null;
      vehicleSpeed = 0;
      status.textContent =
        `log 0.00s · 重置→图像加载 ${resetToScreen.toFixed(0)}ms · ` +
        `记录 ${trialSamples}`;
      running = true;
      if (autoplay) {
        tick(token);
      }
    }
    async function tick(token) {
      if (token !== generation || inFlight) return;
      inFlight = true;
      if (nextTimer !== null) {
        clearTimeout(nextTimer);
        nextTimer = null;
      }
      const started = performance.now();
      const inputStartedAt = pendingInputStartedAt;
      try {
        const keys = [...held].sort().join("");
        const query = encodeURIComponent(keys);
        const autoplayQuery = autoplay ? "1" : "0";
        const response = await fetch(
          "/tick?keys=" + query + "&autoplay=" + autoplayQuery,
          {method: "POST"}
        );
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "request failed");
        vehicleSpeed = result.speed;
        await new Promise((resolve, reject) => {
          view.onload = resolve;
          view.onerror = reject;
          view.src = "/frame.jpg?n=" + (++sequence);
        });
        const loadedAt = performance.now();
        const eventToScreen = loadedAt - started;
        let inputToScreen = null;
        if (inputStartedAt !== null) {
          inputToScreen = loadedAt - inputStartedAt;
          if (pendingInputStartedAt === inputStartedAt) {
            pendingInputStartedAt = null;
          }
        }
        const displayedSampleCount = trialSamples + 1;
        recordSample({
          sequence,
          keys,
          client_unix_ms: Date.now(),
          browser_request_to_image_ms: eventToScreen,
          browser_input_to_image_ms: inputToScreen,
          server: result
        });
        status.textContent =
          `log ${result.time.toFixed(2)}s / ${result.duration.toFixed(2)}s · ` +
          `请求→图像加载 ${eventToScreen.toFixed(0)}ms`;
        if (inputToScreen !== null) {
          status.textContent += ` · 输入→图像加载 ${inputToScreen.toFixed(0)}ms`;
        }
        status.textContent += ` · 记录 ${displayedSampleCount}`;
        if (result.time >= result.duration) {
          running = false;
          autoplay = false;
          updateAutoplayButton();
          status.textContent += " · 已到终点，按 R 重置";
        }
      } catch (error) {
        status.textContent = error.toString();
        running = false;
      } finally {
        inFlight = false;
      }
      scheduleNextTick(token, started);
    }
    updateAutoplayButton();
    if (autoplay) {
      tick(generation);
    } else {
      status.textContent = "人工控制 · ↑/W 加速 · ↓/S 减速 · ←/A 左转 · →/D 右转 · R 重置";
    }
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

    profile = logged_movement_profile(args.movement_profile)
    renderer = SplatADLoggedRenderer(
        args.config,
        output_scale=args.output_scale,
        limits=profile.limits,
    )
    renderer.load()
    if renderer.checkpoint_step != 7999:
        raise RuntimeError(
            f"expected accepted static step 7999, got {renderer.checkpoint_step}"
        )
    controller = LoggedEgoOffsetController.from_profile(
        renderer.logged_duration,
        profile,
    )
    rig = CameraRig(tuple(CameraSpec(name) for name in CAMERAS))
    recorder = BrowserTrialRecorder(
        scene="040",
        movement_profile=profile.name,
        output_scale=args.output_scale,
        dt_seconds=args.dt,
        checkpoint_step=renderer.checkpoint_step,
        logged_duration_seconds=renderer.logged_duration,
        output_path=args.trial_output,
    )
    runtime: dict[str, object] = {
        "state": controller.reset(),
        "jpeg": b"",
        "metadata": {},
    }

    def render_frame() -> None:
        state = runtime["state"]
        assert isinstance(state, EgoState)
        observation = renderer.render("040", state, rig)
        mosaic = make_mosaic(
            Image,
            ImageDraw,
            dict(observation.frames),
            state,
            profile.name,
        )
        buffer = io.BytesIO()
        mosaic.save(buffer, format="JPEG", quality=88)
        runtime["jpeg"] = buffer.getvalue()
        runtime["metadata"] = dict(observation.metadata)

    def state_payload(started: float | None = None) -> dict[str, object]:
        state = runtime["state"]
        metadata = runtime["metadata"]
        assert isinstance(state, EgoState)
        assert isinstance(metadata, dict)
        payload: dict[str, object] = {
            "time": state.time,
            "duration": renderer.logged_duration,
            "x": state.x,
            "y": state.y,
            "yaw_degrees": math.degrees(state.yaw),
            "speed": state.speed,
            "renderer_ms": float(metadata["render_seconds"]) * 1000.0,
            "movement_profile": profile.name,
            "logical_frame": int(metadata["logical_frame"]),
            "frame_selection_error_ms": float(
                metadata["frame_selection_error_ms"]
            ),
            "camera_time_spread_ms": float(metadata["camera_time_spread_ms"]),
            "output_scale": args.output_scale,
            "time_mode": args.time_mode,
        }
        if started is not None:
            payload["server_control_to_jpeg_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
        return payload

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
                ).replace(
                    "__TIME_MODE__",
                    args.time_mode,
                ).replace(
                    "__MODE_HELP__",
                    mode_help_text(args.time_mode),
                )
                self.send_bytes(200, "text/html; charset=utf-8", page.encode())
            elif path == "/frame.jpg":
                self.send_bytes(200, "image/jpeg", runtime["jpeg"])  # type: ignore[arg-type]
            elif path == "/trial.json":
                self.send_bytes(
                    200,
                    "application/json",
                    recorder.report_bytes(),
                )
            else:
                self.send_bytes(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            started = time.perf_counter()
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/trial-sample":
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if content_length > 65536:
                        raise ValueError("trial sample is too large")
                    raw_payload = self.rfile.read(content_length)
                    payload = json.loads(raw_payload.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("trial sample must be a JSON object")
                    summary = recorder.record_sample(payload)
                    response = {
                        "summary": summary,
                        "trial_output": (
                            str(recorder.output_path)
                            if recorder.output_path
                            else None
                        ),
                    }
                    self.send_bytes(
                        200,
                        "application/json",
                        json.dumps(response).encode(),
                    )
                    return
                if parsed.path == "/trial-review":
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if content_length > 8192:
                        raise ValueError("trial review is too large")
                    raw_payload = self.rfile.read(content_length)
                    payload = json.loads(raw_payload.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("trial review must be a JSON object")
                    summary = recorder.record_manual_review(payload)
                    response = {
                        "summary": summary,
                        "trial_output": (
                            str(recorder.output_path)
                            if recorder.output_path
                            else None
                        ),
                    }
                    self.send_bytes(
                        200,
                        "application/json",
                        json.dumps(response).encode(),
                    )
                    return
                render_needed = False
                if parsed.path == "/reset":
                    runtime["state"] = controller.reset()
                    render_needed = True
                elif parsed.path == "/tick":
                    query = parse_qs(parsed.query)
                    keys = set(query.get("keys", [""])[0])
                    if not keys <= {"w", "s", "a", "d"}:
                        raise ValueError("keys must contain only W/S/A/D")
                    autoplay_value = query.get(
                        "autoplay",
                        ["1" if args.time_mode == "auto" else "0"],
                    )[0]
                    if autoplay_value not in {"0", "1"}:
                        raise ValueError("autoplay must be 0 or 1")
                    autoplay = autoplay_value == "1"
                    state = runtime["state"]
                    assert isinstance(state, EgoState)
                    should_advance = should_advance_log_time(
                        "manual",
                        keys,
                        autoplay,
                        state.speed,
                    )
                    if should_advance:
                        runtime["state"] = controller.step(
                            state,
                            control_for_keys(keys),
                            args.dt,
                        )
                        render_needed = True
                else:
                    self.send_bytes(404, "text/plain", b"not found")
                    return
                if render_needed:
                    render_frame()
                payload = state_payload(started)
                if parsed.path == "/reset":
                    recorder.record_reset(payload)
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
    print(
        "movement_profile="
        f"{profile.name} "
        f"limits=+/-{profile.limits.max_abs_forward_meters:.2f}m forward, "
        f"+/-{profile.limits.max_abs_left_meters:.2f}m left, "
        f"+/-{math.degrees(profile.limits.max_abs_yaw_radians):.1f}deg yaw"
    )
    print(f"time_mode={args.time_mode}")
    if args.trial_output:
        print(f"trial_output={Path(args.trial_output).expanduser().resolve()}")
    print("trial_report_endpoint=/trial.json")
    print("controls=W/S/A/D, reset=R")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping browser driving loop")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
