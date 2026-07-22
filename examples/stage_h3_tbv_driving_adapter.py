#!/usr/bin/env python3
"""Minimal route-constrained TbV browser with an auditable evidence outlet."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
import time
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    BranchedRouteDrivingAdapter,
    EgoState,
    HumanControl,
    LoggedCenterlineCorridor,
    LoggedCenterlineSample,
    RouteDrivingEvidenceRecorder,
    SupportedRoute,
)
from stage_h3_tbv_world_pose_probe import (  # noqa: E402
    CAMERAS,
    RIGHT_TRAVERSAL,
    STRAIGHT_TRAVERSAL,
    LocalWorldPose,
    RouteSample,
    TbVWorldRenderer,
    pose_at_progress,
)


DEFAULT_CONFIG = Path(
    "/home/yawei/stage3_external/outputs/tbv_h3/"
    "tbv_branch_pair_splatad_static_8000/splatad/"
    "2026-07-22_resume_2k_to_8k/config.yml"
)
DEFAULT_EVIDENCE = Path(
    "/home/yawei/stage3_external/artifacts/"
    "tbv_branch_pair_driving_adapter/tbv_driving_evidence.json"
)
COMMON_START_METERS = -20.0
BRANCH_ANCHOR_METERS = 0.0
STRAIGHT_END_METERS = 40.0
RIGHT_END_METERS = 30.0
CORRIDOR_HALF_WIDTH_METERS = 1.0
CORRIDOR_HEADING_LIMIT_DEGREES = 30.0
SELECTION_WINDOW_METERS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-scale", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--evidence-output", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--expected-checkpoint-step", type=int, default=7999)
    return parser.parse_args()


def control_for_keys(keys: set[str]) -> HumanControl:
    braking = "s" in keys
    return HumanControl(
        steer=float("a" in keys) - float("d" in keys),
        throttle=float("w" in keys and not braking),
        brake=float(braking),
    )


def route_segment(
    route: tuple[RouteSample, ...], start: float, end: float
) -> tuple[LocalWorldPose, ...]:
    if not start < end:
        raise ValueError("route segment start must precede end")
    poses = [pose_at_progress(route, start)]
    poses.extend(
        LocalWorldPose(sample.x, sample.y, sample.z, sample.yaw)
        for sample in route
        if start < sample.progress < end
    )
    poses.append(pose_at_progress(route, end))
    deduplicated: list[LocalWorldPose] = []
    for pose in poses:
        if deduplicated and math.hypot(
            pose.x - deduplicated[-1].x, pose.y - deduplicated[-1].y
        ) <= 1e-6:
            continue
        deduplicated.append(pose)
    if len(deduplicated) < 2:
        raise RuntimeError("route segment contains fewer than two distinct poses")
    return tuple(deduplicated)


def supported_route(
    *,
    name: str,
    renderer_profile: str,
    route: tuple[RouteSample, ...],
    start: float,
    end: float,
) -> SupportedRoute:
    poses = route_segment(route, start, end)
    samples = tuple(
        LoggedCenterlineSample(
            logical_frame=index,
            log_time=float(index),
            x=pose.x,
            y=pose.y,
            yaw=pose.yaw,
        )
        for index, pose in enumerate(poses)
    )
    return SupportedRoute(
        name=name,
        renderer_profile=renderer_profile,
        corridor=LoggedCenterlineCorridor(
            samples,
            half_width=CORRIDOR_HALF_WIDTH_METERS,
            max_heading_error=math.radians(CORRIDOR_HEADING_LIMIT_DEGREES),
        ),
        start_progress_from_anchor=start,
    )


def make_adapter(renderer: TbVWorldRenderer) -> BranchedRouteDrivingAdapter:
    right_route = renderer.routes[RIGHT_TRAVERSAL]
    straight_route = renderer.routes[STRAIGHT_TRAVERSAL]
    common = supported_route(
        name="common",
        renderer_profile=RIGHT_TRAVERSAL,
        route=right_route,
        start=COMMON_START_METERS,
        end=BRANCH_ANCHOR_METERS,
    )
    straight = supported_route(
        name="straight",
        renderer_profile=STRAIGHT_TRAVERSAL,
        route=straight_route,
        start=COMMON_START_METERS,
        end=STRAIGHT_END_METERS,
    )
    right = supported_route(
        name="right",
        renderer_profile=RIGHT_TRAVERSAL,
        route=right_route,
        start=COMMON_START_METERS,
        end=RIGHT_END_METERS,
    )
    spawn = pose_at_progress(right_route, COMMON_START_METERS)
    return BranchedRouteDrivingAdapter(
        common_route=common,
        branches={"straight": straight, "right": right},
        spawn_state=EgoState(x=spawn.x, y=spawn.y, yaw=spawn.yaw),
        selection_window_meters=SELECTION_WINDOW_METERS,
    )


def route_height(
    renderer: TbVWorldRenderer,
    renderer_profile: str,
    progress_from_anchor: float,
) -> float:
    route = renderer.routes[renderer_profile]
    progress = min(route[-1].progress, max(route[0].progress, progress_from_anchor))
    return pose_at_progress(route, progress).z


def make_mosaic(
    image_module: Any,
    draw_module: Any,
    frames: dict[str, Any],
    state: EgoState,
    support: dict[str, object],
) -> Any:
    order = (
        "ring_front_left",
        "ring_front_center",
        "ring_front_right",
        "ring_side_left",
        "ring_side_right",
        "ring_rear_left",
        "ring_rear_right",
    )
    canvas_width = 1560
    top_height = 520
    bottom_height = 300
    status_height = 50
    canvas = image_module.new(
        "RGB", (canvas_width, top_height + bottom_height + status_height), "black"
    )
    boxes = (
        (0, 0, 520, top_height),
        (520, 0, 520, top_height),
        (1040, 0, 520, top_height),
        (0, top_height, 390, bottom_height),
        (1170, top_height, 390, bottom_height),
        (390, top_height, 390, bottom_height),
        (780, top_height, 390, bottom_height),
    )
    for name, (x, y, width, height) in zip(order, boxes):
        tile = image_module.fromarray(frames[name]).convert("RGB")
        tile.thumbnail((width, height), image_module.Resampling.LANCZOS)
        draw_module.Draw(tile).text(
            (7, 6), name, fill=(0, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0)
        )
        canvas.paste(
            tile,
            (x + (width - tile.width) // 2, y + (height - tile.height) // 2),
        )
    status = (
        f"TbV evidence-only | {support['phase']} | "
        f"branch={support['selected_branch'] or '-'} | "
        f"progress={float(support['progress_from_anchor_meters']):+.1f}m | "
        f"offset={float(support['lateral_offset_meters']):+.2f}m | "
        f"speed={state.speed:.2f}m/s"
    )
    draw_module.Draw(canvas).text(
        (10, top_height + bottom_height + 14), status, fill=(255, 216, 77)
    )
    return canvas


WEB_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TbV 路线约束驾驶 v0</title>
  <style>
    body { margin:0; background:#0d0d0d; color:#eee; font:15px sans-serif; text-align:center; }
    h1 { font-size:18px; margin:5px; }
    #view { display:block; width:calc(100vw - 8px); max-height:calc(100vh - 128px); object-fit:contain; margin:auto; border:1px solid #444; }
    #status { min-height:22px; color:#ffd84d; margin:4px; }
    button { margin:2px; padding:7px 12px; font-size:14px; }
    #branches { color:#ffb27a; }
    #branches[hidden] { display:none; }
    a { color:#8ed8ff; }
  </style>
</head>
<body>
  <h1>TbV shared entrance -20m · ±1m evidence-only pilot</h1>
  <img id="view" src="/frame.jpg" alt="seven reconstructed driving cameras">
  <div id="status">W 油门 · S 刹车 · A/D 转向 · 到锚点后选择分支 · R 重置</div>
  <div id="branches" hidden>
    已到共享锚点：<button data-branch="straight">1 / 直行</button>
    <button data-branch="right">2 / 右转</button>
  </div>
  <div><button id="reset">R 重置</button> <a href="/evidence.json" target="_blank">可信证据 JSON</a></div>
  <script>
    const view = document.getElementById("view");
    const status = document.getElementById("status");
    const branches = document.getElementById("branches");
    const held = new Set();
    const aliases = new Map([["arrowup","w"],["arrowdown","s"],["arrowleft","a"],["arrowright","d"]]);
    const driving = new Set(["w","a","s","d"]);
    const tickPeriodMs = __TICK_PERIOD_MS__;
    let speed = 0, inFlight = false, timer = null, imageSequence = 0, generation = 0;
    let blocked = false, selectionRequired = false, pendingInputAt = null;

    function keyName(value) { const key=value.toLowerCase(); return aliases.get(key)||key; }
    function setHeld(key, active) {
      const changed = held.has(key) !== active;
      if (active) held.add(key); else held.delete(key);
      if (changed) { pendingInputAt=performance.now(); requestTick(); }
    }
    function requestTick() {
      if (inFlight || blocked || selectionRequired) return;
      if (timer !== null) { clearTimeout(timer); timer=null; }
      tick(generation);
    }
    function schedule(token, started) {
      if (blocked || selectionRequired || token !== generation) return;
      if (!held.has("w") && speed <= 1e-6) return;
      timer=setTimeout(()=>tick(token), Math.max(0,tickPeriodMs-(performance.now()-started)));
    }
    async function loadFrame() {
      await new Promise((resolve,reject)=>{
        view.onload=resolve; view.onerror=reject; view.src="/frame.jpg?n="+(++imageSequence);
      });
    }
    function stateText(result, requestMs, inputMs) {
      const route=result.route_support;
      let text=`${route.phase} · ${result.selected_branch||"未选路"} · 进度 ${route.progress_from_anchor_meters.toFixed(1)}m · `+
        `横向 ${route.lateral_offset_meters.toFixed(2)}m · 速度 ${result.speed_mps.toFixed(2)}m/s · 请求→画面 ${requestMs.toFixed(0)}ms`;
      if (inputMs !== null) text+=` · 输入→画面 ${inputMs.toFixed(0)}ms`;
      if (result.boundary_hit) text+=` · 已停止：${result.boundary_reason}`;
      return text;
    }
    async function saveBrowserTiming(result, requestMs, inputMs) {
      const response=await fetch("/evidence-sample", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
        sequence:result.sequence, client_unix_ms:Date.now(), browser_request_to_image_ms:requestMs,
        browser_input_to_image_ms:inputMs
      })});
      if (!response.ok) throw new Error((await response.json()).error||"证据写入失败");
    }
    async function tick(token) {
      if (token!==generation || inFlight || blocked || selectionRequired) return;
      inFlight=true; if (timer!==null) { clearTimeout(timer); timer=null; }
      const started=performance.now(), inputStarted=pendingInputAt;
      try {
        const keys=[...held].sort().join("");
        const response=await fetch("/tick?keys="+encodeURIComponent(keys),{method:"POST"});
        const result=await response.json(); if (!response.ok) throw new Error(result.error||"控制失败");
        await loadFrame(); const loaded=performance.now();
        const requestMs=loaded-started, inputMs=inputStarted===null?null:loaded-inputStarted;
        await saveBrowserTiming(result,requestMs,inputMs);
        if (pendingInputAt===inputStarted) pendingInputAt=null;
        speed=result.speed_mps; blocked=result.boundary_hit; selectionRequired=result.selection_required;
        branches.hidden=!selectionRequired; status.textContent=stateText(result,requestMs,inputMs);
      } catch(error) { status.textContent=error.toString(); }
      finally { inFlight=false; }
      schedule(token,started);
    }
    async function chooseBranch(branch) {
      if (inFlight) return; inFlight=true; const started=performance.now();
      try {
        const response=await fetch("/branch?name="+branch,{method:"POST"});
        const result=await response.json(); if (!response.ok) throw new Error(result.error||"选路失败");
        await loadFrame(); const loaded=performance.now();
        const timingResponse=await fetch("/evidence-route-timing", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
          event_index:result.route_event_index, client_unix_ms:Date.now(), browser_selection_to_image_ms:loaded-started
        })});
        if (!timingResponse.ok) throw new Error((await timingResponse.json()).error||"选路证据写入失败");
        speed=result.speed_mps; selectionRequired=false; blocked=false; branches.hidden=true;
        status.textContent=`已选择 ${branch} · 切换渲染 ${result.server_route_selection_to_jpeg_ms.toFixed(0)}ms`;
      } catch(error) { status.textContent=error.toString(); }
      finally { inFlight=false; }
      requestTick();
    }
    async function reset() {
      held.clear(); ++generation; blocked=false; selectionRequired=false; branches.hidden=true;
      if (timer!==null) { clearTimeout(timer); timer=null; }
      const response=await fetch("/reset",{method:"POST"}); const result=await response.json();
      if (!response.ok) { status.textContent=result.error||"重置失败"; return; }
      await loadFrame(); speed=0; pendingInputAt=null; status.textContent="已重置到公共入口 -20m";
    }
    document.addEventListener("keydown",event=>{
      const key=keyName(event.key);
      if (driving.has(key)) { event.preventDefault(); setHeld(key,true); }
      if (key==="r"&&!event.repeat) { event.preventDefault(); reset(); }
      if (selectionRequired&&key==="1") chooseBranch("straight");
      if (selectionRequired&&key==="2") chooseBranch("right");
    });
    document.addEventListener("keyup",event=>{ const key=keyName(event.key); if(driving.has(key)){event.preventDefault();setHeld(key,false);} });
    window.addEventListener("blur",()=>{ if(held.size){held.clear();pendingInputAt=performance.now();requestTick();} });
    document.querySelectorAll("button[data-branch]").forEach(button=>button.addEventListener("click",()=>chooseBranch(button.dataset.branch)));
    document.getElementById("reset").addEventListener("click",reset);
  </script>
</body>
</html>
"""


def render_web_page(dt: float) -> str:
    return WEB_PAGE.replace("__TICK_PERIOD_MS__", f"{dt * 1000.0:.6f}")


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0 or length > 64_000:
        raise ValueError("JSON request body is empty or too large")
    value = json.loads(handler.rfile.read(length))
    if not isinstance(value, dict):
        raise ValueError("JSON request body must be an object")
    return value


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("--dt must be finite and positive")
    if not 0.0 < args.output_scale <= 1.0:
        raise ValueError("--output-scale must be in (0, 1]")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")

    renderer = TbVWorldRenderer(args.config, args.output_scale)
    adapter: BranchedRouteDrivingAdapter
    evidence: RouteDrivingEvidenceRecorder
    runtime: dict[str, Any] = {
        "state": EgoState(), "jpeg": b"", "render": {}, "sequence": -1,
        "boundary_hit": False, "boundary_reason": None, "selection_required": False,
    }

    def payload(extra: dict[str, object] | None = None) -> dict[str, object]:
        state: EgoState = runtime["state"]
        support = adapter.support(state)
        value: dict[str, object] = {
            "backend": "tbv_route_driving_adapter_v0",
            "sequence": runtime["sequence"],
            "simulation_time_seconds": state.time,
            "x_meters": state.x,
            "y_meters": state.y,
            "yaw_degrees": math.degrees(state.yaw),
            "speed_mps": state.speed,
            "selected_branch": adapter.selected_branch,
            "selection_required": runtime["selection_required"],
            "boundary_hit": runtime["boundary_hit"],
            "boundary_reason": runtime["boundary_reason"],
            "route_support": support.as_dict(),
            "branch_options": (
                {
                    name: adapter.branch_support(name, state).as_dict()
                    for name in sorted(adapter.branches)
                }
                if support.selection_required
                else None
            ),
            "renderer_profile": support.renderer_profile,
            "renderer_ms": float(runtime["render"].get("render_seconds", 0.0)) * 1000.0,
            "frozen_scene_time_seconds": renderer.model_scene_time,
            "evidence_url": "/evidence.json",
            "certified_drivable_corridor": False,
        }
        if extra:
            value.update(extra)
        return value

    def render_state(state: EgoState) -> tuple[bytes, dict[str, object]]:
        from PIL import Image, ImageDraw

        support = adapter.support(state)
        z = route_height(
            renderer, support.renderer_profile, support.progress_from_anchor_meters
        )
        observation = renderer.render(
            support.renderer_profile,
            LocalWorldPose(state.x, state.y, z, state.yaw),
        )
        frames = dict(observation["frames"])
        mosaic = make_mosaic(Image, ImageDraw, frames, state, support.as_dict())
        buffer = io.BytesIO()
        mosaic.save(buffer, format="JPEG", quality=88)
        return buffer.getvalue(), {
            "render_seconds": float(observation["render_seconds"]),
            "scene_time_seconds": float(observation["scene_time_seconds"]),
            "renderer_profile": support.renderer_profile,
            "camera_count": len(frames),
        }

    def commit(
        state: EgoState,
        jpeg: bytes,
        render: dict[str, object],
        *,
        boundary_hit: bool = False,
        boundary_reason: str | None = None,
        selection_required: bool = False,
    ) -> None:
        runtime.update(
            state=state,
            jpeg=jpeg,
            render=render,
            boundary_hit=boundary_hit,
            boundary_reason=boundary_reason,
            selection_required=selection_required,
        )

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status: int, content_type: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, status: int, value: object) -> None:
            self.send_bytes(status, "application/json", json.dumps(value).encode())

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(200, "text/html; charset=utf-8", render_web_page(args.dt).encode())
            elif path == "/frame.jpg":
                self.send_bytes(200, "image/jpeg", runtime["jpeg"])
            elif path == "/state.json":
                self.send_json(200, payload())
            elif path == "/evidence.json":
                self.send_bytes(200, "application/json", evidence.report_bytes())
            else:
                self.send_bytes(404, "text/plain", b"not found")

        def do_POST(self) -> None:  # noqa: N802
            started = time.perf_counter()
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/tick":
                    if runtime["boundary_hit"]:
                        raise ValueError("reconstruction boundary reached; reset is required")
                    if runtime["selection_required"]:
                        raise ValueError("branch selection is required before driving")
                    keys = set(parse_qs(parsed.query).get("keys", [""])[0])
                    if not keys <= set("wasd"):
                        raise ValueError("keys must contain only W/S/A/D")
                    state: EgoState = runtime["state"]
                    update = adapter.step(state, control_for_keys(keys), args.dt)
                    jpeg, render = render_state(update.state)
                    runtime["sequence"] += 1
                    commit(
                        update.state,
                        jpeg,
                        render,
                        boundary_hit=update.boundary_hit,
                        boundary_reason=update.boundary_reason,
                        selection_required=update.selection_required,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    response = payload({"server_control_to_jpeg_ms": elapsed_ms})
                    support = adapter.support(update.state)
                    evidence.record_server_sample(
                        {
                            "sequence": runtime["sequence"],
                            "control_keys": "".join(sorted(keys)),
                            "simulation_time_seconds": update.state.time,
                            "x_meters": update.state.x,
                            "y_meters": update.state.y,
                            "yaw_degrees": math.degrees(update.state.yaw),
                            "speed_mps": update.state.speed,
                            "route_support": support.as_dict(),
                            "renderer_profile": support.renderer_profile,
                            "frozen_scene_time_seconds": renderer.model_scene_time,
                            "camera_count": render["camera_count"],
                            "all_camera_frames_finite": True,
                            "renderer_ms": float(render["render_seconds"]) * 1000.0,
                            "server_control_to_jpeg_ms": elapsed_ms,
                            "frame_sha256": hashlib.sha256(jpeg).hexdigest(),
                            "boundary_hit": update.boundary_hit,
                            "boundary_reason": update.boundary_reason,
                        }
                    )
                    if update.selection_required and not any(
                        event["event"] == "branch_selection_required"
                        for event in evidence.route_events[-1:]
                    ):
                        evidence.record_route_event(
                            "branch_selection_required",
                            {
                                "simulation_time_seconds": update.state.time,
                                "route_support": support.as_dict(),
                                "branch_options": {
                                    name: adapter.branch_support(name, update.state).as_dict()
                                    for name in sorted(adapter.branches)
                                },
                            },
                        )
                    self.send_json(200, response)
                    return
                if parsed.path == "/branch":
                    branch = parse_qs(parsed.query).get("name", [""])[0]
                    state = runtime["state"]
                    before_hash = hashlib.sha256(runtime["jpeg"]).hexdigest()
                    before_profile = adapter.support(state).renderer_profile
                    support = adapter.select_branch(branch, state)
                    jpeg, render = render_state(state)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    commit(state, jpeg, render)
                    event = evidence.record_route_event(
                        "branch_selected",
                        {
                            "branch": branch,
                            "simulation_time_seconds": state.time,
                            "route_support": support.as_dict(),
                            "renderer_profile_before": before_profile,
                            "renderer_profile_after": support.renderer_profile,
                            "frame_sha256_before": before_hash,
                            "frame_sha256_after": hashlib.sha256(jpeg).hexdigest(),
                            "server_route_selection_to_jpeg_ms": elapsed_ms,
                        },
                    )
                    self.send_json(
                        200,
                        payload(
                            {
                                "server_route_selection_to_jpeg_ms": elapsed_ms,
                                "route_event_index": event["event_index"],
                            }
                        ),
                    )
                    return
                if parsed.path == "/reset":
                    state = adapter.reset()
                    jpeg, render = render_state(state)
                    commit(state, jpeg, render)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    evidence.record_reset(
                        {
                            "simulation_time_seconds": state.time,
                            "route_support": adapter.support(state).as_dict(),
                            "frame_sha256": hashlib.sha256(jpeg).hexdigest(),
                            "server_reset_to_jpeg_ms": elapsed_ms,
                        }
                    )
                    self.send_json(200, payload({"server_reset_to_jpeg_ms": elapsed_ms}))
                    return
                if parsed.path == "/evidence-sample":
                    value = read_json_body(self)
                    evidence.record_browser_timing(
                        sequence=int(value.get("sequence", -1)),
                        browser_request_to_image_ms=value.get("browser_request_to_image_ms"),
                        browser_input_to_image_ms=value.get("browser_input_to_image_ms"),
                        client_unix_ms=value.get("client_unix_ms"),
                    )
                    self.send_json(200, evidence.summary())
                    return
                if parsed.path == "/evidence-route-timing":
                    value = read_json_body(self)
                    evidence.record_route_browser_timing(
                        event_index=int(value.get("event_index", -1)),
                        browser_selection_to_image_ms=value.get(
                            "browser_selection_to_image_ms"
                        ),
                        client_unix_ms=value.get("client_unix_ms"),
                    )
                    self.send_json(200, evidence.summary())
                    return
                self.send_bytes(404, "text/plain", b"not found")
            except Exception as error:
                self.send_json(400, {"error": str(error)})

        def log_message(self, format: str, *values: object) -> None:
            print(f"web: {format % values}")

    # Reserve the port before loading the 1.65 GB checkpoint. Unknown services
    # are never killed automatically.
    server = HTTPServer((args.host, args.port), Handler)
    try:
        renderer.load()
        if renderer.checkpoint_step != args.expected_checkpoint_step:
            raise RuntimeError(
                f"expected checkpoint step {args.expected_checkpoint_step}, "
                f"got {renderer.checkpoint_step}"
            )
        adapter = make_adapter(renderer)
        assert renderer.checkpoint_path is not None
        evidence = RouteDrivingEvidenceRecorder(
            scene="tbv_miami_shared_entrance_straight_right",
            config_path=args.config,
            checkpoint_path=renderer.checkpoint_path,
            checkpoint_step=renderer.checkpoint_step,
            output_scale=args.output_scale,
            dt_seconds=args.dt,
            camera_names=CAMERAS,
            route_contract={
                "spawn_progress_from_anchor_meters": COMMON_START_METERS,
                "branch_anchor_progress_meters": BRANCH_ANCHOR_METERS,
                "straight_end_progress_meters": STRAIGHT_END_METERS,
                "right_end_progress_meters": RIGHT_END_METERS,
                "corridor_half_width_meters": CORRIDOR_HALF_WIDTH_METERS,
                "maximum_heading_error_degrees": CORRIDOR_HEADING_LIMIT_DEGREES,
                "selection_window_meters": SELECTION_WINDOW_METERS,
                "boundary_policy": "fail_closed_keep_last_valid_pose_and_stop",
            },
            limitations=(
                "Evidence-only static route-following pilot; not a certified simulator.",
                "No ground truth exists for counterfactual lateral poses.",
                "Vehicles are baked into static geometry and cannot respond.",
                "Branch selection may switch traversal-specific appearance/sensor profiles.",
                "Browser timing excludes monitor scan-out.",
            ),
            output_path=args.evidence_output,
        )
        initial = adapter.reset()
        jpeg, render = render_state(initial)
        commit(initial, jpeg, render)
        evidence.record_reset(
            {
                "simulation_time_seconds": initial.time,
                "route_support": adapter.support(initial).as_dict(),
                "frame_sha256": hashlib.sha256(jpeg).hexdigest(),
                "server_reset_to_jpeg_ms": None,
                "reason": "server_start",
            }
        )
        print(f"TbV route adapter: http://{args.host}:{args.port}")
        print(f"evidence: {args.evidence_output.expanduser().resolve()}")
        print("controls: W/S/A/D, R reset, select 1=straight or 2=right at anchor")
        print("scope: static evidence-only route following; +/-1m fail-closed support")
        print("warning: trusted localhost/tunnel only; one shared driving state")
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping TbV route driving adapter")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
