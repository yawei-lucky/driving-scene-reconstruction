#!/usr/bin/env python3
"""Run the dependency-free human-drivable simulator smoke loop."""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    CameraRig,
    CameraSpec,
    EgoState,
    HumanControl,
    RenderedObservation,
    SimpleVehicleModel,
)


class DummyRenderer:
    """Return pose metadata in place of rendered pixels."""

    def render(
        self,
        scene: object,
        ego_state: EgoState,
        camera_rig: CameraRig,
    ) -> RenderedObservation:
        frames = {
            camera.name: {
                "kind": "placeholder",
                "ego_x": round(ego_state.x, 3),
                "ego_y": round(ego_state.y, 3),
                "ego_yaw": round(ego_state.yaw, 3),
                "camera_yaw": round(camera.yaw, 3),
            }
            for camera in camera_rig.cameras
        }
        return RenderedObservation(
            timestamp=ego_state.time,
            frames=frames,
            metadata={"scene": scene, "backend": "dummy"},
        )


def main() -> None:
    state = EgoState()
    model = SimpleVehicleModel()
    renderer = DummyRenderer()
    camera_rig = CameraRig(
        cameras=(
            CameraSpec("front", yaw=0.0),
            CameraSpec("left", yaw=math.pi / 2.0),
            CameraSpec("right", yaw=-math.pi / 2.0),
            CameraSpec("rear", yaw=math.pi),
        )
    )
    controls = (
        HumanControl(throttle=0.7),
        HumanControl(steer=0.25, throttle=0.7),
        HumanControl(steer=0.25, throttle=0.4),
        HumanControl(steer=-0.2, throttle=0.3),
        HumanControl(brake=0.5),
    )

    print(f"initial state={state}")
    for step_index, control in enumerate(controls, start=1):
        state = model.step(state, control, dt=0.2)
        observation = renderer.render("scene_094_placeholder", state, camera_rig)
        camera_names = ",".join(observation.frames)
        print(
            f"step={step_index} time={state.time:.1f}s "
            f"pose=({state.x:.3f}, {state.y:.3f}, {state.yaw:.3f}) "
            f"speed={state.speed:.3f}m/s cameras={camera_names} "
            f"metadata={dict(observation.metadata)}"
        )


if __name__ == "__main__":
    main()
