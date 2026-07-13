"""Standard-library tests for the simulator MVP."""

from __future__ import annotations

import sys
import unittest
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


class PlaceholderRenderer:
    def render(
        self,
        scene: object,
        ego_state: EgoState,
        camera_rig: CameraRig,
    ) -> RenderedObservation:
        return RenderedObservation(
            timestamp=ego_state.time,
            frames={camera.name: None for camera in camera_rig.cameras},
            metadata={"scene": scene},
        )


class SimpleVehicleModelTest(unittest.TestCase):
    def test_step_updates_time(self) -> None:
        state = EgoState(time=2.0)

        next_state = SimpleVehicleModel().step(state, HumanControl(), dt=0.25)

        self.assertAlmostEqual(next_state.time, 2.25)

    def test_throttle_moves_vehicle_forward(self) -> None:
        state = EgoState()

        next_state = SimpleVehicleModel().step(
            state,
            HumanControl(throttle=1.0),
            dt=1.0,
        )

        self.assertGreater(next_state.x, state.x)
        self.assertGreater(next_state.speed, state.speed)
        self.assertAlmostEqual(next_state.y, 0.0)

    def test_control_is_clamped_at_model_boundary(self) -> None:
        model = SimpleVehicleModel()

        clamped = model.step(EgoState(), HumanControl(throttle=1.0), dt=0.5)
        oversized = model.step(EgoState(), HumanControl(throttle=3.0), dt=0.5)

        self.assertEqual(oversized, clamped)


class RendererInterfaceTest(unittest.TestCase):
    def test_placeholder_renderer_returns_camera_names(self) -> None:
        rig = CameraRig(cameras=(CameraSpec("front"), CameraSpec("rear")))

        observation = PlaceholderRenderer().render("test_scene", EgoState(), rig)

        self.assertEqual(tuple(observation.frames), ("front", "rear"))
        self.assertEqual(observation.metadata["scene"], "test_scene")


if __name__ == "__main__":
    unittest.main()
