"""Standard-library tests for the simulator MVP."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from driving_scene_reconstruction.sim import (  # noqa: E402
    CameraRig,
    CameraSpec,
    EgoState,
    HumanControl,
    NearbyPoseLimits,
    NerfstudioRenderer,
    RenderedObservation,
    SceneReferenceFrame,
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

    def test_steering_changes_heading_and_lateral_position(self) -> None:
        next_state = SimpleVehicleModel().step(
            EgoState(speed=5.0),
            HumanControl(steer=0.5),
            dt=0.5,
        )

        self.assertGreater(next_state.yaw, 0.0)
        self.assertGreater(next_state.y, 0.0)

    def test_step_is_deterministic(self) -> None:
        model = SimpleVehicleModel()
        state = EgoState(x=1.0, y=2.0, yaw=0.3, speed=4.0, time=5.0)
        control = HumanControl(steer=-0.2, throttle=0.4)

        first = model.step(state, control, dt=0.1)
        second = model.step(state, control, dt=0.1)

        self.assertEqual(first, second)

    def test_non_finite_control_is_rejected(self) -> None:
        controls = (
            HumanControl(steer=math.nan),
            HumanControl(throttle=math.inf),
            HumanControl(brake=-math.inf),
        )

        for control in controls:
            with self.subTest(control=control):
                with self.assertRaises(ValueError):
                    SimpleVehicleModel().step(EgoState(), control, dt=0.1)

    def test_non_finite_timestep_is_rejected(self) -> None:
        for dt in (math.nan, math.inf, -math.inf):
            with self.subTest(dt=dt):
                with self.assertRaises(ValueError):
                    SimpleVehicleModel().step(EgoState(), HumanControl(), dt=dt)

    def test_non_finite_model_parameter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SimpleVehicleModel(wheelbase=math.nan)

    def test_non_finite_state_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SimpleVehicleModel().step(
                EgoState(x=math.nan),
                HumanControl(),
                dt=0.1,
            )


class RendererInterfaceTest(unittest.TestCase):
    def test_placeholder_renderer_returns_camera_names(self) -> None:
        rig = CameraRig(cameras=(CameraSpec("front"), CameraSpec("rear")))

        observation = PlaceholderRenderer().render("test_scene", EgoState(), rig)

        self.assertEqual(tuple(observation.frames), ("front", "rear"))
        self.assertEqual(observation.metadata["scene"], "test_scene")


class SceneCoordinateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.front_camera = (
            (0.0, 0.0, -1.0, 1.0),
            (-1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 3.0),
        )
        self.reference = SceneReferenceFrame.from_front_camera(
            self.front_camera,
            rig_origin=(1.0, 2.0, 3.0),
            scene_units_per_meter=0.5,
        )

    def test_front_camera_defines_forward_and_left_axes(self) -> None:
        self.assertEqual(self.reference.forward, (1.0, 0.0, 0.0))
        self.assertEqual(self.reference.left, (-0.0, 1.0, 0.0))

    def test_forward_and_left_displacement_use_scene_scale(self) -> None:
        transformed = self.reference.transform_camera(
            self.front_camera,
            EgoState(x=2.0, y=1.0),
        )

        self.assertAlmostEqual(transformed[0][3], 2.0)
        self.assertAlmostEqual(transformed[1][3], 2.5)
        self.assertAlmostEqual(transformed[2][3], 3.0)

    def test_yaw_rotates_camera_about_rig_origin(self) -> None:
        offset_camera = (
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 1.0, 0.0, 2.0),
            (0.0, 0.0, 1.0, 3.0),
        )

        transformed = self.reference.transform_camera(
            offset_camera,
            EgoState(yaw=math.pi / 2.0),
        )

        self.assertAlmostEqual(transformed[0][3], 1.0)
        self.assertAlmostEqual(transformed[1][3], 3.0)
        self.assertAlmostEqual(transformed[0][0], 0.0, places=7)
        self.assertAlmostEqual(transformed[1][0], 1.0)

    def test_nearby_pose_limits_reject_large_queries(self) -> None:
        limits = NearbyPoseLimits(
            max_abs_forward_meters=2.0,
            max_abs_left_meters=0.5,
            max_abs_yaw_radians=math.radians(5.0),
        )

        limits.validate(EgoState(x=2.0, y=-0.5, yaw=math.radians(5.0)))
        for state in (
            EgoState(x=2.01),
            EgoState(y=-0.51),
            EgoState(yaw=math.radians(5.1)),
        ):
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    limits.validate(state)


class NerfstudioRendererConfigurationTest(unittest.TestCase):
    def test_lazy_construction_does_not_import_heavy_dependencies(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yml") as config:
            renderer = NerfstudioRenderer(config.name)

        self.assertFalse(renderer.is_loaded)

    def test_invalid_output_scale_is_rejected_before_loading(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yml") as config:
            with self.assertRaises(ValueError):
                NerfstudioRenderer(config.name, output_scale=0.0)


if __name__ == "__main__":
    unittest.main()
