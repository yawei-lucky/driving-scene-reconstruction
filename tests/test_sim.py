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
    H3_PROVISIONAL_WORLD_DRIVING_LIMITS,
    H3_WORLD_POSE_PROBE_LIMITS,
    LoggedCenterlineCorridor,
    LoggedCenterlineSample,
    LoggedEgoOffsetController,
    LoggedMovementProfile,
    NearbyPoseLimits,
    NerfstudioRenderer,
    RenderedObservation,
    SceneReferenceFrame,
    SimpleVehicleModel,
    SplatADLoggedRenderer,
    SplatADWorldRenderer,
    WorldDrivingController,
    logged_movement_profile,
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

    def test_optional_speed_limit_caps_forward_speed(self) -> None:
        model = SimpleVehicleModel(max_speed=2.0)

        next_state = model.step(
            EgoState(speed=1.9),
            HumanControl(throttle=1.0),
            dt=1.0,
        )

        self.assertAlmostEqual(next_state.speed, 2.0)

    def test_invalid_optional_speed_limit_is_rejected(self) -> None:
        for max_speed in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(max_speed=max_speed):
                with self.assertRaises(ValueError):
                    SimpleVehicleModel(max_speed=max_speed)

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


class WorldDrivingControllerTest(unittest.TestCase):
    @staticmethod
    def straight_corridor() -> LoggedCenterlineCorridor:
        return LoggedCenterlineCorridor(
            (
                LoggedCenterlineSample(0, 0.0, -2.0, 0.0, 0.0),
                LoggedCenterlineSample(1, 0.1, 0.0, 0.0, 0.0),
                LoggedCenterlineSample(2, 0.2, 10.0, 0.0, 0.0),
            ),
            half_width=1.0,
            max_heading_error=math.radians(30.0),
        )

    def test_throttle_and_steering_change_absolute_future_pose(self) -> None:
        controller = WorldDrivingController()
        state = controller.reset()

        for _ in range(10):
            update = controller.step(
                state,
                HumanControl(throttle=1.0, steer=0.4),
                dt=0.1,
            )
            self.assertFalse(update.boundary_hit)
            state = update.state

        self.assertGreater(state.x, 0.0)
        self.assertGreater(state.y, 0.0)
        self.assertGreater(state.yaw, 0.0)
        self.assertGreater(state.speed, 0.0)

    def test_releasing_throttle_coasts_and_brake_stops(self) -> None:
        controller = WorldDrivingController()
        accelerated = controller.step(
            controller.reset(),
            HumanControl(throttle=1.0),
            dt=0.5,
        ).state
        coasting = controller.step(accelerated, HumanControl(), dt=0.1).state
        state = coasting
        for _ in range(10):
            state = controller.step(
                state,
                HumanControl(brake=1.0),
                dt=0.1,
            ).state

        self.assertGreater(coasting.x, accelerated.x)
        self.assertLess(coasting.speed, accelerated.speed)
        self.assertAlmostEqual(state.speed, 0.0)

    def test_boundary_rejects_pose_instead_of_clamping_it(self) -> None:
        controller = WorldDrivingController()
        state = EgoState(
            x=H3_PROVISIONAL_WORLD_DRIVING_LIMITS.max_abs_forward_meters - 0.01,
            speed=2.0,
        )

        update = controller.step(state, HumanControl(throttle=1.0), dt=0.1)

        self.assertTrue(update.boundary_hit)
        self.assertIsNotNone(update.boundary_reason)
        self.assertAlmostEqual(update.state.x, state.x)
        self.assertAlmostEqual(update.state.y, state.y)
        self.assertAlmostEqual(update.state.yaw, state.yaw)
        self.assertAlmostEqual(update.state.speed, 0.0)
        self.assertAlmostEqual(update.state.time, state.time + 0.1)

    def test_reset_is_exact(self) -> None:
        controller = WorldDrivingController()

        self.assertEqual(controller.reset(), EgoState())
        self.assertEqual(controller.reset(), controller.reset())

    def test_logged_corridor_replaces_fixed_six_meter_rectangle(self) -> None:
        controller = WorldDrivingController(corridor=self.straight_corridor())

        update = controller.step(
            EgoState(x=5.95, speed=2.0),
            HumanControl(),
            dt=0.1,
        )

        self.assertFalse(update.boundary_hit)
        self.assertGreater(update.state.x, 6.0)
        measurement = controller.corridor_measurement(update.state)
        self.assertIsNotNone(measurement)
        assert measurement is not None
        self.assertAlmostEqual(measurement.distance, 0.0)

    def test_logged_corridor_can_spawn_at_recorded_start(self) -> None:
        corridor = self.straight_corridor()
        spawn = corridor.pose_at_progress(0.0)
        controller = WorldDrivingController(
            corridor=corridor,
            spawn_state=spawn,
        )

        self.assertEqual(controller.reset(), spawn)
        update = controller.step(
            controller.reset(),
            HumanControl(throttle=1.0),
            dt=0.1,
        )
        self.assertFalse(update.boundary_hit)
        self.assertGreater(update.state.x, spawn.x)

    def test_logged_corridor_rejects_leaving_observed_tube(self) -> None:
        corridor = self.straight_corridor()

        corridor.validate(EgoState(x=3.0, y=1.0))
        with self.assertRaises(ValueError):
            corridor.validate(EgoState(x=3.0, y=1.01))
        with self.assertRaises(ValueError):
            corridor.validate(EgoState(x=3.0, yaw=math.radians(31.0)))

    def test_logged_corridor_reports_progress_and_signed_offset(self) -> None:
        corridor = self.straight_corridor()

        measurement = corridor.measure(EgoState(x=4.0, y=0.5))

        self.assertAlmostEqual(corridor.length, 12.0)
        self.assertAlmostEqual(measurement.progress, 6.0)
        self.assertAlmostEqual(measurement.lateral_offset, 0.5)
        self.assertAlmostEqual(measurement.distance, 0.5)

    def test_logged_corridor_interpolates_pose_by_progress(self) -> None:
        corridor = self.straight_corridor()

        state = corridor.pose_at_progress(7.0, simulation_time=2.0, speed=1.5)

        self.assertAlmostEqual(state.x, 5.0)
        self.assertAlmostEqual(state.y, 0.0)
        self.assertAlmostEqual(state.yaw, 0.0)
        self.assertAlmostEqual(state.time, 2.0)
        self.assertAlmostEqual(state.speed, 1.5)
        with self.assertRaises(ValueError):
            corridor.pose_at_progress(corridor.length + 0.01)


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


class SplatADLoggedRendererConfigurationTest(unittest.TestCase):
    def test_lazy_construction_does_not_import_heavy_dependencies(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yml") as config:
            renderer = SplatADLoggedRenderer(config.name)

        self.assertFalse(renderer.is_loaded)

    def test_invalid_output_scale_is_rejected_before_loading(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yml") as config:
            with self.assertRaises(ValueError):
                SplatADLoggedRenderer(config.name, output_scale=0.0)

    def test_nearest_frame_selection_is_deterministic(self) -> None:
        times = (0.0, 0.1, 0.2)

        self.assertEqual(SplatADLoggedRenderer._nearest_index(times, 0.04), 0)
        self.assertEqual(SplatADLoggedRenderer._nearest_index(times, 0.06), 1)
        self.assertEqual(SplatADLoggedRenderer._nearest_index(times, 0.15), 1)


class SplatADWorldRendererConfigurationTest(unittest.TestCase):
    def test_lazy_construction_does_not_import_heavy_dependencies(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yml") as config:
            renderer = SplatADWorldRenderer(config.name)

        self.assertFalse(renderer.is_loaded)
        self.assertFalse(renderer.is_world_ready)
        self.assertAlmostEqual(renderer.anchor_log_time, 4.0)

    def test_invalid_anchor_time_is_rejected_before_loading(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yml") as config:
            with self.assertRaises(ValueError):
                SplatADWorldRenderer(config.name, anchor_log_time=-0.1)

    def test_probe_limits_include_requested_lateral_offsets(self) -> None:
        H3_WORLD_POSE_PROBE_LIMITS.validate(
            EgoState(y=-3.0, yaw=math.pi, time=20.0)
        )
        H3_WORLD_POSE_PROBE_LIMITS.validate(
            EgoState(y=3.0, yaw=-math.pi, time=20.0)
        )
        with self.assertRaises(ValueError):
            H3_WORLD_POSE_PROBE_LIMITS.validate(EgoState(y=3.01))


class LoggedEgoOffsetControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = LoggedEgoOffsetController(logged_duration=1.0)

    def test_log_time_advances_without_control(self) -> None:
        state = self.controller.step(EgoState(), HumanControl(), 0.1)

        self.assertAlmostEqual(state.time, 0.1)
        self.assertAlmostEqual(state.x, 0.0)

    def test_steering_is_visible_at_zero_relative_speed(self) -> None:
        state = self.controller.step(
            EgoState(),
            HumanControl(steer=1.0),
            0.1,
        )

        self.assertGreater(state.yaw, 0.0)

    def test_throttle_adds_bounded_forward_offset(self) -> None:
        state = EgoState()
        for _ in range(20):
            state = self.controller.step(
                state,
                HumanControl(throttle=1.0),
                0.05,
            )

        self.assertLessEqual(state.x, self.controller.limits.max_abs_forward_meters)

    def test_brake_from_rest_does_not_reverse(self) -> None:
        state = self.controller.step(
            EgoState(),
            HumanControl(brake=1.0),
            0.1,
        )

        self.assertAlmostEqual(state.speed, 0.0)
        self.assertAlmostEqual(state.x, 0.0)

    def test_speed_persists_after_throttle_is_released(self) -> None:
        accelerated = self.controller.step(
            EgoState(),
            HumanControl(throttle=1.0),
            0.1,
        )
        coasting = self.controller.step(
            accelerated,
            HumanControl(),
            0.1,
        )

        self.assertAlmostEqual(coasting.speed, accelerated.speed)
        self.assertGreater(coasting.x, accelerated.x)

    def test_non_finite_control_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.controller.step(
                EgoState(),
                HumanControl(throttle=math.nan),
                0.1,
            )

    def test_control_is_clamped_at_device_boundary(self) -> None:
        normalized = self.controller.step(
            EgoState(),
            HumanControl(throttle=1.0),
            0.1,
        )
        oversized = self.controller.step(
            EgoState(),
            HumanControl(throttle=10.0),
            0.1,
        )

        self.assertEqual(normalized, oversized)

    def test_end_of_log_stops_without_looping(self) -> None:
        state = self.controller.step(
            EgoState(time=0.95, speed=0.2),
            HumanControl(throttle=1.0),
            0.1,
        )
        stopped = self.controller.step(state, HumanControl(throttle=1.0), 0.1)

        self.assertAlmostEqual(state.time, 1.0)
        self.assertAlmostEqual(stopped.time, 1.0)
        self.assertAlmostEqual(stopped.speed, 0.0)

    def test_reset_is_complete_and_repeatable(self) -> None:
        self.assertEqual(self.controller.reset(), EgoState())
        self.assertEqual(self.controller.reset(), self.controller.reset())

    def test_visible_profile_moves_more_than_safe_profile(self) -> None:
        safe = logged_movement_profile("safe")
        visible = logged_movement_profile("visible")
        safe_controller = LoggedEgoOffsetController.from_profile(1.0, safe)
        visible_controller = LoggedEgoOffsetController.from_profile(1.0, visible)
        safe_state = EgoState()
        visible_state = EgoState()
        control = HumanControl(throttle=1.0, steer=1.0)

        for _ in range(10):
            safe_state = safe_controller.step(safe_state, control, 0.1)
            visible_state = visible_controller.step(visible_state, control, 0.1)

        self.assertGreater(visible_state.x, safe_state.x)
        self.assertGreater(abs(visible_state.yaw), abs(safe_state.yaw))
        self.assertGreater(
            visible.limits.max_abs_forward_meters,
            safe.limits.max_abs_forward_meters,
        )
        self.assertGreater(
            visible.limits.max_abs_yaw_radians,
            safe.limits.max_abs_yaw_radians,
        )

    def test_unknown_logged_movement_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            logged_movement_profile("not-a-profile")

    def test_profile_probe_must_fit_limits(self) -> None:
        with self.assertRaises(ValueError):
            LoggedMovementProfile(
                name="bad",
                limits=NearbyPoseLimits(max_abs_forward_meters=0.5),
                max_relative_speed=1.0,
                relative_acceleration=1.0,
                relative_braking=1.0,
                max_yaw_rate_radians=1.0,
                probe_forward_meters=0.6,
                probe_left_meters=0.0,
                probe_yaw_radians=0.0,
            )


if __name__ == "__main__":
    unittest.main()
