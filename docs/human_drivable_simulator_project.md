# Human-Drivable Log-Based Panoramic Simulator

## 1. Project Goal

The project goal is to build a **human-drivable simulator from real driving logs**.

The first target is not an autonomous-driving agent benchmark. The first target is:

> A human can control the ego vehicle, and the reconstructed 360-degree / multi-view driving scene updates according to the ego state.

The core loop is:

```text
real driving log
→ scene reconstruction / scene representation
→ human control input
→ ego state update
→ 360-degree / multi-view rendering
→ human observes and continues driving
```

Autonomous-driving model integration can come later by replacing the human control input with an agent action.

## 2. Why This Is Different From Fixed Video Generation

A fixed driving video is open-loop:

```text
trajectory / prompt / log
→ generated video
→ playback only
```

A human-drivable simulator is closed-loop:

```text
human steering / throttle / brake
→ ego pose changes
→ rendered observations change
→ human reacts
→ next ego pose changes again
```

The simulator therefore needs an internal state, not only a video output.

Minimum internal state:

```text
ego_x
ego_y
ego_yaw
ego_speed
time
camera_rig_pose
scene_state
```

## 3. Why Scene Reconstruction Is Needed

If the system only replays the original log, the ego vehicle can only follow the original trajectory.

Human driving requires at least small deviations:

- slightly different steering;
- different speed;
- small lateral offset;
- earlier or later braking;
- slightly different viewpoint / head direction.

Once the ego state deviates from the original logged pose, the simulator must render observations that were not directly recorded. That requires a scene representation or reconstruction backend.

## 4. Initial MVP Scope

The first MVP should be intentionally limited:

```text
short real driving log
+ simple human-control interface
+ simple ego state update
+ renderer interface
+ dummy renderer smoke test
```

It does not need full photorealistic rendering yet.

The first MVP should answer:

> Can the repository represent the closed-loop simulator architecture cleanly before choosing a heavy renderer?

## 5. Initial Simulator Interface

The first implementation should define these concepts:

### EgoState

Represents current ego vehicle state:

```text
x
y
yaw
speed
time
```

### HumanControl

Represents control input:

```text
steer
throttle
brake
```

### VehicleModel

Updates ego state:

```text
next_state = vehicle_model.step(state, control, dt)
```

The first version can use a simple kinematic bicycle model or yaw-rate approximation.

### CameraRig / CameraSpec

Defines camera views to render:

```text
front
left
right
rear
panorama
```

### Renderer

Abstract renderer interface:

```text
frames = renderer.render(scene, ego_state, camera_rig)
```

The first renderer can be a dummy renderer that returns metadata only.

## 6. Renderer Backends

The simulator should support multiple renderer backends over time.

### ReplayRenderer

Replays original log frames. Useful for sanity checks, but it does not support true ego deviation.

### PanoramaRenderer

Uses panoramic or surround-view projection. Useful when the input already contains 360-degree imagery.

### ReconstructionRenderer

Uses a reconstructed scene representation such as 3DGS, NeRF, NeuRAD, SplatAD, mesh, or depth-based rendering.

This is where the existing WayveScenes101 + Splatfacto smoke run fits: it is a future reconstruction renderer backend test, not the project goal by itself.

### HybridRenderer

Combines geometry-based rendering with repair, stabilization, or inpainting for missing regions.

## 7. What Is Out of Scope Right Now

Do not prioritize these in the immediate next step:

- world model integration;
- autonomous-driving model integration;
- full Splatfacto / NeRF training;
- full 101-scene dataset download;
- photorealistic rendering quality;
- traffic-agent response modeling;
- large output video generation;
- committing datasets, checkpoints, or rendered videos.

## 8. How Existing Codex Work Fits

Codex already verified that the reconstruction-backend path is technically feasible:

```text
WayveScenes101 scene_094
→ Nerfstudio conversion
→ fisheye camera_model compatibility fix
→ Splatfacto 1-iteration smoke run
```

This should be treated as a renderer-backend smoke test.

Do not continue heavy Splatfacto training until the human-drivable simulator interface exists.

## 9. Immediate Next Step

The next task is Stage H0:

```text
Define the human-drivable log-based panoramic simulator MVP.
```

Expected result:

```text
src/driving_scene_reconstruction/sim/
  state.py
  control.py
  vehicle_model.py
  renderer_interface.py
examples/sim_loop_smoke.py
```

The smoke loop should:

```text
initialize ego state
apply fake human controls
update ego state for several steps
call dummy renderer
print updated state and placeholder frame metadata
```
