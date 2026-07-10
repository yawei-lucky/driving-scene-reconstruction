# Codex Next Task — Stage H0

## Task Title

Stage H0: define human-drivable log-based panoramic simulator MVP

## Repository

```text
yawei-lucky/driving-scene-reconstruction
```

## Project Correction

The immediate goal is not only novel-view synthesis, not only a fixed driving-video generator, and not yet an autonomous-agent evaluation benchmark.

The first simulator target is **human-drivable**:

```text
real driving log
→ scene reconstruction / scene representation
→ human control input
→ ego state update
→ 360-degree / multi-view sensor rendering
→ display / record driving session
```

A human should eventually be able to control the ego vehicle and see the reconstructed panoramic / multi-view driving scene update in closed loop.

Do not add a world-model track in this task.

## Why This Task Comes Before More Training

A previous Codex run already verified that WayveScenes101 + Nerfstudio / Splatfacto can be used as a reconstruction-backend smoke test. It prepared `scene_094` and completed a 1-iteration Splatfacto run.

That is useful, but the project should not continue heavy reconstruction training before the simulator interface is defined.

This task should define the human-drivable simulator skeleton first.

## Rules

- Do not train NeRF, 3DGS, Splatfacto, Nerfacto, NeuRAD, or SplatAD.
- Do not download datasets.
- Do not add world-model integration.
- Do not connect an autonomous-driving model yet.
- Do not commit videos, checkpoints, rendered outputs, or large binaries.
- Keep the implementation dependency-light.
- Prefer dataclasses and simple Python interfaces.
- Commit only lightweight code, docs, examples, or tests.

## Read First

Read these files before editing:

```text
README.md
PROJECT_STATE.md
docs/human_drivable_simulator_project.md
experiments/stage1_codex_execution.md
scripts/prepare_stage1_wayvescenes101_nerfstudio.py
```

## Required Deliverables

### 1. Simulator package skeleton

Create:

```text
src/driving_scene_reconstruction/sim/__init__.py
src/driving_scene_reconstruction/sim/state.py
src/driving_scene_reconstruction/sim/control.py
src/driving_scene_reconstruction/sim/vehicle_model.py
src/driving_scene_reconstruction/sim/renderer_interface.py
```

### 2. Ego state model

In `state.py`, define an `EgoState` dataclass.

Suggested fields:

```text
x: float
y: float
yaw: float
speed: float
time: float
```

Use SI units:

```text
x, y: meters
yaw: radians
speed: meters / second
time: seconds
```

Add a short docstring explaining that this state is for the initial human-drivable simulator loop.

### 3. Human control model

In `control.py`, define a `HumanControl` dataclass.

Suggested fields:

```text
steer: float
throttle: float
brake: float
```

Use normalized controls for now:

```text
steer: -1.0 to 1.0
throttle: 0.0 to 1.0
brake: 0.0 to 1.0
```

Include a small clamp / validation helper if useful.

### 4. Simple vehicle model

In `vehicle_model.py`, implement a simple vehicle model.

Acceptable options:

```text
kinematic bicycle model
or simple yaw-rate approximation
```

Suggested API:

```python
class SimpleVehicleModel:
    def step(self, state: EgoState, control: HumanControl, dt: float) -> EgoState:
        ...
```

Requirements:

- deterministic;
- dependency-free;
- documented assumptions;
- good enough for smoke testing human control and ego pose update;
- not intended to be a high-fidelity vehicle dynamics model.

### 5. Renderer interface

In `renderer_interface.py`, define lightweight renderer abstractions.

Suggested dataclasses / protocols:

```text
CameraSpec
CameraRig
RenderedObservation
Renderer
```

Suggested renderer API:

```python
class Renderer(Protocol):
    def render(self, scene: object, ego_state: EgoState, camera_rig: CameraRig) -> RenderedObservation:
        ...
```

The first renderer can be a dummy renderer in the example, not in the interface file.

### 6. Smoke example

Create:

```text
examples/sim_loop_smoke.py
```

The smoke example should:

1. initialize an ego state;
2. define a simple camera rig;
3. create a dummy renderer that returns placeholder metadata;
4. apply a short sequence of fake human controls;
5. update ego state for several steps;
6. call the dummy renderer each step;
7. print state updates and rendered placeholder metadata.

It should run with only the Python standard library plus the package code.

Example command:

```bash
python examples/sim_loop_smoke.py
```

### 7. Optional lightweight test

If the repository already has a test structure or if it is easy to add one, add a small test for:

```text
vehicle model updates time;
vehicle model moves forward under throttle;
renderer interface example returns expected camera names.
```

Do not introduce heavy test dependencies.

## Expected Final Explanation

In your final response, explain:

- what files were added;
- how to run the smoke example;
- what this simulator skeleton can do;
- what it cannot do yet;
- how the existing WayveScenes101 / Splatfacto smoke run fits as a future renderer backend;
- why no world model or heavy training was added.

## Commit Message

Use a clear commit message, for example:

```text
Add human-drivable simulator MVP skeleton
```
