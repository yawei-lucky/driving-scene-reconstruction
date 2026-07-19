# Experiments

This directory contains historical experiment records, not active task
instructions. Preserve recorded commands, environment details, decisions, and
results as evidence of what was done at the time. For current scope, status,
and next actions, use the repository root `README.md` and `PROJECT_STATE.md`.

Recorded experiments:

- `stage1_public_resources.md`: public dataset and codebase selection;
- `stage1_codex_execution.md`: environment preparation and early smoke runs;
- `stage_h1_scene_094_reconstruction.md`: 8,000-step WayveScenes101 baseline,
  held-out metrics, visual findings, and external artifact inventory.
- `stage_h3_environment.md`: isolated neurad-studio/SplatAD environment,
  pinned revisions, synthetic camera/LiDAR GPU acceptance, and direction risks.
- `stage_h3_dataset_foundation.md`: PandaSet source, license, archive/storage,
  per-sequence packaging, verified acquisition, scene selection, and
  calibration gate.
- `stage_h3_scene_040_smoke.md`: scene 040 data/timing evidence, SplatAD
  100-step checkpoint, held-out rendering, metrics, and visual decision.
- `stage_h3_scene_040_pilot.md`: 2,000-step six-camera/Pandar64 pilot, fixed
  temporal holdout, full checkpoint evidence, and visual quality decision.

Stage H2 connects that H1 checkpoint to the simulator interface. Its design and
GPU validation record live in `docs/stage_h2_reconstruction_renderer.md`.

The next H3 experiment is a fixed nearby-pose, depth, and temporal validation
of the accepted 2,000-step checkpoint. The 8k baseline remains gated on those
geometry, timing, and dynamic-object findings.
