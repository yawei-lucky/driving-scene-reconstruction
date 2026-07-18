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

Stage H2 connects that H1 checkpoint to the simulator interface. Its design and
GPU validation record live in `docs/stage_h2_reconstruction_renderer.md`.

The next experiment should evaluate a fixed grid of nearby ego poses and record
quality, geometry failures, and warmed rendering latency.
