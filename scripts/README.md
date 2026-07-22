# Scripts

The scripts now cover five workflows:

- public-resource acquisition and initial Wayve/PandaSet preparation;
- Stage H1 WayveScenes101 split, Splatfacto training, rendering, and evaluation;
- reference-video construction and integrity validation;
- Stage H2 nearby-pose checkpoint rendering and interactive display;
- Stage H3 isolated SplatAD environment preparation, PandaSet inspection,
  100-step smoke training, and checkpoint rendering.

## Stage H3

Before extracting another sequence, scan the existing verified PandaSet ZIP
for repeat, offset, and multi-direction trajectories:

```bash
python3 scripts/analyze_stage_h3_pandaset_trajectories.py \
  --output-json /home/yawei/stage3_external/artifacts/pandaset_multi_trajectory_inventory/trajectory_inventory.json
```

The scan reads metadata directly from the ZIP and does not extract sensor
payloads. The optional front-camera contact sheet uses Pillow from the H3
environment:

```bash
/home/yawei/stage3_external/envs/h3_splatad/bin/python \
  scripts/analyze_stage_h3_pandaset_trajectories.py \
  --contact-sheet /home/yawei/stage3_external/artifacts/pandaset_multi_trajectory_inventory/front_contact_sheet.jpg
```

Prepare or verify the separate H3 environment:

```bash
scripts/setup_stage_h3_environment.sh
scripts/check_stage_h3_environment.sh
```

The setup command pins the audited neurad-studio, SplatAD gsplat, viser,
PandaSet devkit, and tiny-cuda-nn revisions under
`/home/yawei/stage3_external`. It does not download PandaSet. Use `--repair`
only when the existing environment needs its tested packages reapplied.

After the verified scene-040 extraction exists outside Git:

```bash
scripts/run_stage_h3_pandaset_040.sh data-gate
scripts/run_stage_h3_pandaset_040.sh smoke
scripts/run_stage_h3_pandaset_040.sh render-smoke
scripts/run_stage_h3_pandaset_040.sh pilot
scripts/run_stage_h3_pandaset_040.sh render-pilot
scripts/run_stage_h3_pandaset_040.sh paths
```

`data-gate` writes a JSON sensor/timing report, a six-camera contact sheet, and
Pandar64 camera overlays outside Git. `smoke` is fixed to 100 iterations and
reuses the accepted checkpoint unless `H3_ALLOW_RETRAIN=1` is set deliberately.
`render-smoke` likewise reuses existing held-out renders unless
`H3_ALLOW_RERENDER=1` is set. These commands reproduce the Level 1 integration
gate; they do not claim stable scene quality. `pilot` is the accepted
2,000-step, 0.9-train-split, 750,000-seed Level 2 run. `render-pilot` reloads
its checkpoint and renders the fixed 48-view holdout. Both reuse completed
artifacts by default.

Compact progress images can be rebuilt with the H3 Python environment:

```bash
/home/yawei/stage3_external/envs/h3_splatad/bin/python \
  scripts/build_stage_h3_render_summary.py \
  --smoke-root /home/yawei/stage3_external/artifacts/scene_040_smoke_100_render \
  --pilot-root /home/yawei/stage3_external/artifacts/scene_040_pilot_2000_render \
  --output-dir /home/yawei/stage3_external/artifacts/scene_040_pilot_2000_render
```

## Stage H2

The wrapper selects the existing `wayve_scenes_env`, CUDA 12.1, and RTX 4090
architecture settings:

```bash
scripts/run_stage_h2_scene_094.sh smoke
scripts/run_stage_h2_scene_094.sh interactive
```

All arguments after the mode are forwarded to the corresponding example.

Examples:

```bash
scripts/run_stage_h2_scene_094.sh smoke \
  --forward 0.5 --left 0.2 --yaw-degrees 2 \
  --cameras front-forward left-forward right-forward

scripts/run_stage_h2_scene_094.sh interactive \
  --output-scale 0.125

scripts/run_stage_h2_scene_094.sh interactive \
  --web \
  --output-scale 0.25

scripts/run_stage_h2_scene_094.sh interactive \
  --web \
  --output-scale 0.5 \
  --cameras front-forward

scripts/run_stage_h2_scene_094.sh interactive \
  --headless-steps 2 \
  --output-dir /tmp/dsr_stage_h2_interactive
```

Generated images remain outside Git by default.
