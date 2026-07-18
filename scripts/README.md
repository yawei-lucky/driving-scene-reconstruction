# Scripts

The scripts now cover five workflows:

- public-resource acquisition and initial Wayve/PandaSet preparation;
- Stage H1 WayveScenes101 split, Splatfacto training, rendering, and evaluation;
- reference-video construction and integrity validation;
- Stage H2 nearby-pose checkpoint rendering and interactive display.
- Stage H3 isolated SplatAD environment preparation and GPU acceptance.

## Stage H3

Prepare or verify the separate H3 environment:

```bash
scripts/setup_stage_h3_environment.sh
scripts/check_stage_h3_environment.sh
```

The setup command pins the audited neurad-studio, SplatAD gsplat, viser,
PandaSet devkit, and tiny-cuda-nn revisions under
`/home/yawei/stage3_external`. It does not download PandaSet. Use `--repair`
only when the existing environment needs its tested packages reapplied.

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
