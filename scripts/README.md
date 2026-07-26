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

Audit the selected 610 m repeated TbV route, generate its seven-tile contract,
sample eight source images, and list the exact adjacent-pair payload without
downloading it:

```bash
env MPLCONFIGDIR=/tmp \
  /home/yawei/stage3_external/envs/h3_splatad/bin/python \
  scripts/audit_stage_h3_tbv_long_route_tiles.py \
  --download-source-images \
  --plan-tile-payload
```

The command refuses to overwrite a non-empty output directory. Its default
artifacts remain outside Git under `/home/yawei/stage3_external/artifacts`.

Run the selected adjacent-pair download, independent checkpoints, and matched
world-pose seam probes:

```bash
scripts/run_stage_h3_tbv_long_route_tiles.sh download
scripts/run_stage_h3_tbv_long_route_tiles.sh smoke-2
scripts/run_stage_h3_tbv_long_route_tiles.sh smoke-3
scripts/run_stage_h3_tbv_long_route_tiles.sh seam
scripts/run_stage_h3_tbv_long_route_tiles.sh pilot-2
scripts/run_stage_h3_tbv_long_route_tiles.sh pilot-3
scripts/run_stage_h3_tbv_long_route_tiles.sh seam-500
scripts/run_stage_h3_tbv_long_route_tiles.sh quality-2
scripts/run_stage_h3_tbv_long_route_tiles.sh quality-3
scripts/run_stage_h3_tbv_long_route_tiles.sh seam-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh continuous-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh static-2
scripts/run_stage_h3_tbv_long_route_tiles.sh static-3
scripts/run_stage_h3_tbv_long_route_tiles.sh seam-8000
scripts/run_stage_h3_tbv_long_route_tiles.sh continuous-8000
scripts/run_stage_h3_tbv_long_route_tiles.sh mask-data
scripts/run_stage_h3_tbv_long_route_tiles.sh masked-2
scripts/run_stage_h3_tbv_long_route_tiles.sh masked-3
scripts/run_stage_h3_tbv_long_route_tiles.sh masked-seam-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh masked-continuous-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-mask-audit-2
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-masked-2
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-masked-3
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-masked-seam-2000
scripts/run_stage_h3_tbv_long_route_tiles.sh joint-masked-continuous-2000
```

The 100-step pair tests integration; the bounded 500-step pair establishes
whether the same adjacent tiles warrant a 2,000-step continuous seam video.
The 2,000-step runs are independent training; `static-2/static-3` exactly
restore their optimizer, scheduler, model, and global-step state for the
bounded 8,000-step comparison. The continuous probes render the real 18.75 m
overlap at 12 m/s and 20 fps, compare hard switching with a 5.6 m blend, and
add `-1/0/+1 m` front views. Completed checkpoints are reused unless
`H3_ALLOW_RETRAIN=1` is set.

The masked modes form a bounded static-background discriminator. `mask-data`
uses torchvision Mask R-CNN to write mirrored valid-pixel PNGs; the two
masked training modes pass them through the TbV parser and align them with
SplatAD's existing image crop. The 2,000-step seam and continuous probes report
how many training masks were loaded. They intentionally do not resume to 8,000
steps: the image-only pilot reduced visible vehicle residue but worsened
cross-tile consistency.

The `joint-*` modes are the completed follow-up. They select the nearest image
from each ring camera within 50 ms, use AV2 motion-compensated projection, and
exclude a LiDAR return when any valid projection lands in a traffic-mask
pixel. `joint-mask-audit-2` verifies coverage/removal before training. The
joint 2,000-step pair removed 7.66%/9.41% of tile-2/3 returns but worsened
matched-pose cross-tile RGB MAE to 13.77 / 255, so it is rejected before 8,000
steps. The next experiment is cross-traversal city-frame 3D persistence, not
more direct 2D-silhouette deletion.

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
