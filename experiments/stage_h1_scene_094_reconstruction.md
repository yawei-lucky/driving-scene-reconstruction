# Stage H1 WayveScenes101 Scene 094 Reconstruction

Date: 2026-07-14

## Scope

- Dataset: WayveScenes101 `scene_094` (dataset license: non-commercial research use).
- Tool: Nerfstudio 1.1.5, `splatfacto-big` preset.
- Official split: four side/rear cameras for training (800 images), `front-forward` held out (200 images).
- Raw data, checkpoints, renders, and videos remain outside Git under `/home/yawei/stage1_external`.

## Environment

- Host: `stf-precision-3680` (`shidi`)
- GPU: NVIDIA RTX 4090 D, 24 GB; peak observed training allocation about 7.3 GB
- CUDA toolkit: 12.1; `TORCH_CUDA_ARCH_LIST=8.9`
- Disk: outputs placed on `/home`; `/data` was intentionally rejected because it had only about 11 GB free
- Python: 3.10.14 in `wayve_scenes_env`

## Validation And Training

The split builder validated all five cameras, 200 frames per camera, image/mask containment, unique paths, and produced an atomic split view. Nerfstudio parsed it as 800 train and 200 test images.

Runs:

1. `run_v1`, 3,000 steps: completed; checkpoint `step-000002999.ckpt` (2,007,667,602 bytes).
2. A resume attempt from `run_v1` reached the step-3000 evaluation callback and failed in upstream Splatfacto 1.1.5 with an opacity tensor shape assertion. The original checkpoint remained intact.
3. `run_v2`, fresh 8,000 steps: completed; checkpoint `step-000007999.ckpt` (2,408,704,338 bytes).

Final config:

```text
/home/yawei/stage1_external/outputs/wayvescenes101_h1/scene_094_h1_big/splatfacto/run_v2/config.yml
```

## Results

Nerfstudio held-out evaluation over all 200 `front-forward` images:

| Step | PSNR | SSIM | LPIPS |
| ---: | ---: | ---: | ---: |
| 3,000 | 14.1631 | 0.3938 | 0.8096 |
| 8,000 | 15.5961 | 0.5343 | 0.5630 |

Official Wayve evaluator on the held-out `front-forward` camera:

| PSNR | SSIM | LPIPS | FID |
| ---: | ---: | ---: | ---: |
| 13.2109 | 0.3022 | 0.7058 | 10.5488 |

The upstream FID helper stacks every full-resolution image in memory and was killed after the per-image metrics completed. The wrapper therefore uses the same TorchMetrics `FrechetInceptionDistance(feature=64)` updates in batches of eight. This is mathematically equivalent to the upstream metric while avoiding a roughly 50+ GB pair of input stacks and subsequent copies.

Visual inspection of the midpoint held-out comparison confirms that static scene geometry and camera pose are recognizable, while moving vehicles and fine texture remain blurred. The result is a usable first baseline, not a production-quality reconstruction.

## Artifacts

Artifact root:

```text
/home/yawei/stage1_external/artifacts/scene_094/scene_094_h1_big_run_v2
```

Important files:

- `metrics/metrics.json`: Nerfstudio held-out metrics.
- `metrics/wayve_metrics.json`: official split metrics with memory-safe FID.
- `dataset/`: 1,000 predictions plus 1,000 ground-truth renders at original resolution.
- `videos/artifact_manifest.json`: hashes, dimensions, frame counts, and decode probes.
- `videos/scene_094_front-forward_gt_vs_reconstruction.mp4`: ground truth left, prediction right.
- Five `scene_094_<camera>_reconstruction.mp4` videos, one per camera.
- `samples/front_forward_gt_vs_reconstruction_mid.jpg`: inspected midpoint frame.

All six videos contain exactly 200 frames and passed full ffmpeg decode plus ffprobe frame-count and width checks. File sizes range from 4.5 MB to 17 MB.

## Next Action

Keep `run_v2` as the Stage H1 baseline. Before longer training, investigate the Nerfstudio 1.1.5 Splatfacto resume assertion or upgrade in a separate environment, then compare 15k/30k checkpoints against these fixed metrics and videos. Dynamic-object modeling is the next substantive quality limitation.
