# Stage H3 HUGSIM Official PandaSet-040 Probe

Date: 2026-07-28

## Question

ProPainter removed traffic silhouettes but produced soft, unverifiable road
completion and did not improve foliage. Can HUGSIM provide a more useful
quality direction without first building a TbV adapter or repeating a 30,000
step reconstruction?

This is an inference-only capability gate. It does not promote a new renderer
or claim that HUGSIM repairs an existing SplatAD checkpoint.

## Exact-Scene Shortcut

The official HUGSIM release includes an exported checkpoint for **PandaSet
scene 040**, which is the same raw sequence used by the current SplatAD
static-8k regression:

- official code: https://github.com/hyzhou404/HUGSIM
- official released PandaSet scenes:
  https://huggingface.co/datasets/XDimLab/HUGSIM/tree/main/scenes/pandaset
- selected asset:
  https://huggingface.co/datasets/XDimLab/HUGSIM/blob/main/scenes/pandaset/040.zip

The release therefore bypasses semantic prediction, monocular-depth
preparation, ground training, scene training, and export for this first gate.
No HUGSIM source file was changed.

Local provenance:

| Item | Value |
| --- | --- |
| HUGSIM commit | `adeca402cad4af8635e13d0a105e2fee6a14de85` |
| ZIP path | `/home/yawei/HUGSIM_assets/scenes/pandaset/040.zip` |
| ZIP bytes | 584,191,938 |
| ZIP SHA-256 / release LFS oid | `1cb27bf799d775aaa4c051c65681cc27dcdba4099af04fdbd1b01ed22b79e0ad` |
| Exported `scene.pth` bytes | 751,925,176 |
| Exported `scene.pth` SHA-256 | `67ceb9a124f06a7588f464f0fc1aa2fa8767e971d9005f4c8a936c67c99745f5` |
| Export iteration | 30,000 |
| Native dynamic checkpoints | 11 |

The existing unrelated untracked
`/home/yawei/HUGSIM/pixi.toml.smoke-backup` was left untouched.

## Environment And Command

This probe uses HUGSIM's existing PyTorch `2.4.1+cu121` / CUDA 12.1 pixi
runtime. It must not be launched with the Stage H3 SplatAD CUDA 11.8
environment. The actual host, rather than a sandbox-only CUDA probe, reported
an NVIDIA GeForce RTX 4090 D with about 22.6 GiB free before the run.

The new inference probe is:

```bash
cd /home/yawei/HUGSIM
CUDA_VISIBLE_DEVICES=0 \
  /home/yawei/HUGSIM/.pixi/envs/default/bin/python \
  /home/yawei/driving-scene-reconstruction/scripts/probe_stage_h3_hugsim_pandaset.py \
  --video \
  --output-dir \
    /home/yawei/stage3_external/artifacts/scene_040_hugsim_official_probe_20260728_v2
```

It renders:

- all 80 logged front-camera poses with native dynamics;
- the same 80 poses with HUGSIM's native dynamic layer omitted;
- frames 19, 39, and 59 at camera-left offsets `-3/-1/+1/+3 m`;
- an eight-second visual comparison against the existing SplatAD static-8k
  temporal output.

Positive lateral offset means camera-left. Dynamic objects remain in their
declared world poses while the ego camera moves.

## Artifacts

```text
/home/yawei/stage3_external/artifacts/scene_040_hugsim_official_probe_20260728_v2/hugsim_scene040_vs_splatad_10fps.mp4
/home/yawei/stage3_external/artifacts/scene_040_hugsim_official_probe_20260728_v2/hugsim_scene040_probe_contact_sheet.jpg
/home/yawei/stage3_external/artifacts/scene_040_hugsim_official_probe_20260728_v2/hugsim_scene040_probe.json
```

The video is H.264, 1920x314, 10 fps, 80 frames, and eight seconds. Its columns
are source RGB, SplatAD static-8k, HUGSIM factual, and the same HUGSIM render
without native dynamics. The SplatAD column is cropped from the already
generated labeled comparison video; numeric SplatAD values below come from its
original evaluator JSON, not from that presentation JPEG.

## Measurements

All HUGSIM values below use 80 front-camera frames at 960x540:

| Screen | Result |
| --- | ---: |
| HUGSIM factual PSNR, mean / p50 | 27.7007 / 27.7179 dB |
| HUGSIM factual PSNR, min / max | 24.5404 / 30.4944 dB |
| HUGSIM static-only PSNR, mean / p50 | 27.6639 / 27.6945 dB |
| HUGSIM factual render latency, p50 / p95 | 16.69 / 18.63 ms |
| HUGSIM factual first warm-up maximum | 383.82 ms |
| HUGSIM static-only render latency, p50 / p95 | 11.29 / 13.80 ms |
| Exported scene plus dynamic-model load | 0.692 s |
| Frames with a `>8/255` factual/static pixel difference | 39 / 80 |
| Dynamic changed-pixel fraction, mean / max | 0.0841% / 0.2822% |
| Selected lateral renders | 12 / 12 finite |
| Maximum selected lateral near-black fraction | 0.00251% |

The existing SplatAD evaluator reports 27.1015 dB mean front-camera PSNR over
the same 80 logical frames. This is not an architecture ranking: HUGSIM and
SplatAD use different preprocessing, render resolutions, training policies,
and camera implementations. The values only show that the downloaded HUGSIM
asset is a real reconstruction in the same broad logged-pose quality range,
not an unrelated low-quality demo.

## Visual Finding

Direct inspection of the video, selected PNGs, and contact sheet finds:

- HUGSIM reconstructs coherent road, lane, building, traffic-light, and parked
  vehicle structure at logged front poses.
- Omitting HUGSIM's native dynamics leaves a reusable static 3D background;
  it does not run an image-space deletion or invent a new inpainted road.
- In this scene's front camera, the native dynamic contribution is small.
  Thirty-nine frames change, but no frame changes more than 0.283% of pixels.
  This run therefore does not prove high-quality reconstruction of a large
  nearby moving vehicle.
- Parked cars are correctly part of the static scene, but close cars still
  soften and stretch. Foliage, thin poles, and some building edges remain
  blurry. HUGSIM does not solve general texture quality.
- All tested `+/-3 m` views render without black collapse, so the representation
  has materially more pose reach than an image-space completion. However,
  close vegetation, façades, and cars deform more at the extremes. With no
  real RGB at those counterfactual poses, this is coverage evidence, not a
  `+/-3 m` driving-truth pass.

## Decision

HUGSIM passes the **same-scene inference and structured-background gate**. It
is a better architectural direction than ProPainter for a simulator because
the background, dynamics, semantics, and depth remain tied to one renderable
3D scene.

It does **not** pass as a quality promotion or a direct TbV repair:

1. logged-pose quality is comparable to the current SplatAD result, not a
   decisive visual jump;
2. vegetation and close static vehicles remain soft or deformed;
3. the visible front-camera dynamic layer is too small to validate difficult
   moving-vehicle reconstruction;
4. the official asset is PandaSet 040 and cannot be applied to the TbV Miami
   tiles.

Keep SplatAD static-8k as the default. Do not spend the next cycle reproducing
the full HUGSIM 30k training pipeline or building a TbV adapter solely to seek
sharper pixels.

If the HUGSIM line continues, the smallest useful next demo is to connect the
official 040 asset to the existing simulated vehicle controller, use its
static-only background over the 64.6 m route, and insert one explicit
3DRealCar actor. That directly tests the promising part—ghost-free
compositional traffic—before any new reconstruction training.
