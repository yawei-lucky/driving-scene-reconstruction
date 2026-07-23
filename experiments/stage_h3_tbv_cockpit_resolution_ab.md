# Stage H3 TbV Cockpit Resolution A/B

Date: 2026-07-23

## Question

Can the fixed TbV step-7,999 checkpoint provide a visibly clearer human-driving
view without exceeding the existing 100 ms p95 server control-to-JPEG budget?

## Change before the comparison

The cylindrical cockpit uses only `ring_front_left`, `ring_front_center`, and
`ring_front_right`. The normal driving loop now renders only those cameras.
The complete seven-camera reconstruction view remains at `/diagnostic`, but it
renders only when the page is opened or manually refreshed. It is not included
in normal driving latency.

No checkpoint, route, camera calibration, panorama geometry, or JPEG output
size changed. No new training was run.

## Fixed-pose comparison

Runtime:

- host: `stf-precision-3680`;
- GPU: NVIDIA GeForce RTX 4090 D, driver 580.95.05;
- checkpoint:
  `tbv_branch_pair_splatad_static_8000/.../step-000007999.ckpt`;
- pose: reset/common-approach spawn at route progress -20 m;
- samples: 12 warmed no-motion HTTP control samples per output scale;
- driving cameras: three front cameras;
- final cockpit JPEG: 1600x668 at quality 88.

| Output scale | Renderer p50 / p95 / max (ms) | Server control-to-JPEG p50 / p95 / max (ms) | Main-view Laplacian variance | Main-view mean squared Sobel magnitude |
| --- | ---: | ---: | ---: | ---: |
| 0.50 | 33.00 / 40.58 / 47.69 | 73.73 / 81.25 / 87.86 | 53.62 | 3569.52 |
| 0.75 | 43.87 / 53.65 / 63.85 | 84.49 / 94.99 / 103.80 | 94.57 | 4629.34 |
| 1.00 | 61.98 / 70.75 / 79.73 | 102.63 / 111.38 / 120.10 | 137.16 | 5302.24 |

The edge diagnostics use the driver-visible JPEG region
`x=[0,1300), y=[40,620)`, excluding the trajectory inset, top label, and bottom
status bar. They measure edge energy after the complete render/projection/JPEG
path. They do not establish geometric correctness, realism, or driver task
success.

Relative to 0.50, scale 0.75 increased the main-view Laplacian statistic by
about 76% and the Sobel statistic by about 30%. Direct inspection found a
modest but useful improvement in foliage, building, wire, and road-edge
definition. Scale 1.00 retained more edges again, but its 111.38 ms server p95
failed the declared 100 ms gate.

## Selected-scale motion smoke

Scale 0.75 was then exercised with 25 consecutive throttle samples from the
reset pose:

- reached and held the 4.0 m/s configured speed cap;
- all three front frames were finite for all samples;
- zero support violations and zero boundary hits;
- maximum absolute lateral offset: 0.0704 m;
- minimum remaining distance margin: 0.9296 m;
- renderer latency: 43.08 ms p50, 47.21 ms p95, 58.10 ms maximum;
- server control through cockpit JPEG: 84.51 ms p50, 88.61 ms p95,
  99.58 ms maximum.

This confirms that the selected scale retains useful latency margin under a
short moving-pose smoke. It does not include browser image-load timing,
physical keyboard input, monitor scan-out, branch-profile transition, or a
sustained human steering trial.

## Decision

Use 0.75 as the TbV driving adapter default. It is the clearest tested scale
that passes the existing server p95 budget. Keep 1.00 as an offline inspection
option, not the default interactive setting.

Do not add more static training yet. The remaining blur is partly checkpoint
quality, but this comparison proves that the old 0.50 driving path was also
discarding useful rendered detail. The next discriminating gate is the
straight/right physical driving trial at 0.75. Train or change the model only
if driving-relevant blur, floaters, seams, or baked vehicles still alter lane,
curb, branch, or obstacle interpretation.

## Artifacts outside Git

```text
/home/yawei/stage3_external/artifacts/tbv_cockpit_resolution_ab/
├── scale_050/
│   ├── cockpit.jpg
│   ├── diagnostic.jpg
│   ├── state.json
│   └── tbv_driving_evidence.json
├── scale_075/
│   ├── cockpit.jpg
│   ├── diagnostic.jpg
│   ├── state.json
│   └── tbv_driving_evidence.json
├── scale_100/
│   ├── cockpit.jpg
│   ├── diagnostic.jpg
│   ├── state.json
│   └── tbv_driving_evidence.json
└── selected_scale_075_motion/
    ├── cockpit_4mps.jpg
    ├── state.json
    └── tbv_driving_evidence.json
```
