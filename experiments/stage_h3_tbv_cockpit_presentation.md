# Stage H3 TbV Cockpit Presentation

Date: 2026-07-23

## Question

Can the existing route-constrained TbV adapter present one useful human-driving
view while keeping route support and raw reconstruction diagnostics distinct?

## Implemented scope

- the default `/` page uses one 1600x620 calibrated cylindrical front view;
- `ring_front_left`, `ring_front_center`, and `ring_front_right` are projected
  from their trained pinhole intrinsics and rig rotations into a 150 degree
  horizontal field of view;
- calibrated overlaps are incidence-weighted and feathered; no generated
  completion or overhead RGB camera is introduced;
- an ego-up inset draws the logged trajectory support, +/-1 m active tube,
  straight/right paths, ego pose, lateral offset, and remaining margin;
- `/diagnostic` retains all seven cameras at their original aspect ratios;
- the adapter's configurable default speed cap is 4.0 m/s
  (`H3_TBV_MAX_SPEED_MPS`);
- `/state.json` and the persisted route contract identify the display mode,
  front field of view, projection coverage, support-map source, diagnostic URL,
  and speed cap.

The inset is trajectory evidence only. It is not an overhead photograph, a
LiDAR occupancy product, or a free-space certificate.

## Actual host smoke

Runtime:

- host: `stf-precision-3680`;
- GPU: NVIDIA GeForce RTX 4090 D, driver 580.95.05;
- checkpoint:
  `tbv_branch_pair_splatad_static_8000/.../step-000007999.ckpt`;
- output scale: 0.5;
- simulation step: 0.1 s.

Calibration coverage:

- right-turn traversal profile: 0.9998155;
- straight traversal profile: 0.9998024.

Both profiles rendered successfully and were visually inspected at the branch
anchor. The straight/right route geometry in the inset changed consistently
with the selected profile. The separate seven-camera diagnostic retained every
camera and its complete aspect ratio.

The final 25-sample acceleration smoke reached and held 4.0 m/s:

- seven finite camera frames for every sample;
- zero route-support violations and zero boundary hits;
- maximum absolute lateral offset: 0.0704 m;
- minimum remaining distance margin: 0.9296 m;
- renderer latency: 57.35 ms p50, 60.33 ms p95, 73.53 ms maximum;
- server control through cylindrical cockpit JPEG: 97.40 ms p50,
  100.34 ms p95, 113.43 ms maximum.

This is a host GPU/HTTP result. Browser request-to-image, physical
input-to-image, monitor scan-out, and sustained human steering were not
measured.

## Artifacts outside Git

```text
/home/yawei/stage3_external/artifacts/tbv_branch_pair_driving_adapter/
├── tbv_driving_evidence.json
└── cockpit/
    ├── common_approach_4mps.jpg
    ├── straight_profile_anchor.jpg
    └── seven_camera_diagnostic.jpg
```

## Visual decision and limits

Accept the presentation split for the first physical driving trial:

- the forward road occupies one continuous wide view;
- the support inset communicates route topology and margin without pretending
  to be reconstructed overhead truth;
- the raw seven-camera output remains available without distracting the
  driver.

Do not treat this as visual acceptance. The calibrated camera overlap still
shows parallax seams and traversal/sensor-profile colour differences. Static
reconstruction blur, floaters, and baked vehicles also remain. During the
human trial, reject any interval where these artifacts change lane, curb,
branch, or obstacle interpretation.
