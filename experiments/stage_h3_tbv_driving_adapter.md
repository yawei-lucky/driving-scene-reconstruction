# Stage H3 TbV Route Driving Adapter

Date: 2026-07-22

## Question

Can the accepted TbV step-7,999 checkpoint support a minimal continuous
driving loop from the shared entrance, with an explicit straight/right choice
and enough machine-readable evidence to distinguish route support from visual
credibility?

## Implemented scope

- spawn at branch-local progress -20 m;
- W/S/A/D control through the existing kinematic vehicle model;
- a fail-closed +/-1 m centreline tube and +/-30 degree heading limit;
- stop within 0.5 m of the shared anchor until straight or right is selected;
- traversal-specific seven-camera rendering at one frozen scene time;
- aspect-ratio-preserving mosaic with the three forward cameras prioritized;
- `/state.json`, `/frame.jpg`, and `/evidence.json` endpoints;
- persisted state, control, branch, support-margin, frame-hash, renderer, JPEG,
  reset, and optional browser-timing evidence.

The boundary never clamps an invalid candidate onto the route. It retains the
last valid pose, stops the vehicle, and requires reset. The evidence schema is
explicitly `evidence_only_not_certified`.

## Actual host run

Host environment:

- host: `stf-precision-3680`;
- GPU: NVIDIA GeForce RTX 4090 D, driver 580.95.05;
- checkpoint: `step-000007999.ckpt` from
  `tbv_branch_pair_splatad_static_8000`;
- output scale: 0.5;
- simulation step: 0.1 s.

The final sequential HTTP rehearsal contains:

- 241 control samples and two resets;
- both `straight` and `right` branch selections;
- seven finite camera frames for every committed sample;
- zero committed route-support violations;
- one intentional right-branch boundary event: the 1.013 m candidate was
  rejected, the last valid 0.979 m-offset pose was retained, and a later
  control request received HTTP 400 until reset;
- renderer latency 57.61 ms p50, 59.77 ms p95, 74.31 ms maximum;
- server control through normalized-mosaic JPEG latency 82.55 ms p50,
  85.01 ms p95, 99.46 ms maximum.

At the anchor reached by the straight-throttle machine schedule, both branch
candidates were inside support: right had 0.1515 m remaining distance margin
and straight had 0.7162 m. This is a route-boundary result, not a visual truth
claim.

## Display correction found during the run

The first browser mosaic assumed equal image sizes. TbV's portrait front-centre
and landscape ring cameras have different shapes, so the front-centre tile was
cropped to its sky region even though the renderer output was valid. The
mosaic now normalizes each tile into a fixed viewport without changing aspect
ratio. The preserved initial and straight frames show the full forward road.

## Artifacts outside Git

```text
/home/yawei/stage3_external/artifacts/tbv_branch_pair_driving_adapter/
├── tbv_driving_evidence.json
├── initial_normalized_mosaic.jpg
└── straight_smoke_frame.jpg
```

## Decision and remaining gate

Accept the adapter and evidence outlet as machine-validated plumbing. Do not
claim human drivability yet:

- the rehearsal used HTTP controls, not physical keyboard-to-display input;
- browser timing coverage is therefore correctly recorded as 0.0;
- traversal-profile switching is hashed and timed but still needs a human
  continuity judgment;
- vehicles remain baked into static geometry;
- no lateral ground truth or collision model exists.

Next, run straight and right in separate human reset trials. Preserve browser
request-to-image and input-to-image timing, and reject any segment where road
topology, branch identity, baked traffic, or temporal artifacts change the
driving decision. Do not train beyond 8k before that gate.
