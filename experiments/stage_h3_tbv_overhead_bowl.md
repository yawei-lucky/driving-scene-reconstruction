# Stage H3 TbV Visual-Only Bowl Overhead

Date: 2026-07-23

## Question

Can the existing cockpit retain the 0.75-scale forward surround as its primary
view while adding a useful AVM-style overhead inset without reconstructing a
3D environment, completing unknown pixels, or exceeding the 100 ms p95
server-control-to-forward-image budget?

## Implemented scope

The three display roles are now named explicitly:

- **forward surround**: the primary 150-degree cylindrical view built from the
  three front cameras;
- **overhead**: a small visual-only AVM-style projection with trusted route
  support overlaid;
- **original camera views**: the separate manual seven-camera diagnostic.

The overhead follows the classic calibrated-remap approach rather than LiDAR
mapping or 3D reconstruction:

1. define a fixed vehicle-local virtual bowl;
2. project every bowl pixel into each undistorted camera using the calibrated
   camera-to-rig transform and pinhole intrinsics;
3. precompute per-traversal OpenCV remap tables and feathering weights;
4. blend only covered pixels;
5. leave uncovered pixels black and place an opaque vehicle mask over the
   central blind zone;
6. draw the logged route, +/-1 m support tube, lateral offset, and remaining
   margin above the image.

No generative completion, LiDAR free-space inference, collision claim, or
environment mesh is introduced.

## Projection configuration

- output inset: 284x284 pixels;
- vehicle-local extent: 16.0 m forward, 5.0 m rearward, and 10.5 m to each
  side;
- near ground height: -1.43 m relative to the renderer's camera-rig centre,
  taken from the released camera calibration;
- flat bowl radius: 4.0 m;
- maximum bowl rise: 1.10 m;
- primary front cameras: 0.75 output scale;
- background side/rear cameras: 0.375 output scale;
- update policy: one background update per forward control frame;
- stale-ground handling: SE(2) motion compensation into the current ego pose.

The side/rear render runs on one background worker after the synchronous
three-front render. In the observed 4.0 m/s run, the displayed source was
normally one control frame, or 0.4 m, behind before ground-motion compensation.
This keeps the additional GPU work concurrent with forward-view CPU
composition instead of placing four more cameras on the critical response
path.

## Actual host evidence

Runtime:

- host: `stf-precision-3680`;
- GPU: NVIDIA GeForce RTX 4090 D, driver 580.95.05;
- checkpoint:
  `tbv_branch_pair_splatad_static_8000/.../step-000007999.ckpt`;
- primary output scale: 0.75;
- simulation step: 0.1 s.

Projection coverage:

- right-turn traversal profile: 0.880889;
- straight traversal profile: 0.879290.

Both profiles produced finite overhead images with no background-render error.
One profile-switch run reached the shared anchor in 60 throttle samples and
successfully rebuilt the overhead for the straight traversal.

The clean 25-sample acceleration run reached and held 4.0 m/s:

- all three latency-critical front frames were finite;
- zero support violations and zero boundary hits;
- maximum absolute lateral offset: 0.0704 m;
- minimum remaining corridor margin: 0.9296 m;
- front renderer latency: 43.72 ms p50, 45.40 ms p95,
  58.94 ms maximum;
- server control through forward-surround JPEG: 88.84 ms p50,
  92.97 ms p95, 102.82 ms maximum;
- zero skipped overhead updates and no reported background error.

The immediately preceding no-overhead 0.75-scale motion smoke measured
84.51/88.61 ms server p50/p95. The overhead pilot therefore added about
4.3 ms at p50 and 4.4 ms at p95 while retaining the declared p95 gate. The
102.82 ms single-sample maximum is recorded but is not the project's p95
acceptance statistic.

Observed background source work was approximately 27-33 ms for the four
reduced-scale cameras plus about 9.7 ms for bowl composition. These are
individual state observations, not latency distributions.

## Visual decision and limits

Accept the bowl overhead as a comfort-only auxiliary:

- the road surface, curb relationship, vehicle pose, and route direction are
  easier to read than on the trajectory-only inset;
- the opaque centre mask hides the vehicle blind zone;
- uncovered areas remain visibly black;
- the forward surround remains the dominant image and its 0.75-scale
  sharpness is unchanged.

Do not treat the overhead as image truth. Poles, trees, walls, and vehicles
stretch or lean because the compositor projects them onto a virtual bowl.
Camera seams, static-model ghosts, and one-frame non-ground motion error also
remain. The route and +/-1 m overlay are evidence-bearing; the camera texture
underneath is not.

Physical browser input-to-image timing and a sustained human steering review
remain open.

## Artifacts outside Git

```text
/home/yawei/stage3_external/artifacts/tbv_overhead_bowl_pilot_final/
├── initial_cockpit.jpg
├── initial_overhead.jpg
├── motion_cockpit.jpg
├── motion_overhead.jpg
├── motion_state.json
└── tbv_driving_evidence.json

/home/yawei/stage3_external/artifacts/tbv_overhead_bowl_profiles/
├── right_profile_overhead.jpg
├── right_profile_state.json
├── straight_profile_cockpit.jpg
├── straight_profile_overhead.jpg
├── straight_profile_state.json
└── tbv_driving_evidence.json
```
