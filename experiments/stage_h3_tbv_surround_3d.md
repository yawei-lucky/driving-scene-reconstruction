# Stage H3 TbV Fixed-Bathtub 360° 3D Surround

Date: 2026-07-23

## Question

Can the TbV cockpit add a compact, fixed 360° 3D surround presentation without
introducing a depth pipeline or rebuilding the environment?

## Minimum implementation

The 0.75-scale cylindrical forward surround remains the primary driving view.
The auxiliary view now uses:

1. the seven RGB cameras from the latest completed same-profile pose snapshot;
2. one fixed vehicle-local bathtub display mesh;
3. calibrated camera projection into that mesh through precomputed lookup
   tables;
4. one fixed `rear_left` three-quarter virtual viewpoint.

Only the projection lookup tables are precomputed. Camera imagery is generated
for each background pose snapshot, which can trail the forward driving frame
by one update. The display path uses no depth, LiDAR, scene point cloud,
environment mesh, or generated completion.

## Display configuration

- vehicle-local extent: 8 m forward, 6 m rearward, and 7 m to each side;
- fixed ground height near the vehicle: -1.43 m relative to the camera-rig
  centre;
- flat near region followed by a smooth bathtub rise;
- maximum outer rise: 4 m;
- side/rear source-camera output scale: 0.375;
- virtual-camera field of view: 68 degrees;
- uncovered and near-vehicle blind regions: black;
- simple opaque vehicle proxy and logged-route support overlay: retained.

The mesh is a display surface, not a reconstruction of the surrounding world.

## Actual host evidence

Projection coverage of the fixed viewpoint at both traversal profiles:

| Virtual viewpoint | Right/common profile | Straight profile |
| --- | ---: | ---: |
| rear-left | 0.8028 | 0.7985 |

Observed component costs:

- four reduced-scale side/rear source-camera renders: 26.56/27.70 ms p50/p95;
- composition of the fixed virtual viewpoint: 9.91/17.01 ms p50/p95.

A clean 25-step straight-driving run on the actual RTX 4090 D host reached
4.0 m/s without a boundary hit or background skip:

| Timing | p50 | p95 | maximum |
| --- | ---: | ---: | ---: |
| three-camera renderer | 43.56 ms | 44.50 ms | 45.65 ms |
| server control to JPEG | 88.93 ms | 91.48 ms | 91.79 ms |
| JPEG presentation work | 45.65 ms | 47.13 ms | 47.24 ms |

The final state reported the background surround source one driving frame, or
0.4 m, behind the current forward state. This is an observed source lag, not
motion-compensated image truth. Browser decoding, monitor scan-out, and a human
visual acceptance trial were not measured by this host-side run.

## Visual decision and limits

Keep the fixed-bathtub view as a small visual aid:

- it provides one recognizable rear-left three-quarter view around the ego
  vehicle;
- projecting vertical content onto the bathtub walls is less misleading than
  forcing all pixels onto one flat ground plane;
- the central blind region remains visibly black instead of being invented;
- the forward surround remains the image used for steering decisions.

Do not treat this as a true 3D environment view. With RGB-only source cameras
and one fixed display surface, the compositor cannot recover the real shape,
depth, or occlusion of poles, trees, walls, and vehicles. It also provides no
free-space, collision, or route-support certificate. The logged-route overlay,
not the camera texture, retains the evidence-bearing support semantics.

The next gate is a concise real-operator visual trial of the common approach,
straight branch, and right branch. Avoid more static training or a broad
re-audit until that trial determines whether the primary forward view is
decision-safe.

## Artifacts outside Git

```text
/home/yawei/stage3_external/artifacts/tbv_surround_3d_pilot/
```
