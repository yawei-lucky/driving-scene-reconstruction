# Driver Attention And Dynamic-Traffic Requirements

Date: 2026-07-21

## Product Principle

The simulator operator's only job is to drive. The operator must not be asked
to compensate for reconstruction defects, inspect debug views, identify which
objects are trustworthy, or perform any side task while driving.

Visual softness is acceptable only when it does not change the driving task.
A reconstruction defect becomes a product blocker when it can:

- hide or break the readable road, lane, curb, signal, or drivable corridor;
- create a false obstacle or remove a real obstacle;
- make another road user appear static, duplicated, displaced, or moving in
  the wrong direction;
- cause the driver to hesitate, reinterpret the scene, or divert attention
  from steering and speed control.

## Dynamic Traffic Is Deferred, Not Waived

The current human-drivable MVP may use the accepted static reconstruction to
prove logged-time progression, ego control, multi-camera rendering, latency,
and reset behavior. This does not make background-baked traffic an acceptable
final state.

Before the simulator is used for strict autonomous-driving evaluation or for
driving scenarios in which traffic affects the route, dynamic vehicles and
pedestrians must have coherent position, visibility, motion, and collision
semantics. A blurred appearance alone is not a failure; incorrect driving
meaning is.

Dynamic reconstruction returns to the critical path immediately when any of
these conditions holds:

1. a blurred or baked object obscures a road or lane boundary;
2. a static residue can be mistaken for a real obstacle;
3. the simulator begins closed-loop autonomous-driving evaluation.

## Current Decision

Freeze the accepted PandaSet static-8k checkpoint as the visual baseline and
build the drivable loop around it. Do not require the driver to work around its
known traffic artifacts. Select a low-interference segment for the first MVP,
record every known artifact, and keep dynamic-traffic correctness as an
explicit later acceptance gate rather than an open-ended image-quality task.
