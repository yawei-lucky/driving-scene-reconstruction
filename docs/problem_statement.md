# Problem Statement: Driving Scene Reconstruction and View Extrapolation

## 1. Core Problem

Real-driving scene reconstruction aims to recover a usable representation of a driving scene from real sensor observations. In this repository, the initial focus is **view extrapolation**:

> Given observed camera views from a real driving scene, generate plausible and driving-consistent views from unobserved or weakly observed viewpoints.

This is especially important for 360-degree visualization, remote driving, and sensor-level simulation.

## 2. Why This Is Hard

Driving scenes are difficult because they contain:

- large viewpoint changes;
- nearby dynamic objects;
- strong occlusion;
- thin structures such as poles, lane markings, fences, and traffic signs;
- reflective and transparent surfaces;
- exposure changes between cameras;
- fast ego-motion;
- safety-critical spatial relations.

A generated view can look visually acceptable while still being wrong for driving.

## 3. Failure Modes

Typical failures include:

| Category | Failure |
|---|---|
| Image quality | blur, smear, ghosting, texture collapse |
| Geometry | bent lanes, distorted road boundaries, shifted objects |
| Object consistency | hallucinated objects, missing objects, wrong scale |
| Occlusion | wrong visible / hidden relation |
| Temporal stability | flicker, popping, frame-to-frame deformation |
| Driving relation | wrong left-right, front-rear, same-lane, adjacent-lane relation |

## 4. Why Image Quality Is Not Enough

For driving applications, visual realism is not the final objective. A reconstructed view should preserve task-relevant relations:

- where the road is;
- where the lanes are;
- where other vehicles and pedestrians are;
- whether an object is in the ego lane or adjacent lane;
- whether an object is approaching or receding;
- whether an obstacle is truly visible or only hallucinated.

Therefore, evaluation should go beyond PSNR, SSIM, LPIPS, or visual inspection.

## 5. Initial Project Goal

The first goal is not to build a full reconstruction model. The first goal is to define a measurable benchmark-style case:

> Use real multi-camera driving logs to test how reliable large-angle view extrapolation is, and identify where it fails.

This will create a clear foundation for later model selection and implementation.
