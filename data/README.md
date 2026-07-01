# Data

This directory documents expected data formats. Large raw data should not be committed directly to GitHub.

## Expected Minimal Data

For the first MVP, each clip should provide:

```text
clip_id/
  images/
    cam_front/
    cam_front_left/
    cam_front_right/
    cam_rear_left/
    cam_rear_right/
    cam_rear/
  calibration/
    intrinsics.json
    extrinsics.json
  timestamps.csv
```

## Optional Data

Optional but useful:

```text
  ego_pose.csv
  lidar/
  depth/
  boxes_3d.json
  lanes.json
  can.csv
```

## Data Policy

- Do not commit large videos, raw logs, or private customer data.
- Commit only small toy samples or metadata templates when needed.
- Use external storage for real driving logs.
