## Retarget Robot Motion from RGB-D Images with ArUco Marker

### Overview

This example detects human hand pose from an **RGB-D image sequence** and retargets it to a robot dexterous hand
using **position retargeting**. An **ArUco marker** (DICT_4X4_50) placed on the table surface provides the
camera-to-table coordinate transformation, so the output joint trajectory includes absolute wrist position and
orientation in the table frame.

The output is designed for direct replay in **Isaac Gym** (or other simulators).

### Pipeline

```
RGB image  ──→  MediaPipe Hands  ──→  2D hand keypoints (21 joints)
                                           │
Depth image ─→  Back-projection  ──→  3D keypoints in camera frame
                                           │
RGB image  ──→  ArUco detect     ──→  T_cam2table (4x4)
                (DICT_4X4_50)              │
                                           ▼
                              3D keypoints in table frame (meters)
                                           │
                                           ▼
                              Position Retargeting (with 6D free joint)
                                           │
                                           ▼
                              qpos = [tx, ty, tz, rx, ry, rz, j0, j1, ...]
```

### Prerequisites

Install the additional dependencies on top of the base `dex-retargeting` package:

```shell
pip install mediapipe opencv-python opencv-contrib-python tyro tqdm pytransform3d
```

> **Note**: `opencv-contrib-python` provides the `cv2.aruco` module. If you already have
> `opencv-python` installed, you may need to uninstall it first and install
> `opencv-contrib-python` instead, or use `opencv-python` >= 4.7 which includes ArUco.

### Preparing Your Data

#### Camera Intrinsics

Provide a JSON file with camera intrinsics and distortion coefficients
(same format as `camera_params/color_intrinsics.json`):

```json
{
    "fx": 615.0,
    "fy": 615.0,
    "cx": 320.0,
    "cy": 240.0,
    "distortion": {
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0
    }
}
```

#### Image Data

Organize your RGB-D data as two directories with matching filenames sorted alphabetically:

```
my_data/
├── rgb/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
└── depth/
    ├── 000000.png    # 16-bit PNG in millimeters
    ├── 000001.png    # or .npy files in meters
    └── ...
```

- **RGB**: `.png` or `.jpg`, standard 8-bit color images.
- **Depth**: `.png` (16-bit, values in **millimeters**) or `.npy` (float32, values in **meters**).

#### ArUco Marker

- Print a **DICT_4X4_50** ArUco marker and place it **flat on the table surface**.
- Measure its physical side length in meters (default: 0.04 m = 4 cm).
- You can verify marker detection with the included test script:

```shell
python test_aruco_pose.py
```

### Running the Retargeting

```shell
cd example/position_retargeting

python detect_from_rgbd.py \
    --robot-name linker_l20 \
    --rgb-dir my_data/rgb \
    --depth-dir my_data/depth \
    --output-path output/result.pickle \
    --camera-params-json camera_params/color_intrinsics.json \
    --marker-length 0.04 \
    --hand-type right

python detect_from_rgbd_table_fixed.py \
  --robot-name linker_l20 \
  --rgb-dir ./my_data/rgb \
  --depth-dir ./my_data/depth \
  --output-path ./output/result_table_fixed.pickle \
  --config-path /home/maker01/Code/dex-retargeting/src/dex_retargeting/configs/offline/linker_l20_hand_right.yml

```

**Arguments:**

| Argument | Description |
|---|---|
| `--robot-name` | Robot hand identifier: `allegro`, `shadow`, `leap`, `inspire`, etc. |
| `--rgb-dir` | Directory containing RGB images. |
| `--depth-dir` | Directory containing depth images. |
| `--output-path` | Output pickle file path. |
| `--camera-params-json` | Path to camera intrinsics JSON (default: `camera_params/color_intrinsics.json`). |
| `--marker-length` | Physical side length of the ArUco marker in meters (default: 0.04). |
| `--hand-type` | `right` or `left` (default: `right`). |
| `--config-path` | (Optional) Custom retargeting config YAML. Defaults to the built-in position config. |

For the full list of options:

```shell
python detect_from_rgbd.py --help
```

### Output Format

The output pickle file contains a dictionary with two keys:

```python
{
    "data": [
        {
            "qpos": np.ndarray,            # Robot joint positions (including 6D wrist free joint)
            "wrist_pos": np.ndarray,       # (3,) wrist position in table frame (meters)
            "wrist_rot": np.ndarray,       # (3,3) wrist rotation matrix (MANO convention)
            "joint_pos_table": np.ndarray, # (21,3) all joint positions in table frame
            "T_cam2table": np.ndarray,     # (4,4) camera-to-table transform
        },
        ...  # one entry per frame
    ],
    "meta_data": {
        "config_path": str,
        "dof": int,
        "joint_names": list,
        "camera_matrix": list,
        "dist_coeffs": list,
        "marker_length": float,
    }
}
```

The `qpos` array is structured as:

```
qpos[0:3]  -> wrist translation (tx, ty, tz) in meters
qpos[3:6]  -> wrist rotation (rx, ry, rz) as intrinsic XYZ Euler angles
qpos[6:]   -> finger joint angles in radians
```

### Loading in Isaac Gym

```python
import pickle
import numpy as np

with open("output/result.pickle", "rb") as f:
    result = pickle.load(f)

joint_names = result["meta_data"]["joint_names"]

for frame in result["data"]:
    qpos = frame["qpos"]

    # Wrist 6D pose (from the dummy free joints)
    wrist_translation = qpos[0:3]
    wrist_euler_xyz = qpos[3:6]

    # Finger joint angles
    finger_qpos = qpos[6:]

    # Set actor root state with wrist_translation + euler-to-quaternion conversion
    # Set DOF position targets with finger_qpos
```

### Supported Robots

All robots with a built-in `position` retargeting config are supported:

- Allegro Hand (`allegro`)
- Shadow Hand (`shadow`)
- LEAP Hand (`leap`)
- Inspire Hand (`inspire`)
- Ability Hand (`ability`)
- Schunk SVH Hand (`svh`)
- Panda Gripper (`panda`)

### Notes

1. **ArUco marker placement**: The marker must lie flat on the table surface so that the
   detected marker pose directly represents the table coordinate frame.
2. **ArUco dictionary**: Uses `DICT_4X4_50` by default, matching `test_aruco_pose.py`.
3. **Depth quality**: Hand edges often have noisy or missing depth values. The detector
   applies median filtering and neighborhood search to mitigate this.
4. **Coordinate frame**: The output is in the **table frame** defined by the ArUco marker.
5. **Warm start**: The first valid frame automatically initializes the wrist pose via
   `warm_start()`, which provides a good initial guess for the optimizer.
6. **Marker occlusion**: If the marker is temporarily occluded by the hand, the last
   successfully detected transform is reused.