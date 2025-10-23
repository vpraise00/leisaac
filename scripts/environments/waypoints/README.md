# Waypoint-Based Data Collection

This directory contains tools for automated demonstration data collection using waypoint-based control for bi-arm SO-101 robot manipulation tasks in IsaacLab.

## Overview

The waypoint system allows you to:
1. Define robot trajectories as sequences of target end-effector positions
2. Automatically execute waypoints with different controller backends
3. Record demonstration data in HDF5 format with camera observations
4. Convert datasets to LeRobot format and upload to HuggingFace

## System Architecture

### Controller Modules

- **`waypoint_controller_dik.py`**: DifferentialIK-based controller (default)
  - Fast and simple Jacobian-based inverse kinematics
  - Suitable for most manipulation tasks
  - Supports position-only or full pose control

- **`waypoint_controller_osc.py`**: OperationalSpaceController-based controller (experimental)
  - Impedance control with smoother motion generation
  - May provide better stability for complex interactions
  - Currently under development

### Data Collection Script

- **`bi_arm_waypoint_data_collection.py`**: Main execution script
  - Loads waypoints from JSON files
  - Executes trajectories with selected controller
  - Records observations, actions, and camera data
  - Supports automatic episode management and success labeling

## Quick Start

### 1. Define Waypoints

Create a JSON file with waypoint sequence (e.g., `playground/waypoints/bi_arm_demo.json`):

```json
[
  {
    "relative": false,
    "left": {
      "position": [0.22, -0.45, 0.22],
      "gripper": 0.0,
      "wrist_flex": 1.57
    },
    "right": {
      "position": [0.55, -0.45, 0.22],
      "gripper": 0.0,
      "wrist_flex": 1.57
    },
    "hold_steps": 45
  },
  {
    "relative": false,
    "left": {
      "position": [0.22, -0.45, 0.16],
      "gripper": 0.6,
      "wrist_flex": 1.57
    },
    "right": {
      "position": [0.55, -0.45, 0.16],
      "gripper": 0.6,
      "wrist_flex": 1.57
    },
    "hold_steps": 30
  }
]
```

**Waypoint Parameters:**
- `position`: End-effector target position in world frame [x, y, z] (meters)
- `orientation`: (Optional) Target orientation as quaternion [w, x, y, z]
- `gripper`: Gripper opening (0.0 = closed, 1.0 = open)
- `wrist_flex`: (Optional) Override wrist_flex joint angle in radians
- `hold_steps`: Number of steps to hold at waypoint before proceeding

### 2. Collect Data

**Basic usage (playback only):**
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_demo.json" \
    --controller_type=dik \
    --enable_cameras
```

**Record demonstrations:**
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_demo.json" \
    --controller_type=dik \
    --record \
    --dataset_file="datasets/my_demos.hdf5" \
    --num_demos=10 \
    --enable_cameras
```

**Using batch file:**
```bash
# Edit playground/running_scripts/collect_waypoint_data.bat with your settings
playground\running_scripts\collect_waypoint_data.bat
```

### 3. Command-Line Arguments

#### Controller Selection
- `--controller_type`: Controller backend
  - `dik` (default): DifferentialIK controller
  - `osc` (experimental): OperationalSpaceController

#### Waypoint Parameters
- `--waypoint_file`: Path to waypoint JSON file
- `--step_hz`: Environment stepping rate (default: 30 Hz)
- `--hold_steps`: Global override for waypoint hold durations
- `--position_tol`: Position convergence tolerance in meters (default: 0.05)
- `--pose_interp_gain`: Pose interpolation smoothing (0-1, default: 0.3)

#### DifferentialIK Specific
- `--command_type`: IK control mode
  - `position`: Position-only control (faster, recommended)
  - `pose`: Full 6-DOF pose control
- `--interp_gain`: Joint interpolation smoothing (0-1, default: 0.3)

#### OperationalSpaceController Specific (Experimental)
- `--motion_stiffness`: Impedance stiffness (default: 150.0)
- `--motion_damping_ratio`: Damping ratio (default: 1.0)

#### Wrist Control
- `--force_wrist_down`: Force wrist_flex to point downward (default: enabled)
- `--wrist_flex_angle`: Target wrist_flex angle in radians (default: 1.57 = 90°)

#### Recording
- `--record`: Enable data recording
- `--dataset_file`: Output HDF5 file path
- `--num_demos`: Number of demonstrations to collect
- `--enable_cameras`: Enable camera observations (required for VLM datasets)

## Dataset Pipeline

### HDF5 Dataset Structure

Recorded demonstrations are saved in HDF5 format:
```
dataset.hdf5
└── data/
    ├── demo_0/
    │   ├── actions              # (T, 12) joint positions
    │   ├── obs/
    │   │   ├── joint_pos        # (T, 12)
    │   │   ├── joint_vel        # (T, 12)
    │   │   ├── left_wrist       # (T, H, W, 3) left camera RGB
    │   │   ├── right_wrist      # (T, H, W, 3) right camera RGB
    │   │   └── top              # (T, H, W, 3) top camera RGB
    │   └── attrs: {num_samples, seed, success}
    └── demo_1/
        └── ...
```

### Convert to LeRobot Format

**Prerequisites:**
```bash
# Create separate LeRobot environment
conda create -n lerobot python=3.10
conda activate lerobot
pip install lerobot==0.3.3  # Tested version
```

**Convert and upload:**
```bash
conda activate lerobot

python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo YourUsername/dataset_name \
    --robot_type so101_follower \
    --fps 30 \
    --hdf5_files datasets/my_demos.hdf5 \
    --task "Pick up objects and place them in the box" \
    --push_to_hub
```

**Parameters:**
- `--dataset_path_or_repo`: HuggingFace dataset repository name
- `--robot_type`: Robot configuration (`so101_follower` or `bi_so101_follower`)
- `--fps`: Video frame rate (default: 30)
- `--hdf5_files`: Space-separated list of HDF5 files to merge
- `--task`: Task description for dataset metadata
- `--push_to_hub`: Upload to HuggingFace Hub (requires authentication)

**Authentication:**
```bash
# First time setup
huggingface-cli login
```

### Complete Workflow Example

```bash
# 1. Collect demonstrations
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_demo.json" \
    --controller_type=dik \
    --record \
    --dataset_file="datasets/clean_table_demos.hdf5" \
    --num_demos=50 \
    --enable_cameras

# 2. Convert to LeRobot and upload
conda activate lerobot
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo myusername/clean_table_so101 \
    --robot_type bi_so101_follower \
    --fps 30 \
    --hdf5_files datasets/clean_table_demos.hdf5 \
    --task "Clean toys from table surface using bi-arm coordination" \
    --push_to_hub
```

## Advanced Usage

### Multiple Waypoint Files

Collect data from different scenarios and merge:
```bash
# Collect scenario 1
python ... --dataset_file="datasets/scenario1.hdf5"

# Collect scenario 2
python ... --dataset_file="datasets/scenario2.hdf5"

# Merge during conversion
python scripts/convert/isaaclab2lerobot.py \
    --hdf5_files datasets/scenario1.hdf5 datasets/scenario2.hdf5 \
    --push_to_hub
```

### Debugging Waypoints

Run without recording to test trajectories:
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/test.json" \
    --controller_type=dik \
    --enable_cameras
```

Press `Ctrl+C` to stop execution.

## Troubleshooting

### GPU Crashes During Data Collection
- Try reducing `--step_hz` (e.g., from 30 to 20 Hz)
- Decrease `--pose_interp_gain` for smoother transitions
- Ensure first waypoint doesn't have `wrist_flex` to allow smooth reset transitions
- Consider using `--controller_type=osc` (experimental)

### Gripper Not Pointing Downward
- Ensure `--force_wrist_down` is enabled (default)
- Add explicit `wrist_flex: 1.57` to waypoint JSON
- Check `--wrist_flex_angle` matches desired orientation

### Waypoint Not Converging
- Increase `--position_tol` (e.g., from 0.05 to 0.1)
- Increase waypoint `hold_steps` in JSON
- Verify target positions are reachable (no singularities)

### LeRobot Conversion Errors
- Ensure LeRobot version compatibility (`pip install lerobot==0.3.3`)
- Check HDF5 file contains successful episodes (`success=True`)
- Verify camera keys match expected format (left_wrist, right_wrist, top)

## See Also

- [Main Repository README](../../../README.md)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
- [IsaacLab Tutorials](https://isaac-sim.github.io/IsaacLab/)
