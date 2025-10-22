# Bi-Arm Waypoint Control

This directory contains tools for executing pre-defined waypoint sequences on bi-arm robots, with optional data recording for demonstration collection.

## Overview

The waypoint system allows you to:
- Define a sequence of target poses (position + optional orientation) for both arms
- Execute waypoints automatically using differential IK control
- Record demonstrations in HDF5 format for training
- Run waypoints in playback mode for testing/debugging

## Architecture

### Core Components

1. **`waypoint_controller.py`** - Reusable waypoint execution module
   - `BiArmWaypointController`: Main controller class
   - `load_waypoints_from_json()`: Waypoint file parser
   - Handles IK computation, convergence checking, and action generation

2. **`bi_arm_waypoint_data_collection.py`** - Main execution script
   - Supports two modes: playback (no recording) and data collection (with recording)
   - Automatic episode management
   - Configurable via command-line arguments

3. **Waypoint JSON files** - Define waypoint sequences
   - Location: `playground/waypoints/*.json`
   - Format: Array of waypoint objects

## Waypoint JSON Format

```json
[
  {
    "relative": false,
    "left": {
      "position": [x, y, z],
      "orientation": [w, x, y, z],  // Optional: quaternion (world frame)
      "gripper": 0.0                 // 0.0 = open, 1.0 = closed
    },
    "right": {
      "position": [x, y, z],
      "orientation": [w, x, y, z],  // Optional
      "gripper": 0.0
    },
    "hold_steps": 45                 // Steps to hold at this waypoint
  },
  // ... more waypoints
]
```

**Notes:**
- `position`: [x, y, z] in meters (world frame)
- `orientation`: [w, x, y, z] quaternion (world frame) - only used in pose mode
- `gripper`: 0.0 (open) to 1.0 (closed)
- `hold_steps`: Number of simulation steps to hold at waypoint before moving to next
- If `orientation` is omitted, controller uses position-only mode (robot chooses natural orientation)

## Usage

### Mode 1: Playback Only (No Recording)

Run waypoints in a loop without saving data. Useful for testing and visualization.

**Command:**
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_relative_demo.json" \
    --step_hz=30 \
    --hold_steps=10 \
    --position_tol=0.05 \
    --command_type=position \
    --enable_cameras
```

**Or use batch file:**
```bash
playground\running_scripts\run_waypoints_playback.bat
```

**Behavior:**
- Executes all waypoints in sequence
- Waits 2 seconds after last waypoint
- Resets and loops infinitely
- No data is recorded

### Mode 2: Data Collection (Recording)

Record demonstrations by running waypoints multiple times.

**Command:**
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_relative_demo.json" \
    --record \
    --dataset_file="datasets/waypoint_demos.hdf5" \
    --num_demos=10 \
    --step_hz=30 \
    --hold_steps=10 \
    --position_tol=0.05 \
    --command_type=position \
    --enable_cameras
```

**Or use batch file:**
```bash
playground\running_scripts\collect_waypoint_data.bat
```

**Behavior:**
- Executes waypoint sequence
- Waits 2 seconds after last waypoint
- Marks episode as successful and saves to HDF5
- Resets and starts next episode
- Stops after collecting `num_demos` demonstrations

## Key Parameters

### Required
- `--task`: Task environment name (e.g., `LeIsaac-SO101-CleanToyTable-BiArm-v0`)
- `--waypoint_file`: Path to waypoint JSON file

### Waypoint Control
- `--command_type`: `position` (position-only) or `pose` (position+orientation)
  - `position`: Only controls end-effector position, orientation follows naturally
  - `pose`: Controls both position and orientation (requires `orientation` in JSON)
- `--hold_steps`: Override hold steps for all waypoints (optional)
- `--position_tol`: Position convergence tolerance in meters (default: 0.05)
- `--orientation_tol`: Orientation convergence tolerance (default: 0.02)
- `--pose_interp_gain`: Interpolation gain for pose commands (0-1, default: 0.3)
- `--interp_gain`: Interpolation gain for joint targets (0-1, default: 0.3)
- `--episode_end_delay`: Delay in seconds after last waypoint (default: 2.0)

### Recording (only with `--record`)
- `--record`: Enable data recording (flag)
- `--dataset_file`: Output HDF5 file path
- `--num_demos`: Number of demonstrations to collect

### Simulation
- `--step_hz`: Simulation frequency in Hz (default: 30)
- `--enable_cameras`: Enable camera rendering (for vision data)
- `--seed`: Random seed for environment

## Examples

### Example 1: Test Waypoints (Playback)
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/my_waypoints.json" \
    --step_hz=30 \
    --position_tol=0.05
```

### Example 2: Collect 50 Demonstrations
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_relative_demo.json" \
    --record \
    --dataset_file="datasets/clean_table_demos.hdf5" \
    --num_demos=50 \
    --enable_cameras
```

### Example 3: Pose Mode with Orientation Control
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/oriented_waypoints.json" \
    --command_type=pose \
    --orientation_tol=0.02 \
    --record \
    --dataset_file="datasets/oriented_demos.hdf5" \
    --num_demos=10
```

## Creating Waypoint Files

### Step 1: Determine Target Positions

Use teleoperation or manual control to find desired end-effector positions:

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --teleop_device=keyboard \
    --num_envs=1
```

Note the positions displayed in the terminal.

### Step 2: Create JSON File

Create a JSON file with your waypoints:

```json
[
  {
    "relative": false,
    "left": {
      "position": [0.22, -0.45, 0.22],
      "gripper": 0.0
    },
    "right": {
      "position": [0.55, -0.45, 0.22],
      "gripper": 0.0
    },
    "hold_steps": 45
  },
  {
    "relative": false,
    "left": {
      "position": [0.22, -0.45, 0.11],
      "gripper": 0.6
    },
    "right": {
      "position": [0.55, -0.45, 0.11],
      "gripper": 0.6
    },
    "hold_steps": 60
  }
]
```

### Step 3: Test Playback

```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/my_waypoints.json"
```

## Output Data Format

When using `--record`, data is saved to HDF5 format compatible with LeRobot:

```
dataset.hdf5
└── data/
    ├── demo_0/
    │   ├── actions          # (N, 12) - joint positions [left_arm(5), left_gripper(1), right_arm(5), right_gripper(1)]
    │   ├── obs/
    │   │   ├── joint_pos    # (N, 12) - current joint positions
    │   │   ├── front        # (N, H, W, 3) - front camera images (if enabled)
    │   │   └── wrist        # (N, H, W, 3) - wrist camera images (if enabled)
    │   └── attrs: {num_samples, seed, success}
    └── demo_1/
        └── ...
```

### Converting to LeRobot Format and Uploading to HuggingFace

After collecting demonstrations in HDF5 format, you can convert them to LeRobot's Parquet format and upload to HuggingFace Hub for training.

#### Prerequisites

The conversion script requires the LeRobot library (v0.3.3). Install it in a separate conda environment:

```bash
# Create LeRobot environment
conda create -n lerobot python=3.10
conda activate lerobot

# Install LeRobot v0.3.3
pip install lerobot==0.3.3

# Install additional dependencies
pip install datasets Pillow opencv-python tqdm
```

**Important:** The conversion script must be run in the `lerobot` environment, not the `leisaac` environment.

#### Conversion Command

```bash
# Activate LeRobot environment
conda activate lerobot

# Run conversion (bi-arm example)
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo your-username/clean-toy-table \
    --robot_type bi_so101_follower \
    --fps 30 \
    --hdf5_files datasets/waypoint_demos.hdf5 \
    --task "Clean toys from table using bi-arm robot" \
    --push_to_hub
```

#### Key Parameters

- `--dataset_path_or_repo`: HuggingFace repository name (format: `username/dataset-name`)
- `--robot_type`: Robot configuration type
  - `so101_follower`: Single-arm SO-101 robot (6 DOF: 5 arm joints + 1 gripper)
  - `bi_so101_follower`: Bi-arm SO-101 robot (12 DOF: 2 arms × 6 DOF each)
- `--fps`: Frames per second (must match `--step_hz` from data collection)
- `--hdf5_files`: Space-separated list of HDF5 files to convert (can merge multiple files)
- `--task`: Task description (used for dataset metadata)
- `--push_to_hub`: Upload to HuggingFace Hub (requires authentication)

#### Authentication Setup

Before pushing to HuggingFace Hub, authenticate:

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=your_token_here
```

#### Examples

**Example 1: Single-Arm Dataset**
```bash
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo myusername/pick-orange-demos \
    --robot_type so101_follower \
    --fps 30 \
    --hdf5_files datasets/orange_demos_1.hdf5 datasets/orange_demos_2.hdf5 \
    --task "Pick up orange and place on plate" \
    --push_to_hub
```

**Example 2: Bi-Arm Dataset (from waypoints)**
```bash
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo myusername/clean-table-waypoint \
    --robot_type bi_so101_follower \
    --fps 30 \
    --hdf5_files datasets/waypoint_demos.hdf5 \
    --task "Clean toy table using waypoint-based bi-arm control" \
    --push_to_hub
```

**Example 3: Local Conversion Only (no upload)**
```bash
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo local-test-dataset \
    --robot_type bi_so101_follower \
    --fps 30 \
    --hdf5_files datasets/waypoint_demos.hdf5 \
    --task "Test dataset"
# Omit --push_to_hub to save locally only
```

#### Conversion Process

The script performs the following steps:

1. **Load HDF5 Data**: Reads all successful episodes from input HDF5 files
2. **Skip Initial Frames**: Removes first 5 frames of each episode (stabilization period)
3. **Convert Actions**: Transforms joint positions from IsaacLab format (radians, [-π, π]) to LeRobot format (degrees, normalized ranges)
4. **Process Images**: Compresses camera images as video files (if `--enable_cameras` was used during collection)
5. **Create Parquet Files**: Saves data in LeRobot's columnar format for efficient loading
6. **Generate Metadata**: Creates `meta_data/info.json` with dataset statistics and configuration
7. **Upload to Hub**: Pushes dataset to HuggingFace repository (if `--push_to_hub` specified)

#### Output Structure

The converted dataset has the following structure:

```
local_data/your-username/dataset-name/
├── meta_data/
│   └── info.json              # Dataset metadata (fps, robot_type, task description)
├── videos/
│   ├── chunk-000/
│   │   ├── observation.images.front_000000.mp4
│   │   ├── observation.images.wrist_000000.mp4
│   │   └── ...
│   └── chunk-001/
│       └── ...
└── data/
    ├── chunk-000/
    │   └── train-00000-of-00001.parquet
    └── chunk-001/
        └── train-00001-of-00002.parquet
```

#### Verifying Conversion

After conversion, verify the dataset:

```bash
# In lerobot environment
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('your-username/dataset-name')
print(f'Total frames: {len(dataset)}')
print(f'Features: {dataset.features}')
"
```

#### Troubleshooting Conversion

**Issue: "Dataset not found" error**
- Solution: Ensure `--push_to_hub` is used and HuggingFace token is set
- Or check `local_data/` directory for local files

**Issue: "Robot type mismatch" error**
- Solution: Use `bi_so101_follower` for bi-arm tasks, `so101_follower` for single-arm
- Check HDF5 file has 12 joints (bi-arm) or 6 joints (single-arm)

**Issue: "FPS mismatch" warning**
- Solution: Ensure `--fps` matches `--step_hz` from data collection
- Common values: 30 Hz (default), 20 Hz (slower hardware)

**Issue: Video encoding fails**
- Solution: Install ffmpeg: `sudo apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (macOS)
- Or disable video conversion by modifying the script

## Troubleshooting

### Waypoints Not Converging

**Symptom:** Robot oscillates or hold counter keeps resetting

**Solutions:**
- Increase `--position_tol` (e.g., `0.05` → `0.10`)
- Decrease `--pose_interp_gain` for smoother motion (e.g., `0.3` → `0.2`)
- Decrease `--interp_gain` for more stable IK (e.g., `0.3` → `0.2`)
- Check waypoint positions are reachable (not in collision or outside workspace)

### Robot Flips or Unstable in Pose Mode

**Symptom:** Robot makes sudden large movements when using `--command_type=pose`

**Solutions:**
- Switch to `--command_type=position` mode
- Verify quaternions in JSON are normalized
- Use quaternion values from actual robot poses (see teleoperation logs)
- Increase `--orientation_tol` (e.g., `0.02` → `0.05`)

### Recording Not Starting

**Symptom:** `--record` flag doesn't save data

**Solutions:**
- Ensure `--record` flag is present
- Check `--dataset_file` path is valid and writable
- Verify `--num_demos` is greater than 0
- Check terminal output for StreamingRecorderManager messages

### Low Frame Rate

**Symptom:** Simulation runs slowly

**Solutions:**
- Decrease `--step_hz` (e.g., `30` → `20`)
- Disable cameras if not needed (remove `--enable_cameras`)
- Reduce number of environments (should already be 1 for waypoints)

## Tips and Best Practices

1. **Start with Playback Mode**: Always test waypoints without `--record` first to verify they work correctly

2. **Use Position Mode**: Unless you specifically need orientation control, use `--command_type=position` for better stability

3. **Tune Tolerances**: Adjust `--position_tol` based on your task requirements:
   - Precise manipulation: `0.01` - `0.02` m
   - General movement: `0.05` - `0.10` m

4. **Hold Steps**: Use longer hold times for:
   - Grasping operations (60+ steps)
   - Final placement (60+ steps)
   - Waypoints can use shorter holds (10-30 steps)

5. **Episode End Delay**: Adjust `--episode_end_delay` to ensure:
   - Object has settled after manipulation
   - Last action has fully executed
   - Typical range: 1-3 seconds

6. **Camera Rendering**: Only use `--enable_cameras` when needed for vision-based policies:
   - Adds computational overhead
   - Increases dataset size significantly

7. **Data Collection**: For robust datasets:
   - Collect at least 10-50 demonstrations per task
   - Use domain randomization in task config if available
   - Verify data quality by replaying HDF5 file

## File Structure

```
scripts/environments/waypoints/
├── README.md                              # This file
├── waypoint_controller.py                 # Core waypoint execution module
└── bi_arm_waypoint_data_collection.py    # Main execution script

playground/waypoints/
├── bi_arm_relative_demo.json             # Example waypoint file
└── *.json                                # Your custom waypoint files

playground/running_scripts/
├── run_waypoints_playback.bat            # Playback mode (no recording)
└── collect_waypoint_data.bat             # Data collection mode (with recording)
```

## Related Documentation

- [LeIsaac Project README](../../../README.md)
- [Teleoperation Guide](../teleoperation/)
- [Data Conversion Guide](../../convert/)
- [IsaacLab Controllers](https://isaac-sim.github.io/IsaacLab/source/api/isaaclab/controllers.html)
