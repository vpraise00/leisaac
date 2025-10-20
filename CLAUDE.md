# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LeIsaac integrates LeRobot teleoperation hardware with NVIDIA IsaacLab simulation for robotic manipulation. It enables data collection, policy training, and sim2real transfer using the SO101Leader/Follower system.

**Key Workflow:**
1. Teleoperate robot in IsaacLab simulation using SO101Leader device
2. Record demonstrations to HDF5 format
3. Convert HDF5 data to LeRobot Dataset format
4. Fine-tune policies (GR00T N1.5, LeRobot models)
5. Deploy policies in simulation or on real hardware

## Installation & Setup

**Core Installation:**
```bash
# Install IsaacLab first (see README.md for full instructions)
conda create -n leisaac python=3.10
conda activate leisaac
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Install IsaacLab
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.1.1  # Match IsaacSim 4.5
./isaaclab.sh --install

# Install LeIsaac
cd leisaac
pip install -e source/leisaac
```

**Optional Dependencies for Policy Inference:**
```bash
pip install -e "source/leisaac[gr00t]"        # For GR00T N1.5 inference
pip install -e "source/leisaac[lerobot-async]" # For LeRobot async inference
```

**Asset Download:**
Download scene assets from GitHub releases and extract to `assets/` directory. See README.md Asset Preparation section.

## Common Commands

### Teleoperation & Data Collection

**Basic teleoperation:**
```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=so101leader \
    --port=/dev/ttyACM0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/dataset.hdf5
```

**Keyboard teleoperation (for testing):**
```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --teleop_device=keyboard \
    --num_envs=1
```

**Bi-arm teleoperation:**
```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-BiArm-AssembleHamburger-v0 \
    --teleop_device=bi-so101leader \
    --left_arm_port=/dev/ttyACM0 \
    --right_arm_port=/dev/ttyACM1 \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/biarm_dataset.hdf5
```

**Controls during teleoperation:**
- `b` key: Start teleoperation
- `r` key: Reset environment (mark as failed)
- `n` key: Reset environment (mark as successful)

### Dataset Operations

**Replay recorded dataset:**
```bash
python scripts/environments/teleoperation/replay.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --replay_mode=action \
    --dataset_file=./datasets/dataset.hdf5 \
    --select_episodes 1 2
```

**Convert HDF5 to LeRobot format:**
```bash
# Run in LeRobot environment (separate conda env)
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo EverNorif/so101_pick_orange \
    --robot_type so101_follower \
    --fps 30 \
    --hdf5_files ./datasets/dataset1.hdf5 ./datasets/dataset2.hdf5 \
    --task "Pick up the orange and place it on the plate" \
    --push_to_hub
```

**Convert LeRobot to HDF5 format:**
```bash
python scripts/convert/lerobot2isaaclab.py
```

### Policy Inference

**GR00T N1.5 inference:**
```bash
# First, start GR00T server (see Isaac-GR00T repo)
# Then run inference:
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --eval_rounds=10 \
    --policy_type=gr00tn1.5 \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Pick up the orange and place it on the plate" \
    --device=cuda
```

**LeRobot async inference:**
```bash
# First, start LeRobot async server
# Then run inference:
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=localhost \
    --policy_port=8080 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --policy_checkpoint_path=outputs/smolvla/checkpoints/last/pretrained_model \
    --policy_action_horizon=50 \
    --device=cuda \
    --enable_cameras
```

### MimicGen Data Augmentation

**Complete MimicGen workflow:**
```bash
# 1. Convert joint actions to IK-based actions
python scripts/mimic/eef_action_process.py \
    --input_file ./datasets/mimic-lift-cube-example.hdf5 \
    --output_file ./datasets/processed_mimic-lift-cube-example.hdf5 \
    --to_ik --headless

# 2. Annotate subtasks (manual or automatic)
python scripts/mimic/annotate_demos.py \
    --device cuda \
    --task LeIsaac-SO101-LiftCube-Mimic-v0 \
    --input_file ./datasets/processed_mimic-lift-cube-example.hdf5 \
    --output_file ./datasets/annotated_mimic-lift-cube-example.hdf5 \
    --enable_cameras \
    --auto  # For automatic annotation

# 3. Generate augmented data
python scripts/mimic/generate_dataset.py \
    --device cuda \
    --num_envs 1 \
    --generation_num_trials 10 \
    --input_file ./datasets/annotated_mimic-lift-cube-example.hdf5 \
    --output_file ./datasets/generated_mimic-lift-cube-example.hdf5 \
    --enable_cameras

# 4. Convert back to joint actions
python scripts/mimic/eef_action_process.py \
    --input_file ./datasets/generated_mimic-lift-cube-example.hdf5 \
    --output_file ./datasets/final_generated_mimic-lift-cube-example.hdf5 \
    --to_joint --headless
```

## Architecture

### Project Structure

```
source/leisaac/leisaac/
├── assets/          # Robot and scene configurations (USD files, ArticulationCfg)
├── devices/         # Teleoperation device interfaces
│   ├── keyboard/    # Keyboard SE3 control
│   └── lerobot/     # SO101Leader/BiSO101Leader hardware interfaces
├── enhance/         # Extended IsaacLab functionality
│   ├── datasets/    # Streaming HDF5 dataset writer
│   ├── envs/        # DigitalTwin and MimicGen environment types
│   └── managers/    # Custom managers (e.g., StreamingRecorderManager)
├── policy/          # Policy inference clients
│   ├── base.py      # ZMQServicePolicy base class
│   ├── service_policy_clients.py  # GR00T and LeRobot clients
│   └── lerobot/     # gRPC transport for LeRobot async
├── tasks/           # Task environment configurations
│   ├── pick_orange/ # Example task with env_cfg.py
│   ├── lift_cube/
│   ├── assemble_hamburger/
│   └── */mdp/       # MDP components (observations, rewards, events)
└── utils/           # Shared utilities

scripts/
├── environments/    # Teleoperation and environment scripts
├── evaluation/      # Policy inference scripts
├── mimic/           # MimicGen workflow scripts
└── convert/         # Data format conversion scripts
```

### Task Configuration Pattern

All tasks follow the IsaacLab ManagerBasedRLEnv pattern. Each task directory contains:

1. **`__init__.py`**: Registers task with gym using `@configclass` and `gym.register()`
2. **`{task}_env_cfg.py`**: Core environment configuration
   - `SceneCfg`: Scene assets, robot, cameras, lights
   - `ActionsCfg`: Arm and gripper action terms (set via `use_teleop_device()`)
   - `ObservationsCfg`: Policy and subtask observations
   - `EventCfg`: Reset events
   - `TerminationsCfg`: Success criteria and timeouts
   - `{Task}EnvCfg`: Main config class inheriting from `ManagerBasedRLEnvCfg`
3. **`mdp/`**: MDP functions for observations, rewards, terminations, events

**Key methods in task config:**
- `use_teleop_device(teleop_device)`: Initializes action configuration based on device
- `preprocess_device_action(action, teleop_device)`: Converts device input to environment action

### Device Interface Pattern

All teleoperation devices inherit from `Device` base class in `leisaac/devices/device_base.py`:

- `reset()`: Reset device state
- `add_callback(key, func)`: Bind keyboard shortcuts
- `get_device_state()`: Read current device state
- `input2action()`: Convert device input to action dict
- `advance()`: Main loop - returns action tensor or None

Action dict format:
```python
{
    'started': bool,     # Whether teleoperation has started
    'reset': bool,       # Whether to reset environment
    'arm_action': np.ndarray,    # Arm joint positions
    'gripper_action': np.ndarray # Gripper position
}
```

### Data Recording Architecture

**StreamingHDF5DatasetFileHandler** (`enhance/datasets/hdf5_dataset_file_handler.py`):
- Extends IsaacLab's `HDF5DatasetFileHandler` with streaming capability
- Uses single-threaded executor for async writes
- Configurable compression: `None` (fast), `lzf` (balanced), `gzip` (slow, high compression)
- Chunked dataset storage for efficient incremental writes

**StreamingRecorderManager** (`enhance/managers/streaming_recorder_manager.py`):
- Replaces default IsaacLab recorder during teleoperation
- Flushes data periodically (configurable `flush_steps`)
- Tracks successful episode count for stopping condition

**HDF5 Structure:**
```
dataset.hdf5
└── data/
    ├── demo_0/
    │   ├── actions
    │   ├── obs/
    │   │   ├── joint_pos
    │   │   ├── front (images)
    │   │   └── wrist (images)
    │   └── attrs: {num_samples, seed, success}
    └── demo_1/
        └── ...
```

### Policy Inference Architecture

Two service-based policy client implementations:

1. **Gr00tServicePolicyClient** (ZMQ-based):
   - Connects to GR00T N1.5 inference server
   - Converts IsaacLab observations to GR00T format (video.*, state.*)
   - Handles single_arm and gripper modalities
   - Returns action chunks as tensors

2. **LeRobotServicePolicyClient** (gRPC-based):
   - Connects to LeRobot async inference server
   - Uses protobuf for serialization
   - Sends `TimedObservation` with images and joint states
   - Receives action chunks and caches last action

**Coordinate Conversion:**
- `convert_leisaac_action_to_lerobot()`: IsaacLab (radians, [-π, π]) → LeRobot (degrees, normalized ranges)
- `convert_lerobot_action_to_leisaac()`: LeRobot → IsaacLab
- Joint ranges defined in `scripts/convert/isaaclab2lerobot.py`: `ISAACLAB_JOINT_POS_LIMIT_RANGE` and `LEROBOT_JOINT_POS_LIMIT_RANGE`

### Enhanced Environment Types

**DigitalTwin Environment** (`enhance/envs/manager_based_rl_digital_twin_env.py`):
- Extends `ManagerBasedRLEnvCfg` with greenscreen-style background replacement
- Configuration fields:
  - `rgb_overlay_paths`: Dict mapping camera names to background image paths
  - `rgb_overlay_mode`: "none", "debug" (50% blend), or "background" (full overlay)
  - `render_objects`: List of scene entities to render in foreground
- Enables sim2real transfer by overlaying real backgrounds

**MimicGen Environment** (`enhance/envs/manager_based_rl_leisaac_mimic_env.py`):
- Integrates IsaacLab MimicGen for demonstration augmentation
- Workflow: Record demos → Convert to IK actions → Annotate subtasks → Generate augmented data
- See MimicGen commands section for usage

## Important Constraints & Patterns

### Task Naming Convention
- Single-arm tasks: `LeIsaac-SO101-{TaskName}-v0`
- Bi-arm tasks: `LeIsaac-BiArm-{TaskName}-v0` (MUST use `bi-so101leader` device)

### Device Port Permissions
SO101Leader requires USB port access:
```bash
sudo chmod 666 /dev/ttyACM0
# or
sudo usermod -aG dialout $USER
```

### Scene Asset Management
- USD files parsed using `parse_usd_and_create_subassets()` to dynamically create scene entities
- Domain randomization via `domain_randomization()` with `randomize_object_uniform()` and `randomize_camera_uniform()`
- Assets must be downloaded separately from GitHub releases

### Data Format Compatibility
- **HDF5 → LeRobot**: Only successful episodes converted (check `success` attribute)
- **LeRobot version**: Tested with v0.3.3 (rapid development, compatibility not guaranteed with latest)
- **Joint limits**: IsaacLab and LeRobot use different ranges - conversion required
- **Frame skip**: First 5 frames skipped during conversion to LeRobot format

### Policy Inference Requirements
- Service-based policies (GR00T, LeRobot) require external server running first
- Action horizon determines number of actions predicted per inference call
- Language instruction required for VLA-based policies
- Camera data must match training configuration (same keys and resolutions)

### Observation Group Pattern
Task observations organized into groups:
- `PolicyCfg`: Concatenated or separate tensors for policy input (joint_pos, images, etc.)
- `SubtaskCfg`: Task-specific state observations (e.g., object grasped, on plate)
- Set `concatenate_terms=False` for dictionary-style observations (required for image data)

### Episode Length & Termination
- `episode_length_s`: Maximum episode duration (default 8-60s depending on task)
- `decimation`: Action repeat factor (usually 1 for teleoperation)
- During teleoperation, time_out and success terminations are disabled (set to None)
- Success termination re-enabled only when user presses 'n' key

## Version Compatibility

**IsaacSim 4.5 Stack:**
- Python 3.10
- CUDA 11.8
- PyTorch 2.5.1
- IsaacSim 4.5.0
- IsaacLab 2.1.1

**IsaacSim 5.0 Stack (50 series GPU):**
- Python 3.11
- CUDA 12.8
- PyTorch 2.7.0
- IsaacSim 5.0
- IsaacLab 2.2.0
