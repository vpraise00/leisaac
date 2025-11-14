#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get repository root (two levels up from running_scripts)
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to repository root
cd "$REPO_ROOT"

# Ensure repo sources are importable without installation
SOURCE_PATH="$REPO_ROOT/source"
if [ -d "$SOURCE_PATH" ]; then
    export PYTHONPATH="$SOURCE_PATH:$PYTHONPATH"
fi

# Resolve a Python interpreter (supporting docker + conda flows)
POSSIBLE_PYTHONS=()
if [ -n "$PYTHON_BIN" ]; then
    POSSIBLE_PYTHONS+=("$PYTHON_BIN")
else
    POSSIBLE_PYTHONS+=("/workspace/isaaclab/_isaac_sim/python.sh" "python" "python3")
fi

PYTHON_BIN=""
for candidate in "${POSSIBLE_PYTHONS[@]}"; do
    if [ -x "$candidate" ]; then
        PYTHON_BIN="$candidate"
        break
    fi
    RESOLVED_PATH="$(command -v "$candidate" 2>/dev/null || true)"
    if [ -n "$RESOLVED_PATH" ] && [ -x "$RESOLVED_PATH" ]; then
        PYTHON_BIN="$RESOLVED_PATH"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No Python interpreter is available!"
    echo "Please ensure you are in the correct environment:"
    echo "  - For conda: conda activate leisaac"
    echo "  - For docker: source the Isaac Sim environment (python.sh)"
    exit 1
fi

echo "Quad-Arm Waypoint Data Collection (Headless + WebRTC)"
echo "======================================================"
echo "This script will collect demonstration data by running waypoints repeatedly."
echo ""
echo "Controller Type: DifferentialIK (DIK)"
echo "Working directory: $REPO_ROOT"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda environment: $CONDA_DEFAULT_ENV"
else
    echo "Python interpreter: $PYTHON_BIN"
fi
echo ""

# "$PYTHON_BIN" scripts/environments/waypoints/quad_arm_waypoint_data_collection.py \
#     --controller_type=dik \
#     --task=LeIsaac-SO101-LiftDesk-QuadArm-v0 \
#     --step_hz=30 \
#     --waypoint_file="playground/waypoints/quad_arm_demo.json" \
#     --record \
#     --dataset_file="datasets/quad_waypoint_demos_50epis_nograv.hdf5" \
#     --num_demos=50 \
#     --position_tol=0.09 \
#     --pose_interp_gain=0.25 \
#     --interp_gain=0.25 \
#     --command_type=position \
#     --episode_timeout=90 \
#     --enable_cameras \
#     --rendering_mode=performance \
#     --wrist_flex_min=-2.0 \
#     --wrist_flex_max=0.5 \
#     --headless \
#     --livestream 2

"$PYTHON_BIN" scripts/environments/waypoints/quad_arm_waypoint_data_collection.py \
    --controller_type=dik \
    --task=LeIsaac-QuadArm-CollaborateDemo-v0 \
    --step_hz=30 \
    --waypoint_file="playground/waypoints/quad_arm_demo.json" \
    --record \
    --dataset_file="datasets/quad_waypoint_demos_50epis_nograv.hdf5" \
    --num_demos=50 \
    --position_tol=0.09 \
    --pose_interp_gain=0.25 \
    --interp_gain=0.25 \
    --command_type=position \
    --episode_timeout=90 \
    --enable_cameras \
    --wrist_flex_min=-2.0 \
    --wrist_flex_max=0.5 \
    --headless \
    --livestream 2
