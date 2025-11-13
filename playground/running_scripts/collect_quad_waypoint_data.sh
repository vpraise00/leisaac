#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get repository root (two levels up from running_scripts)
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to repository root
cd "$REPO_ROOT"

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "ERROR: No conda environment is activated!"
    echo "Please activate the leisaac environment first:"
    echo "  conda activate leisaac"
    exit 1
fi

echo "Quad-Arm Waypoint Data Collection"
echo "=================================="
echo "This script will collect demonstration data by running waypoints repeatedly."
echo ""
echo "Controller Type: DifferentialIK (DIK)"
echo "Working directory: $REPO_ROOT"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

python scripts/environments/waypoints/quad_arm_waypoint_data_collection.py \
    --controller_type=dik \
    --task=LeIsaac-SO101-LiftDesk-QuadArm-v0 \
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
    --rendering_mode=performance \
    --wrist_flex_min=-2.0 \
    --wrist_flex_max=0.5
