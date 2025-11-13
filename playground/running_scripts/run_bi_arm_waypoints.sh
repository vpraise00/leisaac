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

echo "Working directory: $REPO_ROOT"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo ""

python scripts/environments/waypoints/bi_arm_waypoint_runner.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --device=cuda \
    --step_hz=30 \
    --waypoint_file="playground/waypoints/bi_arm_relative_demo.json" \
    --hold_steps=10 \
    --enable_cameras \
    --stay_alive \
    --logging_pos \
    --pose_interp_gain=0.3 \
    --interp_gain=0.3 \
    --position_tol=0.1

# left : (0.22 -0.45 0.22) (0.22 -0.45 0.11) (0.22 -0.45 0.22)
# right : (0.55 -0.45 0.22) (0.55 -0.45 0.11) (0.55 -0.45 0.22)
