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

python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo=vpraise00/leisaac_ma_lift_sticktask_10epi \
    --robot_type=bi_so101_follower \
    --fps=30 \
    --hdf5_files=datasets/waypoint_demos_10epis.hdf5 \
    --task="Lift the stick with both arms" \
    --push_to_hub
