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

# python scripts/evaluation/policy_inference.py \
#     --task=LeIsaac-SO101-PickOrange-v0 \
#     --eval_rounds=10 \
#     --policy_type=gr00tn1.5 \
#     --policy_host=localhost \
#     --policy_port=5555 \
#     --policy_timeout_ms=5000 \
#     --policy_action_horizon=16 \
#     --policy_language_instruction="Pick up the orange and place it on the plate" \
#     --device=cuda

python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=127.0.0.1 \
    --policy_port=8080 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction="Pick the orange to the plate" \
    --policy_checkpoint_path=vpraise00/lerobot_singletask_10k_iter \
    --policy_action_horizon=50 \
    --device=cuda \
    --enable_cameras

# LeIsaac-SO101-LiftCube-v0

# python scripts/evaluation/policy_inference.py \
#     --task=LeIsaac-SO101-LiftCube-v0 \
#     --policy_type=lerobot-smolvla \
#     --policy_host=127.0.0.1 \
#     --policy_port=8080 \
#     --policy_timeout_ms=5000 \
#     --policy_language_instruction="Lift the cube" \
#     --policy_checkpoint_path=vpraise00/le_svla \
#     --policy_action_horizon=50 \
#     --device=cuda \
#     --enable_cameras
