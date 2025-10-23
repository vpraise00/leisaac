@echo off

@REM python scripts/evaluation/policy_inference.py ^
@REM     --task=LeIsaac-SO101-PickOrange-v0 ^
@REM     --eval_rounds=10 ^
@REM     --policy_type=gr00tn1.5 ^
@REM     --policy_host=localhost ^
@REM     --policy_port=5555 ^
@REM     --policy_timeout_ms=5000 ^
@REM     --policy_action_horizon=16 ^
@REM     --policy_language_instruction="Pick up the orange and place it on the plate" ^
@REM     --device=cuda

python scripts/evaluation/policy_inference.py ^
    --task=LeIsaac-SO101-PickOrange-v0 ^
    --policy_type=lerobot-smolvla ^
    --policy_host=127.0.0.1 ^
    --policy_port=8080 ^
    --policy_timeout_ms=5000 ^
    --policy_language_instruction="Pick the orange to the plate" ^
    --policy_checkpoint_path=vpraise00/lerobot_singletask_10k_iter ^
    --policy_action_horizon=50 ^
    --device=cuda ^
    --enable_cameras

@REM LeIsaac-SO101-LiftCube-v0

@REM python scripts/evaluation/policy_inference.py ^
@REM     --task=LeIsaac-SO101-LiftCube-v0 ^
@REM     --policy_type=lerobot-smolvla ^
@REM     --policy_host=127.0.0.1 ^
@REM     --policy_port=8080 ^
@REM     --policy_timeout_ms=5000 ^
@REM     --policy_language_instruction="Lift the cube" ^
@REM     --policy_checkpoint_path=vpraise00/le_svla ^
@REM     --policy_action_horizon=50 ^
@REM     --device=cuda ^
@REM     --enable_cameras