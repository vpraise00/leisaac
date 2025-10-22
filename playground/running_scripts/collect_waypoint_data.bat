@echo off
echo Bi-Arm Waypoint Data Collection
echo ================================
echo This script will collect demonstration data by running waypoints repeatedly.
echo.

python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py ^
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 ^
    --step_hz=30 ^
    --waypoint_file="playground\waypoints\bi_arm_relative_demo.json" ^
    --record ^
    --dataset_file="datasets\waypoint_demos.hdf5" ^
    --num_demos=3 ^
    --hold_steps=10 ^
    --position_tol=0.05 ^
    --pose_interp_gain=0.3 ^
    --interp_gain=0.3 ^
    --command_type=position ^
    --force_wrist_down ^
    --wrist_flex_angle=1.57 ^
    --enable_cameras

pause
