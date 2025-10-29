@echo off
echo Quad-Arm Waypoint Data Collection
echo ==================================
echo This script will collect demonstration data by running waypoints repeatedly.
echo.
echo Controller Type: DifferentialIK (DIK)
echo.

python scripts/environments/waypoints/quad_arm_waypoint_data_collection.py ^
    --controller_type=dik ^
    --task=LeIsaac-SO101-LiftDesk-QuadArm-v0 ^
    --step_hz=30 ^
    --waypoint_file="playground\waypoints\quad_arm_demo.json" ^
    --record ^
    --dataset_file="datasets\quad_waypoint_demos_50epis.hdf5" ^
    --num_demos=50 ^
    --position_tol=0.09 ^
    --pose_interp_gain=0.25 ^
    --interp_gain=0.25 ^
    --command_type=position ^
    --episode_timeout=90 ^
    --enable_cameras ^
    --rendering_mode=performance

pause
