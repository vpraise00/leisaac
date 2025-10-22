@echo off
echo Bi-Arm Waypoint Playback (No Recording)
echo =========================================
echo This script will run waypoints in a loop without recording data.
echo.

python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py ^
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 ^
    --step_hz=30 ^
    --waypoint_file="playground\waypoints\bi_arm_relative_demo.json" ^
    --hold_steps=10 ^
    --position_tol=0.05 ^
    --pose_interp_gain=0.3 ^
    --interp_gain=0.3 ^
    --command_type=position ^
    --enable_cameras

pause
