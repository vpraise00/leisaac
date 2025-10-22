@echo off

python scripts/environments/waypoints/bi_arm_waypoint_runner.py ^
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 ^
    --device=cuda ^
    --step_hz=30 ^
    --waypoint_file="playground\waypoints\bi_arm_relative_demo.json" ^
    --hold_steps=10 ^
    --enable_cameras ^
    --stay_alive ^
    --logging_pos ^
    --pose_interp_gain=0.3 ^
    --interp_gain=0.3 ^
    --position_tol=0.1

@REM left : (0.22 -0.45 0.22) (0.22 -0.45 0.11) (0.22 -0.45 0.22)
@REM right : (0.55 -0.45 0.22) (0.55 -0.45 0.11) (0.55 -0.45 0.22)