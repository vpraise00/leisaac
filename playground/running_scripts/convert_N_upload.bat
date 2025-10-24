@echo off

python scripts/convert/isaaclab2lerobot.py ^
    --dataset_path_or_repo=vpraise00/leisaac_ma_waypoint_test_3epi ^
    --robot_type=bi_so101_follower ^
    --fps=30 ^
    --hdf5_files=datasets/waypoint_demos_test.hdf5 ^
    --task="Bi-arm waypoint demonstration" ^
    --push_to_hub