@echo off

python scripts/convert/isaaclab2lerobot.py ^
    --dataset_path_or_repo=vpraise00/leisaac_ma_lift_sticktask_10epi ^
    --robot_type=bi_so101_follower ^
    --fps=30 ^
    --hdf5_files=datasets/waypoint_demos_10epis.hdf5 ^
    --task="Lift the stick with both arms" ^
    --push_to_hub