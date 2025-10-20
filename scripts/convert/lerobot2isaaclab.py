# ===== Example usage ======
"""
python convert_lerobot_to_hdf5.py \
  --dataset_path_or_repo path_or_repo \
  --out_hdf5_path ./exported.hdf5 \
  --robot_type so101_follower
"""


import os
import h5py
import numpy as np
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ====== IsaacLab <-> LeRobot 스케일/단위 정의 (네 기존 스크립트의 역변환) ======
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10.0, 100.0),
]
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (0.0, 100.0),
]

def inverse_preprocess_joint_pos(joint_pos_deg_like: np.ndarray) -> np.ndarray:
    """
    LeRobot 범위(각 관절별 min/max)로 스케일된 '각도 단위' 값을
    IsaacLab 범위로 역스케일한 뒤 라디안으로 변환.
    joint_pos_deg_like: shape (T, 6) or (T, 12)  [단일팔/양팔]
    """
    # 팔이 하나면 6, 양팔이면 12. 6개 블록 단위로 처리.
    n = joint_pos_deg_like.shape[1]
    assert n in (6, 12), f"Expected 6 or 12 DoF, got {n}"

    out = joint_pos_deg_like.copy().astype(np.float32).reshape(-1, n)
    blocks = 2 if n == 12 else 1

    for b in range(blocks):
        for i in range(6):
            idx = b * 6 + i
            ler_min, ler_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
            isa_min, isa_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
            # LeRobotRange -> IsaacRange
            out[:, idx] = (out[:, idx] - ler_min) / (ler_max - ler_min) * (isa_max - isa_min) + isa_min
            # deg -> rad
            out[:, idx] = out[:, idx] * np.pi / 180.0
    return out

# ====== 유틸: 에피소드로 묶기 ======
def group_indices_by_episode(dataset: LeRobotDataset):
    """
    LeRobotDataset의 프레임들을 episode_index 기준으로 묶어서
    {episode_idx: [frame_indices...]} 딕셔너리 반환.
    v0.3.3에서 dataset[i]에 'episode_index'가 없으면,
    'next.done'을 이용해 끊어주는 방식으로 대체.
    """
    ep_to_frames = {}
    cur_ep, cur_list = None, []

    # 먼저 episode_index가 존재하는지 한 프레임 체크
    sample = dataset[0]
    has_episode_index = "episode_index" in sample

    if has_episode_index:
        # 전수 스캔(필요시 tqdm)
        for i in range(len(dataset)):
            item = dataset[i]
            epi = int(item["episode_index"])
            ep_to_frames.setdefault(epi, []).append(i)
    else:
        # next.done으로 에피소드 경계를 추정
        epi = 0
        cur_list = []
        for i in range(len(dataset)):
            item = dataset[i]
            cur_list.append(i)
            done = bool(item.get("next.done", False))
            if done:
                ep_to_frames[epi] = cur_list
                epi += 1
                cur_list = []
        # 마지막 남은 프레임 flush
        if cur_list:
            ep_to_frames[epi] = cur_list
    return ep_to_frames

# ====== 이미지 키 자동 탐색(선택) ======
def guess_image_keys(sample: dict):
    """
    LeRobotDataset 한 프레임(dict)에서 이미지 키를 추정.
    보통 'observation.images.' prefix를 가짐.
    """
    image_keys = []
    for k, v in sample.items():
        if k.startswith("observation.images.") and isinstance(v, np.ndarray):
            # 기대: HxWxC uint8
            if v.ndim == 3 and v.shape[-1] in (1, 3, 4):
                image_keys.append(k)
    return image_keys

# ====== 메인 변환 ======
def convert_lerobot_to_hdf5(
    dataset_path_or_repo: str,
    out_hdf5_path: str = "./exported_dataset.hdf5",
    robot_type: str = "so101_follower",   # or "bi_so101_follower"
    map_images: dict | None = None,       # 예: {"observation.images.front": "front", "observation.images.wrist":"wrist"}
    map_state_key: str = "observation.state",
    map_action_key: str = "action",
    set_success_attr: bool | None = None, # None이면 추정하지 않음. True/False로 강제 가능.
):
    """
    LeRobotDataset → HDF5 변환.
    - robot_type에 따라 state/action 차원(6/12) 관리
    - map_images로 HDF5 내 obs/<name> 키 이름 지정
    """
    os.makedirs(os.path.dirname(out_hdf5_path) or ".", exist_ok=True)
    ds = LeRobotDataset(dataset_path_or_repo)
    print(f"Loaded LeRobotDataset from: {dataset_path_or_repo}")
    print(f"Total frames: {len(ds)}")

    # 이미지 키 자동 추정(명시 미제공 시)
    sample = ds[0]
    if map_images is None:
        image_keys = guess_image_keys(sample)  # ex) ["observation.images.front", "observation.images.wrist"]
        # 기본 매핑 추정 규칙
        default_map = {}
        for k in image_keys:
            short = k.split("observation.images.", 1)[-1]
            default_map[k] = short
        map_images = default_map
        print("Auto-detected image keys:", map_images)

    # 에피소드별 프레임 인덱스 그룹핑
    ep_to_frames = group_indices_by_episode(ds)
    print(f"Found {len(ep_to_frames)} episodes")

    # HDF5 생성
    with h5py.File(out_hdf5_path, "w") as f:
        g_data = f.create_group("data")

        for epi, frame_indices in tqdm(sorted(ep_to_frames.items()), desc="Episodes"):
            demo_name = f"demo_{epi:05d}"
            g_demo = g_data.create_group(demo_name)

            # 프레임 수 T
            T = len(frame_indices)

            # ----- 액션/스테이트 모으기 -----
            actions_list = []
            state_list = []
            images_buf = {dst: [] for dst in map_images.values()}

            for idx in frame_indices:
                item = ds[idx]

                # action/state (numpy array 보장 가정)
                act = np.asarray(item[map_action_key])
                st  = np.asarray(item[map_state_key])

                actions_list.append(act)
                state_list.append(st)

                # 이미지들
                for src, dst in map_images.items():
                    frame_img = item[src]  # 기대: HxWxC uint8
                    if not isinstance(frame_img, np.ndarray):
                        # 일부 버전/설정에선 PIL.Image로 올 수도 있음 → np로 변환
                        from PIL import Image
                        if isinstance(frame_img, Image.Image):
                            frame_img = np.array(frame_img)
                        else:
                            raise TypeError(f"Unsupported image type for key {src}: {type(frame_img)}")
                    images_buf[dst].append(frame_img)

            actions = np.stack(actions_list, axis=0)
            states  = np.stack(state_list, axis=0)

            # ===== 역스케일 & 단위변환 =====
            # - 네 기존 파이프라인이 Isaac(rad)→deg→range-scale→LeRobot 이었으므로
            # - 여기서는 LeRobot→range-unscale→deg→rad
            if robot_type == "so101_follower":
                actions = inverse_preprocess_joint_pos(actions)        # (T,6) deg-like → rad
                states  = inverse_preprocess_joint_pos(states)         # (T,6) deg-like → rad
            elif robot_type == "bi_so101_follower":
                actions = inverse_preprocess_joint_pos(actions)        # (T,12) rad로
                states  = inverse_preprocess_joint_pos(states)         # (T,12) rad로
            else:
                raise ValueError("robot_type must be 'so101_follower' or 'bi_so101_follower'")

            # ----- HDF5에 쓰기 -----
            # 액션/스테이트
            g_demo.create_dataset("actions", data=actions, compression="gzip")
            if robot_type == "so101_follower":
                g_demo.create_dataset("obs/joint_pos", data=states, compression="gzip")
            else:
                # 양팔일 때는 좌/우로 쪼개 저장 (너의 HDF5 관례에 맞춰 조정)
                left = states[:, :6]
                right = states[:, 6:]
                g_demo.create_dataset("obs/left_joint_pos", data=left, compression="gzip")
                g_demo.create_dataset("obs/right_joint_pos", data=right, compression="gzip")

            # 이미지들
            for dst_key, frames in images_buf.items():
                arr = np.stack(frames, axis=0)  # (T,H,W,C)
                # 보통 uint8 권장
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                g_demo.create_dataset(f"obs/{dst_key}", data=arr, compression="gzip")

            # 성공 여부 attribute(선택)
            if set_success_attr is not None:
                g_demo.attrs["success"] = bool(set_success_attr)

    print(f"Saved HDF5 to: {out_hdf5_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert LeRobotDataset to HDF5 for IsaacLab")
    parser.add_argument("dataset_path_or_repo", type=str, help="Path or repo ID of the LeRobotDataset")
    parser.add_argument("--out_hdf5_path", type=str, default="./exported_dataset.hdf5", help="Output HDF5 file path")
    parser.add_argument("--robot_type", type=str, choices=["so101_follower", "bi_so101_follower"], default="so101_follower", help="Robot type")
    parser.add_argument("--map_state_key", type=str, default="observation.state", help="Key for state in dataset")
    parser.add_argument("--map_action_key", type=str, default="action", help="Key for action in dataset")
    parser.add_argument("--set_success_attr", type=lambda x: (str(x).lower() == 'true'), default=None, help="Set success attribute (True/False)")

    args = parser.parse_args()

    convert_lerobot_to_hdf5(
        dataset_path_or_repo=args.dataset_path_or_repo,
        out_hdf5_path=args.out_hdf5_path,
        robot_type=args.robot_type,
        map_state_key=args.map_state_key,
        map_action_key=args.map_action_key,
        set_success_attr=args.set_success_attr,
    )


