# 양팔 로봇 Waypoint 제어

이 디렉토리는 양팔 로봇에서 사전 정의된 waypoint 시퀀스를 실행하고, 데모 수집을 위한 데이터 기록 기능을 제공하는 도구를 포함합니다.

## 개요

Waypoint 시스템의 주요 기능:
- 양팔의 목표 포즈(위치 + 선택적 방향) 시퀀스 정의
- Differential IK 제어를 이용한 자동 waypoint 실행
- 학습용 HDF5 형식 데모 기록
- 테스트/디버깅을 위한 플레이백 모드 실행

## 아키텍처

### 핵심 구성 요소

1. **`waypoint_controller.py`** - 재사용 가능한 waypoint 실행 모듈
   - `BiArmWaypointController`: 메인 컨트롤러 클래스
   - `load_waypoints_from_json()`: Waypoint 파일 파서
   - IK 계산, 수렴 확인, 액션 생성 처리

2. **`bi_arm_waypoint_data_collection.py`** - 메인 실행 스크립트
   - 두 가지 모드 지원: 플레이백(기록 없음)과 데이터 수집(기록 포함)
   - 자동 에피소드 관리
   - 명령줄 인자를 통한 설정

3. **Waypoint JSON 파일** - Waypoint 시퀀스 정의
   - 위치: `playground/waypoints/*.json`
   - 형식: Waypoint 객체의 배열

## Waypoint JSON 형식

```json
[
  {
    "relative": false,
    "left": {
      "position": [x, y, z],
      "orientation": [w, x, y, z],  // 선택사항: quaternion (월드 프레임)
      "gripper": 0.0,                // 0.0 = 열림, 1.0 = 닫힘
      "wrist_flex": 1.57             // 선택사항: wrist_flex 각도 (라디안)
    },
    "right": {
      "position": [x, y, z],
      "orientation": [w, x, y, z],  // 선택사항
      "gripper": 0.0,
      "wrist_flex": 1.57             // 선택사항
    },
    "hold_steps": 45                 // 이 waypoint에서 유지할 스텝 수
  },
  // ... 추가 waypoint
]
```

**참고:**
- `position`: [x, y, z] 미터 단위 (월드 프레임)
- `orientation`: [w, x, y, z] quaternion (월드 프레임) - pose 모드에서만 사용
- `gripper`: 0.0 (열림) ~ 1.0 (닫힘)
- `wrist_flex`: 조인트 각도 라디안 단위 (선택사항) - 지정 시 전역 `--wrist_flex_angle` 값을 오버라이드
- `hold_steps`: 다음 waypoint로 이동하기 전 현재 지점에서 유지할 시뮬레이션 스텝 수
- `orientation`이 생략되면 position-only 모드 사용 (로봇이 자연스러운 방향 선택)

## 사용법

### 모드 1: 플레이백 전용 (기록 없음)

데이터를 저장하지 않고 waypoint를 루프로 실행합니다. 테스트 및 시각화에 유용합니다.

**명령어:**
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_relative_demo.json" \
    --step_hz=30 \
    --hold_steps=10 \
    --position_tol=0.05 \
    --command_type=position \
    --enable_cameras
```

**또는 배치 파일 사용:**
```bash
playground\running_scripts\run_waypoints_playback.bat
```

**동작:**
- 모든 waypoint를 순차적으로 실행
- 마지막 waypoint 후 2초 대기
- 리셋 후 무한 루프
- 데이터 기록 없음

### 모드 2: 데이터 수집 (기록)

Waypoint를 여러 번 실행하여 데모를 기록합니다.

**명령어:**
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_relative_demo.json" \
    --record \
    --dataset_file="datasets/waypoint_demos.hdf5" \
    --num_demos=10 \
    --step_hz=30 \
    --hold_steps=10 \
    --position_tol=0.05 \
    --command_type=position \
    --enable_cameras
```

**또는 배치 파일 사용:**
```bash
playground\running_scripts\collect_waypoint_data.bat
```

**동작:**
- Waypoint 시퀀스 실행
- 마지막 waypoint 후 2초 대기
- 에피소드를 성공으로 표시하고 HDF5에 저장
- 리셋 후 다음 에피소드 시작
- `num_demos` 개수만큼 수집 후 종료

## 주요 파라미터

### 필수
- `--task`: 태스크 환경 이름 (예: `LeIsaac-SO101-CleanToyTable-BiArm-v0`)
- `--waypoint_file`: Waypoint JSON 파일 경로

### Waypoint 제어
- `--command_type`: `position` (위치만) 또는 `pose` (위치+방향)
  - `position`: 엔드 이펙터 위치만 제어, 방향은 자연스럽게 따라감
  - `pose`: 위치와 방향 모두 제어 (JSON에 `orientation` 필요)
- `--hold_steps`: 모든 waypoint의 hold_steps 오버라이드 (선택사항)
- `--position_tol`: 위치 수렴 허용 오차 미터 단위 (기본값: 0.05)
- `--orientation_tol`: 방향 수렴 허용 오차 (기본값: 0.02)
- `--pose_interp_gain`: Pose 명령 보간 게인 (0-1, 기본값: 0.3)
- `--interp_gain`: 조인트 타겟 보간 게인 (0-1, 기본값: 0.3)
- `--episode_end_delay`: 마지막 waypoint 후 지연 시간(초) (기본값: 2.0)
- `--force_wrist_down`: wrist_flex 조인트를 아래 방향으로 강제 설정 (기본값: 활성화)
- `--wrist_flex_angle`: wrist_flex 조인트 목표 각도 라디안 단위 (기본값: 1.57 ≈ 90°)

### 기록 (`--record` 사용 시에만)
- `--record`: 데이터 기록 활성화 (플래그)
- `--dataset_file`: 출력 HDF5 파일 경로
- `--num_demos`: 수집할 데모 개수

### 시뮬레이션
- `--step_hz`: 시뮬레이션 주파수 Hz 단위 (기본값: 30)
- `--enable_cameras`: 카메라 렌더링 활성화 (비전 데이터용)
- `--seed`: 환경 랜덤 시드

## 예제

### 예제 1: Waypoint 테스트 (플레이백)
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/my_waypoints.json" \
    --step_hz=30 \
    --position_tol=0.05
```

### 예제 2: 50개 데모 수집
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_relative_demo.json" \
    --record \
    --dataset_file="datasets/clean_table_demos.hdf5" \
    --num_demos=50 \
    --enable_cameras
```

### 예제 3: 방향 제어를 포함한 Pose 모드
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/oriented_waypoints.json" \
    --command_type=pose \
    --orientation_tol=0.02 \
    --record \
    --dataset_file="datasets/oriented_demos.hdf5" \
    --num_demos=10
```

## Waypoint 파일 생성

### 1단계: 목표 위치 결정

원격 조작 또는 수동 제어를 사용하여 원하는 엔드 이펙터 위치를 찾습니다:

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --teleop_device=keyboard \
    --num_envs=1
```

터미널에 표시되는 위치를 메모합니다.

### 2단계: JSON 파일 생성

Waypoint를 포함한 JSON 파일을 생성합니다:

```json
[
  {
    "relative": false,
    "left": {
      "position": [0.22, -0.45, 0.22],
      "gripper": 0.0
    },
    "right": {
      "position": [0.55, -0.45, 0.22],
      "gripper": 0.0
    },
    "hold_steps": 45
  },
  {
    "relative": false,
    "left": {
      "position": [0.22, -0.45, 0.11],
      "gripper": 0.6
    },
    "right": {
      "position": [0.55, -0.45, 0.11],
      "gripper": 0.6
    },
    "hold_steps": 60
  }
]
```

### 3단계: 플레이백 테스트

```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/my_waypoints.json"
```

## 출력 데이터 형식

`--record` 사용 시 데이터는 LeRobot 호환 HDF5 형식으로 저장됩니다:

```
dataset.hdf5
└── data/
    ├── demo_0/
    │   ├── actions          # (N, 12) - 조인트 위치 [왼팔(5), 왼쪽그리퍼(1), 오른팔(5), 오른쪽그리퍼(1)]
    │   ├── obs/
    │   │   ├── joint_pos    # (N, 12) - 현재 조인트 위치
    │   │   ├── front        # (N, H, W, 3) - 전면 카메라 이미지 (활성화 시)
    │   │   └── wrist        # (N, H, W, 3) - 손목 카메라 이미지 (활성화 시)
    │   └── attrs: {num_samples, seed, success}
    └── demo_1/
        └── ...
```

### LeRobot 형식으로 변환 및 HuggingFace 업로드

HDF5 형식으로 데모를 수집한 후, LeRobot의 Parquet 형식으로 변환하고 학습을 위해 HuggingFace Hub에 업로드할 수 있습니다.

#### 사전 준비

변환 스크립트는 LeRobot 라이브러리(v0.3.3)가 필요합니다. 별도의 conda 환경에 설치하세요:

```bash
# LeRobot 환경 생성
conda create -n lerobot python=3.10
conda activate lerobot

# LeRobot v0.3.3 설치
pip install lerobot==0.3.3

# 추가 종속성 설치
pip install datasets Pillow opencv-python tqdm
```

**중요:** 변환 스크립트는 `leisaac` 환경이 아닌 `lerobot` 환경에서 실행해야 합니다.

#### 변환 명령어

```bash
# LeRobot 환경 활성화
conda activate lerobot

# 변환 실행 (양팔 예제)
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo your-username/clean-toy-table \
    --robot_type bi_so101_follower \
    --fps 30 \
    --hdf5_files datasets/waypoint_demos.hdf5 \
    --task "양팔 로봇으로 테이블에서 장난감 정리하기" \
    --push_to_hub
```

#### 주요 파라미터

- `--dataset_path_or_repo`: HuggingFace 저장소 이름 (형식: `username/dataset-name`)
- `--robot_type`: 로봇 구성 타입
  - `so101_follower`: 단일팔 SO-101 로봇 (6 DOF: 팔 조인트 5개 + 그리퍼 1개)
  - `bi_so101_follower`: 양팔 SO-101 로봇 (12 DOF: 2팔 × 각 6 DOF)
- `--fps`: 초당 프레임 수 (데이터 수집 시 `--step_hz`와 일치해야 함)
- `--hdf5_files`: 변환할 HDF5 파일 목록(공백으로 구분, 여러 파일 병합 가능)
- `--task`: 태스크 설명 (데이터셋 메타데이터에 사용)
- `--push_to_hub`: HuggingFace Hub에 업로드 (인증 필요)

#### 인증 설정

HuggingFace Hub에 푸시하기 전에 인증하세요:

```bash
# HuggingFace 로그인
huggingface-cli login

# 또는 토큰을 환경 변수로 설정
export HF_TOKEN=your_token_here
```

#### 예제

**예제 1: 단일팔 데이터셋**
```bash
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo myusername/pick-orange-demos \
    --robot_type so101_follower \
    --fps 30 \
    --hdf5_files datasets/orange_demos_1.hdf5 datasets/orange_demos_2.hdf5 \
    --task "오렌지를 집어 접시에 놓기" \
    --push_to_hub
```

**예제 2: 양팔 데이터셋 (waypoint 기반)**
```bash
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo myusername/clean-table-waypoint \
    --robot_type bi_so101_follower \
    --fps 30 \
    --hdf5_files datasets/waypoint_demos.hdf5 \
    --task "Waypoint 기반 양팔 제어로 장난감 테이블 정리" \
    --push_to_hub
```

**예제 3: 로컬 변환만 (업로드 없음)**
```bash
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo local-test-dataset \
    --robot_type bi_so101_follower \
    --fps 30 \
    --hdf5_files datasets/waypoint_demos.hdf5 \
    --task "테스트 데이터셋"
# 로컬에만 저장하려면 --push_to_hub 생략
```

#### 변환 과정

스크립트는 다음 단계를 수행합니다:

1. **HDF5 데이터 로드**: 입력 HDF5 파일에서 모든 성공한 에피소드 읽기
2. **초기 프레임 건너뛰기**: 각 에피소드의 처음 5프레임 제거 (안정화 기간)
3. **액션 변환**: 조인트 위치를 IsaacLab 형식(라디안, [-π, π])에서 LeRobot 형식(도, 정규화된 범위)으로 변환
4. **이미지 처리**: 카메라 이미지를 비디오 파일로 압축 (데이터 수집 시 `--enable_cameras` 사용한 경우)
5. **Parquet 파일 생성**: 효율적인 로딩을 위해 LeRobot의 컬럼 형식으로 데이터 저장
6. **메타데이터 생성**: 데이터셋 통계 및 설정이 포함된 `meta_data/info.json` 생성
7. **Hub 업로드**: HuggingFace 저장소에 데이터셋 푸시 (`--push_to_hub` 지정 시)

#### 출력 구조

변환된 데이터셋의 구조:

```
local_data/your-username/dataset-name/
├── meta_data/
│   └── info.json              # 데이터셋 메타데이터 (fps, robot_type, 태스크 설명)
├── videos/
│   ├── chunk-000/
│   │   ├── observation.images.front_000000.mp4
│   │   ├── observation.images.wrist_000000.mp4
│   │   └── ...
│   └── chunk-001/
│       └── ...
└── data/
    ├── chunk-000/
    │   └── train-00000-of-00001.parquet
    └── chunk-001/
        └── train-00001-of-00002.parquet
```

#### 변환 확인

변환 후 데이터셋을 확인하세요:

```bash
# lerobot 환경에서
python -c "
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('your-username/dataset-name')
print(f'전체 프레임: {len(dataset)}')
print(f'특성: {dataset.features}')
"
```

#### 변환 문제 해결

**문제: "Dataset not found" 오류**
- 해결: `--push_to_hub` 사용 및 HuggingFace 토큰 설정 확인
- 또는 `local_data/` 디렉토리에서 로컬 파일 확인

**문제: "Robot type mismatch" 오류**
- 해결: 양팔 태스크에는 `bi_so101_follower`, 단일팔에는 `so101_follower` 사용
- HDF5 파일이 12개 조인트(양팔) 또는 6개 조인트(단일팔)를 가지고 있는지 확인

**문제: "FPS mismatch" 경고**
- 해결: `--fps`가 데이터 수집 시 `--step_hz`와 일치하는지 확인
- 일반적인 값: 30 Hz (기본값), 20 Hz (느린 하드웨어)

**문제: 비디오 인코딩 실패**
- 해결: ffmpeg 설치: `sudo apt-get install ffmpeg` (Linux) 또는 `brew install ffmpeg` (macOS)
- 또는 스크립트를 수정하여 비디오 변환 비활성화

## 문제 해결

### Waypoint가 수렴하지 않음

**증상:** 로봇이 진동하거나 hold 카운터가 계속 리셋됨

**해결책:**
- `--position_tol` 증가 (예: `0.05` → `0.10`)
- 더 부드러운 동작을 위해 `--pose_interp_gain` 감소 (예: `0.3` → `0.2`)
- 더 안정적인 IK를 위해 `--interp_gain` 감소 (예: `0.3` → `0.2`)
- Waypoint 위치가 도달 가능한지 확인 (충돌 없음, 작업 공간 내)

### Pose 모드에서 로봇이 뒤집히거나 불안정함

**증상:** `--command_type=pose` 사용 시 로봇이 갑작스럽게 큰 움직임을 보임

**해결책:**
- `--command_type=position` 모드로 전환
- JSON의 quaternion이 정규화되었는지 확인
- 실제 로봇 포즈의 quaternion 값 사용 (원격 조작 로그 참조)
- `--orientation_tol` 증가 (예: `0.02` → `0.05`)

### 기록이 시작되지 않음

**증상:** `--record` 플래그가 데이터를 저장하지 않음

**해결책:**
- `--record` 플래그가 존재하는지 확인
- `--dataset_file` 경로가 유효하고 쓰기 가능한지 확인
- `--num_demos`가 0보다 큰지 확인
- 터미널 출력에서 StreamingRecorderManager 메시지 확인

### 낮은 프레임 레이트

**증상:** 시뮬레이션이 느리게 실행됨

**해결책:**
- `--step_hz` 감소 (예: `30` → `20`)
- 필요하지 않으면 카메라 비활성화 (`--enable_cameras` 제거)
- 환경 수 감소 (waypoint의 경우 이미 1이어야 함)

## 팁 및 모범 사례

1. **플레이백 모드로 시작**: 항상 `--record` 없이 먼저 waypoint를 테스트하여 올바르게 작동하는지 확인

2. **Position 모드 사용**: 방향 제어가 특별히 필요하지 않은 한 더 나은 안정성을 위해 `--command_type=position` 사용

3. **허용 오차 조정**: 태스크 요구사항에 따라 `--position_tol` 조정:
   - 정밀한 조작: `0.01` - `0.02` m
   - 일반 이동: `0.05` - `0.10` m

4. **Hold Steps**: 다음의 경우 더 긴 hold 시간 사용:
   - 파지 작업 (60+ steps)
   - 최종 배치 (60+ steps)
   - Waypoint는 더 짧은 hold 사용 가능 (10-30 steps)

5. **에피소드 종료 지연**: 다음을 보장하도록 `--episode_end_delay` 조정:
   - 조작 후 물체가 안정화됨
   - 마지막 액션이 완전히 실행됨
   - 일반적인 범위: 1-3초

6. **카메라 렌더링**: 비전 기반 정책에 필요한 경우에만 `--enable_cameras` 사용:
   - 계산 오버헤드 추가
   - 데이터셋 크기 크게 증가

7. **데이터 수집**: 강건한 데이터셋을 위해:
   - 태스크당 최소 10-50개 데모 수집
   - 가능한 경우 태스크 설정에서 도메인 랜덤화 사용
   - HDF5 파일 재생으로 데이터 품질 검증

## 파일 구조

```
scripts/environments/waypoints/
├── README.md                              # 영문 문서
├── README_KOR.md                          # 본 문서
├── waypoint_controller.py                 # 핵심 waypoint 실행 모듈
└── bi_arm_waypoint_data_collection.py    # 메인 실행 스크립트

playground/waypoints/
├── bi_arm_relative_demo.json             # 예제 waypoint 파일
└── *.json                                # 사용자 정의 waypoint 파일

playground/running_scripts/
├── run_waypoints_playback.bat            # 플레이백 모드 (기록 없음)
└── collect_waypoint_data.bat             # 데이터 수집 모드 (기록 포함)
```

## 관련 문서

- [LeIsaac 프로젝트 README](../../../README.md)
- [원격 조작 가이드](../teleoperation/)
- [데이터 변환 가이드](../../convert/)
- [IsaacLab Controllers](https://isaac-sim.github.io/IsaacLab/source/api/isaaclab/controllers.html)
