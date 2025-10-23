# Waypoint 기반 데이터 수집

이 디렉토리는 IsaacLab에서 양팔 SO-101 로봇 조작 태스크를 위한 waypoint 기반 제어를 사용한 자동화된 시연 데이터 수집 도구를 포함합니다.

## 개요

Waypoint 시스템을 통해 다음을 수행할 수 있습니다:
1. 목표 end-effector 위치 시퀀스로 로봇 궤적 정의
2. 다양한 컨트롤러 백엔드로 자동으로 waypoint 실행
3. 카메라 관측을 포함한 HDF5 형식의 시연 데이터 기록
4. 데이터셋을 LeRobot 형식으로 변환하고 HuggingFace에 업로드

## 시스템 아키텍처

### 컨트롤러 모듈

- **`waypoint_controller_dik.py`**: DifferentialIK 기반 컨트롤러 (기본값)
  - 빠르고 간단한 Jacobian 기반 역기구학
  - 대부분의 조작 태스크에 적합
  - 위치만 또는 전체 자세 제어 지원

- **`waypoint_controller_osc.py`**: OperationalSpaceController 기반 컨트롤러 (실험적)
  - 더 부드러운 모션 생성을 위한 임피던스 제어
  - 복잡한 상호작용에서 더 나은 안정성 제공 가능
  - 현재 개발 중

### 데이터 수집 스크립트

- **`bi_arm_waypoint_data_collection.py`**: 메인 실행 스크립트
  - JSON 파일에서 waypoint 로드
  - 선택한 컨트롤러로 궤적 실행
  - 관측, 액션, 카메라 데이터 기록
  - 자동 에피소드 관리 및 성공 라벨링 지원

## 빠른 시작

### 1. Waypoint 정의

waypoint 시퀀스가 포함된 JSON 파일 생성 (예: `playground/waypoints/bi_arm_demo.json`):

```json
[
  {
    "relative": false,
    "left": {
      "position": [0.22, -0.45, 0.22],
      "gripper": 0.0,
      "wrist_flex": 1.57
    },
    "right": {
      "position": [0.55, -0.45, 0.22],
      "gripper": 0.0,
      "wrist_flex": 1.57
    },
    "hold_steps": 45
  },
  {
    "relative": false,
    "left": {
      "position": [0.22, -0.45, 0.16],
      "gripper": 0.6,
      "wrist_flex": 1.57
    },
    "right": {
      "position": [0.55, -0.45, 0.16],
      "gripper": 0.6,
      "wrist_flex": 1.57
    },
    "hold_steps": 30
  }
]
```

**Waypoint 파라미터:**
- `position`: 월드 프레임에서의 end-effector 목표 위치 [x, y, z] (미터 단위)
- `orientation`: (선택사항) 목표 방향을 quaternion으로 표현 [w, x, y, z]
- `gripper`: 그리퍼 개방 정도 (0.0 = 닫힘, 1.0 = 열림)
- `wrist_flex`: (선택사항) wrist_flex 관절 각도 오버라이드 (라디안 단위)
- `hold_steps`: 다음 waypoint로 진행하기 전 현재 위치에서 유지할 스텝 수

### 2. 데이터 수집

**기본 사용법 (재생 전용):**
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_demo.json" \
    --controller_type=dik \
    --enable_cameras
```

**시연 데이터 기록:**
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_demo.json" \
    --controller_type=dik \
    --record \
    --dataset_file="datasets/my_demos.hdf5" \
    --num_demos=10 \
    --enable_cameras
```

**배치 파일 사용:**
```bash
# playground/running_scripts/collect_waypoint_data.bat 파일을 설정에 맞게 수정
playground\running_scripts\collect_waypoint_data.bat
```

### 3. 명령줄 인자

#### 컨트롤러 선택
- `--controller_type`: 컨트롤러 백엔드
  - `dik` (기본값): DifferentialIK 컨트롤러
  - `osc` (실험적): OperationalSpaceController

#### Waypoint 파라미터
- `--waypoint_file`: waypoint JSON 파일 경로
- `--step_hz`: 환경 스테핑 속도 (기본값: 30 Hz)
- `--hold_steps`: waypoint hold 기간 전역 오버라이드
- `--position_tol`: 위치 수렴 허용 오차 (미터 단위, 기본값: 0.05)
- `--pose_interp_gain`: 자세 보간 스무딩 (0-1, 기본값: 0.3)

#### DifferentialIK 전용
- `--command_type`: IK 제어 모드
  - `position`: 위치만 제어 (더 빠름, 권장)
  - `pose`: 전체 6-DOF 자세 제어
- `--interp_gain`: 관절 보간 스무딩 (0-1, 기본값: 0.3)

#### OperationalSpaceController 전용 (실험적)
- `--motion_stiffness`: 임피던스 강성 (기본값: 150.0)
- `--motion_damping_ratio`: 댐핑 비율 (기본값: 1.0)

#### 손목 제어
- `--force_wrist_down`: wrist_flex를 아래쪽으로 향하도록 강제 (기본값: 활성화)
- `--wrist_flex_angle`: 목표 wrist_flex 각도 (라디안 단위, 기본값: 1.57 = 90°)

#### 기록
- `--record`: 데이터 기록 활성화
- `--dataset_file`: 출력 HDF5 파일 경로
- `--num_demos`: 수집할 시연 횟수
- `--enable_cameras`: 카메라 관측 활성화 (VLM 데이터셋에 필수)

## 데이터셋 파이프라인

### HDF5 데이터셋 구조

기록된 시연은 HDF5 형식으로 저장됩니다:
```
dataset.hdf5
└── data/
    ├── demo_0/
    │   ├── actions              # (T, 12) 관절 위치
    │   ├── obs/
    │   │   ├── joint_pos        # (T, 12)
    │   │   ├── joint_vel        # (T, 12)
    │   │   ├── left_wrist       # (T, H, W, 3) 왼쪽 카메라 RGB
    │   │   ├── right_wrist      # (T, H, W, 3) 오른쪽 카메라 RGB
    │   │   └── top              # (T, H, W, 3) 탑 카메라 RGB
    │   └── attrs: {num_samples, seed, success}
    └── demo_1/
        └── ...
```

### LeRobot 형식으로 변환

**사전 준비:**
```bash
# 별도의 LeRobot 환경 생성
conda create -n lerobot python=3.10
conda activate lerobot
pip install lerobot==0.3.3  # 테스트된 버전
```

**변환 및 업로드:**
```bash
conda activate lerobot

python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo YourUsername/dataset_name \
    --robot_type bi_so101_follower \
    --fps 30 \
    --hdf5_files datasets/my_demos.hdf5 \
    --task "물건을 집어서 상자에 넣기" \
    --push_to_hub
```

**파라미터:**
- `--dataset_path_or_repo`: HuggingFace 데이터셋 리포지토리 이름
- `--robot_type`: 로봇 설정 (`so101_follower` 또는 `bi_so101_follower`)
- `--fps`: 비디오 프레임 레이트 (기본값: 30)
- `--hdf5_files`: 병합할 HDF5 파일 목록 (공백으로 구분)
- `--task`: 데이터셋 메타데이터용 태스크 설명
- `--push_to_hub`: HuggingFace Hub에 업로드 (인증 필요)

**인증:**
```bash
# 최초 설정
huggingface-cli login
```

### 완전한 워크플로우 예제

```bash
# 1. 시연 데이터 수집
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/bi_arm_demo.json" \
    --controller_type=dik \
    --record \
    --dataset_file="datasets/clean_table_demos.hdf5" \
    --num_demos=50 \
    --enable_cameras

# 2. LeRobot으로 변환 및 업로드
conda activate lerobot
python scripts/convert/isaaclab2lerobot.py \
    --dataset_path_or_repo myusername/clean_table_so101 \
    --robot_type bi_so101_follower \
    --fps 30 \
    --hdf5_files datasets/clean_table_demos.hdf5 \
    --task "양팔 협업을 사용한 테이블 표면 장난감 정리" \
    --push_to_hub
```

## 고급 사용법

### 여러 Waypoint 파일

다양한 시나리오에서 데이터 수집 후 병합:
```bash
# 시나리오 1 수집
python ... --dataset_file="datasets/scenario1.hdf5"

# 시나리오 2 수집
python ... --dataset_file="datasets/scenario2.hdf5"

# 변환 시 병합
python scripts/convert/isaaclab2lerobot.py \
    --hdf5_files datasets/scenario1.hdf5 datasets/scenario2.hdf5 \
    --push_to_hub
```

### Waypoint 디버깅

기록 없이 궤적 테스트 실행:
```bash
python scripts/environments/waypoints/bi_arm_waypoint_data_collection.py \
    --task=LeIsaac-SO101-CleanToyTable-BiArm-v0 \
    --waypoint_file="playground/waypoints/test.json" \
    --controller_type=dik \
    --enable_cameras
```

`Ctrl+C`를 눌러 실행 중지.

## 문제 해결

### 데이터 수집 중 GPU 크래시
- `--step_hz`를 낮춰보기 (예: 30에서 20 Hz로)
- 더 부드러운 전환을 위해 `--pose_interp_gain` 감소
- 부드러운 리셋 전환을 위해 첫 번째 waypoint에 `wrist_flex` 없도록 확인
- `--controller_type=osc` 시도 (실험적)

### 그리퍼가 아래를 향하지 않음
- `--force_wrist_down`이 활성화되어 있는지 확인 (기본값)
- waypoint JSON에 명시적으로 `wrist_flex: 1.57` 추가
- `--wrist_flex_angle`이 원하는 방향과 일치하는지 확인

### Waypoint가 수렴하지 않음
- `--position_tol` 증가 (예: 0.05에서 0.1로)
- JSON에서 waypoint `hold_steps` 증가
- 목표 위치가 도달 가능한지 확인 (특이점 없음)

### LeRobot 변환 오류
- LeRobot 버전 호환성 확인 (`pip install lerobot==0.3.3`)
- HDF5 파일에 성공한 에피소드가 포함되어 있는지 확인 (`success=True`)
- 카메라 키가 예상 형식과 일치하는지 확인 (left_wrist, right_wrist, top)

## 참고 자료

- [메인 리포지토리 README](../../../README.md)
- [LeRobot 문서](https://github.com/huggingface/lerobot)
- [IsaacLab 튜토리얼](https://isaac-sim.github.io/IsaacLab/)
