import pickle
import torch
import grpc
import time
import numpy as np

from .base import ZMQServicePolicy, Policy
from .lerobot.transport import services_pb2_grpc, services_pb2
from .lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from .lerobot.helpers import RemotePolicyConfig, TimedObservation

from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES


class Gr00tServicePolicyClient(ZMQServicePolicy):
    """
    Service policy client for GR00T N1.5: https://github.com/NVIDIA/Isaac-GR00T
    Target Commit: https://github.com/NVIDIA/Isaac-GR00T/commit/4ea96a16b15cfdbbd787b6b4f519a12687281330
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 5000,
        camera_keys: list[str] = ['front', 'wrist'],
        modality_keys: list[str] = ["single_arm", "gripper"],
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            camera_keys: Keys of the cameras.
            timeout_ms: Timeout of the policy server.
            modality_keys: Keys of the modality.
        """
        super().__init__(host=host, port=port, timeout_ms=timeout_ms, ping_endpoint="ping")
        self.camera_keys = camera_keys
        self.modality_keys = modality_keys

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        obs_dict = {f"video.{key}": observation_dict[key].cpu().numpy().astype(np.uint8) for key in self.camera_keys}

        if "single_arm" in self.modality_keys:
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            obs_dict["state.single_arm"] = joint_pos[:, 0:5].astype(np.float64)
            obs_dict["state.gripper"] = joint_pos[:, 5:6].astype(np.float64)
        # TODO: add bi-arm support

        obs_dict["annotation.human.task_description"] = [observation_dict["task_description"]]

        """
            Example of obs_dict for single arm task:
            obs_dict = {
                "video.front": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "video.wrist": np.zeros((1, 480, 640, 3), dtype=np.uint8),
                "state.single_arm": np.zeros((1, 5)),
                "state.gripper": np.zeros((1, 1)),
                "annotation.human.action.task_description": [observation_dict["task_description"]],
            }
        """

        # get the action chunk via the policy server
        action_chunk = self.call_endpoint("get_action", obs_dict)

        """
            Example of action_chunk for single arm task:
            action_chunk = {
                "action.single_arm": np.zeros((1, 5)),
                "action.gripper": np.zeros((1, 1)),
            }
        """
        concat_action = np.concatenate(
            [action_chunk["action.single_arm"], action_chunk["action.gripper"][:, None]],
            axis=1,
        )
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        return torch.from_numpy(concat_action[:, None, :])


class LeRobotServicePolicyClient(Policy):
    """
    Service policy client for Lerobot: https://github.com/huggingface/lerobot
    Target Commit: https://github.com/huggingface/lerobot/tree/v0.3.3
    """

    def __init__(
        self,
        host: str,
        port: int,
        timeout_ms: int = 5000,
        camera_infos: dict[str, dict] = {},
        task_type: str = 'so101leader',
        policy_type: str = 'smolvla',
        pretrained_name_or_path: str = 'checkpoints/last/pretrained_model',
        actions_per_chunk: int = 50,
        device: str = 'cuda',
    ):
        """
        Args:
            host: Host of the policy server.
            port: Port of the policy server.
            timeout_ms: Timeout of the policy server.
            camera_infos: List of camera information. {camera_key: (height, width)}
            task_type: Type of task.
            policy_type: Type of policy.
            pretrained_name_or_path: Path to the pretrained model in the remote policy server.
            actions_per_chunk: Number of actions per chunk.
            device: Device to use.
        """
        super().__init__("service")
        service_address = f'{host}:{port}'
        self.timeout_ms = timeout_ms
        self.task_type = task_type
        self.actions_per_chunk = actions_per_chunk

        lerobot_features: dict[str, dict] = {}
        self.last_action = None
        self.primary_image_key: str | None = None
        if task_type == 'so101leader':
            lerobot_features['observation.state'] = {
                'dtype': 'float32',
                'shape': (6,),
                'names': [f'{joint_name}.pos' for joint_name in SINGLE_ARM_JOINT_NAMES],
            }
            self.last_action = np.zeros((1, 6))
        # TODO: add bi-arm support

        self.camera_keys = list(camera_infos.keys())
        self.image_key_map: dict[str, str] = {}
        policy_image_keys = self._infer_policy_image_keys(pretrained_name_or_path)
        if not policy_image_keys:
            policy_image_keys = [f'observation.images.{key}' for key in self.camera_keys]

        for policy_key in policy_image_keys:
            mapped_camera = self._resolve_policy_camera_key(policy_key, camera_infos)
            if mapped_camera is None:
                continue
            self.image_key_map[policy_key] = mapped_camera
            camera_image_shape = camera_infos[mapped_camera]
            lerobot_features[policy_key] = {
                'dtype': 'image',
                'shape': (camera_image_shape[0], camera_image_shape[1], 3),
                'names': ['height', 'width', 'channels'],
            }

        if self.image_key_map:
            first_policy_key = next(iter(self.image_key_map))
            self.primary_image_key = self.image_key_map[first_policy_key]

        self.policy_config = RemotePolicyConfig(
            policy_type,
            pretrained_name_or_path,
            lerobot_features,
            actions_per_chunk,
            device,
        )
        print(f"[LeRobotServicePolicyClient] available cameras: {self.camera_keys}")
        print(f"[LeRobotServicePolicyClient] policy image keys: {list(self.image_key_map.keys())}")
        print(f"[LeRobotServicePolicyClient] camera mapping: {self.image_key_map}")
        self.channel = grpc.insecure_channel(
            service_address, grpc_channel_options()
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.latest_action_step = 0
        self.skip_send_observation = False

        self._init_service()

    def _init_service(self):
        try:
            self.stub.Ready(services_pb2.Empty())

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            print("Sending policy instructions to policy server, it may take a while...")
            self.stub.SendPolicyInstructions(policy_setup)
            print("Policy server is ready.")

        except grpc.RpcError:
            raise RuntimeError("Failed to connect to policy server")

    def _send_observation(self, observation_dict: dict):
        raw_observation: dict[str, object] = {}
        frames: dict[str, np.ndarray] = {}
        for camera_key in self.camera_keys:
            if camera_key not in observation_dict:
                continue
            frames[camera_key] = observation_dict[camera_key].cpu().numpy().astype(np.uint8)[0]
            raw_observation[camera_key] = frames[camera_key]
            raw_observation[f"observation.images.{camera_key}"] = frames[camera_key]

        for policy_key, camera_key in self.image_key_map.items():
            if camera_key not in frames:
                continue
            frame = frames[camera_key]
            if policy_key.startswith("observation.images."):
                alias = policy_key[len("observation.images.") :]
                raw_observation[alias] = frame
                raw_observation[policy_key] = frame
            elif policy_key == "observation.image":
                raw_observation["observation.image"] = frame

        raw_observation["task"] = observation_dict["task_description"]
        raw_observation["observation.task"] = observation_dict["task_description"]

        if self.task_type == 'so101leader':
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
            for joint_name in SINGLE_ARM_JOINT_NAMES:
                joint_value = joint_pos[0, SINGLE_ARM_JOINT_NAMES.index(joint_name)].item()
                raw_observation[f"{joint_name}.pos"] = joint_value
                raw_observation[f"observation.state.{joint_name}.pos"] = joint_value
        # TODO: add bi-arm support

        """
            Example of raw_observation for single arm task:
            raw_observation = {
                "front": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 0.0,
                "task": "pick_and_place",
            }
        """
        self.latest_action_step += 1
        observation = TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=self.latest_action_step,
        )
        if self.latest_action_step == 1:
            print(f"[LeRobotServicePolicyClient] raw observation keys: {list(raw_observation.keys())}")

        # send observation to policy server
        observation_bytes = pickle.dumps(observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        _ = self.stub.SendObservations(observation_iterator)

    def _receive_action(self) -> dict:
        actions_chunk = self.stub.GetActions(services_pb2.Empty())
        if len(actions_chunk.data) == 0:
            print("Received `Empty` from policy server, waiting for next call")
            return None
        return pickle.loads(actions_chunk.data)

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        if not self.skip_send_observation:
            self._send_observation(observation_dict)
        action_chunk = self._receive_action()
        if action_chunk is None:
            self.skip_send_observation = True
            return torch.from_numpy(self.last_action).repeat(self.actions_per_chunk, 1)[:, None, :]

        action_list = [action.get_action()[None, :] for action in action_chunk]
        concat_action = torch.cat(action_list, dim=0)
        concat_action = convert_lerobot_action_to_leisaac(concat_action)

        self.last_action = concat_action[-1, :]
        self.skip_send_observation = False

        return torch.from_numpy(concat_action[:, None, :])

    def _infer_policy_image_keys(self, pretrained_name_or_path: str) -> list[str]:
        keys: list[str] = []
        try:
            import os
            import json
            if os.path.isdir(pretrained_name_or_path):
                cfg_path = os.path.join(pretrained_name_or_path, "config.json")
                if not os.path.isfile(cfg_path):
                    return keys
            else:
                from huggingface_hub import hf_hub_download  # type: ignore
                cfg_path = hf_hub_download(
                    repo_id=pretrained_name_or_path,
                    filename="config.json",
                )
            with open(cfg_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            for key, ft in config.get("input_features", {}).items():
                if ft.get("type") == "VISUAL":
                    keys.append(key)
        except Exception as exc:  # pragma: no cover - best-effort fallback
            print(f"[LeRobotServicePolicyClient] Unable to infer policy image keys: {exc}")
        return keys

    def _resolve_policy_camera_key(
        self,
        policy_key: str,
        camera_infos: dict[str, tuple[int, int, int]],
    ) -> str | None:
        if not camera_infos:
            return None
        candidate_keys = list(camera_infos.keys())
        if policy_key.startswith("observation.images."):
            target = policy_key[len("observation.images.") :]
            if target in camera_infos:
                return target
            # heuristic fallbacks
            if target.lower() == "top":
                if "top" in camera_infos:
                    return "top"
                if "wrist" in camera_infos:
                    return "wrist"
                if "front" in camera_infos:
                    return "front"
            if target.lower() == "wrist" and "wrist" not in camera_infos and candidate_keys:
                return candidate_keys[0]
            if candidate_keys:
                return candidate_keys[0]
        elif policy_key == "observation.image":
            return candidate_keys[0]
        else:
            return candidate_keys[0]
        return None
