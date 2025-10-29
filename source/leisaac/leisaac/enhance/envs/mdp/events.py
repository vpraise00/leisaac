import torch

from typing import Literal

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv


def randomize_camera_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation uniformly within the given ranges.

    * It samples the camera position and rotation from the given ranges and adds them to the 
      default camera position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    asset: Camera = env.scene[asset_cfg.name]

    ori_pos_w = asset.data.pos_w[env_ids]
    if convention == "ros":
        ori_quat_w = asset.data.quat_w_ros[env_ids]
    elif convention == "opengl":
        ori_quat_w = asset.data.quat_w_opengl[env_ids]
    elif convention == "world":
        ori_quat_w = asset.data.quat_w_world[env_ids]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = ori_pos_w[:, 0:3] + rand_samples[:, 0:3]  # camera usually spawn with robot, so no need to add env_origins
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)


<<<<<<< Updated upstream
def randomize_particle_object_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
):
    """Reset the particle object to a random position and rotation uniformly within the given ranges.

    * It samples the particle object position and rotation from the given ranges and adds them to the
      default particle object position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    particle_object = env.scene.particle_objects[asset_cfg.name]
    ori_world_pos, ori_world_quat = particle_object.get_world_poses()

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)

    positions = ori_world_pos + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_world_quat, orientations_delta)

    particle_object.set_world_poses(positions, orientations)
=======
def randomize_rigid_object_position_gaussian(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    position_std: dict[str, float],
):
    """Randomize rigid object position with Gaussian noise.

    This function adds Gaussian noise to the object's default position.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        asset_cfg: Configuration for the asset to randomize.
        position_std: Dictionary with standard deviations for each axis (x, y, z).
                     Keys should be 'x', 'y', 'z'. Missing keys default to 0.0 (no noise).

    Example:
        position_std={'x': 0.1, 'y': 0.1, 'z': 0.0}  # Randomize x,y with std=0.1, z fixed
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get default position from asset configuration
    default_pos = asset.data.default_root_state[env_ids, :3]

    # Get standard deviations for each axis
    std_x = position_std.get('x', 0.0)
    std_y = position_std.get('y', 0.0)
    std_z = position_std.get('z', 0.0)

    # Generate Gaussian noise
    noise = torch.randn(len(env_ids), 3, device=asset.device)
    noise[:, 0] *= std_x
    noise[:, 1] *= std_y
    noise[:, 2] *= std_z

    # Apply noise to default position
    new_positions = default_pos + noise

    # Get current orientation (keep it unchanged)
    current_orientation = asset.data.default_root_state[env_ids, 3:7]

    # Set new pose
    asset.write_root_pose_to_sim(
        torch.cat([new_positions, current_orientation], dim=1), env_ids
    )
>>>>>>> Stashed changes
