# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms

import torch.nn.functional as F

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    # print(f"################### Object ee distance {object_ee_distance}:")

    # End-effector velocity (num_envs, 3)
    # ee_velocity = ee_frame.data.target_lin_vel_w[..., 0, :]
    # ee_speed = torch.norm(ee_velocity, dim=1)  # Speed scalar (num_envs,)

    # based reward with smooth decay
    # reward_distance = torch.exp(- (object_ee_distance / std) )
    reward_distance = torch.exp(-(object_ee_distance / 0.2))
    # print(f"object ee distance 2: {reward_distance}")

    # reward_distance += (object_ee_distance < 0.60) * 1.5
    # reward_distance += (object_ee_distance < 0.38) * 5.0

    # Velocity penalty when close to the object
    # velocity_penalty = (object_ee_distance < 0.40) * (ee_speed * -0.5)
    # reward_distance += velocity_penalty

    return reward_distance 


def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix. Shape: (N, 4) -> (N, 3, 3)"""
    # Normalize quaternion
    quat = F.normalize(quat, dim=1)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.shape[0]
    rot = torch.empty((B, 3, 3), device=quat.device)

    rot[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    rot[:, 0, 1] = 2 * (x * y - z * w)
    rot[:, 0, 2] = 2 * (x * z + y * w)
    rot[:, 1, 0] = 2 * (x * y + z * w)
    rot[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    rot[:, 1, 2] = 2 * (y * z - x * w)
    rot[:, 2, 0] = 2 * (x * z - y * w)
    rot[:, 2, 1] = 2 * (y * z + x * w)
    rot[:, 2, 2] = 1 - 2 * (x**2 + y**2)

    return rot

def orientation_alignment(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the alignment between EE Z-axis and the direction to the object."""

    object = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    # Positions
    cube_pos = object.data.root_pos_w               # (num_envs, 3)
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)

    # Vector from EE to object
    vec_to_obj = F.normalize(cube_pos - ee_pos, dim=1)

    # Convert quaternion to rotation matrix
    quat = ee_frame.data.target_quat_w[..., 0, :]  # (num_envs, 4)
    rot = quaternion_to_rotation_matrix(quat)      # (num_envs, 3, 3)

    # Z-axis of EE (approach direction)
    ee_z = rot[:, :, 2]  # (num_envs, 3)

    # Alignment as dot product
    alignment = torch.sum(ee_z * vec_to_obj, dim=1)

    # Optional: smooth reward
    reward = (alignment + 1.0) / 2.0  # Scale to [0, 1]

    #print(f"reward aligment: {reward}")

    return reward


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
