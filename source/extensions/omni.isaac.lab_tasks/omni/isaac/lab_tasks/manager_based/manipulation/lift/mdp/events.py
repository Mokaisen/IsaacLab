import random
import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg

def biased_reset_root_state_uniform(
        env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
):
    if random.random() < 0.7:
        pose_range = {
            "x": (-0.1, 0.1),
            "y": (0.0, 0.25),
            "z": (0.0, 0.0),
        }
    else:
        pose_range = { 
            "x": (-0.1, 0.1),
            "y": (-0.25, 0.0),
            "z": (0.0, 0.0),
        }
    
    velocity_range = {}

    return reset_root_state_uniform(
        env=env,
        env_ids=env_ids,
        pose_range=pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )