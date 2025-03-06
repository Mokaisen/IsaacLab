# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots of brownbotics.

Reference: https://github.com/ros-industrial/universal_robot
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##


BROWNBOT05_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/isaac-sim/workspaces/isaac_sim_scene/UR5_brown_02.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "arm_00": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "arm_01": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "arm_02": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "arm_03": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "arm_04": ImplicitActuatorCfg(
            joint_names_expr=["wrist_2_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "arm_05": ImplicitActuatorCfg(
            joint_names_expr=["wrist_3_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "arm_06": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=0.1125,
            damping=0.001,
        ),
    },
)
"""Configuration of UR-10 arm using implicit actuator models."""