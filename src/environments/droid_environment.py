import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import numpy as np

from typing import List
from pathlib import Path


from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg

DATA_PATH = Path(__file__).parent / "../../assets"

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    sphere_light = AssetBaseCfg(
        prim_path="/World/spehre",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn = sim_utils.UsdFileCfg(
                usd_path=str(DATA_PATH / "nvidia_scene/mesh.usd"),
                activate_contact_sensors=False,
                ),
            )

    banana = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/scene/banana",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.35, 0.19, 0.08),
                rot=(1.0, 0.0, 0.0, 0.0),
                )
            )
    bowl = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/scene/bowl",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.33, -0.1, 0.11),
                rot=(0.67, -0.74, 0.0, 0.0), 
                )
            )

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DATA_PATH / "my_droid.usdz"),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=36,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1, 0, 0, 0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -1 / 5 * np.pi,
                "panda_joint3": 0.0,
                "panda_joint4": -4 / 5 * np.pi,
                "panda_joint5": 0.0,
                "panda_joint6": 3 / 5 * np.pi,
                "panda_joint7": 0,
                "finger_joint": 0.0,
                "right_outer.*": 0.0,
                "left_outer.*": 0.0,
                "left_inner_finger_knuckle_joint": 0.0,
                "right_inner_finger_knuckle_joint": 0.0,
                "left_inner_finger_joint": 0.0,
                "right_inner_finger_joint": 0.0,
            },
        ),
        soft_joint_pos_limit_factor=1,
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=87.0,
                velocity_limit=2.175,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=12.0,
                velocity_limit=2.61,
                stiffness=400.0,
                damping=80.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint"],
                stiffness=15,
                damping=5,
                velocity_limit=0.5,
                effort_limit=120,
            ),
            "inner_finger": ImplicitActuatorCfg(
                joint_names_expr=[".*_inner_finger_joint"],
                stiffness=0.2,
                damping=0.02,
                velocity_limit=3.0,
                effort_limit=0.5,
            ),
        },
    )

    external_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.57, 0.66), rot=(-0.393, -0.195, 0.399, 0.805), convention="opengl"
        ),
    )
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074), rot=(-0.420, 0.570, 0.576, -0.409), convention="opengl"
        ),
    )

class BinaryJointPositionZeroToOneAction(BinaryJointPositionAction):
    # override
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # compute the binary mask
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions > 0.5
        # compute the command
        self._processed_actions = torch.where(
            binary_mask, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class BinaryJointPositionZeroToOneActionCfg(BinaryJointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type = BinaryJointPositionZeroToOneAction


class TargetJointPositionStaticAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    """The configuration of the action term."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[
                :, self._joint_ids
            ].clone()
        # self._default_actions = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._default_actions = self._asset.data.default_joint_pos[
            :, self._joint_ids
        ].clone()
        self._default_actions[:] = torch.tensor(cfg.target)

    @property
    def action_dim(self) -> int:
        return 0

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(
            self._default_actions, joint_ids=self._joint_ids
        )


@configclass
class TargetJointPositionStaticActionCfg(mdp.JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    target: List[float] = [0.0]

    class_type = TargetJointPositionStaticAction
    use_default_offset: bool = True
    preserve_order: bool = True


@configclass
class ActionCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        preserve_order=True,
        use_default_offset=False,
    )

    finger_joint = BinaryJointPositionZeroToOneActionCfg(
        asset_name="robot",
        joint_names=["finger_joint"],
        open_command_expr={"finger_joint": -np.pi / 4},
        # open_command_expr = {"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )

    compliant_joints = TargetJointPositionStaticActionCfg(
        asset_name="robot",
        joint_names=["left_inner_finger_joint", "right_inner_finger_joint"],
        target=[-np.pi / 4, np.pi / 4],
    )


def arm_joint_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    # get joint inidices
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]
    return joint_pos


def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos

def gripper_pos(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot = env.scene[asset_cfg.name]
    joint_names = ["finger_joint"]
    joint_indices = [
        i for i, name in enumerate(robot.data.joint_names) if name in joint_names
    ]
    joint_pos = robot.data.joint_pos[0, joint_indices]

    # rescale
    joint_pos = joint_pos / (np.pi / 4)

    return joint_pos

@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(
            func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1)
        )
        external_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )
        wrist_cam = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("wrist_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                    }
                )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

@configclass
class CommandsCfg:
    """Command terms for the MDP."""


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=1, env_spacing=7.0)

    observations = ObservationCfg()
    actions = ActionCfg()
    rewards = RewardsCfg()

    terminations = TerminationsCfg()
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
        self.episode_length_s = 30

        self.viewer.eye = (4.5, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.decimation = 4 * 2
        self.sim.dt = 1 / (60 * 2)
        self.sim.render_interval = 4 * 2

        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.sim.physx.enable_ccd = True
        self.rerender_on_reset = True

        # self.sim.render.enable_reflections = False
        # self.sim.render.enable_shadows = False
        # self.sim.render.enable_direct_lighting = False
        # self.sim.render.enable_ambient_occlusion = False
