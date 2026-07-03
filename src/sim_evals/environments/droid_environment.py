import torch
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import numpy as np
import os

from typing import NamedTuple
from pathlib import Path

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass, noise
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg

from .nvidia_droid import NVIDIA_DROID

ASSET_ROOT_ENV_VAR = "SIM_EVALS_ASSET_ROOT"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "assets"


class DroidSceneSpec(NamedTuple):
    instruction: str
    asset_paths: tuple[str, ...]


DROID_SCENES = {
    "DROID-CubeInBowl": DroidSceneSpec("put the cube in the bowl", ("scene1.usd",)),
    "DROID-CanInMug": DroidSceneSpec("put the can in the mug", ("scene2.usd",)),
    "DROID-BananaInBin": DroidSceneSpec("put banana in the bin", ("scene3.usd",)),
    "DROID-Berkeley-Task1": DroidSceneSpec(
        "complete the Berkeley task 1",
        ("Berkeley/Berkeley_Task1/Scene.usd", "Berkeley_Task1/Scene.usd"),
    ),
    "DROID-UPenn-FrankaKitchen": DroidSceneSpec(
        "complete the Franka kitchen task",
        ("UPenn/TASK-1-Levine457-FrankaKitchen/Scene.usd", "TASK-1-Levine457-FrankaKitchen/Scene.usd"),
    ),
    "DROID-UPenn-TeaRoom": DroidSceneSpec(
        "complete the tea room task",
        ("UPenn/TASK-2-Levine459-TeaRoom/Scene.usd", "TASK-2-Levine459-TeaRoom/Scene.usd"),
    ),
    "DROID-UPenn-LivingRoom": DroidSceneSpec(
        "complete the living room task",
        ("UPenn/TASK-3-AGH-LivingRoom/Scene.usd", "TASK-3-AGH-LivingRoom/Scene.usd"),
    ),
    "DROID-UTAustin-Task1": DroidSceneSpec(
        "complete the UT Austin task 1",
        ("UT-Austin/UT-Austin-Task1/Scene.usd", "UT-Austin-Task1/Scene.usd"),
    ),
    "DROID-UTAustin-Task2": DroidSceneSpec(
        "complete the UT Austin task 2",
        ("UT-Austin/UT-Austin-Task2/Scene.usd", "UT-Austin-Task2/Scene.usd"),
    ),
    "DROID-UTAustin-Task3": DroidSceneSpec(
        "complete the UT Austin task 3",
        ("UT-Austin/UT-Austin-Task3/Scene.usd", "UT-Austin-Task3/Scene.usd"),
    ),
    "DROID-Yonsei-Task1": DroidSceneSpec(
        "complete the Yonsei task 1",
        ("Yonsei/Task1_20260424/Scene.usd", "Task1_20260424/Scene.usd"),
    ),
    "DROID-Yonsei-Task2": DroidSceneSpec(
        "complete the Yonsei task 2",
        ("Yonsei/Task2/Scene.usd", "Task2/Scene.usd"),
    ),
    "DROID-MILA-Task1": DroidSceneSpec(
        "complete the MILA task 1",
        ("MILA/Task1/Scene.usd", "Task1/Scene.usd"),
    ),
    "DROID-MILA-Task2": DroidSceneSpec(
        "complete the MILA task 2",
        ("MILA/Task2/Scene.usd", "Task2/Scene.usd"),
    ),
    "DROID-FrodoBots-Task1": DroidSceneSpec(
        "complete the FrodoBots task 1",
        ("FrodoBots/Task1/Scene.usd", "Task1/Scene.usd"),
    ),
    "DROID-FrodoBots-Task2": DroidSceneSpec(
        "complete the FrodoBots task 2",
        ("FrodoBots/Task2/Scene.usd", "Task2/Scene.usd"),
    ),
}


def get_data_path() -> Path:
    return Path(os.environ.get(ASSET_ROOT_ENV_VAR, DEFAULT_DATA_PATH)).expanduser().resolve()


def resolve_scene_asset_path(scene_spec: DroidSceneSpec) -> Path:
    candidates = [get_data_path() / asset_path for asset_path in scene_spec.asset_paths]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find asset for {scene_spec.env_id}. Checked:\n  - {candidate_text}\n"
        f"Set {ASSET_ROOT_ENV_VAR} to the directory containing the downloaded assets."
    )


def get_scene_spec(env_id: str) -> DroidSceneSpec:
    try:
        return DROID_SCENES[env_id]
    except KeyError as exc:
        valid = ", ".join(DROID_SCENES)
        raise ValueError(f"Unknown DROID environment {env_id!r}. Valid env ids: {valid}") from exc


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    sphere_light = AssetBaseCfg(
        prim_path="/World/spehre",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

    robot = NVIDIA_DROID

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

    external_cam_2 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam_2",
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
            pos=(0.05, -0.57, 0.66), rot=(0.805, 0.399, -0.195, -0.393), convention="opengl"
        ),
    )

    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
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

    def dynamic_scene(self, env_id: str) -> dict:
        scene_spec = get_scene_spec(env_id)
        environment_path = resolve_scene_asset_path(scene_spec)

        scene = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/scene",
                spawn = sim_utils.UsdFileCfg(
                    usd_path=str(environment_path),
                    ),
                )
        self.scene = scene

        return {
            "scene_env_id": env_id,
            "scene_instruction": scene_spec.instruction,
            "scene_asset_path": str(environment_path),
        }


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
        open_command_expr = {"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
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
        external_cam_2 = ObsTerm(
                func=mdp.observations.image,
                params={
                    "sensor_cfg": SceneEntityCfg("external_cam_2"),
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

        self.decimation = 8
        self.sim.dt = 1 / (15 * 8)
        self.sim.render_interval = self.decimation

        self.sim.physx.enable_ccd = True
        self.sim.physx.gpu_temp_buffer_capacity = 2**30
        self.sim.physx.gpu_heap_capacity = 2**30
        self.sim.physx.gpu_collision_stack_size = 2**30
        self.rerender_on_reset = True

    
    def set_scene(self, env_id: str):
        scene_spec = get_scene_spec(env_id)
        self.scene_env_id = env_id
        self.language_instruction = scene_spec.instruction
        scene_metadata = self.scene.dynamic_scene(env_id)
        self.scene_asset_path = scene_metadata["scene_asset_path"]
