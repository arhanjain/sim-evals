"""Scene-agnostic base environment config for the DROID sim-eval tasks.

:class:`DroidEnvCfg` describes everything shared across scenes -- the robot,
cameras, action space, observations, and MDP defaults. It carries no task USD
or instruction on its own; a concrete scene is composed onto it with
:meth:`DroidEnvCfg.set_scene`, which is what the Gym registration in
``__init__`` calls. Per-scene reward / termination overlays declared on the
:class:`~sim_evals.environments.droid.scenes.DroidSceneSpec` are applied there
too.
"""

import numpy as np

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, noise

from .mdp import BinaryJointPositionZeroToOneActionCfg, arm_joint_pos, gripper_pos
from .robot import NVIDIA_DROID
from .scenes import get_scene_spec, resolve_scene_asset_path


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Robot, lighting and cameras shared by every DROID scene.

    The task USD is not part of the static config; it is attached per scene by
    :meth:`dynamic_scene`.
    """

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
        """Attach the task USD for ``env_id`` and return its scene metadata."""
        scene_spec = get_scene_spec(env_id)
        environment_path = resolve_scene_asset_path(scene_spec)

        self.scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(environment_path),
            ),
        )

        return {
            "scene_env_id": env_id,
            "scene_instruction": scene_spec.instruction,
            "scene_asset_path": str(environment_path),
        }


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
        open_command_expr={"finger_joint": 0.0},
        close_command_expr={"finger_joint": np.pi / 4},
    )


@configclass
class ObservationCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        arm_joint_pos = ObsTerm(func=arm_joint_pos)
        gripper_pos = ObsTerm(func=gripper_pos, noise=noise.GaussianNoiseCfg(std=0.05), clip=(0, 1))
        external_cam = ObsTerm(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("external_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        external_cam_2 = ObsTerm(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("external_cam_2"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        wrist_cam = ObsTerm(
            func=mdp.observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
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
    """Reward terms for the MDP.

    Empty by default -- the eval loop does not consume reward. Scenes with a
    success condition attach their own reward group via ``DroidSceneSpec``.
    """


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum configuration."""


@configclass
class DroidEnvCfg(ManagerBasedRLEnvCfg):
    """Scene-agnostic base config for the DROID environments.

    Instantiate and call :meth:`set_scene` to obtain a concrete, registered
    scene, e.g. ``DroidEnvCfg().set_scene("DROID-CubeInBowl")``.
    """

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

    def set_scene(self, env_id: str) -> "DroidEnvCfg":
        """Compose a concrete scene onto this base config.

        Loads the scene's USD and instruction, and applies any per-scene reward
        / termination overlays declared on its ``DroidSceneSpec``. Returns
        ``self`` so calls can be chained.
        """
        scene_spec = get_scene_spec(env_id)
        self.scene_env_id = env_id
        self.language_instruction = scene_spec.instruction

        scene_metadata = self.scene.dynamic_scene(env_id)
        self.scene_asset_path = scene_metadata["scene_asset_path"]

        # Optional per-scene task overlays; keep the base defaults when unset.
        if scene_spec.rewards is not None:
            self.rewards = scene_spec.rewards()
        if scene_spec.terminations is not None:
            self.terminations = scene_spec.terminations()

        return self
