from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from pxr import Usd, UsdPhysics

from sim_evals.environments.droid_environment import (
    DATA_PATH,
    EnvCfg as DroidEnvCfg,
    EventCfg as DroidEventCfg,
    SceneCfg as DroidSceneCfg,
)

from .rewards import milestone_reward, reset_milestones
from .tasks import get_mila_task


MILA_ASSET_PATH = DATA_PATH / "mila"


@configclass
class MilaSceneCfg(DroidSceneCfg):
    def dynamic_task_scene(self, task_id: str):
        task = get_mila_task(task_id)
        environment_path = MILA_ASSET_PATH / task.scene_asset
        self._load_scene_usd(environment_path)

    def _load_scene_usd(self, environment_path: Path):
        scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(usd_path=str(environment_path)),
        )
        self.scene = scene

        stage = Usd.Stage.Open(str(environment_path))
        if stage is None:
            raise FileNotFoundError(f"Could not open MILA scene asset: {environment_path}")

        scene_prim = stage.GetPrimAtPath("/World")
        for child in scene_prim.GetChildren():
            if not UsdPhysics.RigidBodyAPI(child):
                continue

            name = child.GetName()
            print(f"Found rigid body: {name}")
            pos = child.GetAttribute("xformOp:translate").Get()
            rot = child.GetAttribute("xformOp:orient").Get()
            rot = (rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2])
            asset = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/scene/{name}",
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
            )
            setattr(self, name, asset)


@configclass
class MilaRewardsCfg:
    milestones = RewTerm(
        func=milestone_reward,
        weight=1.0,
        params={"task_id": "place_spoon_in_sink"},
    )


@configclass
class MilaEventCfg(DroidEventCfg):
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_milestones = EventTerm(func=reset_milestones, mode="reset")


@configclass
class MilaDroidEnvCfg(DroidEnvCfg):
    scene = MilaSceneCfg(num_envs=1, env_spacing=7.0)
    rewards = MilaRewardsCfg()
    events = MilaEventCfg()
    task_id = "place_spoon_in_sink"

    def set_task(self, task_id: str):
        task = get_mila_task(task_id)
        self.task_id = task.task_id
        self.scene.dynamic_task_scene(task.task_id)
        self.rewards.milestones.params["task_id"] = task.task_id
        self.episode_length_s = task.max_timesteps * self.sim.dt * self.decimation
