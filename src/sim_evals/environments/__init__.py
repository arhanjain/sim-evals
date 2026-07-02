import gymnasium as gym
from .droid_environment import (
    DROID_SCENES,
    EnvCfg as DroidEnvCfg,
    make_env_cfg_class,
)
from isaaclab.envs import ManagerBasedRLEnv

_DROID_ENV_CFGS = {"DROID": DroidEnvCfg}
_DROID_ENV_CFGS.update(
    {scene_spec.env_id: make_env_cfg_class(scene_spec.env_id) for scene_spec in DROID_SCENES.values()}
)


def _register_droid_env(env_id: str, env_cfg_entry_point: type[DroidEnvCfg]) -> None:
    if env_id in gym.envs.registry:
        return

    gym.register(
        id=env_id,
        entry_point=ManagerBasedRLEnv,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
        },
        disable_env_checker=True,
    )


for _env_id, _env_cfg_entry_point in _DROID_ENV_CFGS.items():
    _register_droid_env(_env_id, _env_cfg_entry_point)
