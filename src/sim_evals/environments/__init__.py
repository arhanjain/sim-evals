import gymnasium as gym
from .droid_environment import (
    BananaInBinEnvCfg,
    CanInMugEnvCfg,
    CubeInBowlEnvCfg,
    EnvCfg as DroidEnvCfg,
)
from isaaclab.envs import ManagerBasedRLEnv

_DROID_ENV_CFGS = {
    "DROID": DroidEnvCfg,
    "DROID-CubeInBowl": CubeInBowlEnvCfg,
    "DROID-CanInMug": CanInMugEnvCfg,
    "DROID-BananaInBin": BananaInBinEnvCfg,
}


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
