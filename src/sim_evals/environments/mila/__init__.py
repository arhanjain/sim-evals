import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

from .droid_environment import MilaDroidEnvCfg


gym.register(
    id="MILA-DROID",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": MilaDroidEnvCfg,
    },
    disable_env_checker=True,
)
