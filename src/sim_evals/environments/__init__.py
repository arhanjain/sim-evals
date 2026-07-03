import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from .droid_environment import EnvCfg as DroidEnvCfg


@configclass
class DroidCubeInBowlEnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-CubeInBowl")


@configclass
class DroidCanInMugEnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-CanInMug")


@configclass
class DroidBananaInBinEnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-BananaInBin")


@configclass
class DroidBerkeleyTask1EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-Berkeley-Task1")


@configclass
class DroidUPennFrankaKitchenEnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-UPenn-FrankaKitchen")


@configclass
class DroidUPennTeaRoomEnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-UPenn-TeaRoom")


@configclass
class DroidUPennLivingRoomEnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-UPenn-LivingRoom")


@configclass
class DroidUTAustinTask1EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-UTAustin-Task1")


@configclass
class DroidUTAustinTask2EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-UTAustin-Task2")


@configclass
class DroidUTAustinTask3EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-UTAustin-Task3")


@configclass
class DroidYonseiTask1EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-Yonsei-Task1")


@configclass
class DroidYonseiTask2EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-Yonsei-Task2")


@configclass
class DroidMILATask1EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-MILA-Task1")


@configclass
class DroidMILATask2EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-MILA-Task2")


@configclass
class DroidFrodoBotsTask1EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-FrodoBots-Task1")


@configclass
class DroidFrodoBotsTask2EnvCfg(DroidEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.set_scene("DROID-FrodoBots-Task2")


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


_register_droid_env("DROID-CubeInBowl", DroidCubeInBowlEnvCfg)
_register_droid_env("DROID-CanInMug", DroidCanInMugEnvCfg)
_register_droid_env("DROID-BananaInBin", DroidBananaInBinEnvCfg)
_register_droid_env("DROID-Berkeley-Task1", DroidBerkeleyTask1EnvCfg)
_register_droid_env("DROID-UPenn-FrankaKitchen", DroidUPennFrankaKitchenEnvCfg)
_register_droid_env("DROID-UPenn-TeaRoom", DroidUPennTeaRoomEnvCfg)
_register_droid_env("DROID-UPenn-LivingRoom", DroidUPennLivingRoomEnvCfg)
_register_droid_env("DROID-UTAustin-Task1", DroidUTAustinTask1EnvCfg)
_register_droid_env("DROID-UTAustin-Task2", DroidUTAustinTask2EnvCfg)
_register_droid_env("DROID-UTAustin-Task3", DroidUTAustinTask3EnvCfg)
_register_droid_env("DROID-Yonsei-Task1", DroidYonseiTask1EnvCfg)
_register_droid_env("DROID-Yonsei-Task2", DroidYonseiTask2EnvCfg)
_register_droid_env("DROID-MILA-Task1", DroidMILATask1EnvCfg)
_register_droid_env("DROID-MILA-Task2", DroidMILATask2EnvCfg)
_register_droid_env("DROID-FrodoBots-Task1", DroidFrodoBotsTask1EnvCfg)
_register_droid_env("DROID-FrodoBots-Task2", DroidFrodoBotsTask2EnvCfg)
