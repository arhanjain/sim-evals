import gymnasium as gym
from .droid_environment import EnvCfg as DroidEnvCfg
from .lw_environment import EnvCfg as LWEnvCfg
from .dynamic_manager_based_rl_env import DynamicManagerBasedRLEnv
from isaaclab.envs import ManagerBasedRLEnv
import os
from pathlib import Path
from . import mdp


DATA_PATH = Path(os.environ.get("POLARIS_DATA_PATH", Path(__file__).parent.parent.parent.parent / "assets"))

gym.register(
    id="DROID",
    entry_point=ManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": DroidEnvCfg,
    },
    disable_env_checker=True,
)

# TODO: check that final points are awarded when object is not grasped
gym.register(
    id="Test",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "scene1.usd"),
    },
    disable_env_checker=True,
)

gym.register(
    id="BlockStack",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "lw_block_stack/scene.usda"),
        "progress_criteria": [
            # mdp.reach("green_cube", threshold=0.2),
            # mdp.reach("wood_cube", threshold=0.2),
            # (mdp.lift("green_cube", default_height=0.06, threshold=0.03), [0]),
            # (mdp.lift("wood_cube", default_height=0.06, threshold=0.03), [1]),
            mdp.point_in_obb("green_cube", "tray", check_axes=(0, 1), require_gripper_open=True),
            mdp.point_in_obb("wood_cube", "tray", check_axes=(0, 1), require_gripper_open=True),
            (mdp.point_in_obb("green_cube", "wood_cube", check_axes=(0, 1), require_gripper_open=True), [0, 1]),
        ]
    },
    disable_env_checker=True,
)

gym.register(
    id="FoodBussing",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "lw_food_bus/scene.usda"),
        "progress_criteria": [
            # mdp.reach("ice_cream_", threshold=0.2),
            # mdp.reach("grapes", threshold=0.2),
            # (mdp.lift("ice_cream_", threshold=0.06), [0]),
            # (mdp.lift("grapes", threshold=0.06), [1]),
            mdp.point_in_obb("ice_cream_", "bowl", check_axes=(0, 1, 2), require_gripper_open=True),
            mdp.point_in_obb("grapes", "bowl", check_axes=(0, 1, 2), require_gripper_open=True),
        ]
    },
    disable_env_checker=True,
)

gym.register(
    id="PanClean",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "lw_pan_clean/scene.usda"),
        "progress_criteria": [
            mdp.proximity("sponge", "pan", threshold=0.1, check_axes=(0, 1), require_gripper_open=False),
        ]
    },
    disable_env_checker=True,
)

gym.register(
    id="OrganizeTools",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "OrganizeTools_Sence/scene.usda"),
        "progress_criteria": [
            # mdp.reach("Scissor052", threshold=0.2),
            # (mdp.lift("Scissor052", threshold=0.04), [0]),
            mdp.point_in_obb("Scissor052", "Box191", check_axes=(0, 1), require_gripper_open=True),
        ]
    },
    disable_env_checker=True,
)

gym.register(
    id="MoveLatteCup",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "MoveLatteCup_Sence/scene.usda"),
        "progress_criteria": [
            # mdp.reach("Cup063_02", threshold=0.2),
            # (mdp.lift("Cup063_02", threshold=0.04), [0]),
            mdp.point_in_obb("Cup063_02", "ChoppingBoard001_02", check_axes=(0, 1), require_gripper_open=True),
        ]
    },
    disable_env_checker=True,
)

gym.register(
    id="TapeIntoContainer",
    entry_point=DynamicManagerBasedRLEnv,
    kwargs={
        "env_cfg_entry_point": LWEnvCfg,
        "usd_file": str(DATA_PATH / "TapeIntoContainer_Scene/scene.usda"),
        "progress_criteria": [
            # mdp.reach("PackagingTape011", threshold=0.2),
            # (mdp.lift("PackagingTape011", threshold=0.04), [0]),
            mdp.point_in_obb("PackagingTape011", "Box193", check_axes=(0, 1), require_gripper_open=True),
        ]
    },
    disable_env_checker=True,
)

# gym.register(
#     id="DROID-MoveLatteCup",
#     entry_point=ManagerBasedRLSplatEnv,
#     disable_env_checker=True,
#     order_enforce=False,
#     kwargs={
#         "env_cfg_entry_point": DroidCfg,
#         "usd_file": str(DATA_PATH / "move_latte_cup/scene.usda"),
#         "rubric": Rubric(
#             criteria=[
#                 checkers.reach("latteartcup_eval", threshold=0.2),
#                 (checkers.lift("latteartcup_eval", threshold=0.04), [0]),
#                 (checkers.is_within_xy("latteartcup_eval", "cuttingboard_eval", percent_threshold=0.8), [1]),
#             ]
#         ),
#     },
# )

# gym.register(
#     id="DROID-OrganizeTools",
#     entry_point=ManagerBasedRLSplatEnv,
#     disable_env_checker=True,
#     order_enforce=False,
#     kwargs={
#         "env_cfg_entry_point": DroidCfg,
#         "usd_file": str(DATA_PATH / "organize_tools/scene.usda"),
#         "rubric": Rubric(
#             criteria=[
#                 checkers.reach("scissor", threshold=0.2),
#                 (checkers.lift("scissor", threshold=0.04), [0]),
#                 (checkers.is_within_xy("scissor", "container_01", percent_threshold=0.8), [1]),
#             ]
#         ),
#     },
# )

# gym.register(
#     id="DROID-TapeIntoContainer",
#     entry_point=ManagerBasedRLSplatEnv,
#     disable_env_checker=True,
#     order_enforce=False,
#     kwargs={
#         "env_cfg_entry_point": DroidCfg,
#         "usd_file": str(DATA_PATH / "tape_into_container/scene.usda"),
#         "rubric": Rubric(
#             criteria=[
#                 checkers.reach("tape_00", threshold=0.2),
#                 (checkers.lift("tape_00", threshold=0.04), [0]),
#                 (checkers.is_within_xy("tape_00", "container_02", percent_threshold=0.8), [1]),
#             ]

from . import mila  # noqa: F401, E402
