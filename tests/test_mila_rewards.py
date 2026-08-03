from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
REWARDS_PATH = ROOT / "src" / "sim_evals" / "environments" / "mila" / "rewards.py"


def _load_rewards_module(task):
    package_name = "mila_reward_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REWARDS_PATH.parent)]
    sys.modules[package_name] = package

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_envs = types.ModuleType("isaaclab.envs")
    isaaclab_envs.ManagerBasedRLEnv = object
    isaaclab.envs = isaaclab_envs
    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.envs"] = isaaclab_envs

    geometry = types.ModuleType(f"{package_name}.geometry")
    geometry.get_object_pos = lambda env: torch.zeros((env.num_envs, 3))
    for name in (
        "get_gripper_open",
        "get_gripper_pos",
        "get_object_point_pos",
        "get_target_point_pos",
        "get_target_pos",
        "object_keypoints_inside_target_region",
        "object_point_above_or_inside_target_region",
        "object_point_inside_target_region",
        "target_is_upright",
    ):
        setattr(geometry, name, lambda *args, **kwargs: None)
    sys.modules[geometry.__name__] = geometry

    tasks = types.ModuleType(f"{package_name}.tasks")
    tasks.get_mila_task = lambda task_id: task
    sys.modules[tasks.__name__] = tasks

    module_name = f"{package_name}.rewards"
    spec = importlib.util.spec_from_file_location(module_name, REWARDS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {REWARDS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MilaRewardTest(unittest.TestCase):
    def test_final_reward_and_success_require_consecutive_valid_steps(self):
        task = SimpleNamespace(
            task_id="test_task",
            milestones=(
                SimpleNamespace(name="reached"),
                SimpleNamespace(name="placed"),
            ),
            final_milestone="placed",
            success_hold_steps=3,
        )
        rewards = _load_rewards_module(task)
        env = SimpleNamespace(
            cfg=SimpleNamespace(task_id=task.task_id),
            num_envs=1,
            device=torch.device("cpu"),
            step_dt=0.1,
        )
        final_checks = iter((False, True, True, False, True, True, True))
        rewards._milestone_checks = lambda env, task_id: {
            "reached": torch.tensor([True]),
            "placed": torch.tensor([next(final_checks)]),
        }

        step_rewards = []
        success = []
        for _ in range(7):
            step_rewards.append(float(rewards.milestone_reward(env, task.task_id)[0]))
            success.append(bool(rewards.task_success_reached(env, task.task_id)[0]))

        self.assertEqual(step_rewards, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        self.assertEqual(success, [False, False, False, False, False, False, True])

        rewards.reset_milestones(env, torch.tensor([0]))
        self.assertFalse(bool(rewards.task_success_reached(env, task.task_id)[0]))
        self.assertEqual(int(env._mila_final_success_hold_count[0]), 0)


if __name__ == "__main__":
    unittest.main()
