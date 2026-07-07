from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .geometry import (
    get_gripper_open,
    get_gripper_pos,
    get_object_pos,
    get_target_pos,
    object_keypoints_inside_target_region,
)
from .tasks import get_mila_task


def reset_milestones(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    _ensure_mila_buffers(env)
    env._mila_milestones_reached[env_ids] = False
    env._mila_initial_object_z[env_ids] = get_object_pos(env)[:, 2][env_ids]


def milestone_reward(env: ManagerBasedRLEnv, task_id: str) -> torch.Tensor:
    task = get_mila_task(task_id)
    _ensure_mila_buffers(env)
    checks = _milestone_checks(env, task_id)
    newly_reached_count = torch.zeros(env.num_envs, device=env.device)

    for milestone_index, milestone in enumerate(task.milestones):
        milestone_name = milestone.name
        previous_reached = (
            torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            if milestone_index == 0
            else env._mila_milestones_reached[:, milestone_index - 1]
        )
        reached_now = checks[milestone_name] & previous_reached
        newly_reached = reached_now & ~env._mila_milestones_reached[:, milestone_index]
        env._mila_milestones_reached[:, milestone_index] |= reached_now
        newly_reached_count += newly_reached.float()

    return newly_reached_count / env.step_dt


def _ensure_mila_buffers(env: ManagerBasedRLEnv):
    task = get_mila_task(env.cfg.task_id)
    num_milestones = len(task.milestones)
    if (
        hasattr(env, "_mila_milestones_reached")
        and env._mila_milestones_reached.shape[1] == num_milestones
    ):
        return

    env._mila_milestones_reached = torch.zeros(
        (env.num_envs, num_milestones),
        dtype=torch.bool,
        device=env.device,
    )
    env._mila_initial_object_z = get_object_pos(env)[:, 2].clone()


def _milestone_checks(env: ManagerBasedRLEnv, task_id: str) -> dict[str, torch.Tensor]:
    task = get_mila_task(task_id)
    object_pos = get_object_pos(env)
    target_pos = get_target_pos(env)
    gripper_open = get_gripper_open(env, task.release_gripper_threshold)

    object_to_gripper = torch.linalg.norm(object_pos - get_gripper_pos(env), dim=1)
    object_to_target_xy = torch.linalg.norm(object_pos[:, :2] - target_pos[:, :2], dim=1)
    lifted = object_pos[:, 2] > env._mila_initial_object_z + task.lift_height
    above_target = (
        (object_to_target_xy < task.target_xy_radius)
        & (object_pos[:, 2] > target_pos[:, 2] + task.above_target_height)
    )
    above_target_anywhere = object_pos[:, 2] > target_pos[:, 2] + task.above_target_height
    in_target = (
        (object_to_target_xy < task.target_xy_radius)
        & (object_pos[:, 2] < target_pos[:, 2] + task.in_target_height)
        & gripper_open
    )

    reached_object = object_to_gripper < task.reach_distance
    object_inside_target_region = object_keypoints_inside_target_region(env, task)

    return {
        "reached_spoon": reached_object,
        "lifted_up_spoon": lifted,
        "above_sink": above_target,
        "completely_in_sink": in_target,
        "above_utensil_holder": above_target,
        "released_in_utensil_holder": in_target,
        "reached_red_bowl": reached_object,
        "lifted_up_red_bowl": lifted,
        "above_grey_bowl": above_target_anywhere,
        "placed_in_grey_bowl": object_inside_target_region & gripper_open,
    }
