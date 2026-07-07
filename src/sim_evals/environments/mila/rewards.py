from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .tasks import get_mila_task


def reset_milestones(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    _ensure_mila_buffers(env)
    env._mila_milestones_reached[env_ids] = False
    env._mila_initial_object_z[env_ids] = _object_pos(env)[:, 2][env_ids]


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
    env._mila_initial_object_z = _object_pos(env)[:, 2].clone()


def _milestone_checks(env: ManagerBasedRLEnv, task_id: str) -> dict[str, torch.Tensor]:
    task = get_mila_task(task_id)
    object_pos = _object_pos(env)
    target_pos = _target_pos(env)
    gripper_pos = _gripper_pos(env)
    gripper_open = _gripper_open(env, task.release_gripper_threshold)

    object_to_gripper = torch.linalg.norm(object_pos - gripper_pos, dim=1)
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
    object_inside_target_region = _object_bbox_inside_target_region(env, task)

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


def _object_bbox_inside_target_region(env: ManagerBasedRLEnv, task) -> torch.Tensor:
    if (
        task.object_bbox_half_extents is None
        or task.target_region_extent_min is None
        or task.target_region_extent_max is None
    ):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    object_pos = _object_pos(env)
    object_quat = _object_quat(env)
    target_pos = _target_pos(env)
    target_quat = _target_quat(env)

    bbox_points = _bbox_corner_points(
        task.object_bbox_half_extents,
        device=env.device,
        dtype=object_pos.dtype,
    )
    object_points_w = object_pos[:, None, :] + _quat_rotate(object_quat[:, None, :], bbox_points[None, :, :])
    object_points_target = _quat_rotate(_quat_conjugate(target_quat)[:, None, :], object_points_w - target_pos[:, None, :])

    tolerance = task.target_region_tolerance
    region_min = torch.tensor(task.target_region_extent_min, device=env.device, dtype=object_pos.dtype) - tolerance
    region_max = torch.tensor(task.target_region_extent_max, device=env.device, dtype=object_pos.dtype) + tolerance

    inside_per_axis = (object_points_target >= region_min) & (object_points_target <= region_max)
    return inside_per_axis.all(dim=-1).all(dim=-1)


def _bbox_corner_points(
    half_extents: tuple[float, float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    hx, hy, hz = half_extents
    return torch.tensor(
        [
            (-hx, -hy, -hz),
            (-hx, -hy, hz),
            (-hx, hy, -hz),
            (-hx, hy, hz),
            (hx, -hy, -hz),
            (hx, -hy, hz),
            (hx, hy, -hz),
            (hx, hy, hz),
        ],
        device=device,
        dtype=dtype,
    )


def _object_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
    return env.scene[task.object_name].data.root_quat_w


def _target_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
    return env.scene[task.target_name].data.root_quat_w


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] *= -1
    return result


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_xyz, vec = torch.broadcast_tensors(quat[..., 1:], vec)
    quat_w = quat[..., :1].expand(vec.shape[:-1] + (1,))
    uv = torch.cross(quat_xyz, vec, dim=-1)
    uuv = torch.cross(quat_xyz, uv, dim=-1)
    return vec + 2.0 * (quat_w * uv + uuv)


def _object_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
    return env.scene[task.object_name].data.root_pos_w


def _target_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
    return env.scene[task.target_name].data.root_pos_w


def _gripper_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    candidate_names = ("base_link", "panda_hand", "panda_link8")
    for name in candidate_names:
        if name in robot.data.body_names:
            body_id = robot.data.body_names.index(name)
            return robot.data.body_pos_w[:, body_id]
    return robot.data.root_pos_w


def _gripper_open(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_id = robot.data.joint_names.index("finger_joint")
    gripper_pos = robot.data.joint_pos[:, joint_id] / (torch.pi / 4)
    return gripper_pos < threshold
