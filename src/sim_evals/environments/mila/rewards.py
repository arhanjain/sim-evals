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
    gripper_open = _gripper_open(env, task.release_gripper_threshold)

    object_to_gripper = _gripper_object_bbox_distance(env, task)
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


def _gripper_object_bbox_distance(env: ManagerBasedRLEnv, task) -> torch.Tensor:
    if task.object_bbox_half_extents is None:
        return torch.linalg.norm(_object_pos(env) - _gripper_pos(env), dim=1)

    fingertip_pos = _gripper_fingertip_pos(env)
    if fingertip_pos is None:
        return torch.linalg.norm(_object_pos(env) - _gripper_pos(env), dim=1)

    object_pos = _object_pos(env)
    object_quat = _object_quat(env)
    fingertip_points_object = _quat_rotate(
        _quat_conjugate(object_quat)[:, None, :],
        fingertip_pos - object_pos[:, None, :],
    )
    half_extents = torch.tensor(
        task.object_bbox_half_extents,
        device=env.device,
        dtype=object_pos.dtype,
    )
    outside_distance = torch.clamp(torch.abs(fingertip_points_object) - half_extents, min=0.0)
    fingertip_to_bbox = torch.linalg.norm(outside_distance, dim=-1)
    return fingertip_to_bbox.min(dim=1).values


def _gripper_fingertip_pos(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    robot = env.scene["robot"]
    candidate_names = (
        "left_inner_finger",
        "right_inner_finger",
        "left_outer_finger",
        "right_outer_finger",
    )
    body_ids = [robot.data.body_names.index(name) for name in candidate_names if name in robot.data.body_names]
    if not body_ids:
        return None
    return robot.data.body_pos_w[:, body_ids, :]


def _object_bbox_inside_target_region(env: ManagerBasedRLEnv, task) -> torch.Tensor:
    if task.target_region_extent_min is None or task.target_region_extent_max is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    object_pos = _object_pos(env)
    object_quat = _object_quat(env)
    target_pos = _target_pos(env)
    target_quat = _target_quat(env)

    object_points = _object_containment_points(task, device=env.device, dtype=object_pos.dtype)
    if object_points is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    object_points_w = object_pos[:, None, :] + _quat_rotate(object_quat[:, None, :], object_points[None, :, :])
    object_points_target = _quat_rotate(_quat_conjugate(target_quat)[:, None, :], object_points_w - target_pos[:, None, :])

    tolerance = task.target_region_tolerance
    region_min = torch.tensor(task.target_region_extent_min, device=env.device, dtype=object_pos.dtype) - tolerance
    region_max = torch.tensor(task.target_region_extent_max, device=env.device, dtype=object_pos.dtype) + tolerance

    inside_per_axis = (object_points_target >= region_min) & (object_points_target <= region_max)
    points_inside_region = inside_per_axis.all(dim=-1).all(dim=-1)
    return points_inside_region & _object_mouth_above_bottom(env, task, object_points_w)


def _object_containment_points(
    task,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if task.object_bottom_center is not None and task.object_mouth_center is not None:
        return torch.tensor((task.object_bottom_center, task.object_mouth_center), device=device, dtype=dtype)
    if task.object_bbox_half_extents is not None:
        return _bbox_corner_points(task.object_bbox_half_extents, device=device, dtype=dtype)
    return None


def _object_mouth_above_bottom(env: ManagerBasedRLEnv, task, object_points_w: torch.Tensor) -> torch.Tensor:
    if task.object_bottom_center is None or task.object_mouth_center is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    bottom_center_z = object_points_w[:, 0, 2]
    mouth_center_z = object_points_w[:, 1, 2]
    return mouth_center_z > bottom_center_z


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
