from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .tasks import get_mila_task


def object_keypoints_inside_target_region(env: ManagerBasedRLEnv, task) -> torch.Tensor:
    if task.target_region_extent_min is None or task.target_region_extent_max is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    object_pos = get_object_pos(env)
    object_quat = get_object_quat(env)
    target_pos = get_target_pos(env)
    target_quat = get_target_quat(env)

    object_points = _object_containment_points(task, device=env.device, dtype=object_pos.dtype)
    if object_points is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    object_points_w = object_pos[:, None, :] + quat_rotate(
        object_quat[:, None, :],
        object_points[None, :, :],
    )
    object_points_target = quat_rotate(
        quat_conjugate(target_quat)[:, None, :],
        object_points_w - target_pos[:, None, :],
    )

    tolerance = task.target_region_tolerance
    region_min = (
        torch.tensor(task.target_region_extent_min, device=env.device, dtype=object_pos.dtype)
        - tolerance
    )
    region_max = (
        torch.tensor(task.target_region_extent_max, device=env.device, dtype=object_pos.dtype)
        + tolerance
    )

    inside_per_axis = (object_points_target >= region_min) & (object_points_target <= region_max)
    points_inside_region = inside_per_axis.all(dim=-1).all(dim=-1)
    orientation_valid = (
        torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        if task.object_region_points
        else _object_mouth_above_bottom(env, task, object_points_w)
    )
    return points_inside_region & orientation_valid


def object_point_inside_target_region(
    env: ManagerBasedRLEnv,
    task,
    local_point: tuple[float, float, float],
) -> torch.Tensor:
    """Return whether one object-local point lies in the authored target region."""
    point_target = _object_point_in_target_frame(env, task, local_point)
    if point_target is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    region_min, region_max = _target_region_bounds(env, task, point_target.dtype)
    return ((point_target >= region_min) & (point_target <= region_max)).all(dim=-1)


def object_point_above_or_inside_target_region(
    env: ManagerBasedRLEnv,
    task,
    local_point: tuple[float, float, float],
) -> torch.Tensor:
    """Return whether a point is horizontally aligned and not below the region."""
    point_target = _object_point_in_target_frame(env, task, local_point)
    if point_target is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    region_min, region_max = _target_region_bounds(env, task, point_target.dtype)
    inside_xy = (
        (point_target[:, :2] >= region_min[:2])
        & (point_target[:, :2] <= region_max[:2])
    ).all(dim=-1)
    return inside_xy & (point_target[:, 2] >= region_min[2])


def target_is_upright(env: ManagerBasedRLEnv, minimum_up_z: float = 0.9) -> torch.Tensor:
    """Require the target's local up axis to remain close to world up."""
    target_quat = get_target_quat(env)
    local_up = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=target_quat.dtype)
    target_up_w = quat_rotate(target_quat, local_up.expand(target_quat.shape[0], -1))
    return target_up_w[:, 2] >= minimum_up_z


def get_object_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
    return env.scene[task.object_name].data.root_pos_w


def get_target_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
    if task.fixed_target_pos_robot is not None:
        return _fixed_target_pos(env, task)
    return env.scene[task.target_name].data.root_pos_w


def get_object_point_pos(env: ManagerBasedRLEnv, local_point: tuple[float, float, float]) -> torch.Tensor:
    object_pos = get_object_pos(env)
    object_quat = get_object_quat(env)
    point = torch.tensor(local_point, device=env.device, dtype=object_pos.dtype)
    return object_pos + quat_rotate(object_quat, point.expand_as(object_pos))


def get_target_point_pos(env: ManagerBasedRLEnv, local_point: tuple[float, float, float]) -> torch.Tensor:
    target_pos = get_target_pos(env)
    target_quat = get_target_quat(env)
    point = torch.tensor(local_point, device=env.device, dtype=target_pos.dtype)
    return target_pos + quat_rotate(target_quat, point.expand_as(target_pos))


def get_gripper_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    candidate_names = ("base_link", "panda_hand", "panda_link8")
    for name in candidate_names:
        if name in robot.data.body_names:
            body_id = robot.data.body_names.index(name)
            return robot.data.body_pos_w[:, body_id]
    return robot.data.root_pos_w


def get_gripper_open(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_id = robot.data.joint_names.index("finger_joint")
    gripper_pos = robot.data.joint_pos[:, joint_id] / (torch.pi / 4)
    return gripper_pos < threshold


def get_object_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
    return env.scene[task.object_name].data.root_quat_w


def get_target_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
    if task.fixed_target_pos_robot is not None:
        quat = _quat_multiply(task.robot_base_rot, task.fixed_target_rot_robot)
        return torch.tensor(quat, device=env.device, dtype=get_object_pos(env).dtype).repeat(env.num_envs, 1)
    return env.scene[task.target_name].data.root_quat_w


def _fixed_target_pos(env: ManagerBasedRLEnv, task) -> torch.Tensor:
    pos = _transform_pos(task.robot_base_pos, task.robot_base_rot, task.fixed_target_pos_robot)
    return torch.tensor(pos, device=env.device, dtype=get_object_pos(env).dtype).repeat(env.num_envs, 1)


def _object_point_in_target_frame(env: ManagerBasedRLEnv, task, local_point):
    if task.target_region_extent_min is None or task.target_region_extent_max is None:
        return None
    point_w = get_object_point_pos(env, local_point)
    return quat_rotate(
        quat_conjugate(get_target_quat(env)),
        point_w - get_target_pos(env),
    )


def _target_region_bounds(env: ManagerBasedRLEnv, task, dtype: torch.dtype):
    tolerance = task.target_region_tolerance
    region_min = torch.tensor(task.target_region_extent_min, device=env.device, dtype=dtype) - tolerance
    region_max = torch.tensor(task.target_region_extent_max, device=env.device, dtype=dtype) + tolerance
    return region_min, region_max


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _transform_pos(base_pos, base_rot, local_pos):
    q = torch.tensor(base_rot, dtype=torch.float64)
    point = torch.tensor(local_pos, dtype=torch.float64)
    rotated = quat_rotate(q, point)
    return tuple(float(base_pos[index] + rotated[index]) for index in range(3))


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] *= -1
    return result


def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_xyz, vec = torch.broadcast_tensors(quat[..., 1:], vec)
    quat_w = quat[..., :1].expand(vec.shape[:-1] + (1,))
    uv = torch.cross(quat_xyz, vec, dim=-1)
    uuv = torch.cross(quat_xyz, uv, dim=-1)
    return vec + 2.0 * (quat_w * uv + uuv)


def _object_containment_points(
    task,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if task.object_region_points:
        return torch.tensor(task.object_region_points, device=device, dtype=dtype)
    if task.object_bottom_center is None or task.object_mouth_center is None:
        return None
    return torch.tensor(
        (task.object_bottom_center, task.object_mouth_center),
        device=device,
        dtype=dtype,
    )


def _object_mouth_above_bottom(env: ManagerBasedRLEnv, task, object_points_w: torch.Tensor) -> torch.Tensor:
    if task.object_bottom_center is None or task.object_mouth_center is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    bottom_center_z = object_points_w[:, 0, 2]
    mouth_center_z = object_points_w[:, 1, 2]
    return mouth_center_z > bottom_center_z
