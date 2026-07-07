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
    return points_inside_region & _object_mouth_above_bottom(env, task, object_points_w)


def get_object_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
    return env.scene[task.object_name].data.root_pos_w


def get_target_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    task = get_mila_task(env.cfg.task_id)
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
    return env.scene[task.target_name].data.root_quat_w


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
