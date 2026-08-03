from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from sim_evals.environments.droid_environment import (
    DATA_PATH,
    EnvCfg as DroidEnvCfg,
    EventCfg as DroidEventCfg,
    SceneCfg as DroidSceneCfg,
)

from .rewards import milestone_reward, reset_milestones
from .tasks import CameraCfg, MilaTask, SpawnBounds, SpawnConstraint, get_mila_task


MILA_ASSET_PATH = DATA_PATH / "mila"
DEFAULT_TASK_ID = "place_spoon_in_sink"

def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_rotate(q, v):
    q_conj = (q[0], -q[1], -q[2], -q[3])
    rotated = _quat_multiply(_quat_multiply(q, (0.0, *v)), q_conj)
    return rotated[1:]


def _transform_pos(base_pos, base_rot, local_pos):
    rotated = _quat_rotate(base_rot, local_pos)
    return tuple(base_pos[i] + rotated[i] for i in range(3))


def _set_xform_pose(prim: Usd.Prim, pos, rot):
    """Author one local pose op, replacing referenced wrapper xform ops."""
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    pose_op = xformable.AddTransformOp(
        precision=UsdGeom.XformOp.PrecisionDouble,
        opSuffix="milaTaskPose",
    )
    transform = Gf.Transform()
    transform.SetTranslation(Gf.Vec3d(*pos))
    transform.SetRotation(Gf.Rotation(Gf.Quatd(rot[0], Gf.Vec3d(*rot[1:]))))
    pose_op.Set(transform.GetMatrix())


def _spawn_mila_scene(
    prim_path: str,
    cfg,
    translation=None,
    orientation=None,
    *,
    task_id: str,
    **kwargs,
):
    """Compose Arhan's scene and align nested movable wrappers pre-physics."""
    prim = sim_utils.spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    task = get_mila_task(task_id)
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    actor_specs = (
        (task.object_prim_path, task.object_init_pos_robot, task.object_init_rot_robot),
        (task.target_prim_path, task.target_init_pos_robot, task.target_init_rot_robot),
    )
    for actor_path, pos_robot, rot_robot in actor_specs:
        if actor_path is None or pos_robot is None or rot_robot is None:
            continue
        wrapper_name = actor_path.split("/", 1)[0]
        pos = _transform_pos(task.robot_base_pos, task.robot_base_rot, pos_robot)
        rot = _quat_multiply(task.robot_base_rot, rot_robot)
        matches = [
            candidate
            for candidate in stage.Traverse()
            if str(candidate.GetPath()).startswith("/World/envs/env_")
            and str(candidate.GetPath()).endswith(f"/scene/{wrapper_name}")
        ]
        if not matches:
            raise RuntimeError(f"MILA movable-actor wrapper is missing: {wrapper_name}")
        for wrapper_prim in matches:
            _set_xform_pose(wrapper_prim, pos, rot)
            print(
                f"Aligned MILA movable-actor wrapper before physics: "
                f"{wrapper_prim.GetPath()} pos={pos} rot={rot}",
                flush=True,
            )
    return prim


def configure_mila_task_scene(env, env_ids, task_id: str):
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    task = get_mila_task(task_id)
    for env_index in range(env.num_envs):
        scene_root = f"/World/envs/env_{env_index}/scene"
        for relative_path in task.hidden_prim_paths:
            prim = stage.GetPrimAtPath(f"{scene_root}/{relative_path}")
            if prim:
                prim.SetActive(False)


def _sample_rigid_asset_pose(env, env_ids, asset_name: str, robot_base_rot, bounds: SpawnBounds):
    asset = env.scene[asset_name]
    root_states = asset.data.default_root_state[env_ids].clone()
    sample_ranges = torch.tensor(
        (bounds.x, bounds.y, bounds.yaw),
        dtype=root_states.dtype,
        device=asset.device,
    )
    samples = math_utils.sample_uniform(
        sample_ranges[:, 0],
        sample_ranges[:, 1],
        (len(env_ids), 3),
        device=asset.device,
    )

    local_offsets = torch.zeros((len(env_ids), 3), dtype=root_states.dtype, device=asset.device)
    local_offsets[:, :2] = samples[:, :2]
    base_rot = torch.tensor(robot_base_rot, dtype=root_states.dtype, device=asset.device).repeat(len(env_ids), 1)
    world_offsets = math_utils.quat_apply(base_rot, local_offsets)

    positions = root_states[:, :3] + env.scene.env_origins[env_ids] + world_offsets
    zeros = torch.zeros(len(env_ids), dtype=root_states.dtype, device=asset.device)
    yaw_delta = math_utils.quat_from_euler_xyz(zeros, zeros, samples[:, 2])
    orientations = math_utils.quat_mul(yaw_delta, root_states[:, 3:7])

    return positions, orientations, root_states


def _default_rigid_asset_pose(env, env_ids, asset_name: str):
    asset = env.scene[asset_name]
    root_states = asset.data.default_root_state[env_ids].clone()
    return (
        root_states[:, :3] + env.scene.env_origins[env_ids],
        root_states[:, 3:7],
        root_states,
    )


def _segment_circle_clearance_valid(
    segment_pos: torch.Tensor,
    segment_quat: torch.Tensor,
    circle_pos: torch.Tensor,
    constraint: SpawnConstraint,
) -> torch.Tensor:
    points_local = torch.tensor(
        constraint.asset_a_points,
        dtype=segment_pos.dtype,
        device=segment_pos.device,
    )
    count = segment_pos.shape[0]
    points_w = segment_pos[:, None, :] + math_utils.quat_apply(
        segment_quat[:, None, :].expand(-1, len(points_local), -1).reshape(-1, 4),
        points_local[None, :, :].expand(count, -1, -1).reshape(-1, 3),
    ).reshape(count, len(points_local), 3)
    start = points_w[:, 0, :2]
    end = points_w[:, 1, :2]
    segment = end - start
    segment_length_squared = (segment * segment).sum(dim=-1).clamp_min(1e-12)
    projection = (((circle_pos[:, :2] - start) * segment).sum(dim=-1) / segment_length_squared).clamp(0.0, 1.0)
    closest = start + projection[:, None] * segment
    distance = torch.linalg.norm(circle_pos[:, :2] - closest, dim=-1)
    required = float(constraint.asset_b_radius or 0.0) + constraint.clearance
    return distance >= required


def _spawn_constraints_valid(candidate_poses, constraints) -> torch.Tensor:
    first_position = next(iter(candidate_poses.values()))[0]
    valid = torch.ones(first_position.shape[0], dtype=torch.bool, device=first_position.device)
    for constraint in constraints:
        pos_a, quat_a, _state_a = candidate_poses[constraint.asset_a]
        pos_b, _quat_b, _state_b = candidate_poses[constraint.asset_b]
        if constraint.kind == "minimum_planar_distance":
            valid &= torch.linalg.norm(pos_a[:, :2] - pos_b[:, :2], dim=-1) >= float(
                constraint.minimum_distance or 0.0
            )
        elif constraint.kind == "segment_circle_clearance":
            valid &= _segment_circle_clearance_valid(pos_a, quat_a, pos_b, constraint)
        else:
            raise ValueError(f"Unsupported spawn constraint: {constraint.kind}")
    return valid


def randomize_mila_task_assets(env, env_ids: torch.Tensor, task_id: str):
    task = get_mila_task(task_id)
    randomized_bounds = {
        name: bounds
        for name, bounds in (
            (task.object_name, task.object_spawn_bounds),
            (task.target_name, task.target_spawn_bounds),
        )
        if bounds is not None
    }
    if not randomized_bounds:
        return

    involved_assets = set(randomized_bounds)
    for constraint in task.spawn_constraints:
        involved_assets.update((constraint.asset_a, constraint.asset_b))
    involved_asset_names = sorted(involved_assets)

    accepted = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
    accepted_poses = {}
    for _attempt in range(128):
        candidate_poses = {
            asset_name: (
                _sample_rigid_asset_pose(
                    env,
                    env_ids,
                    asset_name,
                    task.robot_base_rot,
                    randomized_bounds[asset_name],
                )
                if asset_name in randomized_bounds
                else _default_rigid_asset_pose(env, env_ids, asset_name)
            )
            for asset_name in involved_asset_names
        }
        valid = _spawn_constraints_valid(candidate_poses, task.spawn_constraints)
        newly_accepted = valid & ~accepted
        for asset_name in randomized_bounds:
            positions, orientations, root_states = candidate_poses[asset_name]
            if asset_name not in accepted_poses:
                accepted_poses[asset_name] = (
                    positions.clone(),
                    orientations.clone(),
                    root_states.clone(),
                )
            else:
                accepted_poses[asset_name][0][newly_accepted] = positions[newly_accepted]
                accepted_poses[asset_name][1][newly_accepted] = orientations[newly_accepted]
                accepted_poses[asset_name][2][newly_accepted] = root_states[newly_accepted]
            if _attempt == 0:
                accepted_poses[asset_name][0][~newly_accepted] = positions[~newly_accepted]
                accepted_poses[asset_name][1][~newly_accepted] = orientations[~newly_accepted]
                accepted_poses[asset_name][2][~newly_accepted] = root_states[~newly_accepted]
        accepted |= valid
        if bool(accepted.all()):
            break
    if not bool(accepted.all()):
        raise RuntimeError(f"Could not sample valid {task_id} object spawns after 128 attempts")

    for asset_name, (positions, orientations, root_states) in accepted_poses.items():
        asset = env.scene[asset_name]
        asset.write_root_pose_to_sim(
            torch.cat((positions, orientations), dim=-1), env_ids=env_ids
        )
        asset.write_root_velocity_to_sim(
            torch.zeros_like(root_states[:, 7:13]), env_ids=env_ids
        )


def _make_dome_light(task: MilaTask):
    return AssetBaseCfg(
        prim_path="/World/MilaPolicyDomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=task.dome_light.intensity,
            color=task.dome_light.color,
            texture_file=str(MILA_ASSET_PATH / task.dome_light.texture_file),
            texture_format="latlong",
            visible_in_primary_ray=task.dome_light.visible_in_primary_ray,
        ),
    )


def _make_sphere_light(task: MilaTask):
    if not task.sphere_light.enabled:
        return None
    return AssetBaseCfg(
        prim_path="/World/MilaOverheadLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=task.sphere_light.intensity,
            color=task.sphere_light.color,
            radius=task.sphere_light.radius,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=task.sphere_light.pos),
    )


def _configured_camera(camera, name: str, config: CameraCfg):
    if config.inherited:
        return camera
    camera = deepcopy(camera)
    camera.prim_path = f"{{ENV_REGEX_NS}}/robot/{name}"
    camera.spawn.focal_length = config.focal_length
    camera.spawn.horizontal_aperture = config.horizontal_aperture
    camera.spawn.vertical_aperture = config.vertical_aperture
    camera.offset.pos = config.pos
    camera.offset.rot = config.rot
    return camera


@configclass
class MilaSceneCfg(DroidSceneCfg):
    # Suppress DroidSceneCfg's always-on sphere light. MILA's optional sphere
    # is instantiated below only when the active task YAML enables it.
    sphere_light: AssetBaseCfg | None = None
    policy_dome_light = _make_dome_light(get_mila_task(DEFAULT_TASK_ID))
    overhead_light: AssetBaseCfg | None = None

    def dynamic_task_scene(self, task_id: str):
        task = get_mila_task(task_id)
        self.policy_dome_light = _make_dome_light(task)
        self.overhead_light = _make_sphere_light(task)
        self._apply_task_robot_frame(task)
        environment_path = MILA_ASSET_PATH / task.scene_asset
        self._load_scene_usd(environment_path, task)

    def _apply_task_robot_frame(self, task: MilaTask):
        self.robot = deepcopy(self.robot)
        self.robot.init_state.pos = task.robot_base_pos
        self.robot.init_state.rot = task.robot_base_rot
        if task.robot_init_joint_pos is not None:
            for joint_index, joint_position in enumerate(task.robot_init_joint_pos, start=1):
                self.robot.init_state.joint_pos[f"panda_joint{joint_index}"] = joint_position

        self.external_cam = _configured_camera(
            self.external_cam, "external_cam", task.cameras["external_cam"]
        )
        self.external_cam_2 = _configured_camera(
            self.external_cam_2, "external_cam_2", task.cameras["external_cam_2"]
        )

    def _load_scene_usd(self, environment_path: Path, task: MilaTask):
        scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(environment_path),
                func=partial(_spawn_mila_scene, task_id=task.task_id),
            ),
        )
        self.scene = scene

        stage = Usd.Stage.Open(str(environment_path))
        if stage is None:
            raise FileNotFoundError(f"Could not open MILA scene asset: {environment_path}")

        self._register_rigid_body(
            stage,
            task,
            task.object_name,
            task.object_prim_path,
            task.object_init_pos_robot,
            task.object_init_rot_robot,
        )
        if task.target_prim_path is not None:
            self._register_rigid_body(
                stage,
                task,
                task.target_name,
                task.target_prim_path,
                task.target_init_pos_robot,
                task.target_init_rot_robot,
            )

    def _register_rigid_body(
        self,
        stage: Usd.Stage,
        task: MilaTask,
        alias: str,
        relative_prim_path: str,
        init_pos_robot: tuple[float, float, float] | None,
        init_rot_robot: tuple[float, float, float, float] | None,
    ):
        prim = stage.GetPrimAtPath(f"/World/{relative_prim_path}")
        if not prim:
            raise ValueError(f"MILA scene is missing prim '/World/{relative_prim_path}' for alias '{alias}'")
        if not UsdPhysics.RigidBodyAPI(prim):
            raise ValueError(f"MILA prim '/World/{relative_prim_path}' for alias '{alias}' is not a rigid body")

        print(f"Found rigid body: {alias} -> {relative_prim_path}")
        world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pos = tuple(world_transform.ExtractTranslation())
        quat = world_transform.ExtractRotationQuat()
        imag = quat.GetImaginary()
        rot = (quat.GetReal(), imag[0], imag[1], imag[2])
        if init_pos_robot is not None:
            pos = _transform_pos(task.robot_base_pos, task.robot_base_rot, init_pos_robot)
        if init_rot_robot is not None:
            rot = _quat_multiply(task.robot_base_rot, init_rot_robot)

        asset = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/scene/{relative_prim_path}",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        )
        setattr(self, alias, asset)


@configclass
class MilaRewardsCfg:
    milestones = RewTerm(
        func=milestone_reward,
        weight=1.0,
        params={"task_id": DEFAULT_TASK_ID},
    )


@configclass
class MilaEventCfg(DroidEventCfg):
    configure_task_scene = EventTerm(
        func=configure_mila_task_scene,
        mode="startup",
        params={"task_id": DEFAULT_TASK_ID},
    )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_task_assets = EventTerm(
        func=randomize_mila_task_assets,
        mode="reset",
        params={"task_id": DEFAULT_TASK_ID},
    )
    reset_milestones = EventTerm(func=reset_milestones, mode="reset")


@configclass
class MilaDroidEnvCfg(DroidEnvCfg):
    scene = MilaSceneCfg(num_envs=1, env_spacing=7.0)
    rewards = MilaRewardsCfg()
    events = MilaEventCfg()
    task_id = DEFAULT_TASK_ID

    def __post_init__(self):
        super().__post_init__()
        self.wait_for_textures = True
        self.rerender_on_reset = True

        import carb

        settings = carb.settings.get_settings()
        settings.set("rtx.post.tonemap.op", 7)
        settings.set("rtx/post/tonemap/irayReinhard/crushBlacks", 0.2)
        settings.set("rtx/post/tonemap/irayReinhard/burnHighlights", 0.1)
        settings.set("rtx/post/tonemap/enableSrgbToGamma", False)
        settings.set("rtx/post/tonemap/cm2Factor", 1.2)
        settings.set("rtx/raytracing/fractionalCutoutOpacity", True)

    def set_task(self, task_id: str):
        task = get_mila_task(task_id)
        self.task_id = task.task_id
        self.scene.dynamic_task_scene(task.task_id)
        self.rewards.milestones.params["task_id"] = task.task_id
        self.events.configure_task_scene.params["task_id"] = task.task_id
        self.events.randomize_task_assets.params["task_id"] = task.task_id
        self.seed = task.seed
        self.episode_length_s = task.max_timesteps * self.sim.dt * self.decimation
