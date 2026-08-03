from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs" / "mila"


@dataclass(frozen=True)
class Milestone:
    name: str
    description: str


@dataclass(frozen=True)
class SpawnBounds:
    """Robot-frame reset offsets around an S3-matched nominal pose."""

    x: tuple[float, float] = (0.0, 0.0)
    y: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class SpawnConstraint:
    kind: str
    asset_a: str
    asset_b: str
    minimum_distance: float | None = None
    asset_a_points: tuple[tuple[float, float, float], ...] = ()
    asset_b_radius: float | None = None
    clearance: float = 0.0


@dataclass(frozen=True)
class CameraCfg:
    inherited: bool
    pos: tuple[float, float, float] | None = None
    rot: tuple[float, float, float, float] | None = None
    focal_length: float | None = None
    horizontal_aperture: float | None = None
    vertical_aperture: float | None = None


@dataclass(frozen=True)
class DomeLightCfg:
    intensity: float
    color: tuple[float, float, float]
    texture_file: str
    visible_in_primary_ray: bool


@dataclass(frozen=True)
class SphereLightCfg:
    enabled: bool
    intensity: float
    color: tuple[float, float, float]
    radius: float
    pos: tuple[float, float, float]


@dataclass(frozen=True)
class MilaTask:
    institution: str
    task_id: str
    scene_asset: str
    language_instruction: str
    success_criteria: str
    max_timesteps: int
    seed: int
    success_hold_steps: int
    milestones: tuple[Milestone, ...]
    final_milestone: str
    object_name: str
    target_name: str
    object_prim_path: str
    target_prim_path: str | None
    robot_base_pos: tuple[float, float, float]
    robot_base_rot: tuple[float, float, float, float]
    robot_init_joint_pos: tuple[float, ...]
    object_init_pos_robot: tuple[float, float, float]
    object_init_rot_robot: tuple[float, float, float, float]
    target_init_pos_robot: tuple[float, float, float] | None
    target_init_rot_robot: tuple[float, float, float, float] | None
    object_spawn_bounds: SpawnBounds | None
    target_spawn_bounds: SpawnBounds | None
    spawn_constraints: tuple[SpawnConstraint, ...]
    cameras: dict[str, CameraCfg]
    dome_light: DomeLightCfg
    sphere_light: SphereLightCfg
    fixed_target_pos_robot: tuple[float, float, float] | None
    fixed_target_rot_robot: tuple[float, float, float, float]
    hidden_prim_paths: tuple[str, ...]
    target_top_center: tuple[float, float, float] | None
    target_bottom_center: tuple[float, float, float] | None
    object_bottom_center: tuple[float, float, float] | None
    object_mouth_center: tuple[float, float, float] | None
    object_region_points: tuple[tuple[float, float, float], ...]
    target_region_prim_path: str | None
    target_region_extent_min: tuple[float, float, float] | None
    target_region_extent_max: tuple[float, float, float] | None
    target_region_tolerance: float
    reach_distance: float
    lift_height: float
    target_xy_radius: float
    above_target_height: float
    in_target_height: float
    in_target_xy_radius: float | None
    release_gripper_threshold: float
    config_path: Path


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _vector(value: Any, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} numbers")
    result = tuple(float(item) for item in value)
    return result


def _optional_vector(value: Any, length: int, field: str):
    return None if value is None else _vector(value, length, field)


def _range(value: Any, field: str) -> tuple[float, float]:
    result = _vector(value, 2, field)
    if result[0] > result[1]:
        raise ValueError(f"{field} minimum must not exceed maximum")
    return result


def _spawn_bounds(raw: Any, field: str) -> SpawnBounds | None:
    if raw is None:
        return None
    data = _mapping(raw, field)
    translation = _mapping(data.get("translation_m", {}), f"{field}.translation_m")
    yaw_degrees = _range(data.get("yaw_deg", (0.0, 0.0)), f"{field}.yaw_deg")
    degrees_to_radians = 3.141592653589793 / 180.0
    return SpawnBounds(
        x=_range(translation.get("x", (0.0, 0.0)), f"{field}.translation_m.x"),
        y=_range(translation.get("y", (0.0, 0.0)), f"{field}.translation_m.y"),
        yaw=(yaw_degrees[0] * degrees_to_radians, yaw_degrees[1] * degrees_to_radians),
    )


def _constraint(raw: Any, field: str) -> SpawnConstraint:
    data = _mapping(raw, field)
    kind = str(data["type"])
    assets = _vector_names(data.get("assets"), 2, f"{field}.assets")
    if kind == "minimum_planar_distance":
        return SpawnConstraint(
            kind=kind,
            asset_a=assets[0],
            asset_b=assets[1],
            minimum_distance=float(data["minimum_distance_m"]),
        )
    if kind == "segment_circle_clearance":
        points = data.get("segment_points_local_m")
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError(f"{field}.segment_points_local_m must contain two points")
        return SpawnConstraint(
            kind=kind,
            asset_a=assets[0],
            asset_b=assets[1],
            asset_a_points=tuple(
                _vector(point, 3, f"{field}.segment_points_local_m") for point in points
            ),
            asset_b_radius=float(data["circle_radius_m"]),
            clearance=float(data.get("clearance_m", 0.0)),
        )
    raise ValueError(f"Unsupported spawn constraint type: {kind}")


def _vector_names(value: Any, length: int, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} asset names")
    return tuple(str(item) for item in value)


def _camera(raw: Any, field: str) -> CameraCfg:
    data = _mapping(raw, field)
    inherited = bool(data.get("inherited", False))
    if inherited:
        return CameraCfg(inherited=True)
    return CameraCfg(
        inherited=False,
        pos=_vector(data["pos_robot"], 3, f"{field}.pos_robot"),
        rot=_vector(data["rot_wxyz_robot"], 4, f"{field}.rot_wxyz_robot"),
        focal_length=float(data["focal_length"]),
        horizontal_aperture=float(data["horizontal_aperture"]),
        vertical_aperture=float(data["vertical_aperture"]),
    )


def load_mila_task(path: str | Path) -> MilaTask:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = _mapping(raw, str(config_path))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported MILA config schema in {config_path}")
    if data.get("randomization_scope") != "movable_task_objects_only":
        raise ValueError(f"{config_path} must use movable_task_objects_only randomization")

    robot = _mapping(data["robot"], "robot")
    actors = _mapping(data["actors"], "actors")
    object_data = _mapping(actors["object"], "actors.object")
    target_data = _mapping(actors["target"], "actors.target")
    randomization = _mapping(data["randomization"], "randomization")
    random_assets = _mapping(randomization.get("assets", {}), "randomization.assets")
    geometry = _mapping(data.get("geometry", {}), "geometry")
    thresholds = _mapping(data["thresholds"], "thresholds")
    lighting = _mapping(data["lighting"], "lighting")
    dome = _mapping(lighting["dome"], "lighting.dome")
    sphere = _mapping(lighting["sphere"], "lighting.sphere")
    camera_data = _mapping(data["cameras"], "cameras")
    milestones = tuple(
        Milestone(name=str(item["name"]), description=str(item["description"]))
        for item in data["milestones"]
    )
    milestone_names = tuple(item.name for item in milestones)
    final_milestone = str(data["final_milestone"])
    if not milestones or len(milestone_names) != len(set(milestone_names)):
        raise ValueError(f"{config_path} must define unique milestones")
    if final_milestone != milestone_names[-1]:
        raise ValueError(f"{config_path} final_milestone must be the final ordered milestone")

    object_name = str(object_data["name"])
    target_name = str(target_data["name"])
    allowed_assets = {object_name, target_name}
    unknown_random_assets = set(random_assets) - allowed_assets
    if unknown_random_assets:
        raise ValueError(f"Unknown randomized assets in {config_path}: {sorted(unknown_random_assets)}")

    cameras = {name: _camera(item, f"cameras.{name}") for name, item in camera_data.items()}
    if set(cameras) != {"external_cam", "external_cam_2", "wrist_cam"}:
        raise ValueError(f"{config_path} must configure exactly the three MILA cameras")

    task = MilaTask(
        institution=str(data["institution"]),
        task_id=str(data["task_id"]),
        scene_asset=str(data["scene_asset"]),
        language_instruction=str(data["language_instruction"]),
        success_criteria=str(data["success_criteria"]),
        max_timesteps=int(data["max_timesteps"]),
        seed=int(data["seed"]),
        success_hold_steps=int(data["success_hold_steps"]),
        milestones=milestones,
        final_milestone=final_milestone,
        object_name=object_name,
        target_name=target_name,
        object_prim_path=str(object_data["prim_path"]),
        target_prim_path=(None if target_data.get("prim_path") is None else str(target_data["prim_path"])),
        robot_base_pos=_vector(robot["base_pos_world"], 3, "robot.base_pos_world"),
        robot_base_rot=_vector(robot["base_rot_wxyz_world"], 4, "robot.base_rot_wxyz_world"),
        robot_init_joint_pos=_vector(robot["init_joint_pos"], 7, "robot.init_joint_pos"),
        object_init_pos_robot=_vector(object_data["nominal_pos_robot"], 3, "actors.object.nominal_pos_robot"),
        object_init_rot_robot=_vector(object_data["nominal_rot_wxyz_robot"], 4, "actors.object.nominal_rot_wxyz_robot"),
        target_init_pos_robot=_optional_vector(target_data.get("nominal_pos_robot"), 3, "actors.target.nominal_pos_robot"),
        target_init_rot_robot=_optional_vector(target_data.get("nominal_rot_wxyz_robot"), 4, "actors.target.nominal_rot_wxyz_robot"),
        object_spawn_bounds=_spawn_bounds(random_assets.get(object_name), f"randomization.assets.{object_name}"),
        target_spawn_bounds=_spawn_bounds(random_assets.get(target_name), f"randomization.assets.{target_name}"),
        spawn_constraints=tuple(
            _constraint(item, f"randomization.constraints[{index}]")
            for index, item in enumerate(randomization.get("constraints", []))
        ),
        cameras=cameras,
        dome_light=DomeLightCfg(
            intensity=float(dome["intensity"]),
            color=_vector(dome["color"], 3, "lighting.dome.color"),
            texture_file=str(dome["texture_file"]),
            visible_in_primary_ray=bool(dome["visible_in_primary_ray"]),
        ),
        sphere_light=SphereLightCfg(
            enabled=bool(sphere["enabled"]),
            intensity=float(sphere["intensity"]),
            color=_vector(sphere["color"], 3, "lighting.sphere.color"),
            radius=float(sphere["radius"]),
            pos=_vector(sphere["pos_world"], 3, "lighting.sphere.pos_world"),
        ),
        fixed_target_pos_robot=_optional_vector(geometry.get("fixed_target_pos_robot"), 3, "geometry.fixed_target_pos_robot"),
        fixed_target_rot_robot=_vector(geometry.get("fixed_target_rot_wxyz_robot", (1.0, 0.0, 0.0, 0.0)), 4, "geometry.fixed_target_rot_wxyz_robot"),
        hidden_prim_paths=tuple(str(item) for item in data.get("hidden_prim_paths", [])),
        target_top_center=_optional_vector(geometry.get("target_top_center"), 3, "geometry.target_top_center"),
        target_bottom_center=_optional_vector(geometry.get("target_bottom_center"), 3, "geometry.target_bottom_center"),
        object_bottom_center=_optional_vector(geometry.get("object_bottom_center"), 3, "geometry.object_bottom_center"),
        object_mouth_center=_optional_vector(geometry.get("object_mouth_center"), 3, "geometry.object_mouth_center"),
        object_region_points=tuple(
            _vector(point, 3, f"geometry.object_region_points[{index}]")
            for index, point in enumerate(geometry.get("object_region_points", []))
        ),
        target_region_prim_path=(None if geometry.get("target_region_prim_path") is None else str(geometry["target_region_prim_path"])),
        target_region_extent_min=_optional_vector(geometry.get("target_region_extent_min"), 3, "geometry.target_region_extent_min"),
        target_region_extent_max=_optional_vector(geometry.get("target_region_extent_max"), 3, "geometry.target_region_extent_max"),
        target_region_tolerance=float(geometry.get("target_region_tolerance", 0.01)),
        reach_distance=float(thresholds["reach_distance"]),
        lift_height=float(thresholds["lift_height"]),
        target_xy_radius=float(thresholds["target_xy_radius"]),
        above_target_height=float(thresholds["above_target_height"]),
        in_target_height=float(thresholds["in_target_height"]),
        in_target_xy_radius=(None if thresholds.get("in_target_xy_radius") is None else float(thresholds["in_target_xy_radius"])),
        release_gripper_threshold=float(thresholds["release_gripper_threshold"]),
        config_path=config_path,
    )
    if task.seed != 42:
        raise ValueError(f"{config_path} must use the approved base seed 42")
    if task.success_hold_steps < 1:
        raise ValueError(f"{config_path} success_hold_steps must be positive")
    if task.scene_asset != "Task1/Scene.usd":
        raise ValueError(f"{config_path} must use the replacement Task1 scene")
    for constraint in task.spawn_constraints:
        if constraint.asset_a not in allowed_assets or constraint.asset_b not in allowed_assets:
            raise ValueError(f"Unknown constrained asset in {config_path}: {constraint}")
    return task


def load_mila_tasks(config_dir: str | Path = CONFIG_DIR) -> dict[str, MilaTask]:
    directory = Path(config_dir)
    tasks: dict[str, MilaTask] = {}
    for path in sorted(directory.glob("*.yaml")):
        task = load_mila_task(path)
        if task.task_id in tasks:
            raise ValueError(f"Duplicate MILA task ID: {task.task_id}")
        tasks[task.task_id] = task
    if not tasks:
        raise ValueError(f"No MILA task YAML files found in {directory}")
    return tasks


MILA_TASKS = load_mila_tasks()


def get_mila_task(task_id: str) -> MilaTask:
    try:
        return MILA_TASKS[task_id]
    except KeyError as exc:
        supported_tasks = ", ".join(sorted(MILA_TASKS))
        raise ValueError(f"Unsupported MILA task '{task_id}'. Supported tasks: {supported_tasks}") from exc
