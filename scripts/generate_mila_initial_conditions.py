"""Generate reproducible MILA initial-condition lists for cross-model evals.

The ranges and rejection rules in this file are generation-only. The runtime
environment enumerates the committed JSON poses and performs no random sampling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "mila"
DEFAULT_SEED = 42
DEFAULT_CONDITION_COUNT = 20

Pose = tuple[float, float, float, float, float, float, float]
Quaternion = tuple[float, float, float, float]
Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class SpawnBounds:
    """Robot-frame offsets around an S3-matched nominal world pose."""

    x: tuple[float, float] = (0.0, 0.0)
    y: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class ActorGenerationSpec:
    nominal_pose_world: Pose
    bounds: SpawnBounds | None = None


@dataclass(frozen=True)
class SpawnConstraint:
    kind: str
    asset_a: str
    asset_b: str
    minimum_distance: float | None = None
    asset_a_points: tuple[Vector3, ...] = ()
    asset_b_radius: float | None = None
    asset_b_pose_world: Pose | None = None
    asset_b_extent_min: Vector3 | None = None
    asset_b_extent_max: Vector3 | None = None
    clearance: float = 0.0


@dataclass(frozen=True)
class TaskGenerationSpec:
    instruction: str
    robot_base_rot_world: Quaternion
    actors: Mapping[str, ActorGenerationSpec]
    constraints: tuple[SpawnConstraint, ...] = ()


SPOON_ROTATION = (
    0.7070936560630797,
    0.004307101615473282,
    0.09899682861478236,
    0.7001425843035376,
)

TASK_SPECS = {
    "place_spoon_in_sink": TaskGenerationSpec(
        instruction="Pick up the spoon and place it in the sink.",
        robot_base_rot_world=(0.0, 0.0, 0.0, 1.0),
        actors={
            "spoon": ActorGenerationSpec(
                nominal_pose_world=(
                    -1.5439088634560734,
                    -0.7703773909792926,
                    0.8908113887519167,
                    *SPOON_ROTATION,
                ),
                bounds=SpawnBounds(
                    y=(-0.07, 0.07),
                    yaw=(-math.pi, math.pi),
                ),
            ),
        },
        constraints=(
            SpawnConstraint(
                kind="segment_aabb_clearance",
                asset_a="spoon",
                asset_b="sink",
                asset_a_points=((0.0, -0.18, 0.0), (0.0, 0.18, 0.0)),
                asset_b_pose_world=(
                    -1.4941656673262144,
                    -1.1148039289647322,
                    0.8058775928746968,
                    0.7071067811865475,
                    0.0,
                    0.0,
                    0.7071067811865476,
                ),
                asset_b_extent_min=(
                    -0.17917540669441223,
                    -0.15094169974327087,
                    -0.10113153606653214,
                ),
                asset_b_extent_max=(
                    0.17917540669441223,
                    0.15094169974327087,
                    0.10113153606653214,
                ),
                clearance=0.015,
            ),
        ),
    ),
    "place_spoon_in_utensil_holder": TaskGenerationSpec(
        instruction="Pick up the spoon and place it in the utensil holder.",
        robot_base_rot_world=(0.0, 0.0, 0.0, 1.0),
        actors={
            "spoon": ActorGenerationSpec(
                nominal_pose_world=(
                    -1.4058683975369632,
                    -0.657037382188836,
                    0.8908113887519138,
                    *SPOON_ROTATION,
                ),
                bounds=SpawnBounds(
                    y=(-0.15, 0.0),
                    yaw=(-math.pi, math.pi),
                ),
            ),
            "utensil_holder": ActorGenerationSpec(
                nominal_pose_world=(
                    -1.3535490247772108,
                    -0.828725046255258,
                    0.957649024555536,
                    0.027235030398994692,
                    0.00020374952328353055,
                    -0.0002934064171910318,
                    -0.9996289939362364,
                ),
            ),
        },
        constraints=(
            SpawnConstraint(
                kind="segment_circle_clearance",
                asset_a="spoon",
                asset_b="utensil_holder",
                asset_a_points=((0.0, -0.18, 0.0), (0.0, 0.18, 0.0)),
                asset_b_radius=0.07,
                clearance=0.01,
            ),
        ),
    ),
    "stack_red_bowl_into_grey_bowl": TaskGenerationSpec(
        instruction="Pick up the red bowl and place it in the grey bowl.",
        robot_base_rot_world=(0.0, 0.0, 0.0, 1.0),
        actors={
            "red_bowl": ActorGenerationSpec(
                nominal_pose_world=(
                    -1.4559153752056169,
                    -0.4500163892408599,
                    0.9007219075693947,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                bounds=SpawnBounds(y=(0.0, 0.15)),
            ),
            "grey_bowl": ActorGenerationSpec(
                nominal_pose_world=(
                    -1.6277208213255852,
                    -0.6622602687274793,
                    0.9111174353900668,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                bounds=SpawnBounds(x=(-0.10, 0.10), y=(0.0, 0.166)),
            ),
        },
        constraints=(
            SpawnConstraint(
                kind="minimum_planar_distance",
                asset_a="red_bowl",
                asset_b="grey_bowl",
                minimum_distance=0.191,
            ),
        ),
    ),
}


def _quat_multiply(first: Quaternion, second: Quaternion) -> Quaternion:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_conjugate(quat: Quaternion) -> Quaternion:
    return quat[0], -quat[1], -quat[2], -quat[3]


def _quat_rotate(quat: Quaternion, vector: Vector3) -> Vector3:
    rotated = _quat_multiply(
        _quat_multiply(quat, (0.0, *vector)),
        _quat_conjugate(quat),
    )
    return rotated[1], rotated[2], rotated[3]


def _normalize_quat(quat: Quaternion) -> Quaternion:
    norm = math.sqrt(sum(value * value for value in quat))
    return tuple(value / norm for value in quat)  # type: ignore[return-value]


def _uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    return bounds[0] if bounds[0] == bounds[1] else rng.uniform(*bounds)


def _sample_actor_pose(
    rng: random.Random,
    actor: ActorGenerationSpec,
    robot_base_rot: Quaternion,
) -> Pose:
    if actor.bounds is None:
        return actor.nominal_pose_world
    x_offset = _uniform(rng, actor.bounds.x)
    y_offset = _uniform(rng, actor.bounds.y)
    yaw = _uniform(rng, actor.bounds.yaw)
    world_offset = _quat_rotate(robot_base_rot, (x_offset, y_offset, 0.0))
    nominal_pos = actor.nominal_pose_world[:3]
    nominal_rot = actor.nominal_pose_world[3:]
    position = tuple(
        nominal_pos[index] + world_offset[index] for index in range(3)
    )
    yaw_rotation = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    orientation = _normalize_quat(_quat_multiply(yaw_rotation, nominal_rot))
    return (*position, *orientation)


def _actor_points_world(pose: Pose, points: tuple[Vector3, ...]) -> list[Vector3]:
    position = pose[:3]
    orientation = pose[3:]
    return [
        tuple(
            position[index] + _quat_rotate(orientation, point)[index]
            for index in range(3)
        )
        for point in points
    ]


def _segment_intersects_aabb_2d(
    start: Vector3,
    end: Vector3,
    extent_min: Vector3,
    extent_max: Vector3,
    clearance: float,
) -> bool:
    entry = 0.0
    exit = 1.0
    for axis in range(2):
        minimum = extent_min[axis] - clearance
        maximum = extent_max[axis] + clearance
        direction = end[axis] - start[axis]
        if abs(direction) < 1e-12:
            if start[axis] < minimum or start[axis] > maximum:
                return False
            continue
        first = (minimum - start[axis]) / direction
        second = (maximum - start[axis]) / direction
        entry = max(entry, min(first, second))
        exit = min(exit, max(first, second))
        if entry > exit:
            return False
    return True


def _constraint_valid(
    poses: Mapping[str, Pose],
    constraint: SpawnConstraint,
) -> bool:
    pose_a = poses[constraint.asset_a]
    if constraint.kind == "minimum_planar_distance":
        pose_b = poses[constraint.asset_b]
        distance = math.hypot(pose_a[0] - pose_b[0], pose_a[1] - pose_b[1])
        return distance >= float(constraint.minimum_distance or 0.0)

    points = _actor_points_world(pose_a, constraint.asset_a_points)
    if constraint.kind == "segment_circle_clearance":
        pose_b = poses[constraint.asset_b]
        start, end = points
        segment_x = end[0] - start[0]
        segment_y = end[1] - start[1]
        length_squared = max(segment_x * segment_x + segment_y * segment_y, 1e-12)
        projection = (
            (pose_b[0] - start[0]) * segment_x
            + (pose_b[1] - start[1]) * segment_y
        ) / length_squared
        projection = min(max(projection, 0.0), 1.0)
        closest_x = start[0] + projection * segment_x
        closest_y = start[1] + projection * segment_y
        required = float(constraint.asset_b_radius or 0.0) + constraint.clearance
        return math.hypot(pose_b[0] - closest_x, pose_b[1] - closest_y) >= required

    if constraint.kind == "segment_aabb_clearance":
        if (
            constraint.asset_b_pose_world is None
            or constraint.asset_b_extent_min is None
            or constraint.asset_b_extent_max is None
        ):
            raise ValueError("segment_aabb_clearance requires target pose and extents")
        box_pose = constraint.asset_b_pose_world
        inverse_rotation = _quat_conjugate(box_pose[3:])
        points_box = []
        for point in points:
            relative = tuple(point[index] - box_pose[index] for index in range(3))
            points_box.append(_quat_rotate(inverse_rotation, relative))
        return not _segment_intersects_aabb_2d(
            points_box[0],
            points_box[1],
            constraint.asset_b_extent_min,
            constraint.asset_b_extent_max,
            constraint.clearance,
        )

    raise ValueError(f"Unsupported spawn constraint: {constraint.kind}")


def generate_poses(
    task: TaskGenerationSpec,
    *,
    seed: int = DEFAULT_SEED,
    count: int = DEFAULT_CONDITION_COUNT,
) -> list[dict[str, list[float]]]:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    conditions: list[dict[str, list[float]]] = []
    for _condition_index in range(count):
        for _attempt in range(128):
            poses = {
                name: _sample_actor_pose(rng, actor, task.robot_base_rot_world)
                for name, actor in task.actors.items()
            }
            if all(
                _constraint_valid(poses, constraint)
                for constraint in task.constraints
            ):
                conditions.append(
                    {name: list(pose) for name, pose in poses.items()}
                )
                break
        else:
            raise RuntimeError("Could not generate a collision-free condition")
    return conditions


def generate_document(
    task_id: str,
    *,
    seed: int = DEFAULT_SEED,
    count: int = DEFAULT_CONDITION_COUNT,
) -> dict:
    task = TASK_SPECS[task_id]
    return {
        "instruction": task.instruction,
        "poses": generate_poses(task, seed=seed, count=count),
    }


def _document_text(document: dict) -> str:
    return json.dumps(document, indent=2) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--conditions", type=int, default=DEFAULT_CONDITION_COUNT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    changed = False
    for task_id in TASK_SPECS:
        output_path = CONFIG_DIR / task_id / "initial_conditions.json"
        expected = _document_text(
            generate_document(task_id, seed=args.seed, count=args.conditions)
        )
        if args.check:
            if not output_path.is_file() or output_path.read_text(encoding="utf-8") != expected:
                changed = True
                print(f"OUT_OF_DATE {output_path}")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(expected, encoding="utf-8")
            print(f"WROTE {output_path}")
    if changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
