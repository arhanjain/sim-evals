from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs" / "mila"
SCENE_ASSET = "Task1/Scene.usd"
BASE_SEED = 42
SUCCESS_HOLD_STEPS = 5
INITIAL_CONDITION_COUNT = 20


@dataclass(frozen=True)
class Milestone:
    name: str
    description: str


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
    object_init_pos_world: tuple[float, float, float]
    object_init_rot_world: tuple[float, float, float, float]
    target_init_pos_world: tuple[float, float, float] | None
    target_init_rot_world: tuple[float, float, float, float] | None
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
    initial_conditions_file: Path
    # Success-criteria selectors (previously hardcoded via ``task_id`` string
    # comparisons in the reward code). ``containment_uses_object_point`` places
    # the object's ``object_bottom_center`` point into the target region -- and,
    # when ``release_requires_upright`` is set, additionally requires the target
    # to stay upright within ``target_upright_min_z`` -- instead of the default
    # surface-overlap check. Region-point release is selected by a non-empty
    # ``object_region_points`` and needs no flag.
    containment_uses_object_point: bool = False
    release_requires_upright: bool = False
    target_upright_min_z: float = 0.9


def _pose(value, field: str):
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError(f"{field} must contain [x, y, z, qw, qx, qy, qz]")
    numbers = tuple(float(item) for item in value)
    return numbers[:3], numbers[3:]


def _initial_conditions(task_id: str, actor_names: tuple[str, ...]):
    """Load the repository's existing instruction plus finite-pose JSON format."""
    path = CONFIG_DIR / task_id / "initial_conditions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    instruction = data.get("instruction")
    poses = data.get("poses")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError(f"{path} must define a non-empty instruction")
    if not isinstance(poses, list) or len(poses) != INITIAL_CONDITION_COUNT:
        raise ValueError(
            f"{path} must define exactly {INITIAL_CONDITION_COUNT} pose mappings"
        )
    for index, pose in enumerate(poses):
        if not isinstance(pose, dict) or set(pose) != set(actor_names):
            raise ValueError(
                f"{path}: poses[{index}] must define poses for {sorted(actor_names)}"
            )
        for name in actor_names:
            _pose(pose[name], f"{path}: poses[{index}].{name}")
    actor_poses = {
        name: _pose(poses[0][name], f"{path}: poses[0].{name}")
        for name in actor_names
    }
    return instruction, actor_poses, path


EXTERNAL_CAMERA = CameraCfg(
    inherited=False,
    pos=(0.17631441295554123, 0.5078738721437033, 0.5882641540000001),
    rot=(-0.28456424578276596, -0.14125179727209242, 0.4210391071710604, 0.8494099069483977),
    focal_length=2.1,
    horizontal_aperture=5.376,
    vertical_aperture=3.024,
)
HOLDER_EXTERNAL_CAMERA = CameraCfg(
    inherited=False,
    pos=(0.18134685930269425, 0.6219124578527772, 0.38105248130496585),
    rot=(-0.28456424578276596, -0.14125179727209242, 0.4210391071710604, 0.8494099069483977),
    focal_length=2.1,
    horizontal_aperture=5.376,
    vertical_aperture=3.024,
)
EXTERNAL_CAMERA_2 = CameraCfg(
    inherited=False,
    pos=(0.2412851373627978, -0.40400305313882967, 0.5012697904933473),
    rot=(0.8503307497842424, 0.5093728677822538, -0.06294131031153728, -0.11625602077887143),
    focal_length=2.1,
    horizontal_aperture=5.376,
    vertical_aperture=3.024,
)
WRIST_CAMERA = CameraCfg(inherited=True)
SHARED_DOME_LIGHT = DomeLightCfg(
    intensity=800.0,
    color=(0.6988417, 0.6934452, 0.6934452),
    texture_file="Task1/HDRI004.hdr",
    visible_in_primary_ray=False,
)
SPHERE_LIGHT_OFF = SphereLightCfg(
    enabled=False,
    intensity=8500.0,
    color=(1.0, 0.98, 0.95),
    radius=0.4,
    pos=(-1.5, -0.4, 1.6),
)


sink_instruction, sink_poses, sink_initial_conditions = _initial_conditions(
    "place_spoon_in_sink", ("spoon",)
)
holder_instruction, holder_poses, holder_initial_conditions = _initial_conditions(
    "place_spoon_in_utensil_holder", ("spoon", "utensil_holder")
)
bowl_instruction, bowl_poses, bowl_initial_conditions = _initial_conditions(
    "stack_red_bowl_into_grey_bowl", ("red_bowl", "grey_bowl")
)


MILA_TASKS = {
    "place_spoon_in_sink": MilaTask(
        institution="MILA",
        task_id="place_spoon_in_sink",
        scene_asset=SCENE_ASSET,
        language_instruction=sink_instruction,
        success_criteria="The spoon ends up lying flat in the sink and the gripper releases it.",
        max_timesteps=300,
        seed=BASE_SEED,
        success_hold_steps=SUCCESS_HOLD_STEPS,
        milestones=(
            Milestone("reached_spoon", "Gripper made contact with the spoon."),
            Milestone("lifted_up_spoon", "Spoon was lifted clearly off the countertop."),
            Milestone("above_sink", "The majority of the spoon was moved above the sink region before final placement."),
            Milestone("completely_in_sink", "Spoon ends up completely in the sink and the gripper releases it."),
        ),
        final_milestone="completely_in_sink",
        object_name="spoon",
        target_name="sink",
        object_prim_path="Spoon054/Spoon054",
        target_prim_path=None,
        robot_base_pos=(-1.0184789338318788, -0.8314798528096464, 0.806119064555536),
        robot_base_rot=(0.0, 0.0, 0.0, 1.0),
        robot_init_joint_pos=(0.00112558354, -0.609502673, 0.000848179392, -2.50074935, -0.0229695588, 1.89955163, -0.0205767453),
        object_init_pos_world=sink_poses["spoon"][0],
        object_init_rot_world=sink_poses["spoon"][1],
        target_init_pos_world=None,
        target_init_rot_world=None,
        cameras={"external_cam": EXTERNAL_CAMERA, "external_cam_2": EXTERNAL_CAMERA_2, "wrist_cam": WRIST_CAMERA},
        dome_light=SHARED_DOME_LIGHT,
        sphere_light=SPHERE_LIGHT_OFF,
        fixed_target_pos_robot=(0.4756867334943357, 0.2833240761550858, -0.0002414716808392),
        fixed_target_rot_robot=(0.7071067811865476, 0.0, 0.0, -0.7071067811865475),
        hidden_prim_paths=("Bowl060", "Bowl061", "Bowl062", "CannedFood048", "Cup082", "TeaBox001"),
        target_top_center=(0.0, 0.0, 0.10113153606653214),
        target_bottom_center=(0.0, 0.0, -0.10113153606653214),
        object_bottom_center=None,
        object_mouth_center=None,
        object_region_points=((0.0, -0.18, 0.0), (0.0, 0.18, 0.0)),
        target_region_prim_path="Sink003/Sink003/Sites/reg_basin",
        target_region_extent_min=(-0.17917540669441223, -0.15094169974327087, -0.10113153606653214),
        target_region_extent_max=(0.17917540669441223, 0.15094169974327087, 0.10113153606653214),
        target_region_tolerance=0.0,
        reach_distance=0.20,
        lift_height=0.05,
        target_xy_radius=0.10,
        above_target_height=0.08,
        in_target_height=0.05,
        in_target_xy_radius=0.15,
        release_gripper_threshold=0.5,
        initial_conditions_file=sink_initial_conditions,
    ),
    "place_spoon_in_utensil_holder": MilaTask(
        institution="MILA",
        task_id="place_spoon_in_utensil_holder",
        scene_asset=SCENE_ASSET,
        language_instruction=holder_instruction,
        success_criteria="The spoon ends up in the utensil holder, the gripper releases it, and the holder remains upright.",
        max_timesteps=400,
        seed=BASE_SEED,
        success_hold_steps=SUCCESS_HOLD_STEPS,
        milestones=(
            Milestone("reached_spoon", "Gripper made contact with the spoon."),
            Milestone("lifted_up_spoon", "Spoon was lifted clearly off the countertop."),
            Milestone("above_utensil_holder", "The spoon's insertion end was moved above or into the holder region."),
            Milestone("released_in_utensil_holder", "Spoon is inside the upright holder and the gripper releases it."),
        ),
        final_milestone="released_in_utensil_holder",
        object_name="spoon",
        target_name="utensil_holder",
        object_prim_path="Spoon054/Spoon054",
        target_prim_path="Bowl060/Bowl060",
        robot_base_pos=(-0.9930789338318788, -0.5520798528096464, 0.9144153361701479),
        robot_base_rot=(0.0, 0.0, 0.0, 1.0),
        robot_init_joint_pos=(0.01643991, -0.59407169, -0.01652975, -2.50808907, -0.02607333, 1.91518295, -0.0163574),
        object_init_pos_world=holder_poses["spoon"][0],
        object_init_rot_world=holder_poses["spoon"][1],
        target_init_pos_world=holder_poses["utensil_holder"][0],
        target_init_rot_world=holder_poses["utensil_holder"][1],
        cameras={"external_cam": HOLDER_EXTERNAL_CAMERA, "external_cam_2": EXTERNAL_CAMERA_2, "wrist_cam": WRIST_CAMERA},
        dome_light=SHARED_DOME_LIGHT,
        sphere_light=SphereLightCfg(
            enabled=True,
            intensity=8500.0,
            color=(1.0, 0.98, 0.95),
            radius=0.4,
            pos=(-1.3535490247772108, -0.828725046255258, 1.6),
        ),
        fixed_target_pos_robot=None,
        fixed_target_rot_robot=(1.0, 0.0, 0.0, 0.0),
        hidden_prim_paths=("Bowl061", "Bowl062", "CannedFood048", "Cup082", "TeaBox001"),
        target_top_center=(0.0, 0.0, 0.2589967),
        target_bottom_center=(0.0, 0.0, -0.08577694),
        object_bottom_center=(0.0, -0.18, 0.0),
        object_mouth_center=None,
        object_region_points=(),
        target_region_prim_path="Bowl060/Bowl060/Sites/bowl_liquid",
        target_region_extent_min=(-0.05687966, -0.06451707, -0.07497805),
        target_region_extent_max=(0.05687965, 0.04574309, 0.06924114),
        target_region_tolerance=0.01,
        reach_distance=0.20,
        lift_height=0.05,
        target_xy_radius=0.05,
        above_target_height=0.08,
        in_target_height=0.05,
        in_target_xy_radius=0.05,
        release_gripper_threshold=0.5,
        initial_conditions_file=holder_initial_conditions,
        containment_uses_object_point=True,
        release_requires_upright=True,
    ),
    "stack_red_bowl_into_grey_bowl": MilaTask(
        institution="MILA",
        task_id="stack_red_bowl_into_grey_bowl",
        scene_asset=SCENE_ASSET,
        language_instruction=bowl_instruction,
        success_criteria="The red bowl ends up inside the grey bowl and the gripper releases it.",
        max_timesteps=600,
        seed=BASE_SEED,
        success_hold_steps=SUCCESS_HOLD_STEPS,
        milestones=(
            Milestone("reached_red_bowl", "Gripper made contact with the red bowl."),
            Milestone("lifted_up_red_bowl", "Red bowl was lifted clearly off the countertop."),
            Milestone("above_grey_bowl", "Red bowl was lifted above and aligned with the grey bowl."),
            Milestone("placed_in_grey_bowl", "Red bowl is contained inside the grey bowl and the gripper releases it."),
        ),
        final_milestone="placed_in_grey_bowl",
        object_name="red_bowl",
        target_name="grey_bowl",
        object_prim_path="Bowl061/Bowl061",
        target_prim_path="Bowl062/Bowl062",
        robot_base_pos=(-1.0184789338318788, -0.5012798528096464, 0.831519064555536),
        robot_base_rot=(0.0, 0.0, 0.0, 1.0),
        robot_init_joint_pos=(0.00500921, -0.60578179, 0.00545613, -2.52051044, -0.02542639, 1.91960347, 0.03166837),
        object_init_pos_world=bowl_poses["red_bowl"][0],
        object_init_rot_world=bowl_poses["red_bowl"][1],
        target_init_pos_world=bowl_poses["grey_bowl"][0],
        target_init_rot_world=bowl_poses["grey_bowl"][1],
        cameras={"external_cam": EXTERNAL_CAMERA, "external_cam_2": EXTERNAL_CAMERA_2, "wrist_cam": WRIST_CAMERA},
        dome_light=SHARED_DOME_LIGHT,
        sphere_light=SPHERE_LIGHT_OFF,
        fixed_target_pos_robot=None,
        fixed_target_rot_robot=(1.0, 0.0, 0.0, 0.0),
        hidden_prim_paths=("Bowl060", "CannedFood048", "Cup082", "Spoon054", "TeaBox001"),
        target_top_center=None,
        target_bottom_center=(0.0, 0.0, -0.040000003),
        object_bottom_center=(0.0, 0.0, -0.029655244),
        object_mouth_center=(0.0, 0.0, 0.029655248),
        object_region_points=(),
        target_region_prim_path="Bowl062/Bowl062/Sites/bowl_liquid",
        target_region_extent_min=(-0.089, -0.089, -0.034),
        target_region_extent_max=(0.089, 0.089, 0.034),
        target_region_tolerance=0.01,
        reach_distance=0.12,
        lift_height=0.05,
        target_xy_radius=0.10,
        above_target_height=0.08,
        in_target_height=0.08,
        in_target_xy_radius=None,
        release_gripper_threshold=0.5,
        initial_conditions_file=bowl_initial_conditions,
    ),
}


def get_mila_task(task_id: str) -> MilaTask:
    try:
        return MILA_TASKS[task_id]
    except KeyError as exc:
        supported_tasks = ", ".join(sorted(MILA_TASKS))
        raise ValueError(
            f"Unsupported MILA task '{task_id}'. Supported tasks: {supported_tasks}"
        ) from exc
