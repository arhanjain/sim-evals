from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Milestone:
    name: str
    description: str


@dataclass(frozen=True)
class MilaTask:
    institution: str
    task_id: str
    scene_asset: str
    language_instruction: str
    success_criteria: str
    max_timesteps: int
    milestones: tuple[Milestone, ...]
    object_name: str
    target_name: str
    object_prim_path: str
    target_prim_path: str
    target_top_center: tuple[float, float, float] | None = None
    target_bottom_center: tuple[float, float, float] | None = None
    object_bottom_center: tuple[float, float, float] | None = None
    object_mouth_center: tuple[float, float, float] | None = None
    target_region_prim_path: str | None = None
    target_region_extent_min: tuple[float, float, float] | None = None
    target_region_extent_max: tuple[float, float, float] | None = None
    target_region_tolerance: float = 0.01
    reach_distance: float = 0.03
    lift_height: float = 0.05
    target_xy_radius: float = 0.18
    above_target_height: float = 0.08
    in_target_height: float = 0.05
    in_target_xy_radius: float | None = None
    release_gripper_threshold: float = 0.5


MILA_TASKS: dict[str, MilaTask] = {
    "place_spoon_in_sink": MilaTask(
        institution="MILA",
        task_id="place_spoon_in_sink",
        scene_asset="Task2/Scene.usd",
        language_instruction="Pick up the spoon and place it in the sink.",
        success_criteria="The spoon ends up lying flat in the sink and the gripper releases it.",
        max_timesteps=300,
        milestones=(
            Milestone(
                name="reached_spoon",
                description="Gripper made contact with the spoon.",
            ),
            Milestone(
                name="lifted_up_spoon",
                description="Spoon was lifted clearly off the countertop.",
            ),
            Milestone(
                name="above_sink",
                description=(
                    "The majority of the spoon was moved above the sink region before final placement."
                ),
            ),
            Milestone(
                name="completely_in_sink",
                description="Spoon ends up completely in the sink and the gripper releases it.",
            ),
        ),
        object_name="spoon",
        target_name="sink",
        object_prim_path="Spoon054/Spoon054",
        target_prim_path="Sink081/Sink081",
        target_top_center=(0.0, 0.0, 0.19044651),
        target_bottom_center=(0.0, 0.0, -0.19044651),
        reach_distance=0.20,
        target_xy_radius=0.10,
        in_target_xy_radius=0.15,
    ),
    "place_spoon_in_utensil_holder": MilaTask(
        institution="MILA",
        task_id="place_spoon_in_utensil_holder",
        scene_asset="Task1/Scene.usd",
        language_instruction="Pick up the spoon and place it in the utensil holder.",
        success_criteria="The spoon ends up in the utensil holder and the gripper releases it.",
        max_timesteps=400,
        milestones=(
            Milestone(
                name="reached_spoon",
                description="Gripper made contact with the spoon.",
            ),
            Milestone(
                name="lifted_up_spoon",
                description="Spoon was lifted clearly off the countertop.",
            ),
            Milestone(
                name="above_utensil_holder",
                description=(
                    "Spoon's bottom end was moved above or into the utensil holder region before release."
                ),
            ),
            Milestone(
                name="released_in_utensil_holder",
                description=(
                    "Spoon is in the utensil holder and the gripper releases it. "
                    "The utensil holder remains upright."
                ),
            ),
        ),
        object_name="spoon",
        target_name="utensil_holder",
        object_prim_path="Spoon054/Spoon054",
        target_prim_path="Bowl060/Bowl060",
        target_top_center=(0.0, 0.0, 0.2589967),
        target_bottom_center=(0.0, 0.0, -0.08577694),
        reach_distance=0.20,
        target_xy_radius=0.05,
        in_target_height=0.05,
        in_target_xy_radius=0.05,
    ),
    "stack_red_bowl_into_grey_bowl": MilaTask(
        institution="MILA",
        task_id="stack_red_bowl_into_grey_bowl",
        scene_asset="Task1/Scene.usd",
        language_instruction="Pick up the red bowl and place it in the grey bowl.",
        success_criteria="The red bowl ends up inside the grey bowl.",
        max_timesteps=600,
        milestones=(
            Milestone(
                name="reached_red_bowl",
                description="Gripper made contact with the red bowl.",
            ),
            Milestone(
                name="lifted_up_red_bowl",
                description="Red bowl was lifted clearly off the countertop.",
            ),
            Milestone(
                name="above_grey_bowl",
                description="Red bowl was lifted vertically above the grey bowl height before final placement.",
            ),
            Milestone(
                name="placed_in_grey_bowl",
                description="Red bowl is inside the grey bowl.",
            ),
        ),
        object_name="red_bowl",
        target_name="grey_bowl",
        object_prim_path="Bowl061/Bowl061",
        target_prim_path="Bowl062/Bowl062",
        target_bottom_center=(0.0, 0.0, -0.040000003),
        reach_distance=0.12,
        object_bottom_center=(0.0, 0.0, -0.029655244),
        object_mouth_center=(0.0, 0.0, 0.029655248),
        target_region_prim_path="Bowl062/Bowl062/Sites/bowl_liquid",
        target_region_extent_min=(-0.089, -0.089, -0.034),
        target_region_extent_max=(0.089, 0.089, 0.034),
        target_region_tolerance=0.01,
        lift_height=0.05,
        target_xy_radius=0.10,
        above_target_height=0.08,
        in_target_height=0.08,
    ),
}


def get_mila_task(task_id: str) -> MilaTask:
    try:
        return MILA_TASKS[task_id]
    except KeyError as exc:
        supported_tasks = ", ".join(sorted(MILA_TASKS))
        raise ValueError(f"Unsupported MILA task '{task_id}'. Supported tasks: {supported_tasks}") from exc
