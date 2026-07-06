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
    reach_distance: float = 0.07
    lift_height: float = 0.05
    target_xy_radius: float = 0.18
    above_target_height: float = 0.08
    in_target_height: float = 0.05
    release_gripper_threshold: float = 0.2


MILA_TASKS: dict[str, MilaTask] = {
    "place_spoon_in_sink": MilaTask(
        institution="MILA",
        task_id="place_spoon_in_sink",
        scene_asset="place_spoon_in_sink.usd",
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
                    "The majority of the spoon (in the gripper or dropped) is above the sink, "
                    "but not placed fully in the sink (either it's standing up or not released yet)."
                ),
            ),
            Milestone(
                name="completely_in_sink",
                description="Spoon ends up completely in the sink and the gripper releases it.",
            ),
        ),
        object_name="spoon",
        target_name="sink",
    ),
    "place_spoon_in_utensil_holder": MilaTask(
        institution="MILA",
        task_id="place_spoon_in_utensil_holder",
        scene_asset="place_spoon_in_utensil_holder.usd",
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
                    "Spoon's bottom end is lifted vertically above the utensil holder, somewhere closeby, "
                    "or placed in the container but knocked it over."
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
        target_xy_radius=0.16,
        above_target_height=0.10,
        in_target_height=0.12,
    ),
    "stack_red_bowl_into_grey_bowl": MilaTask(
        institution="MILA",
        task_id="stack_red_bowl_into_grey_bowl",
        scene_asset="stack_red_bowl_into_grey_bowl.usd",
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
                description="Red bowl is lifted above the grey bowl vertically in height (anywhere).",
            ),
            Milestone(
                name="placed_in_grey_bowl",
                description="Red bowl is inside the grey bowl.",
            ),
        ),
        object_name="red_bowl",
        target_name="grey_bowl",
        reach_distance=0.08,
        lift_height=0.05,
        target_xy_radius=0.18,
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
