"""Scene registry for the DROID sim-eval environments.

A "scene" is mostly pure data: a USD asset plus the language instruction handed
to the policy. Scenes are composed onto the scene-agnostic
:class:`~sim_evals.environments.droid.env_cfg.DroidEnvCfg` at registration time
(see :func:`~sim_evals.environments.droid.set_scene`), so adding a new scene is
a one-line entry here rather than a new env-cfg subclass.

Scenes may also differ by task logic. When a scene needs its own success
condition, attach a reward and/or termination config class via the optional
``rewards`` / ``terminations`` fields; leaving them ``None`` keeps the base
env's defaults (empty rewards, time-out-only terminations).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

ASSET_ROOT_ENV_VAR = "SIM_EVALS_ASSET_ROOT"
# .../src/sim_evals/environments/droid/scenes.py -> parents[4] is the repo root.
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[4] / "assets"


class DroidSceneSpec(NamedTuple):
    """Static description of a single DROID scene.

    Attributes:
        env_id: Gym id the scene is registered under.
        instruction: Default language instruction handed to the policy.
        asset_paths: Candidate USD paths relative to the asset root, tried in
            order; the first one that exists is loaded.
        rewards: Optional reward-group config class applied on top of the base
            env. ``None`` keeps the base's (empty) rewards.
        terminations: Optional termination-group config class applied on top of
            the base env. ``None`` keeps the base's time-out-only terminations.
    """

    env_id: str
    instruction: str
    asset_paths: tuple[str, ...]
    rewards: type | None = None
    terminations: type | None = None


_SCENES: tuple[DroidSceneSpec, ...] = (
    DroidSceneSpec("DROID-CubeInBowl", "put the cube in the bowl", ("scene1.usd",)),
    DroidSceneSpec("DROID-CanInMug", "put the can in the mug", ("scene2.usd",)),
    DroidSceneSpec("DROID-BananaInBin", "put banana in the bin", ("scene3.usd",)),
    DroidSceneSpec(
        "DROID-Berkeley-Task1",
        "complete the Berkeley task 1",
        ("Berkeley/Berkeley_Task1/Scene.usd", "Berkeley_Task1/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-UPenn-FrankaKitchen",
        "complete the Franka kitchen task",
        ("UPenn/TASK-1-Levine457-FrankaKitchen/Scene.usd", "TASK-1-Levine457-FrankaKitchen/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-UPenn-TeaRoom",
        "complete the tea room task",
        ("UPenn/TASK-2-Levine459-TeaRoom/Scene.usd", "TASK-2-Levine459-TeaRoom/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-UPenn-LivingRoom",
        "complete the living room task",
        ("UPenn/TASK-3-AGH-LivingRoom/Scene.usd", "TASK-3-AGH-LivingRoom/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-UTAustin-Task1",
        "complete the UT Austin task 1",
        ("UT-Austin/UT-Austin-Task1/Scene.usd", "UT-Austin-Task1/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-UTAustin-Task2",
        "complete the UT Austin task 2",
        ("UT-Austin/UT-Austin-Task2/Scene.usd", "UT-Austin-Task2/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-UTAustin-Task3",
        "complete the UT Austin task 3",
        ("UT-Austin/UT-Austin-Task3/Scene.usd", "UT-Austin-Task3/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-Yonsei-Task1",
        "complete the Yonsei task 1",
        ("Yonsei/Task1_20260424/Scene.usd", "Task1_20260424/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-Yonsei-Task2",
        "complete the Yonsei task 2",
        ("Yonsei/Task2/Scene.usd", "Task2/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-MILA-Task1",
        "complete the MILA task 1",
        ("MILA/Task1/Scene.usd", "Task1/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-MILA-Task2",
        "complete the MILA task 2",
        ("MILA/Task2/Scene.usd", "Task2/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-FrodoBots-Task1",
        "complete the FrodoBots task 1",
        ("FrodoBots/Task1/Scene.usd", "Task1/Scene.usd"),
    ),
    DroidSceneSpec(
        "DROID-FrodoBots-Task2",
        "complete the FrodoBots task 2",
        ("FrodoBots/Task2/Scene.usd", "Task2/Scene.usd"),
    ),
)

# Public registry keyed by env id. Built from ``_SCENES`` so the id lives in one place.
DROID_SCENES: dict[str, DroidSceneSpec] = {spec.env_id: spec for spec in _SCENES}


def get_data_path() -> Path:
    """Root directory the scene assets are resolved against."""
    return Path(os.environ.get(ASSET_ROOT_ENV_VAR, DEFAULT_DATA_PATH)).expanduser().resolve()


def resolve_scene_asset_path(scene_spec: DroidSceneSpec) -> Path:
    """Return the first existing USD path for ``scene_spec``.

    Raises:
        FileNotFoundError: If none of the candidate paths exist.
    """
    candidates = [get_data_path() / asset_path for asset_path in scene_spec.asset_paths]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find asset for {scene_spec.env_id}. Checked:\n  - {candidate_text}\n"
        f"Set {ASSET_ROOT_ENV_VAR} to the directory containing the downloaded assets."
    )


def get_scene_spec(env_id: str) -> DroidSceneSpec:
    """Look up a scene by env id, raising a helpful error if it is unknown."""
    try:
        return DROID_SCENES[env_id]
    except KeyError as exc:
        valid = ", ".join(DROID_SCENES)
        raise ValueError(f"Unknown DROID environment {env_id!r}. Valid env ids: {valid}") from exc
