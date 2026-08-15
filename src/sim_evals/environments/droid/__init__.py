"""Registers the DROID scenes as Gymnasium environments.

Every scene in :data:`~sim_evals.environments.droid.scenes.DROID_SCENES` is
registered under its own Gym id, all sharing the scene-agnostic
:class:`~sim_evals.environments.droid.env_cfg.DroidEnvCfg`. Rather than defining
a config subclass per scene, each id uses a factory ``env_cfg_entry_point``:
IsaacLab's ``load_cfg_from_registry`` accepts a callable and invokes it to build
the config, so the scene's USD / instruction / task overlays are composed onto
the base at load time via :meth:`DroidEnvCfg.set_scene`.
"""

import gymnasium as gym

from .env_cfg import DroidEnvCfg
from .scenes import (
    DROID_SCENES,
    DroidSceneSpec,
    get_data_path,
    get_scene_spec,
    resolve_scene_asset_path,
)

__all__ = [
    "DroidEnvCfg",
    "DroidSceneSpec",
    "DROID_SCENES",
    "get_scene_spec",
    "get_data_path",
    "resolve_scene_asset_path",
]


def _make_env_cfg(scene_id: str):
    """Return a zero-arg factory that builds the base cfg composed with ``scene_id``."""

    def _factory() -> DroidEnvCfg:
        return DroidEnvCfg().set_scene(scene_id)

    return _factory


for _scene_id in DROID_SCENES:
    if _scene_id in gym.registry:
        continue
    gym.register(
        id=_scene_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _make_env_cfg(_scene_id)},
    )
