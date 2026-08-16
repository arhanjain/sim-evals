"""Registers the MILA parity tasks as Gymnasium environments.

The three MILA tasks share one scene USD but differ by rewards, milestones,
hidden prims, robot frame, cameras, lighting, and initial conditions (see
:data:`~sim_evals.environments.mila.tasks.MILA_TASKS`). Each is registered under
``DROID-MILA-<task_id>`` using a factory ``env_cfg_entry_point`` -- the same
data-driven pattern the DROID scenes use -- so the task overlay is composed onto
:class:`MilaDroidEnvCfg` at load time via :meth:`MilaDroidEnvCfg.set_task`.
"""

import gymnasium as gym

from .mila_droid_environment import MilaDroidEnvCfg
from .tasks import MILA_TASKS, get_mila_task

__all__ = ["MilaDroidEnvCfg", "MILA_TASKS", "get_mila_task"]


def _make_env_cfg(task_id: str):
    """Return a zero-arg factory that builds the MILA cfg composed with ``task_id``."""

    def _factory() -> MilaDroidEnvCfg:
        return MilaDroidEnvCfg().set_task(task_id)

    return _factory


for _task_id in MILA_TASKS:
    _env_id = f"DROID-MILA-{_task_id}"
    if _env_id in gym.registry:
        continue
    gym.register(
        id=_env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": _make_env_cfg(_task_id)},
    )
