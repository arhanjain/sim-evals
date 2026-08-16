"""sim-eval environments.

Importing this package registers every sim-eval Gym environment as a side
effect. Import it (``import sim_evals.environments``) before calling
``gym.make`` / ``parse_env_cfg``.
"""

from . import droid  # noqa: F401  (registers the DROID scenes on import)
from . import mila  # noqa: F401  (registers the MILA parity tasks on import)

__all__ = ["droid", "mila"]
