"""Soccer task MDP terms — re-exported from sub-modules.

Self-contained MDP functions for observations, rewards, terminations,
reset events, domain randomization, and soccer-specific ball resets.
Mirrors the mjlab.tasks.velocity.mdp multi-file pattern.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from .domain_randomization import *  # noqa: F403
from .observations import *  # noqa: F403
from .reset_events import *  # noqa: F403
from .rewards import *  # noqa: F403
from .soccer_reset import *  # noqa: F403
from .terminations import *  # noqa: F403
