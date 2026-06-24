"""Soccer task MDP terms — re-exported from sub-modules.

Self-contained MDP functions for observations, rewards, terminations,
reset events, domain randomization, and soccer-specific ball resets.
Mirrors the mjlab.tasks.velocity.mdp multi-file pattern.
"""

from mjlab.envs.mdp import *  # noqa: F401, F403

from .shooter_commands import MultiMotionSoccerCommand, MultiMotionSoccerCommandCfg  # noqa: F401
from .shared_domain_randomization import *  # noqa: F403
from .shooter_kick_detection import KickContactTracker, KickContactEvent, ContactFootInfo  # noqa: F401
from .shared_obs import *  # noqa: F403
from .opponent_obs import *  # noqa: F403
from .shared_reset import *  # noqa: F403
from .shared_rewards import *  # noqa: F403
from .goalkeeper_ball_reset import *  # noqa: F403
from .shared_terminations import *  # noqa: F403
from . import goalkeeper_obs  # noqa: F401
from . import goalkeeper_rewards  # noqa: F401
from . import goalkeeper_student_obs  # noqa: F401
from . import shooter_obs  # noqa: F401
from . import shooter_rewards  # noqa: F401
from . import student_shooter_commands  # noqa: F401
from . import student_shooter_obs  # noqa: F401
from . import student_shooter_rewards  # noqa: F401
