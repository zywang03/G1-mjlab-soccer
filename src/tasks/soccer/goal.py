"""Goal entity configuration for soccer tasks."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.entity import EntityCfg

GOAL_XML: Path = SRC_PATH / "assets" / "soccer" / "goal.xml"
assert GOAL_XML.exists()


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GOAL_XML))
  return spec


def get_goal_cfg(
  pos: tuple[float, float, float] = (0, 0, 0),
  rot: tuple[float, float, float, float] | None = None,
) -> EntityCfg:
  """Get goal entity configuration with custom position and rotation.

  The goal consists of two vertical posts (y=±1.5, z∈[0,1.8]) and a
  horizontal crossbar (z=1.8, y∈[-1.5, 1.5]). All bodies are static.

  By default, the goal opening faces ±x (posts along y-axis).
  To face ±y (posts along x-axis), rotate π/2 about z:
    rot = (0.7071, 0, 0, 0.7071).

  Args:
      pos: World position (x, y, z) offset for the entire goal assembly.
      rot: Optional quaternion (w, x, y, z) rotation. Default is identity.

  Returns:
      EntityCfg configured for the goal.
  """
  kwargs = {"pos": pos}
  if rot is not None:
    kwargs["rot"] = rot
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(**kwargs),
    spec_fn=get_spec,
  )
