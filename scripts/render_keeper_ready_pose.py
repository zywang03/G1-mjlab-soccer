"""Render goalkeeper default stance and a proposed low ready stance."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from src.assets.robots.unitree_g1.g1_constants import G1_XML
from src.tasks.soccer.mdp.goalkeeper_obs import _GK_JOINT_NAMES, _REF_DEFAULT_DOF_POS


LOW_READY_POSE = {
  "left_hip_pitch_joint": -0.78,
  "left_hip_roll_joint": 0.42,
  "left_knee_joint": 1.45,
  "left_ankle_pitch_joint": -0.78,
  "left_ankle_roll_joint": -0.30,
  "right_hip_pitch_joint": -0.78,
  "right_hip_roll_joint": -0.42,
  "right_knee_joint": 1.45,
  "right_ankle_pitch_joint": -0.78,
  "right_ankle_roll_joint": 0.30,
  "waist_pitch_joint": 0.22,
}


def _joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  if joint_id < 0:
    raise KeyError(f"joint not found: {joint_name}")
  return model.jnt_qposadr[joint_id]


def _set_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float], root_xy: tuple[float, float]) -> None:
  data.qpos[:] = model.key_qpos[0] if model.nkey else 0.0
  data.qvel[:] = 0.0
  data.qpos[0] = root_xy[0]
  data.qpos[1] = root_xy[1]
  data.qpos[2] = 0.8
  data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
  for name, value in zip(_GK_JOINT_NAMES, _REF_DEFAULT_DOF_POS):
    data.qpos[_joint_qpos_addr(model, name)] = value
  for name, value in pose.items():
    data.qpos[_joint_qpos_addr(model, name)] = value
  mujoco.mj_forward(model, data)
  floor_z = float(np.min(data.xpos[:, 2]))
  data.qpos[2] += 0.01 - floor_z
  mujoco.mj_forward(model, data)


def _render_pair(out_path: Path) -> None:
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=480, width=640)
  camera = mujoco.MjvCamera()
  camera.azimuth = 135
  camera.elevation = -13
  camera.distance = 2.65
  try:
    _set_pose(model, data, {}, root_xy=(0.0, 0.55))
    mujoco.mj_forward(model, data)
    default_qpos = data.qpos.copy()

    _set_pose(model, data, LOW_READY_POSE, root_xy=(0.0, -0.55))
    low_qpos = data.qpos.copy()

    data.qpos[:] = default_qpos
    mujoco.mj_forward(model, data)
    camera.lookat[:] = np.array([0.0, 0.0, 0.50])
    renderer.update_scene(data, camera=camera)
    img_default = renderer.render()

    data.qpos[:] = low_qpos
    mujoco.mj_forward(model, data)
    camera.lookat[:] = np.array([0.0, 0.0, 0.46])
    renderer.update_scene(data, camera=camera)
    img_low = renderer.render()
  finally:
    renderer.close()

  gap = np.full((img_default.shape[0], 16, 3), 255, dtype=np.uint8)
  combined = np.concatenate([img_default, gap, img_low], axis=1)
  canvas = Image.fromarray(combined)
  draw = ImageDraw.Draw(canvas)
  draw.rectangle((0, 0, 260, 36), fill=(255, 255, 255))
  draw.rectangle((img_default.shape[1] + gap.shape[1], 0, img_default.shape[1] + gap.shape[1] + 300, 36), fill=(255, 255, 255))
  draw.text((12, 10), "default keeper stance", fill=(0, 0, 0))
  draw.text((img_default.shape[1] + gap.shape[1] + 12, 10), "deep ready stance", fill=(0, 0, 0))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  iio.imwrite(out_path, np.asarray(canvas))


def main() -> None:
  out_path = Path("logs/visualizations/keeper_ready_pose_default_vs_deep.png")
  _render_pair(out_path)
  print(out_path)


if __name__ == "__main__":
  main()
