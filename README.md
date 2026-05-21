# CS 2810 — Humanoid Robot Soccer

A perception-guided humanoid soccer shooting and intercepting project for Unitree G1, using reinforcement learning with motion tracking on MuJoCo physics.  Built on the [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) framework.

## Overview

- **Task**: G1 humanoid shoots a stationary ball or intercept the moving ball
- **Physics**: MuJoCo with fluid air-drag on the ball, 50 Hz control
- **Robot**: Unitree G1, 29-DoF PD position control, armature-based stiffness

## Setup

See [doc/setup_en.md](doc/setup_en.md) for environment installation.

## Quick Start

```bash
# List all registered tasks
python scripts/list_envs.py

# Visualize the shooter scene (zero-agent)
python scripts/eval_naive_shooter.py

# Visualize the goalkeeper scene (zero-agent)
python scripts/eval_naive_goalkeeper.py

# Shooter headless batch evaluation with metrics
python scripts/eval_naive_shooter.py --num-trials=50 --headless

# Goalkeeper headless batch evaluation with metrics
python scripts/eval_naive_goalkeeper.py --num-trials=50 --headless

# Load a trained checkpoint
python scripts/eval_naive_shooter.py \
    --headless \
    --num-trials=50 \
    --checkpoint <file-path>

# Record a video
python scripts/eval_naive_shooter.py --video --video-length=500
python scripts/eval_naive_goalkeeper.py --video --video-length=150
```

## Observation Space

### Shooter (Eval)

```
 o_t = (o^prop_t,  o^ref_t,  o^soc_t)

 o^prop : proprioception   — projected_gravity (3) + base_ang_vel (3)
                            + joint_pos (29) + joint_vel (29) + last_action (29)   =  93 D
 o^ref  : motion reference — ref_joint_pos (29) + ref_joint_vel (29)
                            + ref_anchor_ang_vel (3)                               =  61 D
 o^soc  : soccer perception — ball_in_robot_frame (3) + goal_in_robot_frame (3)   =   6 D
 ─────────────────────────────────────────────────────────────────────────────────────────
 Actor total                                                                        160 D
```

### Goalkeeper (Eval)

Single-frame actor terms (matching paper's actor inputs):

| Term | Dim | Paper notation |
|------|-----|----------------|
| base_ang_vel (IMU) | 3 | ωbase |
| projected_gravity | 3 | gbase |
| joint_pos (relative) | 29 | q |
| joint_vel (relative) | 29 | q̇ |
| last_action | 29 | alast |
| **ball_pos_local** | 3 | Oball ∈ R³ |
| **Total (single frame)** | **96** | |

Actor observation: 96 D × **T = 10** history frames = **960 D** (matching paper's HIMPPO history encoder input).


## Evaluation Protocol

### Shooter

Ball position is fixed at the penalty spot (4 m from goal). The robot root pose is randomized each episode (±0.5 m xy, ±0.5 rad yaw).

**Metrics** (matching HumanoidSoccer Section IV‑B):
- Success Rate — fraction of episodes where the ball enters the goal
- Kick Accuracy — cosine similarity between ball velocity direction and the
  ball-to-goal-center vector

### Goalkeeper

Ball uses a **6-region parabolic trajectory model** matching the paper's `assign_ball_states`. Each episode randomly selects one of 6 landing regions and computes the launch velocity to hit the target point.

**Metrics** (matching Humanoid-Goalkeeper Section IV):
- Block Rate (Esucc) — fraction of episodes where ball velocity drops > 2 m/s
  when behind the robot (intercepted)
- Min ball-robot distance — closest approach in xy-plane
- Ball speed at robot crossing

```bash
# Shooter eval
python scripts/eval_naive_shooter.py # Interactive viewer
python scripts/eval_naive_shooter.py --headless --num-trials=100
python scripts/eval_naive_shooter.py --headless --num-trials=100 --checkpoint <path>

# Goalkeeper eval
python scripts/eval_naive_goalkeeper.py # Interactive viewer
python scripts/eval_naive_goalkeeper.py --headless --num-trials=100
python scripts/eval_naive_goalkeeper.py --headless --num-trials=100 --checkpoint <path>
```

## Project Structure

```
src/
  assets/soccer/
    ball.xml, goal.xml, ground.xml     # MuJoCo entity models
    motions/                           # Retargeted kick trajectories (.npz)
  tasks/soccer/
    ball.py, goal.py, ground.py        # Entity config factories
    soccer_env_cfg.py                  # Base env-cfg factory
    motion_data.py                     # Motion dataset loading & playback
    mdp/
      __init__.py                      # Re-exports all MDP terms from sub-modules
      observations.py                  # Proprioceptive, ball-local, motion-ref, soccer-perception
      rewards.py                       # is_terminated
      terminations.py                  # time_out, fell_over, motion-reference terminations
      reset_events.py                  # reset_root_state_uniform, reset_joints_by_offset
      domain_randomization.py          # push_robot_base, perturb_ball_velocity
      soccer_reset.py                  # RegionBallVelCfg + 6-region parabolic trajectory reset
    config/
      settings.yaml                    # Single source of truth for soccer parameters
      soccer_settings.py               # Typed settings loader (dataclass-backed)
      g1/env_cfgs.py                   # G1 shooter & goalkeeper configs
      g1/rl_cfg.py                     # PPO config
      eval/eval_shooter_cfg.py         # Shooter eval config (paper obs + motion ref + DR)
      eval/eval_goalkeeper_cfg.py      # Goalkeeper eval config (T=10 history, 6-region ball)
scripts/
  play.py                              # Task-agnostic scene viewer
  eval_naive_shooter.py                # Shooter eval (headless stats or interactive)
  eval_naive_goalkeeper.py             # Goalkeeper eval (headless stats or interactive)
```

## Acknowledgements

Built for CS 2810 (Spring 2026).  This project uses motion data and design references from the [HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer) and [Humanoid-Goalkeeper] repositories. If you use this template, please site 

```
@article{ren2025humanoidgoalkeeper,
  title={Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints},
  author={Ren, Junli, Long, Jungfeng, Huang, Tao and Wang, Huayi, Wang, Zirui and Jia, Feiyu, Zhang, Wentao and Wang, Jingbo, Ping Luo and Pang, Jiangmiao},
  year={2025}
}
@misc{kong2026learningsoccerskillshumanoid,
  title={Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework},
  author={Jipeng Kong and Xinzhe Liu and Yuhang Lin and Jinrui Han and Sören Schwertfeger and Chenjia Bai and Xuelong Li},
  year={2026},
  eprint={2602.05310},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2602.05310}
}
```
