# CS 2810 — Humanoid Robot Soccer

A perception-guided humanoid soccer shooting and intercepting project for Unitree G1, using reinforcement learning with motion tracking on MuJoCo physics. Built on the [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) framework.

## Overview

- **Task**: G1 humanoid shoots a stationary ball toward a goal (shooter) and intercepts incoming balls (goalkeeper)
- **Physics**: MuJoCo with fluid air-drag on the ball, 50 Hz control
- **Robot**: Unitree G1, 29-DoF PD position control, armature-based stiffness
- **Training**: Two-stage PPO (motion tracking → perception-guided kicking), asymmetric actor-critic
- **Eval**: Paper-matching metrics (Success Rate, Kick Accuracy, Block Rate)

## Setup

See [doc/setup_en.md](doc/setup_en.md) for environment installation.

## Quick Start

```bash
# List all tasks
python scripts/list_envs.py

# Play (visualize with zero agent)
python scripts/play.py Unitree-G1-Shooter-Stage1 --agent zero --viewer native
python scripts/play.py Unitree-G1-Shooter-Stage2 --agent zero --viewer native

# Train shooter (two-stage)
bash shell/train_shooter.sh my_exp 4096

# Eval shooter (paper metrics)
python scripts/eval_naive_shooter.py --headless --num-trials 100
python scripts/eval_naive_shooter.py --headless --num-trials 100 --checkpoint <path>

# Eval goalkeeper
python scripts/eval_naive_goalkeeper.py --headless --num-trials 50
python scripts/eval_naive_goalkeeper.py --headless --num-trials 50 --checkpoint <path>
```

## Training Architecture

Two-stage PPO pipeline ported from HumanoidSoccer:

```
Stage I: Motion Tracking (MLP)       Stage II: Perception-Guided Kicking (LSTM)
  adaptive sampling                    uniform sampling (frame 0)
  MLP [512,256,128] ELU                LSTM(2×128) + MLP [128,64,32] ELU
  154D actor / 292D critic             160D actor / 298D critic
  8 reward terms (tracking only)       17 reward terms (+soccer kick)
        ↓                                      ↓
        └──── checkpoint transfer (iteration counter only) ──┘
                           ↓
                eval_naive_shooter.py  (same obs order)
```

**Observation space** (identical across Stage II and eval, 160D):
```
command(58) → projected_gravity(3) → motion_ref_ang_vel(3) → base_ang_vel(3) →
joint_pos(29) → joint_vel(29) → actions(29) → target_point_pos(3) → target_destination_pos(3)
```

Critic adds privileged body poses (anchor pos/ori, 14 body pos/ori, base_lin_vel) = 298D.

### Stage I — Motion Tracking

Robot learns 10 standard kick motion references without perception.

| Aspect | Config |
|--------|--------|
| Model | **MLP [512, 256, 128]** ELU |
| Sampling | Adaptive (failure-histogram driven) |
| Rewards (8) | 6× tracking (anchor/body pos/ori/vel, all weight=1.0) + action_rate(-0.1) + joint_limit(-10.0) |
| Terminations | timeout(10s) + fell_over(70°) + anchor_pos_z(0.25m) + anchor_ori(0.8) + ee_body_pos(0.25m) |
| DR | push_robot (interval 1-3s) |
| Entropy coef | 0.005 |

### Stage II — Perception-Guided Kicking

Builds on Stage I checkpoint. Model switches to LSTM for ball trajectory prediction.

| Change | Detail |
|--------|--------|
| Model | LSTM(2×128) + MLP [128, 64, 32] ELU |
| Sampling | Uniform (frame 0) |
| Ball placement | Motion endpoint + arc offset (±0.25m, ±π/9) |
| track_anchor_pos | **0.0** (robot free to pursue ball) |
| track_anchor_ori | **1.0** (unchanged from Stage I) |
| body tracking (pos/ori/vel) | **1.0** (unchanged, body pos/ori filtered — excludes ankles) |
| foot_pos tracking | 1.0 (ankles only) |
| Kick rewards | proximity(1), contact(50), sideways_kick(50), vel_align(30), speed(10) |
| Stabilization | foot_distance(0.2), pelvis_orientation(-1), waist_action_rate(-0.25) |
| Ball init vel | **Disabled** (stationary, penalty kick) |

### PPO Config

- Stage I: MLP [512, 256, 128], ELU; Stage II: LSTM(2×128) + MLP [128, 64, 32], ELU
- Gaussian scalar std, init_std=1.0
- Adaptive KL (desired=0.01), clip=0.2, lr=1e-3, entropy=0.005
- 24 steps/env, 5 epochs, 4 mini-batches


## Evaluation

### Shooter

Scene: G1 near origin facing -y (motion-local coords, identical to training/play).
Goal placed at (0, -5, 0) rotated 90° to face G1, ball placed dynamically by
the command system (same as Stage II training). No `motion_origin_offset` /
`motion_yaw_offset` transform — eval uses the exact same coordinate system
as training and play mode via `unitree_g1_stage2_env_cfg(play=True)`.

**Metrics** (matching HumanoidSoccer §IV-B):
- **Success Rate** — fraction of episodes where ball crosses goal plane (y≤-5, |x|≤1.5m, z≤1.8m)
- **Kick Accuracy** — cosine similarity between ball velocity direction and ball→goal-center vector
- **Kick Speed** — ball speed when first > 1 m/s

### Goalkeeper

Ball uses a **6-region parabolic trajectory model** matching the Humanoid-Goalkeeper paper.

**Metrics** (matching Humanoid-Goalkeeper §IV):
- Block Rate — fraction of episodes where ball velocity drops > 2 m/s behind robot
- Min ball-robot xy distance, mean ball speed at robot crossing

## Project Structure

```
src/
  assets/soccer/
    ball.xml, goal.xml, ground.xml     # MuJoCo entity models
    motions/                           # Retargeted kick trajectories (.npz, 13 files)
  tasks/soccer/
    ball.py, goal.py, ground.py        # Entity config factories
    soccer_env_cfg.py                  # Base env-cfg factory
    mdp/
      commands.py                      # MultiMotionSoccerCommand (mjlab CommandTerm)
      kick_detection.py                # KickContactTracker (shared contact detection)
      training_rewards.py              # Soccer/kick reward functions (9 funcs)
      training_obs.py                  # Privileged critic + soccer perception obs
      observations.py                  # Eval observation functions
      rewards.py, terminations.py      # Basic reward/termination functions
      reset_events.py                  # Reset + DR functions
      soccer_reset.py                  # 6-region parabolic ball trajectory
    config/
      settings.yaml                    # Central parameter source of truth
      soccer_settings.py               # Typed settings loader (dataclass-backed)
      g1/
        env_cfgs.py                    # Naive shooter & goalkeeper configs
        training_env_cfgs.py           # G1 training configs (Stage I/II wrappers)
        rl_cfg.py                      # PPO config
      eval/
        eval_shooter_cfg.py            # Eval shooter (reuses Stage II play config + goal)
        eval_goalkeeper_cfg.py         # Eval goalkeeper (T=10 history, 960D)
      training/
        stage1_env_cfg.py              # Stage I factory (motion tracking)
        stage2_env_cfg.py              # Stage II factory (perception-guided kick)
scripts/
  train.py                             # Training entrypoint
  play.py                              # Interactive visualization
  eval_naive_shooter.py                # Shooter eval (headless stats or viewer)
  eval_naive_goalkeeper.py             # Goalkeeper eval (headless stats or viewer)
shell/
  train_shooter.sh                     # Two-stage training orchestration
```

## Settings (`config/settings.yaml`)

```yaml
ball:              # radius=0.10, mass=0.35
goal:              # width=3.0, height=1.8
penalty_spot:      # distance_from_goal=4.0
scene:
  goal_pos: [0,0,0]
  shooter_behind_ball: 1.0
  motion_origin_offset: [-5.6, 0, 0]    # training: not used; play/eval: default (0,0,0)
  motion_yaw_offset: 1.5708             # training: not used; play/eval: default 0
  eval_ball_pos: [0, -1.5, 0.11]       # eval ball position (motion-local coords)
  eval_goal_pos: [0, -5.5, 0]          # eval goal position (motion-local coords)
episode_length_s: 10.0                 # shooter
goalkeeper_episode_length_s: 3.0       # goalkeeper
```

## Acknowledgements

Built for CS 2810 (Spring 2026). This project uses motion data and design references from [HumanoidSoccer](https://github.com/TeleHuman/HumanoidSoccer) and [Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper). If you use this template, please cite:

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
