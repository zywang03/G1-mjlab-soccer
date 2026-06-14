# Goalkeeper Improvements

This document describes the current goalkeeper training path after aligning the
course template more closely with the Humanoid-Goalkeeper reference method.

The goal is not a byte-for-byte reproduction of the original IsaacGym project.
It is a course-ready MuJoCo/mjlab implementation of the key idea from:

> Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints

## Summary

The goalkeeper method now has three pieces:

1. **Target-conditioned ball reset**: each parabolic launch stores the sampled
   catch-plane interception target.
2. **Target-conditioned critic/reward**: the critic sees the true end target,
   and `ee_reach` rewards the region-selected hand reaching that target.
3. **Debug/eval loop**: `debug_goalkeeper_rollout.py` reports why a checkpoint
   blocks or fails, including contact-like distance, target-hand distance, fall
   posture, and stop-ball reward timing.

## Method

### Ball Launch And Target

The eval/training ball reset samples:

- start position from +x in front of the robot
- landing region among six goal regions
- end position behind the robot at -x
- parabolic flight time

During reset, the code now also computes a catch-plane target at local `x=0.1`.
This follows the reference idea that the policy should be conditioned on where
the robot should intercept the ball, not only the instantaneous ball position.

Implemented in:

- `src/tasks/soccer/mdp/goalkeeper_ball_reset.py`

### Observation

The critic observation term `end_target_pos` now uses the sampled catch-plane
target stored by reset. It falls back to current ball position only for older
reset paths that do not provide a target.

Implemented in:

- `src/tasks/soccer/mdp/goalkeeper_obs.py`

### Reward

The `ee_reach` reward now selects the hand by region:

- regions `0, 2, 4`: left hand
- regions `1, 3, 5`: right hand

It rewards the selected hand reaching the sampled end target. This replaces the
previous behavior that rewarded all hands/feet for chasing the current ball.

Two coordinate bugs were also fixed:

- `stop_ball` now requires the ball to pass behind the robot toward `-x`.
- `no_retreat` now penalizes retreating toward the goal at `-x`.

Implemented in:

- `src/tasks/soccer/mdp/goalkeeper_rewards.py`

## Training Task

The goalkeeper training config is now registered:

```bash
python scripts/list_envs.py
```

Expected task:

```text
Unitree-G1-Goalkeeper-Train
```

Smoke test on Mac/CPU:

```bash
python scripts/train.py Unitree-G1-Goalkeeper-Train \
  --gpu-ids None \
  --agent.max-iterations 0 \
  --agent.logger tensorboard \
  --agent.run-name smoke_goalkeeper
```

Server training starter:

```bash
python scripts/train.py Unitree-G1-Goalkeeper-Train \
  --env.scene.num-envs 2048 \
  --agent.logger tensorboard \
  --agent.run-name gk_target_conditioned_v1
```

The Mac command needs `--gpu-ids None` because the default is GPU id `0`.

## Debugging

Run a diagnostic rollout:

```bash
python scripts/debug_goalkeeper_rollout.py \
  --checkpoint src/assets/soccer/weight/goalkeeper.pt \
  --num-trials 50 \
  --csv outputs/goalkeeper_debug.csv
```

Important fields:

- `blocked`: course eval outcome for the trial.
- `contact_like`: whether a hand/foot came near the ball.
- `fell_like`: whether posture crossed the fall-like gravity threshold.
- `min_ee_dist`: closest hand/foot distance to the ball.
- `min_target_hand_dist`: closest selected-hand distance to the catch target.
- `old_front_stop_step`: old front-side stop-ball trigger timing.
- `reward_stop_step`: corrected stop-ball reward trigger timing.

Use this before and after training. A better goalkeeper should improve block
rate while reducing target-hand distance and avoiding excessive fall-like cases.

## Evaluation

Course Phase 1 goalkeeper eval remains:

```bash
python scripts/eval_naive_goalkeeper.py \
  --headless \
  --num-trials 50 \
  --checkpoint <checkpoint.pt>
```

The debug script is not the grading script. It is for explaining failures and
checking whether training is learning the intended target-conditioned behavior.

## Current Status

Validated locally:

- Python compile check for modified goalkeeper modules.
- `Unitree-G1-Goalkeeper-Train` appears in `scripts/list_envs.py`.
- `train.py Unitree-G1-Goalkeeper-Train --agent.max-iterations 0` initializes.
- `debug_goalkeeper_rollout.py` reports finite `target_hand` distances.

Not yet validated:

- Full GPU training after the target-conditioned reward change.
- 50-trial block-rate improvement from a newly trained checkpoint.
- Phase 2 performance against PR2 shooter.

## Next Steps

1. Run a short GPU training job with `Unitree-G1-Goalkeeper-Train`.
2. Evaluate every checkpoint with `eval_naive_goalkeeper.py --num-trials 50`.
3. Diagnose failures with `debug_goalkeeper_rollout.py --num-trials 50`.
4. If target-hand distance improves but block rate does not, tune `stop_ball`,
   posture/fall penalties, and domain randomization schedule.
