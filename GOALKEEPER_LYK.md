# Goalkeeper LYK: ballistic residual policy

This branch keeps the strongest existing path from `keeper-qbs` and adds one
targeted improvement: a frozen distilled goalkeeper plus a small PPO-trained
residual that receives explicit ballistic timing features.

## Why this change

The distilled goalkeeper already dives coherently but misses hard regions
because the policy must infer the ball's future crossing point from history.
`GoalkeeperBallisticResidual` keeps the distilled 960D MLP frozen and trains only
a bounded residual head.  Internally it computes:

- time and `(y, z)` at the keeper plane `x = 0`
- time and `(y, z)` at the goal plane `x = -0.5`
- incoming `vx` and ball speed

The residual is zero-initialized, so training starts exactly at the base policy.

## Train

```bash
MUJOCO_GL=egl python scripts/train_ballistic_residual.py \
  --base src/assets/soccer/weight/goalkeeper_distilled_v3.pt \
  --out logs/lyk/goalkeeper_ballistic_residual.pt \
  --num-envs 1024 --warmup 30 --block-iters 20 --blocks 40 \
  --eval-resets 3 --lr 1e-4 --std 0.06 --residual-scale 0.25
```

## Evaluate

```bash
MUJOCO_GL=egl python scripts/eval_naive_goalkeeper.py \
  --headless --num-trials 50 \
  --checkpoint logs/lyk/goalkeeper_ballistic_residual.pt
```

For failure structure:

```bash
MUJOCO_GL=egl python scripts/diagnose_gk.py \
  --checkpoint logs/lyk/goalkeeper_ballistic_residual.pt \
  --ballistic-residual --num-envs 256 --batches 8
```

## Deploy

`scripts/api_server.py` now detects this checkpoint via its
`ballistic_residual` metadata, rebuilds the frozen base path, and uses the same
keeper observation/history layout as training.
