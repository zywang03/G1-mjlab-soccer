# Phase 1 Goalkeeper Evaluation

- Generated: 2026-06-26 09:34:22 UTC
- Checkpoint: `logs/keeper_targeted_repair/distilled/targeted_moe6_residual_scale0p01_bc0p6.pt`
- Eval task: `Eval-Goalkeeper`
- Seeds: `42, 2810, 202606`
- Trials per seed: `50`
- Total trials: `150`
- Max steps per trial: `150`

## Metric

A goalkeeper trial is counted as successful when the ball does not enter the goal frame before timeout. The reported block rate here is the pooled success count divided by total trials.

```text
block_rate = success_times / total_trials
phase1_score = max(0, (block_rate - 0.8) / 0.2) * 30
```

## Result

- Block rate: `142/150 = 94.67%`
- Mean block rate across seeds: `94.67%`
- Phase 1 GK score from pooled block rate: `22.00/30.00`
- Evaluator score from mean seed rate: `22.00/30.00`

## Per-Seed Breakdown

| Seed | Successes | Trials | Block rate |
| ---: | ---: | ---: | ---: |
| 42 | 50 | 50 | 100.00% |
| 2810 | 45 | 50 | 90.00% |
| 202606 | 47 | 50 | 94.00% |

## Reproduction Command

```bash
/data/mjlab-cu126/bin/python scripts/eval_goalkeeper_official_seeds.py --checkpoint logs/keeper_targeted_repair/distilled/targeted_moe6_residual_scale0p01_bc0p6.pt --seeds 42 2810 202606 --trials-per-seed 50 --max-steps 150 --score-points 30.0 --score-threshold 0.8 --out /data/G1-mjlab-soccer/doc/keeper_phase1_goalkeeper_eval.json --log /data/G1-mjlab-soccer/logs/keeper_phase1_eval/targeted_moe6_residual_scale0p01_bc0p6_3seeds_50each.log --task-id Eval-Goalkeeper --parallel-seeds --seed-gpus 0 1 2
```

## Artifacts

- Raw JSON: `/data/G1-mjlab-soccer/doc/keeper_phase1_goalkeeper_eval.json`
- Full log: `/data/G1-mjlab-soccer/logs/keeper_phase1_eval/targeted_moe6_residual_scale0p01_bc0p6_3seeds_50each.log`
