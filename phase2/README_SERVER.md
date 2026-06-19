# Phase 2 Tournament Server

This document is for the teaching-team machine that runs the tournament web
console and match executor.

## Start The Web Console

```bash
/home/xiaojx/miniconda3/envs/proj-mjlab/bin/python phase2/tournament_server.py
```

Console URL:

```text
http://10.15.89.71:54905
```

## Ports

Internal container ports and external proxy URLs:

| Purpose | Internal | External |
|---|---:|---|
| Web console | 6006 | http://10.15.89.71:54905 |
| Viser slot 0 | 7000 | http://10.15.89.71:43862 |
| Viser slot 1 | 7001 | http://10.15.89.71:37448 |
| Viser slot 2 | 7002 | http://10.15.89.71:65025 |
| Viser slot 3 | 7003 | http://10.15.89.71:17192 |

## Start A Match

Open the web console and submit:

- Shooter team name
- Shooter API URL
- Goalkeeper team name
- Goalkeeper API URL
- Number of trials, default `10`

The API URLs must be reachable from this tournament server.

## Match Behavior

- Each episode lasts `5s`.
- The runner waits on the Viser `Start Trials` button before official trials
  begin.
- Up to 4 matches run concurrently, one per Viser slot.
- Viser displays a Match Scoreboard panel with current trial and prior trial
  outcomes: `S` for shooter goal and `G` for goalkeeper save.

## Student API Requirement

Each student policy server must expose:

```text
POST /reset
POST /act
```

`POST /act` receives raw state:

```json
{
  "shooter": {
    "root_pos": [],
    "root_quat": [],
    "root_lin_vel": [],
    "root_ang_vel": [],
    "joint_pos": [],
    "joint_vel": [],
    "last_action": []
  },
  "goalkeeper": {
    "root_pos": [],
    "root_quat": [],
    "root_lin_vel": [],
    "root_ang_vel": [],
    "joint_pos": [],
    "joint_vel": [],
    "last_action": []
  },
  "ball": {
    "pos": [],
    "vel": []
  }
}
```

and returns:

```json
{
  "action": [[0.0, 0.0, 0.0]]
}
```

The returned action must contain exactly 29 floats.

## Results

Each match writes:

```text
phase2/results/<match_id>.json
phase2/logs/<match_id>.log
```

The JSON includes timestamp, team names, API URLs, minimal config audit,
per-trial results, and summary.

## Client Smoke Test

On a separate client/student machine, run:

```bash
python phase2/eval/zero_policy_eval.py --role shooter --host 0.0.0.0 --port 8000
python phase2/eval/zero_policy_eval.py --role goalkeeper --host 0.0.0.0 --port 8001
```

Then submit the client-reachable URLs in the web console.
