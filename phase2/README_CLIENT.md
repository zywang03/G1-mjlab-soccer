# Phase 2 Client API

This document is for students or test clients that submit policies to the
tournament server.

## What To Run

Run one API server for the shooter policy and one API server for the goalkeeper
policy:

```bash
python phase2/api_server.py --checkpoint <shooter.pt> --host 0.0.0.0 --port 8000 --task shooter
python phase2/api_server.py --checkpoint <goalkeeper.pt> --host 0.0.0.0 --port 8001 --task goalkeeper
```

The tournament server must be able to reach these URLs. If your runtime exposes
ports through a proxy, submit the external proxy URLs, not `localhost`.

## Protocol

Your server must provide:

```text
POST /reset
POST /act
```

`POST /reset` clears recurrent hidden state, history buffers, or any other
episode-local policy state:

```json
{"status": "ok"}
```

`POST /act` receives raw MuJoCo state for both robots and the ball. You should
convert this raw state into the observation format used by your own policy.

The response must be:

```json
{
  "action": [[0.0, 0.0, 0.0]]
}
```

The inner list must contain exactly 29 floats.

## Zero-Policy API Test

For a no-checkpoint API smoke test, run:

```bash
python phase2/eval/zero_policy_eval.py --role shooter --host 0.0.0.0 --port 8000
python phase2/eval/zero_policy_eval.py --role goalkeeper --host 0.0.0.0 --port 8001
```

Submit the two client-reachable URLs in the tournament web console.

## Notes

- The tournament server runs the MuJoCo match. Your API server only returns
  actions.
- Each official Phase 2 episode lasts `5s`.
- Falling over does not terminate the episode.
- Observation spaces are decoupled; customize `compute_shooter_obs()` and
  `compute_goalkeeper_obs()` in `phase2/api_server.py` to match your training.
