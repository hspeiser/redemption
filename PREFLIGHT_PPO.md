# PPO Sim-2-Sim Pre-Flight (VQ2)

## Artifacts (data/models/)
| artifact | surrogate (clean) | surrogate (structured reloc noise) | character |
|---|---|---|---|
| `vq2_ppo_conservative.pt` | 99.3% @ 20.97s | 96.4% | Henry-like, zero saturation — **attempt 1** |
| `vq2_ppo_champion.pt` | 99.4% @ 18.07s | 97.4% | bang-bang racer — attempt 2 |
| ~~`vq2_ppo_flight.pt`~~ | 97.0% | 94.8% | fine-tune REJECTED: faster (17.9s) but less reliable (exploration-std blowup on resume degraded the deterministic policy). Flight lineup stays conservative -> champion. |

Structured noise = coast-and-snap estimator error measured on the real
certified lap (0.4–1.5 m reloc jumps every 6–14 s + 2–15 cm OU) — heavier
than the real lap exhibits (5 jumps ≤1.47 m in ~50 s).

## Verified before any sim contact
- Harness constructs the live stack EXACTLY as the v57 trainer does
  (VisionRX + 4 checkpoints + calib + v29 process-isolation config that
  produced 2 stale episodes in 500).
- Dry-run: policy pipeline verified on the 1,164 logged (obs, action)
  pairs; conservative correlates + on all axes with Henry's actions;
  all outputs finite; no saturation (conservative).
- Map: `data/vq2_map_final_live.json` = click-certified descent-aware
  geometry (EKF holds 3.4 cm / 10.6 cm p90 on the real lap) + the live
  anchor field `spawn_to_gate0`. NOTE: the SAC trainer's runtime map
  diverges from certified geometry by up to 36.8 m on gates 10–16 — do
  NOT fly the policy with that map.

## Preconditions (Henry)
1. Stop the live SAC trainer (v57+): it holds the single-instance mutex
   and both UDP ports. `pkill -f train_vq2_sac_live` or Ctrl-C its window.
2. No manual RC session, no frame tap on UDP 5600.
3. Simulator running at the VQ2 course, healthy real-time (idle GPU).

## Flight command (attempt 1)
```
.venv-train\Scripts\python.exe scripts\eval_vq2_ppo_live.py ^
  --episodes 5 --policy data\models\vq2_ppo_conservative.pt
```
Logs: console + `data/ppo_live_eval.jsonl` (+ the env's own telemetry).

## Abort criteria
- Any episode ends `failure: localizer_exception` twice in a row → stop,
  inspect; do not keep resetting.
- `stale_sensor_stream` on >1 episode → machine contention; stop and
  check GPU/CPU load before continuing.
- Physical damage impossible (sim), so otherwise let all 5 episodes run.

## Escalation ladder
1. Conservative artifact (this flight).
2. `vq2_ppo_flight.pt` / champion (faster; only after conservative
   completes laps).
3. If estimator-shaped failures (sigma spikes, reloc jumps preceding
   loss): retrain with recorded error traces — extraction scripts ready,
   lrspeiser's 92 closed-loop runs + our traces are the source.
4. If a specific gate fails repeatedly: drill it with random starts in
   the surrogate (minutes per iteration).
