# racer_state — overnight run notes

State-based SAC on the NEW sim (ground-truth odometry + gate positions). Single local process,
online SAC, MLP (~72k-param actor). Trains on exact drone→gate distance — no detector, no noise.

## How it was launched (DETACHED — survives the Claude/terminal session)

Launched via WMI `Win32_Process.Create` so the run keeps going even if the terminal/session closes
(an earlier tracked launch got killed when the session backgrounded). Output goes to log FILES:
- Training log: `racer_state/runs/train.log`
- Dashboard log: `racer_state/runs/dash.log`

**Check status:** `Get-Content racer_state/runs/train.log -Tail 30`
**Find the procs:** `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` (look for `racer_state.train` / `.dashboard`)
**Stop it:** kill those python pids (`Stop-Process -Id <pid>`), or the `cmd.exe` wrappers.
**Relaunch (detached):** re-run the WMI `Create` with
`cmd.exe /c "set PYTHONIOENCODING=utf-8 && "<aigp-venv python>" -u -m racer_state.train > runs\train.log 2>&1"`
and cwd = `PyAIPilotExample-v1`. Add `--resume` to continue from `runs/actor.pt`.

## What's running

- **Training**: `python -m racer_state.train` (ai-grand-prix venv). Flies gate 0 → chains toward
  gate 1+ (episode continues on a pass). Fresh run launched ~overnight.
- **Dashboard**: `python -m racer_state.dashboard --port 8050` → **http://localhost:8050**
  (local, no tunnel). Charts: min-distance, success rate, reward, losses, α, entropy, reward-EMA,
  adaptive pitch-penalty weight, outcome mix, caps, and timing (ms/update, ms/step, throughput).

## What happens automatically overnight

- **Every 25 episodes**: a greedy self-eval (10 episodes), logs gate-0 and gate-1 pass counts.
- **Gate-0 milestone** (≥8/10 greedy passes): saves `runs/gate0_reached.pt` + records
  `runs/gate0_reached.mp4` (camera frames of the flight). It then keeps training toward gate 1.
- **Gate-1 milestone** (≥5/10 reach gate 1): saves `runs/gate1_reached.pt` + `gate1_reached.mp4`.
- **Every 100 episodes**: a progress-reel video `runs/videos/epNNNNN_g{gates}_d{min_dist}.mp4`.
- **Checkpoint** `runs/actor.pt` every 10 episodes (+ `train_state.json` for `--resume`).
- **Fallback**: if it still can't pass gate 0 by episode 300 (greedy gate-0 < 3/10), it auto-adds
  **thrust** to the adaptive pitch+roll penalty (live, no restart) to damp oscillation.

## Reward shape (what it maximizes)

`r = progress(meters closed to active gate) − time − adaptive_pitch+roll_penalty`, plus:
- **+25** per gate passed, **+50** finish (last gate).
- **Failure** (crash/away/stray) = **distance-scaled**: `+2` right at the gate → ~0 at 4 m →
  exponentially worse far (clipped −30). Rewards committing; punishes bailing far. Failures are
  TRUE terminals (fixes an earlier critic divergence).
- **away** terminal = 5 consecutive frames receding from the gate.

## Key knobs (`config.py`)

- Thrust HOVER-CENTERED (`thrust_hover=0.28`): neutral action hovers → it can actually fly.
- Pitch+roll: no hard cap; adaptive exp penalty, weight scales with the reward EMA (tightens when
  doing worse, loosens when better). Yaw on a slow cap.
- `elite_frac=0.25`: 25% of each SAC batch from an elite buffer of successful/close trajectories.

## Morning checklist

1. Dashboard http://localhost:8050 — success-rate and min-distance curves; is gate 0 solved?
2. `ls runs/*.pt runs/*.mp4 runs/videos/` — milestone checkpoints + videos; watch the progress reel.
3. Training log tail — look for `*** MILESTONE gate0_reached` / `gate1_reached`, `[EVAL epN]` lines,
   `[FALLBACK]` (thrust penalty engaged?), and that c_loss stayed bounded (no divergence).
4. If gate 0 is solved and it's working gate 1, it's already chaining; if stuck, the eval/fallback
   lines say where.

## Gotchas handled

- **Gate layout decode** occasionally garbles (rare reassembly race) → **consensus across several
  resets** at startup (a lone garbage decode is outvoted). If a run ever insta-strays every episode
  with a weird gate-0 position, that's this — restart.
- Windows stdout is cp1252: run with `PYTHONIOENCODING=utf-8` (α etc.).
- Vision port 5600 is free during training (control uses only MAVLink 14550) → videos don't disrupt.
