# racer_mujoco — a fast MuJoCo twin of the VQ1 racer for RL iteration

> **TL;DR — FULL TRACK SOLVED (stack D + learner thread):** all **6 gates 20/20 greedy in 73.9
> minutes / 803k steps from scratch** — fixed spawn, no curriculum, no expert, single continuous
> run on CPU. Command (run_full.bat):
> `--envs 64 --learner 1 --utd_cap 2.0 --min_utd 0.06 --cdrop 0 --sil_every 2 --batch 1024
>  --ent -2 --away 15 --nstep 5 --her 2 --silw 1.0 --gates 6 --episode_s 40 --mirror 1`
> Milestones: g0 20.0 min, g1 +7.3, g2/g3 +17.8 (together), g4/g5 +28.8 (together).
> Final flight: 164 m descending track in ~13.5 s (~12 m/s). Videos: gate5_solved.mp4,
> full_track_follow.mp4 / full_track_iso.mp4 (record.py). Earlier stack C (single-gate proof):
> gate 0 in 20 min — see "Stack C" below. All buffer-side machinery -> VQ1-portable.

## Why this exists

The VQ1 sim is faithful but **slow** (real-time UDP telemetry, one env, reset settle delays). SAC needs
hundreds of thousands of environment steps, so every reward/curriculum idea took hours to test on VQ1.
This package is a **MuJoCo twin** that matches VQ1 where it matters (the same action interface, the same
15-dim observation, the same reward) but runs **~70× real-time single-env and ~600+ steps/s vectorized
across 64 envs on CPU**. We iterate the RL recipe here in minutes, then port the *winning recipe* (not the
weights) back to VQ1.

The twin is a *methodology accelerator*, not the deployment target. Anything that solves gate 0 here should
solve it on VQ1, because the observation is gate-relative and physics-agnostic — see "Porting to VQ1".

## The pieces

| file | what it is |
|---|---|
| `env.py` | `QuadEnv`: a rate-controlled quadrotor + gate 0, MuJoCo physics. Fast, headless. |
| `train_mj.py` | Vectorized SAC training loop (N parallel envs), curriculum, greedy eval, video, metrics. |
| `dashboard.py` | Live web dashboard on `:8060`, reads `runs_mj/metrics.jsonl`. |
| `run_curr.bat` / `run_dash.bat` | Detached launchers (survive the CLI session — see "Running"). |
| `runs_mj/` | Output: `metrics.jsonl`, `*.mp4` videos, `mj_gate0.pt` (solved actor). |

It reuses `racer_state/` for the parts that must stay identical to VQ1: `config.py` (all knobs),
`sac.py` (the SAC agent + replay/elite buffers), `reward.py` (`step_reward`), `geom.py`
(`rot_world_to_body`), `nets.py` (actor/critic MLPs).

## The environment (`env.py`)

- **Frame**: z-up (standard MuJoCo). Gate 0 sits `23.3 m` ahead at spawn height, a `2.7 m` opening
  (`GATE_HALF = 1.35`) whose normal is the x-axis.
- **Action** `a ∈ (-1,1)^4` = `(roll_rate, pitch_rate, yaw_rate, thrust)` — identical policy units to VQ1.
  Rates map to `a[:3] * RATE_SCALE (4 rad/s)` and are tracked by a body-rate P controller
  (`KP_RATE=25`) applied as a torque through `xfrc_applied`. Thrust is **hover-centered**:
  `thrust_norm = HOVER(0.28) + a[3]*THRUST_SPAN(0.35)`, so `a[3]=0` ≈ hover — this is what lets the
  policy discover level flight instead of falling out of the sky while it explores.
- **Observation** (15-dim, body-frame, built in `train_mj.obs_of`): `[rel_gate_pos/10, vel_body/5,
  body_rates/3, gravity_down_body, gate_normal_body]`. Everything is expressed **relative to the gate
  in the body frame** via `rot_world_to_body(quat, ...)`. This is the key to generalization: the policy
  never sees absolute world coordinates, so "fly through the gate in front of me" is the same task at any
  spawn distance and (later) for any gate.
- **Gate pass** (`env.step`): the drone crosses the gate plane forward (`prev_side < 0 <= side`) inside
  the opening radius (`radial < GATE_HALF`). Direction matters — this was a bug once (crossing is
  negative→positive), now verified with a teleport test.
- **Crash**: any contact (`ncon > 0`) or `z < 0.15` (hit the ground).
- **Curriculum hook**: `spawn_dist` controls how far *behind* the gate the drone spawns. `reset()` places
  it at `gate - spawn_dist` along x with small jitter. Default = full `23.3 m`.

## The reward (`racer_state/reward.py`)

Ground-truth, no detector noise, no EMA needed — odometry distance is exact:

```
r = w_prog*(prev_dist - curr_dist) - step_penalty     # dense progress toward the gate
  + gate_bonus (=50)                if passed          # the big carrot
  + fail_reward(curr_dist)          if crash/stray     # distance-scaled failure (see below)
```

`fail_reward(d)` (config): a **slight positive** near the gate (`fail_near_reward=2.0`) decaying
**exponentially worse the farther away** the failure happens (`-fail_scale*(exp(d/fail_tau)-1)`, clipped
at `-fail_clip`). This was the user's ask: "don't treat all failures the same — punish crashing far from
the gate hard, but a crash *on* the gate is almost fine (you were committing)."

## The learning recipe: distance curriculum (the important part)

**The core problem is discovery, not control.** A hand-coded expert crosses gate 0 ~100% of the time, so
the env is solvable; but SAC from scratch rarely stumbles onto the `+50` pass reward from 23 m out, and
once it learns "fly toward the gate" it parks *just short* rather than committing through (the last meter
has no gradient until you actually cross).

We tried the offline-RL fix — behavior-clone the expert (`seed_demos` + `bc_warmstart`), warm the critic,
regularize SAC toward the demos. **It doesn't hold**: BC gets 20/20 but is variance-prone, and plain SAC
destroys it immediately (offline→online critic overestimation on out-of-distribution actions + the entropy
term randomizing the policy). That's a genuine research rabbit hole; we stopped chasing it.

**The curriculum sidesteps all of it.** Spawn the drone *close* to the gate (`curr_start=4 m`), where
crossing is easy to stumble into → pure RL discovers the `+50` with no expert. Each time the greedy policy
passes `curr_thresh` (16/20), move the spawn back `curr_step` (2.5 m), carrying the learned crossing skill
outward. Because the observation is gate-relative, the task is identical at every distance — only the
approach gets longer. Repeat until spawn reaches the full 23.3 m at 20/20. Pure RL, no BC variance,
no offline-to-online gap.

Knobs (in `train_mj.py` args): `--curr` (enable), `--curr_start 4`, `--curr_step 2.5`, `--curr_thresh 16`.
Exploration is set by `--ent` (target entropy): too high (`-1`) and the greedy mean won't sharpen enough
to commit; too low (`-4`) and it never explores enough to discover. `--ent -2` is the working middle.

## Running

The CLI session **backgrounds and kills** child processes on its own, so we launch **detached** via WMI
`Win32_Process.Create` wrapping a `.bat` (this survives). To start everything:

```powershell
# training (detached) — edit run_curr.bat to change knobs
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c "<...>\racer_mujoco\run_curr.bat"' }
# dashboard (detached)
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c "<...>\racer_mujoco\run_dash.bat"' }
```

Current `run_curr.bat` command:
```
python -m racer_mujoco.train_mj --envs 64 --updates 8 --curr --curr_start 4 --curr_step 2.5 --away 15 --ent -2
```

- **Dashboard**: http://localhost:8060 — 9 live charts (greedy eval passes/20, train pass rate, min-dist,
  reward, throughput, Q, critic loss, alpha, outcome mix).
- **Videos** (`--video_every 1000`, default): a chase-cam greedy mp4 is saved to `runs_mj/` periodically,
  on every curriculum advance (`mastered_d<dist>_ep<ep>.mp4`), and on solve (`gate0_solved.mp4`). Best-effort
  — if the GL renderer fails, video self-disables and training continues.
- **Logs**: `runs_mj/curr.log` (stdout), `runs_mj/metrics.jsonl` (per-episode records).
- **Solved artifact**: `runs_mj/mj_gate0.pt` (`{"actor": state_dict}`) is written when greedy hits 20/20 at
  full distance.

## Success criterion

Gate 0 is "solved" when a **greedy** (deterministic mean-action) eval passes **20/20** episodes at the
**full 23.3 m** spawn distance. The curriculum-advance ladder (4 → 6.5 → 9 → ... → 23.3 m) is the progress
signal along the way.

## Result (SOLVED)

Run: `--envs 64 --updates 8 --curr --curr_start 4 --curr_step 2.5 --away 15 --ent -2`, CPU, ~90 steps/s.
Gate 0 reached **20/20 greedy at the full 23.3 m** at episode 5000 (~460k env steps, well under an hour).
The ladder:

| spawn distance | solved by episode | note |
|---|---|---|
| 4.0 m  | 600  | first discovery of the pass |
| 6.5 m  | 1200 | |
| 9.0 m  | 1600 | |
| 11.5 m | 1800 | fast (skill transferring) |
| 14.0 m | 2800 | harder cliff, dipped then recovered |
| 16.5 m | 3800 | mid-rung churn (mind spiked to 15 m) then recovered — **transient, not collapse** |
| 19.0 m | 4000 | train pass% 88 |
| 21.5 m | 4200 | train pass% 100 |
| **23.3 m** | **5000** | **20/20 full distance — solved** |

Artifacts in `runs_mj/`: `mj_gate0.pt` (actor), `gate0_solved.mp4`, and a `mastered_d<dist>_ep<ep>.mp4`
chase-cam clip at every rung. Each rung resets greedy to ~0/20 (the longer approach is a genuinely harder
task) then re-clears in a few hundred episodes; the dips at 14 m and 16.5 m were transient SAC churn (Q,
critic loss, alpha all stayed healthy), so **patience beat knob-twiddling** — no restart or retune was needed.
Checkpoint/resume (`rung_d<dist>.pt`, `--resume`) was added afterward so future runs preserve each rung.

## Stack C — the curriculum-free recipe (current best)

The curriculum worked but was slow (85 min) and needs spawn control VQ1 doesn't have. Profiling the
curriculum run showed: 97% of wall time in dispatch-bound SAC updates (73 ms at batch 256, saturating
~8.7k samples/s at batch 1024+); 62 of 85 min lost to abrupt all-envs distance jumps + 200-ep eval
gating; a reward exploit (the near-gate "commitment" bonus also paid for RETREATING via the away
terminal, making "approach then hover at 1.6 m" a positive local optimum); and the elite buffer
never being fed in the vectorized loop. Stack C fixes all four and replaces the curriculum with
buffer-side machinery:

1. **HER virtual-gate relabeling** (`--her 1`, `her_relabel`): every FAILED episode gets a virtual
   gate planted on its actual flight path (center = a point it crossed, normal = its velocity
   there); obs/rewards are rebuilt wrt that gate and the episode becomes a successful +50 pass of
   the virtual gate. Because the obs is purely gate-relative, "thread the gate wherever it is"
   transfers to the real gate. Discovery signal from episode one, **no spawn control needed**.
   It forms an implicit ladder: early relabels teach short hops -> flights extend -> virtual gates
   land farther out -> compounding (visible as mind 23.3 -> 17 -> 10.7 -> 4.5 -> 2.3 over the run).
2. **n-step returns** (`--nstep 5`, `nstep_pack`): episodes are flushed at episode end as k-step
   transitions with per-transition bootstrap discount `gamma^k` stored in the buffer (`g` column),
   so the +50 terminal reaches 5 states back immediately. 1-step (VQ1) and 5-step data coexist.
3. **Advantage-filtered self-imitation** (`--silw 1.0`, in `sac.update`): elite transitions carry
   their realized return-to-go (`ret` column); the actor clones an elite action ONLY where that
   return beats the critic's estimate of the current policy (`ret > minQ(o, pi(o))`). Anchors only
   to provably-better behavior and anneals itself away — this is what plain BC couldn't do (it
   collapsed). Effect: greedy snapped 0/20 -> 20/20 in ONE eval window the moment real passes
   landed in the elite buffer.
4. **Away-exploit fix**: away-terminal reward is now `min(0, fail_reward(d))` — retreating never
   pays; the slight-positive near-gate failure stays only for genuine crashes (commitment).
5. **Big-batch restructure** (`--batch 1024 --updates 3`): ~2.4x samples/s on this dispatch-bound
   box for ~1.4x the wall speed.

**Result:** 20/20 greedy at the fixed 23.3 m spawn at ep 2000 / **168k steps / 20.0 min**, stable
across three consecutive evals (20/20, 19/20 at 76% train pass, 20/20). vs curriculum: **2.8x fewer
steps, 4.3x less wall-clock, and no sim-side spawn help.** Artifacts: `mj_gate0_stackC.pt`,
`metrics_stackC.jsonl`, `gate0_solved.mp4`.

## Scaling to gate 1 and beyond

The whole design anticipates multi-gate. Here is the concrete path:

1. **Add the gates to `env.py`.** The VQ1 track is 6 gates; their NED positions are captured by
   `racer_state/mav_io.capture_gates`. Add gate bodies to the MJCF and an ordered `gate_list`. Keep the
   observation gate-relative to the **active** gate only (already the case) so the network input shape and
   meaning never change.
2. **Re-target on pass, don't terminate.** Today gate-0 pass ends the episode (`passed → done`). For
   multi-gate, on pass **increment `active_gate`** and keep flying — recompute `gate_pos/gate_normal` for
   the new active gate and reset the progress baseline (`prev_dist = dist to new gate`). This is exactly the
   `active_gate` machinery already in VQ1's `train.py` and half-present in `env.py` (`self.active_gate = 1`
   on pass). **Critical correctness note the user flagged:** reward and episode-end must key off the gate
   the drone is *currently* chasing — only *after* it passes gate 0 does gate 1's position enter the reward.
   Never let the next gate's geometry leak into scoring before the current one is passed. (This was a real
   bug on VQ1, fixed by invalidating `active_gate` on reset and gating obs/reward on the current index.)
3. **Curriculum, per gate.** The same distance curriculum generalizes: once gate `k` is solved at full
   distance, the "spawn" for gate `k+1` is simply *starting the episode already through gate k* — i.e. the
   drone arrives at gate `k+1` with whatever approach the policy produces. In practice: freeze the gate-0
   recipe, extend the episode cap (`--steps`/`episode_s`), and let progress reward + re-targeting pull it
   through gate 1. If gate 1 proves hard, apply the *same* close-spawn curriculum locally to gate 1 (spawn
   the drone just past gate 0, close to gate 1) and ladder back.
4. **Keep-next-gate-visible shaping (later gates).** Distant gates need the drone to exit gate `k` already
   oriented toward gate `k+1`. Add a small yaw-toward-next-gate bonus or a "next gate in view" term so it
   doesn't over/undershoot the line. Keep it small so progress still dominates.
5. **Curriculum target knob.** Train until `N/20` reach gate `k`, then advance the target to `k+1`
   (`success_gate_target`). Document the observed `k → k+1` transition and the episode-cap / entropy values
   that worked at each stage.

## Porting the winning recipe to VQ1

Port the **recipe, not the weights** (physics differs slightly, so weights won't transfer, but the *method*
does):
- Same 15-dim gate-relative obs (already shared code in `racer_state`).
- Same reward (`step_reward`, shared).
- Same distance curriculum: VQ1's `SIM_RESET` can place the drone; if VQ1 can't spawn arbitrarily close,
  emulate the curriculum by only *counting/rewarding* from a moving virtual start, or start from VQ1's
  existing `gate0_reached.pt` and fine-tune outward.
- Same `--ent -2` exploration and `curr_thresh`/`curr_step` schedule.
VQ1's slower wall-clock is the only cost; the recipe is identical.

## History / dead-ends (so we don't repeat them)

- **BC warm-start + SAC**: 20/20 after BC, collapses to 0/20 under SAC. Tried bc-anchor weight up to 50,
  critic-only warm-up, pinned-low alpha — all collapse. Offline→online extrapolation. Abandoned for the curriculum.
- **`--ent -4` / `alpha0 0.1`**: over-sharpened, killed exploration, never discovered the pass. Too low.
- **`--ent -1`**: good discovery (train pass% climbed to ~38%) but greedy mean parked ~1.7 m short — too
  stochastic to commit. `--ent -2` is the compromise.
- **Gate-pass direction bug**: `passed` originally checked positive→negative crossing; the drone crosses
  negative→positive. Fixed + verified.
- **SAC update dispatch overhead**: ~13 ms fixed per update regardless of batch/device on this box; fixed by
  vectorizing 64 envs so one update amortizes over many transitions.
