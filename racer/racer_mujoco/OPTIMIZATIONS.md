# Throughput optimizations (queued for after the stack-D full-track run)

Timing analysis 2026-07-23 (benchmarked under CPU contention with the live run — absolute ms ~2x
inflated, shares robust). Per 64-step tick: SAC updates ~88%, buffer adds ~5%, env ~4%, obs ~1%.
Observed 51 st/s under stack D (vs 140 for stack C — DroQ+SIL quadrupled per-update cost).

Ranked by expected win / effort:

1. **Batched buffer add** (easy, immediate). `buffer.add` costs ~383 us/transition — 7 one-row
   tensor writes + as_tensor conversions in Python, called 2-6x per env step (mirror x2, HER x2).
   Add `ReplayBuffer.add_batch(o, a, r, no, d, g, ret)` taking numpy arrays and writing with slice
   assignment (handle ring wrap with two slices). Flush each episode as ONE batched call (main,
   mirrored, HER packs each). ~100x faster on the add path (~5% overall + scales with multipliers).

2. **Learner/collector overlap** (medium effort, biggest structural win — also the VQ1 design).
   Move SAC updates off the env-stepping path. Thread version first (~30 lines): a learner thread
   loops `agent.update()` while the main thread steps envs; torch CPU kernels and mj_step both
   release the GIL, so overlap should be substantial. If GIL contention disappoints, upgrade to a
   learner PROCESS with shared-memory replay (true parallelism; env-side sps then governed only by
   env+obs+act ≈ 250+ st/s). On VQ1 this same split converts real-time control-loop dead time into
   unlimited gradient steps — build it once, use it in both places.

3. **torch.compile the update** (one experiment). The update is hundreds of tiny CPU ops (LN,
   dropout, cat, stack, 3 backwards) — inductor fusion could give 2-4x on the DOMINANT cost. Try
   `torch.compile(mode="reduce-overhead")` on actor/critic forwards or wrap the update; verify
   numerics + Windows CPU support. If it works, this stacks with #2.

4. **Batch/UTD retune** (trivial). b=2048 still beats b=1024 on samples/s (2720 vs 2054). Try
   `--updates 2 --batch 2048` (same samples/tick as 4x1024, ~25% less wall) or `--updates 3
   --batch 2048` (1.5x samples, similar wall). DroQ tolerates the reduced grad-step count.

5. **SIL every k-th update** (trivial). SIL adds 2 forwards per update; running it on every 2nd-4th
   update keeps the anchor with ~half the overhead.

6. **Cache next_obs** (easy). `obs_of` runs twice per env step (transition next_obs + next tick's
   obs). Cache per env at step end and reuse -> halves obs cost (~1%).

7. **Euler integrator experiment** (trivial, verify quality). RK4 does 4 dynamics evals/step;
   `integrator="Euler"` (or implicitfast) at timestep 0.004 likely fine for a box+forces ->
   env.step ~3x cheaper (6-gate model doubled it to 0.6 ms).

8. **sps metric hygiene** (trivial): exclude greedy-eval wall time from the steps_per_s calc so
   the dashboard throughput chart reflects training throughput only.

Notes:
- Keep an ablation eye on DroQ itself: if stack D's sample-efficiency gain doesn't out-earn its
  ~2x update cost, consider LayerNorm-only critics (cheaper, most of the stabilization) or
  dropping critic_dropout to 0 and keeping the Polyak eval actor as the stability mechanism.
- All numbers re-measure cleanly once the box is idle; re-run the microbench in train_mj history
  (bench snippet in session notes) before/after each change.
