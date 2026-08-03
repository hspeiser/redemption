# VQ2 straight-speed candidate v1

## Outcome

The protected gate-0-to-10 prefix is preserved byte-for-byte.  A cap-12
reference suffix replaces gates 11-16, with a 30-row cubic-Hermite bridge at
the gate-11 handoff.  Gates 11-13 request 1.05x reference velocity.  Gate 15
uses a 0.25 trajectory blend, 2.0x lateral feedback, and -0.02 lateral trim to
avoid the right-border drift seen in the first live probe.

Frozen artifacts:

- `data/vq2_hybrid_fastprefix_a2suffix_g11_v2.npz`
- `data/vq2_straight_speed_candidate_v1.json`
- `data/vq2_straight_speed_candidate_v1_audit.json`

## Evidence

Exact-command 256-world audits on the three-model learned dynamics pool:

| Configuration | Finish rate | Median finish |
| --- | ---: | ---: |
| Protected champion | 20.7% | 36.07 s |
| Straight-speed candidate v1 | 27.0% | 35.17 s |

The candidate's final 256-world audit had no failures at gate 15.  Its finish
p90 was 35.81 s.

A first live version with full trajectory authority reached gate 15 in 32.88 s
before drifting 0.92 m right of its reference and colliding.  The gate-15
repair reduced offline gate-15 crossing-error p90 from about 0.76 m to 0.42 m
and eliminated gate-15 failures on its 128-world selection audit.

The repaired candidate has not yet received a useful live full-course probe.
The subsequent simulator session became unhealthy: eight consecutive
candidate/control runs failed before gate 6, so those runs contain no evidence
about gates 11-16.  A manual simulator relaunch is required before the next
ABBA probe; the harness must not restart the simulator process itself.

## Next live probe

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .remote\launch_live_full17_fastprefix_abba.ps1 `
  -CandidateConfig C:\Users\henry\Desktop\ai-gp\data\vq2_straight_speed_candidate_v1.json `
  -ChampionConfig D:\ai-gp\champions\vq2_36s_20260802\training_session\config.json `
  -Cycles 2
```

Use the protected-candidate-candidate-protected sequence.  Promote only after
the candidate finishes and its gate-10-to-13 split is faster in a session where
at least one protected arm also reaches gate 11.

## Rejected approaches

- Gate 3-4 is already flown near 11-12 m/s; only about 0.2 s remains there.
- Time-compressing the existing demo did not materially accelerate tracking.
- Phase pitch pulses saved at most about 0.37 s and reduced reliability.
- A late-gate CEM residual schedule saved about 0.37 s; this exposed the
  residual controller's ceiling but did not reach the multi-second target.
- Straight-speed PPO did not create a coordinated faster entry in the bounded
  run.
- Velocity scales above 1.05 traded reliability too aggressively: 1.10,
  1.15, and 1.20 achieved 34.83 s, 34.50 s, and 34.27 s medians but fell to
  21.9%, 17.7%, and 7.3% finish rates on the 96-world sweep.
