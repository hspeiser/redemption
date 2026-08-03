# VQ2 straight-speed candidate v2

## Result

Candidate v2 completed a live official VQ2 run in **35.374649 seconds** on
2026-08-02.  The simulator's official elapsed time is recorded, not inferred
from the harness wall clock.  This improves the prior 37.62-second record by
about 2.25 seconds.

The successful run is episode 0 of session `20260802_191537`.  It was timing
healthy and used the multigate localizer, GPU dense vision at 10 Hz, crop
tracking, the protected fast prefix through gate 10, and the cap-12 suffix.

## Frozen artifacts

- `data/vq2_hybrid_fastprefix_a2suffix_g11_v2.npz`
- `data/vq2_straight_speed_candidate_v2.json`
- `data/vq2_straight_speed_candidate_v2_audit.json`
- `data/vq2_straight_speed_candidate_v2_live_record.json`

The live session and full camera/MAVLink archive are preserved at:

- `D:\ai-gp\training\vq2_full17_fastprefix_abba_v1\20260802_191537`
- `D:\ai-gp\raw_sessions\vq2_20260802_191537`

## What changed

1. The demonstrated prefix through gate 10 remains byte-identical to the
   protected reference.
2. Gates 11-16 use the cap-12 optimized suffix, joined by a 30-row
   cubic-Hermite bridge.
3. Gates 11-13 use 1.05x desired reference velocity.
4. Gate 15 uses trajectory blend 0.25, lateral gain 2.5, and -0.05 lateral
   trim.  The live crossing landed at +0.107 m lateral and -0.108 m vertical.
5. The live overspeed guard is 12.5 m/s.  The old 12.0 m/s guard falsely
   terminated otherwise healthy gate-3 runs at roughly 12.4 m/s.

## Final matched audit

The final audit used the same 12.5 m/s speed cap as live deployment and 256
fresh worlds across the v25/v28/v30 learned dynamics ensembles:

- finish rate: 26.95%
- median finish: 35.10 s
- p90 finish: 35.68 s
- gate-15 failures: zero

## Reproduction

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .remote\launch_live_full17_fastprefix_abba.ps1 `
  -CandidateConfig C:\Users\henry\Desktop\ai-gp\data\vq2_straight_speed_candidate_v2.json `
  -ChampionConfig C:\Users\henry\Desktop\ai-gp\data\vq2_straight_speed_candidate_v2.json `
  -Cycles 2
```

Full recording must remain enabled.  Do not restart the simulator process from
the harness; use drone resets between attempts and manually relaunch the
simulator only when session health degrades.
