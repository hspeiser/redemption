# Protected VQ2 champion

The current protected champion is the v7/multigate reference controller that
completed deterministic VQ2 laps in 39.264468 s and 39.427090 s on 2026-08-01.

- Protected artifact: `D:\ai-gp\champions\vq2_39s_20260801`
- Original run: `D:\ai-gp\training\vq2_fullmap_livechamp_v7_multigate\20260801_212358`
- Original raw recording: `D:\ai-gp\raw_sessions\vq2_20260801_212358`
- Fastest finish: `episode_0001.npz` (17 gates, 39.264468 seconds)
- Second finish: `episode_0003.npz` (17 gates, 39.427090 seconds)
- Fastest episode SHA-256:
  `0C9E28C89F85BBA2DFA2EF4ADF9C909031E840348D6A5D98A9CC66A8189C3646`
- Full artifact manifest SHA-256:
  `AB952DAE4FDBA43FC35DA0390EA52AB6744F84D4C3EB84F7E6BE535EBE7446AD`
- Runtime map: `data/vq2_runtime_map_g9g15fix.json`
- Primary detector: `data/models/gatenet_v7_best.pt`
- Localization: multigate plus crop tracker, with 10 Hz GPU dense vision

The protected directory contains the complete training session, all raw camera
and MAVLink recordings, and copies of every runtime asset. Its manifest covers
9,564 files and 588,384,645 bytes. Do not train into or overwrite it.

The previous 39.9897-second champion remains preserved at
`D:\ai-gp\champions\vq2_40s_20260731` for regression history.

## Promotion rule

A candidate does not replace the champion merely because it reaches a later
gate once. It must first preserve the champion configuration except for the
single change under test, then:

1. complete at least one deterministic full lap;
2. complete a five-lap deterministic regression set without timing faults;
3. avoid regressions at gates 1, 5, 9, 10, 11, 15, and 16;
4. beat 39.264 seconds, or materially improve completion rate at comparable
   speed;
5. retain its checkpoint, config, episode NPZ files, and full raw recording.

## Expert corpus

The offline expert dataset at
`D:\ai-gp\expert_datasets\vq2_13finish_demo_v1.npz` contains the fast local
lap followed by 12 independently verified official finishes from GipsyDanger.
It has 24,950 transitions and 13 terminal completed trajectories. The slower
GipsyDanger laps are support data; they do not replace the fast lap as the
runtime reference trajectory.
