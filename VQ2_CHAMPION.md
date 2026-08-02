# Protected VQ2 champion

The current protected champion is the fully recorded multigate hybrid stack
that completed an accepted official VQ2 lap in **36.826946258 seconds** on
2026-08-02.

- Protected artifact: `D:\ai-gp\champions\vq2_36s_20260802`
- Original training run: `D:\ai-gp\training\vq2_full17_fastprefix_abba_v1\20260802_125354`
- Original raw recording: `D:\ai-gp\raw_sessions\vq2_20260802_125354`
- Fastest finish: `episode_0002.npz`
- Harness duration: 38.511688232 seconds
- Finish episode SHA-256: `092103751D8C83FE5A883E6EBC636A1C7BE09D05F30C96965D5E536E7550068A`
- Full artifact manifest SHA-256: `40232920AF6B8273998C6DA56206C6DE7B723FC87E9A05582BC79AAC1C8D6BAE`
- Runtime map: `data/vq2_runtime_map_g9g15fix.json`
- Vision: v7 primary, v13 on gates 3-5, crop tracker, 10 Hz GPU dense path
- Localization: multigate
- Control: reference controller plus primary and late-gate PPO residual actors

The protected directory contains the complete eight-flight training session,
all raw camera/MAVLink recordings, the exact launcher, and copies of every
runtime asset. Do not train into, overwrite, or mutate it.

The +0.10 m gate-5 / -0.10 m gate-6 diagnostic produced accepted 36.963943481
and 36.888019561 second finishes, but it is not yet promoted. It completed 2
of 7 timing-healthy diagnostic flights; the evidence is useful but below the
required regression sample.

Previous champions remain preserved at:

- `D:\ai-gp\champions\vq2_39s_20260801`
- `D:\ai-gp\champions\vq2_40s_20260731`

## Promotion rule

A challenger does not replace this champion because of one fast lap. It must:

1. complete at least one deterministic official lap;
2. pass a timing-healthy interleaved champion/challenger calibration set;
3. complete a larger locked deterministic regression set;
4. avoid new gate-specific failure concentrations;
5. improve official time or completion reliability repeatedly;
6. retain its config, hashes, episode files, and full raw recording.

## Expert corpus

The offline expert dataset at
`D:\ai-gp\expert_datasets\vq2_13finish_demo_v1.npz` contains the fast local
lap followed by 12 independently verified official Gipsydanger finishes. The
slower laps are support data and do not replace the protected runtime line.
