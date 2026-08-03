# VQ2 exact-controller suffix speed search

Status: balanced v2 candidate certified on two large final pools and one fresh
paired 768-world audit; first live ABBA campaign was prefix-limited and
produced no candidate suffix exposure.

The frozen live record is 35.374649 s. Its gate-10-to-finish suffix is
approximately 15.1 s. Every result below uses the deployment-parity
`BatchedLiveTeacher`, the v25/v28/v30 residual ensemble, the 13-state empirical
gate-10 handoff pool, multigate-10-Hz audit noise, and paired random worlds.

## Selection discipline

- Gate 13 and gate 14 velocity scales are frozen at the record values because
  the reversal is perception-bound.
- Searchable velocity scales are limited to target gates 11, 12, 15, and 16.
- Geometry is frozen to the exact-controller reliability candidate.
- Speed mode is lexicographic: candidates below the reliability floor cannot
  beat feasible candidates; time is the objective within the feasible set.
- CEM generations rotate world seeds.
- Multiple unique finalists are reranked on one untouched common-random-number
  pool. The full feasible frontier is retained in `speed_report.json`.

## Rejected unrestricted result

The first speed candidate changed all suffix scales. On a fresh paired
512-world audit it saved 0.467 s but reduced completion from 68.55% to 64.45%
(-4.10 points; 95% CI -8.20 to 0). New failures appeared at gates 13-15, so it
was rejected without a live flight.

## Straight-only winner

Artifact: `worldmodel/suffix_exact_straights_v1/speed_config.json`

Velocity scales for gates 11-16:

```text
g11 0.999634797
g12 1.084754587
g13 1.050000000  (frozen)
g14 1.000000000  (frozen)
g15 1.100648720
g16 1.154693402
```

Fresh paired audit, seed 20261303, 512 worlds:

| Metric | Frozen record config | Candidate | Paired change |
|---|---:|---:|---:|
| Completion | 68.75% | 75.98% | +7.23 points |
| Completion delta 95% CI | | | +3.13 to +11.52 points |
| Median suffix time | 15.133 s | 14.800 s | -0.367 s |
| Time delta 95% CI | | | -0.400 to -0.367 s |

Finished-lap segment medians:

| Segment | Baseline | Candidate | Change |
|---|---:|---:|---:|
| g10 to g11 | 2.733 s | 2.867 s | +0.134 s |
| g11 to g12 | 2.500 s | 2.433 s | -0.067 s |
| g12 to g13 | 2.700 s | 2.733 s | +0.033 s |
| g13 to g14 | 2.067 s | 2.067 s | 0.000 s |
| g14 to g15 | 2.400 s | 2.267 s | -0.133 s |
| g15 to g16 | 2.800 s | 2.467 s | -0.333 s |

The candidate deliberately gives up 0.134 s before gate 11 and halves gate-11
failures (130 to 64). It then earns the time back primarily at gates 15-16.
The perception-bound g13-to-g14 reversal is unchanged.

SHA-256:

```text
speed_config.json
970672718BD1E60ECE6A7F0CD28779B5BA36D97FD9616A615C86D11FEFE62CBC

speed_best.npz
8D0D227608D6551AD505949ADB20FCAF972E95F836D95F88C3AC491DFEB68573

paired_baseline_512.npz
724D34904C49423B23CE6A44A603050BF2672ADD88A8015094B1CAAAB8754BDA

paired_candidate_512.npz
C849A521BFABCD5AC21D352BE6B96915D45447DBCF0B347F9FD75C75AA62CD60
```

## Next gates

1. Stop all offline compute, manually relaunch the simulator, and run the v2
   balanced candidate in protected ABBA:
   record champion, candidate, candidate, record champion.
2. Promote only with official timing, timing-health pass, no safety regression,
   and hash-frozen config.
3. If more reference time is needed, retime the existing suffix path directly;
   do not rebuild its spline from only the suffix gates.

## Straight-scale plus action-lead frontier

Gipsy ran the exact deployed controller with four concurrent evaluators. The
search changed only velocity scales and integer action leads at gates 11, 12,
15, and 16. Gates 13 and 14 and all suffix geometry remained frozen. Its eight
finalists were reranked together on 768 worlds; then the record baseline, the
prior straight-only winner, the fastest feasible point, and the reliability
knee were audited on a different untouched 768-world seed (20261306).

Fresh paired results against the frozen record suffix:

| Arm | Completion | Change | Median suffix | Change |
|---|---:|---:|---:|---:|
| Frozen record | 65.23% | - | 15.167 s | - |
| Prior straight-only | 78.26% | +13.02 points | 14.833 s | -0.367 s |
| Fast frontier | 70.31% | +5.08 points | 14.633 s | -0.533 s |
| Reliability knee | 82.03% | +16.80 points | 14.833 s | -0.333 s |

Paired bootstrap intervals exclude zero for every completion and time
improvement. The fast arm's completion delta is +2.34 to +7.68 points (95% CI)
and its time delta is -0.533 to -0.517 s. The reliability arm's completion
delta is +13.67 to +20.05 points and its time delta is exactly -0.333 s at the
30 Hz timing resolution.

The fast arm's suffix settings are:

```text
velocity: g11 1.07720512, g12 1.02973040, g13 1.05, g14 1.0,
          g15 1.04317720, g16 1.24178291
leads:    g11 0, g12 1, g13 0, g14 0, g15 0, g16 0
```

The reliability knee's suffix settings are:

```text
velocity: g11 1.01462406, g12 1.03124815, g13 1.05, g14 1.0,
          g15 1.06227591, g16 1.19829365
leads:    g11 2, g12 1, g13 0, g14 0, g15 0, g16 2
```

Directly, the fast arm is 0.200 s faster than the reliability knee but gives
up 11.72 completion points. Both merit a live slot: reliability first to
increase suffix exposure, then fast for the record attempt once session health
is proven. The protected 35.374649 s champion remains frozen.

## Balanced v2 winner

A larger second search started from the reliability knee, enforced an 80%
search-world floor, evaluated 160 CEM candidates, and reranked 12 finalists on
1,536 untouched worlds. It selected:

```text
velocity: g11 1.02562627, g12 1.05550588, g13 1.05, g14 1.0,
          g15 1.07718446, g16 1.25211584
leads:    g11 1, g12 3, g13 0, g14 0, g15 0, g16 2
```

Its final-search result was 82.03% completion at 14.700 s. A separate harsh
paired audit (seed 20261321, 768 worlds, start noise 0.4 m/0.8 m/s) produced:

| Arm | Completion | Change vs record | Median suffix | Paired time change |
|---|---:|---:|---:|---:|
| Frozen record | 66.15% | - | 15.167 s | - |
| v1 fast | 68.62% | +2.47 points | 14.633 s | -0.533 s |
| v1 reliability | 78.91% | +12.76 points | 14.867 s | -0.333 s |
| **v2 balanced** | **75.91%** | **+9.77 points** | **14.667 s** | **-0.500 s** |

The v2 completion improvement has a paired 95% CI of +6.64 to +12.89
percentage points. Its paired time improvement is -0.533 to -0.500 s. On the
same worlds, v2 matches v1-fast's paired finish time while improving completion
by 7.29 points (CI +4.69 to +9.90). Compared with v1-reliability, it buys
0.167 s for a 2.99-point completion reduction. This is the primary live ABBA
candidate; v1-reliability remains the fallback.

Finished-lap v2 segment medians show the reversal stayed unchanged:

```text
g10->11 2.833 s   g11->12 2.500 s   g12->13 2.733 s
g13->14 2.067 s   g14->15 2.300 s   g15->16 2.300 s
```

## Geometry-preserving retime status

The deployment-parity line-space machinery now supports restricted speed
segments, parallel candidate audits, fixed profile accelerations, and exporting
the winning demo. Its first smoke test deliberately ran before a full search
and rejected the naive construction: rebuilding a spline from only gates
11-16 changed the path at the splice, achieved only 9.38% completion, and
failed mainly at gate 15. No full search was run. A valid follow-up must resample
the existing saved suffix path itself so geometry is identical, then recompute
velocity, attitude, rates, and thrust for the new timeline.

## First live ABBA campaign

The candidate was tested in protected A-B-B-A order with full recordings and
automatic timing-health aborts. Across the first 23 attempted flights, only one
run reached the suffix: a protected-champion arm reached gate 12 and crashed.
No candidate arm reached gate 11, so there is no valid live suffix-time
comparison yet. Candidate and champion use identical controls before gate 11;
the early gate-2/4/5 failures therefore do not identify a candidate regression.

The final block correctly aborted after two consecutive unhealthy episodes:
sim-step p95 rose from approximately 42 ms to 72 ms and then 117 ms, with a
365 ms maximum. Those episodes are quarantined. The simulator must be manually
relaunched before the next live ABBA block; the harness does not restart it.
