# VQ2 co-visibility suffix deployable audit

## Decision

The geometry-only and geometry-plus-timing suffixes are **rejected for live
ABBA**.  They improved completion under the line optimizer's geometric
tracker, but that result does not transfer to the frozen live-teacher
controller used by the 35.374649-second champion.

No simulator flight should be spent on either candidate in its current form.

## Conversion verification

`scripts/build_vq2_suffix_reference_demo.py` converts the saved suffix
reference arrays into the SAC demo schema consumed by the existing 30-row
Hermite splicer and live teacher.

- All prefix rows through target gate 10 are byte-identical to the champion.
- Geometry bridge peak acceleration: 3.935 m/s^2.
- Timing bridge peak acceleration: 3.905 m/s^2.
- Reconstructed feed-forward actions match actions rebuilt directly from the
  original line parameters to less than 1.2e-6 maximum absolute error for
  both candidates.

The rejection is therefore not a conversion or feed-forward reconstruction
artifact.

## Matched smoke audit

All arms used the same 64 worlds and seed (`20260829`), the frozen late
residual actor, 10 Hz multigate vision simulation, and the pooled
v25/v28/v30 learned dynamics ensembles.

| Arm | Finishes | Finish rate | Median finish | Main late failure |
| --- | ---: | ---: | ---: | --- |
| Frozen 35.37 champion | 17/64 | 26.56% | 35.23 s | mixed |
| Co-vis geometry | 0/64 | 0.00% | n/a | gate 15, 1.688 m median crossing error |
| Co-vis timing | 3/64 | 4.69% | 37.90 s | gate 15, 1.153 m median crossing error |

The geometry arm produced 12 gate-15 failures among the 12 worlds reaching
that section.  The timing arm was slower than the champion and retained a
large gate-15 transfer error.

Audit artifacts are under
`worldmodel/covis_suffix_deployable_audit_v1/` and are intentionally not
promotion artifacts.

## Root cause and next iteration

The suffix CEM search evaluated candidates with `BatchedFlatRefController`.
The deployed composition tracks demo rows through the campaign's live-teacher
controller, residual routing, cursor rules, and per-gate feedback.  The
candidate references exploit behavior specific to the former tracker; their
reported reliability does not survive the latter controller.

The next suffix search must put the exact live-teacher controller inside its
candidate evaluation loop, including the frozen residual routing and gate-15
repair.  Co-visibility remains a useful secondary objective, but promotion
must be selected on deployable-controller outcomes rather than geometric
tracker outcomes.
