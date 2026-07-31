# Handoff: vision + map upgrades for the SAC/controller campaign

Two adoptions, both validated offline and in PPO flights. Zero code
changes to the trainer required.

## 1. Corrected map (gates 9 and 15)

    --map C:\Users\henry\Desktop\ai-gp\data\vq2_runtime_map_g9g15fix.json

and set the gate-9 right-lateral bias to 0 (the +0.50 right steering
bias was empirically compensating a 0.46m map survey error at gate 9 —
your measured bias matches the audit's delta to 2cm).

Evidence: offset-vector audit over 11 debug archives; gate 9 delta
0.46-0.51m at consistency 0.95-0.98 across all sessions, gate 15 0.21m
replicated to 1cm across independent session groups; all other gates
<=0.07m. Hold-out validation collapses gate-9 innovations from +22.8px
to -2.4px. Expected effect: gate-9 dense fusion 38% -> ~90%+, chicane
landmark-age spikes eliminated, cleaner gate-10 entry state.

## 2. v13 primary detector (close-range recall)

    --primary C:\Users\henry\Desktop\ai-gp\data\models\gatenet_v13drought_best.pt

v7 + 17.6k belief-projected labels + 19.1k pure-projection drought
labels (the occlusion/close views the original labeling loop could not
produce because it required the detector to already almost-see).

Evidence: on CLEAN held-out sessions (zero training overlap), blind
frames recovered: v7 44% -> v13 61%; held-out corner precision equal
or better (0.74px median vs 0.76). Gipsy copy:
/home/henry/aigp/data/models/gatenet_v13drought_best.pt.

## Cautions

- Direct position pins: fine for your reference-following controller
  (keep them on); they destabilize end-to-end policies only.
- The jet + pillar beside the gate-1 approach have been there since at
  least July 30 (frame-matched); they are not new scenery.
- If you ever eval your own teacher standalone: argparse defaults
  diverge from campaign configs in 49 keys (teacher_blend, residual
  scale/gates, all biases). Replicate the session config exactly.
