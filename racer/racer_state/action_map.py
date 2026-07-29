"""Policy action a in (-1,1)^4 -> body-rate command (roll,pitch,yaw rate + thrust).

Pitch + roll: fixed scale (no cap) — aggressive rate use is shaped by the adaptive reward penalty.
Yaw: slow curriculum cap (heading rarely needs to swing).
Thrust: HOVER-CENTERED — a3=0 hovers, a3 modulates around hover by +/- span. Raw action is
stored/trained; the mapping lives only here."""
from __future__ import annotations
import numpy as np


def action_to_cmd(a, cfg, ep: int):
    a = np.clip(np.asarray(a, float), -1.0, 1.0)
    roll_rate = a[0] * cfg.roll_scale
    pitch_rate = a[1] * cfg.pitch_scale
    yaw_rate = a[2] * cfg.rollyaw_cap(ep)      # yaw keeps the slow cap
    thrust = float(np.clip(cfg.thrust_hover + a[3] * cfg.thrust_span(ep), 0.0, 1.0))
    return float(roll_rate), float(pitch_rate), float(yaw_rate), thrust
