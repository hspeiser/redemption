"""Ground-truth reward: exact distance progress + gate-pass bonus + crash/stray terminals.

No detector, no EMA, no clip, no lost-gate — the odometry distance is exact and smooth, so progress
is just (prev_dist - curr_dist) in meters. A small step penalty encourages speed.
"""
from __future__ import annotations


def step_reward(prev_dist, curr_dist, passed, collided, strayed, cfg):
    r = cfg.w_prog * (prev_dist - curr_dist) - cfg.step_penalty
    done, reason = False, ""
    if passed:
        r += cfg.gate_bonus
    if collided:
        r += cfg.fail_reward(curr_dist)     # slight + at the gate, exponentially worse far away
        done, reason = True, "crash"
    elif strayed:
        r += cfg.fail_reward(curr_dist)
        done, reason = True, "stray"
    return r, done, reason
