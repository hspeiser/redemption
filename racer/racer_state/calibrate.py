"""Measure VQ1's actual command->response scales so twin-policy actions map to VQ1 commands in
PHYSICAL units (rad/s, thrust-to-weight) instead of guessed constants.

The twin and VQ1 only ever agreed on the hover POINT (and even that moved: 0.28 -> 0.19). The
twin's thrust model is F = norm * mg/0.28, i.e. a known accel-per-cmd SLOPE; VQ1's slope was never
measured, so a3 bought a different acceleration in each sim and the transferred policy had to
re-learn its own authority. Same for rates: the +/-1.2 cmd clip silently capped VQ1 at ~3 rad/s
vs the twin's 4.

Probes (a few minutes, sim must be running):
  1. THRUST: hold level, step the thrust cmd across a ladder, measure the initial vertical
     acceleration -> fit specific_force(cmd) = a + b*cmd. Gives true hover cmd AND the slope.
  2. RATES: per axis, pulse the cmd at several amplitudes, measure the steady body rate ->
     per-axis signed gain, linearity range, and the achievable rate ceiling.

Writes runs/calib.json, consumed by train2's action adapter.

  python -m racer_state.calibrate
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from racer_state.config import CFG
from racer_state.mav_io import MavLink
from racer_state.geom import rot_world_to_body, rot_body_to_world
from racer_state.train import wait_reset

G = 9.81
_DOWN = np.array([0.0, 0.0, 1.0])          # NED down
PRIOR_GAIN = np.array([-2.48, 2.52, -2.31])    # true cmd->rate gains (pitch is NOT inverted;
                                               # the odometry report was — see probe_frames)
KC = 1.5
H0 = CFG.thrust_hover                      # working hover estimate for stabilization


def vz_world(mav):
    """NED world vertical velocity (down +). ODOMETRY vel is body-frame (probed)."""
    return float(rot_body_to_world(mav.quat, np.asarray(mav.vel, np.float64))[2])


def grav_body(mav):
    return rot_world_to_body(mav.quat, _DOWN)      # level FRD -> ~[0,0,1]


def hold(mav, hb, des_frd, thrust, dur, rec=None):
    """Stream for dur s; des_frd (rad/s) tracked via the prior-gain inner loop."""
    t0 = time.time()
    while time.time() - t0 < dur:
        err = des_frd - np.asarray(mav.rates, np.float64)
        cmd = np.clip((des_frd + KC * err) / PRIOR_GAIN, -1.2, 1.2)
        mav.att(float(cmd[0]), float(cmd[1]), float(cmd[2]), float(thrust))
        if time.time() - hb[0] > 0.3:
            mav.hb(); hb[0] = time.time()
        if rec is not None:
            g = grav_body(mav)
            rec.append((time.time() - t0, vz_world(mav), float(g[2])))
        time.sleep(0.004)


def hold_raw(mav, hb, cmd3, thrust, dur, rec=None):
    """Open-loop rate cmds (we are measuring the plant, no inner loop)."""
    t0 = time.time()
    while time.time() - t0 < dur:
        mav.att(float(cmd3[0]), float(cmd3[1]), float(cmd3[2]), float(thrust))
        if time.time() - hb[0] > 0.3:
            mav.hb(); hb[0] = time.time()
        if rec is not None:
            rec.append((time.time() - t0, np.asarray(mav.rates, np.float64).copy()))
        time.sleep(0.004)


def leveler_des(mav):
    # FRD: rolled-right -> g1>0, correct with negative roll rate; nose-up -> g0<0, correct with
    # negative (nose-down) pitch rate => des_pitch = +3*g0. (The old -3*g0 was tuned to the
    # pre-fix loop that tracked the NEGATIVE of desired pitch — two sign errors cancelling.)
    g = grav_body(mav)
    return np.array([-3.0 * float(g[1]), 3.0 * float(g[0]), 0.0])


def settle(mav, hb, z_ref, dur_max=3.0, vz_tol=0.35):
    """Level + arrest vertical AND horizontal speed + crude altitude pull toward z_ref (NED).
    Horizontal braking tilts gently into the velocity (target gravity component ~ -k*v_body);
    without it drift accumulates across probe rungs and corrupts the accel fits."""
    t0 = time.time()
    while time.time() - t0 < dur_max:
        vz = vz_world(mav)
        vb = np.asarray(mav.vel, np.float64)                 # body FRD
        g0_t = float(np.clip(-0.05 * vb[0], -0.25, 0.25))    # nose-up brakes +x speed
        g1_t = float(np.clip(-0.05 * vb[1], -0.25, 0.25))    # bank-left brakes +y speed
        g = grav_body(mav)
        des = np.array([-3.0 * (float(g[1]) - g1_t), 3.0 * (float(g[0]) - g0_t), 0.0])
        thr = H0 + np.clip(0.10 * vz, -0.10, 0.10) \
                 + np.clip(0.03 * (float(mav.pos[2]) - z_ref), -0.05, 0.05)
        hold(mav, hb, des, float(np.clip(thr, 0.02, 0.9)), 0.03)
        g = grav_body(mav)
        vh = float(np.hypot(vb[0], vb[1]))
        if abs(vz) < vz_tol and vh < 1.0 and abs(float(g[0])) < 0.1 and abs(float(g[1])) < 0.1:
            return True
    return False


def fit_accel(rec):
    """Least-squares dvz/dt from (t, vz, cos_tilt) samples, skipping the initial transient and
    high-speed (drag-contaminated) samples. Returns (az_ned, mean_cos_tilt, n)."""
    arr = np.array([(t, v, c) for t, v, c in rec if t > 0.06 and abs(v) < 4.5])
    if len(arr) < 8:
        return None, None, 0
    t, v, c = arr[:, 0], arr[:, 1], arr[:, 2]
    A = np.stack([t, np.ones_like(t)], 1)
    az = float(np.linalg.lstsq(A, v, rcond=None)[0][0])
    return az, float(np.mean(c)), len(arr)


def main():
    mav = MavLink(CFG.mav_addr)
    print(f"connected sys={mav.sys}", flush=True)
    wait_reset(mav); time.sleep(CFG.reset_settle_s)
    t0 = time.time()
    while mav.t_us == 0 and time.time() - t0 < 3:
        mav.hb(); time.sleep(0.03)
    hb = [0.0]
    mav.arm(True)

    z0 = float(mav.pos[2])
    print(f"spawn z={z0:.2f}; climbing to probe altitude", flush=True)
    t0 = time.time()
    while time.time() - t0 < 4.0:
        hold(mav, hb, leveler_des(mav), H0 + 0.07, 0.03)
        if z0 - float(mav.pos[2]) > 4.0:
            break
    z_ref = z0 - 4.0
    settle(mav, hb, z_ref)

    # ---------------- thrust ladder ----------------
    ladder = [0.19, 0.26, 0.13, 0.33, 0.09, 0.42, 0.55, 0.70, 0.04]
    tpoints = []
    for cmd in ladder:
        ok = False
        for _ in range(3):
            if settle(mav, hb, z_ref):
                ok = True; break
        if not ok:
            print(f"thrust {cmd:.2f}: never settled, rung skipped", flush=True)
            continue
        rec = []
        hold(mav, hb, leveler_des(mav), cmd, 0.35, rec=rec)
        az, cos_t, n = fit_accel(rec)
        if az is None or n < 25:
            print(f"thrust {cmd:.2f}: too few clean samples ({n}), skipped", flush=True)
            continue
        spec = (G - az) / max(0.85, cos_t)         # specific force along body z, m/s^2
        tpoints.append((cmd, spec))
        print(f"thrust {cmd:.2f}: az={az:+6.2f} m/s^2 cos_tilt={cos_t:.3f} "
              f"-> spec={spec:6.2f} m/s^2 ({spec / G:.2f} g)  [{n} samples]", flush=True)

    if len(tpoints) < 4:
        print(f"THRUST: only {len(tpoints)} clean rungs — aborting (fix stability first)",
              flush=True)
        mav.arm(False); mav.stop(); return
    arr = np.array(tpoints)
    A = np.stack([arr[:, 0], np.ones(len(arr))], 1)
    (b, a), res, *_ = np.linalg.lstsq(A, arr[:, 1], rcond=None)
    pred = A @ [b, a]
    ss_res = float(np.sum((arr[:, 1] - pred) ** 2))
    ss_tot = float(np.sum((arr[:, 1] - arr[:, 1].mean()) ** 2))
    r2 = 1.0 - ss_res / max(1e-9, ss_tot)
    hover_cmd = (G - a) / b
    print(f"THRUST FIT: spec = {a:+.2f} + {b:.2f}*cmd  (r2={r2:.4f})  hover_cmd={hover_cmd:.3f}",
          flush=True)
    print(f"  twin-equivalent check: a3=+1 wants {2.25 * G:.1f} m/s^2 -> cmd "
          f"{(2.25 * G - a) / b:.3f}; a3=0 wants g -> cmd {hover_cmd:.3f}", flush=True)

    # ---------------- rate pulses ----------------
    names = ["roll", "pitch", "yaw"]
    amps = [0.4, 0.9, 1.4, 1.9]
    rpoints = {0: [], 1: [], 2: []}
    for axis in range(3):
        for amp in amps + [-0.9]:                  # one negative pulse to verify symmetry
            settle(mav, hb, z_ref)
            cmd3 = [0.0, 0.0, 0.0]; cmd3[axis] = amp
            rec = []
            hold_raw(mav, hb, cmd3, hover_cmd, 0.30, rec=rec)
            cmd3[axis] = -0.5 * amp
            hold_raw(mav, hb, cmd3, hover_cmd, 0.15)          # counter-pulse
            hold(mav, hb, leveler_des(mav), hover_cmd, 0.5)
            steady = np.array([r[axis] for t, r in rec if t > 0.15])
            if len(steady) < 5:
                print(f"{names[axis]} amp {amp:+.1f}: too few samples", flush=True)
                continue
            rate = float(np.mean(steady))
            rpoints[axis].append((amp, rate))
            print(f"{names[axis]:5s} cmd {amp:+.1f} -> rate {rate:+6.2f} rad/s "
                  f"(gain {rate / amp:+.2f})", flush=True)

    gains, rate_max, cmd_max = [], [], []
    for axis in range(3):
        pts = rpoints[axis]
        small = [r / c for c, r in pts if abs(c) <= 0.95]
        gain = float(np.mean(small))
        rmax = float(max(abs(r) for c, r in pts))
        cmax = 1.9
        for c, r in sorted(pts, key=lambda p: abs(p[0])):
            if c > 0 and abs(r - gain * c) > 0.30 * abs(gain * c):
                cmax = abs(c); break
        gains.append(gain); rate_max.append(rmax); cmd_max.append(cmax)
        print(f"{names[axis]:5s}: gain {gain:+.2f} rad/s per cmd, ceiling {rmax:.2f} rad/s, "
              f"linear to |cmd|~{cmax:.1f}", flush=True)

    mav.arm(False)
    for _ in range(6):
        mav.att(0, 0, 0, 0); mav.hb(); time.sleep(0.02)

    calib = {"g": G,
             "thrust": {"intercept": float(a), "slope": float(b),
                        "hover_cmd": float(hover_cmd), "r2": float(r2),
                        "points": [[float(c), float(s)] for c, s in tpoints]},
             "rates": {"gain": gains, "rate_max": rate_max, "cmd_max": cmd_max,
                       "points": {names[k]: [[float(c), float(r)] for c, r in rpoints[k]]
                                  for k in range(3)}},
             "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = os.path.join(CFG.run_dir, "calib.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=1)
    print(f"wrote {out}", flush=True)
    for k in range(3):
        if rate_max[k] < 3.9:
            print(f"WARNING: {names[k]} ceiling {rate_max[k]:.2f} < twin's 4.0 rad/s — the twin "
                  f"policy expects authority VQ1 can't deliver on this axis (consider retraining "
                  f"the twin with RATE_SCALE={min(rate_max):.1f})", flush=True)
    mav.stop()


if __name__ == "__main__":
    main()
