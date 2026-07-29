"""MAVLink link for the NEW sim (VQ1): body-rate control + arm/reset, and an rx thread that
captures full ground-truth state (ODOMETRY: pos, quat, vel, body rates), RACE_STATUS
(active_gate_index), COLLISION, and the TRACK_INFO gate layout (sent after a SIM_RESET).

Ground truth replaces the old detector/EKF entirely — no vision needed for state-based training.
"""
from __future__ import annotations
import struct, threading, time
from collections import deque, Counter
import numpy as np
from pymavlink import mavutil

_ATT_MASK = mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
_RESET_CMD = 31000
_RACE_STATUS_ID = 1
_TRACK_INFO_ID = 2


def _gates_sane(gates) -> bool:
    """Reject corrupted decodes: gate frames are ~2.7 m and live within ~300 m of the origin."""
    for g in gates:
        if not np.all(np.isfinite(g["pos"])) or np.linalg.norm(g["pos"]) > 300.0:
            return False
        if not (0.3 <= g["w"] <= 15.0 and 0.3 <= g["h"] <= 15.0):
            return False
    return True


class MavLink:
    def __init__(self, addr="udpin:0.0.0.0:14550"):
        self.c = mavutil.mavlink_connection(addr)
        self.hb()
        self.c.wait_heartbeat(timeout=10)
        self.sys = self.c.target_system
        self.comp = self.c.target_component
        # latest ground-truth state
        self.pos = np.zeros(3)          # NED position
        self.quat = np.array([1.0, 0, 0, 0])   # w,x,y,z (body->world)
        self.vel = np.zeros(3)          # NED velocity
        self.rates = np.zeros(3)        # body roll/pitch/yaw rate
        self.t_us = 0
        self.reset_count = 0
        self.active_gate = 0
        self.collision_epoch = 0
        self.gates = None               # list of dicts: id, pos(3), quat(4), w, h
        self._gate_decodes = deque(maxlen=24)   # recent valid decodes (for consensus capture)
        self._track_chunks = {}
        self._expected = {}
        self.run = True
        threading.Thread(target=self._rx, daemon=True).start()

    def hb(self):
        self.c.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                  mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def _rx(self):
        while self.run:
            m = self.c.recv_match(blocking=False)
            if m is None:
                time.sleep(0.0005); continue
            t = m.get_type()
            if t == "ODOMETRY":
                self.pos = np.array([m.x, m.y, m.z], np.float64)
                self.quat = np.array([m.q[0], m.q[1], m.q[2], m.q[3]], np.float64)
                self.vel = np.array([m.vx, m.vy, m.vz], np.float64)
                # ODOMETRY pitchspeed is SIGN-FLIPPED vs the true body rotation (verified against
                # quaternion deltas, probe_frames 2026-07-28; roll/yaw report correctly). Store
                # proper right-handed FRD rates so obs + rate-loop feedback see the real motion.
                self.rates = np.array([m.rollspeed, -m.pitchspeed, m.yawspeed], np.float64)
                self.t_us = m.time_usec
                self.reset_count = m.reset_counter
            elif t == "COLLISION":
                self.collision_epoch += 1
            elif t == "ENCAPSULATED_DATA":
                raw = bytes(m.data)
                if not raw:
                    continue
                if raw[0] == _RACE_STATUS_ID:
                    try:
                        _, _, _, _, agi, _ = struct.unpack_from("<BQqqIq", raw)
                        self.active_gate = int(agi)
                    except struct.error:
                        pass
                elif raw[0] == _TRACK_INFO_ID:
                    self._track_chunk(m, raw)
            elif t == "DATA_TRANSMISSION_HANDSHAKE":
                self._track_chunks[m.width] = {}
                self._expected[m.width] = m.packets

    def _track_chunk(self, m, raw):
        try:
            _, tid = struct.unpack_from("<BH", raw)
        except struct.error:
            return
        if tid not in self._expected:
            return
        self._track_chunks[tid][m.seqnr] = raw[3:]
        if len(self._track_chunks[tid]) == self._expected[tid]:
            full = b"".join(self._track_chunks[tid][i] for i in range(self._expected[tid]))
            del self._track_chunks[tid]; del self._expected[tid]
            self._decode_gates(full)

    def _decode_gates(self, payload):
        try:
            ng, = struct.unpack_from("<H", payload)
            if not (1 <= ng <= 100):
                return
            payload = payload[2:]
            gates = []
            for _ in range(ng):
                v = struct.unpack_from("<Hfffffffff", payload)
                gates.append({"id": v[0], "pos": np.array(v[1:4], np.float64),
                              "quat": np.array(v[4:8], np.float64), "w": v[8], "h": v[9]})
                payload = payload[38:]
            if _gates_sane(gates):          # reject corrupted / partial decodes
                self.gates = gates
                self._gate_decodes.append(gates)
        except struct.error:
            pass

    def att(self, roll_rate, pitch_rate, yaw_rate, thrust):
        self.c.mav.set_attitude_target_send(
            0, self.sys, self.comp, _ATT_MASK, [1, 0, 0, 0],
            float(roll_rate), float(pitch_rate), float(yaw_rate), float(thrust))

    def arm(self, on=True):
        self.c.mav.command_long_send(self.sys, self.comp,
                                     mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                     0, 1 if on else 0, 0, 0, 0, 0, 0, 0)

    def reset(self):
        self.c.mav.command_long_send(self.sys, self.comp, _RESET_CMD, 0, 0, 0, 0, 0, 0, 0, 0)

    def capture_gates(self, tries=7, timeout=2.5) -> bool:
        """Reset triggers the sim to broadcast the track layout (our reassembly yields ~1 decode per
        reset). Do several resets and take the CONSENSUS gate-0 position, so a rare garbage decode
        is outvoted. One-time cost at startup."""
        decodes = []
        for _ in range(tries):
            self._track_chunks.clear(); self._expected.clear(); self.gates = None
            self.reset()
            t0 = time.time()
            while self.gates is None and time.time() - t0 < timeout:
                self.hb(); time.sleep(0.03)
            if self.gates is not None:
                decodes.append(self.gates)
        if not decodes:
            return False
        sig = lambda gs: tuple(np.round(gs[0]["pos"], 0))   # gate-0 position (stable rounding)
        counts = Counter(sig(gs) for gs in decodes)
        best_sig, n = counts.most_common(1)[0]
        for gs in decodes:                     # representative decode of the winning layout
            if sig(gs) == best_sig:
                self.gates = gs
                return n >= 2                   # winning gate-0 must appear in >= 2 resets
        return False

    def stop(self):
        self.run = False
