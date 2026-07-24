"""MAVLink receive/transmit hub for the AI-GP simulator.

Buffers every telemetry message the sim sends (full rate, unbounded within
deque limits sized for hours of flight) and records every command we transmit,
each stamped with a wall-clock receive/send time so streams can be aligned
against the camera's sim_time_ns later.
"""

import struct
import threading
import time
from collections import deque

from pymavlink import mavutil

RACE_STATUS_ID = 1
TRACK_INFO_ID = 2
CMD_SIM_RESET = 31000

# sized for ~4+ hours at full telemetry rate — we keep everything
_BUF = 4_000_000


class MavIO:
    def __init__(self, ip="127.0.0.1", port=14550, timesync_hz=5.0):
        self.conn = mavutil.mavlink_connection(f"udpin:{ip}:{port}")
        print("Waiting for heartbeat...", flush=True)
        self.conn.wait_heartbeat()
        print(f"Connected to system {self.conn.target_system}", flush=True)

        self.lock = threading.Lock()
        self.send_lock = threading.Lock()

        # --- telemetry buffers (tuples end with wall_ns receive time) ---
        # (t_us, x,y,z, qw,qx,qy,qz, vx,vy,vz, wr,wp,wy, reset_counter, wall_ns)
        self.odom = deque(maxlen=_BUF)
        # (t_ms, roll, pitch, yaw, rollspeed, pitchspeed, yawspeed, wall_ns)
        self.attitude = deque(maxlen=_BUF)
        # (t_us, ax, ay, az, gx, gy, gz, wall_ns)
        self.imu = deque(maxlen=_BUF)
        # (t_ms, x, y, z, vx, vy, vz, wall_ns)
        self.local_pos = deque(maxlen=_BUF)
        # (t_us, m0, m1, m2, m3, wall_ns)
        self.actuator = deque(maxlen=_BUF)
        # (wall_ns, collision_id, threat_level, impulse)
        self.collisions = deque(maxlen=100_000)
        # (wall_ns, tc1, ts1)
        self.timesync_msgs = deque(maxlen=1_000_000)
        # (wall_ns, sim_boot_ms, race_start_ms, race_finish_ns, active_gate, last_gate_time)
        self.race_status_log = deque(maxlen=1_000_000)
        # (wall_ns, base_mode, system_status)
        self.heartbeats = deque(maxlen=100_000)
        # every command we send: (wall_ns, kind, p0..p3)
        self.sent_log = deque(maxlen=_BUF)

        self.race_status = None
        self.gate_map = None  # list of dicts once track data arrives
        self._track_chunks = {}
        self._track_expected = {}

        self.running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        self._ts_thread = threading.Thread(
            target=self._timesync_loop, args=(timesync_hz,), daemon=True
        )
        self._ts_thread.start()

    # ------------------------------------------------------------------ RX

    def _rx_loop(self):
        while self.running:
            try:
                msg = self.conn.recv_match(blocking=False)
            except ConnectionResetError:
                print("WARNING: MAVLink ConnectionResetError; rx stopped", flush=True)
                return
            if msg is None:
                time.sleep(0.0005)
                continue
            wall = time.time_ns()
            t = msg.get_type()
            if t == "ODOMETRY":
                # Quat convention fix: sim emits Y-flipped left-handed quats
                # (Unreal). Correct body->world reading = (w, -x, y, -z).
                # Verified against PnP-true camera rotation on banked frames.
                self.odom.append((
                    msg.time_usec, msg.x, msg.y, msg.z,
                    msg.q[0], -msg.q[1], msg.q[2], -msg.q[3],
                    msg.vx, msg.vy, msg.vz,
                    msg.rollspeed, msg.pitchspeed, msg.yawspeed,
                    msg.reset_counter, wall,
                ))
            elif t == "ATTITUDE":
                self.attitude.append((
                    msg.time_boot_ms, msg.roll, msg.pitch, msg.yaw,
                    msg.rollspeed, msg.pitchspeed, msg.yawspeed, wall,
                ))
            elif t == "HIGHRES_IMU":
                self.imu.append((
                    msg.time_usec, msg.xacc, msg.yacc, msg.zacc,
                    msg.xgyro, msg.ygyro, msg.zgyro, wall,
                ))
            elif t == "LOCAL_POSITION_NED":
                self.local_pos.append((
                    msg.time_boot_ms, msg.x, msg.y, msg.z,
                    msg.vx, msg.vy, msg.vz, wall,
                ))
            elif t == "ACTUATOR_OUTPUT_STATUS":
                self.actuator.append((
                    msg.time_usec, msg.actuator[0], msg.actuator[1],
                    msg.actuator[2], msg.actuator[3], wall,
                ))
            elif t == "COLLISION":
                self.collisions.append((
                    wall, msg.id, msg.threat_level, msg.horizontal_minimum_delta,
                ))
                print(f"COLLISION id={msg.id} threat={msg.threat_level} "
                      f"impulse={msg.horizontal_minimum_delta:.2f}", flush=True)
            elif t == "TIMESYNC":
                self.timesync_msgs.append((wall, msg.tc1, msg.ts1))
            elif t == "HEARTBEAT":
                self.heartbeats.append((wall, msg.base_mode, msg.system_status))
            elif t == "ENCAPSULATED_DATA":
                self._on_encapsulated(msg, wall)
            elif t == "DATA_TRANSMISSION_HANDSHAKE":
                self._track_expected[msg.width] = msg.packets
                self._track_chunks[msg.width] = {}

    def _on_encapsulated(self, msg, wall):
        raw = bytes(msg.data)
        if raw[0] == RACE_STATUS_ID:
            vals = struct.unpack_from("<BQqqIq", raw)
            self.race_status = {
                "sim_boot_ms": vals[1], "race_start_ms": vals[2],
                "race_finish_ns": vals[3], "active_gate": vals[4],
                "last_gate_time": vals[5], "wall_ns": wall,
            }
            self.race_status_log.append((wall, *vals[1:]))
        elif raw[0] == TRACK_INFO_ID:
            _, transfer_id = struct.unpack_from("<BH", raw)
            if transfer_id not in self._track_expected:
                return
            self._track_chunks[transfer_id][msg.seqnr] = raw[3:]
            if len(self._track_chunks[transfer_id]) == self._track_expected[transfer_id]:
                chunks = self._track_chunks.pop(transfer_id)
                del self._track_expected[transfer_id]
                payload = b"".join(chunks[i] for i in range(len(chunks)))
                self._decode_track(payload)

    def _decode_track(self, payload):
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = []
        for _ in range(num_gates):
            gid, x, y, z, qw, qx, qy, qz, w, h = struct.unpack_from("<Hfffffffff", payload)
            payload = payload[38:]
            gates.append({
                "gate_id": gid, "pos": [x, y, z],
                "quat_wxyz": [qw, qx, qy, qz], "width": w, "height": h,
            })
        gates.sort(key=lambda g: g["gate_id"])
        with self.lock:
            self.gate_map = gates
        print(f"Track data received: {num_gates} gates", flush=True)

    # ------------------------------------------------------------------ TX

    def _log_send(self, kind, *params):
        self.sent_log.append((time.time_ns(), kind, *(list(params) + [0.0] * (4 - len(params)))[:4]))

    def arm(self, arm=True):
        with self.send_lock:
            self.conn.mav.command_long_send(
                self.conn.target_system, self.conn.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1 if arm else 0, 0, 0, 0, 0, 0, 0)
        self._log_send("arm", 1 if arm else 0)

    def reset_sim(self):
        with self.send_lock:
            self.conn.mav.command_long_send(
                self.conn.target_system, self.conn.target_component,
                CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0)
        self._log_send("reset")

    _VEL_MASK = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )

    def send_velocity(self, vx, vy, vz, yaw_rate=0.0):
        """World-frame NED velocity setpoint + yaw rate."""
        with self.send_lock:
            self.conn.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self.conn.target_system, self.conn.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                self._VEL_MASK,
                0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate)
        self._log_send("vel", vx, vy, vz, yaw_rate)

    def send_attitude_rates(self, roll_rate, pitch_rate, yaw_rate, thrust):
        with self.send_lock:
            self.conn.mav.set_attitude_target_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self.conn.target_system, self.conn.target_component,
                mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE,
                [1, 0, 0, 0], roll_rate, pitch_rate, yaw_rate, thrust)
        self._log_send("rates", roll_rate, pitch_rate, yaw_rate)

    _QUAT_MASK = (
        mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_ROLL_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_PITCH_RATE_IGNORE
        | mavutil.mavlink.ATTITUDE_TARGET_TYPEMASK_BODY_YAW_RATE_IGNORE
    )

    def send_attitude_quat(self, q_wxyz, thrust):
        """Attitude setpoint (sim's internal attitude loop tracks it)."""
        with self.send_lock:
            self.conn.mav.set_attitude_target_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self.conn.target_system, self.conn.target_component,
                self._QUAT_MASK,
                list(q_wxyz), 0, 0, 0, thrust)
        self._log_send("att_quat", q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3])

    def _timesync_loop(self, hz):
        while self.running:
            with self.send_lock:
                try:
                    self.conn.mav.timesync_send(time.time_ns(), 0)
                except Exception:
                    pass
            time.sleep(1.0 / hz)

    # ------------------------------------------------------------------ helpers

    def latest_odom(self):
        if not self.odom:
            return None
        o = self.odom[-1]
        return {
            "t_us": o[0], "pos": o[1:4], "quat_wxyz": o[4:8],
            "vel": o[8:11], "rates": o[11:14], "reset_counter": o[14],
            "wall_ns": o[15],
        }

    def latest_attitude(self):
        if not self.attitude:
            return None
        a = self.attitude[-1]
        return {"t_ms": a[0], "roll": a[1], "pitch": a[2], "yaw": a[3], "wall_ns": a[7]}

    def wait_gate_map(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if self.gate_map:
                    return self.gate_map
            time.sleep(0.1)
        return None

    def close(self):
        self.running = False
        time.sleep(0.05)
        self.conn.close()
