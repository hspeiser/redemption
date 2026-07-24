"""Live browser overlay: GateNet annotates the sim camera stream in real time.

    .venv-train\\Scripts\\python.exe scripts\\live_overlay.py
    -> open http://localhost:8899

Draws per frame:
  - decoded corner peaks (green = inner-opening, yellow = outer-panel)
  - per-gate PnP wireframe + gate id + range (magenta), via pose-head-prior
    association against the gate map anchors
  - HUD: pose head estimate, next gate, inference latency, fps

NOTE: binds UDP 5600 — stop the RC recorder first (one listener per port).
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.vision_io import VisionRX
from aigp.vision.model import GateNet, rot6d_to_matrix
from scripts.train_net import orange_channel, decode_corners, POS_SCALE

W, H = 640, 360
PORT = 8899
GATE_COLORS = [(255, 0, 255), (255, 128, 0), (0, 200, 255),
               (128, 255, 0), (255, 0, 128), (0, 128, 255)]

# TRUE gate model (hole-centered local frame): 1.50 m hole + 2.72 m panel
_h, _px = 0.75, 1.324
_zt, _zb = -2.416 - (-1.0745), 0.267 - (-1.0745)
OBJ8 = np.array([[-_h, 0, -_h], [_h, 0, -_h], [_h, 0, _h], [-_h, 0, _h],
                 [-_px, 0, _zt], [_px, 0, _zt], [_px, 0, _zb], [-_px, 0, _zb]])

# rotate gate plane (y=0) into IPPE's Z=0 convention
_RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def solve_branches(corner_dict, K):
    """All planar-ambiguity branch candidates for a >=6-corner set:
    list of (rvec, tvec, rms)."""
    idx = sorted(corner_dict.keys())
    if len(idx) < 6:
        return []
    obj = np.ascontiguousarray(OBJ8[idx], np.float64)
    imgp = np.ascontiguousarray([corner_dict[k] for k in idx],
                                np.float64).reshape(-1, 1, 2)
    obj_p = np.ascontiguousarray(obj @ _RX90.T)
    out = []
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj_p, imgp, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return out
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj_p, imgp, K, None, rvec, tvec)
        except cv2.error:
            continue
        proj, _ = cv2.projectPoints(obj_p, rvec, tvec, K, None)
        rms = float(np.sqrt(((proj - imgp) ** 2).sum(axis=2).mean()))
        if rms > 3.5:
            continue
        R, _ = cv2.Rodrigues(rvec)
        R = R @ _RX90            # back to the gate's own local frame
        rv, _ = cv2.Rodrigues(R)
        out.append((rv, tvec.reshape(3, 1), rms))
    return out


class OneEuro:
    """One-Euro filter: heavy smoothing at rest, low lag during motion."""

    def __init__(self, min_cutoff=1.5, beta=0.05, dcutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, dcutoff
        self.x = None
        self.dx = 0.0
        self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dc, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mc + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


class GateTrack:
    """Temporal track for one gate: filtered corners + smoothed PnP pose."""

    MAX_MISSES = 8
    MAX_JUMP_M = 2.5     # reject single-frame pose teleports

    def __init__(self, gid):
        self.gid = gid
        self.rvec = None
        self.tvec = None
        self.corner_f = {}   # class -> (OneEuro u, OneEuro v)
        self.pose_f = [OneEuro(1.0, 0.3) for _ in range(6)]
        self.misses = 0
        self.last_smoothed = {}

    def force_pose(self, rvec, tvec, t_now):
        """Branch override: reset pose + filters to the given solution."""
        self.rvec = rvec.reshape(3, 1).copy()
        self.tvec = tvec.reshape(3, 1).copy()
        self.pose_f = [OneEuro(1.0, 0.3) for _ in range(6)]
        state = np.concatenate([self.rvec.ravel(), self.tvec.ravel()])
        for i in range(6):
            self.pose_f[i](state[i], t_now)

    def apply_motion(self, R_dc, t_dc):
        """Propagate the track pose by known inter-frame camera motion
        (X_cam' = R_dc X_cam + t_dc). Keeps prediction locked at speed."""
        if self.rvec is None:
            return
        R, _ = cv2.Rodrigues(self.rvec)
        R_new = R_dc @ R
        t_new = (R_dc @ self.tvec.ravel()) + t_dc
        rv, _ = cv2.Rodrigues(R_new)
        self.rvec = rv.reshape(3, 1)
        self.tvec = t_new.reshape(3, 1)
        # reseed pose filter positions without losing responsiveness
        state = np.concatenate([self.rvec.ravel(), self.tvec.ravel()])
        for i in range(6):
            if self.pose_f[i].x is not None:
                self.pose_f[i].x = state[i]

    def predict_corners(self, K):
        if self.rvec is None:
            return None
        proj, _ = cv2.projectPoints(OBJ8, self.rvec, self.tvec, K, None)
        return proj.reshape(-1, 2)

    def update(self, corners, K, t_now):
        """corners: dict class -> raw (u, v). Returns True if pose updated."""
        smoothed = {}
        for c, (u, v) in corners.items():
            if c not in self.corner_f:
                self.corner_f[c] = (OneEuro(), OneEuro())
            fu, fv = self.corner_f[c]
            smoothed[c] = (fu(u, t_now), fv(v, t_now))
        # drop stale corner filters
        for c in list(self.corner_f):
            if c not in corners:
                del self.corner_f[c]
        self.last_smoothed = dict(smoothed)
        idx = sorted(smoothed.keys())
        if len(idx) < 6:
            self.misses += 1
            return False
        obj = np.ascontiguousarray(OBJ8[idx], np.float64)
        img = np.ascontiguousarray([smoothed[c] for c in idx],
                                   np.float64).reshape(-1, 1, 2)
        if self.rvec is not None:
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, None, rvec=self.rvec.copy(),
                tvec=self.tvec.copy(), useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, None,
                                          flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None,
                                                      rvec, tvec)
                except cv2.error:
                    pass
        if not ok:
            self.misses += 1
            return False
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        rms = float(np.sqrt(((proj - img) ** 2).sum(axis=2).mean()))
        if rms > 3.0:
            self.misses += 1
            return False
        if self.tvec is not None and \
                np.linalg.norm(tvec.ravel() - self.tvec.ravel()) > self.MAX_JUMP_M:
            self.misses += 1
            return False
        # smooth pose (rotvec + tvec through One-Euro)
        state = np.concatenate([rvec.ravel(), tvec.ravel()])
        sm = np.array([self.pose_f[i](state[i], t_now) for i in range(6)])
        self.rvec = sm[:3].reshape(3, 1)
        self.tvec = sm[3:].reshape(3, 1)
        self.misses = 0
        return True

HTML = b"""<!doctype html><html><head><title>AI-GP Live Gate Overlay</title>
<style>body{background:#0b0b12;color:#ddd;font-family:monospace;text-align:center}
img{width:960px;max-width:98vw;image-rendering:auto;border:1px solid #333}
h3{color:#f60}</style></head>
<body><h3>AI-GP GateNet live overlay</h3>
<img src="/stream"><p>green = inner corners &nbsp; yellow = outer corners
&nbsp; colored quads = PnP-solved gates (id @ range)</p></body></html>"""


class Annotator:
    def __init__(self):
        calib = json.loads((REPO / "data" / "calib" / "calib.json").read_text())
        self.K = np.array([[calib["fx"], 0, calib["cx"]],
                           [0, calib["fy"], calib["cy"]], [0, 0, 1]])
        self.R_cb = np.array(calib["R_cam_from_body"])
        gates_f = REPO / "data" / "episodes" / "calib01" / "gates.json"
        self.gates = json.loads(gates_f.read_text()) if gates_f.exists() else []
        # world-frame gate rotation + hole-center (for absolute localization)
        from scipy.spatial.transform import Rotation as _Rot
        self.gate_Rw, self.gate_cw = [], []
        for g in self.gates:
            qw, qx, qy, qz = g["quat_wxyz"]
            Rg = _Rot.from_quat([qx, qy, qz, qw]).as_matrix()
            self.gate_Rw.append(Rg)
            self.gate_cw.append(np.asarray(g["pos"]) + Rg @ np.array([0, 0, -1.0745]))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        import os
        ckpt_path = os.environ.get(
            "GATENET_CKPT", str(REPO / "data" / "models" / "gatenet_v4_best.pt"))
        ck = torch.load(ckpt_path, map_location=self.device,
                        weights_only=False)
        print(f"checkpoint: {ckpt_path}", flush=True)
        self.model = GateNet().to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        print(f"model loaded (epoch {ck['epoch']}) on {self.device}", flush=True)

        self.jpeg = None            # latest annotated jpeg bytes
        self.lock = threading.Lock()
        self.fps = 0.0
        self.infer_ms = 0.0
        self._last_fid = None
        self.tracks = {}            # gate_id -> GateTrack
        self.state = {}             # live estimates served at /state.json

    # ---------------- geometry helpers ----------------

    def _project_anchor(self, gate, pos, Rwb):
        Xb = Rwb.T @ (np.asarray(gate["pos"]) - pos)
        Xc = self.R_cb @ Xb
        if Xc[2] < 1.0:
            return None
        u = self.K[0, 0] * Xc[0] / Xc[2] + self.K[0, 2]
        v = self.K[1, 1] * Xc[1] / Xc[2] + self.K[1, 2]
        return np.array([u, v]), Xc[2]

    def _solve_gate(self, corners):
        """corners: dict class->(u,v). Needs >=6 of 8."""
        idx = sorted(corners.keys())
        if len(idx) < 6:
            return None
        obj = np.ascontiguousarray(OBJ8[idx], np.float64)
        img = np.ascontiguousarray([corners[k] for k in idx],
                                   np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, None,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, self.K, None, rvec, tvec)
        except cv2.error:
            pass
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, None)
        rms = float(np.sqrt(((proj - img) ** 2).sum(axis=2).mean()))
        if rms > 3.0:
            return None
        return rvec, tvec, rms

    # ---------------- main per-frame work ----------------

    def process(self, bgr, motion=None):
        """motion: optional (R_dc(3,3), t_dc(3,)) camera motion since the
        previous frame (from gyro/EKF or replay truth) used to propagate
        track predictions before matching."""
        t0 = time.perf_counter()
        orange = orange_channel(bgr)
        img = np.concatenate([bgr.astype(np.float32) / 255.0,
                              orange[..., None]], 2).transpose(2, 0, 1)
        x = torch.from_numpy(img).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=self.device == "cuda"):
            out = self.model(x)
        dec = decode_corners(out["hm"][0].float().cpu(),
                             out["off"][0].float().cpu(), thresh=0.12)
        pos = out["pos"][0].float().cpu().numpy() * POS_SCALE
        Rwb = rot6d_to_matrix(out["rot6"].float())[0].cpu().numpy()
        gate_idx = int(out["gate_cls"][0].argmax().item())
        self.infer_ms = (time.perf_counter() - t0) * 1000

        vis = bgr.copy()
        t_now = time.perf_counter()
        # corner peaks (raw net output, faint)
        for c in range(8):
            col = (0, 180, 0) if c < 4 else (0, 180, 180)
            for (u, v, s) in dec[c]:
                cv2.circle(vis, (int(u), int(v)), 2, col, -1)

        used = [set() for _ in range(8)]

        def claim(c, pi):
            used[c].add(pi)

        # ---- 1. update existing tracks: match against their own prediction
        for gid, tr in list(self.tracks.items()):
            if motion is not None:
                tr.apply_motion(motion[0], motion[1])
            pred = tr.predict_corners(self.K)
            corners = {}
            if pred is not None:
                span = float(np.ptp(pred[:, 0]) + np.ptp(pred[:, 1])) / 2
                rad = np.clip(0.25 * span, 16.0, 120.0)
                for c in range(8):
                    best = None
                    for pi, (u, v, s) in enumerate(dec[c]):
                        if pi in used[c]:
                            continue
                        d = np.hypot(u - pred[c, 0], v - pred[c, 1])
                        if d < rad and (best is None or d < best[0]):
                            best = (d, pi, u, v)
                    if best is not None:
                        corners[c] = (best[2], best[3])
            ok = tr.update(corners, self.K, t_now)
            if ok:
                for c in corners:
                    for pi, (u, v, s) in enumerate(dec[c]):
                        if (u, v) == corners[c]:
                            claim(c, pi)
            if tr.misses > GateTrack.MAX_MISSES:
                del self.tracks[gid]

        # ---- 2. acquire new tracks for untracked gates via pose-head anchors
        for gi, g in enumerate(self.gates):
            if g["gate_id"] in self.tracks:
                continue
            pr = self._project_anchor(g, pos, Rwb)
            if pr is None:
                continue
            ac, depth = pr
            if not (-100 < ac[0] < W + 100 and -100 < ac[1] < H + 100):
                continue
            rad = max(50.0, 1.6 * self.K[0, 0] * 2.6 / max(depth, 2.0))
            corners = {}
            for c in range(8):
                best = None
                for pi, (u, v, s) in enumerate(dec[c]):
                    if pi in used[c]:
                        continue
                    d = np.hypot(u - ac[0], v - ac[1])
                    if d < rad and (best is None or d < best[0]):
                        best = (d, pi, u, v)
                if best is not None:
                    corners[c] = (best[2], best[3])
            sol = self._solve_gate(corners)
            if sol is None:
                continue
            tr = GateTrack(g["gate_id"])
            tr.rvec, tr.tvec = sol[0], sol[1]
            if isinstance(tr.rvec, np.ndarray) and tr.rvec.shape != (3, 1):
                tr.rvec = tr.rvec.reshape(3, 1)
            self.tracks[g["gate_id"]] = tr
            for c in corners:
                for pi, (u, v, s) in enumerate(dec[c]):
                    if (u, v) == corners[c]:
                        claim(c, pi)

        # ---- 2b. planar-branch disambiguation: the two IPPE solutions
        # reproject nearly identically but sit meters apart in 3D; pick the
        # branch whose implied ABSOLUTE position matches the pose-head prior.
        for gid, tr in self.tracks.items():
            if tr.rvec is None or not getattr(tr, "last_smoothed", None):
                continue
            cands = solve_branches(tr.last_smoothed, self.K)
            if len(cands) < 2:
                continue

            def world_of(rv, tv, _gid=gid):
                R_rel, _ = cv2.Rodrigues(rv)
                cig = -R_rel.T @ tv.ravel()
                return self.gate_cw[_gid] + self.gate_Rw[_gid] @ cig

            cur_w = world_of(tr.rvec, tr.tvec)
            best = min(cands,
                       key=lambda c: np.linalg.norm(world_of(c[0], c[1]) - pos))
            best_w = world_of(best[0], best[1])
            if (np.linalg.norm(best_w - pos) + 1.0
                    < np.linalg.norm(cur_w - pos)
                    and np.linalg.norm(best_w - cur_w) > 0.5):
                tr.force_pose(best[0], best[1], t_now)

        # ---- 3. draw tracked gates from their SMOOTHED poses
        for gid, tr in self.tracks.items():
            if tr.rvec is None or tr.misses > 2:
                continue
            col = GATE_COLORS[gid % len(GATE_COLORS)]
            for quad in (OBJ8[:4], OBJ8[4:]):
                proj, _ = cv2.projectPoints(np.ascontiguousarray(quad),
                                            tr.rvec, tr.tvec, self.K, None)
                cv2.polylines(vis, [proj.reshape(-1, 2).astype(np.int32)],
                              True, col, 2)
            rng = float(np.linalg.norm(tr.tvec))
            proj, _ = cv2.projectPoints(OBJ8[4].reshape(1, 3), tr.rvec,
                                        tr.tvec, self.K, None)
            tx, ty = proj.reshape(2).astype(int)
            cv2.putText(vis, f"G{gid} {rng:.1f}m",
                        (max(2, tx), max(14, ty - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # ---- multi-gate JOINT absolute solve: corners from ALL tracked
        # gates against their world coordinates. Non-planar, wide baseline —
        # kills the single-gate tilt ambiguity that costs meters at range.
        joint_world = None
        jw_obj, jw_img = [], []
        for gid, tr in self.tracks.items():
            sm = getattr(tr, "last_smoothed", None)
            if tr.rvec is None or tr.misses > 2 or not sm:
                continue
            for c, (u, v) in sm.items():
                jw_obj.append(self.gate_cw[gid] + self.gate_Rw[gid] @ OBJ8[c])
                jw_img.append([u, v])
        if len(jw_obj) >= 10:
            objw = np.ascontiguousarray(jw_obj, np.float64)
            imgw = np.ascontiguousarray(jw_img, np.float64).reshape(-1, 1, 2)
            ok, rvec, tvec = cv2.solvePnP(objw, imgw, self.K, None,
                                          flags=cv2.SOLVEPNP_SQPNP)
            if ok:
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(objw, imgw, self.K,
                                                      None, rvec, tvec)
                except cv2.error:
                    pass
                proj, _ = cv2.projectPoints(objw, rvec, tvec, self.K, None)
                rms = float(np.sqrt(((proj - imgw) ** 2).sum(axis=2).mean()))
                if rms < 3.0:
                    Rcw, _ = cv2.Rodrigues(rvec)
                    # camera->world rotation as body->world (R_wb = R_wc R_cb)
                    R_wb = Rcw.T @ self.R_cb
                    joint_world = (-Rcw.T @ tvec.ravel(), rms, R_wb)

        # ---- absolute localization from each tracked gate (static map) ----
        gate_states = []
        for gid, tr in self.tracks.items():
            if tr.rvec is None or tr.misses > 2:
                continue
            R_rel, _ = cv2.Rodrigues(tr.rvec)
            t_rel = tr.tvec.ravel()
            cam_in_gate = -R_rel.T @ t_rel
            # camera/world: body pos (camera at body origin); note cam frame
            # offset vs body handled by R_cb being pure rotation at origin
            p_world = self.gate_cw[gid] + self.gate_Rw[gid] @ cam_in_gate
            gate_states.append({
                "gid": int(gid), "range_m": float(np.linalg.norm(t_rel)),
                "cam_world": [float(v) for v in p_world],
                "corners": {str(k): [float(u), float(v)] for k, (u, v)
                            in getattr(tr, "last_smoothed", {}).items()},
            })
        self.state = {
            "wall_ns": time.time_ns(),
            "pose_head_world": [float(v) for v in pos],
            "joint_world": ([float(v) for v in joint_world[0]]
                            if joint_world else None),
            "joint_rms_px": (round(joint_world[1], 3) if joint_world else None),
            "joint_R_wb": ([[float(x) for x in row] for row in joint_world[2]]
                           if joint_world else None),
            "next_gate": int(gate_idx),
            "gates": gate_states,
            "infer_ms": round(self.infer_ms, 1),
            "fps": round(self.fps, 1),
        }

        hud = (f"pose ({pos[0]:6.1f},{pos[1]:6.1f},{pos[2]:6.1f})  "
               f"next gate {gate_idx}  {self.infer_ms:4.1f}ms  "
               f"{self.fps:4.1f}fps")
        cv2.putText(vis, hud, (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", vis,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()

    def run(self, vis_rx):
        n, t0 = 0, time.time()
        while True:
            latest = vis_rx.latest
            if latest is None or latest[0] == self._last_fid:
                time.sleep(0.003)
                continue
            self._last_fid = latest[0]
            frame = cv2.imdecode(np.frombuffer(latest[2], np.uint8),
                                 cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))
            self.process(frame)
            n += 1
            if time.time() - t0 >= 2.0:
                self.fps = n / (time.time() - t0)
                n, t0 = 0, time.time()


ANN = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML)
        elif self.path == "/state.json":
            body = json.dumps(ANN.state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/snap.jpg":
            with ANN.lock:
                j = ANN.jpeg
            if j is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(j)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            try:
                while True:
                    with ANN.lock:
                        j = ANN.jpeg
                    if j is None or j is last:
                        time.sleep(0.01)
                        continue
                    last = j
                    self.wfile.write(b"--frame\r\n"
                                     b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(j)
                    self.wfile.write(b"\r\n")
            except (ConnectionAbortedError, ConnectionResetError,
                    BrokenPipeError):
                return
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global ANN
    ANN = Annotator()
    vis_rx = VisionRX()
    t = threading.Thread(target=ANN.run, args=(vis_rx,), daemon=True)
    t.start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"live overlay at http://localhost:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
