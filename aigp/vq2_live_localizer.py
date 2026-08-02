"""Live VQ2 V7+V10+V11 visual-inertial localizer.

This is the causal counterpart of ``scripts/vq2_align.py``.  It consumes only
HIGHRES_IMU, camera JPEGs, official active-gate events, and a relative map.
V7+V10 supplies continuous corner updates.  V11 is event-triggered only after
the regular corner update fails and can make a conservative translation pin;
it never rotates gyro attitude.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from aigp.calib.detect import detect_gates
from aigp.ekf import GateEKF
from aigp.mavlink_io import MavIO
from aigp.vision.crop_gate import (
    CropGateNet,
    decode_crop_corners,
    orange_channel as crop_orange_channel,
    proposal_channel,
    warp_gate_crop,
)
from aigp.vision.labels import load_calib
from aigp.vision.model import GateNet
from aigp.vision_io import VisionRX
from aigp.vq2_map import gate_quads_world_vq2
from scripts.train_net import decode_corners, orange_channel


HOLE = 0.75
PANEL = 1.35
OBJ8 = np.array([
    [-HOLE, 0, -HOLE], [HOLE, 0, -HOLE],
    [HOLE, 0, HOLE], [-HOLE, 0, HOLE],
    [-PANEL, 0, -PANEL], [PANEL, 0, -PANEL],
    [PANEL, 0, PANEL], [-PANEL, 0, PANEL],
], np.float64)
RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
OBJ_APPARENT_INNER = np.array([
    [-HOLE, HOLE, 0.0], [HOLE, HOLE, 0.0],
    [HOLE, -HOLE, 0.0], [-HOLE, -HOLE, 0.0],
], np.float64)
FLIP = (1, 0, 3, 2, 5, 4, 7, 6)


def _set_worker_affinity(mask_text: str | None) -> None:
    if not mask_text or mask_text.lower() in {"all", "none"}:
        return
    mask = int(mask_text, 0)
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.SetProcessAffinityMask(
            kernel32.GetCurrentProcess(), mask
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        # Dense vision is allowed to finish late; it must never preempt the
        # simulator or the real-time MAVLink/control process.
        below_normal_priority_class = 0x00004000
        kernel32.SetPriorityClass.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(), below_normal_priority_class
        )
    else:
        os.sched_setaffinity(
            0,
            {
                index for index in range(os.cpu_count() or 1)
                if mask >> index & 1
            },
        )


def _dense_corner_peaks(
    image: np.ndarray,
    primary: GateNet,
    refiner: GateNet,
    device: torch.device,
    threshold: float,
    refine_radius_px: float,
):
    orange = orange_channel(image)
    network_input = np.concatenate([
        image.astype(np.float32) / 255.0,
        orange[..., None],
    ], axis=2).transpose(2, 0, 1)
    tensor = torch.from_numpy(network_input).unsqueeze(0).to(device)

    def run(model):
        with torch.no_grad(), torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(tensor)
        return decode_corners(
            output["hm"][0].float().cpu(),
            output["off"][0].float().cpu(),
            thresh=threshold,
        )

    primary_peaks = run(primary)
    refined_peaks = run(refiner)
    snapped = []
    for corner_class, class_peaks in enumerate(primary_peaks):
        ring = range(4) if corner_class < 4 else range(4, 8)
        candidates = [
            point for refine_class in ring
            for point in refined_peaks[refine_class]
        ]
        rows = []
        for u, v, score in class_peaks:
            nearest = min(
                candidates,
                key=lambda point: np.hypot(
                    point[0] - u, point[1] - v
                ),
                default=None,
            )
            if nearest is not None and np.hypot(
                nearest[0] - u, nearest[1] - v
            ) <= refine_radius_px:
                rows.append((nearest[0], nearest[1], score))
            else:
                rows.append((u, v, score))
        snapped.append(rows)
    return snapped


def _merge_corner_peaks(
    left: list[list[tuple[float, float, float]]],
    right: list[list[tuple[float, float, float]]],
    dedupe_radius_px: float = 2.0,
) -> list[list[tuple[float, float, float]]]:
    """Union complementary GateNet peaks without duplicating one corner."""
    merged = []
    for left_class, right_class in zip(left, right, strict=True):
        kept: list[tuple[float, float, float]] = []
        for point in sorted(
            [*left_class, *right_class],
            key=lambda row: float(row[2]),
            reverse=True,
        ):
            if all(
                np.hypot(point[0] - other[0], point[1] - other[1])
                > dedupe_radius_px
                for other in kept
            ):
                kept.append(point)
        merged.append(kept)
    return merged


def _dense_worker_main(
    request_queue,
    result_queue,
    primary_checkpoint: str,
    refine_checkpoint: str,
    gate_primary_checkpoint: str | None,
    gate_primary_gates: tuple[int, ...],
    threshold: float,
    refine_radius_px: float,
    device_text: str,
    worker_threads: int,
    affinity_mask: str | None,
) -> None:
    """Run dense GateNet away from telemetry, EKF, and flight control."""
    try:
        _set_worker_affinity(affinity_mask)
        torch.set_num_threads(max(1, int(worker_threads)))
        torch.set_num_interop_threads(1)
        device = torch.device(device_text)
        models = []
        checkpoints = [primary_checkpoint, refine_checkpoint]
        if gate_primary_checkpoint is not None:
            checkpoints.append(gate_primary_checkpoint)
        for checkpoint_path in checkpoints:
            payload = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model = GateNet().to(device)
            model.load_state_dict(payload["model"])
            model.eval()
            models.append(model)
        result_queue.put({
            "worker_ready": True,
            "worker_pid": os.getpid(),
        }, timeout=5.0)
        while True:
            request = request_queue.get()
            if request is None:
                return
            started = time.perf_counter()
            active_gate = int(request.get("active_gate", -1))
            use_gate_primary = (
                len(models) > 2
                and active_gate in gate_primary_gates
            )
            ensemble_gate_primary = bool(
                use_gate_primary
                and os.environ.get("AIGP_GATE_PRIMARY_ENSEMBLE") == "1"
            )
            if ensemble_gate_primary:
                peaks = _merge_corner_peaks(
                    _dense_corner_peaks(
                        request["image"], models[0], models[1], device,
                        threshold, refine_radius_px,
                    ),
                    _dense_corner_peaks(
                        request["image"], models[2], models[1], device,
                        threshold, refine_radius_px,
                    ),
                )
            else:
                peaks = _dense_corner_peaks(
                    request["image"],
                    models[2] if use_gate_primary else models[0],
                    models[1],
                    device,
                    threshold,
                    refine_radius_px,
                )
            result = {
                "generation": int(request["generation"]),
                "frame_id": int(request["frame_id"]),
                "peaks": peaks,
                "inference_ms": (
                    time.perf_counter() - started
                ) * 1000.0,
                "worker_pid": os.getpid(),
                "gate_primary_used": bool(use_gate_primary),
                "gate_primary_ensemble_used": bool(
                    ensemble_gate_primary
                ),
            }
            try:
                result_queue.put(result, timeout=0.25)
            except queue.Full:
                # The controller owns freshness. Never block vision compute
                # behind an unconsumed old result.
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                result_queue.put_nowait(result)
    except BaseException as error:
        try:
            result_queue.put({
                "worker_error": repr(error),
                "worker_pid": os.getpid(),
            }, timeout=0.25)
        except BaseException:
            pass


@dataclass(frozen=True)
class LiveLocalizerState:
    position: np.ndarray
    velocity: np.ndarray
    quat_wxyz: np.ndarray
    gyro_raw: np.ndarray
    position_sigma_m: float
    visual_age_s: float
    corners_fused: int
    frame_id: int | None
    initialized: bool


def _pnp_gate(det: dict, intrinsics: np.ndarray):
    if det["inner"] is None:
        return None
    image = np.ascontiguousarray(
        np.concatenate([det["inner"], det["outer"]]), np.float64
    ).reshape(-1, 1, 2)
    object_points = np.ascontiguousarray(OBJ8 @ RX90.T)
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image,
            intrinsics,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, image, intrinsics, None, rvec, tvec
            )
        except cv2.error:
            continue
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, intrinsics, None
        )
        rms = float(np.sqrt(np.mean(
            np.sum((projected.reshape(-1, 2) - image.reshape(-1, 2)) ** 2,
                   axis=1)
        )))
        translation = np.asarray(tvec, float).reshape(3)
        if translation[2] <= 0.0:
            continue
        if best is None or rms < best[1]:
            best = translation, rms
    return best


def _pnp_apparent_inner(points: np.ndarray, intrinsics: np.ndarray):
    image = np.ascontiguousarray(points, np.float64).reshape(-1, 1, 2)
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            OBJ_APPARENT_INNER,
            image,
            intrinsics,
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return []
    solutions = []
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                OBJ_APPARENT_INNER,
                image,
                intrinsics,
                None,
                rvec,
                tvec,
            )
        except cv2.error:
            continue
        translation = np.asarray(tvec, float).reshape(3)
        if translation[2] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            OBJ_APPARENT_INNER, rvec, tvec, intrinsics, None
        )
        rms = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(4, 2) - points) ** 2, axis=1
        ))))
        solutions.append((translation, rms))
    return sorted(solutions, key=lambda row: row[1])


def _pnp_points_all(
    indices: list[int],
    points: list[tuple[float, float]],
    intrinsics: np.ndarray,
):
    if len(indices) < 6:
        return []
    object_points = np.ascontiguousarray(
        OBJ8[indices] @ RX90.T
    )
    image = np.ascontiguousarray(
        points, np.float64
    ).reshape(-1, 1, 2)
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image,
            intrinsics,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return []
    solutions = []
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, image, intrinsics, None, rvec, tvec
            )
        except cv2.error:
            continue
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, intrinsics, None
        )
        rms = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(-1, 2) - image.reshape(-1, 2)) ** 2,
            axis=1,
        ))))
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, float).reshape(3)
        if translation[2] > 0.0:
            solutions.append((rotation @ RX90, translation, rms))
    return sorted(solutions, key=lambda row: row[2])


class LiveVQ2Localizer:
    def __init__(
        self,
        *,
        mavlink: MavIO,
        vision: VisionRX,
        map_path: Path,
        primary_checkpoint: Path,
        refine_checkpoint: Path,
        gate_primary_checkpoint: Path | None = None,
        gate_primary_gates: tuple[int, ...] | None = None,
        crop_checkpoint: Path,
        proposal_checkpoint: Path,
        calibration_path: Path,
        threshold: float = 0.12,
        refine_radius_px: float = 2.0,
        crop_padding: float = 2.6,
        crop_interval_s: float = 0.10,
        async_interval_s: float = 0.10,
        max_async_result_age_s: float = 0.30,
        dense_device: str | None = None,
        dense_process_isolation: bool = False,
        dense_worker_threads: int = 4,
        dense_worker_affinity: str | None = None,
        direct_position_pins: bool = False,
        crop_direct_position_pins: bool = False,
        crop_track_enabled: bool | None = None,
        crop_track_interval_s: float = 0.10,
        crop_track_gates: tuple[int, ...] | None = None,
    ) -> None:
        self.mavlink = mavlink
        self.vision = vision
        map_payload = json.loads(Path(map_path).read_text())
        self.map_template = map_payload["gates"]
        fixed_spawn_gate0 = map_payload.get("spawn_to_gate0")
        self.fixed_spawn_gate0 = (
            np.asarray(fixed_spawn_gate0, float)
            if fixed_spawn_gate0 is not None else None
        )
        calibration = load_calib(calibration_path)
        fx, fy, cx, cy = calibration["K"]
        self.K = np.array([
            [fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]
        ])
        self.R_cb = np.asarray(calibration["R_cb"], float)
        self.threshold = float(threshold)
        self.refine_radius_px = float(refine_radius_px)
        self.crop_padding = float(crop_padding)
        self.crop_interval_s = float(crop_interval_s)
        self.async_interval_s = max(float(async_interval_s), 0.0)
        self.max_async_result_age_s = max(
            float(max_async_result_age_s), 0.0
        )
        self.direct_position_pins = bool(direct_position_pins)
        self.crop_direct_position_pins = bool(crop_direct_position_pins)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dense_device = torch.device(
            dense_device if dense_device is not None else self.device
        )
        self.dense_process_isolation = bool(dense_process_isolation)
        self.dense_worker_threads = max(1, int(dense_worker_threads))
        self.dense_worker_affinity = dense_worker_affinity
        self.primary_checkpoint = str(primary_checkpoint)
        self.refine_checkpoint = str(refine_checkpoint)
        self.gate_primary_checkpoint = (
            str(gate_primary_checkpoint)
            if gate_primary_checkpoint is not None else None
        )
        self.gate_primary_gates = frozenset(
            int(gate) for gate in (gate_primary_gates or ())
        )

        self.primary = None
        self.refiner = None
        self.gate_primary = None
        if not self.dense_process_isolation:
            self.primary = self._load_gatenet(primary_checkpoint)
            self.refiner = self._load_gatenet(refine_checkpoint)
            if gate_primary_checkpoint is not None:
                self.gate_primary = self._load_gatenet(
                    gate_primary_checkpoint
                )
        crop_payload = torch.load(
            crop_checkpoint, map_location=self.device, weights_only=False
        )
        self.crop = CropGateNet().to(self.device)
        self.crop.load_state_dict(crop_payload["model"])
        self.crop.eval()
        self.crop_size = int(crop_payload.get("crop_size", 256))
        self.crop_prior = proposal_channel(self.crop_size)
        from ultralytics import YOLO
        self.proposal = YOLO(str(proposal_checkpoint))
        self.proposal_device = str(self.device)
        if self.device.type == "cuda":
            try:
                from torchvision.ops import nms
                nms(
                    torch.zeros((1, 4), device=self.device),
                    torch.ones(1, device=self.device),
                    0.5,
                )
            except (NotImplementedError, RuntimeError):
                self.proposal_device = "cpu"

        self.ekf: GateEKF | None = None
        self.gates: list[dict] = []
        self.gate_world: list[np.ndarray] = []
        self.last_imu_us: int | None = None
        self.last_frame_id: int | None = None
        self.last_visual_wall = -np.inf
        self.last_crop_wall = -np.inf
        self.last_pin_wall = -np.inf
        self.last_generic_pin: tuple[float, int, np.ndarray] | None = None
        self.last_direct_pin: tuple[float, int, np.ndarray] | None = None
        self.last_direct_pin_applied = False
        self.last_relocalization: tuple[
            float, int, np.ndarray
        ] | None = None
        self.last_fused = 0
        self.last_update_source = "none"
        self.update_counts = {
            "v7_v10": 0,
            "v7_v10_direct": 0,
            "gate_primary_v10": 0,
            "gate_primary_v10_direct": 0,
            "v11": 0,
            "classical_pin": 0,
            "net_relocalize": 0,
            "stale_vision_drop": 0,
            "dense_gate_mismatch": 0,
            "none": 0,
            "crop_track": 0,
            "crop_track_direct": 0,
            "crop_track_none": 0,
            "crop_track_stale": 0,
            "crop_track_gate_mismatch": 0,
            "gate_ensemble_v10": 0,
            "gate_ensemble_v10_direct": 0,
        }
        self.dense_attempts_by_gate: dict[int, int] = {}
        self.dense_fusions_by_gate: dict[int, int] = {}
        self.crop_attempts_by_gate: dict[int, int] = {}
        self.crop_fusions_by_gate: dict[int, int] = {}
        self.last_gate_event_correction: dict | None = None
        self.async_inference_ms = 0.0
        # Opt-in 10Hz crop tracker (AIGP_CROP_TRACKER=1): between dense
        # V7/V10 fixes, place a crop from the EKF's own corner projections
        # (no YOLO proposal) and fuse CropGateNet corners.  Dense inference
        # costs ~220ms on CPU and caps vision at ~3Hz; the crop net costs
        # ~50-70ms, so this holds landmark age near the crop cadence where
        # the 3Hz dense cadence lets belief coast ~1s on fast approaches.
        self.crop_track_enabled = (
            os.environ.get("AIGP_CROP_TRACKER") == "1"
            if crop_track_enabled is None
            else bool(crop_track_enabled)
        )
        self.crop_track_gates = (
            None
            if crop_track_gates is None
            else frozenset(int(gate) for gate in crop_track_gates)
        )
        self.crop_track_interval_s = max(
            float(
                os.environ.get(
                    "AIGP_CROP_TRACKER_INTERVAL",
                    str(crop_track_interval_s),
                )
                if crop_track_enabled is None
                else crop_track_interval_s
            ),
            0.02,
        )
        self.crop_track_min_span_px = 30.0
        self.crop_track_max_age_s = 1.2
        self.crop_track_max_result_age_s = 0.35
        self.crop_track_ms = 0.0
        self._crop_lock = threading.Lock()
        self._crop_thread = None
        self._crop_stop_event = None
        self._crop_job = None
        self._crop_result = None
        self._crop_busy = False
        self._crop_last_dispatched = None
        self._crop_last_dispatch_wall = -np.inf
        self._async_lock = threading.Lock()
        self._async_job = None
        self._async_result = None
        self._async_running = False
        self._async_busy = False
        self._async_thread: threading.Thread | None = None
        self._async_stop_event: threading.Event | None = None
        self._async_last_dispatched: int | None = None
        self._async_last_dispatch_wall = -np.inf
        self._async_generation = 0
        self._async_pending: dict[tuple[int, int], dict] = {}
        self._dense_context = None
        self._dense_request_queue = None
        self._dense_result_queue = None
        self._dense_process = None
        self._dense_ready = False
        self.last_gyro = np.zeros(3)
        self.anchor_diagnostics: dict = {}
        self._debug_lock = threading.Lock()
        self._debug_image: np.ndarray | None = None
        self._debug_payload: dict = {}
        if self.dense_process_isolation:
            # Load and warm the expensive worker before any episode can arm.
            # Process startup is never allowed to occur during live flight.
            self._ensure_dense_process()

    def _load_gatenet(self, checkpoint: Path) -> GateNet:
        payload = torch.load(
            checkpoint, map_location=self.dense_device, weights_only=False
        )
        model = GateNet().to(self.dense_device)
        model.load_state_dict(payload["model"])
        model.eval()
        return model

    @staticmethod
    def _decode_jpeg(frame_tuple) -> np.ndarray | None:
        if frame_tuple is None:
            return None
        array = np.frombuffer(frame_tuple[2], np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def _corner_peaks(
        self, image: np.ndarray, active_gate: int | None = None
    ):
        if self.primary is None or self.refiner is None:
            raise RuntimeError(
                "synchronous dense inference is unavailable while the "
                "process-isolated GateNet worker is enabled"
            )
        use_gate_primary = bool(
            self.gate_primary is not None
            and active_gate in self.gate_primary_gates
        )
        if (
            use_gate_primary
            and os.environ.get("AIGP_GATE_PRIMARY_ENSEMBLE") == "1"
        ):
            return _merge_corner_peaks(
                _dense_corner_peaks(
                    image, self.primary, self.refiner,
                    self.dense_device, self.threshold,
                    self.refine_radius_px,
                ),
                _dense_corner_peaks(
                    image, self.gate_primary, self.refiner,
                    self.dense_device, self.threshold,
                    self.refine_radius_px,
                ),
            )
        return _dense_corner_peaks(
            image,
            self.gate_primary if use_gate_primary else self.primary,
            self.refiner,
            self.dense_device,
            self.threshold,
            self.refine_radius_px,
        )

    def _v11_anchor_translation(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        """Gate-0 translation before an EKF/map prediction exists."""
        result = self.proposal.predict(
            source=image,
            conf=0.05,
            iou=0.5,
            imgsz=640,
            device=self.proposal_device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        candidates = []
        for box in sorted(
            boxes,
            key=lambda row: -float(
                (row[2] - row[0]) * (row[3] - row[1])
            ),
        )[:3]:
            center = 0.5 * (box[:2] + box[2:])
            proposal_span = float(max(
                box[2] - box[0], box[3] - box[1]
            ))
            side = max(18.0, proposal_span * self.crop_padding)
            crop, _forward, inverse = warp_gate_crop(
                image, center, side, self.crop_size
            )
            orange = crop_orange_channel(crop)
            network_input = np.concatenate([
                crop.astype(np.float32) / 255.0,
                orange[..., None],
                self.crop_prior[..., None],
            ], axis=2).transpose(2, 0, 1)
            tensor = torch.from_numpy(network_input).unsqueeze(0).to(
                self.device
            )
            with torch.no_grad(), torch.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                output = self.crop(tensor)
            decoded = decode_crop_corners(
                output, self.crop_size, inverse_affine=inverse
            )
            points = np.asarray(decoded["corners"], np.float32)[:4]
            accepted = (
                (np.asarray(decoded["scores"])[:4] >= 0.05)
                & (np.asarray(decoded["visibility"])[:4] >= 0.20)
                & (float(decoded["presence"]) >= 0.20)
            )
            if not accepted.all():
                continue
            for translation, rms in _pnp_apparent_inner(points, self.K):
                depth = float(np.linalg.norm(translation))
                if rms <= 3.0 and 5.0 <= depth <= 20.0:
                    candidates.append((
                        rms,
                        -float(decoded["presence"]),
                        translation,
                    ))
        if not candidates:
            return None
        rms, _negative_presence, translation = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        return translation, rms

    def initialize(
        self,
        timeout_s: float = 2.8,
        keepalive: Callable[[], None] | None = None,
    ) -> LiveLocalizerState:
        """Initialize gravity attitude and re-anchor the relative map at spawn."""
        start_wall_ns = time.time_ns()
        deadline = time.time() + timeout_s
        translations = []
        rms_rows = []
        anchor_sources = []
        seen_frame = None
        while time.time() < deadline:
            if keepalive is not None:
                keepalive()
            frame = self.vision.latest
            if frame is not None and frame[0] != seen_frame:
                seen_frame = frame[0]
                image = self._decode_jpeg(frame)
                if image is not None:
                    solution = self._v11_anchor_translation(image)
                    source = "v11"
                    if solution is None:
                        detections = [
                            detection for detection in detect_gates(
                                image, min_area=500
                            )
                            if detection["inner"] is not None
                        ]
                        if detections:
                            solution = _pnp_gate(
                                max(
                                    detections,
                                    key=lambda row: row["area"],
                                ),
                                self.K,
                            )
                            source = "classical"
                    if solution is not None and solution[1] <= 2.5:
                        translations.append(solution[0])
                        rms_rows.append(solution[1])
                        anchor_sources.append(source)
            if len(translations) >= 10 and time.time() + 0.15 >= deadline:
                break
            time.sleep(0.01)

        imu_rows = [
            row for row in list(self.mavlink.imu)
            if row[-1] >= start_wall_ns
        ]
        if not imu_rows:
            raise RuntimeError("no post-reset IMU samples for VQ2 initialization")
        acceleration = np.asarray([row[1:4] for row in imu_rows], float)
        gyro = np.asarray([row[4:7] for row in imu_rows], float)
        stationary = (
            (np.linalg.norm(acceleration, axis=1) > 8.0)
            & (np.linalg.norm(acceleration, axis=1) < 12.0)
            & (np.linalg.norm(gyro, axis=1) < 0.08)
        )
        if stationary.sum() < 20:
            stationary = np.ones(len(acceleration), dtype=bool)
        specific_force = np.median(acceleration[stationary], axis=0)
        pitch = np.arcsin(np.clip(specific_force[0] / 9.81, -1.0, 1.0))
        roll = np.arctan2(-specific_force[1], -specific_force[2])
        initial_rotation = Rotation.from_euler(
            "ZYX", [0.0, pitch, roll]
        )
        if len(translations) < 3:
            raise RuntimeError(
                f"spawn gate anchor failed: {len(translations)} valid frames"
            )
        translations_all = np.asarray(translations)
        # Crop PnP can briefly occupy a second planar-depth branch while the
        # arena lights/reset settle.  Select the largest dense translation
        # mode before taking the median; never average two depth branches.
        distances = np.linalg.norm(
            translations_all[:, None, :]
            - translations_all[None, :, :],
            axis=2,
        )
        support = np.sum(distances <= 0.35, axis=1)
        seed = int(np.argmax(support))
        cluster_mask = distances[seed] <= 0.50
        if int(cluster_mask.sum()) >= 5:
            translations_selected = translations_all[cluster_mask]
            rms_selected = np.asarray(rms_rows)[cluster_mask]
            sources_selected = [
                source for source, keep in zip(
                    anchor_sources, cluster_mask
                ) if keep
            ]
        else:
            translations_selected = translations_all
            rms_selected = np.asarray(rms_rows)
            sources_selected = anchor_sources
        camera_to_world = initial_rotation.as_matrix() @ self.R_cb.T
        observed_gate0_visual = camera_to_world @ np.median(
            translations_selected, axis=0
        )
        observed_gate0 = observed_gate0_visual
        if self.fixed_spawn_gate0 is not None:
            anchor_error = float(np.linalg.norm(
                observed_gate0_visual - self.fixed_spawn_gate0
            ))
            if anchor_error > 0.50:
                raise RuntimeError(
                    "spawn gate visual anchor disagrees with canonical "
                    f"transform by {anchor_error:.2f} m"
                )
            observed_gate0 = self.fixed_spawn_gate0.copy()

        self.gates = copy.deepcopy(self.map_template)
        map_gate0 = np.asarray(self.gates[0]["pos"], float)
        yaw_map = float(np.arctan2(map_gate0[1], map_gate0[0]))
        yaw_observed = float(np.arctan2(
            observed_gate0[1], observed_gate0[0]
        ))
        yaw_delta = yaw_observed - yaw_map
        anchor_rotation = Rotation.from_euler("Z", yaw_delta).as_matrix()
        for gate in self.gates:
            gate["pos"] = (
                observed_gate0
                + anchor_rotation
                @ (np.asarray(gate["pos"], float) - map_gate0)
            ).tolist()
            qw, qx, qy, qz = gate["quat_wxyz"]
            gate_rotation = Rotation.from_quat(
                [qx, qy, qz, qw]
            ).as_matrix()
            quaternion = Rotation.from_matrix(
                anchor_rotation @ gate_rotation
            ).as_quat()
            gate["quat_wxyz"] = [
                float(quaternion[3]), float(quaternion[0]),
                float(quaternion[1]), float(quaternion[2]),
            ]
        self.gate_world = [
            np.concatenate(gate_quads_world_vq2(gate))
            for gate in self.gates
        ]

        latest_imu = imu_rows[-1]
        self.ekf = GateEKF(self.K, self.R_cb, sigma_px=1.5)
        quaternion = initial_rotation.as_quat()
        self.ekf.init_state(
            np.zeros(3),
            np.zeros(3),
            [quaternion[3], quaternion[0], quaternion[1], quaternion[2]],
            latest_imu[0] * 1e-6,
        )
        self.last_imu_us = int(latest_imu[0])
        self.last_frame_id = seen_frame
        self.last_visual_wall = time.time()
        self.last_crop_wall = -np.inf
        self.last_pin_wall = -np.inf
        self.last_generic_pin = None
        self.last_relocalization = None
        self.last_direct_pin = None
        self.last_direct_pin_applied = False
        self.last_fused = 0
        self.last_update_source = "none"
        self.update_counts = {
            "v7_v10": 0,
            "v7_v10_direct": 0,
            "gate_primary_v10": 0,
            "gate_primary_v10_direct": 0,
            "v11": 0,
            "classical_pin": 0,
            "net_relocalize": 0,
            "stale_vision_drop": 0,
            "dense_gate_mismatch": 0,
            "none": 0,
            "crop_track": 0,
            "crop_track_direct": 0,
            "crop_track_none": 0,
            "crop_track_stale": 0,
            "crop_track_gate_mismatch": 0,
            "gate_ensemble_v10": 0,
            "gate_ensemble_v10_direct": 0,
        }
        self.dense_attempts_by_gate = {}
        self.dense_fusions_by_gate = {}
        self.crop_attempts_by_gate = {}
        self.crop_fusions_by_gate = {}
        self.async_inference_ms = 0.0
        self._async_last_dispatched = None
        with self._crop_lock:
            self._crop_job = None
            self._crop_result = None
        self._crop_last_dispatched = None
        self._crop_last_dispatch_wall = -np.inf
        self.last_gyro = np.asarray(latest_imu[4:7], float)
        spread = np.linalg.norm(
            translations_selected - np.median(
                translations_selected, axis=0
            ),
            axis=1,
        )
        self.anchor_diagnostics = {
            "frames": int(len(translations_selected)),
            "frames_raw": int(len(translations_all)),
            "sources": {
                source: sources_selected.count(source)
                for source in sorted(set(sources_selected))
            },
            "pnp_rms_px": float(np.median(rms_selected)),
            "translation_spread_p90_m": float(np.percentile(spread, 90)),
            "pitch_deg": float(np.degrees(pitch)),
            "roll_deg": float(np.degrees(roll)),
            "gate0": observed_gate0.tolist(),
            "gate0_visual": observed_gate0_visual.tolist(),
            "gate0_visual_error_m": float(np.linalg.norm(
                observed_gate0_visual - observed_gate0
            )),
            "fixed_spawn_gate0": self.fixed_spawn_gate0 is not None,
            "map_yaw_delta_deg": float(np.degrees(yaw_delta)),
        }
        return self.state()

    def _propagate_imu(self) -> None:
        if self.ekf is None:
            return
        if not self.mavlink.imu:
            return
        latest = self.mavlink.imu[-1]
        if self.last_imu_us is not None and latest[0] < self.last_imu_us:
            raise RuntimeError("sim clock reset while VQ2 localizer was active")
        # Walk backward only over the unseen tail.  Converting the entire
        # multi-hour telemetry deque on every camera frame eventually makes
        # a long SAC run quadratic in elapsed time.
        unseen_reversed = []
        for row in reversed(self.mavlink.imu):
            if self.last_imu_us is not None and row[0] <= self.last_imu_us:
                break
            unseen_reversed.append(row)
        for row in reversed(unseen_reversed):
            time_us = int(row[0])
            self.ekf.propagate(
                time_us * 1e-6,
                np.asarray(row[1:4], float),
                np.asarray(row[4:7], float),
            )
            self.last_imu_us = time_us
            self.last_gyro = np.asarray(row[4:7], float)

    def _regular_corner_update(
        self,
        image: np.ndarray,
        active_gate: int,
        peaks=None,
    ) -> int:
        assert self.ekf is not None
        self.last_direct_pin_applied = False
        if peaks is None:
            peaks = self._corner_peaks(image, active_gate)
        position_sigma = float(np.sqrt(max(
            np.trace(self.ekf.P[0:3, 0:3]), 0.0
        )))
        radius = float(np.clip(
            3.0 * self.K[0, 0] * position_sigma / 6.0 + 20.0,
            25.0,
            120.0,
        ))
        observations = []
        debug_expected = []
        debug_matches = []
        gate_hypotheses = []
        # opt-in course-wide association (AIGP_MULTIGATE=1): every
        # plausibly visible gate, global one-to-one peak assignment, and
        # (with AIGP_MULTIGATE_ATT=1) attitude correction when at least
        # two well-separated gates are accepted. Default path unchanged.
        multigate_on = os.environ.get("AIGP_MULTIGATE") == "1"
        # Some launch/early-course gates are already exceptionally stable
        # with the conservative active/adjacent-gate association, while
        # later close-range droughts benefit from the course-wide solver.
        # Keep AIGP_MULTIGATE=1 backward compatible, but allow a live harness
        # to restrict it to selected active gates (for example ``3-16``).
        multigate_gate_spec = os.environ.get(
            "AIGP_MULTIGATE_GATES", ""
        ).strip()
        if multigate_on and multigate_gate_spec:
            enabled_gates: set[int] = set()
            for chunk in multigate_gate_spec.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if "-" in chunk:
                    lo_text, hi_text = chunk.split("-", 1)
                    lo, hi = int(lo_text), int(hi_text)
                    if hi < lo:
                        lo, hi = hi, lo
                    enabled_gates.update(range(lo, hi + 1))
                else:
                    enabled_gates.add(int(chunk))
            multigate_on = int(active_gate) in enabled_gates
        multigate_constellation = None
        update_attitude = False
        if multigate_on:
            from aigp.vq2_multigate import associate_multigate
            far_deweight = os.environ.get("AIGP_FAR_DEWEIGHT") == "1"
            gyro_hot = False
            if far_deweight and self.mavlink.imu:
                gyro_hot = float(np.linalg.norm(
                    np.asarray(self.mavlink.imu[-1][4:7], float)
                )) > 1.5
            observations, debug_matches, multigate_constellation = \
                associate_multigate(
                    self.ekf, self.gate_world, peaks, radius,
                    exclude_far=far_deweight and gyro_hot,
                )
            # attitude is ON by default when the constellation supports
            # it: replay bench showed position-only multigate diverges
            # (1.8m/21m on the ep32 benchmark) because a refused
            # attitude correction gets squeezed into position until an
            # association slips. AIGP_MULTIGATE_POSONLY=1 exists for
            # experiments only.
            update_attitude = bool(
                multigate_constellation["attitude_ok"]
                and os.environ.get("AIGP_MULTIGATE_POSONLY") != "1"
            )
            for gate_index in multigate_constellation["visible_gates"]:
                for physical, world_corner in enumerate(
                    self.gate_world[gate_index]
                ):
                    projected, _ = self.ekf.predict_pixel(world_corner)
                    if projected is not None:
                        debug_expected.append({
                            "gate": int(gate_index),
                            "corner": int(physical),
                            "pixel": [
                                float(projected[0]),
                                float(projected[1]),
                            ],
                        })
        for gate_index in () if multigate_on else (
            active_gate - 1, active_gate, active_gate + 1
        ):
            if not 0 <= gate_index < min(17, len(self.gates)):
                continue
            for physical, world_corner in enumerate(
                self.gate_world[gate_index]
            ):
                projected, _ = self.ekf.predict_pixel(world_corner)
                if projected is not None:
                    debug_expected.append({
                        "gate": int(gate_index),
                        "corner": int(physical),
                        "pixel": [
                            float(projected[0]),
                            float(projected[1]),
                        ],
                    })
            hypotheses = []
            for flipped in (False, True):
                rows = []
                matches = []
                cost = 0.0
                for corner_class in range(8):
                    physical = FLIP[corner_class] if flipped else corner_class
                    world_corner = self.gate_world[gate_index][physical]
                    projected, _ = self.ekf.predict_pixel(world_corner)
                    if projected is None:
                        continue
                    best = min(
                        (
                            (
                                float(np.hypot(
                                    point[0] - projected[0],
                                    point[1] - projected[1],
                                )),
                                point,
                            )
                            for point in peaks[corner_class]
                        ),
                        default=None,
                        key=lambda row: row[0],
                    )
                    if best is not None and best[0] < radius:
                        rows.append((
                            world_corner,
                            np.asarray(best[1][:2], float),
                        ))
                        matches.append({
                            "gate": int(gate_index),
                            "class": int(corner_class),
                            "physical": int(physical),
                            "flipped": bool(flipped),
                            "predicted": [
                                float(projected[0]),
                                float(projected[1]),
                            ],
                            "observed": [
                                float(best[1][0]),
                                float(best[1][1]),
                            ],
                            "distance_px": float(best[0]),
                            "score": float(best[1][2]),
                        })
                        cost += best[0]
                hypotheses.append((len(rows), cost, rows, matches))
            count, _cost, rows, matches = min(
                hypotheses, key=lambda row: (-row[0], row[1])
            )
            if count >= 2:
                gate_hypotheses.append({
                    "gate": int(gate_index),
                    "count": int(count),
                    "cost": float(_cost),
                    "rows": rows,
                    "matches": matches,
                })

        # Association proximity alone is not enough to override the EKF.
        # Visually plausible four-corner sets can still reuse peaks from the
        # previous/next gate. Preserve the proven conservative innovation
        # gate; the dashboard separately marks which associations survived it.
        # racing-speed fix (opt-in via AIGP_FAR_DEWEIGHT=1): far-gate
        # corners observed during fast rotation carry attitude-lag errors
        # of meters at 25-45m range (the bias behind the false
        # spawn-level back half in the map campaign). Drop far gates'
        # rows while the gyro is hot; close-range fusion is unaffected.
        far_deweight = (not multigate_on) and \
            os.environ.get("AIGP_FAR_DEWEIGHT") == "1"
        gyro_hot = False
        if far_deweight and self.mavlink.imu:
            gyro_hot = float(np.linalg.norm(
                np.asarray(self.mavlink.imu[-1][4:7], float)
            )) > 1.5
        for hypothesis in gate_hypotheses:
            if far_deweight and gyro_hot:
                gate_center = np.mean(
                    self.gate_world[hypothesis["gate"]], axis=0
                ) if isinstance(hypothesis.get("gate"), int) else None
                if gate_center is not None:
                    dist = float(np.linalg.norm(
                        gate_center - self.ekf.x[0:3]
                    ))
                    if dist > 18.0:
                        continue
            observations.extend(hypothesis["rows"])
            debug_matches.extend(hypothesis["matches"])
        chi2_gate = 9.0
        sigma_px = 1.5
        visual_consensus = "conservative"

        fused, accepted_indices = self.ekf.update_corners(
            observations,
            chi2_gate=chi2_gate,
            update_attitude=update_attitude,
            sigma_px=sigma_px,
            return_indices=True,
        )
        accepted_set = set(accepted_indices)
        for match_index, match in enumerate(debug_matches):
            match["fused"] = match_index in accepted_set
        direct_pin = (
            self._direct_active_gate_pin(
                debug_matches,
                accepted_set,
                int(active_gate),
            )
            if self.direct_position_pins
            else None
        )
        peak_rows = []
        for corner_class, class_peaks in enumerate(peaks):
            for u, v, score in class_peaks[:8]:
                peak_rows.append({
                    "class": int(corner_class),
                    "pixel": [float(u), float(v)],
                    "score": float(score),
                })
        with self._debug_lock:
            self._debug_image = image.copy()
            self._debug_payload = {
                "active_gate": int(active_gate),
                "association_radius_px": radius,
                "peaks": peak_rows,
                "expected": debug_expected,
                "matches": debug_matches,
                "fused": int(fused),
                "visual_consensus": visual_consensus,
                "measurement_sigma_px": float(sigma_px),
                "chi2_gate": float(chi2_gate),
                "direct_pin": direct_pin,
                "multigate": multigate_constellation,
                "attitude_updated": bool(update_attitude),
            }
        return fused

    def _direct_active_gate_pin(
        self,
        matches: list[dict],
        accepted_indices: set[int],
        active_gate: int,
    ) -> dict | None:
        """Use a complete green inner-corner match as a strong position pin.

        The regular corner EKF update is intentionally conservative and can
        retain tens of centimetres of stale course-frame error even while all
        four active-gate corners are correctly associated. Near an aperture,
        that residual is the difference between a clean crossing and a frame
        strike. A complete accepted inner quad gives a direct PnP translation,
        so use it strongly while retaining gyro attitude and EKF velocity.
        """
        assert self.ekf is not None
        if not 0 <= active_gate < min(17, len(self.gates)):
            return None
        by_physical: dict[int, dict] = {}
        for index, match in enumerate(matches):
            physical = int(match["physical"])
            if (
                index not in accepted_indices
                or int(match["gate"]) != active_gate
                or not 0 <= physical < 4
            ):
                continue
            previous = by_physical.get(physical)
            if previous is None or float(match["score"]) > float(
                previous["score"]
            ):
                by_physical[physical] = match
        if len(by_physical) != 4:
            return None
        points = np.asarray([
            by_physical[index]["observed"] for index in range(4)
        ], np.float32)
        camera_to_world = self.ekf.q.as_matrix() @ self.R_cb.T
        candidates = []
        gate_position = np.asarray(
            self.gates[active_gate]["pos"], float
        )
        for translation, rms in _pnp_apparent_inner(points, self.K):
            depth = float(np.linalg.norm(translation))
            if rms > 2.5 or not 1.0 <= depth <= 15.0:
                continue
            position = gate_position - camera_to_world @ translation
            jump = float(np.linalg.norm(position - self.ekf.p))
            candidates.append((jump, rms, depth, position))
        if not candidates:
            return None
        jump, rms, depth, position = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        if jump > 1.50:
            return {
                "accepted": False,
                "reason": "jump",
                "jump_m": jump,
                "rms_px": rms,
                "depth_m": depth,
            }
        blend = float(np.interp(depth, [1.0, 15.0], [0.90, 0.65]))
        self.ekf.p = (1.0 - blend) * self.ekf.p + blend * position
        self.ekf.P[0:3, :] *= 0.5
        self.ekf.P[:, 0:3] *= 0.5
        self.ekf.P[0:3, 0:3] += np.eye(3) * 0.05**2
        self.last_direct_pin = (
            time.time(), active_gate, position.copy()
        )
        self.last_direct_pin_applied = True
        return {
            "accepted": True,
            "jump_m": jump,
            "rms_px": rms,
            "depth_m": depth,
            "blend": blend,
            "position": position.tolist(),
        }

    def _direct_crop_position_pin(
        self,
        observations: list[tuple[np.ndarray, np.ndarray]],
        active_gate: int,
    ) -> dict | None:
        """Apply the dense quad PnP pin recipe to a complete crop quad.

        CropGateNet observations already carry the matched world corner, so
        recover the physical inner-corner index and reuse the same guarded PnP
        implementation as the dense path.  Partial three-corner crops remain
        conservative EKF updates.
        """
        if not 0 <= active_gate < min(17, len(self.gate_world)):
            return None
        inner = np.asarray(self.gate_world[active_gate][:4], float)
        matches = []
        used: set[int] = set()
        for world_corner, pixel in observations:
            distances = np.linalg.norm(
                inner - np.asarray(world_corner, float), axis=1
            )
            physical = int(np.argmin(distances))
            if float(distances[physical]) > 1e-4 or physical in used:
                continue
            used.add(physical)
            matches.append({
                "gate": int(active_gate),
                "physical": physical,
                "observed": np.asarray(pixel, float),
                "score": 1.0,
            })
        if len(matches) != 4:
            return None
        return self._direct_active_gate_pin(
            matches, set(range(4)), int(active_gate)
        )

    def dashboard_snapshot(self) -> tuple[np.ndarray | None, dict]:
        """Return a coherent copy of the latest frame-association debug data."""
        with self._debug_lock:
            image = (
                None if self._debug_image is None
                else self._debug_image.copy()
            )
            payload = copy.deepcopy(self._debug_payload)
        return image, payload

    def _net_relocalize(self, peaks, active_gate: int) -> bool:
        """PnP-relocalize from 6+ classified peaks after a vision gap."""
        assert self.ekf is not None
        gap = float(time.time() - self.last_visual_wall)
        if gap <= 0.50:
            return False
        strongest = {
            corner_class: max(
                peaks[corner_class], key=lambda point: point[2]
            )[:2]
            for corner_class in range(8)
            if peaks[corner_class]
        }
        indices = sorted(strongest)
        branches = _pnp_points_all(
            indices,
            [strongest[index] for index in indices],
            self.K,
        )
        if not branches:
            return False
        body_rotation = self.ekf.q.as_matrix()
        camera_to_world = body_rotation @ self.R_cb.T
        candidates = []
        for gate_index in (
            active_gate, active_gate + 1, active_gate - 1
        ):
            if not 0 <= gate_index < min(17, len(self.gates)):
                continue
            qw, qx, qy, qz = self.gates[gate_index]["quat_wxyz"]
            gate_rotation = Rotation.from_quat(
                [qx, qy, qz, qw]
            ).as_matrix()
            for gate_to_camera, translation, rms in branches:
                if rms > 3.0:
                    continue
                implied_body_rotation = (
                    gate_rotation @ gate_to_camera.T @ self.R_cb
                )
                cosine = (
                    np.trace(
                        implied_body_rotation.T @ body_rotation
                    ) - 1.0
                ) / 2.0
                angle = float(np.degrees(np.arccos(np.clip(
                    cosine, -1.0, 1.0
                ))))
                if angle > 15.0:
                    continue
                position = (
                    np.asarray(self.gates[gate_index]["pos"], float)
                    - camera_to_world @ translation
                )
                jump = float(np.linalg.norm(position - self.ekf.p))
                candidates.append((
                    jump, angle, rms, gate_index, position,
                ))
        if not candidates:
            return False
        plausible = [
            row for row in candidates
            if row[0] < 1.0 + 0.6 * gap
        ]
        if not plausible and gap <= 3.0:
            return False
        jump, _angle, _rms, gate_index, position = min(
            plausible or candidates,
            key=lambda row: (row[1], row[0], row[2]),
        )
        velocity = self.ekf.v.copy()
        stamp = float(self.ekf.t)
        if self.last_relocalization is not None:
            previous_stamp, previous_gate, previous_position = (
                self.last_relocalization
            )
            delta = stamp - previous_stamp
            if previous_gate == gate_index and 0.05 < delta < 0.7:
                measured_velocity = (position - previous_position) / delta
                if np.linalg.norm(measured_velocity) < 25.0:
                    velocity = measured_velocity
        self.last_relocalization = (
            stamp, gate_index, position.copy()
        )
        quaternion = self.ekf.q.as_quat()
        self.ekf.init_state(
            position,
            velocity,
            [quaternion[3], quaternion[0], quaternion[1], quaternion[2]],
            stamp,
            pos_std=0.4,
            vel_std=0.8,
            ang_std=0.03,
        )
        return True

    def _crop_track_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            with self._crop_lock:
                job = self._crop_job
                if job is not None:
                    self._crop_job = None
                    self._crop_busy = True
            if job is None:
                time.sleep(0.002)
                continue
            started = time.perf_counter()
            try:
                job["obs"] = self._crop_track_infer(job)
            except Exception:
                job["obs"] = None
            job["crop_ms"] = (time.perf_counter() - started) * 1000.0
            with self._crop_lock:
                self._crop_result = job
                self._crop_busy = False

    def _crop_track_infer(
        self, job: dict
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """CropGateNet corners for the active gate, placed from the EKF.

        Unlike _v11_position_pin this needs no YOLO proposal: the snapshot
        EKF projects the gate's known world corners, and the crop window is
        placed around that prediction.  Valid exactly when landmark age is
        small enough that the projection lands near the true gate -- the
        regime the dispatcher enforces (crop_track_max_age_s).
        """
        ekf = job["ekf"]
        gate_index = int(job["active_gate"])
        if not 0 <= gate_index < min(17, len(self.gate_world)):
            return None
        expected = []
        for corner in self.gate_world[gate_index]:
            pixel, _ = ekf.predict_pixel(corner)
            if pixel is None:
                return None
            expected.append(pixel)
        expected = np.asarray(expected, np.float32)
        span = float(np.ptp(expected, axis=0).max())
        if span < self.crop_track_min_span_px:
            return None
        inner_expected = expected[:4]
        center = inner_expected.mean(axis=0)
        image = self._decode_jpeg(job["frame"])
        if image is None:
            return None
        height, width = image.shape[:2]
        if not (
            -0.15 * width <= center[0] <= 1.15 * width
            and -0.15 * height <= center[1] <= 1.15 * height
        ):
            return None
        side = max(18.0, span * self.crop_padding)
        crop, _forward, inverse = warp_gate_crop(
            image, center, side, self.crop_size
        )
        orange = crop_orange_channel(crop)
        network_input = np.concatenate([
            crop.astype(np.float32) / 255.0,
            orange[..., None],
            self.crop_prior[..., None],
        ], axis=2).transpose(2, 0, 1)
        tensor = torch.from_numpy(network_input).unsqueeze(0).to(
            self.device
        )
        with torch.no_grad(), torch.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            output = self.crop(tensor)
        decoded = decode_crop_corners(
            output, self.crop_size, inverse_affine=inverse
        )
        points = np.asarray(decoded["corners"], np.float32)[:4]
        accepted = (
            (np.asarray(decoded["scores"])[:4] >= 0.10)
            & (np.asarray(decoded["visibility"])[:4] >= 0.25)
            & (float(decoded["presence"]) >= 0.25)
        )
        if accepted.sum() < 3:
            return None
        if np.linalg.norm(points.mean(axis=0) - center) > \
                max(15.0, 0.35 * span):
            return None
        # D4 alignment against the EKF projection resolves the crop net's
        # corner-class ordering and guards against locking a wrong gate.
        best_error = np.inf
        best_mapping = None
        base = np.arange(4)
        for shift in range(4):
            for mapping in (
                np.roll(base, shift),
                np.roll(base[::-1], shift),
            ):
                error = float(np.mean(np.linalg.norm(
                    points - inner_expected[mapping], axis=1
                )))
                if error < best_error:
                    best_error = error
                    best_mapping = mapping
        if best_error > max(12.0, 0.25 * span):
            return None
        observations = []
        for point_index in np.flatnonzero(accepted):
            world_corner = self.gate_world[gate_index][
                int(best_mapping[point_index])
            ]
            observations.append((
                np.asarray(world_corner, float),
                points[point_index].astype(float),
            ))
        return observations

    def _v11_position_pin(
        self, image: np.ndarray, active_gate: int
    ) -> bool:
        assert self.ekf is not None
        now = time.time()
        # A single cornerless frame is normal under blur.  CPU YOLO can take
        # hundreds of milliseconds and must not block the control loop unless
        # dense V7/V10 has genuinely been absent across several frames.
        if now - self.last_visual_wall < 0.18:
            return False
        if now - self.last_crop_wall < self.crop_interval_s:
            return False
        self.last_crop_wall = now
        result = self.proposal.predict(
            source=image,
            conf=0.05,
            iou=0.5,
            imgsz=640,
            device=self.proposal_device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return False
        boxes = result.boxes.xyxy.detach().cpu().numpy()

        predicted = {}
        for gate_index in (active_gate - 1, active_gate, active_gate + 1):
            if not 0 <= gate_index < min(17, len(self.gates)):
                continue
            points = []
            for corner in self.gate_world[gate_index][:4]:
                pixel, _ = self.ekf.predict_pixel(corner)
                if pixel is None:
                    points = []
                    break
                points.append(pixel)
            if points:
                points = np.asarray(points)
                predicted[gate_index] = (
                    points,
                    points.mean(axis=0),
                    float(np.ptp(points, axis=0).max()),
                )
        if not predicted:
            return False
        assignments = []
        for box_index, box in enumerate(boxes):
            center = 0.5 * (box[:2] + box[2:])
            gate_index, distance = min(
                (
                    (gate_index, float(np.linalg.norm(center - values[1])))
                    for gate_index, values in predicted.items()
                ),
                key=lambda row: row[1],
            )
            if distance <= max(40.0, predicted[gate_index][2]):
                assignments.append((distance, box_index, box, center, gate_index))
        if not assignments:
            return False

        candidates = []
        for _, _box_index, box, center, gate_index in sorted(assignments):
            expected, expected_center, expected_span = predicted[gate_index]
            proposal_span = float(max(box[2] - box[0], box[3] - box[1]))
            side = max(18.0, proposal_span * self.crop_padding)
            crop, _forward, inverse = warp_gate_crop(
                image, center, side, self.crop_size
            )
            orange = crop_orange_channel(crop)
            network_input = np.concatenate([
                crop.astype(np.float32) / 255.0,
                orange[..., None],
                self.crop_prior[..., None],
            ], axis=2).transpose(2, 0, 1)
            tensor = torch.from_numpy(network_input).unsqueeze(0).to(
                self.device
            )
            with torch.no_grad(), torch.autocast(
                "cuda",
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                output = self.crop(tensor)
            decoded = decode_crop_corners(
                output, self.crop_size, inverse_affine=inverse
            )
            points = np.asarray(decoded["corners"], np.float32)[:4]
            accepted = (
                (np.asarray(decoded["scores"])[:4] >= 0.05)
                & (np.asarray(decoded["visibility"])[:4] >= 0.20)
                & (float(decoded["presence"]) >= 0.20)
            )
            if not accepted.all():
                continue
            if np.linalg.norm(points.mean(axis=0) - expected_center) > \
                    max(15.0, 0.35 * expected_span):
                continue
            # D4 agreement protects against a wrong visible gate/rotation.
            best_error = np.inf
            base = np.arange(4)
            for shift in range(4):
                for mapping in (
                    np.roll(base, shift),
                    np.roll(base[::-1], shift),
                ):
                    error = float(np.mean(np.linalg.norm(
                        points - expected[mapping], axis=1
                    )))
                    best_error = min(best_error, error)
            if best_error > max(16.0, 0.25 * expected_span):
                continue
            for translation, rms in _pnp_apparent_inner(points, self.K):
                if rms > 3.0:
                    continue
                camera_to_world = (
                    self.ekf.q.as_matrix() @ self.R_cb.T
                )
                position = (
                    np.asarray(self.gates[gate_index]["pos"], float)
                    - camera_to_world @ translation
                )
                jump = float(np.linalg.norm(position - self.ekf.p))
                candidates.append((
                    jump, rms, gate_index, position,
                    float(np.linalg.norm(translation)),
                ))
        if not candidates or now - self.last_pin_wall < 0.25:
            return False
        jump, _rms, _gate_index, measurement, depth = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        current_sigma = float(np.sqrt(max(
            np.trace(self.ekf.P[0:3, 0:3]), 0.0
        )))
        maximum_jump = max(0.60, min(1.50, 0.35 + 2.5 * current_sigma))
        if jump > maximum_jump:
            return False
        measurement_sigma = float(np.clip(
            0.12 + 0.020 * depth, 0.15, 0.45
        ))
        H = np.zeros((3, 9), float)
        H[:, :3] = np.eye(3)
        Rm = np.eye(3) * measurement_sigma**2
        S = H @ self.ekf.P @ H.T + Rm
        gain = self.ekf.P @ H.T @ np.linalg.inv(S)
        gain[6:9, :] = 0.0
        correction = gain @ (measurement - self.ekf.p)
        self.ekf.p += correction[:3]
        self.ekf.v += correction[3:6]
        identity = np.eye(9)
        residual = identity - gain @ H
        self.ekf.P = residual @ self.ekf.P @ residual.T + \
            gain @ Rm @ gain.T
        self.ekf.P = 0.5 * (self.ekf.P + self.ekf.P.T)
        self.last_pin_wall = now
        return True

    def _generic_classical_pin(
        self, image: np.ndarray, active_gate: int
    ) -> bool:
        """Recover translation from a complete classical gate detection.

        This reproduces the clean offline stack's gyro-attitude recovery:
        gate identity comes from official race status, PnP contributes only
        camera-to-gate translation, and continuity rejects the previous or
        following gate.  It is deliberately available only after dense
        V7/V10 and crop V11 both failed.
        """
        assert self.ekf is not None
        if not 0 <= active_gate < min(17, len(self.gates)):
            return False
        gap = float(time.time() - self.last_visual_wall)
        if gap <= 0.15:
            return False
        detections = [
            row for row in detect_gates(image, min_area=250)
            if row["inner"] is not None
        ]
        if not detections:
            return False
        camera_to_world = self.ekf.q.as_matrix() @ self.R_cb.T
        candidates = []
        for detection in detections:
            solution = _pnp_gate(detection, self.K)
            if solution is None or solution[1] > 3.0:
                continue
            position = (
                np.asarray(self.gates[active_gate]["pos"], float)
                - camera_to_world @ solution[0]
            )
            candidates.append((
                float(np.linalg.norm(position - self.ekf.p)),
                solution[1],
                position,
            ))
        if not candidates:
            return False
        jump, _rms, position = min(
            candidates, key=lambda row: (row[0], row[1])
        )
        if jump >= min(4.0, 1.5 + 0.8 * gap):
            return False
        velocity = self.ekf.v.copy()
        stamp = float(self.ekf.t)
        if self.last_generic_pin is not None:
            previous_stamp, previous_gate, previous_position = (
                self.last_generic_pin
            )
            delta = stamp - previous_stamp
            if previous_gate == active_gate and 0.06 < delta < 0.7:
                measured_velocity = (position - previous_position) / delta
                if np.linalg.norm(measured_velocity) < 25.0:
                    velocity = 0.7 * velocity + 0.3 * measured_velocity
        self.last_generic_pin = (stamp, active_gate, position.copy())
        quaternion = self.ekf.q.as_quat()
        self.ekf.init_state(
            0.2 * self.ekf.p + 0.8 * position,
            velocity,
            [quaternion[3], quaternion[0], quaternion[1], quaternion[2]],
            stamp,
            pos_std=0.25,
            vel_std=0.7,
            ang_std=0.02,
        )
        return True

    def update(self, active_gate: int) -> LiveLocalizerState:
        if self.ekf is None:
            raise RuntimeError("live VQ2 localizer is not initialized")
        self._propagate_imu()
        frame = self.vision.latest
        fused = 0
        if frame is not None and frame[0] != self.last_frame_id:
            self.last_frame_id = frame[0]
            image = self._decode_jpeg(frame)
            if image is not None:
                peaks = self._corner_peaks(image, int(active_gate))
                fused = self._regular_corner_update(
                    image, int(active_gate), peaks=peaks
                )
                source = (
                    "v7_v10_direct"
                    if fused and self.last_direct_pin_applied
                    else "v7_v10" if fused
                    else "none"
                )
                if not fused and self._v11_position_pin(
                    image, int(active_gate)
                ):
                    fused = 4
                    source = "v11"
                if not fused and self._generic_classical_pin(
                    image, int(active_gate)
                ):
                    fused = 4
                    source = "classical_pin"
                if not fused and self._net_relocalize(
                    peaks, int(active_gate)
                ):
                    fused = 4
                    source = "net_relocalize"
                if fused:
                    self.last_visual_wall = time.time()
                self.last_update_source = source
                self.update_counts[source] += 1
        self.last_fused = fused
        return self.state()

    def _async_inference_loop(
        self, stop_event: threading.Event
    ) -> None:
        while not stop_event.is_set():
            with self._async_lock:
                job = self._async_job
                if job is not None:
                    self._async_job = None
                    self._async_busy = True
            if job is None:
                time.sleep(0.001)
                continue
            started = time.perf_counter()
            peaks = self._corner_peaks(
                job["image"], int(job["active_gate"])
            )
            inference_ms = (time.perf_counter() - started) * 1000.0
            if stop_event.is_set():
                return
            job["peaks"] = peaks
            job["inference_ms"] = inference_ms
            with self._async_lock:
                self._async_result = job
                self._async_busy = False

    def _ensure_dense_process(self) -> None:
        if not self.dense_process_isolation:
            return
        if self._dense_process is not None:
            if self._dense_process.is_alive() and self._dense_ready:
                return
            self._dense_process.join(timeout=0.1)
        self._dense_context = mp.get_context("spawn")
        self._dense_request_queue = self._dense_context.Queue(maxsize=1)
        self._dense_result_queue = self._dense_context.Queue(maxsize=1)
        self._dense_process = self._dense_context.Process(
            target=_dense_worker_main,
            args=(
                self._dense_request_queue,
                self._dense_result_queue,
                self.primary_checkpoint,
                self.refine_checkpoint,
                self.gate_primary_checkpoint,
                tuple(sorted(self.gate_primary_gates)),
                self.threshold,
                self.refine_radius_px,
                str(self.dense_device),
                self.dense_worker_threads,
                self.dense_worker_affinity,
            ),
            name="vq2-dense-gatenet",
            daemon=True,
        )
        self._dense_process.start()
        ready_deadline = time.time() + 30.0
        while time.time() < ready_deadline:
            if not self._dense_process.is_alive():
                raise RuntimeError(
                    "process-isolated GateNet exited during startup"
                )
            try:
                startup = self._dense_result_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if "worker_error" in startup:
                raise RuntimeError(
                    "process-isolated GateNet failed during startup: "
                    f"{startup['worker_error']}"
                )
            if startup.get("worker_ready"):
                self._dense_ready = True
                break
        if not self._dense_ready:
            self._dense_process.terminate()
            self._dense_process.join(timeout=1.0)
            raise RuntimeError(
                "process-isolated GateNet did not become ready in 30 seconds"
            )
        self._async_busy = False
        self._async_pending.clear()

    def _poll_dense_process_result(self):
        if not self.dense_process_isolation:
            return None
        self._ensure_dense_process()
        try:
            worker_result = self._dense_result_queue.get_nowait()
        except queue.Empty:
            return None
        self._async_busy = False
        if "worker_error" in worker_result:
            raise RuntimeError(
                "process-isolated GateNet failed: "
                f"{worker_result['worker_error']}"
            )
        key = (
            int(worker_result["generation"]),
            int(worker_result["frame_id"]),
        )
        job = self._async_pending.pop(key, None)
        if (
            job is None
            or key[0] != self._async_generation
            or not self._async_running
        ):
            return None
        job["peaks"] = worker_result["peaks"]
        job["inference_ms"] = float(worker_result["inference_ms"])
        job["worker_pid"] = int(worker_result["worker_pid"])
        job["gate_primary_used"] = bool(
            worker_result.get("gate_primary_used", False)
        )
        job["gate_primary_ensemble_used"] = bool(
            worker_result.get("gate_primary_ensemble_used", False)
        )
        return job

    def start_async(self) -> None:
        """Start non-blocking V7/V10 inference for live control."""
        self.stop_async()
        self._async_running = True
        self._async_job = None
        self._async_result = None
        self._async_generation += 1
        self._async_last_dispatched = None
        self._async_last_dispatch_wall = -np.inf
        if self.crop_track_enabled:
            self._crop_stop_event = threading.Event()
            self._crop_thread = threading.Thread(
                target=self._crop_track_loop,
                args=(self._crop_stop_event,),
                name="vq2-crop-tracker",
                daemon=True,
            )
            self._crop_thread.start()
        if self.dense_process_isolation:
            self._ensure_dense_process()
            return
        self._async_busy = False
        self._async_stop_event = threading.Event()
        self._async_thread = threading.Thread(
            target=self._async_inference_loop,
            args=(self._async_stop_event,),
            daemon=True,
        )
        self._async_thread.start()

    def stop_async(self) -> None:
        self._async_running = False
        if self._crop_stop_event is not None:
            self._crop_stop_event.set()
        if self._crop_thread is not None:
            self._crop_thread.join(timeout=3.0)
        self._crop_thread = None
        self._crop_stop_event = None
        with self._crop_lock:
            self._crop_job = None
            self._crop_result = None
            self._crop_busy = False
        if self.dense_process_isolation:
            return
        if self._async_stop_event is not None:
            self._async_stop_event.set()
        if self._async_thread is not None:
            self._async_thread.join(timeout=3.0)
            if self._async_thread.is_alive():
                raise RuntimeError(
                    "GateNet worker did not stop within 3 seconds"
                )
        self._async_thread = None
        self._async_stop_event = None
        with self._async_lock:
            self._async_job = None
            self._async_result = None
            self._async_busy = False

    def close(self) -> None:
        """Release the persistent process-isolated dense-vision worker."""
        self.stop_async()
        if self._dense_process is None:
            return
        if self._dense_process.is_alive():
            try:
                self._dense_request_queue.put(None, timeout=0.25)
            except (queue.Full, OSError, ValueError):
                self._dense_process.terminate()
        self._dense_process.join(timeout=3.0)
        if self._dense_process.is_alive():
            self._dense_process.terminate()
            self._dense_process.join(timeout=1.0)
        for worker_queue in (
            self._dense_request_queue,
            self._dense_result_queue,
        ):
            if worker_queue is not None:
                worker_queue.cancel_join_thread()
                worker_queue.close()
        self._dense_process = None
        self._dense_request_queue = None
        self._dense_result_queue = None
        self._async_busy = False
        self._async_pending.clear()
        self._dense_ready = False

    def update_async(self, active_gate: int) -> LiveLocalizerState:
        """Propagate IMU immediately; fuse completed vision out-of-sequence.

        Each inference job owns a snapshot of the EKF from dispatch.  When its
        corner peaks return, that snapshot receives the visual update and all
        buffered IMU after the snapshot is replayed to the present.  A GPU
        rendering/inference stall therefore cannot hold a control command.
        """
        if self.ekf is None:
            raise RuntimeError("live VQ2 localizer is not initialized")
        self._propagate_imu()
        if self.dense_process_isolation:
            result = self._poll_dense_process_result()
        else:
            with self._async_lock:
                result = self._async_result
                if result is not None:
                    self._async_result = None
        dropped_stale = False
        if result is not None:
            self.async_inference_ms = float(result["inference_ms"])
            # Rate-limit from completion/consumption rather than dispatch.
            # A long-running GPU job must not immediately launch another.
            self._async_last_dispatch_wall = time.monotonic()
            result_age_s = (
                time.time_ns() - int(result["frame_wall_ns"])
            ) * 1e-9
            if result_age_s > self.max_async_result_age_s:
                dropped_stale = True
                result = None
                self.last_update_source = "stale_vision_drop"
                self.update_counts["stale_vision_drop"] += 1
                with self._debug_lock:
                    self._debug_payload.update({
                        "source": "stale_vision_drop",
                        "fused": 0,
                        "result_age_s": float(result_age_s),
                        "inference_ms": float(self.async_inference_ms),
                    })
        if (
            result is not None
            and int(result["active_gate"]) != int(active_gate)
        ):
            # The official gate event is authoritative.  A vision result
            # computed for the gate that was just passed must not replace
            # the current EKF with its older snapshot.
            result = None
            self.last_update_source = "dense_gate_mismatch"
            self.update_counts["dense_gate_mismatch"] += 1
        fused = 0
        if result is not None:
            result_gate = int(result["active_gate"])
            self.dense_attempts_by_gate[result_gate] = (
                self.dense_attempts_by_gate.get(result_gate, 0) + 1
            )
            self.ekf = result["ekf"]
            self.last_imu_us = result["last_imu_us"]
            self.last_gyro = result["last_gyro"]
            fused = self._regular_corner_update(
                result["image"],
                int(result["active_gate"]),
                peaks=result["peaks"],
            )
            source = (
                "gate_ensemble_v10_direct"
                if fused
                and result.get("gate_primary_ensemble_used", False)
                and self.last_direct_pin_applied
                else "gate_ensemble_v10"
                if fused and result.get(
                    "gate_primary_ensemble_used", False
                )
                else "gate_primary_v10_direct"
                if fused
                and result.get("gate_primary_used", False)
                and self.last_direct_pin_applied
                else "gate_primary_v10"
                if fused and result.get("gate_primary_used", False)
                else "v7_v10_direct"
                if fused and self.last_direct_pin_applied
                else "v7_v10" if fused
                else "none"
            )
            # Cheap geometric recoveries are safe here.  Crop V11/YOLO stays
            # off the control path; it remains the startup anchor and can be
            # moved to its own delayed worker later.
            if not fused and self._generic_classical_pin(
                result["image"], int(result["active_gate"])
            ):
                fused = 4
                source = "classical_pin"
            if not fused and self._net_relocalize(
                result["peaks"], int(result["active_gate"])
            ):
                fused = 4
                source = "net_relocalize"
            self._propagate_imu()
            self.last_frame_id = int(result["frame_id"])
            self.async_inference_ms = float(result["inference_ms"])
            self.last_update_source = source
            self.update_counts[source] += 1
            if fused:
                self.dense_fusions_by_gate[result_gate] = (
                    self.dense_fusions_by_gate.get(result_gate, 0) + 1
                )
                self.last_visual_wall = float(result["frame_wall_ns"]) * 1e-9
            state_after = self.state()
            with self._debug_lock:
                self._debug_payload.update({
                    "frame_id": int(result["frame_id"]),
                    "source": source,
                    "fused": int(fused),
                    "position": state_after.position.tolist(),
                    "velocity": state_after.velocity.tolist(),
                    "position_sigma_m": state_after.position_sigma_m,
                    "landmark_age_s": state_after.visual_age_s,
                    "inference_ms": float(result["inference_ms"]),
                })
        elif not dropped_stale:
            self.last_update_source = "imu_only"

        crop_result = None
        if self.crop_track_enabled:
            with self._crop_lock:
                crop_result = self._crop_result
                if crop_result is not None:
                    self._crop_result = None
        if crop_result is not None and result is None:
            crop_age_s = (
                time.time_ns() - int(crop_result["frame_wall_ns"])
            ) * 1e-9
            self.crop_track_ms = float(crop_result.get("crop_ms", 0.0))
            observations = crop_result.get("obs")
            if int(crop_result["active_gate"]) != int(active_gate):
                # Never rewind the filter with a crop of the gate that was
                # just passed.  At racing speed the result can arrive after
                # the official gate event has advanced the target.
                self.update_counts["crop_track_gate_mismatch"] += 1
            elif (
                crop_result["dispatch_visual_wall"]
                < self.last_visual_wall - 1e-6
            ):
                # A dense fix landed after this job's snapshot; rebasing to
                # the snapshot would discard it.  Drop the crop result --
                # the tracker refires within one interval.
                self.update_counts["crop_track_stale"] += 1
            elif (
                observations
                and crop_age_s <= self.crop_track_max_result_age_s
            ):
                crop_gate = int(crop_result["active_gate"])
                self.crop_attempts_by_gate[crop_gate] = (
                    self.crop_attempts_by_gate.get(crop_gate, 0) + 1
                )
                self.ekf = crop_result["ekf"]
                self.last_imu_us = crop_result["last_imu_us"]
                self.last_gyro = crop_result["last_gyro"]
                self.last_direct_pin_applied = False
                crop_fused = self.ekf.update_corners(
                    observations,
                    chi2_gate=6.0,
                    update_attitude=False,
                    sigma_px=2.0,
                )
                crop_direct_pin = (
                    self._direct_crop_position_pin(
                        observations, int(crop_result["active_gate"])
                    )
                    if self.crop_direct_position_pins and crop_fused >= 2
                    else None
                )
                self._propagate_imu()
                if crop_fused >= 2:
                    self.crop_fusions_by_gate[crop_gate] = (
                        self.crop_fusions_by_gate.get(crop_gate, 0) + 1
                    )
                    fused = crop_fused
                    self.last_visual_wall = (
                        float(crop_result["frame_wall_ns"]) * 1e-9
                    )
                    self.last_frame_id = int(crop_result["frame_id"])
                    if (
                        crop_direct_pin is not None
                        and crop_direct_pin.get("accepted", False)
                    ):
                        self.last_update_source = "crop_track_direct"
                        self.update_counts["crop_track_direct"] += 1
                    else:
                        self.last_update_source = "crop_track"
                        self.update_counts["crop_track"] += 1
                else:
                    self.update_counts["crop_track_none"] += 1
            else:
                self.update_counts["crop_track_none"] += 1

        frame = self.vision.latest
        if self.dense_process_isolation:
            worker_idle = not self._async_busy
        else:
            with self._async_lock:
                worker_idle = (
                    not self._async_busy
                    and self._async_job is None
                    and self._async_result is None
                )
        if (
            worker_idle
            and frame is not None
            and frame[0] != self._async_last_dispatched
            and time.monotonic() - self._async_last_dispatch_wall
            >= self.async_interval_s
        ):
            image = self._decode_jpeg(frame)
            if image is not None:
                job = {
                    "generation": self._async_generation,
                    "frame_id": int(frame[0]),
                    "frame_wall_ns": int(frame[3]),
                    "image": image,
                    "active_gate": int(active_gate),
                    "ekf": copy.deepcopy(self.ekf),
                    "last_imu_us": self.last_imu_us,
                    "last_gyro": self.last_gyro.copy(),
                }
                if self.dense_process_isolation:
                    key = (
                        int(job["generation"]),
                        int(job["frame_id"]),
                    )
                    request = {
                        "generation": key[0],
                        "frame_id": key[1],
                        "image": image,
                        "active_gate": int(active_gate),
                    }
                    try:
                        self._dense_request_queue.put_nowait(request)
                    except queue.Full:
                        pass
                    else:
                        self._async_pending[key] = job
                        self._async_busy = True
                else:
                    with self._async_lock:
                        self._async_job = job
                self._async_last_dispatched = int(frame[0])
                self._async_last_dispatch_wall = time.monotonic()
        crop_gate_enabled = (
            self.crop_track_gates is None
            or int(active_gate) in self.crop_track_gates
        )
        if self.crop_track_enabled and crop_gate_enabled and frame is not None:
            with self._crop_lock:
                crop_idle = (
                    not self._crop_busy
                    and self._crop_job is None
                    and self._crop_result is None
                )
            if (
                crop_idle
                and frame[0] != self._crop_last_dispatched
                and time.monotonic() - self._crop_last_dispatch_wall
                >= self.crop_track_interval_s
                and time.time() - self.last_visual_wall
                <= self.crop_track_max_age_s
            ):
                crop_job = {
                    "frame": frame,
                    "frame_id": int(frame[0]),
                    "frame_wall_ns": int(frame[3]),
                    "active_gate": int(active_gate),
                    "ekf": copy.deepcopy(self.ekf),
                    "last_imu_us": self.last_imu_us,
                    "last_gyro": self.last_gyro.copy(),
                    "dispatch_visual_wall": self.last_visual_wall,
                }
                with self._crop_lock:
                    self._crop_job = crop_job
                self._crop_last_dispatched = int(frame[0])
                self._crop_last_dispatch_wall = time.monotonic()
        self.last_fused = fused
        return self.state()

    def vision_fusion_by_gate(self) -> dict[str, dict[str, float | int]]:
        """Per-target detector acceptance rates for episode diagnostics."""
        gates = sorted(
            set(self.dense_attempts_by_gate)
            | set(self.crop_attempts_by_gate)
        )
        summary: dict[str, dict[str, float | int]] = {}
        for gate in gates:
            dense_attempts = self.dense_attempts_by_gate.get(gate, 0)
            dense_fusions = self.dense_fusions_by_gate.get(gate, 0)
            crop_attempts = self.crop_attempts_by_gate.get(gate, 0)
            crop_fusions = self.crop_fusions_by_gate.get(gate, 0)
            summary[str(gate)] = {
                "dense_attempts": dense_attempts,
                "dense_fusions": dense_fusions,
                "dense_rate": (
                    dense_fusions / dense_attempts
                    if dense_attempts else 0.0
                ),
                "crop_attempts": crop_attempts,
                "crop_fusions": crop_fusions,
                "crop_rate": (
                    crop_fusions / crop_attempts
                    if crop_attempts else 0.0
                ),
            }
        return summary

    def state(self) -> LiveLocalizerState:
        if self.ekf is None:
            return LiveLocalizerState(
                position=np.zeros(3),
                velocity=np.zeros(3),
                quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                gyro_raw=self.last_gyro.copy(),
                position_sigma_m=np.inf,
                visual_age_s=np.inf,
                corners_fused=0,
                frame_id=self.last_frame_id,
                initialized=False,
            )
        quaternion = self.ekf.q.as_quat()
        return LiveLocalizerState(
            position=self.ekf.p.copy(),
            velocity=self.ekf.v.copy(),
            quat_wxyz=np.array([
                quaternion[3], quaternion[0], quaternion[1], quaternion[2]
            ]),
            gyro_raw=self.last_gyro.copy(),
            position_sigma_m=float(np.sqrt(max(
                np.trace(self.ekf.P[0:3, 0:3]), 0.0
            ))),
            visual_age_s=float(time.time() - self.last_visual_wall),
            corners_fused=self.last_fused,
            frame_id=self.last_frame_id,
            initialized=True,
        )

    def apply_gate_plane_event(
        self,
        crossed_gate: int,
        forward_offset_m: float = 0.15,
        sigma_m: float = 0.20,
    ) -> dict | None:
        """Apply the official gate pass as a one-dimensional pose fix.

        VQ2's gate event is authoritative and tells us that the vehicle has
        just crossed the physical gate plane.  Vision/IMU can otherwise carry
        metres of longitudinal error despite having a plausible lateral gate
        lock.  Correct only position along the gate normal: the event does not
        observe aperture location, velocity, or attitude.
        """
        if (
            self.ekf is None
            or not 0 <= int(crossed_gate) < len(self.gates)
        ):
            return None
        gate_index = int(crossed_gate)
        gate = self.gates[gate_index]
        qw, qx, qy, qz = np.asarray(gate["quat_wxyz"], float)
        rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        normal = rotation[:, 1].copy()
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        previous = (
            np.asarray(self.gates[gate_index - 1]["pos"], float)
            if gate_index > 0 else np.zeros(3, float)
        )
        incoming = np.asarray(gate["pos"], float) - previous
        direction = 1.0 if float(np.dot(normal, incoming)) >= 0.0 else -1.0
        target_plane_m = direction * max(float(forward_offset_m), 0.0)
        gate_position = np.asarray(gate["pos"], float)
        before_m = float(np.dot(self.ekf.p - gate_position, normal))
        correction_m = target_plane_m - before_m
        self.ekf.p += correction_m * normal

        # A hard event constraint removes uncertainty only along the observed
        # plane axis.  Preserve lateral/vertical position, all velocity, and
        # attitude covariance and add realistic packet/vehicle-depth noise.
        projection = np.eye(9)
        projection[:3, :3] -= np.outer(normal, normal)
        self.ekf.P = projection @ self.ekf.P @ projection.T
        self.ekf.P[:3, :3] += (
            max(float(sigma_m), 1e-3) ** 2 * np.outer(normal, normal)
        )
        self.ekf.P = 0.5 * (self.ekf.P + self.ekf.P.T)
        payload = {
            "gate": gate_index,
            "before_plane_m": before_m,
            "after_plane_m": target_plane_m,
            "correction_m": correction_m,
        }
        self.last_gate_event_correction = payload
        self.update_counts["gate_event_plane"] = (
            self.update_counts.get("gate_event_plane", 0) + 1
        )
        return payload
