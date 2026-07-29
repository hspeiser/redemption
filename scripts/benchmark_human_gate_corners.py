"""Benchmark gate-corner models against Henry's human corner fits.

This is intentionally independent of the auto-label training/validation
split.  The default benchmark uses:

* the clean ``rc_20260724_003101`` lap (journals 9 and 10), and
* the later no-contact "gift" lap (journal 11).

Each journal row contains four clicked aperture corners in TL, TR, BR, BL
order.  The corresponding trace supplies only the image path; no trace pose
is used by the benchmark.

For GateNet, association is oracle-assisted in the same way the runtime map
prior works: each corner class is associated to the nearest expected corner.
For YOLO-pose models, the nearest gate instance is selected by image centre.
Both backends are then judged by strict pixel thresholds and by the resulting
1.50 m aperture PnP translation error.

Examples
--------
GateNet V7/V8/V9:

    .venv-train\\Scripts\\python.exe scripts\\benchmark_human_gate_corners.py ^
      --gatenet v7=data/models/gatenet_v7_best.pt ^
      --gatenet v8=data/models/gatenet_v8_best.pt ^
      --gatenet v9=data/models/gatenet_v9_best.pt

Lars's YOLO gate-pose models (requires ultralytics):

    python scripts\\benchmark_human_gate_corners.py ^
      --yolo pose_v4=data/lrspeiser_vision/gatepose_v4_best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.vision.model import GateNet  # noqa: E402
from aigp.vision.crop_gate import (  # noqa: E402
    CropGateNet,
    decode_crop_corners,
    orange_channel as crop_orange_channel,
    proposal_channel,
    warp_gate_crop,
)
from scripts.train_net import decode_corners, orange_channel  # noqa: E402

W, H = 640, 360
K = np.array([[320.0, 0.0, 320.0],
              [0.0, 320.0, 180.0],
              [0.0, 0.0, 1.0]], np.float64)
HALF_APERTURE_M = 0.75
OBJ_SQUARE = np.array([
    [-HALF_APERTURE_M, HALF_APERTURE_M, 0.0],
    [HALF_APERTURE_M, HALF_APERTURE_M, 0.0],
    [HALF_APERTURE_M, -HALF_APERTURE_M, 0.0],
    [-HALF_APERTURE_M, -HALF_APERTURE_M, 0.0],
], np.float64)
OUTER_HALF_M = 1.36
OBJ_OUTER = np.array([
    [-OUTER_HALF_M, OUTER_HALF_M, 0.0],
    [OUTER_HALF_M, OUTER_HALF_M, 0.0],
    [OUTER_HALF_M, -OUTER_HALF_M, 0.0],
    [-OUTER_HALF_M, -OUTER_HALF_M, 0.0],
], np.float64)
OBJ_EIGHT = np.concatenate([OBJ_SQUARE, OBJ_OUTER])


@dataclass(frozen=True)
class HumanSample:
    session: str
    journal: str
    frame: int
    gate: int
    image_path: str
    corners: np.ndarray
    click_rms_px: float


DEFAULT_SETS = (
    (
        "003101",
        (
            REPO / "data" / "vq2_map_human9.json.journal.jsonl",
            REPO / "data" / "vq2_map_human10.json.journal.jsonl",
        ),
        REPO / "data" / "vq2_trace_v3_101.npz",
    ),
    (
        "gift",
        (REPO / "data" / "vq2_map_human11.json.journal.jsonl",),
        REPO / "data" / "vq2_trace_gift.npz",
    ),
)

CONTACT_SET = (
    "003153_contact",
    (REPO / "data" / "vq2_map_human8.json.journal.jsonl",),
    REPO / "data" / "vq2_trace_slam2.npz",
)


def parse_named_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = spec.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name.strip(), Path(raw_path).expanduser().resolve()


def parse_named_pair(spec: str) -> tuple[str, Path, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "expected NAME=PRIMARY_PATH,REFINER_PATH"
        )
    name, raw_paths = spec.split("=", 1)
    if "," not in raw_paths:
        raise argparse.ArgumentTypeError(
            "expected NAME=PRIMARY_PATH,REFINER_PATH"
        )
    primary, refiner = raw_paths.split(",", 1)
    if not name.strip() or not primary.strip() or not refiner.strip():
        raise argparse.ArgumentTypeError(
            "expected non-empty NAME=PRIMARY_PATH,REFINER_PATH"
        )
    return (
        name.strip(),
        Path(primary).expanduser().resolve(),
        Path(refiner).expanduser().resolve(),
    )


def load_samples(include_contact: bool, max_click_rms: float) -> list[HumanSample]:
    sets = list(DEFAULT_SETS)
    if include_contact:
        sets.append(CONTACT_SET)
    samples: list[HumanSample] = []
    seen: set[tuple[str, int, int]] = set()
    for session, journals, trace_path in sets:
        with np.load(trace_path, allow_pickle=False) as trace:
            paths = np.asarray(trace["path"])
        for journal_path in journals:
            for line in journal_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("ok"):
                    continue
                click_rms = float(row.get("rms", math.inf))
                if click_rms > max_click_rms:
                    continue
                frame = int(row["frame"])
                gate = int(row["gate"])
                key = (session, frame, gate)
                if key in seen or not (0 <= frame < len(paths)):
                    continue
                corners = np.asarray(row["clicks"], np.float32)
                if corners.shape != (4, 2) or not np.isfinite(corners).all():
                    continue
                image_path = str(paths[frame])
                if not Path(image_path).is_file():
                    continue
                seen.add(key)
                samples.append(HumanSample(
                    session=session,
                    journal=journal_path.name,
                    frame=frame,
                    gate=gate,
                    image_path=image_path,
                    corners=corners,
                    click_rms_px=click_rms,
                ))
    return sorted(samples, key=lambda x: (x.session, x.frame, x.gate))


def pnp_candidates(corners: np.ndarray) -> list[tuple[np.ndarray, float]]:
    image_points = np.ascontiguousarray(corners, np.float64).reshape(-1, 1, 2)
    try:
        _n, rvecs, tvecs, _errs = cv2.solvePnPGeneric(
            OBJ_SQUARE, image_points, K, None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return []
    out = []
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                OBJ_SQUARE, image_points, K, None, rvec, tvec,
            )
        except cv2.error:
            continue
        t = np.asarray(tvec, np.float64).reshape(3)
        if not np.isfinite(t).all() or t[2] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            OBJ_SQUARE, rvec, tvec, K, None,
        )
        residual = projected.reshape(4, 2) - image_points.reshape(4, 2)
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        out.append((t, rms))
    return out


def best_pnp_translation(corners: np.ndarray, prior: np.ndarray | None = None):
    candidates = pnp_candidates(corners)
    if not candidates:
        return None
    if prior is None:
        return min(candidates, key=lambda x: x[1])[0]
    return min(candidates, key=lambda x: np.linalg.norm(x[0] - prior))[0]


def project_outer_corners(inner_corners: np.ndarray) -> np.ndarray:
    """Project the known 2.72 m coplanar panel from a clicked 1.50 m hole."""
    source = np.ascontiguousarray(OBJ_SQUARE[:, :2], np.float32)
    target = np.ascontiguousarray(inner_corners, np.float32)
    homography = cv2.getPerspectiveTransform(source, target)
    outer_xy = np.ascontiguousarray(
        OBJ_OUTER[:, :2].reshape(1, 4, 2), np.float32,
    )
    return cv2.perspectiveTransform(outer_xy, homography).reshape(4, 2)


def best_runtime_pnp_translation(
    points: np.ndarray, prior: np.ndarray,
) -> np.ndarray | None:
    """Mirror the runtime solve contract: 4 hole points or >=5 mixed."""
    finite = np.isfinite(points).all(axis=1)
    indices = np.flatnonzero(finite)
    if len(points) == 4:
        return best_pnp_translation(points, prior=prior) if finite.all() else None
    if finite[:4].all() and len(indices) == 4:
        return best_pnp_translation(points[:4], prior=prior)
    if len(indices) < 5:
        return None

    obj = np.ascontiguousarray(OBJ_EIGHT[indices], np.float64)
    image = np.ascontiguousarray(points[indices], np.float64).reshape(-1, 1, 2)
    candidates = []
    try:
        _n, rvecs, tvecs, _errs = cv2.solvePnPGeneric(
            obj, image, K, None, flags=cv2.SOLVEPNP_SQPNP,
        )
        candidates.extend(zip(rvecs, tvecs))
    except cv2.error:
        pass
    try:
        ok, rvec, tvec = cv2.solvePnP(
            obj, image, K, None, flags=cv2.SOLVEPNP_SQPNP,
        )
        if ok:
            candidates.append((rvec, tvec))
    except cv2.error:
        pass

    solved = []
    for rvec, tvec in candidates:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj, image, K, None, rvec, tvec,
            )
        except cv2.error:
            continue
        t = np.asarray(tvec, np.float64).reshape(3)
        if np.isfinite(t).all() and t[2] > 0.0:
            solved.append(t)
    return min(solved, key=lambda t: np.linalg.norm(t - prior)) if solved else None


class GateNetBackend:
    def __init__(
        self,
        checkpoint: Path,
        threshold: float,
        device: str,
        refiner_checkpoint: Path | None = None,
        refine_radius: float = 2.0,
    ):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.model = GateNet().to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.refiner = None
        if refiner_checkpoint is not None:
            refiner_ckpt = torch.load(
                refiner_checkpoint, map_location=device, weights_only=False
            )
            self.refiner = GateNet().to(device)
            self.refiner.load_state_dict(refiner_ckpt["model"])
            self.refiner.eval()
        self.device = device
        self.threshold = threshold
        self.refine_radius = refine_radius

    def decode(self, model, tensor):
        with torch.autocast(
            "cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")
        ):
            output = model(tensor)
        return decode_corners(
            output["hm"][0].float().cpu(),
            output["off"][0].float().cpu(),
            thresh=self.threshold,
            topk=24,
        )

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray, expected: np.ndarray):
        orange = orange_channel(bgr)
        inp = np.concatenate([
            bgr.astype(np.float32) / 255.0,
            orange[..., None],
        ], axis=2).transpose(2, 0, 1)
        tensor = torch.from_numpy(inp).unsqueeze(0).to(self.device)
        decoded = self.decode(self.model, tensor)
        if self.refiner is not None:
            refined = self.decode(self.refiner, tensor)
            snapped = []
            for corner_class, class_peaks in enumerate(decoded):
                ring = range(0, 4) if corner_class < 4 else range(4, 8)
                candidates = [
                    point for refine_class in ring
                    for point in refined[refine_class]
                ]
                class_snapped = []
                for u, v, score in class_peaks:
                    nearest = min(
                        candidates,
                        key=lambda point: math.hypot(
                            point[0] - u, point[1] - v
                        ),
                        default=None,
                    )
                    if nearest is not None and math.hypot(
                        nearest[0] - u, nearest[1] - v
                    ) <= self.refine_radius:
                        class_snapped.append(
                            (nearest[0], nearest[1], score)
                        )
                    else:
                        class_snapped.append((u, v, score))
                snapped.append(class_snapped)
            decoded = snapped
        span = float(max(np.ptp(expected[:, 0]), np.ptp(expected[:, 1])))
        association_radius = max(16.0, 0.25 * span)
        predicted = np.full((8, 2), np.nan, np.float32)
        scores = np.zeros(8, np.float32)
        # Journal clicks are in apparent TL,TR,BR,BL order. GateNet classes
        # are physical gate-local corners, whose apparent order rotates and
        # mirrors with gate orientation/face. Runtime knows that mapping from
        # the gate map. The benchmark recovers the equivalent best dihedral
        # mapping so class convention is not mistaken for localization error.
        permutations = []
        base = np.arange(4)
        for shift in range(4):
            permutations.append(np.roll(base, shift))
            permutations.append(np.roll(base[::-1], shift))

        def nearest(corner_class, click_index):
            candidates = decoded[int(corner_class)]
            if not candidates:
                return None
            return min(
                candidates,
                key=lambda p: math.hypot(
                    p[0] - expected[click_index, 0],
                    p[1] - expected[click_index, 1],
                ),
            )

        def permutation_cost(permutation):
            cost = 0.0
            for click_index, corner_class in enumerate(permutation):
                for ring_offset in (0, 4):
                    candidate = nearest(
                        int(corner_class) + ring_offset,
                        click_index + ring_offset,
                    )
                    if candidate is None:
                        cost += 2.0 * association_radius
                    else:
                        cost += min(
                            2.0 * association_radius,
                            math.hypot(
                                candidate[0] - expected[
                                    click_index + ring_offset, 0
                                ],
                                candidate[1] - expected[
                                    click_index + ring_offset, 1
                                ],
                            ),
                        )
            return cost

        mapping = min(permutations, key=permutation_cost)
        for click_index, corner_class in enumerate(mapping):
            for ring_offset in (0, 4):
                output_index = click_index + ring_offset
                candidate = nearest(
                    int(corner_class) + ring_offset, output_index,
                )
                if candidate is None:
                    continue
                distance = math.hypot(
                    candidate[0] - expected[output_index, 0],
                    candidate[1] - expected[output_index, 1],
                )
                if distance <= association_radius:
                    predicted[output_index] = candidate[:2]
                    scores[output_index] = candidate[2]
        return predicted, scores


class YoloPoseBackend:
    def __init__(self, checkpoint: Path, threshold: float, device: str):
        from ultralytics import YOLO

        self.model = YOLO(str(checkpoint))
        self.threshold = threshold
        self.device = device
        if device.startswith("cuda"):
            # Some development PyTorch builds ship torchvision without its
            # CUDA NMS kernel. Inference remains valid on CPU in that case.
            try:
                from torchvision.ops import nms

                nms(
                    torch.zeros((1, 4), device=device),
                    torch.ones(1, device=device),
                    0.5,
                )
            except (NotImplementedError, RuntimeError):
                print("torchvision CUDA NMS unavailable; YOLO using CPU")
                self.device = "cpu"

    def predict(self, bgr: np.ndarray, expected: np.ndarray):
        result = self.model.predict(
            bgr, conf=self.threshold, imgsz=640, verbose=False,
            device=self.device,
        )[0]
        if result.keypoints is None or result.boxes is None:
            return np.full((4, 2), np.nan, np.float32), np.zeros(4, np.float32)
        keypoints = result.keypoints.xy.cpu().numpy()
        if result.keypoints.conf is None:
            scores = np.ones(keypoints.shape[:2], np.float32)
        else:
            scores = result.keypoints.conf.cpu().numpy()
        expected_inner = expected[:4]
        gate_center = expected_inner.mean(axis=0)
        span = float(max(
            np.ptp(expected_inner[:, 0]), np.ptp(expected_inner[:, 1]),
        ))
        association_radius = max(25.0, 0.8 * span)
        best = None
        for corners, corner_scores in zip(keypoints, scores):
            distance = float(np.linalg.norm(corners.mean(axis=0) - gate_center))
            if distance <= association_radius and (
                best is None or distance < best[0]
            ):
                best = (distance, corners, corner_scores)
        if best is None:
            return np.full((4, 2), np.nan, np.float32), np.zeros(4, np.float32)
        return best[1].astype(np.float32), best[2].astype(np.float32)


class CropGateBackend:
    """YOLO gate proposal followed by instance-aware V11 corner refinement."""

    def __init__(
        self,
        checkpoint: Path,
        proposal_checkpoint: Path,
        proposal_threshold: float,
        device: str,
        proposal_padding: float = 3.2,
        inner_only: bool = False,
    ):
        from ultralytics import YOLO

        crop_checkpoint = torch.load(
            checkpoint, map_location=device, weights_only=False
        )
        self.model = CropGateNet().to(device)
        self.model.load_state_dict(crop_checkpoint["model"])
        self.model.eval()
        self.crop_size = int(crop_checkpoint.get("crop_size", 256))
        self.prior = proposal_channel(self.crop_size)
        self.proposal = YOLO(str(proposal_checkpoint))
        self.proposal_threshold = proposal_threshold
        self.proposal_padding = proposal_padding
        self.inner_only = inner_only
        self.device = device
        self.proposal_device = device
        if device.startswith("cuda"):
            try:
                from torchvision.ops import nms

                nms(
                    torch.zeros((1, 4), device=device),
                    torch.ones(1, device=device),
                    0.5,
                )
            except (NotImplementedError, RuntimeError):
                print("torchvision CUDA NMS unavailable; YOLO using CPU")
                self.proposal_device = "cpu"

    @torch.inference_mode()
    def predict(self, bgr: np.ndarray, expected: np.ndarray):
        result = self.proposal.predict(
            bgr,
            conf=self.proposal_threshold,
            imgsz=640,
            verbose=False,
            device=self.proposal_device,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return (
                np.full((8, 2), np.nan, np.float32),
                np.zeros(8, np.float32),
            )
        boxes = result.boxes.xyxy.cpu().numpy()
        expected_center = expected[:4].mean(axis=0)
        expected_span = float(max(
            np.ptp(expected[:4, 0]), np.ptp(expected[:4, 1])
        ))
        association_radius = max(30.0, 1.25 * expected_span)
        selected = None
        for box in boxes:
            center = np.asarray([
                (box[0] + box[2]) * 0.5,
                (box[1] + box[3]) * 0.5,
            ], np.float32)
            distance = float(np.linalg.norm(center - expected_center))
            if distance <= association_radius and (
                selected is None or distance < selected[0]
            ):
                selected = (distance, box, center)
        if selected is None:
            return (
                np.full((8, 2), np.nan, np.float32),
                np.zeros(8, np.float32),
            )
        _distance, box, center = selected
        side = max(
            18.0,
            float(max(box[2] - box[0], box[3] - box[1]))
            * self.proposal_padding,
        )
        crop, _forward, inverse = warp_gate_crop(
            bgr, center, side, self.crop_size
        )
        orange = crop_orange_channel(crop)
        network_input = np.concatenate([
            crop.astype(np.float32) / 255.0,
            orange[..., None],
            self.prior[..., None],
        ], axis=2).transpose(2, 0, 1)
        tensor = torch.from_numpy(network_input).unsqueeze(0).to(self.device)
        with torch.autocast(
            "cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")
        ):
            output = self.model(tensor)
        decoded = decode_crop_corners(
            output, self.crop_size, inverse_affine=inverse
        )
        predicted = np.asarray(decoded["corners"], np.float32)
        heatmap_scores = np.asarray(decoded["scores"], np.float32)
        visibility = np.asarray(decoded["visibility"], np.float32)
        presence = float(decoded["presence"])
        accepted = (
            (heatmap_scores >= 0.05)
            & (visibility >= 0.20)
            & (presence >= 0.20)
        )
        # Mirror runtime map-prior association. A valid proposal may contain
        # two visible gates; only corners consistent with the projected target
        # are allowed to reach PnP/EKF.
        corner_radius = max(16.0, 0.25 * expected_span)
        accepted &= np.linalg.norm(
            predicted - expected, axis=1
        ) <= corner_radius
        predicted_center = np.nanmean(predicted[:4], axis=0)
        center_radius = max(15.0, 0.35 * expected_span)
        if np.linalg.norm(predicted_center - expected_center) > center_radius:
            accepted[:] = False
        if accepted[:4].all() and accepted[4:].all():
            inner_area = abs(float(cv2.contourArea(predicted[:4])))
            outer_area = abs(float(cv2.contourArea(predicted[4:])))
            outer_center_error = float(np.linalg.norm(
                predicted[:4].mean(axis=0) - predicted[4:].mean(axis=0)
            ))
            if (
                inner_area < 4.0
                or not (1.8 < outer_area / inner_area < 6.0)
                or outer_center_error > 0.30 * math.sqrt(outer_area) + 3.0
            ):
                accepted[4:] = False
        if self.inner_only:
            accepted[4:] = False
        predicted[~accepted] = np.nan
        return predicted, heatmap_scores * visibility * presence


def percentile(values, q):
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, float), q)), 4)


def range_bucket(distance_m: float) -> str:
    if distance_m < 5.0:
        return "0-5m"
    if distance_m < 10.0:
        return "5-10m"
    if distance_m < 20.0:
        return "10-20m"
    return "20m+"


def summarize(rows: list[dict]) -> dict:
    corner_errors = [
        error
        for row in rows
        for error in row["inner_corner_errors_px"]
        if error is not None
    ]
    all_corner_errors = [
        error
        for row in rows
        for error in row["all_corner_errors_px"]
        if error is not None
    ]
    pnp_errors_cm = [
        row["pnp_position_error_m"] * 100.0
        for row in rows
        if row["pnp_position_error_m"] is not None
    ]
    result = {
        "samples": len(rows),
        "all_four_inner_candidates": round(
            sum(row["all_four_inner_candidates"] for row in rows)
            / max(len(rows), 1),
            4,
        ),
        "inner_corner_error_px": {
            "p50": percentile(corner_errors, 50),
            "p90": percentile(corner_errors, 90),
            "p99": percentile(corner_errors, 99),
        },
        "all_available_corner_error_px": {
            "p50": percentile(all_corner_errors, 50),
            "p90": percentile(all_corner_errors, 90),
            "p99": percentile(all_corner_errors, 99),
        },
        "pnp_availability": round(
            len(pnp_errors_cm) / max(len(rows), 1), 4,
        ),
        "pnp_position_error_cm": {
            "p50": percentile(pnp_errors_cm, 50),
            "p90": percentile(pnp_errors_cm, 90),
            "p99": percentile(pnp_errors_cm, 99),
        },
    }
    for threshold in (2.0, 4.0, 8.0, 16.0):
        key = f"{int(threshold)}px"
        result.setdefault("corner_recall", {})[key] = round(
            sum(
                error is not None and error <= threshold
                for row in rows
                for error in row["inner_corner_errors_px"]
            ) / max(4 * len(rows), 1),
            4,
        )
        result.setdefault("four_corner_gate_recall", {})[key] = round(
            sum(
                all(error is not None and error <= threshold
                    for error in row["inner_corner_errors_px"])
                for row in rows
            ) / max(len(rows), 1),
            4,
        )
    return result


def evaluate_backend(name: str, backend, samples: list[HumanSample]):
    rows = []
    for index, sample in enumerate(samples):
        bgr = cv2.imread(sample.image_path)
        if bgr is None:
            continue
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H))
        expected = np.concatenate([
            sample.corners,
            project_outer_corners(sample.corners),
        ])
        predicted, scores = backend.predict(bgr, expected)
        errors = []
        for corner, truth in zip(predicted, expected):
            errors.append(
                float(np.linalg.norm(corner - truth))
                if np.isfinite(corner).all() else None
            )
        gt_t = best_pnp_translation(sample.corners)
        pred_t = None
        if gt_t is not None:
            pred_t = best_runtime_pnp_translation(predicted, prior=gt_t)
        rows.append({
            "session": sample.session,
            "journal": sample.journal,
            "frame": sample.frame,
            "gate": sample.gate,
            "image_path": sample.image_path,
            "click_rms_px": sample.click_rms_px,
            "range_m": float(np.linalg.norm(gt_t)) if gt_t is not None else None,
            "inner_corner_errors_px": errors[:4],
            "all_corner_errors_px": errors,
            "corner_scores": [float(x) for x in scores],
            "candidate_points": int(np.isfinite(predicted).all(axis=1).sum()),
            "all_four_inner_candidates": bool(
                np.isfinite(predicted[:4]).all()
            ),
            "pnp_position_error_m": (
                float(np.linalg.norm(pred_t - gt_t))
                if pred_t is not None and gt_t is not None else None
            ),
        })
        if (index + 1) % 25 == 0:
            print(f"{name}: {index + 1}/{len(samples)}", flush=True)

    by_session = {
        session: summarize([r for r in rows if r["session"] == session])
        for session in sorted({r["session"] for r in rows})
    }
    by_range = {}
    ranged_rows = [r for r in rows if r["range_m"] is not None]
    for bucket in ("0-5m", "5-10m", "10-20m", "20m+"):
        selected = [r for r in ranged_rows if range_bucket(r["range_m"]) == bucket]
        if selected:
            by_range[bucket] = summarize(selected)
    by_gate = {
        str(gate): summarize([r for r in rows if r["gate"] == gate])
        for gate in sorted({r["gate"] for r in rows})
    }
    return {
        "overall": summarize(rows),
        "by_session": by_session,
        "by_range": by_range,
        "by_gate": by_gate,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gatenet", action="append", default=[], type=parse_named_path,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--gatenet-refiner",
        action="append",
        default=[],
        type=parse_named_pair,
        metavar="NAME=PRIMARY_PATH,REFINER_PATH",
    )
    parser.add_argument(
        "--yolo", action="append", default=[], type=parse_named_path,
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--crop-gatenet",
        action="append",
        default=[],
        type=parse_named_pair,
        metavar="NAME=CROP_CHECKPOINT,PROPOSAL_CHECKPOINT",
    )
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--proposal-threshold", type=float, default=0.05)
    parser.add_argument("--proposal-padding", type=float, default=3.2)
    parser.add_argument("--crop-inner-only", action="store_true")
    parser.add_argument("--refine-radius", type=float, default=2.0)
    parser.add_argument("--max-click-rms", type=float, default=1.0)
    parser.add_argument("--include-contact", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data" / "models" / "human_corner_benchmark.json",
    )
    args = parser.parse_args()
    if (
        not args.gatenet
        and not args.gatenet_refiner
        and not args.yolo
        and not args.crop_gatenet
    ):
        parser.error(
            "provide at least one GateNet, crop-GateNet, or YOLO model"
        )

    samples = load_samples(args.include_contact, args.max_click_rms)
    if not samples:
        raise SystemExit("no benchmark samples found")
    print(
        f"human benchmark: {len(samples)} samples, "
        f"sessions={sorted({s.session for s in samples})}, "
        f"gates={sorted({s.gate for s in samples})}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {
        "benchmark": {
            "samples": len(samples),
            "max_click_rms_px": args.max_click_rms,
            "includes_contact_lap": args.include_contact,
            "sessions": sorted({s.session for s in samples}),
            "gates": sorted({s.gate for s in samples}),
            "association": (
                "oracle-assisted by expected gate projection; matches runtime "
                "map-prior association and does not measure false positives"
            ),
        },
        "models": {},
    }
    for name, checkpoint in args.gatenet:
        print(f"\nloading GateNet {name}: {checkpoint}", flush=True)
        backend = GateNetBackend(checkpoint, args.threshold, device)
        report["models"][name] = evaluate_backend(name, backend, samples)
    for name, primary, refiner in args.gatenet_refiner:
        print(
            f"\nloading GateNet refiner {name}: {primary} + {refiner}",
            flush=True,
        )
        backend = GateNetBackend(
            primary,
            args.threshold,
            device,
            refiner_checkpoint=refiner,
            refine_radius=args.refine_radius,
        )
        report["models"][name] = evaluate_backend(name, backend, samples)
    for name, checkpoint in args.yolo:
        print(f"\nloading YOLO pose {name}: {checkpoint}", flush=True)
        backend = YoloPoseBackend(checkpoint, args.threshold, device)
        report["models"][name] = evaluate_backend(name, backend, samples)
    for name, crop_checkpoint, proposal_checkpoint in args.crop_gatenet:
        print(
            f"\nloading crop GateNet {name}: "
            f"{crop_checkpoint} + {proposal_checkpoint}",
            flush=True,
        )
        backend = CropGateBackend(
            crop_checkpoint,
            proposal_checkpoint,
            args.proposal_threshold,
            device,
            proposal_padding=args.proposal_padding,
            inner_only=args.crop_inner_only,
        )
        report["models"][name] = evaluate_backend(name, backend, samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    for name, model_report in report["models"].items():
        print(name, json.dumps(model_report["overall"], indent=2))
    print(f"\nfull report -> {args.output}")


if __name__ == "__main__":
    main()
