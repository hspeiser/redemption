"""PnP pose recovery + evaluation (the non-trained final target).

For each detected gate: take the 4 predicted inner corners (+ confidences),
solve for the gate's 6-DoF pose, and compare against the ground-truth pose from
the dataset sidecars. The key deliverable is *reconstruction error vs training
checkpoint* -- computed by :func:`evaluate_run`.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

from .camera import PinholeCamera
from .config import DotDict, load_all
from .dataset import dataset_root, split_paths
from .gate import Gate
from .infer import infer_image, load_model
from .metrics import (corner_rmse, match_by_center, reprojection_rmse,
                      rotation_error_deg, translation_error)
from .utils import ensure_dir, get_logger, read_json, write_json

_SOLVERS = {
    "IPPE": cv2.SOLVEPNP_IPPE,
    "IPPE_SQUARE": cv2.SOLVEPNP_IPPE_SQUARE,
    "ITERATIVE": cv2.SOLVEPNP_ITERATIVE,
}


def _reproj_residuals(params, obj, img, K, dist, w):
    rvec, tvec = params[:3], params[3:]
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return ((proj.reshape(-1, 2) - img) * w[:, None]).ravel()


def solve_pnp(object_pts: np.ndarray, image_pts: np.ndarray, K: np.ndarray,
              dist: np.ndarray, method: str, refine_lm: bool,
              weight_by_conf: bool, conf: np.ndarray) -> dict | None:
    """Solve a single gate's pose. Returns dict(rvec, tvec, R, reproj_rmse) or None."""
    flag = _SOLVERS.get(method.upper(), cv2.SOLVEPNP_IPPE)
    obj = np.asarray(object_pts, np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_pts, np.float64).reshape(-1, 1, 2)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=flag)
    except cv2.error:
        return None
    if not ok:
        return None

    img2 = img.reshape(-1, 2)
    if refine_lm:
        if weight_by_conf:
            w = np.sqrt(np.clip(np.asarray(conf, float), 1e-3, None))
            res = least_squares(
                _reproj_residuals, np.concatenate([rvec.ravel(), tvec.ravel()]),
                args=(object_pts.astype(np.float64), img2, K, dist, w), method="lm",
            )
            rvec, tvec = res.x[:3].reshape(3, 1), res.x[3:].reshape(3, 1)
        else:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    reproj = reprojection_rmse(object_pts, img2, rvec, tvec, K, dist)
    return {"rvec": rvec.ravel(), "tvec": tvec.ravel(), "R": R, "reproj_rmse": reproj}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def _epoch_of(path: Path) -> int:
    """Sort key for checkpoint files: epochN -> N, last -> big, best -> big+1."""
    name = path.stem
    m = re.search(r"epoch(\d+)", name)
    if m:
        return int(m.group(1))
    if name == "last":
        return 10_000
    if name == "best":
        return 10_001
    return 9_999


def find_checkpoints(run_dir: Path, which) -> list[Path]:
    weights = run_dir / "weights"
    if not weights.exists():
        return []
    ckpts = sorted(weights.glob("*.pt"), key=_epoch_of)
    if isinstance(which, list):
        wanted = {int(w) for w in which}
        ckpts = [c for c in ckpts if _epoch_of(c) in wanted]
    return ckpts


def latest_run(project_root: Path, name_prefix: str = "gates_pose") -> Path | None:
    """Most-recently-modified run dir under ``project_root`` (robust to nesting).

    A "run dir" is any directory containing a ``weights`` subdir. Ultralytics may
    nest runs (e.g. ``runs/pose/gates_pose``), so we search recursively and prefer
    dirs whose name starts with ``name_prefix``.
    """
    root = Path(project_root)
    if not root.exists():
        return None
    candidates = [w.parent for w in root.rglob("weights") if w.is_dir()]
    if not candidates:
        return None
    named = [c for c in candidates if c.name.startswith(name_prefix)]
    pool = named or candidates
    return max(pool, key=lambda d: d.stat().st_mtime)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_checkpoint(weights: Path, cfg: DotDict, max_images: int | None = None,
                        device=None) -> dict:
    """Run inference + PnP over the eval split for one checkpoint.

    ``max_images`` limits how many split images are evaluated (used by the live
    watcher for a fast per-epoch estimate); ``device`` overrides the inference
    device (e.g. ``"cpu"`` to avoid contending with a running training job).
    """
    camera = PinholeCamera.from_config(cfg.camera)
    gate = Gate.from_config(cfg.gate)
    K = camera.K
    dist = camera.dist_coeffs
    obj = gate.object_points

    pnp = cfg.pnp
    split = pnp.data.split
    root = dataset_root(cfg.datagen)
    paths = split_paths(root, split)
    meta_files = sorted(paths["meta"].glob("*.json"))
    if max_images:
        meta_files = meta_files[:int(max_images)]

    model = load_model(str(weights))
    device = cfg.train.train.device if device is None else device
    imgsz = int(cfg.train.train.imgsz)

    records: list[dict] = []
    for mf in meta_files:
        meta = read_json(mf)
        img_path = paths["images"] / meta["image"]
        dets = infer_image(model, str(img_path), conf=float(pnp.confidence.min_detection_conf),
                           imgsz=imgsz, device=device)
        gts = meta["gates"]
        if not dets or not gts:
            continue

        pred_centers = np.array([d.center_px for d in dets])
        gt_centers = np.array([np.mean(g["corners_px"], axis=0) for g in gts])
        pairs = match_by_center(pred_centers, gt_centers, float(pnp.matching.max_center_dist_px))

        for pi, gi in pairs:
            d = dets[pi]
            g = gts[gi]
            keep = d.kpt_conf >= float(pnp.confidence.min_keypoint_conf)
            if int(keep.sum()) < int(pnp.confidence.min_corners):
                continue
            sol = solve_pnp(obj, d.kpts_px, K, dist, str(pnp.solver.method),
                            bool(pnp.solver.refine_lm), bool(pnp.confidence.weight_by_confidence),
                            d.kpt_conf)
            if sol is None:
                continue
            R_gt, _ = cv2.Rodrigues(np.asarray(g["rvec"], float))
            records.append({
                "mean_conf": float(np.mean(d.kpt_conf)),
                "min_conf": float(np.min(d.kpt_conf)),
                "box_conf": float(d.box_conf),
                "corner_rmse": corner_rmse(d.kpts_px, np.asarray(g["corners_px"], float)),
                "trans_err": translation_error(sol["tvec"], np.asarray(g["tvec"], float)),
                "rot_err": rotation_error_deg(sol["R"], R_gt),
                "reproj_rmse": sol["reproj_rmse"],
                "gt_depth": float(g["depth"]),
            })

    return {
        "checkpoint": weights.name,
        "epoch": _epoch_of(weights),
        "n_matched": len(records),
        "records": records,
        **_aggregate(records),
    }


def _aggregate(records: list[dict]) -> dict:
    if not records:
        return {"mean_trans_err": None, "mean_rot_err": None, "mean_corner_rmse": None,
                "wmean_trans_err": None, "wmean_rot_err": None, "mean_reproj_rmse": None}
    arr = {k: np.array([r[k] for r in records], float)
           for k in ("trans_err", "rot_err", "corner_rmse", "reproj_rmse", "mean_conf")}
    w = arr["mean_conf"]
    wsum = w.sum() + 1e-9
    return {
        "mean_trans_err": float(arr["trans_err"].mean()),
        "mean_rot_err": float(arr["rot_err"].mean()),
        "mean_corner_rmse": float(arr["corner_rmse"].mean()),
        "mean_reproj_rmse": float(arr["reproj_rmse"].mean()),
        # confidence-weighted pose errors (per the user's request to use confidence)
        "wmean_trans_err": float((arr["trans_err"] * w).sum() / wsum),
        "wmean_rot_err": float((arr["rot_err"] * w).sum() / wsum),
    }


def evaluate_run(cfg: DotDict | None = None) -> dict:
    """Evaluate PnP across all requested checkpoints of a training run."""
    log = get_logger()
    cfg = cfg or load_all()
    pnp = cfg.pnp

    run_dir = Path(pnp.model.run_dir) if pnp.model.run_dir else None
    if run_dir is None:
        run_dir = latest_run(Path(cfg.train.train.project), cfg.train.train.name)
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError("No training run found for PnP eval. Train a model first.")

    ckpts = find_checkpoints(run_dir, pnp.eval.checkpoints)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints under {run_dir/'weights'}")

    log.info(f"PnP eval over {len(ckpts)} checkpoint(s) from {run_dir}")
    per_ckpt = []
    for ck in tqdm(ckpts, desc="checkpoints"):
        per_ckpt.append(evaluate_checkpoint(ck, cfg))

    out_dir = ensure_dir(run_dir / "pnp_eval")
    result = {"run_dir": str(run_dir), "split": pnp.data.split, "checkpoints": per_ckpt}
    # Strip bulky per-record lists from the on-disk curve summary; keep for report caller.
    curve = [{k: v for k, v in c.items() if k != "records"} for c in per_ckpt]
    write_json(out_dir / "pnp_curve.json", {"run_dir": str(run_dir), "curve": curve})
    result["curve"] = curve
    log.info("PnP eval complete")
    return result
