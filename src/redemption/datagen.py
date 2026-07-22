"""Synthetic dataset generation: sample -> render -> filter -> write.

Only FULLY-VISIBLE gates (all 4 inner corners in front, in-frame and unoccluded
by a nearer gate) become labelled instances; other placements are dropped and
images may end up label-free (useful hard negatives). Ground-truth poses are
written to per-image sidecars for later distribution plots and PnP evaluation.

Generation is parallelized across processes (rendering is CPU-bound). Each worker
reloads config from disk and writes its own image/label/meta files.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .camera import PinholeCamera
from .config import DotDict, load_all
from .dataset import (SPLITS, dataset_root, format_label_line, prepare_dirs,
                      split_paths, write_data_yaml)
from .gate import Gate
from .geometry import polygon_contains, sample_pose
from .render import GateProjection, render_scene
from .utils import (ensure_dir, get_logger, resolve_workers, rng_for, timer,
                    timestamp, write_json)

# Per-worker globals (populated by the pool initializer).
_WORKER: dict = {}


def _polygon_area(pts: np.ndarray) -> float:
    """Absolute polygon area via the shoelace formula."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _silhouette(proj: GateProjection) -> np.ndarray:
    """Outer silhouette (front+back outer hull) used for occlusion tests."""
    pts = np.vstack([proj.front_outer, proj.back_outer]).astype(np.float32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    return hull.reshape(-1, 2)


def compute_instances(
    camera: PinholeCamera,
    gate: Gate,
    poses: list,
    projections: list[GateProjection],
    vis_cfg: DotDict,
) -> list[dict]:
    """Return label+meta dicts for every gate that passes the visibility filter."""
    W, H = camera.width, camera.height
    instances: list[dict] = []

    # Pre-compute silhouettes/holes for occlusion tests.
    sils = [_silhouette(p) for p in projections]

    for i, pj in enumerate(projections):
        if not np.all(pj.inner_in_front):
            continue
        pix = pj.front_inner  # (4,2) keypoints

        if bool(vis_cfg.require_all_corners_in_frame):
            if (pix[:, 0].min() < 0 or pix[:, 0].max() >= W
                    or pix[:, 1].min() < 0 or pix[:, 1].max() >= H):
                continue

        if _polygon_area(pix) < float(vis_cfg.min_area_px):
            continue

        if bool(vis_cfg.require_unoccluded):
            occluded = False
            for k, pk in enumerate(projections):
                if k == i or pk.center_depth >= pj.center_depth:
                    continue  # only strictly-nearer gates can occlude
                for c in pix:
                    if polygon_contains(sils[k], c) and not polygon_contains(pk.front_inner, c):
                        occluded = True
                        break
                if occluded:
                    break
            if occluded:
                continue

        instances.append(_make_instance(poses[i], pj, camera))

    return instances


def _make_instance(pose, proj: GateProjection, camera: PinholeCamera) -> dict:
    W, H = camera.width, camera.height
    pix = proj.front_inner

    # Bounding box from the clipped outer silhouette, widened to cover keypoints.
    sil = np.vstack([proj.front_outer, proj.back_outer])
    x0 = min(sil[:, 0].min(), pix[:, 0].min())
    x1 = max(sil[:, 0].max(), pix[:, 0].max())
    y0 = min(sil[:, 1].min(), pix[:, 1].min())
    y1 = max(sil[:, 1].max(), pix[:, 1].max())
    x0, x1 = float(np.clip(x0, 0, W)), float(np.clip(x1, 0, W))
    y0, y1 = float(np.clip(y0, 0, H)), float(np.clip(y1, 0, H))
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    bbox_norm = ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, bw / W, bh / H)

    kpts_norm = [(float(x) / W, float(y) / H, 2) for x, y in pix]
    rvec, tvec = pose.rvec_tvec()
    meta = {
        "pitch_deg": pose.pitch_deg,
        "yaw_deg": pose.yaw_deg,
        "center_cam": pose.center.tolist(),
        "depth": pose.depth,
        "rvec": rvec.tolist(),
        "tvec": tvec.tolist(),
        "corners_px": pix.tolist(),
    }
    return {"bbox_norm": bbox_norm, "kpts_norm": kpts_norm, "meta": meta}


def generate_scene(camera, gate, dg: DotDict, rng: np.random.Generator):
    """Sample gates, render one image, and compute its visible instances."""
    scene = dg.scene
    n = int(rng.integers(int(scene.min_gates), int(scene.max_gates) + 1))
    poses = [sample_pose(camera, dg.pose, rng) for _ in range(n)]
    image, projections = render_scene(camera, gate, poses, dg, rng)
    instances = compute_instances(camera, gate, poses, projections, dg.visibility)
    return image, instances


# ---------------------------------------------------------------------------
# Worker plumbing
# ---------------------------------------------------------------------------
def _init_worker() -> None:
    cfg = load_all()
    _WORKER["camera"] = PinholeCamera.from_config(cfg.camera)
    _WORKER["gate"] = Gate.from_config(cfg.gate)
    _WORKER["dg"] = cfg.datagen
    _WORKER["root"] = dataset_root(cfg.datagen)


def _gen_one(task: tuple[str, int, int]) -> tuple[str, int]:
    """Generate + persist a single image. Returns (split, n_instances)."""
    split, index, global_index = task
    camera = _WORKER["camera"]
    gate = _WORKER["gate"]
    dg = _WORKER["dg"]
    root: Path = _WORKER["root"]

    rng = rng_for(int(dg.dataset.seed), global_index)
    image, instances = generate_scene(camera, gate, dg, rng)

    stem = f"{split}_{index:06d}"
    paths = split_paths(root, split)
    cv2.imwrite(str(paths["images"] / f"{stem}.png"), image)

    lines = [format_label_line(0, inst["bbox_norm"], inst["kpts_norm"]) for inst in instances]
    (paths["labels"] / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    write_json(paths["meta"] / f"{stem}.json", {
        "image": f"{stem}.png",
        "split": split,
        "width": camera.width,
        "height": camera.height,
        "K": camera.K.tolist(),
        "gates": [inst["meta"] for inst in instances],
    })
    return split, len(instances)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate(cfg: DotDict | None = None) -> dict:
    """Generate the full dataset described by the configs. Returns a summary dict."""
    log = get_logger()
    cfg = cfg or load_all()
    dg = cfg.datagen
    camera = PinholeCamera.from_config(cfg.camera)

    root = dataset_root(dg)
    log.info(f"Generating dataset '{dg.dataset.name}' at {root}")
    log.info(f"Camera: {camera.width}x{camera.height}  HFoV={camera.hfov_deg:.1f}  "
             f"VFoV={camera.vfov_deg:.1f}")
    prepare_dirs(root, bool(dg.runtime.overwrite))
    write_data_yaml(root)

    counts = {"train": int(dg.dataset.n_train), "val": int(dg.dataset.n_val),
              "test": int(dg.dataset.n_test)}
    tasks: list[tuple[str, int, int]] = []
    gi = 0
    for split in SPLITS:
        for i in range(counts[split]):
            tasks.append((split, i, gi))
            gi += 1

    workers = resolve_workers(int(dg.runtime.workers))
    log.info(f"{len(tasks)} images across {workers} worker(s)")

    per_split_instances = {s: 0 for s in SPLITS}
    per_split_images = {s: 0 for s in SPLITS}
    labelled_images = {s: 0 for s in SPLITS}

    with timer(log, "rendering"):
        if workers == 1:
            _init_worker()
            results = (_gen_one(t) for t in tasks)
            for split, n_inst in tqdm(results, total=len(tasks)):
                _tally(split, n_inst, per_split_instances, per_split_images, labelled_images)
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
                for split, n_inst in tqdm(ex.map(_gen_one, tasks, chunksize=8), total=len(tasks)):
                    _tally(split, n_inst, per_split_instances, per_split_images, labelled_images)

    summary = {
        "dataset": dg.dataset.name,
        "root": str(root.resolve()),
        "timestamp": timestamp(),
        "counts": counts,
        "instances": per_split_instances,
        "images": per_split_images,
        "labelled_images": labelled_images,
        "labelled_fraction": {
            s: (labelled_images[s] / per_split_images[s]) if per_split_images[s] else 0.0
            for s in SPLITS
        },
    }
    ensure_dir(root)
    write_json(root / "datagen_summary.json", summary)
    log.info(f"Instances: {per_split_instances}  labelled-fraction: "
             + ", ".join(f"{s}={summary['labelled_fraction'][s]:.2f}" for s in SPLITS))
    return summary


def _tally(split, n_inst, inst_acc, img_acc, lbl_acc) -> None:
    inst_acc[split] += n_inst
    img_acc[split] += 1
    if n_inst > 0:
        lbl_acc[split] += 1
