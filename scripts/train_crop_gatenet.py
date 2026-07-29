"""Train instance-aware CropGateNet V11 on one proposed gate per crop.

The crop is generated from exact NPZ gate geometry with detector-like centre
and scale jitter.  It predicts grouped apparent-image corners, visibility,
per-corner uncertainty, and gate presence.  Entire sessions remain disjoint
between training and validation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.vision.crop_gate import (  # noqa: E402
    CropGateNet,
    canonical_quad,
    decode_crop_corners,
    initialize_crop_backbone,
    orange_channel,
    proposal_channel,
    transform_points,
    warp_gate_crop,
)
from aigp.vision.model import focal_heatmap_loss  # noqa: E402


CROP_SIZE = 256
HEATMAP_SIZE = CROP_SIZE // 4
VAL_EPISODES = (
    "rc_20260723_024016",
    "rc_20260723_022140",
    "rc_20260723_022654",
    "rc_20260724_003101",
)


def apply_path_maps(path: str, path_maps: list[tuple[str, str]]) -> str:
    for old, new in path_maps:
        if path.startswith(old):
            mapped = new + path[len(old):]
            return mapped.replace("\\", "/") if "/" in new else mapped
    return path


def splat_corner(
    heatmap: np.ndarray,
    offsets: np.ndarray,
    offset_mask: np.ndarray,
    corner_class: int,
    uv: np.ndarray,
):
    u = float(uv[0]) / 4.0
    v = float(uv[1]) / 4.0
    row, column = int(v), int(u)
    if not (
        0 <= row < HEATMAP_SIZE and 0 <= column < HEATMAP_SIZE
    ):
        return
    sigma = 1.5
    radius = 4
    row0, row1 = max(0, row - radius), min(
        HEATMAP_SIZE, row + radius + 1
    )
    col0, col1 = max(0, column - radius), min(
        HEATMAP_SIZE, column + radius + 1
    )
    yy, xx = np.mgrid[row0:row1, col0:col1]
    gaussian = np.exp(
        -((xx - u) ** 2 + (yy - v) ** 2) / (2.0 * sigma * sigma)
    )
    heatmap[corner_class, row0:row1, col0:col1] = np.maximum(
        heatmap[corner_class, row0:row1, col0:col1], gaussian
    )
    heatmap[corner_class, row, column] = 1.0
    offsets[2 * corner_class, row, column] = u - column
    offsets[2 * corner_class + 1, row, column] = v - row
    offset_mask[corner_class, row, column] = 1.0


def motion_blur(image: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length) | 1)
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D(
        (length * 0.5 - 0.5, length * 0.5 - 0.5), angle_deg, 1.0
    )
    kernel = cv2.warpAffine(kernel, matrix, (length, length))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def add_blue_occlusion(image: np.ndarray) -> tuple[np.ndarray, float]:
    overlay = image.copy()
    height, width = image.shape[:2]
    thickness = random.randint(4, 20)
    points = np.asarray([
        (
            random.randint(-width // 4, width + width // 4),
            random.randint(-height // 4, height + height // 4),
        )
        for _ in range(random.randint(2, 4))
    ], np.int32)
    color = random.choice([
        (255, 160, 0),
        (255, 220, 20),
        (230, 120, 0),
    ])
    cv2.polylines(
        overlay,
        [points],
        False,
        color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    alpha = random.uniform(0.45, 0.9)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0), (
        1.0 + 0.04 * thickness * alpha
    )


class CropGateDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        train: bool,
        path_maps: list[tuple[str, str]],
        negative_ratio: float = 0.20,
    ):
        self.train = train
        self.path_maps = path_maps
        self.negative_ratio = negative_ratio if train else 0.10
        self.items: list[dict] = []
        self.frame_boxes: dict[str, list[tuple[np.ndarray, float]]] = (
            defaultdict(list)
        )
        seen_instances: set[tuple[str, int, int]] = set()
        for npz_path in files:
            with np.load(npz_path, allow_pickle=False) as data:
                paths = np.asarray(data["path"])
                inner = np.asarray(data["inner"], np.float32)
                outer = np.asarray(data["outer"], np.float32)
                vis_inner = np.asarray(data["vis_inner"], bool)
                vis_outer = np.asarray(data["vis_outer"], bool)
                for frame_index, stored_path in enumerate(paths):
                    path = apply_path_maps(str(stored_path), path_maps)
                    for gate_index in range(inner.shape[1]):
                        inner_result = canonical_quad(
                            inner[frame_index, gate_index]
                        )
                        outer_result = canonical_quad(
                            outer[frame_index, gate_index]
                        )
                        if inner_result is None or outer_result is None:
                            continue
                        inner_points, inner_order = inner_result
                        outer_points, outer_order = outer_result
                        visibility = np.concatenate([
                            vis_inner[frame_index, gate_index][inner_order],
                            vis_outer[frame_index, gate_index][outer_order],
                        ])
                        points = np.concatenate(
                            [inner_points, outer_points], axis=0
                        )
                        finite = np.isfinite(points).all(axis=1)
                        visibility &= finite
                        if visibility.sum() < 2:
                            continue
                        span_xy = (
                            np.nanmax(outer_points, axis=0)
                            - np.nanmin(outer_points, axis=0)
                        )
                        span = float(max(span_xy))
                        if not np.isfinite(span) or span < 3.0:
                            continue
                        center = np.nanmean(outer_points, axis=0)
                        key = (
                            path,
                            int(round(float(center[0]))),
                            int(round(float(center[1]))),
                        )
                        if key in seen_instances:
                            continue
                        seen_instances.add(key)
                        item = {
                            "path": path,
                            "points": points.astype(np.float32),
                            "visibility": visibility.astype(np.float32),
                            "center": center.astype(np.float32),
                            "span": span,
                            "session": npz_path.stem,
                        }
                        self.items.append(item)
                        self.frame_boxes[path].append(
                            (center.astype(np.float32), span)
                        )
        self.positive_count = len(self.items)
        self.negative_count = int(self.positive_count * self.negative_ratio)
        if not self.train:
            # Deterministic exact and proposal-error variants.
            variants = []
            for item in self.items:
                for variant in range(3):
                    variants.append({**item, "variant": variant})
            self.items = variants
            self.positive_count = len(self.items)
            self.negative_count = min(
                int(self.positive_count * self.negative_ratio), 1000
            )
        self.prior = proposal_channel(CROP_SIZE)

    def __len__(self):
        return self.positive_count + self.negative_count

    def read_image(self, path: str) -> np.ndarray | None:
        image = cv2.imread(path)
        if image is None:
            return None
        if image.shape[:2] != (360, 640):
            image = cv2.resize(image, (640, 360))
        return image

    def proposal_geometry(self, item: dict) -> tuple[np.ndarray, float]:
        if self.train:
            side = max(18.0, item["span"] * random.uniform(1.45, 2.65))
            center = item["center"] + np.asarray([
                random.uniform(-0.13, 0.13) * side,
                random.uniform(-0.13, 0.13) * side,
            ], np.float32)
            return center, side
        variant = item.get("variant", 0)
        offsets = (
            (0.0, 0.0),
            (0.08, -0.05),
            (-0.07, 0.06),
        )
        multipliers = (2.0, 2.30, 1.75)
        side = max(18.0, item["span"] * multipliers[variant])
        center = item["center"] + side * np.asarray(
            offsets[variant], np.float32
        )
        return center, side

    def negative_geometry(
        self, item: dict, image_shape: tuple[int, int]
    ) -> tuple[np.ndarray, float]:
        height, width = image_shape
        side = random.uniform(24.0, 220.0)
        boxes = self.frame_boxes[item["path"]]
        center = np.asarray([width * 0.5, height * 0.5], np.float32)
        for _ in range(30):
            candidate = np.asarray([
                random.uniform(0.0, width),
                random.uniform(0.0, height),
            ], np.float32)
            if all(
                np.linalg.norm(candidate - gate_center)
                > 0.65 * (side + gate_span)
                for gate_center, gate_span in boxes
            ):
                center = candidate
                break
        return center, side

    def augment(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        uncertainty_multiplier = 1.0
        if random.random() < 0.65:
            gain = random.uniform(0.65, 1.35)
            bias = random.uniform(-24.0, 24.0)
            crop = np.clip(
                crop.astype(np.float32) * gain + bias, 0, 255
            ).astype(np.uint8)
        if random.random() < 0.35:
            noise = np.random.normal(
                0.0, random.uniform(2.0, 9.0), crop.shape
            )
            crop = np.clip(
                crop.astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)
            uncertainty_multiplier *= 1.15
        if random.random() < 0.35:
            length = random.randint(3, 11)
            crop = motion_blur(crop, length, random.uniform(0.0, 180.0))
            uncertainty_multiplier *= 1.0 + 0.08 * length
        if random.random() < 0.40:
            crop, blue_multiplier = add_blue_occlusion(crop)
            uncertainty_multiplier *= blue_multiplier
        if random.random() < 0.15:
            crop = cv2.GaussianBlur(
                crop, (0, 0), sigmaX=random.uniform(0.5, 1.8)
            )
            uncertainty_multiplier *= 1.3
        return crop, uncertainty_multiplier

    def __getitem__(self, index: int):
        negative = index >= self.positive_count
        source_index = (
            (index - self.positive_count) * 7919
            if negative else index
        ) % self.positive_count
        item = self.items[source_index]
        image = self.read_image(item["path"])
        if image is None:
            return self.__getitem__((index + 977) % len(self))

        if negative:
            center, side = self.negative_geometry(item, image.shape[:2])
        else:
            center, side = self.proposal_geometry(item)
        crop, forward, _inverse = warp_gate_crop(
            image, center, side, CROP_SIZE
        )
        uncertainty_multiplier = 1.0
        if self.train:
            crop, uncertainty_multiplier = self.augment(crop)

        heatmap = np.zeros(
            (8, HEATMAP_SIZE, HEATMAP_SIZE), np.float32
        )
        offsets = np.zeros(
            (16, HEATMAP_SIZE, HEATMAP_SIZE), np.float32
        )
        offset_mask = np.zeros(
            (8, HEATMAP_SIZE, HEATMAP_SIZE), np.float32
        )
        corner_xy = np.zeros((8, 2), np.float32)
        visibility = np.zeros(8, np.float32)
        presence = 0.0 if negative else 1.0
        source_px_per_crop_px = float(side / CROP_SIZE)
        sigma_target = np.full(8, 12.0, np.float32)
        if not negative:
            corner_xy = transform_points(item["points"], forward)
            visibility = item["visibility"].copy()
            visibility *= (
                (corner_xy[:, 0] >= 0.0)
                & (corner_xy[:, 0] < CROP_SIZE)
                & (corner_xy[:, 1] >= 0.0)
                & (corner_xy[:, 1] < CROP_SIZE)
            )
            for corner_class in range(8):
                if visibility[corner_class] > 0.5:
                    splat_corner(
                        heatmap,
                        offsets,
                        offset_mask,
                        corner_class,
                        corner_xy[corner_class],
                    )
            base_sigma = np.clip(
                0.75 / max(source_px_per_crop_px, 1e-4), 0.65, 12.0
            )
            sigma_target.fill(
                float(np.clip(
                    base_sigma * uncertainty_multiplier, 0.65, 12.0
                ))
            )

        orange = orange_channel(crop)
        network_input = np.concatenate([
            crop.astype(np.float32) / 255.0,
            orange[..., None],
            self.prior[..., None],
        ], axis=2).transpose(2, 0, 1)
        return {
            "img": torch.from_numpy(network_input),
            "hm": torch.from_numpy(heatmap),
            "off": torch.from_numpy(offsets),
            "om": torch.from_numpy(offset_mask),
            "corner_xy": torch.from_numpy(corner_xy),
            "visibility": torch.from_numpy(visibility),
            "sigma_target": torch.from_numpy(sigma_target),
            "presence": torch.tensor(presence, dtype=torch.float32),
            "source_scale": torch.tensor(
                source_px_per_crop_px, dtype=torch.float32
            ),
        }


@torch.no_grad()
def evaluate(
    model: CropGateNet,
    loader: DataLoader,
    device: str,
    max_batches: int = 160,
) -> dict[str, float]:
    model.eval()
    errors = []
    corner_gt = 0
    presence_hits = 0
    positives = 0
    false_positives = 0
    negatives = 0
    sigma_pairs = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        image = batch["img"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(image)
        batch_size = image.shape[0]
        presence_probability = torch.sigmoid(
            output["presence"].float()
        ).cpu().numpy()
        for sample_index in range(batch_size):
            target_presence = float(batch["presence"][sample_index])
            if target_presence > 0.5:
                positives += 1
                presence_hits += presence_probability[sample_index] >= 0.5
            else:
                negatives += 1
                false_positives += presence_probability[sample_index] >= 0.5
                continue
            decoded = decode_crop_corners(
                {
                    key: value[sample_index:sample_index + 1]
                    for key, value in output.items()
                },
                CROP_SIZE,
            )
            predicted = np.asarray(decoded["corners"])
            scores = np.asarray(decoded["scores"])
            predicted_visibility = np.asarray(decoded["visibility"])
            predicted_sigma = np.asarray(decoded["sigma_crop_px"])
            target = batch["corner_xy"][sample_index].numpy()
            visible = batch["visibility"][sample_index].numpy() > 0.5
            source_scale = float(batch["source_scale"][sample_index])
            for corner_class in np.flatnonzero(visible):
                corner_gt += 1
                if (
                    presence_probability[sample_index] < 0.20
                    or scores[corner_class] < 0.05
                    or predicted_visibility[corner_class] < 0.20
                ):
                    continue
                crop_error = float(np.linalg.norm(
                    predicted[corner_class] - target[corner_class]
                ))
                errors.append(crop_error * source_scale)
                sigma_pairs.append((
                    crop_error * source_scale,
                    predicted_sigma[corner_class] * source_scale,
                ))
    error_array = np.asarray(errors or [999.0], np.float32)
    recall_2 = sum(error <= 2.0 for error in errors) / max(corner_gt, 1)
    recall_4 = sum(error <= 4.0 for error in errors) / max(corner_gt, 1)
    recall_8 = sum(error <= 8.0 for error in errors) / max(corner_gt, 1)
    p90 = float(np.percentile(error_array, 90))
    if sigma_pairs:
        calibration = np.asarray(sigma_pairs, np.float32)
        sigma_coverage = float(
            np.mean(calibration[:, 0] <= 2.0 * calibration[:, 1])
        )
    else:
        sigma_coverage = 0.0
    false_positive_rate = false_positives / max(negatives, 1)
    metrics = {
        "corner_gt": corner_gt,
        "corner_matches": len(errors),
        "corner_candidate_recall": len(errors) / max(corner_gt, 1),
        "corner_recall_2px": recall_2,
        "corner_recall_4px": recall_4,
        "corner_recall_8px": recall_8,
        "corner_px_median": float(np.median(error_array)),
        "corner_px_p90": p90,
        "presence_recall": presence_hits / max(positives, 1),
        "negative_false_positive_rate": false_positive_rate,
        "sigma_2x_coverage": sigma_coverage,
    }
    metrics["selection_score"] = (
        recall_4
        + 0.25 * recall_8
        - 0.001 * min(p90, 100.0)
        - 0.10 * false_positive_rate
    )
    model.train()
    return metrics


def parse_path_maps(values: list[str] | None) -> list[tuple[str, str]]:
    result = []
    for value in values or []:
        old, new = value.split("::", 1)
        result.append((old.rstrip("\\/"), new.rstrip("\\/")))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", action="append", required=True)
    parser.add_argument("--path-map", action="append", default=None)
    parser.add_argument("--val-session", action="append", default=None)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--negative-ratio", type=float, default=0.20)
    parser.add_argument("--init", default=None)
    parser.add_argument("--tag", default="v11crop")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    validation_sessions = tuple(args.val_session or VAL_EPISODES)
    label_dirs = [Path(value).expanduser().resolve() for value in args.labels_dir]
    files = []
    seen_files = set()
    for label_dir in label_dirs:
        for path in sorted(label_dir.glob("*.npz")):
            resolved = path.resolve()
            if resolved not in seen_files:
                files.append(resolved)
                seen_files.add(resolved)
    validation_files = [
        path for path in files
        if any(session in path.stem for session in validation_sessions)
    ]
    training_files = [
        path for path in files if path not in validation_files
    ]
    if not validation_files:
        raise SystemExit("no validation sessions matched; refusing leakage")
    path_maps = parse_path_maps(args.path_map)
    print(
        f"train files={len(training_files)} "
        f"val files={len(validation_files)}",
        flush=True,
    )
    training_set = CropGateDataset(
        training_files,
        train=True,
        path_maps=path_maps,
        negative_ratio=args.negative_ratio,
    )
    validation_set = CropGateDataset(
        validation_files,
        train=False,
        path_maps=path_maps,
        negative_ratio=args.negative_ratio,
    )
    print(
        f"train positives={training_set.positive_count} "
        f"negatives={training_set.negative_count}; "
        f"val variants={validation_set.positive_count} "
        f"negatives={validation_set.negative_count}",
        flush=True,
    )
    training_loader = DataLoader(
        training_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=min(3, args.workers),
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = CropGateNet().to(device)
    if args.init:
        checkpoint = torch.load(
            args.init, map_location=device, weights_only=False
        )
        missing, unexpected = initialize_crop_backbone(model, checkpoint)
        print(
            f"initialized crop backbone from {args.init}; "
            f"new/missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print(f"parameters={parameter_count / 1e6:.2f}M", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(training_loader),
        eta_min=1e-5,
    )
    scaler = torch.amp.GradScaler("cuda")

    output_dir = REPO / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"train_{args.tag}.jsonl"
    best_score = -float("inf")
    with log_path.open("a", encoding="utf-8") as log_file:
        for epoch in range(args.epochs):
            start = time.time()
            aggregates: dict[str, float] = defaultdict(float)
            for batch_index, batch in enumerate(training_loader):
                image = batch["img"].to(device, non_blocking=True)
                target_heatmap = batch["hm"].to(
                    device, non_blocking=True
                )
                target_offsets = batch["off"].to(
                    device, non_blocking=True
                )
                offset_mask = batch["om"].to(
                    device, non_blocking=True
                )
                target_visibility = batch["visibility"].to(
                    device, non_blocking=True
                )
                target_sigma = batch["sigma_target"].to(
                    device, non_blocking=True
                )
                target_presence = batch["presence"].to(
                    device, non_blocking=True
                )
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(image)
                    heatmap_loss = focal_heatmap_loss(
                        output["hm"], target_heatmap
                    )
                    expanded_mask = offset_mask.repeat_interleave(2, dim=1)
                    offset_loss = (
                        F.smooth_l1_loss(
                            output["off"] * expanded_mask,
                            target_offsets * expanded_mask,
                            reduction="sum",
                            beta=0.25,
                        )
                        / expanded_mask.sum().clamp(min=1.0)
                    )
                    visibility_loss = F.binary_cross_entropy_with_logits(
                        output["vis"], target_visibility
                    )
                    presence_loss = F.binary_cross_entropy_with_logits(
                        output["presence"], target_presence
                    )
                    visible_count = target_visibility.sum().clamp(min=1.0)
                    sigma_loss = (
                        F.smooth_l1_loss(
                            output["log_sigma"],
                            torch.log(target_sigma),
                            reduction="none",
                            beta=0.25,
                        )
                        * target_visibility
                    ).sum() / visible_count
                    loss = (
                        heatmap_loss
                        + offset_loss
                        + 0.20 * visibility_loss
                        + 0.35 * presence_loss
                        + 0.08 * sigma_loss
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                for name, value in (
                    ("hm", heatmap_loss),
                    ("off", offset_loss),
                    ("vis", visibility_loss),
                    ("presence", presence_loss),
                    ("sigma", sigma_loss),
                    ("loss", loss),
                ):
                    aggregates[name] += float(value.item())
                if batch_index % 150 == 0:
                    averages = " ".join(
                        f"{name}={value / (batch_index + 1):.4f}"
                        for name, value in aggregates.items()
                    )
                    print(
                        f"ep{epoch} {batch_index}/{len(training_loader)} "
                        f"{averages}",
                        flush=True,
                    )
            metrics = evaluate(model, validation_loader, device)
            record = {
                "epoch": epoch,
                "duration_s": round(time.time() - start, 1),
                **{
                    name: round(value / len(training_loader), 6)
                    for name, value in aggregates.items()
                },
                **{
                    name: round(float(value), 6)
                    for name, value in metrics.items()
                },
            }
            print("EVAL " + json.dumps(record), flush=True)
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "crop_size": CROP_SIZE,
                "corner_order": "apparent_inner_then_outer_TL_TR_BR_BL",
            }
            torch.save(
                checkpoint,
                output_dir / f"crop_gatenet_{args.tag}_last.pt",
            )
            if metrics["selection_score"] > best_score:
                best_score = metrics["selection_score"]
                torch.save(
                    checkpoint,
                    output_dir / f"crop_gatenet_{args.tag}_best.pt",
                )
                print(
                    f"saved best score={best_score:.4f} "
                    f"recall4={metrics['corner_recall_4px']:.3f} "
                    f"recall8={metrics['corner_recall_8px']:.3f}",
                    flush=True,
                )
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
