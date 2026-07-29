"""Instance-aware gate-corner model for high-resolution proposal crops.

Unlike the full-frame GateNet, every input contains at most one proposed gate.
Corner classes are therefore apparent image order (TL, TR, BR, BL) for the
inner ring followed by the outer ring.  The model also predicts corner
visibility, corner uncertainty, and gate presence.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aigp.vision.model import Block, cbs


def canonical_quad(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return apparent TL,TR,BR,BL points and their source indices."""
    points = np.asarray(quad, np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    sums = points.sum(axis=1)
    diffs = points[:, 0] - points[:, 1]
    indices = np.asarray([
        np.argmin(sums),
        np.argmax(diffs),
        np.argmax(sums),
        np.argmin(diffs),
    ], np.int64)
    if len(set(indices.tolist())) != 4:
        return None
    return points[indices], indices


def orange_channel(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[..., 0]
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0
    hue_distance = np.minimum(np.abs(hue - 10.0), 180.0 - np.abs(hue - 10.0))
    hue_weight = np.clip(1.0 - hue_distance / 22.0, 0.0, 1.0)
    return (
        hue_weight
        * np.clip(saturation * 1.6 - 0.25, 0.0, 1.0)
        * np.clip(value * 1.4 - 0.1, 0.0, 1.0)
    ).astype(np.float32)


def crop_affine(
    center_xy: np.ndarray,
    side_px: float,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-frame->crop and crop->full-frame 2x3 affine transforms."""
    scale = output_size / max(float(side_px), 1.0)
    center = np.asarray(center_xy, np.float32)
    forward = np.asarray([
        [scale, 0.0, output_size * 0.5 - scale * center[0]],
        [0.0, scale, output_size * 0.5 - scale * center[1]],
    ], np.float32)
    inverse = cv2.invertAffineTransform(forward).astype(np.float32)
    return forward, inverse


def warp_gate_crop(
    bgr: np.ndarray,
    center_xy: np.ndarray,
    side_px: float,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward, inverse = crop_affine(center_xy, side_px, output_size)
    crop = cv2.warpAffine(
        bgr,
        forward,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(8, 8, 8),
    )
    return crop, forward, inverse


def transform_points(points: np.ndarray, affine: np.ndarray) -> np.ndarray:
    points = np.asarray(points, np.float32)
    homogeneous = np.concatenate(
        [points, np.ones((*points.shape[:-1], 1), np.float32)], axis=-1
    )
    return homogeneous @ np.asarray(affine, np.float32).T


def proposal_channel(size: int, sigma_fraction: float = 0.18) -> np.ndarray:
    """Gaussian channel marking the detector proposal centre."""
    coordinates = np.arange(size, dtype=np.float32) + 0.5
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    center = size * 0.5
    sigma = size * sigma_fraction
    return np.exp(
        -((xx - center) ** 2 + (yy - center) ** 2) / (2.0 * sigma * sigma)
    ).astype(np.float32)


class CropGateNet(nn.Module):
    """Small U-Net with grouped corners and calibrated confidence heads."""

    def __init__(self, in_ch: int = 5):
        super().__init__()
        self.stem = cbs(in_ch, 24)
        self.d1 = nn.Sequential(cbs(24, 48, 2), Block(48))
        self.d2 = nn.Sequential(cbs(48, 96, 2), Block(96))
        self.d3 = nn.Sequential(cbs(96, 192, 2), Block(192))
        self.d4 = nn.Sequential(cbs(192, 256, 2), Block(256))

        self.u3 = cbs(256 + 192, 128)
        self.u2 = cbs(128 + 96, 96)
        self.hm_head = nn.Sequential(cbs(96, 64), nn.Conv2d(64, 8, 1))
        self.off_head = nn.Sequential(cbs(96, 64), nn.Conv2d(64, 16, 1))
        nn.init.constant_(self.hm_head[-1].bias, -4.6)

        self.quality_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 128),
            nn.SiLU(inplace=True),
        )
        self.vis_head = nn.Linear(128, 8)
        self.sigma_head = nn.Linear(128, 8)
        self.presence_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s1 = self.stem(x)
        s2 = self.d1(s1)
        s4 = self.d2(s2)
        s8 = self.d3(s4)
        s16 = self.d4(s8)
        u8 = self.u3(torch.cat([
            F.interpolate(s16, size=s8.shape[2:], mode="nearest"),
            s8,
        ], dim=1))
        u4 = self.u2(torch.cat([
            F.interpolate(u8, size=s4.shape[2:], mode="nearest"),
            s4,
        ], dim=1))
        quality = self.quality_fc(F.adaptive_avg_pool2d(s16, 1).flatten(1))
        return {
            "hm": self.hm_head(u4),
            "off": self.off_head(u4),
            "vis": self.vis_head(quality),
            "log_sigma": self.sigma_head(quality).clamp(-2.5, 4.0),
            "presence": self.presence_head(quality).squeeze(1),
        }


def initialize_crop_backbone(
    model: CropGateNet,
    checkpoint: dict,
) -> tuple[list[str], list[str]]:
    """Load shape-compatible GateNet backbone weights, excluding old heads."""
    source = checkpoint["model"] if "model" in checkpoint else checkpoint
    target = model.state_dict()
    reusable = {
        key: value
        for key, value in source.items()
        if key in target
        and target[key].shape == value.shape
        and not key.startswith(("hm_head", "off_head"))
    }
    stem_key = "stem.0.weight"
    if (
        stem_key in source
        and stem_key in target
        and source[stem_key].shape[0] == target[stem_key].shape[0]
        and source[stem_key].shape[2:] == target[stem_key].shape[2:]
        and source[stem_key].shape[1] + 1 == target[stem_key].shape[1]
    ):
        expanded = target[stem_key].clone()
        expanded[:, :source[stem_key].shape[1]] = source[stem_key]
        expanded[:, -1:] = source[stem_key].mean(dim=1, keepdim=True)
        reusable[stem_key] = expanded
    result = model.load_state_dict(reusable, strict=False)
    return list(result.missing_keys), list(result.unexpected_keys)


@torch.no_grad()
def decode_crop_corners(
    output: dict[str, torch.Tensor],
    crop_size: int,
    inverse_affine: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Decode one crop output into crop or full-frame pixel coordinates."""
    heatmap = torch.sigmoid(output["hm"][0].float())
    offsets = output["off"][0].float()
    height, width = heatmap.shape[-2:]
    stride_x = crop_size / width
    stride_y = crop_size / height
    corners = np.zeros((8, 2), np.float32)
    scores = np.zeros(8, np.float32)
    for corner_class in range(8):
        flat_index = int(torch.argmax(heatmap[corner_class]).item())
        row, column = divmod(flat_index, width)
        du = float(offsets[2 * corner_class, row, column])
        dv = float(offsets[2 * corner_class + 1, row, column])
        corners[corner_class] = (
            (column + du) * stride_x,
            (row + dv) * stride_y,
        )
        scores[corner_class] = float(heatmap[corner_class, row, column])
    if inverse_affine is not None:
        corners = transform_points(corners, inverse_affine)
    return {
        "corners": corners,
        "scores": scores,
        "visibility": torch.sigmoid(output["vis"][0]).cpu().numpy(),
        "sigma_crop_px": torch.exp(output["log_sigma"][0]).cpu().numpy(),
        "presence": float(torch.sigmoid(output["presence"][0]).item()),
    }
