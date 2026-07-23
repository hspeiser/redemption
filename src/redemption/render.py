"""Procedural OpenCV renderer for orange gates + domain randomization.

Given a camera, gate geometry and a list of sampled poses, this renders a single
640x360 image with:

  * a procedural background (dark base + gradient + Perlin-ish noise + clutter),
  * each gate drawn as an extruded orange tube (front ring + darker side faces)
    with simple directional shading, composited far-to-near (painter's order),
  * photometric domain randomization (brightness/contrast, sensor noise, blur,
    motion blur, vignette) so the model generalizes toward real footage.

It also returns, per gate, the projected polygons needed by the labeller to
decide visibility/occlusion -- computed once, here, so datagen never re-projects.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import PinholeCamera
from .config import DotDict
from .gate import Gate
from .geometry import GatePose


@dataclass
class GateProjection:
    """Projected polygons + metadata for one gate (all pixel coords are float)."""

    front_outer: np.ndarray   # (4, 2) outer boundary, front face (Z_obj = 0)
    front_inner: np.ndarray   # (4, 2) inner opening = the keypoints
    back_outer: np.ndarray    # (4, 2) outer boundary, back face (Z_obj = depth)
    inner_in_front: np.ndarray  # (4,) bool, corner is in front of camera
    outer_in_front: np.ndarray  # (4,) bool, outer corner in front (8-kpt visibility)
    inner_depth: np.ndarray   # (4,) camera-frame Z of each inner corner
    center_depth: float       # gate center depth (for painter's sort)
    normal_cam: np.ndarray    # (3,) front-face normal in camera frame


def project_gate(camera: PinholeCamera, gate: Gate, pose: GatePose) -> GateProjection:
    """Project all corner sets of one gate to pixels (single source of truth)."""
    inner_cam = pose.transform(gate.inner_corners(0.0))
    outer_cam = pose.transform(gate.outer_corners(0.0))
    front_outer, _ = camera.project(outer_cam)
    front_inner, _ = camera.project(inner_cam)
    back_outer, _ = camera.project(pose.transform(gate.outer_corners(gate.depth)))
    normal_cam = pose.R @ np.array([0.0, 0.0, 1.0])
    return GateProjection(
        front_outer=front_outer,
        front_inner=front_inner,
        back_outer=back_outer,
        inner_in_front=inner_cam[:, 2] > 1e-6,
        outer_in_front=outer_cam[:, 2] > 1e-6,
        inner_depth=inner_cam[:, 2].copy(),
        center_depth=pose.depth,
        normal_cam=normal_cam,
    )


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------
def _value_noise(h: int, w: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Cheap smooth value-noise in [0,1] (low-res random field, bilinearly upsampled)."""
    small_h = max(2, int(h * scale))
    small_w = max(2, int(w * scale))
    grid = rng.random((small_h, small_w), dtype=np.float64)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)


def render_background(h: int, w: int, cfg: DotDict, rng: np.random.Generator) -> np.ndarray:
    """Build a procedural dark background as an (h, w, 3) BGR uint8 image."""
    bg = cfg.background
    base = float(rng.uniform(bg.base_gray_min, bg.base_gray_max))
    img = np.full((h, w), base, dtype=np.float64)

    if bool(bg.gradient):
        # random linear gradient across a random direction
        ang = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        g = (np.cos(ang) * xx / w + np.sin(ang) * yy / h)
        g = (g - g.min()) / (g.max() - g.min() + 1e-9)
        img += (g - 0.5) * rng.uniform(10, 40)

    if bool(bg.perlin_noise):
        scale = rng.uniform(bg.perlin_scale_min, bg.perlin_scale_max)
        noise = _value_noise(h, w, scale, rng)
        img += (noise - 0.5) * float(bg.perlin_strength)

    img = np.clip(img, 0, 255)
    bgr = np.repeat(img[:, :, None], 3, axis=2).astype(np.uint8)

    clut = bg.clutter
    if bool(clut.enabled):
        n = int(rng.integers(int(clut.count_min), int(clut.count_max) + 1))
        _draw_clutter(bgr, n, bool(clut.allow_orange), rng)
    return bgr


def _draw_clutter(img: np.ndarray, n: int, allow_orange: bool, rng: np.random.Generator) -> None:
    h, w = img.shape[:2]
    for _ in range(n):
        if allow_orange and rng.random() < 0.3:
            color = (int(rng.integers(0, 60)), int(rng.integers(60, 130)), int(rng.integers(180, 255)))
        else:
            # non-orange-ish clutter (avoid confusing the detector's colour cue)
            color = (int(rng.integers(0, 200)), int(rng.integers(0, 200)), int(rng.integers(0, 120)))
        shape = rng.integers(0, 3)
        if shape == 0:  # rectangle
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.rectangle(img, p1, p2, color, -1)
        elif shape == 1:  # circle
            c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.circle(img, c, int(rng.integers(3, 40)), color, -1)
        else:  # line
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.line(img, p1, p2, color, int(rng.integers(1, 6)))


# ---------------------------------------------------------------------------
# Gate drawing
# ---------------------------------------------------------------------------
def _poly(pts: np.ndarray) -> np.ndarray:
    """Sanitize a polygon to int32 pixels, clamped to a sane range for fillPoly."""
    p = np.nan_to_num(pts, nan=0.0, posinf=1e5, neginf=-1e5)
    p = np.clip(p, -1e4, 1e4)
    return np.round(p).astype(np.int32)


def _ring_mask(shape: tuple[int, int], outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [_poly(outer)], 255)
    hole = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(hole, [_poly(inner)], 255)
    mask[hole > 0] = 0
    return mask


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    f = float(np.clip(factor, 0.0, 1.0))
    return tuple(int(np.clip(c * f, 0, 255)) for c in color)


def draw_gate(
    canvas: np.ndarray,
    proj: GateProjection,
    gate: Gate,
    color_bgr: tuple[int, int, int],
    render_cfg: DotDict,
    light_dir: np.ndarray,
) -> None:
    """Draw one gate onto ``canvas`` (BGR uint8) in place."""
    shape = canvas.shape[:2]

    # Directional shading of the front face (simple Lambert w/ ambient term).
    sh = render_cfg.shading
    if bool(sh.enabled):
        ambient = float(sh.ambient)
        lambert = abs(float(np.dot(proj.normal_cam / (np.linalg.norm(proj.normal_cam) + 1e-9), light_dir)))
        front_factor = ambient + (1.0 - ambient) * lambert
    else:
        front_factor = 1.0

    front_color = _shade(color_bgr, front_factor)
    side_color = _shade(color_bgr, front_factor * gate.side_face_darkness)

    # 1) Side faces (extruded tube walls) -- drawn first so the front ring sits on top.
    if bool(render_cfg.draw_side_faces):
        for i in range(4):
            j = (i + 1) % 4
            quad = np.array(
                [proj.front_outer[i], proj.front_outer[j], proj.back_outer[j], proj.back_outer[i]]
            )
            cv2.fillPoly(canvas, [_poly(quad)], side_color)

    # 2) Front frame ring (outer minus inner opening).
    mask = _ring_mask(shape, proj.front_outer, proj.front_inner)
    canvas[mask > 0] = front_color


# ---------------------------------------------------------------------------
# Photometric domain randomization (applied to the final composite)
# ---------------------------------------------------------------------------
def _motion_blur(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(3, 11))
    kernel = np.zeros((k, k), dtype=np.float32)
    ang = rng.uniform(0, np.pi)
    cx = cy = (k - 1) / 2.0
    for t in np.linspace(-cx, cx, k):
        x = int(round(cx + t * np.cos(ang)))
        y = int(round(cy + t * np.sin(ang)))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def apply_photometric(img: np.ndarray, cfg: DotDict, rng: np.random.Generator) -> np.ndarray:
    """Apply brightness/contrast/noise/blur/vignette per ``[augment]`` config."""
    aug = cfg.augment
    if not bool(aug.enabled):
        return img
    out = img.astype(np.float32)

    # brightness + contrast
    b = 1.0 + rng.uniform(-aug.brightness, aug.brightness)
    c = 1.0 + rng.uniform(-aug.contrast, aug.contrast)
    mean = out.mean()
    out = (out - mean) * c + mean * b

    # vignette
    if rng.random() < float(aug.vignette_prob):
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        vig = 1.0 - float(aug.vignette_strength) * np.clip(r, 0, 1) ** 2
        out *= vig[:, :, None]

    out = np.clip(out, 0, 255).astype(np.uint8)

    # blur / motion blur (mutually exclusive-ish)
    if rng.random() < float(aug.blur_prob):
        ksz = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (ksz, ksz), 0)
    elif rng.random() < float(aug.motion_blur_prob):
        out = _motion_blur(out, rng)

    # gaussian sensor noise
    std = float(aug.gaussian_noise_std)
    if std > 0:
        noise = rng.normal(0, std, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def jitter_gate_color(base_bgr: tuple[int, int, int], cfg: DotDict, rng: np.random.Generator) -> tuple[int, int, int]:
    """Per-image jitter of the orange frame colour (hue/saturation)."""
    aug = cfg.augment
    if not bool(aug.enabled):
        return base_bgr
    hsv = cv2.cvtColor(np.uint8([[list(base_bgr)]]), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[0, 0, 0] = (hsv[0, 0, 0] + rng.uniform(-aug.gate_hue_jitter, aug.gate_hue_jitter)) % 180
    hsv[0, 0, 1] = np.clip(hsv[0, 0, 1] * (1.0 + rng.uniform(-aug.gate_sat_jitter, aug.gate_sat_jitter)), 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------
def render_scene(
    camera: PinholeCamera,
    gate: Gate,
    poses: list[GatePose],
    datagen_cfg: DotDict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[GateProjection]]:
    """Render a full scene at supersampled resolution, then downscale.

    Returns the final (H, W, 3) BGR image and the list of :class:`GateProjection`
    (in the SAME order as ``poses``, at final image resolution).
    """
    ss = max(1, int(datagen_cfg.render.supersample))
    H, W = camera.height, camera.width
    Hs, Ws = H * ss, W * ss

    # Supersampled camera => scale intrinsics.
    cam_ss = PinholeCamera(
        width=Ws, height=Hs,
        fx=camera.fx * ss, fy=camera.fy * ss, cx=camera.cx * ss, cy=camera.cy * ss,
        dist_coeffs=camera.dist_coeffs,
    )

    canvas = render_background(Hs, Ws, datagen_cfg, rng)

    # Per-image gate colour + light direction.
    color = jitter_gate_color(tuple(gate.color_bgr), datagen_cfg, rng)
    light_dir = rng.normal(size=3)
    light_dir /= np.linalg.norm(light_dir) + 1e-9

    projections_ss = [project_gate(cam_ss, gate, p) for p in poses]

    # Painter's order: farthest first.
    order = sorted(range(len(poses)), key=lambda i: -projections_ss[i].center_depth)
    for i in order:
        pj = projections_ss[i]
        # Skip gates entirely behind the camera.
        if not np.any(pj.inner_in_front):
            continue
        draw_gate(canvas, pj, gate, color, datagen_cfg.render, light_dir)

    # Downscale for anti-aliasing.
    if ss > 1:
        canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)

    canvas = apply_photometric(canvas, datagen_cfg, rng)

    # Return projections at FINAL resolution (divide the supersampled pixels).
    projections = [project_gate(camera, gate, p) for p in poses]
    return canvas, projections
