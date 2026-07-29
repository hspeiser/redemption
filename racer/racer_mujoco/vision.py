"""Vision pipeline for the pixel policy: gate-color filter, spaced frame ring, and an
index-based (frames stored once) replay buffer for distillation.

Obs = 3 x 84x84 filtered frames spaced `spacing` control steps apart [t, t-s, t-2s], newest first,
clamped at episode start. Filter = bright gate-yellow score + a faint luminance floor so clutter
blocks / the checkered ground stay dimly visible (parallax cues) while gates dominate."""
from __future__ import annotations
import numpy as np
import torch

IMG = 84


def gate_filter(rgb):
    """rgb uint8 (H,W,3) -> filtered uint8 (H,W). Gates (yellow: high R,G / low B) come out
    bright; everything else survives only as a faint luminance floor."""
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    gate = np.clip(np.minimum(r, g) - b, 0.0, 255.0) / 255.0
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    out = np.clip(gate * 1.8 + lum * 0.15, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


class FrameRing:
    """Per-env ring of the last 2*spacing+1 filtered frames + their buffer slot ids."""

    def __init__(self, spacing=5, n=3):
        self.spacing = spacing
        self.n = n
        self.keep = spacing * (n - 1) + 1
        self.frames = []
        self.slots = []

    def reset(self, frame, slot):
        self.frames = [frame]
        self.slots = [slot]

    def push(self, frame, slot):
        self.frames.append(frame); self.slots.append(slot)
        if len(self.frames) > self.keep:
            self.frames = self.frames[-self.keep:]
            self.slots = self.slots[-self.keep:]

    def _idx(self):
        L = len(self.frames)
        return [max(0, L - 1 - k * self.spacing) for k in range(self.n)]   # newest first

    def stack(self):
        return np.stack([self.frames[i] for i in self._idx()], 0)          # (n, H, W) uint8

    def stack_slots(self):
        return np.array([self.slots[i] for i in self._idx()], np.int64)


class VisionBuffer:
    """Frames stored ONCE (uint8 ring); entries reference frame slots by index — no 3x
    duplication. Entries: (frame idx triplet, proprio6, teacher action4)."""

    def __init__(self, cap=250_000, img=IMG, device="cuda"):
        self.frames = np.zeros((cap, img, img), np.uint8)
        self.fpos = 0
        self.idx = np.zeros((cap, 3), np.int64)
        self.proprio = np.zeros((cap, 9), np.float32)
        self.act = np.zeros((cap, 4), np.float32)
        self.pos = 0
        self.size = 0
        self.cap = cap
        self.device = device

    def add_frame(self, frame):
        s = self.fpos
        self.frames[s] = frame
        self.fpos = (self.fpos + 1) % self.cap
        return s

    def add_entry(self, idx3, proprio, act):
        i = self.pos
        self.idx[i] = idx3
        self.proprio[i] = proprio
        self.act[i] = act
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch):
        j = np.random.randint(0, self.size, batch)
        f = self.frames[self.idx[j].reshape(-1)].reshape(batch, 3, IMG, IMG)
        imgs = torch.as_tensor(f, device=self.device).float().div_(255.0)
        prop = torch.as_tensor(self.proprio[j], device=self.device)
        act = torch.as_tensor(self.act[j], device=self.device)
        return imgs, prop, act


def rand_shift(imgs, pad=4):
    """DrQ random-shift augmentation, vectorized via grid_sample (canonical implementation)."""
    import torch.nn.functional as F
    B, C, H, W = imgs.shape
    x = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    eps = 1.0 / (H + 2 * pad)
    ar = torch.linspace(-1.0 + eps, 1.0 - eps, H + 2 * pad, device=imgs.device,
                        dtype=imgs.dtype)[:H]
    ar = ar.unsqueeze(0).repeat(H, 1).unsqueeze(2)
    grid = torch.cat([ar, ar.transpose(1, 0)], dim=2).unsqueeze(0).repeat(B, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (B, 1, 1, 2), device=imgs.device,
                          dtype=imgs.dtype) * (2.0 / (H + 2 * pad))
    return F.grid_sample(x, grid + shift, padding_mode="zeros", align_corners=False)
