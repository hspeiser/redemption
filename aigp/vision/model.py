"""GateNet: multi-head gate perception network.

Inputs:  4ch image (RGB + orange-likelihood), 360x640
Outputs:
  - corner heatmaps  (8, 90, 160)  stride 4: classes 0-3 inner TL,TR,BR,BL
    (in gate-local generation order), 4-7 outer  -> sub-pixel via offsets
  - corner offsets   (16, 90, 160) per-class (dx, dy) within cell
  - pose head (global): drone position (3, /50m), drone rotation (6D),
    velocity (3, /20), next-gate position (3, /50m), active-gate logits (6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cbs(cin, cout, stride=1, k=3):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = cbs(c, c)
        self.b = cbs(c, c)

    def forward(self, x):
        return x + self.b(self.a(x))


class GateNet(nn.Module):
    def __init__(self, in_ch=4, n_gates=6):
        super().__init__()
        self.stem = cbs(in_ch, 24)                    # s1
        self.d1 = nn.Sequential(cbs(24, 48, 2), Block(48))     # s2
        self.d2 = nn.Sequential(cbs(48, 96, 2), Block(96))     # s4
        self.d3 = nn.Sequential(cbs(96, 192, 2), Block(192))   # s8
        self.d4 = nn.Sequential(cbs(192, 256, 2), Block(256))  # s16

        self.u3 = cbs(256 + 192, 128)                 # -> s8
        self.u2 = cbs(128 + 96, 96)                   # -> s4

        self.hm_head = nn.Sequential(cbs(96, 64), nn.Conv2d(64, 8, 1))
        self.off_head = nn.Sequential(cbs(96, 64), nn.Conv2d(64, 16, 1))
        # bias init so initial heatmap prob ~0.01 (focal-loss stability)
        nn.init.constant_(self.hm_head[-1].bias, -4.6)

        self.pose_fc = nn.Sequential(
            nn.Linear(256, 512), nn.SiLU(inplace=True),
            nn.Linear(512, 256), nn.SiLU(inplace=True),
        )
        self.head_pos = nn.Linear(256, 3)
        self.head_rot = nn.Linear(256, 6)
        self.head_vel = nn.Linear(256, 3)
        self.head_next_gate = nn.Linear(256, 3)
        self.head_gate_cls = nn.Linear(256, n_gates)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.d1(s1)
        s4 = self.d2(s2)
        s8 = self.d3(s4)
        s16 = self.d4(s8)

        u8 = self.u3(torch.cat([F.interpolate(s16, size=s8.shape[2:],
                                              mode="nearest"), s8], 1))
        u4 = self.u2(torch.cat([F.interpolate(u8, size=s4.shape[2:],
                                              mode="nearest"), s4], 1))
        hm = self.hm_head(u4)
        off = self.off_head(u4)

        g = F.adaptive_avg_pool2d(s16, 1).flatten(1)
        g = self.pose_fc(g)
        return {
            "hm": hm, "off": off,
            "pos": self.head_pos(g), "rot6": self.head_rot(g),
            "vel": self.head_vel(g), "next_gate": self.head_next_gate(g),
            "gate_cls": self.head_gate_cls(g),
        }


def rot6d_to_matrix(r6):
    """(B,6) -> (B,3,3) via Gram-Schmidt."""
    a, b = r6[:, :3], r6[:, 3:]
    x = F.normalize(a, dim=1)
    y = F.normalize(b - (x * b).sum(1, keepdim=True) * x, dim=1)
    z = torch.cross(x, y, dim=1)
    return torch.stack([x, y, z], dim=2)


def focal_heatmap_loss(logits, gt, alpha=2.0, beta=4.0, valid=None):
    """CenterNet penalty-reduced focal loss. gt is the gaussian-splatted map
    with exact 1.0 at corner cells. valid (B,1,H,W) zeroes the NEGATIVE loss
    inside ignore regions (gates present but with uncertain positions).

    Computed in fp32 regardless of autocast: the negative-term sum over
    ~1.8M cells overflows fp16 once predictions become confident."""
    logits = logits.float()
    gt = gt.float()
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (gt > 0.999).float()
    neg = 1.0 - pos
    if valid is not None:
        neg = neg * valid
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - gt) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    n_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n_pos
