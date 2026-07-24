"""Train GateNet on the auto-labeled dataset.

Run with the CUDA venv:
    .venv-train\\Scripts\\python.exe scripts\\train_net.py
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.vision.model import GateNet, rot6d_to_matrix, focal_heatmap_loss

W, H = 640, 360
HW, HH = W // 4, H // 4          # heatmap size 160 x 90
POS_SCALE, VEL_SCALE = 50.0, 20.0
VAL_EPISODES = ("rc_20260723_024016", "rc_20260723_022140",
                "rc_20260723_022654",
                "rc_20260724_003101")   # VQ2 holdout (full fast lap)


def quat_to_R(q_wxyz):
    w, x, y, z = q_wxyz.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def orange_channel(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    hue_d = np.minimum(np.abs(h - 10.0), np.abs(h - 190.0) % 180)
    hue_w = np.clip(1.0 - hue_d / 22.0, 0, 1)
    return (hue_w * np.clip(s * 1.6 - 0.25, 0, 1) * np.clip(v * 1.4 - 0.1, 0, 1)) \
        .astype(np.float32)


class GateDataset(Dataset):
    def __init__(self, files, train=True, path_map=None):
        """path_map: optional 'old_prefix::new_prefix' applied to stored
        frame paths (for training on a machine other than the recorder's)."""
        self.train = train
        self.items = []
        pm = None
        if path_map:
            old, new = path_map.split("::", 1)
            pm = (old.rstrip("\\/"), new.rstrip("\\/"))
        for f in files:
            d = np.load(f, allow_pickle=False)
            if len(d["path"]) == 0:
                continue
            # materialize each array EXACTLY ONCE: every d["key"] access on an
            # NpzFile decompresses the whole array anew, and row slices pin
            # their parent — doing it per-row exploded RAM to ~20 GB.
            paths = d["path"]
            pos, vel = d["pos"], d["vel"]
            gate_idx, next_gate = d["gate_idx"], d["next_gate_pos"]
            inner, outer = d["inner"], d["outer"]
            vis_inner, vis_outer = d["vis_inner"], d["vis_outer"]
            pv_arr = d["pose_valid"] if "pose_valid" in d else \
                np.ones(len(paths), np.float32)
            R = quat_to_R(d["quat"]).astype(np.float32)
            ig_by_frame = {}
            if "ignore_frame_idx" in d:
                for box, fi in zip(d["ignore_boxes"], d["ignore_frame_idx"]):
                    ig_by_frame.setdefault(int(fi), []).append(box)
            n = len(paths)
            for i in range(n):
                p = str(paths[i])
                if pm is not None and p.startswith(pm[0]):
                    p = pm[1] + p[len(pm[0]):]
                    if "/" in pm[1]:          # POSIX target: fix separators
                        p = p.replace("\\", "/")
                self.items.append({
                    "path": p,
                    "pos": pos[i], "R": R[i],
                    "vel": vel[i], "gate_idx": int(gate_idx[i]),
                    "next_gate_pos": next_gate[i],
                    "inner": inner[i], "outer": outer[i],
                    "vis_inner": vis_inner[i], "vis_outer": vis_outer[i],
                    "pv": float(pv_arr[i]),
                    "ignore": np.array(ig_by_frame.get(i, np.zeros((0, 4))),
                                       np.float32),
                })

    def __len__(self):
        return len(self.items)

    def _splat(self, hm, off, om, cls, uv):
        u, v = uv[0] / 4.0, uv[1] / 4.0
        ci, cj = int(v), int(u)
        if not (0 <= ci < HH and 0 <= cj < HW):
            return
        sigma = 2.0
        rad = 5
        i0, i1 = max(0, ci - rad), min(HH, ci + rad + 1)
        j0, j1 = max(0, cj - rad), min(HW, cj + rad + 1)
        ys, xs = np.mgrid[i0:i1, j0:j1]
        g = np.exp(-(((xs - u) ** 2 + (ys - v) ** 2) / (2 * sigma * sigma)))
        hm[cls, i0:i1, j0:j1] = np.maximum(hm[cls, i0:i1, j0:j1], g)
        hm[cls, ci, cj] = 1.0
        off[2 * cls, ci, cj] = u - cj
        off[2 * cls + 1, ci, cj] = v - ci
        om[cls, ci, cj] = 1.0

    def __getitem__(self, idx):
        it = self.items[idx]
        bgr = cv2.imread(it["path"])
        if bgr is None:
            return self.__getitem__((idx + 977) % len(self))
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H))

        inner = it["inner"].copy()
        outer = it["outer"].copy()
        vis_inner = it["vis_inner"].copy()
        vis_outer = it["vis_outer"].copy()
        ignore_boxes = it["ignore"].copy()
        pose_valid = it.get("pv", 1.0)

        if self.train:
            # ---- zoom augmentation: big-gate views are rare in real data,
            # so synthesize them by crop-zooming. Cropping changes the
            # effective camera, so pose-head losses are masked (pose_valid=0).
            if np.random.rand() < 0.5:
                s = np.random.uniform(1.3, 2.4)
                cw, ch = W / s, H / s
                # bias crop center toward a random labeled gate if any
                centers = []
                for gi in range(inner.shape[0]):
                    if vis_inner[gi].any() or vis_outer[gi].any():
                        c = np.nanmean(outer[gi], axis=0)
                        if np.isfinite(c).all():
                            centers.append(c)
                if centers and np.random.rand() < 0.8:
                    cx0, cy0 = centers[np.random.randint(len(centers))]
                else:
                    cx0 = np.random.uniform(cw / 2, W - cw / 2)
                    cy0 = np.random.uniform(ch / 2, H - ch / 2)
                x0 = float(np.clip(cx0 - cw / 2, 0, W - cw))
                y0 = float(np.clip(cy0 - ch / 2, 0, H - ch))
                crop = bgr[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
                bgr = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)
                sx = W / crop.shape[1]
                sy = H / crop.shape[0]
                for arr in (inner, outer):
                    arr[..., 0] = (arr[..., 0] - x0) * sx
                    arr[..., 1] = (arr[..., 1] - y0) * sy
                vis_inner &= ((inner[..., 0] >= 0) & (inner[..., 0] < W)
                              & (inner[..., 1] >= 0) & (inner[..., 1] < H))
                vis_outer &= ((outer[..., 0] >= 0) & (outer[..., 0] < W)
                              & (outer[..., 1] >= 0) & (outer[..., 1] < H))
                if len(ignore_boxes):
                    ignore_boxes[:, [0, 2]] = (ignore_boxes[:, [0, 2]] - x0) * sx
                    ignore_boxes[:, [1, 3]] = (ignore_boxes[:, [1, 3]] - y0) * sy
                pose_valid = 0.0
            # photometric jitter
            gain = np.random.uniform(0.7, 1.3)
            bias = np.random.uniform(-20, 20)
            bgr = np.clip(bgr.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 6, bgr.shape).astype(np.float32)
                bgr = np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        orange = orange_channel(bgr)
        img = np.concatenate([bgr.astype(np.float32) / 255.0,
                              orange[..., None]], axis=2).transpose(2, 0, 1)

        hm = np.zeros((8, HH, HW), np.float32)
        off = np.zeros((16, HH, HW), np.float32)
        om = np.zeros((8, HH, HW), np.float32)
        valid = np.ones((1, HH, HW), np.float32)   # 0 inside ignore regions
        for (u0, v0, u1, v1) in ignore_boxes:
            i0 = max(0, int(v0 / 4)); i1 = min(HH, int(v1 / 4) + 1)
            j0 = max(0, int(u0 / 4)); j1 = min(HW, int(u1 / 4) + 1)
            if i1 > i0 and j1 > j0:
                valid[0, i0:i1, j0:j1] = 0.0
        n_g = inner.shape[0]
        for gi in range(n_g):
            span = np.nanmax(outer[gi], 0) - np.nanmin(outer[gi], 0)
            if not np.isfinite(span).all() or max(span) < 6.0:
                continue
            for c in range(4):
                if vis_inner[gi, c]:
                    self._splat(hm, off, om, c, inner[gi, c])
                if vis_outer[gi, c]:
                    self._splat(hm, off, om, 4 + c, outer[gi, c])

        return {
            "valid": torch.from_numpy(valid),
            "pose_valid": torch.tensor(pose_valid, dtype=torch.float32),
            "img": torch.from_numpy(img),
            "hm": torch.from_numpy(hm),
            "off": torch.from_numpy(off),
            "om": torch.from_numpy(om),
            "pos": torch.from_numpy(it["pos"] / POS_SCALE),
            "rot6": torch.from_numpy(
                np.concatenate([it["R"][:, 0], it["R"][:, 1]])),
            "vel": torch.from_numpy(it["vel"] / VEL_SCALE),
            "next_gate": torch.from_numpy(it["next_gate_pos"] / POS_SCALE),
            "gate_idx": torch.tensor(it["gate_idx"]),
        }


@torch.no_grad()
def decode_corners(hm_logits, off, thresh=0.25, topk=12):
    """-> list per class of (u, v, score) arrays in full-res pixels."""
    p = torch.sigmoid(hm_logits)
    pmax = F.max_pool2d(p, 3, 1, 1)
    keep = (p == pmax) & (p > thresh)
    out = []
    for c in range(p.shape[0]):
        idx = keep[c].nonzero(as_tuple=False)
        if len(idx) > topk:
            scores = p[c][idx[:, 0], idx[:, 1]]
            sel = scores.topk(topk).indices
            idx = idx[sel]
        pts = []
        for (ci, cj) in idx.tolist():
            du = off[2 * c, ci, cj].item()
            dv = off[2 * c + 1, ci, cj].item()
            pts.append(((cj + du) * 4.0, (ci + dv) * 4.0,
                        p[c, ci, cj].item()))
        out.append(pts)
    return out


@torch.no_grad()
def evaluate(model, loader, device, max_batches=120):
    model.eval()
    px_errs = []
    pos_errs, rot_errs, vel_errs, ng_errs, gate_hits, n_frames = [], [], [], [], 0, 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        img = batch["img"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(img)
        B = img.shape[0]
        Rp = rot6d_to_matrix(out["rot6"].float())
        Rg = rot6d_to_matrix(batch["rot6"].to(device).float())
        tr = torch.einsum("bij,bij->b", Rp, Rg).clamp(-1, 3)
        rot_deg = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
        rot_errs += rot_deg.cpu().tolist()
        pos_errs += (torch.norm(out["pos"] - batch["pos"].to(device), dim=1)
                     * POS_SCALE).cpu().tolist()
        vel_errs += (torch.norm(out["vel"] - batch["vel"].to(device), dim=1)
                     * VEL_SCALE).cpu().tolist()
        ng_errs += (torch.norm(out["next_gate"] - batch["next_gate"].to(device),
                               dim=1) * POS_SCALE).cpu().tolist()
        gate_hits += (out["gate_cls"].argmax(1).cpu()
                      == batch["gate_idx"]).sum().item()
        n_frames += B
        # corner localization on a subset of the batch
        for b in range(min(B, 4)):
            dec = decode_corners(out["hm"][b].float(), out["off"][b].float())
            gt_hm = batch["hm"][b]
            gt_om = batch["om"][b]
            gt_off = batch["off"][b]
            for c in range(8):
                cells = gt_om[c].nonzero(as_tuple=False)
                for (ci, cj) in cells.tolist():
                    gu = (cj + gt_off[2 * c, ci, cj].item()) * 4.0
                    gv = (ci + gt_off[2 * c + 1, ci, cj].item()) * 4.0
                    best = None
                    for (u, v, s) in dec[c]:
                        d = math.hypot(u - gu, v - gv)
                        if best is None or d < best:
                            best = d
                    if best is not None and best < 8.0:
                        px_errs.append(best)
    model.train()
    px = np.array(px_errs) if px_errs else np.array([99.0])
    return {
        "corner_px_median": float(np.median(px)),
        "corner_px_p90": float(np.percentile(px, 90)),
        "corner_matches": len(px_errs),
        "pos_err_m": float(np.median(pos_errs)),
        "rot_err_deg": float(np.median(rot_errs)),
        "vel_err_ms": float(np.median(vel_errs)),
        "next_gate_err_m": float(np.median(ng_errs)),
        "gate_cls_acc": gate_hits / max(n_frames, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", default=None, help="checkpoint to init from")
    ap.add_argument("--tag", default=None,
                    help="version tag: checkpoints/logs get _<tag> names")
    ap.add_argument("--path-map", default=None,
                    help="old_prefix::new_prefix remap for frame paths")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = "cuda"
    labels_dir = REPO / "data" / "labels"
    files = sorted(labels_dir.glob("*.npz"))
    val_f = [f for f in files if f.stem in VAL_EPISODES]
    train_f = [f for f in files if f.stem not in VAL_EPISODES]
    if not val_f:
        val_f = files[::8]
        train_f = [f for f in files if f not in val_f]
    print(f"train episodes: {len(train_f)}  val episodes: {len(val_f)}", flush=True)

    ds_tr = GateDataset(train_f, train=True, path_map=args.path_map)
    ds_va = GateDataset(val_f, train=False, path_map=args.path_map)
    print(f"train frames: {len(ds_tr)}  val frames: {len(ds_va)}", flush=True)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0, drop_last=True)
    va_workers = min(2, args.workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                       num_workers=va_workers, pin_memory=True,
                       persistent_workers=va_workers > 0)

    model = GateNet().to(device)
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"resumed from {args.resume} (epoch {ck['epoch']})", flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"params: {n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(dl_tr), eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    out_dir = REPO / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    log_f = open(out_dir / f"train_log{sfx}.jsonl", "a")
    best = 1e9

    for ep in range(args.epochs):
        t0 = time.time()
        agg = {}
        for bi, batch in enumerate(dl_tr):
            img = batch["img"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(img)
                gt_hm = batch["hm"].to(device, non_blocking=True)
                gt_off = batch["off"].to(device, non_blocking=True)
                gt_om = batch["om"].to(device, non_blocking=True)
                gt_valid = batch["valid"].to(device, non_blocking=True)
                l_hm = focal_heatmap_loss(out["hm"], gt_hm, valid=gt_valid)
                om2 = gt_om.repeat_interleave(2, dim=1)
                l_off = (F.l1_loss(out["off"] * om2, gt_off * om2,
                                   reduction="sum")
                         / om2.sum().clamp(min=1.0))
                pv = batch["pose_valid"].to(device)
                pvn = pv.sum().clamp(min=1.0)

                def masked_mse(a, b):
                    return (((a - b) ** 2).mean(dim=1) * pv).sum() / pvn

                l_pos = masked_mse(out["pos"], batch["pos"].to(device))
                l_rot = masked_mse(out["rot6"], batch["rot6"].to(device))
                l_vel = masked_mse(out["vel"], batch["vel"].to(device))
                l_ng = masked_mse(out["next_gate"],
                                  batch["next_gate"].to(device))
                l_cls = (F.cross_entropy(out["gate_cls"],
                                         batch["gate_idx"].to(device),
                                         reduction="none") * pv).sum() / pvn
                loss = (l_hm + l_off + 2.0 * l_pos + 2.0 * l_rot
                        + 0.5 * l_vel + 2.0 * l_ng + 0.5 * l_cls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in [("hm", l_hm), ("off", l_off), ("pos", l_pos),
                         ("rot", l_rot), ("ng", l_ng), ("loss", loss)]:
                agg[k] = agg.get(k, 0.0) + v.item()
            if bi % 200 == 0:
                print(f"ep{ep} it{bi}/{len(dl_tr)} "
                      + " ".join(f"{k}={agg[k]/(bi+1):.4f}" for k in agg),
                      flush=True)
        metrics = evaluate(model, dl_va, device)
        dur = time.time() - t0
        rec = {"epoch": ep, "dur_s": round(dur, 1),
               **{k: round(v / len(dl_tr), 5) for k, v in agg.items()},
               **{k: round(v, 4) for k, v in metrics.items()}}
        print("EVAL " + json.dumps(rec), flush=True)
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        torch.save({"model": model.state_dict(), "epoch": ep,
                    "metrics": metrics}, out_dir / f"gatenet{sfx}_last.pt")
        if metrics["corner_px_median"] < best:
            best = metrics["corner_px_median"]
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "metrics": metrics}, out_dir / f"gatenet{sfx}_best.pt")
            print(f"saved best (corner median {best:.3f}px)", flush=True)

    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
