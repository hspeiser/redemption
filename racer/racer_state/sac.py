"""SAC agent for state observations: replay buffer + twin-Q + auto entropy temperature.

Self-contained (buffer + learner in one class) for the single-process online loop. State vectors
are tiny, so the whole buffer lives on-device as float tensors.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn.functional as F

from .nets import build_actor, build_critic


class ReplayBuffer:
    def __init__(self, cfg, device, cap=None):
        cap = cap or cfg.buffer_capacity
        self.cap = cap
        self.device = device
        self.o = torch.zeros((cap, cfg.obs_dim), dtype=torch.float32, device=device)
        self.a = torch.zeros((cap, cfg.act_dim), dtype=torch.float32, device=device)
        self.r = torch.zeros((cap, 1), dtype=torch.float32, device=device)
        self.no = torch.zeros((cap, cfg.obs_dim), dtype=torch.float32, device=device)
        self.d = torch.zeros((cap, 1), dtype=torch.float32, device=device)
        # per-transition bootstrap discount: gamma for 1-step, gamma^k for k-step packed transitions
        self.g = torch.full((cap, 1), cfg.gamma, dtype=torch.float32, device=device)
        # realized discounted return-to-go (0 if unknown) — used by advantage-filtered self-imitation
        self.ret = torch.zeros((cap, 1), dtype=torch.float32, device=device)
        self._gamma = cfg.gamma
        self.pos = 0
        self.size = 0

    def add(self, o, a, r, no, d, g=None, ret=0.0):
        i = self.pos
        self.o[i] = torch.as_tensor(o, device=self.device)
        self.a[i] = torch.as_tensor(a, device=self.device)
        self.r[i] = float(r)
        self.no[i] = torch.as_tensor(no, device=self.device)
        self.d[i] = float(d)
        self.g[i] = self._gamma if g is None else float(g)
        self.ret[i] = float(ret)
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def add_batch(self, o, a, r, no, d, g, ret):
        """Vectorized bulk insert (numpy arrays, shape (B, ...)). One slice-write per field instead
        of per-transition Python adds (~100x faster on the add path). Handles ring wrap-around."""
        B = len(o)
        cols = {"o": np.asarray(o, np.float32), "a": np.asarray(a, np.float32),
                "r": np.asarray(r, np.float32).reshape(-1, 1),
                "no": np.asarray(no, np.float32),
                "d": np.asarray(d, np.float32).reshape(-1, 1),
                "g": np.asarray(g, np.float32).reshape(-1, 1),
                "ret": np.asarray(ret, np.float32).reshape(-1, 1)}
        start = self.pos
        first = min(B, self.cap - start)
        for name, arr in cols.items():
            dst = getattr(self, name)
            t = torch.as_tensor(arr, device=self.device)
            dst[start:start + first] = t[:first]
            if first < B:                       # wrap
                dst[0:B - first] = t[first:]
        self.pos = (start + B) % self.cap
        self.size = min(self.size + B, self.cap)

    def sample(self, batch):
        idx = torch.randint(0, self.size, (batch,), device=self.device)
        return (self.o[idx], self.a[idx], self.r[idx], self.no[idx], self.d[idx],
                self.g[idx], self.ret[idx])


class SAC:
    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.actor = build_actor(cfg).to(device)
        # Polyak-averaged actor: slow EMA of actor weights, used for greedy eval/deploy.
        # Smooths per-batch policy churn -> stable evals instead of 19/20 <-> 0/20 flip-flops.
        self.actor_ema = copy.deepcopy(self.actor).to(device)
        for p in self.actor_ema.parameters():
            p.requires_grad_(False)
        self.critic = build_critic(cfg).to(device)
        self.critic_tgt = copy.deepcopy(self.critic).to(device)
        for p in self.critic_tgt.parameters():
            p.requires_grad_(False)
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), cfg.lr)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), cfg.lr)
        self.t_opt = torch.optim.Adam([self.log_alpha], cfg.lr)
        self.buf = ReplayBuffer(cfg, device)
        self.elite = ReplayBuffer(cfg, device, cap=cfg.elite_capacity)   # successful/close trajectories

    def add_elite(self, transitions):
        """transitions: list of (o, a, r, no, term[, g, ret]). Oversampled during updates; when the
        return-to-go (ret) is provided it also feeds advantage-filtered self-imitation."""
        for tr in transitions:
            self.elite.add(*tr)

    @torch.no_grad()
    def act(self, obs, deterministic=False, ema=False):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        net = self.actor_ema if ema else self.actor
        return net.act(o, deterministic).squeeze(0).cpu().numpy()

    def _mixed_batch(self, batch):
        """Draw elite_frac of the batch from the elite buffer (if it has enough), rest from main."""
        ne = int(batch * self.cfg.elite_frac)
        if ne > 0 and self.elite.size >= ne:
            nm = batch - ne
            bm = self.buf.sample(nm)
            be = self.elite.sample(ne)
            return tuple(torch.cat([bm[i], be[i]], 0) for i in range(7))
        return self.buf.sample(batch)

    def update(self, metrics=True, bc=None, bc_weight=0.0, critic_only=False):
        cfg = self.cfg
        o, a, r, no, d, g, _ = self._mixed_batch(cfg.batch_size)
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            na, nlogp = self.actor.sample(no)
            qt = self.critic_tgt(no, na).min(0).values
            target = r + (1 - d) * g * (qt - alpha * nlogp)   # g = gamma^k (n-step aware)
        q = self.critic(o, a)
        c_loss = sum(F.mse_loss(q[i], target) for i in range(q.shape[0]))
        self.c_opt.zero_grad(set_to_none=True); c_loss.backward(); self.c_opt.step()

        if critic_only:                    # warm the critic without disturbing the (BC-trained) actor
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                    tp.mul_(1 - cfg.tau).add_(cfg.tau * p)
            return {} if not metrics else {"c_loss": float(c_loss.detach())}

        pa, plogp = self.actor.sample(o)
        qpi = self.critic(o, pa).min(0).values
        a_loss = (alpha.detach() * plogp - qpi).mean()
        if bc is not None and bc_weight > 0:      # anchor the actor to expert demos (SAC+BC)
            do, da = bc
            mean, _ = self.actor(do)
            a_loss = a_loss + bc_weight * F.mse_loss(torch.tanh(mean), da)
        sil = None
        if getattr(cfg, "sil_weight", 0.0) > 0 and self.elite.size >= 512:
            # advantage-weighted self-imitation (AWR-style soft gate): clone elite actions with
            # weight exp(adv/beta) where adv = realized return-to-go minus the critic's estimate of
            # the current policy. Soft weighting consolidates faster than a binary mask while still
            # annealing away as the policy catches up (immune to the plain-BC collapse).
            eo, ea, _, _, _, _, eret = self.elite.sample(getattr(cfg, "sil_batch", 256))
            with torch.no_grad():
                pa_e, _ = self.actor.sample(eo)
                v_e = self.critic(eo, pa_e).min(0).values
                beta = getattr(cfg, "sil_beta", 10.0)
                w = torch.clamp(torch.exp((eret - v_e) / max(beta, 1e-6)), max=10.0)
            mean_e, _ = self.actor(eo)
            sil = ((torch.tanh(mean_e) - ea).pow(2).mean(1, keepdim=True) * w).sum() \
                / w.sum().clamp(min=1e-3)
            a_loss = a_loss + cfg.sil_weight * sil
        self.a_opt.zero_grad(set_to_none=True); a_loss.backward(); self.a_opt.step()
        with torch.no_grad():      # Polyak-average the actor into the eval/deploy copy
            emat = getattr(cfg, "actor_ema_tau", 0.005)
            for p, tp in zip(self.actor.parameters(), self.actor_ema.parameters()):
                tp.mul_(1 - emat).add_(emat * p)

        t_loss = -(self.log_alpha * (plogp.detach() + cfg.target_entropy)).mean()
        self.t_opt.zero_grad(set_to_none=True); t_loss.backward(); self.t_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                tp.mul_(1 - cfg.tau).add_(cfg.tau * p)
        if not metrics:            # skip the GPU->CPU syncs (float()) on the hot path
            return {}
        return {"c_loss": float(c_loss.detach()), "a_loss": float(a_loss.detach()),
                "alpha": float(alpha.detach()), "q": float(q.mean().detach()),
                "entropy": float(-plogp.mean().detach()),
                "sil": float(sil.detach()) if sil is not None else 0.0}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "actor_ema": self.actor_ema.state_dict(),
                    "critic": self.critic.state_dict(),
                    "log_alpha": self.log_alpha.detach()}, path)

    def load_actor(self, path):
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.actor_ema.load_state_dict(d.get("actor_ema", d["actor"]))

    def load_full(self, path):
        """Resume: actor (+EMA) + critics + temperature (buffer/elite are not persisted)."""
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.actor_ema.load_state_dict(d.get("actor_ema", d["actor"]))
        if "critic" in d:
            self.critic.load_state_dict(d["critic"])
            self.critic_tgt.load_state_dict(d["critic"])
        if "log_alpha" in d:
            with torch.no_grad():
                self.log_alpha.copy_(torch.as_tensor(d["log_alpha"], device=self.device))
