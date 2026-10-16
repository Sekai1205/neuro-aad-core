"""
Nonlinear cross-subject AAD: contrastive dual-encoder CNN.

An EEG encoder and a stimulus encoder (dilated residual 1-D convs) map their inputs
to D latent time series. The match score of an EEG segment with a stream is the
mean Pearson correlation over the D latents. Training (other subjects only)
minimises softplus(-scale * (score_attended - score_unattended)), i.e. it directly
optimises the AAD decision, which is symmetric in the two streams by construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import FS
from .features import SubjectStream, Standardizer

DEVICE = torch.device(os.environ.get("AAD_DEVICE", "mps" if torch.backends.mps.is_available() else "cpu"))


@dataclass(frozen=True)
class NetConfig:
    hidden: int = 32
    latent: int = 8
    dropout: float = 0.1
    steps: int = 4000
    batch: int = 64
    segment_s: float = 5.0
    lr: float = 1e-3
    weight_decay: float = 1e-2
    eval_every: int = 250
    seed: int = 0


class _ResBlock(nn.Module):
    def __init__(self, c: int, dilation: int, p: float, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, dilation=dilation, padding=dilation * (k - 1) // 2)
        self.norm = nn.GroupNorm(1, c)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(F.gelu(self.norm(self.conv(x))))


class _Encoder(nn.Module):
    def __init__(self, cin, hidden, latent, dilations, p, k0=9):
        super().__init__()
        self.inp = nn.Conv1d(cin, hidden, k0, padding=k0 // 2)
        self.blocks = nn.Sequential(*[_ResBlock(hidden, d, p) for d in dilations])
        self.out = nn.Conv1d(hidden, latent, 1)

    def forward(self, x):
        return self.out(self.blocks(F.gelu(self.inp(x))))


def _corr(a, b, eps=1e-6):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)).mean(-1)


class DualEncoder(nn.Module):
    def __init__(self, n_eeg: int, n_env: int, cfg: NetConfig):
        super().__init__()
        self.eeg = _Encoder(n_eeg, cfg.hidden, cfg.latent, (1, 2, 4, 8, 16), cfg.dropout)
        self.env = _Encoder(n_env, cfg.hidden // 2, cfg.latent, (1, 2, 4, 8), cfg.dropout)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, eeg, env_a, env_b):
        e = self.eeg(eeg)
        return _corr(e, self.env(env_a)), _corr(e, self.env(env_b))


@dataclass
class Prepared:
    eeg: np.ndarray
    env_l: np.ndarray
    env_r: np.ndarray
    label: np.ndarray
    trial_id: np.ndarray


def prepare(st: SubjectStream, norm_samples: int | None = None) -> Prepared:
    """z-score EEG with the subject's own statistics (first norm_samples only, if given)."""
    sl = slice(0, norm_samples)
    se = Standardizer.fit(st.eeg[sl])
    sv = Standardizer.fit(np.concatenate([st.env_l, st.env_r]))
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
    return Prepared(f32(se(st.eeg)), f32(sv(st.env_l)), f32(sv(st.env_r)), st.label, st.trial_id)


def _to_dev(arrs):
    return torch.from_numpy(np.stack(arrs)).transpose(1, 2).to(DEVICE)


class _Sampler:
    """Random within-trial segments with their attended / unattended stream."""

    def __init__(self, data: list[Prepared], seg: int, seed: int):
        self.data, self.seg = data, seg
        self.starts = []
        for d in data:
            ok = np.ones(len(d.label) - seg, bool)
            for e in np.flatnonzero(np.diff(d.trial_id)) + 1:
                ok[max(0, e - seg):e] = False
            self.starts.append(np.flatnonzero(ok))
        self.rng = np.random.default_rng(seed)

    def batch(self, B: int):
        X, A, U = [], [], []
        for k in self.rng.integers(len(self.data), size=B):
            d = self.data[k]
            t = self.rng.choice(self.starts[k])
            sl = slice(t, t + self.seg)
            right = d.label[t] == 1
            X.append(d.eeg[sl])
            A.append((d.env_r if right else d.env_l)[sl])
            U.append((d.env_l if right else d.env_r)[sl])
        return _to_dev(X), _to_dev(A), _to_dev(U)


@torch.no_grad()
def score(model: DualEncoder, d: Prepared, starts: np.ndarray, win: int, bs: int = 256) -> np.ndarray:
    """score_right - score_left per window (> 0 -> right ear attended)."""
    model.eval()
    out = []
    for i in range(0, len(starts), bs):
        idx = starts[i:i + bs]
        cut = lambda z: _to_dev([z[s:s + win] for s in idx])
        sl, sr = model(cut(d.eeg), cut(d.env_l), cut(d.env_r))
        out.append((sr - sl).cpu().numpy())
    return np.concatenate(out)


def train(train_data: list[Prepared], val: Prepared, val_starts: np.ndarray, win: int,
          cfg: NetConfig) -> DualEncoder:
    """Train on train_data; keep the checkpoint with the best accuracy on `val`."""
    torch.manual_seed(cfg.seed)
    model = DualEncoder(train_data[0].eeg.shape[1], train_data[0].env_l.shape[1], cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    sampler = _Sampler(train_data, int(cfg.segment_s * FS), cfg.seed)
    y_val = val.label[val_starts + win // 2]
    best, best_state = -1.0, None
    for it in range(cfg.steps):
        model.train()
        x, a, u = sampler.batch(cfg.batch)
        sa, su = model(x, a, u)
        loss = F.softplus(-model.scale * (sa - su)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % cfg.eval_every == 0:
            acc = np.mean((score(model, val, val_starts, win) > 0) == y_val)
            if acc >= best:
                best = acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model
