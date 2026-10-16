"""
4-channel AAD benchmark (KU Leuven, 16 subjects, leave-one-subject-out,
120 s calibration, 10 s decision windows).

Models
  cca_calib_only     CCA decoder trained on the 120 s calibration only (reference)
  cca_cross_subject  CCA + LDA trained on the 15 other subjects, no calibration
  cca_adapted        same, with the 120 s calibration blended into the covariances
  cnn                contrastive dual-encoder CNN trained on the other subjects
  ensemble           calibrated LLR fusion of cca_adapted + cnn (per-window, 10 s)
  ensemble_hmm       ensemble + causal HMM forward filter over consecutive windows

Usage:  python pipelines/run_aad.py [--seeds 3] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aad.data import load_all
from aad.features import FeatureConfig, subject_stream
from aad.linear import CCAConfig
from aad import nets, temporal
from aad.evaluate import CALIB, WIN, linear_scores, window_starts

RESULTS = PROJECT_ROOT / "results"
P_SWITCH_GRID = (0.01, 0.02, 0.05, 0.1, 0.2)

LIN_FEATURES = FeatureConfig()                       # delta+theta EEG, envelope+onset
NET_FEATURES = FeatureConfig(bands=((0.5, 31.0),), env_band=(0.5, 31.0))


def cnn_scores(streams, subjects, seeds, cfg: nets.NetConfig) -> dict:
    """Per test subject: averaged test scores and validation-subject scores."""
    prepared = {s: nets.prepare(streams[s]) for s in subjects}
    out = {}
    for s in subjects:
        t0 = time.time()
        val = subjects[(subjects.index(s) + 1) % len(subjects)]
        train = [prepared[j] for j in subjects if j not in (s, val)]
        test = nets.prepare(streams[s], norm_samples=CALIB)   # test subject: calibration stats only
        t_starts = window_starts(len(test.label), CALIB)
        v_starts = window_starts(len(prepared[val].label), 0)
        te, va = [], []
        for seed in seeds:
            model = nets.train(train, prepared[val], v_starts, WIN, nets.NetConfig(**{**cfg.__dict__, "seed": seed}))
            te.append(nets.score(model, test, t_starts, WIN))
            va.append(nets.score(model, prepared[val], v_starts, WIN))
        out[s] = {"test": np.mean(te, 0), "val": np.mean(va, 0), "val_id": val}
        print(f"  cnn S{s}: {np.mean((out[s]['test'] > 0) == test.label[t_starts + WIN // 2]) * 100:.2f}%"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    return out


def fit_fusion(feats: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logistic regression (no intercept: classes are symmetric) -> LLR weights."""
    w = np.zeros(feats.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-feats @ w))
        H = (feats * (p * (1 - p))[:, None]).T @ feats + 1e-6 * np.eye(len(w))
        w -= np.linalg.solve(H, feats.T @ (p - y))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--quick", action="store_true", help="linear models only")
    args = ap.parse_args()

    print("Loading data ...")
    data = load_all()
    subjects = sorted(data)

    print("Linear CCA/LDA models ...")
    lin_streams = {s: subject_stream(data[s], LIN_FEATURES) for s in subjects}
    lin = linear_scores(lin_streams, LIN_FEATURES, CCAConfig())
    del lin_streams

    cnn = None
    if not args.quick:
        print(f"CNN ({args.seeds} seed(s) per fold) ...")
        net_streams = {s: subject_stream(data[s], NET_FEATURES) for s in subjects}
        cnn = cnn_scores(net_streams, subjects, list(range(args.seeds)), nets.NetConfig(steps=args.steps))
        del net_streams

    rows = []
    for s in subjects:
        L = lin[s]
        y = L["y"]
        train_ids = L["train_ids"]

        # p_switch: chosen on the training subjects' nested CCA score sequences
        tr_scores = np.concatenate([sc for sc, _ in L["train"]])
        tr_y = np.concatenate([yy for _, yy in L["train"]])
        a_lin = temporal.fit_llr_scale(tr_scores, tr_y)
        p_best = max(P_SWITCH_GRID, key=lambda p: np.mean(np.concatenate([
            (temporal.hmm_forward(a_lin * sc, p) > 0) == yy for sc, yy in L["train"]])))

        scores = {k: L[k] for k in ("cca_calib_only", "cca_cross_subject", "cca_adapted")}
        llr = a_lin * L["cca_adapted"]
        if cnn is not None:
            C = cnn[s]
            vi = train_ids.index(C["val_id"])
            v_lin, v_y = L["train"][vi]
            w = fit_fusion(np.stack([v_lin, C["val"]], 1), v_y)
            scores["cnn"] = C["test"]
            llr = np.stack([L["cca_adapted"], C["test"]], 1) @ w
            scores["ensemble"] = llr
        hmm = temporal.hmm_forward(llr, p_best)
        scores["ensemble_hmm" if cnn is not None else "cca_adapted_hmm"] = hmm

        for name, sc in scores.items():
            dec = (sc > 0).astype(int)
            rows.append({"subject": f"S{s}", "model": name, "accuracy_pct": 100 * np.mean(dec == y),
                         "switch_latency_windows": temporal.switch_latency(dec, y),
                         "p_switch": p_best if name.endswith("hmm") else np.nan,
                         "n_windows": len(y)})

    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "aad_benchmark.csv", index=False)
    order = list(dict.fromkeys(df["model"]))
    summary = (df.groupby("model", sort=False)
                 .agg(mean=("accuracy_pct", "mean"), sd=("accuracy_pct", "std"),
                      min=("accuracy_pct", "min"), latency=("switch_latency_windows", "mean"))
                 .loc[order])
    wide = df.pivot(index="subject", columns="model", values="accuracy_pct")[order]
    wide = wide.loc[[f"S{s}" for s in subjects]]
    wide.to_csv(RESULTS / "aad_benchmark_by_subject.csv")
    print("\nPer-subject accuracy (%):")
    print(wide.round(1).to_string())
    print("\nSummary (16 subjects, 10 s windows, 120 s calibration):")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
