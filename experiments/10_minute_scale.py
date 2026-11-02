"""Adim 10 - SEGMENTLER ARASI (DAKIKA OLCEGI) SONDASI, 5x5 (Z10).

Pilot 13_across_segment_probe.py'nin duzeltilmis hali:
  - D1 tabani MUTLAK log guc yerine GORELI log bant gucu (diger tablolarla tutarli)
  - LOSO + havuzlanmis Platt AUC yerine 5x5 katilimci-temelli CV, OOF kaydi
  - SVM ayarlari pilotla ayni (C=10, gamma=scale)
Not: segmentler %50 ortusmeli oldugundan gecikme-1 otokorelasyonu ortusmeden
etkilenir; yorumlarken dikkate alin.
Cikti: outputs/oof_minute.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import common as U


def sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, dtype=np.float64); n = len(x)
    if n < m + 2:
        return 0.0
    r = r * (x.std() + 1e-12)

    def phi(mm):
        z = np.lib.stride_tricks.sliding_window_view(x, mm)
        d = np.abs(z[:, None, :] - z[None, :, :]).max(-1)
        np.fill_diagonal(d, np.inf)
        return (d <= r).sum()
    return float(-np.log((phi(m + 1) + 1e-12) / (phi(m) + 1e-12)))


def feats(x):
    """x (n_seg, 95) goreli log guc zaman serisi -> bloklar."""
    n = len(x); xc = x - x.mean(0)
    t = np.arange(n) - (n - 1) / 2
    return dict(mean=x.mean(0), std=x.std(0, ddof=1),
                ac1=(xc[1:] * xc[:-1]).sum(0) / ((xc ** 2).sum(0) + 1e-12),
                slope=(xc * t[:, None]).sum(0) / (t ** 2).sum(),
                ent=np.array([sample_entropy(x[:200, j]) for j in range(x.shape[1])]))


LEVELS = [("D1 mean", ["mean"]), ("D2 +SD", ["mean", "std"]),
          ("D3 +AC1", ["mean", "std", "ac1"]), ("D4 +slope,entropy", ["mean", "std", "ac1", "slope", "ent"])]


def main():
    df, _, lp, sid, _ = U.load_paa()
    y = df.label.values
    R = U.rbp_feat(lp).astype(np.float64)
    B = [feats(R[sid == s]) for s in range(len(df))]
    B = {k: np.stack([b[k] for b in B]) for k in B[0]}
    rows = []
    for rep, fold, tr_s, te_s in U.splits(y):
        for nm, keys in LEVELS:
            X = np.concatenate([B[k] for k in keys], 1)
            pipe = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True,
                                                       class_weight="balanced", random_state=C.SEED))
            rows += U.oof_rows(nm, rep, fold, te_s, pipe.fit(X[tr_s], y[tr_s]).predict_proba(X[te_s])[:, 1])
    pd.DataFrame(rows).to_csv(C.OUT / "oof_minute.csv", index=False)
    print("tamam", flush=True)


if __name__ == "__main__":
    main()
