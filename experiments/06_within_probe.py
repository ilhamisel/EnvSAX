"""Adim 6 - SEGMENT ICI ZAMANSAL YAPI SONDASI (Tablo 8, Sekil 7a).

Bilgi kaynaklari kademeli eklenir: S1 guc one-hot (760) -> S2 + sirasiz sekil
histogrami (800) -> S3 + bigram (1120) -> S4 + ucte-bir konum (1240).

Z7: pilotta RBF cekirdeginde 760 boyutlu one-hot blok baskindi; kucuk bloklar
gorunmez kalabilirdi. Iki siniflandirici raporlanir:
  RBF   pilotla ayni (SVM-RBF, C=10, gamma=scale), karar degeri ortalamasi
  BLOK  blok-bazli olcekli L2 lojistik regresyon: her blok standardize edilip
        sqrt(blok boyutu)'na bolunur (her blogun toplam varyansi esit);
        C ic dogrulamada secilir (kesim noktalari ic-egitimden, Z3)
Cikti: outputs/oof_within_probe.csv
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import common as U

warnings.filterwarnings("ignore")
NB, W, A, PA = U.NB, U.W, U.A, U.PA
OOF = C.OUT / "oof_within_probe.csv"


def f_power(sym, pw):
    return np.stack([(pw == v) for v in range(PA)], -1).reshape(len(pw), -1).astype(np.float32)


def f_unigram(sym, pw):
    return np.concatenate([np.stack([(sym[:, :, k] == v).mean((1, 2)) for v in range(A)], 1)
                           for k in range(NB)], 1).astype(np.float32)


def f_bigram(sym, pw):
    out = []
    for k in range(NB):
        s = sym[:, :, k].astype(np.int16)
        bg = s[:, :, :-1] * A + s[:, :, 1:]
        out.append(np.stack([(bg == v).mean((1, 2)) for v in range(A * A)], 1))
    return np.concatenate(out, 1).astype(np.float32)


def f_position(sym, pw):
    t = W // 3
    return np.concatenate([np.stack([(sym[:, :, k, lo:hi] == v).mean((1, 2)) for v in range(A)], 1)
                           for lo, hi in [(0, t), (t, 2 * t), (2 * t, W)] for k in range(NB)],
                          1).astype(np.float32)


FNS = [f_power, f_unigram, f_bigram, f_position]
LEVELS = ["S1 power", "S2 +shape hist", "S3 +bigram", "S4 +position"]


def blocks(sym, pw):
    return [f(sym, pw) for f in FNS]


def block_scale(Bs, tr):
    """her blogu egitimden standardize et, sqrt(boyut)'a bol -> birlestir."""
    out = []
    for B in Bs:
        sc = StandardScaler().fit(B[tr])
        out.append(sc.transform(B) / np.sqrt(B.shape[1]))
    return np.concatenate(out, 1)


def main():
    df, paa, lp, sid, y = U.load_paa()
    labels = df.label.values
    rows = pd.read_csv(OOF).to_dict("records") if OOF.exists() else []
    done = {(r["rep"], r["fold"]) for r in rows}
    for rep, fold, tr_s, te_s in U.splits(labels):
        if (rep, fold) in done:
            continue
        t0 = time.time(); seed = U.fold_seed(rep, fold)
        itr_s, va_s = U.inner_val(tr_s, labels, seed)
        f_tr = U.cap(sid, tr_s, C.SEG_TRAIN, seed)
        i_tr = U.cap(sid, itr_s, C.SEG_TRAIN, seed)
        va = np.where(np.isin(sid, va_s))[0]
        te = np.where(np.isin(sid, te_s))[0]
        Bf = blocks(*U.quantize(paa, lp, np.isin(sid, tr_s))[:2])
        Bt = blocks(*U.quantize(paa, lp, np.isin(sid, itr_s))[:2])
        for L in range(1, 5):
            # RBF (pilot ile ayni kurulum)
            F = np.concatenate(Bf[:L], 1)
            sc = StandardScaler().fit(F[f_tr])
            m = SVC(C=10, gamma="scale", class_weight="balanced").fit(sc.transform(F[f_tr]), y[f_tr])
            us, ps = U.subj_agg(m.decision_function(sc.transform(F[te])), sid[te])
            rows += U.oof_rows(f"RBF {LEVELS[L - 1]}", rep, fold, us, ps)
            # blok-olcekli lojistik
            Ft = block_scale(Bt[:L], i_tr)
            best = max(((roc_auc_score(y[va], LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
                                       .fit(Ft[i_tr], y[i_tr]).decision_function(Ft[va])), c)
                        for c in (1e-3, 1e-2, 1e-1, 1.0)))[1]
            Ff = block_scale(Bf[:L], f_tr)
            lr = LogisticRegression(C=best, max_iter=2000, class_weight="balanced").fit(Ff[f_tr], y[f_tr])
            us, ps = U.subj_agg(lr.decision_function(Ff[te]), sid[te])
            rows += U.oof_rows(f"BLOCK {LEVELS[L - 1]}", rep, fold, us, ps)
        pd.DataFrame(rows).to_csv(OOF, index=False)
        print(f"r{rep}f{fold} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
