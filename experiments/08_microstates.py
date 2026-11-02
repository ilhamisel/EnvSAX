"""Adim 8 - MIKRODURUM x SPEKTRAL TAMAMLAYICILIK + FUZYON, 5x5 (Tablo 10, Sekil 8).

Pilot 16/17 betiklerinin duzeltilmis hali:
  Z6  mikrodurumlar pycrostates ile: ortalama referans, 2-20 Hz filtre, GFP tepeleri,
      polarite-bagimsiz ModKMeans; geri atamada zamansal duzlestirme
      (factor=10, half_window_size=10), kisa segment reddi (min_segment_length=5)
      ve kenar reddi. Haritalar her dis katta YALNIZ egitim katilimcilarindan.
  Z5  gec fuzyon agirligi w, egitim katilimcilari icinde ic 5-kat OOF tahminleriyle
      secilir; test hic gorulmez. Esit agirlik da raporlanir.

Ozellikler (katilimci duzeyi, tum kayit):
  SPEC  goreli bant gucu ortalama + SS (190)
  MS_K  her harita icin zaman kapsami, ortalama sure, olusum (3K) + gecis
        olasiliklari (kosegen disi, K(K-1)) + gecis entropisi + LZ karmasikligi
Modeller: SVM-RBF (C=10, gamma=scale), pilotla ayni.
Cikti: outputs/oof_microstate.csv
"""
import sys, time, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from pycrostates.cluster import ModKMeans
from pycrostates.preprocessing import extract_gfp_peaks
from pycrostates.io import ChData
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import common as U

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
PEAKS_PER_SUB = 3000
OOF = C.OUT / "oof_microstate.csv"


def raws(df):
    ch = [str(c) for c in np.load(C.CACHE / "ch_names.npy")]
    info = mne.create_info(ch, C.SFREQ, "eeg")
    out = {}
    for r in df.itertuples():
        raw = mne.io.RawArray(np.load(C.CACHE / f"raw_{r.participant_id}.npy").astype(np.float64) * 1e-6,
                              info, verbose="ERROR")
        raw.set_eeg_reference("average", verbose="ERROR")
        raw.filter(2.0, 20.0, verbose="ERROR")
        out[r.Index] = raw
    return out


def lzc(seq):
    """Lempel-Ziv (1976) karmasikligi, n/log2(n) ile normalize."""
    s = "".join(chr(65 + int(v)) for v in seq)
    n = len(s)
    if n < 2:
        return 0.0
    # Kaspar & Schuster (1987)
    i, c, u, v, vmax = 0, 1, 1, 1, 1
    while u + v <= n:
        if s[i + v - 1] == s[u + v - 1]:
            v += 1
        else:
            vmax = max(v, vmax); i += 1
            if i == u:
                c += 1; u += vmax; v, i, vmax = 1, 0, 1
            else:
                v = 1
    if v != 1:
        c += 1
    return c / (n / np.log2(n))


def ms_features(modk, raw, K):
    seg = modk.predict(raw, factor=10, half_window_size=10, min_segment_length=5,
                       reject_edges=True, verbose="ERROR")
    par = seg.compute_parameters(norm_gfp=True)
    names = modk.cluster_names
    f = [par[f"{n}_timecov"] for n in names] + [par[f"{n}_meandurs"] for n in names] + \
        [par[f"{n}_occurrences"] for n in names]
    T = seg.compute_transition_matrix(stat="probability", ignore_repetitions=True)
    f += list(T[~np.eye(K, dtype=bool)])
    lab = seg.labels[seg.labels >= 0]
    run = lab[np.r_[True, np.diff(lab) != 0]]
    Tc = np.zeros((K, K))
    np.add.at(Tc, (run[:-1], run[1:]), 1)
    p = Tc / max(Tc.sum(), 1)
    f += [-np.sum(p[p > 0] * np.log2(p[p > 0])), lzc(run[:5000])]
    return np.nan_to_num(np.array(f, dtype=np.float64))


def svm():
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True,
                                               class_weight="balanced", random_state=C.SEED))


def inner_w(Xa, Xb, y, seed):
    """egitim katilimcilari icinde ic 5-kat OOF -> en iyi gec fuzyon agirligi."""
    pa, pb = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xa, y):
        pa[va] = svm().fit(Xa[tr], y[tr]).predict_proba(Xa[va])[:, 1]
        pb[va] = svm().fit(Xb[tr], y[tr]).predict_proba(Xb[va])[:, 1]
    ws = np.linspace(0, 1, 21)
    return ws[np.argmax([roc_auc_score(y, w * pa + (1 - w) * pb) for w in ws])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--reps", type=int, default=C.N_REPEATS)
    args = ap.parse_args()

    df, _, lp, sid, _ = U.load_paa()
    y = df.label.values
    p = 10.0 ** lp.astype(np.float64); rb = p / p.sum(-1, keepdims=True)
    SPEC = np.stack([np.r_[rb[sid == s].mean(0).ravel(), rb[sid == s].std(0).ravel()]
                     for s in range(len(df))])
    t0 = time.time()
    R = raws(df)
    peaks = {}
    for i, raw in R.items():
        pk = extract_gfp_peaks(raw, min_peak_distance=2, verbose="ERROR")
        d = pk.get_data()
        sel = np.random.default_rng(C.SEED + i).choice(d.shape[1], min(PEAKS_PER_SUB, d.shape[1]), replace=False)
        peaks[i] = d[:, np.sort(sel)]
    info = pk.info
    print(f"on hazirlik {time.time() - t0:.0f}s", flush=True)

    rows = pd.read_csv(OOF).to_dict("records") if OOF.exists() else []
    done = {(r["model"], r["rep"], r["fold"]) for r in rows}
    for rep, fold, tr_s, te_s in U.splits(y):
        if rep > args.reps:
            break
        seed = U.fold_seed(rep, fold)
        if ("SPEC", rep, fold) not in done:
            rows += U.oof_rows("SPEC", rep, fold, te_s,
                               svm().fit(SPEC[tr_s], y[tr_s]).predict_proba(SPEC[te_s])[:, 1])
        for K in args.K:
            if (f"MS{K}", rep, fold) in done:
                continue
            t1 = time.time()
            modk = ModKMeans(n_clusters=K, n_init=20, max_iter=300, random_state=seed)
            modk.fit(ChData(np.hstack([peaks[s] for s in tr_s]), info), n_jobs=4, verbose="ERROR")
            MS = np.stack([ms_features(modk, R[s], K) for s in range(len(df))])
            pm = svm().fit(MS[tr_s], y[tr_s]).predict_proba(MS[te_s])[:, 1]
            ps = svm().fit(SPEC[tr_s], y[tr_s]).predict_proba(SPEC[te_s])[:, 1]
            EF = np.concatenate([SPEC, MS], 1)
            pe = svm().fit(EF[tr_s], y[tr_s]).predict_proba(EF[te_s])[:, 1]
            w = inner_w(SPEC[tr_s], MS[tr_s], y[tr_s], seed)
            rows += U.oof_rows(f"MS{K}", rep, fold, te_s, pm)
            rows += U.oof_rows(f"EARLY{K}", rep, fold, te_s, pe)
            rows += U.oof_rows(f"LATEeq{K}", rep, fold, te_s, 0.5 * ps + 0.5 * pm)
            rows += U.oof_rows(f"LATEw{K}", rep, fold, te_s, w * ps + (1 - w) * pm)
            pd.DataFrame(rows).to_csv(OOF, index=False)
            print(f"r{rep}f{fold} K={K} ({time.time() - t1:.0f}s) "
                  f"MS={roc_auc_score(y[te_s], pm):.3f} SPEC={roc_auc_score(y[te_s], ps):.3f} "
                  f"EARLY={roc_auc_score(y[te_s], pe):.3f} w={w:.2f}", flush=True)
        pd.DataFrame(rows).to_csv(OOF, index=False)


if __name__ == "__main__":
    main()
