"""S5 (ikincil): Dryad'da HEDEFTE YENİDEN EĞİTİLMİŞ sınıflandırma — sıfır atış transfer DEĞİLDİR.

05_classify.py ile aynı: 5 tekrar x 5 kat katılımcı düzeyi tabakalı CV (seed 42), iç doğrulama %20,
SVM-RBF ızgarası G1, arama için katılımcı başına 40, nihai için 120 segment, katılımcı olasılığı = segment ortalaması.
  M1          log göreli bant gücü (95)
  M1s         katın eğitim katılımcılarından öğrenilen ölçek kodundan çözülmüş log göreli güç (05_classify.py M1s)
  M1s_frozen  dondurulmuş AHEPA ölçek kodundan çözülmüş log göreli güç (ek, etiketli)
Kanal adı kullanılmaz; özellikler sabit sütun sırasıyla (§11.1 varsayımı).
Kesilirse kaldığı kattan devam eder.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import evlib as E
from evlib import C, U, NB

DATA = E.ROOT / "02_DATA" / "dryad_shor_benninger"
OUT = E.ROOT / "04_EXPERIMENTS" / "dryad_classify"; OUT.mkdir(parents=True, exist_ok=True)
OOF = OUT / "oof_classify_dryad.csv"
SEG_SEARCH = 40
G1 = [{"C": c, "gamma": g} for c in (1, 10, 100) for g in ("scale", 0.01, 0.001)]


def quantize_lp(lp, mask):
    """U.quantize'ın ölçek (güç) kısmı: kesim noktaları yalnız mask segmentlerinden."""
    lph = np.zeros(lp.shape, dtype=np.float32)
    for k in range(NB):
        u = lp[mask, :, k].ravel()
        b2 = np.percentile(u, np.linspace(0, 100, U.PA + 1)[1:-1])
        s = np.searchsorted(b2, u)
        reps = np.array([u[s == j].mean() if np.any(s == j) else u.mean() for j in range(U.PA)])
        lph[:, :, k] = reps[np.searchsorted(b2, lp[:, :, k]).astype(np.int64)]
    return lph


def svm(F_tune, F_fin, y, s_tr, va_i, f_tr, te_i):
    best = (-1, None)
    sc = StandardScaler().fit(F_tune[s_tr])
    for g in G1:
        m = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=C.SEED, **g) \
            .fit(sc.transform(F_tune[s_tr]), y[s_tr])
        v = roc_auc_score(y[va_i], m.predict_proba(sc.transform(F_tune[va_i]))[:, 1])
        if v > best[0]:
            best = (v, g)
    sc = StandardScaler().fit(F_fin[f_tr])
    m = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=C.SEED, **best[1]) \
        .fit(sc.transform(F_fin[f_tr]), y[f_tr])
    return m.predict_proba(sc.transform(F_fin[te_i]))[:, 1]


def main():
    man = pd.read_csv(E.ROOT / "04_EXPERIMENTS" / "dryad_frozen" / "participant_manifest_analysis.csv")
    man = man[man.included].reset_index(drop=True)
    labels = man.label.values
    lp, sid = [], []
    for i, r in enumerate(man.itertuples()):
        d = np.load(DATA / "derived" / f"lp_{r.id}.npz")
        keep = ~(d["flat"] | d["big"] | d["nan"])
        lp.append(d["lp"][keep]); sid.append(np.full(keep.sum(), i))
    lp, sid = np.concatenate(lp), np.concatenate(sid)
    y = labels[sid]
    enc = E.FrozenScale(E.ROOT / "03_PROTOCOL" / "frozen_encoder_ahepa.npz")
    RF = U.rbp_feat(lp)
    RFz = U.rbp_feat(enc.decode(enc.encode(lp.astype(np.float64))).astype(np.float32))
    rows = pd.read_csv(OOF).to_dict("records") if OOF.exists() else []
    done = {(r["rep"], r["fold"]) for r in rows}
    for rep, fold, tr_s, te_s in U.splits(labels):
        if (rep, fold) in done:
            continue
        seed = U.fold_seed(rep, fold)
        itr_s, va_s = U.inner_val(tr_s, labels, seed)
        s_tr = U.cap(sid, itr_s, SEG_SEARCH, seed)
        f_tr = U.cap(sid, tr_s, C.SEG_TRAIN, seed)
        va_i = np.where(np.isin(sid, va_s))[0]
        te_i = np.where(np.isin(sid, te_s))[0]
        PF_t = U.rbp_feat(quantize_lp(lp, np.isin(sid, itr_s)))
        PF_f = U.rbp_feat(quantize_lp(lp, np.isin(sid, tr_s)))
        for nm, Ft, Ff in (("M1", RF, RF), ("M1s", PF_t, PF_f), ("M1s_frozen", RFz, RFz)):
            p = svm(Ft, Ff, y, s_tr, va_i, f_tr, te_i)
            us, ps = U.subj_agg(p, sid[te_i])
            rows += U.oof_rows(nm, rep, fold, us, ps)
        pd.DataFrame(rows).to_csv(OOF, index=False)
        print(f"r{rep}f{fold} done", flush=True)
    oof = pd.read_csv(OOF)
    summ, avg = U.summarize(oof, labels)
    a = avg["M1"].sort_index()
    t = labels[a.index.values]
    summ["delong_p_vs_M1"] = [np.nan if m == "M1" else U.delong(t, a.values, avg[m].sort_index().values)[3]
                              for m in summ.model]
    summ.to_csv(OUT / "classify_summary_dryad.csv", index=False)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
