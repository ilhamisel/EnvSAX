"""v2 revizyon - A1: sekil / olcek ayristirma ablasyonu.

Orijinal deneylere dokunmaz; experiments/ modullerini yalnizca ICE AKTARIR ve
ciktilari bu klasore yazar. 06_within_probe.py ile ayni katlar (5x5, tohum 42),
ayni kesim noktasi protokolu (yalniz egitim katilimcilari), ayni iki
siniflandirici (RBF: SVM C=10 gamma=scale; BLOCK: blok olcekli L2 lojistik,
C ic dogrulamada) kullanilir.

Varyantlar:
  SCALE        olcek kodu one-hot (= 06'daki S1; tekrar uretim kontrolu)
  SHAPE        sekil kodu unigram + bigram histogramlari (olcek bilgisi YOK)
  SHAPE+SCALE  ikisi birlikte (= 06'daki S3; tekrar uretim kontrolu)
  ORIG         ozgun log goreli bant gucu (95, surekli)
  DEC          olcek kodundan geri cozulmus log goreli bant gucu (95, surekli)

Cikti: analyses/oof_shape_scale.csv
"""
import sys, time, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

EXP = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXP))
import config as C  # noqa: E402
import common as U  # noqa: E402

spec = importlib.util.spec_from_file_location("probe", EXP / "06_within_probe.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
warnings.filterwarnings("ignore")
OOF = Path(__file__).resolve().parent / "oof_shape_scale.csv"


def variants(paa, lp, mask):
    sym, pw, lph = U.quantize(paa, lp, mask)
    return {"SCALE": [P.f_power(sym, pw)],
            "SHAPE": [P.f_unigram(sym, pw), P.f_bigram(sym, pw)],
            "SHAPE+SCALE": [P.f_power(sym, pw), P.f_unigram(sym, pw), P.f_bigram(sym, pw)],
            "ORIG": [U.rbp_feat(lp)],
            "DEC": [U.rbp_feat(lph)]}


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
        Vf = variants(paa, lp, np.isin(sid, tr_s))
        Vt = variants(paa, lp, np.isin(sid, itr_s))
        for name in Vf:
            F = np.concatenate(Vf[name], 1)
            sc = StandardScaler().fit(F[f_tr])
            m = SVC(C=10, gamma="scale", class_weight="balanced").fit(sc.transform(F[f_tr]), y[f_tr])
            us, ps = U.subj_agg(m.decision_function(sc.transform(F[te])), sid[te])
            rows += U.oof_rows(f"RBF {name}", rep, fold, us, ps)
            Ft = P.block_scale(Vt[name], i_tr)
            best = max(((roc_auc_score(y[va], LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
                                       .fit(Ft[i_tr], y[i_tr]).decision_function(Ft[va])), c)
                        for c in (1e-3, 1e-2, 1e-1, 1.0)))[1]
            Ff = P.block_scale(Vf[name], f_tr)
            lr = LogisticRegression(C=best, max_iter=2000, class_weight="balanced").fit(Ff[f_tr], y[f_tr])
            us, ps = U.subj_agg(lr.decision_function(Ff[te]), sid[te])
            rows += U.oof_rows(f"BLOCK {name}", rep, fold, us, ps)
        pd.DataFrame(rows).to_csv(OOF, index=False)
        print(f"r{rep}f{fold} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
