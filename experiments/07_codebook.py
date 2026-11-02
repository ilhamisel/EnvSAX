"""Adim 7 - BEYIN DURUMU KOD DEFTERI SONDASI, 5x5 (Tablo 9, Sekil 7b).

Pilot 14_codebook_probe.py ile ayni; 5x5 tekrarli CV ve OOF kaydi eklendi.
Her 10 s'lik segment 95 boyutlu log guc profilinden tek bir koda (k-ortalamalar)
indirgenir; kod defteri yalniz egitim katilimcilarindan. Kod torbasi (sirasiz
histogram) -> SVM-RBF. r_d: koddan cozulen profillerin Cohen's d korunumu
(betimsel, tum katilimcilar).
Cikti: outputs/oof_codebook.csv, outputs/T9_codebook_aux.csv
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import common as U

KS = [16, 32, 64, 128, 256, 512]


def main():
    df, _, lp, sid, _ = U.load_paa()
    lp = lp.reshape(len(lp), -1).astype(np.float32)
    labels = df.label.values
    oof, aux = [], []
    for rep, fold, tr_s, te_s in U.splits(labels):
        t0 = time.time()
        tr = np.isin(sid, tr_s)
        Z = StandardScaler().fit(lp[tr]).transform(lp)
        for K in KS:
            km = MiniBatchKMeans(K, random_state=U.fold_seed(rep, fold), n_init=5,
                                 batch_size=4096).fit(Z[tr])
            code = km.predict(Z); Zq = km.cluster_centers_[code]
            H = np.stack([np.bincount(code[sid == s], minlength=K) / (sid == s).sum()
                          for s in range(len(df))]).astype(np.float32)
            pipe = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True,
                                                       class_weight="balanced", random_state=C.SEED))
            p = pipe.fit(H[tr_s], labels[tr_s]).predict_proba(H[te_s])[:, 1]
            oof += U.oof_rows(f"K={K}", rep, fold, te_s, p)
            so = np.array([Z[sid == s].mean(0) for s in range(len(df))])
            sq = np.array([Zq[sid == s].mean(0) for s in range(len(df))])
            aux.append(dict(K=K, rep=rep, fold=fold,
                            rmse=float(np.sqrt(((Z - Zq) ** 2).mean())),
                            r_d=float(np.corrcoef(U.cohens_d(so[labels == 1], so[labels == 0]),
                                                  U.cohens_d(sq[labels == 1], sq[labels == 0]))[0, 1])))
        print(f"r{rep}f{fold} ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(oof).to_csv(C.OUT / "oof_codebook.csv", index=False)
    pd.DataFrame(aux).to_csv(C.OUT / "T9_codebook_aux.csv", index=False)


if __name__ == "__main__":
    main()
