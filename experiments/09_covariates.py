"""Adim 9 - CINSIYET / YAS KARISTIRICI ANALIZI (Z9).

Kohortta cinsiyet dengesiz (AD %67, CN %38 kadin). Dort kontrol:
  (a) DEMOG   yalniz [cinsiyet, yas] ile lojistik regresyon, 5x5 OOF -> "demografi tabani"
  (b) M1 skorunun cinsiyete gore katmanli AUC'si (kadinlar / erkekler), bootstrap GA
  (c) M1fix vs M1res: ayni SVM (C=10, gamma=scale); M1res'te her oznitelik, egitim
      segmentleri uzerinde [cinsiyet, yas]'a dogrusal regresyonla artiklanir
  (d) lojistik regresyon: etiket ~ M1 skoru + cinsiyet + yas (olabilirlik orani testi)
Gereken: outputs/oof_classify.csv (05_classify.py)
Cikti: outputs/oof_covariates.csv, outputs/T12_covariates.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import common as U


def main():
    df, _, lp, sid, y = U.load_paa()
    lab = df.label.values
    sex = (df.gender.values == "F").astype(float)
    age = df.age.values.astype(float)
    D = np.c_[sex, age]
    RF = U.rbp_feat(lp)
    cov_seg = D[sid]
    rows = []
    for rep, fold, tr_s, te_s in U.splits(lab):
        seed = U.fold_seed(rep, fold)
        sc = StandardScaler().fit(D[tr_s])
        lr = LogisticRegression(class_weight="balanced").fit(sc.transform(D[tr_s]), lab[tr_s])
        rows += U.oof_rows("DEMOG", rep, fold, te_s, lr.predict_proba(sc.transform(D[te_s]))[:, 1])
        f_tr = U.cap(sid, tr_s, C.SEG_TRAIN, seed)
        te = np.where(np.isin(sid, te_s))[0]
        reg = LinearRegression().fit(cov_seg[f_tr], RF[f_tr])
        for nm, F in [("M1fix", RF), ("M1res", RF - reg.predict(cov_seg))]:
            s2 = StandardScaler().fit(F[f_tr])
            m = SVC(C=10, gamma="scale", probability=True, class_weight="balanced",
                    random_state=C.SEED).fit(s2.transform(F[f_tr]), y[f_tr])
            us, ps = U.subj_agg(m.predict_proba(s2.transform(F[te]))[:, 1], sid[te])
            rows += U.oof_rows(nm, rep, fold, us, ps)
        print(f"r{rep}f{fold}", flush=True)
    oof = pd.DataFrame(rows)
    oof.to_csv(C.OUT / "oof_covariates.csv", index=False)

    summ, avg = U.summarize(oof, lab)
    m1 = pd.read_csv(C.OUT / "oof_classify.csv")
    m1 = m1[m1.model == "M1"].groupby("sub").prob.mean().sort_index()
    s = m1.index.values; p = m1.values
    strat = []
    for nm, msk in [("female", sex[s] == 1), ("male", sex[s] == 0)]:
        lo, hi = U.boot_auc_ci(lab[s][msk], p[msk])
        strat.append(dict(subgroup=nm, n_AD=int(lab[s][msk].sum()), n_CN=int((1 - lab[s][msk]).sum()),
                          auc=roc_auc_score(lab[s][msk], p[msk]), ci_lo=lo, ci_hi=hi))
    X0 = sm.add_constant(np.c_[sex[s], age[s]])
    X1 = sm.add_constant(np.c_[sex[s], age[s], np.log(p / (1 - p + 1e-9) + 1e-9)])
    f0 = sm.Logit(lab[s], X0).fit(disp=0); f1 = sm.Logit(lab[s], X1).fit(disp=0)
    lr_stat = 2 * (f1.llf - f0.llf)
    from scipy.stats import chi2
    a, b, z, pd_ = U.delong(lab[avg["M1fix"].index], avg["M1fix"].values, avg["M1res"].values)
    U.write_md(C.OUT / "T12_covariates.md", "Table 12. Sex/age confounding checks", [
        summ.round(3).to_markdown(index=False),
        "## M1 (05_classify) AUC by sex\n\n" + pd.DataFrame(strat).round(3).to_markdown(index=False),
        f"## M1fix vs M1res (DeLong)\n\nAUC {a:.3f} vs {b:.3f}, z = {z:.2f}, p = {pd_:.3f}",
        f"## Logit: label ~ sex + age + logit(M1)\n\nLR chi2(1) = {lr_stat:.2f}, "
        f"p = {chi2.sf(lr_stat, 1):.2e}; coef(M1) = {f1.params[-1]:.3f}; "
        f"coef(sex) = {f1.params[1]:.3f} (p = {f1.pvalues[1]:.3f})"])
    print((C.OUT / "T12_covariates.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
