"""A1 ozet: AUC [%95 GA], esik bagimli olcutler ve eslestirilmis testler.

Sondalar karar degeri (decision_function) urettigi icin esik 0'dir
(06/common.summarize'daki 0,5 esigi bu ciktilar icin uygun degildir; AUC etkilenmez).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from statsmodels.stats.multitest import multipletests

EXP = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXP))
import common as U  # noqa: E402

HERE = Path(__file__).resolve().parent
oof = pd.read_csv(HERE / "oof_shape_scale.csv")
lab = U.subjects().label.values
s, avg = U.summarize(oof, lab)
rows = []
for m in s.model:
    p = avg[m]; y = lab[p.index.values]; yh = (p.values > 0).astype(int)
    r = s.set_index("model").loc[m]
    rows.append(dict(model=m, auc=r.auc, ci_lo=r.ci_lo, ci_hi=r.ci_hi, fold_mean=r.auc_fold_mean,
                     fold_sd=r.auc_fold_sd, acc=(yh == y).mean(), f1=f1_score(y, yh, average="macro"),
                     sens=yh[y == 1].mean(), spec=1 - yh[y == 0].mean()))
summ = pd.DataFrame(rows)
summ.to_csv(HERE / "shape_scale_summary.csv", index=False)

out = []
for clf in ("RBF", "BLOCK"):
    pairs = [("SHAPE+SCALE", "SCALE"), ("SHAPE+SCALE", "SHAPE"), ("SCALE", "SHAPE"), ("ORIG", "DEC"),
             ("ORIG", "SHAPE+SCALE")]
    fam = []
    for a, b in pairs:
        A, B = f"{clf} {a}", f"{clf} {b}"
        ss = avg[A].index.values
        _, _, z, p = U.delong(lab[ss], avg[A].values, avg[B].loc[ss].values)
        fa, fb = U.fold_aucs(oof, A, lab), U.fold_aucs(oof, B, lab)
        t, pt = U.corrected_ttest(fa, fb, 52, 13)
        fam.append(dict(clf=clf, A=a, B=b, dAUC=roc_auc_score(lab[ss], avg[A].values) -
                        roc_auc_score(lab[ss], avg[B].loc[ss].values), p_delong=p,
                        fold_dAUC=(fa - fb).mean(), p_corr=pt))
    fam = pd.DataFrame(fam)
    fam["p_delong_holm"] = multipletests(fam.p_delong, method="holm")[1]
    fam["p_corr_holm"] = multipletests(fam.p_corr, method="holm")[1]
    out.append(fam)
cmp_ = pd.concat(out)
cmp_.to_csv(HERE / "shape_scale_tests.csv", index=False)
(HERE / "shape_scale.md").write_text(
    "# A1 - sekil/olcek ablasyonu (5x5, katilimci duzeyi)\n\n" + summ.round(3).to_markdown(index=False)
    + "\n\n## Eslestirilmis testler (aile = siniflandirici)\n\n" + cmp_.round(4).to_markdown(index=False) + "\n",
    encoding="utf-8")
print((HERE / "shape_scale.md").read_text(encoding="utf-8"))
