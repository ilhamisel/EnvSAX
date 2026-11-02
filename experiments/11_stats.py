"""Adim 11 - TUM OOF DOSYALARINDAN ISTATISTIK VE TABLOLAR (Z2).

Birincil olcut: 5 tekrar ortalamasi katilimci olasiligi -> AUC + katilimci-bazli
bootstrap %95 GA. Eslestirilmis karsilastirmalar: DeLong (ortalama olasiliklar) ve
Nadeau-Bengio duzeltilmis tekrarli-CV t testi (25 kat AUC'si); Holm duzeltmesi.
Cikti: outputs/RESULTS_v2.md + T*_v2.csv
"""
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import common as U

NTR, NTE = 52, 13


def compare(oof, pairs, labels):
    rows = []
    _, avg = U.summarize(oof, labels)
    for a, b in pairs:
        s = avg[a].index.values
        auc_a, auc_b, z, p = U.delong(labels[s], avg[a].values, avg[b].loc[s].values)
        fa, fb = U.fold_aucs(oof, a, labels), U.fold_aucs(oof, b, labels)
        t, pt = U.corrected_ttest(fa, fb, NTR, NTE)
        d = fa - fb
        rows.append(dict(A=a, B=b, dAUC=auc_a - auc_b, delong_z=z, p_delong=p,
                         fold_dAUC=d.mean(), dz=d.mean() / (d.std(ddof=1) + 1e-12),
                         t_corr=t, p_corr=pt))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_delong_holm"] = multipletests(out.p_delong, method="holm")[1]
        out["p_corr_holm"] = multipletests(out.p_corr, method="holm")[1]
    return out


def table(oof, labels):
    s, _ = U.summarize(oof, labels)
    s.insert(1, "AUC [95% CI]", [U.fmt(r) for r in s.itertuples()])
    s["AUC rep (mean±SD)"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(s.auc_rep_mean, s.auc_rep_sd)]
    s["AUC fold (mean±SD)"] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(s.auc_fold_mean, s.auc_fold_sd)]
    return s


def show(s):
    return s[["model", "AUC [95% CI]", "AUC rep (mean±SD)", "AUC fold (mean±SD)",
              "acc", "f1", "sens", "spec"]].round(3).to_markdown(index=False)


def main():
    labels = U.subjects().label.values
    blocks = []

    def add(title, *parts):
        blocks.append(f"## {title}\n\n" + "\n\n".join(parts))

    f = C.OUT / "oof_classify.csv"
    if f.exists():
        oof = pd.read_csv(f)
        s = table(oof, labels); s.to_csv(C.OUT / "T5_classification_v2.csv", index=False)
        cmp = compare(oof, [("M1", "M2"), ("M1", "M3"), ("M2", "M3")], labels)
        cmp.to_csv(C.OUT / "T7_statistics_v2.csv", index=False)
        piv = np.stack([U.fold_aucs(oof, m, labels) for m in ("M1", "M2", "M3")], 1)
        fr = friedmanchisquare(*piv.T)
        k, n = 3, len(piv)
        diag = compare(oof, [("M1", "M1s")], labels)
        diag.to_csv(C.OUT / "T6_diagnostic_v2.csv", index=False)
        add("Classification (5x5)", show(s),
            f"Friedman over {n} folds: chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.4f}, "
            f"Kendall W = {fr.statistic / (n * (k - 1)):.3f}",
            cmp.round(4).to_markdown(index=False),
            "### Representation vs model (M1 vs M1s)\n\n" + diag.round(4).to_markdown(index=False))

    f = C.OUT / "oof_within_probe.csv"
    if f.exists():
        oof = pd.read_csv(f)
        s = table(oof, labels); s.to_csv(C.OUT / "T8_within_probe_v2.csv", index=False)
        pairs = []
        for kind in ("RBF", "BLOCK"):
            ms = [m for m in s.model if m.startswith(kind)]
            pairs += [(b, a) for a, b in zip(ms[:-1], ms[1:])]
        add("Within-segment probe", show(s), compare(oof, pairs, labels).round(4).to_markdown(index=False))

    f = C.OUT / "oof_codebook.csv"
    if f.exists():
        oof = pd.read_csv(f)
        s = table(oof, labels)
        aux = pd.read_csv(C.OUT / "T9_codebook_aux.csv").groupby("K")[["rmse", "r_d"]].mean()
        s["rmse"] = aux.rmse.values; s["r_d"] = aux.r_d.values
        s.to_csv(C.OUT / "T9_codebook_v2.csv", index=False)
        add("Codebook probe", show(s), aux.round(4).to_markdown())

    f = C.OUT / "oof_microstate.csv"
    if f.exists():
        oof = pd.read_csv(f)
        s = table(oof, labels); s.to_csv(C.OUT / "T10_microstate_v2.csv", index=False)
        _, avg = U.summarize(oof, labels)
        orc = []
        for K in sorted({int(m[2:]) for m in avg if m.startswith("MS")}):
            a, b = avg["SPEC"], avg[f"MS{K}"].loc[avg["SPEC"].index]
            t = labels[a.index.values]
            ea, eb = np.abs(t - a.values), np.abs(t - b.values)
            por = np.where(ea <= eb, a.values, b.values)
            ca, cb = (a.values > .5) == t, (b.values > .5) == t
            orc.append(dict(K=K, auc_spec=roc_auc_score(t, a.values), auc_ms=roc_auc_score(t, b.values),
                            oracle=roc_auc_score(t, por), err_corr=np.corrcoef(ea, eb)[0, 1],
                            both=int((ca & cb).sum()), spec_only=int((ca & ~cb).sum()),
                            ms_only=int((~ca & cb).sum()), neither=int((~ca & ~cb).sum())))
        orc = pd.DataFrame(orc); orc.to_csv(C.OUT / "T10_oracle_v2.csv", index=False)
        pairs = [(f"{p}{K}", "SPEC") for K in orc.K for p in ("EARLY", "LATEeq", "LATEw")]
        add("Microstate complementarity & fusion", show(s), orc.round(3).to_markdown(index=False),
            compare(oof, pairs, labels).round(4).to_markdown(index=False))

    f = C.OUT / "oof_minute.csv"
    if f.exists():
        oof = pd.read_csv(f)
        s = table(oof, labels); s.to_csv(C.OUT / "T13_minute_v2.csv", index=False)
        ms = list(s.model)
        add("Minute-scale probe (relative power)", show(s),
            compare(oof, [(b, a) for a, b in zip(ms[:-1], ms[1:])], labels).round(4).to_markdown(index=False))

    f = C.OUT / "T12_covariates.md"
    if f.exists():
        add("Covariates", f.read_text(encoding="utf-8").split("\n", 1)[1])

    U.write_md(C.OUT / "RESULTS_v2.md", "Results v2 (5x5 repeated CV, continuous filtering)", blocks)
    print((C.OUT / "RESULTS_v2.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
