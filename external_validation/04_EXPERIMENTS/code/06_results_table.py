"""06_REPORTS/RESULTS_COMPARISON.csv — yalnızca hesaplanmış CSV/JSON dosyalarından üretilir."""
import json
import pandas as pd
import evlib as E

SRC = E.ROOT / "04_EXPERIMENTS" / "source_reference"
DRY = E.ROOT / "04_EXPERIMENTS" / "dryad_frozen"
COLS = {
    "AHEPA all 65 (manuscript)": SRC / "ahepa_all65_fidelity.csv",
    "AHEPA held-out 26 (frozen reference)": SRC / "ahepa_heldout26_fidelity.csv",
    "Dryad external, frozen AHEPA code (PRIMARY)": DRY / "fidelity_primary.csv",
    "Dryad S1 no window rejection": DRY / "fidelity_S1_no_window_rejection.csv",
    "Dryad S2 controls >=60 y": DRY / "fidelity_S2_controls_age_ge60.csv",
    "Dryad S3 4 bands (no delta)": DRY / "fidelity_S3_4band_no_delta.csv",
    "AHEPA held-out 26, S3 4 bands (reference for S3)": DRY / "ahepa_heldout26_fidelity_S3_4band.csv",
    "Dryad S4 label-free target calibration (adapted)": DRY / "fidelity_S4_labelfree_target_calibration.csv",
    "Dryad S6 500-Hz recordings only": DRY / "fidelity_S6_500Hz_only.csv",
}
METRICS = [("n_AD", "AD participants", 0), ("n_CN", "Control participants", 0),
           ("r_d", "r_d", 1), ("spearman", "Spearman", 1), ("ccc", "CCC", 1),
           ("r_d_band_mean_removed", "Band-mean-removed r_d", 1),
           ("r_band_mean_only_profile", "r(original, band-mean-only profile)", 1),
           ("within_r_d_delta", "Delta within-band r_d", 1), ("within_r_d_theta", "Theta within-band r_d", 1),
           ("within_r_d_alpha", "Alpha within-band r_d", 1), ("within_r_d_beta", "Beta within-band r_d", 1),
           ("within_r_d_gamma", "Gamma within-band r_d", 1),
           ("mean_abs_delta_d", "Mean |delta d|", 1), ("mean_abs_d_orig", "Mean |d| original", 1),
           ("mean_abs_d_dec", "Mean |d| decoded", 1),
           ("top10_overlap", "Top-10 overlap", 1), ("top20_overlap", "Top-20 overlap", 1),
           ("null_shuffle_q95", "Channel-shuffle null q95", 0)]


def main():
    out = {}
    for col, p in COLS.items():
        t = pd.read_csv(p, index_col=0)
        v = {}
        for k, lab, withci in METRICS:
            if k not in t.index:
                v[lab] = "n/a"
            elif withci:
                v[lab] = f"{t.loc[k, 'point']:.3f} [{t.loc[k, 'ci_lo']:.3f}, {t.loc[k, 'ci_hi']:.3f}]"
            else:
                v[lab] = f"{t.loc[k, 'point']:.3f}".rstrip("0").rstrip(".")
        out[col] = v
    rep = json.loads((DRY / "report.json").read_text(encoding="utf-8"))
    d = pd.read_csv(DRY / "shift_diagnostics.csv"); a = pd.read_csv(DRY / "ahepa_heldout26_shift_diagnostics.csv")
    for nm, t in (("Dryad external, frozen AHEPA code (PRIMARY)", d), ("AHEPA held-out 26 (frozen reference)", a)):
        for r in t.itertuples():
            out[nm][f"Scale-code saturation {r.band} (levels 0+7)"] = f"{r.saturation:.3f}"
        out[nm]["Log-power MAE (log10 µV²)"] = f"{rep['logpower_MAE' if 'Dryad' in nm else 'ahepa_heldout26_logpower_MAE']:.3f}"
        out[nm]["Relative-power MAE"] = f"{rep['relpower_MAE' if 'Dryad' in nm else 'ahepa_heldout26_relpower_MAE']:.4f}"
    df = pd.DataFrame(out).fillna("")
    df.loc["Out-of-fold r_d (per-fold quantizer; different protocol)"] = ["0.974"] + [""] * (len(COLS) - 1)
    # sütun sırası COLS ile aynı: AHEPA, AHEPA, Dryad x4, AHEPA (S3 referansı), Dryad x2
    df.loc["Channel identity"] = ["named (10-20)"] * 2 + ["UNDOCUMENTED column order (ch01-ch19)"] * 4 \
        + ["named (10-20)"] + ["UNDOCUMENTED column order (ch01-ch19)"] * 2
    df.loc["Eye state"] = ["eyes closed"] * 2 + ["mixed open/closed, unmarked"] * 4 + ["eyes closed"] \
        + ["mixed open/closed, unmarked"] * 2
    cl = pd.read_csv(E.ROOT / "04_EXPERIMENTS" / "dryad_classify" / "classify_summary_dryad.csv").set_index("model")
    for m in cl.index:
        r = cl.loc[m]
        df.loc[f"S5 retrained-in-target AUC {m} (5x5 CV)"] = [""] * 2 + [f"{r.auc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" + ("" if m == "M1" else f"; DeLong p vs M1 = {r.delong_p_vs_M1:.3f}")] + [""] * (len(COLS) - 3)
    df.index.name = "Metric"
    df.to_csv(E.ROOT / "06_REPORTS" / "RESULTS_COMPARISON.csv", encoding="utf-8")
    print(df.to_string())


if __name__ == "__main__":
    main()
