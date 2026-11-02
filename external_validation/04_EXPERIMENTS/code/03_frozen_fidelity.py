"""Birincil dış deney: dondurulmuş AHEPA ölçek kodu -> Dryad AH/kontrol r_d (protokol §6–§8, §11.1).

Kanal adı kullanılmaz (sütun indeksleri). Bütün ölçütler kanal etiketlerinin ortak
permütasyonuna karşı değişmezdir; tek varsayım "bütün katılımcılarda aynı sütun sırası"dır,
bu varsayım etiketsiz olarak denetlenir (channel_consistency.csv).

Çıktı: 04_EXPERIMENTS/dryad_frozen/
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import evlib as E
from evlib import BN, NB, NCH, C, U, rbp, profiles, fidelity, boot_fidelity, channel_shuffle_null

DATA = E.ROOT / "02_DATA" / "dryad_shor_benninger"
OUT = E.ROOT / "04_EXPERIMENTS" / "dryad_frozen"; OUT.mkdir(parents=True, exist_ok=True)
ENC = E.ROOT / "03_PROTOCOL" / "frozen_encoder_ahepa.npz"
SRC = E.ROOT / "04_EXPERIMENTS" / "source_reference"
MIN_CLEAN = 6
L = C.POWER_ALPHABET


def load(man, qc=True):
    lps = []
    for r in man.itertuples():
        d = np.load(DATA / "derived" / f"lp_{r.id}.npz")
        keep = ~(d["flat"] | d["big"] | d["nan"]) if qc else np.ones(len(d["lp"]), bool)
        lps.append(d["lp"][keep].astype(np.float64))
    return lps


def summarize(so, ss, y, bands=None, nboot=2000):
    do, ds = profiles(so, ss, y)
    f = fidelity(do, ds, bands=bands)
    b = boot_fidelity(so, ss, y, nboot=nboot, bands=bands)
    null = channel_shuffle_null(do, ds, nb=len(bands or BN))
    t = pd.DataFrame({"point": pd.Series(f), "ci_lo": b.quantile(.025), "ci_hi": b.quantile(.975)})
    t.loc["n_AD"] = [int((y == 1).sum())] * 3; t.loc["n_CN"] = [int((y == 0).sum())] * 3
    t.loc["null_shuffle_mean"] = [null.mean()] * 3; t.loc["null_shuffle_q95"] = [np.quantile(null, .95)] * 3
    return t, do, ds


def encode_profiles(lps, enc, band_idx=None):
    """-> so, ss (n, 19*nb) participant-mean linear relative power; band_idx: bant alt kümesi."""
    so, ss = [], []
    for lp in lps:
        lph = enc.decode(enc.encode(lp))
        if band_idx is not None:
            lp, lph = lp[..., band_idx], lph[..., band_idx]
        so.append(rbp(lp).mean(0).ravel()); ss.append(rbp(lph).mean(0).ravel())
    return np.array(so), np.array(ss)


class Calibrated(E.FrozenScale):
    """S4: aynı PowerQuantizer tanımı, hedefin bütün katılımcılarından etiketsiz öğrenilir."""

    def __init__(self, lps, seed=C.SEED):
        rng = np.random.default_rng(seed)
        pq = E.S.PowerQuantizer(L)
        v = {b: [] for b in BN}
        for lp in lps:
            i = np.sort(rng.choice(len(lp), size=min(25, len(lp)), replace=False))
            for k, b in enumerate(BN):
                v[b].append(lp[i, :, k].ravel())
        pq.fit({b: np.concatenate(x) for b, x in v.items()})
        self.breaks = np.stack([pq.breaks[b] for b in BN]); self.reps = np.stack([pq.reps[b] for b in BN])


def channel_consistency(man):
    """Etiketsiz: katılımcı log-std ve korelasyon örüntüsünün ortak medyan örüntüyle uyumu.
    Karşılaştırma için aynı katılımcının kanalları rastgele karıştırıldığında beklenen değer."""
    ls, cr = [], []
    for r in man.itertuples():
        d = np.load(DATA / "derived" / f"lp_{r.id}.npz")
        ls.append(d["log_std_ch"]); cr.append(d["corr"])
    ls, cr = np.array(ls), np.array(cr)
    iu = np.triu_indices(NCH, 1)
    ref_s, ref_c = np.median(ls, 0), np.median(cr, 0)
    rng = np.random.default_rng(C.SEED)
    rows = []
    for i, r in enumerate(man.itertuples()):
        p = rng.permutation(NCH)
        rows.append(dict(id=r.id, fs=r.fs,
                         r_logstd=np.corrcoef(ls[i], ref_s)[0, 1],
                         r_corr=np.corrcoef(cr[i][iu], ref_c[iu])[0, 1],
                         r_logstd_permuted=np.corrcoef(ls[i][p], ref_s)[0, 1],
                         r_corr_permuted=np.corrcoef(cr[i][np.ix_(p, p)][iu], ref_c[iu])[0, 1]))
    return pd.DataFrame(rows)


def shift_diagnostics(lps, y, enc, src_lp):
    codes = np.concatenate([enc.encode(lp) for lp in lps])            # (n_win,19,5)
    grp = np.concatenate([np.full(len(lp), yy) for lp, yy in zip(lps, y)])
    tl = np.concatenate(lps)
    lph = enc.decode(codes)
    rows = []
    for k, b in enumerate(BN):
        c = codes[..., k]
        freq = np.array([(c == j).mean() for j in range(L)])
        ent = -(freq[freq > 0] * np.log2(freq[freq > 0])).sum()
        ks = ks_2samp(tl[..., k].ravel(), src_lp[k])
        row = dict(band=b, sat_low=freq[0], sat_high=freq[-1], saturation=freq[0] + freq[-1], entropy_bits=ent,
                   median_lp_target=float(np.median(tl[..., k])), median_lp_source=float(np.median(src_lp[k])),
                   median_shift_decades=float(np.median(tl[..., k]) - np.median(src_lp[k])), ks_D=ks.statistic,
                   **{f"freq_{j}": freq[j] for j in range(L)})
        for g, nm in ((1, "AD"), (0, "CN")):
            cg = c[grp == g]
            row[f"saturation_{nm}"] = float(((cg == 0) | (cg == L - 1)).mean())
            row[f"median_lp_{nm}"] = float(np.median(tl[grp == g][..., k]))
        row["logpower_MAE"] = float(np.abs(tl[..., k] - lph[..., k]).mean())
        rows.append(row)
    relmae = float(np.abs(rbp(tl) - rbp(lph)).mean())
    return pd.DataFrame(rows), relmae, float(np.abs(tl - lph).mean())


def ahepa_reference(enc):
    """AHEPA bekletilmiş 26: doygunluk / MAE ve S3 (4 bant) karşılığı, aynı dondurulmuş kodla."""
    df = U.subjects()
    ev = ~df.fit_split.values
    lps = [np.load(C.CACHE / f"paa_{r.participant_id}.npz")["lp"].astype(np.float64)
           for r in df[ev].itertuples()]
    y = df.label.values[ev]
    fz = np.load(ENC)["fit_lp"]
    diag, relmae, lpmae = shift_diagnostics(lps, y, enc, fz)
    so4, ss4 = encode_profiles(lps, enc, band_idx=[1, 2, 3, 4])
    t4, _, _ = summarize(so4, ss4, y, bands=BN[1:])
    return diag, relmae, lpmae, t4


def main():
    man_all = pd.read_csv(DATA / "participant_manifest.csv")
    enc = E.FrozenScale(ENC)
    src_lp = np.load(ENC)["fit_lp"]
    rep = {}

    # --- birim kuralı (§4/§11.1): katılımcılar arası medyan geniş bant RMS ------------------
    rms_med = float(man_all.rms_median_uv.median())
    rep["unit_rms_median_across_participants"] = rms_med
    rep["unit_rule_pass"] = bool(5 <= rms_med <= 150)
    assert rep["unit_rule_pass"], f"birim çözülemedi: medyan RMS {rms_med}"

    # --- dahil etme -------------------------------------------------------------------------
    man_all["included"] = man_all.n_clean >= MIN_CLEAN
    man_all["exclusion_reason"] = np.where(man_all.included, "", f"<{MIN_CLEAN} temiz pencere")
    man_all.to_csv(OUT / "participant_manifest_analysis.csv", index=False)
    man = man_all[man_all.included].reset_index(drop=True)
    y = man.label.values
    rep["n_AD"], rep["n_CN"] = int((y == 1).sum()), int((y == 0).sum())
    rep["excluded"] = man_all.loc[~man_all.included, ["id", "group", "n_windows", "n_clean"]].to_dict("records")

    cc = channel_consistency(man)
    cc.to_csv(OUT / "channel_consistency.csv", index=False)
    rep["channel_consistency"] = {c: dict(median=float(cc[c].median()), min=float(cc[c].min()),
                                          q05=float(cc[c].quantile(.05)))
                                  for c in ["r_logstd", "r_corr", "r_logstd_permuted", "r_corr_permuted"]}

    # --- birincil ------------------------------------------------------------------------------
    lps = load(man)
    so, ss = encode_profiles(lps, enc)
    prim, do, ds = summarize(so, ss, y)
    prim.to_csv(OUT / "fidelity_primary.csv")
    np.savez(OUT / "profiles_primary.npz", d_o=do, d_s=ds, so=so, ss=ss, y=y)

    diag, relmae, lpmae = shift_diagnostics(lps, y, enc, src_lp)
    diag.to_csv(OUT / "shift_diagnostics.csv", index=False)
    rep["relpower_MAE"], rep["logpower_MAE"] = relmae, lpmae

    a_diag, a_rel, a_lp, a_t4 = ahepa_reference(enc)
    a_diag.to_csv(OUT / "ahepa_heldout26_shift_diagnostics.csv", index=False)
    a_t4.to_csv(OUT / "ahepa_heldout26_fidelity_S3_4band.csv")
    rep["ahepa_heldout26_relpower_MAE"], rep["ahepa_heldout26_logpower_MAE"] = a_rel, a_lp

    # --- bant düzeyi kaynak–hedef özgün profil uyumu (kanal kimliği gerektirmez) --------------
    src = np.load(SRC / "ahepa_all65_profiles.npz")["d_o"].reshape(NCH, NB).mean(0)
    tgt = do.reshape(NCH, NB).mean(0)
    rep["band_mean_d_AHEPA_all65"] = dict(zip(BN, map(float, src)))
    rep["band_mean_d_Dryad"] = dict(zip(BN, map(float, tgt)))
    rep["band_mean_d_sign_agreement"] = int((np.sign(src) == np.sign(tgt)).sum())

    # --- duyarlılık ----------------------------------------------------------------------------
    sens = {}
    lps_noqc = load(man_all[man_all.n_windows >= MIN_CLEAN].reset_index(drop=True), qc=False)
    y_noqc = man_all[man_all.n_windows >= MIN_CLEAN].label.values
    sens["S1_no_window_rejection"] = summarize(*encode_profiles(lps_noqc, enc), y_noqc)[0]
    m2 = (man.label == 1) | (man.age >= 60)
    sens["S2_controls_age_ge60"] = summarize(so[m2], ss[m2], y[m2])[0]
    sens["S3_4band_no_delta"] = summarize(*encode_profiles(lps, enc, band_idx=[1, 2, 3, 4]), y, bands=BN[1:])[0]
    cal = Calibrated(lps)
    sens["S4_labelfree_target_calibration"] = summarize(*encode_profiles(lps, cal), y)[0]
    np.savez(OUT / "S4_calibrated_encoder.npz", scale_breaks=cal.breaks, scale_reps=cal.reps)
    m6 = (man.fs == 500).values
    sens["S6_500Hz_only"] = summarize(so[m6], ss[m6], y[m6])[0]
    for k, t in sens.items():
        t.to_csv(OUT / f"fidelity_{k}.csv")

    keys = ["r_d", "spearman", "ccc", "r_d_band_mean_removed", "mean_abs_delta_d", "within_r_d_delta",
            "within_r_d_theta", "within_r_d_alpha", "within_r_d_beta", "within_r_d_gamma",
            "top10_overlap", "top20_overlap", "null_shuffle_q95", "n_AD", "n_CN"]
    tab = {"primary": prim}
    tab.update(sens)
    rows = []
    for name, t in tab.items():
        for k in keys:
            if k in t.index:
                rows.append(dict(analysis=name, metric=k, point=t.loc[k, "point"],
                                 ci_lo=t.loc[k, "ci_lo"], ci_hi=t.loc[k, "ci_hi"]))
    pd.DataFrame(rows).to_csv(OUT / "summary_long.csv", index=False)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=float)[:3000])
    print(prim.round(4).to_string())
    print(diag[["band", "saturation", "saturation_AD", "saturation_CN", "entropy_bits",
                "median_shift_decades", "ks_D", "logpower_MAE"]].round(3).to_string())
    print(a_diag[["band", "saturation", "entropy_bits", "median_shift_decades", "logpower_MAE"]].round(3).to_string())
    for k, t in sens.items():
        print(k, {m: round(float(t.loc[m, "point"]), 3) for m in ("r_d", "ccc", "r_d_band_mean_removed", "n_AD", "n_CN")},
              "CI", round(float(t.loc["r_d", "ci_lo"]), 3), round(float(t.loc["r_d", "ci_hi"]), 3))


if __name__ == "__main__":
    main()
