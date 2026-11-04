"""V2 birincil deney: Dryad'da katılımcı-ayrık, etiketsiz ölçek kodu kalibrasyonu (protokol §3–§10).

C1 dondurulmuş AHEPA kod defteri, C2 tüm kohort etiketsiz yeniden uyum (ilk deneyin S4 kod defteri),
C3 katılımcı-ayrık OOF kalibrasyon (yeni), C4 AHEPA held-out 26 (ilk deneyden okunur).
Önce PROTOCOL_LOCK.json doğrulanır; kilitli dosyalardan biri değişmişse çalışmaz.

Çıktılar: 02_EXPERIMENTS/{codebooks,features,bootstrap}/, 03_RESULTS/*.csv, DECISION.json
"""
import hashlib
import inspect
import json
import numpy as np
import pandas as pd
import v2lib as V
from v2lib import BN, NB, NCH, L, E, C, rbp, profiles, fidelity

CB_DIR = V.EXP / "codebooks"; FT_DIR = V.EXP / "features"; BT_DIR = V.EXP / "bootstrap"
for d in (CB_DIR, FT_DIR, BT_DIR, V.RES):
    d.mkdir(parents=True, exist_ok=True)
KEYS = ["r_d", "spearman", "ccc", "r_d_band_mean_removed", "mean_abs_delta_d", "mean_abs_d_orig",
        "mean_abs_d_dec", "top10_overlap", "top20_overlap"] + [f"within_r_d_{b}" for b in BN] + \
       [f"within_ccc_{b}" for b in BN] + [f"abs_delta_d_{b}" for b in BN]


def check_lock():
    lock = json.loads((V.PROT / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    for rel, h in lock["sha256"].items():
        assert hashlib.sha256((V.V2 / rel).read_bytes()).hexdigest() == h, f"kilitli dosya değişmiş: {rel}"
    return lock


def run_partition(ids, lps, fold_of, keep=False):
    """fold_of: (n,) 1..5. -> ss (n,95) OOF çözülmüş göreli güç, kod defterleri, [kodlar, lph]."""
    n = len(ids)
    ss = np.zeros((n, NCH * NB)); cbs = {}
    codes, lphs = [None] * n, [None] * n
    for f in sorted(set(fold_of)):
        cal, te = np.where(fold_of != f)[0], np.where(fold_of == f)[0]
        cb = V.Codebook({ids[i]: lps[i] for i in cal}, f)
        assert set(cb.ids) == {ids[i] for i in cal}
        assert not ({ids[i] for i in te} & set(cb.ids)), "held-out katılımcı kalibrasyonda"
        for i in te:
            c = cb.encode(lps[i]); h = cb.decode(c)
            ss[i] = rbp(h).mean(0).ravel()
            if keep:
                codes[i], lphs[i] = c, h
        cbs[f] = cb
    return ss, cbs, codes, lphs


def encode_all(enc, lps):
    codes = [enc.encode(lp) for lp in lps]
    lphs = [enc.decode(c) for c in codes]
    return np.array([rbp(h).mean(0).ravel() for h in lphs]), codes, lphs


def table(f, b, null):
    t = pd.DataFrame({"point": pd.Series(f), "ci_lo": b.quantile(.025), "ci_hi": b.quantile(.975)})
    t.loc["null_shuffle_q95"] = [float(np.quantile(null, .95))] * 3
    t.loc["null_shuffle_mean"] = [float(null.mean())] * 3
    return t


def window_mae(lps, lphs):
    tl, th = np.concatenate(lps), np.concatenate(lphs)
    return float(np.abs(tl - th).mean()), float(np.abs(rbp(tl) - rbp(th)).mean()), \
        [float(np.abs(tl[..., k] - th[..., k]).mean()) for k in range(NB)]


def main():
    lock = check_lock()
    man = V.manifest()
    ids, y = man.id.tolist(), man.label.values
    assert (len(ids), int(y.sum()), int((y == 0).sum())) == (139, 43, 96)
    folds = pd.read_csv(V.PROT / "PARTICIPANT_FOLDS.csv")
    assert folds.id.tolist() == ids
    fold_of = folds.fold.values
    lps = [V.load_lp(p) for p in ids]
    so = np.array([rbp(lp).mean(0).ravel() for lp in lps])
    checks = {}

    # ---------------- C3: birincil OOF -------------------------------------------------------------
    ss3, cbs, codes3, lph3 = run_partition(ids, lps, fold_of, keep=True)

    # sızıntı denetimi (protokol §8)
    rows = []
    for i, pid in enumerate(ids):
        f = int(fold_of[i])
        rows.append(dict(id=pid, label=int(y[i]), feature_fold=f, in_own_codebook=pid in cbs[f].ids,
                         n_codebooks_containing=sum(pid in cb.ids for cb in cbs.values()),
                         n_heldout_windows=len(lps[i])))
    leak = pd.DataFrame(rows)
    assert not leak.in_own_codebook.any()
    assert (leak.n_codebooks_containing == 4).all()
    for f, cb in cbs.items():
        assert set(cb.windows) == set(cb.ids)
        assert all((w < len(lps[ids.index(p)])).all() and len(w) == min(V.N_CAL_WIN, len(lps[ids.index(p)]))
                   for p, w in cb.windows.items())
    leak.to_csv(V.EXP / "LEAKAGE_CHECK_participants.csv", index=False)
    checks["leakage"] = dict(all_participants_heldout_once=bool(sorted(folds.fold.value_counts().index) == [1, 2, 3, 4, 5]
                                                                and len(leak) == 139),
                             any_participant_in_own_codebook=bool(leak.in_own_codebook.any()),
                             each_participant_in_exactly_4_codebooks=bool((leak.n_codebooks_containing == 4).all()),
                             calibration_windows_only_from_calibration_participants=True)

    # etiket bağımsızlığı: Codebook imzası ve sıra değişmezliği
    sig = list(inspect.signature(V.Codebook.__init__).parameters)
    rng = np.random.default_rng(7)
    cal1 = [i for i in range(139) if fold_of[i] != 1]
    perm = {ids[i]: lps[i] for i in rng.permutation(cal1)}
    cb_perm = V.Codebook(perm, 1)
    checks["label_independence"] = dict(
        codebook_signature=sig,
        label_argument_present=any(s in ("y", "label", "labels") for s in sig),
        codebook_identical_under_participant_order_permutation=bool(
            np.array_equal(cb_perm.breaks, cbs[1].breaks) and np.array_equal(cb_perm.reps, cbs[1].reps)))
    assert not checks["label_independence"]["label_argument_present"]
    assert checks["label_independence"]["codebook_identical_under_participant_order_permutation"]

    # ---------------- C1, C2 (aynı kodla) ----------------------------------------------------------
    c1 = E.FrozenScale(V.ENC_FROZEN)
    s4 = np.load(V.V1_DRY / "S4_calibrated_encoder.npz")
    c2 = V.Saved(s4["scale_breaks"], s4["scale_reps"])
    ss1, codes1, lph1 = encode_all(c1, lps)
    ss2, codes2, lph2 = encode_all(c2, lps)
    f1, f2, f3 = (fidelity(*profiles(so, s, y)) for s in (ss1, ss2, ss3))
    assert abs(f1["r_d"] - 0.8037336291093641) < 1e-9, f1["r_d"]
    assert abs(f2["r_d"] - 0.9918461918844368) < 1e-9, f2["r_d"]
    checks["reproduction"] = dict(C1_r_d=f1["r_d"], C2_r_d=f2["r_d"], expected_C1=0.8037336291093641,
                                  expected_C2=0.9918461918844368)

    # birim değişmezliği (+6 log10 güç = sinyal ×1000)
    ss3u, _, _, _ = run_partition(ids, [lp + 6.0 for lp in lps], fold_of)
    ss1u, _, _ = encode_all(c1, [lp + 6.0 for lp in lps])
    checks["unit_invariance"] = dict(
        shift_log10_power=6.0,
        C3_max_abs_diff_decoded_relpower=float(np.abs(ss3u - ss3).max()),
        C3_r_d_shifted=fidelity(*profiles(so, ss3u, y))["r_d"],
        C1_r_d_shifted_for_contrast=fidelity(*profiles(so, ss1u, y))["r_d"])

    # ---------------- belirsizlik ------------------------------------------------------------------
    boots = V.boot_multi(so, {"C1": ss1, "C2": ss2, "C3": ss3}, y)
    pd.concat([b.assign(condition=k, rep=np.arange(len(b))) for k, b in boots.items()]) \
        .to_csv(BT_DIR / "BOOTSTRAP_C1_C2_C3.csv", index=False)
    nulls = {k: E.channel_shuffle_null(*profiles(so, s, y)) for k, s in (("C1", ss1), ("C2", ss2), ("C3", ss3))}
    tabs = {k: table(f, boots[k], nulls[k]) for k, f in (("C1", f1), ("C2", f2), ("C3", f3))}
    assert abs(tabs["C1"].loc["r_d", "ci_lo"] - 0.3092410056791111) < 1e-9   # ilk deneyle aynı örneklemeler

    diffs = []
    for a, b in (("C3", "C1"), ("C3", "C2")):
        for m in ("r_d", "ccc", "r_d_band_mean_removed", "mean_abs_delta_d"):
            d = boots[a][m] - boots[b][m]
            diffs.append(dict(contrast=f"{a} - {b}", metric=m, point=tabs[a].loc[m, "point"] - tabs[b].loc[m, "point"],
                              ci_lo=d.quantile(.025), ci_hi=d.quantile(.975)))
    diffs = pd.DataFrame(diffs)

    # ---------------- MAE, doygunluk, entropi -------------------------------------------------------
    mae = {k: window_mae(lps, h) for k, h in (("C1", lph1), ("C2", lph2), ("C3", lph3))}
    occ = {k: V.occupancy(np.concatenate(c)) for k, c in (("C1", codes1), ("C2", codes2), ("C3", codes3))}
    ah = pd.read_csv(V.V1_DRY / "ahepa_heldout26_shift_diagnostics.csv")
    occ["C4"] = ah[[f"freq_{j}" for j in range(L)]].values
    names = {"C1": "Dryad frozen AHEPA codebook", "C2": "Dryad full-cohort label-free refit",
             "C3": "Dryad participant-disjoint calibration (OOF)", "C4": "AHEPA held-out 26, frozen codebook"}
    sat_rows = []
    for k, o in occ.items():
        for kb, b in enumerate(BN):
            sat_rows.append(dict(condition=k, protocol=names[k], fold="pooled", band=b,
                                 **{f"freq_{j}": o[kb, j] for j in range(L)},
                                 sat_low=o[kb, 0], sat_high=o[kb, -1], saturation=o[kb, 0] + o[kb, -1],
                                 entropy_bits=V.entropy_bits(o[kb])))
    for f in sorted(cbs):
        o = V.occupancy(np.concatenate([codes3[i] for i in np.where(fold_of == f)[0]]))
        for kb, b in enumerate(BN):
            sat_rows.append(dict(condition="C3", protocol=names["C3"], fold=f, band=b,
                                 **{f"freq_{j}": o[kb, j] for j in range(L)},
                                 sat_low=o[kb, 0], sat_high=o[kb, -1], saturation=o[kb, 0] + o[kb, -1],
                                 entropy_bits=V.entropy_bits(o[kb])))
    sat = pd.DataFrame(sat_rows)
    sat.to_csv(V.RES / "SATURATION_ANALYSIS.csv", index=False)

    # ---------------- kod defteri kararlılığı -------------------------------------------------------
    fz = np.load(V.ENC_FROZEN)
    st_rows = []
    for kind, attr, fz_key, n in (("breakpoint", "breaks", "scale_breaks", L - 1), ("decoder_center", "reps", "scale_reps", L)):
        for kb, b in enumerate(BN):
            for j in range(n):
                v = np.array([getattr(cbs[f], attr)[kb, j] for f in sorted(cbs)])
                ref = getattr(c2, attr)[kb, j]
                st_rows.append(dict(kind=kind, band=b, index=j + 1, **{f"fold{f}": v[f - 1] for f in sorted(cbs)},
                                    mean=v.mean(), sd=v.std(ddof=1), range=v.max() - v.min(),
                                    full_cohort_C2=ref, max_abs_diff_vs_C2=np.abs(v - ref).max(),
                                    ahepa_frozen_C1=fz[fz_key][kb, j], mean_shift_vs_C1=v.mean() - fz[fz_key][kb, j]))
    stab = pd.DataFrame(st_rows)
    stab.to_csv(V.RES / "CODEBOOK_STABILITY.csv", index=False)
    max_bp_range = stab[stab.kind == "breakpoint"].groupby("band", sort=False)["range"].max()

    # ---------------- kat bazlı (betimleyici) --------------------------------------------------------
    fold_rows = []
    for f in sorted(cbs):
        m = fold_of == f
        ff = fidelity(*profiles(so[m], ss3[m], y[m]))
        fold_rows.append(dict(fold=f, n_AD=int(y[m].sum()), n_CN=int((y[m] == 0).sum()),
                              n_calibration=len(cbs[f].ids), **{k: ff[k] for k in ("r_d", "ccc", "r_d_band_mean_removed",
                                                                                    "mean_abs_delta_d")}))
    pd.DataFrame(fold_rows).to_csv(V.RES / "FOLD_DESCRIPTIVE.csv", index=False)

    # ---------------- ikincil: 10 farklı bölme -------------------------------------------------------
    rep = pd.read_csv(V.PROT / "REPEATED_FOLDS.csv")
    assert rep.id.tolist() == ids
    rr = []
    for r in range(1, 11):
        s_r, cb_r, _, _ = run_partition(ids, lps, rep[f"rep{r:02d}"].values)
        ff = fidelity(*profiles(so, s_r, y))
        br = np.stack([cb_r[f].breaks for f in sorted(cb_r)])
        rr.append(dict(repeat=r, **{k: ff[k] for k in ("r_d", "spearman", "ccc", "r_d_band_mean_removed",
                                                       "mean_abs_delta_d", "top20_overlap")},
                       max_between_fold_breakpoint_range=float((br.max(0) - br.min(0)).max())))
    rr = pd.DataFrame(rr)
    rr.to_csv(V.RES / "REPEATED_SPLITS.csv", index=False)

    # ---------------- duyarlılık S-cc ----------------------------------------------------------------
    cc = pd.read_csv(V.V1_DRY / "channel_consistency.csv").set_index("id").loc[ids]
    keep = (cc.r_corr >= 0.5).values
    bk = V.boot_multi(so[keep], {"C1": ss1[keep], "C3": ss3[keep]}, y[keep])
    scc = []
    for k, s in (("C1", ss1), ("C3", ss3)):
        f = fidelity(*profiles(so[keep], s[keep], y[keep]))
        for m in ("r_d", "ccc", "r_d_band_mean_removed", "mean_abs_delta_d"):
            scc.append(dict(condition=k, metric=m, point=f[m], ci_lo=bk[k][m].quantile(.025),
                            ci_hi=bk[k][m].quantile(.975), n_AD=int(y[keep].sum()), n_CN=int((y[keep] == 0).sum()),
                            excluded_ids=";".join(np.array(ids)[~keep])))
    pd.DataFrame(scc).to_csv(V.RES / "SENSITIVITY_CHANNEL_CONSISTENCY.csv", index=False)

    # ---------------- kayıtlar ------------------------------------------------------------------------
    for f, cb in cbs.items():
        np.savez(CB_DIR / f"CODEBOOK_fold{f}.npz", scale_breaks=cb.breaks, scale_reps=cb.reps,
                 band_order=np.array(BN), calibration_ids=np.array(cb.ids),
                 heldout_ids=np.array(ids)[fold_of == f],
                 cal_window_index=np.concatenate([cb.windows[p] for p in cb.ids]),
                 cal_window_owner=np.concatenate([np.full(len(cb.windows[p]), p) for p in cb.ids]))
        (CB_DIR / f"CODEBOOK_fold{f}.json").write_text(json.dumps(dict(
            fold=int(f), n_calibration=len(cb.ids), n_heldout=int((fold_of == f).sum()),
            breaks={b: list(map(float, cb.breaks[k])) for k, b in enumerate(BN)},
            decoder={b: list(map(float, cb.reps[k])) for k, b in enumerate(BN)}), indent=1), encoding="utf-8")
    ch = [f"ch{c + 1:02d}" for c in range(NCH)]
    long = []
    for i, pid in enumerate(ids):
        lo, lh = lps[i].mean(0), lph3[i].mean(0)
        ro, rh = rbp(lps[i]).mean(0), rbp(lph3[i]).mean(0)
        cm, sf = codes3[i].mean(0), ((codes3[i] == 0) | (codes3[i] == L - 1)).mean(0)
        for c in range(NCH):
            for kb, b in enumerate(BN):
                long.append((pid, int(y[i]), int(fold_of[i]), ch[c], b, lo[c, kb], lh[c, kb], ro[c, kb], rh[c, kb],
                             cm[c, kb], sf[c, kb]))
    pd.DataFrame(long, columns=["id", "label", "fold", "channel", "band", "logpower_orig", "logpower_dec_oof",
                                "relpower_orig", "relpower_dec_oof", "symbol_mean", "extreme_level_fraction"]) \
        .to_csv(FT_DIR / "PARTICIPANT_FEATURES_long.csv", index=False)
    d_o = profiles(so, ss3, y)[0]
    np.savez(FT_DIR / "OOF_PROFILES.npz", ids=np.array(ids), y=y, fold=fold_of, so=so, ss_C1=ss1, ss_C2=ss2, ss_C3=ss3,
             d_o=d_o, d_C1=profiles(so, ss1, y)[1], d_C2=profiles(so, ss2, y)[1], d_C3=profiles(so, ss3, y)[1])
    np.savez_compressed(FT_DIR / "OOF_WINDOW_CODES.npz", codes=np.concatenate(codes3),
                        sid=np.concatenate([np.full(len(c), i) for i, c in enumerate(codes3)]), ids=np.array(ids),
                        fold=fold_of)

    # ---------------- sonuç tabloları ----------------------------------------------------------------
    t3 = tabs["C3"].loc[KEYS + ["null_shuffle_q95", "null_shuffle_mean"]].copy()
    t3.loc["logpower_MAE"] = [mae["C3"][0], np.nan, np.nan]
    t3.loc["relpower_MAE"] = [mae["C3"][1], np.nan, np.nan]
    for kb, b in enumerate(BN):
        t3.loc[f"logpower_MAE_{b}"] = [mae["C3"][2][kb], np.nan, np.nan]
    t3.loc["n_AD"] = [43, np.nan, np.nan]; t3.loc["n_CN"] = [96, np.nan, np.nan]
    t3.index.name = "metric"
    t3.to_csv(V.RES / "OOF_FIDELITY_RESULTS.csv")
    for k in ("C1", "C2"):
        tabs[k].to_csv(V.RES / f"FIDELITY_{k}_same_code.csv")

    c4 = pd.read_csv(V.V1_SRC / "ahepa_heldout26_fidelity.csv", index_col=0)
    v1rep = json.loads((V.V1_DRY / "report.json").read_text(encoding="utf-8"))
    t2sum = pd.read_csv(V.V2.parent / "experiments" / "outputs" / "T2_fidelity_summary.csv", index_col=0)

    def fmt(t, m):
        return f"{t.loc[m, 'point']:.3f} [{t.loc[m, 'ci_lo']:.3f}, {t.loc[m, 'ci_hi']:.3f}]"

    def satr(k):
        s = sat[(sat.condition == k) & (sat.fold.astype(str) == "pooled")].saturation
        return f"{s.min() * 100:.1f}–{s.max() * 100:.1f}%"

    comp = []
    for k, t, n_ad, n_cn, lpm, rpm, ev in (
            ("C4", c4, 14, 12, v1rep["ahepa_heldout26_logpower_MAE"], v1rep["ahepa_heldout26_relpower_MAE"],
             "Source cohort; quantizer learned on 39 other AHEPA participants; held-out participants"),
            ("C1", tabs["C1"], 43, 96, mae["C1"][0], mae["C1"][1],
             "External cohort; codebook never saw Dryad (frozen transfer); all 139 are independent of the codebook"),
            ("C2", tabs["C2"], 43, 96, mae["C2"][0], mae["C2"][1],
             "External cohort; codebook learned on the same 139 participants it encodes (not participant-disjoint)"),
            ("C3", tabs["C3"], 43, 96, mae["C3"][0], mae["C3"][1],
             "External cohort; each participant encoded by a codebook learned without labels on the other 4 folds")):
        comp.append({"condition": k, "protocol": names[k], "evidence_level": ev, "n_AD": n_ad, "n_CN": n_cn,
                     "r_d": fmt(t, "r_d"), "CCC": fmt(t, "ccc"), "band_mean_removed_r_d": fmt(t, "r_d_band_mean_removed"),
                     "mean_abs_delta_d": fmt(t, "mean_abs_delta_d"), "spearman": fmt(t, "spearman"),
                     "top20_overlap": f"{t.loc['top20_overlap', 'point']:.0f}",
                     "channel_shuffle_q95": f"{t.loc['null_shuffle_q95', 'point']:.3f}",
                     "code_saturation_range_over_bands": satr(k),
                     "logpower_MAE": f"{lpm:.3f}", "relpower_MAE": f"{rpm:.4f}"})
    comp.append({"condition": "C4b", "protocol": "AHEPA all 65, out-of-fold quantizer (manuscript; all training-fold windows)",
                 "evidence_level": "Source cohort; different protocol (5 folds, all windows), point value only",
                 "n_AD": 36, "n_CN": 29, "r_d": f"{float(t2sum.loc['r_d_out_of_fold', 'value']):.3f}"})
    comp = pd.DataFrame(comp)
    comp.to_csv(V.RES / "SOURCE_TARGET_COMPARISON.csv", index=False)
    diffs.to_csv(V.RES / "PAIRED_DIFFERENCES.csv", index=False)

    # ---------------- önceden belirlenmiş karar kuralı (protokol §10) ---------------------------------
    t = tabs["C3"]
    c3sat = sat[(sat.condition == "C3") & (sat.fold.astype(str) == "pooled")].set_index("band").saturation
    crit = {
        "1_rd_ci_lo_gt_shuffle_q95": bool(t.loc["r_d", "ci_lo"] > t.loc["null_shuffle_q95", "point"]),
        "2_ccc_ge_0.90": bool(t.loc["ccc", "point"] >= 0.90),
        "3_bmr_ge_0.90_and_ci_lo_ge_0.70": bool(t.loc["r_d_band_mean_removed", "point"] >= 0.90
                                                and t.loc["r_d_band_mean_removed", "ci_lo"] >= 0.70),
        "4_mean_abs_delta_d_le_0.11": bool(t.loc["mean_abs_delta_d", "point"] <= 0.11),
        "5_saturation_le_0.35_all_bands": bool((c3sat <= 0.35).all()),
        "6_breakpoint_between_fold_range_le_0.10": bool((max_bp_range <= 0.10).all()),
    }
    not_sup = t.loc["r_d", "point"] < t.loc["null_shuffle_q95", "point"] or t.loc["ccc", "point"] < 0.70
    incon = (t.loc["r_d_band_mean_removed", "ci_lo"] < 0.50 and t.loc["r_d_band_mean_removed", "ci_hi"] >= 0.90) or \
            (rr.r_d_band_mean_removed.min() < 0.50 and rr.r_d_band_mean_removed.max() >= 0.90)
    if not_sup:
        verdict = "Not supported"
    elif incon:
        verdict = "Inconclusive"
    elif all(crit.values()):
        verdict = "Participant-disjoint calibration supported"
    elif crit["1_rd_ci_lo_gt_shuffle_q95"] and t.loc["ccc", "point"] >= 0.70:
        verdict = "Partially supported"
    else:
        verdict = "Inconclusive"
    dec = dict(verdict=verdict, criteria=crit, not_supported_trigger=bool(not_sup), inconclusive_trigger=bool(incon),
               values=dict(r_d=t.loc["r_d"].tolist(), ccc=t.loc["ccc"].tolist(),
                           bmr=t.loc["r_d_band_mean_removed"].tolist(), mean_abs_delta_d=t.loc["mean_abs_delta_d"].tolist(),
                           shuffle_q95=float(t.loc["null_shuffle_q95", "point"]),
                           saturation_by_band=c3sat.to_dict(), max_breakpoint_range_by_band=max_bp_range.to_dict(),
                           repeated_bmr_min_max=[float(rr.r_d_band_mean_removed.min()), float(rr.r_d_band_mean_removed.max())]),
               scope="Channel order undocumented: channel-level interpretation exploratory. Not frozen cross-site transfer; "
                     "not clinical generalisability.", checks=checks, protocol_lock=lock["locked_at"])
    (V.RES / "DECISION.json").write_text(json.dumps(dec, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(comp.drop(columns=["evidence_level"]).to_string())
    print(diffs.round(3).to_string())
    print(stab[stab.kind == "breakpoint"].groupby("band", sort=False)[["sd", "range", "max_abs_diff_vs_C2", "mean_shift_vs_C1"]].max().round(3))
    print(sat[sat.fold.astype(str) == "pooled"].pivot(index="band", columns="condition", values="saturation").round(3))
    print(pd.DataFrame(fold_rows).round(3).to_string())
    print(rr.round(3).to_string())
    print(json.dumps(dec, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
