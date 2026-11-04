"""Figures V2-1 … V2-5 (English), SVG + PDF + 600-dpi PNG -> 04_FIGURES/.

Only values written by 02_oof_calibration.py and the first external experiment are plotted.
Text is converted to paths in SVG so the figures render identically without the font installed.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import v2lib as V
from v2lib import BN, NB, NCH, L

V.FIG.mkdir(exist_ok=True)
COL = dict(zip(BN, ["#2a6f97", "#61a5c2", "#e09f3e", "#9e2a2b", "#540b0e"]))
BAND_LABEL = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}
COND = {"C4": ("AHEPA held-out,\nfrozen codebook", "#6c757d"),
        "C1": ("Dryad, frozen\nAHEPA codebook", "#c44536"),
        "C2": ("Dryad, full-cohort\nlabel-free refit", "#8fb8a8"),
        "C3": ("Dryad, participant-disjoint\ncalibration (OOF)", "#1b6f5c")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "path", "pdf.fonttype": 42})


def save(fig, name):
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(V.FIG / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def tables():
    t = {"C3": pd.read_csv(V.RES / "OOF_FIDELITY_RESULTS.csv", index_col=0),
         "C1": pd.read_csv(V.RES / "FIDELITY_C1_same_code.csv", index_col=0),
         "C2": pd.read_csv(V.RES / "FIDELITY_C2_same_code.csv", index_col=0),
         "C4": pd.read_csv(V.V1_SRC / "ahepa_heldout26_fidelity.csv", index_col=0)}
    return t


def ci(t, m):
    return t.loc[m, "point"], t.loc[m, "ci_lo"], t.loc[m, "ci_hi"]


def v2_1(T):
    p = np.load(V.EXP / "features" / "OOF_PROFILES.npz")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for ax, k, title in ((axs[0], "C1", "A. Frozen source (AHEPA) codebook"),
                         (axs[1], "C3", "B. Participant-disjoint target calibration")):
        o, s = p["d_o"].reshape(NCH, NB), p[f"d_{k}"].reshape(NCH, NB)
        for kb, b in enumerate(BN):
            ax.scatter(o[:, kb], s[:, kb], s=13, color=COL[b], label=BAND_LABEL[b], edgecolor="none", alpha=.85)
        lim = 1.25
        ax.plot([-lim, lim], [-lim, lim], color="0.55", lw=.7, ls="--")
        ax.axhline(0, color="0.88", lw=.5); ax.axvline(0, color="0.88", lw=.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        t = T[k]
        r, lo, hi = ci(t, "r_d"); b, blo, bhi = ci(t, "r_d_band_mean_removed")
        ax.text(.03, .97, f"$r_d$ = {r:.3f} [{lo:.3f}, {hi:.3f}]\nCCC = {t.loc['ccc', 'point']:.3f}\n"
                f"band-mean-removed $r_d$ = {b:.3f}\n[{blo:.3f}, {bhi:.3f}]",
                transform=ax.transAxes, va="top", fontsize=6.8)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel("Original Cohen's d (relative power)")
        ax.set_ylabel("Decoded Cohen's d")
    axs[1].legend(frameon=False, fontsize=7, loc="lower right", title="band", title_fontsize=7)
    fig.text(0, -0.04, "Dryad cohort, 43 AD / 96 controls; each point is one channel–band pair (19 × 5). "
             "B: every participant is decoded with a codebook learned without labels on the other four folds.",
             fontsize=6.5)
    fig.tight_layout()
    save(fig, "FigureV2-1_effect_size_scatter")


def v2_2(T):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    order = ["C4", "C1", "C2", "C3"]
    x = np.arange(NB); w = .2
    for j, k in enumerate(order):
        t = T[k]
        for ax, m in ((axs[0], "within_r_d_{}"), (axs[1], "abs_delta_d_{}")):
            v = np.array([ci(t, m.format(b)) for b in BN])
            ax.errorbar(x + (j - 1.5) * w, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", ms=3.5,
                        color=COND[k][1], lw=.8, capsize=1.5, label=COND[k][0].replace("\n", " "))
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([f"{b} ({BAND_LABEL[b]})" for b in BN])
    axs[0].axhline(0, color="0.8", lw=.5)
    axs[0].set_ylabel("Within-band $r_d$ (19 channels)"); axs[0].set_ylim(-0.75, 1.05)
    axs[0].set_title("A. Channel pattern within each band", fontsize=8, loc="left")
    axs[1].set_ylabel("Mean |Δd| (decoded − original)")
    axs[1].set_title("B. Absolute effect-size deviation", fontsize=8, loc="left")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="lower center", ncol=4, bbox_to_anchor=(.5, -.07))
    fig.text(0, -0.13, "Points: estimate; bars: 95% participant bootstrap CI (2,000 replicates). "
             "AHEPA held-out: 14 AD / 12 HC; Dryad: 43 AD / 96 controls. Protocols are not equivalent (see Figure V2-5).",
             fontsize=6.3)
    fig.tight_layout()
    save(fig, "FigureV2-2_bandwise_fidelity")


def v2_3():
    s = pd.read_csv(V.RES / "SATURATION_ANALYSIS.csv")
    s = s[s.fold.astype(str) == "pooled"]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True, gridspec_kw=dict(wspace=.42))
    for ax, k, title in zip(axs, ("C4", "C1", "C3"), ("A. AHEPA held-out,\nfrozen codebook", "B. Dryad,\nfrozen AHEPA codebook",
                                                       "C. Dryad, participant-disjoint\ncalibration (OOF)")):
        m = s[s.condition == k].set_index("band").loc[BN, [f"freq_{j}" for j in range(L)]].values
        im = ax.imshow(m * 100, cmap="Greys", vmin=0, vmax=100, aspect="auto")
        for i in range(NB):
            for j in range(L):
                ax.text(j, i, f"{m[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.8,
                        color="white" if m[i, j] > .55 else "black")
            sat = m[i, 0] + m[i, -1]
            ax.text(L - .4, i, f"{sat * 100:.1f}%", ha="left", va="center", fontsize=6,
                    color="#c44536" if sat > .35 else "#1b6f5c")
        ax.set_xticks(range(L)); ax.set_xticklabels(range(1, L + 1))
        ax.set_yticks(range(NB)); ax.set_yticklabels([f"{b} ({BAND_LABEL[b]})" for b in BN])
        ax.set_xlabel("Scale level"); ax.set_title(title, fontsize=7.5, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in axs:
        ax.text(L - .4, -.75, "1+8", ha="left", fontsize=6)
    cb = fig.colorbar(im, ax=axs, fraction=.02, pad=.08); cb.set_label("% of codes", fontsize=7)
    fig.text(0, -0.08, "Cell values: percentage of window × channel codes at each of the 8 scale levels (held-out data). "
             "Right margin: share at the two extreme levels (nominal 25% for an 8-level equal-frequency codebook).",
             fontsize=6.3)
    save(fig, "FigureV2-3_scale_code_occupancy")


def v2_4():
    st = pd.read_csv(V.RES / "CODEBOOK_STABILITY.csv")
    sa = pd.read_csv(V.RES / "SATURATION_ANALYSIS.csv")
    fig = plt.figure(figsize=(7.4, 5.0))
    gs = fig.add_gridspec(2, 9, height_ratios=[1.15, 1], hspace=.6, wspace=1.2)
    folds = [f"fold{f}" for f in range(1, 6)]
    top = gs[0, :].subgridspec(1, 5, wspace=.45)
    for kb, b in enumerate(BN):
        ax = fig.add_subplot(top[0, kb])
        bp = st[(st.kind == "breakpoint") & (st.band == b)]
        dc = st[(st.kind == "decoder_center") & (st.band == b)]
        for f in folds:
            ax.plot(bp["index"], bp[f], "o", ms=2.6, color=COL[b], alpha=.7)
            ax.plot(dc["index"] - .5, dc[f], "_", ms=5, color="0.35", alpha=.7)
        ax.plot(bp["index"], bp.full_cohort_C2, "-", lw=.7, color=COL[b])
        ax.plot(bp["index"], bp.ahepa_frozen_C1, "--", lw=.8, color="#c44536")
        ax.set_title(f"{b} ({BAND_LABEL[b]})", fontsize=7.5)
        ax.set_xticks(range(1, 8)); ax.tick_params(labelsize=6)
        if kb == 0:
            ax.set_ylabel("log10 band power (µV²)")
        ax.set_xlabel("breakpoint", fontsize=6.5)
    fig.text(0.07, 0.95, "A. Breakpoints of the five fold codebooks (dots), decoder centres (grey ticks), full-cohort refit "
             "(solid) and frozen AHEPA codebook (red dashed)", fontsize=7)
    ax = fig.add_subplot(gs[1, :5])
    data, pos, cols = [], [], []
    for kb, b in enumerate(BN):
        for j, kind in enumerate(("breakpoint", "decoder_center")):
            s = st[(st.kind == kind) & (st.band == b)]
            dev = (s[folds].values - s[folds].values.mean(1, keepdims=True)).ravel()
            data.append(dev); pos.append(kb * 3 + j); cols.append(COL[b] if j == 0 else "0.6")
    bx = ax.boxplot(data, positions=pos, widths=.7, patch_artist=True, showfliers=True,
                    flierprops=dict(ms=2), medianprops=dict(color="black", lw=.8))
    for p_, c in zip(bx["boxes"], cols):
        p_.set_facecolor(c); p_.set_alpha(.75); p_.set_linewidth(.5)
    ax.set_xticks([kb * 3 + .5 for kb in range(NB)]); ax.set_xticklabels([BAND_LABEL[b] for b in BN])
    ax.axhline(0, color="0.8", lw=.5)
    ax.set_ylabel("Deviation from fold mean (log10)")
    ax.set_title("B. Between-fold variation (coloured: breakpoints; grey: decoder centres)", fontsize=7, loc="left")
    ax = fig.add_subplot(gs[1, 6:])
    s3 = sa[(sa.condition == "C3") & (sa.fold.astype(str) != "pooled")]
    for kb, b in enumerate(BN):
        v = s3[s3.band == b].saturation.values * 100
        ax.plot(np.full(len(v), kb) + np.linspace(-.15, .15, len(v)), v, "o", ms=3, color=COL[b])
    fr = sa[(sa.condition == "C1") & (sa.fold.astype(str) == "pooled")].set_index("band").loc[BN].saturation * 100
    ax.plot(range(NB), fr.values, "x", ms=5, color="#c44536", label="frozen AHEPA codebook")
    ax.axhline(25, color="0.6", lw=.6, ls=":", label="nominal 25%")
    ax.set_xticks(range(NB)); ax.set_xticklabels([BAND_LABEL[b] for b in BN])
    ax.set_ylabel("Extreme-level codes (%)"); ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=6, loc="center right")
    ax.set_title("C. Held-out saturation per fold", fontsize=7, loc="left")
    save(fig, "FigureV2-4_codebook_stability")


def v2_5(T):
    mets = [("r_d", "$r_d$"), ("ccc", "Lin's CCC"), ("r_d_band_mean_removed", "Band-mean-removed $r_d$"),
            ("mean_abs_delta_d", "Mean |Δd|")]
    order = ["C4", "C1", "C2", "C3"]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=True)
    for ax, (m, lab) in zip(axs, mets):
        for i, k in enumerate(order):
            p, lo, hi = ci(T[k], m)
            ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", ms=4, color=COND[k][1], lw=1, capsize=2)
            if m == "r_d" and "null_shuffle_q95" in T[k].index:
                ax.plot(T[k].loc["null_shuffle_q95", "point"], i, "|", ms=8, color="0.3", mew=1)
        ax.set_title(lab, fontsize=8)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([COND[k][0] for k in order], fontsize=6.8)
        ax.grid(axis="x", color="0.92", lw=.5)
    axs[0].invert_yaxis()
    axs[0].set_xlim(0.25, 1.01); axs[1].set_xlim(0.25, 1.01); axs[2].set_xlim(-0.1, 1.01)
    fig.text(0, -0.1, "Points: estimate; bars: 95% participant bootstrap CI. Vertical tick on $r_d$: 95th percentile of the "
             "within-band channel-shuffle reference.\nThe four protocols are not equivalent: the AHEPA row is a different "
             "cohort; the full-cohort refit learns the codebook on the participants it encodes;\nthe participant-disjoint "
             "row decodes each participant with a codebook learned without its data. Only the frozen row tests cross-site "
             "transfer.", fontsize=6.2)
    fig.tight_layout()
    save(fig, "FigureV2-5_fidelity_summary")


def main():
    T = tables()
    v2_1(T); v2_2(T); v2_3(); v2_4(); v2_5(T)
    print(sorted(p.name for p in V.FIG.iterdir()))


if __name__ == "__main__":
    main()
