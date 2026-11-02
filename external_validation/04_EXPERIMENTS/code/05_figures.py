"""Figures E1–E5 (English), SVG + PDF + 600-dpi PNG -> 05_FIGURES/.

Only values computed by 00/03 are plotted; nothing is typed in by hand.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import evlib as E
from evlib import BN, NB, NCH

FIG = E.ROOT / "05_FIGURES"; FIG.mkdir(exist_ok=True)
SRC = E.ROOT / "04_EXPERIMENTS" / "source_reference"
DRY = E.ROOT / "04_EXPERIMENTS" / "dryad_frozen"
DATA = E.ROOT / "02_DATA" / "dryad_shor_benninger"
ENC = E.ROOT / "03_PROTOCOL" / "frozen_encoder_ahepa.npz"
COL = dict(zip(BN, ["#2a6f97", "#61a5c2", "#e09f3e", "#9e2a2b", "#540b0e"]))
BAND_LABEL = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42})


def save(fig, name):
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def ci(t, m):
    return t.loc[m, "point"], t.loc[m, "ci_lo"], t.loc[m, "ci_hi"]


def e1():
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.3))
    for ax, (title, prof, fid) in zip(axs, [
            ("AHEPA held-out (n = 26), frozen code", SRC / "ahepa_heldout26_profiles.npz", SRC / "ahepa_heldout26_fidelity.csv"),
            ("Dryad external cohort, frozen code", DRY / "profiles_primary.npz", DRY / "fidelity_primary.csv")]):
        d = np.load(prof); t = pd.read_csv(fid, index_col=0)
        o, s = d["d_o"].reshape(NCH, NB), d["d_s"].reshape(NCH, NB)
        for k, b in enumerate(BN):
            ax.scatter(o[:, k], s[:, k], s=12, color=COL[b], label=BAND_LABEL[b], edgecolor="none", alpha=.85)
        lim = np.abs(np.r_[o.ravel(), s.ravel()]).max() * 1.08
        ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=.7, ls="--")
        ax.axhline(0, color="0.85", lw=.5); ax.axvline(0, color="0.85", lw=.5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        r, lo, hi = ci(t, "r_d")
        ax.set_title(title, fontsize=8)
        ax.text(.03, .97, f"$r_d$ = {r:.3f} [{lo:.3f}, {hi:.3f}]\nCCC = {t.loc['ccc', 'point']:.3f}\n"
                f"n = {int(t.loc['n_AD', 'point'])} AD / {int(t.loc['n_CN', 'point'])} CN",
                transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel("Cohen's d, original relative power"); ax.set_ylabel("Cohen's d, decoded from scale code")
    axs[1].legend(frameon=False, fontsize=7, loc="lower right", title="band", title_fontsize=7)
    save(fig, "E1_profile_scatter")


def e2():
    rows = [("AHEPA all 65 (paper)", SRC / "ahepa_all65_fidelity.csv"),
            ("AHEPA held-out 26, frozen", SRC / "ahepa_heldout26_fidelity.csv"),
            ("Dryad primary, frozen", DRY / "fidelity_primary.csv"),
            ("S1 no window rejection", DRY / "fidelity_S1_no_window_rejection.csv"),
            ("S2 controls ≥ 60 y", DRY / "fidelity_S2_controls_age_ge60.csv"),
            ("S3 4 bands (no δ)", DRY / "fidelity_S3_4band_no_delta.csv"),
            ("S4 label-free target calibration", DRY / "fidelity_S4_labelfree_target_calibration.csv"),
            ("S6 500-Hz recordings only", DRY / "fidelity_S6_500Hz_only.csv")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, m, lab in ((axs[0], "r_d", "$r_d$ (95% participant-bootstrap CI)"), (axs[1], "ccc", "CCC (95% CI)")):
        for i, (name, p) in enumerate(rows):
            t = pd.read_csv(p, index_col=0)
            v, lo, hi = ci(t, m)
            c = "#555555" if name.startswith("AHEPA") else ("#9e2a2b" if "primary" in name else "#2a6f97")
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="o", color=c, ms=4, capsize=2, lw=1)
            if m == "r_d":
                ax.plot(t.loc["null_shuffle_q95", "point"], i, marker="|", color="0.4", ms=8)
        ax.set_xlabel(lab); ax.axvline(1, color="0.85", lw=.5)
        if m == "ccc":
            ax.axvline(.9, color="0.7", lw=.6, ls=":"); ax.axvline(.7, color="0.7", lw=.6, ls=":")
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([r[0] for r in rows]); axs[0].invert_yaxis()
    axs[0].text(.01, -.22, "| = 95th percentile of within-band channel-shuffle null", transform=axs[0].transAxes, fontsize=6.5)
    save(fig, "E2_fidelity_forest")


def e3():
    a = pd.read_csv(SRC / "ahepa_heldout26_fidelity.csv", index_col=0)
    d = pd.read_csv(DRY / "fidelity_primary.csv", index_col=0)
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    x = np.arange(NB)
    for j, (t, lab, c) in enumerate(((a, "AHEPA held-out 26", "#888888"), (d, "Dryad", "#9e2a2b"))):
        for ax, pre in ((axs[0], "within_r_d_"), (axs[1], "abs_delta_d_")):
            v = np.array([t.loc[pre + b, "point"] for b in BN])
            lo = np.array([t.loc[pre + b, "ci_lo"] for b in BN]); hi = np.array([t.loc[pre + b, "ci_hi"] for b in BN])
            ax.bar(x + (j - .5) * .38, v, .38, color=c, label=lab)
            ax.errorbar(x + (j - .5) * .38, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", lw=.7, capsize=1.5)
    for ax, lab in ((axs[0], "within-band $r_d$ (19 channels)"), (axs[1], "mean |Δd| per band")):
        ax.set_xticks(x); ax.set_xticklabels([BAND_LABEL[b] for b in BN]); ax.set_ylabel(lab)
    axs[0].axhline(0, color="0.6", lw=.5); axs[0].legend(frameon=False, fontsize=7)
    save(fig, "E3_bandwise_fidelity")


def e4():
    a = pd.read_csv(DRY / "ahepa_heldout26_shift_diagnostics.csv")
    d = pd.read_csv(DRY / "shift_diagnostics.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.9), sharey=True)
    cmap = plt.get_cmap("viridis", 8)
    for ax, t, title in ((axs[0], a, "AHEPA held-out 26"), (axs[1], d, "Dryad")):
        left = np.zeros(NB)
        for j in range(8):
            v = t[f"freq_{j}"].values
            ax.barh(np.arange(NB), v, left=left, color=cmap(j), edgecolor="white", lw=.3, label=str(j))
            left += v
        for k in range(NB):
            ax.text(1.01, k, f"sat {t.saturation[k] * 100:.0f}%  H {t.entropy_bits[k]:.2f} b", va="center", fontsize=6.5)
        ax.set_title(title, fontsize=8); ax.set_xlim(0, 1); ax.set_xlabel("share of window × channel codes")
    axs[0].set_yticks(range(NB)); axs[0].set_yticklabels([BAND_LABEL[b] for b in BN]); axs[0].invert_yaxis()
    axs[1].legend(title="scale level", ncol=8, frameon=False, fontsize=6, title_fontsize=6.5,
                  loc="upper center", bbox_to_anchor=(-.1, -.22))
    fig.subplots_adjust(wspace=.45)
    save(fig, "E4_scale_code_usage")


def e5():
    z = np.load(ENC); src, br = z["fit_lp"], z["scale_breaks"]
    man = pd.read_csv(DRY / "participant_manifest_analysis.csv"); man = man[man.included]
    tl = {1: [], 0: []}
    for r in man.itertuples():
        dd = np.load(DATA / "derived" / f"lp_{r.id}.npz")
        keep = ~(dd["flat"] | dd["big"] | dd["nan"])
        tl[r.label].append(dd["lp"][keep])
    tl = {g: np.concatenate(v) for g, v in tl.items()}
    fig, axs = plt.subplots(1, NB, figsize=(7.4, 2.2), sharey=False)
    for k, (ax, b) in enumerate(zip(axs, BN)):
        allv = np.r_[src[k], tl[0][..., k].ravel(), tl[1][..., k].ravel()]
        bins = np.linspace(*np.percentile(allv, [.5, 99.5]), 70)
        ax.hist(src[k], bins, density=True, histtype="stepfilled", color="0.75", label="AHEPA (encoder fit)")
        ax.hist(tl[0][..., k].ravel(), bins, density=True, histtype="step", color="#2a6f97", lw=1, label="Dryad CN")
        ax.hist(tl[1][..., k].ravel(), bins, density=True, histtype="step", color="#9e2a2b", lw=1, label="Dryad AD")
        for v in br[k]:
            ax.axvline(v, color="k", lw=.4, alpha=.6)
        ax.set_title(BAND_LABEL[b]); ax.set_yticks([])
    fig.supxlabel("log$_{10}$ band power (µV²); vertical lines = frozen AHEPA scale breakpoints", fontsize=8, y=-.04)
    axs[0].legend(frameon=False, fontsize=6, loc="upper left")
    save(fig, "E5_logpower_shift")


if __name__ == "__main__":
    for f in (e1, e2, e3, e4, e5):
        f(); print(f.__name__, "ok")
