"""v2 revizyon - A2: r_d kararliligi (katilimci bazli bootstrap).

02_fidelity.py'deki global nicemleyiciyi AYNI rastgelelik sirasiyla yeniden
kurar (fit_split kontrol edilir), katilimci profillerini (ozgun / sembolik /
katlar disi) hesaplar ve orijinal r_d degerlerini yeniden uretir. Ardindan
KATILIMCILAR (95 cift degil) gruplar icinde yeniden orneklenir; her ornekte
Cohen d profilleri yeniden hesaplanir. Eslestirilmis ozgun/sembolik degerler ayni
ornekte kalir.

Ek olcutler: Lin uyum katsayisi (CCC; olcek/ofset farkina duyarli), ortalama
|d| ve bant basina ortalama |Δd|.

Cikti: analyses/rd_bootstrap.csv, analyses/rd_bootstrap.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXP = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(EXP))
import config as C  # noqa: E402
import symbolic as S  # noqa: E402
import common as U  # noqa: E402

BN = C.BAND_NAMES
HERE = Path(__file__).resolve().parent
NBOOT = 2000


def rbp(lp):
    p = 10.0 ** lp.astype(np.float64)
    return p / p.sum(-1, keepdims=True)


def ccc(a, b):
    ma, mb = a.mean(), b.mean()
    return 2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (ma - mb) ** 2)


def main():
    df = U.subjects()
    rng = np.random.default_rng(C.SEED)
    fit = []
    for lab in (0, 1):
        s = df[df.label == lab].participant_id.values
        fit += list(rng.choice(s, size=int(round(0.6 * len(s))), replace=False))
    assert set(fit) == set(df[df.fit_split].participant_id), "fit_split yeniden uretilemedi"
    lps = {b: [] for b in BN}
    for r in df[df.fit_split].itertuples():
        z = np.load(C.CACHE / f"zenv_{r.participant_id}.npy", mmap_mode="r")  # rng sirasini korumak icin
        lp = np.load(C.CACHE / f"paa_{r.participant_id}.npz")["lp"]
        i = np.sort(rng.choice(len(lp), size=min(25, len(lp)), replace=False))
        for k, b in enumerate(BN):
            lps[b].append(lp[i, :, k].ravel())
    pq = S.PowerQuantizer(C.POWER_ALPHABET).fit({b: np.concatenate(v) for b, v in lps.items()})

    so, ss = [], []
    for r in df.itertuples():
        lp = np.load(C.CACHE / f"paa_{r.participant_id}.npz")["lp"].astype(np.float64)
        lph = np.stack([pq.inverse(b, pq.transform(b, lp[:, :, k])) for k, b in enumerate(BN)], -1)
        so.append(rbp(lp).mean(0).ravel()); ss.append(rbp(lph).mean(0).ravel())
    so, ss, y = np.array(so), np.array(ss), df.label.values

    _, paa, lpa, sid, _ = U.load_paa(df)
    soof = np.zeros_like(ss)
    for rep, fold, tr_s, te_s in U.splits(y):
        if rep > 1:
            break
        _, _, lph = U.quantize(paa, lpa, np.isin(sid, tr_s))
        for s in te_s:
            soof[s] = rbp(lph[sid == s]).mean(0).ravel()

    def stats(idx):
        yy = y[idx]
        do = U.cohens_d(so[idx][yy == 1], so[idx][yy == 0])
        ds = U.cohens_d(ss[idx][yy == 1], ss[idx][yy == 0])
        df_ = U.cohens_d(soof[idx][yy == 1], soof[idx][yy == 0])
        out = {"r_d": np.corrcoef(do, ds)[0, 1], "r_d_oof": np.corrcoef(do, df_)[0, 1],
               "rho": spearmanr(do, ds).statistic, "ccc": ccc(do, ds), "ccc_oof": ccc(do, df_),
               "mean_abs_d_orig": np.abs(do).mean(), "mean_abs_d_sym": np.abs(ds).mean(),
               "mean_abs_delta": np.abs(ds - do).mean()}
        for k, b in enumerate(BN):
            out[f"abs_delta_{b}"] = np.abs(ds.reshape(19, 5)[:, k] - do.reshape(19, 5)[:, k]).mean()
        return out

    point = stats(np.arange(len(y)))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    br = np.random.default_rng(C.SEED + 7)
    boots = pd.DataFrame([stats(np.r_[br.choice(pos, len(pos)), br.choice(neg, len(neg))])
                          for _ in range(NBOOT)])
    res = pd.DataFrame({"point": pd.Series(point), "ci_lo": boots.quantile(.025), "ci_hi": boots.quantile(.975)})
    res.to_csv(HERE / "rd_bootstrap.csv")
    (HERE / "rd_bootstrap.md").write_text(
        "# A2 - r_d katilimci bootstrap (2000 tekrar, grup ici tabakali)\n\n"
        + res.round(4).to_markdown() + "\n", encoding="utf-8")
    print(res.round(4).to_markdown())


if __name__ == "__main__":
    main()
