"""Adim 2 - TEMSIL SADAKATI (Tablo 2, Sekil 3-4).

Pilot 03_fidelity.py ile ayni olculer; fark: zarf surekli kayitta hesaplandi (Z1).
Ek: ORNEKLEM-DISI r_d. Her katilimcinin sembolik gucu, o katilimciyi gormemis
bir nicemleyiciyle cozulur (5-kat, 1. tekrar); r_d bu kat-disi cozumle de verilir.

Cikti: outputs/T2_fidelity.{csv,md}, cache/fidelity.npz
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import symbolic as S
import common as U

BN = C.BAND_NAMES


def rbp(lp):
    p = 10.0 ** lp.astype(np.float64)
    return p / p.sum(-1, keepdims=True)


def main():
    df = U.subjects()
    rng = np.random.default_rng(C.SEED)
    fit = []
    for lab in (0, 1):
        s = df[df.label == lab].participant_id.values
        fit += list(rng.choice(s, size=int(round(0.6 * len(s))), replace=False))
    df["fit_split"] = df.participant_id.isin(fit)
    df.to_csv(C.CACHE / "subjects.csv", index=False)

    # --- global nicemleyici (fit katilimcilari, kisi basina 25 segment) ------
    zs = {b: [] for b in BN}; lps = {b: [] for b in BN}
    for r in df[df.fit_split].itertuples():
        z = np.load(C.CACHE / f"zenv_{r.participant_id}.npy", mmap_mode="r")
        lp = np.load(C.CACHE / f"paa_{r.participant_id}.npz")["lp"]
        i = np.sort(rng.choice(len(lp), size=min(25, len(lp)), replace=False))
        for k, b in enumerate(BN):
            zs[b].append(np.asarray(z[i, :, k, :], dtype=np.float32).reshape(-1, C.SEG_LEN))
            lps[b].append(lp[i, :, k].ravel())
    sax = {b: S.SAX(C.W, C.ALPHABET).fit(np.concatenate(zs[b])) for b in BN}
    pq = S.PowerQuantizer(C.POWER_ALPHABET).fit({b: np.concatenate(v) for b, v in lps.items()})

    # --- tum katilimcilar ----------------------------------------------------
    shape_r, pw_o, pw_s = [], {b: [] for b in BN}, {b: [] for b in BN}
    rbp_err, so, ss = [], [], []
    for r in df.itertuples():
        z = np.load(C.CACHE / f"zenv_{r.participant_id}.npy")
        lp = np.load(C.CACHE / f"paa_{r.participant_id}.npz")["lp"].astype(np.float64)
        sr = []
        for k, b in enumerate(BN):
            zk = z[:, :, k, :].astype(np.float32)
            rec = sax[b].inverse(sax[b].transform(zk), C.SEG_LEN)
            sr.append(np.corrcoef(zk.ravel(), rec.ravel())[0, 1])
        shape_r.append(sr)
        lph = np.stack([pq.inverse(b, pq.transform(b, lp[:, :, k])) for k, b in enumerate(BN)], -1)
        for k, b in enumerate(BN):
            pw_o[b].append(lp[:, :, k].ravel()); pw_s[b].append(lph[:, :, k].ravel())
        ro, rs = rbp(lp), rbp(lph)
        rbp_err.append(np.abs(ro - rs).reshape(-1, len(BN)))
        so.append(ro.mean(0).ravel()); ss.append(rs.mean(0).ravel())
    shape_r = np.array(shape_r); rbp_err = np.concatenate(rbp_err)
    so, ss, y = np.array(so), np.array(ss), df.label.values
    ev = ~df.fit_split.values

    d_o = U.cohens_d(so[y == 1], so[y == 0]); d_s = U.cohens_d(ss[y == 1], ss[y == 0])
    r_d = np.corrcoef(d_o, d_s)[0, 1]
    r_de = np.corrcoef(U.cohens_d(so[ev & (y == 1)], so[ev & (y == 0)]),
                       U.cohens_d(ss[ev & (y == 1)], ss[ev & (y == 0)]))[0, 1]
    rho = spearmanr(d_o, d_s).statistic
    ov = {k: len(set(np.argsort(-np.abs(d_o))[:k]) & set(np.argsort(-np.abs(d_s))[:k]))
          for k in (10, 20, 30)}

    # --- orneklem-disi r_d: kat-disi nicemleyiciyle cozum ----------------------
    _, paa, lpa, sid, _ = U.load_paa(df)
    so_oof = np.zeros_like(ss)
    for rep, fold, tr_s, te_s in U.splits(y):
        if rep > 1:
            break
        _, _, lph = U.quantize(paa, lpa, np.isin(sid, tr_s))
        for s in te_s:
            so_oof[s] = rbp(lph[sid == s]).mean(0).ravel()
    d_oof = U.cohens_d(so_oof[y == 1], so_oof[y == 0])
    r_d_oof = np.corrcoef(d_o, d_oof)[0, 1]

    rows = []
    for k, b in enumerate(BN):
        po, ps = np.concatenate(pw_o[b]), np.concatenate(pw_s[b])
        rows.append({"Band": b.capitalize(),
                     "Envelope shape r": round(float(shape_r[:, k].mean()), 3),
                     "Log-power r": round(float(np.corrcoef(po, ps)[0, 1]), 4),
                     "Log-power MAE (log10)": round(float(np.abs(po - ps).mean()), 4),
                     "Relative power MAE": round(float(rbp_err[:, k].mean()), 5),
                     "|d_orig - d_sym| (band mean)": round(float(np.abs(
                         d_o.reshape(19, 5)[:, k] - d_s.reshape(19, 5)[:, k]).mean()), 3)})
    tab = pd.DataFrame(rows)
    tab.to_csv(C.OUT / "T2_fidelity.csv", index=False)
    summ = {"r_d": r_d, "r_d_heldout_participants": r_de, "r_d_out_of_fold": r_d_oof,
            "spearman": rho, "top10": ov[10], "top20": ov[20], "top30": ov[30],
            "mean_abs_d_orig": np.abs(d_o).mean(), "mean_abs_d_sym": np.abs(d_s).mean()}
    pd.Series(summ).to_csv(C.OUT / "T2_fidelity_summary.csv", header=["value"])
    band_d = d_o.reshape(19, 5).mean(0)
    U.write_md(C.OUT / "T2_fidelity.md", "Table 2. Fidelity (v2, continuous filtering)", [
        tab.to_markdown(index=False),
        "\n".join(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}"
                  for k, v in summ.items()),
        "Band-mean Cohen's d (original): " + ", ".join(
            f"{b} {v:+.2f}" for b, v in zip(BN, band_d))])
    np.savez(C.CACHE / "fidelity.npz", d_o=d_o, d_s=d_s, d_oof=d_oof,
             ch=np.load(C.CACHE / "ch_names.npy"))
    print((C.OUT / "T2_fidelity.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
