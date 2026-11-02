"""Adim 4 - ZARF ALANINDA SAX vs ABBA (Tablo 4a-b, Sekil 5).

Pilot 07_abba_vs_sax.py + _abba_k_sweep.py; farklar:
  Z8  fit/test ayrimi KATILIMCI duzeyinde (pilotta seri yarilari karisikti)
  Z1  zarf surekli kayitta hesaplanmis onbellekten (zenv)
Not: suruklenme duzeltmeli geri olusturma serinin GERCEK son degerini kullanir;
bu, sembollere ek olarak segment basina bir sayi saklamayi gerektirir.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import symbolic as S
import common as U

BN = C.BAND_NAMES
N_SUB, N_SEG, N_CH = 20, 8, 6
TOLS = [0.05, 0.1, 0.2, 0.4]
ABBA_K = 8


def series(sids, band_idx, n_seg, n_ch, rng):
    out = []
    for sid in sids:
        z = np.load(C.CACHE / f"zenv_{sid}.npy", mmap_mode="r")
        i = np.sort(rng.choice(len(z), min(n_seg, len(z)), replace=False))
        out.append(np.asarray(z[i, :n_ch, band_idx, :], dtype=np.float64).reshape(-1, C.SEG_LEN))
    return np.concatenate(out)


def pick(df, n, rng):
    """tabakali n katilimci -> (fit yarisi, test yarisi), katilimci duzeyinde ayrik."""
    fit, test = [], []
    for lab in (0, 1):
        s = rng.permutation(df[df.label == lab].participant_id.values)[:n // 2]
        fit += list(s[:len(s) // 2]); test += list(s[len(s) // 2:])
    return fit, test


def abba_eval(ztr, zte, tol, k, drift=False):
    ab = S.ABBA(tol, k).fit([S.abba_compress(z, tol) for z in ztr])
    rs, tk = [], []
    t = np.linspace(0, 1, C.SEG_LEN)
    for z in zte:
        sy = ab.transform(S.abba_compress(z, tol))
        rec = ab.inverse(sy, z[0], C.SEG_LEN)
        if drift:
            rec = rec + (z[-1] - rec[-1]) * t
        rs.append(np.corrcoef(z, rec)[0, 1]); tk.append(len(sy))
    return float(np.nanmean(rs)), float(np.mean(tk))


def sax_eval(ztr, zte, w=C.W, a=C.ALPHABET):
    sx = S.SAX(w, a).fit(ztr)
    return float(np.corrcoef(zte.ravel(), sx.inverse(sx.transform(zte), C.SEG_LEN).ravel())[0, 1])


def main():
    df = U.subjects()
    rng = np.random.default_rng(C.SEED)
    fit, test = pick(df, N_SUB, rng)
    rows = []
    for k, b in enumerate(BN):
        ztr, zte = series(fit, k, N_SEG, N_CH, rng), series(test, k, N_SEG, N_CH, rng)
        rows.append(dict(band=b, method="SAX", param=f"w={C.W}, a={C.ALPHABET}", tokens=C.W,
                         compression=round(C.SEG_LEN / C.W, 1), r=round(sax_eval(ztr, zte), 3)))
        for tol in TOLS:
            r, tk = abba_eval(ztr, zte, tol, ABBA_K)
            rows.append(dict(band=b, method="ABBA", param=f"tol={tol}, k={ABBA_K}", tokens=round(tk, 1),
                             compression=round(C.SEG_LEN / max(tk, 1), 1), r=round(r, 3)))
            print(f"  {b:6s} ABBA tol={tol}: tok={tk:.1f} r={r:.3f}", flush=True)
    t4a = pd.DataFrame(rows)

    # k taramasi (alfa), 12 katilimci, katilimci duzeyinde ayrik
    fit, test = pick(df, 12, rng)
    ai = BN.index("alpha")
    ztr, zte = series(fit, ai, 6, 5, rng), series(test, ai, 6, 5, rng)
    rows = [dict(method="SAX", param=f"w={C.W}, a={C.ALPHABET}", tokens=C.W,
                 r=round(sax_eval(ztr, zte), 3), r_drift="-")]
    for tol in (0.1, 0.2):
        for kk in (8, 16, 32, 64):
            r, tk = abba_eval(ztr, zte, tol, kk)
            rd, _ = abba_eval(ztr, zte, tol, kk, drift=True)
            rows.append(dict(method="ABBA", param=f"tol={tol}, k={kk}", tokens=round(tk, 1),
                             r=round(r, 3), r_drift=round(rd, 3)))
            print(f"  sweep tol={tol} k={kk}: tok={tk:.1f} r={r:.3f} drift={rd:.3f}", flush=True)
    t4b = pd.DataFrame(rows)
    t4a.to_csv(C.OUT / "T4a_abba_vs_sax.csv", index=False)
    t4b.to_csv(C.OUT / "T4b_abba_k_sweep.csv", index=False)
    U.write_md(C.OUT / "T4_abba_vs_sax.md", "Table 4. SAX vs ABBA (v2, participant-wise split)",
               [t4a.to_markdown(index=False), "## k sweep (alpha)", t4b.to_markdown(index=False)])


if __name__ == "__main__":
    main()
