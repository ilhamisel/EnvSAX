"""Adim 3 - PARAMETRE DUYARLILIGI (Tablo 3a-c, Sekil 2).

Pilot 04_sensitivity.py ile ayni tarama; fark: filtre ve zarf SUREKLI kayitta (Z1).
Katilimci basina 40 rastgele pencere. Kesim noktalari fit katilimcilarindan.

  (A) token yogunlugu x SAX alfabesi -> zarf sekil sadakati
  (B) guc alfabesi                   -> AD-CN ayirt ediciligi korunumu (r_d)
  (R) referans: ham bant dalga formunda SAX (onerinin ilk hali)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt, hilbert
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import symbolic as S
import common as U

BN = C.BAND_NAMES
N_SEG = 40
TPS_GRID = [1, 2, 4, 8, 16]
ALPHA_GRID = [4, 8, 16, 32]
POWA_GRID = [2, 4, 8, 16, 32]
SOS = {b: S.make_sos(lo, hi, C.SFREQ, C.FILTER_ORDER) for b, (lo, hi) in C.BANDS.items()}


def zscore(x):
    return ((x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-9)).astype(np.float32)


def main():
    df = U.subjects()
    rng = np.random.default_rng(C.SEED)
    starts = {}
    for r in df.itertuples():
        st = C.SEG_STEP * np.arange(r.n_seg)
        starts[r.participant_id] = np.sort(rng.choice(st, size=min(N_SEG, len(st)), replace=False))
    fitmask = np.concatenate([np.full(len(starts[s]), f)
                              for s, f in zip(df.participant_id, df.fit_split)])
    subj = np.concatenate([np.full(len(starts[s]), i) for i, s in enumerate(df.participant_id)])
    y = df.label.values

    def cut(sig, sid):
        idx = starts[sid][:, None] + np.arange(C.SEG_LEN)[None, :]
        return sig[:, idx].transpose(1, 0, 2)                    # (n, 19, 2000)

    def collect(band, fn):
        out = []
        for sid in df.participant_id:
            xb = sosfiltfilt(SOS[band], np.load(C.CACHE / f"raw_{sid}.npy").astype(np.float64), axis=-1)
            out.append(fn(xb, sid))
        return np.concatenate(out)

    rows = []
    # (A) ---------------------------------------------------------------------
    for tps in TPS_GRID:
        w = int(tps * C.SEG_SEC)
        win = max(1, int(round(C.SFREQ / (2.0 * tps))))
        zb = {}
        for b in BN:
            zb[b] = collect(b, lambda xb, sid: zscore(cut(np.log10(uniform_filter1d(
                np.abs(hilbert(xb, axis=-1)), size=win, axis=-1, mode="nearest") + 1e-9), sid)))
        for a in ALPHA_GRID:
            rs = []
            for b in BN:
                sx = S.SAX(w, a).fit(zb[b][fitmask])
                rs.append(np.corrcoef(zb[b].ravel(), sx.inverse(sx.transform(zb[b]), C.SEG_LEN).ravel())[0, 1])
            tok = w * len(BN)
            rows.append(dict(axis="A", tps=tps, alphabet=a, token=tok,
                             compression=round(C.SEG_LEN / tok, 1), shape_r=round(float(np.mean(rs)), 3)))
            print(f"  A tps={tps:2d} a={a:2d} r={np.mean(rs):.3f}", flush=True)
        del zb

    # (B) ---------------------------------------------------------------------
    lp = np.stack([collect(b, lambda xb, sid: np.log10((cut(xb, sid) ** 2).mean(-1) + 1e-12))
                   for b in BN], -1)                              # (n, 19, 5)
    po = 10.0 ** lp; ro = po / po.sum(-1, keepdims=True)
    so = np.array([ro[subj == i].mean(0).ravel() for i in range(len(df))])
    d_o = U.cohens_d(so[y == 1], so[y == 0])
    for L in POWA_GRID:
        pq = S.PowerQuantizer(L).fit({b: lp[fitmask, :, k] for k, b in enumerate(BN)})
        lph = np.stack([pq.inverse(b, pq.transform(b, lp[:, :, k])) for k, b in enumerate(BN)], -1)
        ps = 10.0 ** lph; rs_ = ps / ps.sum(-1, keepdims=True)
        ss = np.array([rs_[subj == i].mean(0).ravel() for i in range(len(df))])
        rd = float(np.corrcoef(d_o, U.cohens_d(ss[y == 1], ss[y == 0]))[0, 1])
        rows.append(dict(axis="B", power_alphabet=L, r_d=round(rd, 4),
                         rbp_mae=round(float(np.abs(ro - rs_).mean()), 5)))
        print(f"  B L={L:2d} r_d={rd:.4f}", flush=True)

    # (R) ---------------------------------------------------------------------
    wave = {b: collect(b, lambda xb, sid: zscore(cut(xb, sid))) for b in BN}
    for rho in [0.15, 0.3, 0.6, 1.0]:
        rs, tok = [], 0
        for b in BN:
            w = max(4, int(round(2 * C.BANDS[b][1] * rho * C.SEG_SEC)))
            tok += w
            sx = S.SAX(w, C.ALPHABET).fit(wave[b][fitmask])
            rs.append(np.corrcoef(wave[b].ravel(), sx.inverse(sx.transform(wave[b]), C.SEG_LEN).ravel())[0, 1])
        rows.append(dict(axis="R", rho=rho, alphabet=C.ALPHABET, token=tok,
                         compression=round(C.SEG_LEN / tok, 1), shape_r=round(float(np.mean(rs)), 3)))
        print(f"  R rho={rho:.2f} r={np.mean(rs):.3f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(C.OUT / "T3_sensitivity.csv", index=False)
    U.write_md(C.OUT / "T3_sensitivity.md", "Table 3. Parameter sensitivity (v2)",
               [out[out.axis == ax].dropna(axis=1, how="all").to_markdown(index=False)
                for ax in ("A", "B", "R")])


if __name__ == "__main__":
    main()
