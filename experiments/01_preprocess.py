"""Adim 1 - ds004504 derivatives -> 200 Hz; bant filtresi ve Hilbert zarfi SUREKLI
kayitta (Z1), ardindan 10 s / %50 ortusmeli pencereleme.

Pilotta filtre ve Hilbert her 10 s'lik segmentte ayri uygulaniyordu; bu, ozellikle
0.5 Hz delta ve zarf icin kenar etkisi yaratir. Burada tum kayit filtrelenir,
zarf tum kayitta hesaplanir ve ancak sonra pencerelenir.

Ciktilar (C.CACHE):
  raw_<sub>.npy   (19, T) float32, uV, 200 Hz        -> mikrodurum, duyarlilik
  paa_<sub>.npz   paa (n,19,5,W) float32 : segment-ici z-skorlu log zarfin PAA'si
                  lp  (n,19,5)   float32 : log10 ortalama bant gucu (pencere icinde)
  zenv_<sub>.npy  (n,19,5,2000) float16  : z-skorlu log zarf (sadakat icin)
  subjects.csv, ch_names.npy
"""
import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from scipy.signal import sosfiltfilt, hilbert
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import symbolic as S

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


def windows(n_t):
    n = 1 + (n_t - C.SEG_LEN) // C.SEG_STEP
    return C.SEG_STEP * np.arange(max(n, 0))


def main():
    parts = pd.read_csv(C.DATASET / "participants.tsv", sep="\t")
    parts = parts[parts.Group.isin(C.PILOT_GROUPS)].reset_index(drop=True)
    print(f"katilimci: {len(parts)} ({(parts.Group == 'A').sum()} AD / "
          f"{(parts.Group == 'C').sum()} CN)", flush=True)
    sos = {b: S.make_sos(lo, hi, C.SFREQ, C.FILTER_ORDER) for b, (lo, hi) in C.BANDS.items()}
    win = max(1, int(round(C.SFREQ / (2.0 * C.ENV_TOKENS_PER_SEC))))   # 25 ornek = 0.125 s

    rows, ch_ref, t0 = [], None, time.time()
    for i, r in parts.iterrows():
        sid = r.participant_id
        raw = mne.io.read_raw_eeglab(C.DERIV / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set",
                                     preload=True, verbose="ERROR")
        ch_ref = ch_ref or raw.ch_names
        assert raw.ch_names == ch_ref, f"{sid}: kanal sirasi farkli"
        raw.resample(C.SFREQ, npad="auto")
        x = (raw.get_data() * 1e6).astype(np.float64)          # (19, T) uV
        np.save(C.CACHE / f"raw_{sid}.npy", x.astype(np.float32))
        st = windows(x.shape[1]); n = len(st)
        idx = st[:, None] + np.arange(C.SEG_LEN)[None, :]       # (n, 2000)

        paa = np.zeros((n, 19, len(C.BAND_NAMES), C.W), dtype=np.float32)
        lp = np.zeros((n, 19, len(C.BAND_NAMES)), dtype=np.float32)
        zenv = np.zeros((n, 19, len(C.BAND_NAMES), C.SEG_LEN), dtype=np.float16)
        for k, b in enumerate(C.BAND_NAMES):
            xb = sosfiltfilt(sos[b], x, axis=-1)                # surekli kayitta filtre
            env = np.abs(hilbert(xb, axis=-1))                  # surekli kayitta zarf
            env = uniform_filter1d(env, size=win, axis=-1, mode="nearest")
            le = np.log10(env + 1e-9)
            seg_le = le[:, idx].transpose(1, 0, 2)              # (n, 19, 2000)
            z = (seg_le - seg_le.mean(-1, keepdims=True)) / (seg_le.std(-1, keepdims=True) + 1e-9)
            paa[:, :, k, :] = S.paa(z, C.W)
            zenv[:, :, k, :] = z.astype(np.float16)
            seg_xb = xb[:, idx].transpose(1, 0, 2)
            lp[:, :, k] = np.log10((seg_xb ** 2).mean(-1) + 1e-12)
        np.savez(C.CACHE / f"paa_{sid}.npz", paa=paa, lp=lp)
        np.save(C.CACHE / f"zenv_{sid}.npy", zenv)
        rows.append(dict(participant_id=sid, group=r.Group, label=int(r.Group == "A"),
                         age=r.Age, gender=r.Gender, mmse=r.MMSE,
                         dur_s=round(x.shape[1] / C.SFREQ, 1), n_seg=n))
        print(f"  [{i + 1:2d}/{len(parts)}] {sid} {r.Group} {rows[-1]['dur_s']:7.1f}s -> "
              f"{n:4d} segment ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(C.CACHE / "subjects.csv", index=False)
    np.save(C.CACHE / "ch_names.npy", np.array(ch_ref))
    print(f"\ntoplam segment {df.n_seg.sum()} (AD {df[df.label == 1].n_seg.sum()} / "
          f"CN {df[df.label == 0].n_seg.sum()}); sure {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
