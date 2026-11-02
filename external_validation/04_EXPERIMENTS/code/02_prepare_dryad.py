"""Dryad Benninger/Shor -> katılımcı başına lp (n,19,5) + pencere QC + manifesto.

Protokol §4, §5 ve §11.1 uygulanır. Etiket bilgisi yalnızca manifestoya yazılır;
ön işleme ve QC grup bilgisi kullanmaz. Kanal adı atanmaz: sütunlar ch01..ch19.

  - yalnız alz_c1_new.mat ve controls_c1_new.mat okunur (MCI/dep/schiz okunmaz)
  - AH: katılımcı başına ilk kayıt (G[0, sütun])
  - baştaki/sondaki 19 kanalın tamamen sıfır olduğu örnekler kesilir, sonra 5 s kenar kırpması
  - 500 Hz -> 200 Hz (MNE resample npad='auto'); 200 Hz kayıtlar yeniden örneklenmez
  - pencere QC (yalnız bariz kayıt hataları): herhangi kanalda std < 0.1, |x| > 500, NaN

Kullanım: python 02_prepare_dryad.py <Dryad .mat dosyalarının klasörü>
Çıktı: 02_DATA/dryad_shor_benninger/derived/lp_<id>.npz, participant_manifest.csv
Kesilirse kaldığı katılımcıdan devam eder.
"""
import sys
import numpy as np
import pandas as pd
import h5py
import evlib as E

RAW = E.Path(sys.argv[1]) if len(sys.argv) > 1 else E.ROOT / "02_DATA" / "dryad_shor_benninger" / "raw"
OUT = E.ROOT / "02_DATA" / "dryad_shor_benninger" / "derived"
OUT.mkdir(parents=True, exist_ok=True)
MAN = E.ROOT / "02_DATA" / "dryad_shor_benninger" / "participant_manifest.csv"
EDGE_S, FLAT_UV, MAX_UV = 5, 0.1, 500.0


def val(f, ref):
    d = f[ref]
    c = d.attrs.get("MATLAB_class", b"").decode()
    if c == "canonical empty" or d.attrs.get("MATLAB_empty", 0):
        return None
    if c == "char":
        return "".join(chr(x) for x in np.asarray(d[()]).ravel())
    if c == "cell":
        return [val(f, r) for r in np.asarray(d[()]).ravel()]
    return d


def records():
    """-> (id, group, label, age, sex, fs, file, h5-dataset-getter)"""
    fa = h5py.File(RAW / "alz_c1_new.mat", "r"); ga = fa["alz_r"]
    for col in range(ga["G"].shape[1]):
        if val(fa, ga["G"][0, col]) is None:
            continue
        fn = val(fa, ga["eeg_files"][0, col])
        age = val(fa, ga["age"][0, col]); fs = val(fa, ga["g"][0, col]); m = val(fa, ga["m"][0, col])
        yield dict(id=fn.split("_")[0], group="AD", label=1, age=float(np.ravel(age[()])[0]),
                   sex=(val(fa, ga["sex"][0, col]) or "").strip(), fs=float(np.ravel(fs[()])[0]),
                   m_field=float(np.ravel(m[()])[0]), file=fn,
                   n_recordings=sum(val(fa, ga["G"][r, col]) is not None for r in range(3))), \
            (lambda c=col: fa[ga["G"][0, c]])
    fc = h5py.File(RAW / "controls_c1_new.mat", "r"); gc = fc["controls_r"]
    for i in range(gc["G"].shape[1]):
        age = val(fc, gc["age"][0, i]); fs = val(fc, gc["g"][0, i])
        yield dict(id=f"CN{i:03d}", group="CN", label=0, age=float(np.ravel(age[()])[0]),
                   sex=(val(fc, gc["sex"][0, i]) or "").strip(), fs=float(np.ravel(fs[()])[0]),
                   m_field=np.nan, file=val(fc, gc["n"][0, i])[0], n_recordings=1), \
            (lambda j=i: fc[np.asarray(fc[gc["G"][0, j]][()]).ravel()[0]])


def prepare(x, fs):
    """x (T,19) -> (19,T') µV 200 Hz, kırpma bilgisi."""
    nz = np.where(np.any(x != 0, axis=1))[0]
    lead, trail = int(nz[0]), int(len(x) - 1 - nz[-1])
    x = x[nz[0]:nz[-1] + 1]
    e = int(EDGE_S * fs)
    x = x[e:len(x) - e].T.astype(np.float64)
    if fs != E.C.SFREQ:
        x = E.resample_to_200(x, fs)
    return x, lead, trail


def window_qc(x):
    st = E.windows(x.shape[1])
    flat, big, nan = [], [], []
    for s in st:
        w = x[:, s:s + E.C.SEG_LEN]
        nan.append(bool(np.isnan(w).any()))
        flat.append(bool((np.nanstd(w, axis=1) < FLAT_UV).any()))
        big.append(bool((np.abs(w) > MAX_UV).any()))
    return np.array(flat), np.array(big), np.array(nan)


def main():
    rows = pd.read_csv(MAN).to_dict("records") if MAN.exists() else []
    done = {r["id"] for r in rows}
    for meta, get in records():
        if meta["id"] in done:
            continue
        d = get()
        x = np.asarray(d[()], dtype=np.float64)
        if x.shape[1] != 19:
            x = x.T
        dur = len(x) / meta["fs"]
        x, lead, trail = prepare(x, meta["fs"])
        flat, big, nan = window_qc(x)
        lp, _, st = E.encode_continuous(np.nan_to_num(x))
        clean = ~(flat | big | nan)
        rms_ch = np.sqrt(((x - x.mean(1, keepdims=True)) ** 2).mean(1))
        np.savez_compressed(OUT / f"lp_{meta['id']}.npz", lp=lp, starts=st, flat=flat, big=big, nan=nan,
                            log_std_ch=np.log10(rms_ch + 1e-12),
                            corr=np.corrcoef(x).astype(np.float32))
        rows.append(dict(meta, duration_s=round(dur, 3), lead_zero_samples=lead, trail_zero_samples=trail,
                         n_windows=len(st), n_flat=int(flat.sum()), n_big=int(big.sum()), n_nan=int(nan.sum()),
                         n_clean=int(clean.sum()), rms_median_uv=float(np.median(rms_ch))))
        pd.DataFrame(rows).to_csv(MAN, index=False)
        print(meta["id"], meta["group"], meta["fs"], f"{dur:.0f}s", len(st), int(clean.sum()),
              f"rms={np.median(rms_ch):.2f}", flush=True)


if __name__ == "__main__":
    main()
