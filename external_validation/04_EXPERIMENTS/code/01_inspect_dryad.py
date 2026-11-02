"""Dryad .mat dosyalarının YAPI denetimi (sonuç hesaplamaz, etiket kullanmaz).

Her dosya için: MAT sürümü (v7.3/HDF5 ya da v5), değişkenler, boyutlar, veri tipi,
öznitelikler; varsa kanal adı / örnekleme hızı / olay (göz açık-kapalı) alanları;
katılımcı başına geçerli (NaN/sıfır olmayan) uzunluk; genlik ölçeği (birim tahmini için).
Büyük diziler yalnızca dilim dilim okunur.

Kullanım: python -I 01_inspect_dryad.py   ->  02_DATA/dryad_shor_benninger/metadata/structure.json
"""
import json
import numpy as np
from evlib import ROOT

import os, sys
# Dryad dosyalarının klasörü argümanla verilebilir (varsayılan: 02_DATA/dryad_shor_benninger/raw).
RAW = __import__("pathlib").Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "02_DATA" / "dryad_shor_benninger" / "raw"
META = ROOT / "02_DATA" / "dryad_shor_benninger" / "metadata"; META.mkdir(parents=True, exist_ok=True)


def is_v73(p):
    with open(p, "rb") as f:
        return b"MATLAB 7.3" in f.read(128)


def h5_walk(p):
    import h5py
    out = {}
    with h5py.File(p, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                rec = dict(shape=obj.shape, dtype=str(obj.dtype), chunks=obj.chunks,
                           attrs={k: str(v)[:200] for k, v in obj.attrs.items()})
                if obj.size <= 512 and obj.dtype.kind in "uifS":
                    v = obj[()]
                    if obj.attrs.get("MATLAB_class", b"") == b"char":
                        v = "".join(chr(c) for c in np.asarray(v).ravel())
                    rec["value"] = str(v)[:500]
                out[name] = rec
        f.visititems(visit)
        big = max(((k, v) for k, v in out.items() if len(v["shape"]) >= 2),
                  key=lambda kv: np.prod(kv[1]["shape"]), default=(None, None))[0]
        if big:
            d = f[big]
            out["_main"] = big
            out["_axis_sizes"] = d.shape
            # her eksen boyunca ilk dilimlerden ölçek ve geçerli uzunluk
            sl = tuple(slice(0, min(s, 4)) if i != int(np.argmax(d.shape)) else slice(None)
                       for i, s in enumerate(d.shape))
            x = np.asarray(d[sl], dtype=np.float64)
            out["_sample_stats"] = dict(shape=x.shape, nan_frac=float(np.isnan(x).mean()),
                                        zero_frac=float((x == 0).mean()),
                                        abs_median=float(np.nanmedian(np.abs(x))),
                                        std=float(np.nanstd(x)), min=float(np.nanmin(x)), max=float(np.nanmax(x)))
    return out


def v5_walk(p):
    import scipy.io as sio
    info = sio.whosmat(p)
    return {"_whosmat": [(n, list(s), c) for n, s, c in info]}


def main():
    res = {}
    for p in [RAW / "alz_c1_new.mat", RAW / "controls_c1_new.mat"]:  # MCI/dep/schiz okunmaz
        res[p.name] = dict(v73=is_v73(p), size=p.stat().st_size,
                           **(h5_walk(p) if is_v73(p) else v5_walk(p)))
    txt = RAW / "AUTHOR_DATASET_SHOR_BENNINGER.txt"
    if txt.exists():
        res["README"] = txt.read_text(encoding="utf-8", errors="replace")
    (META / "structure.json").write_text(json.dumps(res, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
