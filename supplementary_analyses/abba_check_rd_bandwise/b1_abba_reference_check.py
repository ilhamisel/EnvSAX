"""V3 - B1: ABBA uygulamasinin referans uygulamayla karsilastirilmasi.

experiments/04_abba_vs_sax.py'deki ozel ABBA uygulamasi (symbolic.ABBA,
symbolic.abba_compress) ile Elsworth & Guttel'in grubunun yayimladigi fABBA
paketindeki (v1.5.3) referans ABBA bilesenleri AYNI seriler uzerinde karsilastirilir.
Seriler 04 betigindeki rastgelelik sirasi birebir izlenerek yeniden uretilir;
ozel uygulamanin T4a/T4b degerlerini yeniden uretmesi seri esligini dogrular.

Karsilastirilan geri catimlar (hepsi katilimci duzeyinde ayrik egitim/test):
  OZEL      04 betigindeki uygulama (uzunluk tek tek yuvarlanir, T'ye esnetilir)
  REF       fABBA.compress + KMeans(k, n_init="auto", random_state=0; uzunluk ve artis
            std'ye bolunur, scl=1) egitim parcalarinda; test parcalari en yakin
            merkeze atanir; fABBA quantize + inv_compress ile geri catim
            (kumulatif uzunluk yuvarlama). Uzunluk T'den farkliysa kesilir/son
            degerle uzatilir (esnetme yok).
  REF-esn   ayni, ancak ozel uygulamadaki gibi T'ye esnetilir
  SIKISTIRMA yalnizca parcali dogrusal sikistirma (sembollestirme yok) -> ust sinir
  SERI-ICI  fABBA.ABBA test serilerinin 1/4'une (her 4. seri) KENDI kod defteriyle uygulanir (ozgun
            makaledeki kullanim; kod defteri seri basina ayri saklanmalidir)
Yeni siniflandirma deneyi YOKTUR. Cikti: analyses/abba_reference_check.csv/.md
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
import sys
import time
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import fABBA
from fABBA import compress as ref_compress
from fABBA.inverse_t import inv_compress, quantize

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiments"
sys.path.insert(0, str(EXP))
import config as C  # noqa: E402
import common as U  # noqa: E402

spec = importlib.util.spec_from_file_location("abba04", EXP / "04_abba_vs_sax.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
T = C.SEG_LEN
TOLS = [0.05, 0.1, 0.2, 0.4]


def fit_len(rec, stretch):
    rec = np.asarray(rec, dtype=np.float64)
    if stretch:
        return np.interp(np.linspace(0, len(rec) - 1, T), np.arange(len(rec)), rec)
    if len(rec) >= T:
        return rec[:T]
    return np.r_[rec, np.full(T - len(rec), rec[-1])]


def r_of(z, rec):
    return np.corrcoef(z, rec)[0, 1]


def ref_eval(ztr, zte, tol, k):
    P = np.vstack([np.asarray(ref_compress(z, tol=tol))[:, :2] for z in ztr])
    sd = P.std(0); sd[sd == 0] = 1
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(P / sd)
    cen = np.array([P[km.labels_ == c].mean(0) for c in range(k)])
    out = {"REF": [], "REF-esn": [], "SIKISTIRMA": [], "tok": [], "len": []}
    for z in zte:
        p = np.asarray(ref_compress(z, tol=tol))[:, :2]
        lab = km.predict(p / sd)
        rec = inv_compress(quantize(cen[lab]), float(z[0]))
        out["REF"].append(r_of(z, fit_len(rec, False)))
        out["REF-esn"].append(r_of(z, fit_len(rec, True)))
        out["SIKISTIRMA"].append(r_of(z, fit_len(inv_compress(quantize(p), float(z[0])), False)))
        out["tok"].append(len(p)); out["len"].append(len(rec))
    return {k_: float(np.nanmean(v)) for k_, v in out.items()}


def own_eval(zte, tol, k):
    rs = []
    for z in zte[::4]:  # seri ici kullanim: test serilerinin 1/4'u (hiz)
        try:
            ab = fABBA.ABBA(tol=tol, k=k, verbose=0)
            s = ab.fit_transform(z)
            rs.append(r_of(z, fit_len(ab.inverse_transform(s, float(z[0])), False)))
        except Exception:  # parca sayisi < k
            rs.append(np.nan)
    return float(np.nanmean(rs)), int(np.isnan(rs).sum())


def main():
    global T0
    T0 = time.time()
    df = U.subjects()
    rng = np.random.default_rng(C.SEED)
    fit, test = M.pick(df, M.N_SUB, rng)
    t4a = pd.read_csv(C.OUT / "T4a_abba_vs_sax.csv")
    rows = []
    for bi, b in enumerate(C.BAND_NAMES):
        ztr = M.series(fit, bi, M.N_SEG, M.N_CH, rng)
        zte = M.series(test, bi, M.N_SEG, M.N_CH, rng)
        for tol in TOLS:
            r_ozel, tk = M.abba_eval(ztr, zte, tol, M.ABBA_K)
            ref = ref_eval(ztr, zte, tol, M.ABBA_K)
            own, nfail = own_eval(zte, tol, M.ABBA_K)
            pub = t4a[(t4a.band == b) & (t4a.param == f"tol={tol}, k={M.ABBA_K}")].r.iloc[0]
            rows.append(dict(set="T4a", band=b, tol=tol, k=M.ABBA_K, r_T4a=pub, r_OZEL=round(r_ozel, 3),
                             tok_OZEL=round(tk, 1), r_REF=round(ref["REF"], 3), r_REF_esn=round(ref["REF-esn"], 3),
                             r_SIKISTIRMA=round(ref["SIKISTIRMA"], 3), tok_REF=round(ref["tok"], 1),
                             len_REF=round(ref["len"], 0), r_SERI_ICI=round(own, 3), seri_ici_hata=nfail))
            print(round(time.time() - T0), rows[-1], flush=True)
    # alfa k taramasi (04 ile ayni rastgelelik sirasi)
    fit, test = M.pick(df, 12, rng)
    ai = C.BAND_NAMES.index("alpha")
    ztr, zte = M.series(fit, ai, 6, 5, rng), M.series(test, ai, 6, 5, rng)
    t4b = pd.read_csv(C.OUT / "T4b_abba_k_sweep.csv")
    for tol in (0.1, 0.2):
        for kk in (8, 16, 32, 64):
            r_ozel, tk = M.abba_eval(ztr, zte, tol, kk)
            ref = ref_eval(ztr, zte, tol, kk)
            own, nfail = own_eval(zte, tol, kk)
            pub = t4b[t4b.param == f"tol={tol}, k={kk}"].r.iloc[0]
            rows.append(dict(set="T4b", band="alpha", tol=tol, k=kk, r_T4a=pub, r_OZEL=round(r_ozel, 3),
                             tok_OZEL=round(tk, 1), r_REF=round(ref["REF"], 3), r_REF_esn=round(ref["REF-esn"], 3),
                             r_SIKISTIRMA=round(ref["SIKISTIRMA"], 3), tok_REF=round(ref["tok"], 1),
                             len_REF=round(ref["len"], 0), r_SERI_ICI=round(own, 3), seri_ici_hata=nfail))
            print(round(time.time() - T0), rows[-1], flush=True)
    out = pd.DataFrame(rows).rename(columns={"r_T4a": "r_yayimlanan"})
    out.to_csv(HERE / "abba_reference_check.csv", index=False)
    (HERE / "abba_reference_check.md").write_text(
        "# ABBA referans karsilastirmasi (fABBA " + fABBA.__version__ + ")\n\n" + out.to_markdown(index=False) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
