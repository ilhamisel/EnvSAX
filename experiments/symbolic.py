"""Ritim-bant ayristirma + SAX / ABBA sembolizasyonu ve ters donusumleri.

Tasarim notu (hakem Bilimsel-2 / Yontem-10'a karsilik):
  Sembolizasyon iki bileseni AYRI kodlar:
    (1) SEKIL  -> bant sinyalinin birim varyansa olceklenmis halinin SAX sembolleri
    (2) OLCEK  -> bandin log gucunun ayrik seviye sembolu
  Boylece genlik/spektral bilgi sembolik temsilde acikca korunur; "sembolizasyon
  spektral bilgiyi yok eder mi?" sorusu olculebilir hale gelir.

Sizinti onlemi: tum kesim noktalari / kume merkezleri YALNIZ egitim katindan
ogrenilir (fit), teste yalnizca transform uygulanir.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, hilbert
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans

EPS = 1e-12


# --------------------------------------------------------------------------
# ritim bantlari
# --------------------------------------------------------------------------
def make_sos(low, high, sfreq, order=4):
    nyq = sfreq / 2.0
    return butter(order, [low / nyq, min(high / nyq, 0.999)], btype="band", output="sos")


def band_decompose(X, bands, sfreq, order=4):
    """X (..., T) -> {band: (..., T)} zero-phase Butterworth band-pass."""
    return {b: sosfiltfilt(make_sos(lo, hi, sfreq, order), X, axis=-1).astype(np.float32)
            for b, (lo, hi) in bands.items()}


def band_logpower(xb):
    """(..., T) -> (...) log10 ortalama guc."""
    return np.log10(np.mean(xb.astype(np.float64) ** 2, axis=-1) + EPS)


def band_envelope(xb, sfreq, tokens_per_sec=4.0, smooth=True):
    """Bant sinyalinin analitik (Hilbert) genlik zarfi -> log10 zarf serisi.

    Zarf, bandin zaman icindeki spektral gucunu tasir; ham dalga formunun aksine
    yavas degisir ve buyuk oranda sikistirilabilir. AH'nin EEG imzasi (alfa
    gucunde azalma, teta/delta artisi = "slowing") tam olarak bu alanda yasar.
    """
    env = np.abs(hilbert(xb.astype(np.float64), axis=-1))
    if smooth:
        win = max(1, int(round(sfreq / (2.0 * tokens_per_sec))))
        if win > 1:
            env = uniform_filter1d(env, size=win, axis=-1, mode="nearest")
    return np.log10(env + 1e-9).astype(np.float32)


# --------------------------------------------------------------------------
# PAA
# --------------------------------------------------------------------------
def paa(x, w):
    """(..., T) -> (..., w). Kesirli sinirlarda integral ortalamasi."""
    T = x.shape[-1]
    cs = np.concatenate([np.zeros(x.shape[:-1] + (1,), dtype=np.float64),
                         np.cumsum(x.astype(np.float64), axis=-1)], axis=-1)
    edges = np.linspace(0.0, T, w + 1)
    lo = np.floor(edges).astype(int)
    hi = np.minimum(np.ceil(edges).astype(int), T)
    frac = edges - lo
    integ = cs[..., lo] + frac * (cs[..., hi] - cs[..., lo])
    return ((integ[..., 1:] - integ[..., :-1]) / np.diff(edges)).astype(np.float32)


def paa_inverse(p, T):
    """(..., w) -> (..., T) parcali sabit geri acilim."""
    w = p.shape[-1]
    idx = np.minimum((np.arange(T) * w) // T, w - 1)
    return p[..., idx]


# --------------------------------------------------------------------------
# SAX (sekil)
# --------------------------------------------------------------------------
class SAX:
    """Birim varyansli sinyaller icin SAX. Kesim noktalari egitimden ogrenilir."""

    def __init__(self, w, alphabet=8):
        self.w = w
        self.a = alphabet
        self.breaks = None
        self.reps = None

    def fit(self, x_unit):
        p = paa(x_unit, self.w).ravel()
        q = np.linspace(0, 100, self.a + 1)[1:-1]
        self.breaks = np.percentile(p, q).astype(np.float32)
        s = np.searchsorted(self.breaks, p)
        self.reps = np.array([p[s == k].mean() if np.any(s == k) else 0.0
                              for k in range(self.a)], dtype=np.float32)
        return self

    def transform(self, x_unit):
        return np.searchsorted(self.breaks, paa(x_unit, self.w)).astype(np.int8)

    def inverse(self, sym, T):
        return paa_inverse(self.reps[sym.astype(int)], T)


# --------------------------------------------------------------------------
# guc seviyesi kuantalayici (olcek)
# --------------------------------------------------------------------------
class PowerQuantizer:
    """Bant log gucunu ayrik seviyelere kuantalar (bant basina ayri kesim noktasi)."""

    def __init__(self, levels=8):
        self.L = levels
        self.breaks = {}
        self.reps = {}

    def fit(self, logp_by_band):
        for b, v in logp_by_band.items():
            v = np.asarray(v).ravel()
            q = np.linspace(0, 100, self.L + 1)[1:-1]
            br = np.percentile(v, q)
            self.breaks[b] = br
            s = np.searchsorted(br, v)
            self.reps[b] = np.array([v[s == k].mean() if np.any(s == k) else v.mean()
                                     for k in range(self.L)])
        return self

    def transform(self, b, v):
        return np.searchsorted(self.breaks[b], v).astype(np.int8)

    def inverse(self, b, s):
        return self.reps[b][s.astype(int)]


# --------------------------------------------------------------------------
# ABBA
# --------------------------------------------------------------------------
def abba_compress(x, tol=0.1):
    """Adaptif parcali dogrusal yaklasim. -> (m,2) [uzunluk, genlik degisimi]."""
    n = len(x)
    pieces = []
    i = 0
    t2 = tol * tol
    while i < n - 1:
        j = i + 1
        last = j
        while j < n - 1:
            k = j + 1 - i
            seg = x[i:j + 2]
            line = np.linspace(x[i], x[j + 1], k + 1)
            if np.sum((seg - line) ** 2) > t2 * k:
                break
            last = j + 1
            j += 1
        j = max(last, i + 1)
        pieces.append((j - i, x[j] - x[i]))
        i = j
    return np.asarray(pieces, dtype=np.float64) if pieces else np.zeros((0, 2))


class ABBA:
    """ABBA: sikistir (parcali dogrusal) + kumele (k-means) -> sembol dizisi."""

    def __init__(self, tol=0.1, k=8, scl=1.0, seed=0):
        self.tol = tol
        self.k = k
        self.scl = scl
        self.seed = seed
        self.centers = None
        self.mu = None
        self.sd = None

    def _norm(self, P):
        return np.column_stack([self.scl * (P[:, 0] - self.mu[0]) / (self.sd[0] + EPS),
                                (P[:, 1] - self.mu[1]) / (self.sd[1] + EPS)])

    def fit(self, pieces_list):
        P = np.vstack([p for p in pieces_list if len(p)])
        self.mu = P.mean(0)
        self.sd = P.std(0)
        km = KMeans(self.k, n_init=10, random_state=self.seed).fit(self._norm(P))
        lab = km.labels_
        self.centers = np.array([P[lab == c].mean(0) if np.any(lab == c) else P.mean(0)
                                 for c in range(self.k)])
        self._km = km
        return self

    def transform(self, pieces):
        if not len(pieces):
            return np.zeros(0, dtype=np.int8)
        return self._km.predict(self._norm(pieces)).astype(np.int8)

    def inverse(self, sym, x0, T):
        """Sembollerden parcali dogrusal sinyali yeniden kur."""
        if not len(sym):
            return np.full(T, x0, dtype=np.float32)
        C = self.centers[sym.astype(int)]
        lens = np.maximum(np.round(C[:, 0]).astype(int), 1)
        y = [x0]
        for L, inc in zip(lens, C[:, 1]):
            y.extend(np.linspace(y[-1], y[-1] + inc, L + 1)[1:])
        y = np.asarray(y, dtype=np.float32)
        if len(y) >= T:
            return y[:T]
        return np.interp(np.linspace(0, len(y) - 1, T),
                         np.arange(len(y)), y).astype(np.float32)


# --------------------------------------------------------------------------
# yardimci
# --------------------------------------------------------------------------
def unit_scale(xb):
    """(..., T) -> birim std sinyal ve std degeri."""
    sd = xb.std(axis=-1, keepdims=True)
    return (xb / (sd + EPS)).astype(np.float32), sd.squeeze(-1)


def psd(x, sfreq, nperseg=512):
    f, p = welch(x, fs=sfreq, nperseg=min(nperseg, x.shape[-1]), axis=-1)
    return f, p
