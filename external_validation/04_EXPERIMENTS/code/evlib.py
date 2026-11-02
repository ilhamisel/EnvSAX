"""Dış doğrulama ortak kütüphanesi.

Kaynak kod (depo kökündeki experiments/) yalnızca içe aktarılır, değiştirilmez.
Ön işleme 01_preprocess.py ile, sadakat ölçütleri 02_fidelity.py,
09_output_v2/analyses/a2_rd_bootstrap.py ve EnvSAX_TRDIZIN_V3/analyses/b2_rd_bandwise_and_delta_ci.py
ile birebir aynı tanımları kullanır.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import sosfiltfilt, hilbert
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

EXP = Path(__file__).resolve().parents[3] / "experiments"
sys.path.insert(0, str(EXP))
import config as C   # noqa: E402
import symbolic as S  # noqa: E402
import common as U    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]          # EnvSAX_EXTERNAL_VALIDATION
BN = list(C.BAND_NAMES)
NCH, NB = 19, len(BN)
# AHEPA (ds004504) kanal sırası; cache_v2/ch_names.npy ile doğrulanır
AHEPA_CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


# ------------------------------------------------------------------ ön işleme
def windows(n_t):
    n = 1 + (n_t - C.SEG_LEN) // C.SEG_STEP
    return C.SEG_STEP * np.arange(max(n, 0))


def resample_to_200(x, sfreq):
    """(19,T) -> 200 Hz; 01_preprocess.py'deki raw.resample(200, npad='auto') ile aynı yol."""
    import mne
    mne.set_log_level("ERROR")
    info = mne.create_info(NCH, sfreq, "eeg")
    raw = mne.io.RawArray(x.astype(np.float64), info, verbose="ERROR")
    raw.resample(C.SFREQ, npad="auto")
    return raw.get_data()


def encode_continuous(x_uv, keep_shape=False):
    """(19,T) µV, 200 Hz sürekli kayıt -> lp (n,19,5) [, zenv (n,19,5,2000) float16, starts].

    01_preprocess.py ile aynı: filtre ve Hilbert sürekli kayıtta, sonra 10 s / %50 pencere.
    """
    sos = {b: S.make_sos(lo, hi, C.SFREQ, C.FILTER_ORDER) for b, (lo, hi) in C.BANDS.items()}
    win = max(1, int(round(C.SFREQ / (2.0 * C.ENV_TOKENS_PER_SEC))))
    st = windows(x_uv.shape[1]); n = len(st)
    idx = st[:, None] + np.arange(C.SEG_LEN)[None, :]
    lp = np.zeros((n, NCH, NB), dtype=np.float32)
    zenv = np.zeros((n, NCH, NB, C.SEG_LEN), dtype=np.float16) if keep_shape else None
    for k, b in enumerate(BN):
        xb = sosfiltfilt(sos[b], x_uv, axis=-1)
        seg_xb = xb[:, idx].transpose(1, 0, 2)
        lp[:, :, k] = np.log10((seg_xb ** 2).mean(-1) + 1e-12)
        if keep_shape:
            env = uniform_filter1d(np.abs(hilbert(xb, axis=-1)), size=win, axis=-1, mode="nearest")
            seg_le = np.log10(env + 1e-9)[:, idx].transpose(1, 0, 2)
            z = (seg_le - seg_le.mean(-1, keepdims=True)) / (seg_le.std(-1, keepdims=True) + 1e-9)
            zenv[:, :, k, :] = z.astype(np.float16)
    return lp, zenv, st


# ------------------------------------------------------------------ dondurulmuş kodlayıcı
class FrozenScale:
    """Kaydedilmiş PowerQuantizer kesim noktaları + temsil değerleri (bant başına)."""

    def __init__(self, path):
        d = np.load(path)
        self.breaks = d["scale_breaks"]      # (5, L-1)
        self.reps = d["scale_reps"]          # (5, L)

    def encode(self, lp):
        return np.stack([np.searchsorted(self.breaks[k], lp[..., k]) for k in range(NB)], -1).astype(np.int8)

    def decode(self, code):
        return np.stack([self.reps[k][code[..., k].astype(np.int64)] for k in range(NB)], -1)


def rbp(lp):
    p = 10.0 ** np.asarray(lp, dtype=np.float64)
    return p / p.sum(-1, keepdims=True)


# ------------------------------------------------------------------ sadakat ölçütleri
def ccc(a, b):
    ma, mb = a.mean(), b.mean()
    return 2 * np.cov(a, b, ddof=0)[0, 1] / (a.var() + b.var() + (ma - mb) ** 2)


def profiles(so, ss, y):
    return U.cohens_d(so[y == 1], so[y == 0]), U.cohens_d(ss[y == 1], ss[y == 0])


def fidelity(do, ds, bands=None):
    """do, ds: (19*nb,) kanal-major profiller -> ölçüt sözlüğü."""
    bands = bands or BN
    nb = len(bands)
    o, s = do.reshape(-1, nb), ds.reshape(-1, nb)
    r = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    out = {"r_d": r(o, s), "spearman": float(spearmanr(do, ds).statistic), "ccc": float(ccc(do, ds)),
           "r_d_band_mean_removed": r(o - o.mean(0, keepdims=True), s - s.mean(0, keepdims=True)),
           "r_band_mean_only_profile": r(o, np.broadcast_to(o.mean(0, keepdims=True), o.shape)),
           "mean_abs_delta_d": float(np.abs(s - o).mean()),
           "mean_abs_d_orig": float(np.abs(o).mean()), "mean_abs_d_dec": float(np.abs(s).mean())}
    for k, b in enumerate(bands):
        out[f"within_r_d_{b}"] = r(o[:, k], s[:, k])
        out[f"within_ccc_{b}"] = float(ccc(o[:, k], s[:, k]))
        out[f"abs_delta_d_{b}"] = float(np.abs(s[:, k] - o[:, k]).mean())
    for k in (10, 20):
        out[f"top{k}_overlap"] = len(set(np.argsort(-np.abs(do))[:k]) & set(np.argsort(-np.abs(ds))[:k]))
    return out


def boot_fidelity(so, ss, y, nboot=2000, seed=C.SEED + 7, bands=None):
    """Grup içi tabakalı KATILIMCI bootstrap; özgün ve çözülmüş aynı örnekte kalır."""
    import pandas as pd
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(nboot):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        rows.append(fidelity(*profiles(so[i], ss[i], y[i]), bands=bands))
    return pd.DataFrame(rows)


def channel_shuffle_null(do, ds, nperm=10000, seed=C.SEED, nb=NB):
    """b2 ile aynı referans: kanal etiketleri her bant içinde karıştırılır (bant ortalaması korunur)."""
    rng = np.random.default_rng(seed)
    o, s = do.reshape(-1, nb), ds.reshape(-1, nb)
    nch = o.shape[0]
    null = np.empty(nperm)
    for i in range(nperm):
        p = np.stack([s[rng.permutation(nch), k] for k in range(nb)], 1)
        null[i] = np.corrcoef(o.ravel(), p.ravel())[0, 1]
    return null
