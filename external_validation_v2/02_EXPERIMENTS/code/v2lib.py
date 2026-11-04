"""V2 ortak kütüphanesi: ilk dış deneyin evlib'ini (değiştirmeden) içe aktarır.

Kalibrasyon fonksiyonu etiket almaz; pencere seçimi katılımcı kimliğinden türetilen tohumla yapılır,
böylece kod defteri katılımcı sırasından bağımsızdır.
"""
import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[2]                      # EnvSAX_EXTERNAL_VALIDATION_V2
V1 = V2.parent / "EnvSAX_EXTERNAL_VALIDATION"
sys.path.insert(0, str(V1 / "04_EXPERIMENTS" / "code"))
import evlib as E  # noqa: E402
from evlib import BN, NB, NCH, C, U, rbp, profiles, fidelity  # noqa: E402,F401

DATA = V1 / "02_DATA" / "dryad_shor_benninger"
V1_DRY = V1 / "04_EXPERIMENTS" / "dryad_frozen"
V1_SRC = V1 / "04_EXPERIMENTS" / "source_reference"
ENC_FROZEN = V1 / "03_PROTOCOL" / "frozen_encoder_ahepa.npz"
PROT = V2 / "01_PROTOCOL"
EXP = V2 / "02_EXPERIMENTS"
RES = V2 / "03_RESULTS"
FIG = V2 / "04_FIGURES"
L = C.POWER_ALPHABET
N_CAL_WIN = 25


def manifest():
    """İlk deneyin dahil etme kararı (139 katılımcı), aynı sıra."""
    m = pd.read_csv(V1_DRY / "participant_manifest_analysis.csv")
    return m[m.included].reset_index(drop=True)


def load_lp(pid):
    """QC'den geçen pencereler (ilk deneydeki load(qc=True) ile aynı)."""
    d = np.load(DATA / "derived" / f"lp_{pid}.npz")
    keep = ~(d["flat"] | d["big"] | d["nan"])
    return d["lp"][keep].astype(np.float64)


def cal_windows(pid, lp, fold):
    rng = np.random.default_rng([C.SEED, int(fold), zlib.crc32(pid.encode())])
    return np.sort(rng.choice(len(lp), size=min(N_CAL_WIN, len(lp)), replace=False))


class Codebook(E.FrozenScale):
    """PowerQuantizer ile aynı tanım. Girdi: {id: lp}; etiket yok."""

    def __init__(self, lp_by_id, fold):
        pq = E.S.PowerQuantizer(L)
        v = {b: [] for b in BN}
        self.windows = {}
        for pid in sorted(lp_by_id):                       # sıra etiketten ve manifestodan bağımsız
            lp = lp_by_id[pid]
            i = cal_windows(pid, lp, fold)
            self.windows[pid] = i
            for k, b in enumerate(BN):
                v[b].append(lp[i, :, k].ravel())
        pq.fit({b: np.concatenate(x) for b, x in v.items()})
        self.breaks = np.stack([pq.breaks[b] for b in BN])
        self.reps = np.stack([pq.reps[b] for b in BN])
        self.ids = sorted(lp_by_id)


class Saved(E.FrozenScale):
    def __init__(self, breaks, reps):
        self.breaks, self.reps = np.asarray(breaks), np.asarray(reps)


def boot_multi(so, ss_by_name, y, nboot=2000, seed=C.SEED + 7):
    """evlib.boot_fidelity ile aynı RNG dizisi; bütün koşullar aynı yeniden örneklemelerde."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ss_by_name}
    for _ in range(nboot):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        for k, ss in ss_by_name.items():
            rows[k].append(fidelity(*profiles(so[i], ss[i], y[i])))
    return {k: pd.DataFrame(v) for k, v in rows.items()}


def occupancy(codes):
    """codes (n,19,5) -> (5, L) seviye kullanım oranları."""
    return np.array([[(codes[..., k] == j).mean() for j in range(L)] for k in range(NB)])


def entropy_bits(freq):
    f = freq[freq > 0]
    return float(-(f * np.log2(f)).sum())
