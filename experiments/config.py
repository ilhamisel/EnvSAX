"""Deney v2 - ortak yapilandirma (pilot config.py'nin duzeltilmis hali).

Pilot -> v2 farklari:
  Z1  bant filtresi + Hilbert zarfi SUREKLI kayitta hesaplanir, sonra pencerelenir
  Z2  5 x 5 tekrarli katilimci-temelli capraz dogrulama (RepeatedStratifiedKFold)
  Z11 veri ve onbellek yollari ortam degiskeniyle degistirilebilir

Ortam degiskenleri:
  DS004504_ROOT  ds004504 kok klasoru (varsayilan <depo>/data/ds004504)
  TEZ_CACHE      ara ciktilar (varsayilan <depo>/cache)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("DS004504_ROOT", REPO / "data" / "ds004504"))
DERIV = DATASET / "derivatives"
ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("TEZ_CACHE", REPO / "cache"))
OUT = ROOT / "outputs"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# --- ornekleme / segmentasyon ---------------------------------------------
SFREQ = 200.0
SEG_SEC = 10.0
OVERLAP = 0.5
SEG_LEN = int(SEG_SEC * SFREQ)          # 2000
SEG_STEP = int(SEG_LEN * (1 - OVERLAP))  # 1000

# --- ritim bantlari -------------------------------------------------------
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS)
FILTER_ORDER = 4

# --- sembolizasyon --------------------------------------------------------
ENV_TOKENS_PER_SEC = 4.0
W = int(ENV_TOKENS_PER_SEC * SEG_SEC)   # 40
ALPHABET = 8
POWER_ALPHABET = 8

# --- gruplar --------------------------------------------------------------
GROUPS = {"A": "AD", "C": "CN", "F": "FTD"}
PILOT_GROUPS = ["A", "C"]

# --- degerlendirme protokolu ---------------------------------------------
SEED = 42
N_FOLDS = 5
N_REPEATS = 5
VAL_FRAC = 0.2          # ic dogrulama: egitim katilimcilarinin %20'si (tabakali)
SEG_TRAIN = 120         # egitimde katilimci basina en fazla segment
N_BOOT = 1000           # katilimci-bazli bootstrap
