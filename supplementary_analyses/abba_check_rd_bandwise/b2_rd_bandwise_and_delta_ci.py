"""V3 - B2: mevcut çıktılardan iki betimsel hesap (yeni eğitim/deney YOK).

(a) r_d ayrıştırması (cache_v2/fidelity.npz: d_o, d_s, d_oof; 95 = 19 kanal x 5 bant, kanal-major):
    - yalnızca bant ortalamalarından kurulan profilin özgün profille korelasyonu
    - bant ortalaması çıkarılmış r_d (sembolik ve katlar dışı)
    - bant içi r_d (19 kanal)
    - referans dağılım: sembolik profilde kanal etiketleri her bant içinde rastgele karıştırıldığında
      r_d (bant ortalamaları korunur, kanal örüntüsü bozulur; 10.000 permütasyon)
    V2'de STATISTICAL_AUDIT §11'deki değerleri üreten betik pakette yoktu; bu betik onları yeniden üretir.
(b) M1 - M1s AUC farkı için eşleştirilmiş, tabakalı katılımcı bootstrap %95 GA'sı
    (experiments/outputs/oof_classify.csv; tekrar ortalamalı katılımcı olasılıkları; 2.000 tekrar).
Çıktı: analyses/rd_bandwise.csv, analyses/m1_m1s_delta_ci.csv, analyses/b2_summary.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiments"
sys.path.insert(0, str(EXP))
import config as C  # noqa: E402
import common as U  # noqa: E402

BN = list(C.BAND_NAMES)
rng = np.random.default_rng(C.SEED)

# ------------------------------------------------------------------ (a)
f = np.load(C.CACHE / "fidelity.npz")
do, ds, dof = (f[k].reshape(19, 5) for k in ("d_o", "d_s", "d_oof"))
r = lambda a, b: float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
bm = do.mean(0, keepdims=True)
rows = {"r_d (toplam)": r(do, ds), "r_d katlar dışı (toplam)": r(do, dof),
        "yalnız bant ortalaması profili": r(do, np.broadcast_to(bm, do.shape)),
        "bant ort. çıkarılmış r_d": r(do - bm, ds - ds.mean(0, keepdims=True)),
        "bant ort. çıkarılmış r_d, katlar dışı": r(do - bm, dof - dof.mean(0, keepdims=True))}
for k, b in enumerate(BN):
    rows[f"bant içi r_d: {b}"] = float(np.corrcoef(do[:, k], ds[:, k])[0, 1])
    rows[f"bant içi r_d katlar dışı: {b}"] = float(np.corrcoef(do[:, k], dof[:, k])[0, 1])
null = np.empty(10000)
for i in range(len(null)):
    perm = np.stack([ds[rng.permutation(19), k] for k in range(5)], 1)
    null[i] = r(do, perm)
rows["referans: kanal karıştırılmış r_d, ortalama"] = float(null.mean())
rows["referans: kanal karıştırılmış r_d, %95 üst"] = float(np.quantile(null, .95))
rows["referans: kanal karıştırılmış r_d, maks"] = float(null.max())
ra = pd.Series(rows, name="değer").round(3)
ra.to_csv(HERE / "rd_bandwise.csv")

# ------------------------------------------------------------------ (b)
df = U.subjects()
y = df.label.values
oof = pd.read_csv(C.OUT / "oof_classify.csv")
_, avg = U.summarize(oof, y)
p1, p1s = avg["M1"].values, avg["M1s"].values
yy = y[avg.index] if len(avg) == len(y) else y
d0 = roc_auc_score(yy, p1) - roc_auc_score(yy, p1s)
idx1, idx0 = np.where(yy == 1)[0], np.where(yy == 0)[0]
bs = []
for _ in range(2000):
    i = np.r_[rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))]
    bs.append(roc_auc_score(yy[i], p1[i]) - roc_auc_score(yy[i], p1s[i]))
lo, hi = np.quantile(bs, [.025, .975])
rb = pd.DataFrame([dict(dAUC=round(d0, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3), n_boot=2000)])
rb.to_csv(HERE / "m1_m1s_delta_ci.csv", index=False)

(HERE / "b2_summary.md").write_text(
    "# B2: r_d ayrıştırması ve M1–M1s ΔAUC GA'sı\n\n" + ra.to_frame().to_markdown() + "\n\n" +
    rb.to_markdown(index=False) + "\n", encoding="utf-8")
print(ra.to_string()); print(rb.to_string(index=False))
