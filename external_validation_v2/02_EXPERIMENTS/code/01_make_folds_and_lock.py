"""Katılımcı bölmelerini yazar ve protokolü kilitler (sonuçlar hesaplanmadan önce bir kez çalıştırılır).

Etiket yalnızca tabakalama için kullanılır. Kilit, protokol, bölmeler ve deney kodunun sha256 değerlerini içerir;
02_oof_calibration.py çalışmadan önce bunları doğrular.
"""
import hashlib
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import v2lib as V

LOCK = V.PROT / "PROTOCOL_LOCK.json"


def main():
    assert not LOCK.exists(), "protokol zaten kilitli"
    man = V.manifest()
    y = man.label.values
    fold = np.zeros(len(man), int)
    for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y), 1):
        fold[te] = f
    out = pd.DataFrame(dict(id=man.id, group=man.group, fold=fold))
    out.to_csv(V.PROT / "PARTICIPANT_FOLDS.csv", index=False)

    rep = pd.DataFrame(dict(id=man.id))
    for k, (_, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2026)
                                .split(np.zeros(len(y)), y)):
        col = f"rep{k // 5 + 1:02d}"
        if col not in rep:
            rep[col] = 0
        rep.loc[te, col] = k % 5 + 1
    rep.to_csv(V.PROT / "REPEATED_FOLDS.csv", index=False)

    files = ["01_PROTOCOL/OOF_CALIBRATION_PROTOCOL.md", "01_PROTOCOL/PARTICIPANT_FOLDS.csv",
             "01_PROTOCOL/REPEATED_FOLDS.csv", "02_EXPERIMENTS/code/v2lib.py",
             "02_EXPERIMENTS/code/01_make_folds_and_lock.py", "02_EXPERIMENTS/code/02_oof_calibration.py"]
    lock = dict(
        locked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        note="Locked before any V2 result was computed. V2 was designed after the first external evaluation "
             "(frozen r_d 0.804, full-cohort refit r_d 0.992) was known.",
        split=dict(primary="StratifiedKFold(n_splits=5, shuffle=True, random_state=42), manifest order, label used only "
                           "for stratification",
                   secondary="RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2026)"),
        seeds=dict(calibration_windows="default_rng([42, fold, crc32(participant_id)]), 25 windows per participant",
                   bootstrap="default_rng(49), 2000 replicates, within-group participant resampling",
                   channel_shuffle="default_rng(42), 10000 permutations",
                   label_order_check="default_rng(7)"),
        fixed=dict(W=40, a=8, L=8, bands=list(V.BN), n_cal_windows=V.N_CAL_WIN),
        fold_sizes={int(f): dict(n=int((fold == f).sum()), n_AD=int(((fold == f) & (y == 1)).sum()),
                                 n_CN=int(((fold == f) & (y == 0)).sum())) for f in range(1, 6)},
        sha256={f: hashlib.sha256((V.V2 / f).read_bytes()).hexdigest() for f in files})
    LOCK.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(lock, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
