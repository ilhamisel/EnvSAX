"""AHEPA nicemleyicisini 02_fidelity.py ile AYNI rastgelelik sırasıyla yeniden kurar ve dondurur.

Yalnızca AHEPA (cache_v2) kullanılır. Kaynak sonuçların yeniden üretildiği kontrol edilir:
  r_d (65 katılımcı) = 0.97809, r_d (nicemleyici eğitiminde kullanılmayan 26 katılımcı) = 0.98679.
Ardından AHEPA referans değerleri (bootstrap GA'larıyla) dış doğrulamayla aynı kodla hesaplanır.

Çıktılar:
  03_PROTOCOL/frozen_encoder_ahepa.npz   kesim noktaları ve temsil değerleri
  03_PROTOCOL/FROZEN_ENCODER_INFO.json   eğitim katılımcıları, parametreler, sha256
  04_EXPERIMENTS/source_reference/       AHEPA referans profilleri ve ölçütleri
"""
import hashlib, json, time
import numpy as np
import pandas as pd
from evlib import C, S, U, BN, AHEPA_CH, ROOT, FrozenScale, rbp, profiles, fidelity, boot_fidelity, channel_shuffle_null

PROT = ROOT / "03_PROTOCOL"
OUT = ROOT / "04_EXPERIMENTS" / "source_reference"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ch = list(np.load(C.CACHE / "ch_names.npy"))
    assert ch == AHEPA_CH, ch
    df = U.subjects()
    rng = np.random.default_rng(C.SEED)
    fit = []
    for lab in (0, 1):
        s = df[df.label == lab].participant_id.values
        fit += list(rng.choice(s, size=int(round(0.6 * len(s))), replace=False))
    assert set(fit) == set(df[df.fit_split].participant_id), "fit_split yeniden üretilemedi"

    zs = {b: [] for b in BN}; lps = {b: [] for b in BN}
    for r in df[df.fit_split].itertuples():
        z = np.load(C.CACHE / f"zenv_{r.participant_id}.npy", mmap_mode="r")
        lp = np.load(C.CACHE / f"paa_{r.participant_id}.npz")["lp"]
        i = np.sort(rng.choice(len(lp), size=min(25, len(lp)), replace=False))
        for k, b in enumerate(BN):
            zs[b].append(np.asarray(z[i, :, k, :], dtype=np.float32).reshape(-1, C.SEG_LEN))
            lps[b].append(lp[i, :, k].ravel())
    sax = {b: S.SAX(C.W, C.ALPHABET).fit(np.concatenate(zs[b])) for b in BN}
    pq = S.PowerQuantizer(C.POWER_ALPHABET).fit({b: np.concatenate(v) for b, v in lps.items()})
    src_lp = {b: np.concatenate(v) for b, v in lps.items()}

    path = PROT / "frozen_encoder_ahepa.npz"
    np.savez(path,
             scale_breaks=np.stack([pq.breaks[b] for b in BN]), scale_reps=np.stack([pq.reps[b] for b in BN]),
             shape_breaks=np.stack([sax[b].breaks for b in BN]), shape_reps=np.stack([sax[b].reps for b in BN]),
             band_order=np.array(BN), channel_order=np.array(AHEPA_CH),
             fit_lp=np.stack([src_lp[b] for b in BN]))          # E5 için kaynak log-güç dağılımı
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    fz = FrozenScale(path)

    # --- kaynak profilleri (özgün / dondurulmuş ölçek kodundan çözülmüş) -----------
    so, ss, codes = [], [], []
    for r in df.itertuples():
        lp = np.load(C.CACHE / f"paa_{r.participant_id}.npz")["lp"].astype(np.float64)
        code = fz.encode(lp); lph = fz.decode(code)
        so.append(rbp(lp).mean(0).ravel()); ss.append(rbp(lph).mean(0).ravel()); codes.append(code)
    so, ss, y = np.array(so), np.array(ss), df.label.values
    ev = ~df.fit_split.values
    allf = fidelity(*profiles(so, ss, y))
    held = fidelity(*profiles(so[ev], ss[ev], y[ev]))
    assert abs(allf["r_d"] - 0.978090265932632) < 1e-9, allf["r_d"]
    assert abs(held["r_d"] - 0.9867880420286874) < 1e-9, held["r_d"]

    res = {}
    for name, m, f in (("ahepa_all65", np.ones_like(ev), allf), ("ahepa_heldout26", ev, held)):
        b = boot_fidelity(so[m], ss[m], y[m])
        do, ds = profiles(so[m], ss[m], y[m])
        null = channel_shuffle_null(do, ds)
        t = pd.DataFrame({"point": pd.Series(f), "ci_lo": b.quantile(.025), "ci_hi": b.quantile(.975)})
        t.loc["n_AD"] = [int((y[m] == 1).sum())] * 3; t.loc["n_CN"] = [int((y[m] == 0).sum())] * 3
        t.loc["null_shuffle_mean"] = [null.mean()] * 3; t.loc["null_shuffle_q95"] = [np.quantile(null, .95)] * 3
        t.to_csv(OUT / f"{name}_fidelity.csv")
        np.savez(OUT / f"{name}_profiles.npz", d_o=do, d_s=ds, so=so[m], ss=ss[m], y=y[m])
        res[name] = t
    np.savez(OUT / "ahepa_scale_codes.npz", codes=np.concatenate(codes),
             sid=np.concatenate([np.full(len(c), i) for i, c in enumerate(codes)]), y=y, heldout=ev)

    info = dict(
        created=time.strftime("%Y-%m-%d %H:%M:%S"), source="AHEPA / OpenNeuro ds004504 derivatives (cache_v2)",
        encoder_file="03_PROTOCOL/frozen_encoder_ahepa.npz", sha256=sha,
        reconstruction="02_fidelity.py rastgelelik sırası (seed 42): fit_split %60 tabakalı, katılımcı başına 25 segment",
        fit_participants=sorted(fit), n_fit=len(fit),
        n_fit_AD=int(df[df.fit_split].label.sum()), n_fit_CN=int((~df[df.fit_split].label.astype(bool)).sum()),
        heldout_participants=sorted(df[ev].participant_id), n_heldout=int(ev.sum()),
        config=dict(sfreq=C.SFREQ, seg_sec=C.SEG_SEC, overlap=C.OVERLAP, W=C.W, a=C.ALPHABET, L=C.POWER_ALPHABET,
                    bands=C.BANDS, filter="Butterworth 4. derece, sosfiltfilt, sürekli kayıtta",
                    envelope_smoothing_samples=25, channel_order=AHEPA_CH),
        units=dict(voltage="µV (MNE volt * 1e6)", power="µV² (pencere içi ortalama kare)",
                   log="log10(güç + 1e-12)", relative_power="doğrusal p_b / Σ p (5 bant)",
                   scale_code="bant başına log10 güç üzerinde ampirik kuantil kesim noktaları"),
        scale_breaks={b: [float(v) for v in pq.breaks[b]] for b in BN},
        scale_reps={b: [float(v) for v in pq.reps[b]] for b in BN},
        shape_breaks={b: [float(v) for v in sax[b].breaks] for b in BN},
        reproduced=dict(r_d_all65=allf["r_d"], r_d_heldout26=held["r_d"]))
    (PROT / "FROZEN_ENCODER_INFO.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, t in res.items():
        print("==", k); print(t.round(4).to_string())


if __name__ == "__main__":
    main()
