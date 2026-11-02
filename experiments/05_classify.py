"""Adim 5 - AD/CN SINIFLANDIRMA, 5x5 tekrarli CV, esit arama butcesi (Tablo 5-7, Sekil 6).

Modeller (pilot 09_tuned.py + 08_diagnostic.py ile ayni):
  M1   goreli bant gucu (95)                 + SVM-RBF
  M1s  SEMBOLDEN COZULEN goreli bant gucu    + SVM-RBF   (temsil-model tanisi)
  M2   sembolik unigram/bigram/guc histog.   + dogrusal SVM
  M3   sembolik diziler                      + Transformer (Optuna TPE, 10 deneme)

Pilot -> v2 duzeltmeleri:
  Z2  5 tekrar x 5 kat; katilimci-bazli OOF olasiliklari kaydedilir (istatistik 11_stats.py)
  Z3  arama sirasinda kesim noktalari YALNIZ ic-egitim katilimcilarindan ogrenilir
  Z4  nihai Transformer'da erken durdurma yok: ic-egitimde (dogrulama ile) bulunan en iyi
      epok sayisi e* kadar, tum dis-egitim uzerinde sabit epokla egitilir
  Nihai SVM'ler tum dis-egitim katilimcilari uzerinde egitilir (pilotta yalniz ic-egitim).

Cikti: outputs/oof_classify.csv (model, rep, fold, sub, prob), outputs/m3_configs.json
Kesilirse kaldigi kattan devam eder.
"""
import os, gc, sys, json, time, warnings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import common as U

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

NB, NCH, W, A, PA = U.NB, U.NCH, U.W, U.A, U.PA
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_TRIALS = 10
SEG_SEARCH = 40
SEEDS_FINAL = 2
OOF = C.OUT / "oof_classify.csv"
CFG = C.OUT / "m3_configs.json"


class Net(nn.Module):
    def __init__(self, d, nhead, layers, ff, drop, pool, use_pf):
        super().__init__()
        self.use_pf, self.pool = use_pf, pool
        self.tok = nn.Embedding(A, d)
        self.pos = nn.Parameter(torch.zeros(1, W, d))
        self.ch = nn.Embedding(NCH, d)
        self.bd = nn.Embedding(NB, d)
        self.pwe = nn.Embedding(PA, d)
        enc = nn.TransformerEncoderLayer(d, nhead, ff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.attn = nn.Linear(d, 1)
        hid = d + (NCH * NB if use_pf else 0)
        self.head = nn.Sequential(nn.LayerNorm(hid), nn.Dropout(drop), nn.Linear(hid, 2))
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, sym, pw, pf):
        B = sym.shape[0]
        ci = torch.arange(NCH, device=sym.device).view(1, NCH, 1).expand(B, NCH, NB)
        bi = torch.arange(NB, device=sym.device).view(1, 1, NB).expand(B, NCH, NB)
        ctx = (self.ch(ci) + self.bd(bi) + self.pwe(pw)).reshape(B * NCH * NB, 1, -1)
        x = self.tok(sym).reshape(B * NCH * NB, W, -1) + self.pos + ctx
        h = self.enc(x).mean(1).reshape(B, NCH * NB, -1)
        h = (torch.softmax(self.attn(h), 1) * h).sum(1) if self.pool == "attn" else h.mean(1)
        if self.use_pf:
            h = torch.cat([h, pf], 1)
        return self.head(h)


def train_net(cfg, data, y, tr_idx, seed, epochs, va_idx=None, patience=None, te_idx=None):
    """va_idx verilirse erken durdurma (en iyi AUC, en iyi epok); yoksa sabit epok."""
    sym, pw, pf = data
    torch.manual_seed(seed); np.random.seed(seed)
    m = Net(cfg["d"], cfg["nhead"], cfg["layers"], cfg["ff"], cfg["drop"],
            cfg["pool"], cfg["use_pf"]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    w1 = (y[tr_idx] == 0).sum() / max((y[tr_idx] == 1).sum(), 1)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor([1.0, w1], dtype=torch.float32, device=DEV),
                                label_smoothing=cfg["ls"])
    scaler = torch.amp.GradScaler(DEV)

    def bat(idx, bs, sh):
        i = np.random.permutation(idx) if sh else idx
        for k in range(0, len(i), bs):
            j = i[k:k + bs]
            yield (torch.from_numpy(sym[j]).to(DEV).long(), torch.from_numpy(pw[j]).to(DEV).long(),
                   torch.from_numpy(pf[j]).to(DEV), torch.from_numpy(y[j]).to(DEV), j)

    @torch.no_grad()
    def pred(idx):
        m.eval(); o = np.zeros(len(idx))
        for a, b, f, _, j in bat(idx, 512, False):
            with torch.amp.autocast(DEV):
                o[np.searchsorted(idx, j)] = torch.softmax(m(a, b, f).float(), 1)[:, 1].cpu().numpy()
        return o

    best, best_ep, state, bad = -1.0, epochs, None, 0
    for ep in range(1, epochs + 1):
        m.train()
        for a, b, f, t, _ in bat(tr_idx, cfg["bs"], True):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEV):
                loss = lossf(m(a, b, f), t)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if va_idx is not None:
            v = roc_auc_score(y[va_idx], pred(va_idx))
            if v > best:
                best, best_ep, bad = v, ep, 0
                state = {k: x.detach().clone() for k, x in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if state is not None:
        m.load_state_dict(state)
    out = pred(te_idx) if te_idx is not None else None
    del m, opt, state, scaler
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return best, best_ep, out


def svm(F_tune, F_fin, kernel, grid, y, s_tr, va_i, f_tr, te_i):
    """Ic-egitim/dogrulama uzerinde grid; nihai model tum dis-egitimde."""
    best = (-1, None)
    sc = StandardScaler().fit(F_tune[s_tr])
    for g in grid:
        m = SVC(kernel=kernel, probability=True, class_weight="balanced",
                random_state=C.SEED, **g).fit(sc.transform(F_tune[s_tr]), y[s_tr])
        v = roc_auc_score(y[va_i], m.predict_proba(sc.transform(F_tune[va_i]))[:, 1])
        if v > best[0]:
            best = (v, g)
    sc = StandardScaler().fit(F_fin[f_tr])
    m = SVC(kernel=kernel, probability=True, class_weight="balanced",
            random_state=C.SEED, **best[1]).fit(sc.transform(F_fin[f_tr]), y[f_tr])
    return m.predict_proba(sc.transform(F_fin[te_i]))[:, 1], best[1]


def main():
    df, paa, lp, sid, y = U.load_paa()
    labels = df.label.values
    RF = U.rbp_feat(lp)
    print(f"segment {paa.shape} | katilimci {len(df)} | cihaz {DEV}", flush=True)
    rows = pd.read_csv(OOF).to_dict("records") if OOF.exists() else []
    cfgs = json.load(open(CFG)) if CFG.exists() else {}
    done = {(r["rep"], r["fold"]) for r in rows}
    G1 = [{"C": c, "gamma": g} for c in (1, 10, 100) for g in ("scale", 0.01, 0.001)]
    G2 = [{"C": c} for c in (0.01, 0.1, 1, 10)]

    for rep, fold, tr_s, te_s in U.splits(labels):
        if (rep, fold) in done:
            continue
        t0 = time.time(); seed = U.fold_seed(rep, fold)
        itr_s, va_s = U.inner_val(tr_s, labels, seed)
        s_tr = U.cap(sid, itr_s, SEG_SEARCH, seed)
        i_tr = U.cap(sid, itr_s, C.SEG_TRAIN, seed)
        f_tr = U.cap(sid, tr_s, C.SEG_TRAIN, seed)
        va_i = np.where(np.isin(sid, va_s))[0]
        te_i = np.where(np.isin(sid, te_s))[0]

        sym_t, pw_t, lph_t = U.quantize(paa, lp, np.isin(sid, itr_s))     # arama temsili (Z3)
        sym_f, pw_f, lph_f = U.quantize(paa, lp, np.isin(sid, tr_s))      # nihai temsil
        PF_t, PF_f = U.rbp_feat(lph_t), U.rbp_feat(lph_f)

        p1, c1 = svm(RF, RF, "rbf", G1, y, s_tr, va_i, f_tr, te_i)
        p1s, c1s = svm(PF_t, PF_f, "rbf", G1, y, s_tr, va_i, f_tr, te_i)
        p2, c2 = svm(U.sym_feat(sym_t, pw_t), U.sym_feat(sym_f, pw_f), "linear", G2,
                     y, s_tr, va_i, f_tr, te_i)

        dt = (sym_t, pw_t, PF_t)

        def objective(tr):
            cfg = dict(d=tr.suggest_categorical("d", [32, 48, 64]),
                       nhead=tr.suggest_categorical("nhead", [4, 8]),
                       layers=tr.suggest_int("layers", 1, 2),
                       ff=tr.suggest_categorical("ff", [64, 128]),
                       drop=tr.suggest_float("drop", .05, .4),
                       lr=tr.suggest_float("lr", 1e-4, 1e-3, log=True),
                       wd=tr.suggest_float("wd", 1e-4, 1e-1, log=True),
                       ls=tr.suggest_float("ls", .0, .15),
                       bs=tr.suggest_categorical("bs", [64, 96, 128]),
                       pool=tr.suggest_categorical("pool", ["mean", "attn"]),
                       use_pf=tr.suggest_categorical("use_pf", [True, False]))
            v, _, _ = train_net(cfg, dt, y, s_tr, seed, 8, va_i, 2)
            return v

        st = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
        st.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        cfg = st.best_params
        v_ic, e_star, _ = train_net(cfg, dt, y, i_tr, seed, 25, va_i, 5)     # e* (Z4)
        p3 = np.mean([train_net(cfg, (sym_f, pw_f, PF_f), y, f_tr, seed + 1000 * s, e_star,
                                te_idx=te_i)[2] for s in range(SEEDS_FINAL)], 0)
        cfgs[f"r{rep}f{fold}"] = dict(cfg, inner_auc=st.best_value, refit_val_auc=v_ic,
                                      e_star=e_star, M1=c1, M1s=c1s, M2=c2)

        for nm, p in [("M1", p1), ("M1s", p1s), ("M2", p2), ("M3", p3)]:
            us, ps = U.subj_agg(p, sid[te_i])
            rows += U.oof_rows(nm, rep, fold, us, ps)
        pd.DataFrame(rows).to_csv(OOF, index=False)
        json.dump(cfgs, open(CFG, "w"), indent=1)
        cur = pd.DataFrame(rows)
        cur = cur[(cur.rep == rep) & (cur.fold == fold)]
        aucs = {nm: roc_auc_score(labels[g["sub"].values], g.prob.values)
                for nm, g in cur.groupby("model", sort=False)}
        print(f"r{rep}f{fold} ({time.time() - t0:.0f}s) " +
              " ".join(f"{k}={v:.3f}" for k, v in aucs.items()) +
              f" | M3 e*={e_star} pf={cfg['use_pf']} ic={st.best_value:.3f}", flush=True)


if __name__ == "__main__":
    main()
