"""Ortak yardimcilar: veri yukleme, 5x5 bolmeler, nicemleme, katilimci toplama,
kat-disi (OOF) tahmin kaydi ve istatistik (bootstrap GA, DeLong, duzeltilmis t).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import config as C

NB, NCH = len(C.BAND_NAMES), 19
W, A, PA = C.W, C.ALPHABET, C.POWER_ALPHABET


# ------------------------------------------------------------------ veri
def subjects():
    return pd.read_csv(C.CACHE / "subjects.csv")


def load_paa(df=None):
    """-> df, paa (n,19,5,W), lp (n,19,5), sid (n,), y (n,)"""
    df = subjects() if df is None else df
    paa, lp, sid, y = [], [], [], []
    for i, r in enumerate(df.itertuples()):
        d = np.load(C.CACHE / f"paa_{r.participant_id}.npz")
        paa.append(d["paa"]); lp.append(d["lp"])
        n = len(d["lp"]); sid.append(np.full(n, i)); y.append(np.full(n, r.label))
    return (df, np.concatenate(paa), np.concatenate(lp),
            np.concatenate(sid), np.concatenate(y))


# ------------------------------------------------------------------ bolmeler
def splits(labels):
    """5 tekrar x 5 kat, katilimci duzeyinde tabakali. -> (rep, fold, tr_s, te_s)"""
    rskf = RepeatedStratifiedKFold(n_splits=C.N_FOLDS, n_repeats=C.N_REPEATS,
                                   random_state=C.SEED)
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(labels)), labels)):
        yield k // C.N_FOLDS + 1, k % C.N_FOLDS + 1, tr, te


def inner_val(tr_s, labels, seed):
    """Egitim katilimcilarindan tabakali %20 dogrulama -> (ic_egitim, dogrulama)."""
    rng = np.random.default_rng(seed)
    va = np.concatenate([rng.choice(tr_s[labels[tr_s] == l],
                                    max(1, int(round(C.VAL_FRAC * (labels[tr_s] == l).sum()))),
                                    replace=False) for l in (0, 1)])
    return np.setdiff1d(tr_s, va), np.sort(va)


def cap(sid, subs, n, seed):
    """Her katilimcidan en fazla n segment (rastgele) -> sirali indeks."""
    idx = np.concatenate([np.random.default_rng(seed + int(s)).permutation(
        np.where(sid == s)[0])[:n] for s in subs])
    idx.sort()
    return idx


def fold_seed(rep, fold):
    return C.SEED + 100 * rep + fold


# ------------------------------------------------------------------ nicemleme
def quantize(paa, lp, mask):
    """Kesim noktalari YALNIZ mask (egitim) segmentlerinden. -> sym, pw, lph(cozulmus)"""
    sym = np.zeros(paa.shape, dtype=np.int8)
    pw = np.zeros(lp.shape, dtype=np.int8)
    lph = np.zeros(lp.shape, dtype=np.float32)
    for k in range(NB):
        v = paa[mask, :, k, :].ravel()
        b1 = np.percentile(v, np.linspace(0, 100, A + 1)[1:-1])
        sym[:, :, k, :] = np.searchsorted(b1, paa[:, :, k, :]).astype(np.int8)
        u = lp[mask, :, k].ravel()
        b2 = np.percentile(u, np.linspace(0, 100, PA + 1)[1:-1])
        s = np.searchsorted(b2, u)
        reps = np.array([u[s == j].mean() if np.any(s == j) else u.mean() for j in range(PA)])
        pw[:, :, k] = np.searchsorted(b2, lp[:, :, k]).astype(np.int8)
        lph[:, :, k] = reps[pw[:, :, k].astype(np.int64)]
    return sym, pw, lph


def rbp_feat(lp):
    """log bant gucu -> log goreli bant gucu (n, 95)."""
    p = 10.0 ** lp.astype(np.float64)
    return np.log10(p / p.sum(-1, keepdims=True) + 1e-9).reshape(len(lp), -1).astype(np.float32)


def sym_feat(sym, pw):
    """M2: bant basina unigram + bigram + guc sembolu histogramlari."""
    n = len(sym)
    f = []
    for k in range(NB):
        s = sym[:, :, k, :]
        f.append(np.stack([(s == v).mean((1, 2)) for v in range(A)], 1))
        bg = s[:, :, :-1].astype(np.int16) * A + s[:, :, 1:].astype(np.int16)
        f.append(np.stack([(bg == v).mean((1, 2)) for v in range(A * A)], 1))
    f.append(np.stack([(pw == v).mean(1) for v in range(PA)], 1).reshape(n, -1))
    return np.concatenate(f, 1).astype(np.float32)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return (a.mean(0) - b.mean(0)) / (sp + 1e-12)


# ------------------------------------------------------------------ OOF
def subj_agg(p, sid_te):
    """segment olasiliklari -> (katilimci_idx, ortalama olasilik)."""
    us = np.unique(sid_te)
    return us, np.array([p[sid_te == s].mean() for s in us])


def oof_rows(model, rep, fold, subs, probs):
    return [dict(model=model, rep=rep, fold=fold, sub=int(s), prob=float(p))
            for s, p in zip(subs, probs)]


# ------------------------------------------------------------------ istatistik
def boot_auc_ci(y, p, n=C.N_BOOT, seed=C.SEED):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, [2.5, 97.5])


def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x)
    t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def delong(y, p1, p2):
    """Eslestirilmis iki AUC icin DeLong testi -> (auc1, auc2, z, p)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(p)
        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[pos] - tx) / n)
        v01.append(1 - (tz[neg] - ty) / m)
    s10, s01 = np.cov(np.array(v10)), np.cov(np.array(v01))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / m + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) duzeltilmis tekrarli-CV t testi (kat skorlari)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    v = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(max(v, 1e-12))
    return t, 2 * stats.t.sf(abs(t), k - 1)


def summarize(oof, labels):
    """OOF tablosu (model, rep, fold, sub, prob) -> model basina ozet + ortalama olasiliklar.

    Birincil olcut: 5 tekrarin ortalamasi alinmis katilimci olasiligi uzerinden AUC
    ve katilimci-bazli bootstrap %95 GA. Ek: tekrar-bazli ve kat-bazli AUC dagilimi.
    """
    rows, avg = [], {}
    for m, g in oof.groupby("model", sort=False):
        pm = g.groupby("sub").prob.mean()
        s = pm.index.values; p = pm.values; t = labels[s]
        avg[m] = pd.Series(p, index=s)
        lo, hi = boot_auc_ci(t, p)
        rep_auc = [roc_auc_score(labels[gg["sub"].values], gg.prob.values)
                   for _, gg in g.groupby("rep")]
        fold_auc = [roc_auc_score(labels[gg["sub"].values], gg.prob.values)
                    for _, gg in g.groupby(["rep", "fold"])]
        rows.append(dict(model=m, auc=roc_auc_score(t, p), ci_lo=lo, ci_hi=hi,
                         auc_rep_mean=np.mean(rep_auc), auc_rep_sd=np.std(rep_auc, ddof=1),
                         auc_fold_mean=np.mean(fold_auc), auc_fold_sd=np.std(fold_auc, ddof=1),
                         acc=accuracy_score(t, p > .5),
                         f1=f1_score(t, p > .5, average="macro"),
                         sens=((p > .5) & (t == 1)).sum() / (t == 1).sum(),
                         spec=((p <= .5) & (t == 0)).sum() / (t == 0).sum()))
    return pd.DataFrame(rows), avg


def fold_aucs(oof, model, labels):
    g = oof[oof.model == model]
    return np.array([roc_auc_score(labels[gg["sub"].values], gg.prob.values)
                     for _, gg in g.groupby(["rep", "fold"])])


def fmt(r):
    return f"{r.auc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"


def write_md(path, title, blocks):
    path.write_text("\n\n".join([f"# {title}"] + blocks) + "\n", encoding="utf-8")
