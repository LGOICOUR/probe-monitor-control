"""P3: fit the decoder (a regularized logistic-regression probe) on the TA's
recorded activity and score it only on held-out problems.

Design: five rounds of grouped cross-validation (all papers of a problem on
the same side); per-pairing probes plus a transfer test; headline layer chosen
inside the training rounds only; regularization strength tuned inside the
training rounds. Lever "layer picked
in hindsight" reuses the same fits.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def grouped_folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Assign each paper a fold index so that every paper of a problem shares one fold."""
    uniq = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def _fit_score(Xtr, ytr, Xte, C):
    if len(set(ytr.tolist())) < 2:
        return np.full(len(Xte), 0.5)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def _safe_auc(y, s):
    return roc_auc_score(y, s) if len(set(np.asarray(y).tolist())) == 2 else np.nan


def _inner_grid(X, y, g, layer, k_inner, seed):
    """Mean inner-CV AUROC for each C at one layer, using training papers only."""
    folds = grouped_folds(g, k_inner, seed + 1)
    out = []
    for C in C_GRID:
        s = np.zeros(len(y))
        for f in range(k_inner):
            tr, te = folds != f, folds == f
            s[te] = _fit_score(X[tr, layer], y[tr], X[te, layer], C)
        out.append(_safe_auc(y, s))
    return out


def nested_probe(
    X: np.ndarray, y: np.ndarray, g: np.ndarray, *, k: int = 5, k_inner: int = 4, seed: int = 0,
    X_transfer: np.ndarray | None = None, y_transfer: np.ndarray | None = None, g_transfer: np.ndarray | None = None,
    n_jobs: int = -1,
) -> dict:
    """Nested grouped CV over layers and C.

    Returns held-out scores for every paper at every layer (with C tuned inside
    the training rounds for that layer), the headline scores at the layer chosen
    inside the training rounds, and, if a transfer set is given, the same probes'
    scores on transfer papers from the held-out problems only.
    """
    n, L, _ = X.shape
    folds = grouped_folds(g, k, seed)
    layer_scores = np.zeros((L, n))
    headline = np.zeros(n)
    chosen = []
    if X_transfer is not None:
        uniq = sorted(set(g.tolist()))
        fold_of = {u: folds[np.where(g == u)[0][0]] for u in uniq}
        t_folds = np.array([fold_of.get(u, -1) for u in g_transfer])
        t_layer_scores = np.zeros((L, len(y_transfer)))
        t_headline = np.zeros(len(y_transfer))
    for f in range(k):
        tr, te = folds != f, folds == f
        grids = Parallel(n_jobs=n_jobs)(
            delayed(_inner_grid)(X[tr], y[tr], g[tr], layer, k_inner, seed) for layer in range(L)
        )
        grids = np.array(grids)  # [L, len(C_GRID)]
        best_c = [C_GRID[int(np.nanargmax(row))] if not np.all(np.isnan(row)) else C_GRID[0] for row in grids]
        best_layer = int(np.nanargmax(np.nanmax(np.where(np.isnan(grids), -1, grids), axis=1)))
        chosen.append({"fold": f, "layer": best_layer, "C": best_c[best_layer],
                       "inner_auroc": float(np.nanmax(grids[best_layer]))})
        te_idx = np.where(te)[0]
        if X_transfer is not None:
            t_idx = np.where(t_folds == f)[0]
        for layer in range(L):
            Xte = X[te, layer]
            if X_transfer is not None and len(t_idx):
                Xte = np.concatenate([Xte, X_transfer[t_idx, layer]])
            s = _fit_score(X[tr, layer], y[tr], Xte, best_c[layer])
            layer_scores[layer, te_idx] = s[: len(te_idx)]
            if X_transfer is not None and len(t_idx):
                t_layer_scores[layer, t_idx] = s[len(te_idx):]
        headline[te_idx] = layer_scores[best_layer, te_idx]
        if X_transfer is not None and len(t_idx):
            t_headline[t_idx] = t_layer_scores[best_layer, t_idx]
    res = {"folds": folds, "layer_scores": layer_scores, "headline": headline, "chosen": chosen}
    if X_transfer is not None:
        res.update({"t_folds": t_folds, "t_layer_scores": t_layer_scores, "t_headline": t_headline})
    return res


def catch_rate_at_fpr(y: np.ndarray, s: np.ndarray, fpr: float = 0.02) -> float:
    """Fraction of positives scored strictly above the (1 - fpr) quantile of negatives."""
    thr = np.quantile(s[y == 0], 1 - fpr)
    return float(np.mean(s[y == 1] > thr))


def grouped_bootstrap_ci(y, s, g, metric, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI from resampling problems (not papers) with replacement."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(g.tolist())))
    idx_of = {u: np.where(g == u)[0] for u in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[u] for u in pick])
        if len(set(y[idx].tolist())) == 2:
            vals.append(metric(y[idx], s[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def fit_probe(X: np.ndarray, y: np.ndarray, g: np.ndarray, layer: int, k_inner: int = 4, seed: int = 0):
    """Fit one probe at a given layer on ALL given papers, choosing C by grouped
    inner CV. Returns (scaler, classifier, chosen_C). Use for scoring papers from
    problems that are not in X."""
    grid = _inner_grid(X, y, g, layer, k_inner, seed)
    C = C_GRID[int(np.nanargmax(grid))]
    sc = StandardScaler().fit(X[:, layer])
    clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[:, layer]), y)
    return sc, clf, C


def probe_scores(sc, clf, X: np.ndarray, layer: int) -> np.ndarray:
    return clf.decision_function(sc.transform(X[:, layer]))
