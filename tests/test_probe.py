"""Offline tests for the P3 probe machinery on synthetic data."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from probe_monitor.probe import catch_rate_at_fpr, grouped_folds, nested_probe  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def _synthetic(n_problems=40, width=20, layers=3, signal_layer=2, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_problems), 2).astype(str)
    y = np.tile([0, 1], n_problems)
    X = rng.normal(size=(len(y), layers, width))
    X[:, 0, :] = 0.0  # layer 0 identical for every paper, like the template token
    X[:, signal_layer, 0] += 4.0 * y  # signal lives in one direction of one layer
    return X, y, g


def test_folds_never_split_a_problem():
    g = np.repeat(np.arange(30), 3).astype(str)
    folds = grouped_folds(g, 5, 0)
    for u in set(g.tolist()):
        assert len(set(folds[g == u].tolist())) == 1


def test_probe_finds_signal_layer_and_layer0_is_chance():
    X, y, g = _synthetic()
    res = nested_probe(X, y, g, k=4, k_inner=3, n_jobs=1)
    auc_by_layer = [roc_auc_score(y, res["layer_scores"][l]) for l in range(X.shape[1])]
    assert abs(auc_by_layer[0] - 0.5) < 0.05
    assert auc_by_layer[2] > 0.9
    assert all(c["layer"] == 2 for c in res["chosen"])
    assert roc_auc_score(y, res["headline"]) > 0.9


def test_catch_rate():
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.linspace(0, 1, 100), np.full(10, 2.0)])
    assert catch_rate_at_fpr(y, s) == 1.0
