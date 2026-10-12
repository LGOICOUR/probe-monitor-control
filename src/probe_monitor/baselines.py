"""TA-free baselines: the length ruler and the word counter (TF-IDF over code
tokens + logistic regression), scored with the same grouped folds as the probe."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .probe import grouped_folds


def word_counter_scores(texts: list[str], y: np.ndarray, g: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    folds = grouped_folds(g, k, seed)
    s = np.zeros(len(y))
    for f in range(k):
        tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
        vec = TfidfVectorizer(token_pattern=r"[A-Za-z_]+|\S", ngram_range=(1, 2), min_df=2)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, y[tr])
        s[te] = clf.decision_function(vec.transform([texts[i] for i in te]))
    return s
