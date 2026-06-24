"""
models/clustering.py — M4: regime clustering in PROPERTY space.

Fits a Gaussian Mixture Model to the (standardised) response/property values y
and selects the number of regimes K by BIC (REBUILD_SPEC M4).  Returns hard
labels and soft responsibilities r_k used later as MoE gating targets.

Critical (grab #6): clustering is done in PROPERTY space; the pre-image in
recipe space is recovered later *through a model* (the gating classifier in
moe.py), never by point proximity.

R reference: ``mclust::Mclust`` (BIC model selection).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class RegimeResult:
    n_regimes: int
    labels: np.ndarray                 # hard assignment (n,)
    responsibilities: np.ndarray      # soft assignment (n, K)
    means: np.ndarray                 # regime means in ORIGINAL property units (K, m)
    bic_table: pd.DataFrame
    covariance_type: str = "full"

    def to_state(self) -> dict:
        return {
            "n_regimes": int(self.n_regimes),
            "labels": np.asarray(self.labels),
            "responsibilities": np.asarray(self.responsibilities),
            "means": np.asarray(self.means),
            "covariance_type": self.covariance_type,
        }


class GMMRegimes:
    """GMM regime finder with BIC-based selection of K."""

    def __init__(self, k_range: Sequence[int] = range(1, 6),
                 covariance_type: str = "full", n_init: int = 10,
                 seed: Optional[int] = None):
        self.k_range = list(k_range)
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.seed = seed
        self.scaler_: Optional[StandardScaler] = None
        self.gmm_: Optional[GaussianMixture] = None

    # ------------------------------------------------------------------
    def fit(self, Y: Sequence[float]) -> RegimeResult:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        n = Y.shape[0]
        self.scaler_ = StandardScaler().fit(Y)
        Z = self.scaler_.transform(Y)

        rows, best = [], None
        for K in self.k_range:
            if K > n:                  # cannot have more components than samples
                continue
            gmm = GaussianMixture(n_components=K,
                                  covariance_type=self.covariance_type,
                                  n_init=self.n_init, random_state=self.seed)
            gmm.fit(Z)
            bic = gmm.bic(Z)
            rows.append({"K": K, "bic": bic,
                         "loglik": gmm.score(Z) * n})
            if best is None or bic < best[1]:
                best = (gmm, bic, K)

        self.gmm_ = best[0]
        K = best[2]
        labels = self.gmm_.predict(Z)
        resp = self.gmm_.predict_proba(Z)
        means = self.scaler_.inverse_transform(self.gmm_.means_)
        bic_table = pd.DataFrame(rows)
        return RegimeResult(n_regimes=K, labels=labels, responsibilities=resp,
                            means=means, bic_table=bic_table,
                            covariance_type=self.covariance_type)

    # ------------------------------------------------------------------
    def predict_proba(self, Y: Sequence[float]) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        Z = self.scaler_.transform(Y)
        return self.gmm_.predict_proba(Z)
