"""
models/moe.py — M6 (full): Mixture-of-Experts surrogate.

Assembly (REBUILD_SPEC M6):
  * experts  f_k = GPExpert (mean = Scheffe, kernel = Matern 5/2 ARD);
  * gating   g_k(x) from the M4 GMM regimes, but evaluated in RECIPE space
    through a learned classifier (grab #6: pre-image via model, not proximity);
  * prediction:   ŷ(x) = Σ_k g_k(x) μ_k(x);
  * uncertainty:  Var[ŷ] = Σ_k g_k σ_k²   (within-expert uncertainty)
                         + Σ_k g_k (μ_k − ŷ)²   (between-expert disagreement).

Anchor handling (grab #7): if a regime has too few points to fit its expert,
it is augmented with the full data so the expert still knows the boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from sklearn.linear_model import LogisticRegression

from ..core.linalg import n_scheffe_params
from .clustering import GMMRegimes
from .gp_expert import GPExpert


@dataclass
class MoEPrediction:
    mean: np.ndarray
    std: np.ndarray
    uncertainty: np.ndarray            # Σ g_k σ_k²   (within-expert)
    disagreement: np.ndarray          # Σ g_k (μ_k-ŷ)²  (between-expert)
    gating: np.ndarray                # (n, K)
    expert_means: np.ndarray          # (n, K)


class MixtureOfExperts:
    """K-expert MoE with GMM gating and GP experts."""

    def __init__(self, mean_model: Union[str, int] = "quadratic",
                 kernel: str = "matern52", k_range: Sequence[int] = range(1, 5),
                 n_restarts: int = 10, seed: Optional[int] = None,
                 names: Optional[Sequence[str]] = None):
        self.mean_model = mean_model
        self.kernel = kernel
        self.k_range = k_range
        self.n_restarts = n_restarts
        self.seed = seed
        self.names = names
        self.regimes_ = None
        self.gating_ = None
        self.experts_: List[GPExpert] = []
        self.K_: int = 0
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "MixtureOfExperts":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, q = X.shape

        # --- M4: regimes in property space ---
        self.regimes_ = GMMRegimes(k_range=self.k_range, n_init=10,
                                   seed=self.seed).fit(y)
        K = self.regimes_.n_regimes
        labels = self.regimes_.labels
        self.K_ = K

        # --- gating g_k(x): classifier in recipe space ---
        if K == 1:
            self.gating_ = None                      # trivial gating -> ones
        else:
            self.gating_ = LogisticRegression(max_iter=1000, C=10.0)
            self.gating_.fit(X, labels)

            self._gating_classes = self.gating_.classes_

        # --- experts: one GP per regime ---
        min_train = n_scheffe_params(q, self.mean_model) + 3
        self.experts_ = []
        for k in range(K):
            idx = np.where(labels == k)[0]
            if len(idx) < min_train:                 # anchor with full data (#7)
                Xk, yk = X, y
            else:
                Xk, yk = X[idx], y[idx]
            expert = GPExpert(mean_model=self.mean_model, kernel=self.kernel,
                              n_restarts=self.n_restarts, seed=self.seed,
                              names=self.names).fit(Xk, yk)
            self.experts_.append(expert)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def gating_proba(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.K_ == 1 or self.gating_ is None:
            return np.ones((X.shape[0], 1))
        proba = np.zeros((X.shape[0], self.K_))
        proba[:, self._gating_classes] = self.gating_.predict_proba(X)
        return proba

    # ------------------------------------------------------------------
    def predict(self, X) -> MoEPrediction:
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        nt = X.shape[0]
        G = self.gating_proba(X)                     # (nt, K)

        mu = np.zeros((nt, self.K_))
        s2 = np.zeros((nt, self.K_))
        for k, expert in enumerate(self.experts_):
            pred = expert.predict(X, return_std=True)
            mu[:, k] = pred.mean
            s2[:, k] = pred.std ** 2

        yhat = np.sum(G * mu, axis=1)
        uncertainty = np.sum(G * s2, axis=1)
        disagreement = np.sum(G * (mu - yhat[:, None]) ** 2, axis=1)
        var = uncertainty + disagreement
        return MoEPrediction(mean=yhat, std=np.sqrt(np.clip(var, 0, None)),
                             uncertainty=uncertainty, disagreement=disagreement,
                             gating=G, expert_means=mu)

    # ------------------------------------------------------------------
    @property
    def n_regimes(self) -> int:
        return self.K_

    def to_state(self) -> dict:
        self._check_fitted()
        d = {
            "mean_model": self.mean_model, "kernel": self.kernel,
            "K": self.K_, "names": self.names,
            "regimes": self.regimes_.to_state(),
            "experts": [e.to_state() for e in self.experts_],
        }
        if self.gating_ is not None:
            d["gating"] = {
                "coef": np.asarray(self.gating_.coef_),
                "intercept": np.asarray(self.gating_.intercept_),
                "classes": np.asarray(self._gating_classes),
            }
        return d

    @classmethod
    def from_state(cls, d: dict) -> "MixtureOfExperts":
        obj = cls(mean_model=d["mean_model"], kernel=d["kernel"],
                  names=d.get("names"))
        obj.K_ = d["K"]
        obj.experts_ = [GPExpert.from_state(s) for s in d["experts"]]
        if "gating" in d:
            clf = LogisticRegression()

            clf.coef_ = np.asarray(d["gating"]["coef"])
            clf.intercept_ = np.asarray(d["gating"]["intercept"])
            clf.classes_ = np.asarray(d["gating"]["classes"])
            obj.gating_ = clf
            obj._gating_classes = clf.classes_
        else:
            obj.gating_ = None
        obj._fitted = True
        return obj

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("MixtureOfExperts is not fitted; call .fit() first.")
