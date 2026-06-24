"""
models/screening.py — M3 (extension): component screening -> q_eff.

Two complementary screening signals (REBUILD_SPEC M3):
  * ARD-GP lengthscales  -> component importance (the "free screening" of a
    Matern-5/2 GP with one lengthscale per component).
  * Scheffe OLS term significance (p-values) -> term-level importance.

The ARD-GP is fitted with scikit-learn (proven library, decision #1):
    kernel = ConstantKernel * Matern(nu=2.5, anisotropic) + WhiteKernel
A large ARD lengthscale ℓ_i means component i barely changes the response =>
it is a candidate to drop, reducing q -> q_eff (≈ 4..6).

Note (REBUILD_SPEC grab #8 / kernel §5): components live on the simplex
(Σx=1), so one direction is redundant; the *relative* lengthscales across
components remain interpretable for screening.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, WhiteKernel,
)

from .scheffe import ScheffeModel


@dataclass
class ScreeningResult:
    """Outcome of component screening."""
    component_names: List[str]
    lengthscales: np.ndarray            # ARD lengthscale per component
    importance: np.ndarray             # normalised (max=1) importance per component
    active: np.ndarray                 # bool mask of "kept" components
    q_eff: int
    gp_loglik: float
    noise_level: float
    table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def active_indices(self) -> List[int]:
        return [i for i, a in enumerate(self.active) if a]

    def to_state(self) -> dict:
        return {
            "component_names": self.component_names,
            "lengthscales": np.asarray(self.lengthscales),
            "importance": np.asarray(self.importance),
            "active": np.asarray(self.active),
            "q_eff": int(self.q_eff),
            "gp_loglik": float(self.gp_loglik),
            "noise_level": float(self.noise_level),
        }


class ARDScreening:
    """Matern-5/2 ARD Gaussian-process screening of mixture components."""

    def __init__(self, length_scale_bounds=(1e-2, 1e3),
                 noise_bounds=(1e-6, 1e1), n_restarts: int = 12,

                 rel_threshold: float = 0.15, seed: Optional[int] = None):
        self.length_scale_bounds = length_scale_bounds
        self.noise_bounds = noise_bounds
        self.n_restarts = n_restarts
        self.rel_threshold = rel_threshold       # active if importance >= this
        self.seed = seed
        self.gp_: Optional[GaussianProcessRegressor] = None

    # ------------------------------------------------------------------
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float],
            names: Optional[Sequence[str]] = None) -> ScreeningResult:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, q = X.shape
        names = list(names) if names is not None else [chr(ord("A") + i) for i in range(q)]

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(q), nu=2.5,
                     length_scale_bounds=self.length_scale_bounds)
            + WhiteKernel(noise_level=1.0, noise_level_bounds=self.noise_bounds)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.seed,
        )
        gp.fit(X, y)
        self.gp_ = gp

        # Extract ARD lengthscales + noise from the fitted kernel
        matern = gp.kernel_.k1.k2          # ConstantKernel * Matern -> k1; Matern -> k2
        white = gp.kernel_.k2
        lengthscales = np.atleast_1d(matern.length_scale).astype(float)
        if lengthscales.size == 1:         # isotropic fallback
            lengthscales = np.full(q, float(lengthscales))
        noise_level = float(white.noise_level)

        # Importance = inverse lengthscale, normalised so max = 1
        inv = 1.0 / np.maximum(lengthscales, 1e-12)
        importance = inv / inv.max()
        active = importance >= self.rel_threshold
        if not active.any():               # never drop everything
            active[int(np.argmax(importance))] = True
        q_eff = int(active.sum())

        order = np.argsort(-importance)
        table = pd.DataFrame({
            "component": names,
            "lengthscale": lengthscales,
            "importance": importance,
            "active": active,
        }).iloc[order].reset_index(drop=True)

        return ScreeningResult(
            component_names=names, lengthscales=lengthscales,
            importance=importance, active=active, q_eff=q_eff,
            gp_loglik=float(gp.log_marginal_likelihood_value_),
            noise_level=noise_level, table=table,
        )


# ----------------------------------------------------------------------
# Convenience: OLS term significance (from a fitted Scheffe model)
# ----------------------------------------------------------------------

def significant_terms(model: ScheffeModel, alpha: float = 0.05) -> pd.DataFrame:
    """Return Scheffe terms with p-value < alpha (sorted by p-value)."""
    tab = model.coefficient_table()
    sig = tab[tab["p_value"] < alpha].sort_values("p_value").reset_index(drop=True)
    return sig


def screen_components(X, y, names: Optional[Sequence[str]] = None,
                      rel_threshold: float = 0.15, n_restarts: int = 12,
                      seed: Optional[int] = None) -> ScreeningResult:
    """One-shot convenience wrapper around :class:`ARDScreening`."""
    return ARDScreening(rel_threshold=rel_threshold, n_restarts=n_restarts,
                        seed=seed).fit(X, y, names=names)
