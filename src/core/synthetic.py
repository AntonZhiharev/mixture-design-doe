"""
core/synthetic.py — controlled synthetic polygon for verification.

Defines a KNOWN Scheffe mixture function so the pipeline can be measured on
ground truth (REBUILD_SPEC §8: "возьми известную синтетическую функцию").

    y(x) = Σ_t beta_t * term_t(x)  +  N(0, sigma^2)

where the terms are the Scheffe terms of the chosen model order.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from .linalg import scheffe_matrix, scheffe_term_indices, scheffe_term_names


class SyntheticScheffe:
    """Known Scheffe response surface with optional Gaussian noise."""

    def __init__(self, q: int, model: Union[str, int] = "quadratic",
                 coefficients: Sequence[float] | None = None,
                 noise_sd: float = 0.0, seed: int | None = None,
                 names: Sequence[str] | None = None):
        self.q = int(q)
        self.model = model
        self.noise_sd = float(noise_sd)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.term_indices = scheffe_term_indices(q, model)
        self.term_names = scheffe_term_names(q, model, names)
        p = len(self.term_indices)

        if coefficients is None:
            # Reproducible "interesting" coefficients: linear terms decreasing,
            # interaction terms alternating sign.
            cgen = np.random.default_rng(0 if seed is None else seed)
            self.coefficients = np.empty(p)
            for t, idx in enumerate(self.term_indices):
                order = len(idx)
                if order == 1:
                    self.coefficients[t] = 10.0 - 1.5 * idx[0]      # 10,8.5,7,...
                else:
                    sign = -1.0 if (t % 2) else 1.0
                    self.coefficients[t] = sign * cgen.uniform(2.0, 6.0)
        else:
            self.coefficients = np.asarray(coefficients, dtype=float)
            if len(self.coefficients) != p:
                raise ValueError(
                    f"Expected {p} coefficients for model '{model}' (q={q}), "
                    f"got {len(self.coefficients)}.")

    # ------------------------------------------------------------------
    def true(self, X: Sequence[float]) -> np.ndarray:
        """Noiseless response."""
        M = scheffe_matrix(X, self.model)
        y = M @ self.coefficients
        return y

    def evaluate(self, X: Sequence[float]) -> np.ndarray:
        """Noisy response (adds N(0, noise_sd^2))."""
        y = self.true(X)
        if self.noise_sd > 0:
            y = y + self._rng.normal(0.0, self.noise_sd, size=y.shape)
        return y

    # Convenience: callable
    def __call__(self, X: Sequence[float]) -> np.ndarray:
        return self.evaluate(X)

    def coefficient_table(self):
        import pandas as pd
        return pd.DataFrame({"term": self.term_names,
                             "true_coef": self.coefficients})

    def __repr__(self) -> str:
        return (f"SyntheticScheffe(q={self.q}, model='{self.model}', "
                f"p={len(self.coefficients)}, noise_sd={self.noise_sd})")


def default_polygon(noise_sd: float = 0.5, seed: int = 42) -> SyntheticScheffe:
    """Default Iteration-1 test polygon: q=5, Scheffe quadratic, fixed coefs.

    True model (15 terms):
        10A + 8.5B + 7C + 5.5D + 4E
        + (signed) two-way interactions.
    """
    return SyntheticScheffe(q=5, model="quadratic", noise_sd=noise_sd, seed=seed)


class MultiSyntheticScheffe:
    """Multi-response synthetic lab: P independent Scheffe properties.

    Демонстрационная «лаборатория» для мультиотклика (REBUILD_SPEC §12): каждое
    свойство — самостоятельная :class:`SyntheticScheffe` с собственным seed, так
    что свойства различаются (разные «истины»). Меряет сразу все P свойств.

    Methods mirror :class:`SyntheticScheffe` but return arrays shaped (n, P)::

        true(X)     -> noiseless responses,  shape (n, P)
        evaluate(X) -> noisy responses,      shape (n, P)
    """

    def __init__(self, q: int, property_names: Sequence[str],
                 model: Union[str, int] = "quadratic",
                 noise_sd: float = 0.0, seed: int | None = 0,
                 names: Sequence[str] | None = None):
        self.q = int(q)
        self.model = model
        self.noise_sd = float(noise_sd)
        self.property_names = list(property_names)
        if len(self.property_names) == 0:
            raise ValueError("Need at least one property name.")
        base = 0 if seed is None else int(seed)
        # отдельная «истина» на каждое свойство (разные seed → разные функции).
        # Свойство 0 использует базовый seed — совместимо с одно-откликовым
        # SyntheticScheffe(seed=base).
        self.truths = [
            SyntheticScheffe(q, model=model, noise_sd=noise_sd,
                             seed=base + 1009 * i, names=names)
            for i in range(len(self.property_names))
        ]


    @property
    def n_properties(self) -> int:
        return len(self.property_names)

    def true(self, X: Sequence[float]) -> np.ndarray:
        """Noiseless responses, shape (n, P)."""
        cols = [t.true(X) for t in self.truths]
        return np.column_stack(cols)

    def evaluate(self, X: Sequence[float]) -> np.ndarray:
        """Noisy responses, shape (n, P)."""
        cols = [t.evaluate(X) for t in self.truths]
        return np.column_stack(cols)

    def __call__(self, X: Sequence[float]) -> np.ndarray:
        return self.evaluate(X)

    def __repr__(self) -> str:
        return (f"MultiSyntheticScheffe(q={self.q}, P={self.n_properties}, "
                f"props={self.property_names}, noise_sd={self.noise_sd})")


