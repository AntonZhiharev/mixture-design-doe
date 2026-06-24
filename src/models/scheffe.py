"""
models/scheffe.py — M3: Scheffe OLS fit + ANOVA + term significance.

Fits a Scheffe mixture polynomial (explicit model order) by ordinary least
squares and reports coefficients, R2/Adj-R2/RMSE, an ANOVA table and per-term
t-statistics / p-values.

R reference: ``mixexp::MixModel`` + ``lm`` / ``anova``.
Tolerance for golden tests: coefficients atol = 1e-8 (deterministic OLS).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.linalg import scheffe_matrix, scheffe_term_names, n_scheffe_params


class ScheffeModel:
    """Ordinary-least-squares Scheffe mixture model."""

    def __init__(self, model: Union[str, int] = "quadratic",
                 names: Optional[Sequence[str]] = None):
        self.model = model
        self.names = list(names) if names is not None else None
        self.q: Optional[int] = None
        self.term_names: list[str] = []
        self.coefficients: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> "ScheffeModel":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self.q = X.shape[1]
        self.term_names = scheffe_term_names(self.q, self.model, self.names)

        M = scheffe_matrix(X, self.model)
        n, p = M.shape
        if n < p:
            import warnings
            warnings.warn(
                f"n={n} < p={p}: model is under-determined (R^2 will be "
                "misleadingly perfect). Add more runs.", UserWarning, stacklevel=2)

        beta, _, _, _ = np.linalg.lstsq(M, y, rcond=None)

        # Fit statistics
        y_hat = M @ beta
        resid = y - y_hat
        sse = float(resid @ resid)
        sst = float(((y - y.mean()) ** 2).sum())
        df_resid = max(n - p, 0)
        df_model = max(p - 1, 1)

        self._M = M
        self._n, self._p = n, p
        self.coefficients = beta
        self.fitted_values = y_hat
        self.residuals = resid
        self.sse, self.sst = sse, sst
        self.df_resid, self.df_model = df_resid, df_model
        self.r2 = 1.0 - sse / sst if sst > 0 else 1.0
        self.adj_r2 = (1.0 - (sse / df_resid) / (sst / (n - 1))
                       if df_resid > 0 and sst > 0 else float("nan"))
        self.rmse = float(np.sqrt(sse / df_resid)) if df_resid > 0 else 0.0
        self.sigma2 = sse / df_resid if df_resid > 0 else 0.0
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: Sequence[Sequence[float]]) -> np.ndarray:
        self._check_fitted()
        M = scheffe_matrix(X, self.model)
        return M @ self.coefficients

    # ------------------------------------------------------------------
    def coefficient_table(self) -> pd.DataFrame:
        """Per-term estimate, std-error, t-stat and p-value."""
        self._check_fitted()
        XtX = self._M.T @ self._M
        try:
            cov = self.sigma2 * np.linalg.inv(XtX)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            se = np.full(self._p, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = self.coefficients / se
        p = (2 * stats.t.sf(np.abs(t), df=self.df_resid)
             if self.df_resid > 0 else np.full(self._p, np.nan))
        return pd.DataFrame({
            "term": self.term_names,
            "estimate": self.coefficients,
            "std_error": se,
            "t_stat": t,
            "p_value": p,
        })

    # ------------------------------------------------------------------
    def anova(self) -> pd.DataFrame:
        """Regression / Residual / Total ANOVA with F-test."""
        self._check_fitted()
        ss_model = self.sst - self.sse
        ms_model = ss_model / self.df_model if self.df_model > 0 else np.nan
        ms_resid = self.sse / self.df_resid if self.df_resid > 0 else np.nan
        F = ms_model / ms_resid if (ms_resid and ms_resid > 0) else np.nan
        p = (stats.f.sf(F, self.df_model, self.df_resid)
             if np.isfinite(F) else np.nan)
        return pd.DataFrame({
            "source": ["Regression", "Residual", "Total"],
            "SS": [ss_model, self.sse, self.sst],
            "df": [self.df_model, self.df_resid, self._n - 1],
            "MS": [ms_model, ms_resid, np.nan],
            "F": [F, np.nan, np.nan],
            "p_value": [p, np.nan, np.nan],
        })

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        self._check_fitted()
        return {
            "model": self.model,
            "n": self._n,
            "p": self._p,
            "r2": self.r2,
            "adj_r2": self.adj_r2,
            "rmse": self.rmse,
        }

    # ------------------------------------------------------------------
    # Persistence helpers (ProjectState integration)
    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        self._check_fitted()
        return {
            "model": self.model,
            "names": self.names,
            "q": self.q,
            "term_names": self.term_names,
            "coefficients": np.asarray(self.coefficients),
            "r2": self.r2, "adj_r2": self.adj_r2, "rmse": self.rmse,
        }

    @classmethod
    def from_state(cls, d: dict) -> "ScheffeModel":
        m = cls(model=d["model"], names=d.get("names"))
        m.q = d.get("q")
        m.term_names = d.get("term_names", [])
        m.coefficients = np.asarray(d["coefficients"])
        m.r2 = d.get("r2"); m.adj_r2 = d.get("adj_r2"); m.rmse = d.get("rmse")
        m._fitted = True
        return m

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ScheffeModel is not fitted; call .fit() first.")
