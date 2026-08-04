"""
models/gp_expert.py — M6 (single expert, K=1): GP with polynomial mean.

Configuration (REBUILD_SPEC M6 + kernel block):
  * mean function  = Scheffe polynomial (interpretable trend);
  * kernel         = Matern 5/2 ARD (models the *residuals*) + WhiteNoise;
  * hyperparameters by maximize marginal likelihood, multiple restarts (#9);
  * lower bounds on lengthscale (>= min inter-point distance, #11) and on
    noise (>= jitter 1e-6, #11) to avoid oscillation / noise over-fit.

Prediction returns an HONEST predictive std (kernel includes WhiteNoise, so the
test-point variance carries measurement noise).  RBF is available as the
"smooth" comparison option.

R reference: ``DiceKriging::km(covtype="matern5_2")`` / ``GauPro``;
compare mu(x*) and sigma^2(x*) at fixed hyperparameters (atol 1e-6).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, Matern, RBF, WhiteKernel,
)

from ..core.linalg import n_scheffe_params
from .scheffe import ScheffeModel

_ORDER_NAMES = {1: "linear", 2: "quadratic", 3: "cubic",
                4: "quartic", 5: "quintic"}
_NAME_ORDERS = {v: k for k, v in _ORDER_NAMES.items()}


class _ConstantMean:
    """Тренд-заглушка «среднее y» — последний уровень даунгрейда (n < q + запас).

    Честнее интерполирующего полинома: остатки остаются информативными,
    ядро GP учится на реальной вариации, σ не схлопывается в 0.
    """

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.full(X.shape[0], self.value)

    def to_state(self) -> dict:
        return {"model": "constant", "value": self.value}

    @classmethod
    def from_state(cls, d: dict) -> "_ConstantMean":
        return cls(d["value"])


def _min_pairwise_distance(X: np.ndarray) -> float:
    """Smallest Euclidean distance between distinct rows (for ℓ lower bound)."""
    n = X.shape[0]
    if n < 2:
        return 1e-2
    best = np.inf
    for i in range(n - 1):
        d = np.sqrt(((X[i + 1:] - X[i]) ** 2).sum(axis=1))
        m = d.min()
        if m < best:
            best = m
    return float(best) if np.isfinite(best) and best > 0 else 1e-3


@dataclass
class GPPrediction:
    mean: np.ndarray
    std: np.ndarray


class GPExpert:
    """Gaussian-process expert with a Scheffe-polynomial mean."""

    def __init__(self, mean_model: Union[str, int] = "quadratic",
                 kernel: str = "matern52", noise_floor: float = 1e-6,
                 n_restarts: int = 15, seed: Optional[int] = None,
                 names: Optional[Sequence[str]] = None,
                 mean_min_dof: int = 3):
        if kernel not in ("matern52", "rbf"):
            raise ValueError("kernel must be 'matern52' or 'rbf'.")
        self.mean_model = mean_model
        self.kernel = kernel
        self.noise_floor = noise_floor
        self.n_restarts = n_restarts
        self.seed = seed
        self.names = names
        # запас степеней свободы остатков тренда: тренд порядка m допускается
        # только при n ≥ p(m) + mean_min_dof (иначе OLS интерполирует, остатки
        # ≈ 0, ядру учиться нечему ⇒ σ=0 и GP «уверенно врёт»)
        self.mean_min_dof = int(mean_min_dof)
        self.mean_model_effective_: Union[str, int, None] = None
        self.mean_: Optional[ScheffeModel] = None
        self.gp_: Optional[GaussianProcessRegressor] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def _resolve_mean_model(self, n: int, q: int) -> Union[str, int]:
        """Эффективный порядок тренда: даунгрейд, пока n < p + mean_min_dof.

        Возвращает имя/порядок допустимого тренда Шеффе либо ``"constant"``,
        если даже линейный тренд не идентифицируем с запасом.
        """
        req = self.mean_model
        req_order = (_NAME_ORDERS[req] if isinstance(req, str) and req in _NAME_ORDERS
                     else int(req))
        req_order = min(req_order, q)          # порядок Шеффе не выше q
        for m in range(req_order, 0, -1):
            if n >= n_scheffe_params(q, m) + self.mean_min_dof:
                return _ORDER_NAMES.get(m, m)
        return "constant"

    # ------------------------------------------------------------------
    def fit(self, X, y) -> "GPExpert":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        n, q = X.shape

        # --- mean = Scheffe OLS trend (с защитой от n<p: даунгрейд порядка) ---
        eff = self._resolve_mean_model(n, q)
        self.mean_model_effective_ = eff
        if eff != self.mean_model:
            warnings.warn(
                f"GPExpert: тренд '{self.mean_model}' не идентифицируем при "
                f"n={n}, q={q} (нужно n ≥ p + {self.mean_min_dof}); "
                f"использован тренд '{eff}'.", UserWarning, stacklevel=2)
        if eff == "constant":
            self.mean_ = _ConstantMean(float(y.mean()) if n else 0.0)
        else:
            self.mean_ = ScheffeModel(model=eff, names=self.names).fit(X, y)
        resid = y - self.mean_.predict(X)

        # --- kernel on residuals ---
        lb = max(_min_pairwise_distance(X), 1e-3)
        ub = 1e3
        ls0 = np.full(q, np.clip(np.sqrt(q) * lb, lb, ub))
        const = ConstantKernel(np.var(resid) + 1e-6, (1e-6, 1e6))
        if self.kernel == "matern52":
            base = Matern(length_scale=ls0, nu=2.5, length_scale_bounds=(lb, ub))
        else:
            base = RBF(length_scale=ls0, length_scale_bounds=(lb, ub))
        white = WhiteKernel(noise_level=max(np.var(resid) * 0.1, self.noise_floor),
                            noise_level_bounds=(self.noise_floor, 1e2))
        kernel = const * base + white

        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=0.0, normalize_y=False,
            n_restarts_optimizer=self.n_restarts, random_state=self.seed,
        )
        gp.fit(X, resid)
        self.gp_ = gp
        self._X, self._resid = X, resid
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X, return_std: bool = True):
        self._check_fitted()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        trend = self.mean_.predict(X)
        if return_std:
            resid_mean, resid_std = self.gp_.predict(X, return_std=True)
            return GPPrediction(mean=trend + resid_mean, std=resid_std)
        resid_mean = self.gp_.predict(X, return_std=False)
        return trend + resid_mean

    # ------------------------------------------------------------------
    @property
    def lengthscales(self) -> np.ndarray:
        self._check_fitted()
        base = self.gp_.kernel_.k1.k2          # Const * base -> k1 ; base -> k2
        ls = np.atleast_1d(base.length_scale).astype(float)
        return ls

    @property
    def noise_level(self) -> float:
        self._check_fitted()
        return float(self.gp_.kernel_.k2.noise_level)

    @property
    def log_marginal_likelihood(self) -> float:
        self._check_fitted()
        return float(self.gp_.log_marginal_likelihood_value_)

    # ------------------------------------------------------------------
    def to_state(self) -> dict:
        """Serialisable state; refit-free reconstruction via fixed kernel."""
        self._check_fitted()
        return {
            "mean_model": self.mean_model,
            "mean_model_effective": self.mean_model_effective_,
            "kernel": self.kernel,
            "noise_floor": self.noise_floor,
            "names": self.names,
            "scheffe": self.mean_.to_state(),
            "kernel_theta": np.asarray(self.gp_.kernel_.theta),
            "X_train": np.asarray(self._X),
            "resid_train": np.asarray(self._resid),
        }

    @classmethod
    def from_state(cls, d: dict) -> "GPExpert":
        obj = cls(mean_model=d["mean_model"], kernel=d["kernel"],
                  noise_floor=d.get("noise_floor", 1e-6), names=d.get("names"))
        obj.mean_model_effective_ = d.get("mean_model_effective", d["mean_model"])
        if d["scheffe"].get("model") == "constant":
            obj.mean_ = _ConstantMean.from_state(d["scheffe"])
        else:
            obj.mean_ = ScheffeModel.from_state(d["scheffe"])
        X = np.asarray(d["X_train"]); resid = np.asarray(d["resid_train"])
        q = X.shape[1]
        if d["kernel"] == "matern52":
            base = Matern(length_scale=np.ones(q), nu=2.5)
        else:
            base = RBF(length_scale=np.ones(q))
        kernel = ConstantKernel(1.0) * base + WhiteKernel(1.0)
        kernel = kernel.clone_with_theta(np.asarray(d["kernel_theta"]))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0,
                                      optimizer=None, normalize_y=False)
        gp.fit(X, resid)                       # no optimisation, just Cholesky
        obj.gp_ = gp
        obj._X, obj._resid = X, resid
        obj._fitted = True
        return obj

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("GPExpert is not fitted; call .fit() first.")
