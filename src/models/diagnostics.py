"""
models/diagnostics.py — M7/§12: детекторы misspecification (FinalCheckList Блок 7).

Канон §12: bias модели возникает из-за неверной спецификации (misspecification),
а не из-за выбора точек. Поэтому проект должен УМЕТЬ ЗАМЕЧАТЬ, когда общей модели
(per-property MoE) перестаёт хватать, и сигналить о необходимости перестроения.

Здесь — три независимых, чистых (без побочных эффектов) детектора:

  1. «Точка вне всех режимов» (малые gₖ): gating-классификатор не уверен ни в
     одном режиме → max_k gₖ(x) мал. Дополнительно — novelty (экстраполяция):
     насколько точка далека от обучающей базы в пространстве рецептов.
  2. Триггер переразбиения K+1: оптимальное по BIC число режимов на ТЕКУЩИХ
     откликах отличается от того, на котором обучена модель.
  3. Детектор стагнации ветки — в :mod:`src.design.branches`
     (`Branch.is_stagnating`): d_best не растёт `patience` раундов.

Функции принимают уже обученный :class:`MixtureOfExperts` и/или массивы, поэтому
легко тестируются и переживают save/load (не требуют обучающей выборки в модели).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .clustering import GMMRegimes


# ----------------------------------------------------------------------
# Геометрия базы: расстояния / novelty (экстраполяция)
# ----------------------------------------------------------------------
def min_distance(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Минимальное евклидово расстояние от каждой строки ``query`` до ``ref``."""
    query = np.atleast_2d(np.asarray(query, float))
    ref = np.atleast_2d(np.asarray(ref, float))
    if ref.size == 0:
        return np.full(query.shape[0], np.inf)
    d = np.sqrt(((query[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
    return d.min(axis=1)


def nn_scale(ref: np.ndarray) -> float:
    """Типичный масштаб базы: медиана расстояний до ближайшего соседа внутри ``ref``."""
    ref = np.atleast_2d(np.asarray(ref, float))
    if ref.shape[0] < 2:
        return 0.0
    d = np.sqrt(((ref[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return float(np.median(nn))


def novelty_ratio(query: np.ndarray, ref: np.ndarray,
                  eps: float = 1e-9) -> np.ndarray:
    """Безразмерная «новизна» точки = (расстояние до базы) / (масштаб базы).

    ≈1 — точка типична для базы; ≫1 — далеко от всех обучающих точек
    (кандидат на экстраполяцию / вне покрытия модели).
    """
    scale = nn_scale(ref)
    dist = min_distance(query, ref)
    if scale <= eps:
        return np.where(dist > eps, np.inf, 0.0)
    return dist / (scale + eps)


# ----------------------------------------------------------------------
# Детектор 1: «вне всех режимов» (малые gₖ)
# ----------------------------------------------------------------------
def gating_confidence(moe, X: np.ndarray) -> np.ndarray:
    """Уверенность gating: ``max_k gₖ(x)`` для каждой точки (в [0,1]).

    Для одно-режимной модели (K=1) всегда 1.0 — «вне режимов» неприменимо.
    """
    G = moe.gating_proba(X)
    return np.asarray(G, float).max(axis=1)


def out_of_regime_mask(moe, X: np.ndarray, tau: float = 0.6) -> np.ndarray:
    """Маска точек «вне всех режимов»: ``max_k gₖ(x) < tau`` (только при K>1)."""
    if getattr(moe, "K_", 1) <= 1:
        return np.zeros(np.atleast_2d(np.asarray(X, float)).shape[0], dtype=bool)
    return gating_confidence(moe, X) < float(tau)


# ----------------------------------------------------------------------
# Детектор 2: триггер переразбиения (K+1 / перекластеризация)
# ----------------------------------------------------------------------
def recommend_n_regimes(y: np.ndarray, k_range: Sequence[int] = range(1, 6),
                        n_init: int = 10, seed: Optional[int] = None) -> int:
    """Рекомендуемое число режимов по BIC на ТЕКУЩИХ откликах ``y``."""
    reg = GMMRegimes(k_range=k_range, n_init=n_init, seed=seed).fit(
        np.asarray(y, float).ravel())
    return int(reg.n_regimes)


def needs_recluster(moe, y: np.ndarray, k_range: Sequence[int] = range(1, 6),
                    n_init: int = 10, seed: Optional[int] = None
                    ) -> Tuple[bool, int]:
    """Нужно ли переразбиение: BIC-оптимальное K на ``y`` ≠ текущему K модели.

    Возвращает ``(needs, recommended_K)``.
    """
    rec = recommend_n_regimes(y, k_range=k_range, n_init=n_init, seed=seed)
    return (rec != int(getattr(moe, "K_", 1)), rec)


# ----------------------------------------------------------------------
# Сводный отчёт по набору точек
# ----------------------------------------------------------------------
@dataclass
class MisspecReport:
    """Поточечная диагностика misspecification для набора точек."""
    gating_max: np.ndarray          # max_k gₖ(x)
    disagreement: np.ndarray        # Σ gₖ (μₖ−ŷ)²  (between-expert)
    std: np.ndarray                 # предиктивный std MoE
    novelty: np.ndarray             # новизна относительно базы
    out_of_regime: np.ndarray       # bool: max gₖ < tau (K>1)
    extrapolation: np.ndarray       # bool: novelty > novelty_factor

    @property
    def flagged(self) -> np.ndarray:
        """Точка подозрительна, если вне режимов ИЛИ экстраполяция."""
        return np.logical_or(self.out_of_regime, self.extrapolation)

    @property
    def summary(self) -> dict:
        n = int(self.gating_max.shape[0])
        f = lambda m: float(np.mean(m)) if n else 0.0  # noqa: E731
        return {
            "n": n,
            "frac_out_of_regime": f(self.out_of_regime),
            "frac_extrapolation": f(self.extrapolation),
            "frac_flagged": f(self.flagged),
            "mean_gating_max": float(np.mean(self.gating_max)) if n else 1.0,
            "mean_disagreement": float(np.mean(self.disagreement)) if n else 0.0,
            "max_novelty": float(np.max(self.novelty)) if n else 0.0,
        }


def diagnose(moe, X: np.ndarray, ref: np.ndarray, tau: float = 0.6,
             novelty_factor: float = 3.0) -> MisspecReport:
    """Поточечная диагностика набора ``X`` относительно базы ``ref``.

    ``moe``   — обученный per-property суррогат проекта;
    ``ref``   — общая база точек проекта (обучающие рецепты);
    ``tau``   — порог уверенности gating для «вне режимов»;
    ``novelty_factor`` — во сколько раз дальше типичного, чтобы счесть
    точку экстраполяцией.
    """
    X = np.atleast_2d(np.asarray(X, float))
    pred = moe.predict(X)
    gmax = gating_confidence(moe, X)
    nov = novelty_ratio(X, ref)
    oor = out_of_regime_mask(moe, X, tau=tau)
    extra = nov > float(novelty_factor)
    return MisspecReport(
        gating_max=gmax,
        disagreement=np.asarray(pred.disagreement, float),
        std=np.asarray(pred.std, float),
        novelty=nov,
        out_of_regime=oor,
        extrapolation=extra,
    )
