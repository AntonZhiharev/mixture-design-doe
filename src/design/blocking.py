"""
design/blocking.py — Blocking (блокировка) для смесевых планов.

Два режима, соответствующих двум фазам конвейера:

1. **Стартовый blocking** (:func:`assign_blocks_start`) — ОПТИМАЛЬНОЕ разбиение
   уже выбранного D-оптимального плана M2 на блоки (партии/дни) заданных
   размеров. Критерий — log det информации БЛОЧНОЙ модели: термы Шеффе +
   dummy-столбцы блоков 2..B (блок 1 — референс: в модели Шеффе Σxᵢ=1 играет
   роль интерсепта, полный набор dummy был бы коллинеарен). Алгоритм —
   interchange (парные обмены точек между блоками) с мультистартом —
   аналог ``AlgDesign::optBlock``.

2. **Blocking добора** (:func:`augment_block_labels`) — ПОСЛЕДОВАТЕЛЬНЫЙ
   blocking: каждый добор точек (M5 / M7 / ветки) ставится ОТДЕЛЬНОЙ партией
   и получает НОВЫЙ блок (следующий свободный номер). Смещение «новая партия /
   другой день» ловится dummy нового блока и не подмешивается в оценки термов
   смеси. Оптимизировать распределение здесь нечего — партия задана временем
   измерения, а не выбором.

Диагностика (:func:`blocking_diagnostics`) показывает цену блокировки:
D-эффективность блочной модели и потерю информации модельных термов после
учёта блоков (через дополнение Шура) относительно неблокированного плана.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from ..core.linalg import scheffe_matrix, slogdet, d_efficiency


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def balanced_block_sizes(n: int, n_blocks: int) -> List[int]:
    """Сбалансированные размеры блоков: ``n`` опытов на ``n_blocks`` блоков.

    Первые ``n % n_blocks`` блоков получают на один опыт больше.
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks должен быть >= 1 (получено {n_blocks}).")
    base, rem = divmod(int(n), int(n_blocks))
    return [base + 1 if b < rem else base for b in range(n_blocks)]


def block_dummies(labels: Sequence[int]) -> np.ndarray:
    """Dummy-матрица блоков ``(n, B-1)``: столбцы для блоков 2..B.

    Первый (минимальный) блок — референс: в канонической форме Шеффе Σxᵢ=1
    заменяет интерсепт, поэтому полный набор из B индикаторов был бы
    коллинеарен с линейными термами.
    """
    lab = np.asarray(labels, dtype=int).ravel()
    uniq = np.unique(lab)
    if len(uniq) <= 1:
        return np.empty((len(lab), 0))
    return np.column_stack([(lab == u).astype(float) for u in uniq[1:]])


def blocked_scheffe_matrix(X: np.ndarray, labels: Sequence[int],
                           model: Union[str, int]) -> np.ndarray:
    """Модельная матрица блочной модели Шеффе: ``[термы Шеффе | dummy блоков]``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    F = scheffe_matrix(X, model)
    Z = block_dummies(labels)
    return np.hstack([F, Z]) if Z.shape[1] else F


def blocked_logdet(X: np.ndarray, labels: Sequence[int],
                   model: Union[str, int], ridge: float = 1e-10) -> float:
    """log det(MᵀM + ridge·I) блочной модельной матрицы."""
    M = blocked_scheffe_matrix(X, labels, model)
    return slogdet(M.T @ M + ridge * np.eye(M.shape[1]))


# ---------------------------------------------------------------------------
# 1) Стартовый blocking: оптимальное разбиение плана M2 на блоки
# ---------------------------------------------------------------------------

@dataclass
class BlockingResult:
    """Результат оптимального разбиения плана на блоки."""
    labels: np.ndarray                  # метки блоков 1..B на каждый опыт
    n_blocks: int
    block_sizes: List[int]
    logdet: float                       # log det блочной информации
    d_efficiency: float                 # D-эфф. блочной модельной матрицы
    model: Union[str, int] = "quadratic"
    n_restarts: int = 0
    history: List[float] = field(default_factory=list)  # лучший logdet/рестарт

    def to_state(self) -> dict:
        return {
            "labels": np.asarray(self.labels, dtype=int),
            "n_blocks": int(self.n_blocks),
            "block_sizes": [int(s) for s in self.block_sizes],
            "logdet": float(self.logdet),
            "d_efficiency": float(self.d_efficiency),
            "model": self.model,
            "n_restarts": int(self.n_restarts),
        }


def assign_blocks_start(X: np.ndarray, n_blocks: int,
                        model: Union[str, int] = "quadratic",
                        block_sizes: Optional[Sequence[int]] = None,
                        n_restarts: int = 8, max_iter: int = 50,
                        ridge: float = 1e-10,
                        seed: Optional[int] = None) -> BlockingResult:
    """Оптимально разбить точки плана ``X`` на блоки (стартовый blocking).

    Interchange-алгоритм: случайное сбалансированное разбиение → парные обмены
    точек между блоками, пока растёт log det информации БЛОЧНОЙ модели
    (термы Шеффе + dummy блоков). Мультистарт против локальных максимумов.

    Parameters
    ----------
    X           : (n, q) точки уже выбранного плана (D-оптимального).
    n_blocks    : число блоков (партий/дней).
    model       : ЯВНЫЙ порядок модели Шеффе ('linear'|'quadratic'|...).
    block_sizes : явные размеры блоков (сумма = n); None → сбалансированно.
    n_restarts  : число случайных рестартов.
    max_iter    : максимум interchange-проходов на рестарт.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    if n_blocks <= 1:
        labels = np.ones(n, dtype=int)
        M = blocked_scheffe_matrix(X, labels, model)
        return BlockingResult(labels=labels, n_blocks=1, block_sizes=[n],
                              logdet=slogdet(M.T @ M + ridge * np.eye(M.shape[1])),
                              d_efficiency=d_efficiency(M), model=model)

    sizes = (list(block_sizes) if block_sizes is not None
             else balanced_block_sizes(n, n_blocks))
    if len(sizes) != n_blocks or sum(sizes) != n or any(s < 1 for s in sizes):
        raise ValueError(
            f"block_sizes={sizes} несовместимы: нужно {n_blocks} блоков "
            f"с суммой {n} и размером >= 1.")

    F = scheffe_matrix(X, model)
    p = F.shape[1]
    n_par = p + n_blocks - 1
    if n < n_par:
        import warnings
        warnings.warn(
            f"n={n} < p+B-1={n_par}: блочная модель не оценивается "
            "(сингулярна). Уменьшите число блоков или добавьте опыты.",
            UserWarning, stacklevel=2)

    def _logdet(lab: np.ndarray) -> float:
        Z = block_dummies(lab)
        M = np.hstack([F, Z]) if Z.shape[1] else F
        k = M.shape[1]
        return slogdet(M.T @ M + ridge * np.eye(k))

    # шаблон меток по размерам блоков: [1]*s1 + [2]*s2 + ...
    template = np.concatenate([np.full(s, b + 1, dtype=int)
                               for b, s in enumerate(sizes)])

    rng = np.random.default_rng(seed)
    best_lab: Optional[np.ndarray] = None
    best_ld = float("-inf")
    history: List[float] = []

    for _ in range(max(1, n_restarts)):
        lab = np.empty(n, dtype=int)
        lab[rng.permutation(n)] = template
        cur_ld = _logdet(lab)

        improved, it = True, 0
        while improved and it < max_iter:
            improved = False
            it += 1
            for i in range(n):
                for j in range(i + 1, n):
                    if lab[i] == lab[j]:
                        continue
                    lab[i], lab[j] = lab[j], lab[i]
                    ld = _logdet(lab)
                    if ld > cur_ld + 1e-12:
                        cur_ld = ld
                        improved = True
                    else:
                        lab[i], lab[j] = lab[j], lab[i]

        history.append(cur_ld)
        if cur_ld > best_ld:
            best_ld = cur_ld
            best_lab = lab.copy()

    M = blocked_scheffe_matrix(X, best_lab, model)
    return BlockingResult(labels=best_lab, n_blocks=int(n_blocks),
                          block_sizes=[int(s) for s in sizes],
                          logdet=float(best_ld), d_efficiency=d_efficiency(M),
                          model=model, n_restarts=int(n_restarts),
                          history=history)


# ---------------------------------------------------------------------------
# 2) Blocking добора: каждый добор = НОВЫЙ блок (последовательный blocking)
# ---------------------------------------------------------------------------

def augment_block_labels(existing: Optional[Sequence[int]], n_new: int,
                         n_new_blocks: int = 1) -> np.ndarray:
    """Метки блоков после ДОБОРА ``n_new`` точек: добор — это НОВЫЙ блок.

    Последовательный blocking: партия добора измеряется в другое время /
    из другого замеса, чем база, поэтому получает следующий свободный номер
    блока. При ``n_new_blocks > 1`` добор сам делится на несколько
    последовательных новых блоков (сбалансированно, по порядку точек).

    Возвращает ПОЛНЫЙ массив меток (существующие + новые). ``existing=None``
    или пустой → новые точки начинают с блока 1.
    """
    if n_new < 0:
        raise ValueError(f"n_new должен быть >= 0 (получено {n_new}).")
    ex = (np.asarray(existing, dtype=int).ravel()
          if existing is not None else np.empty(0, dtype=int))
    start = int(ex.max()) if ex.size else 0
    if n_new == 0:
        return ex.copy()
    nb = max(1, int(n_new_blocks))
    if nb == 1:
        new = np.full(int(n_new), start + 1, dtype=int)
    else:
        sizes = balanced_block_sizes(int(n_new), nb)
        new = np.concatenate([np.full(s, start + 1 + b, dtype=int)
                              for b, s in enumerate(sizes) if s > 0])
    return np.concatenate([ex, new])


# ---------------------------------------------------------------------------
# Диагностика: цена блокировки
# ---------------------------------------------------------------------------

def blocking_diagnostics(X: np.ndarray, labels: Sequence[int],
                         model: Union[str, int],
                         ridge: float = 1e-10) -> Dict[str, object]:
    """Диагностика блочного плана: во что обходится блокировка.

    * ``d_eff_unblocked``  — D-эфф. модельных термов БЕЗ блоков (ориентир);
    * ``d_eff_blocked``    — D-эфф. полной блочной матрицы [F | Z];
    * ``d_eff_model_adj``  — D-эфф. модельных термов ПОСЛЕ учёта блоков
      (информация = дополнение Шура ``FᵀF − FᵀZ(ZᵀZ)⁻¹ZᵀF`` — ровно то, что
      остаётся на оценку термов смеси при фиксированных эффектах блоков);
    * ``d_loss_pct``       — потеря информации модельных термов из-за
      блокировки, % (0 — блоки ортогональны модели).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    lab = np.asarray(labels, dtype=int).ravel()
    F = scheffe_matrix(X, model)
    n, p = F.shape
    Z = block_dummies(lab)

    d_unb = d_efficiency(F)
    Mb = np.hstack([F, Z]) if Z.shape[1] else F
    d_blk = d_efficiency(Mb)

    if Z.shape[1]:
        ZtZ = Z.T @ Z + ridge * np.eye(Z.shape[1])
        A = F.T @ F - F.T @ Z @ np.linalg.inv(ZtZ) @ Z.T @ F
    else:
        A = F.T @ F
    ld = slogdet(A + ridge * np.eye(p))
    d_adj = float(np.exp(ld / p) / n) if np.isfinite(ld) and n else 0.0
    loss = (100.0 * (1.0 - d_adj / d_unb)) if d_unb > 0 else float("nan")

    uniq, cnt = np.unique(lab, return_counts=True)
    return {
        "n_blocks": int(len(uniq)),
        "block_sizes": {int(u): int(c) for u, c in zip(uniq, cnt)},
        "d_eff_unblocked": float(d_unb),
        "d_eff_blocked": float(d_blk),
        "d_eff_model_adj": float(d_adj),
        "d_loss_pct": float(loss),
    }