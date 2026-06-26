"""
verification/mixture_process_truth.py — контролируемая синтетическая «истина»
над пространством mixture×process (REBUILD_SPEC §8 «известная синтетическая
функция», §13).

Назначение: дать «боевому» бенчмарку ИЗВЕСТНЫЙ отклик η(x, z) над ПРОИЗВЕДЕНИЕМ
симплекса рецепта (MIXTURE) и гиперкуба процессных параметров (PROCESS),
построенный ТЕМ ЖЕ генератором термов, что и модель пайплайна
(``design.block_model``). Тогда аналитический оптимум ветки считается прямо из
коэффициентов истины, а ядро обязано к нему сойтись.

    η = Σ β_i x_i + Σ β_ij x_i x_j        (Scheffé mixture, без intercept)
      + Σ γ_k z_k + Σ γ_kl z_k z_l        (RSM process, код [0,1])
      + Σ δ_ik x_i z_k                    (кросс mixture×process)

Координаты — СОСТАВНЫЕ ``Xc = [x_0..x_{q-1}, z_0..z_{d-1}]`` (process в коде
[0,1]), как в :func:`core.schema.composite_coords`. Мульти-вариант держит по
одному вектору коэффициентов на свойство (включая «цену»).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from ..core.schema import ProjectSchema, ordered_blocks
from ..design.block_model import ModelTerms, build_model_terms, model_matrix


class MixtureProcessTruth:
    """Известная функция отклика η(Xc) над одной схемой (mixture×process).

    ``coefficients`` — вектор длиной p (== числу термов ``build_model_terms``),
    в том же каноническом порядке термов. ``true`` — безшумный отклик,
    ``evaluate`` добавляет гауссов шум ``N(0, noise_sd²)``.
    """

    def __init__(self, schema: ProjectSchema, coefficients: Sequence[float],
                 noise_sd: float = 0.0, seed: Optional[int] = None,
                 terms: Optional[ModelTerms] = None):
        self.schema = schema
        self.terms = terms if terms is not None else build_model_terms(schema)
        coef = np.asarray(coefficients, dtype=float).ravel()
        if coef.size != self.terms.p:
            raise ValueError(
                f"Ожидалось {self.terms.p} коэффициентов (число термов модели), "
                f"получено {coef.size}.")
        self.coefficients = coef
        self.noise_sd = float(noise_sd)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def q(self) -> int:
        return self.terms.q

    @property
    def d(self) -> int:
        return self.terms.d

    def _zero_inactive_mixture(self, Xc: np.ndarray,
                               active_schema: ProjectSchema) -> np.ndarray:
        """Занулить mixture-компоненты, ОТСУТСТВУЮЩИЕ в ``active_schema`` (§15.0.4).

        Append-семантика mixture-компонента: пока компонент (например, C) не
        введён в схему фазы, он НЕ вносит вклад — его реальное значение на грани
        симплекса C=0. Зануление столбца C до сборки модельной матрицы убирает
        ВСЕ термы с C (``β_C·C``, ``A·B·C``, …). ``active_schema=None`` сюда не
        попадает (полная схема — вклад всех компонентов, обратная совместимость).
        """
        active = set(active_schema.mixture_names)
        inactive = [i for i, nm in enumerate(self.schema.mixture_names)
                    if nm not in active]
        if not inactive:
            return Xc
        Xc = np.array(Xc, dtype=float, copy=True)
        Xc[:, inactive] = 0.0
        return Xc

    def true(self, Xc, *, active_schema: Optional[ProjectSchema] = None
             ) -> np.ndarray:
        """Безшумный отклик по составным координатам ``Xc`` (n×(q+d)).

        ``active_schema`` (§15.0.4): если задан, mixture-компоненты, которых в нём
        НЕТ (например, C в фазе 1), зануляются ⇒ их термы не вносят вклад (истина
        считается на грани симплекса C=0, а не при «молчаливом» baseline C=1/3).
        ``None`` → полная схема истины (вклад всех компонентов).
        """
        Xc = np.atleast_2d(np.asarray(Xc, dtype=float))
        if active_schema is not None:
            Xc = self._zero_inactive_mixture(Xc, active_schema)
        M = model_matrix(self.schema, Xc, terms=self.terms)
        return M @ self.coefficients

    def evaluate(self, Xc, *, active_schema: Optional[ProjectSchema] = None
                 ) -> np.ndarray:
        """Зашумлённый отклик (добавляет ``N(0, noise_sd²)``)."""
        y = self.true(Xc, active_schema=active_schema)
        if self.noise_sd > 0:
            y = y + self._rng.normal(0.0, self.noise_sd, size=y.shape)
        return y

    def __call__(self, Xc, *, active_schema: Optional[ProjectSchema] = None
                 ) -> np.ndarray:
        return self.evaluate(Xc, active_schema=active_schema)


    def __repr__(self) -> str:
        return (f"MixtureProcessTruth(q={self.q}, d={self.d}, "
                f"p={self.terms.p}, noise_sd={self.noise_sd})")


class MultiMixtureProcessTruth:
    """P независимых истин над общей схемой: ``{property → coef-вектор}``.

    Мерит сразу все свойства; ``true``/``evaluate`` возвращают массив (n×P) в
    порядке ключей ``coef_by_property``. Одно из свойств обычно — «цена».
    Каждое свойство получает свой seed (разный шум на каждое свойство).
    """

    def __init__(self, schema: ProjectSchema,
                 coef_by_property: Mapping[str, Sequence[float]],
                 noise_sd: float = 0.0, seed: Optional[int] = None):
        if not coef_by_property:
            raise ValueError("Нужно хотя бы одно свойство.")
        self.schema = schema
        self.terms = build_model_terms(schema)
        self.property_names = list(coef_by_property.keys())
        base = 0 if seed is None else int(seed)
        self.truths: Dict[str, MixtureProcessTruth] = {}
        for i, name in enumerate(self.property_names):
            self.truths[name] = MixtureProcessTruth(
                schema, coef_by_property[name], noise_sd=noise_sd,
                seed=base + 1009 * i, terms=self.terms)

    @property
    def n_properties(self) -> int:
        return len(self.property_names)

    def true(self, Xc, *, active_schema: Optional[ProjectSchema] = None
             ) -> np.ndarray:
        """Безшумные отклики всех свойств, форма (n×P).

        ``active_schema`` (§15.0.4) пробрасывается в каждую истину свойства:
        отсутствующие в нём mixture-компоненты не вносят вклад (грань C=0).
        """
        return np.column_stack([self.truths[n].true(Xc, active_schema=active_schema)
                                for n in self.property_names])

    def evaluate(self, Xc, *, active_schema: Optional[ProjectSchema] = None
                 ) -> np.ndarray:
        """Зашумлённые отклики всех свойств, форма (n×P)."""
        return np.column_stack(
            [self.truths[n].evaluate(Xc, active_schema=active_schema)
             for n in self.property_names])

    def __call__(self, Xc, *, active_schema: Optional[ProjectSchema] = None
                 ) -> np.ndarray:
        return self.evaluate(Xc, active_schema=active_schema)


    def __repr__(self) -> str:
        return (f"MultiMixtureProcessTruth(P={self.n_properties}, "
                f"props={self.property_names})")


def composite_random_points(schema: ProjectSchema, n: int,
                            seed: Optional[int] = None) -> np.ndarray:
    """Случайные точки на ПРОИЗВЕДЕНИИ области: симплекс рецепта × куб [0,1]^d.

    Возвращает составные координаты ``[x..., z_code...]`` (n×(q+d)) в каноническом
    порядке блоков (mixture, затем process). Mixture-часть берётся из
    :class:`SimplexRegion` (учитывает границы долей), process-часть — равномерно
    в коде [0,1].
    """
    rng = np.random.default_rng(seed)
    parts = []
    for b in ordered_blocks(schema):
        if b.is_mixture:
            reg = b.as_simplex_region()
            parts.append(np.atleast_2d(reg.random_points(int(n), seed=seed)))
        else:  # process: код [0,1]^d
            parts.append(rng.uniform(0.0, 1.0, size=(int(n), b.size)))
    if not parts:
        return np.empty((int(n), 0))
    return np.hstack(parts)
