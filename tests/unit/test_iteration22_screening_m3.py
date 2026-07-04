# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 22 — M3-минималка: интерпретируемый анализ скрининга (mixture-only).

Проверяем чистый слой :mod:`src.apps.campaign_screening` на реальном
:class:`MixtureProcessRunner` (тот же путь сборки, что в iteration18), с оракулом,
чьи отклики зависят ТОЛЬКО от состава (процесс игнорируется) — так ARD-важность и
Scheffé-fit восстанавливают заведомо известную структуру:

  * ``strength`` доминирует компонент A, ``gloss`` — компонент B;
  * :func:`screening_report` — JSON-сериализуемая сводка на свойство (коэффициенты
    Шеффе, ANOVA, R², значимые термы, ARD-ранжирование, q_eff);
  * :func:`influence_matrix` — матрица «компонент × свойство» восстанавливает
    доминирующий компонент по каждому свойству;
  * :func:`screening_overview` — сводка (матрица + R²) JSON-сериализуема.

Объём минималки: анализ по составу; процесс-оси не участвуют (их роль — за общими
суррогатами GP+MoE, канон §5/§12).
"""
import json
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps import campaign_screening as m3

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ----------------------------------------------------------------------
class _MixOracle:
    """Отклики зависят ТОЛЬКО от состава (процесс игнорируется).

    strength ← доминирует A; gloss ← доминирует B. Детерминированно (без шума):
    mixture-only Scheffé восстанавливает связь почти идеально, а ARD-важность
    ставит доминирующий компонент на первое место (importance = 1)."""

    property_names = ["strength", "gloss"]

    def evaluate(self, Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        A, B, C = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        strength = 10.0 * A + 2.0 * B + 2.0 * C
        gloss = 2.0 * A + 10.0 * B + 2.0 * C
        return np.column_stack([strength, gloss])


def _seeded_runner(n_seed=16):
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    r = MixtureProcessRunner(schema, _MixOracle(), seed=1, n_restarts=2,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    r.seed_initial(n=n_seed, seed=1)
    return r


_ARD = dict(n_restarts=2, seed=0)


# ======================================================================
# mixture_response — только состав + отклик, без NaN
# ======================================================================
def test_mixture_response_shape():
    r = _seeded_runner()
    X, y = m3.mixture_response(r, "strength")
    assert X.shape[1] == 3          # только mixture-доли (q=3)
    assert X.shape[0] == y.shape[0] == len(r.points)
    # доли неотрицательны и Σ≈1 (симплекс)
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-6)


# ======================================================================
# screening_report — структура + JSON-сериализуемость
# ======================================================================
def test_report_structure_and_json():
    r = _seeded_runner()
    rep = m3.screening_report(r, "strength", **_ARD)

    assert rep["property"] == "strength"
    assert rep["components"] == ["A", "B", "C"]
    # quadratic Scheffé, q=3 → p = 3 + 3 = 6 термов
    assert len(rep["coefficients"]) == 6
    assert len(rep["anova"]) == 3            # Regression / Residual / Total
    assert len(rep["component_ranking"]) == 3
    assert 1 <= rep["q_eff"] <= 3
    # отклик = чистая функция состава ⇒ mixture-only фит объясняет почти всё
    assert rep["summary"]["r2"] > 0.95
    assert isinstance(rep["summary"]["underdetermined"], bool)
    # весь отчёт уходит в ассистента/MCP как JSON
    assert isinstance(json.dumps(rep, ensure_ascii=False), str)


# ======================================================================
# influence_matrix — восстанавливает доминирующий компонент
# ======================================================================
def test_influence_matrix_recovers_dominant_component():
    r = _seeded_runner()
    mat = m3.influence_matrix(r, **_ARD)

    assert list(mat.index) == ["A", "B", "C"]
    assert list(mat.columns) == ["strength", "gloss"]
    # A доминирует в strength, B — в gloss
    assert mat["strength"].idxmax() == "A"
    assert mat["gloss"].idxmax() == "B"
    # важности нормированы в [0,1], максимум по свойству = 1
    vals = mat.to_numpy()
    assert np.nanmin(vals) >= -1e-9 and np.nanmax(vals) <= 1.0 + 1e-9
    assert np.isclose(mat["strength"].max(), 1.0)


# ======================================================================
# screening_overview — JSON-сериализуемая сводка (UI/MCP)
# ======================================================================
def test_overview_json_and_shape():
    r = _seeded_runner()
    ov = m3.screening_overview(r, **_ARD)
    assert ov["components"] == ["A", "B", "C"]
    assert ov["properties"] == ["strength", "gloss"]
    assert len(ov["importance"]) == 3 and len(ov["importance"][0]) == 2
    # R² по каждому свойству присутствует и высок (чистая функция состава)
    assert ov["r2"]["strength"] > 0.95 and ov["r2"]["gloss"] > 0.95
    assert isinstance(json.dumps(ov, ensure_ascii=False), str)


# ======================================================================
# явные отказы (A0.6)
# ======================================================================
def test_unknown_property_raises():
    r = _seeded_runner()
    try:
        m3.screening_report(r, "no_such", **_ARD)
        assert False, "ожидался KeyError для неизвестного свойства"
    except KeyError:
        pass
