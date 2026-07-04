# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 21 — C3 (§17.6.1): Excel-выгрузка рекомендованного РЕЦЕПТА ветки.

Проверяем чистые (без Streamlit) хелперы :mod:`src.apps.campaign_ui`:

  * :func:`branch_recipe_dataframe` — рецепт x* ветки одной строкой: имя ветки,
    составные координаты (mixture-доли + процесс в РЕАЛЬНЫХ единицах, замечание 2),
    предсказанные свойства целей ``{свойство} (прогноз)``, итог ``d_overall`` и
    по-целевые ``d[{свойство}]``; при ``batch_kg`` — расход сырья на пробу;
  * :func:`branch_recipe_excel_bytes` — та же строка → валидные xlsx-байты
    (лист «Рецепт»), читаемые обратно pandas/openpyxl.

Рецепт считается по ОБЩИМ суррогатам (GP+MoE) через M8-argmax
(:meth:`MixtureProcessRunner.optimize_xbest`) — канон §5/§12 (одна модель физики).
"""
import io
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.apps import campaign_ui as ui

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ----------------------------------------------------------------------
def _seeded_runner(n_seed=14):
    """Runner {A,B,C}×{T,P} с 2 свойствами и общей базой (процесс в реальных ед.)."""
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[150.0, 1.0],
                                 upper=[200.0, 5.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    p = build_model_terms(schema).p
    rng = np.random.default_rng(0)
    oracle = MultiMixtureProcessTruth(
        schema, {"strength": rng.normal(size=p), "gloss": rng.normal(size=p)},
        noise_sd=0.0)
    r = MixtureProcessRunner(schema, oracle, seed=1, n_restarts=2,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5])
    r.seed_initial(n=n_seed, seed=1)
    return r


def _add_branch(r):
    r.add_branch("premium",
                 {"strength": DesirabilitySpec("max", low=-5, high=5),
                  "gloss": DesirabilitySpec("max", low=-5, high=5)},
                 branch_id="b1")


def _recipe_kwargs():
    """Дешёвые параметры M8-argmax — тесту не нужна полная точность оптимума."""
    return dict(n_candidates=200, refine_iters=20, n_starts=2)


# ======================================================================
# branch_recipe_dataframe — структура строки рецепта
# ======================================================================
def test_recipe_dataframe_columns_and_ranges():
    r = _seeded_runner()
    _add_branch(r)
    df = ui.branch_recipe_dataframe(r, "b1", **_recipe_kwargs())

    assert len(df) == 1
    row = df.iloc[0]
    # имя ветки + все составные координаты
    assert row["ветка"] == "premium"
    for cn in ["A", "B", "C", "T", "P"]:
        assert cn in df.columns
    # mixture-доли в [0,1] и Σ≈1 (симплекс)
    frac = np.array([float(row[c]) for c in ["A", "B", "C"]])
    assert np.all(frac >= -1e-6) and np.all(frac <= 1 + 1e-6)
    assert abs(frac.sum() - 1.0) < 1e-3
    # процесс — в РЕАЛЬНЫХ единицах (замечание 2), а не в коде [0,1]
    assert 150.0 - 1e-6 <= float(row["T"]) <= 200.0 + 1e-6
    assert 1.0 - 1e-6 <= float(row["P"]) <= 5.0 + 1e-6
    # предсказания целей + итог/по-целевые желательности
    assert "strength (прогноз)" in df.columns
    assert "gloss (прогноз)" in df.columns
    assert "d_overall" in df.columns
    assert "d[strength]" in df.columns and "d[gloss]" in df.columns
    assert 0.0 - 1e-9 <= float(row["d_overall"]) <= 1.0 + 1e-9


def test_recipe_dataframe_batch_adds_mass_columns():
    r = _seeded_runner()
    _add_branch(r)
    batch = 10.0
    df = ui.branch_recipe_dataframe(r, "b1", batch_kg=batch, **_recipe_kwargs())
    row = df.iloc[0]
    # расход сырья на пробу = доля · batch; Σ по компонентам ≈ batch
    masses = []
    for c in ["A", "B", "C"]:
        col = f"{c} ({ui.MASS_UNIT})"
        assert col in df.columns
        masses.append(float(row[col]))
    assert abs(sum(masses) - batch) < 1e-2


def test_recipe_dataframe_no_batch_has_no_mass_columns():
    r = _seeded_runner()
    _add_branch(r)
    df = ui.branch_recipe_dataframe(r, "b1", **_recipe_kwargs())
    assert not any(f"({ui.MASS_UNIT})" in c for c in df.columns)


# ======================================================================
# branch_recipe_excel_bytes — валидный xlsx, читаемый обратно
# ======================================================================
def test_recipe_excel_bytes_roundtrip():
    r = _seeded_runner()
    _add_branch(r)
    data = ui.branch_recipe_excel_bytes(r, "b1", batch_kg=5.0, **_recipe_kwargs())
    assert isinstance(data, (bytes, bytearray)) and len(data) > 0

    back = pd.read_excel(io.BytesIO(data), sheet_name="Рецепт")
    assert len(back) == 1
    assert back.iloc[0]["ветка"] == "premium"
    for col in ["A", "B", "C", "T", "P", "d_overall",
                "strength (прогноз)", "gloss (прогноз)"]:
        assert col in back.columns
