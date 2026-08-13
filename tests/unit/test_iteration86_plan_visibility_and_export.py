# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 86 — аудит перед запуском кампании ПВХ (13.08.2026).

Два дефекта, найденные на живом проекте:

1. **preflight падал на пустой базе живого раннера.** У настоящего
   :class:`MixtureProcessRunner` пустая база — это ``X = None``
   (``_rebuild_arrays``), а инструмент ``readonly.preflight`` делал
   ``np.asarray(getattr(runner, "X", np.empty((0,0))))``: ``asarray(None)``
   даёт массив формы (1, 1), guard «база пуста» не срабатывал и ядро падало
   «ожидалось 22 координат на точку, дано 1». Следствие: технолог НЕ ВИДЕЛ
   диагностику предложенного плана до утверждения. Плюс: если черновик плана
   сохранён в проекте (``draft.seed_X``), preflight обязан проверять ИМЕННО
   его, а не свежесгенерированный аналог.

2. **Excel стартового плана.** Расход сырья шёл дублирующими колонками
   «{компонент} (кг)» в основном листе (при q≈18 нечитаемо), итога по
   компонентам на весь план не было, а лист «Навеска» считал граммы от веса
   замеса из паспорта, игнорируя поле «Размер пробы» — два масштаба массы в
   одном файле. Решение технолога: один масштаб — от размера пробы; расход —
   отдельным листом с итогами.
"""
import io
import json

import numpy as np
import pandas as pd
import pytest

from src.apps.campaign_ui import (batch_grams_per_phr, build_setup_runner,
                                  seed_consumption_dataframe,
                                  seed_design_excel_bytes)
from src.assistant.tools import ToolContext, dispatch
from src.design.phr_sampler import PhrSpec

PROJECT = "pvc_edge_v1"

# Референсная v2-спека (та же геометрия, что golden iter42/61).
NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "ESO", "role": "FIXED", "value": 2.5},
    {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["PBNK", "CPE"]},
    {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
     "share_range": [0.0, 0.70], "max_phr": 8.0},
    {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT", "min_phr": 3.0},
    {"name": "UV", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
     "scale": "log", "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
]


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(NODES)


# ----------------------------------------------------------------------
# 1. preflight: пустая база живого раннера (X = None) и черновик плана
# ----------------------------------------------------------------------
class _Runner:
    """Живой раннер в объёме инструмента: X=None на пустой базе (как у
    настоящего MixtureProcessRunner после _rebuild_arrays)."""

    dim = 4

    def __init__(self, spec, X=None):
        self.phr_spec = spec
        self.X = X
        self.seen = None

    def propose_seed(self, n, **kw):
        rng = np.random.default_rng(0)
        return rng.random((int(n), self.dim))

    def preflight(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != self.dim:
            raise ValueError(f"X: ожидалось {self.dim} координат на точку, "
                             f"дано {X.shape[1]}.")
        self.seen = X
        return {"passed": True, "failures": []}


def _ctx(runner, tmp_path=None):
    return ToolContext(spec=_spec(), runner=runner,
                       root=str(tmp_path) if tmp_path else "",
                       project=PROJECT if tmp_path else "")


def test_preflight_none_base_falls_back_to_seed_plan():
    """Регресс: X=None (пустая база живого раннера) — seed-план, не падение."""
    r = _Runner(_spec(), X=None)
    out = dispatch(_ctx(r), "preflight", {})
    assert out["passed"] is True
    assert "seed-план" in out["source"]
    assert r.seen.shape == (16, r.dim)          # дефолт n=16 дошёл до ядра


def test_preflight_none_base_respects_requested_n():
    r = _Runner(_spec(), X=None)
    out = dispatch(_ctx(r), "preflight", {"n": 7})
    assert r.seen.shape == (7, r.dim)
    assert "n=7" in out["source"]


def test_preflight_prefers_saved_draft_plan(tmp_path):
    """Черновик плана из campaign.json проверяется ВМЕСТО генерации нового:
    технолог должен получать диагностику ТОГО плана, что видит в UI."""
    plan = [[0.2, 0.3, 0.5, 0.1],
            [0.4, 0.4, 0.2, 0.9],
            [0.1, 0.6, 0.3, 0.5]]
    d = tmp_path / PROJECT
    d.mkdir()
    (d / "campaign.json").write_text(
        json.dumps({"format": "campaign-v1", "draft": {"seed_X": plan}}),
        encoding="utf-8")
    r = _Runner(_spec(), X=None)
    out = dispatch(_ctx(r, tmp_path), "preflight", {})
    assert "черновик" in out["source"]
    assert out["n_points"] == 3
    assert np.allclose(r.seen, np.asarray(plan, float))


def test_preflight_draft_ignores_requested_n_with_explanation(tmp_path):
    plan = [[0.2, 0.3, 0.5, 0.1], [0.4, 0.4, 0.2, 0.9]]
    d = tmp_path / PROJECT
    d.mkdir()
    (d / "campaign.json").write_text(
        json.dumps({"format": "campaign-v1", "draft": {"seed_X": plan}}),
        encoding="utf-8")
    r = _Runner(_spec(), X=None)
    out = dispatch(_ctx(r, tmp_path), "preflight", {"n": 50})
    assert out["n_points"] == 2
    assert "проигнорирован" in out["source"]


def test_preflight_nonempty_base_untouched():
    """База не пуста — считается по базе (старое поведение не тронуто)."""
    X = np.random.default_rng(1).random((5, 4))
    r = _Runner(_spec(), X=X)
    out = dispatch(_ctx(r), "preflight", {})
    assert out["source"] == "база точек проекта"
    assert np.allclose(r.seen, X)


# ----------------------------------------------------------------------
# 2. Excel стартового плана: без дублей, расход отдельным листом, итоги,
#    единый масштаб массы
# ----------------------------------------------------------------------
def _runner(spec: PhrSpec):
    return build_setup_runner(
        mixture_names=list(spec.component_names), process_names=["T"],
        process_lower=[150.0], process_upper=[200.0], response_names=["y"],
        seed=0)


def _plan(spec: PhrSpec, n: int = 3, seed: int = 0) -> np.ndarray:
    X = spec.sample_candidates(n, seed=seed)
    return np.column_stack([X, np.linspace(0.0, 1.0, n)])


def test_consumption_dataframe_rows_and_total():
    spec = _spec()
    r = _runner(spec)
    X = _plan(spec, n=3)
    df = seed_consumption_dataframe(r, X, 2.0)
    assert len(df) == 4                               # 3 опыта + «Итого»
    assert df["№ опыта"].iloc[-1] == "Итого на план"
    col = f"{spec.component_names[0]} (кг)"
    # строка опыта: кг = доля · размер пробы
    assert float(df[col].iloc[0]) == pytest.approx(
        float(X[0, 0]) * 2.0, abs=1e-3)
    # итог = сумма по опытам
    assert float(df[col].iloc[-1]) == pytest.approx(
        float(X[:, 0].sum()) * 2.0, abs=1e-3)
    # Σ строки опыта ≈ размер пробы (доли нормированы)
    assert float(df["Σ (кг)"].iloc[0]) == pytest.approx(2.0, abs=1e-2)


def test_consumption_dataframe_empty_without_batch():
    spec = _spec()
    assert seed_consumption_dataframe(_runner(spec), _plan(spec), 0.0).empty


def test_excel_no_duplicate_kg_columns_in_main_sheet():
    """Дублирующие колонки «(кг)» ушли из основного листа в «Расход сырья»."""
    spec = _spec()
    r = _runner(spec)
    X = _plan(spec, n=3)
    xf = pd.ExcelFile(io.BytesIO(
        seed_design_excel_bytes(r, X, batch_kg=2.0)))
    assert xf.sheet_names == ["Стартовый дизайн", "Расход сырья"]
    main = xf.parse("Стартовый дизайн")
    assert not any(str(c).endswith("(кг)") for c in main.columns)
    # отклики остались в основном листе
    assert "y (lab)" in main.columns
    cons = xf.parse("Расход сырья")
    assert "Итого на план" in list(cons["№ опыта"].astype(str))


def test_excel_without_batch_has_no_consumption_sheet():
    spec = _spec()
    r = _runner(spec)
    xf = pd.ExcelFile(io.BytesIO(seed_design_excel_bytes(r, _plan(spec))))
    assert xf.sheet_names == ["Стартовый дизайн"]


def test_weighing_sheet_uses_batch_scale_not_passport():
    """Единый масштаб массы: граммы «Навески» — от размера пробы, а не от
    паспортного grams_per_phr (решение технолога 13.08.2026)."""
    spec = _spec()
    r = _runner(spec)
    X = _plan(spec, n=2, seed=4)
    batch = 2.0
    data = seed_design_excel_bytes(r, X, batch_kg=batch, spec=spec,
                                   delta_phr=0.02,
                                   grams_per_phr=25.61)   # паспорт — игнор
    wdf = pd.ExcelFile(io.BytesIO(data)).parse("Навеска")
    gpp = batch_grams_per_phr(spec, batch)
    act = np.asarray(wdf["phr actual"], float)
    grams = np.asarray(wdf["граммы actual"], float)
    assert np.allclose(grams, act * gpp, atol=1e-2)
    assert not np.allclose(grams, act * 25.61, atol=1e-2)


def test_weighing_sheet_keeps_passport_scale_without_batch():
    """Без размера пробы прежнее поведение цело: масштаб — переданный gpp."""
    spec = _spec()
    r = _runner(spec)
    X = _plan(spec, n=2, seed=4)
    data = seed_design_excel_bytes(r, X, spec=spec, delta_phr=0.02,
                                   grams_per_phr=5.0)
    wdf = pd.ExcelFile(io.BytesIO(data)).parse("Навеска")
    act = np.asarray(wdf["phr actual"], float)
    assert np.allclose(np.asarray(wdf["граммы actual"], float), act * 5.0,
                       atol=1e-3)
