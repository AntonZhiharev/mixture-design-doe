# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 24 — §17.2.1: коррекция ошибки ВВОДА измеренных Y (правка опечатки).

Правка технической ошибки внесения отклика, а НЕ переписывание истории (И-1):
координаты состава/процесса, origin-тег и порядок точек («№ опыта») сохраняются;
меняется лишь ошибочно внесённое значение, суррогаты переобучаются, ветки
переоцениваются. Две части (как у существующих кампания-тестов):

  * ЯДРО (:meth:`MixtureProcessRunner.correct_measured`) — правка значения,
    сохранность координат/origin/n_base, явные отказы A0.6;
  * КОНТРОЛЛЕР (:meth:`CampaignController.correct_measured_point`) + чистый
    UI-хелпер (:func:`campaign_ui.measured_responses_editor_df`): переоценка
    веток и запечатывание undo, форма таблицы-редактора.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign as cv
from src.apps import campaign_ui as ui


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _runner():
    # маленькая демо-кампания: общий пул + 2 ветки (premium/rho_focus)
    return ui.build_demo_campaign_runner(seed=3, n_seed=12)


# ======================================================================
# ЯДРО: correct_measured — правка значения, сохранность истории (И-1)
# ======================================================================
def test_correct_measured_changes_only_named_response():
    r = _runner()
    pt = r.points[0]
    coords_before = {k: list(v) for k, v in pt.X.items()}
    origin_before = dict(pt.origin_tag)
    gloss_before = float(pt.Y["gloss"])
    n_before = len(r.points)

    out = r.correct_measured(0, {"strength": 42.0})

    assert float(r.points[0].Y["strength"]) == 42.0        # исправлено
    assert float(r.points[0].Y["gloss"]) == gloss_before   # прочие Y нетронуты
    assert {k: list(v) for k, v in r.points[0].X.items()} == coords_before
    assert dict(r.points[0].origin_tag) == origin_before   # origin/версия целы
    assert len(r.points) == n_before                       # база не урезана (И-1)
    assert out["changed"]["strength"]["new"] == 42.0
    assert out["n_base"] == n_before


def test_correct_measured_refits_surrogates():
    r = _runner()
    # правка меняет обученный суррогат strength (переобучение на новой правде).
    # Координаты первой точки берём из составной матрицы (r.X), не из ключей X.
    x = np.asarray(r.X[0], float).reshape(1, -1)
    before = float(r.surrogates["strength"].predict(x).mean[0])
    r.correct_measured(0, {"strength": float(r.points[0].Y["strength"]) + 50.0})
    after = float(r.surrogates["strength"].predict(x).mean[0])
    assert not np.isclose(before, after)


@pytest.mark.parametrize("idx", [-1, 999])
def test_correct_measured_bad_index(idx):
    r = _runner()
    with pytest.raises(IndexError):
        r.correct_measured(idx, {"strength": 1.0})


def test_correct_measured_unknown_response():
    r = _runner()
    with pytest.raises(KeyError):
        r.correct_measured(0, {"нет_такого": 1.0})


def test_correct_measured_empty_and_nonfinite():
    r = _runner()
    with pytest.raises(ValueError):
        r.correct_measured(0, {})
    with pytest.raises(ValueError):
        r.correct_measured(0, {"strength": float("nan")})


# ======================================================================
# UI-хелпер: таблица-редактор измеренных откликов (чистая, без Streamlit)
# ======================================================================
def test_measured_responses_editor_df_shape():
    r = _runner()
    df = ui.measured_responses_editor_df(r)
    assert len(df) == len(r.points)
    for col in ("№ опыта", "источник", "strength", "gloss", "rho"):
        assert col in df.columns
    # «№ опыта» = 1-based позиция в общей базе; редактор эталон = текущее Y
    assert list(df["№ опыта"]) == list(range(1, len(r.points) + 1))
    assert np.isclose(float(df.iloc[0]["strength"]),
                      float(r.points[0].Y["strength"]))


# ======================================================================
# КОНТРОЛЛЕР: переоценка веток + запечатывание undo (Тр-7.2/7.3)
# ======================================================================
def test_controller_correct_rescores_and_seals_undo():
    ctrl = cv.CampaignController(_runner())
    bid = next(iter(ctrl.runner.branches))
    # накопим обратимую мутацию (есть что откатывать)
    ctrl.set_weights(bid, {list(ctrl.runner.branches[bid].goal)[0]: 2.0})
    assert ctrl.can_undo()

    out = ctrl.correct_measured_point(0, {"strength": 99.0})

    assert float(ctrl.runner.points[0].Y["strength"]) == 99.0
    assert out["changed"]["strength"]["new"] == 99.0
    # изменение измеренной правды — веха: дно undo запечатано (И-1/Тр-7.2)
    assert not ctrl.can_undo()
