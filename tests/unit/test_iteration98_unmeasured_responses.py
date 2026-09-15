# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 98 — НЕПРОВЕДЁННЫЕ измерения отклика с причиной (§13.7 MISSING).

Живая постановка технолога (15.09.2026): на лабораторном экструдере
большая часть рецептов даёт бугристую поверхность (``SurfaceQuality`` 0–5,
годны 4+); оптику (Opacity, Gloss) с такого образца не снимают. Прессовать
«любой ценой» — подмена материала (конфаундер); подставлять число (0,
«худшее») — отравленный GP (обрыв в гладком ядре) и фиктивный режим MoE
(GMM по y выделит столбик сентинелов). Решение: отклик остаётся MISSING
с ОБЯЗАТЕЛЬНОЙ причиной, измеренные отклики опыта сохраняются.

Проверяем:
  * ядро: NaN в Y → MISSING в точке, причина в ``origin_tag``; отказ без
    причины / противоречие «и число, и причина» / неизвестный отклик;
  * суррогат каждого свойства учится ТОЛЬКО на своих измеренных точках
    (n_train), свойство без измерений модели не получает, остальные — да;
  * ``measured_desirability``: точка с непроверенной целью не становится
    рекордом (d=0), полные точки — как прежде (битово);
  * ``mark_unmeasured`` ↔ ``correct_measured`` (обратные операции, причина
    появляется/исчезает);
  * персистентность: MISSING → null → MISSING, причины переживают save/load;
  * UI-хелперы (чистые): парсер причин, сборка per-point причин из
    таблицы, «н/и (причина)» в базе, отчёт, подпись покрытия;
  * промпт ассистента содержит правило про непроведённые измерения.
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps import campaign_ui as ui
from src.apps.campaign import CampaignController
from src.apps.campaign_ui import build_setup_runner
from src.apps.mixture_process_runner import (MISSING_REASONS_TAG,
                                             measured_desirability)
from src.assistant import prompts
from src.core.schema import is_missing
from src.optimize.desirability import Desirability, DesirabilitySpec

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROPS = ["strength", "gloss"]


def _runner():
    return build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T"],
        process_lower=[0.0], process_upper=[1.0],
        response_names=PROPS, seed=3)


def _seed_xy(r, n=8):
    X = np.asarray(r.propose_seed(n, seed=5), float)
    Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
    return X, Y


def _seeded_with_gaps(n=8, gaps=(0, 3)):
    """Раннер с seed'ом, где gloss не измерен у точек ``gaps`` (с причинами)."""
    r = _runner()
    X, Y = _seed_xy(r, n)
    reasons = [None] * n
    for g in gaps:
        Y[g, 1] = np.nan
        reasons[g] = {"gloss": f"образец не получен: SurfaceQuality=1 (#{g})"}
    out = r.commit_seed(X, Y, missing_reasons=reasons)
    return r, X, Y, out


# ======================================================================
# Ядро: фиксация с пропусками
# ======================================================================
class TestCommitWithGaps:
    def test_nan_becomes_missing_with_reason(self):
        r, X, Y, out = _seeded_with_gaps()
        assert out["added"] == 8 and out["n_missing"] == 2
        p0 = r.points[0]
        assert is_missing(p0.Y["gloss"])
        assert p0.Y["strength"] == pytest.approx(Y[0, 0])
        assert p0.origin_tag[MISSING_REASONS_TAG] == {
            "gloss": "образец не получен: SurfaceQuality=1 (#0)"}
        # у точки без пропусков тега причин нет
        assert MISSING_REASONS_TAG not in r.points[1].origin_tag

    def test_nan_without_reason_refused(self):
        r = _runner()
        X, Y = _seed_xy(r)
        Y[2, 1] = np.nan
        with pytest.raises(ValueError, match="причина не указана"):
            r.commit_seed(X, Y)
        with pytest.raises(ValueError, match="причина не указана"):
            r.commit_seed(X, Y, missing_reasons=[None] * 8)
        assert r.points == []                    # ничего не записано

    def test_reason_on_measured_value_is_contradiction(self):
        r = _runner()
        X, Y = _seed_xy(r)
        rows = [None] * 8
        rows[1] = {"gloss": "не измерено"}       # а gloss там измерен
        with pytest.raises(ValueError, match="и измеренное значение, и причина"):
            r.commit_seed(X, Y, missing_reasons=rows)

    def test_unknown_response_in_reasons(self):
        r = _runner()
        X, Y = _seed_xy(r)
        Y[0, 1] = np.nan
        rows = [None] * 8
        rows[0] = {"opacity": "нет образца"}
        with pytest.raises(KeyError, match="не среди свойств"):
            r.commit_seed(X, Y, missing_reasons=rows)

    def test_reasons_length_mismatch(self):
        r = _runner()
        X, Y = _seed_xy(r)
        with pytest.raises(ValueError, match="числу точек"):
            r.commit_seed(X, Y, missing_reasons=[None])

    def test_full_rows_unchanged_without_reasons(self):
        """Без NaN аргумент не нужен — прежний контракт сохранён."""
        r = _runner()
        X, Y = _seed_xy(r)
        out = r.commit_seed(X, Y)
        assert out["n_missing"] == 0
        assert all(MISSING_REASONS_TAG not in p.origin_tag for p in r.points)

    def test_commit_measured_with_gap(self):
        r, X, Y, _ = _seeded_with_gaps()
        r.add_branch("b", {"gloss": DesirabilitySpec("max", low=0, high=10)},
                     branch_id="b1")
        Xb = np.asarray(r.propose_points("b1", n_points=2), float)
        Yb = np.vstack([r._measure(np.asarray(x, float)) for x in Xb])
        Yb[0, 1] = np.nan
        with pytest.raises(ValueError, match="причина не указана"):
            r.commit_measured("b1", Xb, Yb)
        res = r.commit_measured("b1", Xb, Yb,
                                missing_reasons=[{"gloss": "бугристый"}, None])
        assert res["added"] == 2
        assert is_missing(r.points[-2].Y["gloss"])
        assert r.points[-2].origin_tag[MISSING_REASONS_TAG] == {
            "gloss": "бугристый"}


# ======================================================================
# Суррогаты: маска по свойству
# ======================================================================
class TestSurrogateMasking:
    def test_each_property_trains_on_its_measured_rows(self):
        r, X, Y, _ = _seeded_with_gaps(n=8, gaps=(0, 3))
        cov = r.surrogate_coverage()
        assert cov["strength"] == {"n_train": 8, "n_base": 8,
                                   "n_missing": 0, "fitted": True}
        assert cov["gloss"] == {"n_train": 6, "n_base": 8,
                                "n_missing": 2, "fitted": True}
        # NaN в numpy-кэше ровно там, где MISSING
        assert np.isnan(r.Y[:, 1]).sum() == 2
        assert not np.isnan(r.Y[:, 0]).any()
        # GP gloss обучен на 6 строках, strength — на 8
        assert len(r.surrogates["gloss"]._X) == 6
        assert len(r.surrogates["strength"]._X) == 8

    def test_gloss_model_equals_model_on_subset(self):
        """Маскирование эквивалентно обучению на подвыборке (тот же GP)."""
        r, X, Y, _ = _seeded_with_gaps(n=8, gaps=(0, 3))
        keep = [i for i in range(8) if i not in (0, 3)]
        r_ref = _runner()
        r_ref.commit_seed(X[keep], Y[keep])
        grid = np.asarray(r.propose_seed(5, seed=11, reuse_existing=False),
                          float)
        a = r.surrogates["gloss"].predict(grid).mean
        b = r_ref.surrogates["gloss"].predict(grid).mean
        assert np.allclose(a, b, atol=1e-6)

    def test_property_without_any_measurement_has_no_surrogate(self):
        r = _runner()
        X, Y = _seed_xy(r, 6)
        Y[:, 1] = np.nan
        r.commit_seed(X, Y,
                      missing_reasons=[{"gloss": "прибор в ремонте"}] * 6)
        assert "gloss" not in r.surrogates
        assert "strength" in r.surrogates
        assert r.surrogate_coverage()["gloss"]["fitted"] is False
        # предложение точек ветки по gloss честно отказывает
        r.add_branch("b", {"gloss": DesirabilitySpec("max", low=0, high=10)},
                     branch_id="b1")
        with pytest.raises(KeyError):
            r.propose_points("b1", n_points=1)

    def test_propose_and_optimize_work_with_gaps(self):
        r, X, Y, _ = _seeded_with_gaps()
        r.add_branch("b", {"gloss": DesirabilitySpec("max", low=0, high=10),
                           "strength": DesirabilitySpec("max", low=0,
                                                        high=10)},
                     branch_id="b1")
        Xb = r.propose_points("b1", n_points=2)
        assert Xb.shape == (2, r.dim)
        res = r.optimize_xbest("b1", n_candidates=200, refine_iters=20,
                               n_starts=2)
        assert np.isfinite(res.d_overall)


# ======================================================================
# Желательность измеренных точек
# ======================================================================
class TestMeasuredDesirability:
    SPECS = {"a": DesirabilitySpec("max", low=0.0, high=1.0),
             "b": DesirabilitySpec("min", low=0.0, high=1.0)}

    def test_complete_rows_bitwise_equal_to_desirability(self):
        a = np.array([0.2, 0.9, 0.5])
        b = np.array([0.1, 0.3, 0.8])
        d_new = measured_desirability(self.SPECS, {"a": a, "b": b})
        d_old = Desirability(self.SPECS).overall({"a": a, "b": b})
        assert np.array_equal(d_new, d_old)

    def test_nan_row_gets_zero_others_unchanged(self):
        a = np.array([0.2, np.nan, 0.5])
        b = np.array([0.1, 0.3, 0.8])
        d = measured_desirability(self.SPECS, {"a": a, "b": b})
        assert d[1] == 0.0
        full = Desirability(self.SPECS).overall(
            {"a": a[[0, 2]], "b": b[[0, 2]]})
        assert np.array_equal(d[[0, 2]], full)

    def test_all_nan_returns_zeros(self):
        d = measured_desirability(self.SPECS,
                                  {"a": np.full(3, np.nan),
                                   "b": np.zeros(3)})
        assert np.array_equal(d, np.zeros(3))

    def test_unmeasured_goal_never_becomes_d_best(self):
        r, X, Y, _ = _seeded_with_gaps()
        r.add_branch("b", {"gloss": DesirabilitySpec("max", low=0, high=10)},
                     branch_id="b1")
        Xb = np.asarray(r.propose_points("b1", n_points=2), float)
        Yb = np.vstack([r._measure(np.asarray(x, float)) for x in Xb])
        # у "лучшей" точки gloss не измерен, у второй — заведомо плохой
        Yb[0, 1] = np.nan
        Yb[1, 1] = 0.5
        res = r.commit_measured("b1", Xb, Yb,
                                missing_reasons=[{"gloss": "н/о"}, None])
        assert res["d_best"] == pytest.approx(0.05, abs=1e-9)
        assert np.allclose(res["x_best"][:r.dim], Xb[1])


# ======================================================================
# Обратные операции над зафиксированной точкой
# ======================================================================
class TestMarkAndCorrect:
    def test_mark_unmeasured_then_correct_back(self):
        r = _runner()
        X, Y = _seed_xy(r, 6)
        r.commit_seed(X, Y)
        old = float(r.points[1].Y["gloss"])
        out = r.mark_unmeasured(1, {"gloss": "оптика снята с негодного образца"})
        assert out["changed"]["gloss"]["old"] == pytest.approx(old)
        assert out["changed"]["gloss"]["new"] is None
        assert is_missing(r.points[1].Y["gloss"])
        assert r.points[1].origin_tag[MISSING_REASONS_TAG]["gloss"].startswith(
            "оптика")
        assert r.surrogate_coverage()["gloss"]["n_train"] == 5
        # обратно: correct_measured убирает причину
        out2 = r.correct_measured(1, {"gloss": 4.2})
        assert out2["changed"]["gloss"]["old"] is None
        assert r.points[1].Y["gloss"] == 4.2
        assert MISSING_REASONS_TAG not in r.points[1].origin_tag
        assert r.surrogate_coverage()["gloss"]["n_train"] == 6

    def test_mark_unmeasured_requires_reason(self):
        r = _runner()
        X, Y = _seed_xy(r, 4)
        r.commit_seed(X, Y)
        with pytest.raises(ValueError, match="пустая"):
            r.mark_unmeasured(0, {"gloss": "   "})
        with pytest.raises(ValueError, match="Нет откликов"):
            r.mark_unmeasured(0, {})
        with pytest.raises(KeyError):
            r.mark_unmeasured(0, {"opacity": "x"})
        with pytest.raises(IndexError):
            r.mark_unmeasured(9, {"gloss": "x"})

    def test_correct_measured_nan_points_to_mark_unmeasured(self):
        r = _runner()
        X, Y = _seed_xy(r, 4)
        r.commit_seed(X, Y)
        with pytest.raises(ValueError, match="mark_unmeasured"):
            r.correct_measured(0, {"gloss": float("nan")})

    def test_missing_report(self):
        r, *_ = _seeded_with_gaps(n=8, gaps=(0, 3))
        rep = r.missing_report()
        assert [(x["experiment"], x["response"]) for x in rep] == \
            [(1, "gloss"), (4, "gloss")]
        assert all(x["reason"].startswith("образец не получен") for x in rep)

    def test_controller_mark_unmeasured_rescores(self):
        r, X, Y, _ = _seeded_with_gaps(n=8, gaps=())
        ctrl = CampaignController(r)
        r.add_branch("b", {"gloss": DesirabilitySpec(
            "max", low=float(Y[:, 1].min()), high=float(Y[:, 1].max()) + 1)},
            branch_id="b1")
        ctrl._rescore("b1")
        best_idx = int(np.argmax(Y[:, 1]))
        d0 = r.branches["b1"].d_best
        assert d0 > 0
        ctrl.mark_unmeasured_point(best_idx, {"gloss": "перемер"})
        # рекорд ушёл к другой точке — d_best уменьшился
        assert r.branches["b1"].d_best < d0


# ======================================================================
# Персистентность
# ======================================================================
class TestPersistence:
    def test_missing_and_reasons_survive_roundtrip(self):
        r, X, Y, _ = _seeded_with_gaps()
        state = json.loads(json.dumps(cst.runner_to_state(r),
                                      ensure_ascii=False))
        # MISSING на диске — null, не 0
        assert state["runner"]["points"][0]["Y"]["gloss"] is None
        r2 = cst.runner_from_state(state)
        assert is_missing(r2.points[0].Y["gloss"])
        assert r2.points[0].origin_tag[MISSING_REASONS_TAG] == \
            r.points[0].origin_tag[MISSING_REASONS_TAG]
        assert r2.surrogate_coverage()["gloss"]["n_train"] == 6
        assert np.isnan(r2.Y).sum() == 2

    def test_old_save_without_reasons_loads(self):
        r = _runner()
        X, Y = _seed_xy(r, 5)
        r.commit_seed(X, Y)
        state = cst.runner_to_state(r)
        for p in state["runner"]["points"]:
            p["origin_tag"].pop(MISSING_REASONS_TAG, None)
        r2 = cst.runner_from_state(state)
        assert r2.missing_report() == []
        assert r2.active_point_missing_reasons() == [{}] * 5


# ======================================================================
# UI-хелперы (чистые)
# ======================================================================
class TestUiHelpers:
    def test_parse_missing_reasons(self):
        got = ui.parse_missing_reasons(
            "Opacity: образец не получен (SQ=1); gloss60: то же",
            ["Opacity", "Gloss60", "Adhesion"])
        assert got == {"Opacity": "образец не получен (SQ=1)",
                       "Gloss60": "то же"}
        assert ui.parse_missing_reasons("образец не получен", PROPS) == {
            "*": "образец не получен"}
        assert ui.parse_missing_reasons("", PROPS) == {}
        assert ui.parse_missing_reasons(np.nan, PROPS) == {}
        assert ui.parse_missing_reasons(None, PROPS) == {}
        # двоеточие в тексте без имени отклика — общая причина целиком
        assert ui.parse_missing_reasons("причина: прибор сломан", PROPS) == {
            "*": "причина: прибор сломан"}

    def test_missing_reason_rows_from_editor(self):
        df = pd.DataFrame({
            "strength (lab)": [1.0, np.nan, 2.0, np.nan],
            "gloss (lab)": [np.nan, np.nan, 3.0, 4.0],
            ui.MISSING_REASON_COL: ["gloss: нет образца",
                                    "экструзия не пошла",
                                    "",
                                    "strength: разрыв; gloss: лишнее"],
        })
        rows = ui.missing_reason_rows_from_editor(df, PROPS)
        assert rows[0] == {"gloss": "нет образца"}
        # общая причина раздаётся всем пустым
        assert rows[1] == {"strength": "экструзия не пошла",
                           "gloss": "экструзия не пошла"}
        assert rows[2] == {}
        # именованная причина у ИЗМЕРЕННОГО gloss оставлена — раннер откажет
        assert rows[3] == {"strength": "разрыв", "gloss": "лишнее"}

    def test_rows_from_editor_none_without_column(self):
        df = pd.DataFrame({"strength (lab)": [1.0], "gloss (lab)": [np.nan]})
        assert ui.missing_reason_rows_from_editor(df, PROPS) is None

    def test_editor_to_commit_end_to_end(self):
        r = _runner()
        X = np.asarray(r.propose_seed(4, seed=5), float)
        df = ui.seed_design_dataframe(r, X)
        assert ui.MISSING_REASON_COL in df.columns
        Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
        for j, p in enumerate(PROPS):
            df[f"{p} (lab)"] = Y[:, j]
        df.loc[2, "gloss (lab)"] = np.nan
        df.loc[2, ui.MISSING_REASON_COL] = "образец не получен (SQ=0)"
        Yc = np.column_stack([np.asarray(df[f"{p} (lab)"], float)
                              for p in PROPS])
        reasons = ui.missing_reason_rows_from_editor(df, PROPS)
        out = r.commit_seed(X, Yc, missing_reasons=reasons)
        assert out["n_missing"] == 1
        assert r.points[2].origin_tag[MISSING_REASONS_TAG] == {
            "gloss": "образец не получен (SQ=0)"}

    def test_reason_column_not_on_weighing_sheet_but_on_responses_sheet(self):
        r = _runner()
        r.n_blocks_start = 2
        X = r.propose_seed(6)
        plan = ui.seed_plan_by_block_dataframe(r, X, batch_kg=10.0)
        assert ui.MISSING_REASON_COL not in plan.columns
        resp = ui.seed_responses_dataframe(r, X)
        assert ui.MISSING_REASON_COL in resp.columns

    def test_base_dataframe_shows_unmeasured_with_reason(self):
        r, *_ = _seeded_with_gaps(n=8, gaps=(0,))
        df = ui.campaign_base_dataframe(r)
        cell = df.loc[0, "gloss (изм.)"]
        assert isinstance(cell, str) and cell.startswith(ui.UNMEASURED_LABEL)
        assert "образец не получен" in cell
        # измеренные остались числами
        assert float(df.loc[1, "gloss (изм.)"]) == pytest.approx(
            float(r.points[1].Y["gloss"]), abs=1e-4)
        # Excel собирается (смешанный столбец не ломает writer)
        assert len(ui.campaign_base_excel_bytes(r)) > 0

    def test_unmeasured_cell(self):
        assert ui.unmeasured_cell("") == ui.UNMEASURED_LABEL
        assert ui.unmeasured_cell(" x ") == f"{ui.UNMEASURED_LABEL} (x)"


    def test_missing_report_dataframe_and_caption(self):
        r, *_ = _seeded_with_gaps(n=8, gaps=(0, 3))
        df = ui.missing_report_dataframe(r)
        assert list(df.columns) == ["№ опыта", "источник", "отклик", "причина"]
        assert list(df["№ опыта"]) == [1, 4]
        cap = ui.surrogate_coverage_caption(r)
        assert "gloss: модель на 6 из 8" in cap and "2 не измерено" in cap
        # без пропусков — явная фраза о полноте
        r2 = _runner()
        X, Y = _seed_xy(r2, 4)
        r2.commit_seed(X, Y)
        assert "непроведённых измерений нет" in ui.surrogate_coverage_caption(r2)
        assert ui.missing_report_dataframe(r2).empty

    def test_caption_names_property_without_model(self):
        r = _runner()
        X, Y = _seed_xy(r, 4)
        Y[:, 1] = np.nan
        r.commit_seed(X, Y, missing_reasons=[{"gloss": "прибор"}] * 4)
        cap = ui.surrogate_coverage_caption(r)
        assert "gloss" in cap and "модели нет" in cap


# ======================================================================
# Промпт ассистента и read-only инструменты
# ======================================================================
class TestAssistantPrompt:
    def test_missing_block_in_system_prompt(self):
        txt = prompts.architect_system_prompt(project="p", has_runner=True)
        assert "НЕПРОВЕДЁННЫЕ ИЗМЕРЕНИЯ" in txt
        assert "не измерено — причина" in txt
        assert "SurfaceQuality" in txt
        # порядок: правило про данные — до формата ответа
        assert txt.index("НЕПРОВЕДЁННЫЕ ИЗМЕРЕНИЯ") < txt.index(
            prompts.FORMAT_BLOCK[:30])

    def test_get_runs_exposes_unmeasured(self):
        from src.assistant.tools import readonly
        from src.assistant.tools.registry import ToolContext
        r, *_ = _seeded_with_gaps(n=6, gaps=(1,))
        ctx = ToolContext(runner=r)
        out = readonly.get_runs(ctx, limit=10)
        assert out["unmeasured"][0]["experiment"] == 2
        assert out["surrogate_coverage"]["gloss"]["n_train"] == 5
        assert out["runs"][1]["Y"]["gloss"] is None

