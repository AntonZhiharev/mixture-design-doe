# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 55 / UI_REVISION_SPEC P3.1 — КОВАРИАТЫ как столбцы базы (не Y).

Ковариата — телеметрия прогона (M(t)/SME, Die_Pressure, торк, вытяжка,
наработка вала): записывается ПРИ ТОЧКЕ, но НЕ входит в Y/суррогаты (это не
свойство продукта и не несёт желательности). До P3.1 таблица seed принимала
только «свойство (lab)» — телеметрию было некуда записать, и условия прогона
за точкой терялись безвозвратно (A0.6).

Проверяем:
  * объявление столбцов (``set_covariate_names``) — валидация имён (пустые,
    дубли, коллизии с откликами/координатами схемы);
  * запись при фиксации (``commit_seed``/``commit_measured(covariates=…)``)
    и правку постфактум (``set_point_covariates``: merge/удаление/None);
  * суррогаты НЕ переобучаются правкой ковариат («столбцы базы, не Y»);
  * персистентность: ``covariate_names`` в campaign_state; per-point значения
    едут в ``origin_tag`` точек; старый сейв без ключа грузится;
  * чистые UI-хелперы: парсер/round-trip, столбцы «(ковариата)» в таблицах
    seed/базы, редактор ковариат базы, паспорт, префилл сетапа.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps import campaign_ui as ui
from src.apps.campaign import CampaignController
from src.apps.campaign_ui import build_setup_runner
from src.optimize.desirability import DesirabilitySpec

warnings.filterwarnings("ignore", category=ConvergenceWarning)

COV_NAMES = ["SME", "Die_Pressure", "торк"]


def _runner(covariates=COV_NAMES):
    """ABC×T раннер с объявленными ковариатами (база пуста)."""
    r = build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T"],
        process_lower=[0.0], process_upper=[1.0],
        response_names=["strength", "gloss"], seed=3)
    if covariates:
        r.set_covariate_names(covariates)
    return r


def _seeded(covariates=COV_NAMES, n=6, with_covs=True):
    """Раннер с зафиксированным seed'ом; ковариаты — у первых двух точек."""
    r = _runner(covariates)
    X = np.asarray(r.propose_seed(n, seed=5), float)
    Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
    rows = None
    if with_covs and covariates:
        rows = [None] * n
        rows[0] = {"SME": 210.0, "торк": 55.5}
        rows[1] = {"Die_Pressure": 87.0}
    r.commit_seed(X, Y, covariates=rows)
    return r, X


# ======================================================================
# 1. Объявление столбцов: set_covariate_names (валидация A0.6)
# ======================================================================
class TestSetCovariateNames:

    def test_names_stored_in_order(self):
        r = _runner()
        assert r.covariate_names == COV_NAMES

    def test_none_and_empty_clear(self):
        r = _runner()
        r.set_covariate_names(None)
        assert r.covariate_names == []
        r.set_covariate_names(COV_NAMES)
        r.set_covariate_names([])
        assert r.covariate_names == []

    def test_empty_name_rejected(self):
        r = _runner(covariates=None)
        with pytest.raises(ValueError, match="Пустое имя"):
            r.set_covariate_names(["SME", "  "])

    def test_duplicate_rejected(self):
        r = _runner(covariates=None)
        with pytest.raises(ValueError, match="дважды"):
            r.set_covariate_names(["SME", "SME"])

    def test_collision_with_response_rejected(self):
        r = _runner(covariates=None)
        with pytest.raises(ValueError, match="совпадает"):
            r.set_covariate_names(["strength"])

    def test_collision_with_mixture_axis_rejected(self):
        r = _runner(covariates=None)
        with pytest.raises(ValueError, match="совпадает"):
            r.set_covariate_names(["A"])

    def test_collision_with_process_axis_rejected(self):
        r = _runner(covariates=None)
        with pytest.raises(ValueError, match="совпадает"):
            r.set_covariate_names(["T"])

    def test_clearing_names_keeps_point_values(self):
        """Снятие объявления НЕ стирает записанные значения (история И-1)."""
        r, _ = _seeded()
        r.set_covariate_names([])
        assert r.points[0].origin_tag["covariates"]["SME"] == 210.0


# ======================================================================
# 2. Запись при фиксации: commit_seed / commit_measured (covariates=…)
# ======================================================================
class TestCommitWithCovariates:

    def test_commit_seed_writes_covariates(self):
        r, _ = _seeded()
        cov0 = r.points[0].origin_tag["covariates"]
        assert cov0 == {"SME": 210.0, "торк": 55.5}
        assert r.points[1].origin_tag["covariates"] == {"Die_Pressure": 87.0}
        # строки без телеметрии — БЕЗ ключа (честное «не снято», не {})
        assert "covariates" not in r.points[2].origin_tag

    def test_commit_seed_without_covariates_unchanged(self):
        r, _ = _seeded(with_covs=False)
        assert all("covariates" not in p.origin_tag for p in r.points)

    def test_length_mismatch_rejected(self):
        r = _runner()
        X = np.asarray(r.propose_seed(4, seed=5), float)
        Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
        with pytest.raises(ValueError, match="числу точек"):
            r.commit_seed(X, Y, covariates=[{"SME": 1.0}])

    def test_unknown_name_rejected(self):
        r = _runner()
        X = np.asarray(r.propose_seed(4, seed=5), float)
        Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
        with pytest.raises(KeyError, match="не среди объявленных"):
            r.commit_seed(X, Y, covariates=[{"опечатка": 1.0}, None, None,
                                            None])

    def test_non_finite_rejected(self):
        r = _runner()
        X = np.asarray(r.propose_seed(4, seed=5), float)
        Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
        with pytest.raises(ValueError, match="не конечно"):
            r.commit_seed(X, Y, covariates=[{"SME": float("nan")},
                                            None, None, None])

    def test_covariates_without_declaration_rejected(self):
        """Без объявления столбцов запись отвергается — опечатка имени не
        должна молча создать новый столбец (A0.6)."""
        r = _runner(covariates=None)
        X = np.asarray(r.propose_seed(4, seed=5), float)
        Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
        with pytest.raises(ValueError, match="не объявлены"):
            r.commit_seed(X, Y, covariates=[{"SME": 1.0}, None, None, None])

    def test_commit_measured_writes_covariates(self):
        r, _ = _seeded()
        r.add_branch("b", {"strength": DesirabilitySpec("max", low=0.0,
                                                        high=10.0)},
                     budget=10, branch_id="b1")
        Xb = np.asarray(r.propose_points("b1", n_points=2), float)
        Yb = np.vstack([r._measure(np.asarray(x, float)) for x in Xb])
        r.commit_measured("b1", Xb, Yb,
                          covariates=[{"SME": 199.0}, None])
        pts = r.points[-len(Xb):]
        assert pts[0].origin_tag["covariates"] == {"SME": 199.0}
        assert "covariates" not in pts[1].origin_tag

    def test_controller_passthrough(self):
        r = _runner()
        ctrl = CampaignController(r)
        X = np.asarray(ctrl.propose_seed(5, seed=5), float)
        Y = np.vstack([r._measure(np.asarray(x, float)) for x in X])
        ctrl.commit_seed(X, Y, covariates=[{"SME": 1.0}] + [None] * 4)
        assert r.points[0].origin_tag["covariates"] == {"SME": 1.0}


# ======================================================================
# 3. Правка постфактум: set_point_covariates (телеметрия вносится позже)
# ======================================================================
class TestSetPointCovariates:

    def test_write_and_merge(self):
        r, _ = _seeded()
        out = r.set_point_covariates(2, {"SME": 300.0})
        assert out["changed"]["SME"] == {"old": None, "new": 300.0}
        r.set_point_covariates(2, {"торк": 44.0})
        assert r.points[2].origin_tag["covariates"] == {"SME": 300.0,
                                                        "торк": 44.0}

    def test_none_removes_value(self):
        r, _ = _seeded()
        out = r.set_point_covariates(0, {"SME": None})
        assert out["changed"]["SME"]["new"] is None
        assert "SME" not in r.points[0].origin_tag["covariates"]
        # снятие последнего значения удаляет и сам ключ covariates
        r.set_point_covariates(0, {"торк": None})
        assert "covariates" not in r.points[0].origin_tag

    def test_no_surrogate_refit(self):
        """Ковариаты — столбцы базы, НЕ Y модели: суррогаты не трогаются."""
        r, _ = _seeded()
        gp_before = r.surrogates["strength"]
        r.set_point_covariates(0, {"SME": 123.0})
        assert r.surrogates["strength"] is gp_before

    def test_coords_and_y_untouched(self):
        r, _ = _seeded()
        x_before = {k: list(v) for k, v in r.points[0].X.items()}
        y_before = dict(r.points[0].Y)
        r.set_point_covariates(0, {"SME": 500.0})
        assert {k: list(v) for k, v in r.points[0].X.items()} == x_before
        assert r.points[0].Y == y_before

    def test_index_out_of_range(self):
        r, _ = _seeded()
        with pytest.raises(IndexError):
            r.set_point_covariates(99, {"SME": 1.0})

    def test_empty_values_rejected(self):
        r, _ = _seeded()
        with pytest.raises(ValueError, match="Нет значений"):
            r.set_point_covariates(0, {})

    def test_unknown_name_rejected_including_removal(self):
        r, _ = _seeded()
        with pytest.raises(KeyError):
            r.set_point_covariates(0, {"опечатка": 1.0})
        with pytest.raises(KeyError):
            r.set_point_covariates(0, {"опечатка": None})

    def test_getters_aligned(self):
        r, _ = _seeded()
        base = r.point_covariates()
        active = r.active_point_covariates()
        assert len(base) == len(r.points) == len(active)
        assert base[0] == {"SME": 210.0, "торк": 55.5}
        # без миграций/движений области порядок совпадает с базой
        assert base == active
        # геттер отдаёт КОПИИ — правка результата не мутирует точку
        base[0]["SME"] = -1.0
        assert r.points[0].origin_tag["covariates"]["SME"] == 210.0


# ======================================================================
# 4. Персистентность (campaign_state): объявление + per-point значения
# ======================================================================
class TestPersistence:

    def test_state_carries_covariate_names(self):
        r, _ = _seeded()
        state = cst.runner_to_state(r)
        assert state["runner"]["covariate_names"] == COV_NAMES

    def test_roundtrip_names_and_values(self):
        r0, _ = _seeded()
        r1 = cst.runner_from_state(cst.runner_to_state(r0))
        assert r1.covariate_names == COV_NAMES
        assert r1.point_covariates() == r0.point_covariates()

    def test_old_save_without_key_loads_empty(self):
        r0, _ = _seeded()
        state = cst.runner_to_state(r0)
        state["runner"].pop("covariate_names")       # сейв «до P3.1»
        r1 = cst.runner_from_state(state)
        assert r1.covariate_names == []
        # значения в origin_tag точек целы (история И-1)
        assert r1.points[0].origin_tag["covariates"]["SME"] == 210.0

    def test_file_save_load_and_commit_after_load(self, tmp_path):
        r0, _ = _seeded()
        cst.save_campaign(r0, str(tmp_path), "cov")
        r1 = cst.load_campaign(str(tmp_path), "cov")
        assert r1.covariate_names == COV_NAMES
        # сквозной сценарий: добор ПОСЛЕ загрузки принимает телеметрию
        ctrl = CampaignController(r1)
        X = np.asarray(ctrl.propose_seed(3, seed=9, reuse_existing=False),
                       float)
        Y = np.vstack([r1._measure(np.asarray(x, float)) for x in X])
        ctrl.commit_seed(X, Y, covariates=[{"SME": 42.0}, None, None])
        assert r1.points[-3].origin_tag["covariates"] == {"SME": 42.0}


# ======================================================================
# 5. Чистые UI-хелперы (без Streamlit)
# ======================================================================
class TestUiHelpers:

    def test_parse_roundtrip(self):
        txt = ui.covariate_names_to_text(COV_NAMES)
        assert ui.parse_covariate_names(txt) == COV_NAMES
        assert ui.parse_covariate_names("") == []
        assert ui.parse_covariate_names("SME;\n торк ,") == ["SME", "торк"]

    def test_seed_design_dataframe_has_covariate_columns(self):
        r = _runner()
        X = np.asarray(r.propose_seed(4, seed=5), float)
        df = ui.seed_design_dataframe(r, X)
        for nm in COV_NAMES:
            assert f"{nm} (ковариата)" in df.columns
            assert df[f"{nm} (ковариата)"].isna().all()

    def test_seed_design_dataframe_without_declaration(self):
        r = _runner(covariates=None)
        X = np.asarray(r.propose_seed(4, seed=5), float)
        df = ui.seed_design_dataframe(r, X)
        assert not any(str(c).endswith("(ковариата)") for c in df.columns)

    def test_covariate_rows_from_editor(self):
        df = pd.DataFrame({
            "№ опыта": [1, 2],
            "SME (ковариата)": [210.0, np.nan],
            "торк (ковариата)": [np.nan, 44.0],
        })
        rows = ui.covariate_rows_from_editor(df, ["SME", "торк"])
        assert rows == [{"SME": 210.0}, {"торк": 44.0}]

    def test_campaign_base_dataframe_covariate_columns(self):
        r, _ = _seeded()
        df = ui.campaign_base_dataframe(r)
        assert df.loc[0, "SME (ковариата)"] == 210.0
        assert df.loc[1, "Die_Pressure (ковариата)"] == 87.0
        assert np.isnan(df.loc[2, "SME (ковариата)"])

    def test_covariates_editor_df(self):
        r, _ = _seeded()
        df = ui.covariates_editor_df(r)
        assert list(df.columns) == ["№ опыта", "источник", *COV_NAMES]
        assert df.loc[0, "SME"] == 210.0
        assert df.loc[2, "SME"] is None or pd.isna(df.loc[2, "SME"])
        assert len(df) == len(r.points)

    def test_passport_row(self):
        r = _runner()
        df = ui.campaign_passport_dataframe(r)
        row = df[df["параметр"] == "ковариаты базы (телеметрия)"]
        assert len(row) == 1
        assert row.iloc[0]["значение"] == "SME, Die_Pressure, торк"
        # без объявления — честное «—»
        df2 = ui.campaign_passport_dataframe(_runner(covariates=None))
        row2 = df2[df2["параметр"] == "ковариаты базы (телеметрия)"]
        assert row2.iloc[0]["значение"] == "—"

    def test_setup_prefill_has_covariates(self):
        r = _runner()
        pre = ui.setup_prefill_from_runner(r)
        assert pre["setup_covariates"] == "SME, Die_Pressure, торк"
        assert (ui.setup_prefill_from_runner(_runner(covariates=None))
                ["setup_covariates"] == "")