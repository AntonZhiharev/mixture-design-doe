# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 97 — навеска в «Плане по партиям» + лист «Отклики» + ЕДИНИЦА МАССЫ.

Запрос технолога 14.08.2026 (живой проект «Разработка рецептов для изделий из
ПВХ», замес 10 кг, 19 компонентов):

1. **Лист «План по партиям» отдавал ДОЛИ, а не навеску.** С этого листа
   развешивают, и оператор умножал долю «0,0187» на вес замеса вручную для
   каждой позиции — ровно там и рождались ошибки. Нужна готовая цифра.

2. **Отклики мешались с нарядом.** Наряд на развеску печатают в цех, отклики
   вписывает лаборатория; в одном листе строка тянулась на ~40 столбцов, и
   колонки «(lab)» уезжали за край печатной страницы. Нужен отдельный лист в
   ТОМ ЖЕ формате «партия за партией».

3. **Единица массы была константой кода** (``campaign_ui.MASS_UNIT = "кг"``).
   При замесе 10 кг мелкие добавки (UV 0,05 phr ≈ 2 г) в килограммах после
   округления печатались нулями, а лист «Навеска» жил в граммах — в одном
   файле два масштаба записи массы. Решение: единица массы отвеса — ПАРАМЕТР
   ПРОЕКТА (``runner.batch_mass_unit``), задаётся один раз и тащится везде;
   внутри арифметика остаётся в килограммах.
"""
import io
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from src.apps import campaign_state as cs
from src.apps import campaign_ui as ui
from src.apps.campaign_ui import (campaign_base_dataframe,
                                  seed_consumption_dataframe,
                                  seed_design_dataframe,
                                  seed_design_excel_bytes,
                                  seed_plan_by_block_dataframe,
                                  seed_responses_dataframe,
                                  setup_prefill_from_runner)
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.core.mass_units import (DEFAULT_MASS_UNIT, MASS_UNITS,
                                 mass_column_label, mass_from_kg, mass_to_kg,
                                 normalize_mass_unit)
from src.core.schema import ModelSpec, ProjectSchema, VariableBlock

warnings.filterwarnings("ignore")

MIX = ["RESIN", "DINP", "UV"]


class _Oracle:
    property_names = ["strength", "gloss"]

    def evaluate(self, Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        return np.column_stack([3.0 * Xc[:, 0] + 2.0 * Xc[:, 1],
                                1.0 - Xc[:, 2]])


def _runner(n_blocks=2, seed=5, mass_unit=DEFAULT_MASS_UNIT):
    mix = VariableBlock.mixture(MIX)
    proc = VariableBlock.process(["T"], lower=[150.0], upper=[200.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    r = MixtureProcessRunner(schema, _Oracle(), seed=seed, n_restarts=2,
                             n_blocks_start=n_blocks)
    r.set_batch_mass_unit(mass_unit)
    return r


def _data_rows(grouped: pd.DataFrame) -> pd.DataFrame:
    """Только строки ОПЫТОВ (без шапок партий и итогов)."""
    return grouped[pd.to_numeric(grouped["№ опыта"], errors="coerce").notna()]


# ======================================================================
# 1. Ядро единиц массы (core/mass_units) — чистая арифметика показа
# ======================================================================
class TestMassUnitsCore:
    def test_kg_is_default_and_identity(self):
        assert DEFAULT_MASS_UNIT == "кг"
        assert mass_from_kg(2.5) == pytest.approx(2.5)
        assert mass_to_kg(2.5) == pytest.approx(2.5)

    def test_grams_and_tons(self):
        assert mass_from_kg(2.5, "г") == pytest.approx(2500.0)
        assert mass_from_kg(2500.0, "т") == pytest.approx(2.5)
        assert mass_to_kg(2500.0, "г") == pytest.approx(2.5)
        assert mass_to_kg(2.5, "т") == pytest.approx(2500.0)

    def test_roundtrip_through_kg(self):
        """Показ не теряет массу НИ В ОДНОЙ единице.

        Разрешение показа задано в килограммовом эквиваленте (см. ``_DIGITS``),
        поэтому «4 знака» не превращаются в шаг 100 г при переводе в тонны —
        иначе мелкая навеска исчезала бы просто от смены единицы.
        """
        for unit in MASS_UNITS:
            assert mass_to_kg(mass_from_kg(7.2534, unit), unit) == \
                pytest.approx(7.2534, abs=1e-7)

    def test_empty_means_default_not_error(self):
        """Пусто — это «не задано» (старый проект), а не ошибка ввода."""
        assert normalize_mass_unit("") == DEFAULT_MASS_UNIT
        assert normalize_mass_unit(None) == DEFAULT_MASS_UNIT

    def test_aliases_normalized(self):
        assert normalize_mass_unit("kg") == "кг"
        assert normalize_mass_unit(" Граммы ") == "г"
        assert normalize_mass_unit("ТОННА") == "т"

    def test_unknown_unit_is_explicit_refusal(self):
        """A0.6: молчаливый откат на кг напечатал бы не то, что просили."""
        with pytest.raises(ValueError, match="Неизвестная единица массы"):
            normalize_mass_unit("фунт")

    def test_column_label_uses_canonical_unit(self):
        assert mass_column_label("UV", "kg") == "UV (кг)"
        assert mass_column_label("UV", "г") == "UV (г)"

    def test_small_mass_survives_in_grams_but_not_in_kg(self):
        """Корень запроса: мелкая навеска в килограммах округляется в ноль,
        в граммах цифра остаётся живой."""
        kg = 0.00004                       # 0,04 г
        assert mass_from_kg(kg, "кг") == 0.0
        assert mass_from_kg(kg, "г") == pytest.approx(0.04)


# ======================================================================
# 2. Единица массы — ПАРАМЕТР ПРОЕКТА (раннер, save/load, префилл формы)
# ======================================================================
def _patch_saved_state(tmp_path, name, mutate):
    """Правка campaign.json «руками» — имитация старого/битого сейва."""
    path = tmp_path / name / "campaign.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    mutate(state["runner"])
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


class TestProjectMassUnit:
    def test_default_is_kg(self):
        assert _runner(mass_unit="").batch_mass_unit == "кг"

    def test_setter_normalizes_and_refuses_unknown(self):
        r = _runner()
        r.set_batch_mass_unit("g")
        assert r.batch_mass_unit == "г"
        with pytest.raises(ValueError):
            r.set_batch_mass_unit("пуд")

    def test_mass_unit_of_reads_project(self):
        assert ui.mass_unit_of(_runner(mass_unit="г")) == "г"

    def test_mass_unit_of_tolerates_foreign_object(self):
        """Подписи таблиц не место для падений: чужой объект → дефолт."""
        assert ui.mass_unit_of(object()) == DEFAULT_MASS_UNIT

    def test_survives_save_load(self, tmp_path):
        r = _runner(mass_unit="г")
        cs.save_campaign(r, str(tmp_path), "p97")
        assert cs.load_campaign(str(tmp_path), "p97").batch_mass_unit == "г"

    def test_legacy_save_without_key_reads_as_kg(self, tmp_path):
        """Старый проект: ключа нет ⇒ прежнее поведение (кг), без сдвига
        чисел в уже напечатанных нарядах."""
        cs.save_campaign(_runner(mass_unit="г"), str(tmp_path), "p97old")
        _patch_saved_state(tmp_path, "p97old",
                           lambda r: r.pop("batch_mass_unit"))
        assert cs.load_campaign(str(tmp_path),
                                "p97old").batch_mass_unit == "кг"

    def test_broken_value_in_file_does_not_lose_project(self, tmp_path):
        """Параметр ПОКАЗА не имеет права утащить за собой измеренную базу."""
        cs.save_campaign(_runner(mass_unit="г"), str(tmp_path), "p97bad")
        _patch_saved_state(tmp_path, "p97bad",
                           lambda r: r.update(batch_mass_unit="пуд"))
        assert cs.load_campaign(str(tmp_path),
                                "p97bad").batch_mass_unit == "кг"

    def test_prefill_returns_unit_to_form(self):
        """Иначе повторная сборка проекта молча вернула бы килограммы."""
        pre = setup_prefill_from_runner(_runner(mass_unit="г"))
        assert pre["setup_batch_mass_unit"] == "г"


# ======================================================================
# 3. Единица массы тащится ВО ВСЕ таблицы (план, расход, база)
# ======================================================================
class TestUnitEverywhere:
    def test_seed_design_columns_in_project_unit(self):
        r = _runner(mass_unit="г")
        X = r.propose_seed(6)
        df = seed_design_dataframe(r, X, batch_kg=10.0)
        for cn in MIX:
            assert f"{cn} (г)" in df.columns
            assert f"{cn} (кг)" not in df.columns
        # арифметика: доля × замес, переведённая в граммы
        assert float(df["UV (г)"].iloc[0]) == pytest.approx(
            float(X[0, MIX.index("UV")]) * 10.0 * 1000.0, abs=1e-3)

    def test_consumption_sheet_in_project_unit_with_total(self):
        r = _runner(mass_unit="г")
        df = seed_consumption_dataframe(r, r.propose_seed(4), 10.0)
        assert "Σ (г)" in df.columns
        assert df["№ опыта"].iloc[-1] == "Итого на план"
        # Σ строки опыта ≈ вес замеса в граммах (доли нормированы)
        assert float(df["Σ (г)"].iloc[0]) == pytest.approx(10_000.0, abs=1.0)

    def test_campaign_base_in_project_unit(self):
        r = _runner(mass_unit="г")
        X = r.propose_seed(4)
        r.commit_seed(X, np.vstack([r._measure(x) for x in X]))
        df = campaign_base_dataframe(r, batch_kg=10.0)
        for cn in MIX:
            assert f"{cn} (г)" in df.columns

    def test_kg_project_keeps_previous_labels(self):
        """Регресс: проект в килограммах выглядит ровно как до iter97."""
        r = _runner(mass_unit="кг")
        df = seed_design_dataframe(r, r.propose_seed(3), batch_kg=10.0)
        for cn in MIX:
            assert f"{cn} ({ui.MASS_UNIT})" in df.columns


# ======================================================================
# 4. «План по партиям» — НАВЕСКА вместо долей, без откликов
# ======================================================================
class TestPlanByBlockWeights:
    def test_components_hold_weight_not_fraction(self):
        """Навеска = ДОЛЯ ПЛАНА × вес замеса.

        Сверяемся с исходным ``X``, а не с колонкой долей другого листа: там
        доля округлена до 4 знаков (показ), и сравнение проверяло бы точность
        печати, а не арифметику навески.
        """
        r = _runner(mass_unit="г")
        X = r.propose_seed(8)
        j = MIX.index("UV")
        data = _data_rows(seed_plan_by_block_dataframe(r, X, batch_kg=10.0))
        for _, row in data.iterrows():
            frac = float(X[int(row["№ опыта"]) - 1, j])
            assert float(row["UV"]) == pytest.approx(frac * 10.0 * 1000.0,
                                                    abs=1e-3)

    def test_no_duplicate_mass_columns(self):
        """Навеска ЗАМЕНЯЕТ долю: 2×q столбцов при q≈19 нечитаемы."""
        r = _runner(mass_unit="г")
        cols = list(seed_plan_by_block_dataframe(
            r, r.propose_seed(6), batch_kg=10.0).columns)
        assert cols.count("UV") == 1
        assert not [c for c in cols if str(c).endswith("(г)")]

    def test_process_axis_stays_in_real_units(self):
        """Массой становится только состав: T остаётся 150…200 °C."""
        r = _runner(mass_unit="г")
        data = _data_rows(seed_plan_by_block_dataframe(
            r, r.propose_seed(6), batch_kg=10.0))
        assert all(150.0 <= float(v) <= 200.0 for v in data["T"])

    def test_responses_left_the_sheet(self):
        r = _runner(mass_unit="г")
        cols = list(seed_plan_by_block_dataframe(
            r, r.propose_seed(6), batch_kg=10.0).columns)
        assert not [c for c in cols if str(c).endswith("(lab)")]

    def test_without_batch_falls_back_to_fractions(self):
        """Веса замеса нет ⇒ считать навеску нечем; доли — честный отказ."""
        r = _runner(mass_unit="г")
        X = r.propose_seed(6)
        plain = seed_design_dataframe(r, X).set_index("№ опыта")
        row = _data_rows(seed_plan_by_block_dataframe(r, X)).iloc[0]
        assert float(row["UV"]) == pytest.approx(
            float(plain.loc[int(row["№ опыта"])]["UV"]))

    def test_numbers_and_grouping_preserved(self):
        """Контракт iter87 цел: те же ключи, блоки целыми группами."""
        r = _runner(mass_unit="г")
        X = r.propose_seed(10)
        plain = seed_design_dataframe(r, X)
        data = _data_rows(seed_plan_by_block_dataframe(r, X, batch_kg=10.0))
        assert {int(v) for v in data["№ опыта"]} == \
            {int(v) for v in plain["№ опыта"]}
        seq = [int(v) for v in data["Блок"]]
        assert seq == sorted(seq)


# ======================================================================
# 5. Лист «Отклики» — по образцу «Плана по партиям»
# ======================================================================
class TestResponsesSheet:
    def test_columns_are_responses_only(self):
        r = _runner()
        cols = list(seed_responses_dataframe(r, r.propose_seed(6)).columns)
        assert cols[0] == "№ опыта"
        for p in r.property_names:
            assert f"{p} (lab)" in cols
        # координат состава/процесса здесь нет: они в наряде
        for cn in MIX + ["T"]:
            assert cn not in cols

    def test_grouped_like_plan_by_block(self):
        r = _runner(n_blocks=2)
        r.block_factor = "Технолог"
        r.block_names = {1: "Драло", 2: "Казаков"}
        keys = [str(v) for v in
                seed_responses_dataframe(r, r.propose_seed(8))["№ опыта"]]
        assert any(k.startswith("Партия 1 «Драло»") for k in keys)
        assert any(k.startswith("Итого по партии 2:") for k in keys)

    def test_separator_rows_are_blank_not_nan(self):
        r = _runner(n_blocks=2)
        g = seed_responses_dataframe(r, r.propose_seed(6))
        sep = g[pd.to_numeric(g["№ опыта"], errors="coerce").isna()]
        assert len(sep) == 4                       # 2 шапки + 2 итога
        assert list(sep["strength (lab)"]) == [""] * len(sep)

    def test_numbers_match_the_plan(self):
        """Номер опыта — ключ точки: по нему лист сверяется с нарядом."""
        r = _runner(n_blocks=2)
        X = r.propose_seed(8)
        plan = _data_rows(seed_plan_by_block_dataframe(r, X, batch_kg=10.0))
        resp = _data_rows(seed_responses_dataframe(r, X))
        assert [int(v) for v in resp["№ опыта"]] == \
            [int(v) for v in plan["№ опыта"]]

    def test_empty_cells_for_manual_entry(self):
        r = _runner(n_blocks=2)
        data = _data_rows(seed_responses_dataframe(r, r.propose_seed(6)))
        assert data["strength (lab)"].isna().all()

    def test_entered_values_carried(self):
        r = _runner(n_blocks=2)
        X = r.propose_seed(6)
        Ys = np.column_stack([np.arange(len(X), dtype=float),
                              np.arange(len(X), dtype=float) + 100.0])
        data = _data_rows(seed_responses_dataframe(r, X, Ys))
        assert set(np.asarray(data["gloss (lab)"], float)) == \
            set(Ys[:, 1].ravel())

    def test_covariate_columns_present(self):
        r = _runner(n_blocks=2)
        r.set_covariate_names(["SME", "торк"])
        cols = list(seed_responses_dataframe(r, r.propose_seed(6)).columns)
        assert "SME (ковариата)" in cols and "торк (ковариата)" in cols

    def test_flat_list_without_blocking(self):
        """Одна партия: отклики нужны всё равно — лист остаётся плоским."""
        r = _runner(n_blocks=1)
        df = seed_responses_dataframe(r, r.propose_seed(6))
        assert not df.empty
        assert "Блок" not in df.columns
        assert list(df["№ опыта"]) == [1, 2, 3, 4, 5, 6]

    def test_empty_plan_gives_empty_frame(self):
        assert seed_responses_dataframe(_runner(), np.empty((0, 4))).empty


# ======================================================================
# 6. Excel: состав листов и содержимое
# ======================================================================
class TestExcelSheets:
    def _sheets(self, r, X, **kw):
        return pd.ExcelFile(io.BytesIO(seed_design_excel_bytes(r, X, **kw)))

    def test_responses_sheet_always_present(self):
        r = _runner(n_blocks=2)
        assert ui.SHEET_RESPONSES in \
            self._sheets(r, r.propose_seed(6)).sheet_names

    def test_sheet_order_with_batch(self):
        r = _runner(n_blocks=2, mass_unit="г")
        assert self._sheets(r, r.propose_seed(6),
                            batch_kg=10.0).sheet_names == [
            "Стартовый дизайн", ui.SHEET_PLAN_BY_BLOCK, ui.SHEET_RESPONSES,
            "Расход сырья"]

    def test_plan_sheet_carries_weights(self):
        r = _runner(n_blocks=2, mass_unit="г")
        sheet = self._sheets(r, r.propose_seed(6), batch_kg=10.0).parse(
            ui.SHEET_PLAN_BY_BLOCK)
        data = _data_rows(sheet)
        # навеска в граммах: сумма по компонентам ≈ вес замеса
        assert sum(float(data[cn].iloc[0]) for cn in MIX) == \
            pytest.approx(10_000.0, abs=1.0)

    def test_main_sheet_keeps_full_snapshot(self):
        """Решение 14.08.2026: «Стартовый дизайн» — полный снимок плана."""
        r = _runner(n_blocks=2, mass_unit="г")
        main = self._sheets(r, r.propose_seed(6)).parse("Стартовый дизайн")
        assert "strength (lab)" in main.columns
        for cn in MIX:
            assert cn in main.columns

    def test_responses_sheet_readable_back(self):
        r = _runner(n_blocks=2)
        sheet = self._sheets(r, r.propose_seed(6)).parse(ui.SHEET_RESPONSES)
        assert "№ опыта" in sheet.columns
        assert "gloss (lab)" in sheet.columns
        assert not [c for c in sheet.columns if c in MIX]


# ======================================================================
# 7. Черновик seed: размер пробы хранится ВМЕСТЕ с единицей
# ======================================================================
class TestSeedDraftUnit:
    """Число «размер пробы» живёт в единице ПОКАЗА. Если проект перевели из
    килограммов в граммы, восстановленное значение обязано быть ПЕРЕВЕДЕНО, а
    не подставлено как есть — иначе расход сырья уехал бы в 1000 раз."""

    def _session(self, monkeypatch, ctrl_runner, state):
        from src.apps import streamlit_app as app

        class _Ctrl:
            runner = ctrl_runner

        monkeypatch.setattr(app.st, "session_state", state, raising=False)
        state["campaign_ctrl"] = _Ctrl()
        return app

    def test_draft_carries_unit(self, monkeypatch):
        r = _runner(mass_unit="г")
        state = {"setup_seed_X": np.zeros((2, 4)), "setup_seed_batch": 500.0}
        app = self._session(monkeypatch, r, state)
        draft = app._seed_draft_from_session()
        assert draft["seed_batch"] == pytest.approx(500.0)
        assert draft["seed_batch_unit"] == "г"

    def test_restore_converts_when_unit_changed(self, monkeypatch):
        """Сохраняли 500 г, проект теперь в килограммах ⇒ поле = 0,5."""
        r = _runner(mass_unit="кг")
        state = {}
        app = self._session(monkeypatch, r, state)
        assert app._restore_seed_draft(
            {"seed_X": [[0.5, 0.3, 0.2, 0.4]], "seed_batch": 500.0,
             "seed_batch_unit": "г"}, r)
        assert state["setup_seed_batch"] == pytest.approx(0.5)

    def test_restore_keeps_value_when_unit_same(self, monkeypatch):
        r = _runner(mass_unit="г")
        state = {}
        app = self._session(monkeypatch, r, state)
        app._restore_seed_draft(
            {"seed_X": [[0.5, 0.3, 0.2, 0.4]], "seed_batch": 500.0,
             "seed_batch_unit": "г"}, r)
        assert state["setup_seed_batch"] == pytest.approx(500.0)

    def test_legacy_draft_without_unit_taken_as_is(self, monkeypatch):
        """Старый черновик: размерности не знаем — не выдумываем её."""
        r = _runner(mass_unit="г")
        state = {}
        app = self._session(monkeypatch, r, state)
        app._restore_seed_draft(
            {"seed_X": [[0.5, 0.3, 0.2, 0.4]], "seed_batch": 10.0}, r)
        assert state["setup_seed_batch"] == pytest.approx(10.0)
