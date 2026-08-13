# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 87 — путаница в процесс-параметрах + выгрузка плана ПО ПАРТИЯМ.

Живой отказ 13.08.2026 (проект «Разработка рецептов для изделий из ПВХ»): после
загрузки проекта поле осей показывало ``T_plast, T_head, T_mix, rotor_Hz``, а
блок границ — виджеты ``T`` = 150…200 и ``P`` = 1…5. Диагностика на AppTest
показала ДВА независимых дефекта:

1. **Границы молча терялись при смене ЧИСЛА осей.** Ключ виджета несёт ``d`` =
   число осей (``setup_plo_{d}_{i}``), поэтому добавление оси меняло
   идентичность ВСЕХ виджетов границ: Streamlit создавал их заново с дефолтом,
   введённые технологом 165…185 исчезали, а осиротевшие ключи ``setup_plo_2_*``
   оставались в ``session_state`` и уезжали в ``setup_draft.json``.
   Правка: ключ по ИМЕНИ оси + перевод старого формата при префилле.

2. **Позиционные дефолты выдавали бессмысленные диапазоны.** ``{0: (150,200),
   1: (1,5)}`` — вторая ось получала «1…5» независимо от смысла; в ПВХ там
   ``T_head``, и «T_head = 1…5 °C» собиралось МОЛЧА (``VariableBlock`` проверяет
   лишь ``lower <= upper``). Правка: дефолтов нет, вырожденная ось — явный отказ
   сборки (A0.6).

Плюс запрос технолога по выгрузке (та же сессия): план уезжал в Excel в порядке
генерации, а ставится он ПАРТИЯМИ. Нужен лист, где строки разложены по блокам,
но номера опытов СОХРАНЕНЫ — это ключи точек в общей базе.
"""
import io
import warnings

import numpy as np
import pandas as pd
import pytest

from src.apps import campaign_state as cs
from src.apps import campaign_ui as ui
from src.apps.campaign_ui import (build_setup_runner,
                                  legacy_process_bound_keys,
                                  migrate_process_bound_fields,
                                  process_bound_keys,
                                  seed_design_excel_bytes,
                                  seed_plan_by_block_dataframe,
                                  setup_prefill_from_runner,
                                  unknown_process_axis_names)
from src.core.schema import ModelSpec, ProjectSchema, VariableBlock
from src.apps.mixture_process_runner import MixtureProcessRunner

warnings.filterwarnings("ignore")

#: Оси живого ПВХ-проекта — на них и разъехалась форма.
AXES = ["T_plast", "T_head", "T_mix", "rotor_Hz"]
LO = [165.0, 140.0, 95.0, 40.0]
HI = [185.0, 180.0, 125.0, 60.0]


def _runner(names=AXES, lower=LO, upper=HI, seed=0):
    return build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=list(names),
        process_lower=list(lower), process_upper=list(upper),
        response_names=["y"], seed=seed)


# ======================================================================
# 1. Ключи границ — по ИМЕНИ оси, а не по её позиции и числу осей
# ======================================================================
class TestBoundKeysByName:
    def test_key_depends_on_name_only(self):
        assert process_bound_keys("T_plast") == ("setup_plo_T_plast",
                                                 "setup_phi_T_plast")

    def test_key_prefix_respected(self):
        assert process_bound_keys("T", key_prefix="edit") == ("edit_plo_T",
                                                             "edit_phi_T")

    def test_legacy_keys_are_positional(self):
        assert legacy_process_bound_keys(1, 4) == ("setup_plo_4_1",
                                                   "setup_phi_4_1")

    def test_prefill_writes_keys_by_name(self):
        pre = setup_prefill_from_runner(_runner())
        for nm, lo, hi in zip(AXES, LO, HI):
            k_lo, k_hi = process_bound_keys(nm)
            assert pre[k_lo] == pytest.approx(lo)
            assert pre[k_hi] == pytest.approx(hi)
        # позиционных ключей больше НЕ пишем
        assert not [k for k in pre if k.startswith(("setup_plo_4_",
                                                    "setup_phi_4_"))]

    def test_prefill_roundtrip_through_save_load(self, tmp_path):
        r0 = _runner()
        cs.save_campaign(r0, str(tmp_path), "p87")
        r1 = cs.load_campaign(str(tmp_path), "p87")
        assert setup_prefill_from_runner(r1) == setup_prefill_from_runner(r0)


# ======================================================================
# 2. Перевод СТАРЫХ (позиционных) черновиков — загрузка не теряет границы
# ======================================================================
class TestLegacyMigration:
    def test_positional_keys_translated_by_axis_order(self):
        legacy = {"setup_proc": ", ".join(AXES)}
        for i, (lo, hi) in enumerate(zip(LO, HI)):
            k_lo, k_hi = legacy_process_bound_keys(i, len(AXES))
            legacy[k_lo], legacy[k_hi] = lo, hi

        out = migrate_process_bound_fields(legacy)
        for nm, lo, hi in zip(AXES, LO, HI):
            k_lo, k_hi = process_bound_keys(nm)
            assert out[k_lo] == pytest.approx(lo)
            assert out[k_hi] == pytest.approx(hi)
        # позиционные ключи не протекают в состояние формы
        assert not [k for k in out if k.startswith(("setup_plo_4_",
                                                    "setup_phi_4_"))]

    def test_other_fields_untouched(self):
        out = migrate_process_bound_fields(
            {"setup_proc": "T", "setup_mix": "A, B", "setup_seed": 7,
             "setup_plo_1_0": 10.0, "setup_phi_1_0": 20.0})
        assert out["setup_mix"] == "A, B" and out["setup_seed"] == 7
        assert out["setup_plo_T"] == 10.0 and out["setup_phi_T"] == 20.0

    def test_named_keys_win_over_positional(self):
        """Новый формат приоритетен — старый дубль не перетирает его."""
        out = migrate_process_bound_fields(
            {"setup_proc": "T", "setup_plo_T": 165.0, "setup_plo_1_0": 1.0})
        assert out["setup_plo_T"] == 165.0

    def test_stale_positional_keys_for_other_axis_count_dropped(self):
        """Ключи под ДРУГОЕ число осей — чужие значения, их нельзя переносить.

        Ровно этот мусор жил в живом ``session_state``: ``setup_plo_2_*`` от
        прежних двух осей рядом с четырьмя актуальными.
        """
        out = migrate_process_bound_fields(
            {"setup_proc": ", ".join(AXES),
             "setup_plo_2_0": 150.0, "setup_phi_2_0": 200.0})
        assert "setup_plo_2_0" not in out and "setup_phi_2_0" not in out
        assert "setup_plo_T_plast" not in out          # выдумывать не стали

    def test_input_not_mutated(self):
        src = {"setup_proc": "T", "setup_plo_1_0": 5.0}
        migrate_process_bound_fields(src)
        assert src == {"setup_proc": "T", "setup_plo_1_0": 5.0}

    def test_real_pvc_draft_migrates(self):
        """Черновик живого проекта (4 оси) переводится без потерь."""
        draft = {"setup_proc": ", ".join(AXES),
                 "setup_plo_4_0": 165.0, "setup_phi_4_0": 185.0,
                 "setup_plo_4_1": 140.0, "setup_phi_4_1": 180.0,
                 "setup_plo_4_2": 95.0, "setup_phi_4_2": 125.0,
                 "setup_plo_4_3": 40.0, "setup_phi_4_3": 60.0}
        out = migrate_process_bound_fields(draft)
        assert [out[process_bound_keys(n)[0]] for n in AXES] == LO
        assert [out[process_bound_keys(n)[1]] for n in AXES] == HI


# ======================================================================
# 3. Вырожденные границы — ЯВНЫЙ отказ сборки (а не «T_head = 1…5 °C»)
# ======================================================================
class TestDegenerateBoundsRefused:
    def test_zero_range_refused(self):
        with pytest.raises(ValueError, match="Границы процесс-осей не заданы"):
            _runner(names=["T_plast", "T_head"], lower=[165.0, 0.0],
                    upper=[185.0, 0.0])

    def test_message_names_the_axis(self):
        with pytest.raises(ValueError, match="T_head"):
            _runner(names=["T_plast", "T_head"], lower=[165.0, 5.0],
                    upper=[185.0, 5.0])

    def test_inverted_range_refused(self):
        with pytest.raises(ValueError, match="Границы процесс-осей не заданы"):
            _runner(names=["T"], lower=[200.0], upper=[150.0])

    def test_nan_bound_refused(self):
        with pytest.raises(ValueError, match="Границы процесс-осей не заданы"):
            _runner(names=["T"], lower=[float("nan")], upper=[150.0])

    def test_defaults_are_keyed_by_name_not_position(self):
        """Дефолт получает только ШАБЛОННОЕ имя формы, а не «вторая ось».

        Ровно здесь был дефект: позиционная таблица выдавала оси №2 «1…5»,
        и в ПВХ-проекте это доставалось `T_head`.
        """
        assert ui.PROCESS_BOUND_DEFAULTS["T"] == (150.0, 200.0)
        assert ui.PROCESS_BOUND_DEFAULTS["P"] == (1.0, 5.0)
        for nm in AXES:
            assert nm not in ui.PROCESS_BOUND_DEFAULTS

    def test_valid_bounds_still_build(self):
        r = _runner()
        assert list(r.current_schema.process_names) == AXES
        pb = r.current_schema.process_block()
        assert list(pb.lower) == LO and list(pb.upper) == HI


# ======================================================================
# 4. Рассинхрон имён: уровни/связки ссылаются на несуществующую ось
# ======================================================================
class TestUnknownAxisNames:
    def test_unknown_level_axis_detected(self):
        assert unknown_process_axis_names(["T_plast"], "T_mix: 95, 110",
                                          "") == ["T_mix"]

    def test_unknown_link_axes_detected(self):
        out = unknown_process_axis_names(
            ["T_plast"], "", "dT: T_head - T_plast : -35, 5")
        assert out == ["T_head"]

    def test_known_axes_pass(self):
        assert unknown_process_axis_names(
            AXES, "T_mix: 95, 110", "dT: T_head - T_plast : -35, 5") == []

    def test_broken_syntax_is_not_reported_as_unknown(self):
        """Синтаксис — не наша забота: сообщения дают штатные парсеры."""
        assert unknown_process_axis_names(AXES, "мусор без двоеточия", "") == []

    def test_duplicates_collapsed(self):
        out = unknown_process_axis_names([], "X: 1, 2", "d: X - X : 0, 1")
        assert out == ["X"]


# ======================================================================
# 5. Excel: лист «План по партиям» — группировка по блокам, номера СОХРАНЕНЫ
# ======================================================================
class _Oracle:
    property_names = ["y1"]

    def evaluate(self, Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        return (3.0 * Xc[:, 0] + 2.0 * Xc[:, 1]).reshape(-1, 1)


def _blocked_runner(n_blocks=2, seed=5):
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T"], lower=[150.0], upper=[200.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    return MixtureProcessRunner(schema, _Oracle(), seed=seed, n_restarts=2,
                                n_blocks_start=n_blocks)


def _data_rows(grouped: pd.DataFrame) -> pd.DataFrame:
    """Только строки ОПЫТОВ (без заголовков партий и итогов)."""
    return grouped[pd.to_numeric(grouped["№ опыта"], errors="coerce").notna()]


class TestPlanByBlockSheet:
    def test_rows_grouped_by_block_and_numbers_preserved(self):
        r = _blocked_runner(n_blocks=2)
        X = r.propose_seed(10)
        plain = ui.seed_design_dataframe(r, X)
        data = _data_rows(seed_plan_by_block_dataframe(r, X))

        # 1) номера опытов — ТЕ ЖЕ ключи, ни одна точка не потеряна
        assert {int(v) for v in data["№ опыта"]} == \
            {int(v) for v in plain["№ опыта"]}
        assert len(data) == len(plain)

        # 2) блоки идут ЦЕЛЫМИ группами (не чередуются)
        seq = [int(v) for v in data["Блок"]]
        assert seq == sorted(seq)

        # 3) внутри блока номера возрастают
        for b in set(seq):
            part = [int(n) for n, bb in zip(data["№ опыта"], data["Блок"])
                    if int(bb) == b]
            assert part == sorted(part)

    def test_coordinates_follow_their_experiment_number(self):
        """Строка не «съезжает»: координаты остаются при своём номере."""
        r = _blocked_runner(n_blocks=2)
        X = r.propose_seed(8)
        plain = ui.seed_design_dataframe(r, X).set_index("№ опыта")
        data = _data_rows(seed_plan_by_block_dataframe(r, X))
        for _, row in data.iterrows():
            src = plain.loc[int(row["№ опыта"])]
            assert float(row["A"]) == pytest.approx(float(src["A"]))
            assert float(row["T"]) == pytest.approx(float(src["T"]))

    def test_header_and_total_rows_per_block(self):
        r = _blocked_runner(n_blocks=2)
        r.block_factor = "Технолог"
        r.block_names = {1: "Драло", 2: "Казаков"}
        keys = [str(v) for v in
                seed_plan_by_block_dataframe(r, r.propose_seed(8))["№ опыта"]]
        assert any(k.startswith("Партия 1 «Драло»") for k in keys)
        assert any(k.startswith("Партия 2 «Казаков»") for k in keys)
        assert any(k.startswith("Итого по партии 1:") for k in keys)

    def test_header_without_block_names_uses_number(self):
        r = _blocked_runner(n_blocks=2)
        keys = [str(v) for v in
                seed_plan_by_block_dataframe(r, r.propose_seed(6))["№ опыта"]]
        assert any(k.startswith("Партия 1 —") for k in keys)

    def test_separator_rows_are_blank_not_nan(self):
        """В шапке партии и итогах — ПУСТО, а не «NaN» (оператор читает лист).

        Пустая строка в числовой колонке становится NaN, и Excel показывал бы
        «NaN» в разделителях — поэтому лист приводится к object.
        """
        r = _blocked_runner(n_blocks=2)
        g = seed_plan_by_block_dataframe(r, r.propose_seed(6))
        sep = g[pd.to_numeric(g["№ опыта"], errors="coerce").isna()]
        assert len(sep) == 4                      # 2 шапки + 2 итога
        for col in ("Блок", "A", "T"):
            assert list(sep[col]) == [""] * len(sep)

    def test_same_columns_as_main_sheet(self):
        r = _blocked_runner(n_blocks=2)
        X = r.propose_seed(6)
        assert list(seed_plan_by_block_dataframe(r, X).columns) == \
            list(ui.seed_design_dataframe(r, X).columns)

    def test_empty_without_blocking(self):
        r = _blocked_runner(n_blocks=1)
        assert seed_plan_by_block_dataframe(r, r.propose_seed(6)).empty

    def test_empty_plan_gives_empty_frame(self):
        r = _blocked_runner(n_blocks=2)
        assert seed_plan_by_block_dataframe(r, np.empty((0, 4))).empty

    def test_excel_has_sheet_when_blocking_on(self):
        r = _blocked_runner(n_blocks=2)
        xf = pd.ExcelFile(io.BytesIO(
            seed_design_excel_bytes(r, r.propose_seed(8))))
        assert ui.SHEET_PLAN_BY_BLOCK in xf.sheet_names
        sheet = xf.parse(ui.SHEET_PLAN_BY_BLOCK)
        assert "Блок" in sheet.columns and "№ опыта" in sheet.columns

    def test_excel_without_blocking_has_no_sheet(self):
        r = _blocked_runner(n_blocks=1)
        xf = pd.ExcelFile(io.BytesIO(
            seed_design_excel_bytes(r, r.propose_seed(6))))
        assert ui.SHEET_PLAN_BY_BLOCK not in xf.sheet_names

    def test_main_sheet_order_unchanged(self):
        """Основной лист не тронут — новая группировка ДОПОЛНЯЕТ, не заменяет."""
        r = _blocked_runner(n_blocks=2)
        X = r.propose_seed(8)
        main = pd.ExcelFile(io.BytesIO(
            seed_design_excel_bytes(r, X))).parse("Стартовый дизайн")
        assert list(main["№ опыта"]) == \
            list(ui.seed_design_dataframe(r, X)["№ опыта"])

    def test_responses_carried_into_grouped_sheet(self):
        """Внесённые в редакторе отклики уезжают и в лист по партиям."""
        r = _blocked_runner(n_blocks=2)
        X = r.propose_seed(6)
        Ys = np.arange(len(X), dtype=float).reshape(-1, 1)
        data = _data_rows(seed_plan_by_block_dataframe(r, X, Ys))
        assert set(np.asarray(data["y1 (lab)"], float)) == set(Ys.ravel())
