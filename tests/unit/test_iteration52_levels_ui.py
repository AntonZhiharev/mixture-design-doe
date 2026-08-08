# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 52 / UI_REVISION_SPEC P2.1-UI — ДИСКРЕТНЫЕ УРОВНИ process-осей в UI.

Ядро уровней сделано в iter51 (`design/levels.py` + `runner.set_process_levels`
+ снап пула и argmax + персистентность), но ЗАДАТЬ сетку из интерфейса было
нечем и УВИДЕТЬ её после загрузки — тоже: политика «что умеет железо»
существовала только в коде. Это ровно тот случай, против которого A0.6:
план предлагает 673 об/мин, оператор ставит 900, модель учится на 673.

Шаг закрывает три дыры (всё — чистыми хелперами, канон «логика+тест, потом UI»):

  * ввод: ``parse_process_levels`` / ``process_levels_to_text`` (round-trip),
    СМЫСЛОВАЯ валидация делегируется штатному ``set_process_levels``;
  * паспорт/настройки: строка «дискретные уровни process-осей» в
    ``campaign_passport_dataframe`` + колонка «уровни» в
    ``project_settings_dataframe`` + префилл формы после загрузки;
  * план: ``seed_levels_caption`` — «все точки на уровнях» vs «⚠️ N вне сетки»
    (уровни задали ПОСЛЕ построения плана) — сигнал, а не блокировка.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps import campaign_ui as ui

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RPM_LEVELS = [400.0, 900.0]


def _runner(*, levels=None, seed: int = 3):
    """Раннер с ДВУМЯ process-осями: T (непрерывная) и rpm (дискретная)."""
    r = ui.build_setup_runner(
        mixture_names=["A", "B", "C"],
        process_names=["T", "rpm"],
        process_lower=[150.0, 400.0], process_upper=[200.0, 900.0],
        response_names=["y"], seed=seed)
    if levels is not None:
        r.set_process_levels(levels)
    return r


# ======================================================================
# 1. Парсер текста формы (СИНТАКСИС; смысл — за set_process_levels)
# ======================================================================
class TestParseProcessLevels:

    def test_basic(self):
        assert ui.parse_process_levels("rotor_rpm: 400, 900") == {
            "rotor_rpm": [400.0, 900.0]}

    def test_multiline_and_blank_lines_ignored(self):
        txt = "\nrpm: 400, 900\n\nT_zone: 170; 180; 190\n"
        assert ui.parse_process_levels(txt) == {
            "rpm": [400.0, 900.0], "T_zone": [170.0, 180.0, 190.0]}

    def test_empty_text_means_all_continuous(self):
        assert ui.parse_process_levels("") == {}
        assert ui.parse_process_levels(None) == {}

    def test_order_is_kept_as_typed(self):
        """Парсер НЕ сортирует: сортировка — часть нормализации ядра."""
        assert ui.parse_process_levels("rpm: 900, 400")["rpm"] == [900.0, 400.0]

    def test_missing_colon_is_error_with_line_number(self):
        with pytest.raises(ValueError, match="Строка 1"):
            ui.parse_process_levels("rpm 400, 900")

    def test_empty_axis_name_is_error(self):
        with pytest.raises(ValueError, match="пустое имя оси"):
            ui.parse_process_levels(": 400, 900")

    def test_non_numeric_level_is_error(self):
        with pytest.raises(ValueError, match="числа через запятую"):
            ui.parse_process_levels("rpm: 400, быстро")

    def test_axis_without_levels_is_error(self):
        # пустой список неотличим от «оси нет в сетке» — выключать надо
        # отсутствием строки (A0.6)
        with pytest.raises(ValueError, match="не задано ни одного уровня"):
            ui.parse_process_levels("rpm:")

    def test_duplicate_axis_line_is_error(self):
        with pytest.raises(ValueError, match="повторно"):
            ui.parse_process_levels("rpm: 400\nrpm: 900")


class TestLevelsText:

    def test_round_trip(self):
        levels = {"rpm": [400.0, 900.0], "T": [170.0, 180.0]}
        assert ui.parse_process_levels(
            ui.process_levels_to_text(levels)) == levels

    def test_empty_levels_give_empty_text(self):
        assert ui.process_levels_to_text({}) == ""
        assert ui.process_levels_to_text(None) == ""

    def test_text_from_runner_reparses_into_same_policy(self):
        """Текст формы → set_process_levels → тот же словарь (никаких потерь)."""
        r = _runner(levels={"rpm": RPM_LEVELS})
        r2 = _runner()
        r2.set_process_levels(
            ui.parse_process_levels(ui.process_levels_to_text(r.process_levels)))
        assert r2.process_levels == r.process_levels


# ======================================================================
# 2. Валидация — ШТАТНАЯ (единый источник правил, а не копия в UI)
# ======================================================================
class TestValidationDelegatedToRunner:

    def test_unknown_axis_rejected_by_runner(self):
        r = _runner()
        with pytest.raises(KeyError):
            r.set_process_levels(ui.parse_process_levels("нет_такой: 1, 2"))

    def test_level_outside_bounds_rejected_by_runner(self):
        r = _runner()
        with pytest.raises(ValueError):
            r.set_process_levels(ui.parse_process_levels("rpm: 400, 1200"))

    def test_runner_sorts_levels(self):
        r = _runner()
        r.set_process_levels(ui.parse_process_levels("rpm: 900, 400"))
        assert r.process_levels == {"rpm": [400.0, 900.0]}


# ======================================================================
# 3. Видимость: паспорт кампании / настройки проекта / префилл формы
# ======================================================================
class TestVisibility:

    def test_passport_row_shows_levels(self):
        df = ui.campaign_passport_dataframe(
            _runner(levels={"rpm": RPM_LEVELS})).set_index("параметр")
        val = str(df.loc["дискретные уровни process-осей", "значение"])
        assert "rpm" in val and "400" in val and "900" in val

    def test_passport_row_dash_when_continuous(self):
        df = ui.campaign_passport_dataframe(_runner()).set_index("параметр")
        assert df.loc["дискретные уровни process-осей", "значение"] == "—"

    def test_project_settings_marks_discrete_axis(self):
        df = ui.project_settings_dataframe(
            _runner(levels={"rpm": RPM_LEVELS})).set_index("переменная")
        assert df.loc["rpm", "уровни"] == "400, 900"
        # непрерывная ось помечена ЯВНО — «400…900» иначе читается как интервал
        assert df.loc["T", "уровни"] == "непрерывная"
        # у компонентов смеси правило неприменимо — пустая ячейка
        assert df.loc["A", "уровни"] == ""

    def test_prefill_contains_levels_text(self):
        out = ui.setup_prefill_from_runner(_runner(levels={"rpm": RPM_LEVELS}))
        assert out["setup_process_levels"] == "rpm: 400, 900"

    def test_prefill_empty_without_levels(self):
        assert ui.setup_prefill_from_runner(_runner())["setup_process_levels"] == ""

    def test_prefill_survives_save_load(self, tmp_path):
        """Политика уровней доживает до формы после save/load (A0.6)."""
        r0 = _runner(levels={"rpm": RPM_LEVELS})
        cst.save_campaign(r0, str(tmp_path), "lvui")
        r1 = cst.load_campaign(str(tmp_path), "lvui")
        assert (ui.setup_prefill_from_runner(r1)["setup_process_levels"]
                == ui.setup_prefill_from_runner(r0)["setup_process_levels"])


# ======================================================================
# 4. Подпись под планом: «на уровнях» vs «вне сетки»
# ======================================================================
class TestSeedLevelsCaption:

    def test_empty_when_no_levels(self):
        """Без дискретных осей подпись ПУСТА — иначе шум у каждой таблицы."""
        r = _runner()
        assert ui.seed_levels_caption(r, r.propose_seed(6)) == ""

    def test_plan_on_grid(self):
        r = _runner(levels={"rpm": RPM_LEVELS})
        txt = ui.seed_levels_caption(r, r.propose_seed(8))
        assert "rpm" in txt and "стоят на уровнях" in txt
        assert "⚠️" not in txt

    def test_plan_built_before_levels_is_flagged(self):
        """Уровни заданы ПОСЛЕ плана → предупреждение (но не блокировка)."""
        r = _runner()
        X = r.propose_seed(8)                     # непрерывный план
        r.set_process_levels({"rpm": RPM_LEVELS})
        txt = ui.seed_levels_caption(r, X)
        assert "⚠️" in txt and "ВНЕ сетки" in txt

    def test_bad_shape_does_not_crash(self):
        """Кривая форма X — подпись деградирует до перечня осей, не падает."""
        r = _runner(levels={"rpm": RPM_LEVELS})
        txt = ui.seed_levels_caption(r, np.zeros((2, 2)))
        assert "rpm" in txt
