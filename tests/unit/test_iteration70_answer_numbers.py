# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 70 / ASSISTANT_SPEC — раздел «ЧИСЛА»: без дубля и ОТДЕЛЬНО в показе.

Раздел ``## ЧИСЛА`` формата ответа (iter64) введён ради трассируемости: каждое
число подписано инструментом, который его вернул, и прогон можно повторить через
полгода. На практике он выродился в дубль — модель называла те же значения в
``## ОТВЕТ`` «на всякий случай», а UI печатал ответ целиком (``st.markdown``),
поэтому раздел выглядел просто ещё одним заголовком в потоке текста, и
``ToolTurn.sections`` не использовался нигде.

Закрываются обе половины:

  * **B — промпт запрещает дублирование** (:data:`prompts.FORMAT_BLOCK`): полный
    перечень чисел живёт в ``ЧИСЛА``, в ``ОТВЕТ`` остаются одно-два решающих
    значения. Полный запрет чисел в ответе был бы хуже: при свёрнутом блоке
    вывод стал бы недоказуемым («оси неразличимы» — а насколько?);
  * **C — показ разделяет слои** (:func:`prompts.pop_section` →
    :func:`views.answer_view`): числа вырезаются из потока и рисуются свёрнутым
    блоком под ответом. Разбор ЧИСТЫЙ — виджет ничего не режет сам, поэтому
    поведение проверяется тестом, а не глазами.
"""
import inspect

from src.apps import assistant_dock as dock
from src.assistant import views
from src.assistant.prompts import (FORMAT_BLOCK, SECTIONS,
                                   architect_system_prompt, parse_sections,
                                   pop_section)

class _FakeSt:
    """Минимальная заглушка ``streamlit`` для показа артефактов.

    Нужна, чтобы проверить КЛЮЧИ виджетов, которые запросил настоящий код дока,
    без запуска Streamlit: реальный ``st.download_button`` вне сессии не
    работает, а именно на его ключах падала страница.
    """
    def __init__(self, seen_keys: list) -> None:
        self.seen_keys = seen_keys

    def download_button(self, *_args, key: str = "", **_kw) -> bool:
        self.seen_keys.append(key)
        return False

    def caption(self, *_args, **_kw) -> None:
        pass

    def image(self, *_args, **_kw) -> None:
        pass

    def dataframe(self, *_args, **_kw) -> None:
        pass

    def warning(self, *_args, **_kw) -> None:
        pass


ANSWER = ("## ОТВЕТ\n"
          "Жёсткая привязка вшивает монотонный prior: оси неразличимы "
          "(corr +0.982 против +0.154 у трапеции).\n\n"
          "## ЧИСЛА\n"
          "Источник — `simulate_bounds`, seed=20260811, n=400:\n"
          "- corr(DINP, UV) клин = +0.982;\n"
          "- corr(DINP, UV) трапеция = +0.154.\n\n"
          "## OPEN_QUESTIONS\n"
          "Верх UV — PHYSICAL или CONVENTIONAL?\n")


# ======================================================================
# B. Промпт: числа не пересказываются дважды
# ======================================================================
class TestFormatForbidsDuplication:
    def test_duplication_forbidden_in_words(self):
        # Требование должно быть НАЗВАНО, иначе модель продолжит страховаться.
        assert "НЕ пересказывай списком" in FORMAT_BLOCK
        assert "## ЧИСЛА" in FORMAT_BLOCK

    def test_decisive_numbers_still_allowed_in_answer(self):
        """Запрет — на ПЕРЕЧЕНЬ, а не на числа: иначе вывод недоказуем."""
        assert "РЕШАЮЩИХ" in FORMAT_BLOCK

    def test_traceability_required(self):
        # Смысл раздела — повторяемость расчёта, а не «красивая таблица».
        for word in ("источника-инструмента", "seed", "артефакт"):
            assert word in FORMAT_BLOCK

    def test_separate_display_declared(self):
        # Модель должна знать, что раздел показывается отдельно, — тогда
        # пересказ выше теряет смысл и для неё.
        assert "ОТДЕЛЬНО" in FORMAT_BLOCK

    def test_rules_reach_system_prompt(self):
        # Блок собирается в промпт, а не лежит мёртвой строкой.
        text = architect_system_prompt()
        assert "НЕ пересказывай списком" in text
        assert "## ЧИСЛА" in text


# ======================================================================
# C. Разбор: раздел вырезается из потока (чистая функция)
# ======================================================================
class TestPopSection:
    def test_numbers_cut_out(self):
        rest, numbers = pop_section(ANSWER, "ЧИСЛА")
        assert "simulate_bounds" in numbers and "+0.982" in numbers
        assert "## ЧИСЛА" not in rest
        assert "simulate_bounds" not in rest

    def test_neighbours_survive(self):
        """Вырезание не съедает соседние разделы — иначе пропал бы PATCH."""
        rest, _ = pop_section(ANSWER, "ЧИСЛА")
        assert set(parse_sections(rest)) == {"ОТВЕТ", "OPEN_QUESTIONS"}
        assert "PHYSICAL" in rest

    def test_missing_section_passes_through(self):
        # Старая переписка и ответы не по формату показываются как есть.
        plain = "просто текст без разделов"
        rest, numbers = pop_section(plain, "ЧИСЛА")
        assert rest == plain and numbers == ""

    def test_empty_section_dropped(self):
        # Заголовок «ЧИСЛА» без чисел хуже, чем его отсутствие.
        rest, numbers = pop_section("## ОТВЕТ\nтекст\n\n## ЧИСЛА\n", "ЧИСЛА")
        assert numbers == ""
        assert "ЧИСЛА" not in rest

    def test_numbers_is_known_section(self):
        assert "ЧИСЛА" in SECTIONS


# ======================================================================
# C. Представление показа
# ======================================================================
class TestAnswerView:
    def test_numbers_separated(self):
        v = views.answer_view(ANSWER)
        assert v.has_numbers
        assert "+0.982" in v.numbers
        assert "## ЧИСЛА" not in v.text
        # Решающее число из «ОТВЕТ» остаётся видимым без раскрытия блока.
        assert "corr +0.982" in v.text

    def test_title_is_single_wording(self):
        v = views.answer_view(ANSWER)
        assert v.numbers_title == views.NUMBERS_TITLE

    def test_without_numbers(self):
        v = views.answer_view("## ОТВЕТ\nинструменты не звались")
        assert not v.has_numbers and v.numbers == ""
        assert "инструменты не звались" in v.text

    def test_empty_text(self):
        v = views.answer_view("")
        assert v.text == "" and not v.has_numbers


# ======================================================================
# C. Док зовёт разбор, а не режет текст сам
# ======================================================================
class TestDockUsesView:
    def test_render_answer_is_thin(self):
        src = inspect.getsource(dock._render_answer)
        assert "views.answer_view" in src
        assert "st.expander" in src

    def test_both_places_go_through_render_answer(self):
        """История и свежий ход раскладываются ОДИНАКОВО.

        Иначе одно и то же сообщение выглядело бы по-разному до и после
        перезапуска приложения.
        """
        src = inspect.getsource(dock.render_assistant_dock)
        assert src.count("_render_answer(") == 2
        assert "st.markdown(res.text" not in src


# ======================================================================
# Багфикс: ключ кнопки скачивания РОНЯЛ страницу целиком
# ======================================================================
class TestDownloadKeysUnique:
    """``StreamlitDuplicateElementKey`` в ``_render_outputs`` (баг iter68).

    Один и тот же артефакт рисуется ДВАЖДЫ: в ответе хода
    (``views.turn_outputs``) и в панели «Выхлоп песочницы»
    (``views.artifact_outputs``). Ключ строился только из имени файла, поэтому
    после нескольких прогонов подряд свежий файл совпадал сам с собой и
    Streamlit валил ВСЮ страницу — дока не было видно вообще, включая ответ.
    """
    def _keys(self, monkeypatch, tmp_path, names, scope):
        """РЕАЛЬНЫЕ ключи из :func:`dock._render_outputs` (через заглушку ``st``).

        Сверять с переписанной в тесте той же f-строкой смысла нет — такой тест
        проверяет сам себя. Здесь вызывается настоящий код показа, а заглушка
        лишь записывает ключи, которые он попросил у Streamlit.
        """
        seen: list = []
        path = tmp_path / "a.csv"
        path.write_text("x,y\n1,2\n", encoding="utf-8")
        monkeypatch.setattr(dock, "st", _FakeSt(seen), raising=True)
        outputs = [views.OutputFile(name=n, kind="table", path=str(path),
                                    size=7, tool="run_python") for n in names]
        dock._render_outputs(outputs, scope=scope)
        return seen

    def test_same_file_in_two_places(self, monkeypatch, tmp_path):
        # Ровно упавший случай: свежий артефакт хода попал и в панель последних.
        same = ["20260811T125240_corr_demo.csv"]
        keys = (self._keys(monkeypatch, tmp_path, same, "turn")
                + self._keys(monkeypatch, tmp_path, same, "panel"))
        assert len(keys) == 2
        assert len(set(keys)) == 2

    def test_same_name_twice_in_one_place(self, monkeypatch, tmp_path):
        # Артефакты разных прогонов могут носить одно имя — позиция разводит.
        keys = self._keys(monkeypatch, tmp_path, ["plot.csv", "plot.csv"],
                          "panel")
        assert len(set(keys)) == 2

    def test_scope_is_required(self):
        params = inspect.signature(dock._render_outputs).parameters
        assert "scope" in params
        # keyword-only и без умолчания: место показа обязан назвать вызывающий,
        # иначе «panel по умолчанию» тихо вернёт совпадение ключей.
        assert params["scope"].kind is inspect.Parameter.KEYWORD_ONLY
        assert params["scope"].default is inspect.Parameter.empty

    def test_both_call_sites_pass_distinct_scope(self):
        turn = inspect.getsource(dock.render_assistant_dock)
        panel = inspect.getsource(dock._render_artifacts)
        assert 'scope="turn"' in turn
        assert 'scope="panel"' in panel
