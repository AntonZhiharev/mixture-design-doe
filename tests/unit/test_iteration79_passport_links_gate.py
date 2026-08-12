# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 79 — связки осей в паспорте пакета: гейт на dry-run, а не на кнопке.

Живой отказ (сессия 12.08.2026): человек нажал «✅ Принять проект» и получил
трассировку на весь экран —

    PackageError: passport.process_links[] должен быть объектом JSON,
    получено str

Три дефекта в одном:

  1. `parse_passport` проверял только ИМЕНА ключей паспорта, поэтому связка,
     записанная СТРОКОЙ формы («dT: A - B : 10, 60»), проходила dry-run и
     стейдж, а падала внутри `package_to_setup_prefill` — то есть уже ПОСЛЕ
     утверждения человеком. Ровно то, что iter76 закрыл для `preflight_pairs`;
  2. в `apply_project` разовый токен гасился ДО проекции пакета в поля формы:
     сбой проекции сжигал подтверждение;
  3. `PackageError` из проекции не переводился в `ToolError`, а UI ловит
     только его — вместо `st.error` падала вся страница.
"""
import pytest

from src.design.project_package import (PackageError, package_to_setup_prefill,
                                        normalize_process_links,
                                        parse_project_package)

SPEC = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
]
PROCESS = [
    {"name": "T_plast", "range": [165, 185], "unit": "°C"},
    {"name": "T_adapter", "range": [170, 200], "unit": "°C"},
]
CANON = {"name": "dT_head", "minuend": "T_adapter", "subtrahend": "T_plast",
         "lo": 10.0, "hi": 60.0}


def _pkg(links):
    return {"package_kind": "project", "spec": list(SPEC),
            "responses": [{"name": "gloss", "unit": "%"}],
            "process": [dict(p) for p in PROCESS],
            "passport": {"process_links": list(links)}}


# ======================================================================
# 1. Тот самый отказ: связка СТРОКОЙ больше не роняет применение
# ======================================================================
def test_link_as_form_text_is_accepted_and_canonised():
    """Строка формы — законная запись: её пишет и сам интерфейс."""
    pkg = parse_project_package(_pkg(["dT_head: T_adapter - T_plast : 10, 60"]))
    assert pkg.passport["process_links"] == [CANON]


def test_prefill_of_text_link_does_not_raise():
    """Ровно та точка, где падало приложение (package_to_setup_prefill)."""
    pkg = parse_project_package(_pkg(["dT_head: T_adapter - T_plast : 10, 60"]))
    out = package_to_setup_prefill(pkg)
    assert out["setup_process_links"] == "dT_head: T_adapter - T_plast : 10, 60"


def test_prefill_accepts_raw_text_link_without_parse():
    """Защита на самом `_links_text`: строка не роняет проекцию и напрямую."""
    from src.design.project_package import _links_text

    assert _links_text(["dT_head: T_adapter - T_plast : 10, 60"]) \
        == "dT_head: T_adapter - T_plast : 10, 60"


# ======================================================================
# 2. Формы записи объекта
# ======================================================================
def test_object_canon_passes():
    pkg = parse_project_package(_pkg([dict(CANON)]))
    assert pkg.passport["process_links"] == [CANON]


def test_object_synonyms_are_normalised():
    """Модель часто пишет left/right и min/max — принимаем, приводя к канону."""
    pkg = parse_project_package(_pkg([
        {"name": "dT_head", "left": "T_adapter", "right": "T_plast",
         "min": 10, "max": 60}]))
    assert pkg.passport["process_links"] == [CANON]


@pytest.mark.parametrize("text,lo,hi", [
    ("dT_head: T_adapter - T_plast : *, 60", None, 60.0),
    ("dT_head: T_adapter - T_plast : 10, *", 10.0, None),
])
def test_open_band_side_is_star(text, lo, hi):
    """Открытая сторона полосы — «*», а не пропуск ключа."""
    got = parse_project_package(_pkg([text])).passport["process_links"][0]
    assert (got["lo"], got["hi"]) == (lo, hi)


def test_round_trip_text_object_text():
    """Строка → канон → текст формы: значения не теряются и не плывут."""
    pkg = parse_project_package(_pkg(["dT_head: T_adapter - T_plast : *, 60"]))
    assert package_to_setup_prefill(pkg)["setup_process_links"] \
        == "dT_head: T_adapter - T_plast : *, 60"


# ======================================================================
# 3. Гейт: неверная связка отклоняется ДО утверждения человеком
# ======================================================================
@pytest.mark.parametrize("links,needle", [
    (["dT_head: T_nope - T_plast : 10, 60"], "не найдена"),
    (["dT_head: T_adapter - T_plast : 500, 600"], "не пересекает"),
    (["T_plast: T_adapter - T_plast : 10, 60"], "совпадает с именем"),
    (["dT_head: T_adapter - T_adapter : 10, 60"], "совпадают"),
    (["dT_head = T_adapter - T_plast"], "не задана ПОЛОСА"),
    (["dT_head: T_adapter - T_plast : abc, 60"], "не число"),
    ([12345], "объектом JSON"),
    ([{"name": "dT_head", "minuend": "T_adapter"}], "нужны"),
])
def test_bad_link_is_rejected_by_dry_run(links, needle):
    """Отказ обязан случиться на разборе пакета, а не на кнопке применения."""
    with pytest.raises(PackageError) as exc:
        parse_project_package(_pkg(links))
    assert needle in str(exc.value)


def test_rejection_names_available_axes():
    """Сообщение должно показывать, ЧТО допустимо — иначе правка наугад."""
    with pytest.raises(PackageError) as exc:
        parse_project_package(_pkg(["dT_head: T_nope - T_plast : 10, 60"]))
    text = str(exc.value)
    assert "T_plast" in text and "T_adapter" in text


@pytest.mark.parametrize("text", [
    "dT_head = T_adapter - T_plast : 10, 35",
    "dT_head = T_adapter - T_plast : *, 35",
])
def test_equals_sign_is_accepted_as_synonym(text):
    """«=» на месте первого разделителя — живая запись модели, ход не теряем."""
    got = parse_project_package(_pkg([text])).passport["process_links"][0]
    assert (got["name"], got["minuend"], got["subtrahend"]) \
        == ("dT_head", "T_adapter", "T_plast")


def test_link_without_band_names_the_real_reason():
    """Отказ 12.08.2026 №2: причина — НЕТ ПОЛОСЫ, а не «неверный синтаксис».

    Разность названа, пределы железа — нет. Придумать их за технолога нельзя
    (A0.6), поэтому сообщение просит дописать полосу, а не «исправить формат».
    """
    with pytest.raises(PackageError) as exc:
        parse_project_package(_pkg(["dT_head = T_adapter - T_plast"]))
    text = str(exc.value)
    assert "не задана ПОЛОСА" in text
    assert "10, *" in text                    # показан пример открытой стороны
    assert "выдумывать" in text               # и почему числа назвать вам


def test_schema_documents_passport_value_formats():
    """Корень отказа: формат ЗНАЧЕНИЙ паспорта обязан приходить из схемы.

    Пока схема описывала только имена ключей, модель восстанавливала синтаксис
    связки по памяти — и теряла ход на гейте.
    """
    from src.design.project_package import project_package_schema

    vals = project_package_schema(include_example=False)["blocks"]["passport"]
    assert "значения" in vals
    links_doc = vals["значения"]["process_links"]
    for needle in ("minuend", "subtrahend", "lo", "hi", "РЕАЛЬНЫХ"):
        assert needle in links_doc
    # Каждый ключ паспорта документирован — иначе следующий промах повторится.
    from src.design.project_package import PASSPORT_KEYS

    assert set(vals["значения"]) == set(PASSPORT_KEYS)


def test_normalize_process_links_is_the_single_source_of_rules():
    """Правила берутся у ядра linked_axes, а не дублируются в пакете."""
    got = normalize_process_links(
        ["dT_head: T_adapter - T_plast : 10, 60"], PROCESS)
    assert got == [CANON]
    with pytest.raises(PackageError):
        normalize_process_links(["dT_head: T_adapter - T_plast : 10, 5"],
                                PROCESS)


def test_empty_links_keep_passport_untouched():
    """Пустой список связок — не ошибка: связок просто нет."""
    pkg = parse_project_package(_pkg([]))
    assert not pkg.passport.get("process_links")
    assert "setup_process_links" not in package_to_setup_prefill(pkg)


# ======================================================================
# 4. Применение человеком: связка строкой доходит до полей формы
# ======================================================================
class TestHumanApplyWithLinks:
    """Полный путь кнопки «✅ Принять проект» на пакете со связкой-строкой."""

    def _ctx(self, tmp_path):
        from src.assistant.consent import ConsentRegistry
        from src.assistant.session import new_session
        from src.assistant.tools import ToolContext

        return ToolContext(spec=None, runner=None,
                           session=new_session("pvc_edge_v1"),
                           root=str(tmp_path), project="pvc_edge_v1",
                           extra={"consent": ConsentRegistry()})

    def _stage(self, ctx, links):
        from src.assistant.tools import PROPOSE
        from src.assistant.tools.registry import dispatch

        out = dispatch(ctx, "propose_project",
                       {"package": _pkg(links),
                        "rationale": "первичный ввод проекта"},
                       allowed_kinds=[PROPOSE])
        assert out["staged"] is True, out
        return out["project_id"]

    def test_apply_with_text_link_succeeds(self, tmp_path):
        """Ровно сценарий отказа 12.08.2026 — теперь проходит целиком."""
        from src.assistant import context as actx

        ctx = self._ctx(tmp_path)
        pid = self._stage(ctx, ["dT_head: T_adapter - T_plast : 10, 60"])
        out = actx.human_apply_project(ctx, pid, author="человек (тест)")
        assert out["ok"] is True
        assert out["setup_prefill"]["setup_process_links"] \
            == "dT_head: T_adapter - T_plast : 10, 60"

    def test_bad_link_never_reaches_stage(self, tmp_path):
        """Гейт стоит РАНЬШЕ стейджа: утверждать нечего, кнопки не будет."""
        from src.assistant.tools import PROPOSE
        from src.assistant.tools.registry import dispatch

        ctx = self._ctx(tmp_path)
        out = dispatch(ctx, "propose_project",
                       {"package": _pkg(["dT_head: T_nope - T_plast : 10, 60"]),
                        "rationale": "связка на несуществующую ось"},
                       allowed_kinds=[PROPOSE])
        assert out.get("staged") is not True
        assert not ctx.session.staged_projects()

    def test_projection_failure_is_tool_error_not_crash(self, tmp_path):
        """UI ловит ToolError: сырой PackageError уронил бы всю страницу."""
        from src.assistant import context as actx
        from src.assistant.tools import ToolError
        import src.assistant.tools.write as w

        ctx = self._ctx(tmp_path)
        pid = self._stage(ctx, [dict(CANON)])

        def _boom(_pkg_obj):
            raise PackageError("искусственный сбой проекции")

        orig = w.package_to_setup_prefill if hasattr(
            w, "package_to_setup_prefill") else None
        import src.design.project_package as pp
        saved = pp.package_to_setup_prefill
        pp.package_to_setup_prefill = _boom
        try:
            with pytest.raises(ToolError, match="ГЕЙТ ВАЛИДАЦИИ"):
                actx.human_apply_project(ctx, pid)
        finally:
            pp.package_to_setup_prefill = saved
            if orig is not None:
                w.package_to_setup_prefill = orig
        # Дефект 2: подтверждение НЕ израсходовано, пакет остался в стейдже —
        # значит после исправления человек нажимает кнопку снова.
        assert ctx.session.project_by_id(pid).status == "staged"
