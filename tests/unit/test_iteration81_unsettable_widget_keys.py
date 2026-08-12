# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 81 — загрузка проекта роняла приложение: ключ НЕПРИСВАИВАЕМОГО виджета.

Живая сессия 12.08.2026, трассировка пользователя::

    StreamlitValueAssignmentNotAllowedError: Values for the widget with key
    'setup_phr_file' cannot be set using st.session_state
    ... campaign_ui.py:2500 in _render_phr_json_input
        up = st.file_uploader("JSON-файл спеки (опц.)", ...)

Разбор (факты, а не догадки):

  1. ``st.file_uploader`` без выбранного файла держит в ``session_state`` значение
     ``None``. Ключ виджета — ``setup_phr_file``.
  2. :func:`campaign_state.setup_draft_fields` брала ЛЮБОЙ ключ ``setup_*`` со
     скалярным значением ИЛИ ``None`` — то есть и ключ загрузчика. В черновике
     живого проекта на диске лежит ``"setup_phr_file": null``.
  3. При загрузке проекта черновик целиком уходит в ``setup_prefill_pending``, а
     ``render_setup_form`` присваивает КАЖДЫЙ его ключ. Streamlit запрещает
     присваивать значение ``file_uploader`` (``writes_allowed=False``) — падение
     уносило весь ``main()``, а не одну панель.
  4. Причина класса, а не одного ключа: исключения перечислялись по именам
     (кнопки формы), и любой новый неприсваиваемый виджет с ключом ``setup_*``
     протёк бы тем же путём. В том же классе, кроме ``st.file_uploader``:
     ``st.button``, ``st.download_button``, ``st.data_editor``.

Здесь фиксируется:

  * ОДИН барьер (:func:`campaign_state.is_settable_setup_key`) и на записи
    черновика, и на применении префилла — старые черновики на диске править не
    требуется, ключ отбрасывается при применении;
  * АСТ-сканер исходника: каждый неприсваиваемый виджет формы сетапа объявлен в
    :data:`campaign_state.SETUP_UNSETTABLE_KEYS` / ``…_PREFIXES``. Новый виджет
    ловится тестом, а не пользователем;
  * живой ``AppTest``: проект-черновик со «старым» ключом ``setup_phr_file``
    загружается без исключения, а настоящие поля формы восстанавливаются.
"""
import ast
import json
import os
from pathlib import Path

import pytest

from src.apps import campaign_state as cst
from src.core import project_ref as pr

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_UI = os.path.join(_REPO, "src", "apps", "campaign_ui.py")
_APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")

#: Виджеты Streamlit, значение которых НЕЛЬЗЯ задать через ``session_state``
#: (в исходнике 1.58 — ``check_session_state_rules(..., writes_allowed=False)``).
#: Список сверен по ``.venv/Lib/site-packages/streamlit/elements/widgets/``:
#: audio_input, button (+download_button, link_button), camera_input,
#: data_editor, file_uploader, menu_button.
UNSETTABLE_WIDGETS = frozenset({
    "button", "download_button", "link_button", "form_submit_button",
    "file_uploader", "camera_input", "audio_input", "data_editor",
    "menu_button",
})

#: Живой черновик до iter81 (фрагмент реального ``setup_draft.json``): рядом с
#: настоящими полями формы лежит ключ ПУСТОГО загрузчика спеки.
LEGACY_DRAFT = {
    "setup_mix": "PVC_67, PVC_71, DINP",
    "setup_resp": "Gelation, ColorUniformity",
    "setup_comp_mode": "phr-спека (JSON)",
    "setup_phr_src": "JSON / файл",
    "setup_seed": 1,
    "setup_phr_file": None,          # ← ЭТО и роняло приложение
}


# ======================================================================
# 1. БАРЬЕР: какие ключи присваивать нельзя
# ======================================================================
class TestSettableKeyBarrier:
    def test_file_uploader_key_is_refused(self):
        """САМ БАГ: ключ загрузчика JSON-спеки присваивать нельзя."""
        assert cst.is_settable_setup_key("setup_phr_file") is False

    @pytest.mark.parametrize("key", ["setup_build", "setup_propose_seed",
                                     "setup_commit_seed", "setup_fill_demo",
                                     "setup_seed_dl", "setup_seed_editor",
                                     "setup_phr_add_group",
                                     "setup_phr_add_single",
                                     "setup_phr_clear"])
    def test_static_widget_keys_refused(self, key):
        assert cst.is_settable_setup_key(key) is False

    @pytest.mark.parametrize("key", [
        "setup_phr_up_3_1234", "setup_phr_dn_3_1234", "setup_phr_del_3_1234",
        "setup_phr_kids_3_1234_v2", "setup_phr_kids_df_3_1234_v2"])
    def test_dynamic_widget_keys_refused(self, key):
        """Ключи с uid узла спеки — тот же класс, но хвост непредсказуем."""
        assert cst.is_settable_setup_key(key) is False

    @pytest.mark.parametrize("key", [
        "setup_mix", "setup_resp", "setup_phr_json", "setup_phr_src",
        "setup_comp_mode", "setup_seed", "setup_lo_3_0", "setup_phi_4_0",
        "setup_phr_gname_3_1234", "setup_phr_schema"])
    def test_real_form_fields_allowed(self, key):
        """Обратная сторона: настоящие поля формы НЕ должны отсекаться."""
        assert cst.is_settable_setup_key(key) is True


# ======================================================================
# 2. ЗАПИСЬ: черновик больше не сохраняет то, что при загрузке роняет форму
# ======================================================================
class TestDraftSnapshot:
    def test_uploader_key_not_written_to_draft(self):
        state = {"setup_mix": "A, B", "setup_phr_file": None}
        assert cst.setup_draft_fields(state) == {"setup_mix": "A, B"}

    def test_buttons_and_editor_not_written(self):
        state = {"setup_resp": "gloss", "setup_build": False,
                 "setup_seed_dl": False, "setup_seed_editor": {"edited": {}},
                 "setup_phr_add_group": False}
        assert cst.setup_draft_fields(state) == {"setup_resp": "gloss"}

    def test_run_state_not_written(self):
        """Кэш диагностики и номер строки плана — состояние ПРОГОНА."""
        state = {"setup_mix": "A, B", "setup_seed_pf_err": "старая ошибка",
                 "setup_weigh_row": 3, "setup_bounds_row": 3}
        assert cst.setup_draft_fields(state) == {"setup_mix": "A, B"}

    def test_real_fields_survive(self):
        keep = {"setup_mix": "A, B", "setup_phr_json": "{}", "setup_seed": 4,
                "setup_econ_on": True, "setup_pass_weigh_step": 0.1}
        assert cst.setup_draft_fields({**keep, "setup_phr_file": None}) == keep


# ======================================================================
# 3. ПРИМЕНЕНИЕ: старый черновик с диска больше не роняет форму
# ======================================================================
class TestPrefillFilter:
    def test_legacy_draft_is_filtered_on_apply(self):
        """Файлы на диске править не нужно — ключ отсекается при применении."""
        out = cst.settable_setup_fields(LEGACY_DRAFT)
        assert "setup_phr_file" not in out
        assert out["setup_mix"] == LEGACY_DRAFT["setup_mix"]
        assert out["setup_comp_mode"] == "phr-спека (JSON)"
        assert len(out) == len(LEGACY_DRAFT) - 1

    def test_none_value_of_real_field_survives(self):
        """Фильтр — по КЛЮЧУ, а не по значению: None настоящего поля живёт."""
        assert cst.settable_setup_fields({"setup_phr_json": None}) == \
            {"setup_phr_json": None}

    def test_foreign_keys_untouched(self):
        """Ключи вне формы сетапа фильтр не касается (границы функции)."""
        assert cst.settable_setup_fields({"campaign_name": "x"}) == \
            {"campaign_name": "x"}

    def test_empty_and_none_input(self):
        assert cst.settable_setup_fields(None) == {}
        assert cst.settable_setup_fields({}) == {}

    def test_draft_round_trip_through_disk(self, tmp_path):
        """Сквозь диск: сохранённый снимок применяется целиком, без потерь."""
        fields = cst.setup_draft_fields({**LEGACY_DRAFT})
        cst.save_setup_draft(tmp_path, "p1", fields)
        loaded = cst.load_setup_draft(tmp_path, "p1")
        assert cst.settable_setup_fields(loaded) == loaded

    def test_form_applies_prefill_through_the_barrier(self):
        """Форма сетапа применяет префилл ИМЕННО через фильтр, а не напрямую.

        Проверка структурная (AST тела функции), а не по поведению: полный
        рендер формы требует запущенного Streamlit и покрыт живым ``AppTest``
        ниже. Здесь фиксируется, что барьер стоит НА ПУТИ применения — иначе
        рефакторинг вернул бы падение («фильтр есть, но не вызывается»).
        """
        import ast as _ast
        import inspect

        from src.apps import campaign_ui as ui
        src = inspect.getsource(ui.render_setup_form)
        tree = _ast.parse(src.lstrip())
        calls = [n for n in _ast.walk(tree)
                 if isinstance(n, _ast.Call)
                 and isinstance(n.func, _ast.Attribute)
                 and n.func.attr == "settable_setup_fields"]
        assert calls, ("render_setup_form применяет setup_prefill_pending без "
                       "campaign_state.settable_setup_fields — ключ "
                       "неприсваиваемого виджета снова уронит загрузку проекта")


# ======================================================================
# 4. СКАНЕР ИСХОДНИКА: новый неприсваиваемый виджет ловится тестом
# ======================================================================
def _call_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return fn.id if isinstance(fn, ast.Name) else ""


def _owner_map(tree: ast.AST):
    """``{функция → её AST-узел}`` + ``{узел вызова → имя объемлющей функции}``."""
    funcs, owner = {}, {}
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef):
            continue
        funcs[fn.name] = fn
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Call):
                owner[sub] = fn.name
    return funcs, owner


def _arg_node(fn: ast.FunctionDef, call: ast.Call, arg: str):
    """AST-узел значения аргумента ``arg`` в вызове ``call`` функции ``fn``.

    Учитывает и keyword-, и ПОЗИЦИОННУЮ передачу: ``_render_phr_json_input``
    вызывается как ``_render_phr_json_input(key_prefix)`` — без этого разбора
    цепочка префиксов рвалась и ключ загрузчика оставался шаблоном ``*``.
    """
    for kw in call.keywords:
        if kw.arg == arg:
            return kw.value
    names = [a.arg for a in fn.args.args]
    if arg in names:
        i = names.index(arg)
        if i < len(call.args):
            return call.args[i]
    return None


def _prefix_sets(tree: ast.AST):
    """``{функция → множество возможных значений её ``key_prefix``}``.

    Значение ВЫВОДИТСЯ из кода, а не задаётся руками: literal-аргумент вызова,
    дефолт из сигнатуры, либо (транзитивно) префиксы вызывающей функции, если
    аргумент передан как та же переменная ``key_prefix``. Итерация до фикс-точки
    — цепочка ``render_composition_bounds(key_prefix="setup")`` →
    ``_render_phr_json_input(key_prefix)`` даёт ``{"setup"}``, а ``_tab_row``,
    который зовут с ``"ws_tab"``/``"camp_branch_tab"``, к форме сетапа не
    относится и его ключи под проверку не попадают.
    """
    funcs, owner = _owner_map(tree)
    out = {name: set() for name in funcs}
    # дефолты сигнатуры: def f(..., *, key_prefix: str = "setup")
    for name, fn in funcs.items():
        args = fn.args
        pairs = list(zip(args.args[-len(args.defaults):] if args.defaults else [],
                         args.defaults or []))
        pairs += list(zip(args.kwonlyargs, args.kw_defaults or []))
        for a, dflt in pairs:
            if (a is not None and a.arg == "key_prefix"
                    and isinstance(dflt, ast.Constant)
                    and isinstance(dflt.value, str)):
                out[name].add(dflt.value)
    for _ in range(len(funcs) + 2):          # фикс-точка (глубина цепочки мала)
        changed = False
        for call in owner:
            callee = _call_name(call)
            if callee not in out:
                continue
            node = _arg_node(funcs[callee], call, "key_prefix")
            if node is None:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                add = {node.value}
            elif isinstance(node, ast.Name) and node.id == "key_prefix":
                add = set(out.get(owner[call], ()))
            else:
                add = {"*"}
            if not add <= out[callee]:
                out[callee] |= add
                changed = True
        if not changed:
            break
    return out


def _key_pattern(node, prefixes) -> str:
    """Шаблон ключа виджета из AST-узла: неизвестная подстановка → ``*``.

    ``f"{key_prefix}_phr_file"`` разворачивается в ``setup_phr_file`` ТОЛЬКО
    если у объемлющей функции ровно один выведенный префикс (``prefixes``).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            elif (isinstance(part, ast.FormattedValue)
                  and isinstance(part.value, ast.Name)
                  and part.value.id == "key_prefix"
                  and len(prefixes) == 1):
                out.append(next(iter(prefixes)))
            else:
                out.append("*")
        return "".join(out)
    return "*"


def _unsettable_widget_keys(path: str):
    """Все ключи неприсваиваемых виджетов в модуле: ``[(строка, шаблон)]``."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    _, owner = _owner_map(tree)
    pref = _prefix_sets(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) not in UNSETTABLE_WIDGETS:
            continue
        scope = pref.get(owner.get(node, ""), set())
        for kw in node.keywords:
            if kw.arg == "key":
                found.append((node.lineno, _key_pattern(kw.value, scope)))
    return found


@pytest.mark.parametrize("path", [_UI, _APP])
def test_every_unsettable_setup_widget_is_declared(path):
    """Инвариант: ключ ``setup_*`` неприсваиваемого виджета объявлен в барьере.

    Иначе следующий такой виджет уронил бы загрузку проекта ровно так же, как
    ``setup_phr_file`` в живой сессии. Шаблон с динамическим хвостом (``*``)
    должен быть покрыт префиксом — проверяем и сам шаблон, и его начало.
    """
    leaked = [(ln, pat) for ln, pat in _unsettable_widget_keys(path)
              if pat.startswith("setup_")
              and cst.is_settable_setup_key(pat)
              and cst.is_settable_setup_key(pat.split("*")[0])]
    assert not leaked, (
        f"{os.path.basename(path)}: ключи неприсваиваемых виджетов формы "
        f"сетапа не объявлены в campaign_state.SETUP_UNSETTABLE_KEYS / "
        f"SETUP_UNSETTABLE_PREFIXES: {leaked}")


def test_scanner_sees_the_regression_widget():
    """Проверка самого сканера: загрузчик спеки он находит (иначе тест пустой)."""
    pats = [pat for _, pat in _unsettable_widget_keys(_UI)]
    assert "setup_phr_file" in pats
    assert any(p.startswith("setup_phr_kids_") for p in pats)


def test_prefix_inference_resolves_spec_form_to_setup():
    """Основание разворачивания ``f"{key_prefix}_…"`` в сканере — ВЫВОД из кода.

    Формы спеки зовутся только из формы сетапа (``key_prefix="setup"``), поэтому
    их ключи однозначны. Ряд закладок (``_tab_row``) — другие префиксы
    (``ws_tab`` / ``camp_branch_tab``): его кнопки к форме сетапа не относятся и
    под инвариант попадать НЕ должны.
    """
    pref = _prefix_sets(ast.parse(Path(_UI).read_text(encoding="utf-8")))
    for fn in ("render_composition_bounds", "render_process_bounds",
               "_render_phr_json_input", "_render_phr_tree_input"):
        assert pref[fn] == {"setup"}, (fn, pref[fn])
    assert pref["_tab_row"] == {"ws_tab", "camp_branch_tab"}, pref["_tab_row"]


# ======================================================================
# 5. ЖИВОЕ приложение: загрузка «старого» черновика не роняет экран
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

CAMPAIGN_ROOT = os.path.join(_REPO, "project_campaigns")
PROBE = "iter81_legacy_draft_probe"


def test_loading_legacy_draft_does_not_crash_app():
    """Регресс живой ошибки: «📂 Загрузить проект» на черновике до iter81.

    Черновик кладётся на диск КАК БЫЛ (с ``setup_phr_file: null``) — так его
    записала прежняя версия. До фикса первый же ``st.file_uploader`` формы
    сетапа падал ``StreamlitValueAssignmentNotAllowedError`` и уносил ``main()``.
    """
    try:
        cst.delete_campaign(CAMPAIGN_ROOT, PROBE)
    except Exception:  # noqa: BLE001 — проекта могло и не быть
        pass
    # пишем файл напрямую, БЕЗ фильтра снимка: воспроизводим старый формат
    base = Path(CAMPAIGN_ROOT) / PROBE
    base.mkdir(parents=True, exist_ok=True)
    pr.ensure_identity(CAMPAIGN_ROOT, PROBE, label=PROBE)
    (base / pr.SETUP_DRAFT_FILE).write_text(
        json.dumps(LEGACY_DRAFT, ensure_ascii=False, indent=2),
        encoding="utf-8")
    try:
        assert cst.load_setup_draft(CAMPAIGN_ROOT, PROBE) == LEGACY_DRAFT
        assert cst.campaign_exists(CAMPAIGN_ROOT, PROBE) is False

        at = AppTest.from_file(_APP, default_timeout=300).run()
        assert not at.exception
        at.selectbox(key="campaign_select").set_value(PROBE).run()
        [b for b in at.button if b.key == "load_campaign"][0].click().run()

        assert not at.exception, f"загрузка черновика упала: {at.exception}"
        ss = at.session_state
        # Ключ загрузчика в состояние НЕ попал ИЗ ЧЕРНОВИКА. Сам виджет,
        # отрисовавшись, может завести ключ со значением None — поэтому
        # утверждаем именно про значение (файла не выбирали), а не про
        # отсутствие ключа. NB: у ``at.session_state`` нет ``.get`` —
        # обращение к атрибуту уходит в ``__getitem__`` и падает AttributeError.
        assert ("setup_phr_file" not in ss
                or ss["setup_phr_file"] is None)
        # …а настоящие поля формы восстановлены
        assert ss["setup_mix"] == LEGACY_DRAFT["setup_mix"]
        assert ss["setup_resp"] == LEGACY_DRAFT["setup_resp"]
        assert ss["setup_comp_mode"] == LEGACY_DRAFT["setup_comp_mode"]
        assert int(ss["setup_seed"]) == LEGACY_DRAFT["setup_seed"]
    finally:
        try:
            cst.delete_campaign(CAMPAIGN_ROOT, PROBE)
        except Exception:  # noqa: BLE001
            pass
