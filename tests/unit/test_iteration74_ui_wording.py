# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 74 — ЯЗЫК интерфейса и согласованность промпта с экраном.

Запрос пользователя: в интерфейсе не должно быть внутреннего сленга разработки
(«seed», «долить», «undo», «spawn», «стейдж», «выхлоп песочницы», «preflight»,
«суррогаты», «оракул», «гейт»), а помощник обязан понимать ФАКТИЧЕСКИЙ экран —
после iter72 это три зоны и рабочая область на закладках, сайдбара нет.

Почему это ТЕСТ, а не разовая правка: подписи расползаются обратно с каждой
новой секцией, а промпт «отстаёт» от UI молча — модель начинает отправлять
человека к кнопкам, которых на экране нет. Здесь зафиксировано:

1. **ВИДИМЫЕ строки UI** (разбор AST аргументов вызовов ``st.*``) не содержат
   сленговых слов. Проверяется именно текст виджетов, а не комментарии и
   docstring'и: внутреннее имя в коде — норма, на экране — нет.
2. **Подписи кнопок потока** остались операционными (одной «чёрной» проверки
   мало: она прошла бы и на экране, где кнопки переименованы во что угодно).
3. **Промпт согласован с экраном**: промпт архитектора и промпт обзора называют
   три зоны, закладки и отсутствие сайдбара, запрещают отвечать внутренними
   словами; карта интерфейса ``campaign_ui_guide`` описывает раскладку и
   подписи кнопок дословно (и не ссылается на исчезнувшие кнопки).
"""
import ast
import os
import re
from typing import Dict, List, Tuple

import pytest

from src.apps import assistant as ai
from src.apps import workspace as ws
from src.assistant import context as actx
from src.assistant.prompts import UI_BLOCK, architect_system_prompt

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Модули, чьи строки ВИДИТ пользователь.
UI_MODULES = (
    os.path.join(_REPO, "src", "apps", "campaign_ui.py"),
    os.path.join(_REPO, "src", "apps", "assistant_dock.py"),
    os.path.join(_REPO, "src", "apps", "streamlit_app.py"),
)

#: Сленг → операционная замена (для текста ошибки). Ключ — регулярка по
#: видимой строке без учёта регистра.
SLANG: Dict[str, str] = {
    r"seed-?(дизайн|точ|цикл)": "«стартовый план опытов»",
    r"\bдоли(ть|в|л|т)\w*\b": "«добавить точки в базу»",
    r"\bundo\b": "«отменить последнюю настройку»",
    r"\bspawn\w*": "«дочерняя ветка»",
    r"стейдж": "«ждёт применения»",
    r"выхлоп": "«файлы расчётов»",
    r"preflight": "«проверка плана»",
    r"суррогат": "«модели свойств»",
    r"оракул": "«тестовый симулятор»",
    r"\bгейт\w*": "«проверка»",
    r"снапн\w+": "«приведён к шагу весов»",
    r"телеметри\w+": "«условия прогона»",
    r"\bлинза\b": "«выбранная ветка»",
    r"review-сводк\w+": "«сводка»",
    r"раннер\w*": "«движок проекта»",
    r"биндинг\w*": "«активное ограничение»",
    r"\bфит\b": "«подгонка модели»",
}

#: Строки, где внутреннее имя ОСТАЁТСЯ намеренно (имена инструментов и ключей
#: формата данных — по ним человек воспроизводит расчёт).
ALLOWED_SUBSTRINGS: Tuple[str, ...] = (
    "run_python", "savefig", "PhrSpec.from_dicts", "spec_hash",
)

#: Виджеты, чей текст печатается на экране (метка, help, caption, кнопка).
WIDGETS = (
    "markdown", "caption", "button", "write", "subheader", "info", "warning",
    "error", "success", "text_input", "number_input", "selectbox", "checkbox",
    "radio", "expander", "metric", "slider", "multiselect", "toggle",
    "text_area", "data_editor", "download_button", "file_uploader", "popover",
    "spinner", "chat_input", "header",
)


def _st_name(call: ast.Call) -> str:
    """Полное имя вызываемого (``st.caption`` / ``cu[0].button`` → ``button``)."""
    f = call.func
    parts: List[str] = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    parts.reverse()
    return ".".join(parts)


def _string_literals(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            yield n


def visible_strings(path: str) -> List[Tuple[int, str]]:
    """Строки, попадающие в виджеты Streamlit (подписи, help, caption).

    Берутся ТОЛЬКО аргументы вызовов виджетов — то есть текст, который реально
    печатается на экране. Комментарии и docstring'и модулей сюда не попадают:
    внутреннее имя в коде допустимо, на экране — нет. Вызовы
    ``st.session_state.*`` исключены: там лежат КЛЮЧИ виджетов, а не текст.
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    out: List[Tuple[int, str]] = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _st_name(node)
        if "session_state" in name or name.split(".")[-1] not in WIDGETS:
            continue
        # kwargs `key`/`file_name`/`mime`/`format` — служебные идентификаторы,
        # а не текст на экране: сленг в них не важен (ключи трогать нельзя,
        # на них держатся тесты и состояние виджетов).
        args = list(node.args) + [kw.value for kw in node.keywords
                                  if kw.arg not in ("key", "file_name", "mime",
                                                    "format", "file_type")]
        for arg in args:
            for lit in _string_literals(arg):
                txt = lit.value.strip()
                key = (lit.lineno, txt)
                if len(txt) < 3 or key in seen:
                    continue
                seen.add(key)
                out.append(key)
    return out


# ======================================================================
# 1. Видимые строки UI: сленга нет
# ======================================================================
@pytest.mark.parametrize("path", UI_MODULES,
                         ids=[os.path.basename(p) for p in UI_MODULES])
def test_visible_ui_strings_have_no_slang(path):
    hits: List[str] = []
    for lineno, txt in visible_strings(path):
        if any(a in txt for a in ALLOWED_SUBSTRINGS):
            continue
        for pattern, better in SLANG.items():
            if re.search(pattern, txt, flags=re.IGNORECASE):
                hits.append(f"{os.path.basename(path)}:{lineno} "
                            f"«{txt[:70]}» → используйте {better}")
    assert not hits, "сленг в видимых строках UI:\n" + "\n".join(hits)


def test_workspace_labels_and_reasons_are_operational():
    """Подписи закладок и причины недоступности — на языке технолога."""
    blurbs = " ".join(t.blurb for t in ws.WORKSPACE_TABS).lower()
    for bad in ("seed", "undo", "spawn", "ard"):
        assert bad not in blurbs, f"«{bad}» в подписях закладок"
    why = " ".join(ws.WHY.values()).lower()
    assert "seed" not in why
    assert "стартовый план" in why


def test_focus_sections_speak_ui_language():
    """Карта мест ассистента (её текст уходит в промпт) — без сленга."""
    blob = " ".join(
        f"{s.title} {s.doing} " + " ".join(q for _, q in s.asks)
        for s in actx.FOCUS_SECTIONS + (actx.UNKNOWN_SECTION,)).lower()
    for bad in ("seed", "preflight", "undo", "spawn", "оракул", "суррогат"):
        assert bad not in blob, f"«{bad}» в карте шагов ассистента"


# ======================================================================
# 2. Позитивная проверка: операционные подписи на месте
# ======================================================================
def test_flow_buttons_keep_operational_labels():
    """Ключевые кнопки потока подписаны так, как о них говорит помощник.

    Без этой проверки «чёрный список» слов проходил бы и на экране, где кнопки
    переименованы во что угодно.
    """
    src = open(os.path.join(_REPO, "src", "apps", "campaign_ui.py"),
               encoding="utf-8").read()
    for label in ("📐 Предложить стартовый план",
                  "💾 Зафиксировать стартовый план и отклики",
                  "📐 Предложить точки (база не меняется)",
                  "💾 Добавить измеренные точки в базу",
                  "↩️ Отменить последнюю настройку",
                  "🌱 Создать дочернюю ветку",
                  "🧪 Заполнить тестовыми значениями"):
        assert label in src, f"подпись «{label}» пропала из интерфейса"


# ======================================================================
# 3. Промпт согласован с ФАКТИЧЕСКИМ экраном (iter72: три зоны + закладки)
# ======================================================================
def test_architect_prompt_describes_three_zones_and_tabs():
    text = architect_system_prompt(project="p")
    assert UI_BLOCK in text, "блок про экран не собирается в промпт"
    for mark in ("ТРИ зоны", "ЗАКЛАДКАХ", "🌱 Старт", "📚 База опытов",
                 "🌿 Ветки", "📊 Анализ", "🧬 Схема", "🤖 Обзор"):
        assert mark in text, f"промпт не знает про «{mark}»"
    # Сайдбар упразднён в iter72 — модель не должна отправлять туда человека.
    assert "сайдбара) НЕТ" in text


def test_architect_prompt_forbids_internal_jargon_in_answers():
    text = architect_system_prompt()
    assert "ЯЗЫК ОТВЕТА" in text
    for pair in ("не seed", "не undo", "не spawn", "не стейдж",
                 "не preflight", "не суррогаты"):
        assert pair in text, f"в промпте нет замены «{pair}»"


def test_campaign_guide_describes_layout_and_real_buttons():
    guide = ai.campaign_ui_guide()
    assert "layout" in guide and "flow" in guide
    # Проверяем и КЛЮЧИ раскладки (там живёт «сайдбара нет»), и их описания.
    layout_blob = (" ".join(guide["layout"])
                   + " " + " ".join(str(v) for v in guide["layout"].values())
                   ).lower()
    assert "закладк" in layout_blob and "сайдбар" in layout_blob
    assert "боковой панели в приложении нет" in layout_blob
    flow_blob = " ".join(str(v) for v in guide["flow"].values())
    for label in ("📐 Предложить стартовый план",
                  "💾 Добавить измеренные точки в базу",
                  "🏗 Построить проект"):
        assert label in flow_blob, f"карта интерфейса не знает «{label}»"


def test_campaign_prompt_matches_the_screen():
    prompt = ai.campaign_system_prompt()
    assert "ЗАКЛАДКАХ" in prompt
    assert "сайдбара) в приложении НЕТ" in prompt
    assert "ЯЗЫКОМ ИНТЕРФЕЙСА" in prompt
    # Прежний запрет остаётся в силе: стадий M1…M8 в приложении нет.
    assert "M1…M8" in prompt


def test_campaign_guide_has_no_stale_button_labels():
    """Карта не ссылается на подписи кнопок, которых в UI уже нет.

    Именно этот разрыв ломает доверие к помощнику: он уверенно называет кнопку,
    а человек её не находит. Из проверки исключены названия закладок и панелей
    (они рисуются другими модулями), сверяются подписи кнопок/форм потока.
    """
    guide_blob = str(ai.campaign_ui_guide())
    src = open(os.path.join(_REPO, "src", "apps", "campaign_ui.py"),
               encoding="utf-8").read()
    not_buttons = {
        "🌱 Старт", "📚 База опытов", "🌿 Ветки", "📊 Анализ", "🧬 Схема",
        "🤖 Обзор", "📎 Вложения", "🖼 Файлы расчётов помощника",
        "📌 Состояние переписки с помощником", "💬 Помощник по проекту",
        "🏗 Предложенные проекты (пакетом)", "🧬 Предложенные спеки (пакетом)",
        "🧩 Предложенные патчи спеки", "📁 Проект",
    }
    checked = 0
    for label in re.findall(r"«([^»]{4,70})»", guide_blob):
        if label in not_buttons or not re.match(r"[📐💾🧪🏗➕🛠✏️📈🔬]", label):
            continue
        checked += 1
        assert label in src, (f"карта интерфейса ссылается на «{label}», "
                              f"которой нет в campaign_ui.py")
    assert checked >= 5, "проверка не нашла подписей кнопок — регулярка сломалась"
