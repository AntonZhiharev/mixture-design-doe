"""assistant/context.py — контекст хода: проект + фокус UI + спека + вложения (iter65).

Док ассистента живёт в ПРАВОЙ КОЛОНКЕ рядом с потоком кампании и виден на
каждом шаге. Значит, у ассистента появляется то, чего не было у отдельной
вкладки: он знает, ГДЕ сейчас человек — какой шаг открыт, какой узел спеки
разложен, какая ветка выбрана. Этот модуль превращает «место в интерфейсе» в
рабочий контекст хода и держит его ЧИСТЫМ (без Streamlit), чтобы то же самое
могли использовать демо-скрипт и тесты.

Опоры модуля:

1. **Фокус — ФАКТ о месте пользователя, а не разрешение отвечать по памяти.**
   Из фокуса берётся, что «эта ось» — это `DINP` (:func:`resolve_question`), но
   границы `DINP` всё равно считает `explain_node`. Иначе удобство подстановки
   превратилось бы в лазейку для выдумывания чисел.
2. **Фокус читается ЧИСТОЙ функцией** (:func:`focus_from_state`) из обычного
   словаря — того самого, который в приложении зовётся ``st.session_state``.
   Поэтому «контекст по месту» проверяется тестом без запуска Streamlit.
3. **Подсказки по месту генерируются, а не рисуются руками**
   (:func:`suggested_questions`): у каждого шага свои типовые вопросы, и
   каждый обязан маршрутизироваться роутером iter64 в осмысленный вид (тест
   это сверяет). Подсказка, для которой не хватает данных (узел не выбран),
   не исчезает молча — она показывается ВЫКЛЮЧЕННОЙ с причиной (A0.6).
4. **Один ход — одна функция** (:func:`run_turn`): сборка контекста, цикл
   инструментов, запись аудита и ответа в сессию. Док, демо и тест зовут её
   же, поэтому «в демо работает, в интерфейсе нет» невозможно по построению.
5. **Кнопка человека — единственный путь к классу ``write``**
   (:func:`human_apply` / :func:`human_reject`): выдать разовый токен и тут же
   его погасить может только вызывающий UI-код, модели этот путь недоступен
   (iter63).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import llm
from .prompts import (KIND_LABEL, KIND_STATUS, architect_system_prompt,
                      parse_sections, route, with_system)
from .session import AssistantSession, ToolCall

#: Ключ, под которым секции интерфейса публикуют свой фокус в session_state.
FOCUS_KEY = "ui_focus"

#: Метка системного сообщения с фокусом — по ней его видно в разборе хода.
FOCUS_MARK = "[фокус] "

#: Метка системного сообщения с дайджестом вложений.
FILES_MARK = "[вложения] "

#: Бюджет контекста диалога по умолчанию (символы→токены грубо, iter58).
CONTEXT_TOKENS = 24000

#: Сколько символов текста каждого вложения уходит в дайджест контекста.
DIGEST_CHARS = 1200


# ----------------------------------------------------------------------
# Шаги потока (карта мест интерфейса)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FocusSection:
    """Шаг потока кампании как МЕСТО, в котором задают вопросы.

    ``doing`` — что человек здесь делает (уходит в промпт: без этого модель
    отвечает «вообще», а не про текущий экран). ``asks`` — типовые вопросы
    места парами ``(подпись кнопки, шаблон вопроса)``; шаблон может содержать
    ``{node}`` / ``{branch}``.
    """
    key: str
    title: str
    doing: str
    asks: Tuple[Tuple[str, str], ...] = ()


#: Карта шагов. Ключи совпадают с тем, что публикуют секции UI в ``ui_focus``.
FOCUS_SECTIONS: Tuple[FocusSection, ...] = (
    FocusSection(
        key="setup", title="🧩 Сетап схемы",
        doing=("задаёт компоненты смеси (Σ=1), процесс-оси и отклики — "
               "геометрия кампании только формируется"),
        asks=(("Почему границы не такие, как я вводил",
               "Почему эффективные границы отличаются от того, что я вводил?"),
              ("Можно строить план?",
               "Можно уже строить план? Почему preflight ругается?"))),
    FocusSection(
        key="spec", title="🧪 phr-спека (узлы, роли, границы)",
        doing=("разбирает узлы phr-спеки: роли, эффективные границы, "
               "группы и техлимиты — здесь живёт вся геометрия"),
        asks=(("Объясни эту ось", "Объясни эту ось: {node}"),
              ("Почему диапазон не такой",
               "Почему диапазон {node} не такой, как я вводил?"),
              ("Что изменится, если…",
               "Что изменится, если сузить {node}?"))),
    FocusSection(
        key="seed", title="🌱 Стартовый дизайн (seed)",
        doing=("предлагает стартовый план и вносит измеренные отклики — "
               "до фиксации seed база кампании пуста"),
        asks=(("Можно строить план?",
               "Можно уже строить план? Почему preflight ругается?"),
              ("Что изменится, если…",
               "Что изменится, если сузить {node}?"))),
    FocusSection(
        key="weighing", title="⚖️ Навеска (phr → граммы)",
        doing=("сверяет рецепт с разрешением весов: nominal против actual"),
        asks=(("Эта рецептура в границах?",
               "Эта рецептура попадает в границы геометрии?"),
              ("Объясни эту ось", "Объясни эту ось: {node}"))),
    FocusSection(
        key="base", title="📚 Общая база опытов",
        doing=("смотрит уже измеренные точки проекта (origin-теги, отклики)"),
        asks=(("Что в базе?", "Сколько точек в базе и что они покрывают?"),
              ("Эта рецептура в границах?",
               "Эта рецептура попадает в границы геометрии?"))),
    FocusSection(
        key="branch", title="🌿 Ветки и цели",
        doing=("создаёт ветку и ставит цели над откликами (роли, ценовая нога)"),
        asks=(("Можно строить план ветки?",
               "Можно уже строить план ветки {branch}?"),
              ("Что в базе?", "Сколько точек в базе и что они покрывают?"))),
    FocusSection(
        key="workbench", title="🛠 Рабочий стол ветки",
        doing=("предлагает точки, вносит отклики и доливает их в общую базу"),
        asks=(("Эта рецептура в границах?",
               "Эта рецептура попадает в границы геометрии?"),
              ("Что изменится, если…",
               "Что изменится, если сузить {node}?"))),
    FocusSection(
        key="evolution", title="🧬 Эволюция схемы",
        doing=("двигает границы области, раскрывает оси и вводит отклики — "
               "правки задним числом касаются уже собранных точек"),
        asks=(("Что изменится, если…",
               "Что изменится, если расширить {node}?"),
              ("Почему диапазон не такой",
               "Почему диапазон {node} не такой, как я вводил?"))),
    FocusSection(
        key="screening", title="📊 Анализ скрининга",
        doing=("читает эффекты состава по измеренному стартовому дизайну"),
        asks=(("Объясни эту ось", "Объясни эту ось: {node}"),
              ("Что в базе?", "Сколько точек в базе и что они покрывают?"))),
)

SECTIONS_BY_KEY: Dict[str, FocusSection] = {s.key: s for s in FOCUS_SECTIONS}

#: Шаг не определён (пользователь только открыл приложение либо секция не
#: публикует фокус). Спрашивать — честнее, чем угадывать место.
UNKNOWN_SECTION = FocusSection(
    key="", title="— шаг не определён",
    doing=("место в интерфейсе неизвестно: не строй догадок о шаге — "
           "спроси, что именно открыто, если это важно для ответа"),
    asks=(("Можно строить план?",
           "Можно уже строить план? Почему preflight ругается?"),
          ("Что в базе?", "Сколько точек в базе и что они покрывают?")))


def section(key: Any) -> FocusSection:
    """Шаг по ключу; неизвестный ключ — :data:`UNKNOWN_SECTION`, не исключение.

    Интерфейс развивается быстрее спеки контекста: незнакомый ключ секции не
    должен ронять док — он лишь означает «шаг не определён».
    """
    return SECTIONS_BY_KEY.get(str(key or ""), UNKNOWN_SECTION)


# ----------------------------------------------------------------------
# Фокус
# ----------------------------------------------------------------------
@dataclass
class UiFocus:
    """Где сейчас пользователь: шаг потока + предмет разговора.

    ``node`` — узел спеки, разложенный на экране («эта ось»); ``branch`` —
    выбранная ветка; ``recipe_phr`` — рецепт, который человек как раз
    рассматривает (карта навески, точка рабочего стола).
    """
    section_key: str = ""
    node: str = ""
    branch: str = ""
    response: str = ""
    recipe_phr: Optional[List[float]] = None
    note: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> FocusSection:
        return section(self.section_key)

    @property
    def title(self) -> str:
        return self.spec.title

    def is_empty(self) -> bool:
        return not (self.section_key or self.node or self.branch
                    or self.response or self.recipe_phr)

    def to_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"section": self.section_key}
        for k, v in (("node", self.node), ("branch", self.branch),
                     ("response", self.response), ("note", self.note)):
            if v:
                out[k] = v
        if self.recipe_phr:
            out["recipe_phr"] = [float(v) for v in self.recipe_phr]
        if self.extra:
            out["extra"] = dict(self.extra)
        return out


def normalize_focus(obj: Any) -> UiFocus:
    """Что угодно (``None`` / dict / :class:`UiFocus`) → фокус.

    Секции UI публикуют ОБЫЧНЫЙ словарь — чтобы им не требовался импорт слоя
    ассистента ради одной записи в ``session_state``.
    """
    if isinstance(obj, UiFocus):
        return obj
    d = dict(obj or {}) if isinstance(obj, Mapping) else {}
    recipe = d.get("recipe_phr")
    try:
        recipe = [float(v) for v in recipe] if recipe else None
    except (TypeError, ValueError):
        recipe = None
    return UiFocus(section_key=str(d.get("section", d.get("section_key", ""))),
                   node=str(d.get("node", "") or ""),
                   branch=str(d.get("branch", "") or ""),
                   response=str(d.get("response", "") or ""),
                   recipe_phr=recipe,
                   note=str(d.get("note", "") or ""),
                   extra=dict(d.get("extra", {}) or {}))


def focus_from_state(state: Optional[Mapping[str, Any]]) -> UiFocus:
    """Фокус из ``session_state`` приложения (чистая функция).

    Приоритет у явного ``ui_focus``, который публикует активная секция; чего
    там нет — добирается из уже существующих ключей интерфейса (выбранная
    ветка ``camp_branch``) и из ручных селекторов самого дока
    (``assistant_focus_*``). Ручной выбор человека НЕ перебивает то, что
    сказала секция: место, где он стоит, — факт, а селектор — уточнение.
    """
    st = dict(state or {})
    f = normalize_focus(st.get(FOCUS_KEY))
    if not f.section_key:
        f.section_key = str(st.get("assistant_focus_section", "") or "")
    if not f.node:
        f.node = str(st.get("assistant_focus_node", "") or "")
    if not f.branch:
        f.branch = str(st.get("camp_branch", "") or "")
    return f


def focus_caption(f: Any) -> str:
    """Одна строка «где мы» — заголовок дока и вывод демо."""
    f = normalize_focus(f)
    parts = [f"шаг: {f.title}"]
    if f.node:
        parts.append(f"узел: **{f.node}**")
    if f.branch:
        parts.append(f"ветка: {f.branch}")
    if f.response:
        parts.append(f"отклик: {f.response}")
    if f.recipe_phr:
        parts.append(f"рецепт: {len(f.recipe_phr)} узлов")
    return " · ".join(parts)


def focus_block(f: Any) -> str:
    """Фокус словами — блок системного сообщения хода.

    Последняя строка блока принципиальна: подстановка «эта ось → DINP» не
    даёт права на утверждения о геометрии. Место пользователя объясняет, ЧТО
    он спрашивает, а не ОТКУДА берутся числа.
    """
    f = normalize_focus(f)
    sec = f.spec
    lines = ["ФОКУС ИНТЕРФЕЙСА (где сейчас пользователь):",
             f"  • шаг: {sec.title} — {sec.doing}"]
    if f.node:
        lines.append(f"  • узел в фокусе: {f.node} — «эта ось», «этот узел», "
                     f"«здесь» в вопросе означают именно его")
    if f.branch:
        lines.append(f"  • ветка: {f.branch}")
    if f.response:
        lines.append(f"  • отклик: {f.response}")
    if f.recipe_phr:
        lines.append(f"  • рецепт на экране (phr): "
                     + ", ".join(f"{v:g}" for v in f.recipe_phr))
    if f.note:
        lines.append(f"  • примечание секции: {f.note}")
    lines.append("Фокус объясняет, ЧТО спрашивают, но не заменяет инструменты: "
                 "границы, роли и hash по-прежнему считаются вызовом.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Вопрос по месту
# ----------------------------------------------------------------------
_DEICTIC = (
    (re.compile(r"эт(?:у|ой|а|ой)\s+ос(?:ь|и)", re.IGNORECASE), "ось {node}"),
    (re.compile(r"эт(?:от|ого|ому)\s+узел|эт(?:ого|ому)\s+узла", re.IGNORECASE),
     "узел {node}"),
    (re.compile(r"эт(?:от|ого)\s+компонент\w*", re.IGNORECASE),
     "компонент {node}"),
    (re.compile(r"здесь|тут", re.IGNORECASE), "в узле {node}"),
    (re.compile(r"эт(?:у|ой|а)\s+ветк\w+", re.IGNORECASE), "ветку {branch}"),
)


def resolve_question(text: str, f: Any = None) -> str:
    """Подставить в вопрос предмет из фокуса («эту ось» → «ось DINP»).

    Без подстановки половина вопросов «по месту» бессмысленна вне экрана: в
    журнале сессии остаётся «объясни эту ось», и через неделю непонятно, какую.
    Подстановка делается только тем, что ЕСТЬ в фокусе: пустой узел ничего не
    заменяет (лучше честное «какую именно ось?», чем подстановка наугад).
    """
    s = str(text or "")
    f = normalize_focus(f)
    for rx, tmpl in _DEICTIC:
        if "{node}" in tmpl and not f.node:
            continue
        if "{branch}" in tmpl and not f.branch:
            continue
        s = rx.sub(tmpl.format(node=f.node, branch=f.branch), s)
    return s


@dataclass(frozen=True)
class Suggestion:
    """Подсказка «спросить по месту»: кнопка дока + готовый вопрос.

    ``enabled=False`` означает «данных не хватает» (например, узел не выбран);
    подсказка при этом ОСТАЁТСЯ видимой с причиной в ``why`` — исчезнувшая
    кнопка читалась бы как «здесь так спросить нельзя».
    """
    label: str
    question: str
    kind: str = ""
    tools: Tuple[str, ...] = ()
    enabled: bool = True
    why: str = ""

    @property
    def kind_label(self) -> str:
        return KIND_LABEL.get(self.kind, self.kind)


def suggested_questions(f: Any = None, *, has_runner: bool = True
                        ) -> List[Suggestion]:
    """Подсказки текущего шага (чистая функция, без сети и Streamlit).

    Маршрут каждой подсказки считается роутером iter64 — так подсказка и
    ответ на неё используют ОДИН контракт §8, и «кнопка есть, а инструмента
    для неё нет» невозможно.
    """
    f = normalize_focus(f)
    out: List[Suggestion] = []
    for label, tmpl in f.spec.asks:
        need_node = "{node}" in tmpl
        need_branch = "{branch}" in tmpl
        enabled, why = True, ""
        if need_node and not f.node:
            enabled = False
            why = ("узел не выбран: разложите узел в спеке или назовите его "
                   "в вопросе")
        if need_branch and not f.branch:
            enabled = False
            why = "ветка не выбрана: выберите ветку в потоке кампании"
        question = tmpl.format(node=f.node or "…", branch=f.branch or "…")
        r = route(tmpl.format(node=f.node or "узел", branch=f.branch or "ветка"))
        if not has_runner and r.kind == KIND_STATUS:
            why = why or ("проект не собран: ответ будет «не проверено», "
                          "а не «всё хорошо»")
        out.append(Suggestion(label=label, question=question, kind=r.kind,
                              tools=tuple(r.tools), enabled=enabled, why=why))
    return out


# ----------------------------------------------------------------------
# Сборка сообщений хода
# ----------------------------------------------------------------------
def attachments_block(session: AssistantSession, *,
                      per_file_chars: int = DIGEST_CHARS) -> str:
    """Дайджест вложений для контекста (полный текст — по `read_attachment`)."""
    from .views import attachment_digest      # локально: views тянет pandas

    items = attachment_digest(session, per_file_chars=per_file_chars)
    if not items:
        return ""
    lines = ["ВЛОЖЕНИЯ СЕССИИ (начало текста; полностью — `read_attachment`):"]
    for it in items:
        head = " ".join(str(it.get("text", "")).split())[:per_file_chars]
        lines.append(f"  • {it['name']} ({it.get('mime', '')}, "
                     f"{int(it.get('n_chars', 0))} симв."
                     + (", усечён" if it.get("clipped") else "") + ")")
        if head:
            lines.append(f"    {head}")
    return "\n".join(lines)


def build_turn_messages(session: AssistantSession, *, question: str = "",
                        focus: Any = None, spec_hash: str = "",
                        has_runner: bool = True, web: Optional[bool] = None,
                        kinds: Optional[Sequence[str]] = None,
                        max_tokens: int = CONTEXT_TOKENS,
                        extra: str = "",
                        image_urls: Optional[Sequence[str]] = None
                        ) -> List[Dict[str, Any]]:
    """Сообщения одного хода: промпт архитектора + фокус + вложения + хвост.

    Порядок фиксирован: инструкция (ровно одна, iter64) → место пользователя →
    дайджест вложений → усечённый по бюджету хвост диалога (iter58) → новый
    вопрос. Фокус и вложения живут ТОЛЬКО в сборке: в сессию они не пишутся,
    поэтому вчерашнее место пользователя не всплывёт в завтрашнем ходе.

    ``image_urls`` (iter68) — data-URL картинок ЭТОГО хода: они прикрепляются к
    новому вопросу, а не к истории. Скриншот относится к тому, о чём спросили
    сейчас, и повторная отправка его в каждом ходе жгла бы бюджет впустую.
    """
    web_on = bool(session.web_enabled if web is None else web)
    prompt = architect_system_prompt(
        project=session.project, spec_hash=str(spec_hash or ""), web=web_on,
        has_runner=bool(has_runner), n_attachments=len(session.attachments),
        kinds=kinds, extra=extra)
    tail = session.context_messages(max_tokens=max_tokens)
    msgs = with_system(prompt, tail)

    head: List[Dict[str, Any]] = []
    f = normalize_focus(focus)
    if not f.is_empty():
        head.append({"role": "system", "content": FOCUS_MARK + focus_block(f)})
    files = attachments_block(session)
    if files:
        head.append({"role": "system", "content": FILES_MARK + files})
    if head:
        msgs = msgs[:1] + head + msgs[1:]

    q = str(question or "").strip()
    if q or image_urls:
        resolved = resolve_question(q, f) if q else ""
        msgs.append({"role": "user",
                     "content": llm.user_content(resolved, image_urls)})
    return msgs


# ----------------------------------------------------------------------
# Ход ассистента
# ----------------------------------------------------------------------
@dataclass
class TurnResult:
    """Итог одного хода: то, что док рисует, а демо печатает."""
    question: str = ""
    resolved: str = ""
    text: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    kind: str = ""
    tools: Tuple[str, ...] = ()
    calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    web: bool = False
    stopped_reason: str = ""
    new_patches: List[str] = field(default_factory=list)
    #: Артефакты, созданные ЭТИМ ходом (iter68): по ним док рисует график и
    #: таблицу прямо в ответе (:func:`views.turn_outputs`).
    new_artifacts: List[str] = field(default_factory=list)
    #: Изображения, приложенные к вопросу (sha256 вложений сессии).
    images: List[str] = field(default_factory=list)
    #: Картинки, которые НЕ удалось приложить, с причинами — показываем прямо
    #: пользователю: иначе «ничего не вижу на скриншоте» необъяснимо (A0.6).
    image_errors: List[str] = field(default_factory=list)
    ok: bool = True
    error: str = ""
    duration_s: float = 0.0

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def kind_label(self) -> str:
        return KIND_LABEL.get(self.kind, self.kind)


def _image_urls(session: AssistantSession, root: str, project: str,
                image_ids: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Ссылки на вложения → data-URL для запроса + список НЕудач (iter68).

    Отказ по одной картинке не отменяет ход: остальные уходят, а причина
    возвращается вызывающему, чтобы он сказал о ней модели и человеку. Молча
    выкинуть изображение нельзя — ответ «на скриншоте ничего нет» был бы
    необъясним (A0.6).
    """
    from .files import AttachmentError, attachment_data_url

    urls: List[str] = []
    errors: List[str] = []
    for ident in image_ids:
        if not (root and project):
            errors.append(f"{ident}: сессия не привязана к проекту, файл не найти")
            continue
        try:
            urls.append(attachment_data_url(session, root, ident, project=project))
        except (AttachmentError, OSError) as exc:
            errors.append(f"{ident}: {exc}")
    return urls, errors


def run_turn(session: AssistantSession, ctx: Any, question: str, *,
             focus: Any = None, spec_hash: str = "",
             has_runner: Optional[bool] = None, web: Optional[bool] = None,
             model: Optional[str] = None, key: Optional[str] = None,
             kinds: Optional[Sequence[str]] = None,
             max_tokens: int = CONTEXT_TOKENS,
             transport: Optional[Callable[..., Dict[str, Any]]] = None,
             on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
             persist: bool = True, images: Optional[Sequence[str]] = None,
             **loop_kw: Any) -> TurnResult:
    """Провести ход ассистента ПО МЕСТУ и записать его в сессию.

    Одна точка для дока, демо и теста. Что здесь происходит и почему:

    * в сессию попадает ВОПРОС ЧЕЛОВЕКА как есть («объясни эту ось»), а модели
      уходит разрешённый по фокусу («объясни ось DINP»): история не должна
      переписывать сказанное человеком, а модель не должна гадать;
    * каждый вызов инструмента пишется и в сессию, и в ``tool_calls.jsonl``
      проекта — разбор обязан воспроизводиться через неделю (§3.7);
    * отказ модели/сети НЕ роняет док: ответ заменяется человекочитаемым
      сообщением об ошибке, ход помечается ``ok=False`` (A0.6);
    * ``spec_hash`` берётся ИЗВНЕ (его знает вызывающий, у него есть спека) —
      контекст не лезет в ядро сам;
    * ``images`` (iter68) — приложенные к ЭТОМУ вопросу изображения (sha256
      вложений сессии). В сессию пишутся ссылки, в запрос — data-URL: base64
      в переписке раздул бы и файл проекта, и оценку бюджета контекста.
    """
    from .tools import AGENT_KINDS, tool_specs
    from .tools.registry import dispatcher
    from .store import append_log

    kinds = tuple(kinds) if kinds is not None else tuple(AGENT_KINDS)
    if has_runner is None:
        has_runner = getattr(ctx, "runner", None) is not None
    f = normalize_focus(focus)
    q = str(question or "").strip()
    image_ids = [str(x) for x in (images or []) if str(x or "").strip()]
    if not q and not image_ids:
        raise ValueError("Пустой вопрос: ходить к модели не с чем.")
    if not q:
        # Картинка без вопроса — это всё-таки вопрос («посмотри»), но модель
        # не должна угадывать, что от неё хотят: спрашиваем явно.
        q = "Посмотри на приложенное изображение и скажи, что видишь."
    resolved = resolve_question(q, f)

    session.add_message("user", q, images=image_ids)
    root = str(getattr(ctx, "root", "") or "")
    project = str(getattr(ctx, "project", "") or session.project)
    image_urls, image_errors = _image_urls(session, root, project, image_ids)
    msgs = build_turn_messages(session, question="", focus=f,
                               spec_hash=spec_hash, has_runner=has_runner,
                               web=web, kinds=kinds, max_tokens=max_tokens)
    if image_urls:
        # Хвост истории уже содержит текст вопроса (add_message выше), поэтому
        # картинки прикрепляем К НЕМУ, а не отдельным сообщением: провайдеры
        # ждут текст и изображение в одном user-сообщении.
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["content"] = llm.user_content(str(m.get("content", "")),
                                                image_urls)
                break
    if image_errors:
        # A0.6: картинка, которую не удалось приложить, не должна исчезнуть
        # молча — иначе ответ «на скриншоте ничего не вижу» необъясним.
        msgs.append({"role": "system",
                     "content": FILES_MARK + "НЕ УДАЛОСЬ приложить изображения: "
                     + "; ".join(image_errors) +
                     ". Скажи об этом пользователю, не описывай картинку по "
                     "догадке."})
    if resolved != q:
        # Честная пометка: в истории остаётся сказанное человеком, а модель
        # видит, чем именно «эта ось» была на экране.
        msgs.append({"role": "system",
                     "content": FOCUS_MARK + f"вопрос пользователя относится к "
                                             f"фокусу: «{resolved}»"})

    before = {p.id for p in session.patches}
    before_art = {a.id for a in session.artifacts}

    def _audit(rec: Dict[str, Any]) -> None:
        session.add_tool_call(ToolCall(
            tool=str(rec.get("tool", "")), args=dict(rec.get("args", {}) or {}),
            ok=bool(rec.get("ok", True)), error=str(rec.get("error", "")),
            duration_s=float(rec.get("duration_s", 0.0) or 0.0),
            summary=str(rec.get("summary", ""))))
        if root and project:
            try:
                append_log(root, project, "tool_calls", dict(rec))
            except (OSError, ValueError):
                pass          # журнал не должен ронять ответ пользователю

    t0 = time.monotonic()
    res = TurnResult(question=q, resolved=resolved, web=bool(
        session.web_enabled if web is None else web),
        images=list(image_ids), image_errors=list(image_errors))
    r = route(resolved)
    res.kind, res.tools = r.kind, tuple(r.tools)
    try:
        out = llm.run_tool_loop(
            msgs, dispatch=dispatcher(ctx, allowed_kinds=kinds, on_call=_audit),
            tools=tool_specs(kinds), model=model, key=key,
            web=bool(session.web_enabled if web is None else web),
            transport=transport, on_event=on_event, **loop_kw)
    except Exception as exc:                              # noqa: BLE001
        res.ok = False
        res.error = f"{type(exc).__name__}: {exc}"
        res.text = (f"⚠️ Ход не выполнен: {res.error}\n\n"
                    f"Вопрос сохранён в сессии — повторите его, когда причина "
                    f"устранена (ключ, сеть, модель).")
    else:
        res.text = out.text
        res.calls = list(out.calls)
        res.usage = dict(out.usage)
        res.model = out.model
        res.web = bool(out.web)
        res.stopped_reason = out.stopped_reason
    res.duration_s = round(time.monotonic() - t0, 3)
    res.sections = parse_sections(res.text)

    session.add_message("assistant", res.text, model=res.model, web=res.web,
                        usage=res.usage)
    res.new_patches = [p.id for p in session.patches if p.id not in before]
    res.new_artifacts = [a.id for a in session.artifacts
                         if a.id not in before_art]

    if persist and root and project:
        from .store import save_session
        try:
            save_session(session, root, project)
        except (OSError, ValueError):
            pass              # сохранение — не причина потерять ответ
    return res


# ----------------------------------------------------------------------
# Кнопки человека (класс write, iter63)
# ----------------------------------------------------------------------
def human_apply(ctx: Any, patch_id: str, *, note: str = "", author: str = "",
                ttl_s: Optional[float] = None) -> Dict[str, Any]:
    """Применить патч ОТ ИМЕНИ ЧЕЛОВЕКА: выдать разовый токен и погасить его.

    Это и есть «кнопка Применить». Токен выдаётся здесь же и живёт доли
    секунды — привязан к действию, к патчу и к текущему ``spec_hash``: между
    нажатием и применением геометрия измениться не успевает, а модель до
    этого пути не дотягивается (класс ``write`` ей не выдан).
    """
    from .tools import WRITE, dispatch
    from .tools.write import issue_apply_token

    token = issue_apply_token(ctx, patch_id, ttl_s=ttl_s, note=note)
    return dispatch(ctx, "apply_patch",
                    {"patch_id": patch_id, "human_token": token,
                     "note": note, "author": author},
                    allowed_kinds=[WRITE])


def human_apply_spec(ctx: Any, spec_id: str, *, note: str = "",
                     author: str = "", ttl_s: Optional[float] = None
                     ) -> Dict[str, Any]:
    """Применить ПАКЕТ спеки от имени человека — кнопка «Применить» (iter71).

    Тот же путь, что :func:`human_apply`, но для геометрии целиком: первичный
    ввод спеки и её эволюция (добавленный/удалённый узел, смена роли). Токен
    выдаётся здесь же, живёт доли секунды и привязан к отпечатку спеки на
    момент нажатия; модели этот путь недоступен (класс ``write`` ей не выдан).
    """
    from .tools import WRITE, dispatch
    from .tools.write import issue_apply_spec_token

    token = issue_apply_spec_token(ctx, spec_id, ttl_s=ttl_s, note=note)
    return dispatch(ctx, "apply_spec",
                    {"spec_id": spec_id, "human_token": token,
                     "note": note, "author": author},
                    allowed_kinds=[WRITE])


def human_reject_spec(ctx: Any, spec_id: str, reason: str, *,
                      author: str = "", ttl_s: Optional[float] = None
                      ) -> Dict[str, Any]:
    """Отклонить пакет спеки от имени человека — с записью в журнал решений."""
    from .tools import WRITE, dispatch
    from .tools.write import issue_reject_spec_token

    token = issue_reject_spec_token(ctx, spec_id, ttl_s=ttl_s)
    return dispatch(ctx, "reject_spec",
                    {"spec_id": spec_id, "human_token": token,
                     "reason": reason, "author": author},
                    allowed_kinds=[WRITE])


def human_reject(ctx: Any, patch_id: str, reason: str, *, author: str = "",
                 ttl_s: Optional[float] = None) -> Dict[str, Any]:
    """Отклонить патч от имени человека — с записью в журнал решений.

    Отказ фиксируется наравне с применением (iter63): спор «почему тогда не
    расширили границу» разрешает журнал, а не память участников.
    """
    from .tools import WRITE, dispatch
    from .tools.write import issue_reject_token

    token = issue_reject_token(ctx, patch_id, ttl_s=ttl_s)
    return dispatch(ctx, "reject_patch",
                    {"patch_id": patch_id, "human_token": token,
                     "reason": reason, "author": author},
                    allowed_kinds=[WRITE])
