"""assistant/views.py — ЧИСТЫЕ таблицы показа сессии (без Streamlit).

Один источник представлений для дока в интерфейсе (iter65), демо-скрипта
(`run_assistant_demo.py`) и будущего MCP-читателя (iter66). Разъехавшиеся
таблицы врали бы пользователю про одно и то же состояние, поэтому формат
здесь один, а вызывающий лишь рисует.

Канон (`.clinerules`): UI-хелперы — чистые функции, тестируемые напрямую.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from .session import (AssistantSession, PATCH_APPLIED, PATCH_REJECTED,
                      PATCH_STAGED, estimate_tokens)

#: Сколько символов сообщения показывать в таблице ленты (полный текст —
#: в самом чате; таблица нужна для обзора, а не для чтения).
PREVIEW_CHARS = 160

_ROLE_LABEL = {"user": "👤 пользователь", "assistant": "🤖 ассистент",
               "system": "⚙️ система", "tool": "🔧 инструмент"}

_STATUS_LABEL = {PATCH_STAGED: "⏳ предложен",
                 PATCH_APPLIED: "✅ применён",
                 PATCH_REJECTED: "⛔ отклонён"}


def _short(text: Any, n: int = PREVIEW_CHARS) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _hhmm(ts: str) -> str:
    """``2026-08-08T15:04:05+00:00`` → ``15:04`` (дата в таблице не нужна)."""
    s = str(ts or "")
    return s[11:16] if len(s) >= 16 else s


# ----------------------------------------------------------------------
# Таблицы
# ----------------------------------------------------------------------
def messages_dataframe(session: AssistantSession) -> pd.DataFrame:
    """Лента диалога: время / роль / веб / модель / текст (усечён) / токены."""
    rows: List[Dict[str, Any]] = []
    for m in session.messages:
        rows.append({
            "время": _hhmm(m.ts),
            "роль": _ROLE_LABEL.get(m.role, m.role),
            "🌐": "да" if m.web else "",
            "модель": m.model,
            "сообщение": _short(m.content),
            "инструментов": len(m.tool_calls),
            "~токенов": estimate_tokens(m.content),
        })
    return pd.DataFrame(rows, columns=["время", "роль", "🌐", "модель",
                                       "сообщение", "инструментов", "~токенов"])


def attachments_dataframe(session: AssistantSession) -> pd.DataFrame:
    """Приложенные файлы: имя / тип / размер / символов текста / усечён / хеш.

    У изображения (iter68) в колонке «символов» стоит не 0, а «—»: ноль читался
    бы как «файл не прочитался», тогда как текста там нет ПО ЗАМЫСЛУ — картинку
    смотрит сама модель.
    """
    rows: List[Dict[str, Any]] = []
    for a in session.attachments:
        is_image = str(a.mime or "").startswith("image/")
        rows.append({
            "файл": a.name,
            "тип": a.mime or "—",
            "размер, КБ": round(a.size / 1024.0, 1) if a.size else 0.0,
            "символов": "— (картинка)" if is_image else int(a.n_chars),
            "усечён": "да" if a.truncated else "",
            "sha256": a.sha256[:12],
            "примечание": _short(a.note, 60),
        })
    return pd.DataFrame(rows, columns=["файл", "тип", "размер, КБ", "символов",
                                       "усечён", "sha256", "примечание"])


def staged_patches_dataframe(session: AssistantSession, *,
                             only_staged: bool = False) -> pd.DataFrame:
    """Патчи спеки в формате ответа архитектора (`## PATCH`).

    Колонка «хеш» отвечает на главный вопрос ревизора: поедет ли отпечаток
    спеки (а значит, геометрия плана) после применения.
    """
    rows: List[Dict[str, Any]] = []
    for p in session.patches:
        if only_staged and p.status != PATCH_STAGED:
            continue
        rows.append({
            "id": p.id,
            "узел": p.node,
            "поле": p.field_name,
            "было": _short(p.from_value, 40),
            "стало": _short(p.to_value, 40),
            "граница": p.bound_type or "—",
            "знание": p.level or "—",
            "уверенность": p.confidence or "—",
            "хеш": "⚠️ меняется" if p.affects_hash else "не меняется",
            "статус": _STATUS_LABEL.get(p.status, p.status),
            "обоснование": _short(p.rationale, 90),
        })
    return pd.DataFrame(rows, columns=["id", "узел", "поле", "было", "стало",
                                       "граница", "знание", "уверенность",
                                       "хеш", "статус", "обоснование"])


def artifacts_dataframe(session: AssistantSession) -> pd.DataFrame:
    """Артефакты песочницы: время / имя / вид / инструмент / подпись."""
    rows = [{
        "время": _hhmm(a.ts),
        "артефакт": a.name,
        "вид": a.kind,
        "инструмент": a.tool or "—",
        "подпись": _short(a.caption, 80),
    } for a in session.artifacts]
    return pd.DataFrame(rows, columns=["время", "артефакт", "вид",
                                       "инструмент", "подпись"])


@dataclass
class OutputFile:
    """Файл, созданный прогоном песочницы, — для показа в UI (iter68).

    ``kind`` решает, ЧЕМ рисовать: ``image`` → картинкой, ``table`` → таблицей,
    ``text`` → сворачиваемым блоком. Определяется в
    :func:`assistant.sandbox.output_kind` по расширению — содержимое писал код
    модели, других признаков у нас нет.
    """
    name: str
    kind: str
    path: str
    size: int = 0
    #: Инструмент, который создал файл (`run_python`) — видно происхождение.
    tool: str = ""

    @property
    def caption(self) -> str:
        label = {"image": "🖼 график", "table": "📊 таблица"}.get(
            self.kind, "📄 файл")
        return f"{label} · {self.name} · {self.size / 1024.0:.1f} КБ"


#: Виды артефактов, которые UI умеет РИСОВАТЬ (а не только назвать).
SHOWABLE_KINDS = ("image", "table")


def outputs_from_artifacts(artifacts: Sequence[Any]) -> List[OutputFile]:
    """Артефакты → файлы для показа (график/таблица), в порядке появления.

    Источник — артефакты сессии, а не результаты вызовов: в аудит вызова
    (`llm.run_tool_loop`) попадает лишь короткая сводка, а полный путь к файлу
    знает :func:`assistant.tools.sandbox_tools.collect_outputs`, который его же
    и записал в сессию. Так UI не зависит от формата ответа инструмента.

    Пропавший с диска файл в список НЕ попадает: заголовок «график» без
    картинки хуже, чем отсутствие заголовка.
    """
    out: List[OutputFile] = []
    for a in artifacts or []:
        kind = str(getattr(a, "kind", "") or "")
        path = str(getattr(a, "path", "") or "")
        if kind not in SHOWABLE_KINDS or not path or not os.path.exists(path):
            continue
        out.append(OutputFile(name=str(getattr(a, "name", "")), kind=kind,
                              path=path, size=_file_size(path),
                              tool=str(getattr(a, "tool", ""))))
    return out


def turn_outputs(session: AssistantSession, new_artifact_ids: Sequence[str]
                 ) -> List[OutputFile]:
    """Файлы, созданные ЭТИМ ходом: то, что док рисует прямо в ответе.

    Нужна, чтобы посчитанная кривая появлялась в разговоре, а не только в
    отдельной панели артефактов проекта: график, который надо искать, для
    обсуждения почти бесполезен.
    """
    ids = set(str(i) for i in (new_artifact_ids or []))
    return outputs_from_artifacts([a for a in session.artifacts
                                   if str(a.id) in ids])


def artifact_outputs(session: AssistantSession, *, limit: int = 6
                     ) -> List[OutputFile]:
    """Последние показуемые артефакты сессии (график/таблица) для панели.

    Работает и после перезапуска приложения: артефакты живут в проекте, а не в
    памяти процесса.
    """
    shown = outputs_from_artifacts(list(reversed(session.artifacts)))
    return shown[: max(1, int(limit))]


def _file_size(path: str) -> int:
    try:
        return int(os.path.getsize(path))
    except OSError:
        return 0


def tool_calls_dataframe(calls: Sequence[Any]) -> pd.DataFrame:
    """Аудит вызовов: принимает и объекты :class:`ToolCall`, и словари журнала.

    Один формат на оба источника: в сессии вызовы лежат объектами, в
    ``tool_calls.jsonl`` — словарями, а таблица нужна одна и та же.
    """
    rows: List[Dict[str, Any]] = []
    for c in calls or []:
        d = c if isinstance(c, Mapping) else c.to_state()
        rows.append({
            "время": _hhmm(d.get("ts", "")),
            "инструмент": d.get("tool", ""),
            "аргументы": _short(d.get("args", {}), 60),
            "итог": "ok" if d.get("ok", True) else "ошибка",
            "с": round(float(d.get("duration_s", 0.0) or 0.0), 2),
            "результат": _short(d.get("error") or d.get("summary", ""), 90),
        })
    return pd.DataFrame(rows, columns=["время", "инструмент", "аргументы",
                                       "итог", "с", "результат"])


def decisions_dataframe(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Журнал решений компании (ADR) → таблица.

    Решение фиксируется вместе с ``spec_hash`` на момент принятия: без него
    «почему так решили» невозможно сопоставить с геометрией кампании.
    """
    rows: List[Dict[str, Any]] = []
    for r in records or []:
        rows.append({
            "дата": str(r.get("ts", ""))[:10],
            "решение": _short(r.get("title") or r.get("decision", ""), 70),
            "узлы": ", ".join(r.get("nodes", []) or []) or "—",
            "кто": r.get("author", "—"),
            "spec_hash": str(r.get("spec_hash", ""))[:12] or "—",
            "обоснование": _short(r.get("rationale", ""), 90),
        })
    return pd.DataFrame(rows, columns=["дата", "решение", "узлы", "кто",
                                       "spec_hash", "обоснование"])


def consents_dataframe(consents: Sequence[Any]) -> pd.DataFrame:
    """Выданные подтверждения человека (iter63): что именно разрешено и до когда.

    Пользователь должен видеть, на что он «нажал кнопку»: подтверждение
    одноразовое, привязано к действию, цели и отпечатку спеки.
    """
    rows: List[Dict[str, Any]] = []
    for c in consents or []:
        d = c if isinstance(c, Mapping) else c.to_state()
        rows.append({
            "действие": d.get("action", ""),
            "цель": _short(d.get("target", ""), 40),
            "при spec_hash": str(d.get("context_hash", ""))[:12] or "—",
            "живёт, с": round(float(d.get("ttl_s", 0.0) or 0.0)),
            "использован": "да" if float(d.get("used_at", 0) or 0) > 0 else "",
            "токен": str(d.get("token", ""))[:6] + "…",
        })
    return pd.DataFrame(rows, columns=["действие", "цель", "при spec_hash",
                                       "живёт, с", "использован", "токен"])


def scenarios_dataframe(scenarios: Optional[Sequence[Any]] = None
                        ) -> pd.DataFrame:
    """Golden-сценарии маршрутизации (iter64, ASSISTANT_SPEC §8) → таблица.

    Показывает, чем ассистент ОБЯЗАН закрывать типовой вопрос технолога:
    список инструментов здесь — контракт, а не пожелание (тест сверяет с ним
    фактический ход).
    """
    from .prompts import GOLDEN_SCENARIOS  # локально: prompts тянет реестр

    rows: List[Dict[str, Any]] = []
    for sc in (scenarios if scenarios is not None else GOLDEN_SCENARIOS):
        rows.append({
            "№": sc.id,
            "сценарий": sc.title,
            "маршрут": sc.label,
            "инструменты": ", ".join(sc.tools) or "— (кнопка человека)",
            "нельзя": ", ".join(sc.forbidden),
            "правило": _short(sc.rule, 90),
        })
    return pd.DataFrame(rows, columns=["№", "сценарий", "маршрут",
                                       "инструменты", "нельзя", "правило"])


def routing_dataframe(reports: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Итог сверки ходов с golden-маршрутами (:func:`prompts.check_routing`).

    Отдельно называются ДВА разных провала: «не вызваны инструменты» (ответ по
    памяти) и «тронуто запрещённое» (модель полезла применять сама) — лечатся
    они по-разному.
    """
    rows: List[Dict[str, Any]] = []
    for r in reports or []:
        rows.append({
            "№": r.get("id", ""),
            "сценарий": r.get("scenario", ""),
            "маршрут": r.get("kind", ""),
            "вызвано": ", ".join(r.get("called", []) or []) or "—",
            "не хватает": ", ".join(r.get("missing", []) or []) or "—",
            "запрещённое": ", ".join(r.get("forbidden_used", []) or []) or "—",
            "итог": "✅ верно" if r.get("ok") else "⛔ маршрут нарушен",
        })
    return pd.DataFrame(rows, columns=["№", "сценарий", "маршрут", "вызвано",
                                       "не хватает", "запрещённое", "итог"])


def suggestions_dataframe(suggestions: Sequence[Any]) -> pd.DataFrame:
    """Подсказки «спросить по месту» (iter65) → таблица.

    Выключенная подсказка ОСТАЁТСЯ в таблице с причиной: исчезнувшая кнопка
    читалась бы как «здесь так спрашивать нельзя», хотя не хватает всего лишь
    выбранного узла.
    """
    rows: List[Dict[str, Any]] = []
    for s in suggestions or []:
        d = s if isinstance(s, Mapping) else {
            "label": s.label, "question": s.question, "kind": s.kind_label,
            "tools": s.tools, "enabled": s.enabled, "why": s.why}
        rows.append({
            "кнопка": d.get("label", ""),
            "вопрос": _short(d.get("question", ""), 70),
            "маршрут": d.get("kind", ""),
            "инструменты": ", ".join(d.get("tools", []) or []) or "—",
            "доступна": "да" if d.get("enabled", True) else "нет",
            "почему": _short(d.get("why", ""), 70) or "—",
        })
    return pd.DataFrame(rows, columns=["кнопка", "вопрос", "маршрут",
                                       "инструменты", "доступна", "почему"])


# ----------------------------------------------------------------------
# Подписи
# ----------------------------------------------------------------------
def turn_caption(res: Any) -> str:
    """Одна строка об итоге хода ассистента (iter65): маршрут, вызовы, патчи.

    Показывается под ответом в доке: пользователь должен видеть, ЧЕМ был
    закрыт его вопрос (инструментами ядра или разговором) и появилось ли что-то
    в панели патчей — иначе предложение легко не заметить.
    """
    if res is None:
        return "хода не было"
    get = (res.get if isinstance(res, Mapping) else lambda k, d=None:
           getattr(res, k, d))
    parts = [f"маршрут: {get('kind_label', '') or get('kind', '')}",
             f"вызовов инструментов: {len(get('calls', []) or [])}",
             f"{float(get('duration_s', 0.0) or 0.0):.1f} с"]
    if get("web", False):
        parts.append("🌐 веб включён (уровень знания L2)")
    new_patches = list(get("new_patches", []) or [])
    if new_patches:
        parts.append(f"новых патчей в стейдже: {len(new_patches)} — "
                     f"применяет ЧЕЛОВЕК кнопкой")
    reason = str(get("stopped_reason", "") or "")
    if reason and reason != "final":
        parts.append(f"ход прерван: {reason}")
    if not get("ok", True):
        parts.append(f"⛔ {get('error', '')}")
    sections = list((get("sections", {}) or {}))
    if sections:
        parts.append("разделы: " + ", ".join(sections))
    return " · ".join(parts)


#: Сколько символов причины показывать во всплывающем предупреждении: тост
#: живёт секунды, полный текст ошибки остаётся в ленте и в сессии.
TOAST_CHARS = 220

#: Подсказки по классам отказов. Обрыв связи и лимит частоты лечатся ПОВТОРОМ,
#: неверный ключ / модель / пустой счёт — нет, и предлагать «попробуйте снова»
#: в этих случаях значит гонять человека по кругу.
_RETRY_HINTS = (
    ("Сетевая ошибка", True,
     "связь с OpenRouter оборвалась — обычно помогает повтор"),
    ("10054", True, "соединение разорвано на стороне сети — повторите отправку"),
    ("HTTP 429", True, "слишком часто: подождите несколько секунд и повторите"),
    ("HTTP 5", True, "сбой на стороне OpenRouter — повтор обычно проходит"),
    ("timed out", True, "модель не ответила за отведённое время"),
    ("HTTP 401", False, "ключ не принят — проверьте OPENROUTER_API_KEY"),
    ("HTTP 402", False, "на счёте OpenRouter нет средств — повтор не поможет"),
    ("HTTP 404", False, "такой модели нет — проверьте имя и суффикс ':online'"),
    ("OPENROUTER_API_KEY", False, "ключ не задан — укажите его в панели «Модель и ключ»"),
)


@dataclass
class RetryPrompt:
    """Что показать пользователю после НЕУДАЧНОГО хода (iter67).

    Транспорт ассистента делает одну попытку, и разовый обрыв TLS
    (``WinError 10054``) превращался в тупик: ответа нет, а повторить вопрос
    можно только перенабрав его руками. Решение — не молчаливый авторетрай
    (человек не должен платить деньгами и временем за скрытые попытки), а
    ЯВНАЯ кнопка: причина видна, повтор — осознанное действие (A0.6).

    * ``show`` — был ли отказ вообще (успешный ход ничего не рисует);
    * ``question`` — что именно переотправить (слова человека, как сказаны);
    * ``retryable`` — лечится ли отказ повтором; если нет, кнопка остаётся
      доступной (запрещать не наше дело), но подпись честно предупреждает.
    """
    show: bool = False
    question: str = ""
    toast: str = ""
    icon: str = "⚠️"
    button_label: str = "🔄 Повторить отправку"
    retryable: bool = True
    hint: str = ""

    def __bool__(self) -> bool:            # `if prompt:` читается как «есть отказ»
        return bool(self.show)


def retry_prompt(res: Any) -> RetryPrompt:
    """Итог хода → данные для тоста и кнопки повтора (чистая, без Streamlit).

    Успешный ход (или его отсутствие) даёт ``show=False``: кнопка не должна
    висеть после нормального ответа.
    """
    if res is None:
        return RetryPrompt()
    get = (res.get if isinstance(res, Mapping) else lambda k, d=None:
           getattr(res, k, d))
    if get("ok", True):
        return RetryPrompt()

    error = str(get("error", "") or "").strip()
    question = str(get("question", "") or "").strip()
    retryable, hint = True, "повторите отправку того же вопроса"
    for needle, flag, text in _RETRY_HINTS:
        if needle.lower() in error.lower():
            retryable, hint = flag, text
            break

    toast = f"Ответ не получен: {_short(error, TOAST_CHARS) or 'причина неизвестна'}"
    return RetryPrompt(show=True, question=question, toast=toast,
                       icon="⚠️" if retryable else "⛔",
                       button_label="🔄 Повторить отправку" if retryable
                       else "🔄 Отправить ещё раз (причина вряд ли уйдёт сама)",
                       retryable=retryable, hint=hint)


def apply_result_caption(result: Mapping[str, Any]) -> str:
    """Одна строка об итоге применения патча (iter63).

    Сдвиг ``spec_hash`` называется вслух: после него уже собранные точки
    относятся к ПРЕЖНЕЙ геометрии, и молчать об этом нельзя.
    """
    if not result:
        return "патч не применён"
    before = str(result.get("spec_hash_before", ""))[:12]
    after = str(result.get("spec_hash_after", ""))[:12]
    n = len(result.get("changed_intervals", []) or [])
    parts = [f"патч {result.get('patch_id', '—')} · {result.get('status', '')}",
             f"границ изменилось: {n}"]
    parts.append(f"отпечаток: {before}… → {after}…" if result.get("affects_hash")
                 else f"отпечаток не изменился ({before}…)")
    gates = result.get("gates", {}) or {}
    pts = (gates.get("points", {}) or {})
    if pts.get("checked"):
        parts.append(f"точек проверено: {pts.get('n_checked', 0)}, "
                     f"выпало: {pts.get('n_lost', 0)}")
    pre = (gates.get("preflight", {}) or {})
    if pre.get("checked"):
        parts.append("preflight: "
                     + ("не ухудшился" if pre.get("ok") else "УХУДШИЛСЯ"))
    return " · ".join(parts)


def session_caption(session: AssistantSession) -> str:
    """Одна строка о состоянии сессии — заголовок дока и вывод демо."""
    staged = len(session.staged_patches())
    parts = [
        f"проект: **{session.project or '—'}**",
        f"сообщений: {len(session.messages)}",
        f"файлов: {len(session.attachments)}",
        f"патчей в стейдже: {staged}",
        f"артефактов: {len(session.artifacts)}",
        f"вызовов инструментов: {len(session.tool_calls)}",
    ]
    total = int((session.usage or {}).get("total_tokens", 0))
    if total:
        parts.append(f"токенов: {total}")
    parts.append(f"модель: `{session.model or 'по умолчанию'}`")
    parts.append("🌐 интернет: " + ("включён" if session.web_enabled else "выключен"))
    return " · ".join(parts)


def context_caption(session: AssistantSession, *, max_tokens: int = 24000) -> str:
    """Подпись «что уходит в модель»: сколько сообщений и было ли усечение."""
    ctx = session.context_messages(max_tokens=max_tokens)
    omitted = sum(1 for m in ctx if m.get("role") == "system"
                  and str(m.get("content", "")).startswith("[сессия]"))
    n_dialog = len(ctx) - omitted
    txt = (f"в модель уходит сообщений: {n_dialog} из {len(session.messages)} "
           f"(бюджет {max_tokens} токенов)")
    if omitted:
        txt += " — ранние сообщения опущены, полная переписка сохранена в проекте"
    return txt


def attachment_digest(session: AssistantSession, *,
                      per_file_chars: int = 2000) -> List[Dict[str, Any]]:
    """Дайджест вложений для КОНТЕКСТА модели (не для показа).

    Отдаёт по каждому файлу имя/тип/размер и НАЧАЛО извлечённого текста:
    полный документ уходит только по явному запросу инструмента чтения
    (iter59), иначе один паспорт сырья съест весь бюджет.
    """
    out: List[Dict[str, Any]] = []
    for a in session.attachments:
        text = a.text or ""
        clipped = len(text) > per_file_chars
        out.append({
            "name": a.name,
            "mime": a.mime,
            "size": int(a.size),
            "n_chars": int(a.n_chars),
            "text": text[:per_file_chars],
            "clipped": bool(clipped or a.truncated),
        })
    return out
