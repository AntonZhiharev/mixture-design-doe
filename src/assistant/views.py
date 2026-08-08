"""assistant/views.py — ЧИСТЫЕ таблицы показа сессии (без Streamlit).

Один источник представлений для дока в интерфейсе (iter65), демо-скрипта
(`run_assistant_demo.py`) и будущего MCP-читателя (iter66). Разъехавшиеся
таблицы врали бы пользователю про одно и то же состояние, поэтому формат
здесь один, а вызывающий лишь рисует.

Канон (`.clinerules`): UI-хелперы — чистые функции, тестируемые напрямую.
"""
from __future__ import annotations

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
    """Приложенные файлы: имя / тип / размер / символов текста / усечён / хеш."""
    rows: List[Dict[str, Any]] = []
    for a in session.attachments:
        rows.append({
            "файл": a.name,
            "тип": a.mime or "—",
            "размер, КБ": round(a.size / 1024.0, 1) if a.size else 0.0,
            "символов": int(a.n_chars),
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


# ----------------------------------------------------------------------
# Подписи
# ----------------------------------------------------------------------
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
