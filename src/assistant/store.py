"""assistant/store.py — персистентность сессии ассистента в каталоге проекта.

Раскладка (ASSISTANT_SPEC):

    project_campaigns/<проект>/
        campaign.json          # состояние движка (campaign_state, не трогаем)
        assistant/
            session.json       # сессия: переписка, вложения, патчи, аудит
            files/             # приложенные файлы как есть
            artifacts/         # выхлоп песочницы
            tool_calls.jsonl   # аудит вызовов инструментов
            decision_log.jsonl # принятые решения компании (ADR)
            local_facts.jsonl  # L1-факты (добавляет ЧЕЛОВЕК)

Почему рядом, а не внутри ``campaign.json``: сохранение/загрузка кампании
(`campaign_state.save_campaign`) не должны знать про ассистента, а удаление
проекта (`delete_campaign` — rmtree каталога) обязано уносить переписку и
вложения. Каталог даёт и то, и другое без правок движка.

Инварианты:
  * загрузка НЕсуществующей сессии — ПУСТАЯ сессия проекта, не исключение
    (старые проекты открываются как раньше);
  * запись атомарна (tmp + replace): прерванное сохранение не оставляет
    полусериализованный JSON вместо переписки;
  * журналы — append-only jsonl; битая строка при чтении ПРОПУСКАЕТСЯ (одна
    сломанная запись аудита не должна ронять открытие проекта).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .session import AssistantSession, new_session

ASSISTANT_DIRNAME = "assistant"
SESSION_FILE = "session.json"
FILES_DIRNAME = "files"
ARTIFACTS_DIRNAME = "artifacts"

#: Журналы: логическое имя → файл. ``tool_calls`` — аудит вызовов (§3.3),
#: ``decisions`` — решения компании (ADR), ``local_facts`` — L1-знание.
LOG_FILES = {
    "tool_calls": "tool_calls.jsonl",
    "decisions": "decision_log.jsonl",
    "local_facts": "local_facts.jsonl",
}


def _validate_project(name: str) -> str:
    """Имя проекта: те же правила, что у ``campaign_state._validate_name``.

    Дублируется намеренно (4 строки): модуль хранения сессии не тянет за
    собой движок кампании (numpy/sklearn) ради одной проверки.
    """
    name = (name or "").strip()
    if not name or name in (".", "..") or any(s in name for s in ("/", "\\")):
        raise ValueError(f"Недопустимое имя проекта: {name!r}")
    return name


# ----------------------------------------------------------------------
# Пути
# ----------------------------------------------------------------------
def assistant_dir(root: str | Path, project: str) -> Path:
    """Каталог ассистента проекта (может ещё не существовать)."""
    return Path(root) / _validate_project(project) / ASSISTANT_DIRNAME


def session_path(root: str | Path, project: str) -> Path:
    return assistant_dir(root, project) / SESSION_FILE


def files_dir(root: str | Path, project: str) -> Path:
    return assistant_dir(root, project) / FILES_DIRNAME


def artifacts_dir(root: str | Path, project: str) -> Path:
    return assistant_dir(root, project) / ARTIFACTS_DIRNAME


def log_path(root: str | Path, project: str, kind: str) -> Path:
    if kind not in LOG_FILES:
        raise ValueError(f"Неизвестный журнал {kind!r}: "
                         f"допустимы {sorted(LOG_FILES)}.")
    return assistant_dir(root, project) / LOG_FILES[kind]


def ensure_dirs(root: str | Path, project: str) -> Path:
    """Создать каталоги ассистента проекта (идемпотентно)."""
    base = assistant_dir(root, project)
    (base / FILES_DIRNAME).mkdir(parents=True, exist_ok=True)
    (base / ARTIFACTS_DIRNAME).mkdir(parents=True, exist_ok=True)
    return base


# ----------------------------------------------------------------------
# Сессия
# ----------------------------------------------------------------------
def save_session(session: AssistantSession, root: str | Path,
                 project: Optional[str] = None) -> str:
    """Сохранить сессию в ``root/<project>/assistant/session.json``.

    ``project`` по умолчанию — из самой сессии. Запись атомарная: сначала
    ``session.json.tmp``, затем ``os.replace`` (при обрыве останется прежняя
    переписка, а не обрубок).
    """
    project = _validate_project(project or session.project)
    session.project = project
    ensure_dirs(root, project)
    path = session_path(root, project)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(session.to_state(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, path)
    return str(path)


def load_session(root: str | Path, project: str) -> AssistantSession:
    """Загрузить сессию проекта; если её нет — ПУСТАЯ сессия (не ошибка).

    Битый ``session.json`` — тоже не повод рушить открытие проекта: файл
    сохраняется рядом как ``session.corrupt.json`` (чтобы не потерять), а
    работа продолжается с пустой сессией. Молча удалять переписку нельзя
    (A0.6).
    """
    project = _validate_project(project)
    path = session_path(root, project)
    if not path.exists():
        return new_session(project)
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return AssistantSession.from_state(state)
    except (ValueError, OSError):
        try:
            path.replace(path.with_name("session.corrupt.json"))
        except OSError:
            pass
        s = new_session(project)
        s.add_message(
            "system",
            "⚠️ Прежняя сессия ассистента не прочиталась и сохранена как "
            "assistant/session.corrupt.json — переписка начата заново.")
        return s


def session_exists(root: str | Path, project: str) -> bool:
    return session_path(root, project).exists()


def delete_session(root: str | Path, project: str) -> bool:
    """Удалить ТОЛЬКО файл сессии (файлы и журналы остаются)."""
    path = session_path(root, project)
    if not path.exists():
        return False
    path.unlink()
    return True


# ----------------------------------------------------------------------
# Журналы (append-only jsonl)
# ----------------------------------------------------------------------
def append_log(root: str | Path, project: str, kind: str,
               record: Dict[str, Any]) -> str:
    """Дописать запись в журнал (одна строка = один JSON-объект).

    Append-only: журнал не переписывается целиком, поэтому одновременная
    работа приложения и внешнего читателя (MCP-сервер, iter66) безопасна.
    """
    ensure_dirs(root, project)
    path = log_path(root, project, kind)
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return str(path)


def read_log(root: str | Path, project: str, kind: str, *,
             limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Прочитать журнал. Битые строки ПРОПУСКАЮТСЯ (не роняют загрузку).

    ``limit`` — вернуть последние N записей (для показа хвоста в UI).
    """
    path = log_path(root, project, kind)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except ValueError:
                continue          # битая строка — пропуск, а не отказ
            if isinstance(rec, dict):
                out.append(rec)
    if limit is not None and limit > 0:
        return out[-int(limit):]
    return out


def append_logs(root: str | Path, project: str, kind: str,
                records: Iterable[Dict[str, Any]]) -> int:
    n = 0
    for rec in records or []:
        append_log(root, project, kind, rec)
        n += 1
    return n
