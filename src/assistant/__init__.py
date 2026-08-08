"""src/assistant — ИИ-ассистент-архитектор кампании (ASSISTANT_SPEC, iter58+).

Пакет собирает всё, что отличает помощника-архитектора от чата «поверх»
приложения:

* :mod:`session` — сессия диалога, привязанная к ПРОЕКТУ (сообщения, вложения,
  артефакты песочницы, staged-патчи спеки, расход токенов);
* :mod:`store` — персистентность сессии в каталоге проекта
  (``project_campaigns/<проект>/assistant/``) + append-only журналы аудита;
* :mod:`views` — ЧИСТЫЕ таблицы показа (без Streamlit): одни и те же для
  демо-скрипта и для дока в интерфейсе.

Канон (`.clinerules`, REBUILD_SPEC §5/§12): сначала логика + тест, потом UI;
A0.6 — ничего не теряем и не блокируем молча.
"""
from __future__ import annotations

from .session import (Artifact, Attachment, AssistantSession, Message,
                      StagedPatch, ToolCall)
from .store import (append_log, assistant_dir, artifacts_dir, files_dir,
                    load_session, read_log, save_session, session_path)

__all__ = [
    "AssistantSession", "Message", "Attachment", "Artifact", "StagedPatch",
    "ToolCall",
    "assistant_dir", "files_dir", "artifacts_dir", "session_path",
    "save_session", "load_session", "append_log", "read_log",
]
