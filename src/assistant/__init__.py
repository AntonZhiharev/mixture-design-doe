"""src/assistant — ИИ-ассистент-архитектор кампании (ASSISTANT_SPEC, iter58+).

Пакет собирает всё, что отличает помощника-архитектора от чата «поверх»
приложения:

* :mod:`session` — сессия диалога, привязанная к ПРОЕКТУ (сообщения, вложения,
  артефакты песочницы, staged-патчи спеки, расход токенов);
* :mod:`store` — персистентность сессии в каталоге проекта
  (``project_campaigns/<проект>/assistant/``) + append-only журналы аудита;
* :mod:`views` — ЧИСТЫЕ таблицы показа (без Streamlit): одни и те же для
  демо-скрипта и для дока в интерфейсе;
* :mod:`files` — вложения: хранение в проекте, дедуп по sha256, извлечение
  текста (txt/md/csv/json/xlsx/docx/pdf) с явным объяснением при отказе;
* :mod:`consent` — разовые токены подтверждения ЧЕЛОВЕКОМ: без них ни один
  write-инструмент не меняет состояние проекта (iter63).


Канон (`.clinerules`, REBUILD_SPEC §5/§12): сначала логика + тест, потом UI;
A0.6 — ничего не теряем и не блокируем молча.
"""
from __future__ import annotations

from .consent import Consent, ConsentError, ConsentRegistry, issue_token
from .session import (Artifact, Attachment, AssistantSession, Message,
                      StagedNote, StagedPatch, StagedProject, StagedSpec,
                      ToolCall)
from .store import (append_log, append_log_by_ref, assistant_dir,
                    assistant_dir_by_ref, artifacts_dir, dir_for_ref,
                    files_dir, load_session, load_session_by_ref, read_log,
                    read_log_by_ref, ref_of_dir, save_session,
                    save_session_by_ref, session_path, session_path_by_ref)

__all__ = [
    "AssistantSession", "Message", "Attachment", "Artifact", "StagedPatch",
    "StagedSpec", "StagedProject", "StagedNote", "ToolCall",
    "assistant_dir", "files_dir", "artifacts_dir", "session_path",
    "save_session", "load_session", "append_log", "read_log",
    # iter77: доступ по ССЫЛКЕ проекта (переименование не рвёт переписку)
    "dir_for_ref", "ref_of_dir", "assistant_dir_by_ref", "session_path_by_ref",
    "load_session_by_ref", "save_session_by_ref", "append_log_by_ref",
    "read_log_by_ref",
    "Consent", "ConsentError", "ConsentRegistry", "issue_token",
]
