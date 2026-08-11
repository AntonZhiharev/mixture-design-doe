"""assistant/sandbox — песочница ассистента (iter62).

Один интерфейс (:class:`~.base.SandboxBackend`) и одна реализация под него
(:class:`~.subprocess_backend.SubprocessSandbox`). Бэкенд выбирается
:func:`~.subprocess_backend.get_backend` по ``DOE_SANDBOX_BACKEND``, поэтому
переезд на Docker не потребует правок ни в инструментах, ни в тестах
(развилка №1 ASSISTANT_SPEC).

Инструменты класса ``sandbox`` живут в :mod:`.tools` и подключаются реестром
(``assistant.tools``), а не импортируются отсюда — иначе пакет песочницы
зависел бы от реестра, а реестр от него (кольцо).
"""
from __future__ import annotations

from .base import (DEFAULT_PYTEST_TIMEOUT_S, DEFAULT_TIMEOUT_S,
                   IMAGE_SUFFIXES, MAX_COLLECTED_BYTES, MAX_COLLECTED_FILES,
                   MAX_OUTPUT_CHARS, OUTPUT_SUFFIXES, PytestReport,
                   SandboxBackend, SandboxError, SandboxPolicy, SandboxResult,
                   TABLE_SUFFIXES, clip_output, denial_note, detect_denial,
                   output_kind, parse_pytest_output, parse_test_line,
                   progress_caption, timeout_note)
from .guard import NETWORK_MARK, WRITE_MARK, guard_source
from .subprocess_backend import (BACKEND_ENV, BACKENDS, SubprocessSandbox,
                                 get_backend)

__all__ = [
    "SandboxBackend", "SandboxPolicy", "SandboxResult", "SandboxError",
    "PytestReport", "SubprocessSandbox", "get_backend", "BACKEND_ENV",
    "BACKENDS", "progress_caption", "parse_pytest_output", "parse_test_line",
    "clip_output", "detect_denial", "denial_note", "timeout_note",
    "guard_source", "NETWORK_MARK", "WRITE_MARK", "DEFAULT_TIMEOUT_S",
    "DEFAULT_PYTEST_TIMEOUT_S", "MAX_OUTPUT_CHARS",
    "output_kind", "IMAGE_SUFFIXES", "TABLE_SUFFIXES", "OUTPUT_SUFFIXES",
    "MAX_COLLECTED_FILES", "MAX_COLLECTED_BYTES",
]
