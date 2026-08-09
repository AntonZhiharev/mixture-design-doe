"""assistant/tools — инструменты ассистента (реестр + реализации).

Разделение по КЛАССАМ ДОСТУПА (ASSISTANT_SPEC §3):

* ``readonly`` — вызываются свободно: спека, объяснение узла, dry-run патча,
  симуляция границ, preflight, прогоны, факты, решения, вложения;
* ``propose`` (iter63) — модель кладёт патч в СТЕЙДЖ сессии; состояние
  проекта не меняется, поэтому такой вызов ей разрешён;
* ``write`` (iter63) — только через подтверждение человеком (`human_token`):
  применение патча, отклонение, записи в журналы решений и L1-фактов;
* ``sandbox`` (iter62) — изолированное исполнение кода/тестов.

Модели в обычном ходе выдаются ``AGENT_KINDS`` (readonly + propose + sandbox):
класс ``write`` недостижим по построению, а не по тексту промпта.

Реестр (:mod:`.registry`) отдаёт JSON-схемы для модели и исполняет вызовы;
контекст (:class:`.registry.ToolContext`) несёт проект, сессию и движок.
"""
from __future__ import annotations

from .registry import (AGENT_KINDS, PROPOSE, READONLY, SANDBOX, TOOLS, WRITE,
                       ToolContext, ToolDef, ToolError, dispatch, dispatcher,
                       is_long_running, register, tool_names, tool_specs)
from . import readonly as _readonly  # noqa: F401 — регистрация инструментов
from . import sandbox_tools as _sandbox_tools  # noqa: F401 — то же, класс sandbox
from . import write as _write  # noqa: F401 — то же, классы propose/write

__all__ = ["TOOLS", "ToolContext", "ToolDef", "ToolError", "dispatch",
           "dispatcher", "is_long_running", "register", "tool_names",
           "tool_specs", "READONLY", "PROPOSE", "WRITE", "SANDBOX",
           "AGENT_KINDS"]
