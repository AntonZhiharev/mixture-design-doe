"""assistant/tools — инструменты ассистента (реестр + реализации).

Разделение по КЛАССАМ ДОСТУПА (ASSISTANT_SPEC §3):

* ``readonly`` — вызываются свободно: спека, объяснение узла, dry-run патча,
  симуляция границ, preflight, прогоны, факты, решения, вложения;
* ``write`` (iter63) — только через подтверждение человеком (`human_token`);
* ``sandbox`` (iter62) — изолированное исполнение кода/тестов.

Реестр (:mod:`.registry`) отдаёт JSON-схемы для модели и исполняет вызовы;
контекст (:class:`.registry.ToolContext`) несёт проект, сессию и движок.
"""
from __future__ import annotations

from .registry import (READONLY, SANDBOX, TOOLS, WRITE, ToolContext, ToolDef,
                       ToolError, dispatch, register, tool_names, tool_specs)
from . import readonly as _readonly  # noqa: F401 — регистрация инструментов
from . import sandbox_tools as _sandbox_tools  # noqa: F401 — то же, класс sandbox

__all__ = ["TOOLS", "ToolContext", "ToolDef", "ToolError", "dispatch",
           "register", "tool_names", "tool_specs",
           "READONLY", "WRITE", "SANDBOX"]


