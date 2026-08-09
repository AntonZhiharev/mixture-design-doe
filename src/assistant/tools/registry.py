"""assistant/tools/registry.py — реестр инструментов: схемы, контекст, диспетчер.

Один список инструментов на проект: из него собираются и JSON-схемы для модели
(`tool_specs`), и исполнение вызова (`dispatch`). Разъехавшиеся описание и
реализация — классическая причина «модель зовёт то, чего нет», поэтому
описание живёт РЯДОМ с функцией (:func:`register`).

Классы доступа (ASSISTANT_SPEC §3):

* ``readonly`` — свободно;
* ``propose`` — модель ПРЕДЛАГАЕТ (патч уходит в стейдж сессии); состояние
  проекта не меняется, поэтому такие вызовы разрешены ей наравне с чтением
  (iter63);
* ``write`` — только с подтверждением человека (`human_token`, iter63):
  реальная правка спеки и записи в журналы решений/фактов;
* ``sandbox`` — изолированное исполнение (iter62).

Граница между ``propose`` и ``write`` — не «опасность операции», а ответ на
вопрос «кто автор изменения». Предложение автора-модели обратимо и живёт в
стейдже; применение — акт человека, и его нельзя получить удачной
формулировкой запроса.

``long_running=True`` помечает инструменты, которым нужен прогресс-бар в UI
(пользователь не должен думать, что приложение зависло).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

READONLY = "readonly"
PROPOSE = "propose"
WRITE = "write"
SANDBOX = "sandbox"
KINDS = (READONLY, PROPOSE, WRITE, SANDBOX)

#: Что разрешено МОДЕЛИ в обычном ходе: читать, предлагать, считать в
#: песочнице. ``write`` сюда не входит по построению (iter63).
AGENT_KINDS = (READONLY, PROPOSE, SANDBOX)


class ToolError(RuntimeError):
    """Инструмент не может выполниться — с объяснением причины (A0.6).

    Текст уходит МОДЕЛИ как результат вызова (см. :mod:`assistant.llm`), поэтому
    он должен быть содержательным: «проект не собран», «узла нет в спеке»,
    «нужен human_token» — из такого сообщения модель делает следующий шаг.
    """


# ----------------------------------------------------------------------
# Контекст исполнения
# ----------------------------------------------------------------------
@dataclass
class ToolContext:
    """Всё, к чему инструменты имеют доступ.

    ``runner`` — движок кампании (может отсутствовать: проект ещё не собран);
    ``spec`` — активная phr-спека (по умолчанию берётся из раннера);
    ``session`` / ``root`` / ``project`` — память ассистента и её место на
    диске; ``human_token`` (iter63) — разовый токен подтверждения из UI.
    """
    runner: Any = None
    session: Any = None
    root: str = ""
    project: str = ""
    spec: Any = None
    human_token: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def require_runner(self):
        if self.runner is None:
            raise ToolError(
                "Проект кампании не собран в этой сессии: движка нет, отвечать "
                "числами не из чего. Соберите проект (вкладка «🧬 Проект») или "
                "загрузите сохранённый.")
        return self.runner

    def require_spec(self):
        """Активная phr-спека или явный отказ.

        Спека — источник ГЕОМЕТРИИ (роли, границы, hash). Без неё вопросы вида
        «почему такой диапазон» не имеют ответа в терминах ядра, и выдумывать
        его нельзя.
        """
        if self.spec is not None:
            return self.spec
        spec = getattr(self.runner, "phr_spec", None) if self.runner else None
        if spec is None:
            raise ToolError(
                "В проекте не задана phr-спека: роли узлов, эффективные "
                "границы и spec_hash недоступны. Введите спеку в сетапе "
                "(канал «phr-спека (JSON)») — до этого геометрия не определена.")
        return spec

    def require_session(self):
        if self.session is None:
            raise ToolError("Сессия ассистента не передана в контекст "
                            "инструментов (внутренняя ошибка вызова).")
        return self.session


# ----------------------------------------------------------------------
# Определение инструмента
# ----------------------------------------------------------------------
@dataclass
class ToolDef:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Any]
    kind: str = READONLY
    long_running: bool = False

    def spec(self) -> Dict[str, Any]:
        """JSON-схема функции в формате OpenAI/OpenRouter tools."""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}


TOOLS: Dict[str, ToolDef] = {}


def register(name: str, *, description: str,
             parameters: Optional[Dict[str, Any]] = None,
             kind: str = READONLY, long_running: bool = False):
    """Декоратор регистрации инструмента ``fn(ctx, **args)``."""
    if kind not in KINDS:
        raise ValueError(f"Неизвестный класс инструмента {kind!r}: {KINDS}")

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in TOOLS:
            raise ValueError(f"Инструмент '{name}' уже зарегистрирован.")
        TOOLS[name] = ToolDef(
            name=name, description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            fn=fn, kind=kind, long_running=long_running)
        return fn

    return deco


def tool_names(kinds: Sequence[str] = (READONLY,)) -> List[str]:
    return sorted(t.name for t in TOOLS.values() if t.kind in kinds)


def tool_specs(kinds: Sequence[str] = (READONLY,)) -> List[Dict[str, Any]]:
    """Схемы инструментов для передачи модели (по классам доступа)."""
    return [t.spec() for t in TOOLS.values() if t.kind in kinds]


def is_long_running(name: str) -> bool:
    t = TOOLS.get(str(name))
    return bool(t and t.long_running)


def _check_required(tool: ToolDef, args: Dict[str, Any]) -> None:
    required = list((tool.parameters or {}).get("required", []) or [])
    missing = [r for r in required if r not in args]
    if missing:
        raise ToolError(
            f"Инструмент '{tool.name}': не переданы обязательные аргументы "
            f"{missing}. Ожидаются: "
            f"{sorted((tool.parameters or {}).get('properties', {}))}.")


def dispatch(ctx: ToolContext, name: str, args: Optional[Dict[str, Any]] = None,
             *, allowed_kinds: Sequence[str] = (READONLY,)) -> Any:
    """Исполнить инструмент по имени.

    Класс доступа проверяется ЗДЕСЬ: даже если модель узнала имя write-функции
    из истории, без явного разрешения вызывающей стороны она её не выполнит
    (ASSISTANT_SPEC §2 — write только через подтверждение человеком).
    """
    args = dict(args or {})
    tool = TOOLS.get(str(name))
    if tool is None:
        raise ToolError(
            f"Инструмент '{name}' не зарегистрирован. Доступны: "
            f"{tool_names(allowed_kinds)}.")
    if tool.kind not in allowed_kinds:
        raise ToolError(
            f"Инструмент '{name}' относится к классу '{tool.kind}' и в этом "
            f"режиме недоступен (разрешены: {list(allowed_kinds)}). "
            f"Изменения применяет человек кнопкой в интерфейсе.")
    _check_required(tool, args)
    unknown = [k for k in args
               if k not in (tool.parameters or {}).get("properties", {})]
    if unknown:
        raise ToolError(
            f"Инструмент '{name}': неизвестные аргументы {unknown}. "
            f"Допустимы: {sorted((tool.parameters or {}).get('properties', {}))}.")
    return tool.fn(ctx, **args)


def dispatcher(ctx: ToolContext, *, allowed_kinds: Sequence[str] = (READONLY,),
               on_call: Optional[Callable[[Dict[str, Any]], None]] = None
               ) -> Callable[[str, Dict[str, Any]], Any]:
    """Готовый ``dispatch(name, args)`` для :func:`assistant.llm.run_tool_loop`.

    ``on_call`` получает запись аудита (инструмент, аргументы, длительность,
    итог) — её пишут в сессию и в ``tool_calls.jsonl``.
    """
    def _call(name: str, args: Dict[str, Any]) -> Any:
        t0 = time.monotonic()
        try:
            out = dispatch(ctx, name, args, allowed_kinds=allowed_kinds)
        except Exception as exc:  # noqa: BLE001 — аудит пишем и на отказе
            if on_call is not None:
                on_call({"tool": name, "args": args, "ok": False,
                         "error": f"{type(exc).__name__}: {exc}",
                         "duration_s": round(time.monotonic() - t0, 3)})
            raise
        if on_call is not None:
            on_call({"tool": name, "args": args, "ok": True, "error": "",
                     "duration_s": round(time.monotonic() - t0, 3),
                     "summary": str(out)[:200]})
        return out

    return _call
