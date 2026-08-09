"""mcp/campaign_tools.py — чистая логика MCP-сервера `doe-campaign` (iter66).

Тот же слой знания, что у дока ассистента (`src/assistant/tools`), но для
ВНЕШНЕГО агента — Cline в VS Code. Смысл шага: разбор кампании не должен
зависеть от того, открыт ли Streamlit. Cline получает ровно те же ответы «из
ядра» (`explain_node`, `simulate_bounds`, `preflight`, `point_report`, …),
которые видит технолог в доке, — числа считает одна и та же реализация, а не
второй пересказ спеки.

Что здесь принципиально (проверяется тестами
``tests/unit/test_iteration66_assistant_mcp.py``):

1. **Экспортируется РОВНО класс ``readonly``.** ``write`` — акт человека
   (кнопка в интерфейсе, разовый токен iter63); ``propose`` пишет патч в
   стейдж ЧУЖОЙ сессии; ``sandbox`` исполняет код (у Cline своя консоль).
   Список экспортируемого не выписан руками, а вычисляется из реестра
   (:func:`exported_tools`) — новый read-only инструмент появляется в MCP сам,
   новый write-инструмент не появляется никогда.
2. **Обёртки MCP ГЕНЕРИРУЮТСЯ из JSON-схем реестра** (:func:`wrapper_source`):
   имя, аргументы и их типы приходят оттуда же, откуда их берёт модель в доке.
   Руками переписанный список разъехался бы с кодом на первой итерации.
3. **Проект — аргумент, а не глобальное состояние.** Один сервер обслуживает
   все кампании каталога ``project_campaigns``; ``project=""`` разрешается
   автоматически, ТОЛЬКО если проект ровно один (иначе — явный список, а не
   догадка).
4. **«Не собран» ≠ «всё хорошо».** Если у проекта нет ``campaign.json`` (или
   он не читается), контекст отдаётся БЕЗ движка и инструменты честно
   отказывают текстом, а причина видна в :func:`project_status`.
5. **Вызовы пишутся в аудит проекта** (``assistant/tool_calls.jsonl``,
   ``via="mcp"``): разбор через Cline обязан быть виден там же, где разбор
   через док — журнал append-only, поэтому одновременная работа приложения и
   сервера безопасна.

Каталог кампаний берётся из ``DOE_CAMPAIGN_ROOT`` (по умолчанию
``<repo>/project_campaigns``) — так же, как ``DOE_TRACE_ROOT`` у
`doe-introspect`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.assistant.store import (append_log, load_session, session_path)
from src.assistant.tools import (PROPOSE, READONLY, SANDBOX, TOOLS, WRITE,
                                 ToolContext, ToolError, dispatch, tool_names)
from src.assistant.tools.registry import ToolDef

#: Переменная окружения с каталогом проектов-кампаний.
ROOT_ENV = "DOE_CAMPAIGN_ROOT"

#: Классы доступа, которые сервер отдаёт наружу. Ровно один — и это решение
#: сформулировано в терминах авторства: всё, что МЕНЯЕТ состояние (`write`),
#: делает человек в интерфейсе; всё, что пишет в чужую сессию (`propose`), не
#: имеет смысла без дока; исполнение (`sandbox`) у Cline своё.
EXPORTED_KINDS: Tuple[str, ...] = (READONLY,)

#: Имя серверного аргумента «в каком проекте считать».
PROJECT_ARG = "project"

#: Файл состояния кампании (дублируем константу `campaign_state._STATE_FILE`:
#: модуль сервера не тянет за собой numpy/sklearn ради одного имени).
STATE_FILE = "campaign.json"

#: JSON-схема → аннотация Python для генерируемых обёрток.
_PY_TYPES = {"string": "str", "integer": "int", "number": "float",
             "boolean": "bool", "array": "list", "object": "dict"}


# ----------------------------------------------------------------------
# Каталог проектов
# ----------------------------------------------------------------------
def repo_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def campaign_root(root: Optional[str] = None) -> str:
    """Каталог кампаний: аргумент → ``DOE_CAMPAIGN_ROOT`` → ``project_campaigns``."""
    if root:
        return str(root)
    return os.environ.get(ROOT_ENV,
                          str(Path(repo_root()) / "project_campaigns"))


def list_projects(root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Проекты каталога: имя, собран ли движок, есть ли переписка ассистента.

    Показываем и проекты БЕЗ ``campaign.json`` (одна лишь сессия ассистента):
    отсутствие движка — это факт, о котором лучше сказать, чем спрятать
    проект из списка (A0.6).
    """
    base = Path(campaign_root(root))
    if not base.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(base.iterdir(), key=lambda q: q.name):
        if not p.is_dir():
            continue
        has_campaign = (p / STATE_FILE).exists()
        has_session = (p / "assistant" / "session.json").exists()
        if not (has_campaign or has_session):
            continue
        out.append({"project": p.name, "has_campaign": has_campaign,
                    "has_session": has_session,
                    "path": str(p)})
    return out


def project_names(root: Optional[str] = None) -> List[str]:
    return [d["project"] for d in list_projects(root)]


def resolve_project(project: str = "", root: Optional[str] = None) -> str:
    """Имя проекта: пустое разрешается ТОЛЬКО при единственном кандидате.

    Угадывать «наверное, он имел в виду вот этот» нельзя: ответ про чужую
    кампанию выглядит так же уверенно, как правильный.
    """
    name = str(project or "").strip()
    known = project_names(root)
    if name:
        if name not in known:
            raise ToolError(
                f"Проекта '{name}' нет в каталоге кампаний "
                f"{campaign_root(root)}. Доступны: {known or '—'}. "
                f"Проверьте DOE_CAMPAIGN_ROOT сервера doe-campaign.")
        return name
    if not known:
        raise ToolError(
            f"В каталоге {campaign_root(root)} нет ни одного проекта "
            f"кампании. Сохраните проект в интерфейсе (вкладка «🧬 Проект») "
            f"или укажите другой каталог через DOE_CAMPAIGN_ROOT.")
    if len(known) > 1:
        raise ToolError(
            f"Проект не указан, а в каталоге их {len(known)}: {known}. "
            f"Назовите проект явно — угадывать, про какую кампанию вопрос, "
            f"нельзя.")
    return known[0]


# ----------------------------------------------------------------------
# Экспортируемые инструменты
# ----------------------------------------------------------------------
def exported_tools() -> List[ToolDef]:
    """Инструменты, которые сервер отдаёт наружу (класс ``readonly``)."""
    return [TOOLS[n] for n in sorted(TOOLS) if TOOLS[n].kind in EXPORTED_KINDS]


def exported_names() -> List[str]:
    return [t.name for t in exported_tools()]


def hidden_names() -> List[str]:
    """Что НЕ экспортируется — с этим списком сверяется тест контракта."""
    return sorted(tool_names([WRITE, PROPOSE, SANDBOX]))


def is_exported(name: str) -> bool:
    t = TOOLS.get(str(name))
    return bool(t and t.kind in EXPORTED_KINDS)


def tool_catalog() -> List[Dict[str, Any]]:
    """Каталог инструментов сервера словами (для ``list_tools``)."""
    out: List[Dict[str, Any]] = []
    for t in exported_tools():
        props = dict((t.parameters or {}).get("properties", {}))
        req = list((t.parameters or {}).get("required", []) or [])
        out.append({"tool": t.name, "description": t.description,
                    "args": sorted(props), "required": req,
                    "long_running": bool(t.long_running)})
    return out


def _refuse_hidden(name: str) -> str:
    """Текст отказа на не экспортируемый инструмент (A0.6: причина, не «нет»)."""
    t = TOOLS.get(str(name))
    if t is None:
        return (f"Инструмент '{name}' не зарегистрирован. Сервер doe-campaign "
                f"отдаёт: {exported_names()}.")
    if t.kind == WRITE:
        return (f"Инструмент '{name}' относится к классу 'write' и MCP-сервером "
                f"НЕ экспортируется: правку спеки и записи в журналы делает "
                f"ЧЕЛОВЕК кнопкой в интерфейсе кампании (разовый токен "
                f"подтверждения). Через MCP доступно только чтение: "
                f"{exported_names()}.")
    if t.kind == PROPOSE:
        return (f"Инструмент '{name}' кладёт патч в СТЕЙДЖ сессии ассистента и "
                f"через MCP не экспортируется: предложение должно попасть в "
                f"панель, где у человека есть кнопки «Применить»/«Отклонить». "
                f"Сформулируйте правку текстом — применит человек в доке.")
    return (f"Инструмент '{name}' относится к классу '{t.kind}' и через MCP не "
            f"экспортируется (исполнение кода — задача твоей собственной "
            f"консоли, не сервера кампании). Доступно: {exported_names()}.")


# ----------------------------------------------------------------------
# Контекст проекта (движок + спека + сессия), с кэшем по mtime
# ----------------------------------------------------------------------
_CACHE: Dict[Tuple[str, str], Tuple[Tuple[float, float], "ProjectContext"]] = {}


class ProjectContext:
    """Контекст одного проекта: :class:`ToolContext` + причина отсутствия движка."""

    def __init__(self, ctx: ToolContext, *, note: str = "",
                 load_error: str = "") -> None:
        self.ctx = ctx
        self.note = note
        self.load_error = load_error

    @property
    def has_runner(self) -> bool:
        return self.ctx.runner is not None

    @property
    def spec_hash(self) -> str:
        try:
            return str(self.ctx.require_spec().spec_hash())
        except (ToolError, AttributeError):
            return ""


def _stamp(root: str, project: str) -> Tuple[float, float]:
    """Отпечаток состояния проекта на диске (кампания + сессия)."""
    def _m(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    return (_m(Path(root) / project / STATE_FILE),
            _m(Path(session_path(root, project))))


def clear_cache() -> None:
    _CACHE.clear()


def _read_state(root: str, project: str) -> Dict[str, Any]:
    path = Path(root) / project / STATE_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def _spec_from_state(state: Dict[str, Any]):
    """phr-спека прямо из ``campaign.json`` (без сборки движка)."""
    dicts = ((state or {}).get("runner", {}) or {}).get("phr_spec")
    if not dicts:
        return None
    from src.design.phr_sampler import PhrSpec      # локально: тяжёлый импорт
    return PhrSpec.from_dicts(dicts)


def load_context(project: str = "", root: Optional[str] = None, *,
                 use_cache: bool = True) -> ProjectContext:
    """Собрать контекст инструментов для проекта.

    Движок восстанавливается штатным ``campaign_state.load_campaign``
    (суррогаты переобучаются из точек — одна модель физики §5/§12), поэтому
    результат кэшируется по mtime ``campaign.json``/``session.json``: правка в
    интерфейсе видна серверу со следующего вызова, а повторный вопрос не
    переобучает GP заново.

    Если движок не собирается, контекст всё равно возвращается — с ПРИЧИНОЙ и
    (если получится) со спекой из файла: «геометрию прочитать смог, прогоны —
    нет» честнее, чем общий отказ.
    """
    root = campaign_root(root)
    name = resolve_project(project, root)
    key = (str(root), name)
    stamp = _stamp(root, name)
    if use_cache:
        cached = _CACHE.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]

    session = load_session(root, name)
    runner = None
    spec = None
    note = ""
    err = ""
    if (Path(root) / name / STATE_FILE).exists():
        try:
            from src.apps.campaign_state import load_campaign
            runner = load_campaign(root, name)
        except Exception as exc:                       # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            note = (f"Движок проекта '{name}' не восстановился ({err}): "
                    f"вопросы про preflight и прогоны останутся без чисел.")
            try:
                spec = _spec_from_state(_read_state(root, name))
            except Exception:                          # noqa: BLE001
                spec = None
    else:
        note = (f"У проекта '{name}' нет {STATE_FILE}: кампания не сохранена, "
                f"движка и базы точек нет. Это «не проверено», а не «всё "
                f"хорошо».")

    if spec is None and runner is not None:
        spec = getattr(runner, "phr_spec", None)

    ctx = ToolContext(runner=runner, session=session, root=root,
                      project=name, spec=spec)
    pc = ProjectContext(ctx, note=note, load_error=err)
    _CACHE[key] = (stamp, pc)
    return pc


# ----------------------------------------------------------------------
# Состояние проекта
# ----------------------------------------------------------------------
def project_status(project: str = "", root: Optional[str] = None
                   ) -> Dict[str, Any]:
    """Карточка проекта БЕЗ сборки движка: что вообще есть и чего нет.

    Читается сам ``campaign.json`` — статус не должен стоить переобучения
    суррогатов; для чисел есть инструменты ядра.
    """
    root = campaign_root(root)
    name = resolve_project(project, root)
    base = Path(root) / name
    out: Dict[str, Any] = {"project": name, "root": root,
                           "has_campaign": (base / STATE_FILE).exists()}
    session = load_session(root, name)
    out["session"] = {"messages": len(session.messages),
                      "attachments": len(session.attachments),
                      "staged_patches": len(session.staged_patches()),
                      "tool_calls": len(session.tool_calls)}
    if not out["has_campaign"]:
        out["note"] = (f"Проект '{name}' не сохранён как кампания "
                       f"({STATE_FILE} нет): геометрии и базы точек нет. "
                       f"Ответы «не проверено», а не «всё хорошо».")
        return out
    try:
        state = _read_state(root, name)
    except (OSError, ValueError) as exc:
        out["note"] = (f"{STATE_FILE} проекта '{name}' не читается: "
                       f"{type(exc).__name__}: {exc}")
        return out

    r = dict(state.get("runner", {}) or {})
    out["campaign_label"] = str(r.get("campaign_label", "") or "")
    out["property_names"] = list(r.get("property_names", [])
                                 or (state.get("oracle", {}) or {}).get(
                                     "property_names", []) or [])
    out["covariate_names"] = list(r.get("covariate_names", []) or [])
    out["n_points"] = len(r.get("points", []) or [])
    out["branches"] = sorted((r.get("branches", {}) or {}).keys())
    out["schema_version"] = r.get("current_schema_version")
    try:
        spec = _spec_from_state(state)
    except Exception as exc:                            # noqa: BLE001
        spec = None
        out["spec_error"] = f"{type(exc).__name__}: {exc}"
    out["has_phr_spec"] = spec is not None
    out["spec_hash"] = spec.spec_hash() if spec is not None else ""
    out["n_nodes"] = len(getattr(spec, "nodes", []) or []) if spec else 0
    if spec is None:
        out["note"] = ("В проекте не задана phr-спека: роли узлов и "
                       "эффективные границы недоступны — геометрия не "
                       "определена.")
    return out


# ----------------------------------------------------------------------
# Вызов инструмента
# ----------------------------------------------------------------------
def call_tool(project: str, name: str, args: Optional[Dict[str, Any]] = None,
              *, root: Optional[str] = None) -> Dict[str, Any]:
    """Исполнить read-only инструмент ядра в контексте проекта.

    Отказ возвращается ОБЪЯСНЕНИЕМ (``ok=False`` + текст), а не исключением:
    внешнему агенту нужен разбор («класс write применяет человек кнопкой»), из
    которого он сделает следующий шаг, а не стектрейс.
    """
    t0 = time.monotonic()
    root = campaign_root(root)
    name = str(name)
    if not is_exported(name):
        return {"ok": False, "tool": name, "error": _refuse_hidden(name),
                "available": exported_names()}
    try:
        pc = load_context(project, root)
    except ToolError as exc:
        return {"ok": False, "tool": name, "error": str(exc)}

    payload: Dict[str, Any] = {"ok": True, "tool": name,
                               "project": pc.ctx.project}
    try:
        # allowed_kinds — второй рубеж: даже если сюда попадёт имя из другого
        # класса, диспетчер реестра его не выполнит (запрет живёт в коде).
        payload["result"] = dispatch(pc.ctx, name, dict(args or {}),
                                     allowed_kinds=EXPORTED_KINDS)
    except ToolError as exc:
        payload = {"ok": False, "tool": name, "project": pc.ctx.project,
                   "error": str(exc)}
    except Exception as exc:                            # noqa: BLE001
        payload = {"ok": False, "tool": name, "project": pc.ctx.project,
                   "error": f"{type(exc).__name__}: {exc}"}
    payload["duration_s"] = round(time.monotonic() - t0, 3)
    if pc.note:
        payload["note"] = pc.note
    _audit(root, pc.ctx.project, payload, args)
    return payload


def _audit(root: str, project: str, payload: Dict[str, Any],
           args: Optional[Dict[str, Any]]) -> None:
    """Запись вызова в ``assistant/tool_calls.jsonl`` (append-only).

    Помечаем ``via="mcp"``: через неделю должно быть видно, что разбор шёл из
    Cline, а не из дока — иначе журнал кампании рассказывает неполную историю.
    """
    rec = {"tool": payload.get("tool", ""), "args": dict(args or {}),
           "ok": bool(payload.get("ok", False)),
           "error": str(payload.get("error", "")),
           "duration_s": float(payload.get("duration_s", 0.0) or 0.0),
           "summary": str(payload.get("result", ""))[:200],
           "via": "mcp"}
    try:
        append_log(root, project, "tool_calls", rec)
    except (OSError, ValueError):
        pass          # журнал не должен ронять ответ агенту


# ----------------------------------------------------------------------
# Генерация MCP-обёрток из JSON-схем реестра
# ----------------------------------------------------------------------
def python_type(schema: Optional[Dict[str, Any]]) -> str:
    """JSON-тип аргумента → аннотация Python (незнакомый — ``object``)."""
    return _PY_TYPES.get(str((schema or {}).get("type", "")), "object")


def wrapper_signature(tool: ToolDef) -> str:
    """Сигнатура обёртки: аргументы инструмента + ``project`` последним.

    ``project`` идёт в хвосте со значением по умолчанию: обязательные
    аргументы инструмента остаются обязательными и в MCP (иначе схема соврала
    бы про контракт), а имя проекта можно опустить, если он один.
    """
    props = dict((tool.parameters or {}).get("properties", {}))
    if PROJECT_ARG in props:
        raise ValueError(
            f"Инструмент '{tool.name}' объявляет собственный аргумент "
            f"'{PROJECT_ARG}', который занят сервером doe-campaign: обёртку "
            f"не построить без молчаливой подмены значения.")
    required = [r for r in (tool.parameters or {}).get("required", [])
                if r in props]
    optional = [p for p in props if p not in required]
    parts = [f"{r}: {python_type(props[r])}" for r in required]
    parts += [f"{o}: {python_type(props[o])} = None" for o in optional]
    parts.append(f"{PROJECT_ARG}: str = ''")
    return ", ".join(parts)


def wrapper_source(tool: ToolDef) -> str:
    """Исходник обёртки ``fn(<аргументы инструмента>, project='')``.

    Генерация, а не ручной список: имена и типы аргументов приходят из той же
    JSON-схемы, которую видит модель в доке (`@register`), поэтому «в доке
    считает, а через MCP аргумент называется иначе» невозможно.
    """
    props = dict((tool.parameters or {}).get("properties", {}))
    required = [r for r in (tool.parameters or {}).get("required", [])
                if r in props]
    optional = [p for p in props if p not in required]
    lines = [f"def {tool.name}({wrapper_signature(tool)}) -> dict:",
             "    _a = {}"]
    for r in required:
        lines.append(f"    _a['{r}'] = {r}")
    for o in optional:
        lines.append(f"    if {o} is not None:")
        lines.append(f"        _a['{o}'] = {o}")
    lines.append(f"    return _CALL({PROJECT_ARG}, '{tool.name}', _a)")
    return "\n".join(lines)


def build_wrappers(call: Optional[Callable[..., Dict[str, Any]]] = None
                   ) -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Собрать обёртки всех экспортируемых инструментов.

    ``call(project, tool, args)`` подменяется в тестах — генерация проверяется
    без файлов кампании и без пакета ``mcp``.
    """
    call = call or (lambda project, tool, args: call_tool(project, tool, args))
    out: Dict[str, Callable[..., Dict[str, Any]]] = {}
    for t in exported_tools():
        ns: Dict[str, Any] = {"_CALL": call}
        try:
            exec(compile(wrapper_source(t), f"<doe-campaign:{t.name}>", "exec"),
                 ns)                                    # noqa: S102
        except ValueError:
            continue        # конфликт имени аргумента — инструмент пропускаем
        fn = ns[t.name]
        fn.__doc__ = (t.description
                      + f"\n\nАргумент project — имя проекта в каталоге "
                        f"кампаний (можно опустить, если проект один).")
        out[t.name] = fn
    return out


def conflicting_tools() -> List[str]:
    """Инструменты, чью обёртку построить нельзя (имя аргумента занято)."""
    bad: List[str] = []
    for t in exported_tools():
        try:
            wrapper_signature(t)
        except ValueError:
            bad.append(t.name)
    return bad
