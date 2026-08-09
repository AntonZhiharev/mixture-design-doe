"""assistant/tools/sandbox_tools.py — инструменты класса ``sandbox`` (iter62).

Read-only инструменты (iter61) отвечают на вопрос «что сейчас в спеке»;
песочница отвечает на другой — «а проверено ли это». Ассистент-архитектор
обязан уметь ПОКАЗАТЬ, что после патча тесты кампании остаются зелёными, и
посчитать то, для чего готового инструмента нет, — не выдавая при этом
рассуждение за расчёт.

Зарегистрированы:

* ``sandbox_info`` (readonly) — ограничения песочницы словами: чтобы модель
  знала про «сети нет» и «репозиторий на чтение» ДО попытки, а не из отказа;
* ``run_python`` — фрагмент Python в изоляции;
* ``list_tests`` — какие тесты вообще есть у цели (``--collect-only``);
* ``run_pytest`` — прогон С ПРОГРЕССОМ и разобранным отчётом.

Все три исполняющих инструмента имеют ``kind=SANDBOX``: в readonly-режиме
диспетчер их НЕ выполнит (проверка класса живёт в реестре, а не в промпте).
``long_running=True`` — сигнал UI рисовать прогресс-бар (развилка №7).

Выхлоп сохраняется артефактом в кампанию (``assistant/artifacts/``) и
записывается в сессию: разбор должен воспроизводиться позже, а не жить в
одном ответе чата.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence


from ..sandbox import (DEFAULT_PYTEST_TIMEOUT_S, DEFAULT_TIMEOUT_S,
                       SandboxBackend, SandboxError, SandboxPolicy,
                       get_backend)
from ..session import Artifact
from ..store import artifacts_dir, ensure_dirs
from .registry import SANDBOX, ToolContext, ToolError, register

#: Верхний предел тайм-аута, который может запросить модель (сек). Больше —
#: осознанное решение человека в настройках, а не аргумент вызова.
MAX_TIMEOUT_S = 1800.0


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[3])


def backend_for(ctx: ToolContext) -> SandboxBackend:
    """Песочница из контекста или новая по политике проекта.

    Готовый бэкенд кладут в ``ctx.extra['sandbox']`` (UI/тесты): так рабочий
    каталог и его уборка живут ровно столько, сколько нужно вызывающему.
    """
    sb = (ctx.extra or {}).get("sandbox")
    if sb is not None:
        return sb
    policy = SandboxPolicy(repo_root=_repo_root(), timeout_s=DEFAULT_TIMEOUT_S)
    sb = get_backend(policy=policy)
    ctx.extra["sandbox"] = sb           # переиспользуем в пределах хода
    return sb


def _progress(ctx: ToolContext):
    """Обработчик прогресса из контекста (UI подставляет свой)."""
    return (ctx.extra or {}).get("on_progress")


def _check_timeout(value: Any, default: float) -> float:
    if value in (None, "", 0):
        return float(default)
    try:
        t = float(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"timeout_s должен быть числом секунд, получено "
                        f"{value!r}.") from exc
    if t <= 0:
        raise ToolError("timeout_s должен быть > 0: запуск без предела "
                        "времени в песочнице запрещён.")
    if t > MAX_TIMEOUT_S:
        raise ToolError(
            f"timeout_s={t:.0f} с превышает предел {MAX_TIMEOUT_S:.0f} с. "
            f"Долгий прогон — решение человека (настройки), а не аргумент "
            f"вызова: сузьте задачу.")
    return t


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-zА-Яа-я_.-]+", "_", str(name or "artifact"))[:80]


def save_artifact(ctx: ToolContext, name: str, text: str, *, tool: str,
                  caption: str = "", kind: str = "text") -> str:
    """Сохранить выхлоп песочницы в кампанию и отметить его в сессии.

    Без этого разбор существует только в ленте чата: вернуться к нему через
    неделю («а что показал прогон, когда мы двигали границу DINP?») было бы
    нечем — а именно из таких разборов складывается журнал решений.
    """
    if not (ctx.root and ctx.project):
        return ""
    try:
        ensure_dirs(ctx.root, ctx.project)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        fname = f"{ts}_{_safe_name(name)}"
        path = artifacts_dir(ctx.root, ctx.project) / fname
        path.write_text(str(text), encoding="utf-8")
    except OSError as exc:              # не роняем ход из-за артефакта
        return f"(артефакт не сохранён: {exc})"
    if ctx.session is not None:
        ctx.session.add_artifact(Artifact(name=fname, kind=kind,
                                          path=str(path), tool=tool,
                                          caption=caption))
    return str(path)


# ----------------------------------------------------------------------
# Инструменты
# ----------------------------------------------------------------------
@register(
    "sandbox_info",
    description=(
        "Ограничения песочницы: бэкенд, тайм-аут, рабочий каталог, что "
        "разрешено писать, что защищено. Сети в песочнице НЕТ (веб — отдельный "
        "канал ':online'), репозиторий доступен только на чтение, tests/ не "
        "правится вообще."),
    parameters={"type": "object", "properties": {}})
def sandbox_info(ctx: ToolContext) -> Dict[str, Any]:
    return backend_for(ctx).describe()


@register(
    "run_python",
    description=(
        "Выполнить фрагмент Python в изолированном подпроцессе и вернуть его "
        "вывод. Пользуйся, когда нужно ПОСЧИТАТЬ то, для чего нет готового "
        "инструмента (проверить формулу границы, разыграть точки, сверить "
        "числа). Сети нет; писать можно только в рабочий каталог песочницы; "
        "спеку и файлы проекта менять нельзя — для этого есть патч."),
    parameters={"type": "object", "properties": {
        "code": {"type": "string", "description": "код Python"},
        "timeout_s": {"type": "number", "description": "предел времени, сек"}},
        "required": ["code"]},
    kind=SANDBOX, long_running=True)
def run_python(ctx: ToolContext, code: str, timeout_s: Any = None
               ) -> Dict[str, Any]:
    sb = backend_for(ctx)
    limit = _check_timeout(timeout_s, sb.policy.timeout_s or DEFAULT_TIMEOUT_S)
    try:
        res = sb.run_python(str(code), timeout_s=limit)
    except SandboxError as exc:
        raise ToolError(str(exc)) from exc
    out = res.to_dict()
    out["artifact"] = save_artifact(
        ctx, "run_python.txt",
        f"# argv: {' '.join(res.argv)}\n# {res.caption()}\n\n{res.output}",
        tool="run_python", caption=res.caption())
    return out


@register(
    "list_tests",
    description=(
        "Какие тесты есть у указанных файлов (pytest --collect-only). Полезно "
        "перед прогоном: выбрать точечный набор вместо всего каталога."),
    parameters={"type": "object", "properties": {
        "targets": {"type": "array", "items": {"type": "string"},
                    "description": "файлы тестов относительно корня репозитория"},
        "timeout_s": {"type": "number", "description": "предел времени, сек"}},
        "required": ["targets"]},
    kind=SANDBOX)
def list_tests(ctx: ToolContext, targets: Sequence[str],
               timeout_s: Any = None) -> Dict[str, Any]:
    sb = backend_for(ctx)
    limit = _check_timeout(timeout_s, sb.policy.timeout_s or DEFAULT_TIMEOUT_S)
    try:
        ids = sb.collect(targets, timeout_s=limit)
    except SandboxError as exc:
        raise ToolError(str(exc)) from exc
    return {"n": len(ids), "tests": ids,
            "note": ("Пустой список — это НЕ «всё в порядке»: значит, путь или "
                     "фильтр не нашли тестов.") if not ids else ""}


@register(
    "run_pytest",
    description=(
        "Прогнать тесты кампании в песочнице и вернуть отчёт (сколько прошло/"
        "упало, какие именно nodeid, хвост вывода). Так проверяется утверждение "
        "«патч ничего не ломает»: тесты — контракт (golden-числа), править их "
        "нельзя, песочница видит репозиторий только на чтение. Перечисляй файлы "
        "ЯВНО; прогон долгий — пользователю показывается прогресс."),
    parameters={"type": "object", "properties": {
        "targets": {"type": "array", "items": {"type": "string"},
                    "description": "файлы тестов относительно корня репозитория"},
        "k": {"type": "string", "description": "фильтр -k по имени теста"},
        "maxfail": {"type": "integer",
                    "description": "остановиться после N падений"},
        "timeout_s": {"type": "number", "description": "предел времени, сек"}},
        "required": ["targets"]},
    kind=SANDBOX, long_running=True)
def run_pytest(ctx: ToolContext, targets: Sequence[str], k: str = "",
               maxfail: int = 0, timeout_s: Any = None) -> Dict[str, Any]:
    sb = backend_for(ctx)
    limit = _check_timeout(timeout_s, DEFAULT_PYTEST_TIMEOUT_S)
    try:
        report = sb.run_pytest(targets, k=str(k or ""),
                               maxfail=int(maxfail or 0), timeout_s=limit,
                               on_progress=_progress(ctx))
    except SandboxError as exc:
        raise ToolError(str(exc)) from exc
    out = report.to_dict()
    out["artifact"] = save_artifact(
        ctx, "run_pytest.txt",
        f"# {' '.join(report.argv)}\n# {report.caption()}\n\n{report.tail}",
        tool="run_pytest", caption=report.caption())
    if not report.ok and not report.failures and not report.timed_out:
        out["hint"] = ("Прогон не зелёный, но конкретных упавших тестов нет — "
                       "смотри 'tail': скорее всего ошибка импорта/сборки, а "
                       "не провал проверки.")
    return out
