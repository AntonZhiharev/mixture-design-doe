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
                       MAX_COLLECTED_BYTES, MAX_COLLECTED_FILES,
                       OUTPUT_SUFFIXES, SandboxBackend, SandboxError,
                       SandboxPolicy, get_backend, output_kind)
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


def collect_outputs(ctx: ToolContext, sb: SandboxBackend,
                    before: Dict[str, float], *, tool: str
                    ) -> Dict[str, Any]:
    """Забрать ФАЙЛЫ, созданные прогоном, из workdir в кампанию (iter68).

    Зачем: рабочий каталог песочницы временный и удаляется в
    :meth:`SandboxBackend.close`, поэтому график, который построил код модели,
    существовал ровно до конца хода — и «вывод песочницы» выглядел чисто
    текстовым, хотя `matplotlib` в ней есть. Теперь картинка/таблица переезжает
    в ``assistant/artifacts/`` кампании, попадает в сессию и рисуется в доке.

    Что здесь СОЗНАТЕЛЬНО ограничено (A0.6 — ограничение видно, а не молчит):

    * берём только известные расширения (``OUTPUT_SUFFIXES``) — бинарный дамп
      «на всякий случай» кампании не нужен;
    * не больше :data:`MAX_COLLECTED_FILES` файлов и не больше
      :data:`MAX_COLLECTED_BYTES` на файл, о пропуске сообщаем словами;
    * ``*.py`` не собираем: это исходник, который мы же и записали.
    """
    out: Dict[str, Any] = {"files": [], "skipped": []}
    try:
        produced = sb.new_files(before, suffixes=OUTPUT_SUFFIXES)
    except OSError as exc:
        out["skipped"].append(f"каталог прогона не прочитан: {exc}")
        return out
    if not produced:
        return out
    if not (ctx.root and ctx.project):
        out["skipped"].append(
            f"файлов создано {len(produced)}, но сессия не привязана к проекту "
            f"— сохранять их некуда")
        return out

    for path in produced[:MAX_COLLECTED_FILES]:
        src = Path(path)
        try:
            size = src.stat().st_size
        except OSError as exc:
            out["skipped"].append(f"{src.name}: не прочитан ({exc})")
            continue
        if size > MAX_COLLECTED_BYTES:
            out["skipped"].append(
                f"{src.name}: {size / 1048576:.1f} МБ больше лимита "
                f"{MAX_COLLECTED_BYTES / 1048576:.0f} МБ — не сохранён")
            continue
        kind = output_kind(src.name)
        try:
            ensure_dirs(ctx.root, ctx.project)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            fname = f"{ts}_{_safe_name(src.name)}"
            dst = artifacts_dir(ctx.root, ctx.project) / fname
            dst.write_bytes(src.read_bytes())
        except OSError as exc:
            out["skipped"].append(f"{src.name}: не сохранён ({exc})")
            continue
        if ctx.session is not None:
            ctx.session.add_artifact(Artifact(
                name=fname, kind=kind, path=str(dst), tool=tool,
                caption=f"{kind} · {size} байт · создан прогоном"))
        out["files"].append({"name": fname, "kind": kind, "size": size,
                             "path": str(dst), "source_name": src.name})

    if len(produced) > MAX_COLLECTED_FILES:
        out["skipped"].append(
            f"создано файлов: {len(produced)}, сохранены первые "
            f"{MAX_COLLECTED_FILES} — сохраняй меньше файлов за прогон")
    return out


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
        "спеку и файлы проекта менять нельзя — для этого есть патч. "
        "ГРАФИК И ТАБЛИЦУ показывай ФАЙЛОМ: сохрани в текущий каталог "
        "(matplotlib: `plt.savefig('name.png')`; таблица: `df.to_csv('name.csv', "
        "index=False)`) — созданные png/svg/csv/tsv/xlsx/json/html "
        "автоматически переносятся в кампанию и показываются пользователю "
        "картинкой и таблицей. matplotlib используй с backend 'Agg' "
        "(`matplotlib.use('Agg')`), окон в песочнице нет."),
    parameters={"type": "object", "properties": {
        "code": {"type": "string", "description": "код Python"},
        "timeout_s": {"type": "number", "description": "предел времени, сек"}},
        "required": ["code"]},
    kind=SANDBOX, long_running=True)
def run_python(ctx: ToolContext, code: str, timeout_s: Any = None
               ) -> Dict[str, Any]:
    sb = backend_for(ctx)
    limit = _check_timeout(timeout_s, sb.policy.timeout_s or DEFAULT_TIMEOUT_S)
    before = sb.snapshot_workdir()          # чтобы отличить созданное прогоном
    try:
        res = sb.run_python(str(code), timeout_s=limit)
    except SandboxError as exc:
        raise ToolError(str(exc)) from exc
    out = res.to_dict()
    out["artifact"] = save_artifact(
        ctx, "run_python.txt",
        f"# argv: {' '.join(res.argv)}\n# {res.caption()}\n\n{res.output}",
        tool="run_python", caption=res.caption())
    # Файлы забираем ДАЖЕ при падении: график часто успевает сохраниться до
    # ошибки, и он же объясняет, что пошло не так.
    produced = collect_outputs(ctx, sb, before, tool="run_python")
    out["outputs"] = produced["files"]
    if produced["skipped"]:
        out["outputs_skipped"] = produced["skipped"]
    if produced["files"]:
        out["outputs_note"] = (
            "Эти файлы сохранены в кампанию и УЖЕ ПОКАЗАНЫ пользователю "
            "(картинки — изображением, csv/xlsx — таблицей). Не описывай их "
            "содержимое как невидимое и не пересказывай числа таблицы целиком: "
            "ссылайся по имени файла.")
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
