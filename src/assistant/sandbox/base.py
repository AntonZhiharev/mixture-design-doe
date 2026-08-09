"""assistant/sandbox/base.py — интерфейс песочницы и разбор её выхлопа (iter62).

Песочница нужна ассистенту ровно для одного: **проверять свои утверждения
исполнением**, а не убеждением. «Патч не ломает геометрию» — это `run_pytest`,
а не фраза в ответе; «на такой спеке Σphr уезжает» — это посчитанный
подпроцесс, а не память модели.

Развилка №1 ASSISTANT_SPEC решена так: сейчас работает subprocess-бэкенд, но
вызовы идут через ИНТЕРФЕЙС :class:`SandboxBackend`, у которого абстрактен
ровно один метод — :meth:`SandboxBackend.run`. Всё остальное (``run_python``,
``collect``, ``run_pytest``, разбор отчёта, прогресс) написано поверх него,
поэтому переезд на Docker (``DOE_SANDBOX_BACKEND=docker``) не меняет ни
инструменты, ни тесты этого слоя.

Инварианты (проверяются тестами iter62):

* **тайм-аут всегда есть** — зависший подпроцесс убивается, частичный вывод
  сохраняется, а результат честно помечен ``timed_out`` (молчаливое
  ожидание — худший вид отказа для UI);
* **сети нет** — см. :mod:`.guard`; попытка отмечается ``denied='network'``;
* **репозиторий только на чтение**, ``tests/`` не правится вообще —
  ``denied='write'``;
* **вывод усекается С ПОМЕТКОЙ**, а не молча (A0.6);
* **прогресс наружу** — ``on_progress`` получает событие на КАЖДЫЙ тест, UI
  рисует по ним полосу (долгий прогон не должен выглядеть зависанием).
"""
from __future__ import annotations

import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .guard import NETWORK_MARK, WRITE_MARK

#: Тайм-аут по умолчанию на один запуск (сек). Короткий — намеренно: обычная
#: проверка ассистента это «посчитай границы», а не «прогони всё».
DEFAULT_TIMEOUT_S = 60.0

#: Тайм-аут по умолчанию для ``run_pytest``: старт интерпретатора + импорт
#: numpy/sklearn съедают несколько секунд ещё до первого теста.
DEFAULT_PYTEST_TIMEOUT_S = 600.0

#: Предел выхлопа, который берём в память/контекст. Дальше — усечение серединой
#: (хвост важнее: там итог pytest).
MAX_OUTPUT_CHARS = 200_000

#: Переменные окружения с секретами в подпроцесс НЕ передаются: ключ модели
#: не должен быть доступен коду, который написала модель.
SECRET_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


class SandboxError(RuntimeError):
    """Песочница не может выполнить запрос — с объяснением причины (A0.6)."""


# ----------------------------------------------------------------------
# Политика
# ----------------------------------------------------------------------
@dataclass
class SandboxPolicy:
    """Что подпроцессу можно, а что нет.

    ``repo_root`` монтируется/виден ТОЛЬКО НА ЧТЕНИЕ: список
    :attr:`write_roots` его не содержит. ``deny_write`` проверяется РАНЬШЕ
    разрешений — так ``tests/`` остаётся защищённым, даже если кто-то по
    ошибке добавит корень репозитория в разрешённые.
    """
    repo_root: str = ""
    workdir: str = ""
    allow_network: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_output_chars: int = MAX_OUTPUT_CHARS
    allow_temp: bool = True
    write_roots: List[str] = field(default_factory=list)
    deny_write: List[str] = field(default_factory=list)

    def effective_write_roots(self) -> List[str]:
        roots = [self.workdir] if self.workdir else []
        roots += [str(p) for p in self.write_roots]
        if self.allow_temp:
            roots.append(tempfile.gettempdir())
        out: List[str] = []
        for r in roots:
            p = os.path.abspath(str(r))
            if p and p not in out:
                out.append(p)
        return out

    def effective_deny_write(self) -> List[str]:
        deny = [str(p) for p in self.deny_write]
        if self.repo_root:
            # Тесты — контракт (golden-числа iter45–57). Даже если репозиторий
            # когда-нибудь окажется писабельным, этот каталог остаётся закрыт.
            deny.append(os.path.join(os.path.abspath(self.repo_root), "tests"))
        out: List[str] = []
        for d in deny:
            p = os.path.abspath(str(d))
            if p and p not in out:
                out.append(p)
        return out

    def guard_payload(self) -> Dict[str, Any]:
        """Политика в виде, который читает сторож дочернего процесса."""
        return {"allow_network": bool(self.allow_network),
                "write_roots": self.effective_write_roots(),
                "deny_write": self.effective_deny_write()}

    def describe(self) -> Dict[str, Any]:
        """Человекочитаемое описание ограничений (идёт МОДЕЛИ и в UI)."""
        return {
            "repo_root": os.path.abspath(self.repo_root) if self.repo_root else "",
            "repo_access": "только чтение",
            "workdir": self.workdir,
            "network": "запрещена" if not self.allow_network else "разрешена",
            "timeout_s": float(self.timeout_s),
            "write_roots": self.effective_write_roots(),
            "protected": self.effective_deny_write(),
            "note": ("Сети в песочнице нет: веб — отдельный канал (':online'). "
                     "Репозиторий доступен только на чтение, tests/ не правится "
                     "вообще: тесты — контракт кампании."),
        }


# ----------------------------------------------------------------------
# Результат запуска
# ----------------------------------------------------------------------
def clip_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> (str):
    """Усечь вывод СЕРЕДИНОЙ с явной пометкой (хвост важнее — там итог)."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2):]
    return (f"{head}\n…[вывод усечён: {len(text)} символов, показаны начало и "
            f"конец]…\n{tail}")


@dataclass
class SandboxResult:
    """Итог одного запуска в песочнице.

    ``denied`` объясняет ПРИЧИНУ отказа словом (``network``/``write``), а не
    кодом возврата: «упало с кодом 1» и «полезло в сеть» — разные новости для
    пользователя и для модели.
    """
    argv: List[str] = field(default_factory=list)
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    denied: str = ""
    truncated: bool = False
    backend: str = ""
    workdir: str = ""
    note: str = ""

    @property
    def ok(self) -> bool:
        return (not self.timed_out) and self.returncode == 0 and not self.denied

    @property
    def output(self) -> str:
        """stdout + stderr одной лентой (как это видит человек в консоли)."""
        parts = [p for p in (self.stdout, self.stderr) if p]
        return "\n".join(parts)

    def caption(self) -> str:
        if self.timed_out:
            return (f"⏱ прервано по тайм-ауту через {self.duration_s:.1f} с "
                    f"(частичный вывод сохранён)")
        if self.denied == "network":
            return "⛔ попытка выйти в сеть — в песочнице сети нет"
        if self.denied == "write":
            return "⛔ попытка записи вне рабочего каталога — репозиторий на чтение"
        mark = "✅" if self.ok else "⛔"
        return f"{mark} код {self.returncode} · {self.duration_s:.1f} с"

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "returncode": self.returncode,
                "stdout": self.stdout, "stderr": self.stderr,
                "duration_s": round(float(self.duration_s), 3),
                "timed_out": bool(self.timed_out), "denied": self.denied,
                "truncated": bool(self.truncated), "backend": self.backend,
                "workdir": self.workdir, "argv": list(self.argv),
                "note": self.note, "caption": self.caption()}


def detect_denial(text: str) -> str:
    """Что именно запретил сторож: ``network`` / ``write`` / ``''``."""
    s = str(text or "")
    if NETWORK_MARK in s:
        return "network"
    if WRITE_MARK in s:
        return "write"
    return ""


def denial_note(kind: str) -> str:
    """Пояснение отказа, которое уходит МОДЕЛИ и человеку (A0.6)."""
    if kind == "network":
        return ("Подпроцесс попытался выйти в сеть — это запрещено. Интернет у "
                "ассистента есть, но другим каналом (OpenRouter ':online'), и "
                "полученное оттуда — знание уровня L2: локальные факты цеха "
                "(L1) его отменяют.")
    if kind == "write":
        return ("Подпроцесс попытался писать вне рабочего каталога. "
                "Репозиторий доступен только на чтение, а tests/ не правится "
                "вообще: тесты — контракт кампании (golden-числа). Изменение "
                "спеки предлагай патчем — применяет его человек.")
    return ""


def timeout_note(timeout_s: float) -> str:
    return (f"Запуск прерван по тайм-ауту ({timeout_s:.0f} с) и подпроцесс "
            f"убит. Частичный вывод сохранён — по нему видно, на чём всё "
            f"встало. Сузьте задачу (конкретный файл тестов, меньше точек "
            f"розыгрыша) или поднимите timeout_s осознанно.")


# ----------------------------------------------------------------------
# Отчёт pytest
# ----------------------------------------------------------------------
#: Строка вида ``tests/unit/test_x.py::TestY::test_z PASSED [ 33%]`` (pytest -v).
TEST_LINE_RE = re.compile(
    r"^(?P<nodeid>\S+::\S+)\s+(?P<outcome>PASSED|FAILED|ERROR|SKIPPED|XFAIL|"
    r"XPASS)\b(?:.*?\[\s*(?P<pct>\d+)%\s*\])?")

_SUMMARY_RE = re.compile(
    r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
_DURATION_RE = re.compile(r"\bin\s+([\d.]+)s")

_OUTCOME_KEY = {"PASSED": "passed", "FAILED": "failed", "ERROR": "errors",
                "SKIPPED": "skipped", "XFAIL": "xfailed", "XPASS": "xpassed"}


def parse_test_line(line: str) -> Optional[Dict[str, Any]]:
    """Строка ``pytest -v`` → ``{nodeid, outcome, percent}`` или ``None``."""
    m = TEST_LINE_RE.match(str(line or "").strip())
    if not m:
        return None
    pct = m.group("pct")
    return {"nodeid": m.group("nodeid"), "outcome": m.group("outcome"),
            "percent": int(pct) if pct is not None else None}


@dataclass
class PytestReport:
    """Итог прогона тестов — то, на что ассистент имеет право ссылаться.

    ``failures`` — конкретные nodeid: «упало 2 теста» без имён не позволяет ни
    исправить, ни оценить масштаб.
    """
    ok: bool = False
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    deselected: int = 0
    duration_s: float = 0.0
    returncode: int = -1
    timed_out: bool = False
    denied: str = ""
    failures: List[str] = field(default_factory=list)
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    summary_line: str = ""
    tail: str = ""
    note: str = ""
    argv: List[str] = field(default_factory=list)

    def caption(self) -> str:
        if self.timed_out:
            return (f"⏱ тесты прерваны по тайм-ауту через {self.duration_s:.0f} с "
                    f"(успело пройти {self.passed} из {self.total or '?'})")
        if self.denied:
            return f"⛔ прогон остановлен политикой песочницы ({self.denied})"
        mark = "✅" if self.ok else "⛔"
        parts = [f"{self.passed} прошло"]
        if self.failed:
            parts.append(f"{self.failed} упало")
        if self.errors:
            parts.append(f"{self.errors} ошибок")
        if self.skipped:
            parts.append(f"{self.skipped} пропущено")
        return f"{mark} {' · '.join(parts)} за {self.duration_s:.1f} с"

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "total": self.total, "passed": self.passed,
                "failed": self.failed, "errors": self.errors,
                "skipped": self.skipped, "xfailed": self.xfailed,
                "xpassed": self.xpassed, "deselected": self.deselected,
                "duration_s": round(float(self.duration_s), 3),
                "returncode": self.returncode, "timed_out": self.timed_out,
                "denied": self.denied, "failures": list(self.failures),
                "targets": list(self.targets),
                "summary_line": self.summary_line, "tail": self.tail,
                "note": self.note, "caption": self.caption(),
                "argv": list(self.argv)}


def parse_pytest_output(result: SandboxResult, *,
                        outcomes: Optional[Sequence[Dict[str, Any]]] = None,
                        total: int = 0,
                        targets: Sequence[str] = ()) -> PytestReport:
    """Выхлоп pytest → :class:`PytestReport` (ЧИСТАЯ функция).

    Разбор отделён от запуска намеренно: так его тестируют на записанном
    выводе без единого подпроцесса, а Docker-бэкенд получит его даром.
    """
    text = result.output
    lines = text.splitlines()
    recs: List[Dict[str, Any]] = [dict(o) for o in (outcomes or [])]
    if not recs:
        for ln in lines:
            rec = parse_test_line(ln)
            if rec:
                recs.append(rec)

    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0,
              "xfailed": 0, "xpassed": 0, "deselected": 0}
    summary_line = ""
    for ln in reversed(lines):
        if _SUMMARY_RE.search(ln) and ("passed" in ln or "failed" in ln
                                       or "error" in ln or "skipped" in ln
                                       or "no tests ran" in ln):
            summary_line = ln.strip("= ").strip()
            for num, word in _SUMMARY_RE.findall(ln):
                key = "errors" if word.startswith("error") else word
                counts[key] = counts.get(key, 0) + int(num)
            break

    if not summary_line and recs:            # итоговой строки нет (обрыв/тайм-аут)
        for r in recs:
            counts[_OUTCOME_KEY.get(str(r.get("outcome")), "passed")] += 1

    duration = 0.0
    if summary_line:
        m = _DURATION_RE.search(summary_line)
        if m:
            duration = float(m.group(1))
    if not duration:
        duration = float(result.duration_s)

    failures = [str(r.get("nodeid")) for r in recs
                if str(r.get("outcome")) in ("FAILED", "ERROR")]

    n_total = int(total or 0)
    if not n_total:
        n_total = (counts["passed"] + counts["failed"] + counts["errors"]
                   + counts["skipped"] + counts["xfailed"] + counts["xpassed"])
    if not n_total:
        n_total = len(recs)

    note = result.note
    if result.timed_out and not note:
        note = timeout_note(result.duration_s)
    if not result.timed_out and not result.denied and counts["failed"] == 0 \
            and counts["errors"] == 0 and result.returncode not in (0, 5):
        note = (note + " " if note else "") + (
            f"pytest завершился с кодом {result.returncode}, но упавших тестов "
            f"в отчёте нет — вероятна ошибка сборки/импорта (смотри хвост "
            f"вывода), а не провал проверки.")
    if result.returncode == 5:
        note = (note + " " if note else "") + (
            "pytest не нашёл ни одного теста (код 5): проверь путь/-k, это НЕ "
            "«всё зелено».")

    ok = (result.returncode == 0 and not result.timed_out and not result.denied
          and counts["failed"] == 0 and counts["errors"] == 0)

    return PytestReport(
        ok=ok, total=n_total, passed=counts["passed"], failed=counts["failed"],
        errors=counts["errors"], skipped=counts["skipped"],
        xfailed=counts["xfailed"], xpassed=counts["xpassed"],
        deselected=counts["deselected"], duration_s=duration,
        returncode=result.returncode, timed_out=result.timed_out,
        denied=result.denied, failures=failures, outcomes=recs,
        targets=[str(t) for t in targets], summary_line=summary_line,
        tail="\n".join(lines[-40:]), note=note, argv=list(result.argv))


def progress_caption(event: Dict[str, Any]) -> str:
    """Событие песочницы → строка для показа (чистая, без Streamlit).

    Тот же приём, что в :func:`assistant.llm.progress_caption`: UI рисует
    полосу по данным, а не по зашитым названиям инструментов.
    """
    kind = str((event or {}).get("kind", ""))
    if kind == "start":
        tgt = ", ".join(event.get("targets", []) or []) or "—"
        n = event.get("total") or "?"
        return f"🧪 запускаю pytest: {tgt} (тестов: {n})"
    if kind == "test":
        mark = {"PASSED": "✅", "FAILED": "⛔", "ERROR": "💥",
                "SKIPPED": "⏭", "XFAIL": "🟡", "XPASS": "🟡"}.get(
                    str(event.get("outcome", "")), "•")
        done = int(event.get("done", 0) or 0)
        total = int(event.get("total", 0) or 0)
        pct = event.get("percent")
        pct_s = f" {int(pct)}%" if pct is not None else ""
        return (f"{mark} {done}/{total or '?'}{pct_s} · "
                f"{str(event.get('nodeid', ''))}")
    if kind == "done":
        return f"🏁 {str(event.get('caption', ''))}"
    if kind == "exec":
        return f"⚙️ выполняется: {' '.join(event.get('argv', [])[:3])}…"
    return kind


# ----------------------------------------------------------------------
# Интерфейс бэкенда
# ----------------------------------------------------------------------
class SandboxBackend(ABC):
    """Интерфейс песочницы: абстрактен ОДИН метод — :meth:`run`.

    Всё прикладное (``run_python``, ``collect``, ``run_pytest``, прогресс,
    разбор отчёта) реализовано здесь поверх ``run``, поэтому смена бэкенда
    (subprocess → docker) не трогает ни инструменты ассистента, ни их тесты
    (развилка №1 ASSISTANT_SPEC).
    """
    name = "abstract"

    def __init__(self, policy: Optional[SandboxPolicy] = None, **kw: Any):
        self.policy = policy or SandboxPolicy(**kw)

    # -- обязательный примитив ------------------------------------------
    @abstractmethod
    def run(self, argv: Sequence[str], *, timeout_s: Optional[float] = None,
            stdin_text: Optional[str] = None,
            env: Optional[Dict[str, str]] = None,
            on_output: Optional[Callable[[str], None]] = None
            ) -> SandboxResult:
        """Выполнить команду в изоляции и вернуть результат."""

    # -- рабочий каталог -------------------------------------------------
    @property
    def workdir(self) -> str:
        return self.policy.workdir

    @property
    def python(self) -> str:
        """Команда интерпретатора внутри песочницы."""
        return "python"

    def write_scratch(self, name: str, text: str) -> str:
        """Положить файл в РАБОЧИЙ каталог (единственное писабельное место)."""
        safe = os.path.basename(str(name or "snippet.py")) or "snippet.py"
        path = Path(self.workdir) / safe
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(text), encoding="utf-8")
        return str(path)

    # -- прикладные операции ---------------------------------------------
    def run_python(self, code: str, *, timeout_s: Optional[float] = None,
                   filename: str = "snippet.py",
                   on_output: Optional[Callable[[str], None]] = None
                   ) -> SandboxResult:
        """Выполнить фрагмент Python в песочнице.

        Код кладётся файлом в рабочий каталог (а не ``-c``): при падении в
        трассировке видны номера строк, и разбор ошибки становится предметным.
        """
        if not str(code or "").strip():
            raise SandboxError("Пустой код: нечего выполнять в песочнице.")
        path = self.write_scratch(filename, code)
        return self.run([self.python, path], timeout_s=timeout_s,
                        on_output=on_output)

    def _normalize_targets(self, targets: Any) -> List[str]:
        """Проверить цели pytest и привести их к АБСОЛЮТНЫМ путям.

        Абсолютные — потому что рабочий каталог подпроцесса это scratch, а не
        репозиторий: относительный `tests/unit/...` там просто не нашёлся бы,
        и «тестов не найдено» читалось бы как «всё зелено». Rootdir pytest
        определит по самому файлу и подхватит `pyproject.toml` репозитория.
        """
        if isinstance(targets, str):
            targets = [targets]
        items = [str(t).strip() for t in (targets or []) if str(t).strip()]
        if not items:
            raise SandboxError(
                "Не указано, ЧТО прогонять: перечисли файлы тестов явно "
                "(например 'tests/unit/test_iteration61_assistant_tools.py'). "
                "Прогон всего каталога tests/unit заведомо падает на сборке "
                "давней несвязанной проблемы — см. .clinerules.")
        root = os.path.abspath(self.policy.repo_root or os.getcwd())
        out: List[str] = []
        for it in items:
            base, sep, node = it.partition("::")
            path = base if os.path.isabs(base) else os.path.join(root, base)
            path = os.path.abspath(path)
            if not os.path.exists(path):
                raise SandboxError(
                    f"Цель '{it}' не найдена: файла '{path}' нет. Проверь путь "
                    f"относительно корня репозитория ({root}).")
            out.append(path + sep + node if sep else path)
        return out

    def display_targets(self, targets: Sequence[str]) -> List[str]:
        """Пути относительно репозитория — для показа и отчёта."""
        root = os.path.abspath(self.policy.repo_root or os.getcwd())
        out: List[str] = []
        for t in targets:
            base, sep, node = str(t).partition("::")
            try:
                rel = os.path.relpath(base, root)
            except ValueError:                     # другой диск под Windows
                rel = base
            out.append((rel if not rel.startswith("..") else base)
                       + (sep + node if sep else ""))
        return out

    def collect(self, targets: Sequence[str], *,
                timeout_s: Optional[float] = None) -> List[str]:
        """Список nodeid без запуска (``--collect-only``).


        Нужен, чтобы прогресс знал ЗНАМЕНАТЕЛЬ: «идёт тест 40 из 120» — это
        информация, «идёт тест 40» — нет.
        """
        items = self._normalize_targets(targets)
        argv = [self.python, "-m", "pytest", *items, "--collect-only", "-q",
                "--no-header", "-p", "no:cacheprovider", "-W", "ignore"]
        res = self.run(argv, timeout_s=timeout_s or self.policy.timeout_s)
        ids: List[str] = []
        for ln in res.output.splitlines():
            ln = ln.strip()
            if "::" in ln and not ln.startswith(("<", "E ", "ERROR", "=", "!")):
                ids.append(ln)
        return ids

    def run_pytest(self, targets: Sequence[str], *,

                   k: str = "", maxfail: int = 0,
                   extra_args: Sequence[str] = (),
                   timeout_s: Optional[float] = None,
                   count: bool = True,
                   on_progress: Optional[Callable[[Dict[str, Any]], None]] = None
                   ) -> PytestReport:
        """Прогнать тесты С ПРОГРЕССОМ и вернуть разобранный отчёт.

        Прогресс обязателен по развилке №7: прогон профильного файла идёт
        десятки секунд, и без событий пользователь считает, что приложение
        зависло. События (`start` / `test` / `done`) — данные; строку рисует
        :func:`progress_caption`.
        """
        items = self._normalize_targets(targets)
        limit = float(timeout_s or DEFAULT_PYTEST_TIMEOUT_S)
        started = time.monotonic()

        def emit(event: Dict[str, Any]) -> None:
            if on_progress is None:
                return
            try:
                on_progress({"elapsed_s": time.monotonic() - started, **event})
            except Exception:      # noqa: BLE001 — показ не роняет прогон
                pass

        total = 0
        if count:
            try:
                total = len(self.collect(items, timeout_s=min(limit, 120.0)))
            except SandboxError:
                total = 0

        shown = self.display_targets(items)
        emit({"kind": "start", "targets": shown, "total": total})


        argv = [self.python, "-m", "pytest", *items, "-v", "--no-header",
                "--color=no", "-p", "no:cacheprovider", "-W", "ignore"]
        if k:
            argv += ["-k", str(k)]
        if maxfail:
            argv += [f"--maxfail={int(maxfail)}"]
        argv += [str(a) for a in (extra_args or [])]

        outcomes: List[Dict[str, Any]] = []

        def on_output(line: str) -> None:
            rec = parse_test_line(line)
            if rec is None:
                return
            outcomes.append(rec)
            emit({"kind": "test", "done": len(outcomes), "total": total,
                  **rec})

        res = self.run(argv, timeout_s=limit, on_output=on_output)
        report = parse_pytest_output(res, outcomes=outcomes, total=total,
                                     targets=shown)

        emit({"kind": "done", "caption": report.caption(), "ok": report.ok,
              "failed": report.failed, "total": report.total})
        return report

    # -- жизненный цикл ---------------------------------------------------
    def close(self) -> None:
        """Освободить ресурсы (временный каталог, контейнер)."""

    def __enter__(self) -> "SandboxBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def describe(self) -> Dict[str, Any]:
        out = {"backend": self.name}
        out.update(self.policy.describe())
        return out
