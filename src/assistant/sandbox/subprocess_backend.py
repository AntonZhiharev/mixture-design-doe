"""assistant/sandbox/subprocess_backend.py — песочница на подпроцессе (iter62).

Реализация :class:`~.base.SandboxBackend` для машины разработчика: отдельный
процесс Python со сторожем (:mod:`.guard`), собственным рабочим каталогом,
жёстким тайм-аутом и очищенным окружением.

Почему именно так (развилка №1 ASSISTANT_SPEC):

* **процесс, а не exec в текущем** — код, который написала модель, не должен
  делить память, импорты и `sys.path` с приложением: одна `os.chdir` или
  `numpy.seterr` из «проверочного» фрагмента поедет во все последующие
  расчёты кампании;
* **окружение чистится** — ключ OpenRouter и прочие секреты в подпроцесс НЕ
  попадают: инструмент, исполняющий сгенерированный код с доступом к ключу,
  превращает утечку в один `print(os.environ)`;
* **тайм-аут с убийством ДЕРЕВА процессов** — иначе `pytest`, породивший
  подпроцессы, оставляет их висеть после «прерывания», и следующий прогон
  идёт по занятым ресурсам;
* **вывод читается потоково** — прогресс `run_pytest` появляется по ходу, а не
  после (пользователь не должен думать, что приложение зависло).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .base import (SandboxBackend, SandboxError, SandboxPolicy, SandboxResult,
                   SECRET_PATTERNS, clip_output, denial_note, detect_denial,
                   timeout_note)
from .guard import GUARD_FILENAME, POLICY_ENV, guard_source

#: Каталог сторожа внутри рабочего каталога (первый в PYTHONPATH).
GUARD_DIRNAME = "_sandbox_guard"

#: Переменные прокси снимаются: даже при разрешённой сети «случайный» прокси
#: из окружения оператора — не то, что должен наследовать подпроцесс.
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY",
               "http_proxy", "https_proxy", "all_proxy", "ftp_proxy")


def is_secret_key(name: str) -> bool:
    """Похоже ли имя переменной окружения на секрет (тогда не передаём)."""
    up = str(name or "").upper()
    return any(p in up for p in SECRET_PATTERNS)


class SubprocessSandbox(SandboxBackend):
    """Песочница на ``subprocess`` — бэкенд по умолчанию.

    Рабочий каталог создаётся временным, если не задан явно, и удаляется в
    :meth:`close` (артефакты, которые нужно сохранить, копирует инструмент —
    в ``assistant/artifacts/`` кампании, а не в чужой temp).
    """
    name = "subprocess"

    def __init__(self, policy: Optional[SandboxPolicy] = None, *,
                 python: str = "", **kw: Any):
        super().__init__(policy, **kw)
        self._python = python or sys.executable or "python"
        self._owns_workdir = False
        if not self.policy.workdir:
            self.policy.workdir = tempfile.mkdtemp(prefix="doe_sandbox_")
            self._owns_workdir = True
        os.makedirs(self.policy.workdir, exist_ok=True)
        if not self.policy.repo_root:
            self.policy.repo_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self._guard_dir = self._install_guard()

    # ------------------------------------------------------------------
    @property
    def python(self) -> str:
        return self._python

    def _install_guard(self) -> str:
        """Записать ``sitecustomize.py`` сторожа в отдельный каталог."""
        d = Path(self.policy.workdir) / GUARD_DIRNAME
        d.mkdir(parents=True, exist_ok=True)
        (d / GUARD_FILENAME).write_text(guard_source(), encoding="utf-8")
        return str(d)

    def _child_env(self, extra: Optional[Dict[str, str]] = None
                   ) -> Dict[str, str]:
        """Окружение подпроцесса: без секретов, со сторожем и политикой."""
        env = {k: v for k, v in os.environ.items() if not is_secret_key(k)}
        for var in _PROXY_VARS:
            env.pop(var, None)

        paths = [self._guard_dir]
        if self.policy.repo_root:
            paths.append(os.path.abspath(self.policy.repo_root))
        old = env.get("PYTHONPATH", "")
        if old:
            paths.append(old)
        env["PYTHONPATH"] = os.pathsep.join(paths)

        env[POLICY_ENV] = json.dumps(self.policy.guard_payload(),
                                     ensure_ascii=False)
        # Байт-код пишет C-уровень мимо сторожа — не даём ему повода вообще.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # UTF-8 везде, а не только в потоках: под Windows локаль по умолчанию
        # cp1251/cp1252, и `open(..., 'w').write('доля')` из кода модели падал
        # бы UnicodeEncodeError — отчёты кампании пишутся по-русски.
        env["PYTHONUTF8"] = "1"
        env["DOE_SANDBOX"] = "1"

        for k, v in (extra or {}).items():
            env[str(k)] = str(v)
        return env

    # ------------------------------------------------------------------
    def run(self, argv: Sequence[str], *, timeout_s: Optional[float] = None,
            stdin_text: Optional[str] = None,
            env: Optional[Dict[str, str]] = None,
            on_output: Optional[Callable[[str], None]] = None
            ) -> SandboxResult:
        """Запустить команду, потоково читая вывод, с жёстким тайм-аутом."""
        cmd = [str(a) for a in (argv or [])]
        if not cmd:
            raise SandboxError("Пустая команда: нечего запускать в песочнице.")
        limit = float(self.policy.timeout_s if timeout_s is None else timeout_s)
        if limit <= 0:
            raise SandboxError("Тайм-аут песочницы должен быть > 0 секунд: "
                               "запуск без предела времени запрещён.")


        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd, cwd=self.policy.workdir, env=self._child_env(env),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin_text else subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as exc:
            raise SandboxError(
                f"Не удалось запустить команду {cmd[:2]} в песочнице: {exc}. "
                f"Проверь путь к интерпретатору ({self._python}).") from exc

        out_lines: List[str] = []
        err_lines: List[str] = []

        def pump(stream, sink: List[str], notify: bool) -> None:
            try:
                for line in iter(stream.readline, ""):
                    sink.append(line)
                    if notify and on_output is not None:
                        try:
                            on_output(line.rstrip("\r\n"))
                        except Exception:   # noqa: BLE001 — показ не роняет прогон
                            pass
            except (ValueError, OSError):
                pass                      # поток закрыт убийством процесса
            finally:
                try:
                    stream.close()
                except OSError:
                    pass

        threads = [
            threading.Thread(target=pump, args=(proc.stdout, out_lines, True),
                             daemon=True),
            threading.Thread(target=pump, args=(proc.stderr, err_lines, False),
                             daemon=True),
        ]
        for t in threads:
            t.start()

        if stdin_text and proc.stdin:
            try:
                proc.stdin.write(str(stdin_text))
                proc.stdin.close()
            except (OSError, ValueError):
                pass

        timed_out = False
        try:
            returncode = proc.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(proc)
            returncode = proc.poll() if proc.poll() is not None else -9
        for t in threads:
            t.join(timeout=2.0)

        duration = time.monotonic() - started
        limit_chars = int(self.policy.max_output_chars)
        stdout_raw = "".join(out_lines)
        stderr_raw = "".join(err_lines)
        stdout = clip_output(stdout_raw, limit_chars)
        stderr = clip_output(stderr_raw, limit_chars)

        denied = detect_denial(stdout_raw + "\n" + stderr_raw)
        note = ""
        if timed_out:
            note = timeout_note(limit)
        elif denied:
            note = denial_note(denied)

        return SandboxResult(
            argv=cmd, returncode=int(returncode), stdout=stdout, stderr=stderr,
            duration_s=duration, timed_out=timed_out, denied=denied,
            truncated=(len(stdout_raw) > limit_chars
                       or len(stderr_raw) > limit_chars),
            backend=self.name, workdir=self.policy.workdir, note=note)

    # ------------------------------------------------------------------
    @staticmethod
    def _kill_tree(proc: "subprocess.Popen") -> None:
        """Убить процесс ВМЕСТЕ С ПОТОМКАМИ.

        ``proc.kill()`` под Windows снимает только сам процесс: `pytest`,
        успевший породить дочерние, остался бы висеть и держать ресурсы —
        «прервали» превратилось бы в «сделали вид, что прервали».
        """
        if proc.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True, timeout=15, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
        else:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except (OSError, AttributeError):
                pass
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        if self._owns_workdir and self.policy.workdir:
            shutil.rmtree(self.policy.workdir, ignore_errors=True)


# ----------------------------------------------------------------------
# Выбор бэкенда
# ----------------------------------------------------------------------
#: Переменная окружения выбора бэкенда (развилка №1: задел под Docker).
BACKEND_ENV = "DOE_SANDBOX_BACKEND"

BACKENDS = {"subprocess": SubprocessSandbox}


def get_backend(name: str = "", *, policy: Optional[SandboxPolicy] = None,
                **kw: Any) -> SandboxBackend:
    """Создать бэкенд по имени (по умолчанию — из ``DOE_SANDBOX_BACKEND``).

    Незнакомое имя — ЯВНЫЙ отказ со списком доступных, а не тихий откат на
    subprocess: «я думал, что считаю в контейнере» — ровно тот случай, когда
    молчание опаснее ошибки (A0.6).
    """
    key = str(name or os.environ.get(BACKEND_ENV, "") or "subprocess").lower()
    if key == "docker":
        raise SandboxError(
            "Бэкенд 'docker' ещё не реализован (iter62 — subprocess). "
            "Интерфейс SandboxBackend к этому готов: достаточно реализовать "
            "метод run() поверх контейнера, вызовы инструментов и их тесты не "
            "изменятся. Пока снимите DOE_SANDBOX_BACKEND=docker.")
    cls = BACKENDS.get(key)
    if cls is None:
        raise SandboxError(
            f"Неизвестный бэкенд песочницы '{key}'. Доступны: "
            f"{sorted(BACKENDS)} (плюс заявленный 'docker' — не реализован).")
    return cls(policy, **kw)
