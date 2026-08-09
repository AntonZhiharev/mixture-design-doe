# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 62 / ASSISTANT_SPEC — ПЕСОЧНИЦА ассистента.

Песочница существует, чтобы ассистент проверял свои утверждения ИСПОЛНЕНИЕМ.
Но исполнять код, который написала модель, можно только на чётких условиях —
их и фиксируют тесты (DoD шага iter62):

  * **тайм-аут** — зависший подпроцесс убивается, частичный вывод остаётся,
    результат честно помечен (молчаливое ожидание — худший отказ для UI);
  * **отказ сети** — `socket`/`urlopen` в подпроцессе запрещены: интернет у
    ассистента есть, но ДРУГИМ каналом (`:online`), и это разные уровни знания;
  * **отказ записи** — репозиторий виден только на чтение, `tests/` защищён
    отдельно: тесты — контракт кампании (golden-числа iter45–57);
  * **`run_pytest` с прогрессом** — событие на каждый тест + разобранный отчёт
    (сколько прошло/упало, какие nodeid), чтобы «зелено» было проверяемым.

Плюс контракт переносимости: всё прикладное живёт на ОДНОМ абстрактном методе
`run()`, поэтому фейковый бэкенд (как будущий Docker) проходит те же проверки
`run_pytest`/`collect` без единой правки вызовов.
"""
import os
import sys

import pytest

from src.assistant.sandbox import (DEFAULT_TIMEOUT_S, PytestReport,
                                    SandboxBackend, SandboxError,
                                    SandboxPolicy, SandboxResult, clip_output,
                                    detect_denial, get_backend,
                                    parse_pytest_output, parse_test_line,
                                    progress_caption)
from src.assistant.sandbox.guard import NETWORK_MARK, WRITE_MARK, guard_source
from src.assistant.sandbox.subprocess_backend import (BACKEND_ENV,
                                                       SubprocessSandbox,
                                                       is_secret_key)
from src.assistant.session import new_session
from src.assistant.tools import ToolContext, ToolError, dispatch, tool_names
from src.assistant.tools.registry import READONLY, SANDBOX, is_long_running

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

#: Записанный выхлоп `pytest -v` — разбор проверяется БЕЗ запуска подпроцесса.
PYTEST_OUTPUT = """\
tests/unit/test_demo.py::TestA::test_one PASSED                          [ 25%]
tests/unit/test_demo.py::TestA::test_two FAILED                          [ 50%]
tests/unit/test_demo.py::test_three SKIPPED (нужен стенд)                [ 75%]
tests/unit/test_demo.py::test_four PASSED                                [100%]

=================================== FAILURES ===================================
E       assert 0.53 == pytest.approx(0.70)
=========================== short test summary info ============================
FAILED tests/unit/test_demo.py::TestA::test_two - assert 0.53 == 0.70
==================== 2 passed, 1 failed, 1 skipped in 3.21s ====================
"""


def _result(**kw) -> SandboxResult:
    base = dict(argv=["python", "-m", "pytest"], returncode=1,
                stdout=PYTEST_OUTPUT, stderr="", duration_s=4.0,
                backend="fake")
    base.update(kw)
    return SandboxResult(**base)


class ScriptedSandbox(SandboxBackend):
    """Бэкенд «как Docker завтра»: реализован ОДИН метод ``run``.

    Нужен, чтобы проверить, что `run_pytest`/`collect`/прогресс не зависят от
    способа исполнения: при переезде меняется только этот метод.
    """
    name = "scripted"

    def __init__(self, output=PYTEST_OUTPUT, returncode=1, ids=None, **kw):
        super().__init__(SandboxPolicy(repo_root=REPO_ROOT, workdir=REPO_ROOT,
                                       **kw))
        self.output = output
        self.returncode = returncode
        self.ids = ids if ids is not None else [
            "tests/unit/test_demo.py::TestA::test_one",
            "tests/unit/test_demo.py::TestA::test_two",
            "tests/unit/test_demo.py::test_three",
            "tests/unit/test_demo.py::test_four"]
        self.calls = []

    def run(self, argv, *, timeout_s=None, stdin_text=None, env=None,
            on_output=None):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        collecting = "--collect-only" in argv
        text = "\n".join(self.ids) if collecting else self.output
        for line in text.splitlines():
            if on_output is not None:
                on_output(line)
        return SandboxResult(argv=argv, returncode=0 if collecting
                             else self.returncode, stdout=text,
                             duration_s=0.01, backend=self.name)

    def _normalize_targets(self, targets):     # существование файлов не важно
        if isinstance(targets, str):
            targets = [targets]
        items = [str(t) for t in (targets or []) if str(t).strip()]
        if not items:
            return super()._normalize_targets(targets)   # тот же текст отказа
        return items

    def display_targets(self, targets):
        return [str(t) for t in targets]



# ======================================================================
# 1. Разбор выхлопа pytest — чистые функции
# ======================================================================
class TestParsing:

    def test_test_line_parsed_with_percent(self):
        rec = parse_test_line(
            "tests/unit/test_demo.py::TestA::test_one PASSED     [ 25%]")
        assert rec == {"nodeid": "tests/unit/test_demo.py::TestA::test_one",
                       "outcome": "PASSED", "percent": 25}

    def test_non_test_lines_ignored(self):
        for line in ("=== FAILURES ===", "collected 4 items", "",
                     "E   assert 0.53 == 0.70"):
            assert parse_test_line(line) is None

    def test_report_counts_come_from_summary(self):
        rep = parse_pytest_output(_result(), targets=["tests/unit/test_demo.py"])
        assert (rep.passed, rep.failed, rep.skipped) == (2, 1, 1)
        assert rep.ok is False
        assert rep.duration_s == pytest.approx(3.21)   # из строки pytest, не из wall-clock

    def test_failures_are_named(self):
        rep = parse_pytest_output(_result())
        assert rep.failures == ["tests/unit/test_demo.py::TestA::test_two"]

    def test_green_run_is_ok(self):
        out = ("tests/unit/t.py::test_a PASSED  [100%]\n"
               "============ 1 passed in 0.10s ============\n")
        rep = parse_pytest_output(_result(stdout=out, returncode=0))
        assert rep.ok and rep.passed == 1 and rep.failed == 0

    def test_no_tests_collected_is_not_green(self):
        """Код 5 (`no tests ran`) — НЕ «всё зелено», а промах по пути."""
        rep = parse_pytest_output(_result(stdout="no tests ran in 0.01s",
                                          returncode=5))
        assert rep.ok is False
        assert "не нашёл ни одного теста" in rep.note

    def test_collection_error_explained(self):
        """Ненулевой код без упавших тестов = ошибка сборки, так и говорим."""
        rep = parse_pytest_output(_result(
            stdout="ImportError while importing test module", returncode=2))
        assert rep.ok is False and not rep.failures
        assert "ошибка сборки/импорта" in rep.note

    def test_timeout_keeps_partial_outcomes(self):
        """Обрыв по тайм-ауту: итоговой строки нет, но что успело — учтено."""
        partial = ("tests/unit/t.py::test_a PASSED   [ 33%]\n"
                   "tests/unit/t.py::test_b PASSED   [ 66%]\n")
        rep = parse_pytest_output(_result(stdout=partial, returncode=-9,
                                          timed_out=True, duration_s=12.0))
        assert rep.timed_out and rep.passed == 2 and rep.ok is False
        assert "тайм-ауту" in rep.caption()

    def test_report_is_json_ready(self):
        d = parse_pytest_output(_result()).to_dict()
        import json
        assert json.loads(json.dumps(d))["failed"] == 1


class TestCaptions:

    def test_progress_captions_are_human(self):
        assert "pytest" in progress_caption(
            {"kind": "start", "targets": ["tests/unit/t.py"], "total": 4})
        line = progress_caption({"kind": "test", "done": 2, "total": 4,
                                 "percent": 50, "outcome": "FAILED",
                                 "nodeid": "t.py::test_b"})
        assert "2/4" in line and "50%" in line and "t.py::test_b" in line
        assert progress_caption({"kind": "done", "caption": "✅ 4 прошло"})

    def test_output_clipped_with_note(self):
        big = "x" * 5000
        out = clip_output(big, 1000)
        assert len(out) < len(big) and "усечён" in out

    def test_short_output_untouched(self):
        assert clip_output("привет", 1000) == "привет"


# ======================================================================
# 2. Политика: репозиторий на чтение, tests/ защищён отдельно
# ======================================================================
class TestPolicy:

    def _policy(self, tmp_path) -> SandboxPolicy:
        return SandboxPolicy(repo_root=REPO_ROOT, workdir=str(tmp_path))

    def test_repo_is_not_writable(self, tmp_path):
        roots = self._policy(tmp_path).effective_write_roots()
        assert os.path.abspath(REPO_ROOT) not in roots
        assert str(tmp_path) in roots

    def test_tests_dir_is_protected_explicitly(self, tmp_path):
        deny = self._policy(tmp_path).effective_deny_write()
        assert os.path.join(os.path.abspath(REPO_ROOT), "tests") in deny

    def test_network_off_by_default(self, tmp_path):
        payload = self._policy(tmp_path).guard_payload()
        assert payload["allow_network"] is False

    def test_describe_says_the_rules_in_words(self, tmp_path):
        d = self._policy(tmp_path).describe()
        assert d["network"] == "запрещена" and d["repo_access"] == "только чтение"
        assert "':online'" in d["note"]

    def test_denial_detected_by_marker(self):
        assert detect_denial(f"boom {NETWORK_MARK} ...") == "network"
        assert detect_denial(f"boom {WRITE_MARK} ...") == "write"
        assert detect_denial("обычная ошибка") == ""

    def test_guard_source_covers_open_and_socket(self):
        src = guard_source()
        assert "builtins.open" in src and "socket.socket" in src


# ======================================================================
# 3. Выбор бэкенда (задел под Docker)
# ======================================================================
class TestBackendSelection:

    def test_default_is_subprocess(self, tmp_path, monkeypatch):
        monkeypatch.delenv(BACKEND_ENV, raising=False)
        sb = get_backend(policy=SandboxPolicy(repo_root=REPO_ROOT,
                                              workdir=str(tmp_path)))
        try:
            assert sb.name == "subprocess"
        finally:
            sb.close()

    def test_docker_refuses_loudly(self, monkeypatch):
        """Тихий откат на subprocess был бы обманом: «я думал, это контейнер»."""
        monkeypatch.setenv(BACKEND_ENV, "docker")
        with pytest.raises(SandboxError) as exc:
            get_backend()
        assert "docker" in str(exc.value) and "не реализован" in str(exc.value)

    def test_unknown_backend_lists_available(self):
        with pytest.raises(SandboxError) as exc:
            get_backend("qemu")
        assert "subprocess" in str(exc.value)


# ======================================================================
# 4. Подпроцесс: тайм-аут, сеть, запись, секреты
# ======================================================================
@pytest.fixture()
def sandbox(tmp_path):
    sb = SubprocessSandbox(SandboxPolicy(repo_root=REPO_ROOT,
                                         workdir=str(tmp_path / "work"),
                                         timeout_s=60.0))
    yield sb
    sb.close()


class TestSubprocessSandbox:

    def test_runs_code_and_returns_stdout(self, sandbox):
        res = sandbox.run_python("print('привет из песочницы')")
        assert res.ok and "привет из песочницы" in res.stdout

    def test_timeout_kills_and_marks(self, sandbox):
        res = sandbox.run_python("import time\nprint('старт', flush=True)\n"
                                 "time.sleep(30)\nprint('конец')",
                                 timeout_s=2.0)
        assert res.timed_out is True and res.ok is False
        assert res.duration_s < 20                     # процесс действительно убит
        assert "старт" in res.stdout and "конец" not in res.stdout
        assert "тайм-аут" in res.note and "тайм-ауту" in res.caption()

    def test_network_is_refused(self, sandbox):
        res = sandbox.run_python(
            "import socket\ns = socket.socket()\ns.connect(('1.1.1.1', 80))")
        assert res.denied == "network" and res.ok is False
        assert ":online" in res.note

    def test_urlopen_is_refused(self, sandbox):
        res = sandbox.run_python(
            "import urllib.request as u\nu.urlopen('http://example.com')")
        assert res.denied == "network"

    def test_write_into_repo_is_refused(self, sandbox):
        target = os.path.join(REPO_ROOT, "src", "_hacked_by_sandbox.py").replace(
            "\\", "\\\\")
        res = sandbox.run_python(f"open('{target}', 'w').write('x')")
        assert res.denied == "write"
        assert not os.path.exists(os.path.join(REPO_ROOT, "src",
                                               "_hacked_by_sandbox.py"))

    def test_tests_dir_is_untouchable(self, sandbox):
        """Правка tests/ запрещена агенту вообще — это контракт кампании."""
        target = os.path.join(REPO_ROOT, "tests", "unit",
                              "test_iteration45_phr_spec.py").replace("\\", "\\\\")
        res = sandbox.run_python(f"open('{target}', 'a').write('# hack')")
        assert res.denied == "write" and "tests/" in res.note

    def test_delete_outside_workdir_is_refused(self, sandbox, tmp_path):
        victim = tmp_path / "keep_me.txt"      # temp разрешён на запись...
        victim.write_text("данные", encoding="utf-8")
        target = os.path.join(REPO_ROOT, "README.md").replace("\\", "\\\\")
        res = sandbox.run_python(f"import os\nos.remove('{target}')")
        assert res.denied == "write"
        assert os.path.exists(os.path.join(REPO_ROOT, "README.md"))

    def test_workdir_is_writable(self, sandbox):
        res = sandbox.run_python(
            "open('out.txt', 'w').write('ок')\nprint(open('out.txt').read())")
        assert res.ok and "ок" in res.stdout

    def test_repo_is_readable(self, sandbox):
        """Только чтение — значит ЧИТАТЬ можно: инструменты импортируют src."""
        res = sandbox.run_python(
            "from src.assistant.sandbox.base import DEFAULT_TIMEOUT_S\n"
            "print('импорт ок', DEFAULT_TIMEOUT_S)")
        assert res.ok and "импорт ок" in res.stdout

    def test_secrets_are_not_passed(self, sandbox, monkeypatch):
        """Код, написанный моделью, не должен видеть ключ модели."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-очень-секретный")
        res = sandbox.run_python(
            "import os\nprint('KEY=', os.environ.get('OPENROUTER_API_KEY'))")
        assert res.ok and "KEY= None" in res.stdout

    def test_is_secret_key_rules(self):
        assert is_secret_key("OPENROUTER_API_KEY") and is_secret_key("my_token")
        assert not is_secret_key("PATH")

    def test_empty_code_refused(self, sandbox):
        with pytest.raises(SandboxError):
            sandbox.run_python("   ")

    def test_zero_timeout_refused(self, sandbox):
        with pytest.raises(SandboxError):
            sandbox.run([sys.executable, "-c", "pass"], timeout_s=0)


# ======================================================================
# 5. run_pytest: прогресс и отчёт
# ======================================================================
TINY_TESTS = '''\
def test_green_one():
    assert 2 + 2 == 4


def test_green_two():
    assert "phr".upper() == "PHR"


def test_red():
    assert 0.53 == 0.70, "верх доли поехал"
'''


class TestRunPytestScripted:
    """Интерфейс проверяем на фейковом бэкенде — быстро и без подпроцесса."""

    def test_progress_event_per_test(self):
        sb = ScriptedSandbox()
        events = []
        rep = sb.run_pytest(["tests/unit/test_demo.py"], on_progress=events.append)
        kinds = [e["kind"] for e in events]
        assert kinds[0] == "start" and kinds[-1] == "done"
        tests = [e for e in events if e["kind"] == "test"]
        assert len(tests) == 4
        assert [t["done"] for t in tests] == [1, 2, 3, 4]
        assert tests[0]["total"] == 4          # знаменатель взят из collect
        assert rep.failed == 1

    def test_progress_handler_failure_does_not_break_run(self):
        def bad(_event):
            raise RuntimeError("UI упал")

        rep = ScriptedSandbox().run_pytest(["t.py"], on_progress=bad)
        assert rep.passed == 2                 # прогон дошёл до конца

    def test_targets_required(self):
        with pytest.raises(SandboxError) as exc:
            ScriptedSandbox().run_pytest([])
        assert "перечисли файлы тестов явно" in str(exc.value)

    def test_collect_returns_nodeids(self):
        ids = ScriptedSandbox().collect(["tests/unit/test_demo.py"])
        assert len(ids) == 4 and all("::" in i for i in ids)


class TestRunPytestReal:
    """То же самое, но настоящим подпроцессом (DoD: `run_pytest` работает)."""

    def test_real_run_reports_and_progresses(self, sandbox, tmp_path):
        target = tmp_path / "test_sandbox_demo.py"
        target.write_text(TINY_TESTS, encoding="utf-8")

        events = []
        rep = sandbox.run_pytest([str(target)], timeout_s=120,
                                 on_progress=events.append)

        assert isinstance(rep, PytestReport)
        assert (rep.passed, rep.failed) == (2, 1)
        assert rep.ok is False
        assert any(n.endswith("test_red") for n in rep.failures)
        tests = [e for e in events if e["kind"] == "test"]
        assert len(tests) == 3 and tests[-1]["total"] == 3
        assert "test_red" in rep.tail            # хвост объясняет, что упало

    def test_target_missing_explains(self, sandbox):
        with pytest.raises(SandboxError) as exc:
            sandbox.run_pytest(["tests/unit/нет_такого_файла.py"])
        assert "не найдена" in str(exc.value)


# ======================================================================
# 6. Инструменты класса sandbox в реестре
# ======================================================================
class TestSandboxTools:

    def _ctx(self, tmp_path, backend=None, **kw):
        s = new_session("pvc_edge_v1")
        return ToolContext(session=s, root=str(tmp_path / "campaigns"),
                           project="pvc_edge_v1",
                           extra={"sandbox": backend or ScriptedSandbox()},
                           **kw)

    def test_tools_are_registered_as_sandbox_class(self):
        names = tool_names([SANDBOX])
        assert {"run_python", "run_pytest", "list_tests"} <= set(names)
        assert "sandbox_info" in tool_names([READONLY])

    def test_long_running_flag_for_ui(self):
        assert is_long_running("run_pytest") and is_long_running("run_python")

    def test_not_callable_in_readonly_mode(self, tmp_path):
        """Класс доступа проверяет диспетчер, а не текст промпта."""
        ctx = self._ctx(tmp_path)
        with pytest.raises(ToolError) as exc:
            dispatch(ctx, "run_pytest", {"targets": ["t.py"]},
                     allowed_kinds=[READONLY])
        assert "sandbox" in str(exc.value)

    def test_run_pytest_tool_returns_report_and_artifact(self, tmp_path):
        ctx = self._ctx(tmp_path)
        out = dispatch(ctx, "run_pytest", {"targets": ["tests/unit/test_demo.py"]},
                       allowed_kinds=[SANDBOX])
        assert out["failed"] == 1 and out["failures"]
        assert os.path.exists(out["artifact"])
        assert ctx.session.artifacts[-1].tool == "run_pytest"

    def test_progress_handler_taken_from_context(self, tmp_path):
        events = []
        ctx = self._ctx(tmp_path)
        ctx.extra["on_progress"] = events.append
        dispatch(ctx, "run_pytest", {"targets": ["t.py"]}, allowed_kinds=[SANDBOX])
        assert [e["kind"] for e in events][:1] == ["start"]
        assert any(progress_caption(e).startswith(("✅", "⛔"))
                   for e in events if e["kind"] == "test")

    def test_sandbox_info_tells_the_limits(self, tmp_path):
        info = dispatch(self._ctx(tmp_path), "sandbox_info", {})
        assert info["network"] == "запрещена"
        assert info["backend"] in ("scripted", "subprocess")

    def test_timeout_argument_is_bounded(self, tmp_path):
        ctx = self._ctx(tmp_path)
        with pytest.raises(ToolError) as exc:
            dispatch(ctx, "run_python", {"code": "print(1)", "timeout_s": 99999},
                     allowed_kinds=[SANDBOX])
        assert "превышает предел" in str(exc.value)

    def test_unknown_argument_refused(self, tmp_path):
        with pytest.raises(ToolError):
            dispatch(self._ctx(tmp_path), "run_pytest",
                     {"targets": ["t.py"], "coverage": True},
                     allowed_kinds=[SANDBOX])

    def test_run_python_tool_saves_output(self, tmp_path, sandbox):
        ctx = self._ctx(tmp_path, backend=sandbox)
        out = dispatch(ctx, "run_python", {"code": "print('42')"},
                       allowed_kinds=[SANDBOX])
        assert out["ok"] and "42" in out["stdout"]
        assert os.path.exists(out["artifact"])

    def test_list_tests_empty_is_not_silent(self, tmp_path):
        ctx = self._ctx(tmp_path, backend=ScriptedSandbox(ids=[]))
        out = dispatch(ctx, "list_tests", {"targets": ["t.py"]},
                       allowed_kinds=[SANDBOX])
        assert out["n"] == 0 and "НЕ «всё в порядке»" in out["note"]

    def test_defaults_are_sane(self):
        assert DEFAULT_TIMEOUT_S > 0
