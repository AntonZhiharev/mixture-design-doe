# Copyright 2026 DOE contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 91 — ответ помощника ТЕРЯЛСЯ при перезапуске скрипта.

Живой отказ 13.08.2026 (после iter90): человек задал вопрос, помощник начал
работать; человек раскрыл рабочую область — «область помощника как бы
перезагрузилась, запрос скинулся, остался только мой промпт без ответа».

Разбор показал, что iter88–90 лечили симптомы (схлопывание формы, липкость
зон), а корень другой и общий для ЛЮБОГО виджета:

1. ход выполнялся ВНУТРИ прогона скрипта (``assistant_dock`` звал
   ``context.run_turn`` в теле функции отрисовки);
2. ``run_turn`` пишет вопрос в сессию СРАЗУ, а ответ — только после возврата
   из ``llm.run_tool_loop`` (``add_message`` + ``save_session``);
3. показ прогресса (``progress.caption``) — это ``st.*``, а Streamlit на
   каждой постановке ForwardMsg обслуживает отложенные запросы
   (``script_runner._enqueue_forward_msg`` → ``_maybe_handle_execution_control_request``
   → ``RerunException``);
4. **``RerunException`` наследует ``BaseException``**, а не ``Exception``,
   поэтому ``except Exception`` внутри ``run_turn`` его не держал: ход умирал
   ДО записи ответа. Деньги за обращение к модели уже списаны, в переписке —
   один вопрос.

Правка iter91: ход уходит в СВОЙ поток (:mod:`src.assistant.turn_job`), сам
пишет ответ в сессию и на диск, а прогон скрипта только ЧИТАЕТ его состояние.
Живой прогресс сохранён (требование iter65) — его перерисовывает
ФРАГМЕНТ (``st.fragment(run_every=…)``), а фрагментный прогон, в отличие от
полного, выполняющийся скрипт не перебивает.

Что проверяется ниже:

* задание-ход потокобезопасно, копит события и НЕ теряет результат;
* исключение воркера (в т.ч. ``BaseException``) становится причиной в
  ``error``, а ход помечается завершённым — «вечно думающего» состояния нет;
* второй параллельный ход запрещён явным отказом, а не молча;
* ``turn_job`` не знает про Streamlit (воркер не может дать точку прерывания);
* док больше не зовёт ``run_turn`` в теле отрисовки;
* headless-сценарий живого отказа: пока ход идёт, перезапуск скрипта его не
  убивает, а ответ доезжает в переписку.
"""
from __future__ import annotations

import ast
import inspect
import threading
import time
import warnings

import pytest

from src.assistant import turn_job as tj

warnings.filterwarnings("ignore")


# ======================================================================
# 1. Чистая логика задания: ход живёт дольше одного прогона
# ======================================================================
class TestTurnJob:
    def test_result_survives_and_is_readable_after_finish(self):
        """Итог хода доступен главному потоку — за этим весь шаг."""
        def run(question="", images=(), on_event=None):
            on_event({"kind": "llm_request", "iteration": 1})
            return f"ОТВЕТ на «{question}»"

        job = tj.start_turn(run, question="почему такой диапазон?")
        assert job.join(10) is True
        assert job.done and job.running is False
        assert job.result == "ОТВЕТ на «почему такой диапазон?»"
        assert job.error == ""

    def test_question_and_images_reach_the_turn_once(self):
        """Вопрос и картинки подставляет задание: два источника истины не нужны."""
        seen = {}

        def run(question="", images=(), on_event=None):
            seen.update(question=question, images=list(images))
            return "ok"

        job = tj.start_turn(run, question="что видно?", images=["sha1", "sha2"])
        job.join(10)
        assert seen == {"question": "что видно?", "images": ["sha1", "sha2"]}

    def test_events_are_buffered_not_drawn(self):
        """События копятся в задании: воркеру рисовать нечем и незачем."""
        def run(question="", images=(), on_event=None):
            for i in range(3):
                on_event({"kind": "tool_start", "tool": f"t{i}"})
            return "ok"

        job = tj.start_turn(run, question="q")
        job.join(10)
        assert job.n_events == 3
        assert [e["tool"] for e in job.events()] == ["t0", "t1", "t2"]
        assert job.last_event()["tool"] == "t2"

    def test_events_buffer_is_bounded(self):
        """Буфер конечный: показываем последнее, а не всю историю хода."""
        def run(question="", images=(), on_event=None):
            for i in range(tj.EVENTS_LIMIT + 50):
                on_event({"kind": "tool_end", "tool": str(i)})
            return "ok"

        job = tj.start_turn(run, question="q")
        job.join(30)
        assert job.n_events == tj.EVENTS_LIMIT
        # ...и в буфере остался ХВОСТ, а не начало
        assert job.last_event()["tool"] == str(tj.EVENTS_LIMIT + 49)

    def test_events_snapshot_is_a_copy(self):
        """Главный поток не должен портить буфер, который пишет воркер."""
        job = tj.TurnJob(question="q")
        job.add_event({"kind": "a"})
        snap = job.events()
        snap.append({"kind": "подделка"})
        assert job.n_events == 1


# ======================================================================
# 2. Отказы: причина названа, «вечно думающего» состояния нет (A0.6)
# ======================================================================
class TestFailures:
    def test_worker_exception_becomes_named_reason(self):
        def run(question="", images=(), on_event=None):
            raise ValueError("ключ не принят")

        job = tj.start_turn(run, question="q")
        job.join(10)
        assert job.done, "упавший ход обязан завершиться, а не висеть"
        assert job.result is None
        assert job.error == "ValueError: ключ не принят"

    def test_base_exception_is_not_swallowed(self):
        """Гвоздь диагноза: RerunException — BaseException, и её нельзя терять.

        Воркер ловит ``BaseException`` именно поэтому: ``except Exception``
        в ``run_turn`` пропускал управляющий сигнал Streamlit сквозь себя, и
        ход умирал молча, до записи ответа.
        """
        def run(question="", images=(), on_event=None):
            raise KeyboardInterrupt("сигнал управления")

        job = tj.start_turn(run, question="q")
        job.join(10)
        assert job.done and job.result is None
        assert "KeyboardInterrupt" in job.error

    def test_rerun_exception_class_is_not_an_exception(self):
        """Фиксируем факт, на котором стоит весь разбор (Streamlit 1.58)."""
        pytest.importorskip("streamlit")
        from streamlit.runtime.scriptrunner_utils.exceptions import (
            RerunException)
        assert issubclass(RerunException, BaseException)
        assert not issubclass(RerunException, Exception), (
            "если это изменится, старая защита `except Exception` заработает "
            "и разбор iter91 надо перечитать")


# ======================================================================
# 3. Один ход на сессию: два потока в один файл переписки не пишут
# ======================================================================
class _FakeCtx:
    """Контекст инструментов, которого достаточно доку для запуска хода."""
    runner = None
    session = None
    root = ""
    project = "проект"
    spec = None

    def require_spec(self):
        raise AttributeError("спеки нет")     # spec_hash_of вернёт ""


def test_dock_starts_turn_in_background_and_refuses_the_second(monkeypatch):
    """Ход уходит в ФОН, а второй параллельный получает явный отказ.

    Проверяется код дока (``start_background_turn``), а не копия его условия:
    ``st.session_state`` подменён обычным словарём, ``run_turn`` — заглушкой,
    которую можно держать «в полёте».
    """
    pytest.importorskip("streamlit")
    from src.apps import assistant_dock as dock

    state: dict = {}
    monkeypatch.setattr(dock.st, "session_state", state, raising=False)

    gate = threading.Event()
    started = threading.Event()

    def slow_turn(*, session=None, ctx=None, focus=None, spec_hash="",
                  kinds=(), question="", images=(), on_event=None):
        started.set()
        on_event({"kind": "llm_request", "iteration": 1})
        gate.wait(10)
        return f"ОТВЕТ: {question}"

    monkeypatch.setattr(dock.actx, "run_turn", slow_turn)

    job = dock.start_background_turn(None, _FakeCtx(), "почему 165…185?",
                                     focus=None)
    try:
        assert started.wait(10), "воркер не запустился"
        assert state[dock.K_JOB] is job, "ход должен жить в состоянии приложения"
        assert dock.current_job() is job
        assert job.running

        with pytest.raises(tj.TurnBusy):
            dock.start_background_turn(None, _FakeCtx(), "второй вопрос",
                                       focus=None)
        assert dock.current_job() is job, "первый ход не подменён вторым"
    finally:
        gate.set()
    assert job.join(10)
    assert job.result == "ОТВЕТ: почему 165…185?"


# ======================================================================
# 4. Контракты слоёв: воркер не может дать точку прерывания
# ======================================================================
def test_turn_job_layer_knows_nothing_about_streamlit():
    """Импорт Streamlit в воркере вернул бы ровно исходный дефект.

    Проверяем КОД (разбором дерева), а не текст файла: в докстрингах модуль
    как раз обязан объяснять, почему ``st.*`` внутри хода недопустим.
    """
    tree = ast.parse(inspect.getsource(tj))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "streamlit" not in imported
    # ...и ни одного обращения к алиасу `st`
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "st" not in names


def test_dock_no_longer_runs_the_turn_inside_the_script_run():
    """Гвоздь iter91: в теле отрисовки хода больше НЕТ."""
    pytest.importorskip("streamlit")
    from src.apps import assistant_dock as dock

    body = inspect.getsource(dock.render_assistant_dock)
    assert "actx.run_turn" not in body, (
        "ход снова выполняется в прогоне скрипта — любой rerun его убьёт")
    assert "start_background_turn" in body


def test_progress_is_drawn_by_a_fragment_not_by_the_worker():
    """Живой прогресс сохранён (iter65), но рисует его ГЛАВНЫЙ поток."""
    pytest.importorskip("streamlit")
    from src.apps import assistant_dock as dock

    assert dock.PROGRESS_EVERY_S > 0
    tick = inspect.getsource(dock._draw_progress_tick)
    # Прогресс читает состояние задания и переводит событие той же чистой
    # функцией, что и раньше, — словарь событий не раздваивается.
    assert "job_caption" in tick and "progress_caption" in tick
    # Фрагментный прогон не перебивает скрипт, полный — нужен на финише.
    assert 'st.rerun(scope="app")' in tick
    frag = inspect.getsource(dock._turn_progress_fragment)
    assert "st.fragment" in frag and "run_every" in frag


def test_progress_is_not_drawn_into_an_externally_created_container():
    """Фрагмент рисует в СВОЁ место, иначе подписи накапливаются.

    Документированное поведение Streamlit 1.58: элементы фрагмента, отданные в
    созданный СНАРУЖИ контейнер, при фрагментных прогонах не очищаются, а
    копятся до следующего полного прогона. С ``run_every=1.5`` это давало бы
    новую строку «идёт N с» каждые полторы секунды.
    """
    pytest.importorskip("streamlit")
    from src.apps import assistant_dock as dock

    assert not inspect.signature(dock._render_turn_progress).parameters, (
        "прогрессу не нужен внешний контейнер — он рисует под лентой")
    body = inspect.getsource(dock.render_assistant_dock)
    assert "turn_slot" not in body


def test_job_caption_says_the_answer_will_not_be_lost():
    """Пока ход идёт, человек должен видеть, что работать можно дальше."""
    job = tj.TurnJob(question="q")
    job.add_event({"kind": "tool_start", "tool": "preflight"})
    text = tj.job_caption(job, event_caption=lambda e: "🔧 preflight…")
    assert "preflight" in text and "не потеряется" in text
    # Завершённый ход говорит об итоге, а не о «думает».
    job.finish(result="ok")
    assert "получен" in tj.job_caption(job)
    job2 = tj.TurnJob(question="q")
    job2.finish(error="LLMError: ключ не принят")
    assert "прерван" in tj.job_caption(job2) and "ключ" in tj.job_caption(job2)


def test_job_caption_survives_a_broken_event_translator():
    """Показ не имеет права ронять ход (A0.6)."""
    job = tj.TurnJob(question="q")
    job.add_event({"kind": "странное"})

    def boom(_e):
        raise RuntimeError("переводчик сломался")

    assert "помощник думает" in tj.job_caption(job, event_caption=boom)


# ======================================================================
# 5. Сценарий живого отказа: rerun ВО ВРЕМЯ хода не крадёт ответ
# ======================================================================
PROJECT = "pvc_edge_v1"

#: Та же референсная геометрия, что в iter65/84 (golden iter45/49/50).
NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["PBNK", "CPE"]},
    {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
     "share_range": [0.0, 0.70], "max_phr": 8.0},
    {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT", "min_phr": 3.0},
]


def _real_ctx(tmp_path, session):
    from src.assistant.tools import ToolContext
    from src.design.phr_sampler import PhrSpec
    return ToolContext(spec=PhrSpec.from_dicts(NODES), session=session,
                       root=str(tmp_path), project=PROJECT)


def _answer(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"total_tokens": 11}}


def test_answer_reaches_the_session_even_if_the_script_reruns_meanwhile(tmp_path):
    """ГВОЗДЬ ШАГА: перезапуск скрипта во время хода не отменяет ответ.

    Rerun имитируется честно: пока воркер ждёт «модель», главный поток
    поднимает ``RerunException`` — ровно то, что делает Streamlit при правке
    поля. До iter91 этот сигнал прилетал ВНУТРЬ хода (через показ прогресса) и
    убивал его; теперь ход идёт в своём потоке, и сигнал главного потока его
    не касается.
    """
    pytest.importorskip("streamlit")
    from streamlit.runtime.scriptrunner_utils.exceptions import RerunException

    from src.assistant import store
    from src.assistant.context import run_turn
    from src.assistant.session import new_session

    session = new_session(PROJECT)
    ctx = _real_ctx(tmp_path, session)
    hold = threading.Event()
    asked = threading.Event()

    def transport(payload, *, key="", timeout=0):
        asked.set()
        hold.wait(10)                     # «модель думает» — ход в полёте
        return _answer("## ОТВЕТ\nВерх DINP — договорённость цеха.")

    job = tj.start_turn(run_turn, question="Почему верх DINP 14?",
                        session=session, ctx=ctx, spec_hash="",
                        transport=transport)
    assert asked.wait(10), "ход не дошёл до обращения к модели"

    # --- главный поток «перезапускается» столько раз, сколько человек
    # --- шевелит виджеты; ход это переживает
    for _ in range(3):
        with pytest.raises(RerunException):
            raise RerunException(None)
        assert job.running, "ход не должен зависеть от прогонов скрипта"

    hold.set()
    assert job.join(15)
    res = job.result
    assert res is not None and res.ok, f"ход не довёл ответ: {job.error}"
    assert "договорённость" in res.text

    # Ответ лежит в сессии И на диске — переписка не покажет «вопрос без ответа»
    assert session.messages[0].role == "user"
    assert session.messages[-1].role == "assistant"
    assert "договорённость" in session.messages[-1].content
    assert store.session_exists(tmp_path, PROJECT)
    saved = store.load_session(tmp_path, PROJECT)
    assert saved.messages[-1].role == "assistant"
    assert "договорённость" in saved.messages[-1].content


def test_progress_events_are_collected_during_the_real_turn(tmp_path):
    """Прогресс идущего хода читается из задания, а не из прогона скрипта."""
    pytest.importorskip("streamlit")
    from src.assistant import llm
    from src.assistant.context import run_turn
    from src.assistant.session import new_session

    session = new_session(PROJECT)
    ctx = _real_ctx(tmp_path, session)
    hold = threading.Event()
    asked = threading.Event()

    def transport(payload, *, key="", timeout=0):
        asked.set()
        hold.wait(10)
        return _answer("## ОТВЕТ\nготово")

    job = tj.start_turn(run_turn, question="что дальше?", session=session,
                        ctx=ctx, spec_hash="", transport=transport)
    try:
        assert asked.wait(10)
        # Событие «запрос к модели» уже в буфере: подпись собирается той же
        # чистой функцией, что рисовал прежний синхронный ход.
        deadline = time.monotonic() + 5
        while job.n_events == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert job.n_events >= 1
        caption = tj.job_caption(job, event_caption=llm.progress_caption)
        assert "модели" in caption or "думает" in caption
    finally:
        hold.set()
    assert job.join(15)
