"""assistant/turn_job.py — ХОД В ПОЛЁТЕ: ответ переживает rerun (iter91).

Живой отказ 13.08.2026: помощник начал отвечать, человек раскрыл рабочую
область — переписка «перезагрузилась», вопрос остался без ответа. Причина
НЕ в раскрытии формы и не в липкости зон (это правили iter88–90), а в том,
что ход выполнялся внутри прогона скрипта:

1. ``context.run_turn`` пишет вопрос в сессию СРАЗУ, а ответ — только после
   возврата из ``llm.run_tool_loop`` (``add_message`` + ``save_session``);
2. показ прогресса (``progress.caption`` в доке) — это ``st.*``, а Streamlit
   на каждой постановке ForwardMsg обслуживает отложенные запросы
   (``script_runner._enqueue_forward_msg`` → ``RerunException``);
3. ``RerunException`` наследует **BaseException**, а не ``Exception``,
   поэтому защита ``except Exception`` в ``run_turn`` её не держит — ход
   умирал ДО записи ответа. Деньги за запрос к модели уже списаны, а в
   переписке оставался один вопрос.

Решение — не «убрать точки прерывания» (тогда пропадёт живой прогресс,
против iter65), а вынести ход из прогона скрипта: он идёт в СВОЁМ потоке,
пишет результат в сессию сам и переживает сколько угодно перезапусков
скрипта. Тот же приём, что iter84 применил к артефактам: то, что должно
жить дольше одного прогона, не хранится в его памяти.

Модуль намеренно ЧИСТЫЙ (без Streamlit):

* :class:`TurnJob` — состояние хода: вопрос, события прогресса, результат,
  ошибка. Потокобезопасен, читается из главного потока в любой момент;
* :func:`start_turn` — запустить ход в фоне (``threading.Thread``);
* :func:`job_caption` — что показать человеку, пока ход идёт.

Инварианты, которые держит этот слой:

* **воркер не зовёт Streamlit.** В ``src/assistant/**`` нет ``import
  streamlit`` — ни у инструментов ядра, ни здесь; события прогресса
  СКЛАДЫВАЮТСЯ в буфер, а рисует их главный поток;
* **один ход на сессию.** Второй запуск при живом ходе — отказ
  (:class:`TurnBusy`), а не молчаливая потеря первого: два потока писали бы
  в одну сессию и в один файл;
* **отказ объясняет себя** (A0.6). Упавший воркер оставляет причину в
  ``error``, а ход помечается завершённым — «висящего навсегда» состояния
  быть не должно.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

#: Максимум событий прогресса в буфере. Ход ограничен сверху бюджетом
#: (``llm.MAX_ITERATIONS`` / ``TIME_BUDGET_S``), но буфер всё равно держим
#: конечным: показываем мы последнее, а не всю историю.
EVENTS_LIMIT = 200


class TurnBusy(RuntimeError):
    """Ход уже идёт: второй параллельный запуск запрещён."""


@dataclass
class TurnJob:
    """Ход ассистента, выполняющийся в СВОЁМ потоке.

    Живёт в состоянии приложения (у Streamlit — ``session_state``), поэтому
    переживает перезапуски скрипта: главный поток на каждом прогоне видит тот
    же объект и читает его состояние, не мешая воркеру.

    Поля пишет воркер, читает главный поток — доступ через методы под общим
    замком. Чтение ``result`` / ``error`` тоже безопасно: они выставляются
    ОДИН раз, перед тем как ход помечен завершённым.
    """

    question: str = ""
    images: List[str] = field(default_factory=list)
    #: Ход отправлен в ``t0`` (монотонные секунды) — для подписи «идёт N с».
    t0: float = field(default_factory=time.monotonic)
    #: Итог (``context.TurnResult``) — ставится воркером ПЕРЕД ``done``.
    result: Any = None
    #: Причина отказа воркера: ход завершён, но результата нет.
    error: str = ""
    #: Показан ли результат человеку. Ход остаётся в состоянии до показа,
    #: иначе rerun между «готово» и отрисовкой снова потерял бы ответ.
    shown: bool = False

    _events: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _done: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Запись (воркер)
    # ------------------------------------------------------------------
    def add_event(self, event: Dict[str, Any]) -> None:
        """Событие прогресса из воркера — В БУФЕР, не на экран.

        Рисует главный поток: воркеру Streamlit недоступен (у него нет
        ``ScriptRunContext``), а если бы и был — вызов ``st.*`` вернул бы ровно
        тот дефект, от которого мы уходим.
        """
        with self._lock:
            self._events.append(dict(event or {}))
            if len(self._events) > EVENTS_LIMIT:
                del self._events[:-EVENTS_LIMIT]

    def finish(self, result: Any = None, error: str = "") -> None:
        """Пометить ход завершённым. Вызывается воркером РОВНО один раз."""
        with self._lock:
            self.result = result
            self.error = str(error or "")
            self._done = True

    # ------------------------------------------------------------------
    # Чтение (главный поток)
    # ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        with self._lock:
            return self._done

    @property
    def running(self) -> bool:
        """Ход ещё в полёте: воркер работает, результата пока нет."""
        return not self.done

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self.t0)

    def events(self) -> List[Dict[str, Any]]:
        """Снимок буфера событий (копия — воркер продолжает писать)."""
        with self._lock:
            return list(self._events)

    def last_event(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._events[-1]) if self._events else None

    @property
    def n_events(self) -> int:
        with self._lock:
            return len(self._events)

    def join(self, timeout: Optional[float] = None) -> bool:
        """Дождаться воркера (для тестов и демо). ``True`` — ход завершён."""
        th = self._thread
        if th is not None:
            th.join(timeout)
        return self.done


def start_turn(run: Callable[..., Any], *, question: str,
               images: Optional[Sequence[str]] = None,
               thread_wrapper: Optional[Callable[[threading.Thread],
                                                 Optional[threading.Thread]]]
               = None, **run_kw: Any) -> TurnJob:
    """Запустить ход в ФОНЕ и вернуть его состояние.

    ``run`` — функция хода (в приложении это ``context.run_turn``). Ей
    передаются ``question``, ``images`` и ``on_event`` в буфер задания:
    вопрос и картинки задание всё равно помнит (их показывают человеку, пока
    ход идёт), и дублировать их в ``run_kw`` значило бы держать два источника
    истины об одном и том же. Остальное — через ``run_kw``.

    ``thread_wrapper`` — точка для окружения, которому нужно «пометить» поток.
    У Streamlit это ``add_script_run_ctx``: без контекста воркер не увидит
    ``session_state``, а его предупреждения о «missing ScriptRunContext» лезут
    в консоль. Обёртка оставлена ЯВНОЙ, чтобы решение принимал UI-слой, а этот
    модуль не знал про Streamlit.

    Исключение воркера НЕ теряется: причина ложится в ``job.error``, ход
    помечается завершённым. Иначе «вечно думающий» помощник выглядел бы
    зависанием, а причина не была бы названа (A0.6).
    """
    job = TurnJob(question=str(question or ""),
                  images=[str(x) for x in (images or [])])

    def worker() -> None:
        try:
            res = run(question=job.question, images=list(job.images),
                      on_event=job.add_event, **run_kw)
        except BaseException as exc:              # noqa: BLE001
            # BaseException, а не Exception: ровно этим ход и обрывался
            # (RerunException — BaseException). В воркере таких сигналов быть
            # не должно, но проглотить их молча нельзя тем более.
            job.finish(error=f"{type(exc).__name__}: {exc}")
        else:
            job.finish(result=res)

    th = threading.Thread(target=worker, name="doe-assistant-turn", daemon=True)
    if thread_wrapper is not None:
        th = thread_wrapper(th) or th
    job._thread = th
    th.start()
    return job


def job_caption(job: Optional[TurnJob], *,
                event_caption: Optional[Callable[[Dict[str, Any]], str]] = None
                ) -> str:
    """Строка «что сейчас происходит» для показа рядом с лентой.

    Считается ЗДЕСЬ (а не в доке), чтобы поведение проверялось без Streamlit.
    ``event_caption`` — переводчик события в текст (в приложении
    ``llm.progress_caption``): модуль не должен знать словарь событий цикла.
    """
    if job is None:
        return ""
    secs = f"{job.elapsed_s:.0f} с"
    if job.done:
        if job.error:
            return f"⛔ ход прерван: {job.error} ({secs})"
        return f"🏁 ответ получен ({secs})"
    head = ""
    ev = job.last_event()
    if ev is not None and event_caption is not None:
        try:
            head = str(event_caption(ev) or "")
        except Exception:                          # noqa: BLE001
            head = ""                              # показ не роняет ход
    head = head or "🧠 помощник думает…"
    return (f"{head} · идёт {secs} — можно продолжать работу, "
            f"ответ не потеряется")
