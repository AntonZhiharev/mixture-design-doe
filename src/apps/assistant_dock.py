"""apps/assistant_dock.py — ПАНЕЛЬ ДИАЛОГА с ассистентом-архитектором (iter65/69/72).

Отдельная вкладка «💬 Ассистент» была устроена так, что спросить «почему у
этой оси такой диапазон» можно было только уйдя с экрана, где эта ось видна.
Док решает ровно это: он виден на КАЖДОМ шаге потока, а контекст берёт из
``st.session_state['ui_focus']``, который публикует активная закладка рабочей
области.

**iter69 — раскладка и поведение ленты.** Панель переехала в ЛЕВУЮ колонку
(``st.columns(workspace.MAIN_COLUMNS)``, эскиз пользователя) и работает как
переписка в Cline: лента живёт в контейнере ФИКСИРОВАННОЙ высоты со своим
скроллом и автоскроллом к низу (история прокручивается вверх), поле ввода стоит
ПОД лентой и не двигается, а рабочая область справа от ответов ассистента больше
не ползёт. Служебные панели (фокус, интернет, ключ, подсказки) собраны в
свёрнутый экспандер НАД лентой: раньше они стояли между лентой и вводом, и
переписка «плавала». Чистая логика ленты — :mod:`src.apps.workspace`.

**iter72 — три зоны (эскиз пользователя).** Левая колонка целиком отдана
ассистенту: диалог + панели УТВЕРЖДЕНИЯ его предложений (пакеты спеки, патчи) —
применить/отклонить может только человек, и это часть работы с ассистентом.
Постоянная же ДОП-ИНФОРМАЦИЯ (📎 вложения, 🖼 выхлоп песочницы, 📌 состояние
сессии) переехала в ОТДЕЛЬНУЮ правую колонку (:func:`render_assistant_info`):
она нужна на разных закладках рабочей области, а под перепиской её было не
видно, пока не проскроллишь весь диалог.

Слой намеренно ТОНКИЙ. Вся логика — в :mod:`src.assistant.context` и
:mod:`src.assistant.views` (чистые функции, покрытые
``tests/unit/test_iteration65_assistant_ui.py``); здесь только виджеты и
состояние Streamlit:

* сессия живёт в проекте (``project_campaigns/<проект>/assistant``) и
  перечитывается при СМЕНЕ проекта — переписка не «переезжает» из проекта в
  проект вслед за пользователем;
* реестр подтверждений (iter63) лежит в ``st.session_state``: токен не
  переживает перезапуск приложения намеренно — подтверждение относится к
  конкретному сеансу работы человека;
* кнопки «Применить»/«Отклонить» зовут :func:`context.human_apply` /
  :func:`context.human_reject` — единственный путь к классу ``write``;
* долгий ход рисует прогресс (``llm.progress_caption``): пользователь не
  должен думать, что приложение зависло;
* **iter68 — мультимодальный ввод и графический вывод.** Скриншот и голос
  принимает штатный ``st.chat_input`` (``accept_file`` / ``accept_audio``,
  Streamlit ≥1.52), поэтому новых зависимостей нет. Графики и таблицы,
  посчитанные в песочнице, рисуются ``st.image``/``st.dataframe`` прямо в
  ответе;
* **iter93 — вставка скриншота из буфера (Ctrl+V).** Штатный ``chat_input``
  буфер обмена не поддерживает до сих пор (в ``ChatInput.*.js`` Streamlit 1.58
  нет ни ``paste``, ни ``clipboard``), и iter68 отложил вставку как «нужен
  сторонний компонент». Стороннего компонента не понадобилось: картинка из
  буфера подставляется в СКРЫТЫЙ загрузчик файлов того же поля ввода
  (:func:`workspace.chat_paste_js`), а дальше идёт обычный путь загрузки —
  ``_chat_submission`` не отличает вставленный скриншот от выбранного руками;
* **iter70 — числа отдельно от вывода.** Раздел ``## ЧИСЛА`` формата ответа
  (iter64) рисуется свёрнутым блоком ПОД ответом (:func:`_render_answer` →
  :func:`views.answer_view`), а не строкой в общем потоке markdown: там он
  читался как продолжение рассуждения, и измерение с интерпретацией выглядели
  одинаково весомо. Одинаково раскладываются и свежий ход, и история — иначе
  сообщение меняло бы вид после перезапуска приложения.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.apps import workspace as wsx
from src.assistant import config as ai_config
from src.assistant import context as actx
from src.assistant import files as afiles
from src.assistant import llm, session as asess, store, views
from src.assistant import turn_job as tj
from src.assistant.consent import ConsentRegistry
from src.assistant.context import UiFocus
from src.assistant.tools import AGENT_KINDS, ToolContext, ToolError
from src.core import project_ref as pref

#: Ключи session_state дока (все с префиксом — чтобы не столкнуться с потоком).
K_SESSION = "assistant_session"
K_PROJECT = "assistant_session_project"
K_CONSENT = "assistant_consent_registry"
K_LAST_TURN = "assistant_last_turn"
K_PENDING = "assistant_pending_question"
#: Вопрос НЕУДАВШЕГОСЯ хода (iter67): держим отдельно от K_PENDING, потому что
#: K_PENDING отправляется автоматически на следующем прогоне, а этот ждёт
#: явного нажатия кнопки — повтор к модели должен быть решением человека.
K_FAILED = "assistant_failed_question"
#: iter91: ХОД В ПОЛЁТЕ (:class:`src.assistant.turn_job.TurnJob`). Ход идёт в
#: своём потоке, а здесь живёт его состояние — поэтому перезапуск скрипта
#: (правка поля, раскрытая форма, переключённая закладка) больше не обрывает
#: ответ: раньше ход выполнялся ВНУТРИ прогона, и `RerunException`
#: (BaseException!) убивала его до записи ответа в сессию.
K_JOB = "assistant_turn_job"
#: iter76: уведомление о ПЕРЕКЛЮЧЕНИИ переписки при смене имени проекта.
#: Начиная с iter77 смена ИМЕНИ переписку уже не переключает (ключ — ссылка),
#: поэтому сообщение появляется только при смене САМОГО проекта (загрузили
#: другой, создали новый) — то есть когда диалог и правда другой.
K_SWITCH_MSG = "assistant_project_switch_msg"

#: iter77: ССЫЛКА текущего проекта в состоянии приложения. Ключ связи проекта
#: и переписки: имя каталога может совпадать у разных проектов и меняться,
#: ссылка — нет.
K_REF = "campaign_ref"


# ----------------------------------------------------------------------
# Состояние дока
# ----------------------------------------------------------------------
def current_project_ref() -> str:
    """ССЫЛКА текущего проекта (iter77) — истинный ключ связи с перепиской.

    Живёт в ``st.session_state['campaign_ref']`` как состояние ПРИЛОЖЕНИЯ
    (выставляет ``streamlit_app.main``: дефолтный проект получает ссылку ещё до
    сборки). Пустая строка = приложение стартует и ссылку ещё не проставили.
    """
    return str(st.session_state.get(K_REF, "") or "").strip()


def current_project(root: str = "") -> str:
    """Каталог текущего проекта: по ССЫЛКЕ, а не по имени из поля ввода.

    Это и есть багфикс iter77. Раньше здесь читалось поле «Имя проекта»
    (``campaign_name``), поэтому правка имени переключала переписку на другой
    каталог. Теперь имя — подпись, а адрес берётся из идентичности проекта.

    Совместимость: если ссылки в состоянии нет (например, док вызван в обход
    ``main`` — так делают старые тесты AppTest), поведение прежнее — каталог
    по имени. Молчаливого «пусто» здесь быть не должно: без каталога
    переписке некуда лечь.
    """
    ref = current_project_ref()
    if ref and root:
        ident = pref.find_by_ref(root, ref)
        if ident is not None:
            return ident.dirname
    return str(st.session_state.get("campaign_name", "") or "").strip()


def dock_focus() -> UiFocus:
    """Фокус интерфейса из ``st.session_state`` (чистая логика — в context).

    Секции потока рисуются в ЛЕВОЙ колонке раньше дока, поэтому здесь виден
    фокус текущего прогона, а не прошлого.
    """
    return actx.focus_from_state(st.session_state)


def dock_session(root: str, project: str):
    """Сессия ассистента текущего проекта (перечитывается при смене проекта).

    iter77: переписка адресуется ССЫЛКОЙ проекта, поэтому ПЕРЕИМЕНОВАНИЕ её
    больше не переключает (``project`` — каталог, полученный из идентичности,
    а не строка из поля ввода). Сообщение :data:`K_SWITCH_MSG` остаётся, но
    теперь означает то, что и должно: сменился САМ проект (загрузили другой,
    создали новый) — значит и диалог другой.
    """
    prev = st.session_state.get(K_PROJECT)
    if prev != project or K_SESSION not in st.session_state:
        st.session_state[K_SESSION] = store.load_session(root, project) \
            if project else store.load_session(root, "_scratch")
        st.session_state[K_PROJECT] = project
        if prev is not None and prev != project:
            st.session_state[K_SWITCH_MSG] = (
                f"Открыт другой проект: «{prev or '_scratch'}» → "
                f"«{project or '_scratch'}», переписка переключена на него. "
                f"Прежний диалог ЦЕЛ — он сохранён в "
                f"`project_campaigns/{prev or '_scratch'}/assistant/` и "
                f"вернётся вместе со своим проектом. Переименование проекта "
                f"переписку НЕ переключает: она привязана к ссылке проекта, "
                f"а не к имени.")
    return st.session_state[K_SESSION]


def consent_registry() -> ConsentRegistry:
    reg = st.session_state.get(K_CONSENT)
    if not isinstance(reg, ConsentRegistry):
        reg = ConsentRegistry()
        st.session_state[K_CONSENT] = reg
    return reg


def _setup_fields_snapshot() -> Dict[str, Any]:
    """iter76: снимок полей формы «🆕 Новый проект» для инструментов.

    До сборки проекта данные живут ТОЛЬКО в полях формы — без снимка помощник
    их не видел и не мог вносить точечные правки (замкнутый круг несобранного
    проекта). Логика отбора ключей — чистая ``campaign_state.setup_draft_fields``.
    """
    from src.apps import campaign_state as cs
    return cs.setup_draft_fields(st.session_state)


def dock_context(root: str, project: str, runner: Any, session: Any
                 ) -> ToolContext:
    """Контекст инструментов дока: движок + спека + сессия + подтверждения."""
    spec = getattr(runner, "phr_spec", None) if runner is not None else None
    return ToolContext(runner=runner, session=session, root=root,
                       project=project, spec=spec,
                       extra={"consent": consent_registry(),
                              # iter76: помощник видит поля формы сетапа
                              # (get_setup_fields / propose_setup_fields).
                              "setup_fields": _setup_fields_snapshot()})


def spec_hash_of(ctx: ToolContext) -> str:
    try:
        return str(ctx.require_spec().spec_hash())
    except (ToolError, AttributeError):
        return ""


def node_names(ctx: ToolContext) -> List[str]:
    try:
        spec = ctx.require_spec()
    except (ToolError, AttributeError):
        return []
    return [str(getattr(n, "name", n)) for n in getattr(spec, "nodes", [])]


# ----------------------------------------------------------------------
# Панели
# ----------------------------------------------------------------------
def _render_connection() -> None:
    with st.expander("⚙️ Модель и ключ", expanded=not ai_config.llm_available()):
        key_in = st.text_input("OpenRouter API key", type="password",
                               value=os.environ.get("OPENROUTER_API_KEY", ""),
                               key="dock_key")
        model_in = st.text_input("Модель", value=ai_config.model_name(),
                                 key="dock_model")
        if key_in:
            os.environ["OPENROUTER_API_KEY"] = key_in.strip()
        if model_in:
            os.environ["DOE_ASSISTANT_MODEL"] = model_in.strip()
        if st.button("💾 Сохранить в .env", key="dock_save_key"):
            try:
                path = ai_config.save_api_key(key_in, model=model_in)
                st.success(f"Сохранено в `{path}`")
            except (ValueError, OSError) as exc:
                st.error(str(exc))
        _render_limits()


def _render_limits() -> None:
    """Бюджет времени хода и предел шагов (iter92).

    Почему это настройка, а не константа: на кампании из 23 узлов один ход
    честно идёт дольше прежних 180 с (замер 14.08.2026 — 234 с), и упираться в
    предел, который нельзя поднять без правки кода, — тупик для технолога.
    Значения пишутся в тот же локальный `.env` и читаются на КАЖДЫЙ ход.
    """
    budget = st.number_input(
        "⏱ Бюджет времени на один ход, с", min_value=ai_config.MIN_TIME_BUDGET_S,
        max_value=1800.0, step=30.0, value=float(ai_config.time_budget_s()),
        key="dock_budget",
        help="Ход прерывается по этому пределу. Прерванный ход НЕ пропадает: "
             "сделанные вызовы и их результаты остаются в переписке, и повтор "
             "вопроса продолжает работу с этого места.")
    iters = st.number_input(
        "🔧 Предел обращений к инструментам за ход", min_value=1, max_value=40,
        step=1, value=int(ai_config.max_iterations()), key="dock_iters",
        help="Один шаг = один запрос к модели плюс заказанные ею вызовы.")
    if st.button("💾 Сохранить лимиты", key="dock_save_limits"):
        try:
            path = ai_config.save_limits(budget_s=float(budget),
                                         iterations=int(iters))
            st.success(f"Лимиты сохранены в `{path}` — действуют со следующего "
                       f"вопроса.")
        except (ValueError, OSError) as exc:
            st.error(str(exc))


def _render_focus(focus: UiFocus, ctx: ToolContext) -> UiFocus:
    """Где мы сейчас + ручное уточнение узла (фокус секции имеет приоритет)."""
    st.caption("📍 " + actx.focus_caption(focus))
    names = node_names(ctx)
    if names:
        options = ["— не выбран —"] + names
        idx = options.index(focus.node) if focus.node in options else 0
        picked = st.selectbox("Узел в фокусе", options, index=idx,
                              key="assistant_focus_node_pick")
        if picked != "— не выбран —":
            focus.node = picked
    return focus


def _render_suggestions(focus: UiFocus, has_runner: bool) -> Optional[str]:
    """Кнопки «спросить по месту». Возвращает выбранный вопрос (или None)."""
    asked: Optional[str] = None
    sugs = actx.suggested_questions(focus, has_runner=has_runner)
    for i, s in enumerate(sugs):
        if st.button(s.label, key=f"dock_sug_{focus.section_key}_{i}",
                     disabled=not s.enabled, width="stretch",
                     help=(s.question if s.enabled else s.why)):
            asked = s.question
        if not s.enabled and s.why:
            st.caption(f"⛔ {s.why}")
    return asked


def _render_retry(res: Any, *, fresh: bool = False) -> None:
    """Отказ хода → всплывающее предупреждение + подсвеченная кнопка повтора.

    Сознательно НЕ авторетрай: скрытые попытки тратят деньги и время человека,
    а на неверном ключе или пустом счёте крутились бы бесконечно. Причина
    показывается тостом (`st.toast`) и остаётся в ленте, а повтор — явное
    нажатие. Вся логика — в :func:`views.retry_prompt` (чистая, iter67).

    Кнопка держится на ``K_LAST_TURN``, поэтому переживает перезапуски скрипта
    (любое движение виджета в доке), а ``fresh`` отделяет свежий отказ от
    перерисовки: тост всплывает один раз, а не на каждом прогоне.
    """
    prompt = views.retry_prompt(res)
    if not prompt:
        st.session_state.pop(K_FAILED, None)
        return
    st.session_state[K_FAILED] = prompt.question
    if fresh:
        st.toast(prompt.toast, icon=prompt.icon)
    st.warning(f"{prompt.icon} {prompt.toast}\n\n{prompt.hint}")
    if st.button(prompt.button_label, key="dock_retry", type="primary",
                 width="stretch",
                 help=f"Отправить тот же вопрос заново: «{prompt.question}»"):
        st.session_state[K_PENDING] = prompt.question
        st.session_state.pop(K_LAST_TURN, None)
        st.session_state.pop(K_FAILED, None)
        st.rerun()


def _render_patches(ctx: ToolContext, session) -> None:
    """Панель предложений: применить/отклонить может только ЧЕЛОВЕК (iter63)."""
    staged = session.staged_patches()
    st.markdown(f"**🧩 Предложенные патчи спеки: {len(staged)}**")
    if not staged:
        st.caption("Пусто. Правка появляется здесь, когда помощник предлагает "
                   "изменить границу узла — сам он её не применяет.")
        return
    st.dataframe(views.staged_patches_dataframe(session, only_staged=True),
                 width="stretch", hide_index=True)
    for p in staged:
        st.caption(f"`{p.id}` · {p.node}.{p.field_name}: {p.from_value} → "
                   f"{p.to_value}"
                   + (" · ⚠️ отпечаток спеки изменится" if p.affects_hash else ""))
        c = st.columns(2)
        if c[0].button("✅ Применить", key=f"dock_apply_{p.id}"):
            try:
                out = actx.human_apply(ctx, p.id, author="человек (UI)")
            except ToolError as exc:
                st.error(str(exc))
            else:
                # iter73: статус патча живёт в СЕССИИ — без записи на диск он
                # терялся при перезапуске, и применённый патч снова предлагался
                # к применению, хотя решение уже в журнале.
                actx.persist_session(ctx)
                st.success(views.apply_result_caption(out))
                if out.get("warning"):
                    st.warning(out["warning"])
                st.info(out.get("persist_hint", ""))
        reason = st.text_input("причина отказа", key=f"dock_reason_{p.id}",
                               label_visibility="collapsed",
                               placeholder="почему отклоняем")
        if c[1].button("⛔ Отклонить", key=f"dock_reject_{p.id}"):
            if not reason.strip():
                st.error("Отказ тоже идёт в журнал решений — назовите причину.")
            else:
                try:
                    actx.human_reject(ctx, p.id, reason.strip(),
                                      author="человек (UI)")
                except ToolError as exc:
                    st.error(str(exc))
                else:
                    actx.persist_session(ctx)
                    st.success("Патч отклонён, решение записано в журнал.")


def _render_setup_edits(ctx: ToolContext, session, runner: Any) -> None:
    """Панель ПРАВОК ПОЛЕЙ формы сетапа (iter76): применяет человек.

    Пока проект не собран, данные живут в полях формы «🆕 Новый проект» —
    помощник предлагает их точечную правку (``propose_setup_fields``), а
    принятие переносит значения в поля тем же механизмом отложенного
    префилла, что и загрузка проекта.
    """
    staged = session.staged_setups()
    if not staged:
        return                     # пустая панель не рисуется: шаг редкий
    st.markdown(f"**📝 Предложенные правки полей формы: {len(staged)}**")
    if runner is not None:
        # iter94: правка применяется и при собранном проекте (поля — черновик
        # пересборки), но цена называется по СТАДИИ: пустая база ничего не
        # стоит, измеренная — стоит всех опытов. Прежде кнопка была disabled, и
        # правка застревала в стейдже даже там, где терять было нечего.
        n_pts = int(len(getattr(runner, "points", []) or []))
        n_br = int(len(getattr(runner, "branches", {}) or {}))
        if n_pts or n_br:
            st.warning(
                f"Проект ЖИВОЙ: {n_pts} измеренных точек, {n_br} веток. "
                f"Применение правки обновит ПОЛЯ ФОРМЫ; движок не изменится, "
                f"пока вы не нажмёте «🏗 Построить проект» — а сборка создаст "
                f"проект с ПУСТОЙ базой. Чтобы сохранить опыты, правьте живой "
                f"проект панелями «📋 Настройки проекта» и «🧬 Эволюция схемы».")
        else:
            st.info("Проект собран, но не измерен (0 точек, 0 веток): правка "
                    "полей и пересборка сейчас ничего не стоят.")
    for s in staged:
        st.caption(f"`{s.id}` · {s.label or 'без метки'} · полей: "
                   f"{len(s.fields)}" + (f" · {s.rationale}" if s.rationale
                                         else ""))
        st.dataframe(pd.DataFrame(
            [{"поле": k, "новое значение": str(v)}
             for k, v in s.fields.items()]),
            width="stretch", hide_index=True)
        c = st.columns(2)
        if c[0].button("✅ Применить правку", key=f"dock_apply_setup_{s.id}",
                       help="Значения лягут в поля формы «🆕 Новый проект»; "
                            "проект соберёт кнопка «🏗 Построить проект»"):
            try:
                out = actx.human_apply_setup(ctx, s.id, author="человек (UI)")
            except ToolError as exc:
                st.error(str(exc))
            else:
                st.session_state["setup_prefill_pending"] = dict(
                    out.get("setup_prefill", {}) or {})
                st.session_state["camp_project_pkg_msg"] = out.get(
                    "next_step", "")
                actx.persist_session(ctx)
                st.success(out.get("next_step", "Правка применена."))
                # iter94: цена пересборки живого проекта — отдельной строкой,
                # чтобы она не потерялась в success-сообщении.
                if out.get("warning"):
                    st.warning(out["warning"])
                st.rerun()
        reason = st.text_input("причина отказа",
                               key=f"dock_setup_reason_{s.id}",
                               label_visibility="collapsed",
                               placeholder="почему не берём эту правку")
        if c[1].button("⛔ Отклонить правку", key=f"dock_reject_setup_{s.id}"):
            if not reason.strip():
                st.error("Отказ тоже идёт в журнал решений — назовите причину.")
            else:
                try:
                    actx.human_reject_setup(ctx, s.id, reason.strip(),
                                            author="человек (UI)")
                except ToolError as exc:
                    st.error(str(exc))
                else:
                    actx.persist_session(ctx)
                    st.success("Правка отклонена, решение записано в журнал.")


def _render_project_packages(ctx: ToolContext, session, runner: Any) -> None:
    """Панель ПАКЕТОВ ПРОЕКТА: рождение проекта из предложения ассистента (iter73).

    Закрывает отказ, с которого начался шаг: «Применить спеку» в пустой сессии
    отрабатывало «успешно» и ничего не меняло — писать геометрию было некуда,
    потому что проекта не существовало. Здесь принимается ПРОЕКТ целиком, а
    принятие заполняет поля формы «🆕 Новый проект»: собирает проект штатная
    кнопка «🏗 Построить проект», путь сборки в приложении один.

    Манифест показывается ТАБЛИЦЕЙ по блокам (состав / отклики / оси / режимы /
    ковариаты / паспорт) — по требованию: из пакета должно быть видно, что
    именно загружается, потому что ввод идёт в несколько подходов.
    """
    staged = session.staged_projects()
    st.markdown(f"**🏗 Предложенные проекты (пакетом): {len(staged)}**")
    if not staged:
        st.caption("Пусто. Пакет появляется здесь, когда проекта в сессии нет и "
                   "помощник собирает его целиком: состав (phr-спека) + "
                   "отклики + процесс-оси с границами. Одной спекой проект не "
                   "заводится — откликов и осей в ней нет по схеме.")
        return
    st.dataframe(views.staged_projects_dataframe(session, only_staged=True),
                 width="stretch", hide_index=True)
    if runner is not None:
        st.warning("В сессии УЖЕ собран проект: принять пакет проекта нельзя — "
                   "иначе молча пропали бы измеренные точки и ветки. Для правки "
                   "состава нужен пакет спеки; отклики и процесс-оси меняются в "
                   "форме «🆕 Новый проект» при осознанной пересборке.")
    for p in staged:
        m = p.summary or {}
        st.caption(f"`{p.id}` · {p.label or 'без метки'} · "
                   f"{len(list(m.get('components', []) or []))} компонентов · "
                   f"{len(list(m.get('responses', []) or []))} откликов · "
                   f"{len(list(m.get('process', []) or []))} процесс-осей · "
                   f"hash {str(m.get('spec_hash', ''))[:12]}…")
        # Блоки таблицей — «что именно загружается» без чтения JSON глазами.
        st.dataframe(views.project_blocks_dataframe(m),
                     width="stretch", hide_index=True)
        with st.expander(f"JSON пакета проекта {p.id}", expanded=False):
            st.json(p.payload())
        c = st.columns(2)
        if c[0].button("✅ Принять проект", key=f"dock_apply_proj_{p.id}",
                       disabled=runner is not None,
                       help=("Заполнит поля формы «🆕 Новый проект» на закладке "
                             "«🌱 Старт»; собрать проект останется кнопкой "
                             "«🏗 Построить проект»")
                            if runner is None else
                            "Проект уже собран — пакетом он не заводится заново"):
            try:
                out = actx.human_apply_project(ctx, p.id, author="человек (UI)")
            except ToolError as exc:
                st.error(str(exc))
            else:
                # Префилл применяется ОТЛОЖЕННО (тот же механизм, что у
                # загрузки проекта): менять ключ уже созданного виджета
                # Streamlit запрещает, а форма сетапа рисуется в этом же прогоне.
                st.session_state["setup_prefill_pending"] = dict(
                    out.get("setup_prefill", {}) or {})
                st.session_state["camp_project_pkg_msg"] = out.get(
                    "next_step", "")
                actx.persist_session(ctx)
                st.success(views.project_apply_caption(out))
                st.info(out.get("next_step", ""))
                st.rerun()
        reason = st.text_input("причина отказа", key=f"dock_proj_reason_{p.id}",
                               label_visibility="collapsed",
                               placeholder="почему не берём такой проект")
        if c[1].button("⛔ Отклонить проект", key=f"dock_reject_proj_{p.id}"):
            if not reason.strip():
                st.error("Отказ тоже идёт в журнал решений — назовите причину.")
            else:
                try:
                    actx.human_reject_project(ctx, p.id, reason.strip(),
                                              author="человек (UI)")
                except ToolError as exc:
                    st.error(str(exc))
                else:
                    actx.persist_session(ctx)
                    st.success("Пакет проекта отклонён, решение записано в "
                               "журнал.")


def _render_spec_packages(ctx: ToolContext, session) -> None:
    """Панель ПАКЕТОВ спеки: первичный ввод и эволюция геометрии (iter71).

    Отдельно от панели патчей намеренно: патч двигает границу, а пакет меняет
    СОСТАВ и роли — цена решения другая, и показывать их одной таблицей значило
    бы уравнять «поднять верх DINP» и «переписать геометрию кампании».

    JSON пакета доступен целиком (свёрнутым блоком): человек утверждает то, что
    может прочитать, а не то, что ему пересказали.
    """
    staged = session.staged_specs()
    st.markdown(f"**🧬 Предложенные спеки (пакетом): {len(staged)}**")
    if not staged:
        st.caption("Пусто. Пакет появляется здесь, когда помощник собирает "
                   "спеку целиком — первичный ввод состава или его изменение "
                   "(новый узел, удаление, смена роли). Сам он её не применяет.")
        return
    st.dataframe(views.staged_specs_dataframe(session, only_staged=True),
                 width="stretch", hide_index=True)
    for s in staged:
        d = s.summary or {}
        head = (f"`{s.id}` · {s.label or 'без метки'} · "
                f"{'первичный ввод' if d.get('first_spec') else 'эволюция'} · "
                f"{d.get('q_after', 0)} компонентов, dim z = "
                f"{d.get('dim_z_after', 0)}")
        st.caption(head + (" · ⚠️ отпечаток спеки изменится"
                           if d.get("affects_hash") else ""))
        if d.get("removed"):
            st.warning(f"Пакет УДАЛЯЕТ узлы: {', '.join(d['removed'])}. "
                       f"Точки, собранные в прежней геометрии, к новой не "
                       f"относятся.")
        with st.expander(f"JSON пакета {s.id} ({len(s.nodes)} узлов)",
                         expanded=False):
            st.json(s.payload())
        c = st.columns(2)
        if c[0].button("✅ Применить спеку", key=f"dock_apply_spec_{s.id}"):
            try:
                out = actx.human_apply_spec(ctx, s.id, author="человек (UI)")
            except ToolError as exc:
                st.error(str(exc))
            else:
                # iter73: см. панель патчей — статус пакета обязан пережить
                # перезапуск приложения (наблюдалось: applied в журнале, staged
                # в session.json).
                actx.persist_session(ctx)
                st.success(views.spec_apply_caption(out))
                if out.get("warning"):
                    st.warning(out["warning"])
                st.info(out.get("persist_hint", ""))
        reason = st.text_input("причина отказа", key=f"dock_spec_reason_{s.id}",
                              label_visibility="collapsed",
                              placeholder="почему не берём эту геометрию")
        if c[1].button("⛔ Отклонить спеку", key=f"dock_reject_spec_{s.id}"):
            if not reason.strip():
                st.error("Отказ тоже идёт в журнал решений — назовите причину.")
            else:
                try:
                    actx.human_reject_spec(ctx, s.id, reason.strip(),
                                           author="человек (UI)")
                except ToolError as exc:
                    st.error(str(exc))
                else:
                    actx.persist_session(ctx)
                    st.success("Пакет отклонён, решение записано в журнал.")


def _show_attachment_image(session, root: str, project: str, ident: str) -> None:
    """Показать приложенную картинку в ленте (файл лежит в проекте)."""
    att = afiles.find_attachment(session, ident)
    if att is None:
        return
    path = afiles.attachment_path(root, project, att)
    if path.exists():
        st.image(str(path), caption=att.name, width=320)
    else:
        # Ссылка есть, файла нет: молчать нельзя — иначе непонятно, почему
        # ассистент «не видит» картинку (A0.6).
        st.warning(f"Файл изображения «{att.name}» не найден в проекте.")


def _render_message_images(msg, session, root: str, project: str) -> None:
    """Картинки прошлых сообщений: переписка со скриншотами должна читаться."""
    for sha in list(getattr(msg, "images", []) or []):
        _show_attachment_image(session, root, project, sha)


def _render_answer(text: str) -> None:
    """Ответ ассистента: текст, а раздел «ЧИСЛА» — свёрнутым блоком (iter70).

    Раздел формата (iter64) нужен ради трассируемости — каждое число подписано
    инструментом. Но в общем потоке markdown он читался как продолжение
    рассуждения: измерение и интерпретация выглядели одинаково весомо, а модель
    ещё и дублировала значения выше. Здесь числа стоят ПОД ответом и свёрнуты:
    вывод виден сразу, проверка — на один клик. Разбор чистый
    (:func:`views.answer_view`), виджет ничего не режет сам.
    """
    view = views.answer_view(text)
    st.markdown(view.text or "—")
    if view.has_numbers:
        with st.expander(view.numbers_title):
            st.markdown(view.numbers)


def _render_outputs(outputs: List[views.OutputFile], *, scope: str) -> None:
    """Выхлоп песочницы КАРТИНКОЙ и ТАБЛИЦЕЙ, а не только строкой пути (iter68).

    До этого шага график, построенный кодом модели, оставался во временном
    каталоге и до человека не доходил вообще: «вывод песочницы» выглядел чисто
    текстовым. Файлы уже перенесены в кампанию
    (:func:`assistant.tools.sandbox_tools.collect_outputs`), здесь только показ.

    ``scope`` — МЕСТО показа (``turn`` — в ответе хода, ``panel`` — в панели
    артефактов). Один и тот же файл рисуется в ОБОИХ местах, а ключ виджета
    Streamlit обязан быть уникальным на странице: ключ только по имени файла
    ронял страницу целиком (``StreamlitDuplicateElementKey``) как раз тогда,
    когда свежий артефакт попадал ещё и в хвост последних — то есть после
    нескольких прогонов подряд.
    """
    for i, o in enumerate(outputs):
        st.caption(o.caption)
        if o.kind == "image":
            st.image(o.path, width="stretch")
        elif o.kind == "table":
            try:
                df = (pd.read_excel(o.path) if o.path.lower().endswith(".xlsx")
                      else pd.read_csv(o.path,
                                       sep="\t" if o.path.lower().endswith(".tsv")
                                       else ","))
            except (OSError, ValueError) as exc:
                # A0.6: таблица не разобралась — говорим об этом и даём файл,
                # а не показываем пустое место.
                st.warning(f"Таблицу «{o.name}» не удалось прочитать: {exc}")
            else:
                st.dataframe(df, width="stretch", hide_index=True)
        try:
            with open(o.path, "rb") as fh:
                # Ключ уникален по МЕСТУ показа и позиции в списке: имени файла
                # недостаточно — один артефакт виден и в ходе, и в панели.
                st.download_button("⬇️ Скачать", fh.read(), file_name=o.name,
                                   key=f"dock_dl_{scope}_{i}_{o.name}")
        except OSError:
            pass                       # файл исчез — показ уже состоялся


def _chat_submission(session, root: str, project: str):
    """Разобрать ввод чата: текст + картинки + голос → ``(вопрос, [sha256])``.

    Файлы кладутся вложениями сессии (дедуп по sha256 уже есть), голос
    распознаётся ДО хода: в переписке остаётся текст, который человек видит и
    может поправить, а не непрослушиваемая запись.
    """
    # iter93: мост вставки ставится ПЕРЕД полем — скрипт ищет поле по классу
    # `st-key-<ключ>` в уже отрисованном документе, а слушатель висит на
    # `document` и переживает перерисовку ленты. Ключ берётся из слоя раскладки:
    # он же зашит в селектор моста, и расхождение сломало бы вставку молча.
    st.html(wsx.chat_paste_js(wsx.DOCK_INPUT_KEY), unsafe_allow_javascript=True)
    typed = st.chat_input(
        "Спросите про эту ось, границу или следующий шаг… "
        "(📎 скриншот — можно вставить из буфера, 🎤 голос)",
        key=wsx.DOCK_INPUT_KEY, accept_file="multiple",
        file_type=["png", "jpg", "jpeg", "webp", "gif", "txt", "md", "csv",
                   "json", "xlsx", "docx", "pdf"],
        accept_audio=True)
    if typed is None:
        return None, []
    if isinstance(typed, str):          # старое поведение: только текст
        return typed, []

    question = str(getattr(typed, "text", "") or "")
    images: List[str] = []
    for up in list(getattr(typed, "files", []) or []):
        try:
            att = afiles.attach_file(session, root, up.name, up.getvalue(),
                                     project=project)
        except ValueError as exc:
            st.error(f"«{up.name}»: {exc}")
            continue
        store.save_session(session, root, project)
        if afiles.is_image_name(att.name):
            images.append(att.sha256)
        else:
            st.info(f"Документ «{att.name}» приложен к переписке — помощник "
                    f"прочитает его по запросу.")

    audio = getattr(typed, "audio", None)
    if audio is not None:
        try:
            heard = llm.transcribe(audio.getvalue(), fmt="wav")
        except llm.LLMError as exc:
            st.error(f"Речь не распознана: {exc}")
        else:
            st.caption(f"🎤 распознано ({heard['model']}): «{heard['text']}»")
            # Голос ДОПОЛНЯЕТ набранное, а не затирает: человек мог начать
            # печатать и договорить словами.
            question = (question + " " + heard["text"]).strip() if question \
                else heard["text"]
    return (question or None), images


def current_job() -> Optional[tj.TurnJob]:
    """Ход в полёте из состояния приложения (или ``None``)."""
    job = st.session_state.get(K_JOB)
    return job if isinstance(job, tj.TurnJob) else None


def start_background_turn(session, ctx: ToolContext, question: str, *,
                          focus: Any, images: Optional[List[str]] = None
                          ) -> tj.TurnJob:
    """Отправить вопрос в ФОНОВЫЙ ход и запомнить его в состоянии (iter91).

    Ключевая правка шага: ``run_turn`` больше не вызывается в теле прогона.
    Он идёт в своём потоке и сам пишет ответ в сессию и на диск, поэтому
    ``RerunException`` от любого виджета его не касается — перезапускается
    только отрисовка.

    Поток помечается ``add_script_run_ctx``: без контекста Streamlit сыплет в
    консоль «missing ScriptRunContext», а нам он нужен ещё и потому, что
    инструменты хода читают ``ctx``/сессию, собранные в главном потоке. Сам
    воркер ``st.*`` НЕ зовёт (в ``src/assistant/**`` нет ``import streamlit``)
    — иначе вернулся бы исходный дефект.
    """
    if (job := current_job()) is not None and job.running:
        raise tj.TurnBusy("Помощник ещё отвечает на предыдущий вопрос.")

    def _wrap(th):
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx
            return add_script_run_ctx(th, get_script_run_ctx())
        except Exception:                    # noqa: BLE001 — версия/сборка
            return th                        # без метки поток тоже работает

    # `question` и `images` подставляет само задание (оно их помнит и
    # показывает, пока ход идёт), остальное — аргументы run_turn.
    job = tj.start_turn(
        actx.run_turn, question=question, images=list(images or []),
        thread_wrapper=_wrap,
        session=session, ctx=ctx, focus=focus, spec_hash=spec_hash_of(ctx),
        kinds=AGENT_KINDS)
    st.session_state[K_JOB] = job
    return job


def get_script_run_ctx():
    """Контекст прогона (или ``None``) — обёртка ради читаемости и версий."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx as _g
        return _g()
    except Exception:                        # noqa: BLE001
        return None


#: Как часто перерисовывать прогресс идущего хода (с). Это ФРАГМЕНТНЫЙ прогон:
#: он не перебивает скрипт и не трогает остальную страницу — человек в это время
#: спокойно правит поля (`script_requests._fragment_run_should_not_preempt_script`).
PROGRESS_EVERY_S = 1.5


def _supports_fragment() -> bool:
    """Есть ли ``st.fragment`` (1.37+). Без него прогресс просто статичный."""
    return callable(getattr(st, "fragment", None))


def _render_turn_progress() -> None:
    """Живой прогресс идущего хода и показ итога завершённого (iter91).

    Ход выполняется в фоне, поэтому здесь только ЧТЕНИЕ его состояния:

    * идёт — подпись «что сейчас считается» (та же ``llm.progress_caption``,
      что и раньше) обновляется фрагментом раз в :data:`PROGRESS_EVERY_S`;
      как только ход завершился, фрагмент просит ПОЛНЫЙ прогон, чтобы ответ
      встал в ленту;
    * завершён и ещё не показан — подпись итога, аудит вызовов и кнопка
      повтора. Сам текст ответа перерисовывать не нужно: воркер уже дописал
      его в сессию, и лента выше нарисовала его как обычную реплику (iter84 —
      файлы расчёта тоже приезжают вместе с сообщением).

    Прогресс рисуется в СВОЁМ месте (под лентой), а не во внешнем контейнере
    ленты: элементы фрагмента, отданные в созданный СНАРУЖИ контейнер, при
    фрагментных прогонах не очищаются, а накапливаются (docstring
    ``st.fragment``, 1.58) — подписи «идёт N с» копились бы каждые
    :data:`PROGRESS_EVERY_S` секунд.
    """
    job = current_job()
    if job is None:
        # Отказ прошлого хода не должен исчезать при движении виджета: кнопка
        # повтора живёт до успешного ответа или до нового вопроса.
        _render_retry(st.session_state.get(K_LAST_TURN))
        return

    if job.running:
        if _supports_fragment():
            _turn_progress_fragment()
        else:
            st.caption(tj.job_caption(job,
                                      event_caption=llm.progress_caption))
        st.caption("⏳ Помощник отвечает в фоне. Можно листать закладки и "
                   "править поля — ответ придёт в переписку сам и не "
                   "потеряется.")
        return

    # --- ход завершён ---
    res = job.result
    if not job.shown:
        job.shown = True
        st.session_state[K_LAST_TURN] = res
        # Ответ воркер записал в сессию сам, но в ПАМЯТИ этого прогона сессия
        # могла быть перечитана с диска (смена проекта) — перезагружать её тут
        # не надо: следующий прогон возьмёт актуальную.
    if res is None:
        st.error(f"⛔ Ход не выполнен: {job.error or 'причина неизвестна'}")
        st.session_state.pop(K_JOB, None)
        return

    for err in list(getattr(res, "image_errors", []) or []):
        st.warning(f"Изображение не дошло до модели — {err}")
    st.caption(views.turn_caption(res))
    if getattr(res, "calls", None):
        with st.expander("🔧 Что было посчитано"):
            st.dataframe(views.tool_calls_dataframe(res.calls),
                         width="stretch", hide_index=True)
    _render_retry(res)


#: Собранный фрагмент прогресса (создаётся при первом показе, см.
#: :func:`_turn_progress_fragment`): `st.fragment` оборачивает функцию, и
#: пересоздавать обёртку на каждом прогоне не нужно.
_PROGRESS_FRAGMENT = None


def _turn_progress_fragment() -> None:
    """Обёртка: фрагмент создаётся один раз, а вызывается каждый прогон."""
    global _PROGRESS_FRAGMENT
    if _PROGRESS_FRAGMENT is None:
        _PROGRESS_FRAGMENT = st.fragment(run_every=PROGRESS_EVERY_S)(
            _draw_progress_tick)
    _PROGRESS_FRAGMENT()


def _draw_progress_tick() -> None:
    """Тело фрагмента: подпись прогресса, а на финише — полный прогон.

    ``st.rerun(scope="app")`` здесь и есть переход «фон → лента»: пока ход
    идёт, перерисовывается только этот фрагмент; когда воркер закончил, нужен
    полный прогон, иначе новая реплика не появится в ленте.
    """
    job = current_job()
    if job is None:
        return
    if job.done:
        st.rerun(scope="app")
    st.caption(tj.job_caption(job, event_caption=llm.progress_caption))
    st.caption("_ход идёт в фоне — перезагрузка страницы его не прервёт_")


def _supports_feed_container() -> bool:
    """Умеет ли установленный Streamlit ленту с ФИКСИРОВАННОЙ высотой и автоскроллом.

    ``height`` у контейнера появился в 1.29, ``autoscroll`` — позже (1.50+).
    Проверяем оба параметра фактически, а не по номеру версии: на более старом
    Streamlit лента останется частью общего скролла страницы (прежнее
    поведение), но приложение не упадёт.
    """
    try:
        import inspect
        params = inspect.signature(st.container).parameters
        return "height" in params and "autoscroll" in params
    except (TypeError, ValueError):  # pragma: no cover — экзотические сборки
        return False


def _render_attachments(session, root: str, project: str) -> None:
    with st.expander(f"📎 Вложения: {len(session.attachments)}"):
        up = st.file_uploader("Паспорт, ТДС, выгрузка, скриншот (txt/md/csv/"
                              "json/xlsx/docx/pdf/png/jpg)", key="dock_upload")
        if up is not None and st.button("Приложить к сессии", key="dock_attach"):
            try:
                afiles.attach_file(session, root, up.name, up.getvalue(),
                                   project=project)
            except ValueError as exc:
                st.error(str(exc))
            else:
                store.save_session(session, root, project)
                st.success(f"Файл «{up.name}» приложен.")
        if session.attachments:
            st.dataframe(views.attachments_dataframe(session),
                         width="stretch", hide_index=True)


def _render_decisions(ctx: ToolContext, root: str, project: str) -> None:
    """Панель ЖУРНАЛА РЕШЕНИЙ проекта (iter80): показ + ручная запись.

    Закрывает отказ, с которого начался шаг: человек принял пакет проекта,
    запись легла в ``decision_log.jsonl`` — и НИГДЕ на экране не появилась.
    Панели «🏗 Предложенные проекты», «🧬 Предложенные спеки», «🧩 Предложенные
    патчи» показывают ОЖИДАЮЩИЕ утверждения предложения и после применения
    пустеют штатно; журнал же — история принятых решений, и живёт он в файле
    проекта, а не в переписке.

    Ручная запись здесь потому, что до iter80 её не было вовсе: инструмент
    ``record_decision`` существовал, а нажать было нечего — решение «мел до
    100 phr, потому что так делает цех» технолог зафиксировать не мог. Модели
    этот инструмент по-прежнему недоступен (iter63): она предлагает
    формулировку, записывает человек.

    **iter95 — поля собраны в ``st.form``.** Живой отказ 14.08.2026: «начал
    редактировать поле — форма побелела и перезагрузилась, как будто нажал
    „сохранить“». Причина — модель исполнения Streamlit: значение виджета
    фиксируется при ПОТЕРЕ ФОКУСА (дописал «Решение», кликнул в «Почему») и
    каждая фиксация перезапускает ВЕСЬ скрипт — на большом приложении это
    секунды белого экрана на каждое поле. Внутри ``st.form`` виджеты значений
    не отправляют до нажатия кнопки, поэтому перезапуск остаётся ровно один —
    на «✍️ Записать…», как человек и ожидает. Паттерн тот же, что у цен
    проекта (``proj_prices_form`` в ``campaign_ui``).
    """
    recs = store.read_log(root, project, "decisions") if project else []
    with st.expander(f"📚 Журнал решений: {len(recs)}"):
        st.caption(views.decisions_caption(recs))
        if recs:
            st.dataframe(views.decisions_dataframe(recs),
                         width="stretch", hide_index=True)
        else:
            st.caption("Пусто. Сюда попадает КАЖДОЕ принятое и отклонённое "
                       "предложение помощника (проект, состав, границы узла) — "
                       "и решения, которые вы записываете сами. Через полгода "
                       "спор «почему тогда так решили» разрешает этот журнал, "
                       "а не память участников.")
        st.markdown("**✍️ Зафиксировать решение**")
        # iter95: форма — правка полей НЕ перезапускает страницу; полный
        # прогон происходит один раз, на кнопке записи (см. docstring).
        with st.form("dock_dec_form", border=False):
            title = st.text_input("Решение одной строкой", key="dock_dec_title",
                                  placeholder="мел до 100 phr для белых "
                                              "компаундов")
            why = st.text_area("Почему так решили", key="dock_dec_why",
                               height=80,
                               placeholder="на чём основано: опыт цеха, "
                                           "протокол, паспорт, расчёт")
            nodes = st.multiselect("Каких компонентов касается (необязательно)",
                                   options=node_names(ctx),
                                   key="dock_dec_nodes")
            who = st.text_input("Кто решил", key="dock_dec_author",
                                placeholder="фамилия или роль")
            submitted = st.form_submit_button("✍️ Записать решение в журнал")
        if submitted:
            if not title.strip() or not why.strip():
                st.error("Решение без обоснования в журнал не пишется: через "
                         "полгода «почему» будет важнее «что».")
            else:
                try:
                    out = actx.human_record_decision(
                        ctx, title.strip(), why.strip(), nodes=list(nodes),
                        author=who.strip() or "человек (UI)")
                except ToolError as exc:
                    st.error(str(exc))
                else:
                    rec = out.get("decision", {}) or {}
                    if rec.get("persisted"):
                        st.success("Решение записано в журнал проекта.")
                        st.rerun()
                    else:
                        # Молчаливой потери быть не должно: говорим прямо.
                        st.warning("Решение НЕ сохранено на диск: "
                                   + str(rec.get("note", "проект не выбран")))


def _render_local_facts(ctx: ToolContext, root: str, project: str) -> None:
    """Панель ФАКТОВ ЦЕХА (L1) — высший уровень знания (iter80).

    L1 отменяет литературу и справочники, поэтому автор факта — человек и
    только человек: факт, записанный моделью, отменял бы источники от своего
    имени (ASSISTANT_SPEC §370). Помощник читает эти записи (`get_local_facts`)
    и обязан считать их приоритетнее найденного в сети.

    Поля собраны в ``st.form`` (iter95) — по той же причине, что и журнал
    решений: без формы каждая потеря фокуса поля перезапускала весь скрипт.
    """
    recs = store.read_log(root, project, "local_facts") if project else []
    with st.expander(f"🏭 Факты производства (L1): {len(recs)}"):
        st.caption(views.facts_caption(recs))
        if recs:
            st.dataframe(views.local_facts_dataframe(recs),
                         width="stretch", hide_index=True)
        else:
            st.caption("Пусто. Здесь живёт знание вашего производства: "
                       "«смеситель типа A — не выше 120 °C», «плотность "
                       "компаунда измеряется, а не считается». Помощник "
                       "ставит эти записи ВЫШЕ литературы и справочников.")
        st.markdown("**✍️ Записать факт производства**")
        # iter95: форма — правка полей НЕ перезапускает страницу (см.
        # _render_decisions), полный прогон один — на кнопке записи.
        with st.form("dock_fact_form", border=False):
            stmt = st.text_area("Сам факт одной фразой", key="dock_fact_stmt",
                                height=80,
                                placeholder="смеситель типа A — не выше 120 °C")
            scope = st.text_input("К чему относится", key="dock_fact_scope",
                                  placeholder="компонент, свойство, участок, "
                                              "оборудование")
            src_txt = st.text_input("Откуда известно", key="dock_fact_src",
                                    placeholder="кто сказал / протокол / "
                                                "номер опыта")
            who = st.text_input("Кто утверждает", key="dock_fact_author",
                                placeholder="фамилия или роль")
            submitted = st.form_submit_button("✍️ Записать факт в журнал")
        if submitted:
            if not stmt.strip():
                st.error("Пустой факт записать нельзя.")
            else:
                try:
                    out = actx.human_add_local_fact(
                        ctx, stmt.strip(), scope=scope.strip(),
                        source=src_txt.strip(),
                        author=who.strip() or "технолог")
                except ToolError as exc:
                    st.error(str(exc))
                else:
                    rec = out.get("fact", {}) or {}
                    if rec.get("persisted"):
                        st.success("Факт записан. Помощник будет считать его "
                                   "важнее литературы и найденного в сети.")
                        st.rerun()
                    else:
                        st.warning("Факт НЕ сохранён на диск: "
                                   + str(rec.get("note", "проект не выбран")))


def _note_fields_form(ctx: ToolContext, note) -> Dict[str, Any]:
    """Поля предложенной записи как ВИДЖЕТЫ: человек правит их до фиксации.

    Виджеты рисуются по виду записи (``session.NOTE_FIELDS``), значения по
    умолчанию — то, что предложил помощник. Возвращается словарь «поле →
    значение из формы»: он и уходит в ``apply_note``, поэтому в журнал попадает
    именно то, что человек видел на экране.
    """
    src = dict(note.fields or {})
    out: Dict[str, Any] = {}
    if note.kind == asess.NOTE_DECISION:
        out["title"] = st.text_input(
            "Решение одной строкой", value=str(src.get("title", "") or ""),
            key=f"dock_note_title_{note.id}")
        out["rationale"] = st.text_area(
            "Почему так решили", value=str(src.get("rationale", "") or ""),
            height=80, key=f"dock_note_why_{note.id}")
        # Узлы предложения могут не входить в текущую спеку (её могло не быть
        # вовсе) — тогда multiselect с таким значением по умолчанию упал бы.
        proposed = [str(v) for v in (src.get("nodes") or [])]
        options = list(dict.fromkeys(node_names(ctx) + proposed))
        out["nodes"] = st.multiselect(
            "Каких компонентов касается", options=options, default=proposed,
            key=f"dock_note_nodes_{note.id}")
        out["author"] = st.text_input(
            "Кто решил", value=str(src.get("author", "") or ""),
            key=f"dock_note_author_{note.id}",
            placeholder="фамилия или роль")
    else:
        out["statement"] = st.text_area(
            "Сам факт одной фразой", value=str(src.get("statement", "") or ""),
            height=80, key=f"dock_note_stmt_{note.id}")
        out["scope"] = st.text_input(
            "К чему относится", value=str(src.get("scope", "") or ""),
            key=f"dock_note_scope_{note.id}")
        out["source"] = st.text_input(
            "Откуда известно", value=str(src.get("source", "") or ""),
            key=f"dock_note_src_{note.id}",
            placeholder="кто сказал / протокол / номер опыта")
        out["author"] = st.text_input(
            "Кто утверждает", value=str(src.get("author", "") or ""),
            key=f"dock_note_author_{note.id}",
            placeholder="фамилия или роль")
    return out


def _render_note_proposals(ctx: ToolContext, session) -> None:
    """Панель ПРЕДЛОЖЕННЫХ ЗАПИСЕЙ в журналы проекта (iter96).

    Закрывает разрыв, оставшийся после iter80: инструменты записи есть, кнопки
    ручного ввода есть, а предложение помощника доходило до журнала только
    текстом в ответе — формулировку из «## OPEN_QUESTIONS» человек переносил в
    поля посимвольно. На живой ПВХ-сессии это регулярно не делалось: решение
    обсудили, а журнал остался пустым.

    Устройство панели повторяет уже принятый в проекте контур утверждения
    (пакеты спеки/проекта, правки полей формы): помощник кладёт предложение в
    стейдж (``propose_decision`` / ``propose_fact``), человек ВИДИТ и ПРАВИТ
    поля и фиксирует кнопкой. Отличие одно и существенное: поля здесь
    редактируемые, потому что запись идёт от имени человека — он подписывает
    формулировку, а не одобряет чужую.

    Поля собраны в ``st.form`` (iter95): правка текста не перезапускает скрипт,
    полный прогон один — на нажатой кнопке.
    """
    staged = session.staged_notes()
    st.markdown(f"**📝 Предложенные записи в журналы: {len(staged)}**")
    st.caption(views.staged_notes_caption(session))
    if not staged:
        return
    for note in staged:
        head = (f"`{note.id}` · {views.note_kind_label(note.kind)}"
                + (f" · {note.label}" if note.label else ""))
        st.caption(head + (f" · зачем сейчас: {note.why_now}"
                           if note.why_now else ""))
        with st.form(f"dock_note_form_{note.id}", border=True):
            fields = _note_fields_form(ctx, note)
            reason = st.text_input(
                "Причина отказа (нужна только для «Отклонить»)",
                key=f"dock_note_reason_{note.id}",
                placeholder="почему не записываем")
            cols = st.columns(2)
            with cols[0]:
                save = st.form_submit_button(
                    views.note_button(note.kind),
                    help="Запись уйдёт в журнал проекта С ВАШЕЙ правкой полей")
            with cols[1]:
                drop = st.form_submit_button("⛔ Отклонить запись")
        if save:
            try:
                out = actx.human_apply_note(ctx, note.id, fields=fields,
                                            author="человек (UI)")
            except ToolError as exc:
                st.error(str(exc))
            else:
                actx.persist_session(ctx)
                st.success(views.note_apply_caption(out))
                if out.get("note"):
                    st.info(out["note"])
                st.rerun()
        elif drop:
            if not reason.strip():
                st.error("Отказ тоже идёт в журнал решений — назовите причину.")
            else:
                try:
                    actx.human_reject_note(ctx, note.id, reason.strip(),
                                           author="человек (UI)")
                except ToolError as exc:
                    st.error(str(exc))
                else:
                    actx.persist_session(ctx)
                    st.success("Запись отклонена, отказ занесён в журнал "
                               "решений.")
                    st.rerun()


def _render_artifacts(session) -> None:
    """Графики и таблицы прогонов ЭТОГО проекта (живут после перезапуска)."""
    shown = views.artifact_outputs(session)
    with st.expander(f"🖼 Файлы расчётов помощника: {len(shown)}"):
        if not shown:
            st.caption("Пусто. Здесь появляются графики и таблицы, которые "
                       "помощник построил своим расчётом (`run_python` с "
                       "`savefig`/`to_csv`).")
            return
        _render_outputs(shown, scope="panel")


# ----------------------------------------------------------------------
# Док целиком
# ----------------------------------------------------------------------
def render_assistant_dock(runner: Any = None, *, root: str = "") -> None:
    """Левая колонка (iter69): чат архитектора с контекстом ТЕКУЩЕЙ закладки.

    Порядок частей задаёт поведение переписки: настройки → ЛЕНТА (свой скролл,
    автоскролл к низу) → поле ввода → результаты хода. Ввод стоит под лентой и
    не уезжает, потому что растёт не страница, а прокрутка внутри ленты.
    """
    root = root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "project_campaigns")
    project = current_project(root) or "_scratch"
    session = dock_session(root, project)
    ctx = dock_context(root, project, runner, session)
    focus = dock_focus()

    st.subheader("💬 Помощник по проекту")
    # iter76: смена имени проекта переключает переписку — говорим об этом
    # явно, иначе диалог «пропадает» без объяснения (файл на диске цел).
    if st.session_state.get(K_SWITCH_MSG):
        st.warning(st.session_state.pop(K_SWITCH_MSG))
    if runner is None:
        st.info("Проект не собран: расчётной модели и базы опытов нет — на "
                "вопросы о проверке плана и расчётах ответить числами пока "
                "нечем.")

    # iter69: служебные панели (фокус, интернет, ключ, подсказки, вложения,
    # патчи) — НАД лентой и свёрнуты. Раньше они стояли между лентой и полем
    # ввода, и переписка «плавала»: ответ ассистента отодвигал ввод вниз,
    # а до истории приходилось прокручивать всю страницу.
    with st.expander("⚙️ Контекст вопроса и настройки", expanded=runner is None):
        st.caption("Помощник отвечает ЧИСЛАМИ ИЗ РАСЧЁТА: роли, действующие "
                   "границы, отпечаток спеки и проверку плана считает "
                   "программа, а не сама модель.")
        focus = _render_focus(focus, ctx)
        session.web_enabled = st.toggle(
            "🌐 Интернет (:online)", value=bool(session.web_enabled),
            key="dock_web",
            help="Всё, что придёт из сети, — уровень знания L2: локальный факт "
                 "цеха его отменяет.")
        _render_connection()
        # iter78: пока шаг не определён (приложение только открыли), «про
        # открытую закладку» — неправда: закладки в фокусе ещё нет, а кнопки
        # под этой подписью отвечают на вопросы первого входа.
        st.markdown("**С чего начать:**" if not focus.section_key
                    else "**Спросить про открытую закладку:**")
        asked = _render_suggestions(focus, runner is not None)

    # --- ЛЕНТА: свой скролл, история вверх, свежее внизу (как в Cline) ---
    typed, images = None, []
    feed = wsx.feed_items(session.messages)
    box = st.container(height=wsx.CHAT_FEED_HEIGHT, border=True,
                       autoscroll=True) if _supports_feed_container() \
        else st.container()
    with box:
        for pos, item in enumerate(feed):
            with st.chat_message(item.role):
                # Ответ ассистента и в истории раскладывается так же, как свежий
                # (числа — свёрнутым блоком): иначе одно и то же сообщение
                # выглядело бы по-разному до и после перезапуска приложения.
                if item.role == "assistant":
                    _render_answer(item.content)
                else:
                    st.markdown(item.content)
                for sha in item.images:
                    _show_attachment_image(session, root, project, sha)
                # iter84: график и таблица ЭТОГО ответа рисуются в ленте, а не
                # только на том прогоне, где ход выполнялся. Раньше показ шёл
                # исключительно из TurnResult.new_artifacts (память одного
                # прогона Streamlit), поэтому любой rerun — нажатая кнопка,
                # раскрытый экспандер, новый вопрос — убирал их из разговора:
                # файлы оставались в проекте и в панели «🖼 Файлы расчётов», а
                # в переписке ассистент выглядел так, будто потерял свой расчёт.
                if item.artifacts:
                    _render_outputs(views.message_outputs(session,
                                                          item.artifacts),
                                    scope=f"feed{pos}")
    st.caption(wsx.feed_hint(len(feed), wsx.dialog_count(session.messages)))

    # Поле ввода — ПОД лентой и не двигается: лента прокручивается внутри себя.
    typed, images = _chat_submission(session, root, project)
    question = typed or asked or st.session_state.pop(K_PENDING, None)

    # iter91: новый вопрос УХОДИТ В ФОН, а не выполняется в этом прогоне.
    if question or images:
        try:
            start_background_turn(session, ctx, question or "", focus=focus,
                                  images=images)
        except tj.TurnBusy as exc:
            # Два хода в одну сессию писать нельзя (общий файл переписки).
            # Молча проглотить вопрос тоже нельзя — человек его уже задал.
            st.warning(f"{exc} Дождитесь ответа и спросите снова — вопрос "
                       f"НЕ отправлен, повторите его после ответа.")
        else:
            # Rerun сразу: вопрос уже в сессии (run_turn пишет его первым),
            # поэтому лента дорисует его сама, а ниже встанет живой прогресс.
            st.rerun()

    _render_turn_progress()

    # iter73: пакеты ПРОЕКТА — самая верхняя панель утверждения: пока проекта
    # нет, всё остальное (патчи, пакеты спеки) применять некуда, и держать их
    # выше значило бы предлагать человеку кнопки без последствий.
    # iter71: пакеты спеки ВЫШЕ патчей — первичный ввод геометрии это первое,
    # что делается в проекте, и он не должен прятаться под панелью правок.
    # Панели УТВЕРЖДЕНИЯ (проект/спеки/патчи) остаются в зоне ассистента:
    # применить или отклонить предложение — часть работы с ним (iter72).
    _render_project_packages(ctx, session, runner)
    # iter76: правки полей формы — между пакетами проекта и спеки: они
    # относятся к тому же несобранному состоянию, что и пакет проекта.
    _render_setup_edits(ctx, session, runner)
    _render_spec_packages(ctx, session)
    _render_patches(ctx, session)


def render_assistant_info(runner: Any = None, *, root: str = "") -> None:
    """Правая колонка (iter72): постоянная ДОП-ИНФОРМАЦИЯ работы ассистента.

    По эскизу пользователя крайняя правая зона — то, что нужно ПОСТОЯННО на
    разных закладках рабочей области и большей частью связано с ассистентом:
    📎 вложения сессии, 🖼 файлы расчётов, 📚 журнал решений, 🏭 факты
    производства, 📌 состояние сессии. Раньше эти панели жили в левой колонке
    ПОД перепиской — до них приходилось скроллить сквозь весь диалог, а видны
    они были только когда диалог короткий.

    **iter80 — журналы проекта.** «📚 Журнал решений» и «🏭 Факты производства»
    добавлены здесь, а не в левой колонке, намеренно: слева живёт то, что ЖДЁТ
    утверждения (и после нажатия кнопки панель пустеет), а журнал — ИСТОРИЯ
    принятого и отклонённого, она читается из файла проекта и нужна на любом
    шаге. Отсутствие показа и читалось как «решения никуда не пишутся», хотя
    записывались они с iter63.

    Сессия и проект берутся тем же путём, что у дока (:func:`dock_session`):
    обе колонки показывают ОДНО состояние, а не две копии.
    """
    root = root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "project_campaigns")
    project = current_project(root) or "_scratch"
    session = dock_session(root, project)
    # iter80: журналы решений и фактов — те же инструменты, что у дока, поэтому
    # контекст собирается ОДНИМ путём: реестр подтверждений и спека должны быть
    # общими с левой колонкой, иначе токен кнопки относился бы к другому сеансу.
    ctx = dock_context(root, project, runner, session)

    st.subheader("📋 Инфо-панель проекта")
    _render_attachments(session, root, project)
    _render_artifacts(session)
    # Журнал — ПОСЛЕ файлов и ПЕРЕД состоянием переписки: это история проекта,
    # а не служебное состояние сеанса. Нужен он на любой закладке, поэтому
    # живёт в постоянной правой зоне, а не под лентой диалога (iter72).
    # iter96: ПРЕДЛОЖЕННЫЕ записи — прямо над журналами, в которые они лягут.
    # Панель стоит здесь, а не в левой зоне утверждений, намеренно: человек
    # правит формулировку, глядя на историю рядом («это уже записано»), а
    # предложение относится к журналам проекта, а не к геометрии.
    _render_note_proposals(ctx, session)
    _render_decisions(ctx, root, project)
    _render_local_facts(ctx, root, project)

    with st.expander("📌 Состояние переписки с помощником"):
        st.caption(views.session_caption(session))
        st.caption(views.context_caption(session))
        if st.button("🗑️ Очистить переписку (файлы и патчи останутся)",
                     key="dock_clear"):
            session.clear_messages()
            store.save_session(session, root, project)
            st.rerun()
