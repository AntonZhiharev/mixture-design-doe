"""apps/assistant_dock.py — ДОК ассистента-архитектора в правой колонке (iter65).

Отдельная вкладка «💬 Ассистент» была устроена так, что спросить «почему у
этой оси такой диапазон» можно было только уйдя с экрана, где эта ось видна.
Док решает ровно это: он стоит справа (`st.columns([3, 1])`) и виден на КАЖДОМ
шаге потока, а контекст берёт из ``st.session_state['ui_focus']``, который
публикует активная секция.

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
  должен думать, что приложение зависло.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import streamlit as st

from src.assistant import config as ai_config
from src.assistant import context as actx
from src.assistant import files as afiles
from src.assistant import llm, store, views
from src.assistant.consent import ConsentRegistry
from src.assistant.context import UiFocus
from src.assistant.tools import AGENT_KINDS, ToolContext, ToolError

#: Ключи session_state дока (все с префиксом — чтобы не столкнуться с потоком).
K_SESSION = "assistant_session"
K_PROJECT = "assistant_session_project"
K_CONSENT = "assistant_consent_registry"
K_LAST_TURN = "assistant_last_turn"
K_PENDING = "assistant_pending_question"


# ----------------------------------------------------------------------
# Состояние дока
# ----------------------------------------------------------------------
def current_project() -> str:
    """Имя текущего проекта (то же поле, что у сохранения кампании)."""
    return str(st.session_state.get("campaign_name", "") or "").strip()


def dock_focus() -> UiFocus:
    """Фокус интерфейса из ``st.session_state`` (чистая логика — в context).

    Секции потока рисуются в ЛЕВОЙ колонке раньше дока, поэтому здесь виден
    фокус текущего прогона, а не прошлого.
    """
    return actx.focus_from_state(st.session_state)


def dock_session(root: str, project: str):
    """Сессия ассистента текущего проекта (перечитывается при смене проекта)."""
    if st.session_state.get(K_PROJECT) != project or K_SESSION not in st.session_state:
        st.session_state[K_SESSION] = store.load_session(root, project) \
            if project else store.load_session(root, "_scratch")
        st.session_state[K_PROJECT] = project
    return st.session_state[K_SESSION]


def consent_registry() -> ConsentRegistry:
    reg = st.session_state.get(K_CONSENT)
    if not isinstance(reg, ConsentRegistry):
        reg = ConsentRegistry()
        st.session_state[K_CONSENT] = reg
    return reg


def dock_context(root: str, project: str, runner: Any, session: Any
                 ) -> ToolContext:
    """Контекст инструментов дока: движок + спека + сессия + подтверждения."""
    spec = getattr(runner, "phr_spec", None) if runner is not None else None
    return ToolContext(runner=runner, session=session, root=root,
                       project=project, spec=spec,
                       extra={"consent": consent_registry()})


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
                     disabled=not s.enabled, use_container_width=True,
                     help=(s.question if s.enabled else s.why)):
            asked = s.question
        if not s.enabled and s.why:
            st.caption(f"⛔ {s.why}")
    return asked


def _render_patches(ctx: ToolContext, session) -> None:
    """Панель предложений: применить/отклонить может только ЧЕЛОВЕК (iter63)."""
    staged = session.staged_patches()
    st.markdown(f"**🧩 Предложенные патчи спеки: {len(staged)}**")
    if not staged:
        st.caption("Пусто. Патч появляется здесь, когда ассистент предлагает "
                   "правку геометрии — сам он её не применяет.")
        return
    st.dataframe(views.staged_patches_dataframe(session, only_staged=True),
                 use_container_width=True, hide_index=True)
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
                    st.success("Патч отклонён, решение записано в журнал.")


def _render_attachments(session, root: str, project: str) -> None:
    with st.expander(f"📎 Вложения: {len(session.attachments)}"):
        up = st.file_uploader("Паспорт, ТДС, выгрузка (txt/md/csv/json/xlsx/"
                              "docx/pdf)", key="dock_upload")
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
                         use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------
# Док целиком
# ----------------------------------------------------------------------
def render_assistant_dock(runner: Any = None, *, root: str = "") -> None:
    """Правая колонка: чат архитектора с контекстом ТЕКУЩЕГО шага потока."""
    root = root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "project_campaigns")
    project = current_project() or "_scratch"
    session = dock_session(root, project)
    ctx = dock_context(root, project, runner, session)
    focus = dock_focus()

    st.subheader("💬 Архитектор кампании")
    st.caption("Отвечает ЧИСЛАМИ ИЗ ЯДРА: роли, эффективные границы, "
               "spec_hash и preflight считают инструменты, а не память модели.")
    if runner is None:
        st.info("Проект не собран: движка и базы точек нет — вопросы про "
                "preflight и прогоны честно останутся без чисел.")

    focus = _render_focus(focus, ctx)
    session.web_enabled = st.toggle(
        "🌐 Интернет (:online)", value=bool(session.web_enabled),
        key="dock_web",
        help="Всё, что придёт из сети, — уровень знания L2: локальный факт "
             "цеха его отменяет.")
    _render_connection()

    st.markdown("**Спросить по месту:**")
    asked = _render_suggestions(focus, runner is not None)

    for m in session.messages[-20:]:
        if m.role in ("user", "assistant"):
            with st.chat_message(m.role):
                st.markdown(m.content)

    typed = st.chat_input("Спросите про эту ось, границу или следующий шаг…",
                          key="dock_input")
    question = typed or asked or st.session_state.pop(K_PENDING, None)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        box = st.empty()
        with st.chat_message("assistant"):
            with st.spinner("Считаю инструментами ядра…"):
                res = actx.run_turn(
                    session, ctx, question, focus=focus,
                    spec_hash=spec_hash_of(ctx), kinds=AGENT_KINDS,
                    on_event=lambda e: box.caption(llm.progress_caption(e)))
            box.empty()
            st.markdown(res.text or "—")
        st.caption(views.turn_caption(res))
        st.session_state[K_LAST_TURN] = res
        if res.calls:
            with st.expander("🔧 Что было посчитано"):
                st.dataframe(views.tool_calls_dataframe(res.calls),
                             use_container_width=True, hide_index=True)

    _render_patches(ctx, session)
    _render_attachments(session, root, project)

    with st.expander("📌 Состояние сессии"):
        st.caption(views.session_caption(session))
        st.caption(views.context_caption(session))
        if st.button("🗑️ Очистить переписку (файлы и патчи останутся)",
                     key="dock_clear"):
            session.clear_messages()
            store.save_session(session, root, project)
            st.rerun()
