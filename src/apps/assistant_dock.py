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
  Streamlit ≥1.52), поэтому новых зависимостей нет. Ctrl+V из буфера
  Streamlit сам пока НЕ поддерживает — работает выбор файла и drag&drop;
  для настоящей вставки понадобился бы сторонний компонент с JS-бандлом,
  и это решение сознательно отложено. Графики и таблицы, посчитанные в
  песочнице, рисуются ``st.image``/``st.dataframe`` прямо в ответе;
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
#: Вопрос НЕУДАВШЕГОСЯ хода (iter67): держим отдельно от K_PENDING, потому что
#: K_PENDING отправляется автоматически на следующем прогоне, а этот ждёт
#: явного нажатия кнопки — повтор к модели должен быть решением человека.
K_FAILED = "assistant_failed_question"


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
                 use_container_width=True,
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
                   "ассистент собирает его целиком: состав (phr-спека) + "
                   "отклики + процесс-оси с границами. Одной спекой проект не "
                   "заводится — откликов и осей в ней нет по схеме.")
        return
    st.dataframe(views.staged_projects_dataframe(session, only_staged=True),
                 use_container_width=True, hide_index=True)
    if runner is not None:
        st.warning("В сессии УЖЕ собран проект: принять пакет проекта нельзя — "
                   "иначе молча пропали бы измеренные точки и ветки. Для правки "
                   "геометрии нужен пакет спеки, отклики и оси меняются в "
                   "сетапе при осознанной пересборке.")
    for p in staged:
        m = p.summary or {}
        st.caption(f"`{p.id}` · {p.label or 'без метки'} · "
                   f"{len(list(m.get('components', []) or []))} компонентов · "
                   f"{len(list(m.get('responses', []) or []))} откликов · "
                   f"{len(list(m.get('process', []) or []))} процесс-осей · "
                   f"hash {str(m.get('spec_hash', ''))[:12]}…")
        # Блоки таблицей — «что именно загружается» без чтения JSON глазами.
        st.dataframe(views.project_blocks_dataframe(m),
                     use_container_width=True, hide_index=True)
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
        st.caption("Пусто. Пакет появляется здесь, когда ассистент собирает "
                   "спеку целиком — первичный ввод геометрии или эволюция "
                   "(новый узел, удаление, смена роли). Сам он её не применяет.")
        return
    st.dataframe(views.staged_specs_dataframe(session, only_staged=True),
                 use_container_width=True, hide_index=True)
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
            st.image(o.path, use_container_width=True)
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
                st.dataframe(df, use_container_width=True, hide_index=True)
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
    typed = st.chat_input(
        "Спросите про эту ось, границу или следующий шаг… "
        "(📎 скриншот, 🎤 голос)",
        key="dock_input", accept_file="multiple",
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
            st.info(f"Документ «{att.name}» приложен к сессии — ассистент "
                    f"прочитает его инструментом чтения.")

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
                         use_container_width=True, hide_index=True)


def _render_artifacts(session) -> None:
    """Графики и таблицы прогонов ЭТОГО проекта (живут после перезапуска)."""
    shown = views.artifact_outputs(session)
    with st.expander(f"🖼 Выхлоп песочницы: {len(shown)}"):
        if not shown:
            st.caption("Пусто. Здесь появляются графики и таблицы, которые "
                       "ассистент построил в песочнице (`run_python` с "
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
    project = current_project() or "_scratch"
    session = dock_session(root, project)
    ctx = dock_context(root, project, runner, session)
    focus = dock_focus()

    st.subheader("💬 Архитектор кампании")
    if runner is None:
        st.info("Проект не собран: движка и базы точек нет — вопросы про "
                "preflight и прогоны честно останутся без чисел.")

    # iter69: служебные панели (фокус, интернет, ключ, подсказки, вложения,
    # патчи) — НАД лентой и свёрнуты. Раньше они стояли между лентой и полем
    # ввода, и переписка «плавала»: ответ ассистента отодвигал ввод вниз,
    # а до истории приходилось прокручивать всю страницу.
    with st.expander("⚙️ Контекст и настройки хода", expanded=runner is None):
        st.caption("Отвечает ЧИСЛАМИ ИЗ ЯДРА: роли, эффективные границы, "
                   "spec_hash и preflight считают инструменты, а не память "
                   "модели.")
        focus = _render_focus(focus, ctx)
        session.web_enabled = st.toggle(
            "🌐 Интернет (:online)", value=bool(session.web_enabled),
            key="dock_web",
            help="Всё, что придёт из сети, — уровень знания L2: локальный факт "
                 "цеха его отменяет.")
        _render_connection()
        st.markdown("**Спросить по месту:**")
        asked = _render_suggestions(focus, runner is not None)

    # --- ЛЕНТА: свой скролл, история вверх, свежее внизу (как в Cline) ---
    typed, images = None, []
    feed = wsx.feed_items(session.messages)
    box = st.container(height=wsx.CHAT_FEED_HEIGHT, border=True,
                       autoscroll=True) if _supports_feed_container() \
        else st.container()
    with box:
        for item in feed:
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
        # Ход ЭТОГО прогона дорисовывается в ту же ленту ниже (см. run_turn).
        turn_slot = st.container()
    st.caption(wsx.feed_hint(len(feed), wsx.dialog_count(session.messages)))

    # Поле ввода — ПОД лентой и не двигается: лента прокручивается внутри себя.
    typed, images = _chat_submission(session, root, project)
    question = typed or asked or st.session_state.pop(K_PENDING, None)

    if question or images:
        with turn_slot:
            with st.chat_message("user"):
                st.markdown(question or "_(изображение без текста)_")
                for sha in images:
                    _show_attachment_image(session, root, project, sha)
            progress = st.empty()
            with st.chat_message("assistant"):
                with st.spinner("Считаю инструментами ядра…"):
                    res = actx.run_turn(
                        session, ctx, question or "", focus=focus,
                        spec_hash=spec_hash_of(ctx), kinds=AGENT_KINDS,
                        images=images,
                        on_event=lambda e: progress.caption(
                            llm.progress_caption(e)))
                progress.empty()
                _render_answer(res.text)
                # Графики/таблицы, посчитанные в ходе, — сразу в ответе: файл,
                # который надо искать в другой панели, разговору не помогает.
                _render_outputs(views.turn_outputs(session, res.new_artifacts),
                                scope="turn")
        for err in res.image_errors:
            st.warning(f"Изображение не дошло до модели — {err}")
        st.caption(views.turn_caption(res))
        st.session_state[K_LAST_TURN] = res
        if res.calls:
            with st.expander("🔧 Что было посчитано"):
                st.dataframe(views.tool_calls_dataframe(res.calls),
                             use_container_width=True, hide_index=True)
        _render_retry(res, fresh=True)
    else:
        # Отказ прошлого хода не должен исчезать при любом движении виджета:
        # кнопка повтора живёт до успешного ответа или до нового вопроса.
        _render_retry(st.session_state.get(K_LAST_TURN))

    # iter73: пакеты ПРОЕКТА — самая верхняя панель утверждения: пока проекта
    # нет, всё остальное (патчи, пакеты спеки) применять некуда, и держать их
    # выше значило бы предлагать человеку кнопки без последствий.
    # iter71: пакеты спеки ВЫШЕ патчей — первичный ввод геометрии это первое,
    # что делается в проекте, и он не должен прятаться под панелью правок.
    # Панели УТВЕРЖДЕНИЯ (проект/спеки/патчи) остаются в зоне ассистента:
    # применить или отклонить предложение — часть работы с ним (iter72).
    _render_project_packages(ctx, session, runner)
    _render_spec_packages(ctx, session)
    _render_patches(ctx, session)


def render_assistant_info(runner: Any = None, *, root: str = "") -> None:
    """Правая колонка (iter72): постоянная ДОП-ИНФОРМАЦИЯ работы ассистента.

    По эскизу пользователя крайняя правая зона — то, что нужно ПОСТОЯННО на
    разных закладках рабочей области и большей частью связано с ассистентом:
    📎 вложения сессии, 🖼 выхлоп песочницы, 📌 состояние сессии. Раньше эти
    панели жили в левой колонке ПОД перепиской — до них приходилось скроллить
    сквозь весь диалог, а видны они были только когда диалог короткий.

    Сессия и проект берутся тем же путём, что у дока (:func:`dock_session`):
    обе колонки показывают ОДНО состояние, а не две копии.
    """
    root = root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "project_campaigns")
    project = current_project() or "_scratch"
    session = dock_session(root, project)

    st.subheader("📋 Инфо")
    _render_attachments(session, root, project)
    _render_artifacts(session)

    with st.expander("📌 Состояние сессии"):
        st.caption(views.session_caption(session))
        st.caption(views.context_caption(session))
        if st.button("🗑️ Очистить переписку (файлы и патчи останутся)",
                     key="dock_clear"):
            session.clear_messages()
            store.save_session(session, root, project)
            st.rerun()
