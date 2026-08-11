"""streamlit_app.py — единый UI кампании (REBUILD_SPEC_17 §17, C4/финал).

Кампания = ЕДИНЫЙ главный поток на :class:`MixtureProcessRunner` +
:class:`CampaignController`: сетап (компоненты смеси Σ=1 + процесс-параметры +
отклики) → ручной стартовый дизайн (seed) с ручным вводом Y → ручные
мультицелевые ветки (роли + ценовая нога ρ) → рабочий стол (предложить точки →
внести Y → долить) → эволюция схемы в любой момент. Старый mixture-only конвейер
M1…M8 на ``PipelineRunner`` (демо-синтетика, авто-M7) ВЫВЕДЕН из UI (§17.6). Сам
``PipelineRunner`` остаётся в ``src/`` как библиотека/для юнит-тестов ядра.

iter69/iter72 — РАСКЛАДКА ЭКРАНА: ТРИ зоны ``st.columns(workspace.MAIN_COLUMNS)``.
Слева — диалог с помощником (ассистент как инструмент взаимодействия с
программой), в центре — рабочая область на ЗАКЛАДКАХ (второй ряд закладок —
ветки проекта), справа — инфо-панель постоянной дополнительной информации
(вложения, выхлоп песочницы, состояние сессии). Содержимое закладки живёт
в контейнере фиксированной высоты со своим скроллом. Смысл: лента диалога и
рабочая область прокручиваются НЕЗАВИСИМО — ответ ассистента больше не уводит
страницу с таблицы, над которой человек работает. Логика раскладки —
:mod:`src.apps.workspace` (чистая, без Streamlit).

Salvage перенесён на C1–C3 и подключён здесь:
  * C1 — ИИ-ассистент campaign-native (``assistant.campaign_assistant_reply`` +
    ``build_campaign_context``), закладка «🤖 Обзор» рабочей области;
  * C2 — персистентность кампании (``campaign_state.save/load/list/delete``),
    сайдбар «📁 Кампания» (+ удаление под паролём администратора);
  * C3 — выгрузка общей базы/рецепта ветки в Excel живёт внутри ``render_campaign``.

Запуск:
    streamlit run src/apps/streamlit_app.py
    # или:  python run_streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# repo root в sys.path (Streamlit запускает файл напрямую)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.apps import admin  # noqa: E402
from src.apps import assistant as ai  # noqa: E402
from src.apps import campaign as cv  # noqa: E402
from src.apps import campaign_state as cs  # noqa: E402
from src.apps import workspace as wsx  # noqa: E402
from src.apps.assistant_dock import (render_assistant_dock,  # noqa: E402
                                      render_assistant_info)
from src.apps.campaign_ui import (render_campaign,  # noqa: E402
                                   campaign_assistant_overview,
                                   get_campaign_controller,
                                   setup_prefill_from_runner)


# Каталог сохранённых кампаний (в .gitignore — артефакт выполнения).
CAMPAIGN_ROOT = os.path.join(_REPO, "project_campaigns")


# ----------------------------------------------------------------------
# Сайдбар: персистентность проекта (C2, §17.6.1)
# ----------------------------------------------------------------------
def _seed_draft_from_session() -> Optional[Dict[str, Any]]:
    """Черновик стартового дизайна из session_state → JSON-native словарь.

    Пока seed НЕ зафиксирован (``commit_seed``), предложенный план и частично
    внесённые Y живут только в ``setup_seed_X`` / ``setup_seed_Y`` — без этого
    черновика сохранение до фиксации давало «пустой» проект (0 точек), а
    загрузка выглядела как потеря данных. NaN кодируется как null. Размер
    пробы (``setup_seed_batch``, замечание 7) — тоже часть черновика: без
    него после загрузки поле сбрасывалось в 0 и расход сырья терялся."""
    X = st.session_state.get("setup_seed_X")
    if X is None:
        return None
    X = np.atleast_2d(np.asarray(X, float))
    draft: Dict[str, Any] = {
        "seed_X": [[float(v) for v in row] for row in X]}
    Y = st.session_state.get("setup_seed_Y")
    if Y is not None:
        Y = np.atleast_2d(np.asarray(Y, float))
        draft["seed_Y"] = [
            [(float(v) if np.isfinite(v) else None) for v in row] for row in Y]
    batch = st.session_state.get("setup_seed_batch")
    try:
        if batch is not None and float(batch) > 0:
            draft["seed_batch"] = float(batch)
    except (TypeError, ValueError):
        pass
    return draft


def _restore_seed_draft(draft: Optional[Dict[str, Any]]) -> bool:
    """Восстановить черновик seed в session_state (обратное к сбору выше).

    Возвращает True, если черновик был и восстановлен. Ключ виджета-редактора
    Y сбрасывается, чтобы data_editor не наложил старые правки ячеек; сайдбар
    рендерится ДО редактора, поэтому чистить ключ здесь безопасно. Размер
    пробы (``seed_batch``) восстанавливается в ключ виджета — сайдбар идёт
    раньше number_input, менять session_state ещё можно."""
    for k in ("setup_seed_X", "setup_seed_Y", "setup_seed_editor",
              "setup_seed_batch", "setup_seed_df", "setup_seed_df_sig"):
        st.session_state.pop(k, None)
    if not draft or draft.get("seed_X") is None:
        return False
    st.session_state["setup_seed_X"] = np.asarray(draft["seed_X"], float)
    if draft.get("seed_Y") is not None:
        st.session_state["setup_seed_Y"] = np.asarray(
            [[(np.nan if v is None else float(v)) for v in row]
             for row in draft["seed_Y"]], float)
    if draft.get("seed_batch") is not None:
        st.session_state["setup_seed_batch"] = float(draft["seed_batch"])
    return True


def render_campaign_persistence(root: str) -> None:
    """📁 Сохранить/загрузить проект целиком (схема + база точек + ветки).

    Опирается на C2 (``campaign_state``): одна модель физики на проект,
    суррогаты НЕ сохраняются (переобучаются из измеренных точек при загрузке);
    незафиксированный стартовый дизайн сохраняется черновиком (``draft``).
    """
    st.sidebar.header("📁 Проект")
    st.session_state.setdefault("campaign_name", "my_project")
    # Отложенное имя (выставляется при загрузке проекта): применяем ДО
    # инстанцирования виджета text_input — Streamlit запрещает менять
    # session_state ключа виджета после его создания в том же прогоне.
    pending = st.session_state.pop("campaign_name_pending", None)
    if pending:
        st.session_state["campaign_name"] = pending
    ctrl = get_campaign_controller()

    name = st.sidebar.text_input("Имя проекта", key="campaign_name")
    if st.sidebar.button("💾 Сохранить проект", key="save_campaign"):
        if ctrl is None:
            st.sidebar.error("Проект ещё не собран — соберите его во "
                             "вкладке «🧬 Проект» или создайте демо-проект.")
        else:
            try:
                path = cs.save_campaign(ctrl.runner, root, name,
                                        draft=_seed_draft_from_session())
                st.sidebar.success(f"Проект сохранён: {Path(path).parent.name}")
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"Не удалось сохранить: {exc}")

    camps = cs.list_campaigns(root)
    sel = st.sidebar.selectbox("Открыть сохранённый проект",
                               ["— нет —"] + camps, key="campaign_select")
    if st.sidebar.button("📂 Загрузить проект", key="load_campaign") \
            and sel != "— нет —":
        try:
            runner = cs.load_campaign(root, sel)
            draft = cs.load_campaign_draft(root, sel)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Не удалось загрузить '{sel}': {exc}")
        else:
            # Успех — st.rerun() ВНЕ try: RerunException наследует Exception,
            # иначе управляющий сигнал перерисовки был бы проглочен except'ом
            # и показан как «Не удалось загрузить».
            st.session_state["campaign_ctrl"] = cv.CampaignController(runner)
            has_draft = _restore_seed_draft(draft)
            # C2: форма сетапа должна показать НАСТРОЙКИ загруженного проекта
            # (компоненты, доли-границы, процесс-границы, отклики, seed), а не
            # дефолты «A, B, C». Применяется отложенно в render_setup_form —
            # ДО инстанцирования виджетов формы следующего прогона.
            st.session_state["setup_prefill_pending"] = \
                setup_prefill_from_runner(runner)
            st.session_state["campaign_name_pending"] = sel
            st.session_state["camp_loaded_msg"] = (
                f"Проект '{sel}' загружен (общая база: "
                f"{len(runner.points)} точек, веток: {len(runner.branches)}"
                + ("; восстановлен черновик стартового дизайна"
                   if has_draft else "") + ").")
            st.rerun()

    if st.session_state.get("camp_loaded_msg"):
        st.sidebar.success(st.session_state.pop("camp_loaded_msg"))


def render_campaign_deleter(root: str) -> None:
    """🗑 Danger zone: удаление сохранённого проекта под паролём администратора.

    Барьер от случайного удаления (не криптозащита): нужны выбор проекта,
    подтверждение имени и admin-пароль (env ``DOE_ADMIN_PASSWORD``).
    """
    with st.sidebar.expander("🗑 Удалить проект (admin)", expanded=False):
        if st.session_state.get("camp_del_msg"):
            st.success(st.session_state.pop("camp_del_msg"))
        camps = cs.list_campaigns(root)
        if not camps:
            st.caption("Сохранённых проектов нет.")
            return
        st.caption("Удаление безвозвратно. Это барьер от случайного удаления "
                   "(не криптозащита): нужен пароль администратора "
                   "(переменная окружения `DOE_ADMIN_PASSWORD`).")
        target = st.selectbox("Проект для удаления", camps, key="camp_del_select")
        confirm = st.text_input("Подтвердите: впишите имя проекта точно",
                                key="camp_del_confirm")
        pwd = st.text_input("Пароль администратора", type="password",
                            key="camp_del_pwd")
        if st.button("🗑 Удалить навсегда", key="camp_del_button"):
            if confirm != target:
                st.error("Имя для подтверждения не совпадает с выбранным "
                         "проектом — удаление отменено.")
            elif not admin.check_admin_password(pwd):
                st.error("Неверный пароль администратора — удаление отменено.")
            else:
                try:
                    ok = cs.delete_campaign(root, target)
                except ValueError as exc:
                    st.error(f"Удаление отклонено: {exc}")
                else:
                    if not ok:
                        st.error(f"Проект '{target}' не найден.")
                    else:
                        # если удалён загруженный сейчас проект — сбросить сессию
                        if st.session_state.get("campaign_name") == target:
                            st.session_state.pop("campaign_ctrl", None)
                        st.session_state["camp_del_msg"] = \
                            f"Проект '{target}' удалён."
                        st.rerun()


# ----------------------------------------------------------------------
# Вкладка «Ассистент» — campaign-native (C1, §17.6.1)
# ----------------------------------------------------------------------
def render_campaign_assistant() -> None:
    """💬 Встроенный ИИ-ассистент кампании.

    Контекст строится ПРЯМО из сводки кампании (``campaign_assistant_overview``
    → ``build_campaign_context``), системный промпт — ``campaign_system_prompt``.
    Стадий M1…M8 и ``PipelineRunner`` больше нет (§17.6).
    """
    st.subheader("💬 Ассистент проекта: интерпретация состояния и подсказки")
    overview = campaign_assistant_overview()

    # --- настройки подключения (ключ/модель вводятся прямо здесь) -------
    with st.expander("⚙️ Подключение: API-ключ и модель",
                     expanded=not ai.llm_available()):
        st.caption("Ключ OpenRouter (sk-or-…) и имя модели. Кнопка «💾 Сохранить» "
                   "пишет ОБА значения в локальный файл `.env` в корне репозитория "
                   "(он в `.gitignore` — на GitHub не уйдёт, подхватится при "
                   "следующих запусках). Без сохранения они живут только в "
                   "текущей сессии.")
        key_in = st.text_input("OpenRouter API key", type="password",
                               value=os.environ.get("OPENROUTER_API_KEY", ""),
                               key="ai_key")
        model_in = st.text_input("Модель", value=ai.model_name(), key="ai_model")
        if key_in:
            os.environ["OPENROUTER_API_KEY"] = key_in.strip()
        if model_in:
            os.environ["DOE_ASSISTANT_MODEL"] = model_in.strip()
        ckey = st.columns([2, 3])
        if ckey[0].button("💾 Сохранить ключ и модель в .env", key="ai_save_key"):
            try:
                path = ai.save_api_key(key_in, model=model_in)
                st.success(f"Ключ и модель (`{model_in}`) сохранены в `{path}` "
                           "(файл в .gitignore — на GitHub не уйдёт).")
            except ValueError as exc:
                st.error(str(exc))
            except OSError as exc:  # noqa: BLE001
                st.error(f"Не удалось записать .env: {exc}")
        if ai.api_key_persisted():
            ckey[1].caption(f"🔐 Сохранённые ключ и модель найдены: "
                            f"`{ai.env_file_path()}`")
        else:
            ckey[1].caption("Ключ и модель пока не сохранены в .env.")

    # --- статус backend ------------------------------------------------
    cstat = st.columns([2, 3])
    if ai.llm_available():
        cstat[0].success("LLM подключён")
    else:
        cstat[0].warning("Нет OPENROUTER_API_KEY")
    cstat[1].caption(f"Модель: `{ai.model_name()}`")

    with st.expander("👁️ Что сейчас «видит» ассистент (контекст проекта)"):
        st.json(ai.build_campaign_context(overview or {}))

    if overview is None:
        st.info("Проект ещё не собран — соберите его во вкладке "
                "«🧬 Проект» (или создайте демо-проект), чтобы ассистент "
                "видел реальное состояние: ветки, роли, денежный канал ρ.")

    if not ai.llm_available():
        st.caption("Чтобы включить чат, задайте `OPENROUTER_API_KEY` (поле выше "
                   "или переменная окружения). Блок «Что видит ассистент» "
                   "работает и без ключа.")

    # --- история диалога ----------------------------------------------
    history = st.session_state.setdefault("ai_history", [])
    cc = st.columns([1, 5])
    if cc[0].button("🗑️ Очистить", key="ai_clear"):
        st.session_state["ai_history"] = []
        st.rerun()

    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Спросите про проект, ветки, роли или следующий шаг…",
                           key="ai_input", disabled=not ai.llm_available())
    if prompt:
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Думаю над контекстом проекта…"):
                try:
                    reply = ai.campaign_assistant_reply(overview or {},
                                                        history[:-1], prompt)
                except Exception as exc:  # noqa: BLE001
                    reply = f"⚠️ Ошибка обращения к модели: {exc}"
                st.markdown(reply)
        history.append({"role": "assistant", "content": reply})
        st.session_state["ai_history"] = history


# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="DOE — Проект (mixture×process)",
                       layout="wide")
    # Подхватываем сохранённый ключ/модель из локального .env (если есть).
    # Внешние переменные окружения имеют приоритет (override=False).
    ai.load_env_file()
    # iter69: заголовок компактный (без st.title) — высота панелей ниже
    # фиксирована, и крупная шапка отбирала у них место без пользы.
    st.markdown("#### 🧬 DOE — Проект (смесь × процесс)")
    st.caption("Единый поток (§17): сетап → стартовый дизайн (seed) → ветки → "
               "рабочий стол (ручной ввод откликов) → эволюция схемы. Слева — "
               "диалог с помощником (лента прокручивается вверх, ввод внизу), "
               "в центре — рабочая область на закладках (от переписки НЕ "
               "двигается), справа — постоянная инфо-панель: вложения, выхлоп "
               "песочницы, состояние сессии.")

    render_campaign_persistence(CAMPAIGN_ROOT)   # 📁 сохранить/загрузить проект
    render_campaign_deleter(CAMPAIGN_ROOT)       # 🗑 удалить проект (под паролём)

    # iter72: раскладка по эскизу — ТРИ зоны: ДИАЛОГ слева (ассистент —
    # инструмент взаимодействия с программой), РАБОЧАЯ ОБЛАСТЬ в центре
    # (закладки), ИНФО-ПАНЕЛЬ справа (вложения, выхлоп песочницы, состояние
    # сессии — нужны постоянно на разных закладках).
    #
    # Порядок рендера обратный порядку колонок: рабочая область рисуется ПЕРВОЙ,
    # потому что именно она публикует `ui_focus` (активная закладка, ветка), из
    # которого диалог берёт контекст «по месту» (iter65). Streamlit это
    # позволяет: содержимое колонки пишется в её контейнер, а не по порядку
    # вызовов на странице.
    dock_col, work_col, info_col = st.columns(list(wsx.MAIN_COLUMNS))
    with work_col:
        # Обзор ассистента — закладка рабочей области, а не отдельная вкладка
        # верхнего уровня: иначе «💬 Ассистент (обзор)» уводил с проекта целиком.
        render_campaign(overview_renderer=render_campaign_assistant)
    ctrl_now = get_campaign_controller()
    runner_now = getattr(ctrl_now, "runner", None) if ctrl_now else None
    with dock_col:
        render_assistant_dock(runner_now, root=CAMPAIGN_ROOT)
    with info_col:
        render_assistant_info(runner_now, root=CAMPAIGN_ROOT)


if __name__ == "__main__":
    main()
