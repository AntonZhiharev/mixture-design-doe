"""streamlit_app.py — единый UI кампании (REBUILD_SPEC_17 §17, C4/финал).

Кампания = ЕДИНЫЙ главный поток на :class:`MixtureProcessRunner` +
:class:`CampaignController`: сетап (компоненты смеси Σ=1 + процесс-параметры +
отклики) → ручной стартовый дизайн (seed) с ручным вводом Y → ручные
мультицелевые ветки (роли + ценовая нога ρ) → рабочий стол (предложить точки →
внести Y → долить) → эволюция схемы в любой момент. Старый mixture-only конвейер
M1…M8 на ``PipelineRunner`` (демо-синтетика, авто-M7) ВЫВЕДЕН из UI (§17.6). Сам
``PipelineRunner`` остаётся в ``src/`` как библиотека/для юнит-тестов ядра.

Salvage перенесён на C1–C3 и подключён здесь:
  * C1 — ИИ-ассистент campaign-native (``assistant.campaign_assistant_reply`` +
    ``build_campaign_context``), вкладка «💬 Ассистент»;
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

# repo root в sys.path (Streamlit запускает файл напрямую)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import streamlit as st  # noqa: E402

from src.apps import admin  # noqa: E402
from src.apps import assistant as ai  # noqa: E402
from src.apps import campaign as cv  # noqa: E402
from src.apps import campaign_state as cs  # noqa: E402
from src.apps.campaign_ui import (render_campaign,  # noqa: E402
                                   campaign_assistant_overview,
                                   get_campaign_controller)


# Каталог сохранённых кампаний (в .gitignore — артефакт выполнения).
CAMPAIGN_ROOT = os.path.join(_REPO, "project_campaigns")


# ----------------------------------------------------------------------
# Сайдбар: персистентность кампании (C2, §17.6.1)
# ----------------------------------------------------------------------
def render_campaign_persistence(root: str) -> None:
    """📁 Сохранить/загрузить кампанию целиком (схема + база точек + ветки).

    Опирается на C2 (``campaign_state``): одна модель физики на проект,
    суррогаты НЕ сохраняются (переобучаются из измеренных точек при загрузке).
    """
    st.sidebar.header("📁 Кампания")
    st.session_state.setdefault("campaign_name", "my_campaign")
    ctrl = get_campaign_controller()

    name = st.sidebar.text_input("Имя кампании", key="campaign_name")
    if st.sidebar.button("💾 Сохранить кампанию", key="save_campaign"):
        if ctrl is None:
            st.sidebar.error("Кампания ещё не собрана — соберите проект во "
                             "вкладке «🧬 Кампания» или создайте демо-кампанию.")
        else:
            try:
                path = cs.save_campaign(ctrl.runner, root, name)
                st.sidebar.success(f"Кампания сохранена: {Path(path).parent.name}")
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"Не удалось сохранить: {exc}")

    camps = cs.list_campaigns(root)
    sel = st.sidebar.selectbox("Открыть сохранённую кампанию",
                               ["— нет —"] + camps, key="campaign_select")
    if st.sidebar.button("📂 Загрузить кампанию", key="load_campaign") \
            and sel != "— нет —":
        try:
            runner = cs.load_campaign(root, sel)
            st.session_state["campaign_ctrl"] = cv.CampaignController(runner)
            st.session_state["campaign_name"] = sel
            st.session_state["camp_loaded_msg"] = (
                f"Кампания '{sel}' загружена (общая база: "
                f"{len(runner.points)} точек, веток: {len(runner.branches)}).")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Не удалось загрузить '{sel}': {exc}")

    if st.session_state.get("camp_loaded_msg"):
        st.sidebar.success(st.session_state.pop("camp_loaded_msg"))


def render_campaign_deleter(root: str) -> None:
    """🗑 Danger zone: удаление сохранённой кампании под паролём администратора.

    Барьер от случайного удаления (не криптозащита): нужны выбор кампании,
    подтверждение имени и admin-пароль (env ``DOE_ADMIN_PASSWORD``).
    """
    with st.sidebar.expander("🗑 Удалить кампанию (admin)", expanded=False):
        if st.session_state.get("camp_del_msg"):
            st.success(st.session_state.pop("camp_del_msg"))
        camps = cs.list_campaigns(root)
        if not camps:
            st.caption("Сохранённых кампаний нет.")
            return
        st.caption("Удаление безвозвратно. Это барьер от случайного удаления "
                   "(не криптозащита): нужен пароль администратора "
                   "(переменная окружения `DOE_ADMIN_PASSWORD`).")
        target = st.selectbox("Кампания для удаления", camps, key="camp_del_select")
        confirm = st.text_input("Подтвердите: впишите имя кампании точно",
                                key="camp_del_confirm")
        pwd = st.text_input("Пароль администратора", type="password",
                            key="camp_del_pwd")
        if st.button("🗑 Удалить навсегда", key="camp_del_button"):
            if confirm != target:
                st.error("Имя для подтверждения не совпадает с выбранной "
                         "кампанией — удаление отменено.")
            elif not admin.check_admin_password(pwd):
                st.error("Неверный пароль администратора — удаление отменено.")
            else:
                try:
                    ok = cs.delete_campaign(root, target)
                except ValueError as exc:
                    st.error(f"Удаление отклонено: {exc}")
                else:
                    if not ok:
                        st.error(f"Кампания '{target}' не найдена.")
                    else:
                        # если удалена загруженная сейчас кампания — сбросить сессию
                        if st.session_state.get("campaign_name") == target:
                            st.session_state.pop("campaign_ctrl", None)
                        st.session_state["camp_del_msg"] = \
                            f"Кампания '{target}' удалена."
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
    st.subheader("💬 Ассистент кампании: интерпретация состояния и подсказки")
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

    with st.expander("👁️ Что сейчас «видит» ассистент (контекст кампании)"):
        st.json(ai.build_campaign_context(overview or {}))

    if overview is None:
        st.info("Кампания ещё не собрана — соберите проект во вкладке "
                "«🧬 Кампания» (или создайте демо-кампанию), чтобы ассистент "
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

    prompt = st.chat_input("Спросите про кампанию, ветки, роли или следующий шаг…",
                           key="ai_input", disabled=not ai.llm_available())
    if prompt:
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Думаю над контекстом кампании…"):
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
    st.set_page_config(page_title="DOE — Кампания (mixture×process)",
                       layout="wide")
    # Подхватываем сохранённый ключ/модель из локального .env (если есть).
    # Внешние переменные окружения имеют приоритет (override=False).
    ai.load_env_file()
    st.title("🧬 DOE — Кампания (смесь × процесс)")
    st.caption("Единый поток (§17): сетап → стартовый дизайн (seed) → ветки → "
               "рабочий стол (ручной ввод откликов) → эволюция схемы. ОДНА модель "
               "физики на проект; отклики вносит пользователь (реальная "
               "лаборатория), демо-оракул — для прогонов без лаборатории.")

    render_campaign_persistence(CAMPAIGN_ROOT)   # 📁 сохранить/загрузить кампанию
    render_campaign_deleter(CAMPAIGN_ROOT)       # 🗑 удалить кампанию (под паролём)

    tab_camp, tab_ai = st.tabs(["🧬 Кампания", "💬 Ассистент"])
    with tab_camp:
        render_campaign()
    with tab_ai:
        render_campaign_assistant()


if __name__ == "__main__":
    main()
