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
    панель «📁 Проект» на закладке «🌱 Старт» (iter72: сайдбар упразднён;
    + удаление под паролём администратора);
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
from src.apps import assistant_dock as dock  # noqa: E402
from src.apps.assistant_dock import (render_assistant_dock,  # noqa: E402
                                      render_assistant_info)
from src.apps.campaign_ui import (render_campaign,  # noqa: E402
                                   campaign_assistant_overview,
                                   get_campaign_controller,
                                   setup_prefill_from_runner)


# Каталог сохранённых кампаний (в .gitignore — артефакт выполнения).
CAMPAIGN_ROOT = os.path.join(_REPO, "project_campaigns")


# ----------------------------------------------------------------------
# Панель «📁 Проект»: персистентность (C2, §17.6.1; iter72 — закладка «Старт»)
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
    Y сбрасывается, чтобы data_editor не наложил старые правки ячеек; панель
    «📁 Проект» рендерится ДО редактора (обе на закладке «Старт», панель
    первой), а успех загрузки завершается ``st.rerun`` — чистить ключи здесь
    безопасно. Размер пробы (``seed_batch``) восстанавливается в ключ виджета
    аналогично: панель идёт раньше number_input, менять session_state ещё
    можно."""
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


def _load_setup_draft_into_form(root: str, sel: str) -> None:
    """iter76: загрузка проекта-ЧЕРНОВИКА — поля формы вместо раннера.

    Собранного ``campaign.json`` у такого проекта нет; черновик настроек
    (``setup_draft.json``) возвращается в форму «🆕 Новый проект» тем же
    отложенным механизмом ``setup_prefill_pending``, что и префилл из
    загруженного раннера. Успех завершается ``st.rerun()`` (вне ``try`` —
    RerunException наследует Exception и был бы проглочен).
    """
    try:
        setup_draft = cs.load_setup_draft(root, sel)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось загрузить черновик '{sel}': {exc}")
        return
    if not setup_draft:
        st.error(f"'{sel}': нет ни собранного проекта, ни черновика "
                 f"настроек — загружать нечего.")
        return
    st.session_state.pop("campaign_ctrl", None)
    st.session_state["setup_prefill_pending"] = setup_draft
    # iter77: переключаем и ССЫЛКУ (переписка/вложения пойдут за проектом),
    # а в поле имени кладём его ИМЯ, а не имя каталога.
    _ident = cs.pref.read_identity(root, sel)
    if _ident is not None:
        st.session_state["campaign_ref_pending"] = _ident.ref
    st.session_state["campaign_name_pending"] = (
        _ident.label if _ident is not None else sel)
    st.session_state["camp_loaded_msg"] = (
        f"Черновик настроек '{sel}' загружен ({len(setup_draft)} полей "
        f"формы). Проект ещё НЕ собран: проверьте форму «🆕 Новый проект» "
        f"и нажмите «🏗 Построить проект».")
    st.rerun()


def render_campaign_persistence(root: str) -> None:
    """📁 Сохранить/загрузить проект целиком (схема + база точек + ветки).

    Опирается на C2 (``campaign_state``): одна модель физики на проект,
    суррогаты НЕ сохраняются (переобучаются из измеренных точек при загрузке);
    незафиксированный стартовый дизайн сохраняется черновиком (``draft``).

    iter72: панель живёт на закладке «🌱 Старт» рабочей области (обычные
    виджеты, не ``st.sidebar``) — сайдбар упразднён, левая колонка целиком
    отдана ассистенту (эскиз пользователя).
    """
    st.markdown("#### 📁 Проект")
    # Имя проекта живёт как состояние ПРИЛОЖЕНИЯ (пин в main()): виджет ниже
    # рисуется только на закладке «Старт», и без пина Streamlit удалял бы ключ
    # на других закладках.
    st.session_state.setdefault("campaign_name", cs.pref.DEFAULT_LABEL)
    ctrl = get_campaign_controller()

    # iter77: ССЫЛКА проекта — то, чем он опознаётся. Показываем её рядом с
    # именем: без этого «переименование не влияет на переписку» остаётся
    # словами, а человек не видит, с каким проектом работает.
    ref = dock.current_project_ref()
    ident = cs.pref.find_by_ref(root, ref) if ref else None
    if ident is not None:
        st.caption(f"Ссылка проекта: `{ident.ref}` · папка "
                   f"`{ident.dirname}` · переписка и вложения привязаны к "
                   f"ССЫЛКЕ, поэтому переименование их не переключает."
                   + (f" Прежние имена: {', '.join(ident.label_history)}."
                      if ident.label_history else ""))

    # iter73/iter77: подпись говорит, ЧТО это за имя. Раньше имя было ещё и
    # адресом на диске, поэтому его правка молча уводила переписку в другой
    # каталог; теперь это ПОДПИСЬ проекта, а кнопка ниже применяет её.
    name = st.text_input(
        "Имя проекта (подпись; переименование не меняет папку и переписку)",
        key="campaign_name",
        help="Имя — только для человека: проект опознаётся ссылкой (см. "
             "строку выше). Собран проект или нет — сказано строкой в начале "
             "закладки.")
    if ident is not None and st.button("✏️ Переименовать проект",
                                       key="rename_campaign"):
        try:
            renamed = cs.pref.rename_label(root, ident.ref, name)
        except ValueError as exc:
            st.error(f"Переименование отклонено: {exc}")
        else:
            st.success(f"Проект переименован: «{renamed.label}». Папка "
                       f"(`{renamed.dirname}`), переписка, вложения и журналы "
                       f"остались на месте — они привязаны к ссылке "
                       f"`{renamed.ref}`.")
    # iter77: у «Сохранить» два разных смысла, и их надо различать явно.
    #   * имя в поле СОВПАДАЕТ с именем текущего проекта → пишем в него (по
    #     ссылке): это обычное сохранение;
    #   * имя ДРУГОЕ → «сохранить как»: адресуем проект с таким именем, а если
    #     его нет — создаём НОВЫЙ (с новой ссылкой) и переключаемся на него.
    # Переименование БЕЗ создания копии — отдельная кнопка выше. Молча
    # выбирать между этими двумя действиями нельзя: одно меняет подпись, другое
    # плодит проект с собственной перепиской.
    _save = st.button("💾 Сохранить проект", key="save_campaign")
    if _save and (ident is None or name.strip() != ident.label):
        # «Сохранить как»: ищем проект с таким именем, иначе создаём новый.
        try:
            _found = cs.pref.resolve(root, name)
        except ValueError as exc:            # два проекта с этим именем
            st.error(f"{exc} Загрузите нужный проект в списке ниже — тогда "
                     f"сохранение пойдёт именно в него.")
            _save = False
        else:
            ident = _found or cs.pref.create_project(root, name)
            st.session_state[dock.K_REF] = ident.ref
            st.info(f"Сохранение идёт в проект «{ident.label}» (папка "
                    f"`{ident.dirname}`, ссылка `{ident.ref}`)"
                    + (" — он создан только что, поэтому переписка у него "
                       "своя, пустая." if _found is None else "")
                    + " Чтобы сменить только подпись текущего проекта, "
                      "нажмите «✏️ Переименовать проект».")
    if _save:
        if ctrl is None:
            # iter76: до сборки сохраняем ЧЕРНОВИК НАСТРОЕК (поля формы
            # «🆕 Новый проект»). Раньше здесь стоял отказ — и вместе с
            # ошибкой сборки это давало замкнутый круг: «собрать не могу
            # (ошибка), сохранить не могу (не собран)», заполненная форма
            # терялась при закрытии вкладки.
            try:
                fields = cs.setup_draft_fields(st.session_state)
                # iter77: сохраняем В ТЕКУЩИЙ проект (по ссылке), а не в
                # каталог с именем из поля. Иначе правка имени плодила бы
                # новый каталог, а переписка оставалась в прежнем.
                path = (cs.save_setup_draft_by_ref(root, ident.ref, fields)
                        if ident is not None
                        else cs.save_setup_draft(root, name, fields))
                st.success(
                    f"Черновик настроек сохранён: {Path(path).parent.name} "
                    f"({len(fields)} полей формы). Проект ещё НЕ собран — "
                    f"черновик вернёт поля формы «🆕 Новый проект» при "
                    f"загрузке; соберите проект кнопкой «🏗 Построить "
                    f"проект», когда настройки готовы.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Не удалось сохранить черновик: {exc}")
        else:
            try:
                path = (cs.save_campaign_by_ref(
                            ctrl.runner, root, ident.ref,
                            draft=_seed_draft_from_session())
                        if ident is not None
                        else cs.save_campaign(ctrl.runner, root, name,
                                              draft=_seed_draft_from_session()))
                st.success(f"Проект сохранён: {Path(path).parent.name}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Не удалось сохранить: {exc}")

    # iter77: в списке — ИМЯ проекта (человеку), выбор — по КАТАЛОГУ (адресу).
    # Одноимённые проекты теперь возможны, поэтому подпись показывает и папку:
    # иначе две строки «Кромка» были бы неразличимы.
    idents = cs.list_projects(root)
    _by_dir = {i.dirname: i for i in idents}

    def _project_option(dirname: str) -> str:
        if dirname == "— нет —":
            return dirname
        it = _by_dir.get(dirname)
        if it is None:
            return dirname
        tail = "" if it.label == it.dirname else f" · папка «{it.dirname}»"
        return f"{it.label}{tail}"

    sel = st.selectbox("Открыть сохранённый проект",
                       ["— нет —"] + [i.dirname for i in idents],
                       format_func=_project_option, key="campaign_select")
    if st.button("📂 Загрузить проект", key="load_campaign") \
            and sel != "— нет —":
        # Ссылка выбранного проекта уходит в состояние приложения: с этого
        # момента переписка, вложения и журналы берутся по ней.
        _sel_ident = _by_dir.get(sel) or cs.pref.read_identity(root, sel)
        if _sel_ident is not None and _sel_ident.has_ref:
            st.session_state["campaign_ref_pending"] = _sel_ident.ref
        # iter76: проект-ЧЕРНОВИК (собранного campaign.json ещё нет) —
        # возвращаем поля формы «🆕 Новый проект» тем же механизмом
        # отложенного префилла, что у загрузки собранного проекта.
        # Без st.stop(): обрыв прогона здесь оставил бы колонки ассистента
        # неотрисованными.
        if not cs.campaign_exists(root, sel):
            _load_setup_draft_into_form(root, sel)
            return
        try:
            runner = cs.load_campaign(root, sel)
            draft = cs.load_campaign_draft(root, sel)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось загрузить '{sel}': {exc}")
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
            # iter77: в поле имени — ИМЯ проекта (label), а не имя каталога:
            # они теперь могут различаться (проект переименовали).
            st.session_state["campaign_name_pending"] = (
                _sel_ident.label if _sel_ident is not None else sel)
            st.session_state["camp_loaded_msg"] = (
                f"Проект '{sel}' загружен (общая база: "
                f"{len(runner.points)} точек, веток: {len(runner.branches)}"
                + ("; восстановлен незафиксированный стартовый план"
                   if has_draft else "") + ").")
            st.rerun()

    if st.session_state.get("camp_loaded_msg"):
        st.success(st.session_state.pop("camp_loaded_msg"))


def render_campaign_deleter(root: str) -> None:
    """🗑 Danger zone: удаление сохранённого проекта под паролём администратора.

    Барьер от случайного удаления (не криптозащита): нужны выбор проекта,
    подтверждение имени и admin-пароль (env ``DOE_ADMIN_PASSWORD``).
    iter72: живёт на закладке «🌱 Старт» (сайдбар упразднён).
    """
    with st.expander("🗑 Удалить проект (admin)", expanded=False):
        if st.session_state.get("camp_del_msg"):
            st.success(st.session_state.pop("camp_del_msg"))
        idents = cs.list_projects(root)
        if not idents:
            st.caption("Сохранённых проектов нет.")
            return
        st.caption("Удаление безвозвратно. Это барьер от случайного удаления "
                   "(не криптозащита): нужен пароль администратора "
                   "(переменная окружения `DOE_ADMIN_PASSWORD`). Удаление — "
                   "единственный способ прекратить жизнь проекта, включая "
                   "не стартовавший: сам он не исчезает.")
        _labels = {i.dirname: (f"{i.label} · папка «{i.dirname}»"
                               if i.label != i.dirname else i.dirname)
                   for i in idents}
        target = st.selectbox("Проект для удаления",
                              [i.dirname for i in idents],
                              format_func=lambda d: _labels.get(d, d),
                              key="camp_del_select")
        # Подтверждаем ПАПКОЙ: имя может совпадать у двух проектов, и «впишите
        # имя точно» тогда не отличало бы один от другого.
        confirm = st.text_input("Подтвердите: впишите имя ПАПКИ проекта точно",
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
                        # iter77: сравниваем по КАТАЛОГУ текущего проекта (имя
                        # для этого не годится: оно могло быть переименовано и
                        # может совпадать у двух проектов). Ссылку тоже
                        # сбрасываем — иначе состояние ссылалось бы на
                        # удалённый проект, и main() открыл бы его заново.
                        if dock.current_project(root) == target:
                            st.session_state.pop("campaign_ctrl", None)
                            st.session_state.pop(dock.K_REF, None)
                            st.session_state.pop(dock.K_SESSION, None)
                            st.session_state.pop(dock.K_PROJECT, None)
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
    st.subheader("💬 Обзор проекта помощником: состояние и подсказки")
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
        cstat[0].success("Модель подключена")
    else:
        cstat[0].warning("Не задан ключ OPENROUTER_API_KEY")
    cstat[1].caption(f"Модель: `{ai.model_name()}`")

    with st.expander("👁️ Что сейчас «видит» помощник (данные проекта)"):
        st.json(ai.build_campaign_context(overview or {}))

    if overview is None:
        st.info("Проект ещё не собран — соберите его на закладке "
                "«🌱 Старт» (или создайте демо-проект), чтобы помощник "
                "видел реальное состояние: ветки, роли, денежный канал ρ.")

    if not ai.llm_available():
        st.caption("Чтобы включить переписку, задайте `OPENROUTER_API_KEY` "
                   "(поле выше или переменная окружения). Блок «Что видит "
                   "помощник» работает и без ключа.")

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
    st.caption("Порядок работы (§17): настройка области → стартовый план "
               "опытов → ветки → рабочий стол (ручной ввод откликов) → "
               "изменение схемы. Слева — переписка с помощником (история "
               "прокручивается вверх, поле ввода внизу), в центре — рабочая "
               "область на закладках (от переписки НЕ двигается), справа — "
               "инфо-панель: вложения, файлы расчётов помощника, состояние "
               "переписки.")

    # iter77: ИДЕНТИЧНОСТЬ проекта — ссылка, а не имя. Порядок здесь важен:
    #   1) выдать ссылки проектам, созданным до iter77 (безопасно: ничего не
    #      переносится, только дописывается project.json);
    #   2) если проект в сессии ещё не выбран — открыть ДЕФОЛТНЫЙ («my_project»)
    #      и сразу назначить ему ссылку, чтобы к нему можно было подвязать
    #      переписку, вложения и черновик формы ДО сборки движка. Не
    #      стартовавший проект открывается снова как дефолтный с ТОЙ ЖЕ
    #      ссылкой — прекратить его существование может только админское
    #      удаление.
    cs.pref.migrate_root(CAMPAIGN_ROOT)
    _pending_ref = st.session_state.pop("campaign_ref_pending", None)
    if _pending_ref:
        st.session_state[dock.K_REF] = str(_pending_ref)
    if not st.session_state.get(dock.K_REF):
        _ident = cs.pref.ensure_default_project(CAMPAIGN_ROOT)
        st.session_state[dock.K_REF] = _ident.ref
        st.session_state.setdefault("campaign_name", _ident.label)

    # iter72: имя проекта — состояние ПРИЛОЖЕНИЯ, а не только виджета.
    # Виджет «Имя проекта» рисуется лишь на закладке «Старт»; когда открыта
    # другая закладка, Streamlit удаляет ключи неотрисованных виджетов — а от
    # `campaign_name` зависит подпись проекта. Пин (само-присваивание)
    # переводит ключ в app-state и переживает любые закладки. Отложенное имя
    # (`campaign_name_pending`, выставляет загрузчик) применяется тоже здесь —
    # ДО инстанцирования любого виджета с этим ключом.
    #
    # ВАЖНО (iter77): от этого ключа больше НЕ зависит сессия ассистента —
    # переписка адресуется ссылкой (`assistant_dock.current_project_ref`),
    # поэтому правка имени в поле диалог не переключает.
    st.session_state.setdefault("campaign_name", cs.pref.DEFAULT_LABEL)
    _pending_name = st.session_state.pop("campaign_name_pending", None)
    st.session_state["campaign_name"] = (
        _pending_name if _pending_name else st.session_state["campaign_name"])

    # iter72: раскладка по эскизу — ТРИ зоны: ДИАЛОГ слева (ассистент —
    # инструмент взаимодействия с программой), РАБОЧАЯ ОБЛАСТЬ в центре
    # (закладки), ИНФО-ПАНЕЛЬ справа (вложения, выхлоп песочницы, состояние
    # сессии — нужны постоянно на разных закладках). Сайдбар упразднён:
    # персистентность проекта (сохранить/загрузить/удалить) переехала на
    # закладку «🌱 Старт» через project_renderer (аргументом — чтобы поток
    # не импортировал приложение, иначе цикл импорта).
    #
    # Порядок рендера обратный порядку колонок: рабочая область рисуется ПЕРВОЙ,
    # потому что именно она публикует `ui_focus` (активная закладка, ветка), из
    # которого диалог берёт контекст «по месту» (iter65). Streamlit это
    # позволяет: содержимое колонки пишется в её контейнер, а не по порядку
    # вызовов на странице.
    def _project_panel() -> None:
        render_campaign_persistence(CAMPAIGN_ROOT)   # 📁 сохранить/загрузить
        render_campaign_deleter(CAMPAIGN_ROOT)       # 🗑 удалить (под паролём)

    dock_col, work_col, info_col = st.columns(list(wsx.MAIN_COLUMNS))
    with work_col:
        # Обзор ассистента — закладка рабочей области, а не отдельная вкладка
        # верхнего уровня: иначе «💬 Ассистент (обзор)» уводил с проекта целиком.
        render_campaign(overview_renderer=render_campaign_assistant,
                        project_renderer=_project_panel)
    ctrl_now = get_campaign_controller()
    runner_now = getattr(ctrl_now, "runner", None) if ctrl_now else None
    with dock_col:
        render_assistant_dock(runner_now, root=CAMPAIGN_ROOT)
    with info_col:
        render_assistant_info(runner_now, root=CAMPAIGN_ROOT)


if __name__ == "__main__":
    main()
