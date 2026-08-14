"""apps/workspace.py — РАСКЛАДКА рабочей области и ленты диалога (iter69).

До этого шага единственный экран проекта был одной длинной простынёй: сетап,
seed, база, ветки, рабочий стол, скрининг и эволюция схемы рисовались друг под
другом, а справа стоял док ассистента. Практическая цена такой раскладки —
страница целиком уезжает вверх при любом ответе ассистента: чтобы дописать
сообщение, приходилось искать поле ввода, а чтобы вернуться к таблице —
скроллить обратно. Диалог и рабочая область мешали друг другу, потому что
делили ОДИН скролл документа.

Этот модуль — ЧИСТЫЙ (без Streamlit) слой решения:

* **закладки рабочей области** (:data:`WORKSPACE_TABS`) — секции потока §17
  разложены по закладкам, видна ровно одна; поэтому её содержимое влезает в
  контейнер фиксированной высоты со СВОИМ скроллом и не ползёт от чата;
* **гейты доступности** (:func:`tab_states`) — закладка, для которой нет
  данных (база пуста, веток нет), не исчезает молча, а показывается
  ВЫКЛЮЧЕННОЙ с причиной (A0.6);
* **дефолт по состоянию** (:func:`default_tab_key`) — пока seed не измерен,
  открыт «Старт»; после фиксации seed рабочая область сама переходит к
  веткам, а не оставляет человека на уже сделанном шаге;
* **закладки веток** (:func:`branch_labels`) — второй ряд закладок, линза
  контекста (Тр-3.3): значение остаётся ``branch_id``, а подпись человеческая;
* **фокус ассистента** (:func:`focus_section_for`) — активная закладка ЕСТЬ
  место пользователя, и именно её ключ уходит в ``ui_focus`` (iter65), поэтому
  док спрашивает про то, что открыто, а не про «вообще»;
* **высоты панелей** (:data:`WORKSPACE_HEIGHT`, :data:`CHAT_FEED_HEIGHT`) —
  лента диалога живёт в контейнере ФИКСИРОВАННОЙ высоты (история уходит вверх,
  поле ввода стоит под ней), а рабочая область с iter88 высоты НЕ ограничивает:
  она растёт по содержимому и прокручивается общим скроллом страницы
  (:func:`workspace_box_kwargs`);
* **липкие боковые зоны** (:func:`sticky_zones_css`, iter89) — раз центр листает
  всю страницу, диалог слева и инфо-панель справа удерживаются на экране через
  ``position: sticky``, иначе ассистент уезжает вверх вместе с документом;
* **вставка скриншота** (:func:`chat_paste_js`, iter93) — ``st.chat_input``
  буфер обмена не поддерживает вообще, поэтому Ctrl+V со скриншотом молча не
  работал; мост перекладывает картинку из буфера в скрытый загрузчик файлов
  того же поля ввода штатным путём.

Ключи закладок совпадают с ключами шагов
:data:`src.assistant.context.FOCUS_SECTIONS` там, где шаг существует — это
проверяется тестом, чтобы карта мест ассистента и раскладка UI не разъезжались.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Высота рабочей области (px) — ИСТОРИЧЕСКОЕ значение iter69, оставленное для
#: совместимости ссылок и как ориентир «сколько влезает в экран».
#:
#: iter88 (запрос технолога): рабочая область БОЛЬШЕ не обрезается по этой
#: высоте. Идея iter69 была «чат не двигает таблицу», но цена оказалась выше
#: выгоды: длинные панели (форма сетапа с 18 компонентами, план на 135 опытов,
#: рабочий стол ветки) читались через окно в 760 px — внутренний скролл рядом с
#: внешним, и прокручивать приходилось постоянно. Теперь центр — обычная
#: страница с одним скроллом (см. :data:`WORKSPACE_SCROLL`).
WORKSPACE_HEIGHT = 760

#: Ограничивать ли рабочую область по высоте. ``False`` = обычная страница:
#: содержимое растёт как есть, прокрутка — общая для документа. Флаг оставлен
#: ЯВНЫМ (а не удалением кода), чтобы возврат к окну iter69 был правкой одного
#: значения, а не реконструкцией решения по истории git.
WORKSPACE_SCROLL = False

#: Высота ленты диалога (px). Лента скроллится внутри себя (история — вверх),
#: поле ввода стоит ПОД ней и не уезжает.
CHAT_FEED_HEIGHT = 560

#: Сколько последних сообщений показываем в ленте (остальное — в журнале сессии).
FEED_LIMIT = 40

#: Пропорции колонок главного экрана (эскиз пользователя, iter72 — ТРИ зоны):
#: слева ДИАЛОГ с ассистентом (инструмент взаимодействия с программой), в
#: центре РАБОЧАЯ ОБЛАСТЬ на закладках, справа ИНФО-ПАНЕЛЬ — дополнительная
#: информация, которая нужна постоянно на разных закладках (вложения, выхлоп
#: песочницы, состояние сессии). Центр — самый широкий: там таблицы и формы.
MAIN_COLUMNS: Tuple[int, int, int] = (2, 5, 2)


def workspace_box_kwargs(*, supports_height: bool = True) -> Dict[str, Any]:
    """Аргументы контейнера рабочей области (чистая функция, без Streamlit).

    Одно место, где решается «окно с внутренним скроллом или обычная страница»,
    — иначе решение размазывается по вызову виджета и проверяется только
    глазами.

    * :data:`WORKSPACE_SCROLL` ``False`` (iter88, текущее поведение) → ``{}``:
      контейнер рисуется без ``height``, содержимое растёт по своей длине,
      прокрутка одна — страницы. Именно этого просил технолог: длинные панели
      (сетап на 18 компонентов, план на 135 опытов) не читаются через окно.
    * ``True`` → ``{"height": WORKSPACE_HEIGHT, "border": True}`` — поведение
      iter69 (рабочая область не двигается от ответов ассистента).

    ``supports_height=False`` (старый Streamlit без ``st.container(height=…)``)
    всегда даёт ``{}``: вызов виджета не должен падать из-за версии.
    """
    if not (WORKSPACE_SCROLL and supports_height):
        return {}
    return {"height": WORKSPACE_HEIGHT, "border": True}


# ----------------------------------------------------------------------
# Липкие боковые зоны (iter89)
# ----------------------------------------------------------------------
#: Ключи контейнеров боковых зон. Streamlit вешает на контейнер с ``key`` CSS-класс
#: ``st-key-<key>`` (`st.container` docstring, 1.58) — это ЕДИНСТВЕННЫЙ
#: документированный способ адресовать зону из CSS: сам ``st.columns`` ключей не
#: принимает, а порядковые селекторы (`nth-child`) поехали бы от любой правки
#: раскладки.
DOCK_ZONE_KEY = "zone_dock"
INFO_ZONE_KEY = "zone_info"

#: Отступ липкой зоны от верха окна (rem) — под шапкой Streamlit.
STICKY_TOP_REM = 3.0

#: Ниже какой ширины окна (px) липкость ОТКЛЮЧАЕТСЯ. На узком экране Streamlit
#: переносит колонки друг под друга (`@media (max-width: breakpoints.columns)`),
#: и «липкий» блок там перекрыл бы содержимое.
STICKY_MIN_WIDTH_PX = 992

#: Ограничивать ли боковые зоны по высоте окна. ``True``: зона, которая длиннее
#: экрана (переписка + панели утверждения), получает СВОЙ скролл, иначе её низ
#: был бы недосягаем — прилипшая зона не прокручивается вместе со страницей.
STICKY_SIDE_SCROLL = True


def sticky_zones_css(*, top_rem: float = STICKY_TOP_REM,
                     min_width_px: int = STICKY_MIN_WIDTH_PX,
                     side_scroll: bool = STICKY_SIDE_SCROLL) -> str:
    """CSS, делающий боковые зоны липкими при прокрутке центра (iter89).

    Чистая функция: строит текст правил, ничего не рисует. Так решение видно
    целиком в одном месте и проверяется тестом, а не глазами в браузере.

    Как это работает на фактической разметке Streamlit 1.58 (проверено по
    бандлу ``static/js/index.*.js``):

    * скроллится ``section[data-testid="stMain"]`` (``overflow: auto``,
      ``height: 100dvh``) — значит ``position: sticky`` у элемента ВНУТРИ него
      прилипает к этому контейнеру, а не к окну;
    * колонка ряда — ``div[data-testid="stColumn"]``, прямой flex-элемент
      строки ``stHorizontalBlock``; своего ``overflow`` у неё нет, поэтому
      липкость не гасится;
    * адресуем колонку через ``:has(.st-key-<ключ>)`` — то есть «колонка, внутри
      которой лежит наш именованный контейнер». Липким делается САМА КОЛОНКА:
      вешать ``sticky`` на внутренний контейнер бесполезно, его родитель-колонка
      всё равно уедет вверх.

    ``align-self: flex-start`` обязателен: строка колонок растянута
    (``align-items: stretch``), а растянутому на всю высоту элементу прилипать
    некуда — это самая частая причина «sticky не работает».
    """
    dock, info = f".st-key-{DOCK_ZONE_KEY}", f".st-key-{INFO_ZONE_KEY}"
    sel = (f'[data-testid="stColumn"]:has({dock}), '
           f'[data-testid="stColumn"]:has({info})')
    height = (f"\n    max-height: calc(100dvh - {top_rem + 0.5:g}rem);"
              "\n    overflow-y: auto;" if side_scroll else "")
    return f"""<style>
@media (min-width: {min_width_px}px) {{
  {sel} {{
    position: sticky;
    top: {top_rem:g}rem;
    align-self: flex-start;{height}
  }}
}}
</style>"""


# ----------------------------------------------------------------------
# Вставка скриншота в поле ввода диалога (iter93)
# ----------------------------------------------------------------------
#: Ключ поля ввода диалога (``st.chat_input(key=…)``). Живёт здесь, а не
#: литералом в доке, потому что от него зависит ТОЧКА КРЕПЛЕНИЯ вставки:
#: Streamlit вешает на контейнер элемента класс ``st-key-<ключ>``
#: (``stElementContainer`` + ``iV(aV(kg(element)))`` в бандле 1.58) — это
#: единственный документированный способ адресовать конкретный виджет из JS.
DOCK_INPUT_KEY = "dock_input"

#: Флаг «мост уже поставлен» в ``window``. Streamlit перерисовывает страницу на
#: каждом прогоне и вставляет наш скрипт заново — без флага слушатели
#: накапливались бы, и один скриншот вставлялся бы N раз.
PASTE_BRIDGE_FLAG = "__doeChatPasteBridge"

#: Тело моста вставки. Держим ОДНОЙ строкой-шаблоном (а не сборкой из кусков),
#: чтобы решение читалось целиком и проверялось тестом состава.
#:
#: Почему это вообще нужно: ``st.chat_input`` умеет выбор файла и drag&drop, но
#: НЕ умеет буфер обмена — в ``ChatInput.*.js`` (Streamlit 1.58) нет ни одного
#: упоминания ``paste``/``clipboard``. Мост делает ровно то, чего не хватает:
#: перекладывает картинку из ``clipboardData`` в скрытый ``input[type=file]``
#: того же виджета и сообщает об этом штатным событием ``change`` — дальше
#: работает обычный путь загрузки (``react-dropzone`` читает
#: ``e.target.files``), поэтому файл попадает в ``chat_input`` как выбранный
#: руками, без новых зависимостей и своего протокола.
_PASTE_JS = """<script>
(function () {
  var FLAG = "__FLAG__";
  if (window[FLAG]) { return; }
  window[FLAG] = true;
  var ZONE = ".st-key-__KEY__";

  function pictures(cd) {
    var out = [];
    if (!cd) { return out; }
    Array.prototype.slice.call(cd.items || []).forEach(function (it) {
      if (it.kind !== "file") { return; }
      var f = it.getAsFile();
      if (f && String(f.type || "").indexOf("image/") === 0) { out.push(f); }
    });
    return out;
  }

  function named(f, n) {
    // Из буфера файл приходит без осмысленного имени (Chrome даёт
    // "image.png"): в переписке все скриншоты выглядели бы одинаково, а
    // дедуп вложений идёт по sha256 — имя нужно только человеку.
    var type = String(f.type || "image/png");
    var ext = type.split("/")[1] || "png";
    var name = String(f.name || "");
    if (!name || name === "image.png" || name.indexOf(".") === -1) {
      name = "screenshot-" + Date.now() + (n ? "-" + n : "") + "." + ext;
    }
    return new File([f], name, { type: type });
  }

  document.addEventListener("paste", function (ev) {
    var imgs = pictures(ev.clipboardData);
    // Картинок в буфере нет — уходим молча: обычная вставка текста в поле
    // ввода должна работать как всегда.
    if (!imgs.length) { return; }
    var zone = document.querySelector(ZONE);
    var input = zone ? zone.querySelector("input[type=file]") : null;
    if (!input) { return; }
    var dt = new DataTransfer();
    imgs.forEach(function (f, n) { dt.items.add(named(f, n)); });
    input.files = dt.files;
    input.dispatchEvent(new Event("change", { bubbles: true }));
    ev.preventDefault();
  }, true);
})();
</script>"""


def chat_paste_js(input_key: str = DOCK_INPUT_KEY, *,
                  flag: str = PASTE_BRIDGE_FLAG) -> str:
    """JS-мост «Ctrl+V со скриншотом» для поля ввода диалога (iter93).

    Чистая функция: возвращает ТЕКСТ вставки, ничего не рисует — так решение
    видно целиком в одном месте и проверяется тестом, а не глазами в браузере
    (тот же приём, что :func:`sticky_zones_css`).

    Проверенные факты о Streamlit 1.58, на которых мост держится:

    * ``st.chat_input`` буфер обмена НЕ поддерживает: в ``ChatInput.*.js`` нет
      ни ``paste``, ни ``clipboard``, ни ``onPaste`` — работают только выбор
      файла и drag&drop. Это пробел вышестоящей библиотеки, а не наш;
    * скрытый ``input[type=file]`` виджета отдаёт файлы в ``react-dropzone``
      через ``e.target.files``, поэтому программная подстановка ``FileList``
      (``DataTransfer``) плюс событие ``change`` идут ШТАТНЫМ путём загрузки;
    * контейнер элемента с ключом получает класс ``st-key-<ключ>``, что даёт
      устойчивую точку крепления: порядковые селекторы поехали бы от любой
      правки раскладки;
    * ``st.html(..., unsafe_allow_javascript=True)`` не оборачивает вставку в
      iframe и пересоздаёт узлы ``script``, поэтому код работает в ОСНОВНОМ
      документе — то есть видит то же поле ввода, что и человек.

    Мост НЕ перехватывает вставку текста: если в буфере нет картинки, событие
    уходит дальше без изменений.
    """
    return (_PASTE_JS.replace("__KEY__", str(input_key))
            .replace("__FLAG__", str(flag)))


# ----------------------------------------------------------------------
# Закладки рабочей области
# ----------------------------------------------------------------------
#: Требование к состоянию проекта, без которого закладка бессмысленна.
NEED_NONE = ""
NEED_PROJECT = "project"
NEED_POINTS = "points"
NEED_BRANCHES = "branches"

#: Почему закладка выключена (текст показывается человеку — A0.6).
WHY: Dict[str, str] = {
    NEED_PROJECT: ("проект не собран: соберите его на закладке «🌱 Старт» "
                   "или создайте демо-проект"),
    NEED_POINTS: ("стартовый план ещё не измерен: зафиксируйте его на "
                  "закладке «🌱 Старт»"),
    NEED_BRANCHES: "веток пока нет: создайте ветку на закладке «🌿 Ветки»",
}


@dataclass(frozen=True)
class WorkspaceTab:
    """Закладка рабочей области = один шаг потока §17.

    ``focus`` — ключ шага в карте мест ассистента (``ui_focus``); пустая
    строка означает «шаг ассистенту не соответствует» (обзорные закладки).
    ``needs`` — требование к состоянию проекта (см. :data:`WHY`).
    """
    key: str
    title: str
    focus: str = ""
    needs: str = NEED_NONE
    blurb: str = ""


#: Порядок закладок = порядок работы: старт → база → ветки → анализ → схема.
WORKSPACE_TABS: Tuple[WorkspaceTab, ...] = (
    WorkspaceTab(
        key="start", title="🌱 Старт", focus="seed", needs=NEED_NONE,
        # iter72: «Старт» доступен ВСЕГДА — это вход в проект (сетап, сохранить/
        # загрузить). Требовать проект для закладки, на которой он собирается, —
        # курица и яйцо: до iter72 форма жила в сайдбаре, теперь сайдбара нет.
        blurb=("Проект: собрать новый (настройка области §17.4), "
               "сохранить/загрузить, затем стартовый план опытов: предложить "
               "план, внести измеренные отклики, зафиксировать.")),
    WorkspaceTab(
        key="base", title="📚 База опытов", focus="base", needs=NEED_POINTS,
        blurb=("Общая база всех измеренных опытов (И-1): выгрузка в Excel, "
               "исправление откликов, условия прогона.")),
    WorkspaceTab(
        key="branches", title="🌿 Ветки", focus="branch", needs=NEED_POINTS,
        blurb=("Ветки проекта: роли откликов, цели, рабочий стол ветки, "
               "рецепт x*, отмена настройки и дочерние ветки. Второй ряд "
               "закладок — сами ветки.")),
    WorkspaceTab(
        key="screening", title="📊 Анализ", focus="screening",
        needs=NEED_POINTS,
        blurb=("Что дали измеренные опыты: влияние компонентов на свойства "
               "(модель Шеффе + оценка важности).")),
    WorkspaceTab(
        key="evolution", title="🧬 Схема", focus="evolution",
        needs=NEED_POINTS,
        blurb=("Эволюция схемы: раскрыть ось/компонент/отклик, подвинуть "
               "границы области — с явной политикой миграции точек.")),
    WorkspaceTab(
        key="overview", title="🤖 Обзор", focus="", needs=NEED_NONE,
        blurb=("Что именно «видит» ассистент: сводка проекта, ветки, роли, "
               "денежный канал ρ — и чат-обзор без инструментов ядра.")),
)

TABS_BY_KEY: Dict[str, WorkspaceTab] = {t.key: t for t in WORKSPACE_TABS}


@dataclass(frozen=True)
class TabState:
    """Закладка + доступна ли она сейчас + причина отказа (если нет)."""
    tab: WorkspaceTab
    enabled: bool
    why: str = ""

    @property
    def key(self) -> str:
        return self.tab.key

    @property
    def title(self) -> str:
        return self.tab.title


def tab_states(*, has_project: bool, n_points: int = 0, n_branches: int = 0
               ) -> List[TabState]:
    """Состояние ВСЕХ закладок: выключенная закладка остаётся видимой.

    Прятать шаг, для которого не хватает данных, — значит заставлять человека
    гадать, куда он делся; поэтому закладка показывается, но выключена, и при
    ней стоит причина (A0.6).
    """
    out: List[TabState] = []
    for tab in WORKSPACE_TABS:
        need = tab.needs
        if need == NEED_PROJECT and not has_project:
            out.append(TabState(tab, False, WHY[NEED_PROJECT]))
        elif need == NEED_POINTS and (not has_project or int(n_points) <= 0):
            out.append(TabState(
                tab, False,
                WHY[NEED_PROJECT] if not has_project else WHY[NEED_POINTS]))
        elif need == NEED_BRANCHES and int(n_branches) <= 0:
            out.append(TabState(tab, False, WHY[NEED_BRANCHES]))
        else:
            out.append(TabState(tab, True))
    return out


def enabled_keys(*, has_project: bool, n_points: int = 0, n_branches: int = 0
                 ) -> List[str]:
    """Ключи доступных закладок (в порядке :data:`WORKSPACE_TABS`)."""
    return [s.key for s in tab_states(has_project=has_project,
                                      n_points=n_points,
                                      n_branches=n_branches) if s.enabled]


def default_tab_key(*, has_project: bool, n_points: int = 0,
                    n_branches: int = 0) -> str:
    """Закладка по умолчанию = САМЫЙ КОНКРЕТНЫЙ доступный шаг.

    Пока проекта нет — «Старт» (iter72: там сетап и загрузка проекта — это
    вход в работу); пока seed не измерен, работать можно только на «Старте»;
    после измеренного seed интерес человека уходит к веткам (там рабочий
    стол), а не к уже сделанному стартовому дизайну.
    """
    if not has_project:
        return "start"
    if int(n_points) <= 0:
        return "start"
    return "branches"


def resolve_tab(requested: Any, *, has_project: bool, n_points: int = 0,
                n_branches: int = 0) -> Tuple[str, str]:
    """Какую закладку показать по запросу пользователя → ``(ключ, причина)``.

    Выбор человека уважается, пока он ОСТАЁТСЯ возможным: «База» перестаёт
    существовать, если проект пересобран с нуля. В этом случае возвращается
    дефолт и НЕПУСТАЯ причина, которую интерфейс обязан показать: молча
    перекидывать человека на другой экран нельзя (A0.6).
    """
    ok = enabled_keys(has_project=has_project, n_points=n_points,
                      n_branches=n_branches)
    key = str(requested or "")
    if key in ok:
        return key, ""
    fallback = default_tab_key(has_project=has_project, n_points=n_points,
                               n_branches=n_branches)
    if fallback not in ok:                     # страховка: обзор всегда жив
        fallback = ok[0] if ok else "overview"
    if not key:
        return fallback, ""
    tab = TABS_BY_KEY.get(key)
    if tab is None:
        return fallback, f"Закладки «{key}» больше нет — открыта «{fallback}»."
    state = [s for s in tab_states(has_project=has_project, n_points=n_points,
                                   n_branches=n_branches) if s.key == key]
    why = state[0].why if state else ""
    return fallback, (f"Закладка «{tab.title}» недоступна: {why}."
                      if why else f"Закладка «{tab.title}» недоступна.")


#: Фазы проекта: пусто → собран → seed измерен → есть ветки. Фаза меняется
#: ТОЛЬКО действием человека (построить проект, зафиксировать seed, создать
#: ветку), поэтому смена фазы — законный повод открыть следующий шаг.
PHASE_EMPTY = "empty"
PHASE_SETUP = "setup"
PHASE_MEASURED = "measured"
PHASE_BRANCHED = "branched"

#: Что произошло с проектом — для подписи об автопереходе.
PHASE_EVENT: Dict[str, str] = {
    PHASE_SETUP: "проект собран",
    PHASE_MEASURED: "стартовый план измерен",
    PHASE_BRANCHED: "появились ветки",
    PHASE_EMPTY: "проекта в сессии нет",
}


def phase_key(*, has_project: bool, n_points: int = 0, n_branches: int = 0
              ) -> str:
    """Фаза проекта одной строкой (см. :data:`PHASE_EMPTY` и далее)."""
    if not has_project:
        return PHASE_EMPTY
    if int(n_points) <= 0:
        return PHASE_SETUP
    if int(n_branches) <= 0:
        return PHASE_MEASURED
    return PHASE_BRANCHED


@dataclass(frozen=True)
class TabDecision:
    """Какую закладку показать: ключ, текущая фаза и что сказать человеку."""
    key: str
    phase: str
    notice: str = ""
    moved: bool = False


def decide_tab(current: Any, *, prev_phase: Any, has_project: bool,
               n_points: int = 0, n_branches: int = 0) -> TabDecision:
    """Активная закладка с учётом СМЕНЫ фазы проекта.

    Без этого правила человек застревал на закладке, открытой до действия:
    создал проект, стоя на «Обзоре», — и продолжал смотреть обзор, хотя ждал
    форму старта. Поэтому: сменилась фаза (собрали проект, зафиксировали seed,
    создали первую ветку) — открывается дефолтная закладка НОВОЙ фазы; фаза та
    же — уважается выбор человека (:func:`resolve_tab`).

    Автопереход не молчаливый: в ``notice`` уходит текст «что случилось и куда
    попали» (A0.6).
    """
    phase = phase_key(has_project=has_project, n_points=n_points,
                      n_branches=n_branches)
    if str(prev_phase or "") != phase:
        key = default_tab_key(has_project=has_project, n_points=n_points,
                              n_branches=n_branches)
        # iter90: «переход» туда, где человек УЖЕ стоит, — не переход, и
        # рапортовать о нём нечего. Раньше уведомление всё равно вставало в
        # дерево элементов на СЛЕДУЮЩЕМ прогоне (смена фазы фиксируется после
        # отрисовки), то есть на первом же действии пользователя после сборки
        # проекта: вёрстка сдвигалась, stateless-экспандеры перемонтировались,
        # и форма сетапа захлопывалась (живой отказ 13.08.2026).
        if str(current or "") == key:
            return TabDecision(key=key, phase=phase)
        tab = TABS_BY_KEY.get(key)
        title = tab.title if tab is not None else key
        notice = ("" if not prev_phase else
                  f"{PHASE_EVENT.get(phase, 'состояние проекта изменилось')} "
                  f"→ открыта закладка «{title}».")
        return TabDecision(key=key, phase=phase, notice=notice, moved=True)
    key, why = resolve_tab(current, has_project=has_project,
                           n_points=n_points, n_branches=n_branches)
    return TabDecision(key=key, phase=phase, notice=why, moved=bool(why))


def focus_section_for(key: Any) -> str:
    """Ключ шага ассистента (``ui_focus.section``) по закладке рабочей области.

    Пустая строка = у закладки нет соответствующего шага; тогда фокус лучше не
    публиковать вовсе, чем публиковать неверный (ассистент честно скажет «шаг
    не определён»).
    """
    tab = TABS_BY_KEY.get(str(key or ""))
    return tab.focus if tab is not None else ""


def tab_titles(states: Sequence[TabState]) -> List[str]:
    """Подписи закладок для ряда закладок (в том же порядке)."""
    return [s.title for s in states]


# ----------------------------------------------------------------------
# Второй ряд: закладки веток проекта
# ----------------------------------------------------------------------
def branch_labels(branches: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """``{branch_id: подпись}`` для ряда закладок веток.

    Значением закладки остаётся ``branch_id`` (его читают контроллер и
    ``ui_focus``), а человек видит имя ветки. Имя не уникально — поэтому в
    подписи остаётся id: две ветки «premium» иначе неразличимы.
    """
    out: Dict[str, str] = {}
    for bid, br in dict(branches or {}).items():
        name = str(getattr(br, "name", "") or "").strip()
        out[str(bid)] = (f"{name} ({bid})" if name and name != str(bid)
                         else str(bid))
    return out


def resolve_branch(requested: Any, branch_ids: Sequence[str]) -> str:
    """Активная ветка: выбор человека, если он ещё существует, иначе первая.

    Ветку могли удалить (или проект перезагрузили) — тогда сохранённый id
    указывает в пустоту, и линза контекста должна честно вернуться к первой
    существующей ветке. Пустая строка = веток нет вообще.
    """
    ids = [str(b) for b in branch_ids]
    key = str(requested or "")
    if key in ids:
        return key
    return ids[0] if ids else ""


# ----------------------------------------------------------------------
# Лента диалога
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FeedItem:
    """Одна реплика ленты: роль, текст, приложенные картинки, файлы расчёта.

    ``artifacts`` (iter84) — id артефактов, посчитанных ЭТИМ ответом. Нужны,
    чтобы график и таблица оставались В РАЗГОВОРЕ, а не только на том прогоне
    Streamlit, где ход выполнялся: до iter84 они рисовались из
    ``TurnResult.new_artifacts`` (память одного прогона) и пропадали при первом
    же rerun — файл на диске цел, а в переписке его больше нет.
    """
    role: str
    content: str
    images: Tuple[str, ...] = ()
    artifacts: Tuple[str, ...] = ()


def _is_tool_stub(message: Any) -> bool:
    """Реплика ассистента БЕЗ текста, но с заявкой на вызовы (iter92).

    Отдельная функция, а не условие внутри цикла: то же правило нужно
    :func:`dialog_count`, иначе подпись «реплик в ленте N из M» разошлась бы
    с тем, что видно на экране.
    """
    if str(getattr(message, "content", "") or "").strip():
        return False
    return bool(getattr(message, "tool_calls", None))


def feed_items(messages: Optional[Sequence[Any]], *, limit: int = FEED_LIMIT
               ) -> List[FeedItem]:
    """Реплики для показа в ленте — СТАРЫЕ СВЕРХУ, новые снизу.

    Порядок именно такой, потому что лента ведёт себя как переписка: свежий
    ответ появляется внизу (туда же уводит автоскролл контейнера), а история
    остаётся выше — её прокручивают вверх. Служебные роли (``system``/``tool``)
    в ленту не идут: аудит инструментов показывается отдельной панелью хода.
    """
    out: List[FeedItem] = []
    for m in list(messages or []):
        role = str(getattr(m, "role", "") or "")
        if role not in ("user", "assistant"):
            continue
        if role == "assistant" and _is_tool_stub(m):
            # iter92: реплика-ЗАЯВКА на инструменты (текста нет, есть только
            # tool_calls) в переписку не идёт. Такие сообщения появились в
            # сессии вместе с сохранением контекста прерванного хода; в ленте
            # они рисовались бы пустым пузырём «помощник ничего не сказал»,
            # хотя он как раз работал. Что именно он вызывал, видно в панели
            # хода и в аудите вызовов.
            continue
        imgs = tuple(str(s) for s in (getattr(m, "images", None) or []))
        # iter84: файлы расчёта переносятся в ленту вместе с репликой — их
        # показ больше не зависит от того, тот ли это прогон, в котором ход
        # выполнялся.
        arts = tuple(str(s) for s in (getattr(m, "artifacts", None) or []))
        out.append(FeedItem(role=role,
                            content=str(getattr(m, "content", "") or ""),
                            images=imgs, artifacts=arts))
    n = int(limit)
    return out[-n:] if n > 0 else out


def dialog_count(messages: Optional[Sequence[Any]]) -> int:
    """Сколько ВСЕГО реплик диалога в сессии (без служебных ролей).

    Нужно, чтобы честно сказать, что лента показывает только хвост: сравнивать
    с ``len(session.messages)`` нельзя — там ещё системные врезки фокуса и
    ответы инструментов.
    """
    return sum(1 for m in list(messages or [])
               if str(getattr(m, "role", "") or "") in ("user", "assistant")
               and not _is_tool_stub(m))


def feed_hint(n_shown: int, n_total: int) -> str:
    """Подпись под лентой: сколько реплик видно и где остальное.

    Лента ограничена, чтобы длинная переписка не превращала колонку в
    бесконечный документ; но умолчать об урезании нельзя — иначе человек
    решит, что часть разговора потерялась (A0.6).
    """
    if int(n_total) <= int(n_shown):
        return (f"Реплик в ленте: {int(n_shown)}. История прокручивается "
                "вверх, новые сообщения появляются внизу.")
    return (f"Показаны последние {int(n_shown)} из {int(n_total)} реплик "
            "(вся переписка сохранена в сессии проекта).")
