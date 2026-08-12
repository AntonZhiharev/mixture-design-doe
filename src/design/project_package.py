"""design/project_package.py — ПАКЕТ ПРОЕКТА: спека + отклики + процесс-оси (iter73).

Закрываемый отказ (живая сессия 11.08.2026). Ассистент собрал геометрию кампании
верно и положил пакет спеки в стейдж, человек нажал «Применить» — и НИЧЕГО не
произошло: ``apply_spec`` пишет геометрию через ``runner.set_phr_spec``, а
раннера в сессии не было. Причина не в кнопке, а в логике: phr-спека — это
mixture-блок СХЕМЫ проекта, то есть применение спеки при пустой сессии есть не
правка проекта, а его РОЖДЕНИЕ. Инструмент же был написан как правка
существующего.

Одной спеки для рождения проекта, однако, не хватает: ``build_setup_runner``
требует ещё ОТКЛИКИ (свойства) и ПРОЦЕСС-ОСИ с границами, а по схеме
(``spec_schema.not_in_spec``) их в phr-спеке нет и быть не может. Выдумать их за
технолога нельзя (A0.6), поэтому вводится пакет ПРОЕКТА: те же данные, что
человек вбивает в форму «🆕 Новый проект», но одним самодокументируемым JSON.

**Почему пакет самодокументируемый.** Ввод проекта — интерактивный процесс в
несколько подходов: пакет читают глазами и правят частями. Поэтому блоки названы
своими именами (``spec`` / ``responses`` / ``process`` / ``covariates`` /
``passport``), у осей и откликов есть ``unit`` и ``note``, а
:func:`package_manifest` отвечает на вопрос «что именно я сейчас загружаю» до
всякого применения.

**Границы модуля.** Здесь НЕТ Streamlit и НЕТ сборки раннера: пакет только
разбирается, валидируется ядром и проецируется в ЗНАЧЕНИЯ ПОЛЕЙ формы сетапа
(:func:`package_to_setup_prefill`). Раннер рождает штатная кнопка «🏗 Построить
проект» — один путь сборки проекта на всё приложение, а не два расходящихся.
Слой ассистента импортирует этот модуль, слой UI проецирует его результат в
виджеты.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .phr_sampler import PhrSpec

#: Ключ-маркер вида пакета: «проект целиком», а не «только геометрия».
PACKAGE_KIND = "project"

#: Ключи верхнего уровня пакета. Список закрытый: неизвестный ключ — ошибка, а
#: не тихое игнорирование (опечатка в имени блока молча теряла бы половину
#: проекта — ровно тот класс отказа, что закрывал iter71).
TOP_KEYS: Tuple[str, ...] = (
    "package_kind", "spec", "responses", "process", "covariates",
    "passport", "seed", "label", "note",
)

#: Обязательные блоки: без них ``build_setup_runner`` откажет, и отказ лучше
#: получить ДО применения — с указанием, чего именно не хватает.
REQUIRED_KEYS: Tuple[str, ...] = ("spec", "responses", "process")

#: Ключи блока одной процесс-оси.
PROCESS_KEYS: Tuple[str, ...] = ("name", "range", "unit", "levels", "note")

#: Ключи блока одного отклика.
RESPONSE_KEYS: Tuple[str, ...] = ("name", "unit", "note")

#: Ключи паспорта кампании (политика, записываемая ДО первого замера).
PASSPORT_KEYS: Tuple[str, ...] = (
    "campaign_label", "preflight_pairs", "material_lots", "anchor_recipes",
    "weighing_step_g", "grams_per_phr", "process_links",
)


class PackageError(ValueError):
    """Пакет проекта не разбирается — с указанием, ЧТО именно не так.

    Отдельный тип, чтобы слой ассистента мог отличить «пакет собран неверно»
    (ответ по существу, ход не падает) от внутренней поломки.
    """


# ----------------------------------------------------------------------
# Разбор блоков
# ----------------------------------------------------------------------
def _as_mapping(obj: Any, what: str) -> Dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise PackageError(f"{what} должен быть объектом JSON, получено "
                           f"{type(obj).__name__}.")
    return dict(obj)


def _as_pair(value: Any, what: str) -> Tuple[float, float]:
    """``[lo, hi]`` → пара чисел (границы оси в РЕАЛЬНЫХ единицах)."""
    if isinstance(value, Mapping):
        # Частая форма от модели: {"min": 165, "max": 185}. Принимаем, но
        # приводим к канону — молча терять границы нельзя.
        value = [value.get("min", value.get("lo")),
                 value.get("max", value.get("hi"))]
    if (isinstance(value, str) or not isinstance(value, Sequence)
            or len(value) != 2):
        raise PackageError(
            f"{what}: ожидалась пара [нижняя, верхняя] в реальных единицах, "
            f"получено {value!r}.")
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise PackageError(f"{what}: границы должны быть числами, получено "
                           f"{value!r}.") from exc
    if not hi > lo:
        raise PackageError(
            f"{what}: верхняя граница должна быть строго больше нижней, "
            f"получено [{lo:g}, {hi:g}]. Постоянный параметр — не ось проекта: "
            f"уберите его из блока 'process'.")
    return lo, hi


def _names_unique(names: Sequence[str], what: str) -> None:
    seen: Dict[str, int] = {}
    for nm in names:
        seen[nm] = seen.get(nm, 0) + 1
    dupes = sorted(nm for nm, k in seen.items() if k > 1)
    if dupes:
        raise PackageError(f"{what}: имена повторяются — {dupes}.")


def parse_responses(raw: Any) -> List[Dict[str, str]]:
    """Блок ОТКЛИКОВ → список ``{name, unit, note}``.

    Принимаем и краткую форму (``["gloss", "rho"]``), и полную с единицами:
    единицы нужны человеку при вводе Y в лаборатории, но требовать их нельзя —
    у части свойств их нет (индекс, балл).
    """
    if raw is None:
        raise PackageError(
            "Блок 'responses' (отклики/свойства) отсутствует. Без откликов "
            "проект не собирается: измерять будет нечего. В phr-спеке их нет "
            "по схеме — их задаёт технолог.")
    if isinstance(raw, (str, Mapping)):
        raise PackageError(
            "Блок 'responses' должен быть СПИСКОМ откликов: [\"gloss\"] или "
            "[{\"name\": \"gloss\", \"unit\": \"%\"}].")
    out: List[Dict[str, str]] = []
    for i, item in enumerate(list(raw), 1):
        if isinstance(item, str):
            item = {"name": item}
        d = _as_mapping(item, f"Отклик №{i}")
        extra = sorted(set(d) - set(RESPONSE_KEYS))
        if extra:
            raise PackageError(f"Отклик №{i}: неизвестные ключи {extra} "
                               f"(допустимы {list(RESPONSE_KEYS)}).")
        name = str(d.get("name", "")).strip()
        if not name:
            raise PackageError(f"Отклик №{i}: пустое имя.")
        out.append({"name": name, "unit": str(d.get("unit", "") or ""),
                    "note": str(d.get("note", "") or "")})
    if not out:
        raise PackageError(
            "Блок 'responses' пуст: нужен хотя бы один отклик (свойство), "
            "иначе движку нечему обучаться.")
    _names_unique([d["name"] for d in out], "Отклики")
    return out


def parse_process(raw: Any) -> List[Dict[str, Any]]:
    """Блок ПРОЦЕСС-ОСЕЙ → список ``{name, range, unit, levels, note}``.

    ``range`` — в РЕАЛЬНЫХ единицах (°C, об/мин): нормировку в код [0, 1] движок
    делает сам. ``levels`` — дискретная сетка «что умеет железо» (iter52):
    значения обязаны лежать внутри ``range``, иначе план предложит режим,
    которого на линии нет.
    """
    if raw is None:
        raise PackageError(
            "Блок 'process' (процесс-оси с границами) отсутствует. По §17.4 "
            "процесс задаётся СРАЗУ вместе с составом; в phr-спеке его нет по "
            "схеме. Укажите оси с границами в реальных единицах.")
    if isinstance(raw, (str, Mapping)):
        raise PackageError(
            "Блок 'process' должен быть СПИСКОМ осей: "
            "[{\"name\": \"T_plast\", \"range\": [165, 185], \"unit\": \"°C\"}].")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(list(raw), 1):
        d = _as_mapping(item, f"Процесс-ось №{i}")
        extra = sorted(set(d) - set(PROCESS_KEYS))
        if extra:
            raise PackageError(
                f"Процесс-ось №{i}: неизвестные ключи {extra} (допустимы "
                f"{list(PROCESS_KEYS)}). Достижимые режимы — это СПИСОК "
                f"значений 'levels', а не их количество.")
        name = str(d.get("name", "")).strip()
        if not name:
            raise PackageError(f"Процесс-ось №{i}: пустое имя.")
        lo, hi = _as_pair(d.get("range"), f"Процесс-ось '{name}'")
        out.append({"name": name, "range": [lo, hi],
                    "unit": str(d.get("unit", "") or ""),
                    "levels": _parse_levels(d.get("levels"), name, lo, hi),
                    "note": str(d.get("note", "") or "")})
    if not out:
        raise PackageError(
            "Блок 'process' пуст: нужна хотя бы одна процесс-ось (§17.4 — "
            "процесс задаётся сразу вместе с составом).")
    _names_unique([d["name"] for d in out], "Процесс-оси")
    return out


def _parse_levels(raw: Any, name: str, lo: float, hi: float) -> List[float]:
    """Дискретные режимы оси: СПИСОК достижимых значений внутри ``[lo, hi]``.

    Число («levels: 3») отвергается намеренно: количество уровней — политика
    ПЛАНА, а сетка железа — политика проекта. Их путали в живой сессии.
    """
    if raw is None:
        return []
    if isinstance(raw, bool) or isinstance(raw, (int, float)):
        raise PackageError(
            f"Процесс-ось '{name}': 'levels' — это СПИСОК достижимых значений "
            f"(например [400, 900]), а не их количество ({raw!r}). Число "
            f"уровней плана — политика плана, не проекта.")
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise PackageError(f"Процесс-ось '{name}': 'levels' должен быть "
                           f"списком чисел.")
    try:
        levels = [float(v) for v in raw]
    except (TypeError, ValueError) as exc:
        raise PackageError(f"Процесс-ось '{name}': уровни должны быть числами, "
                           f"получено {list(raw)!r}.") from exc
    bad = [v for v in levels if v < lo - 1e-9 or v > hi + 1e-9]
    if bad:
        raise PackageError(
            f"Процесс-ось '{name}': уровни {bad} лежат ВНЕ границ "
            f"[{lo:g}, {hi:g}] — план предложил бы режим, которого на линии "
            f"нет.")
    if len(set(levels)) < len(levels):
        raise PackageError(f"Процесс-ось '{name}': уровни повторяются — "
                           f"{levels}.")
    return levels


def parse_covariates(raw: Any) -> List[str]:
    """Ковариаты базы (телеметрия прогона) — необязательный блок."""
    if raw is None:
        return []
    if isinstance(raw, (str, Mapping)):
        raise PackageError("Блок 'covariates' должен быть списком имён.")
    out = [str(x).strip() for x in list(raw) if str(x).strip()]
    _names_unique(out, "Ковариаты")
    return out


def parse_passport(raw: Any) -> Dict[str, Any]:
    """Паспорт кампании: метка, лоты, anchor-рецепты, разрешение весов, пары."""
    if raw is None:
        return {}
    d = _as_mapping(raw, "Блок 'passport'")
    extra = sorted(set(d) - set(PASSPORT_KEYS))
    if extra:
        raise PackageError(f"Блок 'passport': неизвестные ключи {extra} "
                           f"(допустимы {list(PASSPORT_KEYS)}).")
    return d


# ----------------------------------------------------------------------
# Пакет целиком
# ----------------------------------------------------------------------
class ProjectPackage:
    """Разобранный пакет проекта: спека ЯДРА + отклики + оси + паспорт.

    Собирается только через :func:`parse_project_package`, поэтому существование
    объекта уже означает: спека собрана конструктором ``PhrSpec``, оси и отклики
    непротиворечивы. «Полупринятого» пакета не бывает — иначе человек утверждал
    бы кнопкой то, что применится лишь частично.
    """

    def __init__(self, *, spec: PhrSpec, responses: List[Dict[str, str]],
                 process: List[Dict[str, Any]], covariates: List[str],
                 passport: Dict[str, Any], seed: int, label: str, note: str,
                 raw: Dict[str, Any]):
        self.spec = spec
        self.responses = responses
        self.process = process
        self.covariates = covariates
        self.passport = passport
        self.seed = int(seed)
        self.label = str(label or "")
        self.note = str(note or "")
        self.raw = dict(raw)

    @property
    def response_names(self) -> List[str]:
        return [d["name"] for d in self.responses]

    @property
    def process_names(self) -> List[str]:
        return [d["name"] for d in self.process]

    @property
    def component_names(self) -> List[str]:
        return list(self.spec.component_names)

    @property
    def spec_hash(self) -> str:
        return str(self.spec.spec_hash())

    def process_levels(self) -> Dict[str, List[float]]:
        """Только оси с ЗАДАННОЙ сеткой: пустой список — это «непрерывная»."""
        return {d["name"]: list(d["levels"]) for d in self.process
                if d["levels"]}


def parse_project_package(package: Any) -> ProjectPackage:
    """Разобрать и проверить пакет проекта ЯДРОМ (ничего не применяя).

    Порядок проверок выбран так, чтобы сообщение указывало на самый внешний
    промах: сначала вид пакета, потом обязательные блоки, потом каждый блок
    отдельно. Иначе ошибка в третьей оси маскировала бы отсутствие откликов.
    """
    if isinstance(package, str):
        try:
            package = json.loads(package)
        except json.JSONDecodeError as exc:
            raise PackageError(
                f"Пакет проекта: некорректный JSON (строка {exc.lineno}, "
                f"позиция {exc.colno}): {exc.msg}.") from exc
    d = _as_mapping(package, "Пакет проекта")

    kind = str(d.get("package_kind", "") or "")
    if kind and kind != PACKAGE_KIND:
        raise PackageError(
            f"Это пакет вида '{kind}', а ожидался '{PACKAGE_KIND}'. Пакет ТОЛЬКО "
            f"геометрии (спека) применяется другим инструментом — apply_spec.")
    extra = sorted(set(d) - set(TOP_KEYS))
    if extra:
        raise PackageError(
            f"Пакет проекта: неизвестные ключи верхнего уровня {extra} "
            f"(допустимы {list(TOP_KEYS)}). Узлы состава кладутся ВНУТРЬ блока "
            f"'spec', а не рядом с ним.")
    missing = [k for k in REQUIRED_KEYS if d.get(k) is None]
    if missing:
        raise PackageError(
            f"Пакет проекта неполный: нет блоков {missing}. Проект рождается из "
            f"ТРЁХ обязательных блоков — 'spec' (состав, phr-геометрия), "
            f"'responses' (что меряем) и 'process' (оси процесса с границами). "
            f"Откликов и осей в phr-спеке нет по схеме: их задаёт технолог, "
            f"выдумывать их нельзя.")

    try:
        spec = PhrSpec.from_dicts(d["spec"])
    except PackageError:
        raise
    except Exception as exc:                                # noqa: BLE001
        raise PackageError(
            f"Блок 'spec' не собирается ядром ({type(exc).__name__}: {exc}). "
            f"Формат узлов и инварианты — из spec_schema; проект не тронут."
        ) from exc

    responses = parse_responses(d.get("responses"))
    process = parse_process(d.get("process"))
    covariates = parse_covariates(d.get("covariates"))
    passport = parse_passport(d.get("passport"))
    _check_name_clashes(spec, responses, process, covariates)
    # iter76: пары preflight паспорта валидируются ЗДЕСЬ, против спеки и осей
    # пакета — а не при нажатии «🏗 Построить проект». Раньше пара по имени
    # группы (FILLER | Chalk_95T) проходила dry-run и стейдж, а падала только
    # на кнопке сборки: человек утверждал пакет, который не собирается.
    # Группы разворачиваются в сумму членов сразу — форма получает канон.
    if passport.get("preflight_pairs"):
        passport["preflight_pairs"] = normalize_preflight_pairs(
            passport["preflight_pairs"], spec,
            process_names=[str(p["name"]) for p in process])

    try:
        seed = int(d.get("seed", 1) or 1)
    except (TypeError, ValueError) as exc:
        raise PackageError(f"'seed' должен быть целым числом, получено "
                           f"{d.get('seed')!r}.") from exc

    return ProjectPackage(
        spec=spec, responses=responses, process=process,
        covariates=covariates, passport=passport, seed=seed,
        label=str(d.get("label", "") or ""), note=str(d.get("note", "") or ""),
        raw=d)


def normalize_preflight_pairs(pairs: Any, spec: PhrSpec, *,
                              process_names: Sequence[str] = ()
                              ) -> List[List[List[str]]]:
    """iter76: пары preflight паспорта → канон ``[[левые], [правые]]``.

    Правила те же, что у ядра (``set_preflight_pairs``): сторона пары — имя
    координаты (компонент спеки или процесс-ось пакета), имя УЗЛА-ГРУППЫ
    спеки (``FILLER``, ``SOFT`` — ядро само развернёт её в сумму членов при
    сборке) либо СПИСОК имён (ось-сумма). Имя группы НЕ разворачивается
    здесь: канон пакета остаётся человекочитаемым. Неизвестное имя —
    :class:`PackageError` с перечислением, что допустимо: отказ должен
    случиться на dry-run пакета, а не на кнопке «🏗 Построить проект».
    """
    comps = set(spec.component_names)
    procs = {str(p) for p in process_names}
    groups = (spec.group_members()
              if hasattr(spec, "group_members") else {})

    def _side(value: Any, where: str) -> List[str]:
        names = [value] if isinstance(value, str) else \
            [str(x) for x in (value or [])]
        if not names:
            raise PackageError(f"{where}: пустая сторона пары недопустима.")
        out: List[str] = []
        for nm in names:
            if nm in comps or nm in procs or nm in groups:
                out.append(nm)
            else:
                raise PackageError(
                    f"{where}: имя '{nm}' не является ни компонентом спеки, "
                    f"ни процесс-осью пакета"
                    + (f", ни группой спеки {sorted(groups)}"
                       if groups else "")
                    + ". Ось-сумма задаётся списком имён компонентов.")
        return out

    out: List[List[List[str]]] = []
    for i, item in enumerate(list(pairs or []), 1):
        where = f"passport.preflight_pairs[{i}]"
        if isinstance(item, Mapping):
            left = item.get("left", item.get("a"))
            right = item.get("right", item.get("b"))
        elif isinstance(item, Sequence) and not isinstance(item, str) \
                and len(item) == 2:
            left, right = item[0], item[1]
        else:
            raise PackageError(
                f"{where}: пара должна быть [левое, правое] или "
                f"{{'left': …, 'right': …}}, получено {item!r}.")
        out.append([_side(left, where), _side(right, where)])
    return out


def _check_name_clashes(spec: PhrSpec, responses: List[Dict[str, str]],
                        process: List[Dict[str, Any]],
                        covariates: Sequence[str]) -> None:
    """Имена трёх блоков живут в ОДНОЙ схеме проекта: пересечения запрещены.

    Совпадение сделало бы координату и отклик одним столбцом общей базы. Ловим
    здесь, а не в раннере, — чтобы причина стояла рядом с кнопкой.
    """
    comps = set(spec.component_names)
    resp = {d["name"] for d in responses}
    proc = {d["name"] for d in process}
    clash = sorted(comps & resp)
    if clash:
        raise PackageError(
            f"Имена откликов совпадают с компонентами смеси: {clash}. Отклик — "
            f"то, что МЕРЯЮТ, компонент — то, что дозируют.")
    clash = sorted(comps & proc)
    if clash:
        raise PackageError(
            f"Имена процесс-осей совпадают с компонентами смеси: {clash}.")
    clash = sorted(proc & resp)
    if clash:
        raise PackageError(
            f"Имена процесс-осей совпадают с откликами: {clash}. Ось задаёт "
            f"оператор, отклик измеряется — это разные величины.")
    clash = sorted(set(covariates) & (comps | proc | resp))
    if clash:
        raise PackageError(
            f"Имена ковариат заняты компонентами/осями/откликами: {clash}.")


# ----------------------------------------------------------------------
# «Что именно загружается» — манифест пакета
# ----------------------------------------------------------------------
def package_manifest(pkg: ProjectPackage) -> Dict[str, Any]:
    """ЧТО грузится, по блокам — ответ до всякого применения.

    Требование пользователя: ввод проекта идёт в несколько подходов, поэтому из
    пакета должно быть видно, что именно приедет — состав, отклики, оси с
    границами и единицами. Один общий текст «пакет проекта» здесь бесполезен:
    сверять человек будет построчно, с таблицей технолога в руках.

    Чистая функция (без Streamlit) — питает и UI-таблицу, и ответ инструмента.
    """
    blocks: List[Dict[str, Any]] = [{
        "блок": "состав (phr-спека)",
        "что": f"{len(pkg.component_names)} компонентов, "
               f"{len(pkg.spec.nodes)} узлов, dim z = {int(pkg.spec.dim_z)}",
        "детали": ", ".join(pkg.component_names),
        "единицы": "phr (доли считает ядро)",
    }, {
        "блок": "отклики (что меряем)",
        "что": f"{len(pkg.responses)} шт.",
        "детали": ", ".join(_response_caption(d) for d in pkg.responses),
        "единицы": "по каждому отклику отдельно",
    }, {
        "блок": "процесс-оси (границы)",
        "что": f"{len(pkg.process)} шт.",
        "детали": "; ".join(_axis_caption(d) for d in pkg.process),
        "единицы": "РЕАЛЬНЫЕ (нормировку движок делает сам)",
    }]
    levels = pkg.process_levels()
    blocks.append({
        "блок": "дискретные режимы осей",
        "что": (f"{len(levels)} из {len(pkg.process)} осей на сетке"
                if levels else "нет — все оси непрерывные"),
        "детали": "; ".join(f"{k}: {', '.join(f'{v:g}' for v in vs)}"
                            for k, vs in levels.items()) or "—",
        "единицы": "реальные",
    })
    blocks.append({
        "блок": "ковариаты (телеметрия)",
        "что": f"{len(pkg.covariates)} шт." if pkg.covariates else "нет",
        "детали": ", ".join(pkg.covariates) or "—",
        "единицы": "столбцы базы, в модель не входят",
    })
    blocks.append({
        "блок": "паспорт кампании",
        "что": (", ".join(sorted(pkg.passport)) if pkg.passport
                else "не задан"),
        "детали": str(pkg.passport.get("campaign_label", "") or "—"),
        "единицы": "политика, фиксируется ДО первого замера",
    })
    return {
        "package_kind": PACKAGE_KIND,
        "label": pkg.label,
        "spec_hash": pkg.spec_hash,
        "seed": pkg.seed,
        "blocks": blocks,
        "components": list(pkg.component_names),
        "responses": [dict(d) for d in pkg.responses],
        "process": [dict(d) for d in pkg.process],
        "covariates": list(pkg.covariates),
        "note": pkg.note,
    }


def _response_caption(d: Mapping[str, Any]) -> str:
    unit = str(d.get("unit", "") or "")
    return f"{d['name']}" + (f" [{unit}]" if unit else "")


def _axis_caption(d: Mapping[str, Any]) -> str:
    lo, hi = float(d["range"][0]), float(d["range"][1])
    unit = str(d.get("unit", "") or "")
    txt = f"{d['name']} {lo:g}…{hi:g}" + (f" {unit}" if unit else "")
    lv = list(d.get("levels") or [])
    return txt + (f" (режимы: {', '.join(f'{v:g}' for v in lv)})" if lv else "")


def manifest_caption(pkg: ProjectPackage) -> str:
    """Одна строка о пакете: сколько чего приедет (для подписи в UI и ответа)."""
    levels = pkg.process_levels()
    return (f"состав: {len(pkg.component_names)} компонентов "
            f"(hash {pkg.spec_hash[:12]}…) · отклики: "
            f"{len(pkg.responses)} · процесс-оси: {len(pkg.process)}"
            + (f" (на сетке: {len(levels)})" if levels else "")
            + (f" · ковариаты: {len(pkg.covariates)}" if pkg.covariates else "")
            + (f" · паспорт: {len(pkg.passport)} поля" if pkg.passport else ""))


# ----------------------------------------------------------------------
# Схема пакета как ДАННЫЕ (формат не пересказывается словами)
# ----------------------------------------------------------------------
#: Минимальный ВАЛИДНЫЙ пакет проекта: по одному представителю каждого блока.
#: Пример живёт рядом с валидатором и покрыт тестом — пример, который сам не
#: проходит разбор, хуже отсутствия примера (урок iter71).
PROJECT_EXAMPLE: Dict[str, Any] = {
    "package_kind": PACKAGE_KIND,
    "label": "кромка ПВХ: первичный ввод проекта",
    "seed": 1,
    "spec": {
        "spec_version": 2,
        "group_order": ["SOFT"],
        "nodes": [
            {"name": "RESIN", "role": "FIXED", "value": 100.0},
            {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
            {"name": "SOFT", "role": "GROUP_TOTAL", "range": [3.0, 15.0],
             "members": ["CPE", "PBNK"]},
            {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT",
             "min_phr": 3.0},
            {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
             "share_range": [0.0, 0.70], "max_phr": 8.0},
        ],
    },
    "responses": [
        {"name": "gloss", "unit": "%", "note": "блеск 60°"},
        {"name": "dE", "unit": "ΔE", "note": "цветовое отличие от эталона"},
    ],
    "process": [
        {"name": "T_plast", "range": [165.0, 185.0], "unit": "°C",
         "note": "температура пластикации"},
        {"name": "rotor_rpm", "range": [400.0, 900.0], "unit": "об/мин",
         "levels": [400.0, 900.0], "note": "две передачи экструдера"},
    ],
    "covariates": ["SME", "die_pressure"],
    "passport": {"campaign_label": "PVC-кромка-2026",
                 "weighing_step_g": 0.1, "grams_per_phr": 10.0},
}


def project_package_schema(*, include_example: bool = True) -> Dict[str, Any]:
    """Схема пакета проекта КАК ДАННЫЕ: блоки, обязательность, единицы, пример.

    Тот же приём, что у ``spec_schema`` (iter71): формат, пересказанный словами,
    модель восстанавливает по памяти и промахивается ключами — а промах стоит
    целого хода. Здесь схема собирается из тех же констант, по которым работает
    валидатор, поэтому разъехаться с ним не может.
    """
    out: Dict[str, Any] = {
        "package_kind": PACKAGE_KIND,
        "top_keys": list(TOP_KEYS),
        "required": list(REQUIRED_KEYS),
        "blocks": {
            "spec": {
                "что": "СОСТАВ: phr-спека целиком (тот же формат, что даёт "
                       "spec_schema — обёртка spec_version/nodes/group_order "
                       "или плоский список узлов)",
                "обязателен": True,
                "единицы": "phr; доли для mixture-блока считает ядро "
                           "(fraction_bounds)",
            },
            "responses": {
                "что": "ОТКЛИКИ (свойства), которые меряют в лаборатории",
                "обязателен": True,
                "формат": "список строк или объектов "
                          + str(list(RESPONSE_KEYS)),
                "единицы": "'unit' у каждого отклика (необязательно, но "
                           "полезно при вводе Y)",
            },
            "process": {
                "что": "ПРОЦЕСС-ОСИ с границами (§17.4: процесс задаётся сразу)",
                "обязателен": True,
                "формат": "список объектов " + str(list(PROCESS_KEYS)),
                "единицы": "РЕАЛЬНЫЕ (°C, об/мин); нормировку в [0,1] движок "
                           "делает сам",
                "levels": "СПИСОК достижимых значений (сетка железа), а не их "
                          "количество; значения обязаны лежать внутри 'range'",
            },
            "covariates": {
                "что": "телеметрия прогона — столбцы общей базы",
                "обязателен": False,
                "единицы": "в модель и желательности НЕ входят",
            },
            "passport": {
                "что": "политика кампании ДО первого замера",
                "обязателен": False,
                "формат": "объект с ключами " + str(list(PASSPORT_KEYS)),
            },
            "seed": {"что": "зерно ГСЧ движка проекта", "обязателен": False,
                     "единицы": "целое; на состав и границы не влияет"},
        },
        "invariants": [
            "Имена компонентов, откликов, процесс-осей и ковариат НЕ "
            "пересекаются: это столбцы одной общей базы.",
            "Границы каждой оси: верхняя строго больше нижней (постоянный "
            "параметр — не ось проекта).",
            "'levels' — список значений внутри 'range'; число уровней плана "
            "здесь не задаётся.",
            "Блок 'spec' проверяется тем же конструктором PhrSpec, что и "
            "apply_spec: пакет валиден только целиком.",
        ],
        "hint": "Собрал пакет — зови propose_project: он проверит его ядром и "
                "положит в стейдж. Применяет ЧЕЛОВЕК кнопкой; поля формы "
                "сетапа заполнятся из пакета, проект соберёт кнопка "
                "«🏗 Построить проект».",
    }
    if include_example:
        out["example"] = json.loads(json.dumps(PROJECT_EXAMPLE))
    return out


# ----------------------------------------------------------------------
# Проекция в поля формы сетапа (раннер собирает штатная кнопка)
# ----------------------------------------------------------------------
#: Схема «phr-спека (JSON)» и канал ввода — ярлыки виджетов формы сетапа
#: (``campaign_ui._MODE_PHR`` / ``_PHR_SRC_JSON``). Дублируются здесь строками
#: НАМЕРЕННО: ядро не должно импортировать слой Streamlit. Совпадение ярлыков
#: проверяется тестом — разъезд обнаружится сразу, а не в проде.
SETUP_MODE_PHR = "phr-спека (JSON)"
SETUP_SRC_JSON = "JSON / файл"


def package_to_setup_prefill(pkg: ProjectPackage) -> Dict[str, Any]:
    """Пакет проекта → ЗНАЧЕНИЯ ПОЛЕЙ формы «🆕 Новый проект».

    Раннер здесь НЕ собирается (решение пользователя): пакет заполняет форму, а
    проект рождает штатная кнопка «🏗 Построить проект». Так путь сборки проекта
    в приложении остаётся ОДИН — и то, что человек утвердил, он видит в тех же
    полях, которые заполнял бы руками, с возможностью поправить перед сборкой.

    Формат словаря — тот же, что у ``campaign_ui.setup_prefill_from_runner``
    (ключи виджетов Streamlit), поэтому применяется тем же механизмом
    ``setup_prefill_pending``. Чистая функция: тестируется без UI.
    """
    proc = pkg.process
    d = len(proc)
    out: Dict[str, Any] = {
        # Имена компонентов в режиме phr-спеки берутся ИЗ СПЕКИ (поле формы
        # игнорируется), но заполняем и его — иначе человек видит «A, B, C» и
        # думает, что состав не приехал.
        "setup_mix": ", ".join(pkg.component_names),
        "setup_resp": ", ".join(pkg.response_names),
        "setup_proc": ", ".join(pkg.process_names),
        "setup_seed": int(pkg.seed),
        "setup_comp_mode": SETUP_MODE_PHR,
        "setup_phr_src": SETUP_SRC_JSON,
        "setup_phr_json": json.dumps(pkg.spec.to_dicts(), ensure_ascii=False,
                                     indent=2),
        "setup_covariates": ", ".join(pkg.covariates),
        "setup_process_levels": _levels_text(pkg),
    }
    for i, ax in enumerate(proc):
        out[f"setup_plo_{d}_{i}"] = float(ax["range"][0])
        out[f"setup_phi_{d}_{i}"] = float(ax["range"][1])

    p = pkg.passport
    if "campaign_label" in p:
        out["setup_campaign_label"] = str(p.get("campaign_label", "") or "")
    elif pkg.label:
        out["setup_campaign_label"] = pkg.label
    if "weighing_step_g" in p:
        out["setup_pass_weigh_step"] = float(p.get("weighing_step_g") or 0.0)
    if "grams_per_phr" in p:
        out["setup_pass_weigh_gpp"] = float(p.get("grams_per_phr") or 0.0)
    if p.get("preflight_pairs"):
        out["setup_preflight_pairs"] = _pairs_text(p["preflight_pairs"])
    if p.get("material_lots"):
        out["setup_material_lots"] = "\n".join(
            f"{k}: {v}" for k, v in dict(p["material_lots"]).items())
    if p.get("anchor_recipes"):
        out["setup_anchor_recipes"] = _anchors_text(p["anchor_recipes"])
    if p.get("process_links"):
        out["setup_process_links"] = _links_text(p["process_links"])
    return out


def _levels_text(pkg: ProjectPackage) -> str:
    """Дискретные режимы → текст формы («имя: v, v»); пусто — все непрерывные."""
    return "\n".join(f"{nm}: {', '.join(f'{v:g}' for v in vs)}"
                     for nm, vs in pkg.process_levels().items())


def _pairs_text(pairs: Any) -> str:
    """Обязательные 2D-пары preflight → текст формы («A, B | C»)."""
    lines: List[str] = []
    for item in list(pairs or []):
        if isinstance(item, Mapping):
            left = item.get("left", item.get("a", []))
            right = item.get("right", item.get("b", []))
        elif isinstance(item, Sequence) and not isinstance(item, str) \
                and len(item) == 2:
            left, right = item
        else:
            raise PackageError(
                f"passport.preflight_pairs: пара должна быть [левое, правое] "
                f"или {{'left': …, 'right': …}}, получено {item!r}.")
        lines.append(f"{_names_text(left)} | {_names_text(right)}")
    return "\n".join(lines)


def _names_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return ", ".join(str(x) for x in value)
    return str(value)


def _anchors_text(anchors: Any) -> str:
    """Anchor-рецепты → текст формы («имя: комп=phr, комп=phr»)."""
    lines: List[str] = []
    for name, recipe in dict(anchors or {}).items():
        body = ", ".join(f"{k}={float(v):g}"
                         for k, v in dict(recipe or {}).items())
        lines.append(f"{name}: {body}")
    return "\n".join(lines)


def _links_text(links: Any) -> str:
    """Связанные оси (P3.3) → текст формы «имя: осьA - осьB : lo, hi».

    Канон ключей — тот же, что у ``set_process_links``: ``minuend`` /
    ``subtrahend`` (принимаем и синонимы ``left``/``right``). Открытая сторона
    полосы пишется ``*``: полоса в этом канале обязательна, «нет ограничения» —
    это ``*``, а не пропуск.
    """
    lines: List[str] = []
    for item in list(links or []):
        d = _as_mapping(item, "passport.process_links[]")
        name = str(d.get("name", "") or "")
        a = str(d.get("minuend", d.get("left", "")) or "")
        b = str(d.get("subtrahend", d.get("right", "")) or "")
        if not (name and a and b):
            raise PackageError(
                "passport.process_links[]: нужны 'name', 'minuend', "
                "'subtrahend' (производная ось = minuend − subtrahend).")
        lo = _bound_text(d.get("lo", d.get("min")))
        hi = _bound_text(d.get("hi", d.get("max")))
        lines.append(f"{name}: {a} - {b} : {lo}, {hi}")
    return "\n".join(lines)


def _bound_text(v: Any) -> str:
    """Граница полосы → текст («*» — сторона открыта)."""
    if v is None or (isinstance(v, str) and v.strip() in ("", "*")):
        return "*"
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError) as exc:
        raise PackageError(f"passport.process_links[]: граница {v!r} не число "
                           f"и не «*».") from exc
