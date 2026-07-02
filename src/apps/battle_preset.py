"""apps/battle_preset.py — пресет «Заполнить тестовыми данными» (STEP 6).

Назначение: одним вызовом получить ГОТОВЫЙ снимок конфигурации (`cfg_defaults`
для сайдбар-формы) с ИЗВЕСТНОЙ синтетической истиной, перенесённой из самого
широкого этапа battle-теста (`tests/unit/test_iteration13_battle.py`, STEP 6):
4 компонента {A,B,C,D}, реальные цены, отклики strength/gloss/dry_time/
whiteStrength/ρ.

В интерфейсе этот пресет подаётся кнопкой «🧪 Заполнить тестовыми данными»
(исторически — «Боевой пресет»). Для встроенного ИИ-ассистента предусмотрено
развёрнутое человекочитаемое описание :func:`battle_step6_description`, чтобы он
мог объяснить пользователю, что это за данные и как ими пользоваться.


Маппинг mixture×process → чистый mixture (по согласованию, ответ пользователя):
берём ТОЛЬКО mixture-термы истины STEP 6 (линейные + парные + тройной A*B*C),
процессные термы (T, P, T², P², A:T, C:T) отбрасываем — в pure-mixture Scheffé
у них нет аналога. Тройной терм A*B*C требует модели ``cubic`` (в ядре
``scheffe_matrix`` это special-cubic: linear + парные + тройные).

Коэффициенты задаются по ПОЗИЦИЯМ термов через :func:`scheffe_term_indices`
(ключ — отсортированный кортеж индексов компонентов A=0,B=1,C=2,D=3), без
завязки на строковый формат имён термов.
"""
from __future__ import annotations

from typing import Dict, List

from src.core.linalg import scheffe_term_indices

# Компоненты STEP 6 и их цены [усл.ед/кг] (см. _PRICE4 в battle-тесте).
_COMPONENTS: List[str] = ["A", "B", "C", "D"]
_PRICES: List[float] = [95.0, 200.0, 23.0, 315.0]
_COST_UNIT = "усл.ед/кг"
# Модель ИСТИНЫ синт.лаборатории: cubic (special-cubic) — несёт тройной A*B*C.
_TRUTH_MODEL = "cubic"
# Модель КОНВЕЙЕРА (D-опт/Scheffé-интерпретация M3, форма UI): quadratic.
# Расцеплена с истиной (см. PipelineConfig.truth_model): GP/MoE делают реальную
# математику, а Scheffé quadratic — лишь наглядная аппроксимация cubic-истины.
_MODEL = "quadratic"

_PROPERTIES: List[str] = ["strength", "gloss", "dry_time", "whiteStrength", "rho"]

# Воспроизводимость лаборатории-симулятора и уровень шума отклика по умолчанию.
_SEED = 13
_NOISE_SD = 0.2

# Подпись кнопки в сайдбаре (исторически — «Боевой пресет»).
_BUTTON_LABEL = "🧪 Заполнить тестовыми данными"

# Краткое человекочитаемое описание каждого свойства (для ассистента/подсказок).
_PROPERTY_DESCRIPTIONS: Dict[str, str] = {
    "strength": "прочность покрытия (целевой максимум)",
    "gloss": "глянец/блеск поверхности",
    "dry_time": "время высыхания",
    "whiteStrength": "белизна·прочность (комбинированный показатель)",
    "rho": "плотность ρ (линейная по составу)",
}


# Mixture-проекция истины STEP 6 (_build_econ_truth): свойство → {терм: коэф.},
# где терм — отсортированный кортеж индексов компонентов. Процессные термы
# (T, P, T², P², A:T, C:T) сюда НЕ входят (отброшены).
_TRUTH_BY_TUPLE: Dict[str, Dict[tuple, float]] = {
    "strength": {(0,): 6.0, (1,): 10.0, (2,): 2.0, (3,): 2.0,
                 (0, 1): 9.0, (0, 2): 5.0, (1, 2): 16.0, (1, 3): 4.0,
                 (0, 1, 2): 12.0},
    "gloss": {(0,): 3.0, (1,): 7.0, (2,): 3.0, (3,): 2.0,
              (0, 1): 7.0, (0, 2): 6.0, (1, 2): 14.0, (0, 1, 2): 15.0},
    "dry_time": {(0,): -4.0, (1,): 2.0, (2,): 4.0, (3,): 1.0},
    "whiteStrength": {(0,): 5.0, (1,): 6.0, (2,): 3.0, (3,): 1.0,
                      (0, 3): -5.0, (1, 3): -4.0, (2, 3): -8.0},
    "rho": {(0,): 0.7, (1,): 1.0, (2,): 1.7, (3,): 0.6},
}


def battle_step6_coef_by_property() -> Dict[str, List[float]]:
    """Полные векторы коэф. Scheffé под (q=4, model='cubic') на каждое свойство.

    Раскладывает разреженные термы :data:`_TRUTH_BY_TUPLE` по позициям термов
    модели; отсутствующие термы — 0.0. Длина каждого вектора равна числу термов
    cubic-модели для q=4.
    """
    q = len(_COMPONENTS)
    term_indices = scheffe_term_indices(q, _TRUTH_MODEL)
    out: Dict[str, List[float]] = {}
    for prop, sparse in _TRUTH_BY_TUPLE.items():
        vec = [float(sparse.get(tuple(sorted(idx)), 0.0)) for idx in term_indices]
        out[prop] = vec
    return out


def battle_step6_snapshot(name: str = "battle_step6", seed: int = _SEED,
                          noise_sd: float = _NOISE_SD) -> Dict:
    """Снимок конфигурации (для ``cfg_defaults`` сайдбар-формы) пресета STEP 6.


    Ключи совпадают со снимком :meth:`PipelineRunner.config_snapshot`, поэтому
    форма читает его так же, как при загрузке сохранённого проекта. Включает
    ``truth_coef_by_property`` — известную истину лаборатории.
    """
    q = len(_COMPONENTS)
    return {
        "name": name,
        "q": q,
        "model": _MODEL,
        "truth_model": _TRUTH_MODEL,
        "noise_sd": float(noise_sd),
        "seed": int(seed),
        "names": list(_COMPONENTS),
        "lower": [0.0] * q,
        "upper": [1.0] * q,
        "n_runs_factor": 2.0,
        "n_random": 600,
        "n_restarts": 8,
        "n_blocks": 1,
        "cost_coeffs": list(_PRICES),
        "cost_unit": _COST_UNIT,
        "property_names": list(_PROPERTIES),
        "batch_size": None,
        "batch_unit": None,
        "comp_mode": "fractions",
        "base_index": None,
        "parts_min": None,
        "parts_max": None,
        "truth_coef_by_property": battle_step6_coef_by_property(),
    }


def _term_label(idx: tuple) -> str:
    """Читаемое имя терма по кортежу индексов компонентов: (0,1) → 'A·B'."""
    return "·".join(_COMPONENTS[i] for i in idx)


def battle_step6_truth_readable() -> Dict[str, List[Dict]]:
    """Истина STEP 6 в человекочитаемом виде: свойство → список термов.

    Каждый терм: ``{"term": "A·B", "components": ["A","B"], "coef": 9.0,
    "order": 2}`` (order: 1 — линейный, 2 — парный, 3 — тройной). Удобно для
    ассистента — объяснить «какой компонент/взаимодействие как влияет».
    """
    out: Dict[str, List[Dict]] = {}
    for prop, sparse in _TRUTH_BY_TUPLE.items():
        terms = []
        for idx, coef in sparse.items():
            terms.append({
                "term": _term_label(idx),
                "components": [_COMPONENTS[i] for i in idx],
                "coef": float(coef),
                "order": len(idx),
            })
        out[prop] = terms
    return out


def battle_step6_description() -> Dict:
    """Развёрнутое человекочитаемое описание пресета «Заполнить тестовыми данными».

    JSON-сериализуемая карточка для встроенного ИИ-ассистента: назначение,
    компоненты с ценами, целевые свойства, модель/seed/шум, ИЗВЕСТНАЯ истина
    лаборатории (в читаемых термах) и порядок работы. Опираясь на неё, ассистент
    может объяснить пользователю, что это за тестовые данные и зачем они нужны,
    не выдумывая чисел.
    """
    q = len(_COMPONENTS)
    return {
        "_note": "Описание набора «Заполнить тестовыми данными» (кнопка в "
                 "сайдбаре). Это эталонный демонстрационный пресет с ИЗВЕСТНОЙ "
                 "синтетической истиной лаборатории — на него можно опираться, "
                 "объясняя пользователю, что это за данные и как с ними работать.",
        "button_label": _BUTTON_LABEL,
        "where": "Сайдбар, отдельная кнопка под блоком конфигурации проекта.",
        "purpose": (
            "Одним кликом заполняет форму готовой конфигурацией (4 компонента, "
            "реальные цены, 5 целевых свойств) с заранее заданной "
            "ДЕТЕРМИНИРОВАННОЙ истиной полигона (модель cubic / special-cubic). "
            "Лаборатория-симулятор воспроизводима по seed, поэтому можно сразу "
            "прогонять стадии M1…M8 и сверять результат конвейера с "
            "аналитическим оптимумом во вкладке «Benchmark»."),
        "origin": "Самый широкий этап battle-теста (STEP 6, "
                  "tests/unit/test_iteration13_battle.py).",
        "components": [
            {"name": n, "price": p, "price_unit": _COST_UNIT}
            for n, p in zip(_COMPONENTS, _PRICES)
        ],
        "properties": [
            {"name": p, "meaning": _PROPERTY_DESCRIPTIONS.get(p, "")}
            for p in _PROPERTIES
        ],
        "model_ui": _MODEL,
        "truth_model": _TRUTH_MODEL,
        "seed": _SEED,
        "noise_sd": _NOISE_SD,
        "composition": {
            "mode": "fractions",
            "lower": [0.0] * q,
            "upper": [1.0] * q,
            "note": "Полный симплекс: доля каждого компонента 0…1, сумма = 1.",
        },
        "known_truth": {
            "_note": "Истина — mixture-проекция STEP 6: только mixture-термы "
                     "Шеффе (линейные, парные и тройной A·B·C); процессные термы "
                     "(T, P, …) отброшены. coef — коэффициент терма; order: "
                     "1=линейный (вклад чистого компонента), 2=парное "
                     "взаимодействие, 3=тройное взаимодействие.",
            "terms_by_property": battle_step6_truth_readable(),
        },
        "how_to_use": [
            "Нажать «🧪 Заполнить тестовыми данными» — форма заполнится; при "
            "желании параметры можно поправить.",
            "Нажать «🔧 Создать / сбросить проект».",
            "Пройти вкладки M1…M8 (на M2 отклики можно заполнить кнопкой "
            "симулятора — «🧪 Заполнить тестовыми (симулятор)»).",
            "Открыть вкладку «🎯 Benchmark» и сравнить найденный рецепт с "
            "известным аналитическим оптимумом полигона.",
        ],
    }


