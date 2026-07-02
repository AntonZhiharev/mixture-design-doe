"""verification/battle_truth.py — КАНОНИЧЕСКАЯ синтетическая истина battle-теста.

Единый источник (без дублей/дрейфа, канон .clinerules) для двух «лабораторий»
боевого теста над составной областью mixture×process (REBUILD_SPEC §8/§13):

  * :func:`build_truth_3comp` — 3-компонентный мир {A,B,C} × {T,P}, 5 откликов
    (strength/gloss/dry_time/price/rho). Используется в фазовом сценарии battle
    (поэтапное раскрытие A,B → +T → +C,P) и в STEP 4/5 (floor/relax).
  * :func:`build_truth_econ` — STEP 6: 4-компонентный мир {A,B,C,D} × {T,P},
    5 откликов (strength/gloss/dry_time/whiteStrength/rho). Цена изделия НЕ
    отдельное свойство — собирается на лету ``price_изд = price_состав(x)·ρ(x)``.

Истина — CUBIC (special-cubic) Шеффе по составу + QUADRATIC RSM по процессу +
полный кросс x·z; модель пайплайна — QUADRATIC Шеффе (намеренная мисспецификация,
остаток ловит GP). Координаты — СОСТАВНЫЕ ``Xc = [x_0..x_{q-1}, z_0..z_{d-1}]``
(mixture-доли Σ=1, process в коде [0,1]).

Оба сюжета — источник истины и для ``tests/unit/test_iteration13_battle.py``
(импортирует отсюда), и для ручного хелпера откликов ``tools/response_helper.py``:
одни и те же коэффициенты, никакого расхождения.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence

import numpy as np

from ..core.schema import ModelSpec, ProjectSchema, VariableBlock
from ..design.block_model import build_model_terms
from .mixture_process_truth import MultiMixtureProcessTruth


# ----------------------------------------------------------------------
# Общий конструктор вектора коэффициентов истины по ИМЕНАМ термов
# ----------------------------------------------------------------------
def coef_from_terms(schema: ProjectSchema,
                    contributions: Mapping[str, float]) -> np.ndarray:
    """Вектор коэффициентов истины (длина == числу термов схемы) по именам термов.

    Ключи ``contributions`` — канонические имена термов (``build_model_terms``):
    линейные ``A``/``T``, парные ``A*B``/``A:T``/``T*P``, квадраты процесса
    ``T^2``, тройные ``A*B*C``. Отсутствующие термы получают коэффициент 0.
    """
    terms = build_model_terms(schema)
    v = np.zeros(terms.p)
    for i, name in enumerate(terms.names):
        v[i] = float(contributions.get(name, 0.0))
    return v


# ======================================================================
# 3-компонентный мир {A,B,C} × {T,P} (фазовый сценарий + STEP 4/5)
# ======================================================================
COMPS_3COMP = ["A", "B", "C"]
# Реальные цены компонентов [усл.ед/кг] (для 3-комп ценовой оси STEP 4/5).
PRICE_3COMP: Dict[str, float] = {"A": 95.0, "B": 200.0, "C": 23.0}

# Разреженная истина 3-комп мира: свойство → {имя терма: коэффициент}.
TRUTH_3COMP: Dict[str, Dict[str, float]] = {
    # strength: синергии состава + process-gating (A:T, C:T) — раскрытие T помогает
    "strength": {"A": 6, "B": 6, "C": 5, "A:T": 5, "C:T": 6, "T^2": -3,
                 "A*B": 8, "A*C": 9, "B*C": 9, "A*B*C": 8},
    # gloss: гряда B*C/A*B*C, вогнутая по P (нужен P вне baseline)
    "gloss": {"A": 3, "B": 6, "C": 6, "P": 7, "P^2": -5,
              "A*B": 7, "B*C": 13, "A*C": 9, "A*B*C": 14},
    # dry_time (min): зависит от состава (A быстрый, B/C медленные), T-gated
    "dry_time": {"A": -1, "B": 3, "C": 3, "T": -4, "C:T": -5, "T^2": 2},
    # price (min): мягкий градиент + отрицательный «bland discount» (внутр. минимум)
    "price": {"A": 2.0, "B": 2.2, "C": 2.4,
              "A*B": -1.5, "A*C": -1.5, "B*C": -1.5},
    # rho = ПЛОТНОСТЬ (полноценный отклик), линейный бленд {A,B,C}
    "rho": {"A": 0.7, "B": 1.0, "C": 1.7},
}


def truth_schema_3comp() -> ProjectSchema:
    """Схема ИСТИНЫ 3-комп мира: CUBIC mixture {A,B,C} + QUADRATIC process {T,P}."""
    mix = VariableBlock.mixture(COMPS_3COMP)
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="cubic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def model_schema_3comp() -> ProjectSchema:
    """Схема МОДЕЛИ пайплайна 3-комп мира: QUADRATIC mixture (остаток ловит GP)."""
    mix = VariableBlock.mixture(COMPS_3COMP)
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def build_truth_3comp(noise_sd: float = 0.0) -> MultiMixtureProcessTruth:
    """Истина 3-комп мира {A,B,C} × {T,P} (5 откликов, включая свойство price)."""
    s = truth_schema_3comp()
    coef_by = {prop: coef_from_terms(s, sparse)
               for prop, sparse in TRUTH_3COMP.items()}
    return MultiMixtureProcessTruth(s, coef_by, noise_sd=float(noise_sd))


def comp_price_3comp(Xc) -> np.ndarray:
    """price_состав [усл.ед/кг] 3-комп мира {A,B,C} (детерминирована, БЕЗ D)."""
    Xc = np.atleast_2d(np.asarray(Xc, float))
    w = np.array([PRICE_3COMP[k] for k in COMPS_3COMP], float)
    return Xc[:, :3] @ w


# ======================================================================
# STEP 6: 4-компонентный мир {A,B,C,D} × {T,P} (ценовая цель + ветка white)
# ======================================================================
COMPS_ECON = ["A", "B", "C", "D"]
# Реальные цены компонентов [усл.ед/кг].
PRICE_ECON: Dict[str, float] = {"A": 95.0, "B": 200.0, "C": 23.0, "D": 315.0}

# Разреженная истина STEP 6: свойство → {имя терма: коэффициент}.
TRUTH_ECON: Dict[str, Dict[str, float]] = {
    # strength (max): B критичен через сильные B-синергии; чистый C слаб
    "strength": {"A": 6, "B": 10, "C": 2, "D": 2,
                 "A:T": 5, "C:T": 3, "T^2": -3,
                 "A*B": 9, "A*C": 5, "B*C": 16, "A*B*C": 12, "B*D": 4},
    # gloss (max/target): B-гряда; низкая база от A/C/P → target=8 требует B
    "gloss": {"A": 3, "B": 7, "C": 3, "D": 2, "P": 6, "P^2": -4,
              "A*B": 7, "B*C": 14, "A*C": 6, "A*B*C": 15},
    # dry_time (min): A быстрый, C медленный → fast держит A; T-гейт
    "dry_time": {"A": -4, "B": 2, "C": 4, "D": 1, "T": -4, "C:T": -5, "T^2": 2},
    # whiteStrength (min): D — отбеливатель; A,B,C белят в ПАРЕ с D (интерьер)
    "whiteStrength": {"A": 5, "B": 6, "C": 3, "D": 1,
                      "C*D": -8, "A*D": -5, "B*D": -4},
    # rho = ПЛОТНОСТЬ (отклик опыта), линейный бленд — множитель цены изделия
    "rho": {"A": 0.7, "B": 1.0, "C": 1.7, "D": 0.6},
}


def truth_schema_econ() -> ProjectSchema:
    """Схема ИСТИНЫ STEP 6: CUBIC mixture {A,B,C,D} + QUADRATIC process {T,P}."""
    mix = VariableBlock.mixture(COMPS_ECON)
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="cubic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def model_schema_econ() -> ProjectSchema:
    """Схема МОДЕЛИ пайплайна STEP 6: QUADRATIC mixture {A,B,C,D} + процесс."""
    mix = VariableBlock.mixture(COMPS_ECON)
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def build_truth_econ(noise_sd: float = 0.0) -> MultiMixtureProcessTruth:
    """Истина STEP 6 {A,B,C,D} × {T,P} (5 откликов; цена собирается отдельно)."""
    s = truth_schema_econ()
    coef_by = {prop: coef_from_terms(s, sparse)
               for prop, sparse in TRUTH_ECON.items()}
    return MultiMixtureProcessTruth(s, coef_by, noise_sd=float(noise_sd))


def comp_price_econ(Xc) -> np.ndarray:
    """price_состав [усл.ед/кг] STEP 6 мира {A,B,C,D} (детерминирована)."""
    Xc = np.atleast_2d(np.asarray(Xc, float))
    w = np.array([PRICE_ECON[k] for k in COMPS_ECON], float)
    return Xc[:, :4] @ w


# ----------------------------------------------------------------------
# Цена изделия по истине: price_изд = price_состав(x)·ρ_truth(x) (§15.6 §3)
# ----------------------------------------------------------------------
def item_price_fn(truth: MultiMixtureProcessTruth,
                  comp_price_fn: Callable[[np.ndarray], np.ndarray]
                  ) -> Callable[[np.ndarray], np.ndarray]:
    """Собрать функцию цены изделия ``Xc → price_состав·ρ_truth`` по истине.

    ρ берётся из БЕЗШУМНОЙ истины (``truth.truths['rho'].true``); используется и
    в эталоне (аналитический оптимум с ценой), и в хелпере откликов.
    """
    def _fn(Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        rho = np.asarray(truth.truths["rho"].true(Xc), float).ravel()
        return np.asarray(comp_price_fn(Xc), float).ravel() * rho

    return _fn


# ----------------------------------------------------------------------
# Реестр «миров» для хелпера откликов (единая точка выбора)
# ----------------------------------------------------------------------
def worlds() -> Dict[str, Dict[str, object]]:
    """Карта доступных миров battle-теста для хелпера/плана.

    Каждый мир: имена компонентов смеси, имена процесс-осей, имена откликов,
    билдер истины, функция цены состава. Значения — фабрики/имена, без побочных
    эффектов до вызова.
    """
    return {
        "econ": {
            "label": "STEP 6 — 4 компонента {A,B,C,D} × {T,P}, 5 свойств "
                     "(strength/gloss/dry_time/whiteStrength/rho) + цена изделия",
            "mixture": list(COMPS_ECON),
            "process": ["T", "P"],
            "responses": list(TRUTH_ECON.keys()),
            "build_truth": build_truth_econ,
            "comp_price_fn": comp_price_econ,
            "prices": dict(PRICE_ECON),
        },
        "3comp": {
            "label": "Фазовый мир — 3 компонента {A,B,C} × {T,P}, 5 свойств "
                     "(strength/gloss/dry_time/price/rho)",
            "mixture": list(COMPS_3COMP),
            "process": ["T", "P"],
            "responses": list(TRUTH_3COMP.keys()),
            "build_truth": build_truth_3comp,
            "comp_price_fn": comp_price_3comp,
            "prices": dict(PRICE_3COMP),
        },
    }
