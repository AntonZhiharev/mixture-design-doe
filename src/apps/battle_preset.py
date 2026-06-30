"""apps/battle_preset.py — «боевой» пресет STEP 6 для pipeline M1–M8.

Назначение: одним вызовом получить ГОТОВЫЙ снимок конфигурации (`cfg_defaults`
для сайдбар-формы) с ИЗВЕСТНОЙ синтетической истиной, перенесённой из самого
широкого этапа battle-теста (`tests/unit/test_iteration13_battle.py`, STEP 6):
4 компонента {A,B,C,D}, реальные цены, отклики strength/gloss/dry_time/
whiteStrength/ρ.

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


def battle_step6_snapshot(name: str = "battle_step6", seed: int = 13,
                          noise_sd: float = 0.2) -> Dict:
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
