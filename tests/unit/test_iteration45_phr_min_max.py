# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 45 — шаг B1 ревизии контракта phr-спеки (UI_REVISION_SPEC,
«План работ», 🔴 блокер): ``min_phr``/``max_phr`` на share-узлах как
CONDITIONAL NARROWING, а не бокс по доле.

Кейс из ревизии (PVC, стабилизаторная группа): у ``PBNK_3355`` складской
лимит 8.0 phr, у ``CPE_135A`` техминимум 3.0 phr. Долевые границы обоих
0.30…0.70; тотал группы плавает. Без B1 спека штатно генерирует
нереализуемые рецепты (PBNK до 10.5 phr при тотале 15, CPE от 1.5 phr при
тотале 5) — и это не «углы», а обычные вершины плана.

Проверяемый канон:

  * golden эффективного потолка доли:
    ``hi_φ(T) = min(0.70, 8/T, 1 − 3/T)`` → 0.40 @T=5, полка 0.70 на
    T∈[10, 8/0.7], 0.5333 @T=15 — функция НЕМОНОТОННА (тест на
    монотонность дал бы ложный отказ);
  * сэмплинг/clip: КАЖДАЯ точка соблюдает phr-лимиты (контроль — та же
    спека без лимитов, где нарушения массовые);
  * окно тотала: интервал тотала сужается до значений, при которых доли
    вообще выполнимы; ``fixed``-тотал вне окна — ошибка конфига;
  * ``phr_intervals`` сужаются лимитами (вход слоя навески iter42);
  * ``encode`` отвергает рецепт вне лимитов, round-trip сохраняется;
  * лимиты входят в ``to_dicts``/``spec_hash``, но спеки БЕЗ лимитов
    сериализуются байт-в-байт как до iter45 (хеши iter35/36 не «уехали»).
"""
import numpy as np
import pytest

from src.design.phr_sampler import PhrSpec

# --- спека кейса: STAB = PBNK (max 8.0 phr) + CPE (min 3.0 phr) --------
_STAB_NODES = [
    {"name": "resin", "mode": "fixed", "value": 100.0},
    {"name": "STAB_total", "mode": "absolute", "lo": 5.0, "hi": 15.0},
    {"name": "PBNK_3355", "mode": "share_of", "of": "STAB_total",
     "lo": 0.30, "hi": 0.70, "max_phr": 8.0},
    {"name": "CPE_135A", "mode": "share_of", "of": "STAB_total",
     "lo": 0.30, "hi": 0.70, "min_phr": 3.0},
]


def _nodes_without_limits():
    out = []
    for d in _STAB_NODES:
        d2 = {k: v for k, v in d.items() if k not in ("min_phr", "max_phr")}
        out.append(d2)
    return out


def _spec(nodes=None, **overrides):
    nodes = [dict(d) for d in (nodes or _STAB_NODES)]
    for name, patch in overrides.items():
        for d in nodes:
            if d["name"] == name:
                d.update(patch)
    return PhrSpec.from_dicts(nodes)


# ----------------------------------------------------------------------
# 1. Golden: эффективный потолок доли немонотонен по тоталу
# ----------------------------------------------------------------------
def test_share_bounds_at_total_golden_non_monotone():
    spec = _spec()
    idx_pbnk = 0                      # порядок членов = порядок узлов спеки

    def hi_pbnk(T):
        return spec.share_bounds_at_total("STAB_total", T)[1][idx_pbnk]

    assert hi_pbnk(5.0) == pytest.approx(0.40, abs=1e-12)      # 1 − 3/5
    assert hi_pbnk(10.0) == pytest.approx(0.70, abs=1e-12)     # свой φᵁ
    assert hi_pbnk(8.0 / 0.7) == pytest.approx(0.70, abs=1e-12)   # полка
    assert hi_pbnk(15.0) == pytest.approx(8.0 / 15.0, abs=1e-12)  # 8/T

    # НЕМОНОТОННОСТЬ: растёт до полки, затем падает
    assert hi_pbnk(10.0) > hi_pbnk(5.0)
    assert hi_pbnk(15.0) < hi_pbnk(10.0)

    # нижняя граница CPE: max(φᴸ, 3/T, 1 − hi₀_PBNK) — при T=5 сильнее
    # собственный техминимум (3/5), при T=15 сильнее ПАРТНЁРСКОЕ сужение:
    # PBNK не может дать больше 8/15, значит CPE обязан дать 1 − 8/15
    lo_cpe = spec.share_bounds_at_total("STAB_total", 5.0)[0][1]
    assert lo_cpe == pytest.approx(0.60, abs=1e-12)            # 3/5
    lo_cpe_15 = spec.share_bounds_at_total("STAB_total", 15.0)[0][1]
    assert lo_cpe_15 == pytest.approx(1.0 - 8.0 / 15.0, abs=1e-12)


def test_share_bounds_rejects_non_group_and_zero_total():
    spec = _spec()
    with pytest.raises(ValueError, match="не является родителем"):
        spec.share_bounds_at_total("PBNK_3355", 10.0)
    with pytest.raises(ValueError, match="должен быть > 0"):
        spec.share_bounds_at_total("STAB_total", 0.0)


# ----------------------------------------------------------------------
# 2. Сэмплинг: лимиты соблюдены в КАЖДОЙ точке (контроль — без лимитов)
# ----------------------------------------------------------------------
def test_sampling_respects_phr_limits_and_control_violates():
    spec = _spec()
    P = spec.decode(spec.sample_z(500, seed=7))
    col = {nm: j for j, nm in enumerate(spec.component_names)}
    pbnk = P[:, col["PBNK_3355"]]
    cpe = P[:, col["CPE_135A"]]
    assert pbnk.max() <= 8.0 + 1e-9
    assert cpe.min() >= 3.0 - 1e-9

    control = _spec(_nodes_without_limits())
    Pc = control.decode(control.sample_z(500, seed=7))
    colc = {nm: j for j, nm in enumerate(control.component_names)}
    assert Pc[:, colc["PBNK_3355"]].max() > 8.0        # лимит нарушается
    assert Pc[:, colc["CPE_135A"]].min() < 3.0         # техминимум нарушается


def test_clip_z_projects_into_limits():
    spec = _spec()
    # заведомо недопустимая точка: тотал 15, доля PBNK на своём φᵁ=0.70
    z = np.zeros(spec.dim_z)
    z[spec.z_names.index("STAB_total")] = 15.0
    z[spec.z_names.index("PBNK_3355")] = 0.70
    z[spec.z_names.index("CPE_135A")] = 0.30
    zc = spec.clip_z(z)
    p = spec.decode(zc)
    col = {nm: j for j, nm in enumerate(spec.component_names)}
    assert p[col["PBNK_3355"]] <= 8.0 + 1e-9
    assert p[col["CPE_135A"]] >= 3.0 - 1e-9
    shares = zc[[spec.z_names.index("PBNK_3355"),
                 spec.z_names.index("CPE_135A")]]
    assert shares.sum() == pytest.approx(1.0, abs=1e-9)
    # идемпотентность проекции сохраняется
    assert np.allclose(spec.clip_z(zc), zc, atol=1e-12)


# ----------------------------------------------------------------------
# 3. Окно тотала группы
# ----------------------------------------------------------------------
def test_total_window_narrows_absolute_axis():
    # тотал заявлен от 2.0 — при T < 3/0.70 техминимум CPE недостижим
    spec = _spec(STAB_total={"lo": 2.0})
    lo_t, hi_t = spec.phr_intervals()["STAB_total"]
    assert lo_t == pytest.approx(3.0 / 0.70, rel=1e-9)
    assert hi_t == pytest.approx(15.0, rel=1e-12)

    zlo, zhi = spec.z_bounds()
    j = spec.z_names.index("STAB_total")
    assert zlo[j] == pytest.approx(3.0 / 0.70, rel=1e-9)

    T = spec.decode(spec.sample_z(300, seed=3)).sum(axis=1)  # не тотал, а Σp
    Z = spec.sample_z(300, seed=3)
    assert Z[:, j].min() >= 3.0 / 0.70 - 1e-9
    assert T.size == 300                                     # sanity


def test_fixed_total_outside_window_is_config_error():
    # тотал 3.0 phr: техминимум CPE 3.0 недостижим (доля ≤ 0.70 ⇒ ≤ 2.1 phr).
    # Диагностика указывает КОНКРЕТНЫЙ узел, а не «окно тотала»: при
    # t_lo == t_hi пер-узловая проверка и окно эквивалентны.
    nodes = [dict(d) for d in _STAB_NODES]
    nodes[1] = {"name": "STAB_total", "mode": "fixed", "value": 3.0}
    with pytest.raises(ValueError, match="CPE_135A.*не пересекаются"):
        PhrSpec.from_dicts(nodes)


def test_non_absolute_total_cannot_be_narrowed():
    # тотал — ratio_to-ось: сузить её пер-точечно вверх по DAG не реализовано,
    # поэтому вместо тихого приближения — явная ошибка конфига (A0.6)
    nodes = [
        {"name": "resin", "mode": "fixed", "value": 100.0},
        {"name": "base", "mode": "absolute", "lo": 10.0, "hi": 20.0},
        {"name": "STAB_total", "mode": "ratio_to", "to": "base",
         "lo": 0.5, "hi": 1.5},
        {"name": "PBNK_3355", "mode": "share_of", "of": "STAB_total",
         "lo": 0.30, "hi": 0.70, "min_phr": 5.0},
        {"name": "CPE_135A", "mode": "share_of", "of": "STAB_total",
         "lo": 0.30, "hi": 0.70},
    ]
    with pytest.raises(ValueError, match="сужение такой оси не поддерживается"):
        PhrSpec.from_dicts(nodes)


def test_empty_limit_intersection_is_config_error():
    with pytest.raises(ValueError, match="не пересекаются"):
        _spec(PBNK_3355={"max_phr": 0.5})        # φᴸ·Tᴸ = 1.5 > 0.5


def test_limits_only_on_share_nodes():
    with pytest.raises(ValueError, match="только для share_of"):
        _spec(STAB_total={"max_phr": 10.0})


# ----------------------------------------------------------------------
# 4. Интервалы phr (вход слоя навески iter42)
# ----------------------------------------------------------------------
def test_phr_intervals_narrowed_by_limits():
    spec = _spec()
    iv = spec.phr_intervals()
    assert iv["PBNK_3355"] == pytest.approx((1.5, 8.0))     # было (1.5, 10.5)
    assert iv["CPE_135A"] == pytest.approx((3.0, 10.5))     # было (1.5, 10.5)

    control = _spec(_nodes_without_limits()).phr_intervals()
    assert control["PBNK_3355"] == pytest.approx((1.5, 10.5))
    assert control["CPE_135A"] == pytest.approx((1.5, 10.5))


# ----------------------------------------------------------------------
# 5. encode: рецепт вне лимитов — ошибка данных; round-trip сохранён
# ----------------------------------------------------------------------
def test_encode_rejects_recipe_outside_limits_and_round_trips():
    spec = _spec()
    col = {nm: j for j, nm in enumerate(spec.component_names)}

    good = np.zeros(spec.q)
    good[col["resin"]] = 100.0
    good[col["PBNK_3355"]] = 5.0
    good[col["CPE_135A"]] = 5.0
    z = spec.encode(good)
    assert np.allclose(spec.decode(z), good, atol=1e-9)

    over = good.copy()
    over[col["PBNK_3355"]] = 9.0                # > max_phr при тотале 14
    with pytest.raises(ValueError, match="max_phr"):
        spec.encode(over)

    under = good.copy()                         # доли в границах, но
    under[col["PBNK_3355"]] = 4.0               # CPE = 2 phr < min_phr 3.0
    under[col["CPE_135A"]] = 2.0                # (φ_CPE = 1/3 ∈ [0.3, 0.7])
    with pytest.raises(ValueError, match="min_phr"):
        spec.encode(under)


def test_quantize_valid_recipe_has_no_violations():
    spec = _spec()
    col = {nm: j for j, nm in enumerate(spec.component_names)}
    p = np.zeros(spec.q)
    p[col["resin"]] = 100.0
    p[col["PBNK_3355"]] = 5.0
    p[col["CPE_135A"]] = 5.0
    rep = spec.quantize_recipe(p, 0.1)
    assert rep.ok, rep.violations


# ----------------------------------------------------------------------
# 6. Сериализация и отпечаток
# ----------------------------------------------------------------------
def test_limits_in_to_dicts_and_hash_stable_without_limits():
    spec = _spec()
    dicts = spec.to_dicts()
    by_name = {d["name"]: d for d in dicts}
    assert by_name["PBNK_3355"]["max_phr"] == 8.0
    assert "min_phr" not in by_name["PBNK_3355"]
    assert by_name["CPE_135A"]["min_phr"] == 3.0
    # round-trip: та же спека и тот же отпечаток
    again = PhrSpec.from_dicts(dicts)
    assert again.to_dicts() == dicts
    assert again.spec_hash() == spec.spec_hash()

    # спека БЕЗ лимитов не приобретает новых ключей ⇒ старые хеши валидны
    control = _spec(_nodes_without_limits())
    for d in control.to_dicts():
        assert "min_phr" not in d and "max_phr" not in d
    assert control.spec_hash() != spec.spec_hash()   # лимиты — часть геометрии
