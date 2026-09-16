# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 103 — добор области с множителем измеримости (OPEN §16.2.1.2).

Находка iter101: после сужения области к кромке дыры (``edge_region`` →
``move_region``) ``propose_seed`` на непустой базе = ``propose_augment`` —
жадный maximin от существующих точек. Те сгущены в измеримой зоне, поэтому
maximin отталкивается от них и уводит ВСЕ точки добора в дыру (gate 2.7–3.2):
новый прибор (``yield``) не меряется вовсе. Тест каскада обходил это
``reuse_existing=False``.

Здесь закрывается задача ядра: ``propose_augment(..., feasibility=)`` —
критерий ``d_min(x) · P(измеримо|x)``, тот же множитель, что у explore-члена
acquisition (iter100). Проверяем:

  * на мире :class:`CliffTruth` после сужения области к кромке добор БЕЗ
    множителя действительно проваливается в дыру, а С множителем — нет
    (доля измеримых точек добора растёт, глубина промаха падает);
  * ``feasibility=None`` — прежнее поведение бит-в-бит; ``P ≡ 1`` — тоже;
  * контракты: неверная длина ``P`` — ``ValueError``; мало измеримых
    кандидатов — ``UserWarning``; ``propose_seed(feasibility=)`` пробрасывает
    множитель; на пустой базе без множителя — прежний путь;
  * ``runner.gate_feasibility_fn`` — множитель без ветки; без суррогата
    гейта — явный отказ (``RuntimeError``), не молчаливое ``≡ 1``;
  * ``runner.edge_box_to_deltas`` — бокс ``edge_region`` (process в КОДЕ)
    → дельты ``move_region`` (process в ФИЗИКЕ, iter102): применение бокса
    по T сужает область и исключает точки вне неё.
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps.mixture_process_runner import MixtureProcessRunner
from src.core.schema import ModelSpec, ProjectSchema, VariableBlock
from src.design.branches import edge_region
from src.design.move_bounds import MOVE_RESTRICT
from src.verification.battle_truth import (CLIFF_RESPONSE, GATE_3COMP,
                                           GATE_THRESHOLD_3COMP, GATED_3COMP,
                                           TornLab, build_truth_3comp_cliff,
                                           model_schema_3comp)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

THR = GATE_THRESHOLD_3COMP
GATED = tuple(GATED_3COMP) + (CLIFF_RESPONSE,)
N_SEED, N_AUG, EDGE_WIDTH = 20, 6, 0.3


def _full(r, X):
    return np.vstack([r._to_full(x) for x in np.atleast_2d(X)])


def _schema_real_T() -> ProjectSchema:
    """3-комп мир с process-осями в РЕАЛЬНЫХ единицах (T °C, P бар)."""
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[150.0, 1.0],
                                 upper=[200.0, 5.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _screened(seed=7, *, real_units=False):
    """Скрининг мира CliffTruth в общих границах: суррогат гейта обучен.

    ``real_units`` — схема раннера с T °C / P бар. Оракул при этом тот же:
    :meth:`_to_full` отдаёт ему process в КОДЕ полной схемы (baseline — тоже
    код), а истина ``TornLab`` определена на коде [0,1].
    """
    truth = build_truth_3comp_cliff()
    lab = TornLab(truth, gated=GATED)
    schema = _schema_real_T() if real_units else model_schema_3comp()
    r = MixtureProcessRunner(schema, lab, baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                             seed=seed, n_restarts=2)
    X0 = np.asarray(r.propose_seed(N_SEED, seed=seed), float)
    Y0 = lab.evaluate(_full(r, X0))
    r.commit_seed(X0, Y0, missing_reasons=lab.reasons(Y0))
    assert GATE_3COMP in r.surrogates
    return truth, lab, r


def _edge_box(r, seed=101):
    cands = r._phase_candidates(2000, seed=seed)
    names = (list(r.current_schema.mixture_names)
             + list(r.current_schema.process_names))
    return edge_region(r.surrogates[GATE_3COMP], THR, cands, names,
                       width=EDGE_WIDTH, margin=0.05)


def _gate_dist(truth, r, X):
    """Расстояние до кромки по ИСТИНЕ (>0 — измеримо) для кандидатов схемы."""
    return truth.gate_true(_full(r, X)) - THR


# ======================================================================
# 1. Живой замер: добор после сужения к кромке — без/с множителем
# ======================================================================
@pytest.fixture(scope="module")
def narrowed():
    """Скрининг → бокс кромки → restrict. Возвращает ДВА множителя:

    ``f_before`` — из суррогата гейта ДО сужения (обучен на всей базе: и
    годные, и провалившиеся точки — как в каскаде iter101, где ``edge_region``
    считается до ``move_region``); ``f_after`` — из суррогата ПОСЛЕ сужения
    (``refit_if_possible`` учит его только на АКТИВНОМ пуле; точки дыры,
    выпавшие из области, из обучения гейта пропадают — см. OPEN §16.2.1.4).
    """
    truth, lab, r = _screened(seed=7)
    box = _edge_box(r)
    f_before = r.gate_feasibility_fn(GATE_3COMP, THR, "ge")
    mix_box = {k: v for k, v in box.items()
               if k in r.current_schema.mixture_names}
    mv = r.move_region(mix_box, intent="edge_neighbourhood")
    assert mv.move_type == MOVE_RESTRICT
    f_after = r.gate_feasibility_fn(GATE_3COMP, THR, "ge")
    return truth, lab, r, f_before, f_after


def _report(tag, truth, r, X):
    d = _gate_dist(truth, r, X)
    ok = int((d >= 0).sum())
    print(f"  [{tag}] measurable {ok}/{len(X)}, gate-thr="
          f"{np.round(d, 2).tolist()}")
    return d, ok


def test_augment_without_feasibility_falls_into_the_hole(narrowed):
    truth, lab, r, f_before, f_after = narrowed
    print("\n[augment after edge restrict]")
    d_plain, ok_plain = _report("plain maximin", truth, r,
                                r.propose_augment(N_AUG, seed=8))
    X_feas = r.propose_augment(N_AUG, seed=8, feasibility=f_before)
    d_feas, ok_feas = _report("x P(feasible), gate GP before restrict",
                              truth, r, X_feas)
    d_aft, ok_aft = _report("x P(feasible), gate GP after restrict", truth, r,
                            r.propose_augment(N_AUG, seed=8,
                                              feasibility=f_after))
    # воспроизводим находку iter101: голый maximin уходит в дыру
    assert ok_plain <= N_AUG // 2
    assert d_plain.min() < -1.0
    # множитель из суррогата на всей базе: все точки измеримы, кромка рядом
    assert ok_feas == N_AUG
    assert d_feas.min() > -0.1
    # множитель из суррогата на активном пуле слабее (дыру он видел хуже),
    # но всё равно лучше голого maximin
    assert ok_aft > ok_plain
    assert d_aft.min() > d_plain.min()
    # форма и допустимость
    assert X_feas.shape == (N_AUG, r.dim)
    np.testing.assert_allclose(X_feas[:, :3].sum(axis=1), 1.0, atol=1e-9)
    mb = r.current_schema.mixture_block()
    assert np.all(X_feas[:, :3] >= np.asarray(mb.lower) - 1e-9)
    assert np.all(X_feas[:, :3] <= np.asarray(mb.upper) + 1e-9)
    # точки не дублируются между собой
    d2 = ((X_feas[:, None, :] - X_feas[None, :, :]) ** 2).sum(-1)
    assert d2[~np.eye(N_AUG, dtype=bool)].min() > 1e-6


def test_gate_surrogate_after_restrict_forgets_the_hole(narrowed):
    """Находка iter103: после restrict суррогат гейта учится на активном
    пуле и теряет провалившиеся точки — точность P(измеримо) на пуле
    кандидатов падает. Фиксируем как факт (OPEN §16.2.1.4), не как норму."""
    truth, lab, r, f_before, f_after = narrowed
    pool = r._phase_candidates(600, 8)
    truth_ok = _gate_dist(truth, r, pool) >= 0
    acc_before = float(np.mean((f_before(pool) >= 0.5) == truth_ok))
    acc_after = float(np.mean((f_after(pool) >= 0.5) == truth_ok))
    n_act = len(r._migrated_points())
    print(f"  gate GP accuracy on pool: before restrict {acc_before:.2f} "
          f"(trained on {len(r.points)}), after {acc_after:.2f} "
          f"(trained on {n_act})")
    assert n_act < len(r.points)
    assert acc_before >= 0.95
    assert acc_after < acc_before


def test_feasible_augment_measures_the_new_instrument(narrowed):
    """Смысл добора этапа 2 каскада: новый прибор должен получить данные."""
    truth, lab, r, f_before, f_after = narrowed
    X = r.propose_augment(N_AUG, seed=9, feasibility=f_before)
    Y = lab.evaluate(_full(r, X))
    yi = r.prop_index[CLIFF_RESPONSE]
    n_yield = int(np.isfinite(Y[:, yi]).sum())
    print(f"  yield measured on {n_yield}/{N_AUG} augment points")
    assert n_yield >= N_AUG - 1


# ======================================================================
# 2. Контракты множителя
# ======================================================================
def test_none_and_unit_feasibility_are_bitwise_previous_behaviour(narrowed):
    truth, lab, r, _, _ = narrowed
    X0 = r.propose_augment(N_AUG, seed=8)
    X1 = r.propose_augment(N_AUG, seed=8, feasibility=None)
    X2 = r.propose_augment(N_AUG, seed=8,
                           feasibility=lambda X: np.ones(len(X)))
    np.testing.assert_allclose(X0, X1)
    np.testing.assert_allclose(X0, X2)


def test_feasibility_contract_errors_and_warning(narrowed):
    truth, lab, r, _, _ = narrowed
    with pytest.raises(ValueError, match="на кандидата"):
        r.propose_augment(3, seed=1, feasibility=lambda X: np.ones(5))
    # почти всё неизмеримо → предупреждение, но план всё же выдан
    with pytest.warns(UserWarning, match="P\\(измеримо\\)"):
        X = r.propose_augment(
            4, seed=1,
            feasibility=lambda X: np.where(np.arange(len(X)) < 2, 1.0, 0.0))
    assert X.shape == (4, r.dim)
    # значения вне [0,1] клипуются, а не ломают критерий
    X = r.propose_augment(3, seed=1,
                          feasibility=lambda X: 5.0 * np.ones(len(X)))
    np.testing.assert_allclose(X, r.propose_augment(3, seed=1))


def test_propose_seed_forwards_feasibility(narrowed):
    truth, lab, r, f, _ = narrowed
    np.testing.assert_allclose(r.propose_seed(N_AUG, seed=8, feasibility=f),
                               r.propose_augment(N_AUG, seed=8, feasibility=f))
    # reuse_existing=False — множитель не применяется (план фазы с нуля)
    np.testing.assert_allclose(
        r.propose_seed(N_AUG, seed=8, reuse_existing=False, feasibility=f),
        r._phase_candidates(N_AUG, 8))


def test_empty_base_paths():
    truth = build_truth_3comp_cliff()
    lab = TornLab(truth, gated=GATED)
    r = MixtureProcessRunner(model_schema_3comp(), lab,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                             seed=3, n_restarts=2)
    # без множителя — прежний путь (первые n кандидатов пула)
    np.testing.assert_allclose(r.propose_augment(5, seed=2),
                               r._phase_candidates(600, 2)[:5])
    # с множителем на пустой базе — самый измеримый кандидат первым,
    # далее maximin×P; результат допустим и без дублей
    p = lambda X: np.asarray(X[:, 0], float)          # «измеримо при большом A»
    X = r.propose_augment(5, seed=2, feasibility=p)
    assert X.shape == (5, 5)
    pool = r._phase_candidates(600, 2)
    assert np.allclose(X[0], pool[int(np.argmax(pool[:, 0]))])
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    assert d2[~np.eye(5, dtype=bool)].min() > 1e-6


def test_gate_feasibility_fn_contract():
    truth = build_truth_3comp_cliff()
    lab = TornLab(truth, gated=GATED)
    r = MixtureProcessRunner(model_schema_3comp(), lab,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                             seed=3, n_restarts=2)
    with pytest.raises(KeyError, match="не среди свойств"):
        r.gate_feasibility_fn("нет_такого", 1.0)
    with pytest.raises(ValueError, match="direction"):
        r.gate_feasibility_fn(GATE_3COMP, THR, "between")
    # суррогата гейта ещё нет — явный отказ, не множитель ≡ 1
    with pytest.raises(RuntimeError, match="не обучен"):
        r.gate_feasibility_fn(GATE_3COMP, THR)
    _, _, r2 = _screened(seed=5)
    f = r2.gate_feasibility_fn(GATE_3COMP, THR, "ge")
    cands = r2._phase_candidates(300, seed=4)
    p = f(cands)
    assert p.shape == (300,) and np.all((p >= 0) & (p <= 1))
    truth_ok = truth.gate_true(_full(r2, cands)) >= THR
    assert np.mean((p >= 0.5) == truth_ok) >= 0.85


# ======================================================================
# 3. Бокс кромки по process-осям → move_region в физике (iter102 открыл)
# ======================================================================
def test_edge_box_to_deltas_converts_process_code_to_physics():
    truth, lab, r = _screened(seed=7, real_units=True)
    box = _edge_box(r)
    assert set(box) == {"A", "B", "C", "T", "P"}
    deltas = r.edge_box_to_deltas(box)
    # mixture — дословно
    for k in ("A", "B", "C"):
        assert deltas[k] == tuple(map(float, box[k]))
    # process — код → физика по текущему блоку: T ∈ [150,200], P ∈ [1,5]
    t_lo, t_hi = deltas["T"]
    assert t_lo == pytest.approx(150.0 + 50.0 * box["T"][0])
    assert t_hi == pytest.approx(150.0 + 50.0 * box["T"][1])
    assert 150.0 - 1e-9 <= t_lo <= t_hi <= 200.0 + 1e-9
    p_lo, _ = deltas["P"]
    assert p_lo == pytest.approx(1.0 + 4.0 * box["P"][0])
    # подмножество осей и отказ на неизвестном имени
    assert set(r.edge_box_to_deltas(box, axes=["T"])) == {"T"}
    with pytest.raises(KeyError, match="отсутствует в боксе"):
        r.edge_box_to_deltas(box, axes=["Q"])
    with pytest.raises(KeyError, match="не среди"):
        r.edge_box_to_deltas({"Q": (0.0, 1.0)})


def test_edge_box_applied_to_T_narrows_region_and_excludes_points():
    truth, lab, r = _screened(seed=7, real_units=True)
    n_active0 = len(r._migrated_points())
    box = _edge_box(r)
    # сужение только по T; если кромка по T широкая (истина гейта вогнута
    # по T, полоса может покрыть почти всю ось) — берём середину ±10 °C,
    # чтобы проверить именно механику применения бокса в физике
    t_lo, t_hi = r.edge_box_to_deltas(box, axes=["T"])["T"]
    if t_hi - t_lo > 30.0:
        mid = 0.5 * (t_lo + t_hi)
        t_lo, t_hi = mid - 10.0, mid + 10.0
    mv = r.move_region({"T": (t_lo, t_hi)}, intent="edge_neighbourhood_T")
    assert mv.move_type == MOVE_RESTRICT
    pb = r.current_schema.process_block()
    assert (pb.lower[0], pb.upper[0]) == (t_lo, t_hi)
    # точки вне [t_lo, t_hi] по ФИЗИКЕ исключены из активного пула (iter102),
    # история цела
    active = r._migrated_points()
    assert len(r.points) == N_SEED
    assert 0 < len(active) < n_active0
    assert all(t_lo - 1e-9 <= p.X["PROCESS"][0] <= t_hi + 1e-9 for p in active)
    # добор с множителем в суженной области — кандидаты держат T в физике
    f = r.gate_feasibility_fn(GATE_3COMP, THR, "ge")
    X = r.propose_augment(4, seed=11, feasibility=f)
    T_real = np.asarray([pb.from_code(row)[0] for row in X[:, 3:]])
    assert np.all((T_real >= t_lo - 1e-9) & (T_real <= t_hi + 1e-9))
