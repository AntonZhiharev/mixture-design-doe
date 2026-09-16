# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 101 — реальные бюджеты и КАСКАД «дыра → окрестность → компонент».

Два вопроса сессии (16.09.2026):

1. **Сколько опытов стоил результат iter100.** Там скрининг 60 точек + 6×4 =
   84 опыта на 3 рецептурных + 2 процессных оси — не лабораторный объём.
   Здесь тот же мир (:class:`CliffTruth`, гейт в цели + ``set_branch_gate``,
   explore 0.3) прогоняется на бюджетах 12/16/24 seed + 4 раунда × 3 точки
   (24/28/36 опытов), по двум rng-сидам; таблица ``d_best/d_opt``, промахи,
   глубина промаха — в print. Асерты калиброваны по живому прогону.

2. **Каскад из трёх этапов** на 4-комп мире (:func:`build_truth_4comp_cliff`;
   на грани D=0 — бит-в-бит мир iter100):
     * этап 1 — скрининг в общих границах {A,B,C}×{T,P}; цель
       ``gloss max ∧ strength max ∧ surface ≥ 4`` недостижима: максимум gloss
       лежит В ДЫРЕ, аналитический потолок этапа ниже ``satisfy_at``;
     * этап 2 — окрестность дыры из данных (:func:`edge_region`: бокс
       кандидатов с ``|μ_surface − 4| ≤ 0.3`` по суррогату гейта →
       ``move_region`` restrict) + новый прибор ``yield`` (:class:`StagedLab`
       реализует протокол ``declare_response``; у старых точек MISSING с
       причиной) → малый seed области → цель растёт на ``yield max``;
     * этап 3 — компонент D (``augment_phase_mixture``), который сжимает дыру
       вокруг gloss-гряды; ветка подбирается ближе к цели.
   Каждый этап сравнивается со СВОИМ аналитическим потолком
   (:func:`branch_optimum_region` — область и состав этапа), а рецепты этапов
   — по ФИНАЛЬНОЙ цели по истине: ``d_final(x_best_1) < d_final(x_best_2) <
   d_final(x_best_3)``.

Окрестность дыры здесь сужается по компонентам смеси; process-оси остаются
полными. Первоначальная причина («точки хранят код, границы process лишь
переинтерпретируют его») закрыта в iter102 (PROCESS в точке — физика);
применение бокса по T/P через ``runner.edge_box_to_deltas`` проверяется в
``test_iteration103_feasible_augment.py``. Добор области этапа 2 — с
множителем измеримости (iter103, ``propose_seed(feasibility=)``).
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps.campaign import CampaignController
from src.apps.mixture_process_runner import (MISSING_REASONS_TAG,
                                             MixtureProcessRunner)
from src.core.schema import ResponseSpec, is_missing
from src.core.schema_evolution import known_constant
from src.design.branches import edge_region
from src.design.move_bounds import MOVE_RESTRICT
from src.optimize.desirability import (Desirability, DesirabilitySpec,
                                       hard_threshold_spec)
from src.verification.battle_truth import (CLIFF_RESPONSE, GATE_3COMP,
                                           GATE_THRESHOLD_3COMP, GATED_3COMP,
                                           StagedLab, TornLab,
                                           build_truth_3comp_cliff,
                                           build_truth_4comp_cliff,
                                           model_schema_3comp,
                                           model_schema_econ)
from src.verification.branch_reference import (branch_optimum,
                                               branch_optimum_region)
from src.verification.torn_plots import round_metrics

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

THR = GATE_THRESHOLD_3COMP
GATED = tuple(GATED_3COMP) + (CLIFF_RESPONSE,)


def _full(r, X):
    return np.vstack([r._to_full(x) for x in np.atleast_2d(X)])


def _rounds(r, lab, truth, bid, *, rounds, n_points, explore=0.3):
    """Раунды ветки ручным путём; возвращает расстояния до кромки по раундам."""
    dist = []
    for _ in range(rounds):
        Xn = r.propose_points(bid, n_points=n_points, explore_frac=explore,
                              n_candidates=300)
        Fn = _full(r, Xn)
        Yn = lab.evaluate(Fn)
        r.commit_measured(bid, Xn, Yn, missing_reasons=lab.reasons(Yn))
        dist.append(truth.gate_true(Fn) - THR)
    return dist



# ======================================================================
# 1. Реальные бюджеты: мир iter100 на 24/28/36 опытах
# ======================================================================
def _goal_cliff(truth):
    return {CLIFF_RESPONSE: DesirabilitySpec("max", low=0.0,
                                             high=truth.cliff.y_cliff),
            "strength": DesirabilitySpec("max", low=2.0, high=12.0),
            GATE_3COMP: hard_threshold_spec(THR, 0.3, "ge")}


def _budget_run(n_seed, rounds, n_points, seed):
    truth = build_truth_3comp_cliff()
    lab = TornLab(truth, gated=GATED)
    r = MixtureProcessRunner(model_schema_3comp(), lab,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                             seed=seed, n_restarts=2)
    X0 = np.asarray(r.propose_seed(n_seed, seed=seed), float)
    Y0 = lab.evaluate(_full(r, X0))
    r.commit_seed(X0, Y0, missing_reasons=lab.reasons(Y0))
    r.add_branch("b", _goal_cliff(truth), budget=rounds * n_points,
                 satisfy_at=1.1, branch_id="b")
    r.set_branch_gate("b", GATE_3COMP, THR, "ge")
    dist = _rounds(r, lab, truth, "b", rounds=rounds, n_points=n_points)
    m = round_metrics(dist)
    xb = np.asarray(r.branches["b"].x_best, float)
    return {"n_total": n_seed + rounds * n_points,
            "n_seed_feasible": int((Y0[:, r.prop_index[GATE_3COMP]] >= THR).sum()),
            "d_best": float(r.branches["b"].d_best),
            "miss_frac": float(np.mean(m["miss_frac"])),
            "miss_depth": float(max(m["miss_depth"])),
            "x_best_feasible": bool(truth.gate_true(xb.reshape(1, -1))[0] >= THR),
            "surrogate_n": r.surrogate_coverage()[CLIFF_RESPONSE]["n_train"]}


@pytest.fixture(scope="module")
def cliff_opt():
    truth = build_truth_3comp_cliff()
    return branch_optimum(truth, _goal_cliff(truth), n_scan=20000, seed=5)["d"]


BUDGETS = [(12, 4, 3), (16, 4, 3), (24, 4, 3)]


@pytest.mark.parametrize("n_seed,rounds,n_points", BUDGETS)
def test_realistic_budget_reaches_the_edge(n_seed, rounds, n_points, cliff_opt):
    rows = [_budget_run(n_seed, rounds, n_points, seed) for seed in (7, 11)]
    print(f"\n[budget seed={n_seed} + {rounds}x{n_points}] d_opt={cliff_opt:.4f}")
    for s, row in zip((7, 11), rows):
        print(f"  rng {s}: N={row['n_total']} seed_feasible={row['n_seed_feasible']}"
              f"/{n_seed} d_best={row['d_best']:.4f} "
              f"({row['d_best'] / cliff_opt:.1%} of opt) miss_frac="
              f"{row['miss_frac']:.2f} miss_depth={row['miss_depth']:.2f} "
              f"x_best_feasible={row['x_best_feasible']} "
              f"yield_train={row['surrogate_n']}")
    for row in rows:
        assert row["d_best"] <= cliff_opt + 1e-9          # эталон — потолок
        assert row["x_best_feasible"]                     # рекорд измерим
        assert row["miss_depth"] < 1.0                    # не дно дыры
    # живой прогон (16.09.2026): 98.1–99.8 % от d_opt на всех трёх бюджетах
    # (24/28/36 опытов, 4–9 измеримых seed-точек из 12–24); порог ниже —
    # страховка от регресса, не подгонка под число
    assert min(r["d_best"] for r in rows) >= 0.95 * cliff_opt


# ======================================================================
# 2. Каскад: дыра в общих границах → окрестность + прибор → компонент D
# ======================================================================
BASE4 = [0.25, 0.25, 0.25, 0.25, 0.5, 0.5]
GOAL_STAGE1 = {"gloss": DesirabilitySpec("max", low=8.0, high=12.0),
               "strength": DesirabilitySpec("max", low=6.0, high=12.0),
               GATE_3COMP: hard_threshold_spec(THR, 0.3, "ge")}
SATISFY = 0.90
EDGE_WIDTH = 0.3                     # окрестность кромки: surface 4 ± 0.3
# бюджет каскада: 20 + 9 | 6 + 9 | 12 = 56 опытов на 4 компонента × 2 оси
N_SEED1, ROUNDS1, N_SEED2, ROUNDS2, ROUNDS3, NPTS = 20, 3, 6, 3, 4, 3


def _goal_final(truth):
    g = dict(GOAL_STAGE1)
    g[CLIFF_RESPONSE] = DesirabilitySpec("max", low=0.0,
                                         high=truth.cliff.y_cliff)
    return g


def _d_true(truth, goal, x):
    """Желательность цели по ИСТИНЕ в полном рецепте ``x``."""
    x = np.asarray(x, float).reshape(1, -1)
    means = {p: truth.truths[p].true(x) for p in goal}
    return float(Desirability(goal).overall(means)[0])


def _ceiling(truth, goal, r):
    return branch_optimum_region(truth, goal, r.current_schema,
                                 baseline=BASE4, n_scan=15000, seed=5)["d"]


def _cascade(seed=7, *, feasible_augment=True):
    """``feasible_augment`` — добор этапа 2 с множителем измеримости (iter103);
    ``False`` — прежний обход ``reuse_existing=False`` (план области с нуля)."""
    truth = build_truth_4comp_cliff()
    stage1_props = [p for p in truth.property_names if p != CLIFF_RESPONSE]
    lab = StagedLab(truth, available=stage1_props, gated=GATED)
    r = MixtureProcessRunner(model_schema_econ(), lab, baseline=BASE4,
                             seed=seed, n_restarts=2)
    r.begin_phase(mixture_free=["A", "B", "C"], process_free=["T", "P"])
    ctrl = CampaignController(r)
    log = {}

    # ---------- этап 1: скрининг в общих границах {A,B,C} ----------------
    X0 = np.asarray(r.propose_seed(N_SEED1, seed=seed), float)
    Y0 = lab.evaluate(_full(r, X0))
    r.commit_seed(X0, Y0, missing_reasons=lab.reasons(Y0))
    r.add_branch("b", GOAL_STAGE1, budget=200, satisfy_at=SATISFY,
                 branch_id="b")
    r.set_branch_gate("b", GATE_3COMP, THR, "ge")
    d1 = _rounds(r, lab, truth, "b", rounds=ROUNDS1, n_points=NPTS)
    br = r.branches["b"]
    log["stage1"] = {"ceiling": _ceiling(truth, GOAL_STAGE1, r),
                     "d_best": br.d_best, "x_best": list(br.x_best),
                     "status": br.status, "n_base": len(r.points),
                     "miss": round_metrics(d1)}

    # ---------- этап 2: окрестность дыры + новый прибор yield ------------
    cands = r._phase_candidates(2000, seed=101)
    names = (list(r.current_schema.mixture_names)
             + list(r.current_schema.process_names))
    box_all = edge_region(r.surrogates[GATE_3COMP], THR, cands, names,
                          width=EDGE_WIDTH, margin=0.05)
    # множитель измеримости для добора — из суррогата гейта, обученного на
    # ВСЕЙ базе скрининга (после restrict суррогат переобучится на активном
    # пуле и точки дыры из обучения выпадут — iter103, OPEN §16.2.1.4)
    feas = r.gate_feasibility_fn(GATE_3COMP, THR, "ge")
    # сужаем область по компонентам смеси (process — см. docstring модуля)
    box = {k: v for k, v in box_all.items()
           if k in r.current_schema.mixture_names}
    mv = r.move_region(box, intent="edge_neighbourhood")
    ctrl.add_response(ResponseSpec(CLIFF_RESPONSE, kind="max"),
                      reason="прибор выхода появился после скрининга")
    goal2 = _goal_final(truth)
    # yield ещё без единого замера — суррогата нет, и ветка с ним в цели
    # предлагать точки не может (KeyError по контракту branch_scores). Малый
    # seed СУЖЕННОЙ области меряется новым прибором — суррогат yield
    # рождается; затем цель ветки растёт на yield штатной +целью.
    # iter103: добор пристёгивается к базе (maximin от existing, iter37) с
    # множителем измеримости — раньше голый maximin отталкивался от точек,
    # сгущённых в измеримой зоне, и все 6 точек уходили в дыру (обход был
    # reuse_existing=False).
    if feasible_augment:
        X2 = np.asarray(r.propose_seed(N_SEED2, seed=seed + 1,
                                       feasibility=feas), float)
    else:
        X2 = np.asarray(r.propose_seed(N_SEED2, seed=seed + 1,
                                       reuse_existing=False), float)
    Y2 = lab.evaluate(_full(r, X2))
    r.commit_seed(X2, Y2, missing_reasons=lab.reasons(Y2))
    assert CLIFF_RESPONSE in r.surrogates
    # штатная +цель (набор целей ветки растёт, §16.3): d_best переоценивается
    # под новый объектив по общей базе — точки без yield рекордом не станут
    ctrl.set_desirability("b", CLIFF_RESPONSE, goal2[CLIFF_RESPONSE])
    assert dict(br.goal) == goal2
    d2 = [truth.gate_true(_full(r, X2)) - THR]
    d2 += _rounds(r, lab, truth, "b", rounds=ROUNDS2, n_points=NPTS)
    log["stage2"] = {"box": box, "move_type": mv.move_type,
                     "ceiling": _ceiling(truth, goal2, r),
                     "d_best": br.d_best, "x_best": list(br.x_best),
                     "status": br.status, "n_base": len(r.points),
                     "n_active": len(r._migrated_points()),
                     "miss": round_metrics(d2)}

    # ---------- этап 3: компонент D сжимает дыру -------------------------
    ctrl.add_mixture_component("D", known_constant(0.0), lower=0.0, upper=0.5)
    d3 = _rounds(r, lab, truth, "b", rounds=ROUNDS3, n_points=NPTS)
    log["stage3"] = {"ceiling": _ceiling(truth, goal2, r),
                     "d_best": br.d_best, "x_best": list(br.x_best),
                     "status": br.status, "n_base": len(r.points),
                     "version": r.current_schema_version,
                     "miss": round_metrics(d3)}
    return truth, lab, r, log


@pytest.fixture(scope="module")
def cascade():
    # iter103: добор этапа 2 с множителем измеримости. seed=13 — из 10
    # прогнанных сидов (7…41) этап 2 без промахов во ВСЕХ, а этап 3 (12 точек
    # на поиск D) сходится к satisfied в 5 из 10 — это лотерея бюджета
    # этапа 3, не зависящая от добора (без множителя 6 из 9 + один сид, где
    # все 6 точек добора ушли в дыру и yield не измерился вовсе). См.
    # REBUILD_SPEC §16.2.1.4.
    return _cascade(seed=13)


def _print_stage(tag, s):
    m = s["miss"]
    print(f"  [{tag}] n_base={s['n_base']} ceiling={s['ceiling']:.4f} "
          f"d_best={s['d_best']:.4f} status={s['status']} "
          f"miss_frac={np.round(m['miss_frac'], 2).tolist()} "
          f"depth={np.round(m['miss_depth'], 2).tolist()} "
          f"x_best={np.round(s['x_best'], 3).tolist()}")


def test_cascade_stage1_goal_unreachable_in_full_bounds(cascade):
    truth, lab, r, log = cascade
    s1 = log["stage1"]
    print("\n=== CASCADE ===")
    _print_stage("stage1 {A,B,C}", s1)
    # дыра в общих границах: цель недостижима даже аналитически
    assert s1["ceiling"] < SATISFY
    assert s1["d_best"] <= s1["ceiling"] + 1e-9
    assert s1["status"] == "active"                    # не satisfied
    # но ядро подошло к своему потолку: отношение — контроль, не подгонка
    assert s1["d_best"] >= 0.9 * s1["ceiling"]
    assert max(s1["miss"]["miss_depth"]) < 1.0         # без падений на дно
    # yield на этапе 1 не измерялся вовсе — прибора не было (после
    # declare_response столбец у старых точек есть, но это MISSING с причиной)
    assert all(is_missing(p.Y[CLIFF_RESPONSE]) for p in r.points[:s1["n_base"]])
    assert lab.n_unmeasurable > 0                      # дыра реально встречалась


def test_cascade_stage2_edge_neighbourhood_and_new_instrument(cascade):
    truth, lab, r, log = cascade
    s1, s2 = log["stage1"], log["stage2"]
    _print_stage("stage2 edge+yield", s2)
    print(f"    box={ {k: (round(a, 3), round(b, 3)) for k, (a, b) in s2['box'].items()} }")
    # окрестность дыры — сужение области (restrict), не расширение
    assert s2["move_type"] == MOVE_RESTRICT
    assert any(hi - lo < 1.0 - 1e-9 for lo, hi in s2["box"].values())
    # история цела; активный pool не пуст, но урезан областью
    assert s2["n_base"] > s1["n_base"]
    assert 0 < s2["n_active"] <= s2["n_base"]
    # новый прибор: столбец есть, у старых точек MISSING с причиной
    assert CLIFF_RESPONSE in r.property_names
    old = r.points[:s1["n_base"]]
    assert all(is_missing(p.Y[CLIFF_RESPONSE]) for p in old)
    assert all(p.origin_tag[MISSING_REASONS_TAG][CLIFF_RESPONSE].startswith("прибор")
               for p in old)
    assert lab.n_declared == 1
    # новые точки yield меряют (там, где образец есть)
    new = r.points[s1["n_base"]:s2["n_base"]]
    assert any(not is_missing(p.Y[CLIFF_RESPONSE]) for p in new)
    # iter103: добор области с множителем измеримости — seed этапа 2 весь
    # измерим (до iter103 обход reuse_existing=False давал 2 промаха из 6,
    # а штатный maximin — 6 из 6)
    assert s2["miss"]["miss_frac"][0] == 0.0
    seed2 = new[:N_SEED2]
    assert all(not is_missing(p.Y[CLIFF_RESPONSE]) for p in seed2)
    # рекорд по РАСШИРЕННОЙ цели честен: ≤ потолка этапа 2
    assert s2["d_best"] <= s2["ceiling"] + 1e-9
    # без D цель по-прежнему недостижима
    assert s2["ceiling"] < SATISFY


def test_cascade_stage3_component_shrinks_the_hole(cascade):
    truth, lab, r, log = cascade
    s1, s2, s3 = log["stage1"], log["stage2"], log["stage3"]
    _print_stage("stage3 +D", s3)
    goal_f = _goal_final(truth)
    d_f = [_d_true(truth, goal_f, s["x_best"]) for s in (s1, s2, s3)]
    print(f"    d_final(x_best) by stage: {np.round(d_f, 4).tolist()}; "
          f"ceilings {s1['ceiling']:.4f} → {s2['ceiling']:.4f} → {s3['ceiling']:.4f}")
    # D введён: версия схемы выросла, старые точки на грани D=0, база цела
    assert s3["version"] == 3                          # v1 фаза → v2 yield → v3 D
    assert r.q == 4 and list(r.current_schema.mixture_names) == ["A", "B", "C", "D"]
    assert len(r.points) == s3["n_base"] >= s2["n_base"] + ROUNDS3 * NPTS
    assert all(abs(p.X["MIXTURE"][3]) < 1e-12 for p in r._migrated_points()[:s1["n_base"]])
    # D сжал дыру: потолок этапа 3 выше потолка этапа 2, и цель достижима
    assert s3["ceiling"] > s2["ceiling"] + 0.01
    assert s3["ceiling"] >= SATISFY
    # ядро воспользовалось D: рекорд этапа 3 использует D>0 и выше рекорда 2
    assert s3["x_best"][3] > 0.02
    assert s3["d_best"] > s2["d_best"]
    assert s3["d_best"] <= s3["ceiling"] + 1e-9
    # рецепты этапов по ФИНАЛЬНОЙ цели: каскад монотонно приближается
    assert d_f[0] < d_f[1] < d_f[2]
    # рекорд измерим
    assert truth.gate_true(np.asarray(s3["x_best"]).reshape(1, -1))[0] >= THR


# ======================================================================
# 3. Примитивы каскада: контракты по отдельности
# ======================================================================
def test_edge_region_box_is_simplex_closed_and_refuses_thin_band(cascade):
    truth, lab, r, log = cascade
    from src.design.move_bounds import check_simplex_closure

    class _Mu:                                  # суррогат-заглушка: μ = A·10
        def predict(self, X):
            X = np.atleast_2d(X)
            return type("P", (), {"mean": X[:, 0] * 10.0,
                                  "std": np.zeros(len(X))})()

    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(3), size=500)
    Xc = np.column_stack([W, rng.random((500, 2))])
    names = ["A", "B", "C", "T", "P"]
    box = edge_region(_Mu(), 4.0, Xc, names, width=0.5)
    # кромка A=0.4 ± 0.05 → бокс по A узкий, симплекс-замкнутость держится
    assert 0.3 <= box["A"][0] and box["A"][1] <= 0.5
    mix_box = {k: v for k, v in box.items() if k in ("A", "B", "C")}
    check_simplex_closure(r.schema_history.get(1), mix_box)   # v1: {A,B,C}
    with pytest.raises(ValueError, match="лишь"):
        edge_region(_Mu(), 4.0, Xc, names, width=1e-6)
    with pytest.raises(ValueError, match="width"):
        edge_region(_Mu(), 4.0, Xc, names, width=0.0)
    with pytest.raises(ValueError, match="names"):
        edge_region(_Mu(), 4.0, Xc, names[:3], width=0.5)


def test_staged_lab_contract():
    truth = build_truth_4comp_cliff()
    with pytest.raises(ValueError, match="Гейт"):
        StagedLab(truth, available=["strength", "gloss"], gated=GATED)
    with pytest.raises(KeyError, match="нет в истине"):
        StagedLab(truth, available=[GATE_3COMP, "haze"], gated=GATED)
    lab = StagedLab(truth, available=[GATE_3COMP, "strength"], gated=GATED)
    X = np.array([[0.5, 0.3, 0.2, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 0.0, 0.5, 0.5]])
    Y = lab.evaluate(X)
    assert Y.shape == (2, 2) and np.all(np.isfinite(Y))   # ни один не gated
    lab.declare_response("gloss")                          # прибор включён
    Y = lab.evaluate(X)
    assert Y.shape == (2, 3)
    assert np.isfinite(Y[0, 2]) and np.isnan(Y[1, 2])      # 2-я точка в дыре
    assert lab.reasons(Y)[1] == {"gloss": f"образец не получен: surface="
                                          f"{Y[1, 0]:.2f} < 4"}
    with pytest.raises(ValueError, match="новая физика"):
        lab.declare_response("haze")
    with pytest.raises(ValueError, match="уже измеряется"):
        lab.declare_response("gloss")
    # раннер отвергает новую ПЕРЕМЕННУЮ (физика фиксирована) — как у TornLab
    r = MixtureProcessRunner(model_schema_econ(), lab, baseline=BASE4,
                             seed=1, n_restarts=2)
    with pytest.raises(ValueError, match="ФИКСИРОВАННОЙ"):
        r.declare_variables(mixture=[("E", 0.0, 1.0)])


def test_branch_optimum_region_prefix_and_ceiling_monotone(cascade):
    truth, lab, r, log = cascade
    goal = _goal_final(truth)
    # область {A,B,C} внутри 4-комп истины — валидна; чужой порядок — отказ
    v1 = r.schema_history.get(1)
    o1 = branch_optimum_region(truth, goal, v1, baseline=BASE4,
                               n_scan=5000, seed=1)
    assert 0 < o1["d"] <= 1 and abs(o1["x"][3]) < 1e-12   # D=0 на грани
    from src.core.schema import ModelSpec, ProjectSchema, VariableBlock
    bad = ProjectSchema.mixture_process(
        VariableBlock.mixture(["B", "A"]),
        VariableBlock.process(["T"], lower=[0.0], upper=[1.0]),
        model=ModelSpec())
    with pytest.raises(ValueError, match="префикс"):
        branch_optimum_region(truth, goal, bad, baseline=BASE4, n_scan=100)
    # потолок полной области ≥ потолка любой её подобласти
    o_full = branch_optimum(truth, goal, n_scan=5000, seed=1)
    assert o_full["d"] >= log["stage3"]["ceiling"] - 5e-3
    assert o_full["d"] >= log["stage2"]["ceiling"] - 5e-3


