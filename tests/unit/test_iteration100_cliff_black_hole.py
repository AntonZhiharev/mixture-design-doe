# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 100 — ОБРЫВ: экспоненциальный рост отклика к кромке измеримости.

Вопрос сессии (16.09.2026): если полезный отклик растёт экспоненциально при
приближении к границе области определения и за ней ОБРЫВАЕТСЯ (образец не
получен, MISSING), сможет ли ядро «обтесать» зону роста и аккуратно
подобраться к обрыву — и не станет ли обрыв «чёрной дырой» для GP из-за
максимальной неопределённости за кромкой (суррогат учится только на
измеримой стороне и за ней экстраполирует рост: μ↑ и σ↑ одновременно).

Мир — :class:`CliffTruth` (battle-истина 3-комп мира + гейт ``surface`` +
неполиномиальный ``yield = exp(g_max − surface) − 1`` при ``surface ≥ 4``,
иначе 0). Аналитический оптимум цели «yield max ∧ strength max ∧ surface ≥ 4»
по построению лежит РОВНО на кромке (проверяется ``branch_optimum``).

Что фиксируется (все числа — живой прогон, см. print):
  1. ЭТАЛОН: оптимум на кромке (``surface = 4.000``, ``yield = y_cliff``);
     эталон с мультистартом — строгая верхняя граница измеренного ``d_best``
     при нулевом шуме (iter99 давал 0.878 > 0.877 из-за одного старта).
  2. ПОДКРАДЫВАНИЕ (гейт в цели, explore 0.3): предложения ложатся на кромку
     с обеих сторон (|g−4| ≲ 0.05), промахи МЕЛКИЕ (глубина ≤ 0.1, это не
     дыра, а кромка), ``d_best → d_opt``, ``x_best`` измерим.
  3. ЧЁРНАЯ ДЫРА (explore 0.6 без учёта измеримости): explore-слоты
     систематически уходят в ГЛУБИНУ дыры (глубина > 1 при пороге 4) —
     σ зависимых откликов там максимальна. Лечение (iter100):
     ``set_branch_gate`` → explore-член × ``P(измеримо|x)`` из суррогата
     гейта — промахи исчезают, ``d_best`` тот же, что при explore 0.3.
  4. КАНОН §16.2.1 остаётся: множитель измеримости чинит explore, но НЕ
     заменяет гейт в цели — без гейта в цели argmax сам идёт в дыру.
  5. ГРАФИКА (``output/iter100/*.png``): star-проекция «цветик» (поле σ
     суррогата yield + истинная дыра | точки seed/раундов/x_best/оптимум) и
     срез «yield vs расстояние до кромки» (истина, μ±2σ, точки раундов) —
     для сценариев «дыра» и «с множителем измеримости».
"""
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.design.branches import branch_scores, gate_feasibility
from src.optimize.desirability import DesirabilitySpec, hard_threshold_spec
from src.verification.battle_truth import (CLIFF_RESPONSE, GATE_3COMP,
                                           GATE_THRESHOLD_3COMP, GATED_3COMP,
                                           TornLab, build_truth_3comp_cliff,
                                           model_schema_3comp)
from src.verification.branch_reference import branch_optimum
from src.verification.torn_plots import (cliff_slice, round_metrics,
                                         star_projection)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASELINE = [1 / 3, 1 / 3, 1 / 3, 0.5, 0.5]
NAMES = ["A", "B", "C", "T", "P"]
THR = GATE_THRESHOLD_3COMP
GATED = tuple(GATED_3COMP) + (CLIFF_RESPONSE,)
N_SEED = 60
ROUNDS = 6
N_POINTS = 4
OUT_DIR = Path(__file__).resolve().parents[2] / "output" / "iter100"


def _goal(truth, with_gate=True):
    g = {CLIFF_RESPONSE: DesirabilitySpec("max", low=0.0,
                                          high=truth.cliff.y_cliff),
         "strength": DesirabilitySpec("max", low=2.0, high=12.0)}
    if with_gate:
        g[GATE_3COMP] = hard_threshold_spec(THR, 0.3, "ge")
    return g


def _full(r, X):
    return np.vstack([r._to_full(x) for x in np.atleast_2d(X)])


def _campaign(*, explore, feasibility, gate_in_goal=True, seed=7):
    """Скрининг → ветка → ROUNDS раундов; возвращает всё для метрик/графики."""
    truth = build_truth_3comp_cliff()
    lab = TornLab(truth, gated=GATED)
    r = MixtureProcessRunner(model_schema_3comp(), lab, baseline=BASELINE,
                             seed=seed, n_restarts=2)
    X0 = np.asarray(r.propose_seed(N_SEED, seed=seed), float)
    Y0 = lab.evaluate(_full(r, X0))
    r.commit_seed(X0, Y0, missing_reasons=lab.reasons(Y0))
    r.add_branch("b", _goal(truth, gate_in_goal), budget=ROUNDS * N_POINTS,
                 satisfy_at=1.1, branch_id="b")
    if feasibility:
        r.set_branch_gate("b", GATE_3COMP, THR, "ge")
    rounds = []
    for _ in range(ROUNDS):
        Xn = r.propose_points("b", n_points=N_POINTS, explore_frac=explore,
                              n_candidates=400)
        Fn = _full(r, Xn)
        Yn = lab.evaluate(Fn)
        r.commit_measured("b", Xn, Yn, missing_reasons=lab.reasons(Yn))
        rounds.append({"X": Fn, "g": truth.gate_true(Fn), "Y": Yn})
    dist_rounds = [rd["g"] - THR for rd in rounds]
    return {"truth": truth, "lab": lab, "runner": r, "X0": _full(r, X0),
            "Y0": Y0, "g0": truth.gate_true(_full(r, X0)), "rounds": rounds,
            "metrics": round_metrics(dist_rounds)}


def _print(tag, c):
    m = c["metrics"]
    print(f"\n[{tag}] miss_frac={np.round(m['miss_frac'], 2).tolist()} "
          f"depth={np.round(m['miss_depth'], 2).tolist()} "
          f"nearest|g-4|={np.round(m['nearest_abs'], 3).tolist()} "
          f"d_best={c['runner'].branches['b'].d_best:.4f}")


# ======================================================================
# 1. Эталон: оптимум лежит на кромке; мультистарт — верхняя граница
# ======================================================================
def test_cliff_truth_optimum_is_on_the_edge():
    truth = build_truth_3comp_cliff()
    Xc = np.array([[0.5, 0.3, 0.2, 0.5, 0.5], [0.0, 0.0, 1.0, 0.5, 0.5]])
    y = truth.true(Xc)
    gi, yi = (truth.property_names.index(GATE_3COMP),
              truth.property_names.index(CLIFF_RESPONSE))
    assert y[0, gi] >= THR and y[0, yi] > 0            # измеримая точка
    assert y[1, gi] < THR and y[1, yi] == 0.0           # за кромкой — обрыв
    # ближе к кромке → выше (экспонента)
    d = np.linspace(0.0, 2.0, 5)
    ys = np.exp(truth.cliff.k * (truth.cliff.g_max - (THR + d))) - 1
    assert np.all(np.diff(ys) < 0)

    opt = branch_optimum(truth, _goal(truth), n_scan=20000, seed=5)
    print(f"\nd_opt={opt['d']:.4f} x={np.round(opt['x'], 3).tolist()} "
          f"surface={opt['y'][GATE_3COMP]:.4f} yield={opt['y'][CLIFF_RESPONSE]:.3f}")
    assert abs(opt["y"][GATE_3COMP] - THR) < 5e-3       # РОВНО на кромке
    assert opt["y"][CLIFF_RESPONSE] == pytest.approx(truth.cliff.y_cliff, rel=5e-3)
    # мультистарт не хуже одного старта (эталон — верхняя граница)
    opt1 = branch_optimum(truth, _goal(truth), n_scan=20000, seed=5, n_starts=1)
    assert opt["d"] >= opt1["d"] - 1e-12


# ======================================================================
# 2. Подкрадывание к обрыву (гейт в цели, умеренный explore)
# ======================================================================
def test_creeps_to_the_edge_with_shallow_misses():
    c = _campaign(explore=0.3, feasibility=False)
    _print("creep e=0.3", c)
    m, r, truth = c["metrics"], c["runner"], c["truth"]
    d_opt = branch_optimum(truth, _goal(truth), n_scan=20000, seed=5)["d"]
    print(f"  d_opt={d_opt:.4f}")
    # каждый раунд подошёл к кромке ближе 0.06 по гейту
    assert max(m["nearest_abs"]) < 0.06
    # промахи есть, но МЕЛКИЕ: это кромка, а не дыра
    assert max(m["miss_depth"]) < 0.1
    # рекорд честный, монотонный и близок к аналитическому
    hist = [h["d_best"] for h in r.branches["b"].history]
    assert all(b >= a - 1e-12 for a, b in zip(hist, hist[1:]))
    assert r.branches["b"].d_best <= d_opt + 1e-9
    assert r.branches["b"].d_best >= 0.97 * d_opt
    xb = np.asarray(r.branches["b"].x_best, float)
    assert truth.gate_true(xb.reshape(1, -1))[0] >= THR   # x_best измерим
    # база: у каждого пропуска причина называет гейт
    assert all(GATE_3COMP in row["reason"] for row in r.missing_report())


# ======================================================================
# 3. Чёрная дыра при сильном explore — и её лечение множителем измеримости
# ======================================================================
def test_black_hole_at_high_explore_and_feasibility_fix():
    hole = _campaign(explore=0.6, feasibility=False)
    fix = _campaign(explore=0.6, feasibility=True)
    _print("hole e=0.6", hole)
    _print("fix  e=0.6", fix)
    mh, mf = hole["metrics"], fix["metrics"]
    # дыра: explore-слоты (3 из 4) каждый раунд уходят ГЛУБОКО за кромку
    assert np.mean(mh["miss_frac"]) >= 0.5
    assert np.median(mh["miss_depth"]) > 1.0
    # лечение: промахи редки и мелкие, ни одного глубокого
    assert np.mean(mf["miss_frac"]) <= 0.15
    assert max(mf["miss_depth"]) < 0.1
    # рекорд не пострадал (exploit-слот не трогали)
    assert fix["runner"].branches["b"].d_best >= \
        hole["runner"].branches["b"].d_best - 1e-9
    # состояние гейта переживает save/load
    r2 = cst.runner_from_state(cst.runner_to_state(fix["runner"]))
    assert r2.branch_gate("b") == {"response": GATE_3COMP, "threshold": THR,
                                   "direction": "ge"}
    assert hole["runner"].branch_gate("b") is None


def test_feasibility_multiplier_only_touches_explore_term():
    c = _campaign(explore=0.3, feasibility=False)
    r = c["runner"]
    cands = r._phase_candidates(300, seed=3)
    p = gate_feasibility(r.surrogates[GATE_3COMP], THR, "ge")(cands)
    assert p.shape == (300,) and np.all((p >= 0) & (p <= 1))
    # вероятность согласована с истиной измеримости
    truth_ok = c["lab"].feasible(cands)
    assert p[truth_ok].mean() > 0.8 and p[~truth_ok].mean() < 0.2
    goal = r.branches["b"].goal
    a0, d0, s0 = branch_scores(r.surrogates, goal, cands, explore_frac=0.5)
    a1, d1, s1 = branch_scores(r.surrogates, goal, cands, explore_frac=0.5,
                               feasibility=lambda X: p)
    assert np.allclose(d0, d1) and np.allclose(s0, s1)   # exploit и σ те же
    assert np.all(a1 <= a0 + 1e-12)                       # explore лишь гасится
    # explore_frac=0 → множитель не влияет вовсе
    a2, _, _ = branch_scores(r.surrogates, goal, cands, explore_frac=0.0,
                             feasibility=lambda X: p)
    assert np.allclose(a2, d0)
    with pytest.raises(ValueError, match="на кандидата"):
        branch_scores(r.surrogates, goal, cands, feasibility=lambda X: p[:5])
    with pytest.raises(KeyError, match="не среди свойств"):
        r.set_branch_gate("b", "нет_такого", 1.0)
    with pytest.raises(ValueError, match="direction"):
        r.set_branch_gate("b", GATE_3COMP, 1.0, "between")


# ======================================================================
# 4. Канон: множитель НЕ заменяет гейт в цели ветки
# ======================================================================
def test_feasibility_does_not_replace_gate_in_goal():
    c = _campaign(explore=0.3, feasibility=True, gate_in_goal=False)
    _print("nogate+feas", c)
    # explore чинится (промахи мелкие), но exploit-argmax без гейта в цели
    # сам лезет за кромку — рекорд почти не растёт
    assert max(c["metrics"]["miss_depth"]) < 1.0
    assert np.mean(c["metrics"]["miss_frac"]) >= 0.5



# ======================================================================
# 5. Графика: куда GP звал и куда ставили точки (PNG в output/iter100)
# ======================================================================
def _draw(tag, c, title):
    r, truth, lab = c["runner"], c["truth"], c["lab"]
    grid = r._phase_candidates(3000, seed=101)
    feas = lab.feasible(grid)
    pred = r.surrogates[CLIFF_RESPONSE].predict(grid)
    x_opt = branch_optimum(truth, _goal(truth), n_scan=20000, seed=5)["x"]
    xb = r.branches["b"].x_best
    p_star = star_projection(
        names=NAMES, grid_Xc=grid, grid_field=pred.std, grid_feasible=feas,
        seed_Xc=c["X0"], seed_feasible=c["g0"] >= THR,
        round_Xc=[rd["X"] for rd in c["rounds"]],
        round_feasible=[rd["g"] >= THR for rd in c["rounds"]],
        x_best=None if xb is None else np.asarray(xb, float), x_opt=x_opt,
        field_label=f"σ суррогата {CLIFF_RESPONSE}", title=title,
        path=OUT_DIR / f"star_{tag}.png")
    yi = r.prop_index[CLIFF_RESPONSE]
    p_slice = cliff_slice(
        dist_grid=truth.gate_true(grid) - THR, mu_grid=pred.mean,
        sd_grid=pred.std, truth_grid=truth.cliff.true(grid),
        dist_seed=c["g0"] - THR, y_seed=c["Y0"][:, yi],
        dist_rounds=[rd["g"] - THR for rd in c["rounds"]],
        y_rounds=[rd["Y"][:, yi] for rd in c["rounds"]],
        y_cliff=truth.cliff.y_cliff, response=CLIFF_RESPONSE,
        gate_label=f"{GATE_3COMP} − {THR:g}", title=title,
        path=OUT_DIR / f"cliff_{tag}.png")
    return p_star, p_slice


def test_plots_black_hole_vs_feasibility_aware():
    hole = _campaign(explore=0.6, feasibility=False)
    fix = _campaign(explore=0.6, feasibility=True)
    paths = []
    paths += _draw("hole", hole,
                   "explore 0.6 БЕЗ учёта измеримости: σ за кромкой максимальна "
                   "— explore-слоты падают в дыру")
    paths += _draw("feas", fix,
                   "explore 0.6 × P(измеримо|x) (set_branch_gate): "
                   "точки ложатся на кромку с измеримой стороны")
    print("\nPNG:", *[str(p) for p in paths], sep="\n  ")
    for p in paths:
        assert p.exists() and p.stat().st_size > 20_000

