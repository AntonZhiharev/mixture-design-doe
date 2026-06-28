"""Iteration 13 — БОЕВОЙ ТЕСТ: ядро vs аналитическое решение (REBUILD_SPEC §8).

Эталон — известная синтетическая «лаборатория» над mixture×process:
  * MIXTURE {A,B,C} (Σ=1) + PROCESS {T,P} в коде [0,1];
  * истина — CUBIC Шеффе по составу + RSM по процессу + кросс x·z (богаче модели);
  * 4 свойства, одно из них — ЦЕНА (есть в каждой ветке).

Модель пайплайна — QUADRATIC Шеффе (только тренд/скрининг), дальше остаток
ловит непараметрический GP на составных координатах. То есть истина намеренно
сложнее модели — честная мисспецификация.

3 ВЕТКИ — разные направления целей; «спор» ВНУТРИ ветки (цели тянут рецепт/режим
в разные стороны). ПОЭТАПНОЕ РАСКРЫТИЕ переменных:
  Фаза 1: свободны только 2 mixture (A,B), C и процесс фиксированы на baseline;
  Фаза 2: + процесс T;
  Фаза 3: + остаток mixture (C, полный симплекс) + процесс P.

Проверяем: ядро ПРИБЛИЖАЕТСЯ к аналитическому оптимуму каждой ветки, но НЕ СРАЗУ —
desirability растёт от фазы к фазе и на финале достигает существенной доли
аналитического оптимума (которого без C/процесса в ранних фазах достичь нельзя).
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.optimize.desirability import (DesirabilitySpec, Desirability,
                                        desirability_value)
from src.verification.mixture_process_truth import (MultiMixtureProcessTruth,
                                                    composite_random_points)
from src.verification.branch_reference import branch_optimum, branch_optimum_masked
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.optimize.economic_stop import (decide_stop, evaluate_stop,
                                        ECON_ADVISORY,
                                        expected_price_improvement,
                                        economic_value, price_attributed_value,
                                        price_attribution_alpha, best_of_n_value)





# Косметика: GP на чистой (без шума) истине загоняет noise→0 и часть length-scale
# к границам → sklearn сыплет ConvergenceWarning. Это НЕ ошибка фита и НЕ влияет
# на корректность; глушим ТОЛЬКО предупреждение, границы ядра GPExpert не трогаем.
warnings.filterwarnings("ignore", category=ConvergenceWarning)



# ----------------------------------------------------------------------
def _truth_schema():
    """Истина: CUBIC mixture + quadratic process + полный кросс."""
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="cubic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _model_schema():
    """Модель пайплайна: QUADRATIC mixture (тренд/скрининг); остаток ловит GP."""
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _coef(schema, contributions):
    """Вектор коэффициентов истины по именам термов (остальные = 0)."""
    terms = build_model_terms(schema)
    v = np.zeros(terms.p)
    for i, name in enumerate(terms.names):
        v[i] = float(contributions.get(name, 0.0))
    return v


def _refloor_truth(truth, *, B_idx: int, lo: float):
    """Rebuild the truth on a mixture region with a lower bound floor on one
    component (e.g. B >= lo), reusing the SAME coefficient vectors. Used to get
    the analytic optimum UNDER the step-4 floor (apples-to-apples with the
    floored pipeline x_best)."""
    s0 = truth.schema
    mb0 = s0.mixture_block()
    lower = list(mb0.lower)
    lower[B_idx] = float(lo)
    mix = VariableBlock.mixture(list(mb0.names), lower=lower, upper=list(mb0.upper))
    proc = s0.process_block()
    s = ProjectSchema.mixture_process(mix, proc, model=s0.model)
    coef_by = {name: truth.truths[name].coefficients for name in truth.property_names}
    return MultiMixtureProcessTruth(s, coef_by, noise_sd=0.0)


def _build_truth():
    s = _truth_schema()
    # ИСТИННАЯ мисспецификация: термы ВНЕ Scheffé-quadratic тренда пайплайна,
    # которые способен восстановить только GP-остаток (kernel), а не quadratic-mean:
    #   A*B*C            — спецкубическая синергия состава (degree 3);
    #   T^2, P^2         — ЧИСТЫЕ квадраты процесса (в Scheffé-quadratic их нет).
    # Парные термы (A*B, A:T, T*P, …) — degree 2 и лежат ВНУТРИ тренда (не стресс).
    # INTERIOR trade-off design: every branch optimum keeps A,B,C all > 0 (no
    # recipe component degenerates to a vertex/edge). Achieved purely via truth
    # coefficients (core untouched):
    #   * concave composition synergies (A*B, A*C, B*C, A*B*C) build an interior
    #     ridge so pure vertices are dominated by mixtures;
    #   * equal linear price (A=B=C=2) removes the structural push of one cheap
    #     vertex; trade-offs come from synergies, not from a linear price gradient;
    #   * dry_time is COMPOSITION-DEPENDENT (A fast-drying -2, B/C slow +4) - the
    #     old `3(A+B+C)` was constant on the simplex (Sum x=1) so `fast` ignored
    #     the recipe and sat on an edge; now `fast` genuinely trades composition;
    #   * gloss reaches its target=8 only through B-bearing ridges (no A*C bypass),
    #     so `fast` must keep B > 0.
    # Misspecification (truth richer than the pipeline's quadratic-Scheffe trend)
    # is preserved by A*B*C (degree-3) and pure process squares T^2/P^2, which
    # only the GP residual can recover.
    # Process GATING makes unmasking pay off (ceilings grow across phases):
    #   * C:T in strength and C:T in dry_time, P/-P^2 in gloss -> phase-1 (C=0,
    #     T=P=0.5) cannot reach the optimum, so opening T (phase 2) and C+P
    #     (phase 3) strictly raise the achievable desirability (premium+economy).
    # premium and economy optima are strictly INTERIOR (A,B,C>0); `fast` lets one
    # component (B) degenerate toward an edge - this is INTENTIONAL: STEP 4 below
    # demonstrates the move_bounds primitive by bringing that degenerate component
    # back into play via a lower-bound shift (anti-degeneracy floor, §15.0.3).
    coef_by = {
        # strength: synergies + process gating (A:T, C:T) so T-opening helps
        "strength":  _coef(s, {"A": 6, "B": 6, "C": 5, "A:T": 5, "C:T": 6,
                               "T^2": -3, "A*B": 8, "A*C": 9, "B*C": 9,
                               "A*B*C": 8}),
        # gloss: B*C/A*B*C-led ridge, concave in P (needs P off baseline)
        "gloss":     _coef(s, {"A": 3, "B": 6, "C": 6, "P": 7, "P^2": -5,
                               "A*B": 7, "B*C": 13, "A*C": 9, "A*B*C": 14}),
        # dry_time (min): composition-dependent (A fast, B/C slow), T-gated (C:T)
        "dry_time":  _coef(s, {"A": -1, "B": 3, "C": 3, "T": -4, "C:T": -5,
                               "T^2": 2}),
        # price (min): mild gradient + NEGATIVE pairwise "blend discount" (a mix
        # of two components costs less than either pure). A *constant* price
        # (A=B=C) would be 2*(A+B+C)=2 on the simplex (Sum x=1) -> always zero
        # deviation, meaningless. A pure linear gradient minimises at a vertex
        # (degenerate). The blend discount keeps price VARYING across the simplex
        # (nonzero, branch-specific deviation) with an INTERIOR minimum.
        "price":     _coef(s, {"A": 2.0, "B": 2.2, "C": 2.4,
                               "A*B": -1.5, "A*C": -1.5, "B*C": -1.5}),
        # rho = ПЛОТНОСТЬ (полноценный отклик опыта, GP) для ЭКОНОМИЧЕСКОГО стопа
        # §4 в шагах 4-5; линейный бленд по {A,B,C} БЕЗ компонента D — множитель
        # цены изделия price_состав(A,B,C)·ρ (см. _comp_price3 / _run_with_economic_stop).
        "rho":       _coef(s, {"A": 0.7, "B": 1.0, "C": 1.7}),
    }
    return MultiMixtureProcessTruth(s, coef_by, noise_sd=0.0)


# ----------------------------------------------------------------------
# ЭКОНОМИКА для §4 decide_stop в ШАГАХ 4-5 (3-комп мир {A,B,C}, БЕЗ D/white).
# ρ добавлена в истину как полноценный отклик; цена изделия собирается на лету
# price_состав(A,B,C)·ρ — ровно как в step6, но без компонента D и без ветки
# `white` (она D-зависима и здесь не нужна).
# ----------------------------------------------------------------------
_PRICE3 = {"A": 95.0, "B": 200.0, "C": 23.0}


def _comp_price3(Xc):
    """price_состав [усл.ед/кг] для 3-комп мира {A,B,C} (детерминирована, БЕЗ D)."""
    Xc = np.atleast_2d(np.asarray(Xc, float))
    w = np.array([_PRICE3["A"], _PRICE3["B"], _PRICE3["C"]], float)
    return Xc[:, :3] @ w


# ----------------------------------------------------------------------
# §5 per-property: денежная ценность раунда, АТРИБУТИРОВАННАЯ ценовой оси.
# Чинит objective-agnostic «фантом дешёвого угла» (диагностика §5.3): деньги
# засчитываются только за прирост d_overall ветки, идущий ЧЕРЕЗ цену.
# ----------------------------------------------------------------------
def _props_at(runner, X, names, *, comp_price_fn, price_axis_name, rho_name):
    """Средние предсказания свойств ``names`` на точках X (составная матрица).
    Ось цены: если есть суррогат (Scheffé price, шаги 4-5) — берём его среднее;
    иначе (step6) собираем price_состав(X)·ρ̂ (цена изделия)."""
    X = np.atleast_2d(np.asarray(X, float))
    props = {}
    for nm in names:
        if nm in runner.surrogates:
            props[nm] = np.asarray(runner.surrogates[nm].predict(X).mean, float)
        elif nm == price_axis_name and comp_price_fn is not None:
            rho = np.asarray(runner.surrogates[rho_name].predict(X).mean, float)
            props[nm] = np.asarray(comp_price_fn(X), float).ravel() * rho
        else:
            raise KeyError(f"нет предиктора для оси '{nm}'")
    return props


def _attributed_econ_value(runner, bid, cands, price_savings, *,
                           goal_specs, price_spec, comp_price_fn=None,
                           rho_name="rho", price_axis_name="price"):
    """Денежная ценность раунда ветки, атрибутированная ценовой оси (§5).

    Строит РЕАЛЬНУЮ Desirability ветки (цели + цена), считает d_overall и
    desirability ценовой оси у текущего лучшего рецепта и у кандидатов, затем
    масштабирует сырое удешевление ``price_savings`` (EI по ρ) долей прироста
    d_overall, идущей ЧЕРЕЗ цену (см. :func:`price_attributed_value`)."""
    br = runner.branches[bid]
    full = dict(goal_specs)
    full.setdefault(price_axis_name, price_spec)
    desir = Desirability(full)
    names = list(full.keys())
    # текущий лучший рецепт ветки
    xb = np.asarray(br.x_best, float).reshape(1, -1)
    p_cur = _props_at(runner, xb, names, comp_price_fn=comp_price_fn,
                      price_axis_name=price_axis_name, rho_name=rho_name)
    do_cur = float(desir.overall(p_cur)[0])
    dp_cur = float(desirability_value(p_cur[price_axis_name][0], price_spec))
    # кандидаты
    p_cand = _props_at(runner, cands, names, comp_price_fn=comp_price_fn,
                       price_axis_name=price_axis_name, rho_name=rho_name)
    do_cand = desir.overall(p_cand)
    dp_cand = desirability_value(p_cand[price_axis_name], price_spec)
    return price_attributed_value(
        price_savings, d_overall_cur=do_cur, d_overall_cand=do_cand,
        d_price_cur=dp_cur, d_price_cand=dp_cand,
        price_weight=price_spec.weight,
        total_weight=sum(s.weight for s in full.values()),
        volume=br.volume, horizon=br.horizon)


def _attributed_batch_value(runner, bid, cands, *, comp_price, rho_mean,
                            rho_std, price_best, n_batch, goal_specs, price_spec,
                            comp_price_fn=None, rho_name="rho",
                            price_axis_name="price", seed=None):
    """БАТЧ-версия :func:`_attributed_econ_value` (§4-BATCH): денежная ценность
    РАУНДА из ``n_batch`` опытов = ``E[улучшение лучшей из N]·V·H``, max-тип q-EI,
    атрибутированный ценовой оси (§5). Сравнивать с ``n_batch·c_exp`` (цена раунда).

    Та же α(x), что и в max-single пути (одна реализация —
    :func:`price_attribution_alpha`), но вместо ``max_x`` берётся вогнутый
    best-of-N (:func:`best_of_n_value`): N опытов СТОЯТ N·c_exp и ПРИНОСЯТ
    best-of-N, а не одну точку — так чинятся ЕДИНИЦЫ стоп-критерия."""
    br = runner.branches[bid]
    full = dict(goal_specs)
    full.setdefault(price_axis_name, price_spec)
    desir = Desirability(full)
    names = list(full.keys())
    xb = np.asarray(br.x_best, float).reshape(1, -1)
    p_cur = _props_at(runner, xb, names, comp_price_fn=comp_price_fn,
                      price_axis_name=price_axis_name, rho_name=rho_name)
    do_cur = float(desir.overall(p_cur)[0])
    dp_cur = float(desirability_value(p_cur[price_axis_name][0], price_spec))
    p_cand = _props_at(runner, cands, names, comp_price_fn=comp_price_fn,
                       price_axis_name=price_axis_name, rho_name=rho_name)
    do_cand = desir.overall(p_cand)
    dp_cand = desirability_value(p_cand[price_axis_name], price_spec)
    alpha = price_attribution_alpha(
        d_overall_cur=do_cur, d_overall_cand=do_cand,
        d_price_cur=dp_cur, d_price_cand=dp_cand,
        price_weight=price_spec.weight,
        total_weight=sum(s.weight for s in full.values()))
    return best_of_n_value(comp_price, rho_mean, rho_std, price_best,
                           n_batch=int(n_batch), volume=br.volume,
                           horizon=br.horizon, alpha=alpha, seed=seed)


def _run_with_economic_stop(runner, bid, *, ceil, volume, horizon, cost_exp,
                            goal_specs, comp_price_fn=_comp_price3,
                            rho_name="rho", n_points=5, explore_frac=0.2,
                            n_candidates=500, eps=5e-3, min_rounds=2):

    """Гонять раунды ветки до ПРИНЦИПИАЛЬНОЙ остановки (§4 decide_stop): тот же
    ДВОЙНОЙ критерий, что и в step6 (технический Δd/потолок + экономический
    EI_price·V·H vs c_exp), но в 3-комп мире {A,B,C} (ρ без D, цена изделия =
    price_состав(A,B,C)·ρ). ``min_rounds`` — минимум раундов В ЭТОМ цикле до
    того, как разрешена остановка (прогрев). Возвращает stop_reason."""
    br = runner.branches[bid]
    br.volume, br.horizon, br.cost_exp = float(volume), float(horizon), float(cost_exp)
    rho_i = runner.prop_index[rho_name]
    start = br.spent
    price_best = float("inf")
    prev_d = br.d_best
    final_reason = "budget"
    while br.remaining() > 0:
        res = runner.run_branch_round(bid, n_points=n_points,
                                      explore_frac=explore_frac,
                                      n_candidates=n_candidates)
        Ynew = np.atleast_2d(res["y_new"])
        Xnew = np.atleast_2d(res["x_new"])
        price_best = min(price_best,
                         float(np.min(comp_price_fn(Xnew) * Ynew[:, rho_i])))
        delta = br.d_best - prev_d
        prev_d = br.d_best
        cands = runner._phase_candidates(n_candidates, runner.seed + br.spent)
        pred = runner.surrogates[rho_name].predict(cands)
        # §4-BATCH: ценность РАУНДА из N опытов = E[best-of-N]·V·H (max-тип q-EI),
        # атрибутированная цене (§5). Сравниваем с ЦЕНОЙ РАУНДА N·c_exp — N опытов
        # СТОЯТ N·c_exp и приносят best-of-N, а не одну точку (единицы стопа).
        ev = _attributed_batch_value(
            runner, bid, cands, comp_price=comp_price_fn(cands),
            rho_mean=pred.mean, rho_std=pred.std, price_best=price_best,
            n_batch=n_points, goal_specs=goal_specs,
            price_spec=goal_specs["price"], comp_price_fn=comp_price_fn,
            rho_name=rho_name, seed=runner.seed + br.spent)
        reason = decide_stop(delta_d=delta, d_best=br.d_best, ceil=ceil,
                             economic_value=ev,
                             cost_exp=n_points * br.cost_exp, eps=eps)
        if (br.spent - start) >= int(min_rounds) * int(n_points) \
                and reason is not None:
            final_reason = reason
            break
    return final_reason


def _branch_goals():
    """3 ветки; цена есть в каждой; внутри ветки цели «спорят»."""
    return {
        # Premium: прочность↑ И глянец↑ (A vs B), цена терпима
        "premium": {
            "strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
            "gloss":    DesirabilitySpec("max", low=1.0, high=13.0, weight=1.0),
            "price":    DesirabilitySpec("min", low=1.0, high=5.0, weight=0.3)},
        # Economy: цена↓ (жёстко) + прочность↑ (спор за B) + сушка↓ (через T)
        "economy": {
            "price":    DesirabilitySpec("min", low=1.0, high=5.0, weight=2.0),
            "strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
            "dry_time": DesirabilitySpec("min", low=-3.0, high=3.0, weight=1.0)},
        # Fast: сушка↓ (T) + глянец=target (P, спор за B с ценой) + цена↓ умеренно
        "fast": {
            "dry_time": DesirabilitySpec("min", low=-3.0, high=3.0, weight=1.5),
            "gloss":    DesirabilitySpec("target", low=1.0, high=13.0,
                                         target=8.0, weight=1.0),
            "price":    DesirabilitySpec("min", low=1.0, high=5.0, weight=0.5)},
    }


def _matte_goal():
    """NEW product line (STEP 4): a matte/soft-sheen finish where GLOSS is the
    leading objective and B is its FUNCTIONAL gloss agent. Its analytic optimum
    is strictly interior with B firmly in play (B ~ 0.32) - this is what makes
    the later `B >= floor` honest: B is a mandatory component of this line, not
    an arbitrary decree."""
    return {
        "gloss":    DesirabilitySpec("max", low=1.0, high=13.0, weight=1.5),
        "strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
        "price":    DesirabilitySpec("min", low=1.0, high=5.0, weight=0.5),
    }


def _deep_gloss_goal():
    """NEW product line (STEP 5): a deep high-gloss finish that needs a LARGE
    dose of the C component (its analytic optimum sits at C ~ 0.45). When the
    project currently caps C <= 0.20 (a region-of-interest limit), this optimum
    is OUT OF REACH - the goal can only be met by RELAXING (expanding) C's upper
    bound. That is the honest trigger for a move_bounds RELAX."""
    return {
        "gloss":    DesirabilitySpec("max", low=8.0, high=13.0, weight=2.0),
        "strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
        "price":    DesirabilitySpec("min", low=1.0, high=5.0, weight=0.3),
    }


def _recap_truth(truth, *, var: str, lo: float, hi: float):
    """Rebuild the truth on a mixture region with NEW [lo,hi] bounds on ONE
    component, reusing the SAME coefficient vectors. Used to get the analytic
    optimum UNDER a region cap/floor (apples-to-apples with a region-moved
    pipeline x_best). Generalises :func:`_refloor_truth` to any single bound."""
    s0 = truth.schema
    mb0 = s0.mixture_block()
    lower = list(mb0.lower)
    upper = list(mb0.upper)
    j = list(mb0.names).index(var)
    lower[j] = float(lo)
    upper[j] = float(hi)
    mix = VariableBlock.mixture(list(mb0.names), lower=lower, upper=upper)
    proc = s0.process_block()
    s = ProjectSchema.mixture_process(mix, proc, model=s0.model)
    coef_by = {name: truth.truths[name].coefficients for name in truth.property_names}
    return MultiMixtureProcessTruth(s, coef_by, noise_sd=0.0)


# ----------------------------------------------------------------------
# Pretty-printers for the test trace (ASCII-only: the Windows cp1252 console
# raises UnicodeEncodeError on Cyrillic/Unicode arrows, so we stick to ASCII).
# ----------------------------------------------------------------------
def _fmt_spec(spec):
    """One Derringer-Suich goal as a human-readable condition string."""
    if spec.kind == "target":
        body = (f"TARGET={spec.target:g} (range [{spec.low:g},{spec.high:g}], "
                f"d=0 outside, peak at target)")
    elif spec.kind == "max":
        body = (f"MAXIMIZE (d: 0 at y<={spec.low:g} -> 1 at y>={spec.high:g})")
    else:  # min
        body = (f"MINIMIZE (d: 1 at y<={spec.low:g} -> 0 at y>={spec.high:g})")
    return f"{body}  w={spec.weight:g}"


def _fmt_truth(truth, prop):
    """The truth function eta(prop) as a sum of nonzero terms*coeff (honest:
    read straight from the coefficient vector, so it can never drift)."""
    t = truth.truths[prop]
    parts = [f"{c:+g}*{name}"
             for name, c in zip(t.terms.names, t.coefficients)
             if abs(c) > 1e-12]
    return " ".join(parts) if parts else "0"


def _print_truth_functions(truth, props=None):
    """Print eta(prop) for the requested properties (default: all)."""
    props = props or truth.property_names
    print("  truth functions eta(x,z)  [A,B,C fractions; T,P code in [0,1]]:")
    for p in props:
        print(f"    {p:<9} = {_fmt_truth(truth, p)}")


def _print_goal(bid, goal):
    """Print a branch goal: every property + its desirability condition."""
    print(f"  goal [{bid}] (overall = weighted geo-mean of d_i; any d_i=0 -> veto):")
    for prop, spec in goal.items():
        print(f"    {prop:<9}: {_fmt_spec(spec)}")


def _print_vars(runner, *, label):
    """Print the variables CURRENTLY in play (schema bounds = phase freedom)."""
    s = runner.current_schema
    parts = []
    mb = s.mixture_block()
    if mb is not None:
        for nm, lo, hi in zip(mb.names, mb.lower, mb.upper):
            tag = "" if lo < hi - 1e-12 else " [fixed]"
            parts.append(f"{nm} in [{lo:g},{hi:g}]{tag}")
    pb = s.process_block()
    if pb is not None:
        for nm, lo, hi in zip(pb.names, pb.lower, pb.upper):
            parts.append(f"{nm} in [{lo:g},{hi:g}] (code)")
    absent_mix = [nm for nm in runner.full_schema.mixture_names
                  if mb is None or nm not in mb.names]
    absent_proc = [nm for nm in runner.full_schema.process_names
                   if pb is None or nm not in pb.names]
    absent = absent_mix + absent_proc
    print(f"  {label}: variables in play (schema v{s.version}): "
          + ", ".join(parts))
    if absent:
        print(f"    not yet introduced (baseline-substituted): {', '.join(absent)}")


# ----------------------------------------------------------------------
def test_battle_branches_converge_to_analytic_optimum():
    truth = _build_truth()
    goals = _branch_goals()

    # аналитический оптимум каждой ветки над ПОЛНОЙ областью (эталон сходимости)
    opt = {bid: branch_optimum(truth, goal, n_scan=20000, seed=100 + i)
           for i, (bid, goal) in enumerate(goals.items())}

    # --- synthetic "lab": variables, truth functions, branch goals -----
    print("\n=== SYNTHETIC LAB: variables, truth functions, goals ===")
    print("  MIXTURE {A,B,C} fractions (Sum=1); PROCESS {T,P} code in [0,1]; "
          "baseline=[1/3,1/3,1/3, 0.5,0.5]")
    _print_truth_functions(truth)
    print("  branch goals (3 lines; price appears in each; intra-branch tension):")
    for bid, goal in goals.items():
        _print_goal(bid, goal)

    runner = MixtureProcessRunner(_model_schema(), truth,
                                  baseline=[1/3, 1/3, 1/3, 0.5, 0.5],
                                  seed=7, n_restarts=2)

    # --- Фаза 1: свободны только A,B (схема v1: mixture-only, C заперт bounds) ---
    print("\n--- PHASE 1: only A,B free (C and process not yet introduced) ---")
    runner.begin_phase(mixture_free=["A", "B"], process_free=[])
    _print_vars(runner, label="phase 1")
    runner.seed_initial(n=18, seed=7)

    for bid, goal in goals.items():
        runner.add_branch(bid, goal, budget=60, satisfy_at=1.1, branch_id=bid)
    for bid in goals:
        runner.run_branch_round(bid, n_points=5, explore_frac=0.3,
                                n_candidates=400)
    d1 = {bid: runner.branches[bid].d_best for bid in goals}

    # --- Фаза 2: + процесс T (APPEND в схему, §14: version+1, миграция фазы 1) ---
    print("\n--- PHASE 2: + process T (append to schema, version+1) ---")
    runner.augment_phase_schema(["T"])
    _print_vars(runner, label="phase 2")
    for bid in goals:

        runner.run_branch_round(bid, n_points=5, explore_frac=0.25,
                                n_candidates=400)
    d2 = {bid: runner.branches[bid].d_best for bid in goals}

    # --- Фаза 3: + процесс P (append) + раскрытие C (relax) АТОМАРНО (§15.1.5) ---
    print("\n--- PHASE 3: + process P (append) + open C (full simplex), atomic ---")
    runner.augment_phase_atomic(["P"], ["C"])
    _print_vars(runner, label="phase 3")
    for bid in goals:

        runner.run_branch_round(bid, n_points=7, explore_frac=0.15,
                                n_candidates=600)
    d3 = {bid: runner.branches[bid].d_best for bid in goals}

    # --- result summary (visible with `pytest -s`; ASCII-only to avoid the
    #     Windows cp1252 console UnicodeEncodeError on Cyrillic prints) ---
    seed_n = runner.origin_counts().get("seed", 0)
    print("\n=== BATTLE TEST: branches vs analytic optimum ===")
    print(f"shared seed design: {seed_n} runs - used by ALL branches")
    print(f"{'branch':<9}|{'runs':>7}|{'d_best p1/p2/p3':>22}|"
          f"{'d_opt':>7}|{'gap%':>6}|{'||x-x*||':>9}")
    for bid in goals:
        d_opt = opt[bid]["d"]
        xb = np.asarray(runner.branches[bid].x_best, float)
        xo = np.asarray(opt[bid]["x"], float)
        dist = float(np.linalg.norm(xb - xo))
        gap = 100.0 * (1.0 - d3[bid] / d_opt) if d_opt > 0 else 0.0
        prog = f"{d1[bid]:.2f}/{d2[bid]:.2f}/{d3[bid]:.2f}"
        print(f"{bid:<9}|{runner.branches[bid].spent:>7}|{prog:>22}|"
              f"{d_opt:>7.3f}|{gap:>6.1f}|{dist:>9.3f}")
    total = seed_n + sum(runner.branches[bid].spent for bid in goals)
    print(f"total measurements in shared base: {total}  "
          f"(seed {seed_n} + branches {total - seed_n})")
    print("recipes (A,B,C fractions + T,P code), pipeline x* vs analytic x*:")
    for bid in goals:
        xb = np.round(np.asarray(runner.branches[bid].x_best, float), 3)
        xo = np.round(np.asarray(opt[bid]["x"], float), 3)
        print(f"  {bid:<9} pipeline={xb.tolist()}  analytic={xo.tolist()}")
    print("properties (truth) pipeline vs analytic; dPRICE - price deviation:")
    for bid in goals:
        xb = np.asarray(runner.branches[bid].x_best, float).reshape(1, -1)
        yp = {p: float(truth.truths[p].true(xb)[0]) for p in truth.property_names}
        ya = {p: float(opt[bid]["y"][p]) for p in truth.property_names}
        dpa = yp["price"] - ya["price"]
        dpp = 100.0 * dpa / abs(ya["price"]) if abs(ya["price"]) > 1e-9 else 0.0
        print(f"  {bid:<9} PRICE pipe={yp['price']:.2f} vs anal={ya['price']:.2f} "
              f"d={dpa:+.2f} ({dpp:+.0f}%) | "
              f"strength {yp['strength']:.1f}/{ya['strength']:.1f} "
              f"gloss {yp['gloss']:.1f}/{ya['gloss']:.1f} "
              f"dry {yp['dry_time']:.1f}/{ya['dry_time']:.1f}")


    # phase "ceiling" (best achievable under the mask) - potential of each phase
    baseline = [1/3, 1/3, 1/3, 0.5, 0.5]
    phase_free = [(["A", "B"], []),
                  (["A", "B"], ["T"]),
                  (["A", "B", "C"], ["T", "P"])]
    ceil = {bid: [branch_optimum_masked(truth, goals[bid], baseline=baseline,
                                        mixture_free=mf, process_free=pf,
                                        seed=300 + j)["d"]
                  for j, (mf, pf) in enumerate(phase_free)]
            for bid in goals}
    print("phase ceiling (max achievable under mask) vs pipeline d_best:")
    for bid in goals:
        c = ceil[bid]
        print(f"  {bid:<9} ceil={c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f}  "
              f"pipe={d1[bid]:.2f}/{d2[bid]:.2f}/{d3[bid]:.2f}")

    # ---- проверки ----
    n_strict = 0
    for bid in goals:
        d_opt = opt[bid]["d"]
        # 1) монотонный рост по фазам (d_best только накапливается)
        assert d2[bid] >= d1[bid] - 1e-9
        assert d3[bid] >= d2[bid] - 1e-9
        # 2) санити: нельзя превзойти аналитический оптимум на чистой истине
        assert d3[bid] <= d_opt + 1e-6, (
            f"{bid}: d_best={d3[bid]:.3f} > d_opt={d_opt:.3f}")
        # 3) на финале достигнута существенная доля аналитического оптимума.
        #    При ИСТИННОЙ мисспецификации (ABC/T²/P² вне quadratic-тренда) разрыв
        #    больше — порог мягче (0.6); реальную величину смотрим в печати gap%.
        assert d3[bid] >= 0.6 * d_opt, (
            f"{bid}: финал {d3[bid]:.3f} < 60% от d_opt {d_opt:.3f}")

        # 3b) фазовый потолок: растёт по фазам (раскрытие не сужает достижимое),
        #     финал фазы 3 ~ глобальный оптимум, и пайплайн не превосходит потолок
        #     СВОЕЙ фазы (корректность маски свободы)
        c = ceil[bid]
        assert c[1] >= c[0] - 0.02 and c[2] >= c[1] - 0.02, (
            f"{bid}: потолки фаз не монотонны: {c}")
        assert d_opt >= c[2] - 0.02, f"{bid}: d_opt {d_opt:.3f} < потолок ф3 {c[2]:.3f}"
        assert d1[bid] <= c[0] + 0.02, f"{bid}: ф1 выше потолка {d1[bid]:.3f}>{c[0]:.3f}"
        assert d2[bid] <= c[1] + 0.02, f"{bid}: ф2 выше потолка {d2[bid]:.3f}>{c[1]:.3f}"
        assert d3[bid] <= c[2] + 0.02, f"{bid}: ф3 выше потолка {d3[bid]:.3f}>{c[2]:.3f}"
        # 4) раскрытие реально помогло (строгий рост хотя бы у части веток)
        if d3[bid] > d1[bid] + 0.03:
            n_strict += 1

    # «не сразу»: минимум у двух из трёх веток финал заметно лучше фазы 1
    assert n_strict >= 2, (f"раскрытие переменных почти не помогло: "
                           f"d1={d1}, d3={d3}")

    # общая база выросла, у каждой ветки есть измеренные точки
    counts = runner.origin_counts()
    assert counts.get("seed", 0) == 18
    for bid in goals:
        assert counts.get(f"branch:{bid}", 0) >= 10
        assert runner.branches[bid].x_best is not None

    # ==================================================================
    # РАСШИРЕННЫЙ ЗАМЕР: на что РЕАЛЬНО способен алгоритм в фазе 3
    # (полная свобода) и ВО СКОЛЬКО экспериментов это обходится.
    # Дозабор продолжается, пока ветка СТАГНИРУЕТ (is_stagnating) ИЛИ не
    # исчерпает бюджет. Эталон достижимого — ПОТОЛОК ФАЗЫ 3
    # (branch_optimum_masked при полной свободе, ~ глобальный оптимум).
    # ==================================================================
    PATIENCE, MIN_DELTA = 3, 5e-3
    stop_reason = {}
    for bid in goals:
        br = runner.branches[bid]
        while br.remaining() > 0:
            runner.run_branch_round(bid, n_points=5, explore_frac=0.15,
                                    n_candidates=600)
            if br.is_stagnating(patience=PATIENCE, min_delta=MIN_DELTA):
                stop_reason[bid] = "stagnation"
                break
        else:
            stop_reason[bid] = "budget"

    def _cost_to(history, target):
        """Сколько опытов ВЕТКИ (br.spent) до первого d_best >= target."""
        for h in history:
            if float(h["d_best"]) >= target:
                return int(h["spent"])
        return None

    seed_n = runner.origin_counts().get("seed", 0)
    print("\n=== COST/QUALITY: phase-3 top-up until convergence ===")
    print("achievable reference = PHASE-3 CEILING (full freedom); "
          "runs counted PER BRANCH (shared seed excluded)")
    print(f"{'branch':<9}|{'ceil p3':>8}|{'d3->final':>16}|{'%ceil':>6}|"
          f"{'br.run':>7}|{'80%':>5}|{'90%':>5}|{'95%':>5}|{'stop':>10}")
    d_final = {}
    for bid in goals:
        br = runner.branches[bid]
        c3 = ceil[bid][2]
        d_final[bid] = br.d_best
        pct = 100.0 * br.d_best / c3 if c3 > 0 else 0.0
        c80 = _cost_to(br.history, 0.80 * c3)
        c90 = _cost_to(br.history, 0.90 * c3)
        c95 = _cost_to(br.history, 0.95 * c3)
        def _f(v):
            return "-" if v is None else str(v)
        print(f"{bid:<9}|{c3:>8.3f}|{d3[bid]:>6.2f}->{br.d_best:<8.3f}|{pct:>5.0f}%|"
              f"{br.spent:>7}|{_f(c80):>5}|{_f(c90):>5}|{_f(c95):>5}|"
              f"{stop_reason[bid]:>10}")
    total2 = seed_n + sum(runner.branches[bid].spent for bid in goals)
    print(f"total measurements in shared base after top-up: {total2} "
          f"(seed {seed_n} + branches {total2 - seed_n})")

    # --- FINAL AFTER TOP-UP: recipes and properties pipeline x* vs analytic ---
    print("FINAL recipes (A,B,C fractions + T,P code), pipeline x* vs analytic x*:")
    for bid in goals:
        xb = np.round(np.asarray(runner.branches[bid].x_best, float), 3)
        xo = np.round(np.asarray(opt[bid]["x"], float), 3)
        print(f"  {bid:<9} pipeline={xb.tolist()}  analytic={xo.tolist()}")
    print("FINAL properties (truth) pipeline vs analytic; dPRICE - price deviation:")
    for bid in goals:
        xb = np.asarray(runner.branches[bid].x_best, float).reshape(1, -1)
        yp = {p: float(truth.truths[p].true(xb)[0]) for p in truth.property_names}
        ya = {p: float(opt[bid]["y"][p]) for p in truth.property_names}
        dpa = yp["price"] - ya["price"]
        dpp = 100.0 * dpa / abs(ya["price"]) if abs(ya["price"]) > 1e-9 else 0.0
        print(f"  {bid:<9} PRICE pipe={yp['price']:.2f} vs anal={ya['price']:.2f} "
              f"d={dpa:+.2f} ({dpp:+.0f}%) | "
              f"strength {yp['strength']:.1f}/{ya['strength']:.1f} "
              f"gloss {yp['gloss']:.1f}/{ya['gloss']:.1f} "
              f"dry {yp['dry_time']:.1f}/{ya['dry_time']:.1f}")

    # ---- проверки расширенного замера (робастные, без флака) ----
    for bid in goals:
        c3 = ceil[bid][2]
        # дозабор НЕ ухудшает лучший результат (монотонность d_best)
        assert d_final[bid] >= d3[bid] - 1e-9, (
            f"{bid}: дозабор ухудшил d_best {d3[bid]:.3f}→{d_final[bid]:.3f}")
        # нельзя превзойти физический потолок фазы на чистой истине
        assert d_final[bid] <= c3 + 0.03, (
            f"{bid}: final {d_final[bid]:.3f} > потолок ф3 {c3:.3f}")

    # ==================================================================
    # STEP 4: A NEW GOAL DRIVES AN HONEST ANTI-DEGENERACY FLOOR (§15.0.3).
    #
    # The MOVE_RESTRICT must not be an arbitrary decree - it must ARISE from a
    # real objective. So a NEW product line `matte` arrives whose leading goal
    # is GLOSS and whose FUNCTIONAL gloss agent is component B: its analytic
    # optimum is strictly interior with B firmly in play (B ~ 0.32). We let the
    # pipeline pursue that new goal on the SAME shared physics model (canon:
    # one model per project, a branch is just intent), then read off its x_best.
    #
    # Because the matte line MANDATES B as a functional component, the project
    # now imposes a MINIMUM B DOSE B >= B_FLOOR across ALL recipes (so a
    # degenerate B ~ 0 recipe like `fast` is no longer admissible - B must stay
    # in play to remain matte-compatible on the shared production line). THAT is
    # the honest trigger for the floor.
    #
    # The floor is a REGION move (move_bounds), NOT a schema change:
    #   * schema_version must NOT bump (§15.2.4);
    #   * the shared base (history) is NEVER truncated - points with B<floor stay
    #     in runner.points, only excluded from the ACTIVE pool (policy "exclude",
    #     §15.0.3.3);
    #   * after the floor, M8-argmax (optimize_xbest) must return a recipe with
    #     B >= floor -> the degenerate component is back in play;
    #   * reversibility: dropping the floor restores the excluded points.
    # ==================================================================
    from src.design.move_bounds import MOVE_RESTRICT, MOVE_RELAX

    print("\n=== STEP 4: new GLOSS goal 'matte' needs B -> honest floor ===")
    _print_vars(runner, label="step 4")
    matte_goal = _matte_goal()
    _print_goal("matte", matte_goal)
    _print_truth_functions(truth, ["gloss", "strength", "price"])
    # the new goal's analytic optimum genuinely needs B in play (B ~ 0.32)
    matte_opt = branch_optimum(truth, matte_goal, n_scan=20000, seed=4242)
    print(f"matte ANALYTIC optimum (A,B,C,T,P)="
          f"{np.round(matte_opt['x'], 3).tolist()}  B={matte_opt['x'][1]:.3f} "
          f"d={matte_opt['d']:.3f}")
    assert matte_opt["x"][1] > 0.12, (
        "matte's optimum must genuinely need B in play to justify the floor; "
        f"got B={matte_opt['x'][1]:.3f}")

    # the pipeline pursues the new goal on the SHARED model. Остановка — по тому
    # же §4 decide_stop, что и в step6 (потолок = 99% аналитики matte; экономика
    # EI_price·V·H vs c_exp), в 3-комп мире {A,B,C} (ρ без D, цена без D).
    runner.add_branch("matte", matte_goal, budget=30, satisfy_at=1.1,
                      branch_id="matte")
    matte_ceil = 0.99 * matte_opt["d"]
    matte_stop = _run_with_economic_stop(
        runner, "matte", ceil=matte_ceil, volume=1.0, horizon=12.0,
        cost_exp=1500.0, goal_specs=matte_goal, n_points=5, explore_frac=0.2,
        n_candidates=500, min_rounds=2)

    print(f"step4 matte economic stop: {matte_stop}  "
          f"(d_best={runner.branches['matte'].d_best:.3f} ceil={matte_ceil:.3f})")
    assert matte_stop in {"ceil_reached", "not_economical", "stagnation", "budget"}
    xb_matte = np.asarray(runner.branches["matte"].x_best, float)
    print(f"matte PIPELINE x_best   (A,B,C,T,P)="
          f"{np.round(xb_matte, 3).tolist()}  B={xb_matte[1]:.3f} "
          f"d_best={runner.branches['matte'].d_best:.3f}")
    assert xb_matte[1] > 0.12, (
        f"pipeline did not keep B in play for the matte goal: B={xb_matte[1]:.3f}")

    # this NEW goal is what justifies a project-wide minimum B dose. The
    # degenerate `fast` recipe (B ~ 0) is now inadmissible on the shared line:
    xb_before = np.asarray(runner.branches["fast"].x_best, float)
    print(f"fast x_best before floor (A,B,C,T,P)="
          f"{np.round(xb_before, 3).tolist()}  B={xb_before[1]:.3f} "
          f"(< floor -> must be brought back in play)")

    v_before = runner.current_schema_version
    n_hist_before = len(runner.points)
    n_active_before = len(runner._migrated_points())

    # floor motivated by the matte goal (a minimum functional B dose); we keep
    # it just under matte's optimum so it binds the degenerate `fast`, not matte
    B_FLOOR = 0.12
    mv = runner.move_region({"B": (B_FLOOR, 1.0)}, intent="reach_target")
    n_active_after = len(runner._migrated_points())
    dropped = n_active_before - n_active_after

    print(f"move_type={mv.move_type}  B in [{B_FLOOR}, 1.0]  "
          f"version {v_before}->{runner.current_schema_version}")
    print(f"history (shared base): {n_hist_before} -> {len(runner.points)} "
          f"(never truncated)")
    print(f"active pool: {n_active_before} -> {n_active_after} "
          f"(dropped {dropped} points with B<{B_FLOOR})")

    # region move, NOT schema evolution -> version stays
    assert mv.move_type == MOVE_RESTRICT
    assert runner.current_schema_version == v_before, (
        "move_region must not bump schema_version (§15.2.4)")
    # shared base (history) is never truncated by a region move
    assert len(runner.points) == n_hist_before, (
        "move_region truncated the shared base - history must be preserved")
    # every active point now respects the floor
    for p in runner._migrated_points():
        b = p.X["MIXTURE"][1]
        assert b >= B_FLOOR - 1e-6, f"active point violates B>= {B_FLOOR}: B={b}"

    # M8-argmax on the floored region for ALL branches (the floor is a global
    # region constraint), and the analytic optimum recomputed UNDER THE SAME
    # floor (apples-to-apples) so the post-step-4 summary mirrors the earlier
    # phase summaries (recipes + properties, pipeline x* vs analytic x*).
    floored_truth = _refloor_truth(truth, B_idx=1, lo=B_FLOOR)
    opt4 = {bid: branch_optimum(floored_truth, goals[bid], n_scan=20000,
                                seed=4000 + i)
            for i, bid in enumerate(goals)}
    xb4 = {}
    for bid in goals:
        xb4[bid] = np.asarray(runner.optimize_xbest(bid).x, float)

    print("FINAL+floor recipes (A,B,C + T,P), pipeline x* vs analytic x* "
          f"(B>={B_FLOOR}):")
    for bid in goals:
        xb = np.round(xb4[bid], 3)
        xo = np.round(np.asarray(opt4[bid]["x"], float), 3)
        print(f"  {bid:<9} pipeline={xb.tolist()}  analytic={xo.tolist()}")
    print("FINAL+floor properties (truth) pipeline vs analytic; dPRICE - price dev:")
    for bid in goals:
        xb = xb4[bid].reshape(1, -1)
        yp = {p: float(truth.truths[p].true(xb)[0]) for p in truth.property_names}
        ya = {p: float(opt4[bid]["y"][p]) for p in truth.property_names}
        dpa = yp["price"] - ya["price"]
        dpp = 100.0 * dpa / abs(ya["price"]) if abs(ya["price"]) > 1e-9 else 0.0
        print(f"  {bid:<9} PRICE pipe={yp['price']:.2f} vs anal={ya['price']:.2f} "
              f"d={dpa:+.2f} ({dpp:+.0f}%) | "
              f"strength {yp['strength']:.1f}/{ya['strength']:.1f} "
              f"gloss {yp['gloss']:.1f}/{ya['gloss']:.1f} "
              f"dry {yp['dry_time']:.1f}/{ya['dry_time']:.1f}")

    # the degenerate component is back in play: fast's M8-argmax recipe keeps B
    xb_after = xb4["fast"]
    print(f"fast x_best after floor  (A,B,C,T,P)="
          f"{np.round(xb_after, 3).tolist()}  B={xb_after[1]:.3f}")
    assert xb_after[1] >= B_FLOOR - 1e-6, (
        f"floor failed to bring B back in play: B={xb_after[1]:.3f} < {B_FLOOR}")
    assert abs(xb_after[:3].sum() - 1.0) < 1e-6  # still a valid composition

    # reversibility: drop the floor -> excluded points return to the active pool
    runner.move_region({"B": (0.0, 1.0)}, intent="region_of_interest")
    n_active_restored = len(runner._migrated_points())
    print(f"after dropping floor: active pool {n_active_after} -> "
          f"{n_active_restored} (restored {n_active_restored - n_active_after})")
    assert n_active_restored == n_active_before, (
        "dropping the floor did not restore the excluded points "
        "(history must be reusable - §15.0.3.3 reversibility)")
    assert runner.current_schema_version == v_before  # still a region move

    # ==================================================================
    # STEP 5: A NEW GOAL DEMANDS A BOUNDARY EXPANSION (move_bounds RELAX).
    #
    # The project currently constrains component C to a narrow region-of-interest
    # cap C <= C_CAP (e.g. a supplier/quality limit on the C dose). Now a NEW
    # product line `deep_gloss` arrives: a deep, high-gloss finish whose analytic
    # optimum needs a LARGE dose of C (C ~ 0.45) - WELL BEYOND the current cap.
    #
    # Under the cap that optimum is OUT OF REACH: the best achievable desirability
    # is pinned at the capped ceiling (d ~ 0.64). The ONLY honest way to satisfy
    # the new goal is to RELAX (expand) C's upper bound - a MOVE_RELAX. After the
    # expansion the optimum becomes reachable and the pipeline climbs (d ~ 0.75).
    #
    # Invariants mirror Step 4: a RELAX is a REGION move (no schema bump), the
    # shared base is preserved, and points that the earlier cap had excluded come
    # back into the active pool when the bound re-opens (reversibility).
    # ==================================================================
    def _d_truth(goal, x):
        """Overall desirability BY TRUTH at a pipeline recipe ``x`` (q+d)."""
        xc = np.asarray(x, float).reshape(1, -1)
        means = {p: truth.truths[p].true(xc) for p in goal}
        return float(Desirability(dict(goal)).overall(means)[0])

    print("\n=== STEP 5: new high-C goal 'deep_gloss' needs boundary RELAX ===")
    _print_vars(runner, label="step 5")
    deep_goal = _deep_gloss_goal()
    _print_goal("deep_gloss", deep_goal)
    _print_truth_functions(truth, ["gloss", "strength", "price"])

    # the new goal's analytic optimum needs C far beyond any narrow cap
    deep_opt_full = branch_optimum(truth, deep_goal, n_scan=20000, seed=5151)
    print(f"deep_gloss ANALYTIC optimum (A,B,C,T,P)="
          f"{np.round(deep_opt_full['x'], 3).tolist()}  C={deep_opt_full['x'][2]:.3f} "
          f"d={deep_opt_full['d']:.3f}")

    C_CAP = 0.20
    assert deep_opt_full["x"][2] > C_CAP + 0.1, (
        "deep_gloss optimum must lie well beyond the cap to justify a RELAX; "
        f"got C={deep_opt_full['x'][2]:.3f} vs cap {C_CAP}")

    # analytic ceiling UNDER the cap vs UNDER full freedom (apples-to-apples):
    # the move is NECESSARY only if the cap genuinely lowers the achievable max.
    capped_truth = _recap_truth(truth, var="C", lo=0.0, hi=C_CAP)
    deep_opt_cap = branch_optimum(capped_truth, deep_goal, n_scan=20000, seed=5252)
    print(f"deep_gloss analytic ceiling: capped(C<={C_CAP})={deep_opt_cap['d']:.3f} "
          f"-> full={deep_opt_full['d']:.3f}  (relax lifts the ceiling)")
    assert deep_opt_full["d"] > deep_opt_cap["d"] + 0.05, (
        "the cap must genuinely lower the achievable optimum for the RELAX to be "
        f"necessary (capped {deep_opt_cap['d']:.3f} vs full {deep_opt_full['d']:.3f})")

    runner.add_branch("deep_gloss", deep_goal, budget=60, satisfy_at=1.1,
                      branch_id="deep_gloss")

    # --- impose the current narrow cap on C (MOVE_RESTRICT) ---
    v5_before = runner.current_schema_version
    mv_cap = runner.move_region({"C": (0.0, C_CAP)}, intent="region_of_interest")
    assert mv_cap.move_type == MOVE_RESTRICT
    assert runner.current_schema_version == v5_before  # region move, no bump

    # pipeline pursues the new goal UNDER the cap; achievable is pinned by the cap.
    # Остановка — тот же §4 decide_stop (потолок = 99% capped-аналитики).
    cap_ceil = 0.99 * deep_opt_cap["d"]
    cap_stop = _run_with_economic_stop(
        runner, "deep_gloss", ceil=cap_ceil, volume=2.0, horizon=12.0,
        cost_exp=1500.0, goal_specs=deep_goal, n_points=5, explore_frac=0.2,
        n_candidates=500, min_rounds=2)

    print(f"step5 capped deep_gloss economic stop: {cap_stop}")
    assert cap_stop in {"ceil_reached", "not_economical", "stagnation", "budget"}
    xb_cap = np.asarray(runner.optimize_xbest("deep_gloss").x, float)
    d_cap_pipe = _d_truth(deep_goal, xb_cap)
    print(f"under cap C<={C_CAP}: pipeline x_best="
          f"{np.round(xb_cap, 3).tolist()}  C={xb_cap[2]:.3f}  d_truth={d_cap_pipe:.3f}")
    # the capped recipe respects the cap and cannot beat the capped ceiling
    assert xb_cap[2] <= C_CAP + 1e-6, (
        f"capped pipeline recipe violates C<= {C_CAP}: C={xb_cap[2]:.3f}")
    assert d_cap_pipe <= deep_opt_cap["d"] + 0.03, (
        f"capped pipeline d_truth {d_cap_pipe:.3f} exceeds capped ceiling "
        f"{deep_opt_cap['d']:.3f} (cannot beat the analytic max under the cap)")

    # --- the new goal JUSTIFIES expanding C's upper bound (MOVE_RELAX) ---
    n_active_cap = len(runner._migrated_points())
    mv_relax = runner.move_region({"C": (0.0, 1.0)}, intent="region_of_interest")
    n_active_relaxed = len(runner._migrated_points())
    print(f"move_type={mv_relax.move_type}  C cap {C_CAP} -> 1.0  "
          f"active pool {n_active_cap} -> {n_active_relaxed} "
          f"(restored {n_active_relaxed - n_active_cap} high-C points)")
    assert mv_relax.move_type == MOVE_RELAX
    assert runner.current_schema_version == v5_before  # still a region move
    # relaxing the cap brings the previously-excluded high-C points back (revers.)
    assert n_active_relaxed > n_active_cap, (
        "relaxing C's bound did not restore the excluded high-C points")

    # with the bound re-opened the pipeline can finally chase the high-C optimum.
    # Остановка снова §4 decide_stop (потолок = 99% full-аналитики после RELAX).
    relax_ceil = 0.99 * deep_opt_full["d"]
    relax_stop = _run_with_economic_stop(
        runner, "deep_gloss", ceil=relax_ceil, volume=2.0, horizon=12.0,
        cost_exp=1500.0, goal_specs=deep_goal, n_points=5, explore_frac=0.15,
        n_candidates=600, min_rounds=3)

    print(f"step5 relaxed deep_gloss economic stop: {relax_stop}")
    assert relax_stop in {"ceil_reached", "not_economical", "stagnation", "budget"}
    xb_relax = np.asarray(runner.optimize_xbest("deep_gloss").x, float)
    d_relax_pipe = _d_truth(deep_goal, xb_relax)
    print(f"after RELAX C<=1.0: pipeline x_best="
          f"{np.round(xb_relax, 3).tolist()}  C={xb_relax[2]:.3f}  d_truth={d_relax_pipe:.3f}")

    # the expansion genuinely helped: the recipe moved into the newly opened
    # high-C region and desirability climbed past the capped ceiling
    assert xb_relax[2] > C_CAP + 1e-6, (
        f"boundary RELAX failed to push C beyond the old cap: C={xb_relax[2]:.3f}")
    assert d_relax_pipe > d_cap_pipe + 0.02, (
        f"boundary expansion did not improve the new goal: capped {d_cap_pipe:.3f} "
        f"-> relaxed {d_relax_pipe:.3f}")
    assert abs(xb_relax[:3].sum() - 1.0) < 1e-6  # still a valid composition


# ======================================================================
# STEP 6 (REBUILD_SPEC §15.6 §3): ЦЕНА ИЗДЕЛИЯ КАК ЭКОНОМИЧЕСКАЯ ЦЕЛЬ.
#
# Отдельный 4-компонентный мир {A,B,C,D} с РЕАЛЬНЫМИ ценами компонентов
# (A=95 B=200 C=23 D=315 усл.ед/кг). Ключевой канон Пути A:
#   * ρ (ПЛОТНОСТЬ) — ПОЛНОЦЕННЫЙ моделируемый отклик (GP, как strength/gloss);
#   * цена за изделие  price_изд = price_состав(x)·ρ(x)  НЕ хранится отдельным
#     свойством и НЕ фитится в Шеффе — собирается на лету (``set_branch_cost`` в
#     пайплайне; ``cost_fn`` в эталоне). price_состав детерминирована (известные
#     цены), единственный неизвестный множитель — ρ.
# Новое свойство whiteStrength (MIN) ведёт ветку `white`, которой нужен компонент
# D (отбеливатель) — D реально «в игре» в её оптимуме (интерьерный trade-off).
# Проверяем: пайплайн с ценовой целью приближается к АНАЛИТИЧЕСКОМУ оптимуму
# КАЖДОЙ ветки (с учётом цены), white держит D, а цена реально двигает рецепт
# (аналитика с ценой дешевле аналитики без цены).
# ======================================================================
_COMPS4 = ["A", "B", "C", "D"]
_PRICE4 = {"A": 95.0, "B": 200.0, "C": 23.0, "D": 315.0}
# branch -> (объём т/мес, плотность-ориентир) — экономический контекст
_BRANCH_ECON = {"premium": (1.0, 1.0), "economy": (5.0, 1.1),
                "fast": (10.0, 1.4), "white": (1.0, 1.0)}
_PRICE_W = {"premium": 0.3, "economy": 1.0, "fast": 0.5, "white": 0.5}


def _econ_truth_schema():
    mix = VariableBlock.mixture(_COMPS4)
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="cubic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _econ_model_schema():
    mix = VariableBlock.mixture(_COMPS4)
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _build_econ_truth():
    """4-комп истина {A,B,C,D}: 5 откликов (без свойства price — цена собирается
    из price_состав·ρ). ρ — полноценный отклик. Коэффициенты подобраны так, чтобы
    оптимумы веток были интерьерными (компоненты «в игре», не вырождаются)."""
    s = _econ_truth_schema()
    coef_by = {
        # strength (max): B критичен через сильные B-синергии; чистый C слаб →
        # выбрасывать B дорого (держит B в economy несмотря на дороговизну B)
        "strength":  _coef(s, {"A": 6, "B": 10, "C": 2, "D": 2,
                               "A:T": 5, "C:T": 3, "T^2": -3,
                               "A*B": 9, "A*C": 5, "B*C": 16, "A*B*C": 12,
                               "B*D": 4}),
        # gloss (max/target): B-гряда; низкая база от A/C/P → target=8 требует B
        "gloss":     _coef(s, {"A": 3, "B": 7, "C": 3, "D": 2, "P": 6, "P^2": -4,
                               "A*B": 7, "B*C": 14, "A*C": 6, "A*B*C": 15}),
        # dry_time (min): A быстрый, C медленный → fast держит A; T-гейт
        "dry_time":  _coef(s, {"A": -4, "B": 2, "C": 4, "D": 1,
                               "T": -4, "C:T": -5, "T^2": 2}),
        # whiteStrength (MIN, меньше=лучше): D — отбеливатель; A,B,C белят в ПАРЕ
        # с D (A*D/B*D/C*D < 0) → white хочет реальный БЛЕНД вокруг D (интерьер)
        "whiteStrength": _coef(s, {"A": 5, "B": 6, "C": 3, "D": 1,
                                   "C*D": -8, "A*D": -5, "B*D": -4}),
        # rho = ПЛОТНОСТЬ (отклик опыта), линейный бленд — множитель цены изделия
        "rho":       _coef(s, {"A": 0.7, "B": 1.0, "C": 1.7, "D": 0.6}),
    }
    return MultiMixtureProcessTruth(s, coef_by, noise_sd=0.0)


def _comp_price(Xc):
    """price_состав [усл.ед/кг]: детерминированная функция состава (НЕ свойство)."""
    Xc = np.atleast_2d(np.asarray(Xc, float))
    w = np.array([_PRICE4[k] for k in _COMPS4], float)
    return Xc[:, :4] @ w


def _item_price_truth_fn(truth):
    """cost_fn эталона: цена изделия по ИСТИНЕ = price_состав·ρ_truth."""
    def _fn(Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        rho = np.asarray(truth.truths["rho"].true(Xc), float).ravel()
        return _comp_price(Xc) * rho
    return _fn


def _prop_range(truth, name, n=40000, seed=2):
    Xc = composite_random_points(truth.schema, n, seed=seed)
    y = np.asarray(truth.truths[name].true(Xc), float).ravel() if name != "price" \
        else None
    if name == "price":
        y = _item_price_truth_fn(truth)(Xc)
    return float(y.min()), float(y.max())


def _econ_goals(truth):
    """4 ветки; ЦЕНА у каждой — но как cost-цель (price_изд), не как свойство."""
    plo, phi = _prop_range(truth, "price")
    wlo, whi = _prop_range(truth, "whiteStrength")
    price_spec = {bid: DesirabilitySpec("min", low=plo, high=phi,
                                        weight=_PRICE_W[bid])
                  for bid in _BRANCH_ECON}
    goals = {
        "premium": {
            "strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
            "gloss":    DesirabilitySpec("max", low=1.0, high=13.0, weight=1.0)},
        "economy": {
            "strength": DesirabilitySpec("max", low=2.0, high=12.0, weight=1.0),
            "dry_time": DesirabilitySpec("min", low=-3.0, high=3.0, weight=1.0)},
        "fast": {
            "dry_time": DesirabilitySpec("min", low=-3.0, high=3.0, weight=1.5),
            "gloss":    DesirabilitySpec("target", low=1.0, high=13.0,
                                         target=8.0, weight=1.0)},
        "white": {
            "whiteStrength": DesirabilitySpec("min", low=wlo, high=whi, weight=2.0),
            "strength":      DesirabilitySpec("max", low=2.0, high=12.0,
                                              weight=1.0)},
    }
    return goals, price_spec, (plo, phi), (wlo, whi)


def test_battle_step6_item_price_economy_white_branch():
    """STEP 6: ρ-цена изделия как cost-цель + ветка `white` (нужен D)."""
    truth = _build_econ_truth()
    goals, price_spec, (plo, phi), (wlo, whi) = _econ_goals(truth)
    item_price = _item_price_truth_fn(truth)

    print("\n=== STEP 6: item price = price_sostav * rho; 4-comp {A,B,C,D} ===")
    print(f"  prices [u/kg]: A={_PRICE4['A']:.0f} B={_PRICE4['B']:.0f} "
          f"C={_PRICE4['C']:.0f} D={_PRICE4['D']:.0f}; "
          f"price_izd range=[{plo:.0f},{phi:.0f}]")
    print("  rho is a MODELED response (GP); price NOT a property (assembled).")
    _print_truth_functions(truth)

    # --- аналитический оптимум каждой ветки С УЧЁТОМ ЦЕНЫ (cost_fn) -----
    opt = {}
    for i, (bid, goal) in enumerate(goals.items()):
        opt[bid] = branch_optimum(truth, goal, n_scan=40000, seed=600 + i,
                                  cost_fn=item_price, cost_name="price",
                                  cost_spec=price_spec[bid])

    # --- цена реально двигает рецепт: аналитика С ценой дешевле, чем БЕЗ ----
    econ_free = branch_optimum(truth, goals["economy"], n_scan=40000, seed=777)
    p_econ_cost = item_price(np.asarray(opt["economy"]["x"]).reshape(1, -1))[0]
    p_econ_free = item_price(np.asarray(econ_free["x"]).reshape(1, -1))[0]
    print(f"  economy item-price: with-cost={p_econ_cost:.0f} < "
          f"no-cost={p_econ_free:.0f}  (price truly shifts the recipe)")
    assert p_econ_cost < p_econ_free - 1e-6, (
        f"ценовая цель не удешевила экономную ветку: "
        f"{p_econ_cost:.1f} !< {p_econ_free:.1f}")

    # --- white's optimum genuinely needs D in play (whitener) -------------
    xo_white = np.asarray(opt["white"]["x"], float)
    print(f"  white ANALYTIC (A,B,C,D,T,P)={np.round(xo_white, 3).tolist()}  "
          f"D={xo_white[3]:.3f} d={opt['white']['d']:.3f}")
    assert xo_white[3] > 0.08, (
        f"white's optimum must keep D (whitener) in play: D={xo_white[3]:.3f}")

    # --- пайплайн на ОДНОЙ общей модели; цена изделия как cost-цель ветки ---
    runner = MixtureProcessRunner(_econ_model_schema(), truth,
                                  baseline=[0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
                                  seed=13, n_restarts=2)
    runner.begin_phase(mixture_free=_COMPS4, process_free=["T", "P"])
    runner.seed_initial(n=28, seed=13)
    # экономика ВЕТКИ (§15.6 §2): объём потребления V, цена опыта c_exp, горизонт
    # окупаемости H. Они кормят ДВОЙНОЙ стоп-критерий §4 (см. ниже): без них
    # (V=0) экономическая ценность раунда = 0 и ветка стартует "невыгодной".
    H_HORIZON, C_EXP = 12.0, 1500.0
    for bid, goal in goals.items():
        br = runner.add_branch(bid, goal, budget=30, satisfy_at=1.1,
                               branch_id=bid)
        runner.set_branch_cost(bid, _comp_price, price_spec[bid],
                               rho_property="rho", cost_name="price")
        br.volume = float(_BRANCH_ECON[bid][0])   # V — изд/период
        br.cost_exp = C_EXP                        # c_exp — ₽/опыт
        br.horizon = H_HORIZON                     # H — горизонт окупаемости

    # --- ПРИНЦИПИАЛЬНАЯ остановка (§15.6 §4 decide_stop), НЕ фиксированный цикл ---
    # После КАЖДОГО раунда ветки спрашиваем ДВОЙНОЙ критерий §4 и печатаем причину:
    #   ceil_reached   — d_best упёрся в потолок достижимого (>=99% аналитики):
    #                    улучшать в текущей постановке НЕЧЕГО;
    #   not_economical — улучшать есть куда (d<ceil), но денежно невыгодно:
    #                    max_x EI_price·V·H <= c_exp (экономический стоп §1);
    #   stagnation     — прогресс встал (Δd<eps), хотя потолок не достигнут;
    #   budget         — кончился бюджет ветки раньше любого из критериев.
    # Потолок берём как 99% аналитического оптимума с ценой (за пределами — шум
    # измерения чистой истины GP, дальше двигать смысла нет).
    CEIL_FRAC, EPS, WARMUP = 0.99, 5e-3, 10
    rho_i = runner.prop_index["rho"]
    stop_reason, econ_last = {}, {}
    for bid in goals:
        br = runner.branches[bid]
        ceil = CEIL_FRAC * opt[bid]["d"]
        price_best = float("inf")
        prev_d = br.d_best
        final_reason = "budget"
        while br.remaining() > 0:
            res = runner.run_branch_round(bid, n_points=5, explore_frac=0.2,
                                          n_candidates=500)
            Ynew = np.atleast_2d(res["y_new"])
            Xnew = np.atleast_2d(res["x_new"])
            # лучшая ИЗМЕРЕННАЯ цена изделия ветки = price_состав·ρ_изм (§3)
            pim = _comp_price(Xnew) * Ynew[:, rho_i]
            price_best = min(price_best, float(np.min(pim)))
            delta = br.d_best - prev_d
            prev_d = br.d_best
            # денежная ценность ЕЩЁ ОДНОГО раунда (§6): max_x EI_price·V·H по ρ̂
            cands = runner._phase_candidates(500, runner.seed + br.spent)
            pc = _comp_price(cands)
            pred = runner.surrogates["rho"].predict(cands)
            ei = expected_price_improvement(pc, pred.mean, pred.std,
                                            price_best=price_best,
                                            seed=runner.seed + br.spent)
            # §4-BATCH: ценность РАУНДА из N опытов = E[best-of-N]·V·H (max-тип
            # q-EI), атрибутированная цене (§5.3). Деньги — только за прирост
            # d_overall ветки ЧЕРЕЗ удешевление изделия; «дешёвый угол», который
            # ветка не преследует (whiteStrength тянет в дорогую D-область), ~0.
            # Сравниваем с ЦЕНОЙ РАУНДА N·c_exp (N опытов стоят N·c_exp и приносят
            # best-of-N, а не одну точку) — так чинятся ЕДИНИЦЫ стоп-критерия.
            ev = _attributed_batch_value(
                runner, bid, cands, comp_price=pc, rho_mean=pred.mean,
                rho_std=pred.std, price_best=price_best, n_batch=5,
                goal_specs=goals[bid], price_spec=price_spec[bid],
                comp_price_fn=_comp_price, rho_name="rho",
                seed=runner.seed + br.spent)
            econ_last[bid] = (float(np.max(ei)), float(ev))

            reason = decide_stop(delta_d=delta, d_best=br.d_best, ceil=ceil,
                                 economic_value=ev,
                                 cost_exp=5 * br.cost_exp, eps=EPS)
            if br.spent >= WARMUP and reason is not None:
                final_reason = reason
                break
        stop_reason[bid] = final_reason

    # --- summary (pytest -s): теперь с ПРИЧИНОЙ остановки и экономикой ----
    # NB: prints are ASCII-only (Windows cp1252 console raises UnicodeEncodeError
    # on Cyrillic when stdout is redirected to a file) — see module docstring.
    print("stop reasons (double criterion S4): ceil_reached / not_economical / "
          "stagnation / budget")

    print(f"{'branch':<9}|{'runs':>5}|{'d_best':>7}|{'d_opt':>7}|{'%opt':>5}|"
          f"{'item$':>7}|{'opt$':>7}|{'V':>4}|{'EI$':>6}|{'bestN$':>8}|"
          f"{'Nc_exp':>7}|{'stop':>14}")
    for bid in goals:
        br = runner.branches[bid]
        xb = np.asarray(br.x_best, float)
        pb = float(item_price(xb.reshape(1, -1))[0])
        po = float(item_price(np.asarray(opt[bid]["x"]).reshape(1, -1))[0])
        pct = 100.0 * br.d_best / opt[bid]["d"] if opt[bid]["d"] > 0 else 0.0
        ei_v, ev_v = econ_last.get(bid, (0.0, 0.0))
        print(f"{bid:<9}|{br.spent:>5}|{br.d_best:>7.3f}|{opt[bid]['d']:>7.3f}|"
              f"{pct:>4.0f}%|{pb:>7.0f}|{po:>7.0f}|{br.volume:>4.0f}|"
              f"{ei_v:>6.2f}|{ev_v:>8.1f}|{5 * br.cost_exp:>7.0f}|"
              f"{stop_reason[bid]:>14}")
    print("  read: ceil_reached = hit the ceiling (>=99% of analytic, nothing "
          "left to improve);")
    print("        not_economical = EI_price*V*H <= c_exp (round does not pay "
          "off);")
    print("        stagnation = progress stalled; budget = branch budget spent.")


    # ---- проверки (робастные) ----
    valid_reasons = {"ceil_reached", "not_economical", "stagnation", "budget"}
    for bid in goals:
        br = runner.branches[bid]
        d_opt = opt[bid]["d"]
        # 1) санити: нельзя превзойти аналитический оптимум на чистой истине
        assert br.d_best <= d_opt + 0.03, (
            f"{bid}: d_best={br.d_best:.3f} > d_opt={d_opt:.3f}")
        # 2) достигнута существенная доля оптимума (мисспецификация + цена)
        assert br.d_best >= 0.5 * d_opt, (
            f"{bid}: финал {br.d_best:.3f} < 50% от d_opt {d_opt:.3f}")
        # 3) валидный состав
        assert abs(np.asarray(br.x_best, float)[:4].sum() - 1.0) < 1e-6
        # 4) остановка — по ПРИНЦИПИАЛЬНОМУ критерию §4 (известная причина)
        assert stop_reason[bid] in valid_reasons, (
            f"{bid}: неизвестная причина остановки {stop_reason[bid]}")

    # 4b) сошедшиеся ветки остановились ОСОЗНАННО (не просто по бюджету): хотя бы
    #     одна по принципиальному критерию (потолок/экономика/стагнация)
    assert any(stop_reason[bid] != "budget" for bid in goals), (
        f"ни одна ветка не остановилась по критерию §4: {stop_reason}")


    # 4) white реально держит D в игре (отбеливатель не выродился)
    xb_white = np.asarray(runner.branches["white"].x_best, float)
    print(f"  white PIPELINE (A,B,C,D,T,P)={np.round(xb_white, 3).tolist()}  "
          f"D={xb_white[3]:.3f}")
    assert xb_white[3] > 0.06, (
        f"пайплайн не удержал D для white: D={xb_white[3]:.3f}")
    # whiteStrength у пайплайна лучше (ниже) среднего по симплексу
    ws_pipe = float(truth.truths["whiteStrength"].true(xb_white.reshape(1, -1))[0])
    assert ws_pipe < 0.5 * (wlo + whi), (
        f"white не снизил whiteStrength: {ws_pipe:.2f} (mid={(wlo+whi)/2:.2f})")

    # 5) общая база выросла, у каждой ветки есть измеренные точки
    counts = runner.origin_counts()
    assert counts.get("seed", 0) == 28
    for bid in goals:
        assert counts.get(f"branch:{bid}", 0) >= 10

    # ==================================================================
    # STEP 6b: ОВЕРРАЙД ПОЛИТИКИ ЭКОНОМИКИ (ECON_BINDING -> ECON_ADVISORY).
    #
    # На пассе 1 (выше) ветка остановилась по ЭКОНОМИКЕ (not_economical): её
    # экономическая нога атрибутирована цене (§5) и у оптимума уходит в 0
    # (ценовой рычаг исчерпан), поэтому при низком пороге она ветирует. Под
    # ECON_BINDING это и есть мина §5.3 — price-only нога морозит ветку.
    #
    # Теперь пользователь ЯВНО снимает экономическое ограничение и запускает
    # ВТОРОЙ цикл той же ветки под ECON_ADVISORY: «тянем до потолка/стагнации,
    # не обращая внимания на деньги; технический рычаг ведёт качество». Печатаем
    # ДВЕ СТРОКИ на ветку и МЕЖДУ НИМИ — зафиксированное УКАЗАНИЕ (directive).
    # Инвариант: под ADVISORY экономика НЕ может быть причиной стопа, но её
    # ред-флаг доносится наверх (econ_red_flag=True) — «невыгодно» видно.
    # ==================================================================
    econ_stopped = [bid for bid in goals
                    if stop_reason[bid] == "not_economical"]
    print("\n=== STEP 6b: ECON_ADVISORY override (two passes, directive between) ===")
    print(f"branches stopped by economics on pass 1 (BINDING): {econ_stopped}")
    # сценарий должен реально содержать экономический стоп (premium на оптимуме)
    assert "premium" in econ_stopped, (
        f"ожидали not_economical у premium на пассе 1, получили: {stop_reason}")

    advisory_stop, advisory_flag = {}, {}
    for bid in econ_stopped:
        br = runner.branches[bid]
        ceil = CEIL_FRAC * opt[bid]["d"]
        d_before = br.d_best
        ei_v, ev_v = econ_last.get(bid, (0.0, 0.0))
        # СТРОКА 1 — пасс 1 (BINDING): экономика связала
        print(f"  {bid:<9} pass1(BINDING)  d_best={d_before:.3f} "
              f"ceil={ceil:.3f}  EI$={ei_v:.2f} bestN$={ev_v:.1f} "
              f"Nc_exp={5 * br.cost_exp:.0f}  stop=not_economical")
        # --- ЗАФИКСИРОВАННОЕ УКАЗАНИЕ (между строкой 1 и строкой 2) ---
        print(f"  DIRECTIVE [{bid}]: user disables economic constraint -> "
              f"econ_policy=ECON_ADVISORY (price-only leg may NOT veto live "
              f"technical progress; pull to ceiling/stagnation).")
        prev_d = br.d_best
        price_best = float("inf")
        final_reason, red = "budget", False
        while br.remaining() > 0:
            res = runner.run_branch_round(bid, n_points=5, explore_frac=0.2,
                                          n_candidates=500)
            Ynew = np.atleast_2d(res["y_new"])
            Xnew = np.atleast_2d(res["x_new"])
            pim = _comp_price(Xnew) * Ynew[:, rho_i]
            price_best = min(price_best, float(np.min(pim)))
            delta = br.d_best - prev_d
            prev_d = br.d_best
            cands = runner._phase_candidates(500, runner.seed + br.spent)
            pc = _comp_price(cands)
            pred = runner.surrogates["rho"].predict(cands)
            ev = _attributed_batch_value(
                runner, bid, cands, comp_price=pc, rho_mean=pred.mean,
                rho_std=pred.std, price_best=price_best, n_batch=5,
                goal_specs=goals[bid], price_spec=price_spec[bid],
                comp_price_fn=_comp_price, rho_name="rho",
                seed=runner.seed + br.spent)
            # ЯВНО другая политика — экономика НЕ ветирует живой техпрогресс
            dec = evaluate_stop(delta_d=delta, d_best=br.d_best, ceil=ceil,
                                economic_value=ev, cost_exp=5 * br.cost_exp,
                                eps=EPS, econ_policy=ECON_ADVISORY)
            red = red or bool(dec.econ_red_flag)
            if dec.reason is not None:
                final_reason = dec.reason
                break
        advisory_stop[bid] = final_reason
        advisory_flag[bid] = red
        # СТРОКА 2 — пасс 2 (ADVISORY): стоп ТЕХнический, ред-флаг донесён
        print(f"  {bid:<9} pass2(ADVISORY) d_best {d_before:.3f}->{br.d_best:.3f} "
              f"runs={br.spent}  stop={final_reason}  econ_red_flag={red}")

    # ---- проверки оверрайда ----
    for bid in econ_stopped:
        br = runner.branches[bid]
        # под ADVISORY экономика НЕ может быть причиной стопа (нога не ветирует)
        assert advisory_stop[bid] != "not_economical", (
            f"{bid}: ADVISORY всё ещё ветирует экономикой: {advisory_stop[bid]}")
        assert advisory_stop[bid] in {"ceil_reached", "stagnation", "budget"}
        # экономика НЕ исчезла молча — ред-флаг донесён наверх для показа
        assert advisory_flag[bid] is True, (
            f"{bid}: ред-флаг экономики потерян под ADVISORY")
        # дозабор не ухудшил лучшую desirability и не превзошёл оптимум
        assert br.d_best <= opt[bid]["d"] + 0.03


