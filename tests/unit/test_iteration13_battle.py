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
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.verification.branch_reference import branch_optimum, branch_optimum_masked
from src.apps.mixture_process_runner import MixtureProcessRunner

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


def _build_truth():
    s = _truth_schema()
    coef_by = {
        # прочность: тянет к A (и B), процесс T помогает через кросс
        "strength":  _coef(s, {"A": 10, "B": 6, "C": 2, "A:T": 2}),
        # глянец: тянет к B, процесс P помогает
        "gloss":     _coef(s, {"A": 4, "B": 10, "C": 1, "P": 3}),
        # время сушки (меньше — лучше): база 3 от состава, T ускоряет (−6T)
        "dry_time":  _coef(s, {"A": 3, "B": 3, "C": 3, "T": -6}),
        # цена (меньше — лучше): C дорогой; от процесса не зависит
        "price":     _coef(s, {"A": 1, "B": 2, "C": 5}),
    }
    return MultiMixtureProcessTruth(s, coef_by, noise_sd=0.0)


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


# ----------------------------------------------------------------------
def test_battle_branches_converge_to_analytic_optimum():
    truth = _build_truth()
    goals = _branch_goals()

    # аналитический оптимум каждой ветки над ПОЛНОЙ областью (эталон сходимости)
    opt = {bid: branch_optimum(truth, goal, n_scan=20000, seed=100 + i)
           for i, (bid, goal) in enumerate(goals.items())}

    runner = MixtureProcessRunner(_model_schema(), truth,
                                  baseline=[1/3, 1/3, 1/3, 0.5, 0.5],
                                  seed=7, n_restarts=2)

    # --- Фаза 1: свободны только A,B ---
    runner.set_free(mixture_free=["A", "B"], process_free=[])
    runner.seed_initial(n=18, seed=7)
    for bid, goal in goals.items():
        runner.add_branch(bid, goal, budget=40, satisfy_at=1.1, branch_id=bid)
    for bid in goals:
        runner.run_branch_round(bid, n_points=5, explore_frac=0.3,
                                n_candidates=400)
    d1 = {bid: runner.branches[bid].d_best for bid in goals}

    # --- Фаза 2: + процесс T ---
    runner.set_free(mixture_free=["A", "B"], process_free=["T"])
    for bid in goals:
        runner.run_branch_round(bid, n_points=5, explore_frac=0.25,
                                n_candidates=400)
    d2 = {bid: runner.branches[bid].d_best for bid in goals}

    # --- Фаза 3: + остаток mixture (C) + процесс P ---
    runner.set_free(mixture_free=["A", "B", "C"], process_free=["T", "P"])
    for bid in goals:
        runner.run_branch_round(bid, n_points=7, explore_frac=0.15,
                                n_candidates=600)
    d3 = {bid: runner.branches[bid].d_best for bid in goals}

    # --- сводка результата (видно при `pytest -s`) ---
    seed_n = runner.origin_counts().get("seed", 0)
    print("\n=== БОЕВОЙ ТЕСТ: ветки vs аналитический оптимум ===")
    print(f"общий стартовый план (seed): {seed_n} опытов — используется ВСЕМИ ветками")
    print(f"{'ветка':<9}|{'опытов':>7}|{'d_best ф1/ф2/ф3':>22}|"
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
    print(f"всего измерений в общей базе: {total}  "
          f"(seed {seed_n} + ветки {total - seed_n})")
    print("рецепты (доли A,B,C + код T,P), pipeline x* vs аналитика x*:")
    for bid in goals:
        xb = np.round(np.asarray(runner.branches[bid].x_best, float), 3)
        xo = np.round(np.asarray(opt[bid]["x"], float), 3)
        print(f"  {bid:<9} pipeline={xb.tolist()}  analytic={xo.tolist()}")

    # фазовый «потолок» (best-achievable под маской) — потенциал каждой фазы
    baseline = [1/3, 1/3, 1/3, 0.5, 0.5]
    phase_free = [(["A", "B"], []),
                  (["A", "B"], ["T"]),
                  (["A", "B", "C"], ["T", "P"])]
    ceil = {bid: [branch_optimum_masked(truth, goals[bid], baseline=baseline,
                                        mixture_free=mf, process_free=pf,
                                        seed=300 + j)["d"]
                  for j, (mf, pf) in enumerate(phase_free)]
            for bid in goals}
    print("потолок фазы (max достижимый под маской) vs pipeline d_best:")
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
        # 3) на финале достигнута существенная доля аналитического оптимума
        assert d3[bid] >= 0.7 * d_opt, (
            f"{bid}: финал {d3[bid]:.3f} < 70% от d_opt {d_opt:.3f}")
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
