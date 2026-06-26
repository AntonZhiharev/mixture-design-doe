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
    # ИСТИННАЯ мисспецификация: термы ВНЕ Scheffé-quadratic тренда пайплайна,
    # которые способен восстановить только GP-остаток (kernel), а не quadratic-mean:
    #   A*B*C            — спецкубическая синергия состава (degree 3);
    #   T^2, P^2         — ЧИСТЫЕ квадраты процесса (в Scheffé-quadratic их нет).
    # Парные термы (A*B, A:T, T*P, …) — degree 2 и лежат ВНУТРИ тренда (не стресс).
    coef_by = {
        # прочность: тянет к A; кросс A·T; синергия ABC; вогнутость по T (−T²)
        "strength":  _coef(s, {"A": 10, "B": 6, "C": 2, "A:T": 2,
                               "A*B*C": 8, "T^2": -3}),
        # глянец: тянет к B; процесс P с кривизной (внутренний оптимум P); +ABC
        "gloss":     _coef(s, {"A": 4, "B": 10, "C": 1, "P": 6,
                               "P^2": -4, "A*B*C": 6}),
        # сушка (меньше — лучше): 3 − 6T; +ABC штрафует смесь (тянет к вершине+T=1)
        "dry_time":  _coef(s, {"A": 3, "B": 3, "C": 3, "T": -6, "A*B*C": 6}),
        # цена (меньше — лучше): C дорогой; линейна (в пределах модели — это норм)
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

    # --- Фаза 1: свободны только A,B (схема v1: mixture-only, C заперт bounds) ---
    runner.begin_phase(mixture_free=["A", "B"], process_free=[])
    runner.seed_initial(n=18, seed=7)

    for bid, goal in goals.items():
        runner.add_branch(bid, goal, budget=60, satisfy_at=1.1, branch_id=bid)
    for bid in goals:
        runner.run_branch_round(bid, n_points=5, explore_frac=0.3,
                                n_candidates=400)
    d1 = {bid: runner.branches[bid].d_best for bid in goals}

    # --- Фаза 2: + процесс T (APPEND в схему, §14: version+1, миграция фазы 1) ---
    runner.augment_phase_schema(["T"])
    for bid in goals:

        runner.run_branch_round(bid, n_points=5, explore_frac=0.25,
                                n_candidates=400)
    d2 = {bid: runner.branches[bid].d_best for bid in goals}

    # --- Фаза 3: + процесс P (append) + раскрытие C (relax) АТОМАРНО (§15.1.5) ---
    runner.augment_phase_atomic(["P"], ["C"])
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
    print("свойства (истина) pipeline vs аналитика; Δ ЦЕНА — отклонение по цене:")
    for bid in goals:
        xb = np.asarray(runner.branches[bid].x_best, float).reshape(1, -1)
        yp = {p: float(truth.truths[p].true(xb)[0]) for p in truth.property_names}
        ya = {p: float(opt[bid]["y"][p]) for p in truth.property_names}
        dpa = yp["price"] - ya["price"]
        dpp = 100.0 * dpa / abs(ya["price"]) if abs(ya["price"]) > 1e-9 else 0.0
        print(f"  {bid:<9} ЦЕНА pipe={yp['price']:.2f} vs anal={ya['price']:.2f} "
              f"Δ={dpa:+.2f} ({dpp:+.0f}%) | "
              f"strength {yp['strength']:.1f}/{ya['strength']:.1f} "
              f"gloss {yp['gloss']:.1f}/{ya['gloss']:.1f} "
              f"dry {yp['dry_time']:.1f}/{ya['dry_time']:.1f}")


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
                stop_reason[bid] = "стагнация"
                break
        else:
            stop_reason[bid] = "бюджет"

    def _cost_to(history, target):
        """Сколько опытов ВЕТКИ (br.spent) до первого d_best >= target."""
        for h in history:
            if float(h["d_best"]) >= target:
                return int(h["spent"])
        return None

    seed_n = runner.origin_counts().get("seed", 0)
    print("\n=== ЦЕНА/КАЧЕСТВО: дозабор фазы 3 до сходимости ===")
    print("эталон достижимого = ПОТОЛОК ФАЗЫ 3 (полная свобода); "
          "опыты считаются ПО ВЕТКЕ (seed общий, не входит)")
    print(f"{'ветка':<9}|{'ceil ф3':>8}|{'d3→final':>16}|{'%ceil':>6}|"
          f"{'оп.вет':>7}|{'80%':>5}|{'90%':>5}|{'95%':>5}|{'стоп':>10}")
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
            return "—" if v is None else str(v)
        print(f"{bid:<9}|{c3:>8.3f}|{d3[bid]:>6.2f}→{br.d_best:<9.3f}|{pct:>5.0f}%|"
              f"{br.spent:>7}|{_f(c80):>5}|{_f(c90):>5}|{_f(c95):>5}|"
              f"{stop_reason[bid]:>10}")
    total2 = seed_n + sum(runner.branches[bid].spent for bid in goals)
    print(f"всего измерений в общей базе после дозабора: {total2} "
          f"(seed {seed_n} + ветки {total2 - seed_n})")

    # --- ИТОГ ПОСЛЕ ДОЗАБОРА: рецепты и свойства pipeline x* vs аналитика ---
    print("ИТОГ рецепты (доли A,B,C + код T,P), pipeline x* vs аналитика x*:")
    for bid in goals:
        xb = np.round(np.asarray(runner.branches[bid].x_best, float), 3)
        xo = np.round(np.asarray(opt[bid]["x"], float), 3)
        print(f"  {bid:<9} pipeline={xb.tolist()}  analytic={xo.tolist()}")
    print("ИТОГ свойства (истина) pipeline vs аналитика; Δ ЦЕНА — отклонение по цене:")
    for bid in goals:
        xb = np.asarray(runner.branches[bid].x_best, float).reshape(1, -1)
        yp = {p: float(truth.truths[p].true(xb)[0]) for p in truth.property_names}
        ya = {p: float(opt[bid]["y"][p]) for p in truth.property_names}
        dpa = yp["price"] - ya["price"]
        dpp = 100.0 * dpa / abs(ya["price"]) if abs(ya["price"]) > 1e-9 else 0.0
        print(f"  {bid:<9} ЦЕНА pipe={yp['price']:.2f} vs anal={ya['price']:.2f} "
              f"Δ={dpa:+.2f} ({dpp:+.0f}%) | "
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
