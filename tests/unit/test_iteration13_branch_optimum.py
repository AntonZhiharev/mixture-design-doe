"""Iteration 13 (боевой тест) — аналитический оптимум ветки (эталон сходимости).

Проверяем :func:`branch_optimum` (REBUILD_SPEC §8/§5/§12):
  * чистая линейная цель «max A» → оптимум в вершине A (x≈[1,0,0]), d≈1;
  * «спор» двух целей (max A и max B одинаково) → ВНУТРЕННИЙ оптимум A≈B≈0.5,
    C≈0, overall-desirability ≈0.5 (ни одну цель не выжать на максимум);
  * процессная цель (отклик = 10·T) → оптимум по коду процесса T≈1.
"""
import numpy as np

from src.core.schema import (ProjectSchema, VariableBlock, ModelSpec)
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.verification.branch_reference import branch_optimum, branch_optimum_masked



def _mix_only():
    return ProjectSchema.mixture_only(
        ["A", "B", "C"],
        model=ModelSpec(mixture_order="quadratic", cross_level="additive"))


def _mix_proc():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def test_linear_goal_optimum_at_vertex():
    schema = _mix_only()
    # термы quad: [A, B, C, A*B, A*C, B*C]; отклик = 10*A
    truth = MultiMixtureProcessTruth(schema,
                                     {"strength": [10, 0, 0, 0, 0, 0]})
    goal = {"strength": DesirabilitySpec("max", low=0.0, high=10.0)}
    opt = branch_optimum(truth, goal, n_scan=8000, seed=1)
    assert opt["x"][0] > 0.9       # вершина A
    assert opt["d"] > 0.9


def test_internal_optimum_under_tension():
    schema = _mix_only()
    truth = MultiMixtureProcessTruth(schema, {
        "strength": [10, 0, 0, 0, 0, 0],   # тянет к A=1
        "gloss":    [0, 10, 0, 0, 0, 0]})  # тянет к B=1
    goal = {"strength": DesirabilitySpec("max", low=0.0, high=10.0),
            "gloss":    DesirabilitySpec("max", low=0.0, high=10.0)}
    opt = branch_optimum(truth, goal, n_scan=12000, seed=2)
    x = opt["x"]
    assert abs(x[0] - 0.5) < 0.12 and abs(x[1] - 0.5) < 0.12
    assert x[2] < 0.12             # C выдавлен
    assert 0.35 < opt["d"] < 0.9   # компромисс: ни A, ни B не на максимуме


def test_process_goal_optimum():
    schema = _mix_proc()
    terms = build_model_terms(schema)
    coef = np.zeros(terms.p)
    coef[terms.names.index("T")] = 10.0     # отклик = 10*T (код процесса)
    truth = MultiMixtureProcessTruth(schema, {"cure": coef})
    goal = {"cure": DesirabilitySpec("max", low=0.0, high=10.0)}
    opt = branch_optimum(truth, goal, n_scan=12000, seed=3)
    # составные координаты [A,B,C,T,P]; T — индекс q=3
    assert opt["x"][3] > 0.85
    assert opt["d"] > 0.9


def test_masked_phase_ceiling_below_global_without_process():
    """Потолок фазы под маской: без свободы T достижимое НИЖЕ глобального
    оптимума (T заперт на baseline), а с T — почти дотягивает (REBUILD_SPEC §14)."""
    schema = _mix_proc()
    terms = build_model_terms(schema)
    coef = np.zeros(terms.p)
    coef[terms.names.index("T")] = 10.0          # отклик = 10*T
    truth = MultiMixtureProcessTruth(schema, {"cure": coef})
    goal = {"cure": DesirabilitySpec("max", low=0.0, high=10.0)}
    base = [1/3, 1/3, 1/3, 0.5, 0.5]             # T заперт на 0.5 → отклик 5 → d=0.5

    g = branch_optimum(truth, goal, n_scan=12000, seed=3)["d"]
    no_t = branch_optimum_masked(truth, goal, baseline=base,
                                 mixture_free=["A", "B"], process_free=[],
                                 n_scan=20000, seed=5)["d"]
    with_t = branch_optimum_masked(truth, goal, baseline=base,
                                   mixture_free=["A", "B"], process_free=["T"],
                                   n_scan=20000, seed=6)["d"]
    assert abs(no_t - 0.5) < 0.02                # T=0.5 → ровно половина шкалы
    assert with_t > no_t + 0.3                   # свобода T резко поднимает потолок
    assert with_t <= g + 1e-6                    # но не выше глобального оптимума
    assert with_t > 0.9 * g                      # и почти дотягивает до него


