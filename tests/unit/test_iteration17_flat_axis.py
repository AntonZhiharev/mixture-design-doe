# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 17 / §15.6 ШАГ 6 — A0.7: вырожденная (flat) ось → objective-gap.

Аксиома A0.7: когда оптимум упёрся в границу оси, нельзя слепо репортить «x-gap»
(на сколько подвинуть переменную) — если целевая функция ПЛОСКАЯ по этой оси
(∂d/∂x≡0), переменная неидентифицируема, двигать её бессмысленно. Тогда
репортится OBJECTIVE-GAP (Δd за границей), а ось помечается flat. Эталон:
economy/P, spread = 0.00e+00.

Покрывает:
  * detect_flat_axis — flat-ось: spread≈0, flat=True, репорт objective-gap,
    x_gap=None (двигать нечего); НЕ-flat ось: spread>tol, flat=False, x_gap есть;
  * §3 «Следствие»: ось, ПЛОСКАЯ без ρ, ПЕРЕСТАЁТ быть плоской с ρ(...,P) в цене
    — детектор переоценивает flat-статус на ТЕКУЩЕЙ desirability (с ценой/ρ);
  * Runner.flat_axis_at_border — связка с реальной Desirability ветки (A0.7).
"""
import numpy as np
import pytest

from src.optimize.desirability import (Desirability, DesirabilitySpec,
                                        make_item_cost_fn)
from src.optimize.economic_stop import (detect_flat_axis, FlatAxisResult,
                                         axis_spread)


# ----------------------------------------------------------------------
# Чистый детектор: эталон A0.7 — цель НЕ зависит от оси → flat, objective-gap.
# ----------------------------------------------------------------------
def test_detect_flat_axis_reports_objective_gap_not_x_gap():
    # economy/P-эталон: d не зависит от P (∂d/∂P≡0) → spread = 0.00e+00.
    def objective_fn(ts):
        ts = np.atleast_1d(np.asarray(ts, float))
        return np.full(ts.shape, 0.42)            # плоская по оси

    samples = np.linspace(0.0, 1.0, 21)
    res = detect_flat_axis("P", objective_fn, samples,
                           border_value=1.0, beyond_value=2.0)
    assert isinstance(res, FlatAxisResult)
    assert res.flat is True and res.identifiable is False
    assert res.spread == pytest.approx(0.0, abs=1e-12)   # spread=0.00e+00
    # цель не зависит от оси → за границей d тоже не растёт ⇒ objective_gap≈0,
    # но x_gap ИГНОРИРУЕТСЯ (None) — двигать переменную нечего (A0.7).
    assert res.x_gap is None
    assert res.objective_gap == pytest.approx(0.0, abs=1e-12)


def test_detect_non_flat_axis_keeps_x_gap_and_normal_path():
    # d РЕАЛЬНО растёт с осью → ось различима, обычный путь (x-gap имеет смысл).
    def objective_fn(ts):
        ts = np.atleast_1d(np.asarray(ts, float))
        return 0.2 + 0.6 * ts                     # монотонно растёт

    samples = np.linspace(0.0, 1.0, 21)
    res = detect_flat_axis("C", objective_fn, samples,
                           border_value=1.0, beyond_value=1.5)
    assert res.flat is False and res.identifiable is True
    assert res.spread == pytest.approx(0.6, abs=1e-9)    # > tol
    assert res.x_gap == pytest.approx(0.5)               # beyond − border
    # за границей d продолжает расти ⇒ objective_gap > 0 (справочно)
    assert res.objective_gap == pytest.approx(0.3, abs=1e-9)


def test_axis_spread_helper():
    assert axis_spread([0.5, 0.5, 0.5]) == pytest.approx(0.0)
    assert axis_spread([0.1, 0.9, 0.4]) == pytest.approx(0.8)
    assert axis_spread([]) == 0.0



# ----------------------------------------------------------------------
# §3 «Следствие»: flat БЕЗ ρ → НЕ flat С ρ(...,P) (price_изд как свойство).
# ----------------------------------------------------------------------
def test_flat_axis_reevaluated_when_rho_enters_cost():
    # Цель strength НЕ зависит от P (та же истина) — ось P плоская по strength.
    # Вводим ρ(...,P): price_изд = price_состав·ρ зависит от P → ось различима.
    goal = {"strength": DesirabilitySpec("max", low=0.0, high=10.0)}

    # 4 столбца [A, B, C, P]; оптимум фиксируем, варьируем только P (индекс 3).
    x_opt = np.array([0.5, 0.3, 0.2, 0.0])

    def _set_P(ts):
        ts = np.atleast_1d(np.asarray(ts, float))
        X = np.tile(x_opt, (ts.size, 1))
        X[:, 3] = ts
        return X

    # strength не зависит от P (константа по оси):
    strength_pred = lambda X: np.full(np.atleast_2d(X).shape[0], 6.0)

    # --- БЕЗ ρ: только strength → ось P плоская -----------------------
    desir_no_rho = Desirability(goal)

    def obj_no_rho(ts):
        X = _set_P(ts)
        return desir_no_rho.overall({"strength": strength_pred(X)})

    samples = np.linspace(0.0, 1.0, 21)
    res_no = detect_flat_axis("P", obj_no_rho, samples,
                              border_value=1.0, beyond_value=2.0)
    assert res_no.flat is True                    # без ρ ось вырождена

    # --- С ρ(...,P): price_изд тянет цель → ось ПЕРЕСТАЁт быть плоской --
    comp_price = lambda X: np.full(np.atleast_2d(X).shape[0], 2.0)   # ₽/кг
    rho_pred = lambda X: 0.5 + 0.4 * np.atleast_2d(X)[:, 3]          # ρ растёт с P
    cost_fn = make_item_cost_fn(comp_price, rho_pred)

    specs = dict(goal)
    specs["price"] = DesirabilitySpec("min", low=0.0, high=3.0)
    desir_rho = Desirability(specs)

    def obj_rho(ts):
        X = _set_P(ts)
        props = {"strength": strength_pred(X),
                 "price": np.asarray(cost_fn(X), float).ravel()}
        return desir_rho.overall(props)

    res_rho = detect_flat_axis("P", obj_rho, samples,
                               border_value=1.0, beyond_value=2.0)
    assert res_rho.flat is False                  # с ρ(...,P) ось различима
    assert res_rho.spread > 1e-6
    assert res_rho.x_gap is not None              # обычный путь снова уместен


# ----------------------------------------------------------------------
# Runner-связка: flat_axis_at_border строит objective_fn из реальной
# Desirability ветки по общим суррогатам (A0.7, §3 «Следствие»).
# ----------------------------------------------------------------------
def _runner_with_flat_process():
    """Раннер mixture(A,B,C)×process(P), где strength НЕ зависит от P."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
    from src.verification.mixture_process_truth import MultiMixtureProcessTruth
    from src.apps.mixture_process_runner import MixtureProcessRunner
    from src.design.block_model import build_model_terms

    mix = VariableBlock.mixture(["A", "B", "C"], lower=[0.0, 0.0, 0.0],
                                upper=[1.0, 1.0, 1.0])
    proc = VariableBlock.process(["P"], [0.0], [1.0])
    model = ModelSpec(mixture_order="quadratic", process_order="linear")
    s = ProjectSchema.mixture_process(mix, proc, model=model)
    terms = build_model_terms(s)
    # strength = чистый mixture-отклик (нет P-членов в коэффициентах) → flat по P
    coef = np.zeros(terms.p)
    coef[0] = 8.0          # β_A
    coef[1] = 4.0          # β_B
    coef[2] = 2.0          # β_C
    truth = MultiMixtureProcessTruth(s, {"strength": coef}, noise_sd=0.0)
    r = MixtureProcessRunner(s, truth, seed=0, n_restarts=2)
    r.seed_initial(n=14, seed=0)
    return r


def test_runner_flat_axis_at_border_process_flat():
    r = _runner_with_flat_process()
    br = r.add_branch("eco", {"strength": DesirabilitySpec("max", low=0.0,
                                                           high=10.0)}, budget=6)
    res = r.flat_axis_at_border(br.id, "P", "upper", new_bound=1.0,
                                n_samples=11)
    # strength не зависит от P → ось P вырождена (objective-gap, не x-gap)
    assert res.var == "P"
    assert res.flat is True
    assert res.x_gap is None
    assert res.spread == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# PipelineRunner (mixture-only) — flat_axis_mixture для UI (ШАГ 7).
# ----------------------------------------------------------------------
def _pipeline_runner_flat_comp():
    """Mixture-only PipelineRunner, где целевое свойство НЕ зависит от доли C."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from src.apps.pipeline_runner import PipelineConfig, PipelineRunner

    cfg = PipelineConfig(name="flat", q=3, names=["A", "B", "C"],
                         model="quadratic", noise_sd=0.0, seed=0,
                         property_names=["strength"], n_restarts=2)
    import tempfile
    r = PipelineRunner(cfg, tempfile.mkdtemp())
    r.run_m1(); r.run_m2()
    # отклики: strength зависит ТОЛЬКО от A,B (C-столбец не влияет)
    X = np.asarray(r.design, float)
    y = 8.0 * X[:, 0] + 4.0 * X[:, 1]
    r.Y = y.reshape(-1, 1)
    r.y = r.Y[:, 0]
    r.run_m6()
    return r


def test_pipeline_flat_axis_mixture_detects_flat_C():
    r = _pipeline_runner_flat_comp()
    br = r.add_branch("eco", {"strength": DesirabilitySpec("max", low=0.0,
                                                           high=10.0)}, budget=6)
    res = r.flat_axis_mixture(br.id, "A", "upper", new_bound=1.0, n_samples=11)
    # A РЕАЛЬНО двигает strength → ось различима (обычный путь, x-gap есть)
    assert res.identifiable is True and res.x_gap is not None

