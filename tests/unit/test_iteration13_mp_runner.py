"""Iteration 13 (боевой тест) — механика MixtureProcessRunner.

Проверяем канон §5/§12/§13.8 на уровне механики (без проверки сходимости — она
в боевом тесте шага 4):
  * маска свободы: на ранней фазе «закрытые» координаты держатся на baseline,
    при полном раскрытии — варьируются; Σx=1 сохраняется всегда;
  * раунд ветки растит ОБЩУЮ базу, меряет ВСЕ P свойств, ставит origin-тег;
  * одна модель проекта: словарь суррогатов на ВСЕ свойства (GP);
  * бюджет ветки уважается; ветка — контейнер без своей модели.
"""
import numpy as np

from src.core.schema import ProjectSchema, VariableBlock, ModelSpec
from src.design.block_model import build_model_terms
from src.optimize.desirability import DesirabilitySpec
from src.verification.mixture_process_truth import MultiMixtureProcessTruth
from src.apps.mixture_process_runner import MixtureProcessRunner


def _schema():
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    return ProjectSchema.mixture_process(mix, proc, model=model)


def _oracle(schema, seed=0):
    p = build_model_terms(schema).p
    rng = np.random.default_rng(seed)
    return MultiMixtureProcessTruth(
        schema, {"p0": rng.normal(size=p), "p1": rng.normal(size=p)},
        noise_sd=0.0)


def _runner():
    schema = _schema()
    return MixtureProcessRunner(schema, _oracle(schema), seed=1, n_restarts=3)


def test_phase_mask_holds_closed_coords_at_baseline():
    r = _runner()
    # фаза 1: свободны только A,B; C и весь process — на baseline
    r.set_free(mixture_free=["A", "B"], process_free=[])
    cands = r._masked_candidates(40, seed=5)
    assert np.allclose(cands[:, :3].sum(axis=1), 1.0, atol=1e-9)   # Σx=1
    assert np.allclose(cands[:, 2], 1.0 / 3.0)                     # C фикс
    assert np.allclose(cands[:, 3], 0.5) and np.allclose(cands[:, 4], 0.5)
    assert cands[:, 0].std() > 0.05 and cands[:, 1].std() > 0.05   # A,B варьируют

    # полное раскрытие: всё варьирует
    r.set_free(mixture_free=["A", "B", "C"], process_free=["T", "P"])
    full = r._masked_candidates(40, seed=6)
    assert full[:, 2].std() > 0.05 and full[:, 3].std() > 0.05
    assert np.allclose(full[:, :3].sum(axis=1), 1.0, atol=1e-9)


def test_seed_and_branch_round_grow_shared_base_measure_all_P():
    r = _runner()
    r.set_free(mixture_free=["A", "B"], process_free=[])
    r.seed_initial(n=16, seed=2)
    n0 = len(r.X)
    assert r.Y.shape == (n0, 2)
    assert set(r.surrogates) == {"p0", "p1"}

    br = r.add_branch("opt", {"p0": DesirabilitySpec("max", low=-5, high=5)},
                      budget=4, satisfy_at=2.0)   # недостижимо → не "satisfied"
    out = r.run_branch_round(br.id, n_points=2, n_candidates=200)
    assert out["added"] == 2
    assert len(r.X) == n0 + 2
    assert r.Y.shape == (n0 + 2, 2)               # ВСЕ P измерены
    assert out["y_new"].shape == (2, 2)
    assert r.origin[-2:] == [f"branch:{br.id}"] * 2
    assert r.origin_counts()[f"branch:{br.id}"] == 2
    assert br.spent == 2 and br.remaining() == 2
    assert not hasattr(br, "model") and not hasattr(br, "surrogate")
    # рецепт ветки — валидная составная точка
    xb = np.asarray(br.x_best, float)
    assert np.isclose(xb[:3].sum(), 1.0, atol=1e-6)


def test_branch_budget_respected():
    r = _runner()
    r.seed_initial(n=14, seed=3)
    br = r.add_branch("opt", {"p1": DesirabilitySpec("max", low=-5, high=5)},
                      budget=3, satisfy_at=2.0)
    r.run_branch_round(br.id, n_points=2, n_candidates=150)   # тратит 2
    out = r.run_branch_round(br.id, n_points=2, n_candidates=150)  # остаётся 1
    assert out["added"] == 1
    assert br.remaining() == 0 and br.status == "exhausted"
    out3 = r.run_branch_round(br.id, n_points=2)
    assert out3["added"] == 0
