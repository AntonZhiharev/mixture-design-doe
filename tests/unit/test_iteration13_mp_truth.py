"""Iteration 13 (боевой тест, фундамент) — синтетическая истина mixture×process.

Проверяем «эталон» для боевого бенчмарка (REBUILD_SPEC §8/§13):
  * число коэффициентов == числу термов модели; контроль формы (q, d);
  * на mixture-only схеме истина бит-в-бит совпадает с доверенным
    :class:`SyntheticScheffe` (инвариант §13.9): одни коэффициенты → один отклик;
  * кросс-члены x·z реально влияют на отклик (process не «нейтрален»);
  * мульти-вариант мерит все P свойств, «цена» — отдельное свойство;
  * :func:`composite_random_points` даёт валидные точки (Σx=1, z∈[0,1]).
"""
import numpy as np
import pytest

from src.core.linalg import scheffe_term_indices
from src.core.schema import (ProjectSchema, VariableBlock, ModelSpec,
                             split_composite)
from src.core.synthetic import SyntheticScheffe
from src.design.block_model import build_model_terms
from src.verification.mixture_process_truth import (
    MixtureProcessTruth, MultiMixtureProcessTruth, composite_random_points)


def _mp_schema(cross="full-cross"):
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level=cross, mixture_order="quadratic",
                      process_order="quadratic",
                      main_components=(0, 1, 2) if cross == "cross-main" else ())
    return ProjectSchema.mixture_process(mix, proc, model=model)


def test_coeff_count_must_match_terms():
    schema = _mp_schema()
    p = build_model_terms(schema).p
    with pytest.raises(ValueError):
        MixtureProcessTruth(schema, np.ones(p + 1))
    truth = MixtureProcessTruth(schema, np.ones(p))
    assert truth.q == 3 and truth.d == 2


def test_mixture_only_matches_trusted_scheffe():
    # mixture-only схема + те же коэффициенты → бит-в-бит SyntheticScheffe (§13.9)
    q = 3
    idx = scheffe_term_indices(q, "quadratic")
    coef = np.arange(1, len(idx) + 1, dtype=float)
    ss = SyntheticScheffe(q, model="quadratic", coefficients=coef)
    schema = ProjectSchema.mixture_only(
        ["A", "B", "C"],
        model=ModelSpec(mixture_order="quadratic", cross_level="additive"))
    truth = MixtureProcessTruth(schema, coef)
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(q), size=20)
    assert np.allclose(truth.true(X), ss.true(X))


def test_cross_terms_make_process_matter():
    schema = _mp_schema("full-cross")
    terms = build_model_terms(schema)
    coef = np.zeros(terms.p)
    # включим только кросс-члены x·z: тогда отклик обязан зависеть от z
    for t, cat in enumerate(terms.categories):
        if cat == "cross":
            coef[t] = 5.0
    truth = MixtureProcessTruth(schema, coef)
    x = np.array([0.4, 0.3, 0.3])
    lo = truth.true(np.hstack([x, [0.0, 0.0]]).reshape(1, -1))
    hi = truth.true(np.hstack([x, [1.0, 1.0]]).reshape(1, -1))
    assert not np.isclose(lo[0], hi[0])


def test_multi_truth_shapes_and_price():
    schema = _mp_schema()
    p = build_model_terms(schema).p
    rng = np.random.default_rng(1)
    coef_by = {"strength": rng.normal(size=p),
               "gloss": rng.normal(size=p),
               "dry_time": rng.normal(size=p),
               "price": rng.normal(size=p)}
    truth = MultiMixtureProcessTruth(schema, coef_by, noise_sd=0.0)
    Xc = composite_random_points(schema, 15, seed=3)
    Y = truth.true(Xc)
    assert Y.shape == (15, 4)
    assert truth.n_properties == 4
    assert truth.property_names[-1] == "price"


def test_composite_random_points_valid():
    schema = _mp_schema()
    Xc = composite_random_points(schema, 50, seed=7)
    assert Xc.shape == (50, 5)
    # mixture суммируется в 1, process в коде [0,1]
    assert np.allclose(Xc[:, :3].sum(axis=1), 1.0, atol=1e-6)
    assert (Xc[:, 3:] >= -1e-9).all() and (Xc[:, 3:] <= 1.0 + 1e-9).all()
    blocks = split_composite(schema, Xc[0])
    assert set(blocks) == {"MIXTURE", "PROCESS"}
