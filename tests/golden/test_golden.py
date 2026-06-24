# Copyright 2026 AZH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Golden-тесты против R-эталона (REBUILD_SPEC §6, офлайн-фикстуры).

Сверяют продакшн-модули с закоммиченными дампами `src/verification/golden/*.json`.
R при прогоне НЕ требуется (фикстуры сгенерированы независимым эталоном
`src/verification/reference.py`; R-сниппеты происхождения — в поле `r_reference`).
Регенерация: `python -m src.verification.generate_golden`.
"""
import numpy as np

from src.core.linalg import (scheffe_matrix, d_criterion, d_efficiency,
                             i_criterion)
from src.models.scheffe import ScheffeModel
from src.models.clustering import GMMRegimes
from src.models.gp_expert import GPExpert
from src.optimize.desirability import (DesirabilitySpec, desirability_value,
                                       Desirability)
from src.verification.golden_io import load_fixture, arr, TOLERANCES


# ----------------------------------------------------------------------
def test_scheffe_ols_coefficients():
    fx = load_fixture("scheffe_ols")
    X, y = arr(fx["inputs"]["X"]), arr(fx["inputs"]["y"])
    exp = fx["expected"]
    tol = TOLERANCES[fx["tol"]]

    m = ScheffeModel(model="quadratic").fit(X, y)
    np.testing.assert_allclose(m.coefficients, arr(exp["coefficients"]), atol=tol)
    np.testing.assert_allclose(m.fitted_values, arr(exp["fitted"]), atol=tol)
    assert abs(m.r2 - exp["r2"]) < tol
    assert abs(m.rmse - exp["rmse"]) < tol
    assert abs(m.adj_r2 - exp["adj_r2"]) < tol


# ----------------------------------------------------------------------
def test_d_optimality():
    fx = load_fixture("d_optimality")
    X = arr(fx["inputs"]["X"])
    M = scheffe_matrix(X, "quadratic")
    exp = fx["expected"]

    np.testing.assert_allclose(d_criterion(M), exp["d_criterion"],
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(d_efficiency(M), exp["d_efficiency"],
                               rtol=1e-9, atol=1e-12)


# ----------------------------------------------------------------------
def test_i_optimality():
    fx = load_fixture("i_optimality")
    X = arr(fx["inputs"]["X"])
    moments = arr(fx["inputs"]["moments"])
    M = scheffe_matrix(X, "quadratic")
    exp = fx["expected"]

    np.testing.assert_allclose(i_criterion(M, moments), exp["i_criterion"],
                               rtol=1e-9, atol=1e-12)


# ----------------------------------------------------------------------
def test_desirability():
    fx = load_fixture("desirability")
    inp, exp = fx["inputs"], fx["expected"]
    tol = TOLERANCES[fx["tol"]]

    s_max = DesirabilitySpec("max", low=inp["max"]["low"], high=inp["max"]["high"],
                             s=inp["max"]["s"], weight=inp["max"]["weight"])
    s_min = DesirabilitySpec("min", low=inp["min"]["low"], high=inp["min"]["high"],
                             s=inp["min"]["s"], weight=inp["min"]["weight"])
    s_tgt = DesirabilitySpec("target", low=inp["target"]["low"],
                             high=inp["target"]["high"], target=inp["target"]["target"],
                             s=inp["target"]["s"], s2=inp["target"]["s2"],
                             weight=inp["target"]["weight"])

    d_max = desirability_value(arr(inp["max"]["y"]), s_max)
    d_min = desirability_value(arr(inp["min"]["y"]), s_min)
    d_tgt = desirability_value(arr(inp["target"]["y"]), s_tgt)
    np.testing.assert_allclose(d_max, arr(exp["d_max"]), atol=tol)
    np.testing.assert_allclose(d_min, arr(exp["d_min"]), atol=tol)
    np.testing.assert_allclose(d_tgt, arr(exp["d_target"]), atol=tol)

    # overall via Desirability aggregator (weighted geometric mean)
    desir = Desirability({"P": s_max, "C": s_min, "T": s_tgt})
    overall = desir.overall({"P": arr(inp["max"]["y"]),
                             "C": arr(inp["min"]["y"]),
                             "T": arr(inp["target"]["y"])})
    np.testing.assert_allclose(overall, arr(exp["d_overall"]), atol=tol)


# ----------------------------------------------------------------------
def test_gp_fixed_hyperparameters():
    fx = load_fixture("gp_fixed_hyper")
    st = fx["inputs"]["state"]
    exp = fx["expected"]
    tol = TOLERANCES[fx["tol"]]

    # Reconstruct GPExpert with the fixed kernel theta (optimizer=None path).
    sch = dict(st["scheffe"])
    sch["coefficients"] = arr(sch["coefficients"])
    state = {
        "mean_model": st["mean_model"], "kernel": st["kernel"],
        "noise_floor": st["noise_floor"], "names": st["names"],
        "scheffe": sch,
        "kernel_theta": arr(st["kernel_theta"]),
        "X_train": arr(st["X_train"]), "resid_train": arr(st["resid_train"]),
    }
    gp = GPExpert.from_state(state)
    pred = gp.predict(arr(fx["inputs"]["X_test"]), return_std=True)

    np.testing.assert_allclose(pred.mean, arr(exp["mean"]), atol=tol, rtol=tol)
    np.testing.assert_allclose(pred.std, arr(exp["std"]), atol=tol, rtol=tol)


# ----------------------------------------------------------------------
def test_gmm_regimes_partition():
    fx = load_fixture("gmm_regimes")
    Y = arr(fx["inputs"]["Y"])
    groups = np.asarray(fx["expected"]["groups"], dtype=int)
    exp_k = fx["expected"]["n_regimes"]

    res = GMMRegimes(k_range=fx["inputs"]["k_range"],
                     seed=fx["inputs"]["seed"]).fit(Y)
    assert res.n_regimes == exp_k

    # Responsibilities are a valid distribution.
    np.testing.assert_allclose(res.responsibilities.sum(axis=1),
                               np.ones(len(Y)), atol=1e-8)

    # Partition matches the expected grouping up to label permutation:
    # every pair in the same expected group shares a predicted label, and
    # pairs across expected groups do not.
    labels = res.labels
    for i in range(len(Y)):
        for j in range(i + 1, len(Y)):
            same_expected = groups[i] == groups[j]
            same_pred = labels[i] == labels[j]
            assert same_expected == same_pred


# ----------------------------------------------------------------------
def test_all_fixtures_have_provenance():
    """Каждая фикстура несёт описание и R-сниппет происхождения."""
    for name in ("scheffe_ols", "d_optimality", "i_optimality",
                 "desirability", "gp_fixed_hyper", "gmm_regimes"):
        fx = load_fixture(name)
        assert fx.get("description")
        assert fx.get("r_reference")
