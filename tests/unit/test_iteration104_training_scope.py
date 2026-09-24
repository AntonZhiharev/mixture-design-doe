# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 104 — область обучения суррогата и проектный гейт (§16.2.1.5).

Находка iter103: после ``move_region`` (restrict к кромке дыры) суррогат
гейта учится на активном пуле и теряет провалившиеся точки — весь
отрицательный класс; точность ``P(измеримо)`` падает 1.00 → 0.79, а добор с
«послерестриктным» множителем даёт 4 из 6 вместо 6 из 6. Рабочий приём был —
строить множитель ДО сужения и держать его в переменной; в UI (reruns
Streamlit) так не получится.

Решение: ОБЛАСТЬ ОБУЧЕНИЯ по отклику (``set_training_scope``: ``active`` —
активный пул, как прежде; ``history`` — вся история в координатах текущей
схемы). Объявление гейта (``set_project_gate`` / ``set_branch_gate``)
переводит гейт-отклик на историю: гейт описывает физику измеримости, а не
область интереса. Для остальных откликов — явный выбор технолога, подсказка
— ``training_scope_diagnostics`` (LOO на активных точках: помогает ли
история или снаружи другой режим).

Проверяем:
  * с проектным гейтом суррогат гейта ПОСЛЕ restrict не забывает дыру:
    точность на пуле кандидатов та же, что до сужения, а
    ``gate_feasibility_fn`` после сужения даёт добор 6 из 6 измеримых;
  * без объявления гейта — прежнее поведение (факт iter103 воспроизводится);
  * ``X``/``Y`` (активный пул) не зависят от области обучения;
  * без сужения ``history`` ≡ ``active`` бит-в-бит;
  * контракты сеттеров, ``surrogate_coverage`` (``scope``/``n_base``);
  * ``propose_seed(feasibility=None)`` берёт множитель из проектного гейта,
    ``feasibility=False`` — явно без него; необученный гейт — отказ;
  * ``effective_branch_gate``: собственный > проектный > None;
  * LOO-диагностика: та же физика снаружи → ``history_helps=True``; смена
    режима снаружи → ``False``; без внешних точек → ``None``; закрытая
    форма LOO совпадает с честным refit при фиксированных гиперпараметрах;
  * персистентность через ``campaign_state``; старый сейв без ключей.
"""
import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps.mixture_process_runner import (SCOPE_ACTIVE, SCOPE_HISTORY,
                                             MixtureProcessRunner)
from src.design.branches import edge_region
from src.design.move_bounds import MOVE_RESTRICT
from src.optimize.desirability import DesirabilitySpec
from src.verification.battle_truth import (CLIFF_RESPONSE, GATE_3COMP,
                                           GATE_THRESHOLD_3COMP, GATED_3COMP,
                                           TornLab, build_truth_3comp_cliff,
                                           model_schema_3comp)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

THR = GATE_THRESHOLD_3COMP
GATED = tuple(GATED_3COMP) + (CLIFF_RESPONSE,)
N_SEED, N_AUG, EDGE_WIDTH = 20, 6, 0.3


def _full(r, X):
    return np.vstack([r._to_full(x) for x in np.atleast_2d(X)])


def _screened(seed=7):
    truth = build_truth_3comp_cliff()
    lab = TornLab(truth, gated=GATED)
    r = MixtureProcessRunner(model_schema_3comp(), lab,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                             seed=seed, n_restarts=2)
    X0 = np.asarray(r.propose_seed(N_SEED, seed=seed), float)
    Y0 = lab.evaluate(_full(r, X0))
    r.commit_seed(X0, Y0, missing_reasons=lab.reasons(Y0))
    return truth, lab, r


def _edge_box(r, seed=101):
    cands = r._phase_candidates(2000, seed=seed)
    names = (list(r.current_schema.mixture_names)
             + list(r.current_schema.process_names))
    box = edge_region(r.surrogates[GATE_3COMP], THR, cands, names,
                      width=EDGE_WIDTH, margin=0.05)
    return {k: v for k, v in box.items()
            if k in r.current_schema.mixture_names}


def _gate_dist(truth, r, X):
    return truth.gate_true(_full(r, X)) - THR


def _accuracy(truth, r, f, pool):
    truth_ok = _gate_dist(truth, r, pool) >= 0
    return float(np.mean((f(pool) >= 0.5) == truth_ok))


# ======================================================================
# 1. Живой замер: гейт на истории не забывает дыру после restrict
# ======================================================================
@pytest.fixture(scope="module")
def narrowed_pair():
    """Два раннера одного мира и сида: БЕЗ гейта (область active, факт
    iter103) и С проектным гейтом (область history). У обоих — restrict к
    одному и тому же боксу кромки."""
    truth, lab, r_act = _screened(seed=7)
    _, _, r_hist = _screened(seed=7)
    box = _edge_box(r_act)
    f_before = r_act.gate_feasibility_fn(GATE_3COMP, THR, "ge")
    r_hist.set_project_gate(GATE_3COMP, THR, "ge")
    mv1 = r_act.move_region(box, intent="edge_neighbourhood")
    mv2 = r_hist.move_region(box, intent="edge_neighbourhood")
    assert mv1.move_type == mv2.move_type == MOVE_RESTRICT
    return truth, lab, r_act, r_hist, f_before



def test_gate_on_history_keeps_the_hole_after_restrict(narrowed_pair):
    truth, lab, r_act, r_hist, f_before = narrowed_pair
    f_act = r_act.gate_feasibility_fn(GATE_3COMP, THR, "ge")
    f_hist = r_hist.gate_feasibility_fn(GATE_3COMP, THR, "ge")
    pool = r_hist._phase_candidates(600, 8)
    acc_before = _accuracy(truth, r_act, f_before, pool)
    acc_act = _accuracy(truth, r_act, f_act, pool)
    acc_hist = _accuracy(truth, r_hist, f_hist, pool)
    n_act = len(r_act._migrated_points())
    print(f"\n  gate GP accuracy on narrowed pool: before restrict "
          f"{acc_before:.2f} (n={len(r_act.points)}); after, scope=active "
          f"{acc_act:.2f} (n={n_act}); after, scope=history {acc_hist:.2f} "
          f"(n={len(r_hist.points)})")
    # факт iter103 воспроизводится: активная область забывает дыру
    assert n_act < len(r_act.points)
    assert acc_act < acc_before
    # история — нет: точность как у суррогата до сужения
    assert acc_hist >= acc_before - 1e-9
    assert acc_hist >= 0.95
    # обучающая выборка гейта на истории — вся база
    cov = r_hist.surrogate_coverage()[GATE_3COMP]
    assert cov["scope"] == SCOPE_HISTORY
    assert cov["n_base"] == len(r_hist.points) and cov["fitted"]
    assert "scope" not in r_act.surrogate_coverage()[GATE_3COMP]


def test_augment_with_post_restrict_multiplier_is_all_measurable(narrowed_pair):
    """Смысл для UI: множитель, построенный ПОСЛЕ сужения (как оно и будет
    между reruns), должен работать не хуже построенного до."""
    truth, lab, r_act, r_hist, f_before = narrowed_pair
    X_hist = r_hist.propose_augment(
        N_AUG, seed=8, feasibility=r_hist.project_feasibility_fn())
    d_hist = _gate_dist(truth, r_hist, X_hist)
    X_act = r_act.propose_augment(
        N_AUG, seed=8, feasibility=r_act.gate_feasibility_fn(GATE_3COMP, THR))
    d_act = _gate_dist(truth, r_act, X_act)
    print(f"  augment measurable: scope=history {int((d_hist >= 0).sum())}/"
          f"{N_AUG}, scope=active {int((d_act >= 0).sum())}/{N_AUG}")
    assert int((d_hist >= 0).sum()) == N_AUG
    assert d_hist.min() > -0.1
    assert int((d_hist >= 0).sum()) >= int((d_act >= 0).sum())
    # yield снимается на всех точках добора
    Y = lab.evaluate(_full(r_hist, X_hist))
    assert int(np.isfinite(Y[:, r_hist.prop_index[CLIFF_RESPONSE]]).sum()) \
        >= N_AUG - 1


def test_active_pool_arrays_do_not_depend_on_scope(narrowed_pair):
    truth, lab, r_act, r_hist, _ = narrowed_pair
    np.testing.assert_allclose(r_act.X, r_hist.X)
    np.testing.assert_allclose(r_act.Y, r_hist.Y, equal_nan=True)
    assert len(r_hist.X) == len(r_hist._migrated_points()) < len(r_hist.points)
    # зависимые отклики остались на активном пуле
    for name in r_hist.property_names:
        if name != GATE_3COMP:
            assert r_hist.training_scope(name) == SCOPE_ACTIVE
            assert "scope" not in r_hist.surrogate_coverage()[name]


def test_propose_seed_uses_project_gate_by_default(narrowed_pair):
    truth, lab, r_act, r_hist, _ = narrowed_pair
    f = r_hist.project_feasibility_fn()
    assert f is not None
    np.testing.assert_allclose(
        r_hist.propose_seed(N_AUG, seed=8),
        r_hist.propose_augment(N_AUG, seed=8, feasibility=f))
    # явно без множителя — голый maximin
    np.testing.assert_allclose(
        r_hist.propose_seed(N_AUG, seed=8, feasibility=False),
        r_hist.propose_augment(N_AUG, seed=8))
    # без проектного гейта — прежний путь бит-в-бит
    assert r_act.project_feasibility_fn() is None
    np.testing.assert_allclose(r_act.propose_seed(N_AUG, seed=8),
                               r_act.propose_augment(N_AUG, seed=8))


# ======================================================================
# 2. Контракты
# ======================================================================
def test_history_equals_active_without_restrict():
    truth, lab, r = _screened(seed=5)
    grid = r._phase_candidates(200, seed=3)
    mu0 = r.surrogates[GATE_3COMP].predict(grid).mean
    r.set_training_scope(GATE_3COMP, SCOPE_HISTORY)
    np.testing.assert_allclose(r.surrogates[GATE_3COMP].predict(grid).mean, mu0)
    cov = r.surrogate_coverage()[GATE_3COMP]
    assert cov["scope"] == SCOPE_HISTORY and cov["n_base"] == len(r.points)
    r.set_training_scope(GATE_3COMP, SCOPE_ACTIVE)
    assert r.training_scopes() == {n: SCOPE_ACTIVE for n in r.property_names}
    np.testing.assert_allclose(r.surrogates[GATE_3COMP].predict(grid).mean, mu0)


def test_scope_and_gate_setters_contracts():
    truth, lab, r = _screened(seed=5)
    with pytest.raises(KeyError, match="не среди свойств"):
        r.set_training_scope("нет_такого", SCOPE_HISTORY)
    with pytest.raises(ValueError, match="Область обучения"):
        r.set_training_scope(GATE_3COMP, "everything")
    with pytest.raises(KeyError, match="не среди свойств"):
        r.set_project_gate("нет_такого", 1.0)
    with pytest.raises(ValueError, match="direction"):
        r.set_project_gate(GATE_3COMP, 1.0, "between")
    assert r.project_gate() is None
    r.set_project_gate(GATE_3COMP, THR, "ge")
    assert r.project_gate() == {"response": GATE_3COMP, "threshold": THR,
                                "direction": "ge"}
    assert r.training_scope(GATE_3COMP) == SCOPE_HISTORY
    # снятие гейта область обучения не возвращает (отдельное решение)
    r.set_project_gate(None)
    assert r.project_gate() is None
    assert r.training_scope(GATE_3COMP) == SCOPE_HISTORY
    # гейт ветки тоже объявляет отклик гейтом
    r2 = _screened(seed=5)[2]
    r2.add_branch("b", {"strength": DesirabilitySpec("max", low=0, high=10)},
                  branch_id="b")
    assert r2.training_scope(GATE_3COMP) == SCOPE_ACTIVE
    r2.set_branch_gate("b", GATE_3COMP, THR, "ge")
    assert r2.training_scope(GATE_3COMP) == SCOPE_HISTORY


def test_effective_branch_gate_precedence():
    truth, lab, r = _screened(seed=5)
    r.add_branch("b", {"strength": DesirabilitySpec("max", low=0, high=10)},
                 branch_id="b")
    assert r.effective_branch_gate("b") is None
    assert r._branch_feasibility("b") is None
    r.set_project_gate(GATE_3COMP, THR, "ge")
    g = r.effective_branch_gate("b")
    assert g["source"] == "project" and g["response"] == GATE_3COMP
    assert r.branch_gate("b") is None                  # собственного нет
    assert r._branch_feasibility("b") is not None
    r.set_branch_gate("b", GATE_3COMP, THR + 1.0, "ge")
    g2 = r.effective_branch_gate("b")
    assert g2["source"] == "branch" and g2["threshold"] == THR + 1.0
    r.set_branch_gate("b", None)
    assert r.effective_branch_gate("b")["source"] == "project"
    with pytest.raises(KeyError, match="Нет ветки"):
        r.effective_branch_gate("nope")


def test_project_gate_without_measurements_refuses_multiplier():
    truth = build_truth_3comp_cliff()
    lab = TornLab(truth, gated=GATED)
    r = MixtureProcessRunner(model_schema_3comp(), lab,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                             seed=3, n_restarts=2)
    r.set_project_gate(GATE_3COMP, THR, "ge")           # база пуста — можно
    assert r.project_gate()["response"] == GATE_3COMP
    # пустая база: propose_seed идёт путём плана фазы, множитель не нужен
    X = r.propose_seed(4, seed=1)
    assert X.shape == (4, r.dim)
    # база есть, но гейт не измерен ни разу → отказ, не ≡ 1
    X0 = np.asarray(r.propose_seed(6, seed=2), float)
    Y0 = lab.evaluate(_full(r, X0))
    reasons = [dict(d or {}) for d in lab.reasons(Y0)]
    Y0[:, r.prop_index[GATE_3COMP]] = np.nan
    for d in reasons:
        d[GATE_3COMP] = "прибор оценки поверхности не подключён"
    r.commit_seed(X0, Y0, missing_reasons=reasons)
    with pytest.raises(RuntimeError, match="не обучен"):
        r.propose_seed(3, seed=1)
    # явный отказ от множителя пропускает
    assert r.propose_seed(3, seed=1, feasibility=False).shape == (3, r.dim)



# ======================================================================
# 3. LOO-диагностика: та же физика vs смена режима снаружи
# ======================================================================
class _Smooth:
    """Гладкая истина в общих координатах (A,B,C,T,P → код)."""
    property_names = ["y"]

    def __init__(self, shift_outside=0.0, edge=0.5):
        self.shift, self.edge = float(shift_outside), float(edge)

    def evaluate(self, Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        y = 3.0 * Xc[:, 0] + 1.5 * Xc[:, 1] - 2.0 * Xc[:, 3] + 0.5 * Xc[:, 4]
        # «другой режим» снаружи будущей области A ≤ edge: сдвиг + другая форма
        out = Xc[:, 0] > self.edge
        y = y + out * (self.shift + 4.0 * self.shift * (Xc[:, 0] - self.edge) ** 2)
        return y.reshape(-1, 1)


def _runner_smooth(oracle, seed=4):
    r = MixtureProcessRunner(model_schema_3comp(), oracle,
                             baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                             seed=seed, n_restarts=2)
    X = np.asarray(r.propose_seed(30, seed=seed), float)
    r.commit_seed(X, oracle.evaluate(_full(r, X)))
    return r


def test_loo_diagnostics_same_physics_outside_helps():
    r = _runner_smooth(_Smooth(shift_outside=0.0))
    d0 = r.training_scope_diagnostics("y")
    assert d0["history_helps"] is None and "вне области нет" in d0["note"]
    r.move_region({"A": (0.0, 0.5)}, intent="region_of_interest")
    d = r.training_scope_diagnostics("y")
    print(f"\n  LOO same physics: active {d['active']} history {d['history']}")
    assert d["n_outside"] > 0 and d["scope_now"] == SCOPE_ACTIVE
    assert d["history_helps"] is True
    assert "ЛУЧШЕ" in d["note"]


def test_loo_diagnostics_regime_change_outside_hurts():
    r = _runner_smooth(_Smooth(shift_outside=40.0))
    r.move_region({"A": (0.0, 0.5)}, intent="physical_constraint")
    d = r.training_scope_diagnostics("y")
    print(f"  LOO regime change: active {d['active']} history {d['history']}")
    assert d["n_outside"] > 0
    assert d["history_helps"] is False
    assert "ХУЖЕ" in d["note"]
    with pytest.raises(KeyError, match="не среди свойств"):
        r.training_scope_diagnostics("нет_такого")


def test_loo_scores_closed_form_matches_refit_on_fixed_hyperparameters():
    """Закрытая форма LOO = честный refit при ФИКСИРОВАННЫХ гиперпараметрах."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from src.models.gp_expert import GPExpert
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(3), size=14)
    y = 2 * X[:, 0] - X[:, 1] + 0.1 * rng.standard_normal(14)
    gp = GPExpert(mean_model="linear", n_restarts=1, seed=0).fit(X, y)
    logp, err2 = gp.loo_scores()
    assert logp.shape == err2.shape == (14,)
    r = gp._resid
    for i in (0, 5, 13):
        keep = np.delete(np.arange(14), i)
        g = GaussianProcessRegressor(kernel=gp.gp_.kernel_, alpha=0.0,
                                     optimizer=None, normalize_y=False
                                     ).fit(X[keep], r[keep])
        mu, sd = g.predict(X[[i]], return_std=True)
        assert (r[i] - mu[0]) ** 2 == pytest.approx(err2[i], rel=1e-5, abs=1e-9)
        lp = -0.5 * np.log(2 * np.pi * sd[0] ** 2) - err2[i] / (2 * sd[0] ** 2)
        assert lp == pytest.approx(logp[i], rel=1e-5, abs=1e-7)
    sub_lp, _ = gp.loo_scores([2, 3])
    np.testing.assert_allclose(sub_lp, logp[[2, 3]])


# ======================================================================
# 4. Персистентность
# ======================================================================
def test_state_roundtrip_keeps_project_gate_and_scopes(narrowed_pair):
    truth, lab, r_act, r_hist, _ = narrowed_pair
    r_hist.set_training_scope("strength", SCOPE_HISTORY)
    try:
        state = json.loads(json.dumps(cst.runner_to_state(r_hist),
                                      ensure_ascii=False))
        assert state["runner"]["project_gate"] == {
            "response": GATE_3COMP, "threshold": THR, "direction": "ge"}
        assert state["runner"]["training_scope"] == {
            GATE_3COMP: SCOPE_HISTORY, "strength": SCOPE_HISTORY}
        r2 = cst.runner_from_state(state)
        assert r2.project_gate() == r_hist.project_gate()
        assert r2.training_scopes() == r_hist.training_scopes()
        assert r2.surrogate_coverage() == r_hist.surrogate_coverage()
        grid = r_hist._phase_candidates(100, seed=9)
        np.testing.assert_allclose(
            r2.surrogates[GATE_3COMP].predict(grid).mean,
            r_hist.surrogates[GATE_3COMP].predict(grid).mean, atol=1e-6)
    finally:
        r_hist.set_training_scope("strength", SCOPE_ACTIVE)   # фикстура общая
    # старый сейв без ключей → гейта нет, области active
    state["runner"].pop("project_gate")
    state["runner"].pop("training_scope")
    r3 = cst.runner_from_state(state)
    assert r3.project_gate() is None
    assert set(r3.training_scopes().values()) == {SCOPE_ACTIVE}
    # неизвестная область/отклик в сейве — отказ, не молчание
    bad = json.loads(json.dumps(cst.runner_to_state(r_hist)))
    bad["runner"]["training_scope"] = {GATE_3COMP: "everything"}
    with pytest.raises(ValueError, match="неизвестна"):
        cst.runner_from_state(bad)
    bad["runner"]["training_scope"] = {"нет_такого": SCOPE_HISTORY}
    with pytest.raises(ValueError, match="не среди"):
        cst.runner_from_state(bad)

