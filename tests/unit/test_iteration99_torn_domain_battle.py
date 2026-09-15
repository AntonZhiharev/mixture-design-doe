# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 99 — БОЕВОЙ ТЕСТ С РВАНОЙ ОБЛАСТЬЮ ОПРЕДЕЛЕНИЯ ОТКЛИКОВ.

Живая постановка (аудит 15.09.2026): из 29 откликов на 70 точках плана
полный набор снимается лишь в 10 — большая часть рецептов даёт негодный
образец, и зависимые отклики не измеряются вовсе (iter98: MISSING с
причиной). Вопрос сессии: переживёт ли ядро кампанию, построенную как
«скрининг → границы доступного → ветки под цели → дозаливка / сдвиг границ /
новый компонент / новый отклик», если область определения функции РВАНАЯ.

Лаборатория — :class:`TornLab` над battle-истиной 3-комп мира + гейт-отклик
``surface`` (всегда измерим): ``gloss``/``dry_time`` отдаются как NaN там,
где ``surface < 4``. Измерима МЕНЬШАЯ часть области (~38 %).

Стадии и что проверяется:
  1. СКРИНИНГ (70 точек): ядро принимает план с ~60 % пропусков по gated-
     откликам; у каждого пропуска причина; суррогат каждого свойства учится
     на СВОИХ измеренных; M3-сводка по гейту считается; MISSING переживает
     save/load.
  2. ГРАНИЦЫ ДОСТУПНОГО: суррогат гейта предсказывает измеримость точки
     (accuracy против истины), то есть скрининг реально дал карту дыр.
  3. ВЕТКИ ПОД ЦЕЛИ: канон — гейт-отклик ОБЯЗАН входить в цель ветки
     (``hard_threshold_spec``). Контрольная ветка без гейта сжигает бюджет
     в неизмеримой зоне (acquisition про измеримость не знает: σ там
     максимальна); ветка с гейтом почти не промахивается, ``d_best``
     монотонен и достигает существенной доли аналитического оптимума.
  4. СДВИГ ГРАНИЦ: restrict в измеримую зону — история цела, активный пул
     сжался, суррогаты переобучились; relax обратно — точки вернулись.
  5. НОВЫЙ КОМПОНЕНТ: фаза {A,B} → append C; точки с MISSING мигрируют на
     грань, покрытие согласовано, ветка продолжает расти.
  6. НОВЫЙ ОТКЛИК (iter99): у ручной кампании гейт вводится ПОСЛЕ скрининга —
     у снятых точек MISSING с причиной, новые точки его меряют, суррогат
     рождается, ветка с гейтом в цели работает; у фиксированной истины —
     явный отказ целиком (схема не тронута).

OPEN_QUESTIONS (сознательно НЕ решаются здесь): feasibility-aware
acquisition (P(измеримо|x) как множитель к acq/argmax) — ядро сейчас
полагается на гейт в цели ветки, а не выводит измеримость из паттерна MISSING.
"""
import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps.campaign import CampaignController, response_coverage
from src.apps.campaign_screening import screening_report
from src.apps.campaign_ui import build_setup_runner
from src.apps.mixture_process_runner import (MISSING_REASONS_TAG,
                                             MixtureProcessRunner)
from src.core.schema import ResponseSpec, is_missing
from src.design.move_bounds import MOVE_RELAX, MOVE_RESTRICT
from src.optimize.desirability import DesirabilitySpec, hard_threshold_spec
from src.verification.battle_truth import (GATE_3COMP, GATE_THRESHOLD_3COMP,
                                           GATED_3COMP, TornLab,
                                           build_truth_3comp_gated,
                                           model_schema_3comp)
from src.verification.branch_reference import branch_optimum

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASELINE = [1 / 3, 1 / 3, 1 / 3, 0.5, 0.5]
N_SEED = 70
GATE_SPEC = hard_threshold_spec(GATE_THRESHOLD_3COMP, 0.3, "ge")


def _goal_nogate():
    return {"gloss": DesirabilitySpec("max", low=1.0, high=13.0),
            "strength": DesirabilitySpec("max", low=2.0, high=12.0)}


def _goal_gate():
    return dict(_goal_nogate(), **{GATE_3COMP: GATE_SPEC})


def _lab_and_runner(seed: int = 7):
    lab = TornLab(build_truth_3comp_gated())
    r = MixtureProcessRunner(model_schema_3comp(), lab, baseline=BASELINE,
                             seed=seed, n_restarts=2)
    return lab, r


def _lab_measure(r, lab, X):
    """Лаборатория меряет ПОЛНЫЙ физический вектор (как ``runner._measure``):
    координаты текущей фазы достраиваются гранью/baseline."""
    X = np.atleast_2d(np.asarray(X, float))
    return lab.evaluate(np.vstack([r._to_full(x) for x in X]))


def _screen(r, lab, n=N_SEED, seed=7):
    """Ручной цикл скрининга: план → лаборатория → фиксация с причинами."""
    X = np.asarray(r.propose_seed(n, seed=seed), float)
    Y = _lab_measure(r, lab, X)
    out = r.commit_seed(X, Y, missing_reasons=lab.reasons(Y))
    return X, Y, out


def _branch_rounds(r, lab, bid, *, rounds, n_points, explore_frac=0.3):
    """Раунды ветки ручным путём; возвращает долю предложений в «дыру»."""
    infeasible = 0
    total = 0
    for _ in range(rounds):
        Xn = r.propose_points(bid, n_points=n_points, explore_frac=explore_frac,
                              n_candidates=400)
        Yn = _lab_measure(r, lab, Xn)
        infeasible += int(np.isnan(Yn[:, r.prop_index[GATED_3COMP[0]]]).sum())
        total += len(Xn)
        r.commit_measured(bid, Xn, Yn, missing_reasons=lab.reasons(Yn))
    return infeasible / max(total, 1)


# ======================================================================
# Сквозной сценарий: скрининг → границы → ветки → сдвиг границ
# ======================================================================
def test_torn_domain_campaign_end_to_end():
    lab, r = _lab_and_runner()

    # ---------- 1. СКРИНИНГ: рваный план принимается целиком ------------
    X, Y, out = _screen(r, lab)
    n_full = int(np.isfinite(Y).all(axis=1).sum())
    print(f"\n=== TORN SCREENING: {N_SEED} runs, full rows {n_full}, "
          f"missing cells {out['n_missing']} ===")
    assert out["added"] == N_SEED and len(r.points) == N_SEED
    assert 0.5 <= 1 - n_full / N_SEED <= 0.8, (
        f"калибровка гейта: измеримо {n_full}/{N_SEED} — сценарий должен быть "
        f"«меньшая часть области измерима»")

    cov = r.surrogate_coverage()
    for name in ("strength", "price", "rho", GATE_3COMP):
        assert cov[name]["n_train"] == N_SEED and cov[name]["fitted"]
    for g in GATED_3COMP:
        assert cov[g]["n_train"] == n_full and cov[g]["fitted"]
        assert len(r.surrogates[g]._X) == n_full         # учится на своих
    # у КАЖДОГО пропуска — причина, и она называет гейт
    rep = r.missing_report()
    assert len(rep) == out["n_missing"]
    assert all(GATE_3COMP in row["reason"] for row in rep)
    assert set(row["response"] for row in rep) == set(GATED_3COMP)
    # M3-сводка по гейту считается на полной базе (это и есть скрининг)
    m3 = screening_report(r, GATE_3COMP, n_restarts=2)
    assert m3["summary"]["n"] == N_SEED and m3["n_significant"] >= 1
    # MISSING + причины переживают save/load, покрытие то же
    state = json.loads(json.dumps(cst.runner_to_state(r), ensure_ascii=False))
    r2 = cst.runner_from_state(state)
    assert r2.surrogate_coverage() == cov
    assert r2.missing_report() == rep

    # ---------- 2. ГРАНИЦЫ ДОСТУПНОГО: суррогат гейта знает карту дыр ----
    probe = r._phase_candidates(2000, seed=101)
    truth_ok = lab.feasible(probe)
    pred_ok = r.surrogates[GATE_3COMP].predict(probe).mean >= GATE_THRESHOLD_3COMP
    acc = float((truth_ok == pred_ok).mean())
    print(f"gate surrogate feasibility accuracy: {acc:.3f} "
          f"(feasible frac by truth {truth_ok.mean():.3f})")
    assert acc >= 0.85, f"суррогат гейта не выучил границы доступного: {acc:.3f}"

    # ---------- 3. ВЕТКИ ПОД ЦЕЛИ: гейт в цели — канон ------------------
    r.add_branch("nogate", _goal_nogate(), budget=30, satisfy_at=1.1,
                 branch_id="nogate")
    r.add_branch("gate", _goal_gate(), budget=30, satisfy_at=1.1,
                 branch_id="gate")
    miss_nogate = _branch_rounds(r, lab, "nogate", rounds=4, n_points=5)
    d_hist = []
    miss_gate = 0.0
    for _ in range(4):
        miss_gate += _branch_rounds(r, lab, "gate", rounds=1, n_points=5) / 4
        d_hist.append(r.branches["gate"].d_best)
    d_opt = branch_optimum(lab.truth, _goal_gate(), n_scan=20000, seed=5)["d"]
    print(f"nogate: proposals in the hole {miss_nogate:.2f}, "
          f"d_best={r.branches['nogate'].d_best:.3f}")
    print(f"gate:   proposals in the hole {miss_gate:.2f}, "
          f"d_best trajectory {np.round(d_hist, 3).tolist()}, d_opt={d_opt:.3f}")
    # контроль: без гейта acquisition тянет в дыру (σ там максимальна)
    assert miss_nogate >= 0.5, (
        f"контрольная ветка без гейта должна промахиваться: {miss_nogate:.2f}")
    # канон: с гейтом промахи редки, рекорд честный и растёт
    assert miss_gate <= 0.35, (
        f"ветка с гейтом промахивается слишком часто: {miss_gate:.2f}")
    assert all(b >= a - 1e-9 for a, b in zip(d_hist, d_hist[1:]))
    assert r.branches["gate"].d_best > 0
    # эталон — численный скан+уточнение; допуск как в battle-тесте (потолки фаз)
    assert r.branches["gate"].d_best <= d_opt + 0.02
    assert r.branches["gate"].d_best >= 0.6 * d_opt, (
        f"gate: d_best {r.branches['gate'].d_best:.3f} < 60% от d_opt {d_opt:.3f}")
    # рекорд ветки — только полностью измеренная точка (measured_desirability)
    xb = np.asarray(r.branches["gate"].x_best, float)
    assert bool(lab.feasible(xb.reshape(1, -1))[0])
    # общая база: все точки на месте, пропуски по-прежнему с причинами
    n_base = len(r.points)
    assert n_base == N_SEED + r.branches["nogate"].spent + r.branches["gate"].spent
    assert all(GATE_3COMP in row["reason"] for row in r.missing_report())
    assert response_coverage(r)[GATE_3COMP]["measured"] == n_base

    # ---------- 4. СДВИГ ГРАНИЦ: restrict в измеримую зону, relax назад --
    n_active0 = len(r._migrated_points())
    mv = r.move_region({"C": (0.0, 0.3)}, intent="measurable_zone")
    assert mv.move_type == MOVE_RESTRICT
    n_active1 = len(r._migrated_points())
    assert len(r.points) == n_base                       # И-1: история цела
    assert 0 < n_active1 < n_active0
    cov1 = r.surrogate_coverage()
    assert cov1[GATE_3COMP]["n_base"] == n_active1
    assert cov1["gloss"]["n_train"] + cov1["gloss"]["n_missing"] == n_active1
    # ветка продолжает работать в суженной области
    Xn = r.propose_points("gate", n_points=2, n_candidates=200)
    assert np.all(Xn[:, 2] <= 0.3 + 1e-9)
    mv2 = r.move_region({"C": (0.0, 1.0)}, intent="undo")
    assert mv2.move_type == MOVE_RELAX
    assert len(r._migrated_points()) == n_active0        # вернулись
    assert r.surrogate_coverage()["gloss"]["n_missing"] == \
        sum(1 for p in r.points if is_missing(p.Y["gloss"]))


# ======================================================================
# 5. НОВЫЙ КОМПОНЕНТ: фаза {A,B} → append C, MISSING мигрирует
# ======================================================================
def test_torn_domain_survives_component_append():
    lab, r = _lab_and_runner(seed=3)
    r.begin_phase(mixture_free=["A", "B"], process_free=["T", "P"])
    X, Y, out = _screen(r, lab, n=30, seed=3)
    assert out["n_missing"] > 0
    cov0 = r.surrogate_coverage()
    r.add_branch("gate", _goal_gate(), budget=40, satisfy_at=1.1,
                 branch_id="gate")
    _branch_rounds(r, lab, "gate", rounds=2, n_points=4)
    d_before = r.branches["gate"].d_best
    n_hist = len(r.points)
    v0 = r.current_schema_version

    r.augment_phase_mixture(["C"])                      # грань C=0 для старых

    assert r.current_schema_version == v0 + 1
    assert len(r.points) == n_hist                       # И-1
    mig = r._migrated_points()
    assert len(mig) == n_hist                            # все мигрировали
    assert all(abs(p.X["MIXTURE"][2]) < 1e-12 for p in mig)
    # MISSING и причины переехали дословно
    assert sum(is_missing(p.Y["gloss"]) for p in mig) == \
        sum(is_missing(p.Y["gloss"]) for p in r.points)
    cov1 = r.surrogate_coverage()
    assert cov1["gloss"]["n_train"] == cov0["gloss"]["n_train"] + \
        sum(1 for p in r.points[30:] if not is_missing(p.Y["gloss"]))
    assert r.X.shape[1] == 5                             # {A,B,C} × {T,P}
    # ветка продолжает работать в расширенном симплексе, рекорд не падает
    _branch_rounds(r, lab, "gate", rounds=2, n_points=4)
    assert r.branches["gate"].d_best >= d_before - 1e-9
    assert any(p.X["MIXTURE"][2] > 1e-6 for p in r.points[n_hist:])


# ======================================================================
# 6. НОВЫЙ ОТКЛИК: гейт вводится ПОСЛЕ скрининга (ручная кампания)
# ======================================================================
def test_new_response_refused_for_fixed_truth_lab():
    """TornLab несёт фиксированную физику — объявление отклика отклоняется целиком."""
    lab, r = _lab_and_runner()
    _screen(r, lab, n=12)
    ctrl = CampaignController(r)
    v0 = r.current_schema_version
    with pytest.raises(ValueError, match="не умеет"):
        ctrl.add_response(ResponseSpec("haze"))
    assert r.current_schema_version == v0
    assert "haze" not in r.property_names
    with pytest.raises(ValueError, match="уже занято"):
        r.declare_response("gloss")


def test_gate_response_introduced_after_screening_manual_campaign():
    """Технолог понял, что гейт нужен как отклик, уже ПОСЛЕ скрининга.

    Ручная кампания (истину вносит человек): скрининг снят без ``surface``,
    затем отклик вводится живой операцией — у старых точек MISSING с причиной,
    новые точки его меряют, суррогат рождается, ветка с гейтом в цели
    работает. Проверяется и UI-слой (столбцы листа «Отклики» / базы).
    """
    from src.apps import campaign_ui as ui

    lab = TornLab(build_truth_3comp_gated())
    props_wo_gate = [p for p in lab.property_names if p != GATE_3COMP]
    gi = lab.property_names.index(GATE_3COMP)
    r = build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T", "P"],
        process_lower=[0.0, 0.0], process_upper=[1.0, 1.0],
        response_names=props_wo_gate, seed=7)
    assert list(r.property_names) == props_wo_gate

    # скрининг без гейта: человек вносит Y лаборатории (без столбца surface)
    X = np.asarray(r.propose_seed(40, seed=7), float)
    Yl = lab.evaluate(X)
    Y = np.delete(Yl, gi, axis=1)
    r.commit_seed(X, Y, missing_reasons=lab.reasons(Yl))
    n_hist = len(r.points)
    assert r.surrogate_coverage()["gloss"]["n_missing"] > 0
    df0 = ui.seed_responses_dataframe(r, X[:3])
    assert f"{GATE_3COMP} (lab)" not in df0.columns

    # --- живая операция: ввести гейт как отклик ---------------------------
    ctrl = CampaignController(r)
    v0 = r.current_schema_version
    new = ctrl.add_response(ResponseSpec(GATE_3COMP, kind="max"),
                            reason="оценка поверхности введена после скрининга")
    assert new.version == v0 + 1 and r.current_schema_version == v0 + 1
    assert list(r.property_names) == props_wo_gate + [GATE_3COMP]
    assert r.prop_index[GATE_3COMP] == len(props_wo_gate)
    assert len(r.points) == n_hist                       # И-1
    assert all(is_missing(p.Y[GATE_3COMP]) for p in r.points)
    assert all(p.origin_tag[MISSING_REASONS_TAG][GATE_3COMP].startswith("оценка")
               for p in r.points)
    cov = r.surrogate_coverage()[GATE_3COMP]
    assert cov == {"n_train": 0, "n_base": n_hist, "n_missing": n_hist,
                   "fitted": False}
    # старые измерения не тронуты
    assert np.allclose([p.Y["strength"] for p in r.points], Y[:, 0])
    # UI: столбец появился в листе «Отклики» и в базе («н/и (причина)»)
    df1 = ui.seed_responses_dataframe(r, X[:3])
    assert f"{GATE_3COMP} (lab)" in df1.columns
    base = ui.campaign_base_dataframe(r)
    cell = base.loc[0, f"{GATE_3COMP} (изм.)"]
    assert isinstance(cell, str) and cell.startswith(ui.UNMEASURED_LABEL)
    # ветка на новый отклик создаётся, но предлагать точки пока нечем
    r.add_branch("gate", _goal_gate(), budget=30, satisfy_at=1.1,
                 branch_id="gate")
    with pytest.raises(KeyError):
        r.propose_points("gate", n_points=1)

    # --- дозаливка: новые точки меряют ВСЕ отклики, включая гейт ---------
    X2 = np.asarray(r.propose_seed(20, seed=11), float)
    Y2 = _lab_measure(r, lab, X2)                        # порядок = property_names
    assert lab.property_names == list(r.property_names)
    r.commit_seed(X2, Y2, missing_reasons=lab.reasons(Y2))
    cov2 = r.surrogate_coverage()[GATE_3COMP]
    assert cov2 == {"n_train": 20, "n_base": n_hist + 20, "n_missing": n_hist,
                    "fitted": True}
    assert response_coverage(r)[GATE_3COMP] == {
        "measured": 20, "total": n_hist + 20, "fraction": 20 / (n_hist + 20)}

    # --- ветка с гейтом в цели работает на рождённом суррогате ------------
    miss = _branch_rounds(r, lab, "gate", rounds=3, n_points=4)
    print(f"\nlate-gate branch: proposals in the hole {miss:.2f}, "
          f"d_best={r.branches['gate'].d_best:.3f}")
    assert r.branches["gate"].d_best > 0
    assert miss <= 0.5
    # save/load: расширенный набор откликов и MISSING переживают диск
    state = json.loads(json.dumps(cst.runner_to_state(r), ensure_ascii=False))
    r2 = cst.runner_from_state(state)
    assert list(r2.property_names) == list(r.property_names)
    assert is_missing(r2.points[0].Y[GATE_3COMP])
    assert r2.surrogate_coverage() == r.surrogate_coverage()
