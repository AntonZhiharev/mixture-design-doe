# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 43 (UI_REVISION_SPEC §43, приоритет P1.2) — постановка откликов:
пороги, вероятностные ограничения (chance), binding_report.

Проверяемый канон:

  * **§43.1 хранилище chance ветки** — ``runner.set_branch_chance`` /
    ``branch_chance`` по образцу ценовой ноги (runner-level, ``Branch`` не
    тронут); ``optimize_xbest`` подставляет сохранённые ограничения
    АВТОМАТИЧЕСКИ (иначе заданное из UI ограничение молча не участвовало бы в
    argmax, A0.6), явный аргумент имеет приоритет;
  * **§43.1 персистентность** — round-trip ``branch_chance`` через
    ``campaign_state`` (включая односторонние ``±inf`` → ``null`` → ``∓inf``:
    файл остаётся ВАЛИДНЫМ JSON, без литерала ``Infinity``);
  * **§43.1 обратимость** — ``CampaignController.set_chance`` пишет снимок в
    undo и откатывается (chance — намерение ветки, а не измеренная правда);
  * **§43.2 виды целей** — ``build_goal_spec`` собирает «порог ≥/≤» через
    ``hard_threshold_spec`` (ramp = ШУМ ИЗМЕРЕНИЯ, iter39 замечание 1);
    черновик хранит СЫРЫЕ входы и валидирует их сразу;
  * **§43.2 chance-черновик** — ``draft_add_chance`` / ``draft_remove_chance`` /
    ``draft_chance_constraints`` + отдельная таблица ``chance_editor_dataframe``
    (chance — множитель к d_overall, НЕ цель: роль отклика не меняется);
  * **§43.3 binding_report** — ``binding_report_dataframe`` (тип veto|chance,
    % пула, значение в x*, порог) и ``binding_report_caption``, различающий
    «оптимум НЕ НАЙДЕН» и «оптимум ЗАПРЕЩЁН» (CAMPAIGN_SPEC_PVC §7);
    ``branch_recipe_with_binding`` отдаёт рецепт и отчёт ОДНИМ прогоном.
"""
import numpy as np
import pytest

from src.apps import campaign as cv
from src.apps import campaign_state as cs
from src.apps import campaign_ui as ui
from src.apps.mixture_process_runner import MixtureProcessRunner
from src.core.schema import ModelSpec, ProjectSchema, VariableBlock
from src.optimize.desirability import ChanceConstraint, DesirabilitySpec


# ----------------------------------------------------------------------
# Фикстуры: маленький runner с ветками (mixture-only, дёшево для GP)
# ----------------------------------------------------------------------
class _Oracle:
    """ratio = C/(B+eps) — цель; dE = 10·C — ограничение (не цель)."""

    property_names = ["ratio", "dE"]

    def evaluate(self, Xc):
        Xc = np.atleast_2d(np.asarray(Xc, float))
        ratio = Xc[:, 2] / (Xc[:, 1] + 1e-9)
        return np.column_stack([ratio, 10.0 * Xc[:, 2]])


def _runner(n_seed: int = 12) -> MixtureProcessRunner:
    mix = VariableBlock.mixture(["A", "B", "C"])
    schema = ProjectSchema.mixture_only(
        ["A", "B", "C"], lower=list(mix.lower), upper=list(mix.upper),
        model=ModelSpec(cross_level="additive", mixture_order="quadratic"))
    r = MixtureProcessRunner(schema, _Oracle(), seed=0, n_restarts=1)
    r.seed_initial(int(n_seed))
    r.add_branch("goal", {"ratio": DesirabilitySpec("max", low=0.0, high=2.0)},
                 budget=5, branch_id="b1")
    return r


# ======================================================================
# §43.1 — хранилище chance-ограничений ветки (runner-level)
# ======================================================================
def test_set_branch_chance_roundtrip_in_memory():
    r = _runner()
    assert r.branch_chance("b1") == {}                 # по умолчанию пусто
    con = ChanceConstraint(y_max=5.0, alpha=0.2)
    r.set_branch_chance("b1", {"dE": con})
    got = r.branch_chance("b1")
    assert set(got) == {"dE"} and got["dE"].alpha == pytest.approx(0.2)
    # геттер отдаёт КОПИЮ: правка результата не меняет политику ветки
    got.pop("dE")
    assert set(r.branch_chance("b1")) == {"dE"}
    r.set_branch_chance("b1", None)                    # снятие
    assert r.branch_chance("b1") == {}


def test_set_branch_chance_validates_property_and_type():
    r = _runner()
    with pytest.raises(KeyError, match="не среди свойств оракула"):
        r.set_branch_chance("b1", {"нет_такого": ChanceConstraint(y_max=1.0)})
    with pytest.raises(TypeError, match="ChanceConstraint"):
        r.set_branch_chance("b1", {"dE": (0.0, 5.0)})
    with pytest.raises(KeyError, match="Нет ветки"):
        r.set_branch_chance("нет", {"dE": ChanceConstraint(y_max=1.0)})


def test_optimize_xbest_uses_saved_chance_automatically():
    """§43.1: сохранённое ограничение УЧАСТВУЕТ в argmax без явного аргумента."""
    r = _runner()
    res0 = r.optimize_xbest("b1", n_candidates=120, refine_iters=10, n_starts=1)
    assert res0.binding_report["chance"] == {}          # ограничений нет

    r.set_branch_chance("b1", {"dE": ChanceConstraint(y_max=5.0, alpha=0.2)})
    res1 = r.optimize_xbest("b1", n_candidates=120, refine_iters=10, n_starts=1)
    rep = res1.binding_report["chance"]["dE"]
    assert rep["alpha"] == pytest.approx(0.2)
    assert 0.0 <= rep["prob_at_optimum"] <= 1.0
    assert "dE" in res1.properties                      # mean достроен из суррогата


def test_explicit_chance_argument_overrides_saved():
    """Явный аргумент главнее сохранённого; пустой ``{}`` = «без ограничений»."""
    r = _runner()
    r.set_branch_chance("b1", {"dE": ChanceConstraint(y_max=5.0, alpha=0.2)})
    res = r.optimize_xbest("b1", n_candidates=80, refine_iters=0, n_starts=1,
                           chance_constraints={})
    assert res.binding_report["chance"] == {}


# ======================================================================
# §43.1 — персистентность (round-trip через campaign_state)
# ======================================================================
def test_branch_chance_persistence_roundtrip_one_sided():
    r = _runner()
    r.set_branch_chance("b1", {"dE": ChanceConstraint(y_max=5.0, alpha=0.1)})
    state = cs.runner_to_state(r)
    # ±inf сериализуется как null — файл остаётся ВАЛИДНЫМ JSON (без Infinity)
    saved = state["runner"]["branch_chance"]["b1"]["dE"]
    assert saved["y_min"] is None and saved["y_max"] == pytest.approx(5.0)
    import json
    assert "Infinity" not in json.dumps(state, ensure_ascii=False)

    r2 = cs.runner_from_state(state)
    con = r2.branch_chance("b1")["dE"]
    assert con.y_max == pytest.approx(5.0)
    assert not np.isfinite(con.y_min) and con.y_min < 0
    assert con.alpha == pytest.approx(0.1)


def test_branch_chance_persistence_two_sided_and_absent():
    r = _runner()
    r.set_branch_chance("b1", {"dE": ChanceConstraint(y_min=1.0, y_max=4.0,
                                                     alpha=0.05)})
    r2 = cs.runner_from_state(cs.runner_to_state(r))
    con = r2.branch_chance("b1")["dE"]
    assert (con.y_min, con.y_max) == (pytest.approx(1.0), pytest.approx(4.0))

    # старый сейв БЕЗ ключа (до iter43) грузится без ошибок → ограничений нет
    state = cs.runner_to_state(r)
    state["runner"].pop("branch_chance")
    r3 = cs.runner_from_state(state)
    assert r3.branch_chance("b1") == {}


# ======================================================================
# §43.1 — обратимость через контроллер (undo, §7)
# ======================================================================
def test_controller_set_chance_is_undoable():
    r = _runner()
    ctrl = cv.CampaignController(r)
    ctrl._rescore("b1")                    # оценка по измеренной базе (без chance)
    d_before = float(r.branches["b1"].d_best)

    out = ctrl.set_chance("b1", {"dE": ChanceConstraint(y_max=5.0, alpha=0.2)})
    assert out["op"] == "set_chance" and out["undo_available"]
    assert set(r.branch_chance("b1")) == {"dE"}
    # chance — НЕ цель: goal не тронут, ИЗМЕРЕННЫЙ d_best тоже (И-1) —
    # ограничение живёт только в argmax (множитель к d_overall по суррогату)
    assert set(r.branches["b1"].goal) == {"ratio"}
    assert float(r.branches["b1"].d_best) == pytest.approx(d_before)


    ctrl.undo()
    assert r.branch_chance("b1") == {}


# ======================================================================
# §43.2 — виды целей: пороги через hard_threshold_spec
# ======================================================================
def test_build_goal_spec_threshold_ge_uses_noise_ramp():
    spec = ui.build_goal_spec(ui.GOAL_KIND_GE, threshold=10.0, noise_sd=0.5,
                              weight=2.0)
    assert spec.kind == "max"
    assert (spec.low, spec.high) == (pytest.approx(9.5), pytest.approx(10.0))
    assert spec.weight == pytest.approx(2.0)


def test_build_goal_spec_threshold_le_and_plain_kinds():
    le = ui.build_goal_spec(ui.GOAL_KIND_LE, threshold=2.0, noise_sd=0.25)
    assert le.kind == "min"
    assert (le.low, le.high) == (pytest.approx(2.0), pytest.approx(2.25))

    mx = ui.build_goal_spec("max", low=1.0, high=5.0)
    assert (mx.kind, mx.low, mx.high) == ("max", 1.0, 5.0)
    tg = ui.build_goal_spec("target", low=0.0, high=10.0, target=6.0)
    assert tg.target == pytest.approx(6.0)


def test_build_goal_spec_refuses_incomplete_input():
    with pytest.raises(ValueError, match="порог"):
        ui.build_goal_spec(ui.GOAL_KIND_GE, noise_sd=0.5)
    with pytest.raises(ValueError, match="шума измерения"):
        ui.build_goal_spec(ui.GOAL_KIND_GE, threshold=10.0, noise_sd=0.0)
    with pytest.raises(ValueError, match="low и high"):
        ui.build_goal_spec("max")
    with pytest.raises(ValueError, match="Неизвестный вид"):
        ui.build_goal_spec("порог ≈", threshold=1.0, noise_sd=1.0)


def test_draft_goal_threshold_entry_and_specs():
    d = ui.draft_add_goal([], resp="Adhesion", kind=ui.GOAL_KIND_GE,
                          threshold=10.0, noise_sd=0.5, weight=1.5)
    assert d[0]["threshold"] == pytest.approx(10.0)
    assert d[0]["noise_sd"] == pytest.approx(0.5)
    d = ui.draft_add_goal(d, resp="gloss", kind="max", low=0.0, high=3.0)
    specs = ui.draft_goal_specs(d)
    assert set(specs) == {"Adhesion", "gloss"}
    assert specs["Adhesion"].kind == "max"              # порог ≥ → max с ramp
    assert specs["Adhesion"].low == pytest.approx(9.5)
    assert "порог" in ui.draft_goal_text(d[0])


def test_draft_add_goal_validates_immediately():
    """Ошибка вида всплывает при ДОБАВЛЕНИИ, а не при создании ветки (A0.6)."""
    with pytest.raises(ValueError, match="шума измерения"):
        ui.draft_add_goal([], resp="Adhesion", kind=ui.GOAL_KIND_GE,
                          threshold=10.0)


# ======================================================================
# §43.2 — chance-черновик + таблица ограничений ветки
# ======================================================================
def test_draft_chance_add_replace_remove_and_constraints():
    d = ui.draft_add_chance([], resp="dE", y_max=1.5, alpha=0.1)
    assert len(d) == 1 and d[0]["y_min"] is None
    d = ui.draft_add_chance(d, resp="dE", y_max=1.2, alpha=0.05)  # ЗАМЕНА
    assert len(d) == 1 and d[0]["y_max"] == pytest.approx(1.2)
    d = ui.draft_add_chance(d, resp="ratio", y_min=0.5, y_max=2.0)
    cons = ui.draft_chance_constraints(d)
    assert set(cons) == {"dE", "ratio"}
    assert not np.isfinite(cons["dE"].y_min)            # None → −inf
    assert cons["ratio"].y_min == pytest.approx(0.5)

    d2 = ui.draft_remove_chance(d, 0)
    assert len(d) == 2 and set(ui.draft_chance_constraints(d2)) == {"ratio"}
    assert ui.draft_remove_chance(d2, 9) == d2          # вне диапазона — no-op


def test_draft_add_chance_validates_immediately():
    with pytest.raises(ValueError, match="alpha"):
        ui.draft_add_chance([], resp="dE", y_max=1.0, alpha=0.0)
    with pytest.raises(ValueError, match="finite"):
        ui.draft_add_chance([], resp="dE")              # обе границы бесконечны


def test_chance_editor_dataframe_separate_block():
    r = _runner()
    assert ui.chance_editor_dataframe(r, "b1").empty     # ограничений нет
    r.set_branch_chance("b1", {"dE": ChanceConstraint(y_max=5.0, alpha=0.05)})
    df = ui.chance_editor_dataframe(r, "b1")
    row = df.iloc[0]
    assert row["ограничение (отклик)"] == "dE"
    assert row["y_min"] == "—" and row["y_max"] == pytest.approx(5.0)
    assert "0.950" in row["требование"]
    # роль отклика НЕ изменилась: chance — не цель (§5)
    assert r.response_role("b1", "dE") != "OPTIMIZED"


# ======================================================================
# §43.3 — binding_report: таблица и подпись «не найден» vs «запрещён»
# ======================================================================
_REPORT_OK = {
    "n_pool": 200,
    "specs": {"ratio": {"n_veto": 10, "frac_veto": 0.05, "d_at_optimum": 0.8}},
    "chance": {"dE": {"alpha": 0.05, "n_below": 20, "frac_below": 0.1,
                      "prob_at_optimum": 0.97, "satisfied_at_optimum": True}},
}


def test_binding_report_dataframe_structure():
    df = ui.binding_report_dataframe(_REPORT_OK)
    assert list(df["ограничение"]) == ["ratio", "dE"]
    assert set(df["тип"]) == {"veto (цель)", "вероятностное Pr"}
    ch = df[df["ограничение"] == "dE"].iloc[0]
    assert ch["% пула под биндингом"] == pytest.approx(10.0)
    assert ch["порог"] == pytest.approx(0.95)
    assert ch["выполнено в x*"] == "да"
    assert ui.binding_report_dataframe({}).empty


def test_binding_report_caption_three_states():
    assert ui.binding_report_caption(_REPORT_OK).startswith("✅")

    unmet = {"n_pool": 200, "specs": {},
             "chance": {"dE": {"alpha": 0.05, "frac_below": 0.4,
                               "prob_at_optimum": 0.80,
                               "satisfied_at_optimum": False}}}
    txt = ui.binding_report_caption(unmet)
    assert txt.startswith("⚠️") and "НЕ НАЙДЕН" in txt

    forbidden = {"n_pool": 200,
                 "specs": {"ratio": {"frac_veto": 1.0, "d_at_optimum": 0.0}},
                 "chance": {}}
    txt = ui.binding_report_caption(forbidden)
    assert txt.startswith("⛔") and "ЗАПРЕЩЁН" in txt


def test_branch_recipe_with_binding_single_run():
    r = _runner()
    r.set_branch_chance("b1", {"dE": ChanceConstraint(y_max=5.0, alpha=0.2)})
    df, rep = ui.branch_recipe_with_binding(r, "b1", n_candidates=120,
                                            refine_iters=10, n_starts=1)
    assert len(df) == 1 and "d_overall" in df.columns
    assert rep["n_pool"] > 0 and "dE" in rep["chance"]
    # таблица идентична обычному рецепту по структуре (общий хелпер строки)
    plain = ui.branch_recipe_dataframe(r, "b1", n_candidates=120,
                                       refine_iters=10, n_starts=1)
    assert list(df.columns) == list(plain.columns)
