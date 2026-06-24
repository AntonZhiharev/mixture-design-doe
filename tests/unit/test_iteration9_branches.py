"""Iteration 9 / 3c — project BRANCHES + origin tags (REBUILD_SPEC §5/§12).

Canon checked here:
  * a branch is a container of intent (goal/budget/history/status), NOT a model;
  * one shared project model (surrogates dict), branches read it & write points
    into ONE common base; every point carries an `origin` tag (no copies);
  * a branch round MEASURES ALL P properties and folds the point into the base;
  * budget is respected; branches & origin survive save/load.
"""
import warnings

import numpy as np
import pytest

from src.design.branches import (Branch, branch_scores, propose_by_score,
                                  allocate_budget)
from src.optimize.desirability import DesirabilitySpec
from src.apps.pipeline_runner import PipelineConfig, PipelineRunner

warnings.filterwarnings("ignore")


def _runner(tmp_path, name="branch_proj"):
    cfg = PipelineConfig(q=3, model="linear", property_names=["A", "B"],
                         seed=1, n_restarts=2, noise_sd=0.1)
    r = PipelineRunner(cfg, tmp_path / name)
    r.run_m2(simulate=True)
    return r


# ----------------------------------------------------------------------
def test_branch_is_container_not_model():
    """A Branch carries goal/budget/history — and has no surrogate/model."""
    goal = {"A": DesirabilitySpec("max", low=-5, high=5)}
    b = Branch(id="b1", name="hard", goal=goal, budget=5)
    assert b.remaining() == 5
    assert not hasattr(b, "model")
    assert not hasattr(b, "surrogate")
    assert b.status == "active"


def test_branch_state_roundtrip():
    goal = {"A": DesirabilitySpec("target", low=0, high=10, target=4, weight=2.0),
            "B": DesirabilitySpec("min", low=0, high=3)}
    b = Branch(id="b2", name="g", goal=goal, budget=7, spent=2,
               x_best=[0.3, 0.3, 0.4], d_best=0.5)
    b.history.append({"round": 1})
    b2 = Branch.from_state(b.to_state())
    assert b2.id == "b2" and b2.budget == 7 and b2.spent == 2
    assert b2.d_best == 0.5 and b2.x_best == [0.3, 0.3, 0.4]
    assert set(b2.goal) == {"A", "B"}
    assert b2.goal["A"].kind == "target" and b2.goal["A"].target == 4
    assert b2.goal["A"].weight == 2.0
    assert b2.history == [{"round": 1}]


def test_origin_tags_after_m2(tmp_path):
    r = _runner(tmp_path)
    assert len(r.origin) == len(r.design)
    assert set(r.origin) == {"M2"}
    assert r.origin_counts() == {"M2": len(r.design)}


def test_add_branch_validates_properties(tmp_path):
    r = _runner(tmp_path)
    with pytest.raises(KeyError):
        r.add_branch("bad", {"NOT_A_PROP": DesirabilitySpec("max", low=0, high=1)})
    br = r.add_branch("ok", {"A": DesirabilitySpec("max", low=-5, high=5)},
                      budget=4)
    assert br.id in r.branches
    # duplicate id rejected
    with pytest.raises(ValueError):
        r.add_branch("dup", {"A": DesirabilitySpec("max", low=-5, high=5)},
                     branch_id=br.id)


def test_branch_round_grows_shared_base_and_measures_all_P(tmp_path):
    r = _runner(tmp_path)
    n0 = len(r.design)
    br = r.add_branch("opt", {"A": DesirabilitySpec("max", low=-5, high=5)},
                      budget=4, satisfy_at=2.0)  # unreachable -> never "satisfied"
    out = r.run_branch_round(br.id, n_points=2)

    assert out["added"] == 2
    assert len(r.design) == n0 + 2
    assert r.Y.shape == (n0 + 2, 2)             # ALL P measured
    assert out["y_new"].shape == (2, 2)
    # origin tags appended for the branch, base length consistent
    assert r.origin[-2:] == [f"branch:{br.id}", f"branch:{br.id}"]
    assert len(r.origin) == len(r.design)
    assert r.origin_counts()[f"branch:{br.id}"] == 2
    # one shared model per property (no per-branch model)
    assert set(r.surrogates) == {"A", "B"}
    assert br.spent == 2 and br.remaining() == 2
    assert len(br.history) == 1


def test_branch_budget_is_respected(tmp_path):
    r = _runner(tmp_path)
    br = r.add_branch("opt", {"A": DesirabilitySpec("max", low=-5, high=5)},
                      budget=3, satisfy_at=2.0)
    r.run_branch_round(br.id, n_points=2)       # spends 2
    out = r.run_branch_round(br.id, n_points=2)  # only 1 slot left -> adds 1
    assert out["added"] == 1
    assert br.remaining() == 0
    assert br.status == "exhausted"
    # a further round adds nothing
    out3 = r.run_branch_round(br.id, n_points=2)
    assert out3["added"] == 0 and out3["status"] == "exhausted"


def test_branches_and_origin_survive_save_load(tmp_path):
    r = _runner(tmp_path, name="persist_proj")
    br = r.add_branch("opt", {"A": DesirabilitySpec("max", low=-5, high=5),
                              "B": DesirabilitySpec("min", low=-5, high=5)},
                      budget=5, satisfy_at=2.0)
    r.run_branch_round(br.id, n_points=2)
    n_base = len(r.design)
    counts = r.origin_counts()
    r.save_project()

    r2 = PipelineRunner.from_project(tmp_path, "persist_proj")
    assert set(r2.branches) == {br.id}
    rb = r2.branches[br.id]
    assert rb.spent == 2 and rb.budget == 5
    assert set(rb.goal) == {"A", "B"}
    assert len(r2.origin) == n_base
    assert r2.origin_counts() == counts
    assert r2.Y.shape == (n_base, 2)


def test_allocate_budget_proportional_caps_and_skips():
    # b1 далеко от цели, b2 ближе -> b1 получает больше слотов
    b1 = Branch("b1", "far", {"A": DesirabilitySpec("max", low=0, high=1)},
                budget=10, satisfy_at=1.0, d_best=0.0)
    b2 = Branch("b2", "near", {"A": DesirabilitySpec("max", low=0, high=1)},
                budget=10, satisfy_at=1.0, d_best=0.6)
    alloc = allocate_budget({"b1": b1, "b2": b2}, total_slots=10)
    assert sum(alloc.values()) == 10
    assert alloc["b1"] > alloc["b2"]

    # satisfied / exhausted ветки пропускаются
    b1.refresh_status()  # active (d_best 0)
    sat = Branch("s", "done", {"A": DesirabilitySpec("max", low=0, high=1)},
                 budget=5, satisfy_at=0.5, d_best=0.9)
    sat.refresh_status()
    exh = Branch("e", "spent", {"A": DesirabilitySpec("max", low=0, high=1)},
                 budget=3, spent=3)
    exh.refresh_status()
    alloc2 = allocate_budget({"b1": b1, "s": sat, "e": exh}, total_slots=4)
    assert set(alloc2) == {"b1"} and alloc2["b1"] == 4

    # кэп остатком бюджета: больше, чем доступно суммарно, не назначается
    small = Branch("t", "tiny", {"A": DesirabilitySpec("max", low=0, high=1)},
                   budget=2, spent=0)
    alloc3 = allocate_budget({"t": small}, total_slots=9)
    assert alloc3 == {"t": 2}

    # нет активных / ноль слотов -> пусто
    assert allocate_budget({}, 5) == {}
    assert allocate_budget({"b1": b1}, 0) == {}


def test_portfolio_round_splits_budget_across_branches(tmp_path):
    r = _runner(tmp_path)
    n0 = len(r.design)
    spec = lambda: DesirabilitySpec("max", low=-5, high=5)
    b1 = r.add_branch("b1", {"A": spec()}, budget=10, satisfy_at=2.0)
    b2 = r.add_branch("b2", {"B": spec()}, budget=10, satisfy_at=2.0)
    out = r.run_portfolio_round(total_slots=4)

    assert sum(out["allocation"].values()) == 4
    assert set(out["allocation"]) == {b1.id, b2.id}
    # все 4 точки попали в общую базу, помеченные своими ветками
    assert len(r.design) == n0 + 4
    counts = r.origin_counts()
    assert counts[f"branch:{b1.id}"] + counts[f"branch:{b2.id}"] == 4
    assert b1.spent + b2.spent == 4
    assert out["n_base"] == n0 + 4


def test_branch_scores_blend_and_propose(tmp_path):
    r = _runner(tmp_path)
    r.run_m6()
    goal = {"A": DesirabilitySpec("max", low=-5, high=5)}
    cands = r.region.random_points(50, seed=7)
    acq, d_pred, sigma = branch_scores(r.surrogates, goal, cands, explore_frac=0.0)
    # explore_frac=0 -> acquisition equals the desirability score
    assert np.allclose(acq, d_pred)
    assert acq.shape == (50,)
    pts = propose_by_score(cands, acq, n_points=3, min_dist=0.0)
    assert pts.shape[0] == 3
    # missing surrogate -> explicit error
    with pytest.raises(KeyError):
        branch_scores(r.surrogates, {"ZZZ": goal["A"]}, cands)
