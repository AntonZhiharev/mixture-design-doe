"""Итерация 7 / M9 — тесты чистой логики MCP-интроспекции (src/mcp/queries).

Не требуют установленного пакета `mcp`: проверяют выборку из сохранённого trace.
"""
import numpy as np
import pytest

from src.observability.trace import PipelineTrace
from src.mcp import queries


@pytest.fixture
def trace_root(tmp_path):
    root = str(tmp_path / "trace")
    tr = PipelineTrace(run_id="run1", root=root, meta={"seed": 7})
    tr.log("setup", metrics={"n_props": 4})
    tr.log("M2", outputs={"design": np.zeros((20, 3))},
           metrics={"n": 20, "p": 15, "d_efficiency": 0.9})
    tr.log("AL_round_0", outputs={"x_star": np.array([0.2, 0.3, 0.5])},
           metrics={"n_exp": 21, "d_overall": 0.95, "cost_star": 2.5,
                    "sigma_mean": 0.01, "accepted": True})
    tr.log("AL_round_1", outputs={"x_star": np.array([0.25, 0.30, 0.45])},
           metrics={"n_exp": 25, "d_overall": 0.97, "cost_star": 2.3,
                    "sigma_mean": 0.02, "accepted": True})
    tr.log("opt", outputs={"recipe": np.array([0.25, 0.30, 0.45])},
           metrics={"n_exp": 25, "incumbent_found": True})
    tr.log("benchmark",
           outputs={"recipe_analytical": [0.24, 0.31, 0.45],
                    "recipe_pipeline": [0.25, 0.30, 0.45]},
           metrics={"meets": True, "price_gap_pct": 5.0, "n_exp": 25},
           diagnostics={"specs": [{"name": "P2", "type": "ge"}]})
    tr.save()
    return root


def test_list_runs(trace_root):
    assert queries.list_runs(trace_root) == ["run1"]


def test_run_overview(trace_root):
    ov = queries.run_overview(trace_root, "run1")
    assert ov["run_id"] == "run1"
    assert ov["meta"]["seed"] == 7
    assert ov["n_al_rounds"] == 2
    assert ov["benchmark"]["meets"] is True


def test_get_stage_and_missing(trace_root):
    ev = queries.get_stage(trace_root, "run1", "M2")
    assert ev["metrics"]["p"] == 15
    assert len(ev["outputs"]["design"]) == 20
    with pytest.raises(KeyError):
        queries.get_stage(trace_root, "run1", "nope")


def test_get_metrics(trace_root):
    m = queries.get_metrics(trace_root, "run1")
    assert m["setup"]["n_props"] == 4
    assert m["opt"]["incumbent_found"] is True


def test_get_design(trace_root):
    d = queries.get_design(trace_root, "run1")
    assert d["stage"] == "M2" and d["n_points"] == 20


def test_al_progression(trace_root):
    prog = queries.al_progression(trace_root, "run1")
    assert [r["round"] for r in prog] == [0, 1]
    assert prog[1]["cost_star"] == 2.3


def test_diff_rounds(trace_root):
    d = queries.diff_rounds(trace_root, "run1", 0, 1)
    assert d["delta"]["n_exp"] == 4
    assert d["delta"]["cost_star"] == pytest.approx(-0.2)
    assert d["recipe_shift_l2"] > 0


def test_get_benchmark(trace_root):
    b = queries.get_benchmark(trace_root, "run1")
    assert b["metrics"]["price_gap_pct"] == 5.0
    assert b["specs"][0]["name"] == "P2"
    assert b["recipes"]["pipeline"] == [0.25, 0.30, 0.45]
