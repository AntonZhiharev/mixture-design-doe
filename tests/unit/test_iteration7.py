"""Итерация 7 / M9 — тесты слоя наблюдаемости (PipelineTrace).

Лёгкие тесты без запуска тяжёлого pipeline: проверяем сериализацию numpy,
round-trip save/load, корректность списка прогонов и сводки метрик.
"""
import numpy as np
import pytest

from src.observability.trace import (StageEvent, PipelineTrace, list_runs,
                                      _jsonable)


def test_jsonable_handles_numpy_and_nonfinite():
    out = _jsonable({
        "arr": np.array([1.0, 2.0, 3.0]),
        "int": np.int64(7),
        "flt": np.float64(2.5),
        "bool": np.bool_(True),
        "nested": [np.array([1, 2]), {"x": np.float32(1.5)}],
        "nan": float("nan"),
        "inf": np.float64(np.inf),
    })
    assert out["arr"] == [1.0, 2.0, 3.0]
    assert out["int"] == 7 and isinstance(out["int"], int)
    assert out["flt"] == 2.5 and isinstance(out["flt"], float)
    assert out["bool"] is True
    assert out["nested"][0] == [1, 2]
    assert out["nested"][1]["x"] == pytest.approx(1.5)
    # nan / inf приводятся к None (валидный JSON)
    assert out["nan"] is None
    assert out["inf"] is None


def test_stage_event_round_trip():
    ev = StageEvent(run_id="r1", stage="M2", ts="t",
                    inputs={"n": 20},
                    outputs={"design": np.zeros((2, 3))},
                    metrics={"d_efficiency": np.float64(0.95)},
                    diagnostics={})
    d = ev.to_dict()
    ev2 = StageEvent.from_dict(d)
    assert ev2.run_id == "r1" and ev2.stage == "M2"
    assert ev2.outputs["design"] == [[0, 0, 0], [0, 0, 0]]
    assert ev2.metrics["d_efficiency"] == pytest.approx(0.95)


def test_pipeline_trace_save_load(tmp_path):
    root = str(tmp_path / "trace")
    trace = PipelineTrace(run_id="run42", root=root, meta={"seed": 42})
    trace.log("M2", inputs={"n_runs": 20},
              metrics={"n": 20, "p": 15, "d_efficiency": np.float64(0.9)})
    trace.log("AL_round_0", outputs={"x_star": np.array([0.2, 0.3, 0.5])},
              metrics={"cost_star": 2.5, "accepted": np.bool_(True)})
    trace.log("opt", outputs={"recipe": np.array([0.1, 0.4, 0.5])},
              metrics={"n_exp": 24})

    run_dir = trace.save()
    assert run_dir.endswith("run42")

    loaded = PipelineTrace.load(root, "run42")
    assert loaded.run_id == "run42"
    assert loaded.meta["seed"] == 42
    assert loaded.stages() == ["M2", "AL_round_0", "opt"]

    al = loaded.get("AL_round_0")
    assert al.metrics["accepted"] is True
    assert al.outputs["x_star"] == [0.2, 0.3, 0.5]

    summary = loaded.metrics_summary()
    assert summary["M2"]["p"] == 15
    assert summary["opt"]["n_exp"] == 24


def test_list_runs(tmp_path):
    root = str(tmp_path / "trace")
    assert list_runs(root) == []                      # нет каталога
    for rid in ["runA", "runB"]:
        PipelineTrace(run_id=rid, root=root).save()
    assert list_runs(root) == ["runA", "runB"]


def test_get_missing_stage_raises(tmp_path):
    trace = PipelineTrace(run_id="x", root=str(tmp_path))
    trace.log("M2", metrics={"n": 1})
    with pytest.raises(KeyError):
        trace.get("does_not_exist")
