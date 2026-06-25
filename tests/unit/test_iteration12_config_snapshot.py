# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 12 / §13.1: config_snapshot (воспроизводимость) + версионирование state.

Закрывает остаток §13.11 п.1–2:
  * ``core.config_snapshot.ConfigSnapshot`` — seeds/гиперпараметры/версии +
    ``code↔real`` на блок; обратимость и round-trip сериализации;
  * интеграция версионирования (``schema_history``/``points``/``config_snapshot``)
    в ``core.state.ProjectState`` — аддитивно, со save/load round-trip и
    обратной совместимостью со старыми ``state.json`` (M1–M8).
"""
import numpy as np
import pytest

from src.core.config_snapshot import ConfigSnapshot, detect_versions
from src.core.schema import (MIXTURE, PROCESS, DataPoint, ModelSpec,
                             ProjectSchema, ResponseSpec, VariableBlock)
from src.core.schema_evolution import evolve_schema, known_constant
from src.core.state import ProjectState


# ----------------------------------------------------------------------
# Фикстуры схем
# ----------------------------------------------------------------------
def _mixture_process_schema(version: int = 1) -> ProjectSchema:
    return ProjectSchema.mixture_process(
        VariableBlock.mixture(["A", "B", "C"]),
        VariableBlock.process(["T", "t"], [100, 10], [200, 20]),
        responses=[ResponseSpec("y1", "max")],
        model=ModelSpec(cross_level="cross-main", main_components=(0,)),
        version=version)


# ----------------------------------------------------------------------
# ConfigSnapshot.capture — состав снимка
# ----------------------------------------------------------------------
def test_capture_records_seeds_hyperparams_versions_and_code_real():
    s = _mixture_process_schema()
    snap = ConfigSnapshot.capture(
        s, seeds={"design": 0, "gp": 1, "mc": 42},
        hyperparameters={"kernel": "matern52", "noise_floor": 1e-6})

    assert snap.seeds == {"design": 0, "gp": 1, "mc": 42}
    # model всегда в снимке + явные гиперпараметры доливаются
    assert snap.hyperparameters["model"] == s.model.to_dict()
    assert snap.hyperparameters["kernel"] == "matern52"
    # версии окружения + версия схемы
    assert snap.versions["schema"] == "1"
    assert "python" in snap.versions and "numpy" in snap.versions
    # code↔real на оба блока
    assert set(snap.code_real) == {MIXTURE, PROCESS}
    assert snap.code_real[PROCESS]["names"] == ["T", "t"]
    assert snap.code_real[PROCESS]["lower"] == [100.0, 10.0]
    assert snap.code_real[PROCESS]["upper"] == [200.0, 20.0]


# ----------------------------------------------------------------------
# code↔real воспроизводимо ИЗ СНИМКА (без доступа к исходной схеме, §13.2)
# ----------------------------------------------------------------------
def test_code_real_reproducible_from_snapshot_and_invertible():
    s = _mixture_process_schema()
    snap = ConfigSnapshot.capture(s)
    pb = s.process_block()

    real = [150.0, 15.0]
    code_from_snap = snap.to_code(PROCESS, real)
    # совпадает с преобразованием исходного блока
    np.testing.assert_allclose(code_from_snap, pb.to_code(real))
    np.testing.assert_allclose(code_from_snap, [0.5, 0.5])
    # обратимость: from_code(to_code(real)) == real
    np.testing.assert_allclose(snap.from_code(PROCESS, code_from_snap), real)

    # восстановленный блок эквивалентен исходному
    assert snap.block(PROCESS) == pb


def test_snapshot_roundtrip_serialisation_identity():
    s = _mixture_process_schema()
    snap = ConfigSnapshot.capture(s, seeds={"design": 7},
                                  hyperparameters={"kernel": "rbf"})
    again = ConfigSnapshot.from_dict(snap.to_dict())
    assert again.to_dict() == snap.to_dict()
    assert again.seeds == snap.seeds
    assert again.code_real == snap.code_real


def test_detect_versions_has_core_entries():
    v = detect_versions()
    assert "python" in v and "numpy" in v


# ----------------------------------------------------------------------
# ProjectState: версионирование схемы и общая база точек
# ----------------------------------------------------------------------
def test_add_schema_tracks_versions_and_rejects_duplicates():
    st = ProjectState(name="p")
    s1 = ProjectSchema.mixture_only(["A", "B", "C"])
    s2 = evolve_schema(s1, add_process=[("T", 100, 200), ("t", 10, 20)],
                       migration={"T": known_constant(150), "t": known_constant(15)})
    st.add_schema(s1)
    st.add_schema(s2)
    assert st.current_schema_version == 2
    assert st.latest_schema() is s2
    assert st.schema_for(1) is s1
    with pytest.raises(ValueError):
        st.add_schema(s1)                       # повторная версия запрещена
    with pytest.raises(KeyError):
        st.schema_for(99)


def test_add_point_requires_existing_schema_version():
    st = ProjectState(name="p")
    s1 = ProjectSchema.mixture_only(["A", "B", "C"])
    st.add_schema(s1)
    p = DataPoint(schema_version=1, X={MIXTURE: [0.2, 0.3, 0.5]}, Y={"y1": 1.0})
    st.add_point(p)
    assert len(st.points) == 1
    bad = DataPoint(schema_version=2, X={MIXTURE: [0.2, 0.3, 0.5]})
    with pytest.raises(ValueError):
        st.add_point(bad)                       # версии 2 нет в истории


# ----------------------------------------------------------------------
# save/load round-trip: schema_history + points + config_snapshot
# ----------------------------------------------------------------------
def test_state_persists_versioning_and_snapshot(tmp_path):
    st = ProjectState(name="demo")
    s1 = ProjectSchema.mixture_only(["A", "B", "C"],
                                    responses=[ResponseSpec("y1", "max")])
    s2 = evolve_schema(s1, add_process=[("T", 100, 200), ("t", 10, 20)],
                       migration={"T": known_constant(150), "t": known_constant(15)})
    st.add_schema(s1)
    st.add_schema(s2)
    st.add_point(DataPoint(schema_version=1, X={MIXTURE: [0.2, 0.3, 0.5]},
                           Y={"y1": 0.7}, origin_tag={"stage": "M2"}))
    st.add_point(DataPoint(schema_version=2,
                           X={MIXTURE: [0.1, 0.4, 0.5], PROCESS: [0.5, 0.5]},
                           Y={"y1": 0.9}))
    st.set_config_snapshot(ConfigSnapshot.capture(s2, seeds={"design": 3}))

    folder = tmp_path / "project_demo"
    st.save(folder)
    loaded = ProjectState.load(folder)

    assert loaded.current_schema_version == 2
    assert [s.version for s in loaded.schema_history] == [1, 2]
    assert loaded.schema_for(2).process_names == ("T", "t")
    assert len(loaded.points) == 2
    assert loaded.points[0].schema_version == 1
    assert loaded.points[0].origin_tag == {"stage": "M2"}
    np.testing.assert_allclose(loaded.points[1].X[PROCESS], [0.5, 0.5])

    snap = loaded.get_config_snapshot()
    assert snap is not None
    assert snap.seeds == {"design": 3}
    np.testing.assert_allclose(snap.to_code(PROCESS, [150.0, 15.0]), [0.5, 0.5])


def test_old_state_json_without_versioning_loads_with_empty_defaults():
    """Обратная совместимость: старый снимок (M1–M8) без §13-полей грузится пустым."""
    legacy = {"name": "old", "config": {"q": 3, "seed": 42},
              "stage": "M2_screening_design", "data": {}, "models": {},
              "history": [{"stage": "M1_geometry", "ts": "t"}]}
    st = ProjectState.from_dict(legacy)
    assert st.name == "old"
    assert st.schema_history == []
    assert st.points == []
    assert st.current_schema_version is None
    assert st.config_snapshot == {}
    assert st.get_config_snapshot() is None
