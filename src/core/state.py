"""
core/state.py — ProjectState: persistence & checkpointing (REBUILD_SPEC §5).

A single serialisable object that captures the whole pipeline state so the user
can SAVE at any module (M1..M8) and LOAD from that checkpoint later.

Layout on disk::

    project_<name>/
    ├── state.json        # config, current stage, history, inline arrays
    ├── data/             # designs + responses per stage (CSV)
    ├── models/           # fitted-model parameters (JSON / NPZ)
    └── checkpoints/      # named snapshots of state.json

Arrays are stored inline in ``state.json`` as nested lists for portability;
large tabular data (designs/responses) are additionally dumped to ``data/*.csv``.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config_snapshot import ConfigSnapshot
from .schema import DataPoint, ProjectSchema

# Canonical pipeline stages
STAGES: List[str] = [
    "M1_geometry", "M2_screening_design", "M3_screening_analysis",
    "M4_clustering", "M5_local_design", "M6_moe", "M7_active_learning",
    "M8_optimization",
]


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types/arrays to JSON-friendly structures."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _from_jsonable(obj: Any) -> Any:
    """Inverse of :func:`_to_jsonable`."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.array(obj["__ndarray__"], dtype=obj.get("dtype", float))
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


@dataclass
class ProjectState:
    """Serialisable container for the whole DOE pipeline."""

    name: str = "project"
    config: Dict[str, Any] = field(default_factory=dict)   # q, names, bounds, targets, cost, seed
    stage: str = "M1_geometry"                              # current stage
    data: Dict[str, Any] = field(default_factory=dict)     # designs/responses per stage
    models: Dict[str, Any] = field(default_factory=dict)   # fitted-model parameters
    history: List[Dict[str, Any]] = field(default_factory=list)

    # §13.1 фундамент: версионирование схемы, общая база точек, снимок конфигурации.
    # Поля аддитивны (дефолт — пусто) ⇒ совместимы со старыми state.json (M1–M8):
    # from_dict терпимо относится к их отсутствию.
    schema_history: List[ProjectSchema] = field(default_factory=list)
    current_schema_version: Optional[int] = None
    points: List[DataPoint] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # §13.1 версионирование схемы и общая база точек
    # ------------------------------------------------------------------
    def add_schema(self, schema: ProjectSchema) -> ProjectSchema:
        """Зарегистрировать НОВУЮ версию схемы (неизменяемость версий, §13.7)."""
        if any(s.version == schema.version for s in self.schema_history):
            raise ValueError(
                f"Версия схемы {schema.version} уже в истории (неизменяемость).")
        self.schema_history.append(schema)
        self.current_schema_version = max(s.version for s in self.schema_history)
        return schema

    def schema_for(self, version: int) -> ProjectSchema:
        for s in self.schema_history:
            if s.version == version:
                return s
        raise KeyError(f"Версия схемы {version} не найдена в истории.")

    def latest_schema(self) -> Optional[ProjectSchema]:
        if not self.schema_history:
            return None
        return max(self.schema_history, key=lambda s: s.version)

    def add_point(self, point: DataPoint) -> DataPoint:
        """Дописать точку в общую базу (точки разных версий сосуществуют, §13.1).

        Если история схем непуста, точка обязана ссылаться на существующую версию.
        """
        if (self.schema_history
                and not any(s.version == point.schema_version
                            for s in self.schema_history)):
            raise ValueError(
                f"Точка ссылается на отсутствующую версию схемы "
                f"{point.schema_version}.")
        self.points.append(point)
        return point

    def set_config_snapshot(self, snapshot: Any) -> None:
        """Сохранить §13.1-снимок (``ConfigSnapshot`` или готовый dict)."""
        if isinstance(snapshot, ConfigSnapshot):
            self.config_snapshot = snapshot.to_dict()
        else:
            self.config_snapshot = dict(snapshot)

    def get_config_snapshot(self) -> Optional[ConfigSnapshot]:
        """Восстановить ``ConfigSnapshot`` из снимка (или ``None``, если пуст)."""
        if not self.config_snapshot:
            return None
        return ConfigSnapshot.from_dict(self.config_snapshot)

    # ------------------------------------------------------------------
    # Stage bookkeeping
    # ------------------------------------------------------------------
    def set_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage '{stage}'. Valid: {STAGES}")
        self.stage = stage
        self.history.append({"stage": stage, "ts": datetime.now().isoformat()})

    def put(self, key: str, value: Any, *, section: str = "data") -> None:
        getattr(self, section)[key] = value

    def get(self, key: str, default: Any = None, *, section: str = "data") -> Any:
        return getattr(self, section).get(key, default)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": _to_jsonable(self.config),
            "stage": self.stage,
            "data": _to_jsonable(self.data),
            "models": _to_jsonable(self.models),
            "history": self.history,
            "schema_history": [s.to_dict() for s in self.schema_history],
            "current_schema_version": self.current_schema_version,
            "points": [p.to_dict() for p in self.points],
            "config_snapshot": _to_jsonable(self.config_snapshot),
            "_saved_at": datetime.now().isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectState":
        return cls(
            name=d.get("name", "project"),
            config=_from_jsonable(d.get("config", {})),
            stage=d.get("stage", "M1_geometry"),
            data=_from_jsonable(d.get("data", {})),
            models=_from_jsonable(d.get("models", {})),
            history=d.get("history", []),
            schema_history=[ProjectSchema.from_dict(s)
                            for s in d.get("schema_history", [])],
            current_schema_version=d.get("current_schema_version"),
            points=[DataPoint.from_dict(p) for p in d.get("points", [])],
            config_snapshot=_from_jsonable(d.get("config_snapshot", {})),
        )

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------
    def save(self, folder: str | Path) -> Path:
        """Atomically write state.json (+ ensure subfolders) to ``folder``."""
        folder = Path(folder)
        for sub in ("data", "models", "checkpoints"):
            (folder / sub).mkdir(parents=True, exist_ok=True)
        target = folder / "state.json"
        tmp = folder / "state.json.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, folder: str | Path) -> "ProjectState":
        folder = Path(folder)
        with open(folder / "state.json", "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def checkpoint(self, folder: str | Path, label: Optional[str] = None) -> Path:
        """Save state.json then copy it as a named checkpoint snapshot."""
        folder = Path(folder)
        self.save(folder)
        label = label or self.stage
        snap = folder / "checkpoints" / f"{label}.json"
        shutil.copy2(folder / "state.json", snap)
        return snap

    @classmethod
    def restore(cls, folder: str | Path, label: str) -> "ProjectState":
        """Load a named checkpoint snapshot."""
        folder = Path(folder)
        snap = folder / "checkpoints" / f"{label}.json"
        with open(snap, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def list_checkpoints(self, folder: str | Path) -> List[str]:
        folder = Path(folder) / "checkpoints"
        if not folder.exists():
            return []
        return sorted(p.stem for p in folder.glob("*.json"))
