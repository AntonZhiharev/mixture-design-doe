"""M9 — PipelineTrace: структурный след стадий pipeline.

Каждая стадия эмитит StageEvent(inputs/outputs/metrics/diagnostics). PipelineTrace
накапливает события и сериализует их в JSON:

    {root}/{run_id}/index.json        # метаданные прогона + список стадий
    {root}/{run_id}/{stage}.json      # одно событие на стадию

Принципы (см. REBUILD_SPEC §11):
  * не инвазивно: логируются уже посчитанные величины, без влияния на pipeline;
  * сериализуемо: numpy → стандартные типы (см. _jsonable);
  * дёшево: только запись на диск.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

SCHEMA_VERSION = 1


# ----------------------------------------------------------------------
# JSON-safe конвертация (numpy → стандартные типы)
# ----------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    """Рекурсивно приводит объект к JSON-сериализуемому виду."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None

    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    # последняя попытка — строковое представление
    return str(obj)


# ----------------------------------------------------------------------
# StageEvent
# ----------------------------------------------------------------------
@dataclass
class StageEvent:
    """Одно структурное событие стадии pipeline."""

    run_id: str
    stage: str                       # 'M1'..'M8' | 'AL_round_{i}' | 'opt' | 'benchmark'
    ts: str                          # ISO-время логирования (UTC)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageEvent":
        return cls(
            run_id=d["run_id"],
            stage=d["stage"],
            ts=d.get("ts", ""),
            inputs=d.get("inputs", {}) or {},
            outputs=d.get("outputs", {}) or {},
            metrics=d.get("metrics", {}) or {},
            diagnostics=d.get("diagnostics", {}) or {},
        )


# ----------------------------------------------------------------------
# PipelineTrace
# ----------------------------------------------------------------------
class PipelineTrace:
    """Сборщик событий стадий + сериализация на диск."""

    def __init__(self, run_id: str, root: str = "project_demo/trace",
                 meta: Dict[str, Any] | None = None):
        self.run_id = str(run_id)
        self.root = root
        self.meta: Dict[str, Any] = dict(meta or {})
        self._events: List[StageEvent] = []

    # --- запись ---------------------------------------------------------
    def log(self, stage: str, *, inputs: Dict[str, Any] | None = None,
            outputs: Dict[str, Any] | None = None,
            metrics: Dict[str, Any] | None = None,
            diagnostics: Dict[str, Any] | None = None) -> StageEvent:
        ev = StageEvent(
            run_id=self.run_id,
            stage=str(stage),
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            inputs=dict(inputs or {}),
            outputs=dict(outputs or {}),
            metrics=dict(metrics or {}),
            diagnostics=dict(diagnostics or {}),
        )
        self._events.append(ev)
        return ev

    # --- доступ ---------------------------------------------------------
    def stages(self) -> List[str]:
        return [e.stage for e in self._events]

    def get(self, stage: str) -> StageEvent:
        for e in self._events:
            if e.stage == stage:
                return e
        raise KeyError(f"stage '{stage}' not found in run '{self.run_id}'")

    def events(self) -> List[StageEvent]:
        return list(self._events)

    # --- персистентность ------------------------------------------------
    def _run_dir(self) -> str:
        return os.path.join(self.root, self.run_id)

    def save(self) -> str:
        """Атомарно (по файлам) пишет события + index.json. Возвращает каталог."""
        run_dir = self._run_dir()
        os.makedirs(run_dir, exist_ok=True)
        for ev in self._events:
            self._atomic_write(os.path.join(run_dir, f"{ev.stage}.json"),
                               ev.to_dict())
        index = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "meta": _jsonable(self.meta),
            "stages": self.stages(),
        }
        self._atomic_write(os.path.join(run_dir, "index.json"), index)
        return run_dir

    @staticmethod
    def _atomic_write(path: str, data: Dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, root: str, run_id: str) -> "PipelineTrace":
        run_dir = os.path.join(root, str(run_id))
        index_path = os.path.join(run_dir, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"trace index not found: {index_path}")
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        trace = cls(run_id=index.get("run_id", run_id), root=root,
                    meta=index.get("meta", {}))
        for stage in index.get("stages", []):
            sp = os.path.join(run_dir, f"{stage}.json")
            if not os.path.exists(sp):
                continue
            with open(sp, "r", encoding="utf-8") as f:
                trace._events.append(StageEvent.from_dict(json.load(f)))
        return trace

    # --- сводка ---------------------------------------------------------
    def metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Метрики по всем стадиям: {stage: metrics}."""
        return {e.stage: dict(e.metrics) for e in self._events}


def list_runs(root: str) -> List[str]:
    """Список run_id (подкаталоги с index.json) в каталоге trace."""
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(root)):
        if os.path.exists(os.path.join(root, name, "index.json")):
            out.append(name)
    return out
