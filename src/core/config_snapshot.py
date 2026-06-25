"""
core/config_snapshot.py — §13.1/§13.2 снимок конфигурации (воспроизводимость).

`config_snapshot` из §13.1 — это **отдельная воспроизводимая сущность**, которая
фиксирует ВСЁ, что нужно для битового воспроизведения расчёта поверх собранных
точек:

  * ``seeds``           — именованные сиды (design / gp / mc / …);
  * ``hyperparameters`` — гиперпараметры модели/ядра (включая ``schema.model``);
  * ``versions``        — версии окружения (python/numpy/sklearn/scipy) и схемы;
  * ``code_real``       — параметры преобразования ``code ↔ real`` НА БЛОК
                          (§13.2): по сохранённым ``lower/upper`` блок и его
                          обратимая функция ``to_code``/``from_code``
                          восстанавливаются целиком, без доступа к исходной схеме.

Это НЕ то же самое, что ``apps.pipeline_runner.PipelineRunner.config_snapshot()`` —
тот сериализует legacy ``PipelineConfig`` (M1–M8) для пересборки runner'а. Здесь —
фундаментный §13-снимок поверх ``ProjectSchema`` (блочная схема + версионирование).

Инвариант воспроизводимости (под тестом): из ``ConfigSnapshot`` восстанавливается
тот же ``VariableBlock`` ⇒ ``to_code``/``from_code`` обратимы и совпадают с
исходным блоком; ``from_dict(to_dict())`` — тождественно.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .schema import ProjectSchema, VariableBlock


# ----------------------------------------------------------------------
# Версии окружения (best-effort: отсутствие пакета не роняет снимок)
# ----------------------------------------------------------------------
def detect_versions() -> Dict[str, str]:
    """Версии ключевых библиотек окружения (для воспроизводимости расчёта)."""
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:  # sklearn — ядро GP (M6)
        import sklearn  # noqa: WPS433 (локальный импорт намеренно)

        versions["sklearn"] = sklearn.__version__
    except Exception:  # pragma: no cover - окружение без sklearn
        pass
    try:  # scipy — статистика/линалг
        import scipy  # noqa: WPS433

        versions["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover
        pass
    return versions


def _jsonify(obj: Any) -> Any:
    """Рекурсивно привести numpy-типы к JSON-нативным (для hyperparameters)."""
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


# ----------------------------------------------------------------------
# ConfigSnapshot — §13.1
# ----------------------------------------------------------------------
@dataclass
class ConfigSnapshot:
    """Воспроизводимый снимок конфигурации проекта (§13.1)."""

    seeds: Dict[str, int] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)
    # code↔real на блок: {kind: {"names": [...], "lower": [...], "upper": [...]}}
    code_real: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # -- захват из схемы ----------------------------------------------
    @classmethod
    def capture(cls, schema: ProjectSchema, *,
                seeds: Optional[Dict[str, int]] = None,
                hyperparameters: Optional[Dict[str, Any]] = None,
                versions: Optional[Dict[str, str]] = None) -> "ConfigSnapshot":
        """Собрать снимок из ``ProjectSchema`` + явных сидов/гиперпараметров.

        ``code_real`` извлекается из блоков схемы (их ``lower/upper`` полностью
        задают обратимое преобразование §13.2). ``hyperparameters`` всегда несёт
        ``model`` (Scheffé/RSM/кросс-уровень); ``versions`` дополняется версией
        схемы. Переданные ``hyperparameters``/``versions`` доливаются поверх.
        """
        code_real = {
            b.kind: {"names": list(b.names),
                     "lower": [float(v) for v in b.lower],
                     "upper": [float(v) for v in b.upper]}
            for b in schema.blocks
        }
        hp: Dict[str, Any] = {"model": schema.model.to_dict()}
        if hyperparameters:
            hp.update(_jsonify(hyperparameters))
        ver = detect_versions()
        ver["schema"] = str(schema.version)
        if versions:
            ver.update({str(k): str(v) for k, v in versions.items()})
        return cls(seeds={str(k): int(v) for k, v in (seeds or {}).items()},
                   hyperparameters=hp, versions=ver, code_real=code_real)

    # -- воспроизведение code↔real из снимка --------------------------
    def block(self, kind: str) -> VariableBlock:
        """Восстановить ``VariableBlock`` блока ``kind`` из снимка (без схемы).

        Доказывает воспроизводимость §13.2: преобразование ``code↔real``
        полностью задаётся сохранёнными ``lower/upper``.
        """
        if kind not in self.code_real:
            raise KeyError(f"В снимке нет блока '{kind}'. Есть: {list(self.code_real)}.")
        spec = self.code_real[kind]
        return VariableBlock(kind=kind, names=tuple(spec["names"]),
                             lower=tuple(spec["lower"]), upper=tuple(spec["upper"]))

    def to_code(self, kind: str, real: Sequence[float]) -> np.ndarray:
        """real → code [0,1] по сохранённому в снимке преобразованию блока."""
        return self.block(kind).to_code(real)

    def from_code(self, kind: str, code: Sequence[float]) -> np.ndarray:
        """code [0,1] → real по сохранённому в снимке преобразованию блока."""
        return self.block(kind).from_code(code)

    # -- сериализация (JSON-native, round-trip-инвариант) --------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "seeds": {str(k): int(v) for k, v in self.seeds.items()},
            "hyperparameters": _jsonify(self.hyperparameters),
            "versions": {str(k): str(v) for k, v in self.versions.items()},
            "code_real": {
                k: {"names": list(v["names"]),
                    "lower": [float(x) for x in v["lower"]],
                    "upper": [float(x) for x in v["upper"]]}
                for k, v in self.code_real.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ConfigSnapshot":
        return cls(
            seeds={str(k): int(v) for k, v in d.get("seeds", {}).items()},
            hyperparameters=dict(d.get("hyperparameters", {})),
            versions={str(k): str(v) for k, v in d.get("versions", {}).items()},
            code_real={
                k: {"names": list(v.get("names", [])),
                    "lower": [float(x) for x in v.get("lower", [])],
                    "upper": [float(x) for x in v.get("upper", [])]}
                for k, v in d.get("code_real", {}).items()
            },
        )
