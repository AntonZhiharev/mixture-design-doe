"""
core/schema_evolution.py — §13.7 эволюция схемы + миграция старых точек.

Контракт (REBUILD_SPEC §13.7):
  * Схема ВЕРСИОНИРУЕТСЯ: ``SchemaHistory`` — неизменяемый реестр версий; точки
    разных версий сосуществуют, каждая ссылается на свою версию.
  * Расширение **пространства** (новая ПЕРЕМЕННАЯ) требует ЯВНОЙ политики миграции;
    расширение **модели** (новый ЧЛЕН из уже существующих переменных) — бесплатно
    (член вычислим из имеющихся координат, ``schema_diff_vars`` это различает).
  * Политики (на имя переменной): ``known-constant(v)`` / ``unknown`` /
    ``recompute(fn)``. НИКАКИХ молчаливых 0/средних — только явная политика или
    исключение точки.
  * Новый ОТКЛИК ⇒ у старых точек ``Y[y_new] = MISSING`` (суррогат учится только
    на измеренных, M6 это умеет). Тоже без молчаливых нулей.

Здесь НЕТ генератора дизайна и критерия остановки — ``augmented_design`` (§13.6)
переиспользуется как есть: сначала мигрируем точки этим модулем, затем подаём их
как ``existing`` в добор. Второй критерий остановки не вводится.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .schema import (MIXTURE, MISSING, PROCESS, DataPoint, ProjectSchema,
                     ResponseSpec, VariableBlock, schema_diff_vars)

# Политики миграции — JSON-native dict (сериализуются в ProjectSchema.migration).
KNOWN_CONSTANT = "known-constant"
UNKNOWN = "unknown"
RECOMPUTE = "recompute"


def known_constant(value: float) -> Dict[str, Any]:
    """Старые опыты шли при известном фикс. РЕАЛЬНОМ значении переменной."""
    return {"policy": KNOWN_CONSTANT, "value": float(value)}


def unknown() -> Dict[str, Any]:
    """Значение не записано → точка не годна для членов с этой переменной."""
    return {"policy": UNKNOWN}


def recompute(fn_name: str) -> Dict[str, Any]:
    """Вычислимо из существующих координат точки (по имени функции в реестре)."""
    return {"policy": RECOMPUTE, "fn": fn_name}


# ----------------------------------------------------------------------
# SchemaHistory — неизменяемый реестр версий (§13.1)
# ----------------------------------------------------------------------
@dataclass
class SchemaHistory:
    """Реестр версий схемы. Точка ссылается на версию через ``schema_version``."""

    versions: List[ProjectSchema] = field(default_factory=list)

    @classmethod
    def start(cls, schema: ProjectSchema) -> "SchemaHistory":
        h = cls()
        h.add(schema)
        return h

    def add(self, schema: ProjectSchema) -> ProjectSchema:
        if any(s.version == schema.version for s in self.versions):
            raise ValueError(f"Версия {schema.version} уже в истории (неизменяемость).")
        self.versions.append(schema)
        return schema

    def get(self, version: int) -> ProjectSchema:
        for s in self.versions:
            if s.version == version:
                return s
        raise KeyError(f"Версия схемы {version} не найдена в истории.")

    def latest(self) -> ProjectSchema:
        if not self.versions:
            raise ValueError("История схем пуста.")
        return max(self.versions, key=lambda s: s.version)


# ----------------------------------------------------------------------
# evolve_schema — новая версия (version+1)
# ----------------------------------------------------------------------
def evolve_schema(old: ProjectSchema, *,
                  add_process: Optional[Sequence[Tuple[str, float, float]]] = None,
                  add_responses: Sequence[ResponseSpec] = (),
                  model=None,
                  migration: Optional[Dict[str, Dict[str, Any]]] = None
                  ) -> ProjectSchema:
    """Вернуть НОВУЮ версию схемы (``version+1``).

    ``add_process`` — список ``(name, lower, upper)``; добавляется в КОНЕЦ
    process-блока (или создаётся новый блок). ``add_responses`` — новые отклики.
    ``model`` — новая ``ModelSpec`` (иначе сохраняется старая). ``migration`` —
    политики для НОВЫХ переменных (по имени), доливаются к ``old.migration``.
    """
    blocks = list(old.blocks)
    if add_process:
        new_names = [str(p[0]) for p in add_process]
        new_lo = [float(p[1]) for p in add_process]
        new_hi = [float(p[2]) for p in add_process]
        pb = old.process_block()
        if pb is None:
            merged = VariableBlock.process(new_names, new_lo, new_hi)
            blocks.append(merged)
        else:
            merged = VariableBlock.process(list(pb.names) + new_names,
                                           list(pb.lower) + new_lo,
                                           list(pb.upper) + new_hi)
            blocks = [merged if b.is_process else b for b in blocks]

    responses = list(old.responses) + list(add_responses)
    mig = dict(old.migration)
    if migration:
        mig.update(migration)
    return ProjectSchema(version=old.version + 1, blocks=tuple(blocks),
                         responses=tuple(responses),
                         model=model or old.model, migration=mig)


# ----------------------------------------------------------------------
# Миграция точки под целевую схему (§13.7)
# ----------------------------------------------------------------------
def migrate_point(point: DataPoint, old_schema: ProjectSchema,
                  target: ProjectSchema, *,
                  recompute_fns: Optional[Dict[str, Callable[[DataPoint], float]]] = None
                  ) -> Optional[DataPoint]:
    """Привести старую точку к ``target`` или вернуть ``None`` (точка не годна).

    Новые ПЕРЕМЕННЫЕ резолвятся по ``target.migration``; ``unknown`` / отсутствие
    политики ⇒ ``None``. Новые ОТКЛИКИ ⇒ ``Y[y_new]=MISSING``. Никаких молчаливых
    подстановок 0/средних. Возвращаемая точка помечается ``fixed_in_augment=True``.
    """
    diff = schema_diff_vars(old_schema, target)
    if diff[MIXTURE]:
        return None  # эволюция mixture-переменных меняет симплекс — отдельная политика

    new_X: Dict[str, List[float]] = {}

    # MIXTURE: размер должен совпасть с целевым
    if target.mixture_block() is not None:
        mx = point.X.get(MIXTURE)
        if mx is None or len(mx) != target.n_mixture:
            return None
        new_X[MIXTURE] = [float(v) for v in mx]
    elif MIXTURE in point.X:
        return None  # у цели нет mixture-блока, а у точки есть — несовместимо

    # PROCESS: старые координаты должны быть ПРЕФИКСОМ целевых (evolve добавляет в конец)
    pb = target.process_block()
    if pb is not None:
        target_names = list(target.process_names)
        old_names = list(old_schema.process_names)
        if target_names[:len(old_names)] != old_names:
            return None  # переупорядочивание не поддержано
        old_codes = list(point.X.get(PROCESS, []))
        if len(old_codes) != len(old_names):
            return None  # точка несогласована со своей версией
        codes = [float(c) for c in old_codes]
        for j in range(len(old_names), len(target_names)):
            nm = target_names[j]
            pol = target.migration.get(nm) or {"policy": UNKNOWN}
            kind = pol.get("policy")
            if kind == KNOWN_CONSTANT:
                lo, hi = pb.lower[j], pb.upper[j]
                v = float(pol["value"])
                codes.append((v - lo) / (hi - lo) if hi > lo else 0.0)
            elif kind == RECOMPUTE:
                fn = (recompute_fns or {}).get(pol.get("fn"))
                if fn is None:
                    return None
                codes.append(float(fn(point)))
            else:  # unknown / неизвестная политика
                return None
        new_X[PROCESS] = codes
    elif PROCESS in point.X:
        return None  # у цели нет process-блока, а у точки есть

    # Y: измеренные сохраняем, новые отклики → MISSING (никаких молчаливых нулей)
    new_Y: Dict[str, Any] = {r: point.Y.get(r, MISSING) for r in target.response_names}

    tag = dict(point.origin_tag)
    tag["migrated_from"] = point.schema_version
    return DataPoint(schema_version=target.version, X=new_X, Y=new_Y,
                     origin_tag=tag, fixed_in_augment=True)


def select_fixed_rows(points: Sequence[DataPoint], target_schema: ProjectSchema,
                      history: SchemaHistory, *,
                      recompute_fns: Optional[Dict[str, Callable[[DataPoint], float]]] = None
                      ) -> Tuple[List[DataPoint], List[DataPoint]]:
    """§13.7 канонический отбор: вернуть ``(используемые_мигрированные, отброшенные)``.

    Это полная версия с резолвом старой схемы по версии и политиками миграции
    (``augmented.select_fixed_rows`` — частный случай без истории/миграции).
    """
    used: List[DataPoint] = []
    skipped: List[DataPoint] = []
    for pt in points:
        try:
            old = history.get(pt.schema_version)
        except KeyError:
            skipped.append(pt)
            continue
        migrated = migrate_point(pt, old, target_schema, recompute_fns=recompute_fns)
        if migrated is None:
            skipped.append(pt)
        else:
            used.append(migrated)
    return used, skipped
