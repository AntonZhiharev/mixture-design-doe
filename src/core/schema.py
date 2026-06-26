"""
core/schema.py — §13–14 фундамент: блочная схема переменных + составная
версионированная точка плана.

Это **единый фундамент** для двух фич (REBUILD_SPEC §13–14):

  * **Mixture-Process** — пространство планирования = произведение симплекса
    рецепта (MIXTURE-блок, Σx=1) и гиперкуба процессных параметров
    (PROCESS-блок, независимые интервалы). Режим определяется НАЛИЧИЕМ блоков,
    а не флагом: mixture-only и process-only — частные случаи (один блок пуст).
  * **Schema Evolution + Augmented Design** — схема версионируется, точки
    разных версий сосуществуют в общей базе, расширение схемы (добавить
    process-блок / отклик) не уничтожает старые данные.

Ключевые инварианты (см. §13.1):
  * Точка = составной объект ``{X: {block_kind: [...]}, Y: {resp: val|MISSING}}``
    + ``schema_version`` + ``origin_tag``. **Не плоский массив.**
  * ``MISSING`` — явный сентинел, допустим **только в Y**, не в X.
  * ``X["PROCESS"]`` хранится **в коде [0,1]**; физические единицы — через
    ``code↔real`` блока.
  * Сумма ``=1`` проверяется **только** для ``X["MIXTURE"]``.
  * ≤1 MIXTURE-блок, ≤1 PROCESS-блок, оба пустых запрещены.

Модуль аддитивен: не меняет существующие M1–M8, лишь даёт структуры, на которые
лягут блочный генератор термов (§13.3) и поблочная геометрия (§13.4).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .simplex import SimplexRegion

# Канонические виды блоков
MIXTURE = "MIXTURE"
PROCESS = "PROCESS"
_KINDS = (MIXTURE, PROCESS)

_CROSS_LEVELS = ("additive", "cross-main", "full-cross")
_PROCESS_ORDERS = ("linear", "quadratic")
_RESPONSE_KINDS = ("max", "min", "target")

_TOL = 1e-9


# ----------------------------------------------------------------------
# MISSING — явный сентинел отсутствующего ОТКЛИКА (только в Y)
# ----------------------------------------------------------------------
class _Missing:
    """Синглтон-сентинел «значение отклика не измерено».

    Намеренно НЕ ``None`` и НЕ ``0.0``: отравленную нулём/средним модель
    нельзя отладить (§13.7). Допустим только в ``DataPoint.Y``.
    """

    _instance: "Optional[_Missing]" = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - тривиально
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


def is_missing(v: Any) -> bool:
    """True, если значение — сентинел ``MISSING`` (или его JSON-кодировка None)."""
    return v is MISSING or v is None


# ----------------------------------------------------------------------
# VariableBlock — один блок переменных со своим типом пространства
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class VariableBlock:
    """Блок переменных: MIXTURE (симплекс) или PROCESS (бокс [L,U] на фактор).

    ``lower``/``upper`` — границы в ФИЗИЧЕСКИХ единицах (для MIXTURE — доли
    компонента ``L_i ≤ x_i ≤ U_i``; для PROCESS — реальные интервалы фактора).
    Внутренние расчёты для PROCESS ведутся в коде [0,1] — см. ``to_code``.
    """

    kind: str
    names: Tuple[str, ...]
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "lower", tuple(float(v) for v in self.lower))
        object.__setattr__(self, "upper", tuple(float(v) for v in self.upper))
        if self.kind not in _KINDS:
            raise ValueError(f"kind должен быть одним из {_KINDS}, дано '{self.kind}'.")
        n = len(self.names)
        if n == 0:
            raise ValueError("Блок не может быть пустым (нет переменных).")
        if not (len(self.lower) == len(self.upper) == n):
            raise ValueError("names/lower/upper должны быть одной длины.")
        lo = np.asarray(self.lower, float)
        hi = np.asarray(self.upper, float)
        if np.any(lo > hi + _TOL):
            raise ValueError("Каждая нижняя граница должна быть ≤ верхней.")
        if self.kind == MIXTURE:
            # Делегируем проверку выполнимости симплекса SimplexRegion.
            SimplexRegion(lower=lo, upper=hi, names=list(self.names))

    # -- размеры -------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def is_mixture(self) -> bool:
        return self.kind == MIXTURE

    @property
    def is_process(self) -> bool:
        return self.kind == PROCESS

    # -- code ↔ real (только PROCESS) ----------------------------------
    def to_code(self, real: Sequence[float]) -> np.ndarray:
        """Физические единицы → код [0,1] (покомпонентно, обратимо)."""
        real = np.asarray(real, dtype=float)
        lo = np.asarray(self.lower, float)
        hi = np.asarray(self.upper, float)
        span = np.where(hi - lo > _TOL, hi - lo, 1.0)
        return (real - lo) / span

    def from_code(self, code: Sequence[float]) -> np.ndarray:
        """Код [0,1] → физические единицы (обратное к ``to_code``)."""
        code = np.asarray(code, dtype=float)
        lo = np.asarray(self.lower, float)
        hi = np.asarray(self.upper, float)
        span = np.where(hi - lo > _TOL, hi - lo, 1.0)
        return lo + code * span

    def as_simplex_region(self) -> SimplexRegion:
        """MIXTURE-блок → :class:`SimplexRegion` (переиспользование M1-геометрии)."""
        if not self.is_mixture:
            raise ValueError("as_simplex_region доступен только для MIXTURE-блока.")
        return SimplexRegion(lower=list(self.lower), upper=list(self.upper),
                             names=list(self.names))

    # -- сериализация --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "names": list(self.names),
                "lower": list(self.lower), "upper": list(self.upper)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VariableBlock":
        return cls(kind=d["kind"], names=tuple(d["names"]),
                   lower=tuple(d["lower"]), upper=tuple(d["upper"]))

    # -- удобные конструкторы -----------------------------------------
    @classmethod
    def mixture(cls, names: Sequence[str],
                lower: Optional[Sequence[float]] = None,
                upper: Optional[Sequence[float]] = None) -> "VariableBlock":
        q = len(names)
        lo = [0.0] * q if lower is None else list(lower)
        hi = [1.0] * q if upper is None else list(upper)
        return cls(kind=MIXTURE, names=tuple(names), lower=tuple(lo), upper=tuple(hi))

    @classmethod
    def process(cls, names: Sequence[str],
                lower: Sequence[float], upper: Sequence[float]) -> "VariableBlock":
        return cls(kind=PROCESS, names=tuple(names),
                   lower=tuple(lower), upper=tuple(upper))


# ----------------------------------------------------------------------
# ModelSpec — уровень модели = ВХОД генератора дизайна (§13.3)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Конфигурация модели отклика (определяет p — размер дизайна на M2/M5).

    ``cross_level`` — уровень кросс-членов x_i·z_k:
      * ``additive``   — кросс нет (две почти независимые задачи);
      * ``cross-main`` *(дефолт)* — кросс только с ГЛАВНЫМИ компонентами
        (``main_components``, заполняется после M3); до M3 список пуст →
        ведёт себя как ``additive`` (адаптивный дефолт по фазе, §13.3);
      * ``full-cross`` — все x_i·z_k (когда бюджет точек покрывает p).
    ``mixture_order`` — порядок Scheffé (linear/quadratic/...).
    ``process_order`` — порядок RSM по z (linear/quadratic).
    ``main_components`` — индексы главных mixture-компонентов для ``cross-main``.
    """

    cross_level: str = "cross-main"
    mixture_order: str = "quadratic"
    process_order: str = "quadratic"
    main_components: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "main_components",
                           tuple(int(i) for i in self.main_components))
        if self.cross_level not in _CROSS_LEVELS:
            raise ValueError(f"cross_level ∈ {_CROSS_LEVELS}, дано '{self.cross_level}'.")
        if self.process_order not in _PROCESS_ORDERS:
            raise ValueError(f"process_order ∈ {_PROCESS_ORDERS}, дано '{self.process_order}'.")

    def to_dict(self) -> Dict[str, Any]:
        return {"cross_level": self.cross_level,
                "mixture_order": self.mixture_order,
                "process_order": self.process_order,
                "main_components": list(self.main_components)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSpec":
        return cls(cross_level=d.get("cross_level", "cross-main"),
                   mixture_order=d.get("mixture_order", "quadratic"),
                   process_order=d.get("process_order", "quadratic"),
                   main_components=tuple(d.get("main_components", ())))


# ----------------------------------------------------------------------
# ResponseSpec — описание целевого свойства (контракт DesirabilitySpec, §3/§10)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ResponseSpec:
    name: str
    kind: str = "max"
    target: Optional[float] = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in _RESPONSE_KINDS:
            raise ValueError(f"kind ∈ {_RESPONSE_KINDS}, дано '{self.kind}'.")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind,
                "target": self.target, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResponseSpec":
        return cls(name=d["name"], kind=d.get("kind", "max"),
                   target=d.get("target"), weight=float(d.get("weight", 1.0)))


# ----------------------------------------------------------------------
# ProjectSchema — версионированная схема проекта
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectSchema:
    """Неизменяемая версия схемы: набор блоков + отклики + модель.

    Версии копятся в ``ProjectState.schema_history`` (см. §13.1); точки разных
    версий сосуществуют в общей базе.
    """

    version: int
    blocks: Tuple[VariableBlock, ...]
    responses: Tuple[ResponseSpec, ...] = ()
    model: ModelSpec = field(default_factory=ModelSpec)
    # §13.7: политика миграции на ИМЯ переменной → {"policy": ..., "value": ...}.
    # JSON-native (known-constant/unknown/recompute), сериализуема, без классов.
    migration: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # §15.1.5: причины создания этой версии — список записей вида
    # ``{"append_param": "P"}`` / ``{"relax_bounds": "C"}``. Нужен, чтобы знать,
    # в какой области/составе собраны точки версии (атомарная фаза «схема+область»
    # несёт ОБЕ причины при ОДНОМ bump). Аддитивно, дефолт — пусто.
    change_log: Tuple[Dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "responses", tuple(self.responses))
        object.__setattr__(self, "change_log",
                           tuple(dict(e) for e in self.change_log))
        n_mix = sum(1 for b in self.blocks if b.is_mixture)
        n_proc = sum(1 for b in self.blocks if b.is_process)
        if n_mix > 1:
            raise ValueError("Допустим не более одного MIXTURE-блока.")
        if n_proc > 1:
            raise ValueError("Допустим не более одного PROCESS-блока.")
        if n_mix == 0 and n_proc == 0:
            raise ValueError("Оба блока пустые запрещены — нужен хотя бы один.")

    # -- доступ к блокам ----------------------------------------------
    def mixture_block(self) -> Optional[VariableBlock]:
        for b in self.blocks:
            if b.is_mixture:
                return b
        return None

    def process_block(self) -> Optional[VariableBlock]:
        for b in self.blocks:
            if b.is_process:
                return b
        return None

    @property
    def n_mixture(self) -> int:
        b = self.mixture_block()
        return b.size if b else 0

    @property
    def n_process(self) -> int:
        b = self.process_block()
        return b.size if b else 0

    @property
    def mixture_names(self) -> Tuple[str, ...]:
        b = self.mixture_block()
        return b.names if b else ()

    @property
    def process_names(self) -> Tuple[str, ...]:
        b = self.process_block()
        return b.names if b else ()

    @property
    def response_names(self) -> Tuple[str, ...]:
        return tuple(r.name for r in self.responses)

    def with_changes(self, **kw: Any) -> "ProjectSchema":
        """Вернуть НОВУЮ версию схемы (immutable); версию задаёт вызывающий."""
        return replace(self, **kw)

    # -- сериализация --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "blocks": [b.to_dict() for b in self.blocks],
            "responses": [r.to_dict() for r in self.responses],
            "model": self.model.to_dict(),
            "migration": {k: dict(v) for k, v in self.migration.items()},
            "change_log": [dict(e) for e in self.change_log],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectSchema":
        return cls(
            version=int(d["version"]),
            blocks=tuple(VariableBlock.from_dict(b) for b in d.get("blocks", [])),
            responses=tuple(ResponseSpec.from_dict(r) for r in d.get("responses", [])),
            model=ModelSpec.from_dict(d.get("model", {})),
            migration={k: dict(v) for k, v in d.get("migration", {}).items()},
            change_log=tuple(dict(e) for e in d.get("change_log", [])),
        )

    # -- удобные конструкторы (частные случаи общего механизма) --------
    @classmethod
    def mixture_only(cls, names: Sequence[str], *,
                     lower: Optional[Sequence[float]] = None,
                     upper: Optional[Sequence[float]] = None,
                     responses: Sequence[ResponseSpec] = (),
                     model: Optional[ModelSpec] = None,
                     version: int = 1) -> "ProjectSchema":
        return cls(version=version,
                   blocks=(VariableBlock.mixture(names, lower, upper),),
                   responses=tuple(responses),
                   model=model or ModelSpec())

    @classmethod
    def process_only(cls, names: Sequence[str], lower: Sequence[float],
                     upper: Sequence[float], *,
                     responses: Sequence[ResponseSpec] = (),
                     model: Optional[ModelSpec] = None,
                     version: int = 1) -> "ProjectSchema":
        return cls(version=version,
                   blocks=(VariableBlock.process(names, lower, upper),),
                   responses=tuple(responses),
                   model=model or ModelSpec())

    @classmethod
    def mixture_process(cls, mixture: VariableBlock, process: VariableBlock, *,
                        responses: Sequence[ResponseSpec] = (),
                        model: Optional[ModelSpec] = None,
                        version: int = 1) -> "ProjectSchema":
        return cls(version=version, blocks=(mixture, process),
                   responses=tuple(responses), model=model or ModelSpec())


def ordered_blocks(schema: ProjectSchema) -> List[VariableBlock]:
    """Канонический порядок блоков для составных координат: MIXTURE, затем PROCESS.

    Это фиксированный порядок конкатенации координат точки (§13.1): сначала
    mixture-доли ``x`` (q штук), затем process-код ``z`` (d штук). Генератор
    термов (§13.3) и геометрия (§13.4) опираются на этот порядок.
    """
    out: List[VariableBlock] = []
    mb = schema.mixture_block()
    pb = schema.process_block()
    if mb is not None:
        out.append(mb)
    if pb is not None:
        out.append(pb)
    return out


# ----------------------------------------------------------------------
# DataPoint — составная версионированная точка
# ----------------------------------------------------------------------
@dataclass
class DataPoint:
    """Одна точка плана: координаты по блокам + отклики + происхождение.

    ``X`` — словарь ``{block_kind: [coords...]}``; для PROCESS координаты — в
    коде [0,1]. ``Y`` — словарь ``{response_name: float | MISSING}`` (MISSING
    допустим поколоночно). ``origin_tag`` — например
    ``{"stage": "M2", "branch_id": None, "schema_version": 1}``.
    """

    schema_version: int
    X: Dict[str, List[float]]
    Y: Dict[str, Any] = field(default_factory=dict)
    origin_tag: Dict[str, Any] = field(default_factory=dict)
    fixed_in_augment: bool = False

    # -- доступ к блокам ----------------------------------------------
    def mixture(self) -> Optional[np.ndarray]:
        v = self.X.get(MIXTURE)
        return np.asarray(v, float) if v is not None else None

    def process_code(self) -> Optional[np.ndarray]:
        v = self.X.get(PROCESS)
        return np.asarray(v, float) if v is not None else None

    # -- валидация -----------------------------------------------------
    def validate(self, schema: ProjectSchema, *, tol: float = 1e-6) -> "DataPoint":
        """Поблочная проверка инвариантов (§13.1/§13.4). Возвращает self.

        Σx=1 проверяется ТОЛЬКО на MIXTURE-блоке; интервал [0,1] — ТОЛЬКО на
        PROCESS; ``MISSING`` запрещён в X (допустим только в Y).
        """
        mb = schema.mixture_block()
        pb = schema.process_block()

        # MISSING недопустим в X
        for kind, coords in self.X.items():
            for c in coords:
                if is_missing(c):
                    raise ValueError(f"MISSING недопустим в X (блок {kind}).")

        if mb is not None:
            x = self.X.get(MIXTURE)
            if x is None or len(x) != mb.size:
                raise ValueError("X[MIXTURE] отсутствует или не той длины.")
            x = np.asarray(x, float)
            if abs(x.sum() - 1.0) > tol:
                raise ValueError(f"Σx={x.sum():.6f} ≠ 1 на MIXTURE-блоке.")
            if np.any(x < -tol):
                raise ValueError("Доли компонентов должны быть ≥ 0.")
        elif MIXTURE in self.X:
            raise ValueError("X содержит MIXTURE, но в схеме нет mixture-блока.")

        if pb is not None:
            z = self.X.get(PROCESS)
            if z is None or len(z) != pb.size:
                raise ValueError("X[PROCESS] отсутствует или не той длины.")
            z = np.asarray(z, float)
            if np.any(z < -tol) or np.any(z > 1.0 + tol):
                raise ValueError("PROCESS-координаты должны быть в коде [0,1].")
        elif PROCESS in self.X:
            raise ValueError("X содержит PROCESS, но в схеме нет process-блока.")

        # Y: только float или MISSING
        for name, val in self.Y.items():
            if not (is_missing(val) or isinstance(val, (int, float))):
                raise ValueError(f"Y['{name}'] должен быть числом или MISSING.")
        return self

    # -- сериализация --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        # MISSING кодируется на диск как JSON null (декодируется обратно в MISSING)
        y = {k: (None if is_missing(v) else float(v)) for k, v in self.Y.items()}
        return {
            "schema_version": int(self.schema_version),
            "X": {k: [float(c) for c in v] for k, v in self.X.items()},
            "Y": y,
            "origin_tag": dict(self.origin_tag),
            "fixed_in_augment": bool(self.fixed_in_augment),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataPoint":
        y_raw = d.get("Y", {})
        y = {k: (MISSING if v is None else float(v)) for k, v in y_raw.items()}
        return cls(
            schema_version=int(d["schema_version"]),
            X={k: [float(c) for c in v] for k, v in d.get("X", {}).items()},
            Y=y,
            origin_tag=dict(d.get("origin_tag", {})),
            fixed_in_augment=bool(d.get("fixed_in_augment", False)),
        )


# ----------------------------------------------------------------------
# Составные координаты (mixture x, затем process code) — общий слой
# ----------------------------------------------------------------------
def composite_coords(schema: ProjectSchema, point: DataPoint) -> np.ndarray:
    """Точка → плоский вектор координат в каноническом порядке (x..., z_code...).

    Используется внутренними расчётами (генератор термов, геометрия). MIXTURE —
    доли как есть; PROCESS — код [0,1] как хранится.
    """
    parts: List[np.ndarray] = []
    for b in ordered_blocks(schema):
        coords = point.X.get(b.kind)
        if coords is None or len(coords) != b.size:
            raise ValueError(f"Точка не содержит координат блока {b.kind} нужной длины.")
        parts.append(np.asarray(coords, float))
    if not parts:
        return np.empty(0)
    return np.concatenate(parts)


def composite_matrix(schema: ProjectSchema, points: Sequence[DataPoint]) -> np.ndarray:
    """Список точек → матрица составных координат (n × (q+d))."""
    if len(points) == 0:
        dim = schema.n_mixture + schema.n_process
        return np.empty((0, dim))
    return np.vstack([composite_coords(schema, p) for p in points])


def split_composite(schema: ProjectSchema, vec: Sequence[float]
                    ) -> Dict[str, np.ndarray]:
    """Плоский вектор составных координат → словарь по блокам (обратное к concat)."""
    vec = np.asarray(vec, float)
    out: Dict[str, np.ndarray] = {}
    off = 0
    for b in ordered_blocks(schema):
        out[b.kind] = vec[off:off + b.size]
        off += b.size
    return out


# ----------------------------------------------------------------------
# Diff схем — различие переменных (для миграции/augmented design, §13.7)
# ----------------------------------------------------------------------
def schema_diff_vars(old: ProjectSchema, new: ProjectSchema) -> Dict[str, List[str]]:
    """НОВЫЕ ПЕРЕМЕННЫЕ в ``new`` относительно ``old`` (по блокам).

    Возвращает ``{"MIXTURE": [...], "PROCESS": [...]}`` — имена переменных,
    появившихся в новой схеме. ВАЖНО (§13.7): это про новые *переменные*
    (расширение пространства, требует миграционной политики), а НЕ про новые
    *члены* из уже существующих переменных (расширение модели — переиспользует
    данные почти бесплатно). Различение членов vs переменных живёт здесь:
    новый параметр z → старые точки требуют политику; новый кросс-член x·z из
    старых x,z → вычислим из существующих координат, миграция не нужна.
    """
    diff: Dict[str, List[str]] = {MIXTURE: [], PROCESS: []}
    for kind in _KINDS:
        old_names = set(_block_names(old, kind))
        new_names = _block_names(new, kind)
        diff[kind] = [nm for nm in new_names if nm not in old_names]
    return diff


def _block_names(schema: ProjectSchema, kind: str) -> Tuple[str, ...]:
    for b in schema.blocks:
        if b.kind == kind:
            return b.names
    return ()


def _block_bounds(schema: ProjectSchema, kind: str) -> Dict[str, Tuple[float, float]]:
    """Карта ``var → (lower, upper)`` для блока ``kind`` (пусто, если блока нет)."""
    for b in schema.blocks:
        if b.kind == kind:
            return {nm: (float(lo), float(hi))
                    for nm, lo, hi in zip(b.names, b.lower, b.upper)}
    return {}


def schema_diff_bounds(old: ProjectSchema, new: ProjectSchema, *,
                       tol: float = 1e-12
                       ) -> Dict[str, Dict[str, Tuple[float, float, float, float]]]:
    """ИЗМЕНЕНИЯ ГРАНИЦ переменных, ОБЩИХ для ``old`` и ``new`` (по блокам).

    Возвращает ``{"MIXTURE": {var: (old_lo, old_hi, new_lo, new_hi)}, "PROCESS":
    {...}}`` — только для переменных, у которых границы изменились (§15.0.2 / §15.0
    Предусловие 4). Это ОРТОГОНАЛЬНАЯ ось к :func:`schema_diff_vars`:

      * ``schema_diff_vars`` — added/removed ПЕРЕМЕННЫЕ (расширение пространства,
        требует миграционной политики); НЕ читает границы.
      * ``schema_diff_bounds`` — изменение ОБЛАСТИ при том же составе (например,
        C-релаксация ``[1/3,1/3]→[0,1]``); смотрит ТОЛЬКО пересечение имён
        ``old∩new`` (у новых переменных нет «старых» границ, у удалённых — «новых»).

    Ловит изменение в ЛЮБУЮ сторону: и релаксацию, и сужение (симметрично).
    Намеренно НЕ расширяет ``schema_diff_vars``: на него завязан ``migrate_point``
    (``if diff[MIXTURE]: return None``); смешивание осей рискует отбрасыванием
    валидных точек при C-релаксации.
    """
    diff: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
        MIXTURE: {}, PROCESS: {}}
    for kind in _KINDS:
        ob = _block_bounds(old, kind)
        nb = _block_bounds(new, kind)
        for name in ob.keys() & nb.keys():
            olo, ohi = ob[name]
            nlo, nhi = nb[name]
            if abs(olo - nlo) > tol or abs(ohi - nhi) > tol:
                diff[kind][name] = (olo, ohi, nlo, nhi)
    return diff


