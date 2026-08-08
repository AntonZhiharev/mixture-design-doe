"""
optimize/desirability.py — M8: product optimisation on the simplex.

Derringer–Suich desirability (REBUILD_SPEC M8, §3):

  * per-property desirability d_i(y) in [0, 1]:
      - "max"    (larger-is-better):  ramps 0 -> 1 as y goes low -> high;
      - "min"    (smaller-is-better): ramps 1 -> 0 as y goes low -> high;
      - "target" (target-is-best):    two-sided peak at the target value;
      - "target_range" (P2.2 UI_REVISION_SPEC): two-sided PLATEAU —
        d == 1 on [target_low, target_high], ramps to 0 towards low/high
        (постановка «желатинизация 60–70 %»: внутри допуска все значения
        РАВНОХОРОШИ; пик в точке ("target") искусственно предпочёл бы
        середину плато и тянул бы оптимизатор от дешёвого края допуска);
    each with a shape exponent `s` (s>1 = stricter, s<1 = lenient).

  * overall (weighted geometric mean):
        d_overall = (Π_i d_i^{w_i})^{1/Σ w_i}
    if any d_i == 0  ->  d_overall == 0 (a hard veto, by construction).

Cost is handled as just another property with a "min" spec (grab: cost is a
real objective, not a hack) — see `optimize_desirability(cost_fn=..., cost_spec=...)`.

Iter39 (блокер 2 DECODE_LAYER_PROPOSAL, «до сетапа»): σ-канал до оптимизатора.

  * :class:`ChanceConstraint` — вероятностное ограничение
    ``Pr(y_min ≤ y ≤ y_max) ≥ 1−α`` (постановка для ΔE по колористическим
    группам: недодоз УФ = рекламации в поле, потери асимметричны).
    d-фактор = ``clip(p / (1−α), 0, 1)`` — гладкий по μ и σ (градиент даёт
    Φ), плоского нуля нет: направление возврата в допустимую область
    сохраняется всюду, в отличие от жёсткого veto по среднему.
  * ``sigma_predictors`` — параллельный канал ``name -> callable(X)->σ(X)``.
    ⚠️ Для MoE подавать ПОЛНУЮ предиктивную σ (``MoEPrediction.std``):
    она уже включает и внутриэкспертную дисперсию Σ π_k σ_k², и
    межэкспертное рассогласование Σ π_k (μ_k − μ̄)² — неопределённость
    гейта. Только «внутри» переоценивает Pr на границах зон экспертов —
    ровно там, где идёт оптимизация.
  * :func:`hard_threshold_spec` — порог на ПРЕДСКАЗАННОЕ СРЕДНЕЕ
    (Adhesion/Opacity) с ramp ШИРИНОЙ ШУМА ИЗМЕРЕНИЯ отклика: veto
    практически жёсткое, но наклон сохранён — оптимизатор может выйти из
    недопустимой области (узкий/нулевой ramp даёт плоский нуль без
    направления возврата).
  * binding-отчёт (`DesirabilityResult.binding_report`) — какое ограничение
    бинднулось и на скольких точках глобального пула: «оптимум не найден»
    отличим от «оптимум запрещён».

Optimisation is performed OVER THE CONSTRAINED SIMPLEX (grab #10: no free-R^q
gradient).  We score a feasible candidate set, then locally refine the best
point with feasibility-preserving random perturbations.

R reference: ``desirability::dOverall``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from scipy.special import ndtr

from ..core.simplex import SimplexRegion
# P2.1: проекция на сетку уровней — ОДНА реализация на проект (тай-брейк
# «ближайший, при равенстве меньший» обязан совпадать у сэмплера и argmax,
# иначе план и оптимум разойдутся на границе ячейки). design.levels ничего
# не импортирует из optimize — цикла нет.
from ..design.levels import snap_to_levels as _snap_to_grid

Predictor = Callable[[np.ndarray], np.ndarray]


# ----------------------------------------------------------------------
# Per-property desirability specification
# ----------------------------------------------------------------------
@dataclass
class DesirabilitySpec:
    """One Derringer–Suich desirability transform for a single property.

    Parameters
    ----------
    kind   : "max" | "min" | "target" | "target_range".
    low    : lower bound of the active range.
    high   : upper bound of the active range.
    target : peak location (required for kind="target"; must satisfy low<target<high).
    target_low, target_high : plateau bounds (required for kind="target_range";
             must satisfy low < target_low < target_high < high). d == 1 on
             [target_low, target_high]; ramps to 0 at low / high (P2.2).
    s      : shape exponent for the (lower) ramp  (s>0).
    s2     : shape exponent for the upper ramp of a "target"/"target_range"
             spec (defaults to s).
    weight : importance exponent in the weighted geometric mean (w_i > 0).
    """

    kind: str
    low: float
    high: float
    target: Optional[float] = None
    s: float = 1.0
    s2: Optional[float] = None
    weight: float = 1.0
    target_low: Optional[float] = None
    target_high: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in ("max", "min", "target", "target_range"):
            raise ValueError(f"Unknown kind '{self.kind}' "
                             "(use max|min|target|target_range).")
        if self.high <= self.low:
            raise ValueError("Require high > low.")
        if self.s <= 0 or (self.s2 is not None and self.s2 <= 0):
            raise ValueError("Shape exponents s, s2 must be > 0.")
        if self.weight <= 0:
            raise ValueError("weight must be > 0.")
        if self.kind == "target":
            if self.target is None:
                raise ValueError("kind='target' requires a `target` value.")
            if not (self.low < self.target < self.high):
                raise ValueError("Require low < target < high for kind='target'.")
        # P2.2: плато-таргет. Поля плато валидны ТОЛЬКО у "target_range", а
        # точечный `target` у него запрещён: молча проигнорированное поле —
        # это потерянное намерение пользователя (A0.6).
        if self.kind == "target_range":
            if self.target_low is None or self.target_high is None:
                raise ValueError("kind='target_range' requires both "
                                 "`target_low` and `target_high`.")
            if not (self.low < self.target_low < self.target_high < self.high):
                raise ValueError("Require low < target_low < target_high < high "
                                 "for kind='target_range'.")
            if self.target is not None:
                raise ValueError("kind='target_range' has a plateau, not a peak: "
                                 "`target` must be None (use target_low/high).")
        elif self.target_low is not None or self.target_high is not None:
            raise ValueError("target_low/target_high are only valid for "
                             "kind='target_range'.")
        if self.s2 is None:
            self.s2 = self.s


# ----------------------------------------------------------------------
# Vectorised desirability transform
# ----------------------------------------------------------------------
def desirability_value(y, spec: DesirabilitySpec) -> np.ndarray:
    """Map property values ``y`` to desirabilities in [0, 1] for ``spec``."""
    y = np.asarray(y, dtype=float)
    lo, hi = spec.low, spec.high
    d = np.zeros_like(y, dtype=float)

    if spec.kind == "max":
        d = np.where(y <= lo, 0.0,
                     np.where(y >= hi, 1.0,
                              ((y - lo) / (hi - lo)) ** spec.s))
    elif spec.kind == "min":
        d = np.where(y <= lo, 1.0,
                     np.where(y >= hi, 0.0,
                              ((hi - y) / (hi - lo)) ** spec.s))
    elif spec.kind == "target":
        t = spec.target
        lower = ((y - lo) / (t - lo))
        upper = ((hi - y) / (hi - t))
        d = np.where((y < lo) | (y > hi), 0.0,
                     np.where(y <= t,
                              np.clip(lower, 0.0, 1.0) ** spec.s,
                              np.clip(upper, 0.0, 1.0) ** spec.s2))
    else:  # target_range (P2.2): плато d=1 на [target_low, target_high]
        tl, th = spec.target_low, spec.target_high
        lower = ((y - lo) / (tl - lo))
        upper = ((hi - y) / (hi - th))
        d = np.where((y < lo) | (y > hi), 0.0,
                     np.where(y < tl,
                              np.clip(lower, 0.0, 1.0) ** spec.s,
                              np.where(y > th,
                                       np.clip(upper, 0.0, 1.0) ** spec.s2,
                                       1.0)))
    return np.clip(d, 0.0, 1.0)


# ----------------------------------------------------------------------
# iter39 (блокер 2 DECODE_LAYER_PROPOSAL): σ-канал до оптимизатора
# ----------------------------------------------------------------------
_SIGMA_FLOOR = 1e-12          # σ→0 ⇒ prob вырождается в индикатор среднего


@dataclass
class ChanceConstraint:
    """Вероятностное ограничение ``Pr(y_min ≤ y ≤ y_max) ≥ 1−α``.

    Постановка для ΔE по колористическим группам (блокер 2): потери
    асимметричны (недодоз УФ = рекламации в поле), поэтому ограничение —
    на ХВОСТ предиктивного распределения, а не на среднее (veto по
    среднему для ΔE недостаточно — см. шапку модуля).

    d-фактор = ``clip(p / (1−α), 0, 1)`` — МНОЖИТЕЛЬ к d_overall:

      * ``p ≥ 1−α``  → фактор 1 (ограничение выполнено — не влияет);
      * ``p < 1−α``  → фактор ``p/(1−α)`` ∈ (0,1) — гладкий по μ и σ
        (градиент даёт Φ); плоского нуля нет: направление возврата в
        допустимую область сохраняется всюду.

    ⚠️ σ должна быть ПОЛНОЙ предиктивной. Для MoE это
    ``MoEPrediction.std``: Var[y] = Σ π_k σ_k² (внутри экспертов)
    + Σ π_k (μ_k − μ̄)² (межэкспертное рассогласование — неопределённость
    гейта). Только «внутри» переоценивает p на границах зон
    ответственности экспертов — ровно там, где идёт оптимизация.

    ``y_min=-inf`` / ``y_max=+inf`` — односторонние варианты
    (для ΔE: ``ChanceConstraint(y_max=dE_max, alpha=…)``).
    """

    y_min: float = -np.inf
    y_max: float = np.inf
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if not (self.y_min < self.y_max):
            raise ValueError("Require y_min < y_max.")
        if not (np.isfinite(self.y_min) or np.isfinite(self.y_max)):
            raise ValueError("At least one of y_min / y_max must be finite.")

    def prob(self, mu, sigma) -> np.ndarray:
        """``Pr(y_min ≤ y ≤ y_max)`` при ``y ~ N(μ, σ²)`` (поэлементно).

        σ прижимается к ``_SIGMA_FLOOR`` снизу: при σ→0 вероятность
        вырождается в индикатор «среднее внутри интервала».
        """
        mu = np.atleast_1d(np.asarray(mu, dtype=float))
        sigma = np.maximum(np.atleast_1d(np.asarray(sigma, dtype=float)),
                           _SIGMA_FLOOR)
        upper = (ndtr((self.y_max - mu) / sigma)
                 if np.isfinite(self.y_max) else np.ones_like(mu))
        lower = (ndtr((self.y_min - mu) / sigma)
                 if np.isfinite(self.y_min) else np.zeros_like(mu))
        return np.clip(upper - lower, 0.0, 1.0)

    def dfactor(self, mu, sigma) -> np.ndarray:
        """Множитель к d_overall: ``clip(prob / (1−α), 0, 1)``."""
        return np.clip(self.prob(mu, sigma) / (1.0 - self.alpha), 0.0, 1.0)


def hard_threshold_spec(threshold: float, noise_sd: float,
                        direction: str = "ge", *,
                        width_in_sd: float = 1.0,
                        s: float = 1.0,
                        weight: float = 1.0) -> DesirabilitySpec:
    """Порог на ПРЕДСКАЗАННОЕ СРЕДНЕЕ с ramp шириной ~ шума измерения.

    Для пороговых откликов (Adhesion ≥ T, Opacity ≥ T): d = 1 в допустимой
    области (ограничение «молчит» в геометрическом среднем), d = 0 глубже
    ``width_in_sd·noise_sd`` за порогом (veto), между ними — НАКЛОН.

    Почему ramp не «узкий», а шириной порядка шума измерения отклика:
    узкий/нулевой ramp даёт плоский нуль на всей недопустимой области —
    у оптимизатора нет направления возврата (refine встаёт, глобальный
    пул может вернуть вырожденный выбор среди d=0). Ramp ~ шуму сохраняет
    градиент к допустимой области, оставляя veto практически жёстким:
    различить «чуть ниже порога» и «на пороге» точнее шума всё равно
    нельзя.

    Это ограничение НА СРЕДНЕЕ, не вероятностное: для Adhesion/Opacity
    достаточно, для ΔE — нет (там :class:`ChanceConstraint`).

    Parameters
    ----------
    threshold   : порог допустимости.
    noise_sd    : СКО шума измерения отклика (>0).
    direction   : "ge" — допустимо ``y ≥ threshold``; "le" — ``y ≤ threshold``.
    width_in_sd : ширина ramp в единицах noise_sd (дефолт 1).
    s, weight   : параметры формы/веса :class:`DesirabilitySpec`.
    """
    ramp = float(width_in_sd) * float(noise_sd)
    if ramp <= 0.0:
        raise ValueError("Require noise_sd > 0 and width_in_sd > 0 "
                         "(нулевой ramp = плоский нуль без градиента).")
    if direction in ("ge", ">="):
        return DesirabilitySpec("max", low=threshold - ramp, high=threshold,
                                s=s, weight=weight)
    if direction in ("le", "<="):
        return DesirabilitySpec("min", low=threshold, high=threshold + ramp,
                                s=s, weight=weight)
    raise ValueError(f"Unknown direction '{direction}' (use 'ge' | 'le').")


# ----------------------------------------------------------------------
# Weighted geometric-mean aggregation
# ----------------------------------------------------------------------
def overall_desirability(d_individual: Mapping[str, np.ndarray],
                         weights: Optional[Mapping[str, float]] = None
                         ) -> np.ndarray:
    """Weighted geometric mean of per-property desirabilities.

    ``d_individual`` maps name -> array of d_i values (broadcastable shapes).
    Any zero desirability forces the overall to zero (hard veto).
    """
    names = list(d_individual.keys())
    if not names:
        raise ValueError("No desirabilities to aggregate.")
    D = np.vstack([np.atleast_1d(np.asarray(d_individual[n], float)) for n in names])
    if weights is None:
        w = np.ones(len(names))
    else:
        w = np.array([float(weights.get(n, 1.0)) for n in names])
    w = w / w.sum()

    out = np.zeros(D.shape[1], dtype=float)
    veto = np.any(D <= 0.0, axis=0)
    safe = ~veto
    if np.any(safe):
        log_d = (w[:, None] * np.log(np.clip(D[:, safe], 1e-300, 1.0))).sum(axis=0)
        out[safe] = np.exp(log_d)
    return out


# ----------------------------------------------------------------------
# Aggregator object (specs + weights bundled together)
# ----------------------------------------------------------------------
class Desirability:
    """Bundle of named :class:`DesirabilitySpec` objects."""

    def __init__(self, specs: Mapping[str, DesirabilitySpec]):
        if not specs:
            raise ValueError("Provide at least one desirability spec.")
        self.specs: Dict[str, DesirabilitySpec] = dict(specs)
        self.weights: Dict[str, float] = {n: s.weight for n, s in self.specs.items()}

    @property
    def names(self):
        return list(self.specs.keys())

    def individual(self, properties: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """d_i for every spec, given predicted property values."""
        missing = set(self.specs) - set(properties)
        if missing:
            raise KeyError(f"Missing predicted properties: {sorted(missing)}")
        return {n: desirability_value(properties[n], s) for n, s in self.specs.items()}

    def overall(self, properties: Mapping[str, np.ndarray]) -> np.ndarray:
        return overall_desirability(self.individual(properties), self.weights)


# ----------------------------------------------------------------------
# Optimisation result
# ----------------------------------------------------------------------
@dataclass
class DesirabilityResult:
    x: np.ndarray                       # best recipe (q parts, sums to 1)
    d_overall: float                    # overall desirability at x
    d_individual: Dict[str, float]      # per-property desirability at x
    properties: Dict[str, float]        # predicted property values at x
    n_evaluated: int = 0                # candidates scored
    refined: bool = False               # whether local refinement improved x
    n_starts: int = 1                   # number of multi-start refinements run
    history: list = field(default_factory=list)
    # iter39: {"n_pool", "specs": {name: …}, "chance": {name: …}} — какое
    # ограничение бинднулось и на скольких точках глобального пула
    binding_report: dict = field(default_factory=dict)

    def summary(self) -> str:
        props = ", ".join(f"{k}={v:.4g}" for k, v in self.properties.items())
        dind = ", ".join(f"d[{k}]={v:.3f}" for k, v in self.d_individual.items())
        return (f"d_overall={self.d_overall:.4f}\n"
                f"  recipe   = {np.round(self.x, 4).tolist()}\n"
                f"  props    = {props}\n"
                f"  desir.   = {dind}")


# ----------------------------------------------------------------------
# Optimisation over the constrained simplex
# ----------------------------------------------------------------------
def optimize_desirability(region: SimplexRegion,
                          predictors: Mapping[str, Predictor],
                          specs: Mapping[str, DesirabilitySpec],
                          cost_fn: Optional[Predictor] = None,
                          cost_spec: Optional[DesirabilitySpec] = None,
                          cost_name: str = "cost",
                          n_candidates: int = 4000,
                          refine_iters: int = 400,
                          refine_scale: float = 0.05,
                          n_starts: int = 5,
                          seed: Optional[int] = None,
                          process_lower: Optional[Sequence[float]] = None,
                          process_upper: Optional[Sequence[float]] = None,
                          process_fixed: Optional[Mapping[int, float]] = None,
                          process_levels: Optional[Mapping[int, Sequence[float]]] = None,
                          process_project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                          phr_spec=None,
                          chance_constraints: Optional[Mapping[str, ChanceConstraint]] = None,
                          sigma_predictors: Optional[Mapping[str, Predictor]] = None
                          ) -> DesirabilityResult:
    """Maximise the overall desirability over the constrained mixture simplex,
    optionally PRODUCT-ed with a process box (mixture×process, §15.1.4).

    Parameters
    ----------
    region      : feasible mixture region (M1).
    predictors  : name -> callable(X)->y giving the predicted property mean.
                  Wrap a MoE as ``lambda X: moe.predict(X).mean``. When a process
                  box is given, ``X`` is the COMPOSITE matrix ``[x..., z_code...]``.
    specs       : name -> DesirabilitySpec (must match `predictors` keys).
    cost_fn     : optional callable(X)->cost; folded in as a "min" property.
    cost_spec   : DesirabilitySpec for cost (defaults to plain "min" over the
                  observed cost range of the candidate set).
    n_candidates: feasible candidates to score (global stage).
    refine_iters: local random-search steps around the incumbent (0 disables).
    refine_scale: std of the (pseudocomponent / process-code) perturbation.
    process_lower / process_upper : per-process-coord box bounds in CODE space
                  (length ``d``). ``None`` (default) ⇒ mixture-only — поведение и
                  поток ГСЧ ИДЕНТИЧНЫ прежним (обратная совместимость, §15.1.4).
    process_fixed : ``{idx: value}`` для ЗАКРЫТЫХ фазой process-координат (маска
                  свободы): эти координаты держатся на ``value`` и не варьируются.
    process_levels : P2.1 (UI_REVISION_SPEC) — ``{idx: [уровни в КОДЕ]}``:
                  ДИСКРЕТНЫЕ уровни process-оси. Кандидаты и refine-пробы по
                  такой оси ПРОЕЦИРУЮТСЯ на ближайший уровень, поэтому x*
                  выдаётся в достижимом режиме (иначе argmax предлагал бы
                  «673 об/мин» при доступных 400/900, и модель училась бы на
                  одном, а лаборатория ставила другое — A0.6). ``None``
                  (дефолт) — прежнее поведение бит-в-бит: ни поток ГСЧ, ни
                  число проб не меняются (снап — чистая проекция ПОСЛЕ
                  розыгрыша).
    process_project : P3.3 (UI_REVISION_SPEC) — необязательная ПРОЕКЦИЯ
                  process-части В КОДЕ: ``callable(Z: n×d) -> n×d``,
                  применяется к глобальному пулу (после снапа уровней) и к
                  каждой refine-пробе (после clip/снапа). Канонический
                  случай — связанные оси (``dT_head = A − B ∈ [lo, hi]``,
                  ``runner._snap_links_code``): argmax обязан искать только
                  среди РЕАЛИЗУЕМЫХ пар осей (A0.6). Требования к проекции:
                  идемпотентность и «не трогать оси вне своей
                  ответственности» (в т.ч. закрытые ``process_fixed``).
                  ``None`` (дефолт) — прежнее поведение бит-в-бит.
    phr_spec    : опциональная phr/DAG-спека (``design.phr_sampler.PhrSpec``,
                  duck-typed; iter38, B1). ``None`` (дефолт) — прежний путь
                  бит-в-бит. Задана ⇒ оптимизация уважает phr-геометрию
                  (cap-потолки/трапеции, share-группы, ratio_to) ПО ПОСТРОЕНИЮ:
                  глобальный пул — ``sample_z → decode → to_fractions``, refine —
                  возмущение в z + ``clip_z`` + decode (НЕ rejection: у границы,
                  где лежит оптимум, rejection обваливается — урок iter34).
                  Требование: ``phr_spec.component_names`` соответствует
                  mixture-столбцам региона (``phr_spec.q == region.q``).
    chance_constraints : iter39 (блокер 2) — ``name -> ChanceConstraint``:
                  вероятностные ограничения ``Pr(y∈[y_min,y_max]) ≥ 1−α``.
                  Каждый d-фактор ``clip(p/(1−α),0,1)`` УМНОЖАЕТСЯ на
                  d_overall (и в глобальном пуле, и в refine). Для каждого
                  имени требуется mean-предиктор в ``predictors`` И
                  σ-предиктор в ``sigma_predictors``. ``None`` — прежнее
                  поведение бит-в-бит (RNG-поток не тронут).
    sigma_predictors : ``name -> callable(X)->σ(X)`` — предиктивная σ для
                  chance-constraints. Для MoE подавать ПОЛНУЮ σ
                  (``lambda X: moe.predict(X).std`` — включает
                  неопределённость гейта, см. :class:`ChanceConstraint`).

    Returns a :class:`DesirabilityResult`; при наличии process-бокса ``x`` —
    составной рецепт ``[x..., z_code...]`` (длиной ``q+d``).
    ``binding_report`` — статистика veto/биндинга по глобальному пулу
    и значения ограничений в x*.
    """
    # ---- assemble the full set of named objectives -------------------
    specs = dict(specs)
    predictors = dict(predictors)
    # the cost spec (if any) is served by `cost_fn`, not by `predictors`
    required = set(specs)
    if cost_fn is not None:
        required.discard(cost_name)
    missing = required - set(predictors)
    if missing:
        raise KeyError(f"Specs without a predictor: {sorted(missing)}")

    # ---- iter39: σ-канал (chance-constraints поверх desirability) -----
    cc: Dict[str, ChanceConstraint] = dict(chance_constraints or {})
    sig_preds: Dict[str, Predictor] = dict(sigma_predictors or {})
    if cc:
        no_mean = set(cc) - set(predictors)
        if no_mean:
            raise KeyError(
                f"Chance-constraints без mean-предиктора: {sorted(no_mean)}")
        no_sigma = set(cc) - set(sig_preds)
        if no_sigma:
            raise KeyError(
                f"Chance-constraints без sigma-предиктора: {sorted(no_sigma)}")

    def chance_probs(X: np.ndarray,
                     props: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """``Pr(y∈[y_min,y_max])`` по каждому ограничению (σ считается тут)."""
        X = np.atleast_2d(X)
        return {n: cc[n].prob(props[n],
                              np.asarray(sig_preds[n](X), float).ravel())
                for n in cc}

    def chance_factor(probs: Mapping[str, np.ndarray]) -> np.ndarray:
        """``Π clip(p/(1−α), 0, 1)`` — множитель к d_overall."""
        f: Optional[np.ndarray] = None
        for n, p in probs.items():
            fn = np.clip(p / (1.0 - cc[n].alpha), 0.0, 1.0)
            f = fn if f is None else f * fn
        return f

    def evaluate_props(X: np.ndarray) -> Dict[str, np.ndarray]:
        props = {n: np.asarray(f(X), float).ravel() for n, f in predictors.items()}
        if cost_fn is not None:
            props[cost_name] = np.asarray(cost_fn(X), float).ravel()
        return props

    # ---- process-box setup (d==0 ⇒ строго mixture-only, без лишних draws) ----
    q = region.q
    d = 0 if process_lower is None else len(process_lower)
    plo = np.asarray(process_lower, float) if d else None
    phi = np.asarray(process_upper, float) if d else None
    fixed = {int(k): float(v) for k, v in (process_fixed or {}).items()}
    free_proc = [j for j in range(d) if j not in fixed]
    # P2.1: сетки уровней СВОБОДНЫХ осей (закрытые фазой держатся на value —
    # снапить нечего); уровни отсортированы вызывающей стороной (levels.py).
    grids = {int(k): np.asarray(v, float)
             for k, v in (process_levels or {}).items()
             if int(k) not in fixed}

    def _snap_axis(j: int, values: np.ndarray) -> np.ndarray:
        """Проекция координаты оси ``j`` на её сетку уровней (нет сетки — as is)."""
        lv = grids.get(int(j))
        return values if lv is None else _snap_to_grid(values, lv)

    def _augment(Xmix: np.ndarray, rng_proc) -> np.ndarray:
        """Дополнить mixture-кандидаты (m×q) process-координатами → (m×(q+d))."""
        if d == 0:
            return Xmix
        Xmix = np.atleast_2d(Xmix)
        m = len(Xmix)
        Z = np.empty((m, d), float)
        for j in range(d):
            if j in fixed:
                Z[:, j] = fixed[j]
            else:
                Z[:, j] = _snap_axis(j, rng_proc.uniform(plo[j], phi[j], size=m))
        # P3.3: проекция связок ПОСЛЕ розыгрыша/снапа уровней — пул содержит
        # только реализуемые пары осей.
        if process_project is not None:
            Z = np.asarray(process_project(Z), float)
        return np.hstack([Xmix, Z])

    # ---- global stage: score a feasible candidate set ----------------
    if phr_spec is not None and int(phr_spec.q) != int(q):
        raise ValueError(
            f"optimize_desirability: phr_spec даёт {phr_spec.q} компонентов, "
            f"а mixture-регион — {q}.")
    rng = np.random.default_rng(seed)
    if phr_spec is None:
        cand = region.random_points(n_candidates, seed=seed)
        verts = region.extreme_vertices()
        cent = region.centroid().reshape(1, -1)
        mix_candidates = (np.vstack([cand, verts, cent]) if len(verts)
                          else np.vstack([cand, cent]))
        Zmix = None
        z_width = None
    else:
        # phr-путь (iter38, B1): кандидаты допустимы по построению; вершины/
        # центроид БОКСА не добавляются — это артефакты бокса, в углах
        # которого phr-геометрия и нарушается.
        Zmix = phr_spec.sample_z(n_candidates, seed=seed)
        mix_candidates = phr_spec.to_fractions(phr_spec.decode(Zmix))
        z_lo, z_hi = phr_spec.z_bounds()
        z_width = z_hi - z_lo          # масштаб осей z (единицы разные)
    rng_proc = (np.random.default_rng((0 if seed is None else seed) + 12345)
                if d else None)
    candidates = _augment(mix_candidates, rng_proc)

    props = evaluate_props(candidates)

    # cost spec defaults to "min" over the observed candidate cost range
    if cost_fn is not None and cost_name not in specs:
        c = props[cost_name]
        lo, hi = float(np.min(c)), float(np.max(c))
        if hi <= lo:
            hi = lo + 1.0
        specs[cost_name] = cost_spec or DesirabilitySpec("min", low=lo, high=hi)

    desir = Desirability(specs)
    d_all = desir.overall(props)
    probs_pool = chance_probs(candidates, props) if cc else {}
    if cc:
        d_all = d_all * chance_factor(probs_pool)
    n_eval = len(candidates)

    # MULTI-START: refine from the top `n_starts` distinct global candidates
    # (grab: a single incumbent can sit in a poor basin; restart from several).
    n_starts = max(1, int(n_starts))
    order = np.argsort(-d_all)
    start_indices = [int(i) for i in order[:n_starts]]

    d_global_best = float(d_all[order[0]])
    x_best = candidates[order[0]].copy()
    d_best = d_global_best

    history = [{"stage": "global", "n": n_eval, "d_overall": d_global_best,
                "n_starts": len(start_indices)}]

    # ---- local stage: feasibility-preserving random refinement -------
    # Рабочий вектор (дефолт): [mixture pseudocomponents (q), СВОБОДНЫЕ
    # process-коды]. При d==0 размер шага == q ⇒ поток ГСЧ совпадает с
    # прежним (golden цел). phr-путь: mixture-часть возмущается в z
    # (dim_z осей) с clip_z + decode — допустимость по построению.
    mix_dim = q if phr_spec is None else int(phr_spec.dim_z)
    step_dim = mix_dim + len(free_proc)
    for s_no, gi in enumerate(start_indices):
        x_cur = candidates[gi].copy()
        d_cur = float(d_all[gi])
        if phr_spec is None:
            w_cur = region.to_pseudo(x_cur[:q])  # work in pseudocomponents
        else:
            zmix_cur = Zmix[gi].copy()           # work in z (phr-геометрия)
        z_cur = x_cur[q:].copy() if d else np.empty(0)
        improved = False
        for it in range(int(refine_iters)):
            step = rng.normal(0.0, refine_scale, size=step_dim)
            if phr_spec is None:
                w_try = np.clip(w_cur + step[:q], 0.0, None)
                s = w_try.sum()
                if s <= 0:
                    continue
                w_try = w_try / s
                x_mix_try = region.from_pseudo(w_try)
                if not region.is_feasible(x_mix_try):
                    x_mix_try = region.clip(x_mix_try)
                    if not region.is_feasible(x_mix_try):
                        continue
            else:
                # step ~ N(0, refine_scale) в НОРМИРОВАННЫХ осях → в единицы z
                zmix_try = phr_spec.clip_z(zmix_cur + step[:mix_dim] * z_width)
                x_mix_try = phr_spec.to_fractions(phr_spec.decode(zmix_try))
            if d:
                z_try = z_cur.copy()
                for k, j in enumerate(free_proc):
                    # сначала clip к боксу, затем снап к сетке уровней: шаг по
                    # дискретной оси либо остаётся на месте, либо перепрыгивает
                    # на соседний ДОСТИЖИМЫЙ режим
                    z_try[j] = float(_snap_axis(j, np.asarray(
                        np.clip(z_cur[j] + step[mix_dim + k], plo[j], phi[j]),
                        float)))
                # P3.3: refine-проба тоже проецируется на полосы связок —
                # иначе локальный шаг вывел бы x* из реализуемой области.
                if process_project is not None:
                    z_try = np.asarray(
                        process_project(z_try.reshape(1, -1)), float).ravel()
                x_try = np.concatenate([x_mix_try, z_try])
            else:
                x_try = x_mix_try
            p_try = evaluate_props(x_try.reshape(1, -1))
            d_try = float(desir.overall(p_try)[0])
            if cc:
                d_try *= float(chance_factor(
                    chance_probs(x_try.reshape(1, -1), p_try))[0])
            n_eval += 1
            if d_try > d_cur:
                d_cur, x_cur = d_try, x_try.copy()
                if phr_spec is None:
                    w_cur = region.to_pseudo(x_cur[:q])
                else:
                    zmix_cur = zmix_try.copy()
                z_cur = x_cur[q:].copy() if d else np.empty(0)
                improved = True
        history.append({"stage": "start", "start": s_no, "from_global": gi,
                        "d_overall": d_cur, "improved": improved})
        if d_cur > d_best:
            d_best, x_best = d_cur, x_cur.copy()

    refined = d_best > d_global_best + 1e-15

    # ---- package the winner -----------------------------------------
    props_best = evaluate_props(x_best.reshape(1, -1))
    d_ind = {n: float(v[0]) for n, v in desir.individual(props_best).items()}
    props_scalar = {n: float(v[0]) for n, v in props_best.items()}

    # binding-отчёт (iter39): «оптимум не найден» отличим от «оптимум
    # запрещён». Статистика — по ГЛОБАЛЬНОМУ пулу (refine-пробы не входят),
    # значения ограничений — в возвращаемом x*.
    d_ind_pool = desir.individual(props)
    report: Dict[str, dict] = {
        "n_pool": int(len(candidates)),
        "specs": {n: {"n_veto": int(np.sum(v <= 0.0)),
                      "frac_veto": float(np.mean(v <= 0.0)),
                      "d_at_optimum": d_ind[n]}
                  for n, v in d_ind_pool.items()},
        "chance": {},
    }
    if cc:
        probs_best = chance_probs(x_best.reshape(1, -1), props_best)
        for n, con in cc.items():
            thr = 1.0 - con.alpha
            p_star = float(probs_best[n][0])
            report["chance"][n] = {
                "alpha": float(con.alpha),
                "n_below": int(np.sum(probs_pool[n] < thr)),
                "frac_below": float(np.mean(probs_pool[n] < thr)),
                "prob_at_optimum": p_star,
                "satisfied_at_optimum": bool(p_star >= thr - 1e-12),
            }

    return DesirabilityResult(
        x=x_best, d_overall=d_best, d_individual=d_ind,
        properties=props_scalar, n_evaluated=n_eval,
        refined=refined, n_starts=len(start_indices), history=history,
        binding_report=report,
    )



# ----------------------------------------------------------------------
# §15.6 §3 — Цена за ИЗДЕЛИЕ через плотность ρ (структурный параметр)
# ----------------------------------------------------------------------
# ИСТОЧНИК ИСТИНЫ (physics-трактовка, решение сессии §15.6):
#
#     price_изд(A,B,C,T,P) = price_состав(A,B,C) · ρ(A,B,C,T,P)   [₽/изделие]
#
# Цена за КИЛОГРАММ сырья (``price_состав``) зависит ТОЛЬКО от состава. Масса
# одного изделия ∝ ρ (плотность), поэтому цена за ШТУКУ = ₽/кг · масса ∝
# price_состав · ρ. Меньше ρ (вспенивание/упаковка ПВХ) → легче изделие → больше
# изделий из того же сырья → НИЖЕ цена за штуку (§3, пример с ПВХ).
#
# ⚠️ В §3 displayed-формула записана с делением (``price_состав / ρ``); это
# ОПИСКА — она инвертировала бы знак монотонности (меньше ρ → дороже), что
# противоречит прозаическому пояснению про вспенивание. Здесь зафиксирована
# physics-трактовка (умножение). НЕ менять знак без перечтения §3 целиком.
#
# ⚠️ Новый канал оптимизации (§3): process-переменные (T, P), не влиявшие на
# цену сырья, теперь влияют на цену ИЗДЕЛИЯ ЧЕРЕЗ ρ — числитель (price_состав)
# их не видит, множитель ρ видит. Это часто ГЛАВНЫЙ рычаг в пластиках.
# ----------------------------------------------------------------------
def price_per_item(composition_price, rho) -> np.ndarray:
    """Цена за изделие из цены состава (₽/кг) и плотности ρ (§15.6 §3).

    ``price_изд = price_состав · ρ`` (physics-трактовка, см. блок выше). Обе части
    — поэлементные массивы одинаковой формы (или скаляры). ρ — полноценный отклик
    общего суррогата (GPExpert в кампании / MoE в pipeline, как strength/gloss),
    поэтому в acquisition/оптимизации сюда подаётся его среднее
    ``surrogate.predict(X).mean``; неопределённость ``σ_ρ`` идёт в VoI (§5),
    а не в саму точечную цену.
    """
    pc = np.asarray(composition_price, float)
    r = np.asarray(rho, float)
    return pc * r


def make_item_cost_fn(composition_price_fn: Predictor,
                      rho_predictor: Predictor) -> Predictor:
    """Builder ``cost_fn(X) -> price_изд`` для :func:`optimize_desirability`.

    Собирает цену за изделие (§15.6 §3) из двух источников над СОСТАВНОЙ матрицей
    ``X = [x..., z_code...]``:

      * ``composition_price_fn(X)`` — цена состава ``price_состав(A,B,C)`` (₽/кг);
        зависит ТОЛЬКО от mixture-столбцов (process-столбцы игнорирует);
      * ``rho_predictor(X)``        — среднее ρ общего суррогата
        (``lambda X: runner.surrogates['rho'].predict(X).mean``); зависит и от
        состава, и от режима (T, P) — это и есть новый ценовой рычаг.

    Результат — ``cost_fn``, который складывается в ``optimize_desirability`` как
    обычное ``min``-свойство (цена — реальная цель, не хак: см. шапку модуля).
    """
    def cost_fn(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        pc = np.asarray(composition_price_fn(X), float).ravel()
        r = np.asarray(rho_predictor(X), float).ravel()
        return price_per_item(pc, r)

    return cost_fn


