"""
optimize/desirability.py — M8: product optimisation on the simplex.

Derringer–Suich desirability (REBUILD_SPEC M8, §3):

  * per-property desirability d_i(y) in [0, 1]:
      - "max"    (larger-is-better):  ramps 0 -> 1 as y goes low -> high;
      - "min"    (smaller-is-better): ramps 1 -> 0 as y goes low -> high;
      - "target" (target-is-best):    two-sided peak at the target value;
    each with a shape exponent `s` (s>1 = stricter, s<1 = lenient).

  * overall (weighted geometric mean):
        d_overall = (Π_i d_i^{w_i})^{1/Σ w_i}
    if any d_i == 0  ->  d_overall == 0 (a hard veto, by construction).

Cost is handled as just another property with a "min" spec (grab: cost is a
real objective, not a hack) — see `optimize_desirability(cost_fn=..., cost_spec=...)`.

Optimisation is performed OVER THE CONSTRAINED SIMPLEX (grab #10: no free-R^q
gradient).  We score a feasible candidate set, then locally refine the best
point with feasibility-preserving random perturbations.

R reference: ``desirability::dOverall``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from ..core.simplex import SimplexRegion

Predictor = Callable[[np.ndarray], np.ndarray]


# ----------------------------------------------------------------------
# Per-property desirability specification
# ----------------------------------------------------------------------
@dataclass
class DesirabilitySpec:
    """One Derringer–Suich desirability transform for a single property.

    Parameters
    ----------
    kind   : "max" | "min" | "target".
    low    : lower bound of the active range.
    high   : upper bound of the active range.
    target : peak location (required for kind="target"; must satisfy low<target<high).
    s      : shape exponent for the (lower) ramp  (s>0).
    s2     : shape exponent for the upper ramp of a "target" spec (defaults to s).
    weight : importance exponent in the weighted geometric mean (w_i > 0).
    """

    kind: str
    low: float
    high: float
    target: Optional[float] = None
    s: float = 1.0
    s2: Optional[float] = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ("max", "min", "target"):
            raise ValueError(f"Unknown kind '{self.kind}' (use max|min|target).")
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
    else:  # target
        t = spec.target
        lower = ((y - lo) / (t - lo))
        upper = ((hi - y) / (hi - t))
        d = np.where((y < lo) | (y > hi), 0.0,
                     np.where(y <= t,
                              np.clip(lower, 0.0, 1.0) ** spec.s,
                              np.clip(upper, 0.0, 1.0) ** spec.s2))
    return np.clip(d, 0.0, 1.0)


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
                          process_fixed: Optional[Mapping[int, float]] = None
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

    Returns a :class:`DesirabilityResult`; при наличии process-бокса ``x`` —
    составной рецепт ``[x..., z_code...]`` (длиной ``q+d``).
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
                Z[:, j] = rng_proc.uniform(plo[j], phi[j], size=m)
        return np.hstack([Xmix, Z])

    # ---- global stage: score a feasible candidate set ----------------
    rng = np.random.default_rng(seed)
    cand = region.random_points(n_candidates, seed=seed)
    verts = region.extreme_vertices()
    cent = region.centroid().reshape(1, -1)
    mix_candidates = (np.vstack([cand, verts, cent]) if len(verts)
                      else np.vstack([cand, cent]))
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
    # Рабочий вектор: [mixture pseudocomponents (q), СВОБОДНЫЕ process-коды].
    # При d==0 размер шага == q ⇒ поток ГСЧ совпадает с прежним (golden цел).
    step_dim = q + len(free_proc)
    for s_no, gi in enumerate(start_indices):
        x_cur = candidates[gi].copy()
        d_cur = float(d_all[gi])
        w_cur = region.to_pseudo(x_cur[:q])      # work in pseudocomponents
        z_cur = x_cur[q:].copy() if d else np.empty(0)
        improved = False
        for it in range(int(refine_iters)):
            step = rng.normal(0.0, refine_scale, size=step_dim)
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
            if d:
                z_try = z_cur.copy()
                for k, j in enumerate(free_proc):
                    z_try[j] = float(np.clip(z_cur[j] + step[q + k], plo[j], phi[j]))
                x_try = np.concatenate([x_mix_try, z_try])
            else:
                x_try = x_mix_try
            p_try = evaluate_props(x_try.reshape(1, -1))
            d_try = float(desir.overall(p_try)[0])
            n_eval += 1
            if d_try > d_cur:
                d_cur, x_cur = d_try, x_try.copy()
                w_cur = region.to_pseudo(x_cur[:q])
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

    return DesirabilityResult(
        x=x_best, d_overall=d_best, d_individual=d_ind,
        properties=props_scalar, n_evaluated=n_eval,
        refined=refined, n_starts=len(start_indices), history=history,
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
    GP+MoE (как strength/gloss), поэтому в acquisition/оптимизации сюда подаётся
    его среднее ``surrogate.predict(X).mean``; неопределённость ``σ_ρ`` идёт в VoI
    (§5), а не в саму точечную цену.
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


