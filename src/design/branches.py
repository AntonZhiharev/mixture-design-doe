"""
design/branches.py — iteration-9 (3c): project BRANCHES (REBUILD_SPEC §5/§12).

Canon: there is ONE project-wide physics model (the per-property MoE surrogates).
A *branch* is NOT a model — it is a lightweight container of intent:

    goal       : per-property desirability specs (what "good" means here);
    budget     : how many measurement slots this branch may spend;
    history    : what it has done; status; current best recipe x*.

All branches read the SAME shared surrogates and append their measured points to
the SAME common base (each point carries an `origin` tag, no copies).

Branch acquisition (refinement+search in one score):
    acq(x) = (1-explore)·d_overall(ŷ(x))  +  explore·σ̄_n(x)
where d_overall is the branch desirability of the predicted property means and
σ̄_n is the mean predictive std across the goal properties (normalised to [0,1]).
A new point is then MEASURED ON ALL P PROPERTIES and folded into the base.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..optimize.desirability import DesirabilitySpec, Desirability


# ----------------------------------------------------------------------
@dataclass
class Branch:
    """A goal-driven exploration branch (holds NO surrogate of its own)."""
    id: str
    name: str
    goal: Dict[str, DesirabilitySpec]      # property -> desirability spec
    budget: int = 10                       # total measurement slots
    spent: int = 0                         # slots already consumed
    satisfy_at: float = 0.9                # d_overall threshold -> "satisfied"
    status: str = "active"                 # active | satisfied | exhausted
    x_best: Optional[List[float]] = None   # best recipe found so far
    d_best: float = 0.0                    # best measured branch desirability
    history: List[dict] = field(default_factory=list)
    # -- §15.6 §2: экономические атрибуты ветки (для двойного стопа §4) -
    # Ветка — живой объект; экономика эволюционирует вместе с ней (§2.1):
    #   volume   V      — объём потребления (изд/период), масштаб экономии;
    #   cost_exp c_exp  — стоимость одного эксперимента (₽/опыт), порог выгоды;
    #   horizon  H      — горизонт окупаемости (период), DEFAULT ветки; на раунд
    #                     может быть override (см. resolve_horizon, §2.1).
    # Дефолты нейтральны (V=0 ⇒ экономическая ценность=0): пока экономика не
    # задана, двойной стоп ведёт себя как чисто технический — обратная
    # совместимость со старыми ветками (рождение ветки, §2.1).
    volume: float = 0.0
    cost_exp: float = 0.0
    horizon: float = 0.0



    # -- bookkeeping ---------------------------------------------------
    def remaining(self) -> int:
        return max(0, int(self.budget) - int(self.spent))
    def resolve_horizon(self, override: Optional[float] = None) -> float:
        """Горизонт H для раунда (§2.1): override на раунд ИЛИ default ветки.

        ``override`` — разовый горизонт под дорогой эксперимент конкретного
        клиента (не меняет default ветки). ``None`` ⇒ берётся ``self.horizon``.
        """
        return float(self.horizon if override is None else override)



    def refresh_status(self) -> str:
        if self.d_best >= self.satisfy_at:
            self.status = "satisfied"
        elif self.remaining() <= 0:
            self.status = "exhausted"
        else:
            self.status = "active"
        return self.status

    def is_stagnating(self, patience: int = 2, min_delta: float = 1e-3) -> bool:
        """Детектор стагнации ветки (FinalCheckList Блок 7, §12).

        Ветка считается «застрявшей», если её лучшая измеренная desirability
        ``d_best`` не выросла более чем на ``min_delta`` за последние
        ``patience`` раундов (по записям ``history``). Достигшая цели
        (``satisfied``) ветка стагнацией НЕ считается — она просто готова.
        Требуется минимум ``patience + 1`` раундов истории.
        """
        if self.status == "satisfied":
            return False
        hist = [h for h in self.history if "d_best" in h]
        if len(hist) < int(patience) + 1:
            return False
        window = hist[-(int(patience) + 1):]
        improvement = float(window[-1]["d_best"]) - float(window[0]["d_best"])
        return improvement < float(min_delta)


    # -- (de)serialisation --------------------------------------------
    def to_state(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "goal": {k: asdict(v) for k, v in self.goal.items()},
            "budget": int(self.budget), "spent": int(self.spent),
            "satisfy_at": float(self.satisfy_at), "status": self.status,
            "x_best": list(self.x_best) if self.x_best is not None else None,
            "d_best": float(self.d_best), "history": list(self.history),
            "volume": float(self.volume), "cost_exp": float(self.cost_exp),
            "horizon": float(self.horizon),       # §15.6 §2: экономика ветки
        }

    @classmethod
    def from_state(cls, d: Mapping) -> "Branch":
        goal = {k: DesirabilitySpec(**dict(v)) for k, v in d.get("goal", {}).items()}
        xb = d.get("x_best")
        return cls(
            id=d["id"], name=d.get("name", d["id"]), goal=goal,
            budget=int(d.get("budget", 10)), spent=int(d.get("spent", 0)),
            satisfy_at=float(d.get("satisfy_at", 0.9)),
            status=d.get("status", "active"),
            x_best=list(xb) if xb is not None else None,
            d_best=float(d.get("d_best", 0.0)),
            history=list(d.get("history", [])),
            volume=float(d.get("volume", 0.0)),
            cost_exp=float(d.get("cost_exp", 0.0)),
            horizon=float(d.get("horizon", 0.0)),
        )


# ----------------------------------------------------------------------
def branch_scores(surrogates: Mapping[str, "object"],
                  goal: Mapping[str, DesirabilitySpec],
                  candidates: np.ndarray,
                  explore_frac: float = 0.3):
    """Branch acquisition over a feasible candidate set (higher = better).

    Returns ``(acq, d_pred, sigma)``:
      * acq    : blended exploit/explore score per candidate;
      * d_pred : branch desirability of the predicted property means;
      * sigma  : mean predictive std across the goal properties.
    """
    candidates = np.atleast_2d(np.asarray(candidates, float))
    missing = set(goal) - set(surrogates)
    if missing:
        raise KeyError(f"No surrogate for goal properties: {sorted(missing)}")

    means: Dict[str, np.ndarray] = {}
    sigma = np.zeros(candidates.shape[0], float)
    for name in goal:
        pred = surrogates[name].predict(candidates)
        means[name] = np.asarray(pred.mean, float).ravel()
        sigma += np.asarray(pred.std, float).ravel()
    sigma /= max(len(goal), 1)

    d_pred = Desirability(dict(goal)).overall(means)
    smax = float(sigma.max()) if sigma.size else 0.0
    sigma_n = sigma / smax if smax > 0 else np.zeros_like(sigma)
    explore_frac = float(np.clip(explore_frac, 0.0, 1.0))
    acq = (1.0 - explore_frac) * d_pred + explore_frac * sigma_n
    return acq, d_pred, sigma


def allocate_budget(branches: Mapping[str, Branch],
                    total_slots: int) -> Dict[str, int]:
    """Арбитр бюджета между ветками (3d, REBUILD_SPEC §9/§12, портфель).

    Делит ``total_slots`` между АКТИВНЫМИ ветками (status ``active`` и есть
    остаток бюджета). «Перспективность» ветки = насколько она ещё не дошла до
    своей цели ``max(satisfy_at − d_best, 0)``: дальше от цели → больше слотов.
    Если все ветки одинаково близки к цели (нулевые веса) — делим поровну.
    Раздача по слотам (метод Вебстера/Сент-Лагю), кэп = остаток бюджета ветки;
    лишние слоты, которые некому отдать (все ветки исчерпаны), не назначаются.
    """
    elig = {bid: b for bid, b in branches.items()
            if b.status == "active" and b.remaining() > 0}
    total_slots = int(total_slots)
    if not elig or total_slots <= 0:
        return {}
    weights = {bid: max(float(b.satisfy_at) - float(b.d_best), 0.0)
               for bid, b in elig.items()}
    if sum(weights.values()) <= 0:
        weights = {bid: 1.0 for bid in elig}

    alloc = {bid: 0 for bid in elig}
    left = total_slots
    while left > 0:
        open_b = {bid: weights[bid] for bid in elig
                  if alloc[bid] < elig[bid].remaining()}
        if not open_b:
            break
        # дать слот ветке с наибольшим приоритетом w/(alloc+1) (Webster)
        bid = max(open_b, key=lambda k: open_b[k] / (alloc[k] + 1))
        alloc[bid] += 1
        left -= 1
    return {bid: n for bid, n in alloc.items() if n > 0}


def propose_by_score(candidates: np.ndarray, scores: np.ndarray,
                     n_points: int, min_dist: float = 0.03) -> np.ndarray:
    """Pick up to ``n_points`` diverse candidates by descending score."""
    candidates = np.atleast_2d(np.asarray(candidates, float))
    scores = np.asarray(scores, float).ravel()
    order = np.argsort(-scores)
    chosen: List[int] = []
    for idx in order:
        if len(chosen) >= n_points:
            break
        if all(np.sqrt(((candidates[idx] - candidates[c]) ** 2).sum()) > min_dist
               for c in chosen):
            chosen.append(int(idx))
    if not chosen:
        chosen = [int(order[0])]
    return candidates[chosen]
