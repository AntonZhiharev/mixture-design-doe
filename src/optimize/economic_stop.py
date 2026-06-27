"""
optimize/economic_stop.py — §15.6 ШАГ 4: ДВОЙНОЙ стоп-критерий (технический И
экономический/VoI) + ``stop_reason``.

§4 (источник истины): продолжать, пока выполнено И:

    технический:   Δd ≥ ε  ∧  d < ceil
    экономический: max_x EI_₽(x) · V · H  >  c_exp        (VoI, §5)

Стоп, если нарушено ЛЮБОЕ. ``stop_reason ∈ {ceil_reached, stagnation,
not_economical}``.

----------------------------------------------------------------------
Уточнение реализации EI_₽ (binding, решение сессии §15.6):
----------------------------------------------------------------------
§4 пишет ``EI_₽`` обобщённо, но §6 ЯВНО задаёт денежный путь:
«экономия за границей X ₽/период = Δprice_изд · V». Поэтому экономическая
ценность раунда считается ЧЕРЕЗ цену изделия (а НЕ через d_overall):

    EI_price(x) = E[ max(price_best − price_изд(x), 0) ]      [₽/изделие]
    price_изд   = price_состав(A,B,C) · ρ(...)               (ШАГ 2 / §3)
    economic_value = max_x EI_price(x) · V · H               [₽ за горизонт]
    gate:  economic_value > c_exp                            (стоит ставить опыт)

Единицы сходятся: EI_price [₽/изд] · V [изд/период] = [₽/период];
· H [период] = [₽ за горизонт]; сравнение с c_exp [₽/опыт]. Неопределённость
входит через σ_ρ (honest GP-постериор, фундамент ШАГА 1): ρ сэмплируется
N(μ_ρ, σ_ρ), числитель price_состав детерминирован (зависит лишь от состава).

Разделение ответственности:
  * :func:`decide_stop` — ЧИСТАЯ логика §4 (вход: числа; выход: stop_reason);
  * :func:`expected_price_improvement` / :func:`economic_value` — денежная
    оценка §6 (одна верная реализация, заменяема). x_next по-прежнему даёт
    VoI-движок d_overall (`src/optimize/voi.py`, ШАГ 3) — это РАЗНЫЕ роли:
    куда ставить опыт (инфо-выгода) vs стоит ли вообще (денежная выгода).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# stop_reason — ровно множество §4
STOP_CEIL = "ceil_reached"
STOP_STAGNATION = "stagnation"
STOP_NOT_ECONOMICAL = "not_economical"


# ----------------------------------------------------------------------
# Денежная оценка раунда (§6): EI на цену изделия через σ_ρ
# ----------------------------------------------------------------------
def expected_price_improvement(composition_price, rho_mean, rho_std,
                               price_best: float, *,
                               n_mc: int = 512,
                               seed: Optional[int] = None) -> np.ndarray:
    """EI на УДЕШЕВЛЕНИЕ изделия по кандидатам (§6 «Δprice_изд»).

    Для каждого кандидата: ``price_изд = price_состав · ρ`` (ШАГ 2/§3), ρ
    сэмплируется ``N(μ_ρ, σ_ρ)`` (honest σ из ШАГА 1), числитель детерминирован.

        EI_price = E[ max(price_best − price_изд, 0) ]   [₽/изделие, ≥ 0]

    Parameters (поэлементно по кандидатам, broadcast скаляра допустим)
    ----------
    composition_price : ``price_состав(A,B,C)`` — ₽/кг, зависит от состава.
    rho_mean, rho_std : μ_ρ / σ_ρ общего суррогата плотности на кандидатах.
    price_best        : текущая лучшая (минимальная) цена изделия — порог EI.

    Returns массив ``(n,)`` EI на удешевление (₽/изделие).
    """
    pc = np.atleast_1d(np.asarray(composition_price, float))
    mu = np.atleast_1d(np.asarray(rho_mean, float))
    sd = np.clip(np.atleast_1d(np.asarray(rho_std, float)), 0.0, None)
    n = max(pc.shape[0], mu.shape[0], sd.shape[0])
    pc = np.broadcast_to(pc, (n,))
    mu = np.broadcast_to(mu, (n,))
    sd = np.broadcast_to(sd, (n,))

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((int(n_mc), n))
    rho_samp = mu[None, :] + sd[None, :] * Z          # (S×n)
    price = pc[None, :] * rho_samp                     # ₽/изделие
    gain = np.clip(float(price_best) - price, 0.0, None)
    return gain.mean(axis=0)                            # (n,)


def economic_value(ei_price, volume: float, horizon: float) -> float:
    """Денежная ценность лучшего кандидата за горизонт (§4/§6).

    ``economic_value = max_x EI_price(x) · V · H``  [₽ за горизонт].
    ``ei_price`` — массив по кандидатам (берётся максимум) или скаляр.
    """
    ei = np.atleast_1d(np.asarray(ei_price, float))
    best = float(ei.max()) if ei.size else 0.0
    return best * float(volume) * float(horizon)



# ----------------------------------------------------------------------
# §4 — ЧИСТАЯ логика двойного стоп-критерия
# ----------------------------------------------------------------------
def decide_stop(*, delta_d: float, d_best: float, ceil: float,
                economic_value: float, cost_exp: float,
                eps: float = 5e-3) -> Optional[str]:
    """Двойной стоп §4. Возвращает ``stop_reason`` или ``None`` (продолжать).

    Продолжать тогда и только тогда, когда выполнено **И**:

        технический:   Δd ≥ ε  ∧  d_best < ceil
        экономический: economic_value > c_exp

    Нарушено любое → стоп. Приоритет причин (когда нарушено несколько):

      1. ``ceil_reached``   — упёрлись в потолок достижимого (``d_best ≥ ceil``):
         улучшать НЕЧЕГО в текущей постановке — это сильнейший сигнал (дальше
         только движение границ/раскрытие переменных).
      2. ``not_economical`` — улучшать ЕСТЬ куда (d<ceil), но денежно невыгодно
         (``economic_value ≤ c_exp``): осознанный экономический стоп §1.
      3. ``stagnation``     — технически выдохлись (``Δd < ε``), потолок не
         достигнут и деньги ещё были бы — но прогресс встал.

    Параметры
    ----------
    delta_d        : прирост d_best за последний раунд (Δd ≥ 0 обычно).
    d_best         : текущая лучшая desirability ветки.
    ceil           : потолок достижимого (аналитический/фазовый).
    economic_value : ``max_x EI_price · V · H`` (₽ за горизонт, см. выше).
    cost_exp       : стоимость эксперимента c_exp (₽/опыт) — порог выгодности.
    eps            : технический порог Δd (ε).
    """
    tech_ceiling_ok = float(d_best) < float(ceil)
    tech_progress_ok = float(delta_d) >= float(eps)
    econ_ok = float(economic_value) > float(cost_exp)

    if tech_ceiling_ok and tech_progress_ok and econ_ok:
        return None                                   # все условия — продолжаем

    # приоритет причин (см. docstring)
    if not tech_ceiling_ok:
        return STOP_CEIL
    if not econ_ok:
        return STOP_NOT_ECONOMICAL
    return STOP_STAGNATION
