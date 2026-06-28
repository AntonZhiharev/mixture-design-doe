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

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..design.move_bounds import BORDER_HARD, BORDER_SOFT

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

    ⚠️ ЭТО objective-AGNOSTIC оценка: считает ЛЮБОЕ удешевление, даже если ветка
    туда не пойдёт (её d_overall там не растёт). Для тяжёлой/разреженной ветки это
    переоценивает выгоду «дешёвого угла», который цель не преследует (диагностика
    §5.3). Когда нужна ЧЕСТНАЯ денежная выгода ИМЕННО ветки — см.
    :func:`price_attributed_value` (§5 per-property: деньги только за прирост
    d_overall, атрибутированный ценовой оси).
    """
    ei = np.atleast_1d(np.asarray(ei_price, float))
    best = float(ei.max()) if ei.size else 0.0
    return best * float(volume) * float(horizon)


def price_attributed_value(price_savings, *, d_overall_cur, d_overall_cand,
                           d_price_cur, d_price_cand,
                           price_weight: float, total_weight: float,
                           volume: float, horizon: float,
                           d_floor: float = 1e-6) -> float:
    """§5 per-property: денежная ценность раунда, АТРИБУТИРОВАННАЯ ценовой оси.

    Чинит objective-agnostic переоценку :func:`economic_value` (диагностика §5.3:
    «фантом дешёвого угла»). Сырое удешевление ``price_savings`` (₽/изд, EI по ρ)
    засчитывается ТОЛЬКО там, где (а) ветка реально улучшает свой ``d_overall`` и
    (б) это улучшение идёт ИМЕННО через цену. Долю цены берём из лог-разложения
    ВЗВЕШЕННОГО ГЕОМЕТРИЧЕСКОГО СРЕДНЕГО desirability (§3):

        log d_overall   = Σ_i (w_i/Σw)·log d_i
        Δlog d_overall  = Σ_i (w_i/Σw)·Δlog d_i              (best → кандидат)
        вклад_цены      = (w_price/Σw)·Δlog d_price
        α(x) = clip( вклад_цены / Δlog d_overall , 0, 1 )   если Δlog d_overall>0,
               иначе 0   (нет прироста цели → денег нет)

        economic_value = V·H · max_x [ price_savings(x)·α(x) ]   [₽ за горизонт]

    Поведение на крайних случаях:
      * дешёвый угол (цена падает, но d_overall НЕ растёт — другое свойство
        вето/проседает): Δlog d_overall ≤ 0 ⇒ α=0 ⇒ денег 0 (фантом убран);
      * улучшение чисто через ДРУГУЮ ось (цена не двигается): вклад_цены=0 ⇒ α=0;
      * улучшение чисто через цену: вклад_цены≈Δlog d_overall ⇒ α≈1 ⇒ полная цена.

    Параметры (массивы — поэлементно по кандидатам; ``*_cur`` — скаляры у текущего
    лучшего рецепта ветки)
    ----------
    price_savings   : EI на удешевление изделия по кандидатам (₽/изд, ≥0).
    d_overall_cur/cand : d_overall ветки у текущего лучшего / у кандидатов.
    d_price_cur/cand   : desirability ЦЕНОВОЙ оси у текущего лучшего / кандидатов.
    price_weight    : вес ценовой оси ``w_price`` в гео-среднем ветки.
    total_weight    : сумма весов всех осей ветки ``Σw``.
    volume, horizon : V (изд/период) и H (горизонт) — как в :func:`economic_value`.
    d_floor         : нижняя отсечка desirability перед log (вето/нули → d_floor).
    """
    ps = np.atleast_1d(np.asarray(price_savings, float))
    do_cur = max(float(d_overall_cur), float(d_floor))
    do_cand = np.clip(np.atleast_1d(np.asarray(d_overall_cand, float)),
                      float(d_floor), None)
    dp_cur = max(float(d_price_cur), float(d_floor))
    dp_cand = np.clip(np.atleast_1d(np.asarray(d_price_cand, float)),
                      float(d_floor), None)
    w = float(price_weight) / float(total_weight)
    dlog_overall = np.log(do_cand) - np.log(do_cur)
    dlog_price = np.log(dp_cand) - np.log(dp_cur)
    contrib = w * dlog_price
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(dlog_overall > 0.0,
                         np.clip(contrib / dlog_overall, 0.0, 1.0), 0.0)
    val = ps * alpha
    best = float(np.max(val)) if val.size else 0.0
    return max(best, 0.0) * float(volume) * float(horizon)




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


# ----------------------------------------------------------------------
# §15.6 §6 — Граница-сигнал → ДЕНЕЖНАЯ ТРИАДА (не флаг)
# ----------------------------------------------------------------------
@dataclass
class MoneyTriad:
    """Денежная триада для упора оптимума в границу (§6). НЕ решение — ПРЕДЛОЖЕНИЕ.

    A0.6: система НИКОГДА не двигает границу молча. Триада кладёт перед
    пользователем ТРИ цифры, решение двигать — за ним.

    * ``saving_per_period`` — экономия за границей ``X = Δprice_изд · V`` [₽/период];
    * ``acquisition_cost``  — цена добычи ``N · c_exp`` [₽] (N — оценка опытов до
      сходимости);
    * ``payback_periods``   — окупаемость ``N·c_exp / X`` [период] (сравнить с H);
    * ``worth_it``          — ``payback_periods ≤ H`` (окупается в горизонте);
    * ``var`` / ``side``    — какая граница и с какой стороны упёрлась.
    """

    var: str
    side: str
    saving_per_period: float
    acquisition_cost: float
    payback_periods: float
    horizon: float
    worth_it: bool


class HardBoundaryError(Exception):
    """Попытка предложить движение ``hard``-границы (A0.5): запрещено по
    происхождению. Триада НЕ предлагается — показываем информативный отказ."""


def money_triad(var: str, side: str, *,
                delta_price_item: float, volume: float,
                n_experiments: int, cost_exp: float,
                horizon: float) -> MoneyTriad:
    """Собрать денежную триаду §6 для одной упёршейся ``soft``-границы.

    ``delta_price_item`` — ожидаемое УДЕШЕВЛЕНИЕ изделия за границей (₽/изделие,
    напр. ``max EI_price`` из :func:`expected_price_improvement` за расширенной
    областью). ``n_experiments`` (N) — оценка числа опытов до сходимости.

    Окупаемость = ``N·c_exp / (Δprice·V)``; ∞ если экономии нет (X≤0).
    """
    saving = float(delta_price_item) * float(volume)        # ₽/период
    acq = float(n_experiments) * float(cost_exp)            # ₽
    payback = acq / saving if saving > 0 else float("inf")  # период
    return MoneyTriad(
        var=str(var), side=str(side),
        saving_per_period=saving, acquisition_cost=acq,
        payback_periods=payback, horizon=float(horizon),
        worth_it=bool(payback <= float(horizon)))


def boundary_signal(var: str, side: str, origin: str, *,
                    delta_price_item: float, volume: float,
                    n_experiments: int, cost_exp: float,
                    horizon: float) -> MoneyTriad:
    """Граница-сигнал §6: для ``soft`` — триада; для ``hard`` — отказ (A0.5).

    ``origin ∈ {soft, hard}``. ``hard`` ⇒ :class:`HardBoundaryError` (триада не
    предлагается, движение запрещено по происхождению — информативно «упёрлись в
    hard-лимит»). ``soft`` ⇒ :func:`money_triad` (предложение пользователю, A0.6).
    """
    if origin == BORDER_HARD:
        raise HardBoundaryError(
            f"граница {var}/{side} — hard (физика/закон/бюджет): движение "
            f"запрещено по происхождению (A0.5), триада не предлагается.")
    if origin != BORDER_SOFT:
        raise ValueError(f"origin должен быть '{BORDER_SOFT}'|'{BORDER_HARD}', "
                         f"дано '{origin}'.")
    return money_triad(var, side, delta_price_item=delta_price_item,
                       volume=volume, n_experiments=n_experiments,
                       cost_exp=cost_exp, horizon=horizon)


# ----------------------------------------------------------------------
# §15.6 A0.7 — ВЫРОЖДЕННОЕ (flat) направление цели → objective-gap, НЕ x-gap
# ----------------------------------------------------------------------
@dataclass
class FlatAxisResult:
    """Диагностика направления оси, в которое упёрся оптимум (аксиома A0.7).

    Когда оптимум стоит НА границе оси, наивно репортить «x-gap» (на сколько
    подвинуть переменную) НЕЛЬЗЯ: если целевая функция **плоская** по этой оси
    (∂d/∂x≡0 в окрестности), переменная **неидентифицируема** — двигать её
    бессмысленно. Тогда репортится **objective-gap** (на сколько вырастет d за
    границей), а x-gap игнорируется. Эталон A0.7: economy/P, spread=0.00e+00.

    ⚠️ Flat-статус считается на ТЕКУЩЕЙ цели (включая ``price_изд`` с ρ как
    свойство, §3): ось, плоская БЕЗ ρ, может ПЕРЕСТАТЬ быть плоской с введением
    ρ(...,P) — детектор это переоценивает, потому что ``objective_fn`` прогоняет
    РЕАЛЬНУЮ :meth:`Desirability.overall` текущей постановки (§3 «Следствие»).

    Поля:
      * ``var``           — имя оси (переменной);
      * ``flat``          — ``spread ≤ tol`` ⇒ ось вырождена/неидентифицируема;
      * ``spread``        — ``max(d) − min(d)`` вдоль оси (РЕАЛЬНАЯ desirability);
      * ``objective_gap`` — ``d_за_границей − d_на_границе`` (Δd; репортится
        ВМЕСТО x-gap, когда ось flat — это «на сколько подвинуть границу стоит»);
      * ``x_gap``         — предложение «подвинуть x» (``beyond − border``); для
        flat-оси ``None`` — двигать НЕЧЕГО (цель не зависит от оси);
      * ``identifiable``  — ``not flat`` (ось различима целью).
    """

    var: str
    flat: bool
    spread: float
    objective_gap: float
    x_gap: Optional[float]
    identifiable: bool


def axis_spread(d_values) -> float:
    """``max(d) − min(d)`` по набору desirability вдоль оси (0 для пустого/одного)."""
    d = np.atleast_1d(np.asarray(d_values, float))
    return float(d.max() - d.min()) if d.size else 0.0


def detect_flat_axis(var: str, objective_fn, axis_samples, *,
                     border_value: float, beyond_value: float,
                     tol: float = 1e-9) -> FlatAxisResult:
    """Детектор вырожденного направления цели (A0.7): flat ⇒ objective-gap.

    ``objective_fn`` — ВЕКТОРИЗОВАННАЯ функция ``t -> d_overall`` (массив на
    массив): варьирует ТОЛЬКО эту ось, всё прочее фиксирует у оптимума, и считает
    desirability через РЕАЛЬНУЮ :meth:`Desirability.overall` ТЕКУЩЕЙ постановки
    (с ценой/ρ, §3). Этим flat-статус честно переоценивается при добавлении ρ.

    Алгоритм (§3 «Следствие» + §0 A0.7):
      1. ``spread = max(d) − min(d)`` на ``axis_samples`` внутри текущих границ;
      2. ``spread ≤ tol`` ⇒ ось **flat** (неидентифицируема) ⇒ репортим
         ``objective_gap`` (Δd за границей), ``x_gap = None`` (двигать нечего);
      3. иначе ось различима ⇒ ``x_gap = beyond − border`` (обычный путь,
         триада §6 уместна), ``objective_gap`` — справочно.

    ``border_value`` — текущая граница по оси (где упёрся оптимум);
    ``beyond_value`` — значение за расширенной границей (для objective-gap).
    """
    d_in = np.atleast_1d(np.asarray(objective_fn(np.asarray(axis_samples,
                                                            float)), float))
    spread = axis_spread(d_in)
    d_border = float(np.atleast_1d(
        np.asarray(objective_fn(np.asarray([float(border_value)], float)),
                   float))[0])
    d_beyond = float(np.atleast_1d(
        np.asarray(objective_fn(np.asarray([float(beyond_value)], float)),
                   float))[0])
    objective_gap = d_beyond - d_border
    flat = spread <= float(tol)
    x_gap = None if flat else float(beyond_value) - float(border_value)
    return FlatAxisResult(var=str(var), flat=flat, spread=spread,
                          objective_gap=objective_gap, x_gap=x_gap,
                          identifiable=not flat)
