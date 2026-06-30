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
# Политика использования экономики на входе ядра (§4 уточнение, решение сессии).
# ----------------------------------------------------------------------
# Экономическая нога §4 считается ЧЕРЕЗ цену, атрибутированную ветке (§5). Поэтому
# она ПРИНЦИПИАЛЬНО уходит в 0 там, где ценовой рычаг исчерпан, а качественный
# headroom огромен (дешёвый-плохой угол: attr=0 при d≪ceil). Если позволить такой
# price-only ноге ВЕТИРОВАТЬ, низкий c_exp заморозит ветку в худшей точке на t=0 —
# хотя технический рычаг обязан вести качество НЕЗАВИСИМО. Политика разводит две
# роли экономики:
#
#   * ``ECON_BINDING``  — §4 каноника: экономика СВЯЗЫВАЕТ (``not_economical``
#     может остановить цикл). Дефолт — обратная совместимость.
#   * ``ECON_ADVISORY`` — внутри цикла принято решение тянуть до ПОТОЛКА /
#     СТАГНАЦИИ / СХОЖДЕНИЯ, не обращая внимания на экономику: price-only нога
#     НЕ ветирует живой техпрогресс (``Δd≥ε ∧ d<ceil`` ⇒ продолжаем). Но
#     экономика НЕ исчезает молча — ядро всё равно поднимает ``econ_red_flag``
#     (ред-флаг для показа пользователю), чтобы «невыгодно» было видно.
ECON_BINDING = "binding"
ECON_ADVISORY = "advisory"



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
                           d_floor: float = 1e-6,
                           rho_optimized: bool = False) -> float:
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
    alpha = price_attribution_alpha(
        d_overall_cur=d_overall_cur, d_overall_cand=d_overall_cand,
        d_price_cur=d_price_cur, d_price_cand=d_price_cand,
        price_weight=price_weight, total_weight=total_weight, d_floor=d_floor,
        rho_optimized=rho_optimized)
    val = ps * alpha
    best = float(np.max(val)) if val.size else 0.0
    return max(best, 0.0) * float(volume) * float(horizon)


def price_attribution_alpha(*, d_overall_cur, d_overall_cand,
                            d_price_cur, d_price_cand,
                            price_weight: float, total_weight: float,
                            d_floor: float = 1e-6,
                            rho_optimized: bool = False) -> np.ndarray:
    """Доля прироста d_overall, атрибутированная ЦЕНОВОЙ оси, по кандидатам (§5).

    Вынесено из :func:`price_attributed_value`, чтобы ОДНА реализация α(x) кормила
    и max-single путь, и БАТЧ best-of-N (:func:`best_of_n_value`) — без копий
    формулы (канон §5: деньги только за прирост цели ЧЕРЕЗ цену).

        α(x) = clip( (w_price/Σw)·Δlog d_price / Δlog d_overall , 0, 1 )
               если Δlog d_overall > 0,  иначе 0.

    Возвращает массив ``(n,)`` в ``[0,1]`` (0 там, где цель не растёт / растёт не
    через цену). Параметры — как в :func:`price_attributed_value`.

    ``rho_optimized`` (§5/§12 Гр-1, атрибуция, читающая РОЛЬ) — ρ ветки носит роль
    OPTIMIZED И одновременно питает цену (``price_изд = price_состав·ρ``). Тогда
    ВЕСЬ EI на удешевление течёт через σ_ρ (числитель price_состав детерминирован,
    см. :func:`expected_price_improvement`), а эта σ_ρ-разведка УЖЕ оправдана
    качественной ногой d_ρ. Засчитывать её ещё и деньгами — двойной счёт одной δρ.
    Поэтому ``rho_optimized=True`` зануляет ВЕСЬ ценовой разведочный канал
    (``α≡0``): денежная нога от σ_ρ = 0 (вся неопределённость ушла в качество).
    Ценовая выгода от ВЫБОРА состава (детерминированный price_состав·μ_ρ) — это уже
    технический рычаг desirability, не денежный VoI-гейт. ``Tag`` сам по себе НЕ
    спасает от фантома — спасает именно это чтение роли в атрибуции.
    """
    do_cand = np.atleast_1d(np.asarray(d_overall_cand, float))
    if bool(rho_optimized):
        return np.zeros(do_cand.shape, float)
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
    return np.asarray(alpha, float)


# ----------------------------------------------------------------------
# §4-BATCH — q-EI «улучшение ЛУЧШЕЙ из N точек» (max-тип) + оптимальный размер
# раунда N*. Чинит ЕДИНИЦЫ стопа: раунд из N опытов СТОИТ N·c_exp и приносит
# E[best-of-N], а не одну точку. Связка ОБЯЗАНА быть «best-of-N vs N·c_exp»:
#
#     E[ улучшение лучшей из N ]·V·H   >   N·c_exp
#     └──────── max-тип q-EI ────────┘     └ цена раунда ┘
#
# best-of-N растёт с N, но СУБЛИНЕЙНО (вогнуто), а N·c_exp линейна ⇒ есть N*
# (маржинальная точка перестаёт окупаться). Так маржинальная (1) и батч (2)
# трактовки сшиты. ⚠️ Слева ОБЯЗАН быть батч max-тип, НЕ max_x одной точки —
# иначе «одна точка должна окупить все N» → преждевременный стоп.
# ----------------------------------------------------------------------
def best_of_n_curve_value(composition_price, rho_mean, rho_std,
                          price_best: float, *, n_max: int,
                          volume: float, horizon: float,
                          alpha=None, n_mc: int = 512,
                          seed: Optional[int] = None) -> np.ndarray:
    """Кривая ``value(k) = E[ улучшение лучшей из k ]·V·H`` для ``k=1..n_max``.

    Денежный масштаб — то же удешевление изделия, что и в
    :func:`expected_price_improvement` (gain = ``max(price_best − price_изд, 0)``,
    ρ~N(μ,σ)); при ``alpha`` (§5 per-property) каждый кандидат домножается на свою
    долю ``α(x)`` ДО взятия максимума — деньги только за прирост цели через цену.

    Батч из ``k`` кандидатов выбирается ЖАДНО (submodular greedy): на каждом шаге
    добавляем кандидата, максимизирующего ``E[max]``. Для монотонной submodular
    ``E[max]`` маржинальные приросты НЕ ВОЗРАСТАЮТ ⇒ кривая **вогнута** по построению
    (это и проверяет N*-тест). На сходимости ``best-of-N ≈ max-single`` (всё мелко).

    Возвращает массив ``(min(n_max, n_cand),)`` — ₽ за горизонт, неубывающий.
    """
    pc = np.atleast_1d(np.asarray(composition_price, float))
    mu = np.atleast_1d(np.asarray(rho_mean, float))
    sd = np.clip(np.atleast_1d(np.asarray(rho_std, float)), 0.0, None)
    n = max(pc.shape[0], mu.shape[0], sd.shape[0])
    pc = np.broadcast_to(pc, (n,))
    mu = np.broadcast_to(mu, (n,))
    sd = np.broadcast_to(sd, (n,))
    a = (np.ones(n, float) if alpha is None
         else np.clip(np.broadcast_to(np.atleast_1d(np.asarray(alpha, float)),
                                      (n,)), 0.0, None))

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((int(n_mc), n))
    price = pc[None, :] * (mu[None, :] + sd[None, :] * Z)      # (S×n)
    gain = np.clip(float(price_best) - price, 0.0, None) * a[None, :]   # attr.

    k_max = int(min(int(n_max), n))
    running = np.zeros(int(n_mc), float)
    chosen = np.zeros(n, dtype=bool)
    curve = np.empty(k_max, float)
    for k in range(k_max):
        # E[max] при добавлении каждого ещё-не-выбранного кандидата
        cand_val = np.where(chosen, -np.inf,
                            np.maximum(running[:, None], gain).mean(axis=0))
        j = int(np.argmax(cand_val))
        running = np.maximum(running, gain[:, j])
        chosen[j] = True
        curve[k] = running.mean()
    return curve * float(volume) * float(horizon)


def best_of_n_value(composition_price, rho_mean, rho_std, price_best: float, *,
                    n_batch: int, volume: float, horizon: float,
                    alpha=None, n_mc: int = 512,
                    seed: Optional[int] = None) -> float:
    """Денежная ценность РАУНДА из ``n_batch`` опытов (₽ за горизонт), max-тип q-EI.

    ``= E[ улучшение лучшей из n_batch ]·V·H``. Сравнивать с ``n_batch·c_exp``
    (цена раунда), НЕ с одиночным ``c_exp`` — см. :func:`decide_stop`. Тонкая
    обёртка над :func:`best_of_n_curve_value` (берёт последний элемент кривой).
    """
    if int(n_batch) <= 0:
        return 0.0
    curve = best_of_n_curve_value(composition_price, rho_mean, rho_std,
                                  price_best, n_max=int(n_batch), volume=volume,
                                  horizon=horizon, alpha=alpha, n_mc=n_mc,
                                  seed=seed)
    return float(curve[-1]) if curve.size else 0.0


def optimal_round_size(curve_value, cost_exp: float) -> int:
    """Оптимальный размер раунда ``N*`` по кривой best-of-N (₽ за горизонт).

    ``N* = max{ k : value(k) − value(k−1) > c_exp }`` (``value(0)=0``) — добавляем
    точки в раунд, пока МАРЖИНАЛЬНАЯ окупается (прирост best-of-k·V·H перекрывает
    одиночный ``c_exp``). Поскольку кривая ВОГНУТА (см. :func:`best_of_n_curve_value`),
    маржиналь не возрастает ⇒ ``N*`` корректно определён первым «провалом ниже».
    ``0`` — если уже первая точка не окупается.
    """
    cv = np.atleast_1d(np.asarray(curve_value, float))
    if cv.size == 0:
        return 0
    marg = np.diff(np.concatenate(([0.0], cv)))         # value(k)-value(k-1)
    below = np.where(marg <= float(cost_exp))[0]
    return int(below[0]) if below.size else int(cv.size)


# ----------------------------------------------------------------------
# §4 — ЧИСТАЯ логика двойного стоп-критерия (+ политика экономики)
# ----------------------------------------------------------------------
@dataclass
class StopDecision:
    """Решение ядра §4: СВЯЗЫВАЮЩАЯ причина + ред-флаг экономики (для показа).

    Разделяет два разных вопроса, которые раньше схлопывались в одну строку:

      * ``reason`` — что ОСТАНОВИЛО цикл (``None`` ⇒ продолжать). Это то, на что
        реагирует движок дозабора.
      * ``econ_red_flag`` — экономика говорит «невыгодно» (``economic_value ≤
        c_exp``). Поднимается ВСЕГДА, даже когда цикл продолжается (политика
        ``ECON_ADVISORY``): price-only нога не ветирует живой техпрогресс, но её
        сигнал НЕ исчезает молча — ядро отдаёт его наверх для показа пользователю
        (ред-флаг). При ``ECON_BINDING`` он совпадает с тем, что причина —
        ``not_economical``.
    """

    reason: Optional[str]
    econ_red_flag: bool


def evaluate_stop(*, delta_d: float, d_best: float, ceil: float,
                  economic_value: float, cost_exp: float,
                  eps: float = 5e-3,
                  econ_policy: str = ECON_BINDING) -> StopDecision:
    """Двойной стоп §4 c политикой экономики. Возвращает :class:`StopDecision`.

    Технический рычаг (Δd / потолок) и экономический (``economic_value`` vs
    ``c_exp``) — РАЗНЫЕ роли. ``econ_policy`` решает, СВЯЗЫВАЕТ ли экономика:

    Общий инвариант (ОБЕ политики): ``ceil_reached`` — абсолютный приоритет.
    Упёрлись в потолок достижимого (``d_best ≥ ceil``) ⇒ улучшать НЕЧЕГО, дальше
    только движение границ/раскрытие переменных.

    ``ECON_BINDING`` (дефолт, §4 каноника) — приоритет причин:
      1. ``ceil_reached``;
      2. ``not_economical`` (``economic_value ≤ c_exp``) — улучшать есть куда, но
         денежно невыгодно (осознанный экономический стоп §1);
      3. ``stagnation`` (``Δd < ε``) — технически выдохлись, деньги ещё были бы.

    ``ECON_ADVISORY`` — экономика НЕ ветирует живой техпрогресс. Пока
    ``Δd ≥ ε ∧ d_best < ceil`` ⇒ ``reason=None`` (тянем до потолка/стагнации),
    КАКОЙ БЫ ни была экономика — это обезвреживает «мину» price-only ноги на
    дешёвом-плохом старте (``attr=0`` при ``d≪ceil``: ценовой рычаг исчерпан, а
    качественный headroom огромен — технический рычаг обязан вести качество
    независимо). Когда техпрогресс встал (``Δd < ε``) ⇒ ``stagnation``. ВО ВСЕХ
    случаях ``econ_red_flag = (economic_value ≤ c_exp)`` поднимается для показа.

    Параметры
    ----------
    delta_d        : прирост d_best за последний раунд (Δd ≥ 0 обычно).
    d_best         : текущая лучшая desirability ветки.
    ceil           : потолок достижимого (аналитический/фазовый).
    economic_value : ``max_x EI_price · V · H`` (₽ за горизонт, см. выше).
    cost_exp       : стоимость раунда (``N·c_exp``) — порог выгодности (§4-BATCH).
    eps            : технический порог Δd (ε).
    econ_policy    : ``ECON_BINDING`` | ``ECON_ADVISORY`` (см. модульный коммент).
    """
    if econ_policy not in (ECON_BINDING, ECON_ADVISORY):
        raise ValueError(
            f"econ_policy должен быть '{ECON_BINDING}'|'{ECON_ADVISORY}', "
            f"дано '{econ_policy}'.")

    tech_ceiling_ok = float(d_best) < float(ceil)
    tech_progress_ok = float(delta_d) >= float(eps)
    econ_ok = float(economic_value) > float(cost_exp)
    red_flag = not econ_ok            # «невыгодно» — ВСЕГДА сообщаем (для показа)

    # потолок — абсолютный приоритет в обеих политиках
    if not tech_ceiling_ok:
        return StopDecision(STOP_CEIL, red_flag)

    if econ_policy == ECON_ADVISORY:
        # price-only нога НЕ ветирует живой техпрогресс (обезвреженная мина t=0):
        # пока качество растёт — тянем; экономику несём как ред-флаг.
        if tech_progress_ok:
            return StopDecision(None, red_flag)
        return StopDecision(STOP_STAGNATION, red_flag)

    # ECON_BINDING — §4 каноника (приоритет причин)
    if tech_progress_ok and econ_ok:
        return StopDecision(None, red_flag)
    if not econ_ok:
        return StopDecision(STOP_NOT_ECONOMICAL, red_flag)
    return StopDecision(STOP_STAGNATION, red_flag)


def decide_stop(*, delta_d: float, d_best: float, ceil: float,
                economic_value: float, cost_exp: float,
                eps: float = 5e-3,
                econ_policy: str = ECON_BINDING) -> Optional[str]:
    """Тонкая строковая обёртка §4 над :func:`evaluate_stop` (обратная
    совместимость). Возвращает ``stop_reason`` (``None`` ⇒ продолжать) —
    ``.reason`` решения. Ред-флаг экономики при этом ОТБРАСЫВАЕТСЯ; когда он нужен
    для показа (особенно под ``ECON_ADVISORY``), вызывай :func:`evaluate_stop`
    напрямую и читай ``.econ_red_flag``.
    """
    return evaluate_stop(delta_d=delta_d, d_best=d_best, ceil=ceil,
                         economic_value=economic_value, cost_exp=cost_exp,
                         eps=eps, econ_policy=econ_policy).reason



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
