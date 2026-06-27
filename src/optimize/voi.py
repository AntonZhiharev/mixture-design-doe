"""
optimize/voi.py — §15.6 ШАГ 3: VoI = per-property MC-EI с разложением вклада.

Канон §15.6 §5 (форма дисконта):

    VoI = MC-EI, сэмплирование ПО СВОЙСТВАМ (раздельные GP-постериоры σ_i),
          прогон набора через НАСТОЯЩУЮ d_overall (veto/веса честно соблюдены),
          с РАЗЛОЖЕНИЕМ EI по вкладу каждого свойства:
              EI_total(x) = Σ_i EI^(i)(x)
          (вклад оси i ≈ падение EI при «заморозке» σ_i на μ_i без шума).

Почему НЕ замкнутая формула EI по агрегату (§5 «Подводные камни»):
  1. ``d_overall`` — взвешенное geo-mean с veto, НЕГАУССОВА и рвётся в нуле →
     аналитическая EI к агрегату неприменима → MC обязателен.
  2. сэмплируем КАЖДОЕ свойство из его GP-постериора N(μ_i, σ_i) НЕЗАВИСИМО и
     гоним совместный сэмпл через РЕАЛЬНУЮ :meth:`Desirability.overall` — так
     veto/веса соблюдаются ЧЕСТНО (а не «врут в обе стороны» от агрегирования).

ЗАЩИТА (§5) встроена в саму форму: дутый σ УЖЕ-добранного свойства (d_i≈1) почти
не двигает агрегат (geo-mean/veto держит d_overall у худшей оси), поэтому его EI
автоматически мал — без отдельного штрафа. Разложение вклада делает это явным:
видно, ПО КАКОЙ ОСИ разведка реально окупается (``contributions``/``limiting_axis``).

EI_₽ (денежная шкала §4/§6) — отдельный слой ШАГА 4: ``price_изд`` уже входит в
``d_overall`` min-свойством (ШАГ 2), поэтому d_overall-EI уже тянет к
дешевле-и-лучше; перевод EI·V·H в ₽ и сравнение с c_exp живёт в стоп-слое ветки.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .desirability import Desirability, DesirabilitySpec


# ----------------------------------------------------------------------
# Внутреннее: общие случайные числа (CRN) для честного разложения вклада
# ----------------------------------------------------------------------
def _d_overall_samples(order: Sequence[str],
                       mu: np.ndarray, sd: np.ndarray,
                       desir: Desirability, Z: np.ndarray) -> np.ndarray:
    """d_overall по MC-сэмплам свойств. Формы: ``mu/sd`` (P×n), ``Z`` (S×P×n).

    Каждое свойство сэмплируется НЕЗАВИСИМО: ``y_i = μ_i + σ_i·Z_i``; совместный
    сэмпл (по всем свойствам) гонится через РЕАЛЬНУЮ ``desir.overall``. Возвращает
    массив ``(S×n)`` значений d_overall. ``Z`` фиксирован снаружи (CRN), чтобы
    «заморозка» одной оси меняла ТОЛЬКО её столбец — корректное разложение вклада.
    """
    S, P, n = Z.shape
    props: Dict[str, np.ndarray] = {}
    for pi, name in enumerate(order):
        # (S×n) сэмплы свойства; затем плющим в 1D для desir.overall
        y = mu[pi][None, :] + sd[pi][None, :] * Z[:, pi, :]
        props[name] = y.reshape(-1)
    d_flat = np.asarray(desir.overall(props), float)   # (S*n,)
    return d_flat.reshape(S, n)


def _ei_from_samples(d_samples: np.ndarray, d_best: float) -> np.ndarray:
    """EI = E[max(d − d_best, 0)] по сэмплам ``(S×n)`` → ``(n,)`` на кандидата."""
    gain = np.clip(d_samples - float(d_best), 0.0, None)
    return gain.mean(axis=0)


# ----------------------------------------------------------------------
# Результат VoI-ацквизиции
# ----------------------------------------------------------------------
@dataclass
class VoIResult:
    """Итог VoI-разведки над набором кандидатов (§15.6 §5).

    * ``x_next``        — рецепт argmax EI (составной ``[x..., z...]``);
    * ``ei``            — EI_total в точке ``x_next`` (шкала d_overall);
    * ``ei_all``        — EI_total по ВСЕМ кандидатам ``(n,)``;
    * ``contributions`` — вклад каждой оси в EI в точке ``x_next`` (Σ = ei);
    * ``limiting_axis`` — ось с максимальным вкладом (КУДА вести разведку);
    * ``justified``     — EI обоснован разведкой РЕАЛЬНО лимитирующей оси
      (top-вклад совпал с осью наименьшей текущей d_i), а не дутым σ
      уже-добранного свойства (§5 «ЗАЩИТА»).
    """

    x_next: np.ndarray
    ei: float
    ei_all: np.ndarray
    contributions: Dict[str, float]
    limiting_axis: str
    justified: bool


# ----------------------------------------------------------------------
# Основной движок: per-property MC-EI + разложение вклада
# ----------------------------------------------------------------------
def mc_ei_decomposed(surrogates: Mapping[str, "object"],
                     goal: Mapping[str, DesirabilitySpec],
                     candidates: np.ndarray,
                     d_best: float,
                     *,
                     n_mc: int = 256,
                     seed: Optional[int] = None) -> VoIResult:
    """VoI = per-property MC-EI с разложением вклада по осям (§15.6 §5).

    Parameters
    ----------
    surrogates : ``name -> GP`` с ``predict(X).mean/.std`` (ОБЩИЕ суррогаты
                 проекта; ``goal`` ссылается на их подмножество).
    goal       : ``name -> DesirabilitySpec`` — цель ветки (веса/kind/veto живут
                 здесь и применяются РЕАЛЬНОЙ ``Desirability.overall``).
    candidates : допустимые составные кандидаты ``(n × (q+d))``.
    d_best     : текущая лучшая (измеренная) desirability — порог EI.
    n_mc       : число MC-сэмплов на свойство (общие Z для всех осей — CRN).

    Returns :class:`VoIResult`. Разложение вклада: вклад оси ``i`` =
    ``EI_total − EI_freeze(i)``, где ``EI_freeze(i)`` считается с σ_i→0 (ось
    «заморожена» на μ_i, без шума) при ТЕХ ЖЕ Z — остальные оси сэмплируются как
    обычно. Так дутый σ уже-добранного свойства даёт малый вклад (geo-mean/veto
    не пускает его в агрегат), а добор veto-лимитирующего — большой (§5 ЗАЩИТА).
    """
    candidates = np.atleast_2d(np.asarray(candidates, float))
    n = candidates.shape[0]
    order: List[str] = list(goal)
    missing = set(order) - set(surrogates)
    if missing:
        raise KeyError(f"No surrogate for goal properties: {sorted(missing)}")
    if n == 0:
        raise ValueError("Пустой набор кандидатов для VoI.")

    P = len(order)
    mu = np.empty((P, n), float)
    sd = np.empty((P, n), float)
    for pi, name in enumerate(order):
        pred = surrogates[name].predict(candidates)
        mu[pi] = np.asarray(pred.mean, float).ravel()
        sd[pi] = np.clip(np.asarray(pred.std, float).ravel(), 0.0, None)

    desir = Desirability(dict(goal))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((int(n_mc), P, n))          # CRN: общие для всех осей

    # EI_total: все оси сэмплируются
    d_all = _d_overall_samples(order, mu, sd, desir, Z)
    ei_all = _ei_from_samples(d_all, d_best)            # (n,)

    bi = int(np.argmax(ei_all))
    ei_star = float(ei_all[bi])

    # Разложение вклада в точке argmax: заморозить ось i (σ_i→0) при ТЕХ ЖЕ Z
    raw: Dict[str, float] = {}
    mu_b = mu[:, bi:bi + 1]
    sd_b = sd[:, bi:bi + 1]
    Z_b = Z[:, :, bi:bi + 1]
    d_full_b = _d_overall_samples(order, mu_b, sd_b, desir, Z_b)
    ei_full_b = float(_ei_from_samples(d_full_b, d_best)[0])
    for pi, name in enumerate(order):
        sd_frozen = sd_b.copy()
        sd_frozen[pi] = 0.0                              # ось заморожена на μ_i
        d_frozen = _d_overall_samples(order, mu_b, sd_frozen, desir, Z_b)
        ei_frozen = float(_ei_from_samples(d_frozen, d_best)[0])
        raw[name] = max(ei_full_b - ei_frozen, 0.0)      # вклад разведки оси i

    # нормировка вкладов к ei_full_b (взаимодействия осей → сумма raw ≠ ei)
    total_raw = sum(raw.values())
    if total_raw > 0:
        contributions = {k: ei_full_b * v / total_raw for k, v in raw.items()}
    else:
        contributions = {k: 0.0 for k in order}

    limiting_axis = max(contributions, key=lambda k: contributions[k])

    # ЗАЩИТА (§5): обоснован ли высокий EI разведкой РЕАЛЬНО лимитирующей оси?
    # лимитирующая ось агрегата = свойство с наименьшей текущей d_i (μ-уровень).
    mu_at_b = {name: mu[order.index(name), bi:bi + 1] for name in order}
    d_ind_mu = {name: float(np.asarray(v, float)[0])
                for name, v in desir.individual(mu_at_b).items()}
    worst_axis = min(d_ind_mu, key=lambda k: d_ind_mu[k])
    justified = (limiting_axis == worst_axis) if ei_star > 0 else False

    return VoIResult(
        x_next=candidates[bi].copy(), ei=ei_star, ei_all=ei_all,
        contributions=contributions, limiting_axis=limiting_axis,
        justified=bool(justified))
