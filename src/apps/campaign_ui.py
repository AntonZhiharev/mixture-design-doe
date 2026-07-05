"""apps/campaign_ui.py — Streamlit-вкладка «Кампания» (ТЗ v1.1, §16/§16.1).

ШАГ 4: тонкий UI поверх :mod:`src.apps.campaign` (ШАГ 1–3). Канон «логика+тест,
потом UI» соблюдён: вся честность живёт в campaign-слое; здесь — только показ и
кнопки. Разделение:

  * ЧИСТЫЕ хелперы (``build_demo_campaign_runner``, ``role_table_dataframe``,
    ``spawn_review_dataframe``, ``goal_editor_dataframe``,
    ``workbench_points_dataframe``) НЕ зовут Streamlit — тестируются напрямую;
  * :func:`render_campaign` рисует вкладку через ``st`` (тест — headless AppTest).

UI работает с :class:`CampaignController` поверх ОТДЕЛЬНОГО
:class:`MixtureProcessRunner` (составная область mixture×process с ρ), который не
смешивается с pipeline-runner-ом M1–M8: это другая модель проекта (канон §5/§12,
одна модель физики на проект). Демо-оракул синтетический и детерминированный —
чтобы вкладку можно было запустить без реальной лаборатории.

A0.6: всё, что меняет состояние (смена роли, spawn, раунд, правка целей), делает
ТОЛЬКО явная кнопка пользователя; роли и денежный канал показываются read-only.
"""
from __future__ import annotations

import math
import zlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st


from ..core.schema import ModelSpec, ProjectSchema, VariableBlock
from ..core.simplex import parts_ranges_to_fraction_bounds
from ..optimize.desirability import DesirabilitySpec
from ..apps.mixture_process_runner import MixtureProcessRunner
from ..apps import campaign as cv
from ..apps import campaign_screening as csx
from ..apps import campaign_state as cs

from ..design.branches import ROLE_OPTIMIZED, ROLE_PRICE_INPUT
from ..design.blocking import blocking_diagnostics



# ----------------------------------------------------------------------
# Синтетический детерминированный оракул демо-кампании (без лаборатории)
# ----------------------------------------------------------------------
_DEMO_PRICE = {"A": 95.0, "B": 200.0, "C": 23.0}   # ₽/кг состава (известны)


class _DemoOracle:
    """Оракул демо: 3 свойства от составных координат ``[A,B,C,T,P]``.

    ``rho`` (плотность) — полноценный отклик (множитель цены изделия, §3); цена
    состава детерминирована (:func:`demo_price_fn`). Линейно-разнообразные функции
    достаточно богаты, чтобы роли/каналы было видно, и дёшевы для GP-фита.
    """

    property_names = ["strength", "gloss", "rho"]

    def evaluate(self, Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        A, B, C = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        T = Xc[:, 3] if Xc.shape[1] > 3 else np.zeros(len(Xc))
        P = Xc[:, 4] if Xc.shape[1] > 4 else np.zeros(len(Xc))
        strength = 6.0 * A + 5.0 * B + 4.0 * C + 3.0 * T
        gloss = 3.0 * A + 6.0 * B + 5.0 * C + 4.0 * P
        rho = 0.8 * A + 1.0 * B + 1.4 * C
        return np.column_stack([strength, gloss, rho])


def demo_price_fn(Xc) -> np.ndarray:
    """Цена состава ₽/кг = доли·цены компонентов (детерминирована, без процесса)."""
    Xc = np.atleast_2d(np.asarray(Xc, float))
    w = np.array([_DEMO_PRICE["A"], _DEMO_PRICE["B"], _DEMO_PRICE["C"]], float)
    return Xc[:, :3] @ w


def build_demo_campaign_runner(*, seed: int = 7, n_seed: int = 14
                               ) -> MixtureProcessRunner:
    """Собрать демо-кампанию: runner + общий пул + две КОНТРАСТНЫЕ ветки.

    Контраст под И-5/Гр-1 (виден в UI сразу):
      * ``premium`` — ρ НЕ цель, питает цену ⇒ роль PRICE_INPUT (канал ALIVE);
      * ``rho_focus`` — ρ в цели (min) И питает цену ⇒ роль OPTIMIZED (канал
        ZEROED): денежная нога занулена, двойной счёт δρ убран.
    Обе ветки имеют экономику (V/c_exp/H), чтобы объяснение §16.1 показывало ₽.
    """
    mix = VariableBlock.mixture(["A", "B", "C"])
    proc = VariableBlock.process(["T", "P"], lower=[0.0, 0.0], upper=[1.0, 1.0])
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    runner = MixtureProcessRunner(schema, _DemoOracle(),
                                  baseline=[1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
                                  seed=int(seed), n_restarts=2)
    runner.seed_initial(n=int(n_seed), seed=int(seed))

    price_spec = DesirabilitySpec("min", low=0.0, high=300.0, weight=0.5)
    runner.add_branch(
        "premium", {"strength": DesirabilitySpec("max", low=2.0, high=12.0),
                    "gloss": DesirabilitySpec("max", low=1.0, high=13.0)},
        budget=20, satisfy_at=1.1, branch_id="premium")
    runner.set_branch_cost("premium", demo_price_fn, price_spec,
                           rho_property="rho")
    runner.add_branch(
        "rho_focus", {"strength": DesirabilitySpec("max", low=2.0, high=12.0),
                      "rho": DesirabilitySpec("min", low=0.5, high=1.5)},
        budget=20, satisfy_at=1.1, branch_id="rho_focus")
    runner.set_branch_cost("rho_focus", demo_price_fn, price_spec,
                           rho_property="rho")
    for bid in ("premium", "rho_focus"):
        b = runner.branches[bid]
        b.volume, b.cost_exp, b.horizon = 1.0e4, 1.0e-3, 100.0
    return runner


# ----------------------------------------------------------------------
# §17.4 (Ш3b) — РЕАЛЬНЫЙ сетап: ручной оракул + сборка раннера из формы
# ----------------------------------------------------------------------
class ManualOracle:
    """Оракул РУЧНОГО сетапа (§17.4): несёт имена свойств; истинные Y — от пользователя.

    Реальная лаборатория меряет ВНЕ системы, поэтому :meth:`evaluate` НЕ выдаёт
    себя за настоящую истину — это лишь ДЕТЕРМИНИРОВАННЫЙ демо-генератор для кнопки
    «Заполнить тестовыми» (прогоны без лаборатории). Коэффициенты стабильно
    выводятся из имени свойства (``crc32`` — воспроизводимо между процессами), но
    это синтетика: реальные отклики всегда вносит пользователь (``commit_seed`` /
    ``commit_measured``, A0.6). ``evaluate`` принимает ПОЛНЫЙ составной вектор
    ``Xc`` (n×dim) — как того требует контракт раннера.
    """

    def __init__(self, property_names: Sequence[str]):
        self.property_names: List[str] = list(property_names)

    def evaluate(self, Xc) -> np.ndarray:
        Xc = np.atleast_2d(np.asarray(Xc, float))
        n, dim = Xc.shape
        cols: List[np.ndarray] = []
        for name in self.property_names:
            rng = np.random.default_rng(zlib.crc32(str(name).encode("utf-8")))
            w = rng.uniform(0.5, 2.0, size=dim)
            b = float(rng.uniform(0.0, 1.0))
            cols.append(Xc @ w + b)
        return np.column_stack(cols) if cols else np.empty((n, 0), float)


def is_manual_campaign(runner) -> bool:
    """True, если истину кампании вносит пользователь (:class:`ManualOracle`).

    P0: для таких кампаний авто-оракульные действия (прогон раунда демо-оракулом)
    скрываются — ``ManualOracle.evaluate`` лишь синтетический демо-генератор, и
    авто-раунд молча записал бы в РЕАЛЬНУЮ базу выдуманные Y (A0.6). Загруженные
    с диска кампании тоже manual (``campaign_state`` реконструирует ManualOracle).
    Чистая (без Streamlit)."""
    return isinstance(getattr(runner, "oracle", None), ManualOracle)


def build_setup_runner(*, mixture_names: Sequence[str],
                       process_names: Sequence[str],
                       process_lower: Sequence[float],
                       process_upper: Sequence[float],
                       response_names: Sequence[str],
                       mixture_lower: Optional[Sequence[float]] = None,
                       mixture_upper: Optional[Sequence[float]] = None,
                       baseline: Optional[Sequence[float]] = None,
                       seed: int = 0, n_restarts: int = 2
                       ) -> MixtureProcessRunner:
    """§17.4: собрать ``MixtureProcessRunner`` РЕАЛЬНОГО сетапа (ручной оракул).

    Составная область — симплекс {mixture} × куб {process} СРАЗУ (процесс-параметры
    с самого старта, §17.4); отклики — имена пользователя. Оракул —
    :class:`ManualOracle` (Y вносится вручную через seed/branch-циклы §17.2/§17.4;
    ``evaluate`` — лишь демо-заполнение). База ПУСТА, суррогатов нет: стартовый
    дизайн предлагается (``propose_seed``) и меряется пользователем
    (``commit_seed``). ``baseline`` по умолчанию — равномерная смесь (1/q) +
    середина каждого процесс-интервала.
    """
    mixture_names = [str(s) for s in mixture_names]
    process_names = [str(s) for s in process_names]
    response_names = [str(s) for s in response_names]
    if not mixture_names:
        raise ValueError("Нужен хотя бы один компонент смеси.")
    if not process_names:
        raise ValueError("Нужен хотя бы один процесс-параметр "
                         "(§17.4: процесс-параметры задаются сразу).")
    if not response_names:
        raise ValueError("Нужен хотя бы один отклик (свойство).")
    pl = [float(v) for v in process_lower]
    pu = [float(v) for v in process_upper]
    if len(pl) != len(process_names) or len(pu) != len(process_names):
        raise ValueError("Число границ процесса не совпадает с числом параметров "
                         f"({len(process_names)}).")

    mix = VariableBlock.mixture(mixture_names, lower=mixture_lower,
                                upper=mixture_upper)
    proc = VariableBlock.process(process_names, lower=pl, upper=pu)
    model = ModelSpec(cross_level="full-cross", mixture_order="quadratic",
                      process_order="quadratic")
    schema = ProjectSchema.mixture_process(mix, proc, model=model)
    oracle = ManualOracle(response_names)
    if baseline is None:
        q = len(mixture_names)
        baseline = [1.0 / q] * q + [(lo + hi) / 2.0 for lo, hi in zip(pl, pu)]
    return MixtureProcessRunner(schema, oracle, baseline=list(baseline),
                                seed=int(seed), n_restarts=int(n_restarts))


def setup_coord_names(runner) -> List[str]:
    """Имена составных координат ТЕКУЩЕЙ схемы: mixture-компоненты + process-оси."""
    sch = runner.current_schema
    return list(sch.mixture_names) + list(sch.process_names)


def process_code_to_real(runner, X):
    """Составной ``X`` (процесс в коде [0,1]) → копия с процесс-осями в РЕАЛЬНЫХ
    единицах (замечание 2). Mixture-доли остаются как есть (они уже физические).

    Раннер хранит точки в внутреннем коде [0,1] по каждой процесс-оси; для показа
    и Excel пользователю нужны абсолютные величины (T=150…200 °C, а не 0…1).
    Денормализация — покомпонентная :meth:`VariableBlock.from_code` процесс-блока
    ТЕКУЩЕЙ схемы (обратимо к нормировке движка). Чистая (без Streamlit)."""
    X = np.atleast_2d(np.asarray(X, float)).copy()
    pb = runner.current_schema.process_block()
    q = len(runner.current_schema.mixture_names)
    if pb is not None:
        d = len(pb.names)
        if X.shape[1] >= q + d and d > 0:
            for i in range(len(X)):
                X[i, q:q + d] = pb.from_code(X[i, q:q + d])
    return X



def make_linear_price_fn(prices: Sequence[float]):
    """§17.5: ЦЕНА СОСТАВА ₽/кг из цен компонентов — линейная нога ρ (Ш4).

    Возвращает callable ``Xc → цена состава`` = Σ(доля_i·цена_i) по mixture-долям
    (первые ``len(prices)`` координат составного вектора; процесс-оси на цену
    состава не влияют, как в демо). Детерминирована и чиста (без Streamlit) —
    тестируется напрямую. Используется как ``composition_price_fn`` в
    :meth:`CampaignController.create_branch` (item-цена = состав·ρ, §3/§15.6).
    Единый источник — :func:`campaign_state.linear_price_fn`: результат несёт
    сериализуемый дескриптор ``price_spec``, поэтому ценовая нога переживает
    save/load кампании (C2, §17.6.1).
    """
    return cs.linear_price_fn(prices)


# ----------------------------------------------------------------------
# §17.4 — научная терминология UI (замена жаргона; единый источник ярлыков)
# ----------------------------------------------------------------------
# «Ценовая нога» — внутренний жаргон; в UI показываем операционный термин.
COST_MODEL_LABEL = "Модель себестоимости изделия (плотность ρ → ₽/изд)"
# Единицы по умолчанию (замечание 7): валюта и масса выносятся в подписи/Excel.
CURRENCY_UNIT = "₽"
MASS_UNIT = "кг"


def recommended_seed_size(q: int, d: int) -> int:
    """§17.4 (замечание 4): рекомендуемый размер стартового (скрининг) дизайна.

    Универсальная формула от числа компонентов смеси ``q`` и числа
    процесс-параметров ``d``. Первый этап — СКРИНИНГ: оцениваем главные эффекты и
    смесь-процессные взаимодействия кросс-модели «смесь-линейно × процесс-линейно»,
    число членов которой ``P = q·(1 + d)``. Добавляем ~50 % запаса на остаточную
    дисперсию/lack-of-fit: ``N = P + ⌈P/2⌉``, но не меньше ``q + d + 1`` (иначе
    даже главные эффекты не разрешимы). Пример: q=3, d=2 → P=9 → N=14.
    """
    q = int(q)
    d = int(d)
    if q < 1:
        raise ValueError("Нужен хотя бы один компонент смеси (q ≥ 1).")
    if d < 0:
        raise ValueError("Число процесс-параметров не может быть отрицательным.")
    p_terms = q * (1 + d)
    n = p_terms + math.ceil(p_terms / 2)
    return int(max(n, q + d + 1))


def mixture_amounts_to_fractions(amounts: Sequence[float]) -> np.ndarray:
    """Части (любые ≥0) → доли состава (Σ=1) нормировкой по сумме (замечание 1).

    Программа умеет хранить состав и в ЧАСТЯХ, и в ДОЛЯХ и переводить одно в
    другое: доли = части / Σ(части). Пустой/отрицательный/нулевой ввод — явный
    отказ (A0.6: не нормируем молча мусор)."""
    a = np.asarray(amounts, float).ravel()
    if a.size == 0:
        raise ValueError("Ожидается непустой вектор количеств компонентов.")
    if np.any(a < 0):
        raise ValueError("Количества компонентов не могут быть отрицательными.")
    s = float(a.sum())
    if s <= 0:
        raise ValueError("Сумма количеств должна быть положительной.")
    return a / s


def resolve_mixture_bounds(n: int, lower_txt: str, upper_txt: str, *,
                           mode: str = "fractions"):
    """Границы состава в ДОЛЯХ [0,1] из ввода в долях ИЛИ частях (замечание 1).

    ``mode='fractions'`` — границы уже доли; ``mode='parts'`` — части, переводятся
    в доли через :func:`mixture_utils.convert_parts_to_proportions` (нормировка по
    сумме верхних границ). Если границы не заданы (пусто) — возвращает
    ``(None, None)`` (полный симплекс). Число границ обязано совпасть с ``n``.
    """
    lo = _parse_floats(lower_txt)
    hi = _parse_floats(upper_txt)
    if not lo and not hi:
        return None, None
    if lo is None or hi is None:
        raise ValueError("Границы состава — числа через запятую.")
    if len(lo) != int(n) or len(hi) != int(n):
        raise ValueError(f"Границ состава должно быть по {int(n)} "
                         "(по числу компонентов смеси).")
    if mode == "parts":
        from ..utils.mixture_utils import convert_parts_to_proportions
        pairs = convert_parts_to_proportions(list(zip(lo, hi)))
        lo = [float(p[0]) for p in pairs]
        hi = [float(p[1]) for p in pairs]
    return lo, hi


# ----------------------------------------------------------------------
# C3 (§17.6.1) — выгрузка ОБЩЕЙ базы кампании + расход сырья (доли↔части)
# ----------------------------------------------------------------------
def campaign_base_dataframe(runner, *, batch_kg: Optional[float] = None
                            ) -> pd.DataFrame:
    """§17.6.1 (C3): ОБЩАЯ база кампании → таблица для показа/Excel (read-only).

    По строке на опыт общей базы (``runner.points``, И-1): сквозной «№ опыта»,
    человекочитаемый origin, составные координаты (mixture-доли + процесс в
    РЕАЛЬНЫХ единицах) и измеренные отклики по всем P. При заданном ``batch_kg``
    добавляются столбцы расхода сырья ``{компонент} ({кг})`` = доля·batch —
    сколько взвесить на партию (замечание 7: единицы нужны лаборатории). Пустая
    база → пустой DataFrame.
    """
    coord_names = setup_coord_names(runner)
    props = list(runner.property_names)
    X = getattr(runner, "X", None)
    if X is None or len(X) == 0:
        return pd.DataFrame()
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.atleast_2d(np.asarray(getattr(runner, "Y", np.empty((len(X), 0))),
                                 float))
    origins = list(getattr(runner, "origin", []) or [])
    # blocking: метки партий (блоков) активных точек — порядок совпадает с X
    try:
        blocks = list(runner.point_blocks())
    except Exception:  # noqa: BLE001 — блоки не критичны для показа базы
        blocks = []
    has_block_names = bool(getattr(runner, "block_names", {}) or {})
    mix_names = list(runner.current_schema.mixture_names)
    n_mix = len(mix_names)
    # Показ процесса в АБСОЛЮТНЫХ единицах (замечание 2): mixture-доли не трогаем,
    # процесс-оси денормализуем из внутреннего кода [0,1]. Расход сырья — по долям.
    Xreal = process_code_to_real(runner, X)

    rows: List[Dict[str, Any]] = []
    for i in range(len(X)):
        row: Dict[str, Any] = {"№ опыта": i + 1}
        og = origins[i] if i < len(origins) else ""
        row["источник"] = origin_label(runner, og)
        row["Блок"] = int(blocks[i]) if i < len(blocks) else 1
        if has_block_names:
            row["Партия"] = block_display(runner, row["Блок"])
        for j, cn in enumerate(coord_names[:X.shape[1]]):
            row[cn] = round(float(Xreal[i, j]), 4)

        if batch_kg is not None and float(batch_kg) > 0:
            for j, cn in enumerate(mix_names):
                row[f"{cn} ({MASS_UNIT})"] = round(
                    float(X[i, j]) * float(batch_kg), 4)
        for k, pn in enumerate(props):
            row[f"{pn} (изм.)"] = (round(float(Y[i, k]), 4)
                                   if k < Y.shape[1] else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def measured_responses_editor_df(runner) -> pd.DataFrame:
    """§17.2.1: общая база → таблица-редактор ИЗМЕРЕННЫХ откликов (правка ошибок ввода).

    Строка на точку ``runner.points`` В ПОРЯДКЕ базы (индекс строки = «№ опыта» − 1
    = ``point_index`` для :meth:`MixtureProcessRunner.correct_measured`): сквозной
    номер, человекочитаемый источник и ТЕКУЩЕЕ измеренное значение каждого свойства
    (по ``property_names``). Редактируются только столбцы-отклики; координаты и
    происхождение неизменны (И-1). Чистый хелпер (без Streamlit) — тестируется
    напрямую и служит эталоном сравнения «старое↔новое» для коррекции."""
    props = list(runner.property_names)
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(getattr(runner, "points", []) or []):
        y = getattr(p, "Y", {}) or {}
        og = (p.origin_tag.get("origin", "seed")
              if getattr(p, "origin_tag", None) else "seed")
        row: Dict[str, Any] = {"№ опыта": i + 1,
                               "источник": origin_label(runner, og)}
        for pn in props:
            val = y.get(pn, None)
            try:
                row[pn] = float(val) if val is not None else None
            except (TypeError, ValueError):
                row[pn] = None
        rows.append(row)
    return pd.DataFrame(rows)


def campaign_base_excel_bytes(runner, *, batch_kg: Optional[float] = None
                              ) -> bytes:
    """§17.6.1 (C3): ОБЩАЯ база кампании → xlsx-байты (для кнопки скачивания).

    Лист «База опытов» = :func:`campaign_base_dataframe` (с расходом сырья, если
    задан ``batch_kg``). Чистый хелпер (без Streamlit) — тестируется напрямую;
    возвращает готовые к отдаче байты .xlsx (openpyxl)."""
    import io
    df = campaign_base_dataframe(runner, batch_kg=batch_kg)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        (df if not df.empty else pd.DataFrame({"инфо": ["база пуста"]})).to_excel(
            xw, sheet_name="База опытов", index=False)
    buf.seek(0)
    return buf.getvalue()


def seed_design_dataframe(runner, Xs, Ys=None, *, batch_kg: Optional[float] = None
                          ) -> pd.DataFrame:
    """§17.4 (C3): предложенный СТАРТОВЫЙ дизайн → таблица для показа/Excel.

    По строке на предложенную seed-точку: будущий «№ опыта» (база пока пуста ⇒
    1…N), заблокированные составные координаты (mixture-доли + процесс в РЕАЛЬНЫХ
    единицах) и столбцы откликов ``{свойство} (lab)`` (пустые — места под ручной
    ввод в лаборатории, либо уже внесённые ``Ys``). При заданном ``batch_kg``
    добавляются столбцы расхода сырья ``{компонент} ({кг})`` = доля·batch —
    сколько взвесить на опыт (замечание 7). Чистый хелпер — тестируется напрямую.
    """
    coord_names = setup_coord_names(runner)
    props = list(runner.property_names)
    Xs = np.atleast_2d(np.asarray(Xs, float))
    ncoord = Xs.shape[1]
    mix_names = list(runner.current_schema.mixture_names)
    Ya = np.atleast_2d(np.asarray(Ys, float)) if Ys is not None else None
    # Показ процесса в АБСОЛЮТНЫХ единицах (замечание 2): mixture-доли не трогаем,
    # процесс-оси денормализуем из внутреннего кода [0,1]. Расход сырья считаем по
    # ДОЛЯМ (mixture), поэтому берём их из исходного Xs (real == code для mixture).
    Xreal = process_code_to_real(runner, Xs)

    # blocking: оптимальные метки партий стартового дизайна (детерминированы по
    # seed раннера ⇒ при commit_seed точки получат ЭТИ ЖЕ метки)
    seed_blocks = None
    if int(getattr(runner, "n_blocks_start", 1)) > 1 and len(Xs):
        try:
            seed_blocks = np.asarray(runner.seed_block_labels(Xs), int)
        except Exception:  # noqa: BLE001 — блоки не критичны для показа
            seed_blocks = None

    has_block_names = bool(getattr(runner, "block_names", {}) or {})
    nums = list(experiment_index(len(runner.points), len(Xs)))
    rows: List[Dict[str, Any]] = []
    for i in range(len(Xs)):
        row: Dict[str, Any] = {"№ опыта": nums[i]}
        if seed_blocks is not None:
            row["Блок"] = int(seed_blocks[i])
            if has_block_names:
                row["Партия"] = block_display(runner, int(seed_blocks[i]))
        for j, cn in enumerate(coord_names[:ncoord]):
            row[cn] = round(float(Xreal[i, j]), 4)
        if batch_kg is not None and float(batch_kg) > 0:
            for j, cn in enumerate(mix_names):
                row[f"{cn} ({MASS_UNIT})"] = round(
                    float(Xs[i, j]) * float(batch_kg), 4)
        for k, pn in enumerate(props):
            row[f"{pn} (lab)"] = (round(float(Ya[i, k]), 4)
                                  if Ya is not None and k < Ya.shape[1]
                                  else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)



def block_display(runner, b: int) -> str:
    """Отображаемое имя блока: пользовательское имя или номер (чистая).

    Имена блоков (:attr:`MixtureProcessRunner.block_names`) — метаданные показа
    (оператор / партия сырья / смена): помогают понять ПРИНЦИП блокировки, на
    математику не влияют. Нет имени → строка с номером."""
    names = getattr(runner, "block_names", {}) or {}
    nm = str(names.get(int(b), "")).strip()
    return nm if nm else str(int(b))


def seed_blocking_caption(runner, Xs) -> str:
    """Iteration 28+: ВИДИМАЯ сводка блокировки стартового дизайна (подпись).

    Blocking был «невидим»: при ``n_blocks_start == 1`` (дефолт) таблица
    намеренно не несёт столбца «Блок», и в UI не оставалось НИ ОДНОГО
    упоминания блокировки — пользователь не знал, что она есть и как её
    включить. Возвращает строку-подпись под seed-таблицей:

      * 1 партия  — подсказка, что blocking выключен и как включить;
      * nb > 1    — размеры партий и цена блокировки (потеря D-информации
        модельных термов, :func:`design.blocking.blocking_diagnostics`).

    Чистая (без Streamlit) — тестируется напрямую.
    """
    nb = max(1, int(getattr(runner, "n_blocks_start", 1)))
    Xs = np.atleast_2d(np.asarray(Xs, float))
    if nb <= 1 or len(Xs) == 0:
        return ("🧱 Блокировка выключена: весь стартовый план — ОДНА партия "
                "(столбца «Блок» нет). Если опыты нельзя поставить одной "
                "партией / за один день — увеличьте «Партий (блоков)» выше: "
                "план оптимально разобьётся на партии и появится столбец "
                "«Блок».")
    factor = str(getattr(runner, "block_factor", "") or "").strip()
    factor_txt = f" Фактор блокировки: {factor}." if factor else ""
    try:
        lab = np.asarray(runner.seed_block_labels(Xs), int)
        q = len(runner.current_schema.mixture_names)
        diag = blocking_diagnostics(
            Xs[:, :q], lab, model=getattr(runner, "gp_mean_model", "quadratic"))
        sizes = ", ".join(
            f"блок {b} «{block_display(runner, b)}»: {c} оп."
            if block_display(runner, b) != str(b) else f"блок {b}: {c} оп."
            for b, c in sorted(diag["block_sizes"].items()))
        return (f"🧱 Блокировка включена: {nb} партий ({sizes}); потеря "
                f"D-информации модельных термов ≈ {diag['d_loss_pct']:.1f}% "
                "(метки — в столбце «Блок»; показ = фиксация: при "
                f"commit_seed точки получат ЭТИ ЖЕ блоки).{factor_txt}")
    except Exception:  # noqa: BLE001 — подпись не должна ломать seed-цикл
        return (f"🧱 Блокировка включена: {nb} партий — метки в столбце "
                "«Блок».")


def base_blocking_caption(runner) -> str:
    """Iteration 28+: сводка блокировки ОБЩЕЙ базы кампании (подпись).

    Показывается под таблицей общей базы: сколько партий (блоков), их размеры
    и цена блокировки по :meth:`MixtureProcessRunner.blocking_summary`.
    Одна партия / пустая база → пустая строка (подпись не показывается).
    Чистая (без Streamlit) — тестируется напрямую."""
    try:
        bs = runner.blocking_summary()
        if int(bs.get("n_blocks", 0)) <= 1:
            return ""
        sizes = ", ".join(
            f"блок {b} «{block_display(runner, b)}»: {c} оп."
            if block_display(runner, b) != str(b) else f"блок {b}: {c} оп."
            for b, c in sorted(bs["block_sizes"].items()))
        loss = bs.get("d_loss_pct")
        loss_txt = (f"; потеря D-информации ≈ {float(loss):.1f}%"
                    if loss is not None and np.isfinite(float(loss)) else "")
        factor = str(getattr(runner, "block_factor", "") or "").strip()
        factor_txt = f" Фактор блокировки: {factor}." if factor else ""
        return (f"🧱 Партии (блоки) базы: {bs['n_blocks']} ({sizes}){loss_txt}. "
                f"Каждый добор ветки автоматически получает НОВЫЙ блок.{factor_txt}")
    except Exception:  # noqa: BLE001 — подпись не должна ломать показ базы
        return ""


def seed_design_excel_bytes(runner, Xs, Ys=None, *,
                            batch_kg: Optional[float] = None) -> bytes:
    """§17.4 (C3): предложенный стартовый дизайн → xlsx-байты (кнопка скачивания).

    Лист «Стартовый дизайн» = :func:`seed_design_dataframe` (с расходом сырья, если
    задан ``batch_kg``; пустые «(lab)» — места под ручной ввод откликов). Чистый
    хелпер (без Streamlit) — тестируется напрямую; отдаёт готовые байты .xlsx."""
    import io
    df = seed_design_dataframe(runner, Xs, Ys, batch_kg=batch_kg)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        (df if not df.empty else pd.DataFrame({"инфо": ["дизайн пуст"]})).to_excel(
            xw, sheet_name="Стартовый дизайн", index=False)
    buf.seek(0)
    return buf.getvalue()


def branch_recipe_dataframe(runner, branch_id, *, batch_kg: Optional[float] = None,
                            n_candidates: int = 2000, refine_iters: int = 200,
                            n_starts: int = 5) -> pd.DataFrame:
    """§17.6.1 (C3): рекомендованный РЕЦЕПТ ветки x* → одна строка (показ/Excel).

    Запускает M8-argmax (:meth:`MixtureProcessRunner.optimize_xbest`) по ОБЩИМ
    суррогатам (GP+MoE): максимум desirability целей ветки над составной областью.
    Строка несёт: имя ветки, состав-доли + процесс в РЕАЛЬНЫХ единицах (замечание
    2), предсказанные свойства целей ``{свойство} (прогноз)``, итог ``d_overall`` и
    по-целевые ``d[{свойство}]``; при ``batch_kg`` — расход сырья на пробу. Чистый
    хелпер (без Streamlit) — тестируется напрямую."""
    res = runner.optimize_xbest(branch_id, n_candidates=int(n_candidates),
                                refine_iters=int(refine_iters),
                                n_starts=int(n_starts))
    coord_names = setup_coord_names(runner)
    mix_names = list(runner.current_schema.mixture_names)
    x = np.atleast_2d(np.asarray(res.x, float))
    xr = process_code_to_real(runner, x)[0]  # процесс → реальные единицы
    row: Dict[str, Any] = {"ветка": runner.branches[branch_id].name}
    for j, cn in enumerate(coord_names[:x.shape[1]]):
        row[cn] = round(float(xr[j]), 4)
    if batch_kg is not None and float(batch_kg) > 0:
        for j, cn in enumerate(mix_names):
            row[f"{cn} ({MASS_UNIT})"] = round(float(x[0, j]) * float(batch_kg), 4)
    for pn, val in (res.properties or {}).items():
        row[f"{pn} (прогноз)"] = round(float(val), 4)
    row["d_overall"] = round(float(res.d_overall), 4)
    for pn, dv in (res.d_individual or {}).items():
        row[f"d[{pn}]"] = round(float(dv), 4)
    return pd.DataFrame([row])


def branch_recipe_excel_bytes(runner, branch_id, *,
                              batch_kg: Optional[float] = None,
                              n_candidates: int = 2000, refine_iters: int = 200,
                              n_starts: int = 5) -> bytes:
    """§17.6.1 (C3): рекомендованный рецепт ветки → xlsx-байты (кнопка скачивания).

    Лист «Рецепт» = :func:`branch_recipe_dataframe`. Чистый хелпер (без Streamlit)
    — тестируется напрямую; отдаёт готовые байты .xlsx (openpyxl)."""
    import io
    df = branch_recipe_dataframe(runner, branch_id, batch_kg=batch_kg,
                                 n_candidates=n_candidates,
                                 refine_iters=refine_iters, n_starts=n_starts)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Рецепт", index=False)
    buf.seek(0)
    return buf.getvalue()



# ----------------------------------------------------------------------
# Чистые таблицы для показа (без Streamlit) — тестируемы напрямую
# ----------------------------------------------------------------------



_MONEY_RU = {cv.MONEY_ZEROED: "занулён (ZEROED)", cv.MONEY_ALIVE: "живой (ALIVE)",
             None: "—"}
_CHANGE_RU = {"inherited": "унаследовано как есть",
              "overridden_same_role": "тронуто, роль та же",
              "changed_by_objective": "изменено объективом ветки"}

# Легенда §4-стопа: причина остановки раунда (двойной критерий §4/§6).
_STOP_RU: Dict[Optional[str], str] = {
    None: "▶ продолжать (есть куда и выгодно)",
    "ceil_reached": "🎯 потолок достигнут (ceil_reached)",
    "stagnation": "🛑 прогресс встал (stagnation)",
    "not_economical": "💸 невыгодно (not_economical)",
}

# Подсказка к полю «Значимость цели» (ядровый вес взвеш. геом-среднего d_i).
# Только текст UI — сам параметр в ядре остаётся ``DesirabilitySpec.weight``.
_WEIGHT_HELP = (
    "Значимость цели — её относительный приоритет в компромиссе между откликами. "
    "Итог ветки = взвешенное геометрическое среднее желательностей d_i: "
    "d_overall = (Π d_i^w)^(1/Σw). Больше значимость → оптимум сильнее тянется "
    "выполнить именно эту цель, жертвуя другими. Важны только ОТНОШЕНИЯ "
    "значимостей (равные = равная важность; умножить все на число — без эффекта). "
    "Не спасает от veto: если хоть одна d_i=0, итог=0 при любой значимости."
)



def role_table_dataframe(report: Dict[str, Any]) -> pd.DataFrame:
    """Role-репорт ветки → таблица для показа (контекст ветки уже зашит в report)."""
    rows = []
    for r in report["responses"]:
        rows.append({
            "отклик": r["response"],
            "роль": r["role_label"],
            "код": r["role_code"],
            "в цели": "да" if r["in_goal"] else "—",
            # Колонка строго строковая: у целей — число, у нецелевых — «—».
            # Смешение float/str ломает Arrow-сериализацию st.dataframe (варнинг
            # «Serialization … unsuccessful» и авто-приведение типа колонки «вес»).
            "вес": (f'{float(r["weight"]):g}' if r["weight"] is not None else "—"),
            "ден. канал ρ": _MONEY_RU.get(r["money_channel"], "—"),

            "покрытие": f'{r["coverage_measured"]}/{r["coverage_total"]}',
            "низк. покрытие": "⚠️" if r["low_coverage"] else "",
        })
    return pd.DataFrame(rows)


def spawn_review_dataframe(review: Dict[str, Any]) -> pd.DataFrame:
    """Review-сводка наследования ролей при spawn → таблица (Тр-8.1а/б/в)."""
    rows = []
    for r in review["responses"]:
        rows.append({
            "отклик": r["response"],
            "роль родителя": r["role_parent"],
            "роль ребёнка": r["role_child"],
            "изменение": _CHANGE_RU.get(r["change"], r["change"]),
            "ден. канал ρ (ребёнок)": _MONEY_RU.get(r["money_channel_child"], "—"),
        })
    return pd.DataFrame(rows)


def goal_editor_dataframe(runner, branch_id: str) -> pd.DataFrame:
    """Текущие цели ветки → таблица (§16.3): отклик, вид, диапазон, target, вес.

    Ветка — это НАБОР целей (мультицель): каждая цель несёт свой вид
    (``min``/``max``/``target``), диапазон ``[low, high]`` (и ``target`` для
    target-типа) и вес геом-среднего. Читает ``branch.goal`` (read-only)."""
    br = runner.branches[branch_id]
    rows = []
    for resp, spec in (br.goal or {}).items():
        rows.append({
            "цель (отклик)": resp,
            "вид": spec.kind,
            "low": round(float(spec.low), 4),
            "high": round(float(spec.high), 4),
            "target": (round(float(spec.target), 4)
                       if spec.target is not None else "—"),
            "вес": round(float(spec.weight), 4),
        })
    return pd.DataFrame(rows)


def workbench_points_dataframe(runner, result: Dict[str, Any]) -> pd.DataFrame:
    """Долитые за раунд точки → таблица измеренных откликов (§16.4).

    ``result`` — выхлоп ``run_branch_round`` (через ``CampaignController.run_round``):
    берём ``y_new`` (n×P, порядок ``property_names``) и помечаем origin-тегом ветки
    (И-1: точки уже в общей базе, здесь только показ)."""
    y = np.atleast_2d(np.asarray(result.get("y_new"), float))
    if y.size == 0:
        return pd.DataFrame()
    cols = list(runner.property_names)
    df = pd.DataFrame(y[:, [runner.prop_index[c] for c in cols]], columns=cols)
    df.insert(0, "origin", f"branch:{result.get('branch')}")
    # blocking добора: только что залитые точки — последние в базе, их партия =
    # последний (максимальный) блок общей базы
    try:
        bl = list(runner.point_blocks())
        if len(bl) >= len(df):
            df.insert(1, "Блок", [int(b) for b in bl[-len(df):]])
            if getattr(runner, "block_names", {}) or {}:
                df.insert(2, "Партия", [block_display(runner, int(b))
                                        for b in bl[-len(df):]])
    except Exception:  # noqa: BLE001 — блок не критичен для показа
        pass
    return df


def experiment_index(base_count: int, n: int) -> pd.Index:
    """Сквозные 1-based номера опытов проекта для ``n`` новых точек.

    «Номер опыта» = позиция точки в ЕДИНОЙ общей базе (``runner.points``),
    куда опыты добавляются по порядку (seed, затем commit'ы веток). Для ещё
    не залитых предложенных точек это их БУДУЩИЕ номера: если в базе уже
    ``base_count`` опытов, новые получат ``base_count+1 … base_count+n``.
    Возвращает именованный (``«№ опыта»``) индекс для показа в таблицах.
    """
    start = int(base_count) + 1
    return pd.Index(range(start, start + int(n)), name="№ опыта")


def origin_label(runner, origin: str) -> str:
    """Человекочитаемый ярлык origin-тега для ПОКАЗА (сам тег не меняется).

    Origin-тег точки в общей базе каноничен и завязан на id ветки
    (``"seed"`` / ``"M2"`` / ``"branch:{id}"``, И-1): id стабилен, а имя ветки
    можно переименовать, поэтому в ДАННЫХ хранится id. Для таблиц UI подменяем
    ``"branch:{id}"`` на ``"{имя} ({id})"``; прочие теги — как есть.
    """
    if isinstance(origin, str) and origin.startswith("branch:"):
        bid = origin.split(":", 1)[1]
        br = runner.branches.get(bid)
        if br is not None:
            return f"{br.name} ({bid})"
    return str(origin)



# ----------------------------------------------------------------------
# Streamlit-рендер вкладки (тест — headless AppTest)
# ----------------------------------------------------------------------

def get_campaign_controller() -> Optional["cv.CampaignController"]:
    """Контроллер демо-кампании из session_state (или ``None``, если не создан)."""
    return st.session_state.get("campaign_ctrl")


def _flash(msg: str, kind: str = "success") -> None:
    """P0: отложенное уведомление — переживает ``st.rerun`` мутации.

    Мутации состояния завершаются ``st.rerun()`` (иначе таблицы ВЫШЕ кнопки
    показывали устаревшее состояние до следующего клика), а rerun стирает вывод
    текущего прогона. Сообщение складывается в session_state и показывается в
    начале следующего прогона (:func:`_show_flashes`)."""
    st.session_state.setdefault("camp_flash", []).append((str(kind), str(msg)))


def _show_flashes() -> None:
    """Показать и очистить отложенные уведомления (:func:`_flash`)."""
    for kind, msg in st.session_state.pop("camp_flash", []):
        getattr(st, kind, st.info)(msg)


def _invalidate_branch_caches() -> None:
    """P0: сбросить кеши дорогих расчётов после мутаций состояния.

    Кеши результатов (₽-объяснение канала ρ, рецепт x*) живут под ключами
    ``cache_*`` в session_state; после смены целей/весов/ролей/точек они
    устаревают и должны пересчитываться по кнопке заново."""
    for k in [k for k in st.session_state if str(k).startswith("cache_")]:
        st.session_state.pop(k, None)


def campaign_assistant_overview(
        ctrl: Optional["cv.CampaignController"] = None) -> Optional[Dict[str, Any]]:
    """Сводка кампании для ассистента/MCP (§16.1) или ``None``, если кампании нет.

    Берёт контроллер из ``session_state`` (либо переданный) и отдаёт
    ``campaign_overview`` с объяснением денежного канала ρ (``with_money=True``),
    но БЕЗ дорогого econ-MC (``compute_value=False``) — нужны роли, занулённый/
    живой канал и причина «почему за ρ есть/нет денег» (``reason_code``/``text``),
    а не точная ₽-оценка. Read-only (A0.6); ошибки гасятся в ``None``, чтобы мост
    к ассистенту/Cline никогда не ронял основной UI.
    """
    ctrl = ctrl if ctrl is not None else get_campaign_controller()
    if ctrl is None:
        return None
    try:
        return ctrl.overview(with_money=True, compute_value=False)
    except Exception:  # noqa: BLE001 — мост не должен ломать UI
        return None


def _rho_of(runner, branch_id: str) -> Optional[str]:
    pcfg = cv.branch_price_config(runner, branch_id)
    return pcfg["rho_property"] if pcfg else None


def _parse_names(text: str) -> List[str]:
    """Разобрать имена через запятую/точку-с-запятой в список (без пустых)."""
    return [t.strip() for t in str(text).replace(";", ",").split(",") if t.strip()]


def _parse_floats(text: str) -> Optional[List[float]]:
    """Разобрать числа через запятую/точку-с-запятой; ``None`` при нечисловом вводе."""
    try:
        return [float(v) for v in str(text).replace(";", ",").split(",")
                if str(v).strip()]
    except ValueError:
        return None


def render_composition_bounds(names: Sequence[str], *, key_prefix: str = "setup"):
    """§17.4 (замечание 1): ограничения состава — «Доли (0…1)» ИЛИ «Массовые части
    (база = 100)». Возвращает ``(lower, upper)`` в ДОЛЯХ или ``(None, None)``.

    Форма-близнец сайдбара pipeline (`streamlit_app._composition_bounds`):
    экспандер с радио «Способ ввода» и попарными полями ``L·X`` / ``U·X`` по
    каждому компоненту (а не «через запятую»). В режиме «Массовые части» базовый
    компонент = 100 частей, остальные задаются диапазоном частей → доли считает
    каноничная :func:`core.simplex.parts_ranges_to_fraction_bounds` (tightest box,
    та же математика, что в pipeline). Пустой список компонентов — ``(None, None)``.
    """
    q = len(names)
    if q == 0:
        return None, None
    # NB: без st.expander — форма зовётся ВНУТРИ экспандера сетапа, а Streamlit
    # запрещает вложенные экспандеры; используем контейнер с заголовком.
    st.markdown("**📐 Ограничения состава (опц.)**")
    with st.container():
        mode = st.radio(
            "Способ ввода", ["Доли (0…1)", "Массовые части (база = 100)"],
            key=f"{key_prefix}_comp_mode",
            help="«Массовые части»: базовый компонент = 100 частей, остальные "
                 "задаются диапазоном частей, а доли (и плавающий диапазон доли "
                 "базы) считаются автоматически. «Доли»: границы доли каждого "
                 "компонента 0…1.")
        if mode.startswith("Доли"):
            st.caption("Доли каждого компонента (0…1). Сумма нижних ≤ 1 ≤ сумма "
                       "верхних. Оставьте 0…1, если ограничений нет.")
            lower: List[float] = []
            upper: List[float] = []
            for i in range(q):
                cc = st.columns(2)
                lo_i = cc[0].number_input(
                    f"L · {names[i]}", min_value=0.0, max_value=1.0, value=0.0,
                    step=0.01, format="%.4f", key=f"{key_prefix}_lo_{q}_{i}")
                hi_i = cc[1].number_input(
                    f"U · {names[i]}", min_value=0.0, max_value=1.0, value=1.0,
                    step=0.01, format="%.4f", key=f"{key_prefix}_hi_{q}_{i}")
                lower.append(float(lo_i))
                upper.append(float(hi_i))
            nontrivial = any(l > 0 for l in lower) or any(u < 1 for u in upper)
            return (lower, upper) if nontrivial else (None, None)

        # --- режим массовых частей (база = 100) ---
        base_i = st.selectbox(
            "Базовый компонент (= 100 частей)", list(range(q)),
            format_func=lambda i: names[i], key=f"{key_prefix}_base_{q}")
        st.caption("Диапазон массовых частей для остальных компонентов "
                   "(база фиксирована = 100 частей):")
        pmin = [0.0] * q
        pmax = [0.0] * q
        for i in range(q):
            if i == base_i:
                pmin[i] = pmax[i] = 100.0
                st.markdown(f"**{names[i]}** — база: 100 частей (фиксировано)")
                continue
            cc = st.columns(2)
            pmin[i] = cc[0].number_input(
                f"min частей · {names[i]}", min_value=0.0, value=0.0, step=1.0,
                key=f"{key_prefix}_pmin_{q}_{i}")
            pmax[i] = cc[1].number_input(
                f"max частей · {names[i]}", min_value=0.0, value=100.0, step=1.0,
                key=f"{key_prefix}_pmax_{q}_{i}")
        try:
            lo_arr, hi_arr = parts_ranges_to_fraction_bounds(pmin, pmax)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось пересчитать части в доли: {exc}")
            return None, None
        tbl = pd.DataFrame(
            {"min частей": pmin, "max частей": pmax,
             "доля L": np.round(lo_arr, 4), "доля U": np.round(hi_arr, 4)},
            index=list(names)[:q])
        st.caption("Рассчитанные диапазоны долей для алгоритма:")
        st.dataframe(tbl, use_container_width=True)
        return lo_arr.tolist(), hi_arr.tolist()


def render_process_bounds(names: Sequence[str], *, key_prefix: str = "setup"):
    """§17.4 (замечание 2): границы процесс-параметров — попарные поля L/U на КАЖДЫЙ
    параметр в РЕАЛЬНЫХ единицах (интерфейс-близнец ограничений состава).

    Заменяет ввод «через запятую»: для каждого параметра — своя строка с нижней и
    верхней РЕАЛЬНОЙ границей (T=150…200 °C, P=1…5 бар); понятнее и меньше ошибок.
    Нормировку в код [0,1] делает движок сам. Возвращает ``(lower, upper)``
    списками реальных величин или ``(None, None)``, если параметров нет. Первые
    две оси получают осмысленные дефолты (T=150…200, P=1…5), остальные — 0…1.
    """
    d = len(names)
    if d == 0:
        return None, None
    st.markdown("**⚙️ Границы процесс-параметров (реальные единицы)**")
    st.caption("Для каждого параметра — нижняя и верхняя РЕАЛЬНАЯ граница "
               "(например, T = 150…200 °C, P = 1…5 бар). Нормировку в код [0,1] "
               "программа делает сама (замечание 2).")
    _defaults = {0: (150.0, 200.0), 1: (1.0, 5.0)}
    lower: List[float] = []
    upper: List[float] = []
    for i in range(d):
        dlo, dhi = _defaults.get(i, (0.0, 1.0))
        cc = st.columns([1, 2, 2])
        cc[0].markdown(f"**{names[i]}**")
        lo_i = cc[1].number_input(
            f"нижняя · {names[i]}", value=float(dlo), step=1.0, format="%.4f",
            key=f"{key_prefix}_plo_{d}_{i}")
        hi_i = cc[2].number_input(
            f"верхняя · {names[i]}", value=float(dhi), step=1.0, format="%.4f",
            key=f"{key_prefix}_phi_{d}_{i}")
        lower.append(float(lo_i))
        upper.append(float(hi_i))
    return lower, upper


def setup_prefill_from_runner(runner) -> Dict[str, Any]:
    """C2 (§17.6.1): значения формы сетапа из ЗАГРУЖЕННОГО проекта.

    Обратная проекция «раннер → ключи виджетов формы "🆕 Новый проект"»: имена
    компонентов/процесс-параметров/откликов, границы долей (режим «Доли (0…1)» —
    массовые части не сериализуются, честно показываем рассчитанные доли),
    реальные границы процесс-осей и seed раннера. Без неё форма после загрузки
    проекта показывала дефолты («A, B, C», 0…1, 150…200), и пользователь видел
    «настройки не подтянулись» (A0.6). Чистая (без Streamlit) — тестируется
    напрямую; применяется отложенно через ключ ``setup_prefill_pending``
    (Streamlit запрещает менять ключ виджета, созданного в том же прогоне).
    """
    sch = runner.current_schema
    mix = list(sch.mixture_names)
    proc = list(sch.process_names)
    q, d = len(mix), len(proc)
    out: Dict[str, Any] = {
        "setup_mix": ", ".join(mix),
        "setup_resp": ", ".join(str(p) for p in runner.property_names),
        "setup_proc": ", ".join(proc),
        "setup_seed": int(getattr(runner, "seed", 0)),
        "setup_comp_mode": "Доли (0…1)",
    }
    mb = sch.mixture_block()
    if mb is not None:
        for i in range(q):
            out[f"setup_lo_{q}_{i}"] = float(mb.lower[i])
            out[f"setup_hi_{q}_{i}"] = float(mb.upper[i])
    pb = sch.process_block()
    if pb is not None:
        for i in range(d):
            out[f"setup_plo_{d}_{i}"] = float(pb.lower[i])
            out[f"setup_phi_{d}_{i}"] = float(pb.upper[i])
    # blocking: число партий + фактор/имена блоков (виджеты seed-секции)
    out["setup_seed_blocks"] = int(getattr(runner, "n_blocks_start", 1))
    out["setup_block_factor"] = str(getattr(runner, "block_factor", "") or "")
    for b, nm in (getattr(runner, "block_names", {}) or {}).items():
        out[f"setup_block_name_{int(b)}"] = str(nm)
    return out


def project_settings_dataframe(runner) -> pd.DataFrame:
    """Сводка настроек ТЕКУЩЕГО проекта: переменная / тип / границы L…U.

    Показывается после сборки/загрузки проекта, чтобы пользователь видел, какие
    доли компонентов и реальные границы процесс-параметров действуют в движке
    (единый источник истины — ``runner.current_schema``, а не поля формы).
    Чистая (без Streamlit) — тестируется напрямую."""
    rows: List[Dict[str, Any]] = []
    sch = runner.current_schema
    mb = sch.mixture_block()
    if mb is not None:
        for nm, lo, hi in zip(mb.names, mb.lower, mb.upper):
            rows.append({"переменная": str(nm), "тип": "компонент смеси (доля)",
                         "нижняя": float(lo), "верхняя": float(hi)})
    pb = sch.process_block()
    if pb is not None:
        for nm, lo, hi in zip(pb.names, pb.lower, pb.upper):
            rows.append({"переменная": str(nm),
                         "тип": "процесс-параметр (реальные единицы)",
                         "нижняя": float(lo), "верхняя": float(hi)})
    return pd.DataFrame(rows)


def render_project_settings(runner) -> None:
    """📋 Панель «настройки проекта» (читает движок, не форму).

    C2: после загрузки проекта именно здесь видно, что доли компонентов,
    границы процесс-параметров и отклики ПОДТЯНУЛИСЬ (двойник формы сетапа,
    но по фактическому состоянию раннера)."""
    with st.expander("📋 Настройки проекта"):
        st.caption(
            f"Отклики: {', '.join(runner.property_names)} · схема v"
            f"{int(runner.current_schema_version)} · seed раннера = "
            f"{int(getattr(runner, 'seed', 0))} · общая база: "
            f"{len(runner.points)} точек. Границы ниже — те, по которым "
            "реально работает движок (сохраняются и загружаются с проектом).")
        st.dataframe(project_settings_dataframe(runner),
                     use_container_width=True, hide_index=True)


def render_setup_form() -> None:

    """§17.4 (Ш3b): форма РЕАЛЬНОГО сетапа — mixture + процесс + отклики.


    По кнопке строит :class:`MixtureProcessRunner` (:class:`ManualOracle`, пустая
    база) и кладёт :class:`CampaignController` в ``session_state`` под тем же
    ключом, что и демо-кампания (главный поток §17 — один движок). Реального
    оракула нет: стартовые отклики вносит пользователь в ручном seed-цикле ниже.

    UX: форма живёт в ЛЕВОЙ панели (сайдбар) рядом с «📁 Проект» — основная
    область не загромождается; кнопка «🏗 Построить проект» — ВНУТРИ формы
    (это её submit: собирает раннер из введённых полей), поэтому не висит
    отдельной сиротой на странице.
    """
    # C2: отложенный префилл формы из ЗАГРУЖЕННОГО проекта — применяем ДО
    # инстанцирования виджетов формы (ключ кладёт загрузчик в сайдбаре;
    # менять session_state виджета можно только до его создания в прогоне).
    _pending = st.session_state.pop("setup_prefill_pending", None)
    if _pending:
        for _k, _v in dict(_pending).items():
            st.session_state[_k] = _v
    with st.sidebar.expander("🆕 Новый проект — реальный сетап (§17.4)",
                             expanded=get_campaign_controller() is None):
        st.caption(
            "Составная область СРАЗУ: симплекс компонентов смеси (Σ=1) × куб "
            "процесс-параметров (реальные единицы). Отклики (свойства) меряются "
            "вручную — оракула-симулятора нет (кнопка «Заполнить тестовыми» в "
            "seed-цикле оставлена для прогонов без лаборатории, A0.6).")
        mix_txt = st.text_input("Компоненты смеси (через запятую)",
                                value="A, B, C", key="setup_mix")
        resp_txt = st.text_input("Отклики / свойства (через запятую)",
                                 value="strength, gloss, rho", key="setup_resp")

        # Замечание 1: ограничения состава — форма «Доли / Массовые части» с
        # попарными полями L·X / U·X по каждому компоненту (форма-близнец сайдбара
        # pipeline). Части → доли каноничной parts_ranges_to_fraction_bounds.
        mix_live = _parse_names(mix_txt)
        mlo, mhi = render_composition_bounds(mix_live, key_prefix="setup")

        proc_txt = st.text_input("Процесс-параметры (через запятую)",
                                 value="T, P", key="setup_proc")
        # Замечание 2: границы процесса — попарные поля L/U на каждый параметр в
        # РЕАЛЬНЫХ единицах (форма-близнец ограничений состава), а не «через
        # запятую» — понятнее и меньше ошибок. Нормировку в код [0,1] движок
        # делает сам.
        proc_live = _parse_names(proc_txt)
        plo, phi = render_process_bounds(proc_live, key_prefix="setup")
        seed_v = st.number_input(

            "Seed раннера (зерно ГСЧ проекта)", value=1, step=1, key="setup_seed",
            help="Зерно генератора случайных чисел движка проекта: фиксирует "
                 "воспроизводимость стартового дизайна и внутренних рестартов "
                 "оптимизатора. Одно и то же значение → тот же результат; на "
                 "состав и границы НЕ влияет.")
        if st.button("🏗 Построить проект", key="setup_build"):
            try:
                mix = _parse_names(mix_txt)
                proc = _parse_names(proc_txt)
                resp = _parse_names(resp_txt)
                # plo/phi уже собраны формой render_process_bounds выше (реальные
                # единицы, попарно L/U). Пустой список процесс-параметров → форма
                # вернула (None, None); build_setup_runner отвергнет пустой proc.
                if plo is None or phi is None:
                    raise ValueError("Добавьте хотя бы один процесс-параметр "
                                     "и задайте его границы (§17.4).")
                # mlo/mhi уже посчитаны формой render_composition_bounds выше
                # (доли; None — полный симплекс).

                runner = build_setup_runner(
                    mixture_names=mix, process_names=proc,
                    process_lower=plo, process_upper=phi,
                    response_names=resp, mixture_lower=mlo, mixture_upper=mhi,
                    seed=int(seed_v))
                st.session_state["campaign_ctrl"] = cv.CampaignController(runner)
                for k in ("setup_seed_X", "setup_seed_Y",
                          "setup_seed_df", "setup_seed_df_sig"):
                    st.session_state.pop(k, None)
                st.success(
                    f"Проект собран: смесь {mix} × процесс {proc}, отклики {resp}. "
                    "База пуста — предложите и измерьте стартовый дизайн ниже.")
            except (ValueError, KeyError) as exc:
                st.error(str(exc))


def render_seed_entry(ctrl: "cv.CampaignController") -> None:
    """§17.4: ручной СТАРТОВЫЙ цикл «предложить seed → внести Y → зафиксировать».

    Пока стартовый дизайн не измерен (база пуста), это единственная активная
    секция вкладки: ``propose_seed`` (read-only) → таблица ввода Y (по всем P) →
    ``commit_seed`` (доливает в общую базу origin=seed, обучает суррогаты).
    «Заполнить тестовыми» берёт Y из демо-оракула (``_measure``) — ЯВНОЕ действие
    (A0.6). Составные координаты заблокированы; правятся только столбцы «(lab)».
    """
    runner = ctrl.runner
    props = list(runner.property_names)
    coord_names = setup_coord_names(runner)
    st.markdown("### 🌱 Стартовый дизайн (seed) — ручной ввод откликов (§17.4)")
    st.caption(
        f"Отклики проекта: {', '.join(props)}. Предложите N точек по составной "
        "области, внесите измеренные Y по каждому свойству и зафиксируйте — точки "
        "лягут в ОБЩУЮ базу (origin=seed), суррогаты обучатся (И-1).")
    # Замечание 4: рекомендуемый N скрининга считаем от q компонентов и d
    # процесс-параметров — предлагаем сразу как значение по умолчанию.
    q = len(runner.current_schema.mixture_names)
    d = len(runner.current_schema.process_names)
    rec_n = recommended_seed_size(q, d)
    sc = st.columns([1, 1, 1, 1])
    seed_n = sc[0].number_input(
        "N seed-точек", min_value=2, max_value=200, value=int(rec_n), step=1,
        key="setup_seed_n",
        help=f"Рекомендация для скрининга: N = q·(1+d) + ⌈q·(1+d)/2⌉ = {rec_n} "
             f"(q={q} компонентов, d={d} процесс-параметров) — число членов "
             "кросс-модели «смесь-линейно × процесс-линейно» плюс ~50% запаса на "
             "остаточную дисперсию. Значение можно изменить вручную.")

    seed_design = sc[1].number_input(
        "зерно ГСЧ (воспроизводимость)", value=1, step=1,
        key="setup_seed_design",
        help="Зерно генератора случайных чисел для построения стартового "
             "дизайна. Одно и то же значение → тот же набор из N точек "
             "(воспроизводимо); другое значение → другой случайный вариант. "
             "К числу «seed-точек» отношения не имеет — это разные «seed».")

    # P0-багфикс «+/− срабатывает со второго клика»: НЕ передаём value= вместе
    # с key — меняющийся default конфликтует со state виджета, и Streamlit
    # проглатывает первое нажатие. Ключ инициализируется ОДИН раз из раннера
    # (до создания виджета), дальше состоянием владеет сам ключ.
    if "setup_seed_blocks" not in st.session_state:
        st.session_state["setup_seed_blocks"] = int(
            getattr(runner, "n_blocks_start", 1))
    nb_blocks = sc[2].number_input(
        "Партий (блоков)", min_value=1, max_value=20, step=1,
        key="setup_seed_blocks",
        help="Blocking стартового дизайна: если опыты нельзя поставить одной "
             "партией / за один день, план ОПТИМАЛЬНО разбивается на блоки "
             "(interchange по блочному D-критерию) — эффект партии ловится "
             "блочной моделью и не искажает оценки состава. 1 — без "
             "блокировки. Доборы веток автоматически получают НОВЫЙ блок "
             "(каждая партия добора — отдельный блок).")
    runner.n_blocks_start = max(1, int(nb_blocks))

    if sc[3].button("📐 Предложить seed-дизайн", key="setup_propose_seed"):
        X = np.asarray(ctrl.propose_seed(int(seed_n), seed=int(seed_design)), float)
        st.session_state["setup_seed_X"] = X
        st.session_state.pop("setup_seed_Y", None)
        # Новый дизайн — сброс состояния редактора Y (иначе data_editor наложит
        # СТАРЫЕ правки ячеек на новые строки); кнопка выше редактора, поэтому
        # виджет ещё не создан в этом прогоне и ключ можно чистить.
        st.session_state.pop("setup_seed_editor", None)
        st.session_state.pop("setup_seed_df", None)
        st.session_state.pop("setup_seed_df_sig", None)

    # Названия блоков (опц.): фактор блокировки + имя каждой партии — чтобы был
    # понятен ПРИНЦИП блокировки (оператор / партия сырья / рабочая смена).
    # Метаданные показа/Excel; на оптимальное разбиение не влияют; персистятся
    # в campaign.json (block_factor / block_names).
    nb_now = int(getattr(runner, "n_blocks_start", 1))
    if nb_now > 1:
        with st.expander("🧱 Названия партий (блоков) — что их различает (опц.)"):
            st.caption(
                "Блок — группа опытов, поставленных в одинаковых условиях "
                "(одна партия сырья / один оператор / одна смена). Назовите "
                "фактор и сами партии — имена попадут в таблицы и Excel "
                "(столбец «Партия»); на расчёт блокировки они не влияют.")
            bf = st.text_input(
                "Что различает партии (фактор блокировки)",
                value=str(getattr(runner, "block_factor", "") or ""),
                key="setup_block_factor",
                placeholder="например: оператор / партия сырья / рабочая смена")
            runner.block_factor = bf.strip()
            names_now = dict(getattr(runner, "block_names", {}) or {})
            ncols = st.columns(min(nb_now, 4))
            new_names: Dict[int, str] = {}
            for b in range(1, nb_now + 1):
                nm = ncols[(b - 1) % len(ncols)].text_input(
                    f"Имя блока {b}", value=str(names_now.get(b, "")),
                    key=f"setup_block_name_{b}",
                    placeholder=f"например: смена {b}")
                if nm.strip():
                    new_names[b] = nm.strip()
            runner.block_names = new_names

    Xs = st.session_state.get("setup_seed_X")
    if Xs is None:
        return
    Xs = np.atleast_2d(np.asarray(Xs, float))

    if st.button("🧪 Заполнить тестовыми (демо-оракул)", key="setup_fill_demo"):
        st.session_state["setup_seed_Y"] = np.vstack(
            [runner._measure(np.asarray(x, float)) for x in Xs])
        st.session_state.pop("setup_seed_editor", None)
        st.session_state.pop("setup_seed_df", None)
        st.session_state.pop("setup_seed_df_sig", None)

    # Замечание 7: размер ПРОБЫ (партии) — добавляет столбцы расхода сырья
    # {компонент} (кг) = доля·batch, чтобы понимать, сколько взвесить на опыт.
    mix_names = list(runner.current_schema.mixture_names)
    batch = st.number_input(
        f"Размер пробы, {MASS_UNIT}/опыт (для расхода сырья и Excel)",
        min_value=0.0, value=0.0, step=0.1, key="setup_seed_batch",
        help="0 — только состав в долях; >0 — добавит столбцы расхода сырья "
             f"({MASS_UNIT}) = доля компонента × размер пробы (замечание 7). "
             "На сам дизайн (координаты) не влияет — только показ и выгрузка.")
    batch_kg = float(batch) if batch > 0 else None
    mass_cols = ([f"{c} ({MASS_UNIT})" for c in mix_names]
                 if batch_kg is not None else [])

    Ys = st.session_state.get("setup_seed_Y")
    lab_cols = [f"{p} (lab)" for p in props]
    # Единый источник таблицы (показ = редактор = Excel): чистый хелпер строит
    # «№ опыта» + заблокированные координаты (+ расход сырья) + столбцы «(lab)».
    #
    # БАГФИКС (пропадающие значения): вход data_editor обязан быть СТАБИЛЬНЫМ
    # между прогонами. Раньше df пересобирался КАЖДЫЙ rerun с уже внесённым
    # черновиком Y — для Streamlit изменившиеся данные = НОВЫЙ виджет, и свежая
    # (ещё не запечённая в df) правка сбрасывалась: первая введённая ячейка в
    # каждой следующей колонке «исчезала», повторный ввод принимался. Теперь df
    # кэшируется по сигнатуре (дизайн Xs + размер пробы) и пересобирается только
    # при её смене / явном заполнении (демо-кнопка, новый дизайн, загрузка
    # проекта) — черновик Y вливается в кэш в этот момент, правки не теряются.
    sig = (Xs.tobytes(), Xs.shape, batch_kg,
           int(getattr(runner, "n_blocks_start", 1)),
           tuple(sorted((getattr(runner, "block_names", {}) or {}).items())))
    if (st.session_state.get("setup_seed_df_sig") != sig
            or "setup_seed_df" not in st.session_state):
        st.session_state["setup_seed_df"] = seed_design_dataframe(
            runner, Xs, Ys, batch_kg=batch_kg)
        st.session_state["setup_seed_df_sig"] = sig
        # Пересборка входа → чистим состояние виджета (мы выше редактора по
        # коду, он ещё не создан в этом прогоне — ключ чистить безопасно).
        st.session_state.pop("setup_seed_editor", None)
    df = st.session_state["setup_seed_df"]
    st.caption("Составные координаты заблокированы; заполняются только столбцы "
               "«свойство (lab)» (вручную или кнопкой «Заполнить тестовыми»):")
    blk_cols = [c for c in ("Блок", "Партия") if c in df.columns]
    edited = st.data_editor(df, use_container_width=True, height=320,
                            hide_index=True,
                            disabled=["№ опыта", *blk_cols,
                                      *coord_names[:Xs.shape[1]],
                                      *mass_cols],
                            key="setup_seed_editor")
    # Iteration 28+: blocking должен быть ВИДЕН всегда — и когда включён
    # (размеры партий + цена блокировки), и когда выключен (как включить).
    st.caption(seed_blocking_caption(runner, Xs))
    # C2: черновик Y из редактора живёт в session_state (setup_seed_Y), чтобы
    # частично внесённые отклики можно было сохранить в проект ДО фиксации
    # (commit_seed) и восстановить при загрузке. NaN допустимы (пустые ячейки).
    try:
        st.session_state["setup_seed_Y"] = np.column_stack(
            [np.asarray(edited[c], float) for c in lab_cols])
    except (KeyError, TypeError, ValueError):
        pass
    # Для Excel ниже берём СВЕЖИЙ черновик (включая правки текущего прогона).
    Ys = st.session_state.get("setup_seed_Y", Ys)

    # C3: сохранить ПЛАН стартового эксперимента в Excel (ещё до фиксации) —
    # экспортируем внесённые в редакторе значения (пустые «(lab)» — под ручной
    # ввод в лаборатории), с расходом сырья, если задан размер пробы.
    st.download_button(
        "⬇️ Сохранить план в Excel (.xlsx)",
        data=seed_design_excel_bytes(runner, Xs, Ys, batch_kg=batch_kg),
        file_name="seed_design.xlsx", key="setup_seed_dl",
        mime="application/vnd.openxmlformats-officedocument."
             "spreadsheetml.sheet")

    if st.button("💾 Зафиксировать seed (commit_seed)", key="setup_commit_seed"):

        try:
            Y = np.column_stack([np.asarray(edited[c], float) for c in lab_cols])
            if np.isnan(Y).any():
                raise ValueError(
                    "Заполните измеренные отклики (столбцы «… (lab)») для ВСЕХ "
                    "точек — вручную в таблице или кнопкой «🧪 Заполнить "
                    "тестовыми». Пустые ячейки (None) фиксировать нельзя.")
            out = ctrl.commit_seed(Xs, Y)

            for k in ("setup_seed_X", "setup_seed_Y",
                      "setup_seed_df", "setup_seed_df_sig"):
                st.session_state.pop(k, None)
            # P0: уведомление через _flash — st.success перед st.rerun не
            # доживал до глаз пользователя (rerun стирает вывод прогона).
            _flash(
                f"Seed зафиксирован: +{out['added']} точек (origin=seed), общая "
                f"база = {out['n_base']}, суррогаты обучены. Дальше — создание "
                "веток (Ш4, §17.5).")
            # База стала непустой → сразу перерисовать вкладку, чтобы открылось
            # создание веток (§17.5) без второго клика (иначе ранний return в
            # render_campaign держит seed-секцию до следующего взаимодействия).
            st.rerun()
        except (ValueError, KeyError) as exc:
            st.error(str(exc))



# ----------------------------------------------------------------------
# Чистые хелперы черновика целей ветки (без Streamlit — тестируются напрямую)
# ----------------------------------------------------------------------
def draft_add_goal(draft: Sequence[Dict[str, Any]], *, resp: str, kind: str,
                   low: float, high: float, weight: float,
                   target: Optional[float] = None) -> List[Dict[str, Any]]:
    """Добавить цель в черновик ветки (§17.5). Возвращает НОВЫЙ список.

    Цель по одному и тому же отклику НЕ дублируется: повторное добавление того же
    ``resp`` ЗАМЕНЯЕТ прежнюю запись (иначе при создании ветки дубли молча
    схлопнулись бы в ``goals[resp]`` — тихая потеря, A0.6). ``target`` хранится
    только для вида ``target``.
    """
    entry = {"resp": resp, "kind": kind, "low": float(low), "high": float(high),
             "weight": float(weight),
             "target": (float(target) if kind == "target" and target is not None
                        else None)}
    out = [dict(g) for g in draft]
    for i, g in enumerate(out):
        if g["resp"] == resp:
            out[i] = entry
            return out
    out.append(entry)
    return out


def draft_remove_goal(draft: Sequence[Dict[str, Any]],
                      index: int) -> List[Dict[str, Any]]:
    """Убрать цель по индексу из черновика ветки. Возвращает НОВЫЙ список.

    Индекс вне диапазона — список возвращается без изменений (идемпотентно).
    """
    out = [dict(g) for g in draft]
    if 0 <= index < len(out):
        del out[index]
    return out


def render_branch_creation(ctrl: "cv.CampaignController") -> None:
    """§17.5 (Ш4): ВРУЧНУЮ создать ветку — мультицель + роли + ценовая нога.


    Замена авто-M7 (§17.0): пользователь объявляет намерение ветки сам. Цели
    набираются по одной в session_state (мультицель §16.3, каждая — вид/диапазон/
    вес); опционально включается ценовая нога (ρ-отклик + цены компонентов →
    ``make_linear_price_fn`` + desirability цены) и экономика (V/c_exp/H). Роли
    выводятся ядром из намерения (цель ⇒ OPTIMIZED, ρ без цели ⇒ PRICE_INPUT).
    Всё мутирующее — по кнопке (A0.6); логика/валидация — в
    :meth:`CampaignController.create_branch` (§17.3).
    """
    runner = ctrl.runner
    props = list(runner.property_names)
    mix_names = list(runner.current_schema.mixture_names)
    draft: List[Dict[str, Any]] = st.session_state.setdefault(
        "camp_new_goals", [])

    with st.expander("➕ Создать ветку вручную (§17.5 — цели/роли/ценовая нога)",
                     expanded=not list(runner.branches)):
        st.caption(
            "Ветка — контейнер НАМЕРЕНИЯ (канон §5/§12): несколько целей "
            "(мультицель §16.3) + опц. ценовая нога ρ. Модели у ветки нет — "
            "физика общая. Роли выводятся из намерения: цель ⇒ OPTIMIZED, ρ "
            "ценовой ноги без цели ⇒ PRICE_INPUT (И-5).")
        c = st.columns([2, 2, 1])
        name = c[0].text_input("Имя ветки",
                               value=f"branch{len(runner.branches) + 1}",
                               key="camp_nb_name")
        budget = c[1].number_input(
            "Бюджет ветки (число опытов)", min_value=1, max_value=200,
            value=20, step=1, key="camp_nb_budget",
            help="Максимум лабораторных опытов (точек добора), которые ветка "
                 "может потратить на достижение цели: 1 слот = 1 измеренная "
                 "точка. По исчерпании бюджета ветка останавливается.")
        satisfy = c[2].number_input(
            "Порог достаточности ceil (d_overall)", min_value=0.1,
            value=1.0, step=0.1, key="camp_nb_ceil",
            help="Уровень общей желательности d_overall (0…1+), по достижении "
                 "которого цель считается выполненной и раунды можно "
                 "останавливать (технический критерий §4).")


        # --- набор целей (мультицель) ---
        st.markdown("**🎯 Цели ветки (добавляйте по одной — мультицель §16.3)**")
        st.caption(
            "Ниже — конструктор ОДНОЙ цели. Настройте параметры и нажмите "
            "«➕ Добавить цель в ветку»: цель войдёт в ветку ТОЛЬКО после нажатия "
            "(до этого черновик пуст). Повторное добавление того же отклика "
            "ЗАМЕНЯЕТ прежнюю цель; удалить цель поштучно — кнопкой 🗑 ниже.")
        gc = st.columns([2, 2, 2, 2, 2])
        ng_resp = gc[0].selectbox("Цель (отклик)", props, key="camp_nb_resp")
        ng_kind = gc[1].selectbox("вид", ["max", "min", "target"],
                                  key="camp_nb_kind")
        ng_lo = gc[2].number_input("low", value=0.0, step=0.5, key="camp_nb_lo")
        ng_hi = gc[3].number_input("high", value=10.0, step=0.5, key="camp_nb_hi")
        ng_w = gc[4].number_input("Значимость цели", min_value=0.01, value=1.0,
                                  step=0.5, key="camp_nb_w", help=_WEIGHT_HELP)

        ng_tgt = st.number_input("target (для вида target; low<target<high)",
                                 value=5.0, step=0.5, key="camp_nb_tgt")
        ac = st.columns([2, 2])
        if ac[0].button("➕ Добавить цель в ветку", key="camp_nb_add_goal"):
            st.session_state["camp_new_goals"] = draft_add_goal(
                draft, resp=ng_resp, kind=ng_kind, low=float(ng_lo),
                high=float(ng_hi), weight=float(ng_w),
                target=(float(ng_tgt) if ng_kind == "target" else None))
            draft = st.session_state["camp_new_goals"]
        if ac[1].button("🧹 Очистить цели", key="camp_nb_clear_goals"):
            st.session_state["camp_new_goals"] = []
            draft = []
        if draft:
            st.caption("Цели ветки (черновик) — 🗑 удаляет цель поштучно:")
            for i, g in enumerate(draft):
                rc = st.columns([9, 1])
                tgt_txt = (f", target={g['target']}"
                           if g.get("target") is not None else "")
                rc[0].markdown(
                    f"{i + 1}. **{g['resp']}** — {g['kind']} "
                    f"[{g['low']}, {g['high']}]{tgt_txt}, "
                    f"значимость {g['weight']}")

                if rc[1].button("🗑", key=f"camp_nb_del_goal_{i}"):
                    st.session_state["camp_new_goals"] = draft_remove_goal(draft, i)
                    st.rerun()
        else:
            st.caption("Пока ни одной цели — ветке нужен минимум один объектив "
                       "(§17.3).")


        # --- модель себестоимости изделия (опц.) — замечания 6, 7 ---
        st.markdown(f"**💰 {COST_MODEL_LABEL} (опц., §3/§15.6)**")
        st.caption(
            f"Себестоимость изделия = цена состава ({CURRENCY_UNIT}/{MASS_UNIT}) × "
            "плотность ρ. Плотность ρ — отдельный отклик (роль «вход "
            "себестоимости»): чем точнее ρ, тем точнее себестоимость. Задайте цены "
            "компонентов и верхний приемлемый порог себестоимости.")
        use_price = st.checkbox(
            "Учитывать себестоимость изделия (ρ-отклик + цены компонентов)",
            key="camp_nb_use_price")
        rho_prop = None
        prices_txt = ""
        cost_hi = 300.0
        cur_unit = CURRENCY_UNIT
        if use_price:
            uc = st.columns([1, 2])
            cur_unit = uc[0].text_input(
                "Валюта", value=CURRENCY_UNIT, key="camp_nb_cur",
                help="Обозначение денежной единицы (₽, $, €…). Используется в "
                     "подписях цен, экономики и в Excel-выгрузке.")
            rho_prop = uc[1].selectbox(
                "Плотность ρ (отклик — вход себестоимости)", props,
                key="camp_nb_rho")
            pc = st.columns([3, 2])
            prices_txt = pc[0].text_input(
                f"Цены компонентов {mix_names}, {cur_unit}/{MASS_UNIT} "
                "(через запятую)",
                value=", ".join(["100"] * len(mix_names)), key="camp_nb_prices")
            cost_hi = pc[1].number_input(
                f"Верхний порог себестоимости, {cur_unit}/изд (выше → d=0)",
                min_value=0.0, value=300.0, step=10.0, key="camp_nb_cost_hi")
        st.markdown("**📈 Экономика ветки (опц., для двойного стопа §4/§6)**")
        st.caption(
            "Нужна, чтобы взвесить пользу уточнения против стоимости опытов "
            f"(денежный критерий остановки). Все суммы — в валюте «{cur_unit}».")
        ec = st.columns(3)
        vol = ec[0].number_input(
            "Объём выпуска V, изд/период", min_value=0.0, value=0.0,
            step=100.0, key="camp_nb_vol",
            help="Плановый выпуск изделий за один период. Масштабирует денежный "
                 "эффект снижения себестоимости (эффект = Δсебестоимости × V).")
        cexp = ec[1].number_input(
            f"Стоимость одного опыта, {cur_unit}/опыт", min_value=0.0, value=0.0,
            step=10.0, key="camp_nb_cexp",
            help="Во сколько обходится один лабораторный опыт (реагенты, время). "
                 "Сравнивается с денежной пользой уточнения: если опыт дороже "
                 "ожидаемой пользы — срабатывает экономический стоп.")
        hor = ec[2].number_input(
            "Горизонт планирования H, периодов", min_value=0.0, value=0.0,
            step=1.0, key="camp_nb_hor",
            help="Сколько периодов выпуска учитываем при оценке эффекта: денежная "
                 "польза накапливается за H периодов.")


        if st.button("🌿 Создать ветку", key="camp_nb_create"):
            try:
                if not draft:
                    raise ValueError("Добавьте хотя бы одну цель (§17.3).")
                goals = {}
                for g in draft:
                    goals[g["resp"]] = DesirabilitySpec(
                        g["kind"], low=g["low"], high=g["high"],
                        target=g["target"], weight=g["weight"])
                price_fn = cost_spec = None
                if use_price:
                    prices = _parse_floats(prices_txt)
                    if prices is None or len(prices) != len(mix_names):
                        raise ValueError(
                            f"Нужно {len(mix_names)} цен компонентов "
                            f"{mix_names} (через запятую).")
                    price_fn = make_linear_price_fn(prices)
                    cost_spec = DesirabilitySpec("min", low=0.0,
                                                 high=float(cost_hi), weight=0.5)
                out = ctrl.create_branch(
                    str(name), goals, budget=int(budget),
                    satisfy_at=float(satisfy), price_fn=price_fn,
                    cost_spec=cost_spec, rho_property=rho_prop,
                    volume=(float(vol) if vol > 0 else None),
                    cost_exp=(float(cexp) if cexp > 0 else None),
                    horizon=(float(hor) if hor > 0 else None))
                st.session_state["camp_new_goals"] = []
                _invalidate_branch_caches()
                _flash(
                    f"Ветка «{out['branch_name']}» (`{out['branch_id']}`) создана: "
                    f"{out['n_goals']} цел., ценовая нога = {out['has_price_leg']}"
                    + (f" (ρ={out['rho_property']}, канал занулён="
                       f"{out['price_channel_suppressed']})"
                       if out['has_price_leg'] else "")
                    + f"; d_best={out['d_best']:.3f}.")
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))


def render_workbench(ctrl: "cv.CampaignController", bsel: str) -> None:
    """§17.5 (Ш5): рабочий стол ветки на РУЧНОМ цикле §17.2 (предложить → Y → долить).

    Замена авто-оракула (`run_branch_round`) реальным лабораторным циклом:
    ``validate_ready`` (§17.3 гейт) → ``propose_points`` (read-only) → таблица
    ввода измеренных Y по всем P → ``commit_measured`` (долив в общую базу
    origin=branch:{id}, переобучение суррогатов, §4-стоп). «Заполнить тестовыми»
    (демо-оракул ``_measure``) остаётся ЯВНОЙ кнопкой (A0.6). Составные координаты
    заблокированы; правятся только столбцы «свойство (lab)». Ключи session_state
    привязаны к ветке — смена ветки не тащит чужих кандидатов.
    """
    runner = ctrl.runner
    props = list(runner.property_names)
    coord_names = setup_coord_names(runner)
    kx, ky = f"camp_wb_X_{bsel}", f"camp_wb_Y_{bsel}"
    with st.expander("🛠 Рабочий стол ветки (§17.2/§16.4 — ручной добор)",
                     expanded=True):
        st.caption(
            "Реальный лабораторный цикл (§17.2): предложить N точек (read-only) → "
            "внести измеренные Y по всем свойствам → долить в ОБЩУЮ базу "
            "(origin=branch:{id}, И-1) → переобучить суррогаты → x*/d_best → "
            "§4-стоп. Мерим только по кнопке (A0.6); долив запечатывает undo.")
        br_now = runner.branches[bsel]
        st.caption(f"Ветка «{br_now.name}»: бюджет {br_now.budget}, потрачено "
                   f"{br_now.spent}, осталось {br_now.remaining()}, "
                   f"d_best={br_now.d_best:.3f}, статус {br_now.status}.")

        # §17.6.1 (C3): рекомендованный РЕЦЕПТ ветки x* + скачивание в Excel.
        # M8-argmax по ОБЩИМ суррогатам (GP+MoE) дорогой → считаем ТОЛЬКО по
        # кнопке (A0.6), результат кешируем в session_state (df + готовые
        # xlsx-байты), чтобы rerun не пересчитывал оптимизацию.
        st.markdown("**📋 Рекомендованный рецепт ветки (x*) — по общим суррогатам "
                    "GP+MoE**")
        st.caption(
            "Итог ветки одной строкой: рецепт-состав + процесс в РЕАЛЬНЫХ единицах, "
            "предсказанные свойства целей, общий d_overall и по-целевые d[·]. "
            "Считается по кнопке (M8-argmax дорогой); при размере пробы > 0 "
            "добавляется расход сырья на пробу.")
        rec_batch = st.number_input(
            f"Размер пробы, {MASS_UNIT}/опыт (для расхода сырья и Excel)",
            min_value=0.0, value=0.0, step=0.1, key=f"camp_wb_recipe_batch_{bsel}")
        rkey = f"cache_recipe_{bsel}"
        if st.button("📋 Рассчитать рецепт ветки (x*)",
                     key=f"camp_wb_recipe_btn_{bsel}"):
            import io
            try:
                bk = float(rec_batch) if float(rec_batch) > 0 else None
                df_rec = branch_recipe_dataframe(runner, bsel, batch_kg=bk)
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                    df_rec.to_excel(xw, sheet_name="Рецепт", index=False)
                st.session_state[rkey] = (df_rec, buf.getvalue())
            except (ValueError, KeyError, RuntimeError) as exc:
                st.session_state.pop(rkey, None)
                st.error(f"Не удалось рассчитать рецепт ветки: {exc}")
        cached_rec = st.session_state.get(rkey)
        if cached_rec is not None:
            df_rec, xls_bytes = cached_rec
            st.dataframe(df_rec, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Скачать рецепт ветки в Excel (.xlsx)", data=xls_bytes,
                file_name=f"branch_{bsel}_recipe.xlsx",
                key=f"camp_wb_recipe_dl_{bsel}",
                mime="application/vnd.openxmlformats-officedocument."
                     "spreadsheetml.sheet")

        # §17.3 (Ш2) гейт: перед предложением/пересчётом — проверка полноты данных
        ready = ctrl.validate_ready(bsel)

        if not ready["ok"]:
            st.error("Не хватает данных для добора (§17.3):\n" + ready["text"])

        wc = st.columns([1, 1, 1])
        wb_n = wc[0].number_input("N точек", min_value=1, max_value=20, value=3,
                                  step=1, key=f"camp_wb_n_{bsel}")
        wb_expl = wc[1].slider("explore", 0.0, 1.0, 0.3, 0.05,
                               key=f"camp_wb_expl_{bsel}")
        if wc[2].button("📐 Предложить точки (read-only)",
                        key=f"camp_wb_propose_{bsel}", disabled=not ready["ok"]):
            X = np.asarray(ctrl.propose_points(bsel, n_points=int(wb_n),
                                               explore_frac=float(wb_expl),
                                               n_candidates=200), float)
            st.session_state[kx] = X
            st.session_state.pop(ky, None)

        # P0: сводка ПОСЛЕДНЕГО долива живёт в session_state — переживает
        # st.rerun и остаётся видимой до следующего раунда этой ветки.
        last = st.session_state.get(f"camp_wb_last_{bsel}")
        if last is not None:
            # NB: без вложенного st.expander — рабочий стол сам живёт в
            # экспандере, а Streamlit запрещает вложенные экспандеры.
            st.success(last["msg"])
            st.caption("Измеренные отклики долитых точек (по всем P):")
            st.dataframe(last["points"], use_container_width=True,
                         hide_index=True)
            st.caption("Сводка источников общей базы:")
            st.dataframe(last["origins"], use_container_width=True,
                         hide_index=True)
            st.caption(last["stop"])

        Xs = st.session_state.get(kx)
        if Xs is None:
            return
        Xs = np.atleast_2d(np.asarray(Xs, float))
        if Xs.shape[0] == 0:
            st.info("Бюджет ветки исчерпан — предложить нечего.")
            return

        if st.button("🧪 Заполнить тестовыми (демо-оракул)",
                     key=f"camp_wb_fill_{bsel}"):
            st.session_state[ky] = np.vstack(
                [runner._measure(np.asarray(x, float)) for x in Xs])

        Ys = st.session_state.get(ky)
        df = pd.DataFrame(np.round(Xs, 4), columns=coord_names[:Xs.shape[1]])
        lab_cols = [f"{p} (lab)" for p in props]
        for j, col in enumerate(lab_cols):
            df[col] = (np.round(np.asarray(Ys, float)[:, j], 4)
                       if Ys is not None else np.nan)
        # Сквозная нумерация: предложенные точки ещё не залиты — показываем их
        # будущие номера в общей базе (len(points)+1 … +N). Явный read-only
        # столбец (st.data_editor игнорирует кастомный индекс — см. seed выше).
        df.insert(0, "№ опыта",
                  list(experiment_index(len(runner.points), len(df))))
        st.caption("Предложенные точки: координаты заблокированы, заполняются "
                   "только столбцы «свойство (lab)» (вручную или демо-кнопкой):")
        edited = st.data_editor(df, use_container_width=True, height=280,
                                hide_index=True,
                                disabled=["№ опыта", *coord_names[:Xs.shape[1]]],
                                key=f"camp_wb_editor_{bsel}")

        if st.button("💾 Долить измеренные (commit_measured)",
                     key=f"camp_wb_commit_{bsel}"):
            try:
                d_before = float(br_now.d_best)
                Y = np.column_stack([np.asarray(edited[c], float)
                                     for c in lab_cols])
                if np.isnan(Y).any():
                    raise ValueError(
                        "Заполните измеренные отклики (столбцы «… (lab)») для "
                        "ВСЕХ предложенных точек — вручную или кнопкой "
                        "«🧪 Заполнить тестовыми». Пустые ячейки (None) "
                        "доливать нельзя.")
                res = ctrl.commit_measured(bsel, Xs, Y)

                st.session_state.pop(kx, None)
                st.session_state.pop(ky, None)
                wdf = workbench_points_dataframe(runner, res)
                if not wdf.empty:
                    # Те же сквозные номера, что точки получили в общей базе:
                    # последние len(wdf) опытов проекта — явным столбцом (единый
                    # вид с редакторами выше, без служебного индекса).
                    wdf.insert(0, "№ опыта", list(experiment_index(
                        len(runner.points) - len(wdf), len(wdf))))
                    # origin-тег канонично хранится как branch:{id}; для показа —
                    # человекочитаемое «Имя (id)» (см. origin_label).
                    wdf["origin"] = [origin_label(runner, o) for o in wdf["origin"]]
                    wdf = wdf.rename(columns={"origin": "ветка"})
                oc = pd.DataFrame(
                    [{"источник": origin_label(runner, k), "точек": v}
                     for k, v in runner.origin_counts().items()])
                # §4-стоп (двойной): технический И экономический, читает роль ρ
                delta_d = float(res["d_best"]) - d_before
                dec = runner.branch_stop_decision(
                    bsel, delta_d=delta_d, ceil=br_now.satisfy_at,
                    n_round=int(res["added"]) or 1, n_candidates=200,
                    n_mc=128, seed=0)
                # P0: сводка раунда — в session_state (переживает st.rerun и
                # остаётся видимой), а rerun сразу обновляет капшены/таблицы
                # ВЫШЕ по странице (бюджет, d_best, роли, база).
                st.session_state[f"camp_wb_last_{bsel}"] = {
                    "msg": (f"Долито {res['added']} точек "
                            f"(origin=branch:{bsel}); d_best {d_before:.3f} → "
                            f"{res['d_best']:.3f} (монотонно не убывает); "
                            f"общая база = {res['n_base']} точек."),
                    "points": wdf,
                    "origins": oc,
                    "stop": (
                        f"§4-стоп: **{_STOP_RU.get(dec.reason, dec.reason)}** "
                        f"(Δd={delta_d:+.4f}, d_best={res['d_best']:.3f}, "
                        f"ceil={br_now.satisfy_at:.3f}, "
                        f"econ_red_flag={dec.econ_red_flag})."),
                }
                _invalidate_branch_caches()
                st.rerun()
            except (ValueError, KeyError, RuntimeError) as exc:
                st.error(str(exc))


def render_schema_evolution(ctrl: "cv.CampaignController") -> None:
    """§17.6 / §16.2 (Ш6): эволюция схемы В ЛЮБОЙ МОМЕНТ + пересчёт (UI).

    Штатная операция живого проекта: добавить процесс-переменную / компонент
    смеси / отклик, подвинуть границы области. Миграция старых точек — по ЯВНОЙ
    политике (A0.6: молчаливой миграции нет). Всё делегируется фасаду
    :class:`CampaignController` (логика/валидация там); общая база не урезается
    (И-1), версия схемы растёт (кроме region-move). Read/rare-write — по кнопке.
    """
    from ..core.schema_evolution import known_constant

    runner = ctrl.runner
    full_proc = (list(runner._full_proc.names)
                 if getattr(runner, "_full_proc", None) else [])
    full_mix = (list(runner._full_mix.names)
                if getattr(runner, "_full_mix", None) else [])
    cur_proc = list(runner.current_schema.process_names)
    cur_mix = list(runner.current_schema.mixture_names)
    hidden_proc = [p for p in full_proc if p not in cur_proc]
    hidden_mix = [m for m in full_mix if m not in cur_mix]

    with st.expander("🧬 Эволюция схемы (§16.2 — раскрыть ось/отклик, подвинуть "
                     "границы)"):
        st.caption(
            "Живой проект меняется: раскрыть объявленную процесс-ось/компонент, "
            "ввести новый отклик, подвинуть границы области. Миграция старых точек "
            "— по ЯВНОЙ политике (A0.6). База не урезается (И-1); версия растёт.")

        # --- раскрыть процесс-переменную (known_constant baseline) ---
        st.markdown("**➕ Раскрыть процесс-переменную** (из полной схемы)")
        if hidden_proc:
            pc = st.columns([2, 2, 2])
            pv = pc[0].selectbox("переменная", hidden_proc, key="camp_ev_proc")
            pconst = pc[1].number_input("константа у старых точек (миграция)",
                                        value=0.0, step=0.1, key="camp_ev_proc_c")
            if pc[2].button("➕ Добавить процесс-ось", key="camp_ev_proc_btn"):
                try:
                    ctrl.add_process_var(pv, known_constant(float(pconst)))
                    st.success(f"Процесс-ось «{pv}» раскрыта (миграция "
                               f"known_constant={pconst}); версия схемы поднята.")
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.error(str(exc))
        else:
            st.caption("Все объявленные процесс-оси уже в схеме.")

        # --- раскрыть компонент смеси (Σ-совместимо: known_constant(0)) ---
        st.markdown("**➕ Раскрыть компонент смеси** (грань симплекса C=0)")
        if hidden_mix:
            mc = st.columns([3, 2])
            mv = mc[0].selectbox("компонент", hidden_mix, key="camp_ev_mix")
            if mc[1].button("➕ Добавить компонент", key="camp_ev_mix_btn"):
                try:
                    ctrl.add_mixture_component(mv)
                    st.success(f"Компонент «{mv}» раскрыт (миграция "
                               "known_constant(0.0), грань симплекса); версия "
                               "поднята.")
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.error(str(exc))
        else:
            st.caption("Все объявленные компоненты смеси уже в схеме.")

        # --- ввести новый отклик ---
        st.markdown("**➕ Ввести новый отклик** (у старых точек Y=MISSING)")
        rc = st.columns([3, 2])
        new_resp = rc[0].text_input("имя отклика", value="", key="camp_ev_resp")
        if rc[1].button("➕ Добавить отклик", key="camp_ev_resp_btn"):
            try:
                from ..core.schema import ResponseSpec
                if not new_resp.strip():
                    raise ValueError("Задайте имя отклика.")
                ctrl.add_response(ResponseSpec(name=new_resp.strip()))
                st.success(f"Отклик «{new_resp.strip()}» введён в схему (версия "
                           "поднята; у старых точек значение MISSING, §13.7).")
            except (ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))

        # --- подвинуть границы (region-move, без bump) ---
        st.markdown("**↔ Подвинуть границы области** (relax/restrict, без bump)")
        bc = st.columns([2, 2, 2, 2])
        axes = cur_mix + cur_proc
        mv_ax = bc[0].selectbox("ось", axes, key="camp_ev_bound_ax")
        mv_lo = bc[1].number_input("нижняя", value=0.0, step=0.05,
                                   key="camp_ev_bound_lo")
        mv_hi = bc[2].number_input("верхняя", value=1.0, step=0.05,
                                   key="camp_ev_bound_hi")
        mv_intent = bc[3].selectbox("намерение", ["relax", "restrict"],
                                    key="camp_ev_bound_intent")
        if st.button("↔ Применить движение границ", key="camp_ev_bound_btn"):
            try:
                if mv_intent == "relax":
                    ctrl.relax_bounds(mv_ax, float(mv_lo), float(mv_hi))
                else:
                    ctrl.restrict_bounds(mv_ax, float(mv_lo), float(mv_hi))
                st.success(f"Границы «{mv_ax}» → [{mv_lo}, {mv_hi}] ({mv_intent}); "
                           "область обновлена (история цела, И-1).")
            except (ValueError, KeyError, RuntimeError) as exc:
                st.error(str(exc))


def render_screening_analysis(ctrl: "cv.CampaignController") -> None:
    """M3-минималка (UI): интерпретируемый анализ скрининга после измеренного seed.

    Показывает «что дали опыты» ДО построения веток: (1) сводную матрицу влияний
    «компонент × свойство» (ARD-важность 0…1) и (2) детальный разбор выбранного
    свойства — Scheffé-quadratic по составу: R²/adj-R²/RMSE/q_eff, коэффициенты,
    ANOVA, значимые термы и bar-chart важности компонентов. Анализ — ТОЛЬКО по
    составу (mixture-only, объём минималки); физика в целом остаётся за общими
    суррогатами GP+MoE (канон §5/§12). Дорогие ARD-фиты считаются ТОЛЬКО по кнопке
    (A0.6), результат кешируется в session_state; вся математика — в чистом слое
    :mod:`campaign_screening` (тестируется без Streamlit).
    """
    runner = ctrl.runner
    props = list(runner.property_names)
    with st.expander("📊 Анализ скрининга (M3 — Scheffé + ARD по составу)"):
        st.caption(
            "Интерпретируемая сводка сразу после измеренного стартового дизайна: "
            "какой компонент на какое свойство влияет (ARD-важность) и насколько "
            "квадратичная модель Шеффе по составу объясняет данные (R²). Процесс "
            "здесь не участвует — его роль за общими суррогатами GP+MoE. Считается "
            "по кнопке (ARD-фиты дорогие).")

        # --- сводная матрица влияний (главный итог скрининга) ---
        if st.button("📊 Рассчитать матрицу влияний (компонент × свойство)",
                     key="camp_m3_overview_btn"):
            with st.spinner("ARD-скрининг по каждому свойству…"):
                try:
                    st.session_state["camp_m3_matrix"] = csx.influence_matrix(
                        runner, n_restarts=4, seed=0)
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.session_state.pop("camp_m3_matrix", None)
                    st.error(f"Не удалось построить матрицу влияний: {exc}")
        mat = st.session_state.get("camp_m3_matrix")
        if mat is not None:
            st.caption("ARD-важность компонентов (0…1; максимум по свойству = 1) "
                       "— чем ближе к 1, тем сильнее компонент влияет на свойство:")
            st.dataframe(mat, use_container_width=True)

        # --- детальный разбор одного свойства ---
        st.markdown("**🔬 Детальный разбор свойства (Scheffé-quadratic по составу)**")
        prop = st.selectbox("Свойство", props, key="camp_m3_prop")
        if st.button("🔬 Разобрать свойство (Scheffé-fit + ANOVA + ARD)",
                     key="camp_m3_fit_btn"):
            with st.spinner(f"Фит Шеффе + ARD для «{prop}»…"):
                try:
                    st.session_state["camp_m3_report"] = csx.screening_report(
                        runner, prop, n_restarts=4, seed=0)
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.session_state.pop("camp_m3_report", None)
                    st.error(f"Не удалось разобрать свойство: {exc}")
        rep = st.session_state.get("camp_m3_report")
        if rep is not None and rep.get("property") in props:
            s = rep["summary"]
            mc = st.columns(4)
            mc[0].metric("R²", f"{s['r2']:.3f}")
            adj = s.get("adj_r2")
            mc[1].metric("adj-R²",
                         f"{adj:.3f}" if isinstance(adj, (int, float)) else "—")
            mc[2].metric("RMSE", f"{s['rmse']:.3g}")
            mc[3].metric("q_eff", rep["q_eff"])
            if s.get("underdetermined"):
                st.warning("n < p: модель недоопределена — R² вводит в "
                           "заблуждение, добавьте опыты (§17.4).")
            st.caption(f"Свойство «{rep['property']}» — коэффициенты Шеффе "
                       f"(mixture-only, {rep['model']}):")
            st.dataframe(pd.DataFrame(rep["coefficients"]),
                         use_container_width=True, hide_index=True)
            st.caption("ANOVA (значимость регрессии в целом, F-тест):")
            st.dataframe(pd.DataFrame(rep["anova"]),
                         use_container_width=True, hide_index=True)
            rank = pd.DataFrame(rep["component_ranking"])
            if not rank.empty and "importance" in rank.columns:
                st.caption("Важность компонентов (ARD, 0…1):")
                st.bar_chart(rank.set_index("component")["importance"])
            st.caption(f"Значимых термов (p < {rep['alpha']}): "
                       f"{rep['n_significant']}.")
            if rep["n_significant"]:
                st.dataframe(pd.DataFrame(rep["significant_terms"]),
                             use_container_width=True, hide_index=True)


def render_campaign() -> None:
    """Вкладка «🧬 Кампания»: реальный сетап §17.4 + роли + мультицель §16.3 +
    рабочий стол §16.4 + смена роли §5 + spawn §8 + undo §7 (мутации — по кнопке)."""


    # UX: демо-кнопки — компактно в ЗАГОЛОВКЕ (popover справа от подзаголовка;
    # для старых Streamlit без st.popover — фолбэк-экспандер), а не отдельным
    # широким блоком посреди страницы.
    hdr = st.columns([5, 1])
    hdr[0].subheader("🧬 Проект: per-branch роли откликов и эволюция (ТЗ v1.1)")
    _ctrl_now = get_campaign_controller()
    with hdr[1]:
        if hasattr(st, "popover"):
            _demo_box = st.popover("🧪 Демо")
        else:  # pragma: no cover — старые Streamlit без st.popover
            _demo_box = st.expander("🧪 Демо", expanded=_ctrl_now is None)
    st.caption(
        "Роль отклика — атрибут пары (ветка × отклик): один и тот же ρ может быть "
        "ЦЕЛЬЮ в одной ветке и ЦЕНОЙ-ВХОДОМ в другой. Денежный канал ρ читается из "
        "РЕАЛЬНОЙ атрибуции ядра (И-5/Гр-1): OPTIMIZED ⇒ занулён, PRICE_INPUT ⇒ "
        "живой. Всё, что меняет состояние, делает только ваша кнопка (A0.6).")

    # P0: уведомления мутаций (переживают st.rerun) — показ в начале прогона.
    _show_flashes()

    # §17.4 (Ш3b): форма реального сетапа проекта — в ЛЕВОЙ панели (сайдбар).
    render_setup_form()

    # P0: демо-проект — с явным подтверждением, если кампания уже есть
    # (раньше кнопка молча ЗАТИРАЛА текущую кампанию одним кликом).
    with _demo_box:
        st.caption("Демо-проект (синтетический оракул {A,B,C}×{T,P}) — готовый "
                   "проект для знакомства с интерфейсом: общий пул + "
                   "две контрастные ветки (premium: ρ=PRICE_INPUT, канал живой; "
                   "rho_focus: ρ=OPTIMIZED, канал занулён).")
        if _ctrl_now is not None:
            st.warning("Проект уже есть в сессии: создание демо ЗАМЕНИТ его. "
                       "Несохранённые изменения пропадут — сначала сохраните "
                       "проект в сайдбаре («📁 Проект»).")
            demo_ok = st.checkbox(
                "Понимаю: заменить текущий проект демо-проектом",
                key="camp_create_confirm")
        else:
            demo_ok = True
        if st.button("🧬 Создать / сбросить демо-проект", key="camp_create",
                     disabled=not demo_ok):
            with st.spinner("Сборка демо-проекта (общий пул + 2 ветки)…"):
                runner = build_demo_campaign_runner()
                st.session_state["campaign_ctrl"] = cv.CampaignController(runner)
            _flash("Демо-проект создан: ветки **premium** (ρ=PRICE_INPUT, "
                   "канал живой) и **rho_focus** (ρ=OPTIMIZED, канал занулён).")
            st.rerun()

    ctrl = get_campaign_controller()
    if ctrl is None:
        st.info("Соберите проект в форме «🆕 Новый проект» (левая панель) или "
                "создайте демо-проект — кнопка «🧪 Демо» в заголовке "
                "(синтетический оракул {A,B,C}×{T,P}).")
        return
    runner = ctrl.runner

    # C2: read-only сводка настроек ДЕЙСТВУЮЩЕГО проекта (из движка) — видно,
    # что доли компонентов/границы процесса/отклики подтянулись после загрузки.
    render_project_settings(runner)

    # §17.4 (Ш3b): пока стартовый дизайн НЕ измерен (база пуста) — единственная
    # активная секция это ручной seed-цикл; ветко-UI ниже требует измеренных данных.
    if len(runner.points) == 0:
        render_seed_entry(ctrl)
        return

    # §17.5 (Ш4): ручное создание веток — доступно после измеренного seed.
    render_branch_creation(ctrl)
    # §17.6/§16.2 (Ш6): эволюция схемы в любой момент (штатная операция проекта).
    render_schema_evolution(ctrl)

    # M3-минималка: интерпретируемый анализ скрининга сразу после измеренного seed
    # (Scheffé + ARD по составу) — «что дали опыты» до построения веток.
    render_screening_analysis(ctrl)

    # C3 (§17.6.1): выгрузка ОБЩЕЙ базы опытов кампании в Excel (+ расход сырья).

    with st.expander("⬇️ Выгрузить общую базу опытов в Excel (C3)"):
        st.caption(
            "Общая база всех измеренных опытов (И-1): № опыта, источник, состав "
            "(доли) + процесс (реальные единицы) и измеренные отклики. Укажите "
            f"размер партии ({MASS_UNIT}) — добавится расход каждого компонента на "
            "опыт, чтобы понимать, сколько сырья взвесить (замечание 7).")
        bc = st.columns([1, 2])
        batch = bc[0].number_input(
            f"Размер партии, {MASS_UNIT}/опыт", min_value=0.0, value=0.0,
            step=0.1, key="camp_base_batch",
            help="0 — только состав в долях; >0 — добавит столбцы расхода сырья "
                 f"({MASS_UNIT}) = доля компонента × размер партии.")
        batch_kg = float(batch) if batch > 0 else None
        base_df = campaign_base_dataframe(runner, batch_kg=batch_kg)
        st.dataframe(base_df, use_container_width=True, hide_index=True)
        _blk_txt = base_blocking_caption(runner)
        if _blk_txt:
            st.caption(_blk_txt)
        st.download_button(
            "⬇️ Скачать .xlsx",
            data=campaign_base_excel_bytes(runner, batch_kg=batch_kg),
            file_name="campaign_base.xlsx", key="camp_base_dl",
            mime="application/vnd.openxmlformats-officedocument."
                 "spreadsheetml.sheet")

    # §17.2.1: КОРРЕКЦИЯ ошибки ввода измеренных откликов (правка опечатки Y).
    with st.expander("✏️ Исправить измеренные отклики (коррекция ошибок ввода, "
                     "§17.2.1)"):
        st.caption(
            "Опечатку/ошибку ВВОДА отклика можно исправить: правятся только "
            "значения откликов, состав и «№ опыта» сохраняются (И-1 — история не "
            "урезается и не переупорядочивается). После сохранения суррогаты "
            "переобучаются, ветки переоцениваются. Координаты не редактируются.")
        props_corr = list(runner.property_names)
        edit_df = measured_responses_editor_df(runner)
        edited_corr = st.data_editor(
            edit_df, use_container_width=True, hide_index=True,
            disabled=["№ опыта", "источник"], key="camp_correct_editor")
        if st.button("💾 Сохранить исправления откликов", key="camp_correct_save"):
            try:
                n_fixed = 0
                for ridx in range(len(edited_corr)):
                    changes: Dict[str, float] = {}
                    for pn in props_corr:
                        ov = edit_df.iloc[ridx][pn]
                        nv = edited_corr.iloc[ridx][pn]
                        if pd.isna(nv):
                            continue
                        if pd.isna(ov) or not np.isclose(float(ov), float(nv)):
                            changes[pn] = float(nv)
                    if changes:
                        ctrl.correct_measured_point(ridx, changes)
                        n_fixed += 1
                if n_fixed:
                    _invalidate_branch_caches()
                    _flash(f"Исправлено опытов: {n_fixed}. Суррогаты "
                           "переобучены, ветки переоценены (И-1).")
                    st.rerun()
                else:
                    st.info("Изменений не обнаружено — редактировать нечего.")
            except (ValueError, KeyError, IndexError, RuntimeError) as exc:
                st.error(str(exc))

    bids = list(runner.branches)

    if not bids:
        st.info("Стартовый дизайн измерен, суррогаты обучены (общая база = "
                f"{len(runner.points)} точек). Создайте ветку в форме "
                "«➕ Создать ветку вручную» выше (Ш4, §17.5).")
        return


    # --- линза ветки (Тр-3.3): роли В КОНТЕКСТЕ выбранной ветки ----------

    def _branch_label(bid: str) -> str:
        """Показать имя ветки (+id) в селекторе; значение опции остаётся id."""
        br = runner.branches.get(bid)
        return f"{br.name} ({bid})" if br is not None else str(bid)

    bsel = st.selectbox("Ветка (линза контекста — Тр-3.3)", bids,
                        key="camp_branch", format_func=_branch_label)
    rep = ctrl.role_report(bsel)
    st.caption(f"Линза ветки: **{rep['branch_name']}** (`{bsel}`). Role-tag "
               "валиден ТОЛЬКО в этом контексте; смена ветки меняет теги.")
    st.dataframe(role_table_dataframe(rep), use_container_width=True)

    with st.expander("💰 Почему за ρ есть/нет денег (§16.1)"):
        # P0: MC-оценка (n_mc=128) раньше считалась на КАЖДЫЙ rerun страницы
        # (любой клик по любому виджету) — теперь только по кнопке + кеш
        # (сбрасывается _invalidate_branch_caches после мутаций ветки).
        st.caption("Объяснение денежного канала ρ (MC-оценка) считается по "
                   "кнопке и кешируется до следующего изменения ветки.")
        mkey = f"cache_money_{bsel}"
        if st.button("💰 Объяснить денежный канал ρ",
                     key=f"camp_money_btn_{bsel}"):
            with st.spinner("MC-оценка денежного канала ρ…"):
                try:
                    st.session_state[mkey] = ctrl.money_explanation(
                        bsel, n_candidates=200, n_mc=128, seed=0)["text"]
                except (ValueError, KeyError, RuntimeError) as exc:
                    st.session_state.pop(mkey, None)
                    st.error(f"Не удалось оценить денежный канал: {exc}")
        if st.session_state.get(mkey):
            st.markdown(st.session_state[mkey])

    # --- §16.3: мультицелевой редактор ветки (несколько целей/диапазонов/весов)
    with st.expander("🎯 Редактор целей ветки (§16.3 — мультицель)"):
        st.caption(
            "Ветка — это НАБОР целей: несколько откликов, каждый со своим видом "
            "(min/max/target), диапазоном и весом (снято ограничение «одна ветка — "
            "одна цель»). Цель над откликом делает его роль OPTIMIZED (§16.0); "
            "удаление ПОСЛЕДНЕЙ цели запрещено — ветке нужен объектив. Правки "
            "обратимы (undo, §7) и НЕ трогают измеренную правду (И-1).")
        st.dataframe(goal_editor_dataframe(runner, bsel), use_container_width=True)

        st.markdown("**➕/✏️ Задать или заменить цель над откликом**")
        gc = st.columns([2, 2, 2, 2, 2])
        g_resp = gc[0].selectbox("Цель (отклик)", list(runner.property_names),
                                 key="camp_goal_resp")

        g_kind = gc[1].selectbox("вид", ["max", "min", "target"],
                                 key="camp_goal_kind")
        g_lo = gc[2].number_input("low", value=0.0, step=0.5, key="camp_goal_lo")
        g_hi = gc[3].number_input("high", value=10.0, step=0.5, key="camp_goal_hi")
        g_w = gc[4].number_input("Значимость цели", min_value=0.01, value=1.0,
                                 step=0.5, key="camp_goal_w", help=_WEIGHT_HELP)

        g_tgt = st.number_input("target (только для вида target; low<target<high)",
                                value=5.0, step=0.5, key="camp_goal_tgt")
        if st.button("💾 Задать / заменить цель", key="camp_goal_set"):
            try:
                tgt = float(g_tgt) if g_kind == "target" else None
                spec = DesirabilitySpec(g_kind, low=float(g_lo), high=float(g_hi),
                                        target=tgt, weight=float(g_w))
                res = ctrl.set_desirability(bsel, g_resp, spec)
                shift = res["recommendation_shift"]
                _invalidate_branch_caches()
                _flash(
                    f"Цель «{g_resp}» ({g_kind}) задана; d_best "
                    f"{res['d_best_before']:.3f} → {res['d_best_after']:.3f}"
                    + (f"; рекомендация x* сместилась на ≈{shift:.3f}."
                       if shift is not None else "; x* пересчитана."))
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))

        goals_now = list(runner.branches[bsel].goal or {})
        if goals_now:
            st.markdown("**⚖️ Значимость целей (относительный приоритет в "
                        "геом-среднем d_i)**")
            wcols = st.columns(len(goals_now))
            new_w: Dict[str, float] = {}
            for i, resp in enumerate(goals_now):
                cur_w = float(runner.branches[bsel].goal[resp].weight)
                new_w[resp] = wcols[i].number_input(
                    f"Значимость «{resp}»", min_value=0.01, value=cur_w,
                    step=0.5, key=f"camp_goal_w_{resp}", help=_WEIGHT_HELP)

            if st.button("⚖️ Применить веса", key="camp_goal_weights"):
                try:
                    res = ctrl.set_weights(
                        bsel, {r: float(v) for r, v in new_w.items()})
                    _invalidate_branch_caches()
                    _flash(f"Веса обновлены; d_best → "
                           f"{res['d_best_after']:.3f} (re-score, И-1).")
                    st.rerun()
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))

            st.markdown("**🗑 Удалить цель** (последняя — отказ)")
            dc = st.columns([3, 2])
            del_resp = dc[0].selectbox("цель на удаление", goals_now,
                                       key="camp_goal_del_sel")
            if dc[1].button("🗑 Удалить цель", key="camp_goal_del"):
                try:
                    ctrl.delete_goal(bsel, del_resp)
                    _invalidate_branch_caches()
                    _flash(f"Цель «{del_resp}» удалена (роль → REFERENCE).")
                    st.rerun()
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))

    # --- §17.5 (Ш5): рабочий стол ветки на РУЧНОМ цикле §17.2 (не авто-оракул)
    render_workbench(ctrl, bsel)

    # --- смена роли ρ (§5) ----------------------------------------------

    st.markdown("**🔁 Сменить роль ρ (§5) — переключает денежный канал**")
    rho = _rho_of(runner, bsel)
    if rho is None:
        st.caption("У ветки нет ценовой ноги — переключать роль ρ нечем.")
    else:
        cur = runner.response_role(bsel, rho)
        target = ROLE_PRICE_INPUT if cur == ROLE_OPTIMIZED else ROLE_OPTIMIZED
        st.caption(f"ρ = «{rho}»: текущая роль **{cur}** → станет **{target}** "
                   f"(канал {'ALIVE' if target == ROLE_PRICE_INPUT else 'ZEROED'}).")
        if target == ROLE_OPTIMIZED:
            cc = st.columns(3)
            cc[0].selectbox("вид цели", ["min", "max", "target"],
                            key="camp_sw_kind")
            cc[1].number_input("low", value=0.5, key="camp_sw_lo")
            cc[2].number_input("high", value=1.5, key="camp_sw_hi")
        if st.button(f"Сменить роль ρ → {target}", key="camp_do_switch"):
            try:
                spec = None
                if target == ROLE_OPTIMIZED:
                    kind = st.session_state.get("camp_sw_kind", "min")
                    lo = float(st.session_state.get("camp_sw_lo", 0.5))
                    hi = float(st.session_state.get("camp_sw_hi", 1.5))
                    tgt = (lo + hi) / 2.0 if kind == "target" else None
                    spec = DesirabilitySpec(kind, low=lo, high=hi, target=tgt)
                res = ctrl.switch_role(bsel, rho, target, spec=spec)
                shift = res["recommendation_shift"]
                _invalidate_branch_caches()
                _flash(
                    f"Роль ρ: {res['role_before']} → {res['role_after']}; "
                    f"канал занулён = {res['price_channel_suppressed']}; "
                    + (f"рекомендация x* сместилась на ≈{shift:.3f}."
                       if shift is not None else "рекомендация x* пересчитана."))
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))

    # --- undo (§7) + прогон раунда (запечатывает дно) -------------------
    cu = st.columns(2)
    if ctrl.can_undo():
        if cu[0].button("↩️ Undo последней настройки (§7)", key="camp_undo"):
            u = ctrl.undo()
            _invalidate_branch_caches()
            _flash(f"Откат «{u['undone']}» ветки {u['branch_id']} "
                   f"(undo_available={u['undo_available']}).", kind="info")
            st.rerun()
    else:
        cu[0].caption("Undo пуст: дно — последний снятый раунд (И-1).")
    # P0: авто-раунд зовёт оракул раннера; в РУЧНОЙ кампании (ManualOracle) это
    # записало бы в реальную базу СИНТЕТИЧЕСКИЕ Y — кнопка скрыта (A0.6).
    if is_manual_campaign(runner):
        cu[1].caption("Авто-прогон раунда недоступен: отклики этого проекта "
                      "вносятся ВРУЧНУЮ — используйте «🛠 Рабочий стол ветки» "
                      "выше (предложить → внести Y → долить).")
    elif cu[1].button("▶ Прогнать раунд ветки (демо-оракул; запечатает undo, "
                      "Тр-7.2/7.3)", key="camp_run_round"):
        try:
            ctrl.run_round(bsel, n_points=2, explore_frac=0.2, n_candidates=150)
            _invalidate_branch_caches()
            _flash("Раунд снят: новые измерения в общей базе, дно undo "
                   "обновлено.")
            st.rerun()
        except (ValueError, KeyError, RuntimeError) as exc:
            st.error(str(exc))

    # --- spawn ветки (§8) с наследованием ролей -------------------------
    st.markdown("**🌱 Spawn ветки (§8) — наследование ролей + review-сводка**")
    # NB: локальная переменная НЕ «cs» — иначе затеняется модуль campaign_state.
    spc = st.columns([2, 2, 2])
    parent = spc[0].selectbox("Родитель", bids, key="camp_spawn_parent",
                              format_func=_branch_label)
    cname = spc[1].text_input("Имя ребёнка", value="child", key="camp_spawn_name")
    over = spc[2].checkbox("новая цель над ρ (перебьёт роль, Тр-8.1в)",
                           key="camp_spawn_over")
    prho = _rho_of(runner, parent)
    new_goals = ({prho: DesirabilitySpec("min", low=0.5, high=1.5)}
                 if over and prho else None)

    if st.button("👁 Предпросмотр наследования (без создания)",
                 key="camp_spawn_preview"):
        rev = ctrl.preview_spawn(parent, new_goals=new_goals)
        st.dataframe(spawn_review_dataframe(rev), use_container_width=True)
        if rev["any_role_changed_by_objective"]:
            st.warning("Объектив ветки перебил унаследованную роль ρ — канал "
                       "цены будет занулён (И-5).")
    if st.button("🌱 Создать ветку (spawn)", key="camp_do_spawn"):
        try:
            cid = f"{parent}_child{len(bids)}"
            res = ctrl.spawn_branch(parent, cname, child_id=cid,
                                    new_goals=new_goals)
            _invalidate_branch_caches()
            # P0: review-сводка — в session_state (переживает st.rerun); rerun
            # нужен, чтобы селектор веток/таблицы сразу увидели ребёнка.
            st.session_state["camp_spawn_last"] = res["review"]
            _flash(
                f"Ветка «{res['child_name']}» создана (`{res['child_id']}`); "
                f"канал ρ занулён = {res['price_channel_suppressed']}.")
            st.rerun()
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if st.session_state.get("camp_spawn_last") is not None:
        st.caption("Review-сводка наследования ролей последнего spawn:")
        st.dataframe(spawn_review_dataframe(st.session_state["camp_spawn_last"]),
                     use_container_width=True)
