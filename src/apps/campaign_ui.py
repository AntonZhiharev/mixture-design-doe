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

import zlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st


from ..core.schema import ModelSpec, ProjectSchema, VariableBlock
from ..optimize.desirability import DesirabilitySpec
from ..apps.mixture_process_runner import MixtureProcessRunner
from ..apps import campaign as cv
from ..design.branches import ROLE_OPTIMIZED, ROLE_PRICE_INPUT


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


def role_table_dataframe(report: Dict[str, Any]) -> pd.DataFrame:
    """Role-репорт ветки → таблица для показа (контекст ветки уже зашит в report)."""
    rows = []
    for r in report["responses"]:
        rows.append({
            "отклик": r["response"],
            "роль": r["role_label"],
            "код": r["role_code"],
            "в цели": "да" if r["in_goal"] else "—",
            "вес": (r["weight"] if r["weight"] is not None else "—"),
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
    return df


# ----------------------------------------------------------------------
# Streamlit-рендер вкладки (тест — headless AppTest)
# ----------------------------------------------------------------------
def get_campaign_controller() -> Optional["cv.CampaignController"]:
    """Контроллер демо-кампании из session_state (или ``None``, если не создан)."""
    return st.session_state.get("campaign_ctrl")


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


def render_setup_form() -> None:
    """§17.4 (Ш3b): форма РЕАЛЬНОГО сетапа — mixture + процесс + отклики.

    По кнопке строит :class:`MixtureProcessRunner` (:class:`ManualOracle`, пустая
    база) и кладёт :class:`CampaignController` в ``session_state`` под тем же
    ключом, что и демо-кампания (главный поток §17 — один движок). Реального
    оракула нет: стартовые отклики вносит пользователь в ручном seed-цикле ниже.
    """
    with st.expander("🆕 Новый проект кампании — реальный сетап (§17.4)",
                     expanded=get_campaign_controller() is None):
        st.caption(
            "Составная область СРАЗУ: симплекс компонентов смеси (Σ=1) × куб "
            "процесс-параметров (реальные единицы). Отклики (свойства) меряются "
            "вручную — оракула-симулятора нет (кнопка «Заполнить тестовыми» в "
            "seed-цикле оставлена для прогонов без лаборатории, A0.6).")
        c = st.columns(2)
        mix_txt = c[0].text_input("Компоненты смеси (через запятую)",
                                  value="A, B, C", key="setup_mix")
        resp_txt = c[1].text_input("Отклики / свойства (через запятую)",
                                   value="strength, gloss, rho", key="setup_resp")
        proc_txt = st.text_input("Процесс-параметры (через запятую)",
                                 value="T, P", key="setup_proc")
        pc = st.columns(2)
        plo_txt = pc[0].text_input("Нижние границы процесса (через запятую)",
                                   value="0, 0", key="setup_proc_lo")
        phi_txt = pc[1].text_input("Верхние границы процесса (через запятую)",
                                   value="1, 1", key="setup_proc_hi")
        seed_v = st.number_input("Seed раннера", value=1, step=1, key="setup_seed")
        if st.button("🏗 Построить проект кампании", key="setup_build"):
            try:
                mix = _parse_names(mix_txt)
                proc = _parse_names(proc_txt)
                resp = _parse_names(resp_txt)
                plo = _parse_floats(plo_txt)
                phi = _parse_floats(phi_txt)
                if plo is None or phi is None:
                    raise ValueError("Границы процесса — числа через запятую.")
                runner = build_setup_runner(
                    mixture_names=mix, process_names=proc,
                    process_lower=plo, process_upper=phi,
                    response_names=resp, seed=int(seed_v))
                st.session_state["campaign_ctrl"] = cv.CampaignController(runner)
                for k in ("setup_seed_X", "setup_seed_Y"):
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
    sc = st.columns([1, 1, 1])
    seed_n = sc[0].number_input("N seed-точек", min_value=2, max_value=60,
                                value=12, step=1, key="setup_seed_n")
    seed_design = sc[1].number_input("seed дизайна", value=1, step=1,
                                     key="setup_seed_design")
    if sc[2].button("📐 Предложить seed-дизайн", key="setup_propose_seed"):
        X = np.asarray(ctrl.propose_seed(int(seed_n), seed=int(seed_design)), float)
        st.session_state["setup_seed_X"] = X
        st.session_state.pop("setup_seed_Y", None)

    Xs = st.session_state.get("setup_seed_X")
    if Xs is None:
        return
    Xs = np.atleast_2d(np.asarray(Xs, float))

    if st.button("🧪 Заполнить тестовыми (демо-оракул)", key="setup_fill_demo"):
        st.session_state["setup_seed_Y"] = np.vstack(
            [runner._measure(np.asarray(x, float)) for x in Xs])

    Ys = st.session_state.get("setup_seed_Y")
    df = pd.DataFrame(np.round(Xs, 4), columns=coord_names[:Xs.shape[1]])
    lab_cols = [f"{p} (lab)" for p in props]
    for j, col in enumerate(lab_cols):
        df[col] = (np.round(np.asarray(Ys, float)[:, j], 4)
                   if Ys is not None else np.nan)
    st.caption("Составные координаты заблокированы; заполняются только столбцы "
               "«свойство (lab)» (вручную или кнопкой «Заполнить тестовыми»):")
    edited = st.data_editor(df, use_container_width=True, height=320,
                            disabled=coord_names[:Xs.shape[1]],
                            key="setup_seed_editor")
    if st.button("💾 Зафиксировать seed (commit_seed)", key="setup_commit_seed"):
        try:
            Y = np.column_stack([np.asarray(edited[c], float) for c in lab_cols])
            out = ctrl.commit_seed(Xs, Y)
            for k in ("setup_seed_X", "setup_seed_Y"):
                st.session_state.pop(k, None)
            st.success(
                f"Seed зафиксирован: +{out['added']} точек (origin=seed), общая "
                f"база = {out['n_base']}, суррогаты обучены. Дальше — создание "
                "веток (Ш4, §17.5).")
        except (ValueError, KeyError) as exc:
            st.error(str(exc))


def render_campaign() -> None:
    """Вкладка «🧬 Кампания»: реальный сетап §17.4 + роли + мультицель §16.3 +
    рабочий стол §16.4 + смена роли §5 + spawn §8 + undo §7 (мутации — по кнопке)."""
    st.subheader("🧬 Кампания: per-branch роли откликов и эволюция (ТЗ v1.1)")
    st.caption(
        "Роль отклика — атрибут пары (ветка × отклик): один и тот же ρ может быть "
        "ЦЕЛЬЮ в одной ветке и ЦЕНОЙ-ВХОДОМ в другой. Денежный канал ρ читается из "
        "РЕАЛЬНОЙ атрибуции ядра (И-5/Гр-1): OPTIMIZED ⇒ занулён, PRICE_INPUT ⇒ "
        "живой. Всё, что меняет состояние, делает только ваша кнопка (A0.6).")

    # §17.4 (Ш3b): форма реального сетапа проекта (mixture + процесс + отклики).
    render_setup_form()

    if st.button("🧬 Создать / сбросить демо-кампанию", key="camp_create"):
        with st.spinner("Сборка демо-кампании (общий пул + 2 ветки)…"):
            runner = build_demo_campaign_runner()
            st.session_state["campaign_ctrl"] = cv.CampaignController(runner)
        st.success("Демо-кампания создана: ветки **premium** (ρ=PRICE_INPUT, канал "
                   "живой) и **rho_focus** (ρ=OPTIMIZED, канал занулён).")

    ctrl = get_campaign_controller()
    if ctrl is None:
        st.info("Соберите проект в форме «🆕 Новый проект кампании» или нажмите "
                "«Создать демо-кампанию» (синтетический оракул {A,B,C}×{T,P}).")
        return
    runner = ctrl.runner

    # §17.4 (Ш3b): пока стартовый дизайн НЕ измерен (база пуста) — единственная
    # активная секция это ручной seed-цикл; ветко-UI ниже требует измеренных данных.
    if len(runner.points) == 0:
        render_seed_entry(ctrl)
        return

    bids = list(runner.branches)
    if not bids:
        st.info("Стартовый дизайн измерен, суррогаты обучены (общая база = "
                f"{len(runner.points)} точек). Создание веток (цели/роли/ценовая "
                "нога) — следующий шаг Ш4 (§17.5).")
        return

    # --- линза ветки (Тр-3.3): роли В КОНТЕКСТЕ выбранной ветки ----------

    bsel = st.selectbox("Ветка (линза контекста — Тр-3.3)", bids,
                        key="camp_branch")
    rep = ctrl.role_report(bsel)
    st.caption(f"Линза ветки: **{rep['branch_name']}** (`{bsel}`). Role-tag "
               "валиден ТОЛЬКО в этом контексте; смена ветки меняет теги.")
    st.dataframe(role_table_dataframe(rep), use_container_width=True)

    with st.expander("💰 Почему за ρ есть/нет денег (§16.1)"):
        ex = ctrl.money_explanation(bsel, n_candidates=200, n_mc=128, seed=0)
        st.markdown(ex["text"])

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
        g_resp = gc[0].selectbox("отклик", list(runner.property_names),
                                 key="camp_goal_resp")
        g_kind = gc[1].selectbox("вид", ["max", "min", "target"],
                                 key="camp_goal_kind")
        g_lo = gc[2].number_input("low", value=0.0, step=0.5, key="camp_goal_lo")
        g_hi = gc[3].number_input("high", value=10.0, step=0.5, key="camp_goal_hi")
        g_w = gc[4].number_input("вес", min_value=0.01, value=1.0, step=0.5,
                                 key="camp_goal_w")
        g_tgt = st.number_input("target (только для вида target; low<target<high)",
                                value=5.0, step=0.5, key="camp_goal_tgt")
        if st.button("💾 Задать / заменить цель", key="camp_goal_set"):
            try:
                tgt = float(g_tgt) if g_kind == "target" else None
                spec = DesirabilitySpec(g_kind, low=float(g_lo), high=float(g_hi),
                                        target=tgt, weight=float(g_w))
                res = ctrl.set_desirability(bsel, g_resp, spec)
                shift = res["recommendation_shift"]
                st.success(
                    f"Цель «{g_resp}» ({g_kind}) задана; d_best "
                    f"{res['d_best_before']:.3f} → {res['d_best_after']:.3f}"
                    + (f"; рекомендация x* сместилась на ≈{shift:.3f}."
                       if shift is not None else "; x* пересчитана."))
            except (ValueError, KeyError) as exc:
                st.error(str(exc))

        goals_now = list(runner.branches[bsel].goal or {})
        if goals_now:
            st.markdown("**⚖️ Веса целей (экспоненты геом-среднего d_i)**")
            wcols = st.columns(len(goals_now))
            new_w: Dict[str, float] = {}
            for i, resp in enumerate(goals_now):
                cur_w = float(runner.branches[bsel].goal[resp].weight)
                new_w[resp] = wcols[i].number_input(
                    f"вес «{resp}»", min_value=0.01, value=cur_w, step=0.5,
                    key=f"camp_goal_w_{resp}")
            if st.button("⚖️ Применить веса", key="camp_goal_weights"):
                try:
                    res = ctrl.set_weights(
                        bsel, {r: float(v) for r, v in new_w.items()})
                    st.success(f"Веса обновлены; d_best → "
                               f"{res['d_best_after']:.3f} (re-score, И-1).")
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))

            st.markdown("**🗑 Удалить цель** (последняя — отказ)")
            dc = st.columns([3, 2])
            del_resp = dc[0].selectbox("цель на удаление", goals_now,
                                       key="camp_goal_del_sel")
            if dc[1].button("🗑 Удалить цель", key="camp_goal_del"):
                try:
                    ctrl.delete_goal(bsel, del_resp)
                    st.success(f"Цель «{del_resp}» удалена (роль → REFERENCE).")
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))

    # --- §16.4: рабочий стол ветки — раунд добора точек (внутриветочный цикл)
    with st.expander("🛠 Рабочий стол ветки (§16.4 — раунд добора)"):
        st.caption(
            "Полный внутриветочный цикл: предложить N точек (acquisition/argmax по "
            "области) → измерить (демо-оракул) → долить в общую базу с origin-тегом "
            "ветки → переобучить суррогаты → x*/d_best → §4-стоп. A0.6: мерим "
            "только по кнопке; раунд запечатывает undo (измеренную правду не "
            "откатить, Тр-7.2/7.3).")
        br_now = runner.branches[bsel]
        st.caption(f"Ветка «{br_now.name}»: бюджет {br_now.budget}, потрачено "
                   f"{br_now.spent}, осталось {br_now.remaining()}, "
                   f"d_best={br_now.d_best:.3f}, статус {br_now.status}.")
        wc = st.columns([1, 1, 1])
        wb_n = wc[0].number_input("N точек", min_value=1, max_value=20, value=3,
                                  step=1, key="camp_wb_n")
        wb_expl = wc[1].slider("explore", 0.0, 1.0, 0.3, 0.05, key="camp_wb_expl")
        if wc[2].button("▶ Прогнать раунд добора", key="camp_wb_run"):
            try:
                d_before = float(br_now.d_best)
                res = ctrl.run_round(bsel, n_points=int(wb_n),
                                     explore_frac=float(wb_expl),
                                     n_candidates=200)
                st.success(
                    f"Долито {res['added']} точек (origin=branch:{bsel}); d_best "
                    f"{d_before:.3f} → {res['d_best']:.3f} (монотонно не убывает); "
                    f"общая база = {res['n_base']} точек.")
                st.caption("Измеренные отклики долитых точек (по ВСЕМ P свойствам):")
                st.dataframe(workbench_points_dataframe(runner, res),
                             use_container_width=True)
                oc = pd.DataFrame(
                    {"точек": runner.origin_counts()}).rename_axis("origin")
                st.dataframe(oc, use_container_width=True)
                # §4-стоп (двойной): технический И экономический, читает роль ρ
                delta_d = float(res["d_best"]) - d_before
                dec = runner.branch_stop_decision(
                    bsel, delta_d=delta_d, ceil=br_now.satisfy_at,
                    n_round=int(wb_n), n_candidates=200, n_mc=128, seed=0)
                st.caption(
                    f"§4-стоп: **{_STOP_RU.get(dec.reason, dec.reason)}** "
                    f"(Δd={delta_d:+.4f}, d_best={res['d_best']:.3f}, "
                    f"ceil={br_now.satisfy_at:.3f}, "
                    f"econ_red_flag={dec.econ_red_flag}).")
            except (ValueError, KeyError, RuntimeError) as exc:
                st.error(str(exc))

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
                st.success(
                    f"Роль ρ: {res['role_before']} → {res['role_after']}; "
                    f"канал занулён = {res['price_channel_suppressed']}; "
                    + (f"рекомендация x* сместилась на ≈{shift:.3f}."
                       if shift is not None else "рекомендация x* пересчитана."))
            except (ValueError, KeyError) as exc:
                st.error(str(exc))

    # --- undo (§7) + прогон раунда (запечатывает дно) -------------------
    cu = st.columns(2)
    if ctrl.can_undo():
        if cu[0].button("↩️ Undo последней настройки (§7)", key="camp_undo"):
            u = ctrl.undo()
            st.info(f"Откат «{u['undone']}» ветки {u['branch_id']} "
                    f"(undo_available={u['undo_available']}).")
    else:
        cu[0].caption("Undo пуст: дно — последний снятый раунд (И-1).")
    if cu[1].button("▶ Прогнать раунд ветки (запечатает undo, Тр-7.2/7.3)",
                    key="camp_run_round"):
        ctrl.run_round(bsel, n_points=2, explore_frac=0.2, n_candidates=150)
        st.success("Раунд снят: новые измерения в общей базе, дно undo обновлено.")

    # --- spawn ветки (§8) с наследованием ролей -------------------------
    st.markdown("**🌱 Spawn ветки (§8) — наследование ролей + review-сводка**")
    cs = st.columns([2, 2, 2])
    parent = cs[0].selectbox("Родитель", bids, key="camp_spawn_parent")
    cname = cs[1].text_input("Имя ребёнка", value="child", key="camp_spawn_name")
    over = cs[2].checkbox("новая цель над ρ (перебьёт роль, Тр-8.1в)",
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
            st.success(
                f"Ветка «{res['child_name']}» создана (`{res['child_id']}`); "
                f"канал ρ занулён = {res['price_channel_suppressed']}.")
            st.dataframe(spawn_review_dataframe(res["review"]),
                         use_container_width=True)
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
