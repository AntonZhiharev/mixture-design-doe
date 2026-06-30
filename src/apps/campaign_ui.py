"""apps/campaign_ui.py — Streamlit-вкладка «Кампания» (ТЗ v1.1, §16/§16.1).

ШАГ 4: тонкий UI поверх :mod:`src.apps.campaign` (ШАГ 1–3). Канон «логика+тест,
потом UI» соблюдён: вся честность живёт в campaign-слое; здесь — только показ и
кнопки. Разделение:

  * ЧИСТЫЕ хелперы (``build_demo_campaign_runner``, ``role_table_dataframe``,
    ``spawn_review_dataframe``) НЕ зовут Streamlit — тестируются напрямую;
  * :func:`render_campaign` рисует вкладку через ``st`` (тест — headless AppTest).

UI работает с :class:`CampaignController` поверх ОТДЕЛЬНОГО
:class:`MixtureProcessRunner` (составная область mixture×process с ρ), который не
смешивается с pipeline-runner-ом M1–M8: это другая модель проекта (канон §5/§12,
одна модель физики на проект). Демо-оракул синтетический и детерминированный —
чтобы вкладку можно было запустить без реальной лаборатории.

A0.6: всё, что меняет состояние (смена роли, spawn, раунд), делает ТОЛЬКО явная
кнопка пользователя; роли и денежный канал показываются read-only.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

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
# Чистые таблицы для показа (без Streamlit) — тестируемы напрямую
# ----------------------------------------------------------------------
_MONEY_RU = {cv.MONEY_ZEROED: "занулён (ZEROED)", cv.MONEY_ALIVE: "живой (ALIVE)",
             None: "—"}
_CHANGE_RU = {"inherited": "унаследовано как есть",
              "overridden_same_role": "тронуто, роль та же",
              "changed_by_objective": "изменено объективом ветки"}


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


def render_campaign() -> None:
    """Вкладка «🧬 Кампания»: read-only роли + смена роли §5 + spawn §8 + undo §7."""
    st.subheader("🧬 Кампания: per-branch роли откликов и эволюция (ТЗ v1.1)")
    st.caption(
        "Роль отклика — атрибут пары (ветка × отклик): один и тот же ρ может быть "
        "ЦЕЛЬЮ в одной ветке и ЦЕНОЙ-ВХОДОМ в другой. Денежный канал ρ читается из "
        "РЕАЛЬНОЙ атрибуции ядра (И-5/Гр-1): OPTIMIZED ⇒ занулён, PRICE_INPUT ⇒ "
        "живой. Всё, что меняет состояние, делает только ваша кнопка (A0.6).")

    if st.button("🧬 Создать / сбросить демо-кампанию", key="camp_create"):
        with st.spinner("Сборка демо-кампании (общий пул + 2 ветки)…"):
            runner = build_demo_campaign_runner()
            st.session_state["campaign_ctrl"] = cv.CampaignController(runner)
        st.success("Демо-кампания создана: ветки **premium** (ρ=PRICE_INPUT, канал "
                   "живой) и **rho_focus** (ρ=OPTIMIZED, канал занулён).")

    ctrl = get_campaign_controller()
    if ctrl is None:
        st.info("Нажмите «Создать демо-кампанию», чтобы начать "
                "(синтетический оракул {A,B,C}×{T,P}, ρ=плотность).")
        return
    runner = ctrl.runner
    bids = list(runner.branches)

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
