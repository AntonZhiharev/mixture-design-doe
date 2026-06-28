"""streamlit_app.py — UI пересобранного pipeline M1–M8 (REBUILD_SPEC §9).

Тонкая Streamlit-обёртка над `pipeline_runner.PipelineRunner`: пошаговый прогон
стадий M1…M8, визуализация результатов, save/load чекпоинтов `ProjectState` и
сравнение найденного рецепта с известным оптимумом синтетического полигона.

Запуск:
    streamlit run src/apps/streamlit_app.py
    # или:  python run_streamlit_app.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# repo root в sys.path (Streamlit запускает файл напрямую)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import streamlit as st  # noqa: E402

from src.core.state import ProjectState  # noqa: E402
from src.core.simplex import parts_ranges_to_fraction_bounds  # noqa: E402
from src.models.moe import MixtureOfExperts  # noqa: E402

from src.apps.pipeline_runner import (PipelineConfig, PipelineRunner,  # noqa: E402
                                       list_projects)
from src.optimize.desirability import DesirabilitySpec  # noqa: E402
from src.apps import assistant as ai  # noqa: E402


STAGES = [
    ("M1", "M1 · Геометрия области"),
    ("M2", "M2 · D-оптимальный дизайн"),
    ("M3", "M3 · Анализ (Scheffé + ARD)"),
    ("M4", "M4 · Кластеризация режимов"),
    ("M5", "M5 · I-оптимальный дизайн"),
    ("M6", "M6 · MoE-модель"),
    ("M7", "M7 · Active learning"),
    ("M8", "M8 · Оптимизация продукта"),
]

_XLSX_MIME = ("application/vnd.openxmlformats-officedocument."
              "spreadsheetml.sheet")


# ----------------------------------------------------------------------
def _parse_floats(text: str, n: int):
    try:
        vals = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        return None
    return vals if len(vals) == n else None


def _batch():
    """Текущий размер пробы и единица из session_state."""
    return (float(st.session_state.get("batch_size", 0.0)),
            str(st.session_state.get("batch_unit", "")))


def _excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for nm, d in sheets.items():
            d.to_excel(xl, sheet_name=str(nm)[:31])
    return buf.getvalue()


def _download(sheets: dict, fname: str, key: str,
              label: str = "⬇️ Скачать в Excel"):
    """Кнопка выгрузки одной или нескольких таблиц в .xlsx (fallback CSV)."""
    try:
        st.download_button(label, _excel_bytes(sheets), file_name=fname,
                           mime=_XLSX_MIME, key=key)
    except Exception:  # noqa: BLE001 — нет движка Excel → CSV
        df0 = next(iter(sheets.values()))
        st.download_button("⬇️ Скачать CSV",
                           df0.to_csv().encode("utf-8-sig"),
                           file_name=fname.replace(".xlsx", ".csv"),
                           mime="text/csv", key=key + "_csv")


def _df_design(design, names):
    return pd.DataFrame(np.round(design, 4),
                        columns=names[: design.shape[1]])


def _amounts(df: pd.DataFrame, comp_cols, batch: float, unit: str):
    """Пересчёт долей в количество сырья: доля × размер пробы."""
    amt = df.copy()
    for c in comp_cols:
        amt[c] = np.round(df[c].astype(float) * batch, 4)
    return amt.rename(columns={c: f"{c} ({unit})" if unit else f"{c}"
                               for c in comp_cols})


def _totals(df: pd.DataFrame, comp_cols, batch: float, unit: str):
    """Суммарный расход каждого компонента на весь план + общий вес."""
    sums = {c: round(float(np.asarray(df[c], float).sum()) * batch, 4)
            for c in comp_cols}
    label = f"Итого, {unit}" if unit else "Итого"
    out = pd.DataFrame({label: sums})
    out.index.name = "Компонент"
    out.loc["ВСЕГО"] = round(sum(sums.values()), 4)
    return out


def render_design_table(df: pd.DataFrame, comp_cols, title: str,
                        fname: str, key: str, batch: float, unit: str,
                        height: int | None = None, with_totals: bool = False,
                        start_index: int = 1):
    """Показать таблицу-план + (опц.) количества сырья + кнопку Excel.

    ``with_totals``: добавить таблицу «Итого по компонентам на весь план»
    (как на M2) — полезно для планов из многих опытов (M5).
    ``start_index``: с какого номера нумеровать опыты (для сквозной нумерации
    добора M5, продолжающего опыты M2 — напр. 31, 32, …).
    """
    show = df.copy()
    show.index = np.arange(start_index, start_index + len(show))
    show.index.name = "Опыт"

    # Streamlit >=1.30 запрещает height=None: передаём только валидное значение.
    _h = {"height": height} if isinstance(height, int) and height > 0 else {}
    st.caption(title)
    st.dataframe(show, use_container_width=True, **_h)
    sheets = {"Доли": show}
    if batch and batch > 0:
        amt = _amounts(show, comp_cols, batch, unit)
        st.caption(f"📦 Количество сырья на пробу {batch:g} {unit} "
                   f"(доля × размер пробы):")
        st.dataframe(amt, use_container_width=True, **_h)

        sheets[(f"Количество ({unit})" if unit else "Количество")] = amt
        if with_totals:
            tot = _totals(show, comp_cols, batch, unit)
            st.caption(f"⚖️ Итого сырья на ВЕСЬ план — {len(show)} опытов × "
                       f"{batch:g} {unit} (сколько всего закупить/взвесить):")
            st.dataframe(tot, use_container_width=True)
            sheets["Итого по плану"] = tot
    _download(sheets, fname, key)



# ----------------------------------------------------------------------
def _defaults() -> dict:
    """Текущие значения по умолчанию для виджетов конфигурации.

    Заполняются при загрузке проекта (`render_project_loader`), иначе пусты —
    виджеты используют свои встроенные дефолты.
    """
    return st.session_state.get("cfg_defaults", {})


def _ver() -> int:
    """Нонс версии конфигурации: меняется при загрузке проекта, чтобы все
    виджеты сайдбара пересоздались и подхватили новые значения."""
    return int(st.session_state.get("cfg_ver", 0))


def _composition_bounds(d: dict, v: int, q: int, names: list):
    """Ограничения состава: «Доли (0…1)» ИЛИ «Массовые части (база = 100)».

    В режиме частей база фиксируется в 100 частей, остальные задаются
    диапазоном частей, а диапазоны долей L_i ≤ x_i ≤ U_i (включая «плавающую»
    долю базы) считаются автоматически. Возвращает (lower, upper) — списки
    долей или (None, None), если ограничений нет (тривиальный 0…1).
    """
    d_lo = d.get("lower") or []
    d_hi = d.get("upper") or []
    d_pmin = d.get("parts_min") or []
    d_pmax = d.get("parts_max") or []
    saved_mode_idx = 1 if d.get("comp_mode") == "parts" else 0
    with st.sidebar.expander("📐 Ограничения состава (опц.)"):
        mode = st.radio(
            "Способ ввода",
            ["Доли (0…1)", "Массовые части (база = 100)"],
            index=saved_mode_idx, key=f"bmode_{v}",
            help="«Массовые части»: одна база = 100 частей, остальные задаются "
                 "диапазоном частей; доли (и плавающий диапазон доли базы) "
                 "рассчитываются автоматически.")

        if mode.startswith("Доли"):
            st.caption("Доли каждого компонента (0…1). Сумма нижних ≤ 1 ≤ "
                       "сумма верхних. Оставьте 0…1, если ограничений нет.")
            lower, upper = [], []
            for i in range(q):
                cc = st.columns(2)
                lo_i = cc[0].number_input(
                    f"L · {names[i]}", min_value=0.0, max_value=1.0,
                    value=float(d_lo[i]) if i < len(d_lo) else 0.0,
                    step=0.01, format="%.4f", key=f"lo_{v}_{q}_{i}")
                hi_i = cc[1].number_input(
                    f"U · {names[i]}", min_value=0.0, max_value=1.0,
                    value=float(d_hi[i]) if i < len(d_hi) else 1.0,
                    step=0.01, format="%.4f", key=f"hi_{v}_{q}_{i}")
                lower.append(float(lo_i)); upper.append(float(hi_i))
            nontrivial = any(l > 0 for l in lower) or any(u < 1 for u in upper)
            meta = {"comp_mode": "fractions", "base_index": None,
                    "parts_min": None, "parts_max": None}
            return (lower, upper, meta) if nontrivial else (None, None, meta)

        # --- режим массовых частей ---
        _sb = d.get("base_index")
        base_default = int(_sb) if isinstance(_sb, int) and _sb < q else 0
        base_i = st.selectbox(
            "Базовый компонент (= 100 частей)", list(range(q)),
            index=base_default,
            format_func=lambda i: names[i], key=f"base_{v}_{q}")
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
                f"min частей · {names[i]}", min_value=0.0,
                value=float(d_pmin[i]) if i < len(d_pmin) else 0.0,
                step=1.0, key=f"pmin_{v}_{q}_{i}")
            pmax[i] = cc[1].number_input(
                f"max частей · {names[i]}", min_value=0.0,
                value=float(d_pmax[i]) if i < len(d_pmax) else 10.0,
                step=1.0, key=f"pmax_{v}_{q}_{i}")
        try:
            lo_arr, hi_arr = parts_ranges_to_fraction_bounds(pmin, pmax)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось пересчитать части в доли: {exc}")
            return None, None, {"comp_mode": "parts", "base_index": int(base_i),
                                "parts_min": list(pmin), "parts_max": list(pmax)}
        tbl = pd.DataFrame(
            {"min частей": pmin, "max частей": pmax,
             "доля L": np.round(lo_arr, 4), "доля U": np.round(hi_arr, 4)},
            index=names[:q])
        st.caption("Рассчитанные диапазоны долей для алгоритма:")
        st.dataframe(tbl, use_container_width=True)
        meta = {"comp_mode": "parts", "base_index": int(base_i),
                "parts_min": list(pmin), "parts_max": list(pmax)}
        return lo_arr.tolist(), hi_arr.tolist(), meta


def sidebar_config():

    d = _defaults()
    v = _ver()
    st.sidebar.header("⚙️ Конфигурация проекта")
    name = st.sidebar.text_input("Имя проекта", d.get("name", "ui_project"),
                                 key=f"name_in_{v}")
    q = st.sidebar.slider("Число компонентов q", 2, 12,
                          int(d.get("q", 5)), key=f"q_{v}")
    model_opts = ["quadratic", "linear"]
    model = st.sidebar.selectbox(
        "Модель Scheffé (тренд)", model_opts,
        index=model_opts.index(d["model"]) if d.get("model") in model_opts else 0,
        key=f"model_{v}")
    noise = st.sidebar.slider("Шум лаборатории σ", 0.0, 1.0,
                              float(d.get("noise_sd", 0.2)), 0.05, key=f"noise_{v}")
    seed = st.sidebar.number_input("Seed", value=int(d.get("seed", 42)),
                                   step=1, key=f"seed_{v}")
    factor = st.sidebar.slider("n_runs = factor·p", 1.5, 3.0,
                               float(d.get("n_runs_factor", 2.0)), 0.1,
                               key=f"factor_{v}")
    n_restarts = st.sidebar.slider("GP/опт. restarts", 2, 15,
                                   int(d.get("n_restarts", 6)), key=f"restarts_{v}")
    n_blocks = st.sidebar.slider(
        "🧱 Блоки (партии/дни)", 1, 10, int(d.get("n_blocks", 1)),
        key=f"blocks_{v}",
        help="План экспериментов будет сбалансированно разбит на указанное "
             "число блоков (round-robin) — удобно ставить опыты партиями/по "
             "дням. Метка блока попадает в таблицу M2 и в Excel.")

    # --- Имена компонентов (по умолчанию A, B, C, …) ---
    d_names = d.get("names") or []
    default_names = [chr(ord("A") + i) for i in range(q)]
    with st.sidebar.expander("🏷️ Имена компонентов (опц.)"):
        names = [st.text_input(
                    f"Компонент {i + 1}",
                    (d_names[i] if i < len(d_names) else default_names[i]),
                    key=f"name_{v}_{q}_{i}").strip() or default_names[i]
                 for i in range(q)]

    # --- Целевые свойства (мультиотклик, §12): общая модель проекта ---
    d_props = d.get("property_names") or ["y"]
    with st.sidebar.expander("🎯 Целевые свойства (отклики)"):
        st.caption("Имена измеряемых свойств продукта через запятую. Первое — "
                   "первичное (для M7/M8/benchmark). M3 строит скрининг, а M6 — "
                   "суррогат на КАЖДОЕ свойство (общая модель проекта).")
        props_txt = st.text_input("Свойства (через запятую)",
                                   ", ".join(d_props), key=f"props_{v}")
        property_names = [p.strip() for p in props_txt.split(",") if p.strip()]
        if not property_names:
            property_names = ["y"]


    # --- Ограничения состава: доли (0..1) ИЛИ массовые части (база=100) ---
    lower, upper, comp_meta = _composition_bounds(d, v, q, names)


    # --- Стоимость компонентов: одна единица для всех + поле на компонент ---
    d_cost = d.get("cost_coeffs") or []
    cost_opts = ["— не учитывать", "₽/кг", "₽/л", "$/кг", "$/л", "у.е./ед."]
    cost_unit_default = d.get("cost_unit")
    cost_unit_idx = (cost_opts.index(cost_unit_default)
                     if cost_unit_default in cost_opts else 0)
    with st.sidebar.expander("💰 Стоимость компонентов (опц.)"):
        cost_unit = st.selectbox(
            "Единица цены (одна для всех компонентов)",
            cost_opts, index=cost_unit_idx, key=f"cost_unit_{v}",
            help="Все цены задаются в ОДНОЙ единице измерения. Стоимость "
                 "рецепта = Σ(цена_i · доля_i) и учитывается в M8 как "
                 "критерий «дешевле — лучше».")
        use_cost = not cost_unit.startswith("—") or bool(d_cost)
        cost = None
        if use_cost:
            st.caption(f"Цена за единицу количества для каждого компонента:")
            cost = [st.number_input(
                        f"{lbl}", min_value=0.0,
                        value=float(d_cost[i]) if i < len(d_cost) else 1.0,
                        step=0.1, key=f"cost_{v}_{q}_{i}")
                    for i, lbl in enumerate(names)]

    # --- Размер пробы → пересчёт долей в количество сырья ---
    st.sidebar.header("📦 Размер пробы")
    bs = st.sidebar.number_input(
        "Размер партии/пробы", min_value=0.0,
        value=float(d.get("batch_size") or 100.0), step=10.0,
        key=f"batch_{v}",
        help="Доли компонентов умножаются на этот размер и показываются как "
             "количество сырья на пробу (в таблицах планов и в рецепте).")
    batch_opts = ["г", "кг", "мг", "т", "л", "мл", "ед."]
    batch_unit_default = d.get("batch_unit")
    batch_unit_idx = (batch_opts.index(batch_unit_default)
                      if batch_unit_default in batch_opts else 0)
    bu = st.sidebar.selectbox("Единица количества", batch_opts,
                              index=batch_unit_idx, key=f"batch_unit_{v}")
    st.session_state["batch_size"] = bs
    st.session_state["batch_unit"] = bu

    cfg = PipelineConfig(name=name, q=q, model=model, noise_sd=noise,
                         seed=int(seed), n_runs_factor=factor,
                         n_restarts=n_restarts, n_blocks=int(n_blocks),
                         lower=lower, upper=upper,
                         names=names, cost_coeffs=cost,
                         cost_unit=(None if cost_unit.startswith("—") else cost_unit),
                         property_names=property_names,
                         batch_size=float(bs), batch_unit=bu,
                         comp_mode=comp_meta["comp_mode"],
                         base_index=comp_meta["base_index"],
                         parts_min=comp_meta["parts_min"],
                         parts_max=comp_meta["parts_max"])

    project_dir = os.path.join(_REPO, "project_ui", name)
    return cfg, project_dir


def get_runner(cfg, project_dir, reset=False) -> PipelineRunner:
    if reset or "runner" not in st.session_state:
        st.session_state["runner"] = PipelineRunner(cfg, project_dir)
    return st.session_state["runner"]


def _runner_started(runner: PipelineRunner) -> bool:
    """Проект уже «начат» — есть результаты стадий или загруженные данные."""
    return bool(runner.results) or runner.design is not None


def _sync_runner(cfg, project_dir) -> PipelineRunner:
    """Привязать runner к текущему конфигу сайдбара.

    Пока проект НЕ начат (нет результатов/дизайна) — пересобираем runner из
    конфига на каждом ререндере, чтобы правки q/границ/имён сразу применялись.
    После старта — фиксируем runner, но переименование проекта перенаправляет
    сохранение (semantics «save as») без потери работы.
    """
    runner = st.session_state.get("runner")
    if runner is None or not _runner_started(runner):
        runner = PipelineRunner(cfg, project_dir)
        st.session_state["runner"] = runner
    elif str(runner.project_dir) != str(project_dir):
        runner.project_dir = Path(project_dir)
        runner.cfg.name = cfg.name
        runner.state.name = cfg.name
    return runner



# ----------------------------------------------------------------------
def render_project_loader(root: str):
    """Открыть/загрузить сохранённый проект целиком (config + данные + модели)."""
    st.sidebar.header("📁 Проект")
    projs = list_projects(root)
    sel = st.sidebar.selectbox("Открыть сохранённый проект",
                               ["— новый —"] + projs, key="proj_select")
    if st.sidebar.button("📂 Загрузить проект", key="load_project") \
            and sel != "— новый —":
        try:
            runner = PipelineRunner.from_project(root, sel)
            snap = runner.config_snapshot()
            snap["name"] = sel
            st.session_state["cfg_defaults"] = snap
            st.session_state["cfg_ver"] = _ver() + 1
            st.session_state["runner"] = runner
            st.session_state["loaded_msg"] = (
                f"Проект '{sel}' загружен (стадия {runner.state.stage}).")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Не удалось загрузить '{sel}': {exc}")
    if st.session_state.get("loaded_msg"):
        st.sidebar.success(st.session_state.pop("loaded_msg"))


def render_project_saver(runner: PipelineRunner):
    """Сохранить весь проект на диск (главное сохранение, не чекпоинт стадии)."""
    if st.sidebar.button("💾 Сохранить проект", key="save_project"):
        try:
            path = runner.save_project()
            st.sidebar.success(f"Проект сохранён: {Path(path).parent.name}")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Не удалось сохранить: {exc}")



# ----------------------------------------------------------------------
def render_stage(runner: PipelineRunner, key: str, title: str):
    st.subheader(title)
    # этап уже выполнен (есть данные в сессии или восстановлены с диска)?
    _completed = set(ai.completed_stages(runner))
    is_done = key in _completed or f"{key}_fit" in _completed
    cols = st.columns([1, 2, 2])
    label = (f"♻️ Пересчитать {key} (заново)" if is_done
             else f"▶ Выполнить {key}")
    run = cols[0].button(label, key=f"run_{key}")
    if is_done:
        cols[1].caption("✓ Результаты загружены/посчитаны и показаны ниже. "
                        "Кнопка **пересчитывает этап заново** (данные с диска "
                        "при этом не нужны).")
    else:
        cols[1].caption("Этап ещё не выполнен — нажмите, чтобы рассчитать.")

    if run:

        with st.spinner(f"Выполняется {key}…"):
            try:
                if key == "M1":
                    runner.run_m1()
                elif key == "M2":
                    runner.run_m2(simulate=False)

                elif key == "M3":
                    runner.run_m3_fit(); runner.run_m3_ard()
                elif key == "M4":
                    runner.run_m4()
                elif key == "M5":
                    bar = st.progress(0.0, text="M5: подготовка…")
                    runner.run_m5(progress=lambda m, f: bar.progress(
                        min(max(f, 0.0), 1.0), text=f"M5: {m}"))
                    bar.empty()

                elif key == "M6":
                    runner.run_m6()
                elif key == "M7":
                    runner.run_m7()
                elif key == "M8":
                    runner.run_m8()
                st.success(f"{key} выполнено.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{key}: {exc}")

    _render_result(runner, key)


def _render_m2(runner: PipelineRunner, names, batch: float, unit: str):
    """M2: план эксперимента + ручной ввод откликов y + расход сырья."""
    r = runner.results["M2"]
    c = st.columns(4)
    c[0].metric("Опытов (n)", r["n"]); c[1].metric("p", r["p"])
    c[2].metric("D-eff", f"{r['d_efficiency']:.4f}")
    c[3].metric("Блоков", r.get("n_blocks", 1))

    df = _df_design(r["design"], names)
    comp_cols = list(df.columns)
    df.index = np.arange(1, len(df) + 1); df.index.name = "Опыт"
    has_blocks = r.get("n_blocks", 1) > 1 and r.get("blocks") is not None
    if has_blocks:
        df["Блок"] = np.asarray(r["blocks"])
    # отклики ВСЕХ целевых свойств (мультиотклик, §12), ПРИВЯЗАННЫЕ к плану M2:
    # по столбцу на каждое свойство из матрицы Y (n×P), а не один y[:,0].
    props = list(runner.property_names)
    P = len(props)
    Ymat = np.asarray(r.get("Y"), dtype=float)
    if Ymat.ndim != 2 or Ymat.shape != (len(df), P):
        # fallback: собрать из первичного y, остальное — NaN
        Ymat = np.full((len(df), P), np.nan)
        yv = np.asarray(r.get("y"), dtype=float)
        if yv.ndim == 1 and yv.shape[0] == len(df) and P >= 1:
            Ymat[:, 0] = yv
    prop_cols = []
    for i, prop in enumerate(props):
        col = f"{prop} (lab)"
        df[col] = np.round(Ymat[:, i], 4)
        prop_cols.append(col)

    st.info(
        "ℹ️ Столбцы вида **«свойство (lab)»** — измеренные целевые свойства "
        "продукта для каждого опыта (по столбцу на каждое свойство проекта; "
        f"сейчас их {P}: {', '.join(props)}). Первое свойство — первичное "
        "(для M7/M8/benchmark). Столбцы НЕ заполняются автоматически: впишите "
        "значения вручную и нажмите «💾 Сохранить отклики», либо «🧪 Заполнить "
        "тестовыми» — данные сгенерирует симулятор. Стадии M3–M8 используют "
        "именно эти значения.")
    st.caption("План эксперимента (доли компонентов; редактируются только "
               "столбцы «свойство (lab)»):")
    locked = comp_cols + (["Блок"] if has_blocks else [])
    edited = st.data_editor(df, use_container_width=True, height=320,
                            disabled=locked, key="m2_editor")
    bcols = st.columns(2)
    if bcols[0].button("💾 Сохранить отклики", key="save_y"):
        try:
            Ynew = np.column_stack(
                [np.asarray(edited[c], dtype=float) for c in prop_cols])
            runner.Y = Ynew
            runner.y = Ynew[:, 0]
            runner.state.put("responses", runner.y)
            runner.state.put("responses_multi", runner.Y)
            runner.results["M2"]["Y"] = runner.Y
            runner.results["M2"]["y"] = runner.y
            # отклики изменились → метрики M3–M8 на старых y устарели
            runner.invalidate_metrics()
            st.success(f"Отклики сохранены ({P} свойств) — выполняйте M3–M8.")

        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось сохранить отклики: {exc}")

    if bcols[1].button("🧪 Заполнить тестовыми (симулятор)", key="fill_sim"):
        try:
            runner.design = np.asarray(r["design"])
            runner.simulate_responses()
            st.success("Заполнено симулятором — можно править и выполнять M3–M8.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Не удалось заполнить: {exc}")


    sheets = {"План_доли": df}
    if batch and batch > 0:
        amt = _amounts(df, comp_cols, batch, unit)
        st.caption(f"📦 Количество сырья на ОДНУ пробу {batch:g} {unit} "
                   f"(доля × размер пробы):")
        st.dataframe(amt, use_container_width=True, height=320)
        tot = _totals(df, comp_cols, batch, unit)
        st.caption(f"⚖️ Итого сырья на ВЕСЬ план — {r['n']} опытов × "
                   f"{batch:g} {unit} (сколько всего закупить/взвесить):")
        st.dataframe(tot, use_container_width=True)
        sheets[f"Количество ({unit})" if unit else "Количество"] = amt
        sheets["Итого по плану"] = tot
    else:
        st.caption("ℹ️ Укажите «📦 Размер пробы» в сайдбаре, чтобы увидеть "
                   "количество сырья на пробу и общий вес на весь план.")
    _download(sheets, "M2_design.xlsx", "dl_m2")


def _render_m3_property(runner: PipelineRunner, names, prop: str, idx: int):
    """Анализ M3 (Scheffé + ARD) для ОДНОГО свойства — вкладка страницы M3.

    Данные уже посчитаны для всех свойств (`per_property`), поэтому вкладки
    лишь раскладывают готовые результаты. `idx` нужен для уникальных ключей
    виджетов (иначе кнопки скачивания конфликтуют между вкладками).
    """
    if "M3_fit" in runner.results:
        res = runner.results["M3_fit"]
        r = res.get("per_property", {}).get(prop, res)
        c = st.columns(3)
        c[0].metric("R²", f"{r['r2']:.4f}")
        c[1].metric("Adj-R²", f"{r['adj_r2']:.4f}")
        c[2].metric("RMSE", f"{r['rmse']:.4f}")
        st.caption(f"Коэффициенты Scheffé для «{prop}» "
                   "(интерпретация значимости термов):")
        st.dataframe(r["coef_table"], use_container_width=True)
        _download({"Коэффициенты": r["coef_table"], "ANOVA": r["anova"]},
                  f"M3_{prop}.xlsx", f"dl_m3_{idx}")
        st.caption("ANOVA:")
        st.dataframe(r["anova"], use_container_width=True)
    if "M3_ard" in runner.results:
        res = runner.results["M3_ard"]
        a = res.get("per_property", {}).get(prop, res)
        st.info(f"ARD-GP «{prop}»: q_eff = {a['q_eff']} · активные: "
                f"{', '.join(a['active'])} · logLik={a['gp_loglik']:.2f} "
                f"· noise={a['noise_level']:.4f}")
        imp = pd.DataFrame({"importance": a["importance"]},
                           index=names[: len(a["importance"])])
        st.bar_chart(imp)


def _render_m4_property(info: dict, prop: str):
    """Режимы M4 для ОДНОГО свойства: BIC-кривая + баланс кластеров."""
    c = st.columns([1, 3])
    c[0].metric("Число режимов K (BIC)", info["n_regimes"])
    c[1].caption(f"Кластеризация в пространстве свойства **{prop}**, не "
                 "рецептур. Границы режимов у разных свойств могут не "
                 "совпадать. Прообраз режима в рецептурах — через модель "
                 "(gating), не по близости точек.")
    bt = info.get("bic_table")
    if bt is not None:
        st.line_chart(bt.set_index("K")[["bic"]])
    else:
        st.caption("ℹ️ BIC-кривая не сохраняется на диск (лёгкий режим) — "
                   "видны центры/баланс режимов из кэша. Нажмите «♻️ Пересчитать "
                   "M4», чтобы построить кривую BIC заново.")

    means = np.asarray(info["means"], float).ravel()
    K = len(means)
    counts = np.asarray(info.get("counts", np.zeros(K)), int).ravel()
    weights = np.asarray(info.get("weights", np.zeros(K)), float).ravel()
    stds = np.asarray(info.get("stds", np.zeros(K)), float).ravel()
    tbl = pd.DataFrame({
        f"среднее «{prop}»": np.round(means, 4),
        "вес (доля точек)": np.round(weights, 3),
        "точек": counts,
        "σ внутри режима": np.round(stds, 4),
    }, index=[f"Режим {k + 1}" for k in range(K)])
    st.caption("Состав режимов — баланс кластеров:")
    st.dataframe(tbl, use_container_width=True)
    marg = [k + 1 for k in range(K) if counts[k] < 3]
    if marg and K > 1:
        st.warning(
            f"⚠️ Маргинальные режимы (< 3 точек): "
            f"{', '.join('Режим ' + str(m) for m in marg)}. "
            "Такой кластер может быть артефактом BIC — оцените, нужен ли он, "
            "или доберите точки в этой области (якорные точки вне кластеров).")


def _render_result(runner: PipelineRunner, key: str):
    names = runner.names
    batch, unit = _batch()



    if key == "M1" and "M1" in runner.results:
        r = runner.results["M1"]
        c = st.columns(4)
        c[0].metric("Вершин области", r["n_vertices"])
        c[1].metric("q", r["q"]); c[2].metric("p (термов)", r["p"])
        c[3].metric("Опытов в плане (M2)", r["n_runs"])
        st.info(f"ℹ️ Ниже — **экстремальные вершины области** (геометрия), "
                f"это НЕ план эксперимента. Сам план из **{r['n_runs']} опытов** "
                f"строится на стадии **M2** (D-оптимальный дизайн).")
        render_design_table(_df_design(r["vertices"], names),
                            list(names[: r["q"]]),
                            "Экстремальные вершины области:",
                            "M1_vertices.xlsx", "dl_m1", batch, unit)

    elif key == "M2" and "M2" in runner.results:
        _render_m2(runner, names, batch, unit)

    elif key == "M3":
        # мультиотклик (§12): анализ строится на ВСЕ свойства сразу; результаты
        # раскладываем по вкладкам (по свойству) внутри страницы M3 — удобнее,
        # чем переключать одно свойство селектором.
        props = list(runner.property_names)
        if not (("M3_fit" in runner.results) or ("M3_ard" in runner.results)):
            st.info("Выполните M3 — анализ Scheffé и ARD-скрининг построятся "
                    "сразу по всем свойствам и разложатся по вкладкам.")
        elif len(props) == 1:
            _render_m3_property(runner, names, props[0], 0)
        else:
            for i, tab in enumerate(st.tabs([f"📊 {p}" for p in props])):
                with tab:
                    _render_m3_property(runner, names, props[i], i)


    elif key == "M4" and "M4" in runner.results:
        # мультиотклик (§12): режимы оцениваются для КАЖДОГО свойства (границы
        # режимов могут не совпадать) — раскладываем по вкладкам, как M3/M6.
        r = runner.results["M4"]
        per = r.get("per_property", {})
        if not per:                       # старый одно-откликовый результат
            per = {r.get("property", runner.property_names[0]): r}
        props = [p for p in runner.property_names if p in per] or list(per)
        if len(props) == 1:
            _render_m4_property(per[props[0]], props[0])
        else:
            for i, tab in enumerate(st.tabs([f"🧩 {p}" for p in props])):
                with tab:
                    _render_m4_property(per[props[i]], props[i])



    elif key == "M5" and "M5" in runner.results:
        r = runner.results["M5"]
        design5 = r.get("design")
        n_runs5 = r.get("n_runs")
        if n_runs5 is None and design5 is not None:
            n_runs5 = len(design5)
        n_runs5 = int(n_runs5 or 0)
        existing_n = int(r.get("existing_n", 0))
        c = st.columns(3)
        if r.get("i_optimal") is not None:
            c[0].metric("I-score (M2 + добор)", f"{r['i_optimal']:.4f}")
        if r.get("i_of_d_design") is not None and r.get("i_optimal") is not None:
            # ниже — лучше: добор снижает среднюю дисперсию прогноза
            c[1].metric("I-score (только база M2)", f"{r['i_of_d_design']:.4f}",
                        delta=f"{r['i_optimal'] - r['i_of_d_design']:+.4f}",
                        delta_color="inverse")
        c[2].metric("Новых опытов (добор)", n_runs5)

        # --- диагностика объёма/остановки базового I-дизайна (§5.5) ---
        q_full = r.get("q_full"); q_eff = r.get("q_eff")
        p_quad = r.get("p_quad"); n_total = r.get("n_total")
        stop_reason = r.get("stop_reason"); cond = r.get("cond_number")
        if q_eff is not None and p_quad is not None:
            d = st.columns(4)
            reduced = bool(r.get("reduced"))
            d[0].metric("q_eff / q",
                        f"{q_eff} / {q_full}" if q_full else f"{q_eff}",
                        help="M5 строит модель на активных компонентах (q_eff "
                             "после M3-ARD), а не на полном q (§5.5.1).")
            d[1].metric("p_quad (на q_eff)", p_quad)
            if n_total is not None:
                d[2].metric("n итог / p_quad",
                            f"{n_total} / {p_quad}",
                            help="Достаточность: n ≥ p_quad + запас (§5.5.3).")
            _stop_ru = {"sufficiency": "достаточно n", "rel_gain": "выигрыш мал",
                        "budget": "бюджет N_max", "pool_exhausted": "пул исчерпан"}
            d[3].metric("Стоп-критерий", _stop_ru.get(stop_reason, stop_reason or "—"),
                        help="Почему добор остановлен (§5.5.3): любое из — "
                             "достаточность n / затухание ΔI/I / бюджет.")
            if isinstance(cond, (int, float)) and np.isfinite(cond):
                msg = (f"🔢 cond(XᵀX) итогового плана ≈ {cond:,.0f}"
                       if cond < 1e8 else
                       f"⚠️ cond(XᵀX) ≈ {cond:,.0f} — высокая обусловленность "
                       f"(возможна коллинеарность/вырожденность, §5.5.2)")
                st.caption(msg)
            if reduced and r.get("active"):
                st.caption(f"Активные компоненты (q_eff): {', '.join(r['active'])} "
                           f"— термы Шеффе по остальным исключены (heredity).")

        if n_runs5 == 0:
            # достаточность уже выполнена базой M2 → добор не нужен (§5.5.4)
            st.success(
                f"✅ Добор НЕ требуется: базы M2 ({existing_n} опытов) уже "
                f"достаточно для квадратичной модели на q_eff "
                f"(n ≥ p_quad+запас = {r.get('min_total')}). Это и есть здоровое "
                f"поведение M5 — «фундамент», а не гонка I в пол (§5.5.4). "
                f"Прицельную точность под цели добирает M7/раунды веток.")
        elif existing_n:
            st.info(
                f"ℹ️ Это **I-оптимальный ДОБОР**: {n_runs5} **новых** точек к "
                f"уже имеющимся {existing_n} опытам плана M2. Это новые рецептуры "
                f"(отличные от измеренных) — в таблице они идут со **сквозными "
                f"номерами {existing_n + 1}…{existing_n + n_runs5}**, продолжая "
                f"нумерацию M2. Добор останавливается по затуханию выигрыша / "
                f"достаточности / бюджету (§5.5.3) — без гонки I в пол. Точки пока "
                f"**только предложены**: в общую базу проекта не дописаны.")
        else:
            st.info(
                "ℹ️ M2 ещё не выполнен, поэтому построен самостоятельный "
                "I-оптимальный план «с нуля». Точки только предложены (в базу "
                "не дописаны).")

        if design5 is not None and n_runs5 > 0:
            render_design_table(_df_design(np.asarray(design5), names),
                                list(names[: runner.q]),
                                f"I-оптимальный добор — {n_runs5} новых опытов "
                                f"(точнее прогноз):",
                                "M5_design.xlsx", "dl_m5", batch, unit, height=300,
                                with_totals=True, start_index=existing_n + 1)
        elif design5 is None and n_runs5 > 0:
            st.caption("ℹ️ Координаты плана не сохранены в этом проекте (старый "
                       "формат). Нажмите «♻️ Пересчитать M5», чтобы построить "
                       "I-оптимальный добор заново.")





    elif key == "M6" and "M6" in runner.results:
        # мультиотклик (§12): общий на проект суррогат на каждое свойство.
        res = runner.results["M6"]
        props = list(runner.property_names)
        prop = props[0]
        if len(props) > 1:
            prop = st.selectbox("Свойство (суррогат)", props, key="m6_prop")
        r = res.get("per_property", {}).get(prop, res)
        c = st.columns(3)
        c[0].metric("Экспертов K", r["n_regimes"])
        c[1].metric("Test RMSE", f"{r['test_rmse']:.4f}")
        c[2].metric("σ within / between",
                    f"{r['within']:.3f} / {r['between']:.3f}")
        if len(props) > 1 and res.get("per_property"):
            summary = pd.DataFrame([
                {"свойство": n, "K": v["n_regimes"],
                 "Test RMSE": round(v["test_rmse"], 4),
                 "σ within": round(v["within"], 3),
                 "σ between": round(v["between"], 3)}
                for n, v in res["per_property"].items()
            ]).set_index("свойство")
            st.caption("Суррогаты по всем свойствам (общая модель проекта):")
            st.dataframe(summary, use_container_width=True)

    elif key == "M7" and "M7" in runner.results:
        r = runner.results["M7"]
        c = st.columns(3)
        c[0].metric("Точек: старт→финал", f"{r['n_start']}→{r['n_final']}")
        c[1].metric("Лучшее y", f"{r['y_best']:.4f}")
        c[2].metric("Ранний стоп", "да" if r["stopped_early"] else "нет")
        if r["historyB"]:
            hb = pd.DataFrame(r["historyB"])
            if "best_y" in hb:
                st.line_chart(hb.set_index("iter")[["best_y"]])
        st.caption(f"Лучший рецепт (AL): {np.round(r['x_best'], 4).tolist()}")

    elif key == "M8" and "M8" in runner.results:
        r = runner.results["M8"]
        st.metric("d_overall", f"{r['d_overall']:.4f}")
        rec = np.asarray(r["recipe"])
        rec_df = pd.DataFrame([np.round(rec, 4)], columns=list(names[: len(rec)]))
        render_design_table(rec_df, list(names[: len(rec)]),
                            "Оптимальный рецепт (доли компонентов):",
                            "M8_recipe.xlsx", "dl_m8", batch, unit)
        st.bar_chart(pd.DataFrame({"доля": np.round(rec, 4)},
                                  index=names[: len(rec)]))
        st.caption(f"Свойства в оптимуме: { {k: round(v,3) for k,v in r['properties'].items()} }")
        st.caption(f"d по критериям: { {k: round(v,3) for k,v in r['d_individual'].items()} }")


# ----------------------------------------------------------------------
def render_checkpoints(runner: PipelineRunner, cfg, project_dir):
    st.sidebar.header("💾 Чекпоинты")
    cps = runner.checkpoints()
    st.sidebar.write("Доступно:", cps or "—")
    label = st.sidebar.selectbox("Чекпоинт", cps or ["—"])
    if st.sidebar.button("↩ Загрузить состояние") and cps and label != "—":
        try:
            ps = ProjectState.restore(project_dir, label)
            _restore_runner(cfg, project_dir, ps)
            st.sidebar.success(f"Загружен '{label}' (стадия {ps.stage}).")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Не удалось: {exc}")


def _restore_runner(cfg, project_dir, ps: ProjectState):
    """Пересобрать runner из восстановленного ProjectState (design/y/MoE)."""
    runner = PipelineRunner(cfg, project_dir)
    runner.state = ps
    design = ps.get("design"); y = ps.get("responses")
    if design is not None:
        runner.design = np.asarray(design)
    if y is not None:
        runner.y = np.asarray(y)
    for mkey in ("m7_final_moe", "m6_moe"):
        if mkey in ps.models:
            try:
                runner.moe = MixtureOfExperts.from_state(ps.models[mkey])
                break
            except Exception:  # noqa: BLE001
                pass
    st.session_state["runner"] = runner


def render_benchmark(runner: PipelineRunner):
    st.subheader("🎯 Benchmark: pipeline vs известный оптимум полигона")
    if st.button("▶ Сравнить с истиной"):
        b = runner.benchmark()
        c = st.columns(3)
        c[0].metric("y* (истина)", f"{b['y_true']:.4f}")
        if "y_pipeline_true" in b:
            c[1].metric("y(pipeline)", f"{b['y_pipeline_true']:.4f}",
                        delta=f"{-b['value_gap_pct']:.1f}%")
            c[2].metric("‖Δ рецепта‖", f"{b['recipe_dist']:.4f}")
            cmp = pd.DataFrame({
                "истина": np.round(b["x_true"], 4),
                "pipeline": np.round(b["x_pipeline"], 4),
            }, index=runner.names[: len(b["x_true"])])
            st.bar_chart(cmp)
        else:
            st.info("Сначала выполните M8, чтобы получить рецепт pipeline.")


def render_branches(runner: PipelineRunner):
    """🌿 Менеджер веток (3d, REBUILD_SPEC §5/§12).

    Ветка = цель (desirability по свойству) + бюджет слотов, БЕЗ своей модели.
    Все ветки читают общие суррогаты проекта и пишут измеренные точки в ОДНУ
    общую базу (origin-теги). Портфельный раунд делит слоты арбитром бюджета.
    """
    st.subheader("🌿 Ветки: общая модель проекта + ветко-специфичный сбор точек")
    st.caption("Ветка — это ЦЕЛЬ (desirability по свойству) и БЮДЖЕТ слотов, без "
               "собственной модели. Все ветки используют общие суррогаты проекта "
               "и дописывают измеренные точки в одну общую базу (origin-теги). "
               "Портфельный раунд распределяет слоты между активными ветками.")

    props = list(runner.property_names)
    ready = (runner.design is not None and runner.Y is not None
             and not np.any(np.isnan(np.asarray(runner.Y, dtype=float))))
    if not ready:
        st.info("Сначала выполните M2 и заполните отклики (вручную или "
                "«🧪 Заполнить тестовыми (симулятор)»).")
        return

    # --- создание ветки -------------------------------------------------
    st.markdown("**➕ Новая ветка**")
    c = st.columns([3, 2, 2, 2])
    name = c[0].text_input("Название", value=f"Ветка {len(runner.branches) + 1}",
                           key="branch_name")
    prop = c[1].selectbox("Целевое свойство", props, key="branch_prop")
    kind = c[2].selectbox("Тип цели", ["max", "min", "target"], key="branch_kind")
    budget = c[3].number_input("Бюджет (слотов)", min_value=1, max_value=100,
                               value=6, step=1, key="branch_budget")
    if st.button("➕ Добавить ветку", key="add_branch"):
        col = np.asarray(runner.Y, float)[:, props.index(prop)]
        lo, hi = float(np.min(col)), float(np.max(col))
        if hi <= lo:
            hi = lo + 1.0
        tgt = (lo + hi) / 2.0 if kind == "target" else None
        spec = DesirabilitySpec(kind, low=lo, high=hi, target=tgt)
        try:
            runner.add_branch(name, {prop: spec}, budget=int(budget))
            st.success(f"Ветка «{name}» добавлена (цель: {kind} «{prop}»).")
        except (KeyError, ValueError) as exc:
            st.error(f"Не удалось добавить ветку: {exc}")

    # --- диагностика misspecification (FinalCheckList Блок 7, §12) -------
    # (всегда доступна после M2 — это диагностика ОБЩЕЙ модели проекта, а не ветки)
    with st.expander("🔬 Диагностика модели (misspecification, §12)"):
        st.caption("Замечает, когда общей модели проекта перестаёт хватать: "
                   "точки «вне всех режимов» (малые gₖ), экстраполяция (novelty) "
                   "и триггер переразбиения K+1 (BIC-оптимальное число режимов ≠ "
                   "текущему). Read-only — состояние проекта не меняется.")
        if st.button("🔬 Проверить модель проекта", key="diagnose_base"):
            try:
                rep = runner.diagnose_base()
                drows = []
                for name, info in rep["per_property"].items():
                    drows.append({
                        "свойство": name,
                        "вне режимов, %": round(100 * info["frac_out_of_regime"], 1),
                        "экстраполяция, %": round(100 * info["frac_extrapolation"], 1),
                        "K текущее": info["current_K"],
                        "K по BIC": info["recommended_K"],
                        "переразбить?": "⚠️ да" if info["needs_recluster"] else "ок",
                    })
                st.dataframe(pd.DataFrame(drows).set_index("свойство"),
                             use_container_width=True)
                flags = [n for n, i in rep["per_property"].items()
                         if i["needs_recluster"]]
                if flags:
                    st.warning("Рекомендуется переразбиение (K+1) для свойств: "
                               + ", ".join(flags) + " — добавьте точки/перестройте M6.")
                else:
                    st.success("Число режимов адекватно текущим данным.")
            except RuntimeError as exc:
                st.error(str(exc))
        stagn = [b.name for b in runner.branches.values() if b.is_stagnating()]
        if stagn:
            st.warning("Застрявшие ветки (d_best не растёт): " + ", ".join(stagn)
                       + ". Стоит сменить цель/зону или закрыть ветку.")

    # --- портфельный раунд (арбитр бюджета) -----------------------------
    st.markdown("**▶ Портфельный раунд** — арбитр делит слоты между активными "
                "ветками (дальше от цели → больше слотов).")
    c2 = st.columns([1, 1, 2])
    slots = c2[0].number_input("Слотов на раунд", min_value=1, max_value=50,
                               value=4, step=1, key="portfolio_slots")
    explore = c2[1].slider("Доля exploration", 0.0, 1.0, 0.3, 0.05,
                           key="portfolio_explore")
    if c2[2].button("▶ Прогнать раунд (арбитраж бюджета)", key="run_portfolio"):
        with st.spinner("Сбор точек по веткам…"):
            try:
                out = runner.run_portfolio_round(int(slots),
                                                 explore_frac=float(explore))
                alloc = out["allocation"]
                if alloc:
                    st.success("Слоты распределены: " + ", ".join(
                        f"«{runner.branches[k].name}»={v}"
                        for k, v in alloc.items()))
                else:
                    st.warning("Нет активных веток с бюджетом — слоты не назначены.")
                oc = pd.DataFrame(
                    {"точек": runner.origin_counts()}).rename_axis("origin")
                st.caption("Происхождение точек общей базы:")
                st.dataframe(oc, use_container_width=True)
            except RuntimeError as exc:
                st.error(str(exc))

    # --- §15.6: экономический стоп + граница-сигнал + flat-ось (A0.7) ----
    # ЧИСТО READ-ONLY (A0.6): система НИКОГДА не двигает границы молча — это
    # ПРЕДЛОЖЕНИЯ с цифрами, решение за пользователем.
    with st.expander("💰 §15.6 — стоп/деньги/границы (предложения, read-only)"):
        st.caption("Двойной стоп (технический И экономический), денежная триада "
                   "при упоре в soft-границу и A0.7: на ВЫРОЖДЕННОЙ (flat) оси "
                   "репортится objective-gap (Δd за границей), а не «двигать x». "
                   "Ничего не меняется автоматически — это предложения (A0.6).")
        if not runner.branches:
            st.info("Заведите ветку, чтобы оценить экономику остановки/границ.")
        else:
            from src.optimize.economic_stop import (decide_stop, money_triad,
                                                     STOP_CEIL, STOP_STAGNATION,
                                                     STOP_NOT_ECONOMICAL)
            bnames = {b.name: bid for bid, b in runner.branches.items()}
            ce = st.columns([3, 2, 2])
            bsel = ce[0].selectbox("Ветка", list(bnames), key="econ_branch")
            comp = ce[1].selectbox("Компонент (ось)", list(runner.names),
                                   key="econ_comp")
            new_bound = ce[2].number_input("Граница за пределом (доля)",
                                           min_value=0.0, max_value=1.0,
                                           value=1.0, step=0.05, key="econ_bound")
            ce2 = st.columns([1, 1, 1, 1])
            volume = ce2[0].number_input("V (изд/период)", min_value=0.0,
                                         value=1000.0, step=100.0, key="econ_V")
            cexp = ce2[1].number_input("c_exp (₽/опыт)", min_value=0.0,
                                       value=50.0, step=10.0, key="econ_cexp")
            horizon = ce2[2].number_input("H (период)", min_value=0.0,
                                          value=6.0, step=1.0, key="econ_H")
            dprice = ce2[3].number_input("Δprice_изд за границей (₽/изд)",
                                         min_value=0.0, value=0.2, step=0.05,
                                         key="econ_dprice")
            _render_econ_eval = st.button(
                "💰 Оценить экономику остановки и границы", key="econ_eval")
            if _render_econ_eval:
                bid = bnames[bsel]
                br = runner.branches[bid]
                try:
                    # 1) flat-ось (A0.7): objective-gap vs x-gap
                    flat = runner.flat_axis_mixture(bid, comp, "upper",
                                                    float(new_bound))
                    if flat.flat:
                        st.warning(
                            f"Ось **{comp}** ВЫРОЖДЕНА (flat, неидентифицируема): "
                            f"spread={flat.spread:.2e}. Двигать долю бессмысленно "
                            f"(A0.7) → репортим **objective-gap** = "
                            f"{flat.objective_gap:+.4f} Δd, x-gap игнор.")
                    else:
                        st.info(
                            f"Ось **{comp}** различима: spread={flat.spread:.2e}, "
                            f"x-gap={flat.x_gap:+.4f} (двигать долю осмысленно), "
                            f"objective-gap={flat.objective_gap:+.4f} Δd.")

                    # 2) денежная триада §6 (mixture-only runner → soft, A0.5)
                    t = money_triad(comp, "upper",
                                    delta_price_item=float(dprice),
                                    volume=float(volume),
                                    n_experiments=max(1, br.remaining()),
                                    cost_exp=float(cexp), horizon=float(horizon))
                    st.markdown(
                        f"**Денежная триада (soft-граница {comp}):**\n"
                        f"- экономия: **{t.saving_per_period:.1f} ₽/период** "
                        f"(= Δprice·V)\n"
                        f"- цена добычи: **{t.acquisition_cost:.1f} ₽** "
                        f"(= N·c_exp)\n"
                        f"- окупаемость: **{t.payback_periods:.2f} периодов** "
                        f"(сравнить с H={horizon:g})")
                    if t.worth_it:
                        st.success("Окупается в горизонте → двигать стоит "
                                   "(решение за вами, A0.6).")
                    else:
                        st.warning("НЕ окупается в горизонте → двигать "
                                   "невыгодно (решение за вами, A0.6).")

                    # 3) двойной стоп §4 — причина остановки
                    delta_d = 0.0
                    if len(br.history) >= 2:
                        delta_d = float(br.history[-1].get("d_best", br.d_best)) \
                            - float(br.history[-2].get("d_best", 0.0))
                    econ_val = float(dprice) * float(volume) * float(horizon)
                    reason = decide_stop(delta_d=delta_d, d_best=br.d_best,
                                         ceil=br.satisfy_at,
                                         economic_value=econ_val,
                                         cost_exp=float(cexp))
                    labels = {STOP_CEIL: "🎯 потолок достигнут (ceil_reached)",
                              STOP_STAGNATION: "🛑 прогресс встал (stagnation)",
                              STOP_NOT_ECONOMICAL: "💸 невыгодно (not_economical)",
                              None: "▶ продолжать (есть куда и выгодно)"}
                    st.caption(f"Двойной стоп §4: **{labels[reason]}** "
                               f"(Δd={delta_d:+.4f}, d_best={br.d_best:.3f}, "
                               f"ceil={br.satisfy_at:.3f}, "
                               f"econ_value={econ_val:.1f} ₽ vs c_exp={cexp:g}).")
                except (KeyError, ValueError, RuntimeError) as exc:
                    st.error(str(exc))




    st.session_state["runner"] = runner


# ----------------------------------------------------------------------
def render_assistant(runner: PipelineRunner):
    """💬 Встроенный ИИ-ассистент: видит контекст страниц + мост в trace.

    Ассистент читает живое состояние `runner` из session_state (метрики всех
    стадий, ветки, benchmark) и отвечает через OpenRouter. Тот же снимок
    пишется в trace, чтобы Cline в VS Code наблюдал данные через MCP
    `doe-introspect`.
    """
    st.subheader("💬 Ассистент: интерпретация результатов и подсказки по шагам")

    # --- настройки подключения (ключ/модель вводятся прямо здесь) -------
    with st.expander("⚙️ Подключение: API-ключ и модель",
                     expanded=not ai.llm_available()):
        st.caption("Ключ OpenRouter (sk-or-…) и имя модели. Кнопка «💾 Сохранить» "
                   "пишет ОБА значения (ключ + модель) в локальный файл `.env` в "
                   "корне репозитория — он занесён в `.gitignore`, поэтому НЕ "
                   "попадёт на GitHub и подхватится автоматически при следующих "
                   "запусках. Без сохранения они живут только в текущей сессии.")
        key_in = st.text_input("OpenRouter API key", type="password",
                               value=os.environ.get("OPENROUTER_API_KEY", ""),
                               key="ai_key")
        model_in = st.text_input("Модель", value=ai.model_name(), key="ai_model")
        if key_in:
            os.environ["OPENROUTER_API_KEY"] = key_in.strip()
        if model_in:
            os.environ["DOE_ASSISTANT_MODEL"] = model_in.strip()

        ckey = st.columns([2, 3])
        if ckey[0].button("💾 Сохранить ключ и модель в .env", key="ai_save_key"):
            try:
                path = ai.save_api_key(key_in, model=model_in)
                st.success(f"Ключ и модель (`{model_in}`) сохранены в `{path}` "
                           "(файл в .gitignore — на GitHub не уйдёт). Подхватятся "
                           "при следующем запуске.")
            except ValueError as exc:
                st.error(str(exc))
            except OSError as exc:  # noqa: BLE001
                st.error(f"Не удалось записать .env: {exc}")
        if ai.api_key_persisted():
            ckey[1].caption(f"🔐 Сохранённые ключ и модель найдены: "
                            f"`{ai.env_file_path()}`")
        else:
            ckey[1].caption("Ключ и модель пока не сохранены в .env.")



    # --- статус backend и мост в trace ---------------------------------
    cstat = st.columns([2, 2, 3])
    if ai.llm_available():
        cstat[0].success("LLM подключён")
    else:
        cstat[0].warning("Нет OPENROUTER_API_KEY")
    cstat[1].caption(f"Модель: `{ai.model_name()}`")
    if cstat[2].button("🔄 Опубликовать снапшот для Cline (VS Code)",
                       key="ai_snapshot", use_container_width=True):
        try:
            info = ai.write_live_snapshot(
                runner, chat_history=st.session_state.get("ai_history", []))
            st.session_state["ai_snapshot_msg"] = (
                f"Снапшот записан в trace: run_id=`{info['run_id']}` "
                f"(сообщений в чате: {info.get('n_messages', 0)}; каталог: "
                f"{info['root']}). Cline видит его через MCP `doe-introspect` "
                f"(run_overview / get_stage `assistant_chat`).")

        except Exception as exc:  # noqa: BLE001
            st.session_state["ai_snapshot_msg"] = f"Не удалось записать снапшот: {exc}"
    if st.session_state.get("ai_snapshot_msg"):
        st.info(st.session_state.pop("ai_snapshot_msg"))

    with st.expander("👁️ Что сейчас «видит» ассистент (контекст страниц)"):
        st.json(ai.build_context(runner))

    if not ai.llm_available():
        st.caption("Чтобы включить чат, задайте переменную окружения "
                   "`OPENROUTER_API_KEY` перед запуском приложения. Кнопка выше "
                   "и так публикует контекст в trace — это работает без ключа.")

    # --- история диалога ----------------------------------------------
    history = st.session_state.setdefault("ai_history", [])
    cc = st.columns([1, 5])
    if cc[0].button("🗑️ Очистить", key="ai_clear"):
        st.session_state["ai_history"] = []
        st.rerun()

    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Спросите про результаты, метрики или следующий шаг…",
                           key="ai_input", disabled=not ai.llm_available())
    if prompt:
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Думаю над контекстом…"):
                try:
                    reply = ai.assistant_reply(runner, history[:-1], prompt)
                except Exception as exc:  # noqa: BLE001
                    reply = f"⚠️ Ошибка обращения к модели: {exc}"
                st.markdown(reply)
        history.append({"role": "assistant", "content": reply})
        st.session_state["ai_history"] = history
        # мост: фиксируем для Cline полный диалог (включая последний ответ),
        # чтобы переписку было видно через MCP get_stage `assistant_chat`.
        try:
            ai.write_live_snapshot(runner, chat_history=history)
        except Exception:  # noqa: BLE001 — мост не должен ломать чат
            pass



# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="DOE Pipeline M1–M8", layout="wide")
    # Подхватываем сохранённый ключ/модель из локального .env (если есть).
    # Внешние переменные окружения имеют приоритет (override=False).
    ai.load_env_file()
    st.title("🧪 MoE/GP Pipeline для Mixture DOE — M1…M8")

    st.caption("Пошаговый конвейер пересборки (REBUILD_SPEC). «Лаборатория» — "
               "синтетический полигон Scheffé; на каждой стадии сохраняется чекпоинт.")

    root = os.path.join(_REPO, "project_ui")
    render_project_loader(root)            # 📁 открыть сохранённый проект
    cfg, project_dir = sidebar_config()
    if st.sidebar.button("🔧 Создать / сбросить проект"):
        get_runner(cfg, project_dir, reset=True)
        st.sidebar.success("Проект инициализирован.")

    runner = _sync_runner(cfg, project_dir)
    render_project_saver(runner)           # 💾 сохранить проект целиком
    render_checkpoints(runner, cfg, project_dir)

    # индикатор пройденных стадий — по долговечному состоянию проекта
    # (учитывает загруженные проекты и кэш метрик, не только in-memory results)
    _completed = set(ai.completed_stages(runner))
    done = [k for k, _ in STAGES
            if k in _completed or f"{k}_fit" in _completed]


    st.progress(len(done) / len(STAGES),
                text=f"Пройдено стадий: {len(done)}/{len(STAGES)}  "
                     f"(текущая: {runner.state.stage})")

    tabs = st.tabs([t for _, t in STAGES]
                   + ["🌿 Ветки", "🎯 Benchmark", "💬 Ассистент"])
    for tab, (k, title) in zip(tabs, STAGES):
        with tab:
            render_stage(runner, k, title)
    with tabs[-3]:
        render_branches(runner)
    with tabs[-2]:
        render_benchmark(runner)
    with tabs[-1]:
        render_assistant(runner)


if __name__ == "__main__":
    main()
