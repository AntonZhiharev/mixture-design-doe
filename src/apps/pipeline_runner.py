"""Оркестратор pipeline M1–M8 для UI (REBUILD_SPEC §8/§9).

Чистая логика без зависимости от Streamlit — управляет стадиями M1…M8 поверх
готовых модулей `src/`, ведёт `ProjectState` и сохраняет чекпоинты `after_M*`.
Используется Streamlit-приложением (`streamlit_app.py`) и юнит-тестами.

«Лаборатория» эмулируется синтетическим полигоном (`SyntheticScheffe`, §5): это
позволяет UI прогонять весь конвейер без реального эксперимента и сравнивать
найденный рецепт с известным оптимумом (benchmark).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence



import numpy as np

from src.core.simplex import SimplexRegion
from src.core.synthetic import SyntheticScheffe, MultiSyntheticScheffe

from src.core.state import ProjectState
from src.core.linalg import (scheffe_term_indices, scheffe_matrix,
                             scheffe_active_terms)
from src.design.d_optimal import (build_candidate_pool, d_optimal_design,
                                  d_optimal_for_region)
from src.design.i_optimal import (region_moment_matrix, i_optimal_design,
                                   i_optimal_augment,
                                   i_optimal_augment_sequential)


from src.models.scheffe import ScheffeModel
from src.models.screening import ARDScreening
from src.models.clustering import GMMRegimes
from src.models.moe import MixtureOfExperts
from src.models.diagnostics import diagnose, needs_recluster
from src.design.active_learning import active_learning_loop
from src.design.branches import (Branch, branch_scores, propose_by_score,
                                  allocate_budget)
from src.optimize.desirability import (DesirabilitySpec, Desirability,
                                       optimize_desirability)


# ----------------------------------------------------------------------
@dataclass
class PipelineConfig:
    name: str = "ui_project"
    q: int = 5
    lower: Optional[Sequence[float]] = None
    upper: Optional[Sequence[float]] = None
    names: Optional[Sequence[str]] = None
    model: str = "quadratic"
    noise_sd: float = 0.2
    seed: int = 42
    n_runs_factor: float = 2.0          # n_runs = ceil(factor * p)
    n_random: int = 600                 # candidate pool size
    n_restarts: int = 8
    n_blocks: int = 1                   # число блоков (партий/дней) для плана
    cost_coeffs: Optional[Sequence[float]] = None   # per-component unit cost
    cost_unit: Optional[str] = None                 # единица цены (одна на проект)
    property_names: Optional[Sequence[str]] = None  # имена целевых свойств (P)
    batch_size: Optional[float] = None              # размер пробы (для пересчёта)
    batch_unit: Optional[str] = None                # единица количества пробы
    # исходная форма ввода ограничений состава (чтобы отображать как задал user)
    comp_mode: Optional[str] = None                 # "fractions" | "parts"
    base_index: Optional[int] = None                # база (=100 частей) в режиме parts
    parts_min: Optional[Sequence[float]] = None     # min частей по компонентам
    parts_max: Optional[Sequence[float]] = None     # max частей по компонентам
    # модель ИСТИНЫ синт.лаборатории (опц.): порядок Scheffé генератора отклика.
    # Расцеплена с `model` (та — для D-опт/Scheffé-интерпретации M3): лаборатория
    # может быть cubic (тройные термы), а конвейер аппроксимирует её quadratic —
    # реальная математика на GP/MoE, порядок Scheffé им не важен. None → = model.
    truth_model: Optional[str] = None
    # известная истина синт.лаборатории (опц.): свойство → полный вектор коэф.
    # Scheffé под (q, truth_model). Если задано — лаборатория детерминирована.
    truth_coef_by_property: Optional[Dict[str, Sequence[float]]] = None





    def region(self) -> SimplexRegion:
        if self.lower is not None or self.upper is not None:
            return SimplexRegion(lower=self.lower, upper=self.upper,
                                 names=self.names)
        return SimplexRegion(q=self.q, names=self.names)

    @classmethod
    def from_snapshot(cls, name: str, d: Dict[str, Any]) -> "PipelineConfig":
        """Rebuild a config from a saved ``state.config`` snapshot (see
        :meth:`PipelineRunner.config_snapshot`). Unknown keys fall back to
        dataclass defaults so older snapshots keep loading."""
        d = dict(d or {})
        return cls(
            name=name,
            q=int(d.get("q", 5)),
            lower=d.get("lower"),
            upper=d.get("upper"),
            names=d.get("names"),
            model=str(d.get("model", "quadratic")),
            noise_sd=float(d.get("noise_sd", 0.2)),
            seed=int(d.get("seed", 42)),
            n_runs_factor=float(d.get("n_runs_factor", 2.0)),
            n_random=int(d.get("n_random", 600)),
            n_restarts=int(d.get("n_restarts", 8)),
            n_blocks=int(d.get("n_blocks", 1)),
            cost_coeffs=d.get("cost_coeffs"),
            cost_unit=d.get("cost_unit"),
            property_names=d.get("property_names"),
            batch_size=d.get("batch_size"),
            batch_unit=d.get("batch_unit"),
            comp_mode=d.get("comp_mode"),
            base_index=d.get("base_index"),
            parts_min=d.get("parts_min"),
            parts_max=d.get("parts_max"),
            truth_model=d.get("truth_model"),
            truth_coef_by_property=d.get("truth_coef_by_property"),
        )




# ----------------------------------------------------------------------
class PipelineRunner:
    """Хранит состояние конвейера и выполняет стадии M1…M8 по одной."""

    def __init__(self, config: PipelineConfig, project_dir: str | Path):
        self.cfg = config
        self.project_dir = Path(project_dir)
        self.region = config.region()
        self.q = self.region.q
        self.names = self.region.names
        # целевые свойства (P): мультиотклик (REBUILD_SPEC §12)
        self.property_names = (list(config.property_names)
                               if config.property_names else ["y"])
        # синтетическая «лаборатория»: P независимых истин, мерит все свойства.
        # Модель ИСТИНЫ (truth_model) расцеплена с моделью конвейера (model):
        # лаборатория может быть выше порядком (cubic), чем Scheffé-интерпретация.
        truth_model = config.truth_model or config.model
        self.truth_multi = MultiSyntheticScheffe(
            self.q, self.property_names, model=truth_model,
            noise_sd=config.noise_sd, seed=config.seed,
            coef_by_property=config.truth_coef_by_property)
        # первичная истина (свойство 0) — для M6/M7/benchmark (1D, совместимость)
        self.truth = self.truth_multi.truths[0]
        self.cost_coeffs = (np.asarray(config.cost_coeffs, float)
                            if config.cost_coeffs is not None
                            else np.linspace(1.0, 3.0, self.q))
        # рабочие объекты в памяти
        self.design: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None      # отклики всех свойств (n×P)
        self.y: Optional[np.ndarray] = None      # первичное свойство (n,) = Y[:,0]
        self.blocks: Optional[np.ndarray] = None
        self.moe: Optional[MixtureOfExperts] = None
        # M6: общие на проект суррогаты — по одному MoE на свойство
        # (REBUILD_SPEC §12: общая модель проекта, мультиотклик).
        self.surrogates: Dict[str, MixtureOfExperts] = {}
        # 3c: единая база точек с origin-тегами + ветки без своих моделей (§5/§12)
        self.origin: List[str] = []
        self.branches: Dict[str, Branch] = {}


        self.results: Dict[str, Dict[str, Any]] = {}
        # кэш компактных метрик стадий, переживающий перезагрузку проекта
        # (лёгкая персистентность: скаляры/мелкие массивы, без тяжёлых таблиц)
        self.cached_metrics: Dict[str, Any] = {}
        self.p = len(scheffe_term_indices(self.q, config.model))

        self.n_runs = int(np.ceil(config.n_runs_factor * self.p))
        # полный снимок конфигурации — чтобы проект восстанавливался целиком
        self.state = ProjectState(name=config.name,
                                  config=self.config_snapshot())

    # ------------------------------------------------------------------
    # Сохранение/загрузка ПРОЕКТА целиком (не только чекпоинт стадии)
    # ------------------------------------------------------------------
    def config_snapshot(self) -> Dict[str, Any]:
        """Полный сериализуемый снимок конфигурации проекта.

        Достаточен, чтобы восстановить идентичный `PipelineConfig`
        (см. :meth:`PipelineConfig.from_snapshot`) — включая границы долей,
        имена, стоимость и параметры алгоритма.
        """
        return {
            "q": int(self.q),
            "model": self.cfg.model,
            "noise_sd": float(self.cfg.noise_sd),
            "seed": int(self.cfg.seed),
            "names": list(self.names),
            "lower": self.region.lower.tolist(),
            "upper": self.region.upper.tolist(),
            "n_runs_factor": float(self.cfg.n_runs_factor),
            "n_random": int(self.cfg.n_random),
            "n_restarts": int(self.cfg.n_restarts),
            "n_blocks": int(self.cfg.n_blocks),
            "cost_coeffs": self.cost_coeffs.tolist(),
            "cost_unit": self.cfg.cost_unit,
            "property_names": list(self.property_names),
            "batch_size": self.cfg.batch_size,
            "batch_unit": self.cfg.batch_unit,
            "comp_mode": self.cfg.comp_mode,
            "base_index": self.cfg.base_index,
            "parts_min": (list(self.cfg.parts_min)
                          if self.cfg.parts_min is not None else None),
            "parts_max": (list(self.cfg.parts_max)
                          if self.cfg.parts_max is not None else None),
            "truth_model": self.cfg.truth_model,
            "truth_coef_by_property": (
                {k: list(map(float, v))
                 for k, v in self.cfg.truth_coef_by_property.items()}
                if self.cfg.truth_coef_by_property is not None else None),
        }


    def save_project(self) -> str:
        """Сохранить весь проект (state.json + data/models) в `project_dir`.

        В отличие от чекпоинтов стадий, это «главное» сохранение проекта,
        которое потом подхватывает :meth:`from_project`.
        """
        self.state.config = self.config_snapshot()
        # лёгкий кэш метрик стадий: накапливаем (кэш с диска ∪ свежие из сессии)
        self.cached_metrics = self.stage_metrics()
        self.state.put("stage_metrics", self.cached_metrics)
        if self.design is not None:
            self.state.put("design", self.design)

        if self.y is not None:
            self.state.put("responses", self.y)
        if self.blocks is not None:
            self.state.put("blocks", self.blocks)
        if self.origin:
            self.state.put("origin", list(self.origin))
        if self.branches:
            self.state.put("branches",
                           {bid: b.to_state() for bid, b in self.branches.items()})
        path = self.state.save(self.project_dir)
        return str(path)

    def _restore_from_state(self, ps: ProjectState) -> None:
        """Подтянуть рабочие объекты (design/y/blocks/MoE) из ProjectState."""
        self.state = ps
        # лёгкий кэш метрик стадий — чтобы ассистент/MCP/индикатор видели
        # пройденные стадии и их цифры без дорогого пересчёта (см. save_project)
        cached = ps.get("stage_metrics")
        self.cached_metrics = dict(cached) if cached else {}
        design = ps.get("design")

        y = ps.get("responses")
        Y = ps.get("responses_multi")
        blocks = ps.get("blocks")
        if design is not None:
            self.design = np.asarray(design)
        if y is not None:
            self.y = np.asarray(y)
        if Y is not None:
            self.Y = np.asarray(Y)
        if blocks is not None:
            self.blocks = np.asarray(blocks)
        origin = ps.get("origin")
        if origin is not None:
            self.origin = [str(o) for o in origin]
        elif self.design is not None:
            self.origin = ["M2"] * len(self.design)
        branches = ps.get("branches")
        if branches:
            self.branches = {bid: Branch.from_state(b)
                             for bid, b in dict(branches).items()}

        # суррогаты per-property (M6, §12): m6_moe__{i} → property_names[i]
        self.surrogates = {}
        for i, name in enumerate(self.property_names):
            key = f"m6_moe__{i}"
            if key in ps.models:
                try:
                    self.surrogates[name] = MixtureOfExperts.from_state(
                        ps.models[key])
                except Exception:  # noqa: BLE001 — модель опциональна
                    pass

        for mkey in ("m7_final_moe", "m6_moe"):
            if mkey in ps.models:
                try:
                    self.moe = MixtureOfExperts.from_state(ps.models[mkey])
                    break
                except Exception:  # noqa: BLE001 — модель опциональна
                    pass
        # если первичный суррогат не восстановился из алиаса — берём из словаря
        if self.moe is None and self.property_names[0] in self.surrogates:
            self.moe = self.surrogates[self.property_names[0]]

    def rehydrate_results(self) -> None:
        """Восстановить лёгкие ``results`` завершённых стадий ПОСЛЕ загрузки —
        чтобы UI сразу показывал сохранённые данные, БЕЗ пересчёта.

        Источник — ``cached_metrics`` (с диска) + восстановленные ``design/Y``.
        Тяжёлые артефакты (таблицы Шеффе/ANOVA в M3, BIC-кривая M4) в лёгком
        варианте персистентности НЕ сохраняются и здесь не воспроизводятся:
        для них UI показывает сводку и предлагает пересчёт ради деталей.
        Идемпотентно: уже посчитанные в сессии стадии не перетирает.
        """
        cm = self.cached_metrics or {}
        primary = self.property_names[0]

        # M1 — геометрия области детерминирована, вершины считаем напрямую (дёшево)
        if "M1" in cm and "M1" not in self.results:
            try:
                self.results["M1"] = {**cm["M1"],
                                      "vertices": self.region.extreme_vertices(),
                                      "centroid": self.region.centroid()}
            except Exception:  # noqa: BLE001 — геометрия не критична для load
                pass

        # M2 — план/отклики восстановлены из state (design/Y/blocks)
        if "M2" in cm and self.design is not None and "M2" not in self.results:
            m2 = dict(cm["M2"])
            self.results["M2"] = {
                "n": int(m2.get("n", len(self.design))),
                "p": int(m2.get("p", self.p)),
                "d_efficiency": m2.get("d_efficiency"),
                "n_blocks": int(m2.get("n_blocks", 1)),
                "design": self.design, "Y": self.Y, "y": self.y,
                "blocks": self.blocks,
                "property_names": list(self.property_names)}

        # M4 — сводка режимов по свойствам (без BIC-кривой: она не персистится)
        if "M4" in cm and "M4" not in self.results:
            per = {name: {"property": name, **dict(vals)}
                   for name, vals in cm["M4"].items()}
            top = dict(per.get(primary, {}))
            self.results["M4"] = {"per_property": per,
                                  "property_names": list(self.property_names),
                                  **top}

        # M5 — I-оптимальный добор предложен; координаты + диагностика из кэша
        if "M5" in cm and "M5" not in self.results:
            m5 = dict(cm["M5"])
            d5 = m5.get("design")
            res5 = {
                "i_optimal": m5.get("i_optimal"),
                "i_of_d_design": m5.get("i_of_d_design"),
                "n_runs": m5.get("n_runs"),
                "existing_n": int(m5.get("existing_n", 0)),
                "applied": bool(m5.get("applied", False)),
                "design": (np.asarray(d5, float) if d5 is not None else None)}
            for k in ("q_full", "q_eff", "reduced", "active", "p_quad",
                      "n_total", "n_over_p", "min_total", "n_max", "rel_tol",
                      "stop_reason", "cond_number"):
                if k in m5:
                    res5[k] = m5[k]
            self.results["M5"] = res5



    @classmethod
    def from_project(cls, root: str | Path, name: str) -> "PipelineRunner":
        """Загрузить проект по имени из каталога `root` (например project_ui)."""
        project_dir = Path(root) / name
        ps = ProjectState.load(project_dir)
        cfg = PipelineConfig.from_snapshot(ps.name or name, ps.config)
        runner = cls(cfg, project_dir)
        runner._restore_from_state(ps)
        # сразу восстановить лёгкие результаты завершённых стадий для отображения
        runner.rehydrate_results()
        return runner



    # ------------------------------------------------------------------
    def cost_fn(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return X @ self.cost_coeffs

    def _ckpt(self, label: str) -> str:
        self.state.checkpoint(self.project_dir, label=label)
        return str(self.project_dir / "checkpoints" / f"{label}.json")

    # ===================== M1: геометрия ==============================
    def run_m1(self) -> Dict[str, Any]:
        V = self.region.extreme_vertices()
        centroid = self.region.centroid()
        self.state.set_stage("M1_geometry")
        self.state.put("vertices", V)
        self.state.put("centroid", centroid)
        res = {"n_vertices": int(len(V)), "vertices": V, "centroid": centroid,
               "q": self.q, "p": self.p, "n_runs": self.n_runs,
               "checkpoint": self._ckpt("after_M1")}
        self.results["M1"] = res
        return res

    # ===================== M2: D-optimal design ======================
    def run_m2(self, simulate: bool = True) -> Dict[str, Any]:
        """Построить D-оптимальный план. ``simulate``: заполнить ли отклики
        синтетической «лабораторией» сразу (True — для тестов/демо) или оставить
        пустыми (NaN) под ручной ввод/кнопку заполнения в UI."""
        res = d_optimal_for_region(self.region, n_runs=self.n_runs,
                                   model=self.cfg.model,
                                   n_random=self.cfg.n_random,
                                   n_restarts=self.cfg.n_restarts,
                                   seed=self.cfg.seed)
        self.design = res.design
        n = self.design.shape[0]
        P = len(self.property_names)
        self.Y = (self.truth_multi.evaluate(self.design) if simulate
                  else np.full((n, P), np.nan))
        self.y = self.Y[:, 0]                       # первичное свойство
        self.origin = ["M2"] * n                    # origin-тег каждой точки
        self.blocks = self._assign_blocks(n, max(1, int(self.cfg.n_blocks)))

        self.state.set_stage("M2_screening_design")
        self.state.put("design", self.design)
        self.state.put("responses", self.y)
        self.state.put("responses_multi", self.Y)
        self.state.put("blocks", self.blocks)
        self.state.put("d_efficiency", float(res.d_efficiency))
        out = {"n": int(n), "p": self.p,
               "d_efficiency": float(res.d_efficiency),
               "logdet": float(res.logdet), "design": self.design,
               "y": self.y, "Y": self.Y,
               "property_names": list(self.property_names),
               "blocks": self.blocks,
               "n_blocks": int(max(1, self.cfg.n_blocks)),
               "checkpoint": self._ckpt("after_M2")}

        self.results["M2"] = out
        return out

    @staticmethod
    def _assign_blocks(n: int, n_blocks: int) -> np.ndarray:
        """Сбалансированно распределить n опытов по блокам (партиям/дням).

        Опыты раскладываются «по кругу» (round-robin), чтобы каждый блок
        получил близкое число прогонов и покрывал разные точки дизайна.
        Возвращает массив меток блоков 1..n_blocks длиной n.
        """
        if n_blocks <= 1:
            return np.ones(n, dtype=int)
        return (np.arange(n) % n_blocks).astype(int) + 1


    # ===================== M3a: Scheffe fit ==========================
    def run_m3_fit(self) -> Dict[str, Any]:
        """Scheffe-аппроксимация на КАЖДОЕ свойство (REBUILD_SPEC §12).

        Для каждого свойства строится своя модель Шеффе; `per_property[name]`
        хранит её диагностику. Верхний уровень результата дублирует первичное
        свойство (столбец 0) — для обратной совместимости со старым UI/тестами.
        """
        cols = self._property_columns()
        self.state.set_stage("M3_screening_analysis")
        per: Dict[str, Dict[str, Any]] = {}
        primary: Optional[Dict[str, Any]] = None
        for i, name in enumerate(self.property_names):
            if name not in cols:
                continue
            fit = ScheffeModel(model=self.cfg.model, names=self.names).fit(
                self.design, cols[name])
            self.state.models[f"m3_scheffe__{i}"] = fit.to_state()
            info = {"property": name, "r2": float(fit.r2),
                    "adj_r2": float(fit.adj_r2), "rmse": float(fit.rmse),
                    "coef_table": fit.coefficient_table(), "anova": fit.anova(),
                    "term_names": fit.term_names,
                    "coefficients": fit.coefficients}
            per[name] = info
            if primary is None:
                primary = info
        # alias первичного свойства — совместимость
        if "m3_scheffe__0" in self.state.models:
            self.state.models["m3_scheffe"] = self.state.models["m3_scheffe__0"]
        out = dict(primary or {})
        out["per_property"] = per
        out["property_names"] = list(self.property_names)
        out["checkpoint"] = self._ckpt("after_M3")
        self.results["M3_fit"] = out
        return out

    # ===================== M3b: ARD-GP screening =====================
    def run_m3_ard(self) -> Dict[str, Any]:
        """ARD-GP скрининг на КАЖДОЕ свойство (REBUILD_SPEC §12).

        Активные компоненты могут отличаться от свойства к свойству;
        `per_property[name]` хранит результат скрининга для каждого.
        """
        cols = self._property_columns()
        per: Dict[str, Dict[str, Any]] = {}
        primary: Optional[Dict[str, Any]] = None
        for i, name in enumerate(self.property_names):
            if name not in cols:
                continue
            scr = ARDScreening(seed=self.cfg.seed,
                               n_restarts=self.cfg.n_restarts,
                               rel_threshold=0.15).fit(self.design, cols[name])
            self.state.models[f"m3_ard_screening__{i}"] = scr.to_state()
            active = [scr.component_names[j] for j in scr.active_indices()]
            info = {"property": name, "q_eff": int(scr.q_eff),
                    "active": active, "table": scr.table,
                    "gp_loglik": float(scr.gp_loglik),
                    "noise_level": float(scr.noise_level),
                    "importance": np.asarray(scr.importance)}
            per[name] = info
            if primary is None:
                primary = info
        if "m3_ard_screening__0" in self.state.models:
            self.state.models["m3_ard_screening"] = \
                self.state.models["m3_ard_screening__0"]
        out = dict(primary or {})
        out["per_property"] = per
        out["property_names"] = list(self.property_names)
        self.results["M3_ard"] = out
        return out

    # ===================== M4: GMM regimes ===========================
    @staticmethod
    def _regime_summary(reg, yv: np.ndarray, name: str) -> Dict[str, Any]:
        """Сводка по режимам одного свойства: means/weights/counts/stds.

        `weights` — доля точек на режим (средние responsibilities, mixing
        proportions); `counts` — точки по жёстким меткам; `stds` — разброс
        свойства внутри режима. Нужны, чтобы судить о балансе кластеров
        (см. чек-лист Блок 4/7), а не только о центрах.
        """
        K = int(reg.n_regimes)
        labels = np.asarray(reg.labels).ravel()
        resp = np.asarray(reg.responsibilities, dtype=float)
        yv = np.asarray(yv, dtype=float).ravel()
        counts = np.bincount(labels, minlength=K).astype(int)
        if resp.ndim == 2 and resp.shape[1] == K:
            weights = resp.mean(axis=0)               # мягкие доли (mixing prop.)
        else:
            weights = counts / max(len(labels), 1)
        stds = np.array([float(np.std(yv[labels == k])) if counts[k] > 0 else 0.0
                         for k in range(K)])
        return {"property": name, "n_regimes": K, "bic_table": reg.bic_table,
                "means": np.asarray(reg.means).ravel(),
                "weights": np.asarray(weights, dtype=float).ravel(),
                "counts": counts, "stds": stds}

    def run_m4(self) -> Dict[str, Any]:
        """Кластеризация режимов GMM в пространстве свойств — на КАЖДОЕ свойство.

        По канону мультиотклика (§12, как M3/M6) режимность оценивается для
        каждого свойства отдельно: границы режимов у разных свойств могут не
        совпадать. `per_property[name]` хранит сводку по свойству (число
        режимов, центры, баланс кластеров). Верхний уровень дублирует первичное
        свойство (столбец 0) — для обратной совместимости со старым UI/тестами.
        """
        cols = self._property_columns()
        self.state.set_stage("M4_clustering")
        per: Dict[str, Dict[str, Any]] = {}
        primary: Optional[Dict[str, Any]] = None
        for i, name in enumerate(self.property_names):
            if name not in cols:
                continue
            reg = GMMRegimes(k_range=range(1, 6), seed=self.cfg.seed).fit(
                cols[name])
            self.state.models[f"m4_regimes__{i}"] = reg.to_state()
            info = self._regime_summary(reg, cols[name], name)
            per[name] = info
            if primary is None:
                primary = info
        if "m4_regimes__0" in self.state.models:
            self.state.models["m4_regimes"] = self.state.models["m4_regimes__0"]
        out = dict(primary or {})
        out["per_property"] = per
        out["property_names"] = list(self.property_names)
        out["checkpoint"] = self._ckpt("after_M4")
        self.results["M4"] = out
        return out



    # ===================== M5: I-optimal design ======================
    # Критерий остановки добора (FinalCheckList §5.5.3): добор прекращается по
    # ЛЮБОМУ из условий — затухание выигрыша / достаточность n / бюджет.
    M5_REL_TOL = 0.03          # ΔI/I < ε ⇒ выигрыш на точку затух (5.5.3)
    M5_MARGIN = 12             # запас над p_quad для достаточности (5.5.3, 10–15)

    def _active_indices(self) -> tuple[list[int], int, bool]:
        """Индексы АКТИВНЫХ компонентов (q_eff) из M3-ARD (объединение по свойствам).

        Возвращает ``(indices, q_eff, reduced)``. Если M3-ARD не выполнен —
        ``(все компоненты, q, False)`` (фолбэк на полный q). Берём ОБЪЕДИНЕНИЕ
        активных по всем свойствам: компонент, важный хотя бы для одного
        свойства, остаётся в локальной модели M5 (FinalCheckList §5.5.1).
        """
        src = self.results.get("M3_ard") or (self.cached_metrics or {}).get("M3_ard")
        per = None
        if isinstance(src, dict):
            per = src.get("per_property", src)
        active_names: set = set()
        if isinstance(per, dict):
            for info in per.values():
                if isinstance(info, dict):
                    for a in info.get("active", []) or []:
                        active_names.add(a)
        idx = [i for i, nm in enumerate(self.names) if nm in active_names]
        if not idx:
            return list(range(self.q)), self.q, False
        return idx, len(idx), len(idx) < self.q

    def run_m5(self, n_add: Optional[int] = None,
               progress: Optional[Callable[[str, float], None]] = None
               ) -> Dict[str, Any]:
        """I-оптимальный ДОБОР точек к плану M2 с критерием остановки (§5.5).

        Канон FinalCheckList §5.5:
        - **Размерность (5.5.1).** Локальная квадратичная модель строится на
          ``q_eff`` АКТИВНЫХ компонентах (по M3-ARD), а не на полном ``q`` —
          число параметров ``p_quad`` считается на ``q_eff``. Точки добора —
          полноценные рецептуры (все ``q`` компонентов), но термы Шеффе берутся
          только по активным (с heredity).
        - **Координаты/обусловленность (5.5.2).** I считается через каноническую
          форму Шеффе (без интерсепта); матрица моментов — по ОГРАНИЧЕННОЙ
          области; логируется ``cond(XᵀX)`` итогового плана.
        - **Остановка (5.5.3).** Добор прекращается по ЛЮБОМУ из: относительный
          выигрыш ``ΔI/I < ε``; достаточность ``n ≥ p_quad + запас``; бюджет
          ``N_max``. Никакого абсолютного порога на I.
        - **Роль (5.5.4).** M5 = разумный фундамент (``n ≳ p + запас``), а не
          финальная точность — её добирает M7 под цели веток.

        Точки добора отличны от уже измеренных M2 (исключены из пула) и
        нумеруются сквозным образом (``existing_n + 1, …``). Это РАСЧЁТ плана:
        точки лишь предлагаются, в общую базу пока не дописываются.

        ``n_add`` — необязательный ЖЁСТКИЙ override числа добираемых точек
        (отключает авто-остановку и бюджет). ``progress(msg, frac)`` — коллбэк UI.
        """
        def _pg(msg: str, frac: float) -> None:
            if progress is not None:
                progress(msg, frac)

        existing = (np.asarray(self.design, float)
                    if self.design is not None and len(self.design) else None)
        existing_n = 0 if existing is None else int(len(existing))

        # --- 5.5.1: размерность по q_eff (после M3-ARD) ---
        active_idx, q_eff, reduced = self._active_indices()
        terms = scheffe_active_terms(self.q, active_idx, self.cfg.model)
        p_quad = len(terms)

        _pg("Пул кандидатов", 0.05)
        pool = build_candidate_pool(self.region, n_random=self.cfg.n_random,
                                    seed=self.cfg.seed)
        if existing is not None:
            pool = self._exclude_points(pool, existing)
        _pg("Матрица моментов области (аналитически)", 0.2)
        # 5.5.2: моменты на СТАНДАРТНОМ симплексе (аналитически, §13.5/§13.11) —
        # детерминированно, без MC-смещения сэмплера; на q_eff-редуцированном базисе
        W = region_moment_matrix(self.region, self.cfg.model,
                                 terms=terms, method="analytic")

        min_total = p_quad + self.M5_MARGIN
        # бюджет N_max: жёсткий потолок добора (5.5.3); override n_add отключает
        n_max = int(n_add) if n_add is not None else max(2 * p_quad, 1)
        rel_tol = 0.0 if n_add is not None else self.M5_REL_TOL
        min_total_eff = (existing_n + int(n_add)) if n_add is not None else min_total

        _pg("I-оптимальный добор с критерием остановки", 0.5)
        aug = i_optimal_augment_sequential(
            existing, pool, W, model=self.cfg.model, terms=terms,
            n_max=n_max, min_total=min_total_eff, margin=self.M5_MARGIN,
            rel_tol=rel_tol, seed=self.cfg.seed)

        _pg("Диагностика", 0.95)
        self.state.set_stage("M5_local_design")
        new_design = np.asarray(aug.new_points)
        n_total = existing_n + aug.n_added
        out = {
            "i_optimal": float(aug.i_final),       # I объединённого плана (M2+добор)
            "i_of_d_design": float(aug.i_base),    # I только базы M2
            "design": new_design,
            "n_runs": int(aug.n_added),
            "existing_n": existing_n,
            # --- диагностика §5.5 ---
            "q_full": int(self.q), "q_eff": int(q_eff), "reduced": bool(reduced),
            "active": [self.names[i] for i in active_idx],
            "p_quad": int(p_quad), "n_total": int(n_total),
            "n_over_p": float(n_total / p_quad) if p_quad else float("nan"),
            "min_total": int(min_total), "n_max": int(n_max),
            "rel_tol": float(rel_tol), "stop_reason": aug.stop_reason,
            "cond_number": float(aug.cond_number),
            "i_history": [float(v) for v in aug.i_history],
            "applied": False,
            "checkpoint": self._ckpt("after_M5")}
        self.results["M5"] = out
        _pg("Готово", 1.0)
        return out

    @staticmethod
    def _exclude_points(pool: np.ndarray, used: np.ndarray,
                        atol: float = 1e-7) -> np.ndarray:
        """Убрать из ``pool`` строки, совпадающие с любой точкой ``used``.

        Нужно для I-оптимального добора (M5): новые точки не должны повторять
        уже измеренные точки базы (M2).
        """
        pool = np.atleast_2d(np.asarray(pool, float))
        used = np.atleast_2d(np.asarray(used, float))
        keep = [p for p in pool
                if not np.any(np.all(np.isclose(used, p, atol=atol), axis=1))]
        return np.array(keep) if keep else pool[:0]




    # ===================== M6: MoE surrogate =========================
    def run_m6(self) -> Dict[str, Any]:
        """Суррогат MoE на КАЖДОЕ свойство — общая модель проекта (§12).

        Строит словарь :attr:`surrogates` ``{property → MixtureOfExperts}``;
        `self.moe` — алиас первичного свойства (столбец 0) для M7/M8/benchmark.
        `per_property[name]` хранит диагностику суррогата каждого свойства.
        """
        cols = self._property_columns()
        Xt = self.region.random_points(200, seed=self.cfg.seed + 2)
        self.state.set_stage("M6_moe")
        self.surrogates = {}
        per: Dict[str, Dict[str, Any]] = {}
        primary: Optional[Dict[str, Any]] = None
        for i, name in enumerate(self.property_names):
            if name not in cols:
                continue
            moe = MixtureOfExperts(k_range=range(1, 5), seed=self.cfg.seed,
                                   n_restarts=self.cfg.n_restarts).fit(
                self.design, cols[name])
            self.surrogates[name] = moe
            self.state.models[f"m6_moe__{i}"] = moe.to_state()
            pred = moe.predict(Xt)
            truth_i = self.truth_multi.truths[i].true(Xt)
            rmse = float(np.sqrt(np.mean((pred.mean - truth_i) ** 2)))
            info = {"property": name, "n_regimes": int(moe.n_regimes),
                    "test_rmse": rmse,
                    "within": float(pred.uncertainty.mean()),
                    "between": float(pred.disagreement.mean())}
            per[name] = info
            if primary is None:
                primary = info
        # первичный суррогат — для совместимости (M7/M8/benchmark)
        self.moe = self.surrogates.get(self.property_names[0])
        if "m6_moe__0" in self.state.models:
            self.state.models["m6_moe"] = self.state.models["m6_moe__0"]
        out = dict(primary or {})
        out["per_property"] = per
        out["property_names"] = list(self.property_names)
        out["checkpoint"] = self._ckpt("after_M6")
        self.results["M6"] = out
        return out

    # ===================== M7: active learning =======================
    def run_m7(self, n_iter_refine: int = 6, n_iter_search: int = 8
               ) -> Dict[str, Any]:
        self._require_data()
        oracle = lambda X: self.truth.evaluate(X)
        X0, y0 = self.design, self.y
        grid = self.region.random_points(500, seed=self.cfg.seed + 99)

        resA = active_learning_loop(self.region, oracle, X0, y0,
                                    n_iter=n_iter_refine, acquisition="max_std",
                                    batch=2, n_candidates=400, seed=self.cfg.seed,
                                    model_kwargs={"n_restarts": self.cfg.n_restarts})
        resB = active_learning_loop(self.region, oracle, resA.X, resA.y,
                                    n_iter=n_iter_search, acquisition="ei",
                                    batch=1, n_candidates=600, maximize=True,
                                    acq_tol=1e-4, seed=self.cfg.seed + 1,
                                    model_kwargs={"n_restarts": self.cfg.n_restarts})
        self.moe = resB.model
        self.design, self.y = resB.X, resB.y
        x_best, y_best = resB.best(maximize=True)
        self.state.set_stage("M7_active_learning")
        self.state.models["m7_final_moe"] = resB.model.to_state()
        self.state.put("design", self.design)
        self.state.put("responses", self.y)
        out = {"n_start": int(len(y0)), "n_final": int(len(resB.y)),
               "y_best": float(y_best), "x_best": np.asarray(x_best),
               "stopped_early": bool(resB.stopped_early),
               "historyA": resA.history, "historyB": resB.history,
               "checkpoint": self._ckpt("after_M7")}
        self.results["M7"] = out
        return out

    # ===================== M8: product optimisation ==================
    def run_m8(self, prop_weight: float = 1.0, cost_weight: float = 1.0
               ) -> Dict[str, Any]:
        if self.moe is None:
            self.run_m6()
        y = self.y if self.y is not None else np.array([0.0, 1.0])
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi <= lo:
            hi = lo + 1.0
        specs = {"property": DesirabilitySpec("max", low=lo, high=hi,
                                              weight=prop_weight)}
        cost_spec = DesirabilitySpec("min",
                                     low=float(self.cost_coeffs.min()),
                                     high=float(self.cost_coeffs.max()),
                                     weight=cost_weight)
        predictors = {"property": lambda X: self.moe.predict(X).mean}
        res = optimize_desirability(self.region, predictors, specs,
                                    cost_fn=self.cost_fn, cost_spec=cost_spec,
                                    cost_name="cost", n_candidates=4000,
                                    refine_iters=400, seed=self.cfg.seed)
        self.state.set_stage("M8_optimization")
        self.state.put("m8_recipe", res.x)
        self.state.put("m8_d_overall", float(res.d_overall))
        out = {"recipe": np.asarray(res.x), "d_overall": float(res.d_overall),
               "d_individual": res.d_individual, "properties": res.properties,
               "checkpoint": self._ckpt("after_M8")}
        self.results["M8"] = out
        return out

    # ===================== Benchmark vs аналитический оптимум =========
    def benchmark(self, n_scan: int = 20000) -> Dict[str, Any]:
        scan = self.region.random_points(n_scan, seed=self.cfg.seed + 7)
        truth_vals = self.truth.true(scan)
        best = int(np.argmax(truth_vals))
        x_true = scan[best]
        m8 = self.results.get("M8", {})
        x_pipe = m8.get("recipe")
        out = {"x_true": x_true, "y_true": float(truth_vals[best])}
        if x_pipe is not None:
            x_pipe = np.asarray(x_pipe)
            out["x_pipeline"] = x_pipe
            out["y_pipeline_true"] = float(self.truth.true(x_pipe.reshape(1, -1))[0])
            out["recipe_dist"] = float(np.linalg.norm(x_true - x_pipe))
            denom = abs(out["y_true"]) if abs(out["y_true"]) > 1e-9 else 1.0
            out["value_gap_pct"] = 100.0 * (out["y_true"] - out["y_pipeline_true"]) / denom
        self.results["benchmark"] = out
        return out

    # ------------------------------------------------------------------
    def checkpoints(self) -> List[str]:
        return self.state.list_checkpoints(self.project_dir)

    def _require_data(self) -> None:
        if self.design is None or self.y is None:
            raise RuntimeError("Сначала выполните M2 (нет дизайна/откликов).")
        if np.any(np.isnan(np.asarray(self.y, dtype=float))):
            raise RuntimeError("Отклики y не заполнены: впишите значения в "
                               "столбец «y (lab)» или нажмите «Заполнить "
                               "тестовыми (симулятор)» на стадии M2.")

    def _property_columns(self) -> Dict[str, np.ndarray]:
        """Столбцы откликов по свойствам ``{property → y_vec}`` (REBUILD_SPEC §12).

        Источник — общая матрица откликов проекта ``Y (n×P)``. Для старых
        одно-откликовых проектов (без ``Y``) возвращает единственный столбец
        первичного свойства из ``self.y``. Проверяет, что отклики заполнены.
        """
        self._require_data()
        if self.Y is None:
            return {self.property_names[0]: np.asarray(self.y, dtype=float)}
        Y = np.asarray(self.Y, dtype=float)
        if np.any(np.isnan(Y)):
            raise RuntimeError("Отклики свойств не заполнены: впишите значения "
                               "или нажмите «Заполнить тестовыми (симулятор)» "
                               "на стадии M2.")
        return {name: Y[:, i] for i, name in enumerate(self.property_names)
                if i < Y.shape[1]}

    def simulate_responses(self) -> np.ndarray:
        """Заполнить отклики ВСЕХ свойств синтетической «лабораторией»."""
        if self.design is None:
            raise RuntimeError("Сначала выполните M2 (нет дизайна).")
        self.Y = self.truth_multi.evaluate(self.design)
        self.y = self.Y[:, 0]
        self.state.put("responses", self.y)
        self.state.put("responses_multi", self.Y)
        if "M2" in self.results:
            self.results["M2"]["y"] = self.y
            self.results["M2"]["Y"] = self.Y
        # отклики изменились → метрики M3–M8 на старых y больше не валидны
        self.invalidate_metrics()
        return self.Y


    # ===================== Ветки (3c, REBUILD_SPEC §5/§12) ===========
    def add_branch(self, name: str, goal: Dict[str, DesirabilitySpec],
                   budget: int = 10, satisfy_at: float = 0.9,
                   branch_id: Optional[str] = None) -> Branch:
        """Завести ветку: цель (desirability по свойствам) + бюджет, без модели.

        Ветка читает общие суррогаты проекта и пишет точки в общую базу —
        собственной модели у неё нет (канон §5/§12). Цель ``goal`` —
        словарь ``{property → DesirabilitySpec}``; все свойства должны входить
        в ``property_names`` проекта.
        """
        unknown = set(goal) - set(self.property_names)
        if unknown:
            raise KeyError(f"Цель ветки ссылается на неизвестные свойства: "
                           f"{sorted(unknown)} (есть: {self.property_names}).")
        bid = branch_id or f"b{len(self.branches) + 1}"
        if bid in self.branches:
            raise ValueError(f"Ветка '{bid}' уже существует.")
        br = Branch(id=bid, name=name, goal=dict(goal),
                    budget=int(budget), satisfy_at=float(satisfy_at))
        self.branches[bid] = br
        return br

    def _ensure_origin(self) -> None:
        """Гарантировать, что origin-теги синхронны с числом точек базы."""
        n = 0 if self.design is None else len(self.design)
        if len(self.origin) != n:
            # старые проекты/после M7 — доразметить недостающие как 'M2'
            self.origin = (list(self.origin) + ["M2"] * n)[:n] if self.origin \
                else ["M2"] * n

    def _refit_surrogates(self) -> None:
        """Переобучить общие суррогаты проекта на ТЕКУЩЕЙ общей базе (§12).

        Точки всех веток лежат в одной базе ``design/Y`` — модель одна на проект,
        поэтому после добавления точек любой веткой суррогаты обновляются для
        всех. Обновляет ``surrogates``/``moe`` и их снимки в state.
        """
        cols = self._property_columns()
        self.surrogates = {}
        for i, name in enumerate(self.property_names):
            if name not in cols:
                continue
            moe = MixtureOfExperts(k_range=range(1, 5), seed=self.cfg.seed,
                                   n_restarts=self.cfg.n_restarts).fit(
                self.design, cols[name])
            self.surrogates[name] = moe
            self.state.models[f"m6_moe__{i}"] = moe.to_state()
        self.moe = self.surrogates.get(self.property_names[0])
        if "m6_moe__0" in self.state.models:
            self.state.models["m6_moe"] = self.state.models["m6_moe__0"]

    def run_branch_round(self, branch_id: str, n_points: int = 1,
                         explore_frac: float = 0.3, n_candidates: int = 600,
                         refit: bool = True) -> Dict[str, Any]:
        """Один раунд активного сбора точек ВЕТКИ (M7 для ветки, §12).

        Шаги: (1) суррогаты проекта (обучить при отсутствии); (2) acquisition
        ветки по её desirability-цели + exploration; (3) выбрать ≤``n_points``
        точек в рамках бюджета; (4) ИЗМЕРИТЬ все P свойств; (5) дописать в общую
        базу с origin-тегом ``branch:{id}``; (6) дообучить общие суррогаты.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        self._require_data()
        if not self.surrogates:
            self.run_m6()
        self._ensure_origin()
        br = self.branches[branch_id]
        n_take = min(int(n_points), br.remaining())
        if n_take <= 0:
            br.refresh_status()
            return {"branch": branch_id, "added": 0, "status": br.status,
                    "remaining": br.remaining(), "d_best": br.d_best,
                    "x_best": br.x_best, "note": "бюджет исчерпан"}

        seed = self.cfg.seed + 1000 + br.spent
        cands = self.region.random_points(n_candidates, seed=seed)
        acq, d_pred, sigma = branch_scores(self.surrogates, br.goal, cands,
                                           explore_frac=explore_frac)
        newX = propose_by_score(cands, acq, n_take, min_dist=0.03)

        # ИЗМЕРЕНИЕ ВСЕХ P СВОЙСТВ (новая точка меряется целиком, §12)
        Ynew = self.truth_multi.evaluate(newX)
        self.design = np.vstack([self.design, newX])
        self.Y = np.vstack([self.Y, Ynew])
        self.y = self.Y[:, 0]
        self.origin += [f"branch:{branch_id}"] * len(newX)
        br.spent += len(newX)

        # лучший по ИЗМЕРЕННОЙ desirability цели ветки
        desir = Desirability(dict(br.goal))
        meas = {name: Ynew[:, self.property_names.index(name)] for name in br.goal}
        d_meas = desir.overall(meas)
        bi = int(np.argmax(d_meas))
        if float(d_meas[bi]) > br.d_best:
            br.d_best = float(d_meas[bi])
            br.x_best = newX[bi].tolist()
        br.refresh_status()
        br.history.append({"round": len(br.history) + 1, "added": int(len(newX)),
                           "d_round": float(np.max(d_meas)), "d_best": br.d_best,
                           "spent": br.spent, "status": br.status})

        if refit:
            self._refit_surrogates()

        # обновить общую базу в state
        self.state.put("design", self.design)
        self.state.put("responses", self.y)
        self.state.put("responses_multi", self.Y)
        self.state.put("origin", list(self.origin))
        return {"branch": branch_id, "added": int(len(newX)),
                "x_new": np.asarray(newX), "y_new": np.asarray(Ynew),
                "status": br.status, "remaining": br.remaining(),
                "d_best": br.d_best, "x_best": br.x_best,
                "stagnating": bool(br.is_stagnating()),
                "n_base": int(len(self.design))}

    def run_portfolio_round(self, total_slots: int, explore_frac: float = 0.3,
                            n_candidates: int = 600) -> Dict[str, Any]:
        """Портфельный раунд: арбитр делит бюджет между ветками и гоняет их (3d).

        Слоты распределяются :func:`allocate_budget` (дальше от цели → больше
        слотов; satisfied/exhausted пропускаются), затем для каждой ветки
        выполняется :meth:`run_branch_round`. Возвращает распределение, итоги
        раундов и обновлённую сводку origin.
        """
        for b in self.branches.values():
            b.refresh_status()
        alloc = allocate_budget(self.branches, total_slots)
        rounds: Dict[str, Any] = {}
        for bid, n in alloc.items():
            rounds[bid] = self.run_branch_round(
                bid, n_points=n, explore_frac=explore_frac,
                n_candidates=n_candidates)
        return {"allocation": alloc, "rounds": rounds,
                "n_base": int(0 if self.design is None else len(self.design)),
                "origin_counts": self.origin_counts(),
                "statuses": {bid: b.status for bid, b in self.branches.items()}}

    def origin_counts(self) -> Dict[str, int]:
        """Сводка происхождения точек общей базы ``{origin → count}``."""
        self._ensure_origin()
        out: Dict[str, int] = {}
        for o in self.origin:
            out[o] = out.get(o, 0) + 1
        return out


    # ===================== §15.6 A0.7 — flat-ось (objective-gap) ======
    def flat_axis_mixture(self, branch_id: str, comp: str, side: str,
                          new_bound: float, *, n_samples: int = 21,
                          tol: float = 1e-9):
        """A0.7-детектор для mixture-компонента ``comp`` цели ветки (objective-gap).

        Строит ``objective_fn(t) -> d_overall`` ТЕКУЩЕЙ постановки ветки: варьирует
        ТОЛЬКО долю ``comp`` (остальные перенормируются пропорционально, Σx=1),
        фиксируя прочее у M8-оптимума ветки, и считает desirability через РЕАЛЬНУЮ
        :class:`Desirability` по ОБЩИМ суррогатам проекта. ``spread ≤ tol`` ⇒ ось
        вырождена (неидентифицируема) ⇒ репортится ``objective_gap`` (Δd за
        границей), ``x_gap = None`` (двигать долю нечего, A0.7). Это ДИАГНОСТИКА
        (read-only — состояние проекта не меняется, A0.6).
        """
        from src.optimize.economic_stop import detect_flat_axis
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        if not self.surrogates:
            self.run_m6()
        br = self.branches[branch_id]
        names = list(self.names)
        if comp not in names:
            raise KeyError(f"'{comp}' не компонент состава ({names}).")
        axis = names.index(comp)
        lo = float(self.region.lower[axis])
        hi = float(self.region.upper[axis])

        # опорный рецепт: M8-argmax цели ветки (а не «лучшая измеренная»)
        x_opt = self._branch_xbest(branch_id)
        desir = Desirability(dict(br.goal))

        def _set_comp(t):
            x = np.asarray(x_opt, float).copy()
            t = float(np.clip(t, 0.0, 1.0))
            others = [k for k in range(self.q) if k != axis]
            rest = float(x[others].sum())
            x[axis] = t
            if rest > 1e-12:
                x[others] = x[others] * ((1.0 - t) / rest)
            elif others:
                x[others] = (1.0 - t) / len(others)
            return x

        def objective_fn(ts):
            ts = np.atleast_1d(np.asarray(ts, float))
            X = np.vstack([_set_comp(t) for t in ts])
            props = {n: self.surrogates[n].predict(X).mean for n in br.goal}
            return desir.overall(props)

        samples = np.linspace(lo, hi, int(n_samples))
        border_value = hi if side == "upper" else lo
        return detect_flat_axis(comp, objective_fn, samples,
                                border_value=float(border_value),
                                beyond_value=float(new_bound), tol=float(tol))

    def _branch_xbest(self, branch_id: str) -> np.ndarray:
        """M8-argmax рецепт цели ветки по общим суррогатам (опора для A0.7)."""
        br = self.branches[branch_id]
        if br.x_best is not None:
            return np.asarray(br.x_best, float).ravel()
        predictors = {n: (lambda X, n=n: self.surrogates[n].predict(X).mean)
                      for n in br.goal}
        res = optimize_desirability(self.region, predictors, dict(br.goal),
                                    n_candidates=2000, refine_iters=200,
                                    seed=self.cfg.seed)
        return np.asarray(res.x, float).ravel()



    # ===================== Кэш метрик стадий (лёгкая персистентность) ==
    # Порядок зависимостей стадий: всё правее зависит от откликов/данных.
    _STAGE_ORDER = ["M1", "M2", "M3_fit", "M3_ard", "M4", "M5", "M6", "M7",
                    "M8", "benchmark"]
    # Стадии, НЕ зависящие от откликов (только геометрия области и базовый план).
    # M5 теперь ЗАВИСИТ от откликов: его размерность берётся из q_eff (M3-ARD),
    # а M3-ARD строится на y (FinalCheckList §5.5.1) — поэтому при смене откликов
    # M5 инвалидируется вместе с M3–M8.
    _DATA_INDEPENDENT = ("M1", "M2")


    def stage_metrics_compact(self) -> Dict[str, Any]:
        """Компактные метрики пройденных стадий ИЗ ``self.results``.

        Только скаляры и мелкие массивы — без тяжёлых таблиц (coef_table/anova/
        bic_table) и матриц дизайна. Используется и ассистентом, и кэшем на диск.
        """
        r = self.results or {}
        out: Dict[str, Any] = {}
        if "M1" in r:
            out["M1"] = {k: r["M1"].get(k)
                         for k in ("n_vertices", "q", "p", "n_runs")}
        if "M2" in r:
            m2 = r["M2"]
            out["M2"] = {k: m2.get(k)
                         for k in ("n", "p", "d_efficiency", "n_blocks")}
            # координаты точек плана (доли компонентов) — план небольшой
            if m2.get("design") is not None:
                out["M2"]["design"] = np.round(
                    np.asarray(m2["design"], float), 4).tolist()
            out["M2"]["component_names"] = list(self.names)
            # разбиение опытов по блокам (партиям/дням): {блок → число опытов}
            if m2.get("blocks") is not None:
                bl = np.asarray(m2["blocks"]).astype(int).ravel()
                uniq, cnt = np.unique(bl, return_counts=True)
                out["M2"]["block_sizes"] = {int(u): int(c)
                                            for u, c in zip(uniq, cnt)}
                out["M2"]["blocks"] = bl.tolist()

        if "M3_fit" in r:
            per = r["M3_fit"].get("per_property", {})
            out["M3_fit"] = {n: {"r2": v.get("r2"), "adj_r2": v.get("adj_r2"),
                                 "rmse": v.get("rmse")}
                             for n, v in per.items()}
        if "M3_ard" in r:
            per = r["M3_ard"].get("per_property", {})
            out["M3_ard"] = {n: {"q_eff": v.get("q_eff"),
                                 "active": v.get("active")}
                             for n, v in per.items()}
        if "M4" in r:
            per = r["M4"].get("per_property", {})
            if per:
                out["M4"] = {n: {"n_regimes": v.get("n_regimes"),
                                 "means": np.asarray(v.get("means")).tolist(),
                                 "weights": np.asarray(v.get("weights")).tolist(),
                                 "counts": np.asarray(v.get("counts")).tolist(),
                                 "stds": np.asarray(v.get("stds")).tolist()}
                             for n, v in per.items()}
        if "M5" in r:
            m5 = r["M5"]
            out["M5"] = {"i_optimal": m5.get("i_optimal"),
                         "i_of_d_design": m5.get("i_of_d_design"),
                         "n_runs": m5.get("n_runs"),
                         "existing_n": int(m5.get("existing_n", 0)),
                         "applied": m5.get("applied", False),
                         # диагностика §5.5 (объём/остановка базового I-дизайна)
                         "q_full": m5.get("q_full"), "q_eff": m5.get("q_eff"),
                         "reduced": m5.get("reduced"), "active": m5.get("active"),
                         "p_quad": m5.get("p_quad"), "n_total": m5.get("n_total"),
                         "n_over_p": m5.get("n_over_p"),
                         "min_total": m5.get("min_total"),
                         "n_max": m5.get("n_max"), "rel_tol": m5.get("rel_tol"),
                         "stop_reason": m5.get("stop_reason"),
                         "cond_number": m5.get("cond_number")}

            # координаты предлагаемого I-оптимального плана (доли компонентов)

            if m5.get("design") is not None:
                d5 = np.asarray(m5["design"], float)
                out["M5"]["design"] = np.round(d5, 4).tolist()
                out["M5"]["component_names"] = list(self.names)
                # как этот план разложился бы по блокам проекта (round-robin):
                # сам план пока «не применён» (точки не измерены), но разбиение
                # показывает, как его поставить партиями при сборе откликов
                nb = max(1, int(self.cfg.n_blocks))
                if nb > 1:
                    bl5 = self._assign_blocks(len(d5), nb)
                    uniq, cnt = np.unique(bl5, return_counts=True)
                    out["M5"]["block_sizes"] = {int(u): int(c)
                                                for u, c in zip(uniq, cnt)}
                    out["M5"]["blocks"] = bl5.tolist()

        if "M6" in r:
            per = r["M6"].get("per_property", {})
            out["M6"] = {n: {"n_regimes": v.get("n_regimes"),
                             "test_rmse": v.get("test_rmse"),
                             "within": v.get("within"),
                             "between": v.get("between")}
                         for n, v in per.items()}
        if "M7" in r:
            out["M7"] = {k: r["M7"].get(k)
                         for k in ("n_start", "n_final", "y_best",
                                   "stopped_early")}
        if "M8" in r:
            out["M8"] = {"d_overall": r["M8"].get("d_overall"),
                         "recipe": np.asarray(r["M8"].get("recipe")).tolist()
                         if r["M8"].get("recipe") is not None else None,
                         "properties": r["M8"].get("properties"),
                         "d_individual": r["M8"].get("d_individual")}
        if "benchmark" in r:
            b = r["benchmark"]
            out["benchmark"] = {k: b.get(k) for k in
                                ("y_true", "y_pipeline_true", "recipe_dist",
                                 "value_gap_pct")}
        return out

    def stage_metrics(self) -> Dict[str, Any]:
        """Метрики стадий: кэш с диска, перекрытый свежими из текущей сессии.

        Свежие результаты (``stage_metrics_compact``) приоритетнее кэша — если
        стадию пересчитали в этой сессии, отдаём новые числа.
        """
        out = dict(getattr(self, "cached_metrics", {}) or {})
        out.update(self.stage_metrics_compact())
        return out

    def invalidate_metrics(self, *, keep: Sequence[str] = _DATA_INDEPENDENT
                           ) -> None:
        """Сбросить метрики/результаты стадий, ЗАВИСЯЩИХ от откликов.

        Вызывается при изменении откликов (ручной ввод/симуляция): прежние M3–M8
        построены на старых ``y`` и больше не валидны. Геометрия/план (M1/M2/M5)
        от откликов не зависят и сохраняются. ``keep`` — стадии, которые остаются.
        """
        keep = set(keep)
        self.cached_metrics = {k: v for k, v in
                               (self.cached_metrics or {}).items() if k in keep}
        self.results = {k: v for k, v in self.results.items() if k in keep}


    # ===================== Misspecification (FinalCheckList Блок 7) ===
    def diagnose_base(self, X: Optional[np.ndarray] = None, tau: float = 0.6,
                      novelty_factor: float = 3.0) -> Dict[str, Any]:
        """Диагностика misspecification общей модели проекта (§12, Блок 7).

        Для КАЖДОГО свойства считает по набору точек ``X`` (по умолчанию — вся
        общая база) долю точек «вне всех режимов» (малые gₖ) и экстраполяции
        (novelty), а также сравнивает текущее число режимов с BIC-оптимальным
        на текущих откликах (триггер переразбиения K+1).

        Возвращает ``{"per_property": {name: {...summary, current_K,
        recommended_K, needs_recluster}}, "n_query", "tau"}``. Не меняет
        состояние проекта (read-only).
        """
        self._require_data()
        if not self.surrogates:
            self.run_m6()
        ref = np.asarray(self.design, float)
        Xq = ref if X is None else np.atleast_2d(np.asarray(X, float))
        cols = self._property_columns()
        per: Dict[str, Any] = {}
        for name, moe in self.surrogates.items():
            rep = diagnose(moe, Xq, ref=ref, tau=tau,
                           novelty_factor=novelty_factor)
            need, rec_k = needs_recluster(moe, cols[name], seed=self.cfg.seed)
            info = dict(rep.summary)
            info.update({"current_K": int(getattr(moe, "K_", 1)),
                         "recommended_K": int(rec_k),
                         "needs_recluster": bool(need)})
            per[name] = info
        return {"per_property": per, "n_query": int(len(Xq)),
                "tau": float(tau), "novelty_factor": float(novelty_factor)}



# ----------------------------------------------------------------------
def list_projects(root: str | Path) -> List[str]:
    """Имена сохранённых проектов в каталоге `root` (где есть state.json)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and (p / "state.json").exists())


def delete_project(root: str | Path, name: str) -> bool:
    """Удалить сохранённый проект (каталог ``root/name``) целиком.

    Деструктивная операция — в UI закрыта подтверждением + паролём
    (см. :mod:`src.apps.admin`). Здесь — только защита от ошибок:

    - ``name`` непустой и не содержит разделителей пути/``..`` (анти-traversal);
    - целевой каталог обязан лежать ВНУТРИ ``root`` и быть валидным проектом
      (наличие ``state.json``), иначе ``ValueError`` — чтобы случайно не снести
      постороннюю папку.

    Возвращает ``True`` при успешном удалении, ``False`` — если проекта нет.
    """
    name = (name or "").strip()
    if not name or name in (".", "..") or any(s in name for s in ("/", "\\")):
        raise ValueError(f"Недопустимое имя проекта: {name!r}")
    root = Path(root).resolve()
    target = (root / name).resolve()
    # target должен быть прямым потомком root (анти-traversal)
    if target.parent != root:
        raise ValueError(f"Проект вне каталога проектов: {target}")
    if not target.exists():
        return False
    if not (target / "state.json").exists():
        raise ValueError(f"'{name}' не похож на проект (нет state.json) — "
                         f"удаление отклонено.")
    shutil.rmtree(target)
    return True



