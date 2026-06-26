"""
apps/mixture_process_runner.py — runner ветвящегося поиска над mixture×process
(REBUILD_SPEC §5/§12/§13.8, forward-путь).

Канон (§5/§12): ОДНА модель физики на проект — словарь общих суррогатов
``surrogates: property → GPExpert`` на СОСТАВНЫХ координатах ``[x..., z_code...]``.
Ветка (:class:`Branch`) — контейнер намерения без своей модели; все ветки читают
общий словарь суррогатов и дописывают измеренные точки в ОДНУ общую базу с
origin-тегами. Новая точка меряется по ВСЕМ P свойствам (оракул).

Поэтапное раскрытие переменных («маска свободы», §13.8/§14): на ранней фазе
acquisition варьирует лишь часть составных координат (например, 2 из 3 mixture),
остальные держатся на baseline. По мере раскрытия фаз свобода растёт; общая база
точек переживает фазы (ранние точки имеют константные «закрытые» столбцы — GP это
терпит). Это отделяет ИНТЕРПРЕТАЦИЮ/скрининг (Шеффе) от непараметрического GP,
которому кубические термы не нужны.

Runner ORACLE-AGNOSTIC: оракул — любой объект с ``property_names`` и
``evaluate(Xc)->(n,P)`` (синтетическая истина в тестах или реальная лаборатория).
Персистентность намеренно опущена (вне области боевого бенчмарка).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..core.schema import MIXTURE, PROCESS, DataPoint, ProjectSchema
from ..models.gp_expert import GPExpert
from ..design.branches import (Branch, branch_scores, propose_by_score,
                               allocate_budget)
from ..optimize.desirability import Desirability, DesirabilitySpec


class MixtureProcessRunner:
    """Ветвящийся активный поиск над составной областью (симплекс × куб)."""

    def __init__(self, schema: ProjectSchema, oracle: Any, *,
                 baseline: Optional[Sequence[float]] = None,
                 seed: int = 0, n_restarts: int = 4,
                 gp_mean_model: str = "quadratic", gp_kernel: str = "matern52"):
        self.schema = schema
        self.oracle = oracle
        self.property_names: List[str] = list(oracle.property_names)
        self.prop_index = {n: i for i, n in enumerate(self.property_names)}
        self.q = int(schema.n_mixture)
        self.d = int(schema.n_process)
        self.dim = self.q + self.d
        self.seed = int(seed)
        self.n_restarts = int(n_restarts)
        self.gp_mean_model = gp_mean_model
        self.gp_kernel = gp_kernel

        self._mix_block = schema.mixture_block()
        self._mix_region = (self._mix_block.as_simplex_region()
                            if self._mix_block is not None else None)

        # baseline составных координат для «закрытых» (несвободных) переменных
        if baseline is not None:
            self.baseline = np.asarray(baseline, float).ravel()
            if self.baseline.size != self.dim:
                raise ValueError(f"baseline длины {self.baseline.size}, "
                                 f"ожидалось {self.dim}.")
        else:
            mix = (np.full(self.q, 1.0 / self.q) if self.q else np.empty(0))
            proc = (np.full(self.d, 0.5) if self.d else np.empty(0))
            self.baseline = np.concatenate([mix, proc])

        # маска свободы: по умолчанию ВСЁ свободно (полная область)
        self._mix_free = np.ones(self.q, dtype=bool)
        self._proc_free = np.ones(self.d, dtype=bool)

        # общая база + общая модель + ветки
        self.X: Optional[np.ndarray] = None         # составные координаты (n×dim)
        self.Y: Optional[np.ndarray] = None         # отклики всех свойств (n×P)
        self.origin: List[str] = []
        self.surrogates: Dict[str, GPExpert] = {}
        self.branches: Dict[str, Branch] = {}

        # §15.1.2: ВЕДУЩАЯ база — список DataPoint с версионированной схемой;
        # numpy X/Y/origin — ПРОИЗВОДНЫЕ (пересобираются из points для GP).
        # baseline «закрытых» фазой координат пишется в X точки РЕАЛЬНЫМ значением
        # (не маской) — иначе select_fixed_rows не увидит, например, T=0.5.
        self.points: List[DataPoint] = []
        self.schema_history: List[ProjectSchema] = [schema]
        self.current_schema_version: int = int(schema.version)

    # ------------------------------------------------------------------
    # Фазы раскрытия (маска свободы)
    # ------------------------------------------------------------------
    def set_free(self, mixture_free: Optional[Sequence] = None,
                 process_free: Optional[Sequence] = None) -> "MixtureProcessRunner":
        """Задать СВОБОДНЫЕ переменные фазы (имена или индексы); остальные — на
        baseline. ``None`` — оставить блок как есть (все его переменные свободны
        по умолчанию)."""
        if mixture_free is not None:
            self._mix_free = self._mask(mixture_free, self._mix_block, self.q)
        if process_free is not None:
            pb = self.schema.process_block()
            self._proc_free = self._mask(process_free, pb, self.d)
        return self

    @staticmethod
    def _mask(free, block, size: int) -> np.ndarray:
        mask = np.zeros(size, dtype=bool)
        names = list(block.names) if block is not None else []
        for item in free:
            idx = names.index(item) if isinstance(item, str) else int(item)
            if 0 <= idx < size:
                mask[idx] = True
        return mask

    def free_dims(self) -> np.ndarray:
        """Булева маска свободных СОСТАВНЫХ координат (длиной dim)."""
        return np.concatenate([self._mix_free, self._proc_free])

    # ------------------------------------------------------------------
    # Генерация кандидатов под маской свободы (Σx=1 сохраняется)
    # ------------------------------------------------------------------
    def _masked_candidates(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(n)
        out = np.empty((n, self.dim), float)

        if self.q > 0:
            base_mix = self.baseline[:self.q]
            free = self._mix_free
            held = ~free
            held_sum = float(base_mix[held].sum()) if held.any() else 0.0
            c = np.tile(base_mix, (n, 1))
            if free.any():
                samp = np.atleast_2d(self._mix_region.random_points(n, seed=seed))
                fs = samp[:, free].sum(axis=1, keepdims=True)
                fs = np.where(fs > 1e-12, fs, 1.0)
                c[:, free] = samp[:, free] * (1.0 - held_sum) / fs
            out[:, :self.q] = c

        if self.d > 0:
            base_proc = self.baseline[self.q:]
            z = np.tile(base_proc, (n, 1))
            pf = self._proc_free
            if pf.any():
                zr = rng.uniform(0.0, 1.0, size=(n, self.d))
                z[:, pf] = zr[:, pf]
            out[:, self.q:] = z

        return out

    # ------------------------------------------------------------------
    # Ведущая база точек (DataPoint) ⇄ производные numpy-кэши (§15.1.2)
    # ------------------------------------------------------------------
    def _make_point(self, coords: np.ndarray, y_row: np.ndarray,
                    origin: str) -> DataPoint:
        """Составная строка ``[x..., z_code...]`` + отклики → ``DataPoint``.

        baseline «закрытых» координат уже лежит в ``coords`` реальным значением
        (см. :meth:`_masked_candidates`), поэтому пишется в ``X`` как есть —
        §15.1.2 (baseline-as-value, не маска).
        """
        coords = np.asarray(coords, float).ravel()
        X: Dict[str, List[float]] = {}
        if self.q > 0:
            X[MIXTURE] = [float(v) for v in coords[:self.q]]
        if self.d > 0:
            X[PROCESS] = [float(v) for v in coords[self.q:self.q + self.d]]
        Y = {name: float(y_row[i]) for i, name in enumerate(self.property_names)}
        tag = {"origin": origin, "schema_version": self.current_schema_version}
        return DataPoint(schema_version=self.current_schema_version,
                         X=X, Y=Y, origin_tag=tag)

    def _rebuild_arrays(self) -> None:
        """Пересобрать numpy-кэши (X/Y/origin) из ведущей базы ``points``."""
        if not self.points:
            self.X = None
            self.Y = None
            self.origin = []
            return
        rows: List[List[float]] = []
        ys: List[List[float]] = []
        for p in self.points:
            rows.append(list(p.X.get(MIXTURE, [])) + list(p.X.get(PROCESS, [])))
            ys.append([float(p.Y[name]) for name in self.property_names])
        self.X = np.asarray(rows, float)
        self.Y = np.asarray(ys, float)
        self.origin = [p.origin_tag.get("origin", "seed") for p in self.points]

    # ------------------------------------------------------------------
    # Общая модель проекта (GP на каждое свойство, составные координаты)
    # ------------------------------------------------------------------
    def fit_surrogates(self) -> None:
        if self.X is None or len(self.X) == 0:
            raise RuntimeError("Нет данных: сначала seed_initial().")
        self.surrogates = {}
        for i, name in enumerate(self.property_names):
            gp = GPExpert(mean_model=self.gp_mean_model, kernel=self.gp_kernel,
                          seed=self.seed, n_restarts=self.n_restarts)
            self.surrogates[name] = gp.fit(self.X, self.Y[:, i])

    def seed_initial(self, n: int = 12, seed: Optional[int] = None
                     ) -> Dict[str, Any]:
        """Стартовый набор точек ПОД ТЕКУЩЕЙ маской свободы + измерение + GP."""
        s = self.seed if seed is None else int(seed)
        X0 = self._masked_candidates(n, s)
        Y0 = np.atleast_2d(self.oracle.evaluate(X0))
        self.points = [self._make_point(X0[i], Y0[i], "seed")
                       for i in range(len(X0))]
        self._rebuild_arrays()
        self.fit_surrogates()
        return {"n": int(len(X0)), "P": int(self.Y.shape[1])}

    # ------------------------------------------------------------------
    # Ветки (контейнеры намерения; модель — общая)
    # ------------------------------------------------------------------
    def add_branch(self, name: str, goal: Mapping[str, DesirabilitySpec],
                   budget: int = 10, satisfy_at: float = 0.9,
                   branch_id: Optional[str] = None) -> Branch:
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

    def run_branch_round(self, branch_id: str, n_points: int = 2,
                         explore_frac: float = 0.3, n_candidates: int = 600
                         ) -> Dict[str, Any]:
        """Раунд активного сбора точек ветки на текущей фазе (масштаб §12)."""
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        if not self.surrogates:
            self.fit_surrogates()
        br = self.branches[branch_id]
        n_take = min(int(n_points), br.remaining())
        if n_take <= 0:
            br.refresh_status()
            return {"branch": branch_id, "added": 0, "status": br.status,
                    "d_best": br.d_best, "x_best": br.x_best}

        seed = self.seed + 1000 + br.spent
        cands = self._masked_candidates(n_candidates, seed)
        acq, d_pred, sigma = branch_scores(self.surrogates, br.goal, cands,
                                           explore_frac=explore_frac)
        newX = propose_by_score(cands, acq, n_take, min_dist=0.02)

        Ynew = np.atleast_2d(self.oracle.evaluate(newX))
        for i in range(len(newX)):
            self.points.append(
                self._make_point(newX[i], Ynew[i], f"branch:{branch_id}"))
        self._rebuild_arrays()
        br.spent += len(newX)

        desir = Desirability(dict(br.goal))
        meas = {name: Ynew[:, self.prop_index[name]] for name in br.goal}
        d_meas = np.asarray(desir.overall(meas), float).ravel()
        bi = int(np.argmax(d_meas))
        if float(d_meas[bi]) > br.d_best:
            br.d_best = float(d_meas[bi])
            br.x_best = newX[bi].tolist()
        br.refresh_status()
        br.history.append({"round": len(br.history) + 1, "added": int(len(newX)),
                           "d_round": float(np.max(d_meas)), "d_best": br.d_best,
                           "spent": br.spent, "status": br.status})

        self.fit_surrogates()
        return {"branch": branch_id, "added": int(len(newX)),
                "x_new": newX, "y_new": Ynew, "d_best": br.d_best,
                "x_best": br.x_best, "status": br.status,
                "n_base": int(len(self.X))}

    def run_portfolio_round(self, total_slots: int, explore_frac: float = 0.3,
                            n_candidates: int = 600) -> Dict[str, Any]:
        """Портфельный раунд: арбитр делит бюджет между активными ветками."""
        for b in self.branches.values():
            b.refresh_status()
        alloc = allocate_budget(self.branches, total_slots)
        rounds: Dict[str, Any] = {}
        for bid, n in alloc.items():
            rounds[bid] = self.run_branch_round(
                bid, n_points=n, explore_frac=explore_frac,
                n_candidates=n_candidates)
        return {"allocation": alloc, "rounds": rounds,
                "n_base": int(0 if self.X is None else len(self.X))}

    def origin_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for o in self.origin:
            out[o] = out.get(o, 0) + 1
        return out
