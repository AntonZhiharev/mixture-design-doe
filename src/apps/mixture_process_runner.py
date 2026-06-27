"""
apps/mixture_process_runner.py — runner ветвящегося поиска над mixture×process
(REBUILD_SPEC §5/§12/§13.8 + §14/§15, forward-путь).

Канон (§5/§12): ОДНА модель физики на проект — словарь общих суррогатов
``surrogates: property → GPExpert`` на СОСТАВНЫХ координатах ``[x..., z_code...]``.
Ветка (:class:`Branch`) — контейнер намерения без своей модели; все ветки читают
общий словарь суррогатов и дописывают измеренные точки в ОДНУ общую базу с
origin-тегами. Новая точка меряется по ВСЕМ P свойствам (оракул).

ПОЭТАПНОЕ РАСКРЫТИЕ ПЕРЕМЕННЫХ (§15.1.1) — ДВА РАЗНЫХ механизма, БЕЗ маски:

  * **+process (T, P)** — APPEND В СХЕМУ (:meth:`augment_phase_schema`):
    process-переменная попадает в process-блок текущей схемы, ``schema_version``
    инкрементится, политика миграции старых точек = ``known-constant(baseline)``
    (точки фазы k-1 мерились при baseline этого параметра — это полноценные
    данные, не MISSING). Старые точки переиспользуются как fixed.
  * **+mixture (C)** — APPEND В СХЕМУ (:meth:`augment_phase_mixture`, §15.0.4):
    C — полноценный новый компонент симплекса (Σ переопределяется
    ``A+B=1 → A+B+C=1``), ``schema_version`` инкрементится. Старые
    2-компонентные точки мигрируют на ГРАНЬ ``known-constant(0.0)`` (а НЕ на
    внутренний baseline — отличие от process). Это ОТМЕНЯЕТ прежний
    region-expansion для C (Предусловие 2 снято в §15.0.4).
  * **атомарно «схема+схема»** (:meth:`augment_phase_atomic`, §15.1.5): один
    target несёт И append-process, И append-mixture; ОДИН bump, обе причины в
    ``change_log`` (``append_param`` + ``append_mixture``).


Старого mask+baseline пути (`set_free`/`_masked_candidates`) больше НЕТ: свобода
фазы кодируется САМОЙ схемой (членство в process-блоке) и её bounds (mixture).
Закрытый параметр не «маскируется» — его просто нет в текущей схеме; при измерении
полная физическая координата достраивается baseline'ом для оракула, а точка
хранит координаты ТЕКУЩЕЙ схемы. Общий GP всегда видит точки, мигрированные к
текущей схеме (§13.7 ``select_fixed_rows``: baseline уходит в ``known-constant``).

M8-argmax (§15.1.4): ``x_best`` ветки — argmax desirability по суррогату
(``optimize_desirability`` — мультистарт + локальное уточнение, со стоимостным
членом), а НЕ «лучшая измеренная точка». В каждом branch-раунде последняя из
``n_points`` точек — это exploit-предложение M8-argmax (внутри бюджета раунда),
остальные — acquisition; так argmax давит рецепт к границе/вершине.

Runner ORACLE-AGNOSTIC: оракул — любой объект с ``property_names`` и
``evaluate(Xc)->(n,P)`` (синтетическая истина в тестах или реальная лаборатория),
где ``Xc`` — ПОЛНЫЙ физический составной вектор ``[x..., z_code_full...]``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..core.schema import (MIXTURE, PROCESS, DataPoint, ProjectSchema,
                           VariableBlock, composite_matrix)
from ..core.simplex import SimplexRegion

from ..core.schema_evolution import (SchemaHistory, evolve_schema,
                                     known_constant,
                                     migrate_point, point_in_region)
from ..design.move_bounds import (move_bounds, handle_dropped_fixed,
                                  RegionMove, RegionMoveError,
                                  POLICY_EXCLUDE, POLICY_ERROR, POLICY_BOUNDARY)
from ..models.gp_expert import GPExpert
from ..design.block_model import build_model_terms, model_matrix
from ..design.augmented import build_moments
from ..design.branches import (Branch, branch_scores, propose_by_score,
                               allocate_budget)
from ..optimize.desirability import (Desirability, DesirabilitySpec,
                                     DesirabilityResult, optimize_desirability)


class MixtureProcessRunner:
    """Ветвящийся активный поиск над составной областью (симплекс × куб).

    Конструируется с ПОЛНОЙ (финальной) схемой проекта (mixture + полный
    process-блок); ``begin_phase`` задаёт стартовую ОГРАНИЧЕННУЮ фазу (v1), далее
    фаза растёт через ``augment_phase_*`` / ``expand_region_mixture``.
    """

    def __init__(self, schema: ProjectSchema, oracle: Any, *,
                 baseline: Optional[Sequence[float]] = None,
                 seed: int = 0, n_restarts: int = 4,
                 gp_mean_model: str = "quadratic", gp_kernel: str = "matern52"):
        self.full_schema = schema
        self.oracle = oracle
        self.property_names: List[str] = list(oracle.property_names)
        self.prop_index = {n: i for i, n in enumerate(self.property_names)}

        self._full_mix = schema.mixture_block()
        self._full_proc = schema.process_block()
        self.q_full = int(schema.n_mixture)
        self.d_full = int(schema.n_process)
        self.dim_full = self.q_full + self.d_full

        self.seed = int(seed)
        self.n_restarts = int(n_restarts)
        self.gp_mean_model = gp_mean_model
        self.gp_kernel = gp_kernel

        # baseline ПОЛНЫХ составных координат (mixture-доли + process-КОД [0,1]).
        if baseline is not None:
            self.baseline = np.asarray(baseline, float).ravel()
            if self.baseline.size != self.dim_full:
                raise ValueError(f"baseline длины {self.baseline.size}, "
                                 f"ожидалось {self.dim_full}.")
        else:
            mix = (np.full(self.q_full, 1.0 / self.q_full)
                   if self.q_full else np.empty(0))
            proc = (np.full(self.d_full, 0.5) if self.d_full else np.empty(0))
            self.baseline = np.concatenate([mix, proc])


        # текущая фаза = полная схема по умолчанию (если begin_phase не вызван)
        self.current_schema: ProjectSchema = schema
        self.schema_history = SchemaHistory.start(schema)
        self.current_schema_version: int = int(schema.version)

        # общая база (ведущая) + производные numpy-кэши + общая модель + ветки
        self.points: List[DataPoint] = []
        self.X: Optional[np.ndarray] = None
        self.Y: Optional[np.ndarray] = None
        self.origin: List[str] = []
        self.surrogates: Dict[str, GPExpert] = {}
        self.branches: Dict[str, Branch] = {}
        # §15.0.3: после движения границ области (move_region) точки, выпавшие из
        # НОВОЙ области, легально исключаются из активного pool по политике
        # ``exclude`` (но ОСТАЮТСЯ в self.points — история ≠ активный pool, движение
        # обратимо). Журнал движений области (для интроспекции/обратимости).
        self._region_moves: List[Dict[str, Any]] = []
        self._drop_policy: str = POLICY_EXCLUDE

    # ------------------------------------------------------------------
    # размеры текущей фазы
    # ------------------------------------------------------------------
    @property
    def q(self) -> int:
        """Число mixture-компонентов ТЕКУЩЕЙ схемы (растёт при append C, §15.0.4).

        Фазо-зависим: в фазе 1 симплекс РЕАЛЬНО 2-компонентный {A,B} (C
        отсутствует), после ``augment_phase_mixture(["C"])`` — 3-компонентный.
        """
        return int(self.current_schema.n_mixture)

    @property
    def d(self) -> int:
        """Число process-координат ТЕКУЩЕЙ схемы (растёт при append)."""
        return int(self.current_schema.n_process)


    @property
    def dim(self) -> int:
        return self.q + self.d

    # ------------------------------------------------------------------
    # Стартовая ограниченная фаза (v1) — заменяет старый mask-путь
    # ------------------------------------------------------------------
    def begin_phase(self, mixture_free: Sequence[str],
                    process_free: Sequence[str] = ()
                    ) -> "MixtureProcessRunner":
        """Задать СТАРТОВУЮ ограниченную фазу (схема v1).

        ``mixture_free`` (§15.0.4) — имена варьируемых mixture-компонентов; фаза 1
        РЕАЛЬНО содержит ТОЛЬКО их (остальные ОТСУТСТВУЮТ, а не заперты на
        baseline — C появится append'ом в фазе 2). Должны быть ПРЕФИКСОМ полного
        состава симплекса (append новых компонентов идёт в КОНЕЦ, как process).
        ``process_free`` — имена process-переменных, ПРИСУТСТВУЮЩИХ в v1 (тоже
        ПРЕФИКС полного process-порядка); отсутствующие достраиваются baseline'ом
        при измерении.
        """
        if self._full_mix is None:
            raise ValueError("begin_phase требует mixture-блок в схеме.")
        free = set(mixture_free)
        names = list(self._full_mix.names)
        full_lo = list(self._full_mix.lower)
        full_hi = list(self._full_mix.upper)
        free_names = [nm for nm in names if nm in free]
        if free_names != names[:len(free_names)]:
            raise ValueError(
                "mixture_free должен быть ПРЕФИКСОМ полного состава симплекса "
                "(append mixture-компонента идёт в КОНЕЦ).")
        idx = [names.index(nm) for nm in free_names]
        mix_block = VariableBlock.mixture(
            free_names, lower=[full_lo[i] for i in idx],
            upper=[full_hi[i] for i in idx])


        proc_open = self._ordered_process(process_free)
        blocks: List[VariableBlock] = [mix_block]
        if proc_open:
            idx = [self._full_proc.names.index(nm) for nm in proc_open]
            blocks.append(VariableBlock.process(
                proc_open,
                lower=[self._full_proc.lower[j] for j in idx],
                upper=[self._full_proc.upper[j] for j in idx]))

        v1 = ProjectSchema(version=1, blocks=tuple(blocks),
                           responses=self.full_schema.responses,
                           model=self.full_schema.model)
        self.current_schema = v1
        self.schema_history = SchemaHistory.start(v1)
        self.current_schema_version = 1
        self.points = []
        self.surrogates = {}
        self._rebuild_arrays()
        return self

    def _ordered_process(self, names: Sequence[str]) -> List[str]:
        """Имена process-переменных в каноническом (полном) порядке."""
        if self._full_proc is None:
            return []
        order = list(self._full_proc.names)
        return [nm for nm in order if nm in set(names)]

    # ------------------------------------------------------------------
    # Раскрытие фаз: append-process / relax-mixture / атомарно (§15.1.1)
    # ------------------------------------------------------------------
    def augment_phase_schema(self, process_vars: Sequence[str]
                             ) -> ProjectSchema:
        """§14: APPEND process-переменных в схему (``version+1``).

        Политика миграции старых точек по каждому добавленному параметру =
        ``known-constant(baseline)`` (§15.1.2): точки прежних фаз мерились при
        baseline этого параметра ⇒ валидны для расширенной модели (не MISSING).
        Старые точки переиспользуются как fixed; общая база переживает фазу.
        """
        add = self._process_append_spec(process_vars)
        mig = self._process_migration(process_vars)
        new = evolve_schema(self.current_schema, add_process=add, migration=mig)
        self.schema_history.add(new)
        self.current_schema = new
        self.current_schema_version = int(new.version)
        self.fit_surrogates()
        return new

    def augment_phase_mixture(self, mixture_vars: Sequence[str]
                              ) -> ProjectSchema:
        """§15.0.4: APPEND mixture-компонента(ов) C в схему (``version+1``).

        Заменяет прежний region-expansion (Предусловие 2 отменено для C): C —
        полноценный новый компонент симплекса (Σ переопределяется
        ``A+B=1 → A+B+C=1``), а не релаксация bounds. Старые (2-компонентные)
        точки мигрируют на ГРАНЬ симплекса ``known-constant(0.0)`` — единственное
        значение, при котором Σ сходится (A+B+0=1 ⟺ A+B=1). ОТЛИЧИЕ от
        process-append: РЕАЛЬНАЯ доля, не code-baseline куба.
        """
        add = self._mixture_append_spec(mixture_vars)
        mig = self._mixture_migration(mixture_vars)
        new = evolve_schema(self.current_schema, add_mixture=add, migration=mig)
        self.schema_history.add(new)
        self.current_schema = new
        self.current_schema_version = int(new.version)
        self.fit_surrogates()
        return new

    def augment_phase_atomic(self, process_vars: Sequence[str],
                             mixture_vars: Sequence[str]) -> ProjectSchema:
        """§15.1.5 + §15.0.4: атомарная фаза «схема+схема» — ОДИН target, ОДИН bump.

        Несёт И append-process (T/P), И append-mixture (C) в ОДНОЙ новой версии;
        обе причины — в ``change_log`` (``append_param`` + ``append_mixture``).
        Миграции различают оси: process → ``known-constant(baseline-код)``,
        mixture → ``known-constant(0.0)`` (грань симплекса).
        """
        add_p = self._process_append_spec(process_vars)
        add_m = self._mixture_append_spec(mixture_vars)
        mig = self._process_migration(process_vars)
        mig.update(self._mixture_migration(mixture_vars))
        new = evolve_schema(self.current_schema, add_process=add_p,
                            add_mixture=add_m, migration=mig)
        self.schema_history.add(new)
        self.current_schema = new
        self.current_schema_version = int(new.version)
        self.fit_surrogates()
        return new

    # ------------------------------------------------------------------
    # Движение границ ОБЛАСТИ (§15.0.3) — НЕ эволюция схемы (без bump)
    # ------------------------------------------------------------------
    def move_region(self, deltas: Mapping[str, "tuple"], *,
                    policy: str = POLICY_EXCLUDE,
                    intent: Optional[str] = None) -> RegionMove:
        """Подвинуть границы СУЩЕСТВУЮЩИХ компонентов через примитив move_bounds.

        Это REGION-операция (§15.0.3 / §15.2.4): ``schema_version`` НЕ растёт —
        состав схемы тот же, меняется лишь область (bounds). Версионные bump'ы —
        дело ``augment_phase_*`` (append переменной). Примитив :func:`move_bounds`
        классифицирует движение (relax/restrict/shift; rebalance → NotImplemented)
        и проверяет инварианты симплекс-замкнутости ДО применения.

        Точки общей базы НЕ удаляются (``self.points`` = история). После сужения
        выпавшие из новой области точки легально исключаются из активного pool
        (политика ``exclude``, см. :meth:`_migrated_points`), но восстановятся при
        обратном расширении (обратимость, §15.0.3.3). Для ``policy=error`` движение
        отклоняется, если хоть одна точка с измеренным Y выпадает (без потерь
        данных); ``boundary`` (проекция) для точек с измеренным Y запрещён.

        ``intent`` — необязательная пометка ЗАЧЕМ двигаем (economy/physical/ROI),
        пишется в журнал ``_region_moves``; примитив от неё не зависит.
        """
        move = move_bounds(self.current_schema, dict(deltas))

        # policy=error / boundary: проверить судьбу КАЖДОЙ активной точки заранее,
        # чтобы запретить молчаливую потерю (или запрещённую Y-проекцию) ДО
        # подмены области. handle_dropped_fixed поднимет RegionMoveError сам.
        if policy in (POLICY_ERROR, POLICY_BOUNDARY):
            for p in self._migrated_points():
                handle_dropped_fixed(p, move.region_after, policy=policy)

        self._drop_policy = policy
        self.current_schema = move.region_after
        # История хранит АКТУАЛЬНОЕ определение текущей версии (область после
        # движения) — миграция старых точек bounds-агностична (migrate_point не
        # читает old.bounds), поэтому замена объекта версии безопасна и нужна,
        # чтобы _mixture_region/_phase_candidates видели новую область.
        self.schema_history.versions = [
            move.region_after if s.version == move.region_after.version else s
            for s in self.schema_history.versions]
        self._region_moves.append({
            "move_type": move.move_type, "deltas": dict(deltas),
            "intent": intent, "policy": policy,
            "version": int(self.current_schema_version)})
        self.fit_surrogates()
        return move

    # -- вспомогательные построители схем -------------------------------
    def _process_append_spec(self, process_vars):
        out = []
        for nm in self._ordered_process(process_vars):
            j = self._full_proc.names.index(nm)
            out.append((nm, float(self._full_proc.lower[j]),
                        float(self._full_proc.upper[j])))
        return out

    def _process_migration(self, process_vars):
        mig = {}
        for nm in self._ordered_process(process_vars):
            j = self._full_proc.names.index(nm)
            lo, hi = self._full_proc.lower[j], self._full_proc.upper[j]
            code = float(self.baseline[self.q_full + j])
            real = lo + code * (hi - lo)          # known-constant хранит РЕАЛ
            mig[nm] = known_constant(real)
        return mig

    def _ordered_mixture(self, names: Sequence[str]) -> List[str]:
        """Имена mixture-компонентов в каноническом (полном) порядке."""
        order = list(self._full_mix.names)
        return [nm for nm in order if nm in set(names)]

    def _mixture_append_spec(self, mixture_vars):
        out = []
        for nm in self._ordered_mixture(mixture_vars):
            i = self._full_mix.names.index(nm)
            out.append((nm, float(self._full_mix.lower[i]),
                        float(self._full_mix.upper[i])))
        return out

    def _mixture_migration(self, mixture_vars):
        # §15.0.4: грань симплекса C=0 — РЕАЛЬНАЯ доля (не code-трансформ куба)
        return {nm: known_constant(0.0)
                for nm in self._ordered_mixture(mixture_vars)}

    # ------------------------------------------------------------------
    # Полный физический вектор для оракула / хранение точки (§15.1.2)
    # ------------------------------------------------------------------
    def _to_full(self, coords_cur: np.ndarray) -> np.ndarray:
        """Координаты ТЕКУЩЕЙ схемы → ПОЛНЫЙ физический вектор для оракула.

        Mixture (q) копируется как есть; текущие process-координаты — ПРЕФИКС
        полного process-блока, недостающие достраиваются baseline'ом.
        """
        coords_cur = np.asarray(coords_cur, float).ravel()
        q_cur = self.q
        mix_cur = coords_cur[:q_cur]
        proc_cur = coords_cur[q_cur:]
        # mixture: отсутствующие компоненты достраиваются ГРАНЬЮ симплекса 0
        # (§15.0.4), НЕ baseline — оракул считает истину при C=0 (Σ сходится:
        # A+B+0=1). process: недостающие — baseline (внутренняя точка куба).
        mix_full = np.zeros(self.q_full)
        mix_full[:mix_cur.size] = mix_cur
        proc_full = self.baseline[self.q_full:self.q_full + self.d_full].copy()
        if proc_cur.size:
            proc_full[:proc_cur.size] = proc_cur
        return np.concatenate([mix_full, proc_full])

    def _measure(self, coords_cur: np.ndarray) -> np.ndarray:
        """Измерить набор current-координат оракулом (ПО ВСЕМ P свойствам)."""
        Xc = np.atleast_2d(coords_cur)
        full = np.vstack([self._to_full(r) for r in Xc])
        return np.atleast_2d(self.oracle.evaluate(full))

    def _make_point(self, coords_cur: np.ndarray, y_row: np.ndarray,
                    origin: str) -> DataPoint:
        coords_cur = np.asarray(coords_cur, float).ravel()
        X: Dict[str, List[float]] = {}
        if self.q > 0:
            X[MIXTURE] = [float(v) for v in coords_cur[:self.q]]
        if self.d > 0:
            X[PROCESS] = [float(v) for v in coords_cur[self.q:self.q + self.d]]
        Y = {name: float(y_row[i]) for i, name in enumerate(self.property_names)}
        tag = {"origin": origin, "schema_version": self.current_schema_version}
        return DataPoint(schema_version=self.current_schema_version,
                         X=X, Y=Y, origin_tag=tag)

    # ------------------------------------------------------------------
    # Ведущая база (DataPoint) ⇄ производные numpy-кэши на ТЕКУЩЕЙ схеме
    # ------------------------------------------------------------------
    def _migrated_points(self) -> List[DataPoint]:
        """Активные точки базы, мигрированные к ТЕКУЩЕЙ схеме (§13.7 + §15.0.3).

        Различает ДВА механизма выпадения точки:
          * **сбой миграции** (``migrate_point``→None: нет политики/несовместимый
            состав) — БАГ конфигурации, всегда ``RuntimeError``;
          * **вне текущей области** (миграция прошла, но точка не в
            ``current_schema`` после движения границ ``move_region``) — ЛЕГАЛЬНОЕ
            выпадение по политике ``exclude`` (§15.0.3.3): точка остаётся в
            ``self.points`` (история ≠ активный pool), при обратном расширении
            области снова пройдёт ``point_in_region`` ⇒ вернётся (обратимость).

        baseline закрытых ранее параметров уходит в ``known-constant`` миграции;
        ИЗМЕРЕННЫЕ Y возвращаются из исходных точек дословно (responses схемы
        раннера пусты — свойства от оракула).
        """
        if not self.points:
            return []
        used: List[DataPoint] = []
        migration_failed: List[DataPoint] = []
        for src in self.points:
            old = self.schema_history.get(src.schema_version)
            mig = migrate_point(src, old, self.current_schema)
            if mig is None:
                migration_failed.append(src)            # ось migration — баг
                continue
            if not point_in_region(mig, self.current_schema):
                continue                                # вне области — exclude (история)
            mig.Y = dict(src.Y)                         # измеренные Y дословно
            used.append(mig)
        if migration_failed:
            raise RuntimeError(
                f"{len(migration_failed)} точек не мигрировали к схеме "
                f"v{self.current_schema_version} — проверь политику миграции.")
        return used


    def _rebuild_arrays(self) -> None:
        mig = self._migrated_points()
        if not mig:
            self.X = None; self.Y = None; self.origin = []
            return
        self.X = composite_matrix(self.current_schema, mig)
        self.Y = np.asarray(
            [[float(p.Y[name]) for name in self.property_names] for p in mig],
            float)
        self.origin = [p.origin_tag.get("origin", "seed") for p in mig]

    # ------------------------------------------------------------------
    # Общая модель проекта (GP на каждое свойство, составные координаты)
    # ------------------------------------------------------------------
    def fit_surrogates(self) -> None:
        self._rebuild_arrays()
        if self.X is None or len(self.X) == 0:
            raise RuntimeError("Нет данных: сначала seed_initial().")
        self.surrogates = {}
        for i, name in enumerate(self.property_names):
            gp = GPExpert(mean_model=self.gp_mean_model, kernel=self.gp_kernel,
                          seed=self.seed, n_restarts=self.n_restarts)
            self.surrogates[name] = gp.fit(self.X, self.Y[:, i])

    def seed_initial(self, n: int = 12, seed: Optional[int] = None
                     ) -> Dict[str, Any]:
        """Стартовый набор точек ТЕКУЩЕЙ фазы + измерение + GP."""
        s = self.seed if seed is None else int(seed)
        X0 = self._phase_candidates(n, s)
        Y0 = self._measure(X0)
        self.points = [self._make_point(X0[i], Y0[i], "seed")
                       for i in range(len(X0))]
        self.fit_surrogates()
        return {"n": int(len(X0)), "P": int(self.Y.shape[1])}

    # ------------------------------------------------------------------
    # Генерация кандидатов фазы: область кодируется СХЕМОЙ (без маски)
    # ------------------------------------------------------------------
    def _mixture_region(self, schema: Optional[ProjectSchema] = None
                        ) -> SimplexRegion:
        schema = schema or self.current_schema
        return schema.mixture_block().as_simplex_region()

    def _phase_candidates(self, n: int, seed: int) -> np.ndarray:
        """n допустимых составных кандидатов ТЕКУЩЕЙ схемы (mixture-region ×
        process-куб [0,1]^d текущей размерности). Запертые mixture-bounds и
        членство process-блока полностью задают свободу фазы.

        Запертые bounds'ами компоненты (``lower==upper``, напр. C на baseline в
        фазе 1) держатся на своём значении; СВОБОДНЫЕ сэмплируются Дирихле и
        масштабируются на остаток ``1−Σ_locked`` (Σx=1). Прямой
        ``SimplexRegion.random_points`` тут непригоден: ``from_pseudo`` не уважает
        верхнюю границу запертого компонента и сваливается в центроид.
        """
        rng = np.random.default_rng(int(seed))
        n = int(n)
        mb = self.current_schema.mixture_block()
        lo = np.asarray(mb.lower, float)
        hi = np.asarray(mb.upper, float)
        locked = (hi - lo) < 1e-12
        free = ~locked
        held = lo.copy()
        held_sum = float(held[locked].sum())
        mix = np.tile(held, (n, 1))
        if free.any():
            w = rng.dirichlet(np.ones(int(free.sum())), size=n)
            mix[:, free] = w * (1.0 - held_sum)
        if self.d > 0:
            z = rng.uniform(0.0, 1.0, size=(n, self.d))
            return np.hstack([mix, z])
        return mix


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
        """Раунд активного сбора точек ветки на текущей фазе (§12).

        Бюджет раунда (``n_points``) делится: последняя точка — M8-argmax
        exploit-предложение (:meth:`optimize_xbest`, §15.1.4), остальные —
        acquisition (explore/exploit blend). ``x_best`` ветки берётся из argmax,
        а не «лучшей измеренной»: так рецепт давит к границе/вершине.
        """
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

        # exploit-слот (M8-argmax) — последний из бюджета, если есть место на >1
        n_exploit = 1 if n_take >= 2 else (1 if n_take == 1 else 0)
        n_acq = n_take - n_exploit

        seed = self.seed + 1000 + br.spent
        newX_list: List[np.ndarray] = []
        if n_acq > 0:
            cands = self._phase_candidates(n_candidates, seed)
            acq, d_pred, sigma = branch_scores(self.surrogates, br.goal, cands,
                                               explore_frac=explore_frac)
            acqX = propose_by_score(cands, acq, n_acq, min_dist=0.02)
            newX_list.append(np.atleast_2d(acqX))
        if n_exploit > 0:
            # in-round argmax держим лёгким (вызывается каждый раунд): глубокий
            # мультистарт не нужен — exploit-точка всё равно перемеривается.
            res = self.optimize_xbest(branch_id, n_candidates=400,
                                      refine_iters=80, n_starts=3)
            # рецепт argmax в координатах ТЕКУЩЕЙ схемы (q + d)
            newX_list.append(res.x[:self.dim].reshape(1, -1))


        newX = np.vstack(newX_list)
        Ynew = self._measure(newX)
        for i in range(len(newX)):
            self.points.append(
                self._make_point(newX[i], Ynew[i], f"branch:{branch_id}"))
        br.spent += len(newX)

        desir = Desirability(dict(br.goal))
        meas = {name: Ynew[:, self.prop_index[name]] for name in br.goal}
        d_meas = np.asarray(desir.overall(meas), float).ravel()
        bi = int(np.argmax(d_meas))
        if float(d_meas[bi]) > br.d_best:
            br.d_best = float(d_meas[bi])
            br.x_best = self._to_full(newX[bi]).tolist()
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

    # ------------------------------------------------------------------
    # M8-argmax по суррогату над составной областью текущей фазы (§15.1.4)
    # ------------------------------------------------------------------
    def optimize_xbest(self, branch_id: str, *, n_candidates: int = 2000,
                       refine_iters: int = 200, n_starts: int = 5
                       ) -> DesirabilityResult:
        """M8-argmax: мультистарт-максимум desirability ветки по ОБЩИМ суррогатам
        над составной областью ТЕКУЩЕЙ фазы (mixture-region × process-куб
        [0,1]^d). Возвращает :class:`DesirabilityResult` с рецептом ``x`` длиной
        ``q+d`` (current). Свобода фазы целиком в схеме: запертые mixture
        компоненты сидят в своих ``[v,v]``-bounds региона, отсутствующие process
        просто вне области (достроятся baseline'ом при измерении)."""
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        if not self.surrogates:
            self.fit_surrogates()
        br = self.branches[branch_id]
        region = self._mixture_region()
        predictors = {name: (lambda X, gp=self.surrogates[name]: gp.predict(X).mean)
                      for name in br.goal}
        kw: Dict[str, Any] = dict(
            n_candidates=int(n_candidates), refine_iters=int(refine_iters),
            n_starts=int(n_starts), seed=self.seed + 5000 + br.spent)
        if self.d > 0:
            kw.update(process_lower=[0.0] * self.d,
                      process_upper=[1.0] * self.d)
        return optimize_desirability(region, predictors, dict(br.goal), **kw)

    # ------------------------------------------------------------------
    # Диагностика дизайна (§15.1.3 / §15.2 P4 / §15.2.5)
    # ------------------------------------------------------------------
    def _design_matrix(self, points: Sequence[DataPoint]):
        terms = build_model_terms(self.current_schema)
        if not points:
            return np.empty((0, terms.p)), terms
        X = composite_matrix(self.current_schema, points)
        return model_matrix(self.current_schema, X, terms=terms), terms

    def start_info_matrix_rank(self) -> int:
        """Ранг инфо-матрицы ``EᵀE`` ФИКСИРОВАННЫХ (мигрированных из прежних
        версий) точек на модели текущей схемы (§15.1.3 / §15.2 P4)."""
        fixed = [p for p in self._migrated_points()
                 if p.origin_tag.get("schema_version",
                                     p.schema_version) < self.current_schema_version
                 or p.fixed_in_augment]
        E, _ = self._design_matrix(fixed)
        if E.shape[0] == 0:
            return 0
        return int(np.linalg.matrix_rank(E.T @ E))

    def design_i_value(self, *, ridge: float = 1e-8) -> float:
        """I-критерий ``tr[(XᵀX)⁻¹ W]`` ТЕКУЩЕГО объединённого дизайна на модели
        и моментах текущей схемы (§15.1.5 / §15.2.5). Меньше — точнее прогноз."""
        mig = self._migrated_points()
        X, terms = self._design_matrix(mig)
        if X.shape[0] == 0:
            return float("inf")
        W = build_moments(self.current_schema, terms=terms, method="analytic")
        XtX = X.T @ X + np.eye(X.shape[1]) * ridge
        try:
            inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            return float("inf")
        return float(np.trace(inv @ W))
