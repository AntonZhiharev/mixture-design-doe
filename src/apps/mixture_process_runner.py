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

Ручной оракул (§17.2): измерение расщеплено на :meth:`propose_points` (предложить
кандидатов БЕЗ измерения — read-only) и :meth:`commit_measured` (дописать
ВНЕСЁННЫЕ Y в общую базу). :meth:`run_branch_round` — тонкая обёртка над ними для
синтетического/автопрогона.

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
                                  POLICY_EXCLUDE, POLICY_ERROR, POLICY_BOUNDARY,
                                  BORDER_HARD, BORDER_SOFT, boundary_hits,
                                  _var_bounds)
from ..models.gp_expert import GPExpert
from ..design.block_model import build_model_terms, model_matrix
from ..design.augmented import build_moments
from ..design.branches import (Branch, branch_scores, propose_by_score,
                               allocate_budget,
                               ROLE_OPTIMIZED, ROLE_PRICE_INPUT, ROLE_REFERENCE,
                               ROLE_PRIORITY)
from ..optimize.desirability import (Desirability, DesirabilitySpec,
                                     DesirabilityResult, optimize_desirability,
                                     make_item_cost_fn)



def _expand_delta(schema, var, side, new_bound):
    """(lower, upper) для расширения границы ``var`` в сторону ``side`` (§15.6 §6).

    Берёт текущие bounds переменной и двигает ТОЛЬКО упёршуюся сторону к
    ``new_bound`` (другая остаётся прежней) — формат дельты для ``move_bounds``.
    """
    mb = schema.mixture_block()
    names = list(mb.names)
    if var not in names:
        raise KeyError(f"'{var}' не mixture-компонент текущей схемы.")
    j = names.index(var)
    lo, hi = float(mb.lower[j]), float(mb.upper[j])
    if side == "lower":
        return (float(new_bound), hi)
    return (lo, float(new_bound))


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
        # §15.6 §3: опциональная ЦЕНОВАЯ цель ветки (цена за изделие как min).
        # Хранится в runner (а не в Branch): держит callable composition_price_fn
        # и имя ρ-свойства; cost_fn собирается на лету из ТЕКУЩИХ суррогатов
        # (ρ̂ меняется каждый раунд). {branch_id: {price_fn, cost_spec, cost_name,
        # rho_property}}. Пусто ⇒ ветка чисто техническая (обратная совместимость).
        self._branch_cost: Dict[str, Dict[str, Any]] = {}

        # §15.0.3: после движения границ области (move_region) точки, выпавшие из
        # НОВОЙ области, легально исключаются из активного pool по политике
        # ``exclude`` (но ОСТАЮТСЯ в self.points — история ≠ активный pool, движение
        # обратимо). Журнал движений области (для интроспекции/обратимости).
        self._region_moves: List[Dict[str, Any]] = []
        self._drop_policy: str = POLICY_EXCLUDE
        # §15.6 §6 / A0.5: происхождение границ (hard/soft). ДЕФОЛТ — soft (можно
        # двигать). hard (физика/закон/бюджет) двигать НЕЛЬЗЯ. Храним отдельно от
        # frozen-схемы (политика, не состояние модели): {var: "hard"|"soft"}.
        self._border_origin: Dict[str, str] = {}

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
    def augment_phase_schema(self, process_vars: Sequence[str], *,
                             migration=None, bounds=None
                             ) -> ProjectSchema:
        """§14: APPEND process-переменных в схему (``version+1``).

        Политика миграции старых точек по каждому добавленному параметру =
        ``known-constant(baseline)`` (§15.1.2): точки прежних фаз мерились при
        baseline этого параметра ⇒ валидны для расширенной модели (не MISSING).
        Старые точки переиспользуются как fixed; общая база переживает фазу.
        """
        add = self._process_append_spec(process_vars, bounds=bounds)
        mig = (dict(migration) if migration is not None
               else self._process_migration(process_vars))
        new = evolve_schema(self.current_schema, add_process=add, migration=mig)
        self.schema_history.add(new)
        self.current_schema = new
        self.current_schema_version = int(new.version)
        self.fit_surrogates()
        return new

    def augment_phase_mixture(self, mixture_vars: Sequence[str], *,
                              migration=None, bounds=None
                              ) -> ProjectSchema:
        """§15.0.4: APPEND mixture-компонента(ов) C в схему (``version+1``).

        Заменяет прежний region-expansion (Предусловие 2 отменено для C): C —
        полноценный новый компонент симплекса (Σ переопределяется
        ``A+B=1 → A+B+C=1``), а не релаксация bounds. Старые (2-компонентные)
        точки мигрируют на ГРАНЬ симплекса ``known-constant(0.0)`` — единственное
        значение, при котором Σ сходится (A+B+0=1 ⟺ A+B=1). ОТЛИЧИЕ от
        process-append: РЕАЛЬНАЯ доля, не code-baseline куба.
        """
        add = self._mixture_append_spec(mixture_vars, bounds=bounds)
        mig = (dict(migration) if migration is not None
               else self._mixture_migration(mixture_vars))
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

        A0.5/A0.6: движение границы с происхождением ``hard`` ЗАПРЕЩЕНО (физика/
        закон/бюджет) — :class:`RegionMoveError`. Происхождение задаётся
        :meth:`set_border_origin` (дефолт ``soft``). Система не двигает молча:
        денежную триаду для ``soft``-упора готовит :meth:`border_money_triad`.
        """
        for var in deltas:
            if self.border_origin(var) == BORDER_HARD:
                raise RegionMoveError(
                    f"граница '{var}' помечена как hard (A0.5): движение запрещено "
                    f"по происхождению. Снимите hard-метку осознанно через "
                    f"set_border_origin('{var}', 'soft'), если это действительно "
                    f"мягкое ограничение.")
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

    # ------------------------------------------------------------------
    # §15.6 §6 / A0.5 — происхождение границ (hard/soft) + денежная триада
    # ------------------------------------------------------------------
    def border_origin(self, var: str) -> str:
        """Происхождение границы переменной (дефолт ``soft`` — можно двигать)."""
        return self._border_origin.get(var, BORDER_SOFT)

    def set_border_origin(self, var: str, origin: str) -> None:
        """Пометить границу ``var`` как ``hard`` (нельзя двигать) или ``soft``.

        A0.5: hard = физика/закон/бюджет. A0.6: метка — ОСОЗНАННОЕ решение
        пользователя; пометив hard, он запрещает молчаливое движение этой оси.
        """
        if origin not in (BORDER_HARD, BORDER_SOFT):
            raise ValueError(f"origin должен быть '{BORDER_SOFT}'|'{BORDER_HARD}', "
                             f"дано '{origin}'.")
        self._border_origin[var] = origin

    def border_money_triad(self, branch_id: str, var: str, side: str,
                           new_bound: float, composition_price_fn, *,
                           rho_property: str = "rho",
                           n_experiments: int = 10,
                           horizon: Optional[float] = None,
                           n_candidates: int = 400):
        """Денежная триада §6 для упора оптимума ветки в ``soft``-границу ``var``.

        Сравнивает лучшую цену изделия В ТЕКУЩЕЙ области с лучшей ожидаемой ценой
        ЗА расширенной границей (``var`` до ``new_bound``); экономия за период =
        ``Δprice_изд · V`` ветки, цена добычи = ``N · c_exp``, окупаемость
        сравнивается с горизонтом ветки (override на раунд через ``horizon``).

        ``hard``-граница ⇒ :class:`economic_stop.HardBoundaryError` (триада не
        предлагается, A0.5). Возвращает :class:`economic_stop.MoneyTriad`.

        ``composition_price_fn(Xc)`` — цена состава ₽/кг (зависит от состава);
        ``rho_property`` — имя GP-свойства плотности в общих суррогатах (§3).
        Это ПРЕДЛОЖЕНИЕ пользователю (A0.6): метод НЕ двигает границу.
        """
        from ..optimize.economic_stop import (boundary_signal,
                                               expected_price_improvement)
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        if rho_property not in self.surrogates:
            raise KeyError(f"Нет суррогата плотности '{rho_property}'. "
                           f"ρ должно быть полноценным GP-свойством (§3).")
        br = self.branches[branch_id]

        def _best_price(region_schema):
            reg = region_schema.mixture_block().as_simplex_region()
            mix = reg.random_points(int(n_candidates), seed=self.seed + 909)
            mix = np.atleast_2d(mix)
            if self.d > 0:
                rng = np.random.default_rng(self.seed + 910)
                proc = rng.uniform(0.0, 1.0, size=(mix.shape[0], self.d))
                Xc = np.hstack([mix, proc])
            else:
                Xc = mix
            pc = np.asarray(composition_price_fn(Xc), float).ravel()
            pred = self.surrogates[rho_property].predict(Xc)
            price = pc * np.asarray(pred.mean, float).ravel()
            return float(price.min()), Xc, pc, np.asarray(pred.std, float).ravel()

        price_cur, _, _, _ = _best_price(self.current_schema)
        # область ЗА границей: тот же примитив, но без применения движения
        widened = move_bounds(self.current_schema,
                              {var: _expand_delta(self.current_schema, var, side,
                                                  float(new_bound))}).region_after
        price_new, Xc_n, pc_n, sd_n = _best_price(widened)
        ei_price = float(expected_price_improvement(
            pc_n, self.surrogates[rho_property].predict(Xc_n).mean, sd_n,
            price_best=price_cur, seed=self.seed + 911).max())
        delta = max(price_cur - price_new, ei_price)

        H = br.resolve_horizon(horizon)
        return boundary_signal(
            var, side, self.border_origin(var),
            delta_price_item=delta, volume=br.volume,
            n_experiments=int(n_experiments), cost_exp=br.cost_exp, horizon=H)

    def flat_axis_at_border(self, branch_id: str, var: str, side: str,
                            new_bound: float, *, n_samples: int = 21,
                            cost_fn=None, cost_spec=None, cost_name: str = "cost",
                            tol: float = 1e-9):
        """A0.7-детектор: вырождена ли ось ``var`` для цели ветки (objective-gap).

        Строит ``objective_fn(t) -> d_overall`` ТЕКУЩЕЙ постановки ветки: варьирует
        ТОЛЬКО ось ``var`` (mixture-компонент с пропорциональной перенормировкой
        Σ=1 или process-координату), всё прочее держит у M8-оптимума ветки, и
        считает desirability через РЕАЛЬНУЮ :class:`Desirability` по ОБЩИМ
        суррогатам. Если задан ``cost_fn`` (``price_изд`` с ρ, §3), он входит как
        ``min``-свойство — тогда flat-статус честно ПЕРЕОЦЕНИВАЕТСЯ с ценой/ρ
        (§3 «Следствие»: ось, плоская без ρ, может стать не-плоской с ρ(...,P)).

        Возвращает :class:`economic_stop.FlatAxisResult`: ``flat`` ⇒ ось
        неидентифицируема, репортится ``objective_gap`` (Δd за границей), ``x_gap``
        = ``None`` (двигать переменную нечего, A0.7). Это ДИАГНОСТИКА (read-only).
        """
        from ..optimize.economic_stop import detect_flat_axis
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        if not self.surrogates:
            self.fit_surrogates()
        br = self.branches[branch_id]
        x_opt = np.asarray(self.optimize_xbest(branch_id).x, float).ravel()

        names = list(self.current_schema.mixture_names)
        proc_names = list(self.current_schema.process_names)
        if var in names:
            axis = names.index(var)                 # mixture-компонент
            lo, hi = _var_bounds(self.current_schema, var)
        elif var in proc_names:
            axis = self.q + proc_names.index(var)    # process в коде [0,1]
            lo, hi = 0.0, 1.0
        else:
            raise KeyError(f"'{var}' нет в текущей схеме (mixture/process).")

        is_mixture = var in names
        specs = dict(br.goal)
        if cost_fn is not None:
            from ..optimize.desirability import DesirabilitySpec
            specs[cost_name] = (cost_spec if cost_spec is not None
                                else DesirabilitySpec("min", low=0.0, high=1.0))
        desir = Desirability(specs)

        def _set_axis(t: float) -> np.ndarray:
            x = x_opt.copy()
            if is_mixture and self.q > 1:
                t = float(np.clip(t, 0.0, 1.0))
                others = [k for k in range(self.q) if k != axis]
                rest = float(x[others].sum())
                x[axis] = t
                if rest > 1e-12:
                    x[others] *= (1.0 - t) / rest
                else:
                    x[others] = (1.0 - t) / max(len(others), 1)
            else:
                x[axis] = float(t)
            return x

        def objective_fn(ts) -> np.ndarray:
            ts = np.atleast_1d(np.asarray(ts, float))
            X = np.vstack([_set_axis(t) for t in ts])
            props = {n: self.surrogates[n].predict(X).mean for n in br.goal}
            if cost_fn is not None:
                props[cost_name] = np.asarray(cost_fn(X), float).ravel()
            return desir.overall(props)

        samples = np.linspace(lo, hi, int(n_samples))
        border_value = hi if side == "upper" else lo
        return detect_flat_axis(var, objective_fn, samples,
                                border_value=float(border_value),
                                beyond_value=float(new_bound), tol=float(tol))



    # -- вспомогательные построители схем -------------------------------
    def _process_append_spec(self, process_vars, bounds=None):
        out = []
        for nm in self._ordered_process(process_vars):
            j = self._full_proc.names.index(nm)
            if bounds and nm in bounds:
                lo, hi = bounds[nm]
            else:
                lo, hi = self._full_proc.lower[j], self._full_proc.upper[j]
            out.append((nm, float(lo), float(hi)))
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

    def _mixture_append_spec(self, mixture_vars, bounds=None):
        out = []
        for nm in self._ordered_mixture(mixture_vars):
            i = self._full_mix.names.index(nm)
            if bounds and nm in bounds:
                lo, hi = bounds[nm]
            else:
                lo, hi = self._full_mix.lower[i], self._full_mix.upper[i]
            out.append((nm, float(lo), float(hi)))
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
    # §17.4 (Ш3) РУЧНОЙ СТАРТОВЫЙ ОРАКУЛ: seed расщеплён на две половины, как
    # branch-цикл §17.2 — propose_seed (предложить дизайн БЕЗ измерения, read-only)
    # и commit_seed (дописать ВНЕСЁННЫЕ Y). :meth:`seed_initial` — авто-обёртка над
    # ними для синт.оракула (мерит сам). Ветки на старте ещё нет, поэтому это
    # отдельная от branch-цикла пара; origin точек = "seed".
    # ------------------------------------------------------------------
    def propose_seed(self, n: int = 12, *, seed: Optional[int] = None
                     ) -> np.ndarray:
        """§17.4: ПРЕДЛОЖИТЬ стартовый seed-дизайн БЕЗ измерения (read-only).

        Возвращает ``n`` составных кандидатов ТЕКУЩЕЙ схемы (mixture-region ×
        process-куб) как первую половину ручного стартового цикла: пользователь
        измеряет их сам и фиксирует через :meth:`commit_seed`. НИЧЕГО не измеряет
        и НЕ пишет в общую базу (A0.6). Детерминированно по ``seed``. Аналог
        :meth:`propose_points`, но для стартового дизайна (ветки ещё нет)."""
        s = self.seed if seed is None else int(seed)
        return self._phase_candidates(int(n), s)

    def commit_seed(self, X: Any, Y: Any) -> Dict[str, Any]:
        """§17.4: ЗАФИКСИРОВАТЬ измеренные ``Y`` стартового seed-дизайна.

        Вторая половина ручного стартового цикла: ``X`` — кандидаты из
        :meth:`propose_seed` (координаты текущей схемы, ``n×dim``), ``Y`` —
        измеренные отклики (``n×P`` в порядке ``property_names``; вносит
        пользователь). Точки ДОПИСЫВАЮТСЯ в ОБЩУЮ базу с origin-тегом ``"seed"``
        (И-1, без урезания истории), суррогаты переобучаются. В отличие от
        :meth:`seed_initial` (авто-оракул), Y приходит от пользователя. Пустой
        ``X`` — no-op. Возвращает ``{added, n_base, P}``.
        """
        newX = np.atleast_2d(np.asarray(X, float))
        Ynew = np.atleast_2d(np.asarray(Y, float))
        P = len(self.property_names)
        if newX.shape[1] != self.dim:
            raise ValueError(
                f"X: ожидалось {self.dim} координат на точку, дано {newX.shape[1]}.")
        if Ynew.shape[0] != newX.shape[0]:
            raise ValueError(
                f"Y: строк {Ynew.shape[0]} ≠ числу точек {newX.shape[0]}.")
        if Ynew.shape[1] != P:
            raise ValueError(
                f"Y: ожидалось {P} свойств на строку ({list(self.property_names)}), "
                f"дано {Ynew.shape[1]}.")
        if newX.shape[0] == 0:
            return {"added": 0, "n_base": len(self.points), "P": P}
        for i in range(len(newX)):
            self.points.append(self._make_point(newX[i], Ynew[i], "seed"))
        self.fit_surrogates()
        return {"added": int(len(newX)), "n_base": len(self.points), "P": P}


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

    def set_branch_cost(self, branch_id: str, composition_price_fn,
                        cost_spec: DesirabilitySpec, *,
                        rho_property: str = "rho",
                        cost_name: str = "price") -> None:
        """§15.6 §3: задать ЦЕНОВУЮ цель ветки — цена за изделие как ``min``.

        ``price_изд = composition_price_fn(Xc) · ρ̂(Xc)`` (см. ``make_item_cost_fn``):
        ``composition_price_fn`` — цена состава ₽/кг (зависит от состава), ρ̂ —
        среднее ОБЩЕГО суррогата ``rho_property``. Цена НЕ хранится отдельным
        свойством и НЕ фитится в Шеффе — собирается на лету из ТЕКУЩИХ суррогатов
        (ρ̂ меняется каждый раунд). ``cost_spec`` — фиксированный ``min``-диапазон
        цены (одинаков для acquisition/argmax/измеренного d_best).

        ``rho_property`` обязан быть полноценным GP-свойством (есть в оракуле):
        измеренный d_best считает цену по ИЗМЕРЕННОЙ ρ, а acquisition/argmax — по ρ̂.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        if rho_property not in self.property_names:
            raise KeyError(f"ρ-свойство '{rho_property}' не среди свойств оракула "
                           f"{self.property_names} (§3: ρ — полноценный отклик).")
        self._branch_cost[branch_id] = {
            "price_fn": composition_price_fn, "cost_spec": cost_spec,
            "cost_name": str(cost_name), "rho_property": str(rho_property)}

    def _branch_cost_fn(self, branch_id: str):
        """``(cost_fn, cost_name, cost_spec)`` ценовой цели ветки или ``(None,)*3``.

        ``cost_fn`` собирается из ТЕКУЩИХ суррогатов (ρ̂ = ``surrogates[rho]``) —
        вызывать ПОСЛЕ ``fit_surrogates``. Нет ценовой цели ⇒ ``(None, None, None)``
        (ветка чисто техническая, обратная совместимость)."""
        cfg = self._branch_cost.get(branch_id)
        if cfg is None:
            return None, None, None
        rho = cfg["rho_property"]
        if rho not in self.surrogates:
            raise KeyError(f"Нет суррогата ρ '{rho}' — сначала fit_surrogates().")
        rho_pred = (lambda X, gp=self.surrogates[rho]: gp.predict(X).mean)
        cost_fn = make_item_cost_fn(cfg["price_fn"], rho_pred)
        return cost_fn, cfg["cost_name"], cfg["cost_spec"]

    # ------------------------------------------------------------------
    # §5/§12 РОЛЬ ОТКЛИКА в ветке — атрибут (ветка × отклик), ВЫВОДИТСЯ из
    # текущего намерения (goal + ценовая конфигурация), а НЕ хранится отдельно
    # (нет дубля состояния). Приоритет M2: OPTIMIZED > PRICE_INPUT > REFERENCE —
    # роль всегда однозначна (XOR честности §5). Аналог _border_origin: политика
    # раннера, НЕ во frozen-схеме. Атрибуция branch-local (Гр-3): эти методы
    # читают ТОЛЬКО намерение запрошенной ветки.
    # ------------------------------------------------------------------
    def response_role(self, branch_id: str, response: str) -> str:
        """Роль ``response`` В КОНТЕКСТЕ ветки ``branch_id`` (приоритет M2).

        ``OPTIMIZED`` — отклик есть в ``branch.goal`` (нога качества d_i);
        ``PRICE_INPUT`` — отклик НЕ цель, но это ρ ценовой ноги ветки
        (``set_branch_cost`` ⇒ σ_ρ-разведка засчитывается в деньги, §3);
        ``REFERENCE`` — меряется оракулом, но ни цель, ни ρ-цены (справочный).

        Приоритет OPTIMIZED > PRICE_INPUT снимает двойную роль ρ (Гр-1): если
        ρ одновременно в ``goal`` и питает цену, роль = ``OPTIMIZED`` (вся σ_ρ
        ушла в качество), а ценовой σ_ρ-канал зануляется на шаге атрибуции — см.
        :meth:`price_channel_suppressed`.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        if response not in self.property_names:
            raise KeyError(f"Отклик '{response}' не среди свойств оракула "
                           f"{self.property_names}.")
        if response in self.branches[branch_id].goal:
            return ROLE_OPTIMIZED
        cfg = self._branch_cost.get(branch_id)
        if cfg is not None and response == cfg["rho_property"]:
            return ROLE_PRICE_INPUT
        return ROLE_REFERENCE

    def branch_roles(self, branch_id: str) -> Dict[str, str]:
        """Карта ``{отклик → роль}`` по ВСЕМ свойствам оракула для ветки."""
        return {p: self.response_role(branch_id, p) for p in self.property_names}

    def responses_by_role(self, branch_id: str) -> Dict[str, List[str]]:
        """Обратный индекс ``{роль → [отклики]}`` (порядок свойств оракула)."""
        out: Dict[str, List[str]] = {r: [] for r in ROLE_PRIORITY}
        for p in self.property_names:
            out[self.response_role(branch_id, p)].append(p)
        return out

    def price_channel_suppressed(self, branch_id: str) -> bool:
        """ρ ценовой ноги имеет роль OPTIMIZED ⇒ σ_ρ-ценовой канал занулён (Гр-1).

        ``True`` только когда у ветки есть ценовая нога И её ``rho_property``
        попал в ``goal`` (роль ρ = OPTIMIZED по приоритету M2). Это read-only
        диагностика-крючок для атрибуции (шаг B): денежная нога от σ_ρ-разведки
        в такой ветке = 0 (вся неопределённость ρ уже оправдана качеством),
        ценовая выгода остаётся лишь от ВЫБОРА состава (детерминированный
        price_состав·μ_ρ). ``False`` ⇒ либо нет цены, либо ρ — чистый PRICE_INPUT.
        """
        cfg = self._branch_cost.get(branch_id)
        if cfg is None:
            return False
        return cfg["rho_property"] in self.branches[branch_id].goal

    # ------------------------------------------------------------------
    # §16 D — ФАКТИЧЕСКАЯ денежная нога стопа ветки, ЧИТАЮЩАЯ роль ρ.
    # Тянет зануление σ_ρ-канала (price_channel_suppressed, Гр-1) до РЕАЛЬНОГО
    # денежного VoI-гейта: economic_value ветки считается через
    # price_attributed_value с rho_optimized=price_channel_suppressed(bid).
    # Раньше rho_optimized жил лишь в ЧИСТЫХ функциях economic_stop, а раннер
    # денежную ногу не вызывал вовсе (в §15.6-пути активны только триада/
    # boundary_signal). Атрибуция branch-local: читает намерение ИМЕННО ветки.
    # ------------------------------------------------------------------
    def _branch_reference_recipe(self, branch_id: str) -> np.ndarray:
        """Опорный «инкумбент» ветки в координатах ТЕКУЩЕЙ схемы (длина q+d).

        Это опорная точка для атрибуции прироста (§5): относительно неё считаются
        ``price_best`` (порог EI) и ``d_overall_cur`` (база Δlog d_overall). Если у
        ветки есть измеренный ``x_best``, совместимый с текущим q (mixture-доли
        суммируются в 1) — берём его срез к текущей схеме; иначе нейтральная
        опорная точка: ЦЕНТРОИД mixture-региона + process-baseline (всегда
        допустима, фазо-устойчива — не зависит от того, бежали ли уже раунды).
        """
        proc = (self.baseline[self.q_full:self.q_full + self.d].copy()
                if self.d > 0 else np.empty(0))
        xb = self.branches[branch_id].x_best
        if xb is not None:
            xb = np.asarray(xb, float).ravel()
            mix = xb[:self.q]
            if mix.size == self.q and abs(float(mix.sum()) - 1.0) < 1e-6:
                if self.d > 0 and xb.size >= self.q_full + self.d:
                    proc = xb[self.q_full:self.q_full + self.d]
                return np.concatenate([mix, proc]) if self.d > 0 else mix
        mix = np.asarray(self._mixture_region().centroid(), float).ravel()
        return np.concatenate([mix, proc]) if self.d > 0 else mix

    def branch_economic_value(self, branch_id: str, *, n_candidates: int = 600,
                              n_mc: int = 512, horizon: Optional[float] = None,
                              x_ref: Optional[Sequence[float]] = None,
                              price_best: Optional[float] = None,
                              seed: Optional[int] = None) -> float:
        """§16 D: ЧЕСТНАЯ денежная ценность раунда ветки (₽ за горизонт),
        АТРИБУТИРОВАННАЯ ценовой оси и ЧИТАЮЩАЯ роль ρ (Гр-1).

        Связывает три куска §16 в один денежный сигнал:
          1. σ_ρ-разведку — :func:`expected_price_improvement` (EI на удешевление
             изделия ``price_изд = price_состав·ρ``, ρ~N(μ̂,σ̂) общего суррогата);
          2. атрибуцию §5 — :func:`price_attributed_value` (деньги ТОЛЬКО за
             прирост ``d_overall`` ветки ИМЕННО через цену, не за «фантом дешёвого
             угла»);
          3. РОЛЬ ρ — ``rho_optimized=self.price_channel_suppressed(branch_id)``:
             если ρ носит роль OPTIMIZED (цель И питает цену), ВЕСЬ ценовой
             разведочный канал занулён (``α≡0``) — двойной счёт одной δρ убран.

        Ветка без ценовой ноги (``set_branch_cost`` не задан) ⇒ ``0.0``: денежного
        VoI-гейта нет (чисто техническая ветка, обратная совместимость). Опорный
        инкумбент — :meth:`_branch_reference_recipe` (или ``x_ref`` override, в
        координатах текущей схемы); ``price_best`` по умолчанию — цена изделия в
        опорной точке. Это ДИАГНОСТИКА (read-only): метод ничего не измеряет.
        """
        from ..optimize.economic_stop import (expected_price_improvement,
                                               price_attributed_value)
        from ..optimize.desirability import desirability_value
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        cfg = self._branch_cost.get(branch_id)
        if cfg is None:
            return 0.0                  # нет ценовой ноги ⇒ денежного гейта нет
        if not self.surrogates:
            self.fit_surrogates()
        br = self.branches[branch_id]
        rho = cfg["rho_property"]
        cost_name = cfg["cost_name"]
        cost_spec = cfg["cost_spec"]
        price_fn = cfg["price_fn"]
        s = (self.seed + 2000 + br.spent) if seed is None else int(seed)

        # опорный рецепт (инкумбент) и его цена/качество (по ρ̂ суррогата)
        x_cur = (np.asarray(x_ref, float).ravel() if x_ref is not None
                 else self._branch_reference_recipe(branch_id))
        Xc_cur = x_cur.reshape(1, -1)
        pc_cur = float(np.asarray(price_fn(Xc_cur), float).ravel()[0])
        rho_cur = float(np.asarray(self.surrogates[rho].predict(Xc_cur).mean,
                                   float).ravel()[0])
        price_cur = pc_cur * rho_cur
        pb = price_cur if price_best is None else float(price_best)

        specs = dict(br.goal)
        specs[cost_name] = cost_spec
        desir = Desirability(specs)

        def _d_overall(X, price_item):
            means = {n: np.asarray(self.surrogates[n].predict(X).mean,
                                   float).ravel() for n in br.goal}
            means[cost_name] = np.asarray(price_item, float).ravel()
            return np.asarray(desir.overall(means), float).ravel()

        d_overall_cur = float(_d_overall(Xc_cur, [price_cur])[0])
        d_price_cur = float(np.asarray(desirability_value(price_cur, cost_spec),
                                       float).ravel()[0])

        cand = self._phase_candidates(int(n_candidates), s)
        pc_c = np.asarray(price_fn(cand), float).ravel()
        pred = self.surrogates[rho].predict(cand)
        rho_mean = np.asarray(pred.mean, float).ravel()
        rho_std = np.asarray(pred.std, float).ravel()
        price_item_c = pc_c * rho_mean
        d_overall_cand = _d_overall(cand, price_item_c)
        d_price_cand = np.asarray(desirability_value(price_item_c, cost_spec),
                                  float).ravel()

        price_savings = expected_price_improvement(
            pc_c, rho_mean, rho_std, price_best=pb, n_mc=int(n_mc), seed=s + 7)

        price_weight = float(cost_spec.weight)
        total_weight = (float(sum(sp.weight for sp in br.goal.values()))
                        + price_weight)
        H = br.resolve_horizon(horizon)
        return price_attributed_value(
            price_savings, d_overall_cur=d_overall_cur,
            d_overall_cand=d_overall_cand, d_price_cur=d_price_cur,
            d_price_cand=d_price_cand, price_weight=price_weight,
            total_weight=total_weight, volume=br.volume, horizon=H,
            rho_optimized=self.price_channel_suppressed(branch_id))

    def branch_stop_decision(self, branch_id: str, *, delta_d: float,
                             ceil: float, eps: float = 5e-3,
                             econ_policy: Optional[str] = None,
                             n_round: int = 1, n_candidates: int = 600,
                             n_mc: int = 512, horizon: Optional[float] = None,
                             x_ref: Optional[Sequence[float]] = None,
                             price_best: Optional[float] = None,
                             seed: Optional[int] = None):
        """§16 D: двойной стоп §4 ветки с денежной ногой, ЧИТАЮЩЕЙ роль ρ.

        Денежная нога ``economic_value`` — :meth:`branch_economic_value` (роль ρ
        уже учтена: при OPTIMIZED-ρ канал занулён ⇒ ``economic_value=0``).
        Цена раунда — ``N·c_exp`` (``n_round·br.cost_exp``, §4-BATCH). Возвращает
        :class:`economic_stop.StopDecision`: при занулённом канале денежная нога
        НЕ фантомит — ``econ_red_flag`` поднимается честно (а под ``ECON_BINDING``
        причина — ``not_economical``), вместо мнимой выгоды за σ_ρ.
        """
        from ..optimize.economic_stop import evaluate_stop, ECON_BINDING
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        br = self.branches[branch_id]
        ev = self.branch_economic_value(
            branch_id, n_candidates=n_candidates, n_mc=n_mc, horizon=horizon,
            x_ref=x_ref, price_best=price_best, seed=seed)
        round_cost = float(br.cost_exp) * int(n_round)
        return evaluate_stop(
            delta_d=float(delta_d), d_best=float(br.d_best), ceil=float(ceil),
            economic_value=float(ev), cost_exp=round_cost, eps=float(eps),
            econ_policy=(ECON_BINDING if econ_policy is None else econ_policy))

    # ------------------------------------------------------------------
    # §17.2 (Ш1) РУЧНОЙ ОРАКУЛ: измерение расщеплено на две явные половины —
    # propose_points (предложить БЕЗ измерения, read-only) и commit_measured
    # (дописать ВНЕСЁННЫЕ Y). run_branch_round — тонкая обёртка над ними для
    # синтетического/автопрогона (оракул меряет сам).
    # ------------------------------------------------------------------
    def propose_points(self, branch_id: str, n_points: int = 2,
                       explore_frac: float = 0.3, n_candidates: int = 600,
                       seed: Optional[int] = None) -> np.ndarray:
        """§17.2: ПРЕДЛОЖИТЬ точки ветки БЕЗ измерения (read-only, база не меняется).

        Возвращает матрицу кандидатов ``newX`` (координаты ТЕКУЩЕЙ схемы, длина
        ``dim`` на строку); НИЧЕГО не измеряет и НЕ пишет в общую базу — первая
        половина ручного цикла «предложить → зафиксировать Y» (A0.6: мерим только
        по команде пользователя). Бюджет ``n_points`` делится как в
        :meth:`run_branch_round`: последняя точка — M8-argmax exploit, остальные —
        acquisition (explore/exploit blend). Детерминированно по ``seed`` (по
        умолчанию ``self.seed+1000+spent`` — совпадает с авто-раундом). Нет
        бюджета ⇒ пустой массив ``(0, dim)``.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        if not self.surrogates:
            self.fit_surrogates()
        br = self.branches[branch_id]
        n_take = min(int(n_points), br.remaining())
        if n_take <= 0:
            return np.empty((0, self.dim), float)

        # exploit-слот (M8-argmax) — последний из бюджета, если есть место на >1
        n_exploit = 1 if n_take >= 2 else (1 if n_take == 1 else 0)
        n_acq = n_take - n_exploit

        # §15.6 §3: ценовая цель ветки (если задана) — цена = price_состав·ρ̂
        cost_fn, cost_name, cost_spec = self._branch_cost_fn(branch_id)

        s = (self.seed + 1000 + br.spent) if seed is None else int(seed)
        newX_list: List[np.ndarray] = []
        if n_acq > 0:
            cands = self._phase_candidates(n_candidates, s)
            acq, d_pred, sigma = branch_scores(
                self.surrogates, br.goal, cands, explore_frac=explore_frac,
                cost_fn=cost_fn, cost_name=(cost_name or "cost"),
                cost_spec=cost_spec)
            acqX = propose_by_score(cands, acq, n_acq, min_dist=0.02)
            newX_list.append(np.atleast_2d(acqX))
        if n_exploit > 0:
            # in-round argmax держим лёгким (вызывается каждый раунд): глубокий
            # мультистарт не нужен — exploit-точка всё равно перемеривается.
            res = self.optimize_xbest(branch_id, n_candidates=400,
                                      refine_iters=80, n_starts=3)
            # рецепт argmax в координатах ТЕКУЩЕЙ схемы (q + d)
            newX_list.append(res.x[:self.dim].reshape(1, -1))
        return np.vstack(newX_list)

    def commit_measured(self, branch_id: str, X: Any, Y: Any) -> Dict[str, Any]:
        """§17.2: ЗАФИКСИРОВАТЬ измеренные отклики ``Y`` предложенных точек.

        Вторая половина ручного цикла: ``X`` — кандидаты из :meth:`propose_points`
        (координаты текущей схемы, ``n×dim``), ``Y`` — измеренные отклики
        (``n×P`` в порядке ``property_names``; вносит пользователь или синт.оракул
        через «Заполнить тестовыми»). Точки дописываются в ОБЩУЮ базу с
        origin-тегом ``branch:{id}`` (И-1, без копий), суррогаты переобучаются,
        ``d_best``/``x_best`` пересчитываются по ИЗМЕРЕННЫМ Y (не по суррогату).
        Контракт возврата — как у :meth:`run_branch_round`.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        newX = np.atleast_2d(np.asarray(X, float))
        Ynew = np.atleast_2d(np.asarray(Y, float))
        P = len(self.property_names)
        if newX.shape[1] != self.dim:
            raise ValueError(
                f"X: ожидалось {self.dim} координат на точку, дано {newX.shape[1]}.")
        if Ynew.shape[0] != newX.shape[0]:
            raise ValueError(
                f"Y: строк {Ynew.shape[0]} ≠ числу точек {newX.shape[0]}.")
        if Ynew.shape[1] != P:
            raise ValueError(
                f"Y: ожидалось {P} свойств на строку ({list(self.property_names)}), "
                f"дано {Ynew.shape[1]}.")
        br = self.branches[branch_id]
        if newX.shape[0] == 0:
            br.refresh_status()
            return {"branch": branch_id, "added": 0, "status": br.status,
                    "d_best": br.d_best, "x_best": br.x_best,
                    "n_base": int(0 if self.X is None else len(self.X))}

        for i in range(len(newX)):
            self.points.append(
                self._make_point(newX[i], Ynew[i], f"branch:{branch_id}"))
        br.spent += len(newX)

        # измеренный d_best (§3): цена за изделие — по ИЗМЕРЕННОЙ ρ (в Ynew),
        # а не по суррогату; argmax/acquisition в propose_points — по ρ̂.
        specs = dict(br.goal)
        meas = {name: Ynew[:, self.prop_index[name]] for name in br.goal}
        if branch_id in self._branch_cost:
            cfg = self._branch_cost[branch_id]
            pc = np.asarray(cfg["price_fn"](newX), float).ravel()
            rho_meas = Ynew[:, self.prop_index[cfg["rho_property"]]]
            meas[cfg["cost_name"]] = pc * rho_meas
            specs[cfg["cost_name"]] = cfg["cost_spec"]
        desir = Desirability(specs)
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

    def run_branch_round(self, branch_id: str, n_points: int = 2,
                         explore_frac: float = 0.3, n_candidates: int = 600
                         ) -> Dict[str, Any]:
        """Раунд активного сбора точек ветки на текущей фазе (§12).

        Тонкая обёртка над :meth:`propose_points` + оракул + :meth:`commit_measured`
        (§17.2): предлагает точки, измеряет их синт.оракулом и фиксирует Y. Бюджет
        (``n_points``) делится: последняя точка — M8-argmax exploit (§15.1.4),
        остальные — acquisition; ``x_best`` — из argmax, а не «лучшей измеренной».
        Ручной поток (реальная лаборатория) вызывает две половины напрямую.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Нет ветки '{branch_id}'.")
        br = self.branches[branch_id]
        newX = self.propose_points(branch_id, n_points=n_points,
                                   explore_frac=explore_frac,
                                   n_candidates=n_candidates)
        if newX.shape[0] == 0:
            br.refresh_status()
            return {"branch": branch_id, "added": 0, "status": br.status,
                    "d_best": br.d_best, "x_best": br.x_best}
        Ynew = self._measure(newX)
        return self.commit_measured(branch_id, newX, Ynew)

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
        # §15.6 §3: argmax по desirability+цена (цена = price_состав·ρ̂ из суррогата)
        cost_fn, cost_name, cost_spec = self._branch_cost_fn(branch_id)
        if cost_fn is not None:
            kw.update(cost_fn=cost_fn, cost_name=cost_name, cost_spec=cost_spec)
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
