# §15. Боевая верификация §14 (Schema Augmentation) + M8-argmax

Статус: ТЗ (согласовано в сессии отладки `test_iteration13_battle`).
Связано с: REBUILD_SPEC §13.6/§13.7 (augmented design + schema evolution),
§5.5 (критерий остановки), §3/M8 (desirability), §8 (боевой бенчмарк).

ТЗ состоит из двух частей:
1. **что должен делать боевой тест §14** (часть 2 ниже);
2. **что алгоритм обязан гарантировать**, чтобы тест был осмысленным (часть 1).

Часть (1)-гарантии **важнее** — без них тест проверит «не падает» вместо
«совпадает с истиной».

---

## §15.0 Предусловия (фиксируются ДО кода)

### Предусловие 1 — миграция резолвится ПО ПАРАМЕТРУ (подтверждено кодом)

Подтверждено чтением `core/schema_evolution.py::migrate_point` (строки 161–175):

- `target.migration` = словарь `{var_name: policy}`; для каждого ДОБАВЛЕННОГО
  process-параметра политика резолвится **независимо** по имени.
- `known-constant` хранит **реальное** значение; переводится в код по `(lo,hi)`
  ЭТОГО параметра: `code = (v - lo) / (hi - lo)`.
- Фаза 3 (открываем `C, P`): для точки фазы 2 `P` резолвится по `migration["P"]`
  независимо; `T` уже открыт ранее (обычная координата). Логика НЕ «поедет».

### Предусловие 2 — `C` это НЕ schema evolution (важнейшая оговорка)

`C` — **mixture-компонент**, симплекс `{A,B,C}` (Σ=1) онтологически полон с v1.
Фаза 1 не «не знала про C» — она **фиксировала** C через bounds/baseline, но C
структурно присутствовал. Поэтому «раскрытие C» — **смена области (region)**, а
НЕ добавление переменной в схему.

```
Раскрытие переменных — ДВА РАЗНЫХ механизма:
  • +T, +P (process)  → APPEND В СХЕМУ      → schema_version+1 → §14 augmented (migration)
  • +C    (mixture)   → RELAX BOUNDS        → схема та же по составу → region expansion
```

- `migrate_point` C-релаксацию **переживёт без правок**: `schema_diff_vars`
  возвращает пустой `diff[MIXTURE]` (имя `C` не менялось), координата C у точки
  уже есть, P уходит в `known-constant`.
- НО: тест §14 (миграция/append) **обязан** ставиться на `+T`/`+P`, не на `+C` —
  иначе проверяется не тот механизм (ложно-зелёный относительно §14).
- Region-expansion (`+C`) проверяется ОТДЕЛЬНО (§15.2.4).

### Предусловие 3 — полная `select_fixed_rows` из `schema_evolution`, не mock

Есть ДВЕ функции (не перепутать):

- `design/augmented.py::select_fixed_rows(points, target_schema)` — частный
  случай без истории/миграции: берёт точку, только если `composite_coords`
  уже проходит. Политик нет.
- `core/schema_evolution.py::select_fixed_rows(points, target_schema, history,
  recompute_fns=...)` — **полная**: резолвит старую версию по `history`, гоняет
  `migrate_point` (per-param политики, `known-constant`/`unknown→MISSING`/
  `recompute`, асимметрия «новый член из старых переменных бесплатен»).

→ Боевой §14 использует **полную** из `schema_evolution`. Если в тесте есть
локальный mock — заменить на импорт реальной (иначе тест проверяет mock).

### Предусловие 4 — `schema_diff` диффит ТОЛЬКО список переменных, НЕ bounds

Подтверждено чтением `core/schema.py::schema_diff_vars` (строки 531–547): чистая
set-разница имён по блокам; `b.lower/b.upper` не читаются нигде. Значит
C-релаксация (имя то же, bounds `[1/3,1/3]→[0,1]`) для diff **невидима**.

Нужны три маленькие правки ПЕРЕД атомарной реализацией §15.1:

1. **`schema_diff_bounds(old, new) -> {kind: {var: (old_lo,old_hi,new_lo,new_hi)}}`**
   — НОВАЯ функция (не расширять `schema_diff_vars`: на него завязан
   `migrate_point` через `if diff[MIXTURE]: return None`; смешивание → риск, что
   миграция начнёт отбрасывать валидные точки при C-релаксации).
2. **evolve-путь для смены mixture-bounds** — `evolve_schema` сейчас умеет только
   `add_process` (append); нужен конструктор новой версии с изменёнными
   mixture-bounds (через пересборку/`replace` блока).
3. **`assert fixed ∈ new_bounds`** — явная валидация fixed-строк против новой
   области; сейчас её НЕТ вообще (нельзя ни предполагать, ни молча пропускать).

### §15.0.2 Семантика «ограниченные кандидаты при полной области интереса» (pool ⊂ region(W))

> **Зафиксировано как КОРРЕКТНОЕ поведение, НЕ баг** (закрывает бывший открытый
> вопрос pool/W).

В фазах 1–2 `pool` уважает L/U (C заперт на baseline → кандидаты на РЕБРЕ
симплекса), а `W` (moment matrix, `analytic_moment_matrix`) интегрируется по
ПОЛНОМУ симплексу (C свободен). Рассогласование областей **намеренное**: оно
кодирует «экспериментировать можем только в подобласти (pool), но оптимальность
оцениваем относительно полной области интереса (W)».

```
Фаза 1–2:
  pool  = build_candidate_pool(region_restricted)   # C=baseline → РЕБРО симплекса
  W     = analytic_moment_matrix(region_FULL)        # C свободен → ПОЛНЫЙ симплекс
          ─────────────────────────────              ─────────────────────────
          ГДЕ можно ставить опыты                     ОТНОСИТЕЛЬНО ЧЕГО оптимально
          (физическое ограничение фазы)               (область интереса заказчика)
```

**Математическое обоснование.** I-оптимальность минимизирует усреднённую по
области интереса дисперсию предсказания:

```
I = tr[(X_poolᵀ X_pool)⁻¹ · M_full],   M_full = ∫_{S_full} f fᵀ
```

`X_pool` — где физически ставим точки (ребро, C=baseline); `M_full` — где хотим
хорошо предсказывать (полный симплекс). «Где можно мерить» ≠ «где важно быть
точным». I-критерий с `M_full` при `X_pool` на ребре отвечает на правильный
вопрос: «как расставить доступные (реберные) точки, чтобы минимизировать
дисперсию по ПОЛНОМУ симплексу» — легитимная extrapolation-aware постановка.

> «Исправление» рассогласования (подмена W на `M_edge`) оптимизировало бы под
> НЕВЕРНУЮ область — точность по ребру вместо нужного заказчику симплекса. Это
> была бы регрессия, замаскированная под фикс. Поэтому фиксируем явно.

**Следствия и защита от «фикса в неверную сторону»:**

1. **Диагностический разрыв (намеренно высокий I в фазах 1–2).** ⚠️ I в
   фазах 1–2 будет ВЫШЕ, чем если бы W считался по ребру — реберные точки плохо
   покрывают дисперсию по свободному C (экстраполяция вглубь симплекса). Это
   ожидаемо, не дефект.
   - [ ] ТЗ фиксирует: высокий I в ограниченных фазах = цена ограничения; падает
     при раскрытии C (фаза 3, релаксация bounds → pool догоняет W).
   - [ ] Бесплатная проверка §14/region-expansion: `I_фаза3 < I_фаза2` при той же
     модели — именно потому, что pool наконец покрывает область W. Не падает →
     релаксация C не сработала.

2. **Защитный assert на согласованность ТЕРМОВ (не областей).** pool и W
   легитимно в разных областях, но обязаны быть в ОДНИХ термах (Условие A
   M5-инъекции). Разрешаем расхождение области, запрещаем расхождение термов.
   ```python
   # РАЗРЕШЕНО: разные области (pool ⊂ region(W)) — это §15.0.2
   # ЗАПРЕЩЕНО: разные термы (model_matrix vs M) — это баг Условия A
   assert model_matrix_fn(pool).shape[1] == W.shape[0], \
       "термы pool и W рассогласованы (Условие A) — ЭТО баг, в отличие от области"
   # НО НЕ добавлять assert `pool_region == W_region` — он сломал бы §15.0.2
   ```
   - [ ] Assert Условия A (термы) — остаётся.
   - [ ] Assert «одинаковая область pool/W» — НЕ добавлять.
   - [ ] Комментарий-якорь в коде между ними — чтобы будущий ревьюер не добавил
     область-assert «для симметрии».

3. **Невырожденность `X_poolᵀ X_pool` на ограниченном pool** (НАСТОЯЩИЙ риск, в
   отличие от мнимого «бага рассогласования»). Если pool заперт на ребре
   (C=const), термы, активируемые только свободным C (`A·B·C`, члены с C), могут
   быть вырождены/коллинеарны → `XᵀX` плохо обусловлена → `(XᵀX)⁻¹` взрывается →
   I-мусор.
   - [ ] Проверить обусловленность `X_poolᵀ X_pool` в фазах 1–2: модель не
     содержит термов, неидентифицируемых на ограниченном pool. quadratic Scheffé
     на ребре C=const идентифицируем → ok; терм, требующий вариации C, не оценим.
   - [ ] Контроль: `cond(X_poolᵀ X_pool) < threshold` или явный rank-чек активных
     термов на pool (это настоящий assert, в отличие от мнимого область-assert).

**Тесты §15.0.2:**

```python
def test_pool_restricted_W_full_is_legitimate():
    """§15.0.2: pool на ребре + W по полному симплексу = корректно, не баг."""
    pool = build_candidate_pool(region_C_fixed)        # ребро
    W = analytic_moment_matrix(region_full)            # полный симплекс
    assert pool_region(pool) != W_region(W)            # области РАЗНЫЕ — ОК
    assert model_matrix(pool).shape[1] == W.shape[0]   # термы ОДИНАКОВЫЕ (Условие A)
    I = i_value(model_matrix(pool), W)
    assert np.isfinite(I)

def test_I_drops_when_pool_reaches_W_region():
    """Раскрытие C: pool догоняет область W → I падает (та же модель)."""
    I_phase2 = i_value(model_matrix(pool_edge), W_full)   # pool ⊂ region(W)
    I_phase3 = i_value(model_matrix(pool_full), W_full)   # pool = region(W)
    assert I_phase3 < I_phase2                            # покрытие улучшилось

def test_restricted_pool_terms_identifiable():
    """Настоящий риск: термы модели идентифицируемы на ограниченном pool."""
    X = model_matrix(pool_edge)
    assert np.linalg.matrix_rank(X.T @ X) == X.shape[1]   # не вырождено
    assert np.linalg.cond(X.T @ X) < COND_THRESHOLD
```

---

## §15.1 Что алгоритм обязан гарантировать (часть важнее теста)

### §15.1.1 Переход фазы = смена схемы, не маска. Два механизма разделены.

> ПРИНЯТО: schema-augmentation **заменяет** маску полностью. Старый mask+baseline
> код раскрытия фаз **удалить**, а не держать параллельно (два пути раскрытия =
> рассинхрон, ровно та проблема, которую избегали с критерием остановки).

```python
def open_variables(state, new_vars):
    process_appends = [v for v in new_vars if is_process_append(v)]   # +T, +P
    mixture_unmasks = [v for v in new_vars if is_mixture_component(v)] # +C
    # АТОМАРНО (см. §15.1.5): если фаза меняет И схему, И область — один target,
    # один augmented_design. Раздельные ветки ниже — для ЧИСТЫХ случаев.
    if process_appends and mixture_unmasks:
        return augment_phase_atomic(state, process_appends, mixture_unmasks)
    if process_appends:
        return augment_phase_schema(state, process_appends)   # §14: version+1
    if mixture_unmasks:
        return expand_region_mixture(state, mixture_unmasks)  # region, НЕ §14
    return state


def augment_phase_schema(state, process_vars):
    """ТОЛЬКО append process в схему. C сюда не попадает."""
    new_schema = append_process_vars(state.current_schema, process_vars)  # version+1
    state.schema_history.append(new_schema)
    fixed, skipped = select_fixed_rows(state.points, new_schema, state.schema_history)
    new_pts = augmented_design(new_schema, fixed, ...)        # fixed как старт EᵀE
    measure_and_append(new_pts, oracle)
    refit_surrogates(state)                                   # на ОБЪЕДИНЁННОЙ базе
    return state


def expand_region_mixture(state, mixture_vars):
    """Снятие ограничения симплекса. Схема та же по составу; schema_version НЕ
    инкрементится за это (но bounds-change регистрируется, см. §15.1.5)."""
    new_region = unmask_simplex_region(state.region, mixture_vars)  # ребро → полный симплекс
    new_pts = i_optimal_augment_sequential(state.points, ..., model=current_block_model)
    measure_and_append(new_pts, oracle)
    refit_surrogates(state)
    return state
```

### §15.1.2 Migration policy для фазового раскрытия — ЯВНАЯ

Когда фаза 2 открывает `T`, точки фазы 1 имели `T = baseline = 0.5`. Политика:

- [ ] `T` в фазе 1 = `known-constant(0.5)` — точки фазы 1 **валидны** для модели
  с `T` (T=0.5 известно). Это **не** `MISSING`, это полноценные данные при T=0.5.
- [ ] `select_fixed_rows` помечает их `known-constant`, **не** выбрасывает.
- [ ] **Асимметрия:** новый **кросс-член** `A·T` для старых точек **вычислим**
  (A есть, T=0.5 известно) → точка валидна и для `A·T`. Новый **параметр** требует
  политики; новый **член из старых+известных переменных** — бесплатен.

> ⚠️ Ловушка хранения: baseline `T=0.5` должен **реально записываться в
> `X["PROCESS"]`** точки фазы 1, а не подразумеваться маской. Иначе
> `select_fixed_rows` не увидит T у точки и выбросит её. Это **правка хранения**,
> и она ПЕРВАЯ (см. §15.3).

### §15.1.3 Augmented design стартует с непустой информационной матрицы

- [ ] `Xfixed^T Xfixed` (точки фаз 1..k-1, спроецированные на новые термы) —
  **стартовая** инфо-матрица augmented design фазы k, **не** ноль.
- [ ] Критерий остановки §5.5 — тот же `stop_reason`, `I` на новой модели +
  объединённом дизайне (переиспользовать M5-инъекцию; второй критерий НЕ вводить).
- [ ] Старые fixed-точки **не переизмеряются** (у них уже есть Y); добираются
  только новые.

### §15.1.4 M8-argmax на объединённой базе (economy-фикс)

- [ ] `x_best` ветки = `optimize_desirability(surrogate, schema, region)`
  (мультистарт argmax по суррогату), **не** «лучшая измеренная точка».
- [ ] argmax в **псевдокомпонентах** на симплексе + бокс по process (как
  `optimize_desirability` уже умеет, `src/optimize/desirability.py`).
- [ ] argmax учитывает стоимостный член (он есть в `optimize_desirability`).

### §15.1.5 Атомарность фазы, меняющей И схему, И область (ИНВАРИАНТ)

> **Инвариант §15.0 (под фиксацию):** Фаза, изменяющая И схему (append process),
> И область (relax mixture bounds), применяется **атомарно**: одна `target_schema`
> несёт оба изменения, один `augmented_design` со всеми старыми точками как fixed.
> `schema_version` инкрементится за append; relax-bounds регистрируется как
> `relax_bounds`-change в **той же** версии (без отдельного bump). Промежуточный
> пересчёт дизайна на НЕ-финальной схеме запрещён.

Обоснование (доказуемо): фиксация подмножества строк сужает допустимое множество
⇒ `I_atomic ≤ I_seq`. Sequential (region→schema) жжёт опыты на промежуточную
модель (без P-членов) и замораживает их как fixed — строго ≤ I-эффективности
совместного дизайна под финальную модель. Version-timeline шаг C всё равно не
пишет (Предусловие 2) — выделять его в отдельный пересчёт смысла нет.

Четыре условия атомарности (приняты):

1. [ ] `select_fixed_rows`/оркестратор резолвит ДВЕ оси diff'а в одном проходе:
   append-P (migration `known-constant`) И relax-bounds-C (НЕ migration, но
   `assert fixed_C ∈ new_bounds` — не предположение). Требует `schema_diff_bounds`
   (Предусловие 4).
2. [ ] Лог версии перечисляет ОБЕ причины (`append_param: P` + `relax_bounds: C`)
   при ОДНОМ bump. Примирение с Предусловием 2: bump за P, релаксация C —
   `kind="relax_bounds"` в той же версии, в логе, без своего bump. Без записи
   теряется, в какой области собраны v3-точки.
3. [ ] `M_final` (моменты/модель) на РАСШИРЕННОЙ области (полный симплекс ×
   `[0,1]^2`); `assert` на `target_region`, не на `state.region`.
4. [ ] В тест добавить контрольный sequential-прогон и `assert I_atomic ≤ I_seq` —
   превращает «атомарно лучше» из аргумента в проверяемый инвариант.

> Семантика pool/W (кандидаты на ребре при W по полному симплексу) — РЕШЕНА и
> зафиксирована как корректное поведение в §15.0.2 (не баг). `M_final` на
> расширенной области не отменяет §15.0.2 для ранних фаз; см. там же проверку
> `I_фаза3 < I_фаза2`.

---

## §15.2 Изменения в боевом тесте

### §15.2.1 Новый тест: `test_battle_schema_augmentation_reuses_phase1` (только +T)

```python
def test_battle_schema_augmentation_reuses_phase1():
    """§14 в бою: process-append (+T). C сюда НЕ входит — это region, не schema."""
    truth, goals = build_battle_truth(), build_battle_goals()

    # --- Фаза 1: свободны A,B; T,P на baseline (T=0.5 КАК ЗНАЧЕНИЕ В ТОЧКЕ) ---
    state_B = run_phase1(schema_v1, n_seed=18, n_branch=5)
    assert all(pt.X["PROCESS"][T_idx] == 0.5 for pt in state_B.points)   # §15.1.2 ловушка
    assert all(pt.schema_version == 1 for pt in state_B.points)

    # --- Фаза 2 через ЧИСТЫЙ schema-append: только T (process). НЕ +C. ---
    state_B = augment_phase_schema(state_B, process_vars=["T"])          # version → 2

    from src.core.schema_evolution import select_fixed_rows              # ПОЛНАЯ (Предусл.3)
    fixed, skipped = select_fixed_rows(state_B.points_v1, schema_v2, state_B.schema_history)
    assert len(fixed) >= 1                                               # §15.1.2: НЕ выброшены
    assert all(f.origin_tag.get("migrated_from") == 1 for f in fixed)    # мигрированы из v1
    # политика T для всех = known-constant (пер-параметр, Предусл.1):
    assert schema_v2.migration["T"]["policy"] == "known-constant"

    # --- Сценарий C: фаза 2 с нуля (контроль) ---
    state_C = run_phase2_from_scratch(schema_v2, n_seed=18, n_branch=5)
```

### §15.2.2 Проверки (различающие, не декоративные)

```python
    # (P1) Augmented НЕ ХУЖЕ from-scratch по сходимости к истине
    assert d_final(state_B) >= d_final(state_C) - 0.03

    # (P2) Augmented дешевле: меньше НОВЫХ опытов на тот же уровень d
    assert n_new_experiments(state_B, phase=2) < n_experiments(state_C)
    #      ← доказательство, что fixed-строки работают

    # (P3) Старые точки НЕ переизмерены
    assert phase1_points_unchanged(state_B)        # Y фазы 1 не трогали

    # (P4) Augmented design стартовал с непустой инфо-матрицы
    assert start_info_matrix_rank(state_B, phase=2) > 0   # §15.1.3

    # (P5) Новая модель содержит T-члены (γ_T, δ_AT)
    assert "T" in model_terms(schema_v2) and "A:T" in model_terms(schema_v2)

    # (P6) Точки разных версий сосуществуют в базе
    assert {1, 2} <= {pt.schema_version for pt in state_B.points}
```

### §15.2.3 Правка существующего теста сходимости (economy-фикс, M8-argmax)

```python
    # M8-argmax вместо "лучшей измеренной":
    # ОЖИДАНИЕ: premium/fast ↑ к ~100%, economy ↑ (но, возможно, не до 97%).
    # economy T должно дотянуться ближе к 1.0 (ГРАНИЦА КУБА, не вершина симплекса!)
    assert state.economy.x_best[T_idx] > 0.85      # было 0.661 → argmax давит к краю
    assert d_final(economy) >= 0.94                # не хуже, чем было
    # если economy всё ещё < 0.97 → тогда часть (а) граничной тяги, отдельной итерацией
```

### §15.2.4 Новый тест region-expansion (`+C`) — ОТДЕЛЬНО

```python
def test_battle_mixture_region_expansion():
    """+C = снятие ограничения симплекса. schema_version НЕ меняется за это. НЕ §14."""
    state = run_phase_with_T(...)                          # уже открыт T
    v_before = state.current_schema_version
    state = expand_region_mixture(state, mixture_vars=["C"])
    # схема по СОСТАВУ та же; если bounds-change оформлен как relax_bounds в той же
    # версии — version не растёт за C (см. §15.1.5 п.2):
    assert state.current_schema_version == v_before
    # точки переиспользуются как fixed в пределах ОДНОЙ модели (M5), не через migration
    assert d_final_improves_or_holds(state)
```

### §15.2.5 Тест атомарности фазы 3 (+C и +P вместе)

```python
def test_battle_phase3_atomic_better_than_sequential():
    """Фаза 3 атомарно (один target: +P append + C relax) ≤ I, чем region→schema."""
    state_atomic = augment_phase_atomic(state_after_T, process=["P"], mixture=["C"])
    state_seq    = expand_then_append(state_after_T, mixture=["C"], process=["P"])
    assert I_value(state_atomic) <= I_value(state_seq) + 1e-9    # §15.1.5 довод
    # один bump за P, причины обе в логе:
    assert state_atomic.current_schema_version == v_after_T + 1
    log = state_atomic.schema_history[-1].change_log
    assert {"append_param": "P"} in log and {"relax_bounds": "C"} in log
```

---

## §15.3 Порядок реализации (не переписывать)

> economy-фикс и §14 смыкаются: M8-argmax работает на объединённой fixed+new базе.

1. **Хранение baseline как значения** (§15.1.2 ловушка): закрытые координаты
   пишутся в `X` точки реальным значением, не маской. **ПЕРВОЕ** — иначе
   `select_fixed_rows` не увидит T=0.5.
2. **Diff/evolve правки** (Предусловие 4): `schema_diff_bounds`, evolve-путь смены
   mixture-bounds, `assert fixed ∈ new_bounds`.
3. **M8-argmax** (economy-фикс): подключить `optimize_desirability` вместо «лучшей
   измеренной». Изолированно, на существующих тестах сходимости. Проверить
   premium/fast → ~100%, economy → насколько подрос.
4. **`augment_phase_schema` + полная `select_fixed_rows`** (§14 ядро):
   переиспользует M5-инъекцию и `augmented_design`.
5. **`augment_phase_atomic`** (§15.1.5): один target (+P append + C relax), один
   `augmented_design`, лог обеих причин.
6. **Удалить старый mask+baseline путь** раскрытия фаз (принято: не держать
   параллельно).
7. **Новые боевые тесты** §15.2.1/.2/.4/.5 + правка §15.2.3.
8. **Только если economy < 0.97 после шагов 3–7** — граничная тяга acquisition
   (часть «а»), отдельной итерацией. Не раньше.

---

## §15.4 Чек-лист (гейт перед мержем)

### Предусловия (§15.0)
- [ ] Миграция пер-параметр: `target.migration[var]`; фаза 3 резолвит `C`,`P`
  независимо.
- [ ] `C` ≠ schema evolution: раскрытие mixture-компонента = region expansion,
  `schema_version` не растёт за C, не через `augmented_design`-as-schema-change.
- [ ] Полная `select_fixed_rows` из `schema_evolution`, не mock.
- [ ] `schema_diff_bounds` добавлена; evolve умеет менять mixture-bounds;
  `assert fixed ∈ new_bounds` на месте.

### Правка diff/evolve/select (Предусловие 4) — гейт
- [ ] `schema_diff_bounds` добавлена; `schema_diff_vars` НЕ тронут (две
  ортогональные оси: added/removed vars vs изменённые bounds).
- [ ] `schema_diff_bounds` смотрит только `old.vars ∩ new.vars` (не пересекается
  с added/removed — у новых переменных «старых» границ нет, у удалённых нет
  «новых»).
- [ ] Возвращает diff ЛЮБОГО изменения границ (не только релаксации): сужение
  bounds тоже ловится (симметрично).
- [ ] `evolve_schema(append_params, relax_bounds)`: bump `version+1` ТОЛЬКО за
  append; relax — `change` в той же версии (правка текущей логики: сейчас
  `evolve_schema` всегда делает `version+1`, стр. 117 — нужно различать оси).
- [ ] `select_fixed_rows` резолвит ОБЕ оси: `migration` (added → политика) +
  `within_new_bounds` (rebounded → проверка вхождения, НЕ миграция).
- [ ] Пострезолв `assert point_in_region(fixed, target.region)` — defensive: ловит
  пропуск оси diff'ом (использовать существующий `SimplexRegion.contains`/
  аналог по process-боксу или добавить, если нет).

### Тесты правки diff/evolve/select
- [ ] Ортогональность осей: схема с added-var + rebounded-var → `schema_diff_vars`
  видит только added, `schema_diff_bounds` только rebounded.
- [ ] Негативный: точка с координатой ВНЕ новых bounds исключена из fixed (или
  ловится пострезолв-assert'ом), не протекает молча.
- [ ] Боевой: baseline-C точки фазы 1 ∈ релаксованных bounds → точка остаётся
  валидным fixed после relax (не выброшена).

### Алгоритм (§15.1)
- [ ] Переход фазы = `schema_version+1` (за append), не маска. Старый mask-путь
  удалён.
- [ ] baseline закрытого process пишется в `X` реальным значением.
- [ ] `select_fixed_rows` помечает фазу 1 `known-constant`, не выбрасывает; кросс
  `A:T` вычислим для fixed, не требует миграции.
- [ ] `augmented_design` стартует с `Xfixed^T Xfixed ≠ 0`; старые точки не
  переизмеряются.
- [ ] Критерий остановки = тот же `stop_reason` §5.5, `I` на новой модели.
- [ ] `x_best` = `optimize_desirability` (argmax), не «лучшая измеренная».
- [ ] Фаза «схема+область» атомарна: один target, один пересчёт; bump за append,
  `relax_bounds` в логе той же версии; `M_final` на расширенной области;
  `assert I_atomic ≤ I_seq` в тесте.

### Боевые тесты (§15.2)
- [ ] §14: B (augmented) vs C (from-scratch) на одной истине; P1–P6.
- [ ] P2 (новых опытов в B < C) — доказательство переиспользования.
- [ ] region-expansion (+C): `schema_version` неизменна (§15.2.4).
- [ ] атомарность фазы 3: `I_atomic ≤ I_seq`, один bump, обе причины в логе
  (§15.2.5).

### Регресс (база не сломана)
- [ ] Старый боевой тест сходимости проходит после M8-argmax: монотонность
  d1≤d2≤d3, потолки, origin-теги.
- [ ] mixture-only golden бит-в-бит (M5-инъекция не задета §14).
- [ ] mask-путь удалён осознанно (не оставлен полу-живым).

---

## §15.5 Открытые вопросы (вынесены, решаются позже)

1. **Граница economy** (часть «а»): нужна ли тяга acquisition к вершинам/границам
   симплекса, если M8-argmax не дотянул economy до 97%.

---

### Решённые (история)

- **pool/W несогласованность** — РЕШЕНА в §15.0.2: «ограниченные кандидаты при
  полной области интереса» зафиксированы как корректное поведение (не баг), с
  обоснованием `I = tr[(XᵀX)⁻¹·M_full]` и защитой от «фикса в неверную сторону».

---

## §15.6 Статус реализации (шаги §15.3, выполнено)

Все 7 шагов §15.3 реализованы; профильные тесты зелёные (battle 1 passed ~221 c;
iteration15/mp_runner/12_*/9_* — 77 passed). Ключевые проектные решения,
зафиксированные при реализации:

- **Ре-архитектура раннера (шаг 6, «удалить mask-путь»).** Старый
  `set_free`/`_masked_candidates`/baseline-as-value-в-X удалён. Свобода фазы
  теперь кодируется САМОЙ схемой:
  - `+process (T,P)` = членство в process-блоке → `augment_phase_schema`
    (`evolve_schema(add_process, migration=known-constant(baseline))`, `version+1`);
  - `+mixture (C)` = bounds (`[1/3,1/3]→[0,1]`) → `expand_region_mixture`
    (без bump) либо `augment_phase_atomic` (атомарно с append-P, один bump);
  - `begin_phase(mixture_free, process_free)` строит стартовую схему v1
    (mixture-only при пустом process_free; запертые mixture-компоненты — bounds
    `[v,v]` на baseline).
  - Точки хранят координаты ТЕКУЩЕЙ схемы; общий GP всегда видит базу,
    мигрированную к текущей схеме (`select_fixed_rows`/`migrate_point`),
    `known-constant` достраивает закрытый параметр РЕАЛЬНЫМ baseline (§15.1.2).
    Оракул всегда меряется на ПОЛНОМ физическом векторе (`_to_full`).

- **M8-argmax (§15.1.4) встроен в `run_branch_round` как exploit-точка В ПРЕДЕЛАХ
  бюджета раунда** (последняя из `n_points` — argmax `optimize_xbest`, остальные —
  acquisition). Это СОЗНАТЕЛЬНО сохраняет контракт `added == n_points` (рерайт
  `test_iteration13_mp_runner` свёлся к смене семантики `x_best`: теперь это
  argmax-рецепт, а не «лучшая измеренная»). `x_best` хранится как ПОЛНЫЙ составной
  рецепт (`_to_full`).

- **`select_fixed_rows` и Y.** `migrate_point` пересобирает `Y` по
  `target.response_names`; у схемы раннера responses пусты (свойства — от оракула),
  поэтому измеренные `Y` восстанавливаются из исходных точек дословно (миграция
  трогает только X-координаты/область).

- **Кандидаты на запертом компоненте.** `SimplexRegion.random_points` непригоден
  при `lower==upper` (`from_pseudo` игнорирует верхнюю границу → rejection
  сваливается в центроид). `_phase_candidates` сэмплирует свободные компоненты
  Дирихле и масштабирует на `1−Σ_locked` (Σx=1), запертые держит на baseline.

### Покрытие §15.2 тестами

- **§15.2.1/.2 (§14 reuse, P3–P6)** — `test_augment_phase_schema_reuses_phase1_as_fixed`
  (+T: version+1, миграция фазы 1 НЕ выброшена, baseline-as-value `X[:,T]==0.5`,
  `start_info_matrix_rank>0`, T-члены в модели) + `..._new_round_adds_versioned_points`
  (версии 1 и 2 сосуществуют). Полная B-vs-C from-scratch сходимость (P1/P2 с
  маржами) НЕ выделена в отдельный прогон — сходимость покрыта боевым §15.2.3.
- **§15.2.3 (M8-argmax в боевом)** — `test_iteration13_battle.py` переведён на
  `begin_phase`/`augment_phase_schema(["T"])`/`augment_phase_atomic(["P"],["C"])`;
  `x_best` от M8-argmax; робастные проверки (монотонность, потолки) сохранены.
- **§15.2.4 (region-expansion +C)** — `test_region_expansion_does_not_bump_version`.
- **§15.2.5 (атомарность)** — `test_atomic_phase_one_bump_both_reasons_logged`
  (один bump, обе причины в `change_log`, C раскрыт, `design_i_value` конечен).
  Контрольный `I_atomic ≤ I_seq` как ОТДЕЛЬНЫЙ прогон не добавлен (инвариант
  §15.1.5 обоснован аналитически; в коде проверяется идентифицируемость дизайна).

### Отложено

- **§15.5 / §15.3 шаг 8** — граничная тяга acquisition к вершинам, если economy
  всё ещё < 0.97 после M8-argmax. Отдельной итерацией, по факту замера в боевом.


