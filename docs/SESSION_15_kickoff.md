# Kickoff новой сессии — реализация §15 (Schema Augmentation в бою + M8-argmax)

Это вводная для сессии, которая будет **реализовывать** ТЗ §15. Не пересказывает
ТЗ — указывает, с чего начать, где источник истины и каких ловушек избегать.

## Источник истины

- **ТЗ:** `docs/REBUILD_SPEC_15_battle_augmentation.md` — читать ЦЕЛИКОМ перед
  кодом. Это согласованный контракт (предусловия §15.0, инвариант атомарности
  §15.1.5, семантика pool/W §15.0.2, тесты §15.2, порядок §15.3, гейт §15.4).
- **Базовая спека:** `docs/REBUILD_SPEC.md` — §13.6/§13.7 (augmented design +
  schema evolution), §5.5 (критерий остановки), §3/M8 (desirability), §8 (бенчмарк).
- **Контекст отладки:** `docs/SESSION_battle_handoff.md`, тест
  `tests/unit/test_iteration13_battle.py` (точка входа, что сейчас проверяется).

## Что уже подтверждено чтением кода (НЕ перепроверять с нуля)

- `core/schema_evolution.py::migrate_point` (стр. 161–175): миграция **пер-параметр**,
  `known-constant` хранит реальное значение, переводит в код по `(lo,hi)`.
- `core/schema.py::schema_diff_vars` (стр. 531–547): диффит **только список имён**,
  `lower/upper` НЕ читает → C-релаксация ему невидима.
- `core/schema_evolution.py::evolve_schema` (стр. ~117): **всегда** `version+1`,
  умеет только `add_process` (append). Смену mixture-bounds НЕ умеет.
- `design/augmented.py`: `build_candidate_pool` уважает L/U; `analytic_moment_matrix`
  считает W на ПОЛНОМ симплексе (L/U игнорирует) → это §15.0.2, корректно.
- Две `select_fixed_rows`: упрощённая в `design/augmented.py` и полная в
  `core/schema_evolution.py`. Боевой §14 использует **полную**.

## Порядок реализации (из §15.3 — соблюдать, не переписывать)

1. **Хранение baseline как ЗНАЧЕНИЯ** (§15.1.2 ловушка): закрытые координаты
   (`T=0.5`) пишутся в `X["PROCESS"]` точки реальным значением, не маской.
   **ПЕРВОЕ** — без этого `select_fixed_rows` не увидит T и выбросит точки.
2. **Diff/evolve правки** (Предусловие 4 + гейт §15.4):
   - новая `schema_diff_bounds(old,new)` (НЕ трогать `schema_diff_vars`);
   - evolve-путь смены mixture-bounds (различать bump: append → `version+1`,
     relax → change в той же версии);
   - `assert point_in_region(fixed, target.region)` (проверить наличие
     `SimplexRegion.contains`, иначе добавить).
3. **M8-argmax** (economy-фикс): `x_best = optimize_desirability(...)` вместо
   «лучшей измеренной». Изолированно, на существующих тестах сходимости.
4. **`augment_phase_schema` + полная `select_fixed_rows`** (§14 ядро).
5. **`augment_phase_atomic`** (§15.1.5): один target (+P append + C relax), один
   `augmented_design`, лог обеих причин, bump только за append.
6. **Удалить старый mask+baseline путь** раскрытия фаз (принято: не параллелить).
7. **Новые боевые тесты** §15.2.1/.2/.4/.5 + правка §15.2.3 + тесты §15.0.2.
8. **Только если** economy < 0.97 после шагов 3–7 — граничная тяга acquisition
   (часть «а»), отдельной итерацией.

> Каждый шаг: **сначала логика + unit-тест, потом UI** (канон REBUILD_SPEC §5/§12).

## Ловушки, на которых легко ошибиться

- **`+C` ≠ schema evolution.** Симплекс `{A,B,C}` полон с v1; «открытие C» = relax
  bounds (region), `schema_version` за это НЕ растёт. Тест §14 ставить на `+T/+P`,
  не на `+C`. (Предусловие 2)
- **pool/W:** НЕ добавлять assert «область pool == область W» — он сломает §15.0.2.
  Оставить только assert согласованности **термов** (Условие A) + комментарий-якорь.
- **Атомарность фазы 3:** запрещён промежуточный пересчёт дизайна на не-финальной
  схеме. Один target несёт оба изменения; контрольный `assert I_atomic ≤ I_seq`.
- **Не молчаливые подстановки:** `unknown` → точка исключается; новый отклик →
  `MISSING`; никаких нулей/средних (§13.7).

## Гейт перед мержем

Полный чек-лист — §15.4 ТЗ (предусловия, правка diff/evolve/select, алгоритм,
боевые тесты, регресс). Не мержить, пока не закрыты.

## Команды (правила репозитория, .clinerules)

- Окружение: `.venv\Scripts\python.exe`.
- Тесты — перечислять файлы ЯВНО (glob не разворачивается), напр.:
  ```
  .venv\Scripts\python.exe -m pytest tests/unit/test_iteration13_battle.py -q -W ignore
  ```
  Профильные новые файлы §15 (по мере создания) добавлять в ту же строку.
- НЕ гонять `pytest tests/unit` целиком (заноза `test_precision.py`:
  `ModuleNotFoundError: core`).
- Шелл — PowerShell, команды через `;`. Push тихо:
  `git push --quiet origin main 2>&1; echo DONE`.
- Перед изменениями: `git pull --rebase origin main` (если нет незакоммиченного).
- Коммитить только исходники/тесты/доки; артефакты не коммитить (см. `.gitignore`).
- Формат коммита, напр.: `iter13/§15: baseline-as-value + schema_diff_bounds (REBUILD_SPEC_15 §15.0/§15.3)`.

## Первый шаг сессии

1. Прочитать `docs/REBUILD_SPEC_15_battle_augmentation.md` целиком.
2. Прочитать `tests/unit/test_iteration13_battle.py` и `src/core/state.py`
   (где хранится точка/`X`) — понять, как сейчас пишется baseline.
3. Начать с шага 1 порядка (§15.3): baseline как значение в `X` + unit-тест,
   фиксирующий `pt.X["PROCESS"][T_idx] == 0.5`.
