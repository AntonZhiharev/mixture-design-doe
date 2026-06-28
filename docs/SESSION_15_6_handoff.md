# Handoff / промт для новой сессии — §15.6 (экономический стоп) — ЗАВЕРШЁН

> **СТАТУС: §15.6 полностью реализован (ШАГИ 1–7), все зелёные.** Документ ниже
> оставлен как историческая вводная; новых незакрытых шагов §15.6 нет.

ШАГ 6 (A0.7, flat-ось → objective-gap) и ШАГ 7 (UI) добавлены:
  * `detect_flat_axis`/`FlatAxisResult`/`axis_spread` в `src/optimize/economic_stop.py`;
    runner-связки `flat_axis_at_border` (`MixtureProcessRunner`) и
    `flat_axis_mixture` (`PipelineRunner`); тест `tests/unit/test_iteration17_flat_axis.py`.
  * UI: read-only экспандер «💰 §15.6» в `render_branches` (`src/apps/streamlit_app.py`):
    stop_reason + денежная триада + flat-ось (objective-gap). A0.6 — только
    предложения. Тест `tests/unit/test_iteration17_economic_ui.py`.

---

Это вводная для сессии, которая **продолжила** реализацию §15.6. Шаги 1–5 были
сделаны ранее; ШАГИ 6–7 закрыты в этой сессии.

## Промт (скопировать в начало новой сессии)

> Продолжаем §15.6 (экономический критерий остановки и движения границ).
> Источник истины — `docs/REBUILD_SPEC_15_6_economic_stop.md` (читать ЦЕЛИКОМ,
> особенно блок аксиом §0 и «Уточнения реализации (binding)» в §3/§4/§5/§6 — это
> зафиксированные решения, код не должен их «переоткрывать»). Шаги 1–5 готовы
> (см. карту реализации в конце спеки, отметки ✅ ШАГ N). Сделай **ШАГ 6** — A0.7:
> на ВЫРОЖДЕННОМ (flat) направлении целевой функции репортить **objective-gap, а
> НЕ x-gap**; ось помечать неидентифицируемой (регрессионный кейс economy/P:
> ∂d/∂P≡0, spread=0.00e+00). Канон `.clinerules`: сначала логика+unit-тест, потом
> UI (ШАГ 7). Тесты гонять профильно через `.venv\Scripts\python.exe`, файлы
> перечислять явно, push тихо.

## Что уже сделано (ШАГИ 1–5, всё в `main`)

| Шаг | Что | Код | Тест |
|---|---|---|---|
| 1 | калибровка σ GP (фундамент VoI) | `src/models/gp_expert.py` (как есть, honest σ) | `tests/unit/test_iteration17_sigma_calibration.py` |
| 2 | ρ как GP-свойство; `price_изд = price_состав·ρ` (physics, НЕ деление!) | `price_per_item`/`make_item_cost_fn` в `src/optimize/desirability.py` | `tests/unit/test_iteration17_item_cost.py` |
| 3 | VoI = per-property MC-EI + разложение вклада + `justified` (ЗАЩИТА §5) | `mc_ei_decomposed`/`VoIResult` в `src/optimize/voi.py` | `tests/unit/test_iteration17_voi.py` |
| 4 | двойной стоп + `stop_reason`; V/c_exp/H в ветке + `resolve_horizon` | `decide_stop`/`economic_value`/`expected_price_improvement` в `src/optimize/economic_stop.py`; `Branch` в `src/design/branches.py` | `tests/unit/test_iteration17_economic_stop.py` |
| 5 | hard/soft происхождение границ (A0.5) + денежная триада (§6) | `boundary_hits`+`BORDER_*` в `src/design/move_bounds.py`; `money_triad`/`boundary_signal`/`MoneyTriad`/`HardBoundaryError` в `src/optimize/economic_stop.py`; `border_origin`/`set_border_origin`/`border_money_triad`+hard-guard в `src/apps/mixture_process_runner.py` | `tests/unit/test_iteration17_boundary_triad.py` |

Прогон всех iter17-тестов: **31 passed** (~5.9 c). Регресс move-bounds/16: 32 passed.

## ШАГ 6 — A0.7 (flat-направление → objective-gap), план

**Суть (§0 A0.7 + §3 «Следствие»):** когда оптимум упёрся в границу оси, нельзя
слепо репортить «x-gap» (насколько подвинуть переменную) — если целевая функция
**плоская** по этой оси (∂d/∂x≡0 в окрестности), переменная **неидентифицируема**,
двигать её бессмысленно. Надо репортить **objective-gap** (насколько вырастет d за
границей) и помечать ось `flat`/неидентифицируемой. Эталон: economy/P, spread по P
= 0.00e+00.

**Важная связка (§3):** ось, найденная плоской БЕЗ ρ (economy/P-флэт), может
ПЕРЕСТАТЬ быть плоской с введением ρ(...,P) — flat-статус **переоценивается** при
добавлении ρ. Детектор flat должен смотреть на ТЕКУЩИЙ d_overall ветки (включая
price_изд с ρ как свойство), а не на «истину без цены».

**Где жить логике:** чистый детектор рядом с триадой — кандидат
`src/optimize/economic_stop.py` или новый маленький модуль. Идея:
- по набору кандидатов вдоль оси (варьируем только её, остальное фиксируем у
  оптимума) считаем `spread = max(d) − min(d)` через РЕАЛЬНУЮ `Desirability.overall`;
- `spread ≤ tol` ⇒ ось `flat`/неидентифицируема ⇒ репортим **objective-gap**
  (Δd за границей, не Δx), x-gap игнорируем;
- иначе — обычный путь (триада §6 уместна);
- связать с `boundary_hits`/`border_money_triad`: для flat-оси триаду по x не
  предлагать (двигать нечего), вернуть objective-gap-диагностику.

**Тест (новый файл, напр. `tests/unit/test_iteration17_flat_axis.py`):**
- flat-ось: goal/предиктор, где d НЕ зависит от P → `spread≈0`, флаг `flat=True`,
  репортится objective-gap, x-gap игнор;
- НЕ-flat ось: spread>tol, флаг False, обычный путь;
- (если поднимается) переоценка: без ρ ось flat, с ρ-свойством в d — НЕ flat.

## ШАГ 7 — UI (после логики шага 6)

Панель Streamlit (`src/apps/streamlit_app.py`, рядом с секцией веток/стагнации,
~строка 960): показать `stop_reason`, денежную триаду (3 цифры + worth_it) при
упоре в soft-границу, информативный отказ для hard, и flat-ось (objective-gap
вместо предложения двигать x). UI — read-only предложения (A0.6). Тесты UI — по
образцу `test_iteration9_branches_ui.py`.


## Ловушки (на чём уже спотыкались в этой сессии)

1. **Шелл-таймауты.** `git log`/`dir` без флагов открывают пейджер и висят 30 c.
   Использовать `git --no-pager ...`; python звать через
   `& 'd:\DOE\.venv\Scripts\python.exe' ...` (PowerShell требует call-оператор `&`
   для путей в кавычках). Тесты — с `-p no:cacheprovider -W ignore --no-header`,
   файлы перечислять ЯВНО. НЕ гонять `pytest tests/unit` целиком (заноза
   `test_precision.py`). Тяжёлые наборы (branches/misspec, battle) могут не
   уложиться в 30 c — гонять по одному файлу.
2. **CRLF.** Файлы в CRLF; `editor` с `old_text` по многострочью иногда не матчит
   (LF↔CRLF) — тогда вставлять через `insert_line`. ОСТОРОЖНО: вставка `insert_line`
   внутрь функции может ВЫТЕСНИТЬ соседнюю строку. В этой сессии так пропал
   `return STOP_STAGNATION` в `decide_stop` — поймал тест шага 4. После любой
   вставки в существующую функцию — перечитать затронутый диапазон.
3. **§3 знак цены — binding:** `price_изд = price_состав · ρ` (УМНОЖЕНИЕ; деление в
   формуле §3 — описка). Не менять.
4. **Роли не смешивать:** VoI-движок d_overall (шаг 3) = КУДА ставить опыт;
   экономический слой (шаг 4) = СТОИТ ЛИ вообще (через цену, §6). origin границ —
   политика раннера, НЕ во frozen-схеме.

## Команды (правила репозитория, .clinerules)

- Окружение: `.venv\Scripts\python.exe`. Шелл — PowerShell, команды через `;`.
- Тесты, напр.:
  `& 'd:\DOE\.venv\Scripts\python.exe' -m pytest tests/unit/test_iteration17_flat_axis.py tests/unit/test_iteration17_economic_stop.py -q -W ignore --no-header -p no:cacheprovider`
- Push тихо: `git push --quiet origin main 2>&1; echo DONE`.
- Коммитить исходники/тесты/доки; артефакты не коммитить (`_battle_out.txt` и пр.).
- Формат коммита: `iter17/15.6: <что> (REBUILD_SPEC_15_6 sec.<§> step<N>)`.

## Первый шаг новой сессии

1. Прочитать `docs/REBUILD_SPEC_15_6_economic_stop.md` целиком (особенно §0, §3
   «Следствие для A0.7», карта реализации в конце).
2. Прочитать `src/optimize/economic_stop.py`, `src/optimize/voi.py`,
   `src/design/move_bounds.py` (`boundary_hits`), `src/optimize/desirability.py`
   (`Desirability.overall`/`individual`) — понять, на что опереть flat-детектор.
3. Сделать ШАГ 6: логика flat-оси + objective-gap + unit-тест; затем ШАГ 7 (UI).
