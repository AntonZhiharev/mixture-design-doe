# ТЗ: ревизия UI кампании под decode-слой (iter40–44)

Статус: **утверждено к работе 05.08.2026** (ревизия UI после iter32–39:
DECODE_LAYER_PROPOSAL + CAMPAIGN_SPEC_PVC). Идём по шагам; каждый шаг —
отдельная итерация с тестами и коммитом.

Основание ревизии (проверено grep'ом по коду 05.08.2026):

* `src/apps/campaign_ui.py` из новой волны интегрировал ТОЛЬКО preflight
  iter32 (caption + экспандер деталей в seed-цикле);
* `src/apps/campaign.py` (CampaignController): 0 упоминаний
  phr/chance/preflight_pairs/campaign_label/quantize/premix/binding;
* `src/apps/campaign_state.py`: НЕ сериализует `phr_spec`,
  `campaign_label`, `preflight_pairs` — молчаливая потеря политики
  кампании при save/load (против A0.6 и требования CAMPAIGN_SPEC_PVC §3
  «записать ДО первого замера, задним числом не восстанавливается»).

## Канон исполнения (обязателен для каждого шага)

1. **Сначала логика + тест, потом UI** (REBUILD_SPEC §5/§12, .clinerules).
   Все новые UI-хелперы — чистые функции (без Streamlit), тестируемые
   напрямую (образец: `seed_preflight_caption`, `preflight_details_dataframe`).
2. **A0.6**: ничего не блокируем молча и не теряем молча. Диагностика —
   сигнал, решение — за пользователем.
3. Тесты: `tests/unit/test_iteration{N}_*.py`, перед коммитом гонять
   профильные + регресс: `test_iteration35_phr_spec_campaign.py`,
   `test_iteration37_campaign_gaps.py`, `test_iteration38_phr_optimizer.py`,
   `test_iteration39_sigma_channel.py`, `test_iteration9_branches.py`
   (файлы перечислять явно, не glob).
4. Коммит на шаг: `iter{N}/ui: <суть> (UI_REVISION_SPEC §{N})`.

---

## iter40 — Персистентность политики кампании (П1, блокер)

Без этого шага любой UI-ввод последующих шагов бессмыслен: save/load
молча откатит сэмплер/оптимизатор на бокс и перестанет писать
`spec_hash`/`campaign` в `origin_tag` новых точек.

**Файл:** `src/apps/campaign_state.py`.

`runner_to_state` — добавить в раздел `"runner"`:

```json
"phr_spec": [<PhrSpec.to_dicts()>] | null,
"campaign_label": "<str>" | "",
"preflight_pairs": [[["осьA1", ...], ["осьB1", ...]], ...]
```

`runner_from_state` — восстановление ШТАТНЫМИ сеттерами (они валидируют
против схемы): `set_phr_spec(PhrSpec.from_dicts(...))`,
`set_campaign_label(...)` (только непустая), `set_preflight_pairs(...)`.
Старые сейвы без ключей → `None` / `""` / `[]` (без ошибок — как
`sampling_groups` iter31).

**Инварианты (тесты, `test_iteration40_campaign_policy_persistence.py`):**

1. round-trip phr-спеки: `spec_hash()` до сохранения == после загрузки
   (референс — `RECIPE_DICTS` из тестов iter35, hash
   `049d0e35…360450`);
2. round-trip `campaign_label` + `preflight_pairs` (включая оси-суммы
   `["PMPlus_8", "DL_531"]`);
3. старый сейв (без новых ключей) грузится, поля пустые;
4. точка, добавленная ПОСЛЕ load, получает `spec_hash`/`campaign`
   в `origin_tag` (сквозной сценарий save → load → добор);
5. `preflight()` после load видит пары (pair-coverage не пустой).

---

## iter41 — Сетап: ввод phr-спеки + паспорт кампании (П2, П3)

**Файлы:** `src/apps/campaign_ui.py` (`render_setup_form`,
`render_composition_bounds`, `setup_prefill_from_runner`,
`project_settings_dataframe`/`render_project_settings`) + чистые хелперы.

### 41.1 Третий режим ввода состава: «phr-спека (JSON)»

В радио «Способ ввода» (`render_composition_bounds`) добавить режим
**«phr-спека (JSON)»**: `st.text_area` с JSON-списком узлов в формате
`PhrSpec.from_dicts` (тот же формат, что `to_dicts` → входит в hash) +
опциональный `st.file_uploader` (.json). Конструктор `PhrSpec` уже
делает статическую валидацию (циклы, ссылки, пустые пересечения) —
ошибки показывать `st.error` как есть.

Чистые хелперы (тестируемые):

* `parse_phr_spec_json(text: str) -> PhrSpec` — json.loads + from_dicts,
  человекочитаемые ошибки;
* `phr_spec_summary_dataframe(spec) -> pd.DataFrame` — узел / режим /
  lo / hi / ref / cap_to / cap_ratio / «компонент смеси?» (лист);
* показ `spec.phr_intervals()` и `spec.fraction_bounds()` (рассчитанные
  доли для mixture-блока — аналог таблицы режима «Массовые части»).

Связка при «🏗 Построить проект» в этом режиме:

* **имена компонентов смеси берутся из `spec.component_names`** (поле
  «Компоненты смеси» игнорируется/заполняется от спеки с подписью) —
  иначе рассинхрон имён, и раннер молча (warning) откатится на бокс;
* mixture-блок схемы — из `spec.fraction_bounds()`;
* после сборки раннера — `runner.set_phr_spec(spec)`;
* функциональные группы (iter31) в этом режиме НЕ применяются к
  сэмплингу (при заданной спеке `_phase_candidates` идёт phr-путём) —
  поле групп скрыть/подписать.

### 41.2 Паспорт кампании (сетап, до первого замера)

Секция «🪪 Паспорт кампании» в форме сетапа (и доступна после сборки):

* «Метка кампании» (`st.text_input`) → `runner.set_campaign_label`;
* «Обязательные 2D-пары» (`st.text_area`) → `runner.set_preflight_pairs`.
  Формат: одна строка = пара, стороны через `|`, ось-сумма — имена
  через запятую:

  ```
  UV_CSFCP | TiO2_BLR895
  dT | PMPlus_8, DL_531
  DINP | TiO2_BLR895
  ```

  Чистые хелперы `parse_preflight_pairs(text)` /
  `preflight_pairs_to_text(pairs)` (round-trip, образец —
  `parse_sampling_groups`/`sampling_groups_to_text`);
* показ `spec_hash` активной спеки (`st.code`, полный hex) с подписью
  «зафиксируйте хеш и лоты сырья ДО первого замера (CAMPAIGN_SPEC_PVC §3)».

### 41.3 Отражение после загрузки

* `setup_prefill_from_runner`: + `campaign_label`, `preflight_pairs`
  (текст), режим «phr-спека» с JSON `to_dicts`, если спека активна;
* `project_settings_dataframe`/`render_project_settings`: + строки о
  phr-спеке (q, dim_z, hash-префикс 12 симв.), метке, парах.

**Тесты (`test_iteration41_setup_phr_ui.py`):** парсинг JSON
(валид/ошибка конструктора наружу), summary-таблица на референсной
спеке iter35, `parse_preflight_pairs` round-trip (+ оси-суммы),
prefill из раннера с активной спекой, «имена из спеки» при сборке.

---

## iter42 — Слой навески: phr, квантование, премикс (П4)

**Файлы:** `src/design/phr_sampler.py` (одна новая функция),
`src/apps/campaign_ui.py` (таблицы seed/рецепта).

### 42.1 Логика: доли → phr

Новый метод `PhrSpec.fractions_to_phr(x) -> np.ndarray` (обратное к
`to_fractions`): масштаб от fixed-листа (`T = value_fixed / x_fixed`;
если fixed-листа нет — ValueError с объяснением, что тотал неопределим).
Тест: round-trip `fractions_to_phr(to_fractions(p)) == p` на
референсном anchor iter35.

### 42.2 UI: ввод разрешения весов

В секциях seed-дизайна и рецепта ветки (при АКТИВНОЙ phr-спеке):
поля «шаг весов, г» и «г на 1 phr» → `delta_phr = шаг / (г/phr)`
(показ рассчитанного δ; пример из CAMPAIGN_SPEC_PVC §5: 0.1 г,
5 г/phr → δ=0.02).

### 42.3 Чистый хелпер таблицы навески

`recipe_weighing_dataframe(spec, x_fractions, delta_phr)`:

| колонка | источник |
|---|---|
| компонент | `spec.component_names` |
| phr nominal | `fractions_to_phr(x)` |
| phr actual | `quantize_recipe(p, δ).p_actual` |
| граммы actual | `p_actual · г/phr` |
| премикс | `premix_required(δ, lo, hi)` по `phr_intervals()`; fixed-оси — «—» |
| нарушение | строки `violations`, относящиеся к узлу (пусто = ок) |

Подпись обязательна (CAMPAIGN_SPEC_PVC §5): **«дозируйте и фиксируйте
ФАКТИЧЕСКИЕ значения (actual); модель должна видеть actual, а не
nominal»**. `violations` показывать как `st.warning` (A0.6 — не
блокировать).

### 42.4 Интеграция

* `branch_recipe_dataframe`/`…excel_bytes`: при активной спеке и
  заданном δ — добавить лист/колонки навески (nominal/actual/премикс);
* `seed_design_dataframe`/`…excel_bytes`: то же для каждой строки плана;
* **фиксация actual**: при активной спеке и заданном δ предложенный
  план ПЕРЕД показом снапится к δ-сетке (`quantize_recipe` построчно),
  и `commit_seed`/`commit_measured` получают actual-доли
  (`to_fractions(p_actual)`) — с явной подписью, что зафиксирован
  actual-план. Violations не блокируют, но показываются.

**Тесты (`test_iteration42_weighing_ui.py`):** round-trip 42.1;
таблица навески на референсной спеке (SBM_55/UV_CSFCP → премикс,
DINP → прямая — golden §5); violations прокидываются; снап плана
перед фиксацией меняет X на actual.

---

## iter43 — Постановка откликов: пороги, chance, binding (П5)

**Файлы:** `src/apps/mixture_process_runner.py` (хранилище chance),
`src/apps/campaign_state.py` (персистентность), `src/apps/campaign_ui.py`
(редакторы целей + показ binding_report).

### 43.1 Хранилище chance-ограничений ветки

По образцу ценовой ноги (`_branch_cost` — runner-level, сериализуется
в campaign_state):

* `runner.set_branch_chance(branch_id, {prop: ChanceConstraint})`
  (+ геттер/очистка; валидация prop против `property_names`);
* `optimize_xbest` при наличии сохранённых chance для ветки подставляет
  их в `chance_constraints=` АВТОМАТИЧЕСКИ (явный аргумент вызова
  имеет приоритет);
* персистентность: `"branch_chance": {bid: {prop: {"y_min":…, "y_max":…,
  "alpha":…}}}` (asdict/распаковка `ChanceConstraint` — dataclass),
  round-trip тест.

### 43.2 Редакторы целей: новые виды

В обоих редакторах (создание ветки `render_branch_creation` и
«🎯 Редактор целей ветки») selectbox «вид» расширить:

* `max` / `min` / `target` — как есть;
* **«порог ≥» / «порог ≤»** — ввод: порог + СКО шума измерения
  (`noise_sd > 0`); построение — `hard_threshold_spec(threshold,
  noise_sd, "ge"/"le")` → обычный `DesirabilitySpec` (persists штатно).
  Подсказка: ramp = шум измерения, НЕ «узкий» (iter39, замечание 1);
* **«вероятностный Pr(y ≤ max) ≥ 1−α»** — ввод: `y_max` (и/или `y_min`),
  `alpha` → `ChanceConstraint` через `set_branch_chance` (43.1).
  В таблице целей ветки chance-ограничения показывать ОТДЕЛЬНЫМ блоком
  (это множитель к d_overall, не цель).

`draft_add_goal`/`goal_editor_dataframe` расширить соответственно
(чистые — тестируются напрямую).

### 43.3 binding_report — обязателен к просмотру

Чистый хелпер `binding_report_dataframe(report) -> pd.DataFrame`
(ограничение / тип (veto|chance) / % точек пула под биндингом /
значение в x* / порог). Показ:

* после «Рецепт ветки» (`branch_recipe_dataframe` возвращает и report —
  либо отдельная функция `branch_recipe_with_binding(...)`, чтобы не
  ломать сигнатуру/Excel);
* caption-строка: «оптимум не найден» vs «оптимум запрещён»
  (CAMPAIGN_SPEC_PVC §7).

**Тесты (`test_iteration43_goal_ui.py`):** set_branch_chance +
персистентность round-trip; `optimize_xbest` подхватывает сохранённые
chance (binding_report непустой); билдер «порог» выдаёт спеку
`hard_threshold_spec`-эквивалент; `binding_report_dataframe` структура.

---

## iter44 — Видимость (П6, фон)

* seed-цикл: caption «план пристёгнут к N существующим точкам
  (maximin-аугментация)» при `reuse_existing`-пути (чистый хелпер
  `seed_augment_caption(runner, Xs)`);
* `origin_label`: расшифровка ключей `campaign` / `spec_hash`
  (префикс 8 симв.) / `block` в ярлыке;
* preflight-caption: если заданы пары — упоминание pair-coverage.

Тесты — расширение UI-хелпер тестов (файл iter44 или дополнение iter41).

---

## Порядок и Definition of Done

| Шаг | Зависимости | DoD |
|---|---|---|
| iter40 | — | тесты 40 + регресс зелёные; save/load сохраняет политику |
| iter41 | 40 | из UI собирается проект с phr-спекой PVC; hash виден; пары/метка живут после reload |
| iter42 | 41 | таблицы seed/рецепта показывают phr nominal/actual/премикс; фиксируется actual |
| iter43 | 40 | цели-пороги и chance задаются из UI; binding_report виден после каждой оптимизации |
| iter44 | 41 | подписи аугментации/origin/pair-coverage |

## Открытые вопросы (решить по ходу, до соответствующего шага)

1. **iter41**: file_uploader для JSON спеки — нужен, или достаточно
   textarea? — **РЕШЕНО 05.08.2026: оба** (textarea + file_uploader;
   загруженный файл имеет приоритет над textarea, о чём caption);
2. **iter42**: снап предложенного плана к actual ДО фиксации — ок как
   дефолт при заданном δ, или нужен переключатель «фиксировать
   nominal»? (предлагаю дефолт actual + caption, без переключателя);
3. **iter43**: chance-ограничения на ветке — runner-level по образцу
   `_branch_cost` (предложено выше) или расширять `Branch`? (предлагаю
   runner-level: не трогаем `Branch.to_state` и канон контейнера).