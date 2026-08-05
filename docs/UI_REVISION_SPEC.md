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

---

## Ревизия контракта phr-спеки «pvc_edge_v1» (решения 05.08.2026)

Итог перепроверки таблицы 19 компонентов PVC и контракта UI→ядро,
сверено с `PhrSpec` / `_phr_spec_pvc.json`. Численно таблица корректна
(Σ phr 121,12–172,75; closure-диапазоны; немонотонности подтверждены
расчётом). Ниже — зафиксированные решения по контракту (C1–C5)
и план работ (B1–B8).

### Роли узлов (9)

| Роль | Координата | Ключи |
|---|---|---|
| `FIXED` | — | `value` |
| `ABSOLUTE` | phr | `range`, `scale?` |
| `ABSOLUTE_CAPPED` | phr | `range`, `scale?`, `cap_to[]`, `cap_ratio` |
| `GROUP_TOTAL` | phr | `range`, `members[]` |
| `GROUP_TOTAL_FIXED` | — | `value`, `members[]` |
| `SHARE_FREE` | доля | `group`, `share_range`, `min_phr?`, `max_phr?` |
| `SHARE_CLOSURE` | **нет** | `group`, `min_phr?`, `max_phr?` — только k=2 |
| `SHARE_SIMPLEX` | доля (совместно) | `group`, `share_range`, `min_phr?`, `max_phr?` — k≥3 |
| `RATIO_TO` | множитель | `reference`, `range` |

dim z = 16 (MIXTURE) + 4 (PROCESS).

### Инварианты валидатора

1. k=2 → ровно один `SHARE_CLOSURE` + один `SHARE_FREE`.
2. k≥3 → все члены `SHARE_SIMPLEX`; `SHARE_CLOSURE` запрещён (C1:
   замыкание при k≥3 — внутреннее свойство сэмплера на `SimplexRegion`,
   не роль узла; группа даёт k−1 свободных координат; отдельные
   `min_share`/`max_share` не нужны).
3. `SHARE_CLOSURE` и `FIXED` — без `range`/`share_range`; наличие —
   ошибка валидации, не тихое игнорирование.
4. `group_order` — ТОЧНАЯ перестановка множества групп с ролью
   `GROUP_TOTAL` (C2). `GROUP_TOTAL_FIXED` (RESIN) исключается: тотал
   детерминирован, стратифицировать нечего. Несовпадение множеств —
   ошибка (не «лишние игнорируются»).
5. `min_phr`/`max_phr` разрешены на любом share-узле, задаются в phr,
   в ядре превращаются в conditional narrowing (не box).
6. Share-бокс группы (C5, вместо per-node range):
   `Σφᴸ ≤ 1 ≤ Σφᵁ` и `φᵢᵁ ≤ 1 − Σ_{j≠i} φⱼᴸ ∀i` — сравнения
   **нестрогие** (LUB впритык: 0,60 = 1 − 0,40; строгое `<` даст ложный
   отказ). Для k=2 диапазон closure — производный
   `[1−φᵁ_free, 1−φᴸ_free]`, он вычисляется, не задаётся.

### Примечания к таблице компонентов

* LUB (k=3): `DL_60` 0,30–0,70; `AKLUB_K_435` 0,10–0,60; `OPE`
  0,10–0,60 — все `SHARE_SIMPLEX`; уровни не определены —
  «непрерывно (2 своб. коорд.)» (C3); `LUB.total` — отдельная ось,
  2 уровня.
* UI-пометка: `φ_AKLUB = 0,60` достижимо лишь в единственной вершине
  симплекса (остальные прижаты к нижним бондам) — мера нуль; формально
  диапазон 0,04–0,72 phr верен, практически верх недостижим (объяснить,
  иначе «почему не видел таких точек»).
* Hash `049d0e35…` — от `RECIPE_DICTS` iter35; в тесты нового контракта
  НЕ закладывать (C4). Hash плоской legacy-спеки `_phr_spec_pvc.json` был
  `44f0f96a…` (валиден до B2); после B2 файл мигрирован на схему v2
  (hash `7769ce2152788efb…`, валиден до B5); после B5 (scale='log' у
  TiO2/UV) текущий hash — `eadaca2eb84c39c3…` (изменится после B4).
  Финальный хеш пересчитать и записать **в момент сетапа**, вместе с лотами сырья
  и anchor-рецептами (CAMPAIGN_SPEC_PVC §3).

### План работ (по одному шагу на итерацию, iter45+)

| Приоритет | Шаг | Содержание |
|---|---|---|
| ✅ **сделано** (iter45, коммит ниже) | B1 | `min_phr`/`max_phr` в `PhrNode` + narrowing на share-узлах (считается ПОСЛЕ розыгрыша тотала, машинерия per-point потолка как у cap-узлов). Без этого спека генерирует нереализуемые рецепты: `PBNK_3355` до 10,5 phr (складской лимит 8,0), `CPE_135A` от 1,5 phr (техминимум 3,0) — и это штатные вершины плана, не углы. Golden: `hi_φ(T) = min(0,70; 8/T; 1−3/T)` → 0,40 @T=5; полка 0,70 на T∈[10; 11,4286]; 0,5333 @T=15 — функция **немонотонна**, тест на монотонность даст ложный отказ. Статическая валидация: непустое пересечение `[φᴸTᴸ, φᵁTᵁ]` с `min_phr`/`max_phr`. |
| ✅ **сделано** (iter46, заметки ниже) | B2+B6+B8 | Роли `SHARE_CLOSURE`/`SHARE_SIMPLEX`, исключение closure из z (22 → 16), новая JSON-схема (`role`/`group`/`members`/`reference`/`scale`/`spec_version`), валидация «closure без range». Причина: 22 координаты содержат 6 точных линейных зависимостей (пары (φ, 1−φ) идеально коллинеарны) → rank(Z)=16 при dim=22, cond→∞, VIF→∞; ARD-длины по парам не идентифицируются, preflight-гейт считает не то пространство. |
| ✅ **сделано** (iter47) | B5 | Лог-сэмплинг `TiO2_BLR895` и `UV_CSFCP`: ln z равномерно, границы логарифмируются; cap на УФ применяется **после** экспоненцирования (потолок в phr, не в логах). Причина: доля точек TiO2 < 1 phr — 9,1 % uniform против 36,7 % log; UV < 0,12 phr — 28 % против 49 %; отклик по УФ экстремально сатурирующий (при 0,12 phr A₃₄₁=3,3) — вся информация в нижней декаде. Итог: z-координата log-оси — `ln phr` (`sample_z`/`z_bounds`/`clip_z` — в лог-шкале, `decode` экспоненцирует, `encode` логарифмирует после проверок в phr); у cap-осей логарифмируется уже суженная граница `min(hi, cap_ratio·Σref)` пер-точечно; сериализация/хеш не тронуты (scale в хеше с iter46); `_phr_spec_pvc.json` мигрирован (`scale='log'` у TiO2/UV — хеш спеки изменился, это ожидаемо); маргинали проверены аналитически (36,7 %/48,9 %) — `tests/unit/test_iteration47_log_sampling.py`. |
| 🔴 | B4 (iter48) | `group_order` в модель и в `spec_hash`. Порядок влияет на меру (первая группа Dₙ≈0,02, поздние ≈0,38) — без него хеш не воспроизводит план. |
| 🟡 | B7 (iter49) | Контракт-ответ ядра на точку: `effective_bounds` (с `active` — какое ограничение сработало), `premix_required`, `phr_nominal` vs `phr_actual` раздельно. |
| — | финал | Пересчёт `spec_hash` эталонной спеки → запись при сетапе с лотами и anchor'ами. |

Замечание о взаимном порядке с невыполненными iter42–44: слой навески
(iter42) строит таблицы по `phr_intervals()`/долям — делать его ДО
B1/B2 значит переделывать после смены эффективных диапазонов и схемы.
Рекомендуемый порядок: iter45–46 (B1, B2 — ✅ сделаны) → iter42 → далее
по плану. После B4 геометрия спеки закрыта окончательно.

### Что зафиксировано реализацией B1 (iter45, `src/design/phr_sampler.py`)

Тесты: `tests/unit/test_iteration45_phr_min_max.py` (13 шт.); регресс
iter34/35/37/38/39/40/41/9 — 162 зелёных.

1. `min_phr`/`max_phr` допустимы **только на `share_of`**-узлах: у
   `absolute`/`ratio_to` границы уже заданы в собственных координатах,
   дублирующий лимит в phr там был бы вторым источником истины.
2. Публичный `PhrSpec.share_bounds_at_total(parent, T)` — эффективные
   границы долей с учётом лимитов узла И партнёров по группе
   (`loᵢ = max(lo₀ᵢ, 1 − Σ_{j≠i} hi₀ⱼ)`). Партнёрское сужение работает и
   СНИЗУ: при T=15 `lo_CPE = 1 − 8/15 = 0,4667`, а не `φᴸ = 0,30` —
   потолок партнёра поднимает чужой пол.
3. **Окно тотала.** Лимиты в phr ограничивают не только долю, но и сам
   тотал: `T ≥ min_phr/φᵁ`, `T ≤ max_phr/φᴸ`, плюс `Σlo₀(T) ≤ 1 ≤ Σhi₀(T)`
   (обе суммы невозрастают по T ⇒ бисекция даёт интервал). Окно применяется
   к `sample_z`/`clip_z`/`z_bounds`/`encode` и к `phr_intervals()`.
4. **Сужать можно только `absolute`-ось без cap.** Если окно у́же
   заявленного интервала, а тотал задан `ratio_to`/`share_of`/`absolute+cap`,
   пришлось бы сужать пер-точечно вверх по DAG — не реализовано, поэтому
   явная ошибка конфига вместо тихого приближения (A0.6).
5. **Диагностика от частного к общему**: сначала «узел X: лимиты не
   пересекаются с достижимым диапазоном», затем групповые сообщения. Для
   `fixed`-тотала пер-узловая проверка и окно эквивалентны (`t_lo == t_hi`),
   поэтому отдельной fixed-ветки нет — она была бы мёртвым кодом.
6. Лимиты входят в `to_dicts()`/`spec_hash()`, но **только когда заданы** —
   спеки без лимитов сериализуются как до iter45, хеши iter35/36 не «уехали»
   (`44f0f96a…` для `_phr_spec_pvc.json` остаётся валиден до B2).
7. `encode` отвергает рецепт вне лимитов (ошибка данных), `quantize_recipe`
   добавляет лимиты в `violations` как страховку (обычно перекрывается уже
   суженным `phr_intervals`).

### Что зафиксировано реализацией B2+B6+B8 (iter46, `src/design/phr_sampler.py`)

Тесты: `tests/unit/test_iteration46_phr_roles.py` (44 шт.); регресс
iter33/34/35/37/38/39/40/41/45/9 — 236 зелёных.

1. **Две схемы сериализации сосуществуют.** v1 (legacy, `mode`) работает
   без изменений: старые сейвы, `RECIPE_DICTS` iter35 и hash `049d0e35…`
   валидны; v2 (роли) распознаётся по ключу `role` (плюс обёртка
   `{"spec_version": 2, "nodes": [...]}`). Смешивать `mode` и `role` в
   одном списке нельзя; спека v2 не принимает legacy `share_of`.
   `PhrSpec.schema_version` (1|2) определяет формат `to_dicts()` —
   round-trip и hash стабильны в обеих схемах. Авто-конверсия v1→v2 НЕ
   делается: у legacy k=2-группы обе доли несут независимые диапазоны, и
   выбор «кто closure» изменил бы меру сэмплера молча (A0.6).
2. **Производные члены групп.** k=2: `SHARE_CLOSURE` без z-оси, диапазон
   производный `[1−φᵁ_free, 1−φᴸ_free]`; k≥3: все `SHARE_SIMPLEX`,
   зависимая координата — **ПОСЛЕДНИЙ член группы** в порядке спеки
   (порядок узлов и так часть спеки/hash). Доли группы разыгрываются
   целиком (`_narrowing_split` по всем k, та же мера, что раньше), но в z
   пишутся только свободные члены; decode восстанавливает производный как
   `1 − Σ партнёров`; clip_z реконструирует полный вектор долей, проецирует
   и пишет назад свободные координаты (идемпотентность сохранена);
   encode валидирует производные члены (включая их min/max_phr), но z для
   них не пишет. PVC-спека: dim_z 22 → 16, rank(Z)=16=dim (тест).
3. **B8-строгость схемы v2**: per-role списки обязательных/допустимых
   ключей (`_ROLE_TABLE`); `range`/`share_range` у `SHARE_CLOSURE`/`FIXED`/
   `GROUP_TOTAL_FIXED` — ошибка с объяснением; лишние ключи (в т.ч. legacy
   `lo`/`hi`/`of`/`to`) — ошибка; `members` обязаны ТОЧНО совпадать
   (состав И порядок) с узлами `group=…`; `cap_to` — только список.
4. **C5** (`φᵢᵁ ≤ 1 − Σ_{j≠i} φⱼᴸ`, нестрого) проверяется у simplex-групп;
   LUB впритык (0.60 = 1 − 0.40) проходит. Родитель группы нового
   контракта — только absolute без cap (GROUP_TOTAL) или fixed
   (GROUP_TOTAL_FIXED), в линейной шкале.
5. **`min_phr`/`max_phr` работают на всех share-ролях** (включая closure):
   единый роутинг статических границ долей через `_share_base`
   (per-point narrowing/окно тотала iter45 не менялись).
6. **`scale`**: схема принимает `linear`/`log` (`log` — только absolute,
   lo>0), ключ входит в `to_dicts`/hash; геометрические операции
   (`sample_z`/`clip_z`/`z_bounds`) до iter47 (B5) отвергали log-оси явной
   ошибкой — с iter47 лог-сэмплинг реализован (гейт снят).
7. **UI (точечно)**: `parse_phr_spec_json` принимает v2-обёртку;
   `phr_spec_summary_dataframe` показывает производный диапазон closure
   (не сентинель 0,0); иерархический редактор дерева — legacy-only
   (для v2 — явная ошибка, префилл идёт JSON-каналом). Полноценный
   v2-редактор в дереве — отдельная задача при необходимости.
