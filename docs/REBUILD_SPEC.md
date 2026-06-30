# 🔧 REBUILD SPEC — MoE/GP Pipeline для Mixture DOE (ПВХ)

> Итоговое самодостаточное ТЗ для глобальной пересборки приложения.
> **Единый источник истины:** §0–§14 (база) + §15.0–§15.5 (боевая аугментация,
> M8-argmax) + §15.6 (экономический стоп / движение границ / A0.7) — всё в этом
> файле (ранее §15.x жили в отдельных `REBUILD_SPEC_15_*.md`, теперь вшиты сюда).
> Версия 1.6 (синхронизирована с текущей сборкой: реализованы M1–M8, итерации 1–6; **M9 (PipelineTrace + MCP `doe-introspect`)** — §11, итерация 7; **golden-тесты против R 4.6.0** — §6; реализовано **Streamlit-приложение pipeline M1–M8 + чекпоинты** (итерация 8); см. §8. **§13–14 (Mixture-Process + Schema Evolution + Augmented Design)** — спроектировано, в реализации).




> Язык ядра: **Python** (`numpy / scipy / scikit-learn`), UI — Streamlit.

---

## 0. Сводка решений (зафиксировано)

| # | Решение | Реализация |
|---|---|---|
| 1 | Используем проверенную кодовую базу | Перенос рабочих M1–M3 (Scheffé OLS, ANOVA, Lenth, heredity, геометрия симплекса, parts/псевдокомпоненты). Для новых модулей — `scikit-learn` (GMM/GP). Ручную линалгебру → `numpy/scipy.linalg`. |
| 2 | R-эталон — офлайн | Фикстуры в JSON/CSV в `src/verification/golden/`. Тесты сверяются с дампами. |
| 3 | Новое ядро + перенос M1–M3 | Чистые подпакеты в `src/`. Старый код остаётся как референс. UI — Streamlit под новый pipeline. |
| 4 | **Checkpointing** | Save/Load на каждом уровне M1…M8 через `ProjectState`. |
| 5 | Синтетический полигон | q=5, истинная функция = Scheffé-quadratic (15 членов) + гауссов шум. |
| 6 | Старт | Итерация 1: linalg + M1 + M2 + M3 + golden-тесты. |

---

## 1. Архитектура pipeline (M1–M8)

```
ВХОД: компоненты q=10..12, границы L_i ≤ x_i ≤ U_i, целевые свойства, стоимость
  │
  ▼
M1 ГЕОМЕТРИЯ ОБЛАСТИ      constrained simplex → extreme vertices → псевдокомпоненты/ILR
  ▼
M2 SCREENING DESIGN      D-optimal, BATCH coordinate-exchange + restarts (модель = явный вход)
  ▼  [эксперименты]
M3 SCREENING ANALYSIS    Scheffé fit → значимость; ARD-GP lengthscales → q → q_eff ≈ 4..6
  ▼
M4 КЛАСТЕРИЗАЦИЯ РЕЖИМОВ  GMM по нормированным свойствам y → K режимов + responsibilities (BIC)
  ▼
M5 ЛОКАЛЬНЫЕ ДИЗАЙНЫ     quadratic локально, I-optimal (точный прогноз)
  ▼  [эксперименты]
M6 MoE-МОДЕЛЬ            эксперты f_k = GP(mean=Scheffé, kernel=Matérn5/2-ARD)
                         gating g_k(x) = GMM responsibilities
                         ŷ(x) = Σ g_k(x)·μ_k(x);  Var = неуверенность + разногласие
  ▼
M7 ACTIVE LEARNING       acquisition по фазе: I-opt/max-var (уточнение) или EI/LCB (поиск)
                         argmax ПО СИМПЛЕКСУ (constrained) → новые точки → назад в M6
  ▼
M8 ОПТИМИЗАЦИЯ ПРОДУКТА  desirability d_overall = (Π d_i)^(1/m) + стоимость, на симплексе
```

**Две петли обратной связи:**
- M6 ↔ M7 (active learning): дозабор точек, пока max-неопределённость > порога ИЛИ EI даёт улучшение > ε.
- M2 ← M3/M4: если q_eff велик или найдены неожиданные режимы — вернуться к M2 с уточнённой моделью.

---

## 2. Сводная таблица «что чем считать»

| Модуль | Метод | Критерий | R-эталон | Python |
|---|---|---|---|---|
| M1 геометрия | extreme vertices, псевдокомпоненты/ILR | покрытие области | `mixexp::Xvert` | `core/simplex.py` |
| M2 screening design | **D-optimal** batch coord-exchange | значимость коэффициентов | `AlgDesign::optFederov("D")` | `design/d_optimal.py` |
| M3 screening fit | Scheffé OLS + ARD-GP | p-values + lengthscales | `mixexp::MixModel`, `lm`, `anova` | `models/scheffe.py`, `models/screening.py` |
| M4 кластеры | **GMM** по свойствам | BIC | `mclust::Mclust` | `models/clustering.py` (sklearn) |
| M5 local design | **I-optimal** | точность прогноза | `AlgDesign::optFederov("I")` | `design/i_optimal.py` |
| M6 эксперты | **GP** (mean=Scheffé) | marg. likelihood + restarts | `DiceKriging`, `GauPro` | `models/gp_expert.py` (sklearn) |
| M6 gating | GMM responsibilities | — (из M4) | `mclust` | `models/moe.py` |
| M7 active learning | EI / LCB / max-variance | по фазе | `DiceOptim::EI` | `design/active_learning.py` |
| M8 оптимизация | desirability | d_overall max | `desirability::dOverall` | `optimize/desirability.py` |

---

## 3. Главные формулы

**Контроль сложности:** `p_quad = q + C(q,2)`, `n ≳ p + 5..10 + реплики`.

**Дисперсия прогноза (полином):** `ν(x) = xᵀ(XᵀX)⁻¹x`, `det(X_newᵀX_new) = det(XᵀX)(1+ν(x))`.

**GP прогноз:**
`μ(x*) = k*ᵀ(K+σ_n²I)⁻¹y`, `σ²(x*) = k** − k*ᵀ(K+σ_n²I)⁻¹k*`.

**Matérn 5/2 (ARD):**
`k(x,x') = σ_f²·(1+√5 r + 5r²/3)·exp(−√5 r)`, где `r = √(Σ_i (x_i−x_i')²/ℓ_i²)`.

**RBF (опция, «гладкий»):** `k(x,x') = σ_f²·exp(−r²/2)`, `r` как выше.

**MoE прогноз и неопределённость:**
`ŷ(x) = Σ_k g_k(x)μ_k(x)`,
`Var[ŷ] = Σ_k g_k σ_k²  (неуверенность)  +  Σ_k g_k(μ_k−ŷ)²  (разногласие)`.

**Expected Improvement (минимизация y*):**
`EI(x) = (y*−μ)Φ(z) + σφ(z)`, `z = (y*−μ)/σ` (для максимизации — знак меняется).

**Desirability:** `d_overall = (Π_{i=1}^m d_i)^{1/m}` (Derringer–Suich one/two-sided).

---

## 4. Выбор ядра GP (M6) — контракт о гладкости

Ядро — **единственное место, где явно зашивается предположение о гладкости** физического отклика. Ошибка здесь → GP либо «уверенно врёт», либо осциллирует.

**RBF vs Matérn — что отличается.** Оба стационарны (зависят от `r = ‖x−x'‖`), оба дают гладкую интерполяцию; разница — в *степени* предполагаемой гладкости. У Matérn её задаёт параметр ν:

| ν | Гладкость функции | Физический смысл |
|---|---|---|
| ν → ∞ | бесконечно гладкая = **RBF** | идеальная парабола |
| ν = 5/2 | дважды дифференцируема | гладкая, но с реалистичными изломами |
| ν = 3/2 | один раз дифференцируема | заметно «шершавее» |
| ν = 1/2 | недифференцируема (Орнштейн–Уленбек) | почти случайное блуждание |

**Дефолт для ПВХ — Matérn 5/2 с ARD**, не RBF:
- физсвойства композиций не бесконечно гладкие (пороги, насыщение, перегибы) → RBF занижает σ между точками → active learning «думает, что всё знает» и перестаёт исследовать;
- Matérn 5/2 дважды дифференцируема (хватает производных для оптимизации), честнее по неопределённости, лучше ловит локальные изломы.

**Конфигурация по умолчанию:**
- **mean = Scheffé quadratic** (интерпретируемый тренд);
- **kernel = Matérn5/2-ARD (на остатках) + WhiteNoise(σ_n²)**;
- координаты: **псевдокомпоненты или ILR** (q−1), иначе K вырождается из-за Σx_i=1;
- ARD-lengthscales ℓ_i — вторично как screening-индикатор (большой ℓ_i → компонент неважен; масштаб относится к *преобразованным* координатам).

**ARD обязателен** (отдельный ℓ_i на компонент): у каждого компонента свой масштаб влияния — стабилизатор резко в малых дозах (малый ℓ_i), инертный наполнитель плавно (большой ℓ_i). Один общий ℓ усредняет это в кашу.

**Гиперпараметры:** maximize marginal likelihood, **restarts ≥ 10**. Границы: ℓ_i ≥ мин. межточечное расстояние (иначе осцилляции) и ≤ размер области; σ_f² ≈ var(y); σ_n² ≥ 1e-6 (jitter — иначе K необратима и переобучение шума), верхняя ≈ оценка ошибки из реплик.

**Композиция ядер** (`+Linear`, сумма Matérn разных ℓ) — только если простое ядро дало плохой fit. Альтернатива `+Linear` — уже заложенное полиномиальное среднее Шеффе: тренд ловит mean, ядро моделирует только остатки (чище и интерпретируемее).

**Опция RBF** — для сравнения / режима «гладкий отклик».

**Реализация:** Scheffé OLS (mean, переиспользуем M3) + `sklearn GaussianProcessRegressor(Matern(nu=2.5, length_scale=[ℓ_i]) + WhiteKernel)` на остатках в ILR/псевдо-координатах. R-эталон сверки: `DiceKriging::km(covtype="matern5_2")` / `GauPro` — сверять μ(x*), σ²(x*) при фикс. гиперпараметрах (atol 1e-6).

---

## 5. Слой персистентности (checkpointing)

`ProjectState` — единое сериализуемое состояние проекта:
```
project_<name>/
├── state.json        # стадия, конфиг (q, имена, границы, свойства, цели, стоимость), seed, history
├── data/             # общая база точек X + отклики Y (n×P) + origin-теги (CSV)
├── models/           # параметры общих суррогатов проекта (property → модель; JSON / NPZ)
└── checkpoints/      # снимки состояния на каждом уровне M1..M8
```
- Каждый модуль: `to_state(state)` / `from_state(state)`.
- `ProjectState.save(path)` / `ProjectState.load(path)` — атомарная запись.
- Streamlit: на каждой стадии «💾 Сохранить чекпоинт» / «📂 Загрузить чекпоинт»; `st.session_state` ↔ диск.
- Остановка после любого M-модуля и продолжение ровно с точки сохранения.

**Общая база точек проекта (мультиотклик/ветки, §12) — first-class сущность.**
Состояние хранит **единую базу** `X` всех экспериментов проекта вместе с
матрицей откликов `Y ∈ ℝ^{n×P}` (столбец на свойство) и именами свойств. База
общая: на ней обучаются **общие суррогаты** проекта (словарь `property → модель`
в `models/`), а не отдельные модели веток.
- **Origin-теги.** Каждая точка несёт тег происхождения (`origin`: `init` —
  стартовый D-/I-дизайн, либо `branch:<имя>` — добор конкретной ветки). Это **одна
  точка с тегом**, а не копии: для обучения используется вся база, тег нужен лишь
  для истории/арбитража. Заводить две копии точки запрещено.
- **Ветки — без собственных моделей.** Ветка хранит только цель
  (`DesirabilitySpec` по свойствам), бюджет, статус и ссылки на свои добранные
  точки (через origin-тег), но **не** контейнер модели — прогнозы всегда из общих
  суррогатов проекта. Цели/истории/рецепты веток сериализуются в `state.json`.
- Снимок конфигурации (`config_snapshot`) включает свойства и ветки, чтобы
  проект, общая модель и его ветки восстанавливались целиком.

---

## 6. Верификация через R (golden tests, офлайн)

Принцип: фиксируем вход (seed + данные) → гоняем свой модуль и R-эталон → сверяем с допуском.
Офлайн: один раз в R генерим фикстуры → дамп в `src/verification/golden/*.json` → тесты сверяются с дампами (R больше не нужен).

**Допуски:**
| Величина | Допуск |
|---|---|
| коэффициенты Scheffé (OLS) | atol = 1e-8 |
| D-/I-эффективность (по det) | ~1–2% (алгоритм стохастический, не точка-в-точку) |
| GP-прогноз μ, σ (при фикс. гиперпарам.) | atol = 1e-6 |
| GMM responsibilities | метки + вероятности |
| desirability | atol = 1e-8 |

**Edge cases:** вырожденные ограничения, почти коллинеарные компоненты, L_i=0, вершины симплекса, n≈p.

**Статус реализации (боевой масштаб, эталон — R 4.6.0).** Фикстуры генерируются
один раз штатным **R** (`src/verification/r/compute_reference.R`, base R: `lm` /
`qr` / `det` / `solve` + явные формулы Деррингера–Сюича и GP-постериора Matérn5/2
ARD — без внешних пакетов, чтобы не зависеть от сетевой установки) через
оркестратор `generate_golden.py` (обмен Python↔R по CSV). Если R недоступен —
fallback на независимый Python-эталон `reference.py` (тот же мат-контракт, числа
совпадают). Результат коммитится в `src/verification/golden/*.json` (поле
`engine` фиксирует источник: `R-base`), тесты `tests/golden/test_golden.py`
сверяют продакшн с дампами и **R при прогоне не требуют**. Регенерация:
`python -m src.verification.generate_golden` (авто-поиск Rscript; либо
`--engine python`, либо `DOE_RSCRIPT=<путь>`).

**Боевой масштаб покрытия:** Scheffé quadratic **q=5 → p=15** (OLS-коэффициенты,
`lm`, atol 1e-8; n=30 точек), D-/I-критерии (det/trace, 15×15), desirability
(Деррингер–Сюич, 1e-8), **GP Matérn5/2 ARD с 5 длинами** при фикс. θ (μ/σ vs R,
1e-6), GMM в 2D-пространстве свойств (BIC → K=3). Итог: `pytest tests/golden/` →
**7 passed**.



---

## 7. Критические инженерные решения (грабли)

1. Модель — **ВХОД** генератора дизайна (не зашивать квадрат в алгоритм).
2. **BATCH** coordinate-exchange, не greedy sequential.
3. Sequential (active learning) — только дозабор/поиск, не построение базового дизайна.
4. `n > p` с запасом; предупреждать при `n≈p` (иначе R²=1 обманывает).
5. **D** для screening, **I** для уточнения.
6. Кластеры — в пространстве **свойств**; прообраз в рецептурах — через модель, не по близости.
7. Якорные точки вне кластеров (модель должна знать границы режимов).
8. Работа в **псевдокомпонентах / q−1 / ILR** (иначе XᵀX и K вырождены).
9. **Multiple restarts** везде (D-opt, GMM/EM, GP-гиперпараметры).
10. argmax acquisition — **на симплексе с ограничениями** (candidate set или проекция).
11. Нижние границы на ℓ и σ_n² у GP.
12. GP с полиномиальным средним (интерпретируемость + честная σ).
13. Cubic — только точечно после screening на q_eff.

---

## 8. MVP-план (итерации) и статус сборки

| Итерация | Содержание | Модули | Статус |
|---|---|---|---|
| **1 (скелет)** | linalg + геометрия + D-opt batch + Scheffé fit | M1, M2, M3 | ✅ реализовано |
| **2 (снижение размерности)** | ARD-GP screening → q_eff + I-optimal | M3(ARD), M5 | ✅ реализовано |
| **3 (гибкая модель)** | GP-эксперт (K=1, без gating) | M6 (GP) | ✅ реализовано |
| **4 (режимы)** | GMM + полный MoE (K>1) | M4, M6 (MoE) | ✅ реализовано |
| **5 (active learning)** | дозабор/поиск (max-var, затем EI) на симплексе | M7 | ✅ реализовано |
| **6 (продукт)** | desirability + мультикритерий + стоимость | M8 | ✅ реализовано |
| **7 (наблюдаемость)** | PipelineTrace + MCP-сервер `doe-introspect` | M9 | ✅ реализовано |
| **8 (UI)** | Streamlit pipeline M1–M8 + чекпоинты + benchmark | UI | ✅ реализовано |


> Полигон Итерации 1: q=5, истинная Scheffé-quadratic с заданными коэффициентами + шум → измеряем восстановление.
> Примечание по нумерации: M7 (active learning) вынесен в отдельную итерацию 5, M8 (продукт) — в итерацию 6 (изначально оба планировались в одной итерации 5).

**Соответствие модуль → файл → тест → демо (текущая сборка):**

| Модуль | Файл | Тест | Демо |
|---|---|---|---|
| M1 | `core/simplex.py` (+ `linalg.py`, `synthetic.py`, `state.py`) | `test_iteration1.py` | `run_iteration1_demo.py` |
| M2 | `design/d_optimal.py` | `test_iteration1.py` | `run_iteration1_demo.py` |
| M3 | `models/scheffe.py`, `models/screening.py` | `test_iteration1/2.py` | `run_iteration1/2_demo.py` |
| M5 | `design/i_optimal.py` | `test_iteration2.py` | `run_iteration2_demo.py` |
| M6 эксперт | `models/gp_expert.py` | `test_iteration3.py` | `run_iteration3_demo.py` |
| M4 | `models/clustering.py` | `test_iteration4.py` | `run_iteration4_demo.py` |
| M6 MoE | `models/moe.py` | `test_iteration4.py` | `run_iteration4_demo.py` |
| M7 | `design/active_learning.py` | `test_iteration5.py` | `run_iteration5_demo.py` |
| M8 | `optimize/desirability.py` | `test_iteration6.py` | `run_iteration6_demo.py` |
| M9 | `observability/trace.py`, `mcp/queries.py`, `mcp/introspect_server.py` ✅ | `test_iteration7.py`, `test_iteration7_mcp.py` ✅ | `run_iteration7_demo.py` ✅ |
| UI | `apps/streamlit_app.py`, `apps/pipeline_runner.py` ✅ | `test_iteration8_app.py`, `test_iteration8_streamlit.py` ✅ | `run_streamlit_app.py` ✅ |


Прогон: `pytest tests/unit/test_iteration*.py tests/golden/` → **84 passed**
(M1–M8: 52; M9: 13; UI: 12 — `test_iteration8_app` 10 + `test_iteration8_streamlit` 2; golden: 7).

Чекпоинты `ProjectState`: `after_M1`…`after_M8` (демо — в `project_demo/checkpoints/`,
UI — в `project_ui/<name>/checkpoints/`).

**Статус по спеке:** все пункты ТЗ реализованы (M1–M9 + golden + Streamlit-UI).
Возможные развития (вне ТЗ): сквозные integration-тесты, реальные R-пакеты
(`DiceKriging`/`mclust`) для golden, экспорт отчётов из UI.



---

## 9. Целевая структура репозитория

Легенда: ✅ реализовано · ⬜ запланировано.

```
src/
├── core/
│   ├── linalg.py        # ✅ numpy/scipy линалгебра (замена ручной)
│   ├── simplex.py       # ✅ M1: vertices, псевдокомпоненты, ILR
│   ├── synthetic.py     # ✅ синтетический полигон (известная Scheffé-функция)
│   └── state.py         # ✅ ProjectState (персистентность/checkpointing)
├── design/
│   ├── d_optimal.py     # ✅ M2: D-opt batch coordinate-exchange + restarts
│   ├── i_optimal.py     # ✅ M5: I-opt локальный
│   └── active_learning.py  # ✅ M7: EI / LCB / max-variance на симплексе
├── models/
│   ├── scheffe.py       # ✅ M3: Scheffé OLS + ANOVA
│   ├── screening.py     # ✅ M3: значимость (Lenth/heredity/FS) + ARD-GP
│   ├── gp_expert.py     # ✅ M6: GP-эксперт (mean=Scheffé, kernel=Matérn5/2-ARD)
│   ├── clustering.py    # ✅ M4: GMM по свойствам + BIC
│   └── moe.py           # ✅ M6: MoE сборка + неопределённость
├── optimize/
│   └── desirability.py  # ✅ M8: d_overall + стоимость на симплексе
├── observability/
│   └── trace.py         # ✅ M9: PipelineTrace — структурные события стадий + запись на диск
├── mcp/
│   ├── queries.py            # ✅ M9: чистая логика выборки из trace (без зависимости от mcp)
│   └── introspect_server.py  # ✅ M9: MCP-сервер doe-introspect (tools/resources, read-only)

├── verification/        # ✅ §6: офлайн golden против R 4.6.0 (боевой q=5)
│   ├── r/compute_reference.R # ✅ R-эталон (base R: lm/det/solve/GP/desirability)
│   ├── reference.py         # ✅ Python-эталон fallback (QR-OLS, D/I, desirability, GP)
│   ├── golden_io.py         # ✅ загрузка/сохранение фикстур + реестр допусков
│   ├── generate_golden.py   # ✅ оркестратор Python↔R (CSV); R-сниппеты в provenance
│   └── golden/              # ✅ закоммиченные R-фикстуры (*.json, engine=R-base)

└── apps/
    ├── pipeline_runner.py # ✅ UI-оркестратор M1–M8 (без Streamlit, тестируемый)
    └── streamlit_app.py   # ✅ UI: стадии M1–M8 + чекпоинты + benchmark

tests/
├── unit/                # ✅ test_iteration1..6 (M1..M8); test_iteration7(+_mcp) (M9); test_iteration8(+_streamlit) (UI)
├── integration/         # ⬜ сквозной pipeline
└── golden/              # ✅ test_golden.py — сверка продакшна с R-фикстурами



project_<name>/
└── trace/               # ✅ M9: события стадий (JSON) + index.json (источник для MCP)

```


> Отличия от первоначального плана: `active/acquisition.py` → `design/active_learning.py`;
> `models/gp.py` → `models/gp_expert.py`; `models/gmm.py` → `models/clustering.py`.

---

## 10. Контракты модулей (сигнатуры)

```python
# core/simplex.py (M1)
class SimplexRegion:
    def __init__(self, lower, upper): ...
    def extreme_vertices(self) -> np.ndarray: ...
    def to_pseudo(self, x) -> np.ndarray: ...
    def from_pseudo(self, w) -> np.ndarray: ...
    def to_ilr(self, x) -> np.ndarray: ...
    def from_ilr(self, z) -> np.ndarray: ...
    def is_feasible(self, x) -> bool: ...

# design/d_optimal.py (M2)
def d_optimal_design(candidates, n_runs, model, n_restarts=10, seed=None) -> DesignResult
# model: 'linear' | 'quadratic' | 'cubic' | список членов — ЯВНЫЙ вход

# models/scheffe.py (M3)
class ScheffeModel:
    def fit(self, X, y, model='quadratic') -> self
    def predict(self, X) -> np.ndarray
    def anova(self) -> pd.DataFrame
    @property coefficients, r2, adj_r2, rmse

# models/screening.py (M3): значимость членов (Lenth/heredity/FS) + ARD-GP lengthscales → q_eff
# design/i_optimal.py (M5): def i_optimal_design(candidates, n_runs, model, ...) -> DesignResult

# models/gp_expert.py (M6)
class GPExpert:     # mean=Scheffé, kernel=Matérn5/2-ARD на остатках (+ RBF опция)
    def fit(self, X, y) -> self
    def predict(self, X, return_std=True) -> GPPrediction(mean, std)
    @property lengthscales            # ARD → screening
    def to_state() / from_state()     # checkpointing

# models/clustering.py (M4)
class GMMRegimes:                     # кластеры в пространстве СВОЙСТВ, выбор K по BIC
    def fit(self, y) -> self
    @property n_regimes, labels, responsibilities
    def to_state() / from_state()

# models/moe.py (M6)
class MixtureOfExperts:               # GMM-gating (в recipe-пространстве) + GP-эксперты
    def fit(self, X, y) -> self
    def predict(self, X) -> MoEPrediction(mean, std, uncertainty, disagreement, gating, expert_means)
    @property n_regimes
    def to_state() / from_state()

# design/active_learning.py (M7)
def active_learning_loop(region, oracle, X0, y0, n_iter, acquisition, batch, ...) -> ActiveLearningResult
# acquisition: 'max_std' | 'max_disagreement' | 'ei' | 'lcb'; argmax на candidate set симплекса

# optimize/desirability.py (M8)
@dataclass
class DesirabilitySpec:   # kind: 'max' | 'min' | 'target'; low, high, target, s, s2, weight
def optimize_desirability(region, predictors, specs, cost_fn=None, cost_spec=None,
                          n_candidates=4000, refine_iters=400, seed=None) -> DesirabilityResult
# d_overall = (Π d_i^{w_i})^{1/Σw_i};  стоимость = ещё одно 'min'-свойство;  argmax на симплексе

# observability/trace.py (M9)
@dataclass
class StageEvent:                     # одно структурное событие стадии pipeline
    run_id: str; stage: str; ts: str  # stage: 'M1'..'M8' | 'AL_round_{i}' | 'opt'
    inputs: dict                      # снимок конфига/параметров стадии
    outputs: dict                     # дизайн-точки, коэффициенты, прогнозы, рецепт
    metrics: dict                     # D-eff/I-eff, R²/adjR², BIC, q_eff, d_overall, cost, ...
    diagnostics: dict                 # σ, disagreement, residuals, acquisition, near-miss-флаги

class PipelineTrace:                  # сборщик событий + запись на диск (read-only снаружи)
    def __init__(self, run_id, root='project_<name>/trace'): ...
    def log(self, stage, *, inputs, outputs, metrics, diagnostics) -> StageEvent
    def save(self) -> None            # JSON на стадию + index.json
    @classmethod
    def load(cls, root, run_id) -> 'PipelineTrace'
    def stages(self) -> list[str]
    def get(self, stage) -> StageEvent

# mcp/introspect_server.py (M9) — MCP-сервер 'doe-introspect' (read-only)
# Resources:  doe://runs                         список прогонов
#             doe://run/{id}/stages              стадии прогона
#             doe://run/{id}/stage/{stage}       полное событие стадии
# Tools:      list_runs()                        -> [run_id, ...]
#             get_stage(run_id, stage)           -> StageEvent (inputs/outputs/metrics/diag)
#             get_metrics(run_id)                -> сводка метрик по всем стадиям
#             get_design(run_id, stage)          -> точки дизайна + отклики
#             get_predictions(run_id, points)    -> μ/σ MoE в заданных точках
#             diff_rounds(run_id, a, b)          -> дельта метрик/рецепта между раундами AL
#             get_benchmark(run_id)              -> аналитический оптимум vs pipeline (gap, ‖Δ‖)
```

---

## 11. Слой наблюдаемости и MCP-интроспекция (M9)

**Зачем.** Pipeline M1–M8 — это последовательность стадий, на каждой из которых принимаются статистические и инженерные решения (какой дизайн, сколько режимов, где дозабирать, какой рецепт принять). Чтобы накапливать опыт и системно дорабатывать алгоритм, нужен **сквозной структурный след (trace)** каждой стадии и **программный доступ к нему**, который ассистент использует в двух ролях:
1. **Аналитик** — комментирует статистическое здоровье стадии: достаточно ли `n` против `p`, не вырожден ли дизайн (D-eff), адекватно ли число режимов K (BIC), не застрял ли active learning, нет ли near-miss по спекам, какова цена/качество прогноза.
2. **Разработчик** — по тем же данным находит алгоритмические узкие места (например: safety-margin сажает в дорогой угол; шумовой ложный приём инкумбента) и формулирует конкретные правки кода. Накопленные наблюдения → backlog оптимизаций.

**Принципы.**
- **Не инвазивно и read-only.** M9 только *читает* и *логирует*; он не меняет решения pipeline. Источник истины для алгоритма остаётся `ProjectState` (§5); trace — его наблюдаемый сосед.
- **Структурно и сериализуемо.** Каждая стадия эмитит `StageEvent` (inputs / outputs / metrics / diagnostics) → JSON в `project_<name>/trace/` + `index.json`. Формат совместим с чекпоинтами (§5): одни и те же seed/конфиг → воспроизводимый trace.
- **Дёшево.** Логирование — это сериализация уже посчитанных величин (без повторных тяжёлых вычислений); тяжёлые объекты (модели) сохраняются ссылкой на `models/`.

**Содержимое `StageEvent` по стадиям (минимум):**

| Стадия | metrics | diagnostics |
|---|---|---|
| M1 геометрия | n_vertices, объём области | покрытие/вырожденность ограничений |
| M2 D-opt | D-eff, n, p, n/p | условие XᵀX, дубликаты |
| M3 fit | R², adjR², RMSE, p-values | остатки, n≈p флаг |
| M3 ARD | q_eff, ℓ_i | ранжирование компонентов |
| M4 GMM | K, BIC | размеры кластеров, перекрытие |
| M5 I-opt | I-eff | прогнозная дисперсия |
| M6 MoE | per-expert marg.LL | σ (uncertainty) + disagreement |
| M7 AL раунд | acquisition max, Δ цели | стагнация, σ̃ по раундам |
| M8 продукт | d_overall, cost, рецепт | d_i по свойствам, near-miss-флаги |
| benchmark | price_gap, ‖Δrecipe‖, meets | analytical vs pipeline |

**MCP-сервер `doe-introspect`** (контракт — §10): отдаёт trace как resources (по URI) и tools (точечные запросы). Это позволяет ассистенту, не имея прямого доступа к процессу, получать данные любой стадии и сравнивать раунды/прогоны.

**Транспорт/реализация.** Python MCP SDK, stdio-сервер; конфиг в настройках MCP. Зависимость изолирована (`mcp` в extras, не тянется в ядро pipeline). Тесты (`test_iteration7.py`): round-trip `PipelineTrace.save/load`, корректность метрик стадий, контрактные ответы tools на фикстуре одного прогона `run_pipeline_benchmark`.

**План (итерация 7):**
1. `observability/trace.py` — `StageEvent` + `PipelineTrace` (log/save/load) и встроить логирование в `run_pipeline_benchmark` (и далее в реальный pipeline) — **без изменения логики**.
2. `mcp/introspect_server.py` — сервер `doe-introspect` с resources/tools (§10).
3. `test_iteration7.py` + `run_iteration7_demo.py` (прогон → trace → выборка через tools).
4. Зарегистрировать сервер в MCP-конфиге, проверить сквозной доступ.

---

## 12. Мультиотклик и ветки-рецепты (итерация 9)

**Зачем.** Реальная задача — не один отклик, а **несколько целевых показателей**
(P свойств продукта). Цель пользователя — получать **рецептуры под конкретные
наборы целевых значений**. Единого «первого свойства» нет: сначала общий
*screening* по всем свойствам выявляет главные эффекты, затем работа расходится
на **ветки** — каждая ветка есть целевой набор (таргеты по свойствам) и
независимо дорабатывается, пока модель не даст приемлемый результат.

**Ключевой архитектурный принцип.** **Модель физики — единая и общая на проект**
(словарь `property → суррогат`). Она обучается на **объединении точек всех веток**
`X = ⋃_b X_b`. Ветка — это **целевая траектория сбора точек**, а не контейнер для
собственной модели. «Специализация под рецепт» выражается в **acquisition-функции
ветки** (desirability её цели), а не в отдельной модели физики.
Обоснование: несмещённость суррогата не зависит от стратегии выбора точек —
расположение точек влияет на дисперсию прогноза `σ(x)`, но не на смещение `ŷ(x)`;
поэтому «предвзятый» под цель сбор точек не портит модель, а объединение точек
разных целей **улучшает покрытие** области и удешевляет каждую следующую ветку.

**Данные.** Эксперимент даёт матрицу откликов `Y ∈ ℝ^{n×P}` (один столбец на
свойство). Имена свойств — конфиг проекта. Демонстрационная «лаборатория» —
`MultiSyntheticScheffe`: P независимых Scheffé-функций (разные seed), мерит сразу
все свойства; в UI отклики НЕ заполняются автоматически (кнопка «Заполнить
тестовыми»).

**Общие стадии (M1–M6) — по всем свойствам сразу:**
- M1 геометрия — без изменений.
- M2 D-opt дизайн + сбор `Y` (P редактируемых столбцов `y`).
- M3 screening — Scheffé-fit + ARD **на каждое свойство**; сводка главных
  эффектов: какие компоненты на какие свойства влияют.
- M4 режимы — GMM в P-мерном пространстве свойств (уже многомерно).
- M5 I-opt — геометрия, без изменений.
- M6 суррогаты — **MoE/GP на каждое свойство** (словарь `property → модель`),
  **общий на проект** (см. принцип выше).

**Ветки (развилка на M7).** Ветка = `{имя, спецификации цели по свойствам,
ограничения, бюджет, история собственных точек, статус}`, где спецификация
свойства — `kind ∈ {max,min,target}` + границы/таргет + вес (контракт
`DesirabilitySpec`, §3/§10). Ветка **хранит свою цель и историю своего добора
точек, но НЕ хранит собственную модель** — прогнозы всегда берутся из общих
суррогатов проекта. В каждой ветке:
- **Active learning** ведёт acquisition по **desirability целевого набора ветки**
  `d_overall(x) = (Π_i d_i(ŷ_i(x))^{w_i})^{1/Σw}`, где `ŷ_i(x)` — прогнозы из
  **общих** суррогатов всех свойств. Новые точки оракул меряет по ВСЕМ свойствам,
  они добавляются в **общую базу проекта** (и одновременно учитываются в истории
  ветки как её добор). **Общие суррогаты дообучаются на объединённой базе всех
  веток** — улучшение от точек одной ветки доступно всем веткам. Итерации — пока
  прирост desirability/неопределённость не станут приемлемыми (порог/максимум).
- **Контроль misspecification.** Если добранная точка попадает в зону вне всех
  режимов (все `g_k(x)` малы) — флаг «возможный новый режим»: кандидат на
  добавление эксперта (`K+1`) и/или перекластеризацию M4. Несмещённость держится
  лишь при верной модели; выход за освоенные режимы — единственный реальный
  источник смещения.
- **M8 ветки** — финальный рецепт (argmax desirability на симплексе) + достигнутые
  свойства и `d_i` по критериям. Поиск argmax — **мультистартовый** (целевая зона
  рецептур может иметь несвязный прообраз).
- Ветки **сравниваются между собой по достигнутой desirability**; они **разделяют
  общую модель и общую базу точек**, но имеют независимые цели, истории добора и
  итоговые рецепты.

**Обновление общей модели.** При добавлении точки — инкрементальное обновление
суррогатов при фиксированных гиперпараметрах (`O(n²)`); полная переоптимизация
гиперпараметров (с restarts) — периодически или по триггеру (накоплено N новых
точек / сработал флаг misspecification).

**Арбитраж бюджета (если веток несколько, а слотов мало).** Следующий эксперимент
отдаётся ветке с максимальным нормированным на приоритет/бюджет значением
acquisition (`EI`/прирост desirability). Набор веток работает как **единый портфель
active learning** над общей моделью. *(Детализация — отдельный блок, при
необходимости.)*

**Персистентность.** Свойства, **общая база точек проекта** и ветки (их цели,
добранные точки, статус, рецепты) хранятся в `ProjectState` (§5) и попадают в
снимок конфигурации (`config_snapshot`), чтобы проект, общая модель и его ветки
восстанавливались целиком.

**План реализации (под-итерации):**
- **3a** ядро мультиотклика: `MultiSyntheticScheffe`, `Y (n×P)`, имена свойств; M2 собирает `Y`.
- **3b** M3 per-property screening + M6 суррогат на каждое свойство.
- **3c** модель ветки + branch active learning (acquisition по desirability) + M8-рецепт ветки; персист веток.
- **3d** UI: конфиг свойств, M2 с несколькими `y`, менеджер веток (создать целевые наборы → запустить → сравнить рецепты).

---

## 13–14. Mixture-Process + Sequential Augmented Design (объединённая итерация)

> §13 (Mixture-Process) и §14 (Schema Evolution + Augmented Design) сведены в
> **один реализуемый контракт**: обе фичи пересекаются в одной точке — требуют,
> чтобы **точка плана была составным версионированным объектом, а генератор
> дизайна — поблочным с поддержкой fixed-строк**. Это общий фундамент, на котором
> стоят обе фичи; реализовывать по отдельности нельзя — будет два несовместимых
> формата точки.

**Зачем.** До сих пор пространство планирования = симплекс рецепта (Σx=1). Реальные
задачи почти всегда содержат **процессные параметры** (температура, время, давление,
скорость, pH…), не связанные массовым ограничением. Обобщение пространства до
**произведения симплекса и гиперкуба** превращает узкий рецептурный инструмент в
общий **mixture-process DoE-движок**. Граничные случаи покрываются тем же механизмом:

| Режим | mixture-блок | process-блок | Эквивалент |
|---|---|---|---|
| **Mixture-only** (текущий) | есть | нет | Scheffé / симплекс-дизайн |
| **Mixture-process** (целевой) | есть | есть | mixture×process designs |
| **Process-only** («жёсткий») | нет | есть | классический RSM / факторный дизайн на кубе |

> **Инвариант.** mixture-only и process-only — **частные случаи** общего поблочного
> механизма, а НЕ отдельные ветки кода. Если появилось `if process_only: …` на
> уровне алгоритмов дизайна/модели — это ошибка архитектуры. Различие живёт только
> в **наборе блоков** и их правилах.

### 13.0 Главный объединяющий инвариант

```
ЕДИНЫЙ ФУНДАМЕНТ:
  • Точка = составной объект {blocks: X, responses: Y} + schema_version + origin_tag
  • Генератор дизайна = поблочный (mixture/process) + поддержка FIXED-строк
  • Критерий остановки = §5.5 (ΔI/I), I на ТЕКУЩЕЙ модели и ТЕКУЩЕМ объединённом дизайне

Mixture-Process = «какие блоки есть в схеме»
Augmented design = «как добрать точки к уже собранным при той же/новой схеме»
```

### 13.1 Структуры данных (фундамент обеих фич)

```python
# ---- Схема (версионируется) ----
class VariableBlock:
    kind: Literal["MIXTURE", "PROCESS"]
    names: list[str]
    # MIXTURE: L_i, U_i покомпонентные (опц.), constraint = simplex
    # PROCESS: (L_j, U_j) на фактор, constraint = box; + code<->real
    bounds: dict[str, tuple[float, float]]

class ResponseSpec:
    name: str
    kind: Literal["max", "min", "target"]
    target: float | None
    weight: float

class ModelSpec:
    cross_level: Literal["additive", "cross-main", "full-cross"]  # дефолт cross-main
    process_order: Literal["linear", "quadratic"]                 # дефолт quadratic

class ProjectSchema:
    version: int
    blocks: list[VariableBlock]        # 0..1 MIXTURE, 0..1 PROCESS (оба пустых запрещены)
    responses: list[ResponseSpec]
    model: ModelSpec

# ---- Точка (составная, версионированная) ----
MISSING = object()  # явный сентинел, НЕ None, НЕ 0.0

class DataPoint:
    schema_version: int                       # в какой схеме собрана
    X: dict[str, list[float]]                 # {"MIXTURE": [...], "PROCESS": [...]}  (PROCESS — в коде [0,1])
    Y: dict[str, float | MISSING]             # MISSING допустим ПОКОЛОНОЧНО
    origin_tag: dict                          # branch_id, stage, schema_version
    fixed_in_augment: bool = False            # пометка при использовании как опорной

# ---- Состояние проекта ----
class ProjectState:
    schema_history: list[ProjectSchema]       # все версии, неизменяемые
    current_schema_version: int
    points: list[DataPoint]                   # точки разных версий сосуществуют
    surrogates: dict[str, GPModel]            # per-response, на ОБЪЕДИНЁННОЙ базе
    branches: list[Branch]
    config_snapshot: dict                     # seeds, гиперпараметры, версии
```

**Инварианты структуры (assert при сборке точки):**
- `MISSING` допустим **только в `Y`**, не в `X` (missing-in-inputs — out of scope).
- `X["PROCESS"]` хранится **в коде [0,1]**; физические единицы — через `code↔real` блока.
- сумма `=1` проверяется **только** для `X["MIXTURE"]`; если mixture-блока нет — проверки нет.
- каждая точка ссылается на **существующую** версию в `schema_history`.

### 13.2 Нормировка process-переменных (обязательно)

> Симплекс уже в [0,1] по природе. Process-переменные — нет. Несопоставимость метрик
> исказит GP-ARD, матрицу моментов I-opt и acquisition.

- Все process-переменные **кодируются в [0,1]** (единый стандарт) для **всех
  внутренних расчётов**: генерация дизайна, GP-ядро, моменты, EI/desirability.
- Исходные физические единицы — **только** в UI, рецепте и `Y`-сборе.
- Преобразование `code ↔ real` — единая функция на блок, обратимая, в `config_snapshot`.

### 13.3 Модель отклика — генератор термов (вход дизайна!)

Полная форма (mixture-process):

η = Σ β_i x_i + Σ β_ij x_i x_j   (Scheffé, mixture)
  + Σ γ_k z_k + Σ γ_kl z_k z_l   (process, RSM)
  + Σ δ_ik x_i z_k               (кросс mixture×process)

```python
def build_model_terms(schema: ProjectSchema) -> ModelTerms:
    """
    Единый генератор для ВСЕХ режимов. Никаких if mode==...
    Режим определяется НАЛИЧИЕМ блоков, не флагом.
    """
    mix = get_block(schema, "MIXTURE")     # может быть None  → process-only
    proc = get_block(schema, "PROCESS")    # может быть None  → mixture-only
    terms = []

    if mix:
        terms += scheffe_linear(mix)                    # Σ β_i x_i
        terms += scheffe_quadratic(mix)                 # Σ β_ij x_i x_j (без intercept!)
    if proc:
        terms += process_linear(proc)                   # Σ γ_k z_k
        if schema.model.process_order == "quadratic":
            terms += process_quadratic(proc)            # γ_kk z_k^2, γ_kl z_k z_l
        if not mix:
            terms += [INTERCEPT]                        # process-only → нужен intercept
    if mix and proc:
        terms += cross_terms(mix, proc, schema.model.cross_level)  # δ_ik x_i z_k
    return ModelTerms(terms)
```

Уровни модели (конфиг, по убыванию числа параметров):

| Уровень | Mixture | Process | Кросс | Когда |
|---|---|---|---|---|
| `additive` | Scheffé | linear/quad | нет | две почти независимые задачи (редко) |
| `cross-main` *(дефолт)* | Scheffé | linear/quad | x_i z_k только для **главных** компонентов (после M3) | баланс |
| `full-cross` | Scheffé | quad | все x_i z_k | когда точек хватает |

> **Критично (§13.3):** `cross_terms` присутствуют при `cross-main`/`full-cross`.
> Если их нет — задачи склеены. `cross-main` берёт x_i z_k только для главных
> компонентов после M3.
> **Scheffé без intercept** (симплекс уже несёт константу через Σx=1). Process-only —
> **с** intercept. Это частая ошибка — assert на это.
> **Дефолт кросс-членов адаптивен по фазе:** до M3-screening главных компонентов нет
> → `cross-main` на старте M2 падает в `additive`; после M3 включается `cross-main`
> по реально отобранным главным компонентам; `full-cross` — когда бюджет n покрывает p.
> `p` **считается и логируется** для выбранного уровня (контроль n vs p).

### 13.4 Поблочный геометрический слой (основной рефакторинг)

```python
class BlockGeometry:
    """Полиморфно по типу блока. Каждый метод применяет правило ТИПА."""

    def project(self, block: VariableBlock, coords):
        if block.kind == "MIXTURE": return simplex_project(coords, block.bounds)
        if block.kind == "PROCESS": return box_clip(coords, [0,1])  # в коде всегда [0,1]

    def validate(self, block, coords, eps=1e-9):
        if block.kind == "MIXTURE":
            return abs(sum(coords) - 1) < eps and all(c >= -eps for c in coords)
        if block.kind == "PROCESS":
            return all(0 - eps <= c <= 1 + eps for c in coords)

    def exchange_step(self, block, coords, rng):
        if block.kind == "MIXTURE": return cox_direction_step(coords, block.bounds, rng)
        if block.kind == "PROCESS": return interval_step(coords, [0,1], rng)
```

> **Правило (§13.4):** симплекс-проекция трогает **только** x; clip — **только** z.
> Никогда наоборот — самый частый баг при наивной реализации.
> Все assert Σx=1 работают **только** на mixture-блоке, не на всём векторе.

### 13.5 Моменты и I-критерий (на произведении области)

```python
def moment_matrix(schema, terms) -> np.ndarray:
    """M = ∫_D f f^T,  D = simplex × [0,1]^d.  Произведение → факторизуется по блокам."""
    # mixture-моменты по правильному симплексу (псевдокомпоненты)
    # process-моменты по нормированному кубу [0,1]^d
    # кросс-моменты = произведение соответствующих интегралов (независимость блоков!)
    ...

def i_value(X_model, M, jitter=1e-10):
    XtX = X_model.T @ X_model + jitter * np.eye(X_model.shape[1])
    return np.trace(np.linalg.solve(XtX, M))     # tr[(X'X)^{-1} M]
```

> Сверяется golden-тестом группы A (R-эталон, atol 1e-8). Координаты — псевдокомпоненты
> для mixture, [0,1] для process. Тот же jitter, что в R.

### 13.6 Sequential Augmented Design (ядро §14, переиспользует §5.5)

```python
def augmented_design(state: ProjectState,
                     target_schema: ProjectSchema,
                     region,
                     stop_cfg: StopConfig) -> list[DataPoint]:
    """
    Добор точек к УЖЕ СОБРАННЫМ под (возможно новую) схему/модель.
    Старые валидные точки = FIXED (стартовая информационная матрица), НЕ переделываем.
    """
    terms = build_model_terms(target_schema)
    p = len(terms.list)

    fixed = select_fixed_rows(state.points, target_schema)   # см. migration policy §13.7
    X_fixed = model_matrix(fixed, terms)                     # их вклад в X'X

    M = moment_matrix(target_schema, terms)
    geom = assemble_point_geometry(target_schema)

    new_rows = []
    I_prev = i_value(X_fixed, M) if len(fixed) else np.inf
    while True:
        cand = coordinate_exchange_one(
            fixed_info = X_fixed_info(X_fixed, new_rows),     # X'X учитывает fixed+new
            terms = terms, M = M, geom = geom,
            n_restarts = stop_cfg.restarts, rng = stop_cfg.rng,
        )
        new_rows.append(cand)
        I_now = i_value(model_matrix(fixed + new_rows, terms), M)

        n_total = len(fixed) + len(new_rows)
        rel_gain = (I_prev - I_now) / I_prev if I_prev < np.inf else 1.0
        reason = stop_reason(rel_gain, n_total, p, len(new_rows), stop_cfg)
        if reason:
            log_stop(reason, n_total, p, I_now)               # наблюдаемость §5.5
            break
        I_prev = I_now

    return [to_datapoint(r, target_schema) for r in new_rows]


def stop_reason(rel_gain, n_total, p, n_new, cfg):
    # §5.5: ОТНОСИТЕЛЬНЫЙ критерий, НЕ абсолютный порог на I.
    if n_total < p + cfg.margin:        return None            # не хватает на модель
    if n_new   >= cfg.n_max:            return "budget"
    if rel_gain < cfg.eps:              return "relative_gain" # ← основной выход
    return None
```

> **Связь §5.5 ↔ augmented:** критерий остановки **тот же самый** (ΔI/I<ε, бюджет,
> n≥p+margin). Меняются только две вещи:
> 1. I считается на **полной целевой модели** `target_schema` (с новыми кросс/process-членами);
> 2. стартовая информационная матрица — **не ноль**, а X_fixedᵀ·X_fixed (вклад уже собранных точек).
> Никакого нового критерия не пишем — переиспользуем `stop_reason`. Если в коде
> появился второй критерий остановки «для augment» — это дубль, ошибка.

### 13.7 Migration policy — что со старыми точками при смене схемы (§14)

```python
def select_fixed_rows(points, target_schema) -> list[DataPoint]:
    """
    Решает, какие старые точки годятся как FIXED для новой модели.
    НИКАКИХ молчаливых нулей/средних. Только явные политики.
    """
    fixed = []
    for pt in points:
        old = get_schema(pt.schema_version)
        added_vars = schema_diff_vars(old, target_schema)     # новые ПЕРЕМЕННЫЕ (не члены)
        if not added_vars:
            fixed.append(pt)                                   # схема не расширялась по X
            continue
        resolved = resolve_added_vars(pt, added_vars, target_schema.migration)
        if resolved is APPLICABLE:
            fixed.append(with_resolved_vars(pt, added_vars))   # known-constant / recompute
        # else: точка НЕ годится для членов с новым параметром → не в fixed
    return fixed
```

**Политики миграции (явные, на блок/параметр).**

Добавлен параметр z_new:

| Политика | Что делает | Результат для точки |
|---|---|---|
| `known-constant(v)` | старые опыты шли при известном фикс. значении | z_new=v, точка **валидна** |
| `unknown` | значение не записано | точка **не в fixed** для членов с z_new (но годна для старой подмодели) |
| `recompute(fn)` | вычислимо из существующих X | пересчитать |

Добавлен отклик y_new:
- старые точки: `Y[y_new] = MISSING` → суррогат y_new учится **только** на точках с
  измеренным значением (per-response, M6 это уже умеет).

**Запрещено:** подставлять `0.0` / среднее / `None` молча. Только `MISSING` или явная
политика. Каждое решение — в `schema_history` лог.

> **Асимметрия «новый член из старых переменных» vs «новый параметр».** Когда
> `target_schema` добавляет лишь **новый член из уже существующих переменных**
> (например `cross-main` после нового screening подтянул x_3·z_1, где x_3 и z_1 уже
> были в схеме) — старые fixed-точки дают столбец x_3·z_1 **автоматически** (он
> вычислим из их x_3 и z_1). То есть расширение **модели** (больше членов из тех же
> переменных) почти бесплатно переиспользует данные, а расширение **пространства**
> (новый параметр z) — требует миграционной политики. `schema_diff_vars` обязан
> различать эти два случая: новые ПЕРЕМЕННЫЕ vs новые ЧЛЕНЫ из старых переменных,
> чтобы не помечать точку невалидной там, где член просто вычислим.

### 13.8 Влияние на M1–M8 (сводно, обе фичи)

| Модуль | Mixture-Process (§13) | Augmented/Schema (§14) |
|---|---|---|
| **M1 геометрия** | произведение блоков; mixture/process-only = частные случаи | — |
| **M2 D-opt / M5 I-opt** | поблочный exchange, модель §13.3 как вход | **fixed-строки** в exchange (опорная инфо-матрица) |
| **M3 screening** | ARD на x и z; отбор главных кросс-членов | переучивается на расширенной схеме |
| **M4 GMM (output)** | без изменений | переучивается при новом y |
| **M5 / §5.5** | моменты на произведении области | критерий тот же, I на целевой модели + fixed |
| **M6 per-response GP** | ARD по x и норм. z; mean §13.3 | новый y = новый суррогат; MISSING → не в обучении |
| **M7 ветки / AL** | acquisition на S×куб (поблочно) | работает на current_schema; старые точки с MISSING вне обучения |
| **M8 desirability** | без изменений; рецепт = {состав + условия} | новый y входит в желательность |

### 13.9 Объединённый чек-лист реализации

> **Легенда статуса (сверено на итерации 12, коммиты `d7c2ed7`·`93200e8`·`a0e1f57`·`4ee079`):**
> `[x]` сделано и под тестом · `[~]` частично (см. примечание) · `[ ]` не сделано.
> Детальная сверка с привязкой к тестам и остаток — в §13.11.

**Фундамент (общий для §13 и §14) — строить ПЕРВЫМ**
- [x] Точка = составной объект `{X: {block: [...]}, Y: {resp: val|MISSING}}` + `schema_version` + `origin_tag`. **Не плоский массив.** *(schema.DataPoint)*
- [x] `MISSING` — явный сентинел, допустим **только в `Y`**, не в `X`. *(DataPoint.validate)*
- [x] Схема версионируется: `schema_history` неизменяема, точки разных версий сосуществуют. *(schema_evolution.SchemaHistory + evolve_schema)*
- [x] Запрет «оба блока пустые»; ≤1 MIXTURE, ≤1 PROCESS. *(ProjectSchema.__post_init__)*
- [x] `config_snapshot`: seeds, гиперпараметры, версии + `code↔real` на блок — воспроизводимость. *(core.config_snapshot.ConfigSnapshot; интегрирован в ProjectState; test_iteration12_config_snapshot)*

**Mixture-Process (§13)**
- [~] Проекция/clip/валидатор/exchange — **поблочные**, по `kind`. *(project/clip/validate поблочные; кандидаты = декартово произведение блоков; покоординатный block-exchange не вводился — добор pool-based, §13.6)*
- [x] Симплекс-проекция → только x; clip → только z. Не наоборот. *(block_geometry)*
- [x] Σ=1 проверяется только на MIXTURE-блоке; process-only: нет проверки Σ=1; mixture-only: нет process-clip. *(block_geometry.validate)*
- [x] z внутри всех расчётов ∈ [0,1]; `code↔real` обратима; в снимке. *(to_code/from_code обратимы; сохраняются в `ConfigSnapshot.code_real` и воспроизводятся из снимка)*
- [x] Кросс-термы x_i z_k присутствуют при `cross-main`/`full-cross`. *(block_model.build_model_terms)*
- [x] Scheffé **без** intercept; process-only **с** intercept (assert). *(build_model_terms)*
- [x] p считается/логируется; режим = по наличию блоков, **нет** `if mode`. *(count_params/resolve_model_for_budget)*
- [x] M по произведению области (симплекс × норм. куб); кросс-моменты = произведение интегралов. *(block_moments.block_moment_matrix)*

**Schema Evolution + Augmented Design (§14)**
- [x] Политики `known-constant` / `unknown→MISSING` / `recompute` — явные, в логе. *(schema_evolution; решение — в origin_tag.migrated_from)*
- [x] **Никаких молчаливых 0/средних** в X или Y. *(migrate_point: unknown→точка исключается; новый отклик→MISSING)*
- [x] Новый отклик → старые точки `Y[y_new]=MISSING`, суррогат учится только на измеренных. *(migrate_point)*
- [x] `select_fixed_rows` отбирает валидные старые точки для целевой модели. *(schema_evolution.select_fixed_rows: резолв старой схемы по версии + политики; augmented.* — частный случай без истории)*
- [x] Coordinate-exchange учитывает X_fixedᵀ·X_fixed как стартовую инфо-матрицу (не ноль). *(i_optimal_augment_sequential: M_acc=EᵀE)*
- [x] Старые fixed-точки **не двигаются**, добираются только новые. *(forward-selection по пулу)*
- [x] `schema_diff_vars` различает новые ПЕРЕМЕННЫЕ vs новые ЧЛЕНЫ из старых переменных. *(schema.schema_diff_vars)*
- [x] `stop_reason` — единая функция; **нет** второго критерия для augment. *(переиспользован критерий §5.5)*
- [x] Выход по `relative_gain` (ΔI/I), НЕ по абсолютному порогу I; не останавливаться при n_total < p+margin. *(§5.5)*
- [x] I считается на **целевой** модели и **объединённом** (fixed+new) дизайне; лог остановки: reason, n_total, p, I. *(model_matrix_fn + IAugmentResult)*

**Регресс (доказательство сохранности базы)**
- [x] **mixture-only бит-в-бит** со старым golden (дизайн по I/det ±1%, Scheffé atol 1e-8, GP μ/σ atol 1e-6). *(model_matrix==scheffe и W==region_moment_matrix бит-в-бит; GP-постериор μ/σ на составных координатах сверен с независимым `gp_posterior` на mixture/process/mixture-process, atol/rtol 1e-6 — test_golden_gp_composite)*
- [x] **augment без смены схемы** == старое поведение §5.5: инъекция `model_matrix_fn` для mixture-only бит-в-бит совпала с legacy Scheffé-путём (indices/i_history/stop_reason). *(test_iteration12_augment)*
- [x] golden-группа A (значение I vs R) проходит на mixture, process, mixture-process. *(аналитические моменты vs независимый эталон, atol 1e-8; test_golden_product_moments)*

> **Топ-3 «протекающих» места этой итерации:**
> 1. **Формат точки** — составной версионированный объект. Если плоский массив — обе фичи сломаются при первом расширении схемы.
> 2. **Единый критерий остановки** — augment переиспользует `stop_reason` из §5.5, не пишет свой. Иначе дубль и рассинхрон.
> 3. **`MISSING` только в `Y`, никаких молчаливых нулей** — отравленная модель иначе неотлаживаема.

### 13.10 Порядок реализации (чтобы не переписывать)

1. **Фундамент** (структуры §13.1) — точка, схема, версионирование. Без этого остальное ляжет на неверный формат.
2. **Генератор термов** (§13.3) + **поблочная геометрия** (§13.4) — общая база для дизайна.
3. **Моменты + i_value** (§13.5) → прогнать **golden-группу A** (фундамент верен).
4. **Augmented design** (§13.6) поверх существующего exchange + `select_fixed_rows` (§13.7).
5. **Регресс mixture-only и augment-без-схемы** — гейт перед мержем.
6. Миграционные политики (§13.7) — расширяешь по мере появления реальных сценариев.

### 13.11 Статус сверки чек-листа (итерация 12) и остаток

**Сделано и под тестом** (коммиты `d7c2ed7`, `93200e8`, `a0e1f57`, `4ee079`; тесты `tests/unit/test_iteration12_{schema,blocks,moments,augment}.py` — 35 проверок):
- Фундамент: блочная схема (`core/schema.py`), составная версионированная точка, `MISSING` только в Y, инварианты блоков.
- §13.3/§13.4: единый генератор термов (`design/block_model.py`) + поблочная геометрия (`core/block_geometry.py`).
- §13.5: моменты на произведении области + `i_value` (`design/block_moments.py`); mixture-only `W` бит-в-бит == `i_optimal.region_moment_matrix`.
- §13.6: блочный добор (`design/augmented.py`) переиспользует критерий §5.5 через неинвазивную инъекцию `model_matrix_fn` в `i_optimal_augment_sequential` (дефолт=None ⇒ mixture-only Scheffé бит-в-бит).
- §13.5/golden-A: аналитические моменты `analytic_moment_matrix` (закрытая форма) vs независимый эталон `reference.ref_product_moment_matrix` (atol 1e-8) на mixture/process/mixture-process; `i_value`==`ref_i_criterion` (`tests/golden/test_golden_product_moments.py`, коммит `ee39ab1`).
- §13.7 schema-evolution: `core/schema_evolution.py` (SchemaHistory + evolve_schema + migrate_point + select_fixed_rows; политики known-constant/unknown/recompute; новый отклик→MISSING) + поле `ProjectSchema.migration`; прогрессия mixture→mixture+process→+параметр+отклик под тестом (`test_iteration12_evolution.py`).
- §13.5/§13.11 АНАЛИТИЧЕСКИЕ моменты как ДЕФОЛТ (устранение MC-смещения golden-A): блочный добор (`design/augmented.augmented_design` + новый публичный `build_moments`) и M5 (`apps/pipeline_runner.run_m5`) берут `W` из закрытой формы (`block_moments.analytic_moment_matrix` / `i_optimal.region_moment_matrix(method="analytic")`) — детерминированно, без смещения `SimplexRegion.random_points`. MC оставлен опцией (`moment_method="mc"` / `method="mc"`) для быстрых прикидок; дефолт `region_moment_matrix` остаётся `mc` (бит-в-бит регресс §13.9 не сломан). Две независимые реализации закрытой формы сверены (atol 1e-12) и инвариант результата к `n_mc` — `tests/unit/test_iteration12_analytic_moments.py`.

**Сделано (итерация 13) — закрыт остаток §13.11 п.1–3** (тесты `tests/unit/test_iteration12_config_snapshot.py`, `tests/golden/test_golden_gp_composite.py`):
- **`config_snapshot`** (§13.1/§13.2): `core/config_snapshot.py::ConfigSnapshot` — seeds/гиперпараметры/версии (окружение+схема) + `code↔real` на КАЖДЫЙ блок (по сохранённым `lower/upper` блок и его обратимые `to_code`/`from_code` восстанавливаются из снимка, без доступа к схеме); round-trip `from_dict(to_dict())` тождественен.
- **Персистентность версионирования:** `core/state.py::ProjectState` аддитивно несёт `schema_history`/`current_schema_version`/`points`/`config_snapshot` (save/load round-trip; старые `state.json` M1–M8 грузятся с пустыми дефолтами — обратная совместимость под тестом). Хелперы: `add_schema`/`add_point`/`schema_for`/`latest_schema`/`set_config_snapshot`/`get_config_snapshot`.
- **Полный GP-golden μ/σ на составных координатах:** продакшн `GPExpert` (sklearn) при фикс. θ сверен с независимым Cholesky-эталоном `reference.gp_posterior` на mixture-only/process-only/mixture-process (atol/rtol 1e-6). Нюанс зафиксирован: составной Scheffé-базис при mixture-process РАНГ-ДЕФИЦИТЕН (Σ_i x_i·z_k = z_k из-за Σx=1) ⇒ тренд берётся из продакшн `mean_` (lstsq/SVD устойчив к неполному рангу), а НЕЗАВИСИМО сверяется именно GP-постериор остатков (μ_resid, σ) — ровно цель п.3.

**Остаток:**
1. *(СДЕЛАНО — см. «Сделано (итерация 13)»)* ~~`config_snapshot`~~ · ~~персистентность версионирования~~ · ~~полный GP-golden μ/σ на составных координатах~~.
2. *(СДЕЛАНО ранее)* ~~Неравномерность `SimplexRegion.random_points`~~: путь моментов (M5 и блочный добор) переведён на аналитику, MC сохранён опцией. Остаточно (не критично): сам сэмплер `random_points` (смещён к центроиду) не переписан — он остаётся только в генерации пула кандидатов (где важна допустимость, а не равномерность) и в опциональном MC-пути.

> **Предсуществующий несвязанный дефект:** `tests/unit/test_iteration11_metrics_cache.py::test_metrics_survive_save_load_without_results` падает и без правок итерации 12 (доказано `git stash`) — конфликт теста с фичей регидрации результатов (коммит `acdbc7f`). К §13/§14 не относится.

---

<!-- MERGED-SECTION-15 -->
# §15. Боевая верификация §14 (Schema Augmentation) + M8-argmax + экономический стоп

> Разделы §15.0–§15.5 (боевая аугментация) и §15.6 (экономический
> критерий остановки) ВШИТЫ в этот документ из ранее отдельных файлов
> `REBUILD_SPEC_15_battle_augmentation.md` и
> `REBUILD_SPEC_15_6_economic_stop.md` (объединение спецификации в
> один источник истины). Содержимое перенесено без изменений по
> существу; верхние заголовки понижены на один уровень.

---

## §15.0–§15.5 — Боевая аугментация и M8-argmax


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

## §15.0.4 Правка рассогласования Y↔X в фазе 1 (Вариант 1: append-семантика C)

> **СТАТУС: ОТМЕНЯЕТ Предусловие 2 и §15.2.4 в части компонента C.** Решение
> принято осознанно (сессия §15.0.4). Ранее (Предусловие 2) `C` трактовался как
> region-expansion: симплекс `{A,B,C}` онтологически полон с v1, C заперт
> bounds'ами `[1/3,1/3]`, `schema_version` за раскрытие C НЕ растёт. Это давало
> **рассогласование Y↔X в фазе 1**: схема варьировала A+B (C на baseline), а
> оракул считал Y при `C=1/3`, хотя точки концептуально 2-компонентны. Вариант 1
> устраняет рассогласование append-семантикой.

### Суть

```
БЫЛО:  фаза 1: A+B=1 в схеме, НО оракул считает Y при C=1/3   ← рассогласование
СТАЛО: фаза 1: A+B=1, C ОТСУТСТВУЕТ, оракул считает Y при C=0  ← append-согласовано
       фаза 2: +C (version+1), точки фазы 1 = fixed на грани C=0,
               migration known-constant(0.0)
```

Грань `C=0` тождественна старому 2-компонентному симплексу: `A+B+0=1 ⟺ A+B=1`.
Именно поэтому Вариант 1 «чист» — fixed-строки фазы 1 автоматически согласованы
по Σ после переопределения `A+B=1 → A+B+C=1`.

### 1. Оракул принимает `active_schema`

C-вклад считается ТОЛЬКО если C в активной схеме фазы; иначе C=0 → термы `5·C`,
`A·B·C` и любые с C зануляются. `active_schema=None` → полная схема (обратная
совместимость). Это **меняет истину фазы 1 осознанно** — не баг новой истины, а
корректная append-семантика.

### 2. Хранение точек фазы 1 — РЕАЛЬНО 2-компонентны

`phase1_point.X["MIXTURE"] == [A, B]` (A+B=1), `"C" not in schema_v1.vars`,
`schema_version == 1`. Запрет молчаливой дозаписи `C=1/3` (это была бы
«Интерпретация A через заднюю дверь»).

### 3. Append C в фазе 2 + migration

`evolve_schema(v1, append_mixture=["C"])` → `version 1→2`, Σ переопределяется
`A+B=1 → A+B+C=1`. Для fixed-строк фазы 1: `migration["C"] = known-constant(0.0)`
(грань симплекса), `C=0.0`, `A+B+C=1` сходится.

> ⚠️ **append mixture ≠ append process.** Для process (T/P) baseline —
> внутренняя точка куба (`T=0.5`) → `known-constant(0.5)`. Для mixture
> «не варьировали» = **грань симплекса C=0**, а НЕ внутренний baseline →
> `known-constant(0.0)`. Грань C=0 — единственное значение, при котором старая
> 2-компонентная точка остаётся валидной (Σ сходится при неизменных A,B).
> `evolve_schema`/`select_fixed_rows` обязаны различать эти два случая.

### 4. Переход псевдокомпонент при смене размерности симплекса

> ⚠️ **Ловушка, которую вскрывает append mixture (process — нет).**

Псевдокомпонентное преобразование Scheffé зависит от РАЗМЕРНОСТИ симплекса.
Фаза 1 — 2D `{A,B}`; фаза 2 — 3D `{A,B,C}` с пересчётом псевдокомпонент. Точка
фазы 1 `(A,B)` при C=0 в реальных координатах = `(A,B,0)`, но в псевдокомпонентах
3D-симплекса это другой вектор. `select_fixed_rows` должен прогонять fixed-строки
через НОВОЕ псевдокомпонентное преобразование. Assert: `pseudo_3d(f) ↔ real (A,B,0)`
обратимо. Для T/P append это не нужно (process не трогает симплекс).

### 5. Модель: термы с C активируются только в v2+

`"C" not in model_terms(v1)` (нет `5·C`, `A·B·C`); `"C"`, `"A:B:C"` ∈
`model_terms(v2)`. Идентифицируемость (§15.0.2 cond-чек): C-термы на fixed-строках
с C=0 дают НУЛЕВЫЕ столбцы → оцениваются только по новым точкам фазы 2 с C>0.
`augmented_design` ОБЯЗАН добрать точки с C>0; rank-чек на ОБЪЕДИНЁННОЙ базе
(fixed+new), не на fixed.

### 6. Пересчёт golden фазы 1

Истина фазы 1 ИЗМЕНИЛАСЬ (C=0 вместо C=1/3) → golden пересчитать, причину — в
changelog: «истина фазы 1 изменена с C=1/3 на C=0 для append-согласованности C
(§15.0.4)». Это не регрессия, а исправление: старый golden считался при
противоречивом состоянии (Y при 1/3, X без C).

### 7. Чек-лист §15.0.4

- [ ] Оракул принимает `active_schema`, C-вклад только если C в схеме → фаза 1 при C=0.
- [ ] Точки фазы 1 хранят 2 mixture-координаты (A+B=1), C отсутствует; запрет дозаписи C=1/3.
- [ ] `evolve_schema(append_mixture=["C"])`: version+1, Σ переопределяется.
- [ ] `migration["C"] = known-constant(0.0)` (грань симплекса), НЕ baseline-внутренняя точка.
- [ ] Псевдокомпоненты fixed-строк пересчитаны под 3D-симплекс (ловушка смены размерности).
- [ ] C-термы идентифицируемы только на fixed+new (augmented добирает C>0); rank-чек на объединённой базе.
- [ ] Golden фазы 1 пересчитан при C=0, причина в changelog.
- [ ] Тесты: 2-компонентность фазы 1; оракул C=0; fixed на грани; идентифицируемость; псевдокомпонент-remap.

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

---

## §15.6 — Экономический критерий остановки и движения границ


> Источник истины для §15.6. Самодостаточный документ. При противоречии кода
> блоку аксиом (§0) — прав §0 (код не должен «переоткрывать» решения по-своему).
> Связь с базовой спекой: `docs/REBUILD_SPEC.md` (§3/M8 desirability, §5.5 стоп,
> M6 GP/MoE — выше в этом документе) и §15.0.3 движение границ / §15.5 точка
> интеграции / боевой тест (выше, в разделе §15).

## 0. Блок архитектурных аксиом (читать перед реализацией)

> Эти строки фиксируют концепцию, которая **не должна выводиться из паттернов кода**. При противоречии кода этим строкам — прав этот блок.

- **A0.1.** Движок предсказания — **GP + MoE**, непараметрический чёрный ящик. У него **нет** «полиномиальной степени», которую можно недобрать.
- **A0.2.** **Scheffé — только интерпретационный слой** над выходом GP для человека. Он не ограничивает обучаемость движка и не является моделью предсказания. Запрещено диагностировать остатки в терминах степени Scheffé.
- **A0.3.** Недобор в разреженной зоне лечится **данными** (augmented в зону), не ростом параметрической модели.
- **A0.4.** Mixture: $\sum_i x_i=1,\ x_i\ge0$ (симплекс). Process-переменные (T, P) — вне симплекса.
- **A0.5.** Граница имеет **происхождение**: `hard` (физика/закон/бюджет — двигать нельзя) / `soft` (дефолт — можно). Точки с измеренным $Y$ (`fixed`) не проецируются.
- **A0.6.** Система **никогда не двигает границы/точки молча**. Любое движение — предложение пользователю с цифрой. Решение — за пользователем.
- **A0.7. (новая)** На **вырожденном (flat) направлении** целевой функции репортится **objective-gap, не x-gap**; ось помечается неидентифицируемой. *Регрессионный кейс: economy/P, $\partial d/\partial P\equiv0$, spread=0.00e+00.*

## 1. Цель

Заменить чисто технический стоп-критерий на **двойной** (технический И экономический/VoI). Останавливаться, когда **либо** нечего улучшать, **либо** улучшать невыгодно. При упоре в границу — класть перед пользователем **денежную триаду**, не флаг.

## 2. Входные параметры

| Параметр | Обозн. | Назначение | Размещение в схеме |
|---|---|---|---|
| Плотность | $\rho(A,B,C,T,P)$ | цена за **изделие** | **GP-свойство** (зависит от состава И режимов) — см. §3 |
| Объём потребления | $V$ | масштаб экономии, кг/шт за период | атрибут ветки |
| Стоимость эксперимента | $c_{exp}$ | порог выгодности | атрибут ветки (const + переменная часть) |
| Горизонт окупаемости | $H$ | за сколько окупиться | **атрибут ветки (default) + override на раунд** — см. §2.1 |

### 2.1. Горизонт $H$ — атрибут ветки с override на раунд

$H$ **эволюционирует вместе с веткой** (тот же принцип, что schema versioning — ветка живой объект):

```
рождение:  объёмов нет → один проход → серия       (H мал/не задан)
рост:      спрос → V растёт → H появляется и растёт
зрелость:  клиенты просят качество → новые цели/компоненты → ветка обрастает
```

- **default** — на уровне ветки (текущий горизонт продукта);
- **override** — на уровне раунда (разовый дорогой эксперимент под клиента).

## 3. Цена за изделие через плотность (структурный параметр)

$$\text{price}_{\text{изд}}(A,B,C,T,P) = \frac{\text{price}_{\text{состав}}(A,B,C)}{\rho(A,B,C,T,P)} \quad [\text{₽/кг изделия}]$$

- $\rho$ — **полноценный отклик GP+MoE** со своей неопределённостью $\sigma_\rho$ (как strength/gloss).
- Числитель зависит от состава, знаменатель — от состава **и режима**.
- ⚠️ **Новый канал оптимизации:** process-переменные (T, P), не влиявшие на цену в истине, теперь влияют **через $\rho$** (вспенивание/упаковка ПВХ: меньше $\rho$ → больше изделий из того же сырья → ниже цена/штуку). Это часто **главный** рычаг в пластиках.
- ⚠️ **Следствие для A0.7:** вырожденные направления, найденные без $\rho$ (например economy/P-флэт), **могут перестать быть вырожденными** с введением $\rho(...,P)$. Flat-статус оси переоценивается при добавлении $\rho$.

> **Уточнение реализации §3 (решение сессии, ШАГ 2 — binding).** Displayed-формула
> выше записана с делением (`price_состав / ρ`), но это **противоречит** прозе про
> вспенивание («меньше ρ → ниже цена/штуку»). Деление дало бы обратный знак
> (меньше ρ → дороже). В коде зафиксирована **physics-трактовка**:
> $\text{price}_{\text{изд}} = \text{price}_{\text{состав}}(A,B,C)\cdot\rho(A,B,C,T,P)$
> Масса изделия $\propto\rho$, цена за штуку $=$ ₽/кг $\cdot$ масса $\propto$
> `price_состав·ρ`; вспенивание (ниже ρ) удешевляет штуку. Реализовано как
> `price_per_item` / `make_item_cost_fn` в `src/optimize/desirability.py`
> (подключается в `optimize_desirability(cost_fn=...)` как обычное `min`-свойство);
> тест `tests/unit/test_iteration17_item_cost.py`. Знак НЕ менять без перечтения §3.

## 4. Двойной стоп-критерий

Продолжать, пока выполнено **И**:

$$\underbrace{\Delta d \ge \varepsilon \;\wedge\; d < ceil}_{\text{технический}} \quad\wedge\quad \underbrace{\max_x \text{EI}_{\text{₽}}(x)\cdot V\cdot H > c_{exp}}_{\text{экономический (VoI)}}$$

Стоп, если нарушено **любое**. Логируется `stop_reason ∈ {ceil_reached, stagnation, not_economical}`.

> **Уточнение реализации §4 (ШАГ 4 — binding).**
> 1. **$\text{EI}_{₽}$ — денежный путь §6, через цену, НЕ через d_overall.** §6
>    явно задаёт «экономия = Δprice_изд · V», поэтому экономический член считается
>    `economic_value = max_x EI_price(x)·V·H`, где
>    `EI_price = E[max(price_best − price_изд, 0)]` с ρ~`N(μ_ρ,σ_ρ)` (honest σ,
>    ШАГ 1), `price_изд = price_состав·ρ` (ШАГ 2). Это РАЗНАЯ роль с VoI-движком
>    d_overall (ШАГ 3): VoI говорит *КУДА* ставить опыт (инфо-выгода), экономика —
>    *СТОИТ ЛИ ВООБЩЕ* (денежная выгода). Не смешивать.
> 2. **Приоритет причин** (когда нарушено несколько):
>    `ceil_reached` > `not_economical` > `stagnation`. Потолок достигнут —
>    сильнейший сигнал (улучшать нечего → дальше движение границ §6); иначе если
>    невыгодно — `not_economical`; иначе встал прогресс — `stagnation`.
> 3. **Единицы:** EI_price [₽/изд]·V [изд/период]·H [период] = [₽ за горизонт]
>    против c_exp [₽/опыт]. Реализовано: `decide_stop`/`economic_value`/
>    `expected_price_improvement` в `src/optimize/economic_stop.py`; V/c_exp/H —
>    атрибуты `Branch` (+`resolve_horizon` override на раунд, §2.1). Дефолты
>    нейтральны (V=0 ⇒ economic_value=0): без заданной экономики стоп ведёт себя
>    как чисто технический (обратная совместимость). Тест:
>    `tests/unit/test_iteration17_economic_stop.py`.


## 5. VoI — per-property MC-EI с разложением вклада (форма дисконта)

> Принцип A0.3 + урок premium-угла (3/294): неопределённость GP — **причина исследовать** (разведка), а не штраф. VoI идёт в разреженную зону **осознанно**, и лечит баг $d_{best}>ceil$ направленным измерением, а не запретом.

**Базовая идея (Expected Improvement):** ожидаемое улучшение лучшего решения от эксперимента в $x$, усреднённое по возможным исходам (взвешенным GP-постериором). Член разведки $\propto\sigma$ — растёт с неуверенностью.

**Форма реализации — НЕ EI по агрегату $d_{overall}$.** Агрегирование до оценки информации теряет, по какой оси разведка полезна, и из-за geo-mean+veto **врёт в обе стороны** (завышает EI от дутого $\sigma$ уже-добранного свойства; занижает EI от добора veto-лимитирующего). Поэтому:

```
VoI = MC-EI, сэмплирование ПО СВОЙСТВАМ (раздельные GP-постериоры σ_i),
      прогон набора через НАСТОЯЩУЮ d_overall (veto/веса честно соблюдены),
      с РАЗЛОЖЕНИЕМ EI по вкладу каждого свойства:
          EI_total(x) = Σ_i EI^(i)(x)
      (вклад оси i ≈ падение EI при "заморозке" σ_i на μ_i без шума).

ЗАЩИТА: высокий EI принимается, ТОЛЬКО если обоснован разведкой свойства,
        реально лимитирующего d_overall — не дутым σ свойства, которое
        geo-mean/veto уже не даёт двигать агрегат.

ВЫХОД:  x_next = argmax_x EI_₽(x);  EI_₽ через price_изд (с ρ, §3).
БОНУС:  диагностика знает не только "стоит ставить", но и ПО КАКОЙ ОСИ.
```

**Подводные камни (зафиксировать для VS Code):**
1. $d_{overall}$ — geo-mean+veto, **негауссова, рвётся в нуле** → замкнутая формула EI неприменима к агрегату → **MC обязателен**.
2. Многокритериальность: per-property MC через $d_{overall}$ достаточен; строгий EHVI не требуется.
3. **$\sigma$ должна быть калиброванной** — VoI целиком держится на честности $\sigma_i$ GP. Нужна проверка калибровки (premium-угол — тест на честность $\sigma$).
4. $\arg\max_x\text{EI}$ — оптимизация по симплексу+process на раунд; подъёмно.

> **Уточнение реализации §5 (ШАГ 3 — binding).** Движок VoI
> (`mc_ei_decomposed` в `src/optimize/voi.py`) считает EI в шкале **d_overall**
> (per-property MC через РЕАЛЬНУЮ `Desirability.overall`, CRN-разложение вклада,
> поле `justified` = ЗАЩИТА §5). Денежная шкала $\text{EI}_{₽}$ (член $\text{EI}\cdot V\cdot H$
> vs $c_{exp}$) — **отдельный слой ШАГА 4**: $V/H/c_{exp}$ живут атрибутами ветки,
> а `price_изд` (с ρ) уже входит в `d_overall` min-свойством (ШАГ 2), поэтому
> d_overall-EI УЖЕ тянет к «дешевле-и-лучше». НЕ встраивать $V/H/c_{exp}$ внутрь
> движка VoI — он остаётся чистым (вход: суррогаты+goal+d_best; выход:
> EI/contributions/limiting_axis/justified). Тест: `tests/unit/test_iteration17_voi.py`.

## 6. Граница-сигнал → денежная триада

При упоре оптимума в `soft`-границу — пользователю выдаётся **триада** (не флаг):

```
ГРАНИЦА C ≤ 0.2 (soft): оптимум упёрся.
  экономия за границей:  X ₽/период  (= Δprice_изд · V, через VoI/§5, c ρ)
  цена добычи:           N · c_exp    (N — оценка экспериментов до сходимости)
  окупаемость:           N·c_exp / X  периодов  (сравнить с H)
  → двигать C до 1.0? [решение пользователя — A0.6]
```

- `hard`-граница (A0.5): триада **не предлагается**, движение запрещено по происхождению; показываем «упёрлись в hard-лимит» информативно.
- Движение применяется **в конце раунда** (не внутри — иначе раунд гибрид двух постановок).

> **Уточнение реализации §6 (ШАГ 5 — binding).**
> 1. **Происхождение границы — ПОЛИТИКА, не состояние модели.** `hard`/`soft` НЕ
>    зашиваются во frozen `VariableBlock` (иначе ломается сериализация/golden
>    схемы) — реестр держит раннер: `MixtureProcessRunner.border_origin` /
>    `set_border_origin` (дефолт `soft`, A0.5). Геометрия упора — чистый детектор
>    `move_bounds.boundary_hits` (не знает про hard/soft).
> 2. **A0.6 не двигаем молча.** `move_region` ОТКАЗЫВАЕТ двигать `hard`-границу
>    (`RegionMoveError`); триада — это ПРЕДЛОЖЕНИЕ (`MoneyTriad` с полем
>    `worth_it = payback ≤ H`), решение за пользователем. Для `hard` —
>    `HardBoundaryError` (информативный отказ, не флаг).
> 3. **Числа триады:** экономия `X = Δprice_изд·V` [₽/период] (Δprice_изд — макс.
>    EI на удешевление за расширенной областью, через ρ, ШАГ 2/4), цена добычи
>    `N·c_exp`, окупаемость `N·c_exp/X` сравнивается с горизонтом ветки H
>    (override на раунд, §2.1). Реализовано: `money_triad`/`boundary_signal` в
>    `src/optimize/economic_stop.py`, `MixtureProcessRunner.border_money_triad`.
>    Тест: `tests/unit/test_iteration17_boundary_triad.py`.

## 7. Регрессионные кейсы (из battle-теста)

| Кейс | Что проверяет | Эталон |
|---|---|---|
| append T/P/C (phase 1→3) | рост потолка за введением переменных | монотонный рост $d_{best}$ |
| restrict floor B≥0.12 (matte) | политика выпадающих + обратимость | pool 144→89→144, история не урезана |
| relax cap C 0.2→1.0 (deep_gloss) | граница-сигнал, прототип триады | `ceiling 0.639→0.753`, $d$ к аналитике |
| **economy/P (A0.7)** | flat-направление → objective-gap | spread=0.00e+00, x-gap игнор |
| **battle vs post-top-up** | battle-числа промежуточные | оценка по post-top-up (+17%→+0%) |

## 8. Связь с предыдущими §

- §15.0.3 (H1/H3): `reach_target` теперь **оценивает экономику** (VoI) перед предложением.
- §15.5 (точка интеграции): **закрыт** — триггер = упор в `soft`-границу в конце раунда; исполнение = предложение пользователю (A0.6).
- premium-остаток (+4%): **разреженность угла**, не структурная мисспецификация; VoI идёт туда разведкой, $\sigma$ — вход и в диагностику, и в экономику.

---

## Статус развилок

| # | Развилка | Решение |
|---|---|---|
| 1 | Горизонт $H$ | ✅ атрибут ветки (default) + override на раунд (§2.1) |
| 2 | Плотность $\rho$ | ✅ GP-свойство $\rho(A,B,C,T,P)$; новый P-рычаг через цену (§3) |
| 3 | Форма дисконта | ✅ VoI = per-property MC-EI с разложением вклада (§5) |

**Все развилки закрыты.** §15.6 готов к переносу в VS Code.

---

## Заметка реализации (фундамент VoI)

**Единственное скрытое допущение**, на котором держится весь VoI: GP честен о
своей уверенности. Если GP врёт о $\sigma$, VoI врёт следом. Поэтому **первым
шагом** реализации — прогнать **тест калибровки $\sigma$** на premium-угле
(предсказанный $\sigma$ vs реальная ошибка на отложенных точках): проверить
фундамент до того, как на нём строить экономику. Текущий движок —
`src/models/gp_expert.py` (`GPExpert.predict` уже отдаёт honest `mean`/`std`,
ядро включает `WhiteKernel`), что и есть точка приложения этого теста.

### Карта реализации (привязка к текущему коду)

| § | Что вводим | Где (текущий код) |
|---|---|---|
| §2/§2.1 | `V`, `c_exp`, `H` (+ per-round override $H$) | атрибуты `Branch` (`src/design/branches.py`) |
| §3 | $\rho$ как GP-свойство; `price_изд = price_состав · ρ` (physics, см. уточнение §3) | `price_per_item`/`make_item_cost_fn` в `src/optimize/desirability.py` ✅ ШАГ 2 |
| §4 | двойной стоп; `stop_reason ∈ {ceil_reached, stagnation, not_economical}` | `decide_stop`/`economic_value` в `src/optimize/economic_stop.py` + V/c_exp/H в `Branch` ✅ ШАГ 4 |
| §5 | per-property MC-EI с разложением вклада, $x_{next}=\arg\max\text{EI}$ | `mc_ei_decomposed`/`VoIResult` в `src/optimize/voi.py` ✅ ШАГ 3 |
| §5.3 | калибровка $\sigma$ (premium-угол) | `tests/unit/test_iteration17_sigma_calibration.py` ✅ ШАГ 1 |
| §6 | `hard`/`soft` происхождение границ; денежная триада | `boundary_hits`+`BORDER_*` (move_bounds), `money_triad`/`boundary_signal` (economic_stop), runner `border_origin`/`set_border_origin`/`border_money_triad`+hard-guard ✅ ШАГ 5 |
| §7 | регрессионные кейсы | боевой тест (`tests/unit/test_iteration13_battle.py`) + новые `test_iteration17_*` |
| A0.7 | flat-ось → objective-gap (не x-gap) | `detect_flat_axis`/`FlatAxisResult`/`axis_spread` (economic_stop), runner `flat_axis_at_border` (MixtureProcessRunner) / `flat_axis_mixture` (PipelineRunner). Тест: `tests/unit/test_iteration17_flat_axis.py` ✅ ШАГ 6 |
| UI | §15.6 read-only панель (A0.6): stop_reason + триада + flat-ось | `render_branches` (streamlit_app, экспандер «💰 §15.6»). Тест: `tests/unit/test_iteration17_economic_ui.py` ✅ ШАГ 7 |

> Канон (`.clinerules`, REBUILD_SPEC §5/§12): сначала логика + unit-тест, потом UI.
> Порядок шагов: (1) σ-калибровка → (2) $\rho$ как свойство + `price_изд` →
> (3) VoI MC-EI с разложением → (4) двойной стоп + `stop_reason` →
> (5) `hard`/`soft` границы + триада → (6) A0.7 flat-ось → (7) UI.


---

## §16. Роль отклика (ветка × отклик) и атрибуция, читающая роль (итерация 16)

> Развитие §5/§12. Канон (`.clinerules`): ОДНА модель физики на проект, ветка —
> контейнер намерения; сначала логика + unit-тест, потом UI.

### §16.0 Решённое ядро (реализовано, ✅)

**Роль — атрибут ПАРЫ (ветка × отклик), а не глобальное свойство отклика.** Роль
ВЫВОДИТСЯ из намерения ветки (`goal` + ценовая нога), без отдельного хранения.
XOR честности §5 разрешается **приоритетом M2**: `OPTIMIZED > PRICE_INPUT >
REFERENCE` — роль всегда однозначна. Один и тот же отклик носит РАЗНЫЕ роли в
разных ветках (branch-local).

Ключевой случай **Гр-1**: ρ одновременно ЦЕЛЬ и питает цену
(`price_изд = price_состав·ρ`) ⇒ роль = `OPTIMIZED`, а ценовой σ_ρ-канал помечен
к занулению. Атрибуция, читающая роль: при ρ=`OPTIMIZED` весь ценовой
разведочный канал зануляется (`α≡0`) — иначе одна и та же δρ засчитывается
ДВАЖДЫ (через качественную ногу d_ρ И через ценовую d_price). σ_ρ-разведка уже
оправдана качеством, поэтому денежная нога от неё = 0.

| § | Что введено | Где (код) | Тест |
|---|---|---|---|
| §16.A | константы `ROLE_OPTIMIZED/PRICE_INPUT/REFERENCE` + приоритет M2 | `src/design/branches.py` | `test_iteration16_roles.py` ✅ |
| §16.A | `response_role`/`branch_roles`/`responses_by_role`/`price_channel_suppressed` (роль из намерения, branch-local) | `src/apps/mixture_process_runner.py` | 6 passed ✅ |
| §16.B | `rho_optimized` в `price_attribution_alpha`/`price_attributed_value` (Гр-1: `α≡0`) | `src/optimize/economic_stop.py` | discriminating-probe Гр-2 ✅ |
| §16.C | интеграция роль→атрибуция (branch-local Гр-3): `price_channel_suppressed(branch)` кормит `rho_optimized` | (тест) | Гр-3 ✅ (8 passed total) |
| §16.D | ФАКТИЧЕСКАЯ денежная нога стопа ветки `branch_economic_value`/`branch_stop_decision` (роль ρ→`price_attributed_value`, branch-local) | `src/apps/mixture_process_runner.py` | `test_iteration16_money_stop.py` ✅ (10 passed) |

### §16.0.D Денежная нога стопа, читающая роль (реализовано, ✅)

`rho_optimized` доведён из ЧИСТЫХ функций `economic_stop` до РАННЕРА:
:meth:`MixtureProcessRunner.branch_economic_value` считает ₽ за горизонт через
`price_attributed_value` с `rho_optimized=self.price_channel_suppressed(bid)`
(σ_ρ-разведка через `expected_price_improvement`, атрибуция §5 — деньги только
за прирост d_overall ИМЕННО через цену, роль ρ читается branch-local).
:meth:`branch_stop_decision` кормит это в двойной стоп §4 (`evaluate_stop`).
Опорный инкумбент — `_branch_reference_recipe` (измеренный `x_best` ветки или
центроид+baseline). Покрыто матрицей (роль × сценарий): ρ=цель-без-цены / ρ=цена
(PRICE_INPUT, канал жив, ₽>0) / ρ=цель+цена (OPTIMIZED, канал занулён, ₽=0 РОВНО) /
многоного́я ветка / ветка без ρ; + регрессия `evaluate_stop`: при занулённом
канале денежная нога НЕ фантомит (`not_economical`/red-flag честно; ADVISORY не
ветирует, но несёт red-flag).

> Трактовка «несколько ценовых ног»: архитектура (`set_branch_cost`) держит ОДНУ
> ценовую ногу на ветку; «несколько ног» прочитано как многоного́я по КАЧЕСТВУ
> ветка + ценовая нога (знаменатель атрибуции Σw + роль ρ — то, что §16 проверяет).

### §16.1 ОСТАТОК — открытый пункт ТЗ (E+, НЕ реализовано)

Канон тот же: сначала логика + unit-тест (D — закрыт), потом UI.

**E+. UI + AI/MCP (логика D зелёная).**
- [ ] read-only показ роли каждого отклика в ветке и факта зануления σ_ρ-канала
  (`render_branches`, экспандер §15.6/§16), A0.6 — система ничего не меняет молча;
- [ ] прокинуть роль/атрибуцию в `doe-introspect` (MCP) и в assistant —
  объяснение «почему денег за ρ нет» (двойной счёт убран);
- [ ] обновить «Карту реализации» выше строкой §16, когда E+ закрыт.

---

*Документ — основной источник истины пересборки. Обновлять при изменении решений.*



