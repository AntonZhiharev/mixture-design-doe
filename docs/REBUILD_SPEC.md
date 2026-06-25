# 🔧 REBUILD SPEC — MoE/GP Pipeline для Mixture DOE (ПВХ)

> Итоговое самодостаточное ТЗ для глобальной пересборки приложения.
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

*Документ — основной источник истины пересборки. Обновлять при изменении решений.*



