# 🔧 REBUILD SPEC — MoE/GP Pipeline для Mixture DOE (ПВХ)

> Итоговое самодостаточное ТЗ для глобальной пересборки приложения.
> Версия 1.5 (синхронизирована с текущей сборкой: реализованы M1–M8, итерации 1–6; **M9 (PipelineTrace + MCP `doe-introspect`)** — §11, итерация 7; **golden-тесты против R 4.6.0** — §6; реализовано **Streamlit-приложение pipeline M1–M8 + чекпоинты** (итерация 8); см. §8).




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

*Документ — основной источник истины пересборки. Обновлять при изменении решений.*



