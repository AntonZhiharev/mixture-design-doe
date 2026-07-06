# 🧪 Mixture Design DOE — кампания оптимизации смесевых рецептур (MoE/GP pipeline)

Платформа для планирования и последовательной оптимизации **смесевых составов**
(mixture, Σxᵢ = 1) с **процесс-параметрами** (T, P, … — куб вне симплекса):
D-/I-оптимальные дизайны, суррогатные модели **GP (mean = Scheffé) + MoE**,
active learning, desirability-оптимизация, экономический стоп и blocking по партиям.

UI — единое Streamlit-приложение «кампания»; ядро — чистые Python-подпакеты в `src/`
(`numpy / scipy / scikit-learn`), сверенные golden-тестами с эталоном R.

> Источник истины по архитектуре — `docs/REBUILD_SPEC.md` (§0–§15) и
> `docs/REBUILD_SPEC_17_campaign_flow.md` (§17 — главный поток кампании).

---

## 🚀 Быстрый старт

```bash
# 1. Виртуальное окружение
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Зависимости
pip install -r requirements.txt

# 3. Запуск главного приложения
python run_streamlit_app.py
# или напрямую:
streamlit run src/apps/streamlit_app.py
```

Браузер: **http://localhost:8501**

---

## 🧭 Главный поток — кампания (§17)

Приложение работает на одном движке: `MixtureProcessRunner` + `CampaignController`.
Одна модель физики на проект (`property → MoE`), единая общая база точек с
origin-тегами, ветки — контейнеры намерения (цели/бюджет/история/рецепт), без
собственных моделей.

```
1. СЕТАП        компоненты смеси (границы, baseline, Σ=1)
                + процесс-параметры (реальные единицы, code↔real)
                + отклики (свойства); ρ может быть ценовой ногой
2. SEED-ДИЗАЙН  D-optimal стартовый план по составной области (симплекс×куб),
                опц. blocking по партиям; РУЧНОЙ ввод измеренных Y
                («Заполнить тестовыми» — демо-оракул для прогонов без лаборатории)
3. СКРИНИНГ     Scheffé fit + значимость (M3) по накопленной базе
4. ВЕТКИ        ручные, мультицелевые: min/max/target, веса, роли,
                ценовая нога ρ, ограничения области
5. РАБОЧИЙ СТОЛ propose_points (exploit-argmax + explore-acquisition)
                → ручной ввод Y → commit_measured в ОБЩУЮ базу
                → переобучение суррогатов → пересчёт x*/d_best
                → двойной стоп (технический + экономический §15.6)
6. M8 РЕЦЕПТ    argmax desirability на составной области =
                оптимальный рецепт {состав + режимы}
7. ЭВОЛЮЦИЯ     добавить/изменить компонент/параметр/отклик/границы
                в любой момент — миграция точек по явной политике (§16.2)
```

Дополнительно в UI:
- **💬 ИИ-ассистент** с контекстом кампании (`src/apps/assistant.py`);
- **📁 Персистентность** — сохранение/загрузка кампаний в `project_campaigns/`
  (`src/apps/campaign_state.py`), включая черновик seed до фиксации;
- **Blocking** — оптимальные блоки seed-дизайна (interchange по блочному
  D-критерию), каждый добор-раунд = новый блок/партия; колонка «Блок» в таблицах;
- **Excel-выгрузка** общей базы и рецепта ветки; шаблоны ввода откликов
  (`tools/response_helper.py`).

---

## 🏗️ Ядро pipeline M1–M8 (библиотека)

| Модуль | Что делает | Код |
|---|---|---|
| M1 | геометрия constrained-симплекса, extreme vertices, псевдокомпоненты/ILR | `core/simplex.py` |
| M2 | D-optimal screening design (batch coordinate-exchange + restarts) | `design/d_optimal.py` |
| M3 | Scheffé OLS + ANOVA + значимость; ARD-GP lengthscales → q_eff | `models/scheffe.py`, `models/screening.py` |
| M4 | GMM-кластеризация режимов по свойствам (BIC) | `models/clustering.py` |
| M5 | I-optimal локальные дизайны | `design/i_optimal.py` |
| M6 | эксперты GP (mean=Scheffé, kernel=Matérn5/2-ARD) + MoE-gating | `models/gp_expert.py`, `models/moe.py` |
| M7 | active learning: EI / LCB / max-variance, argmax по симплексу | `design/active_learning.py` |
| M8 | desirability (Derringer–Suich) + стоимость | `optimize/desirability.py` |
| — | экономический стоп, VoI, движение границ | `optimize/economic_stop.py`, `optimize/voi.py`, `design/move_bounds.py` |
| — | blocking (стартовый + sequential) | `design/blocking.py`, `design/block_model.py` |
| — | эволюция схемы (компоненты/параметры/отклики) | `core/schema.py`, `core/schema_evolution.py` |
| M9 | трассировка pipeline (PipelineTrace) | `observability/trace.py` |

`PipelineRunner` (mixture-only конвейер M1–M8 с авто-M7) выведен из UI (§17.6),
но остаётся в `src/apps/pipeline_runner.py` как библиотека для юнит-тестов ядра
и бенчмарка.

---

## 🗂️ Структура проекта

```
DOE/
├── run_streamlit_app.py        ← лаунчер главного приложения (кампания)
├── run_pipeline_benchmark.py   ← бенчмарк pipeline против аналитического оптимума
│
├── src/
│   ├── apps/
│   │   ├── streamlit_app.py            ← ГЛАВНОЕ приложение (кампания §17)
│   │   ├── campaign_ui.py              ← рендер кампании (сетап/ветки/рабочий стол)
│   │   ├── campaign.py                 ← CampaignController (валидация, петля)
│   │   ├── campaign_state.py           ← персистентность кампаний (save/load)
│   │   ├── campaign_screening.py       ← скрининг M3 в campaign-flow
│   │   ├── mixture_process_runner.py   ← MixtureProcessRunner (движок кампании)
│   │   ├── pipeline_runner.py          ← PipelineRunner M1–M8 (библиотека)
│   │   ├── assistant.py                ← ИИ-ассистент кампании
│   │   └── admin.py, battle_preset.py
│   │
│   ├── core/       ← симплекс, схема, эволюция схемы, state, linalg, синтетика
│   ├── design/     ← D/I-optimal, active learning, blocking, ветки, границы
│   ├── models/     ← Scheffé, screening, GMM, GP-эксперты, MoE, диагностика
│   ├── optimize/   ← desirability, экономический стоп, VoI
│   ├── observability/ ← PipelineTrace (M9)
│   ├── mcp/        ← MCP-сервер doe-introspect (инспекция прогонов)
│   ├── verification/ ← golden-фикстуры против R + battle-truth оракулы
│   ├── algorithms/ ← легаси-алгоритмы (референс)
│   └── utils/      ← ANOVA, отчёты PDF/DOCX, утилиты
│
├── tools/response_helper.py    ← Excel-шаблоны ввода откликов
├── docs/                       ← REBUILD_SPEC и рабочие конспекты
└── tests/
    ├── unit/                   ← test_iteration{N}_*.py (логика + UI-гейты)
    ├── integration/
    ├── performance/
    └── golden/
```

---

## 🧪 Тесты

Логика всегда идёт впереди UI: каждая итерация = логика + unit-тест, потом UI.

```bash
# Профильные тесты — перечислять файлы ЯВНО (glob в cmd/PowerShell не работает):
.venv\Scripts\python.exe -m pytest tests/unit/test_iteration27_blocking.py tests/unit/test_iteration28_campaign_blocking.py -q -W ignore

# Golden-тесты (сверка с эталоном R 4.6.0, офлайн-фикстуры):
.venv\Scripts\python.exe -m pytest tests/golden/test_golden.py tests/golden/test_golden_gp_composite.py tests/golden/test_golden_product_moments.py -q -W ignore
```

> ⚠️ Не запускать `pytest tests/unit` целиком: `tests/unit/test_precision.py`
> падает на сборке (`ModuleNotFoundError: core`) — известная несвязанная проблема.

---

## 📈 Бенчмарк

```bash
.venv\Scripts\python.exe run_pipeline_benchmark.py
```

Синтетический полигон: q=5, истинная функция = Scheffé-quadratic + шум;
pipeline сравнивается с аналитическим оптимумом (см. `src/verification/`).

---

## 🔌 MCP-сервер doe-introspect

`src/mcp/introspect_server.py` — MCP-сервер для инспекции сохранённых прогонов
pipeline из ИИ-ассистента (Cline и др.): список прогонов, стадии, метрики,
прогресс active-learning, diff раундов, итог бенчмарка.

Установка и регистрация — `docs/MCP_SETUP.md` (машинно-зависимая, в git не хранится).

---

## 📖 Документация

| Файл | Тема |
|------|------|
| `docs/REBUILD_SPEC.md` | **Главное ТЗ**: архитектура M1–M8, формулы, ядра GP, checkpointing, боевая аугментация, экономический стоп |
| `docs/REBUILD_SPEC_17_campaign_flow.md` | §17 — единый поток кампании (текущий главный UI) |
| `docs/BATTLE_PLAN_17.md` | план боевых прогонов и итераций |
| `docs/MCP_SETUP.md` | установка MCP-сервера doe-introspect |
| `docs/getting_started.md` | базовое использование (легаси-часть) |
| остальные `docs/*.md` | исследовательские заметки: ANOVA, false positives, JMP-сравнения, hierarchical screening и др. |

---

## 🗄️ Легаси

Старые Streamlit-приложения (`efficient_sequential_workflow_app.py`,
`doe_sequential_workflow_app.py`, `sequential_reconstruction_app.py`,
`staged_parameter_recovery_app.py`) и лаунчеры (`run_sequential_interface.py`,
`run_efficient_workflow.py`, `run_iteration*_demo.py` и пр.) сохранены как
референс, но **не являются главным потоком** — актуальное приложение одно:
`src/apps/streamlit_app.py`.

---

## 📄 Лицензия

MIT License.