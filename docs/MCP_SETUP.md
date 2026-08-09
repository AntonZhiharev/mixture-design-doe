# Продолжение проекта на другой машине + MCP-серверы `doe-introspect` / `doe-campaign`

Памятка для переноса работы в новую сессию / на другую машину. Всё необходимое
лежит в git; единственное, что НЕ хранится в репозитории — локальная регистрация
MCP-сервера в настройках Cline (она машинно-зависимая). Ниже — как всё поднять.

## 1. Клонирование и окружение
```powershell
git clone https://github.com/AntonZhiharev/mixture-design-doe.git
cd mixture-design-doe

# виртуальное окружение (Windows / PowerShell)
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # включает пакет mcp>=1.2.0
```
macOS/Linux: `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt`.

## 2. Прогон тестов (санити-чек)
```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_iteration9_branches.py tests/unit/test_iteration9_misspec.py tests/unit/test_iteration8_app.py -q -W ignore
```
> Заноза: `tests/unit/test_precision.py` падает на сборке (`No module named 'core'`) —
> давняя несвязанная проблема. Перечисляй нужные файлы явно, не гоняй `tests/unit` целиком.

## 3. Данные для интроспекции (trace)
MCP-сервер читает сохранённые прогоны pipeline из каталога trace. Артефакты
прогонов в git НЕ коммитятся (`project_demo/`, `project_ui/` в `.gitignore`),
поэтому на новой машине их надо СГЕНЕРИРОВАТЬ:
```powershell
.venv\Scripts\python.exe run_iteration7_demo.py     # создаст project_demo/trace/<run_id>
```
Каталог trace задаётся переменной `DOE_TRACE_ROOT` (по умолчанию `<repo>/project_demo/trace`).

## 4. Самопроверка сервера (без MCP-транспорта)
```powershell
.venv\Scripts\python.exe src/mcp/introspect_server.py --selftest
```
Должен напечатать `TRACE_ROOT` и список прогонов. Если пусто — сначала шаг 3.

## 5. Регистрация MCP-сервера в Cline
Открой настройки MCP Cline (файл `cline_mcp_settings.json`):
- Windows: `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
- macOS: `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- Linux: `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

Добавь сервер (замени `<REPO>` на абсолютный путь к клонированному репозиторию):
```json
{
  "mcpServers": {
    "doe-introspect": {
      "command": "<REPO>/.venv/Scripts/python.exe",
      "args": ["<REPO>/src/mcp/introspect_server.py"],
      "env": {
        "DOE_TRACE_ROOT": "<REPO>/project_demo/trace",
        "PYTHONPATH": "<REPO>"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```
Пример для Windows (как на исходной машине):
```json
{
  "mcpServers": {
    "doe-introspect": {
      "command": "d:\\DOE\\.venv\\Scripts\\python.exe",
      "args": ["d:\\DOE\\src\\mcp\\introspect_server.py"],
      "env": {
        "DOE_TRACE_ROOT": "d:\\DOE\\project_demo\\trace",
        "PYTHONPATH": "d:\\DOE"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```
macOS/Linux: `command` → `<REPO>/.venv/bin/python`, пути в POSIX-формате.

После сохранения файла Cline подхватит сервер; появятся инструменты
`list_runs / run_overview / get_stage / get_metrics / get_design /
al_progression / diff_rounds / get_benchmark` (см. `src/mcp/introspect_server.py`,
логика — `src/mcp/queries.py`).

## 6. Встроенный ИИ-ассистент в Streamlit + мост в trace
В приложении (`src/apps/streamlit_app.py`) есть вкладка **«💬 Ассистент»**
(модуль `src/apps/assistant.py`). Это гибрид:

* **Чат прямо в приложении.** Ассистент видит «живой» контекст страниц
  (метрики стадий M1…M8, конфиг, ветки, benchmark из `runner`) и отвечает через
  OpenRouter. Включается переменной окружения **`OPENROUTER_API_KEY`**; модель
  переопределяется `DOE_ASSISTANT_MODEL` (по умолчанию `anthropic/claude-3.5-sonnet`).
  Сетевой вызов — на stdlib `urllib`, новых зависимостей нет.
* **Мост к Cline в VS Code.** Кнопка «🔄 Опубликовать снапшот для Cline» (и
  автоматически — на каждый вопрос в чате) пишет тот же снимок в каталог
  `DOE_TRACE_ROOT` как обычный прогон `PipelineTrace` с `run_id = ui_<проект>`.
  После этого Cline наблюдает ровно те же данные через MCP `doe-introspect`
  (`list_runs` → найти `ui_*`, затем `run_overview` / `get_stage` / `get_design`).

Запуск с ключом (PowerShell):
```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."          # ключ OpenRouter
$env:DOE_TRACE_ROOT = "<REPO>\project_demo\trace"  # тот же, что у MCP-сервера
.venv\Scripts\python.exe run_streamlit_app.py
```
Без ключа чат отключён, но кнопка публикации снапшота в trace работает —
наблюдение через Cline доступно и без LLM.

## 7. MCP-сервер `doe-campaign` (iter66): числа кампании прямо в Cline

`doe-introspect` читает ПРОГОНЫ pipeline (trace). `doe-campaign` — другое:
это те же read-only инструменты, что у ассистента-архитектора в приложении
(`src/assistant/tools`), только для Cline. Смысл — не рассуждать о геометрии
по исходникам: эффективные границы, `spec_hash`, `preflight`, разбор рецепта
считает ЯДРО.

Инструменты: `list_projects`, `project_status`, `list_tools` + весь класс
`readonly` реестра (`get_spec`, `explain_node`, `validate_spec` (dry-run),
`simulate_bounds`, `preflight`, `point_report`, `encode_recipe`, `get_runs`,
`campaign_overview`, `get_local_facts`, `get_decisions`, `list_attachments`,
`read_attachment`, `sandbox_info`). У каждого есть аргумент `project` — имя
проекта в каталоге кампаний; его можно опустить, если проект один.

**Класс `write` не экспортируется вообще**: применение патча, запись решения
и L1-факта — акт человека кнопкой в приложении (разовый токен). Класс
`propose` (патч в стейдж сессии) и `sandbox` тоже не выдаются. Попытка
позвать их возвращает объяснение, а не выполнение.

Самопроверка (пакет `mcp` не нужен):
```powershell
$env:DOE_CAMPAIGN_ROOT = "<REPO>\project_campaigns"
.venv\Scripts\python.exe src/mcp/campaign_server.py --selftest
```
Регистрация в `cline_mcp_settings.json` (рядом с `doe-introspect`):
```json
{
  "mcpServers": {
    "doe-campaign": {
      "command": "d:\\DOE\\.venv\\Scripts\\python.exe",
      "args": ["d:\\DOE\\src\\mcp\\campaign_server.py"],
      "env": {
        "DOE_CAMPAIGN_ROOT": "d:\\DOE\\project_campaigns",
        "PYTHONPATH": "d:\\DOE"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```
Каталог проектов по умолчанию — `<repo>/project_campaigns` (тот же, что у
приложения), переопределяется `DOE_CAMPAIGN_ROOT`. Проекты создаются
сохранением кампании в интерфейсе; проект без `campaign.json` честно
отвечает «движка нет — это не проверено, а не «всё хорошо»».

Каждый вызов дописывается в аудит проекта
(`project_campaigns/<проект>/assistant/tool_calls.jsonl`) с пометкой
`via="mcp"` — разбор через Cline виден там же, где разбор через док;
сессия ассистента при этом не переписывается.

## 8. Где продолжать по плану
- `docs/REBUILD_SPEC.md` — спецификация (канон §5/§12).
- `docs/FinalCheckList.md` + `docs/FinalCheckList_audit.md` — чек-лист и статус по блокам.
- `.clinerules` — правила работы и синхронизации с git в каждой сессии.
- `docs/ASSISTANT_SPEC.md` — слой ассистента (iter58–iter66) и MCP-контракт.
