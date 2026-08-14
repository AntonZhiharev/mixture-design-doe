# Продолжение проекта на другой машине + MCP-сервер `doe-campaign`

Памятка для переноса работы в новую сессию / на другую машину. Всё необходимое
лежит в git; единственное, что НЕ хранится в репозитории — локальная регистрация
MCP-сервера в настройках Cline (она машинно-зависимая). Ниже — как всё поднять.

## 0. Какой сервер актуален (состояние на 11.08.2026)

| Сервер | Что отдаёт | Статус |
|---|---|---|
| **`doe-campaign`** (§7) | read-only инструменты КАМПАНИИ из ядра: `get_spec`, `explain_node`, `simulate_bounds`, `preflight`, `point_report`, `encode_recipe`, … | **рабочий, единственный включённый** |
| `doe-introspect` (§3–5) | прогоны pipeline M1–M8 из каталога trace (`list_runs`, `run_overview`, `get_stage`, …) | **легаси**: `"disabled": true` в настройках Cline; код и тесты оставлены |

Истина по слою ассистента — `docs/ASSISTANT_SPEC.md` (iter58–iter66). По нему
разбор кампании ведётся ТОЛЬКО через `doe-campaign`: числа считает ядро, а не
пересказ исходников. `doe-introspect` — про другой артефакт (trace старого
pipeline-потока) и в кампанийном потоке не нужен; включать его стоит, только
если вернулись к разбору прогонов `run_iteration7_demo.py` / benchmark.
Два включённых сервера с похожими именами — источник путаницы, поэтому
легаси держим выключенным, а не удалённым.

Проверить фактическое состояние (какой файл настроек живой и что в нём):
```powershell
.venv\Scripts\python.exe tools\check_mcp_settings.py
```
Скрипт печатает ОБА хранилища настроек Cline и возвращает код 1, если канон
нарушен. **Важно:** в Cline 4.1.x путь к настройкам ДРУГОЙ, чем в 3.x — см. §5.


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

> **Настраиваешь с нуля? Переходи сразу к §7 (`doe-campaign`).**
> Разделы §3–§5 ниже — про легаси-сервер `doe-introspect`, он выключен.

## 3. Данные для интроспекции (trace) — ЛЕГАСИ
MCP-сервер `doe-introspect` читает сохранённые прогоны pipeline из каталога trace. Артефакты

прогонов в git НЕ коммитятся (`project_demo/`, `project_ui/` в `.gitignore`),
поэтому на новой машине их надо СГЕНЕРИРОВАТЬ:
```powershell
.venv\Scripts\python.exe run_iteration7_demo.py     # создаст project_demo/trace/<run_id>
```
Каталог trace задаётся переменной `DOE_TRACE_ROOT` (по умолчанию `<repo>/project_demo/trace`).

## 4. Самопроверка `doe-introspect` (без MCP-транспорта) — ЛЕГАСИ

```powershell
.venv\Scripts\python.exe src/mcp/introspect_server.py --selftest
```
Должен напечатать `TRACE_ROOT` и список прогонов. Если пусто — сначала шаг 3.

## 5. Где лежит `cline_mcp_settings.json` (ВАЖНО: путь зависит от версии Cline)

> **Грабли, пойманные 11.08.2026.** Cline **4.1.x** перенёс хранилище настроек
> MCP из `globalStorage` расширения в домашний каталог и сменил СХЕМУ (поля
> `command`/`args`/`env` теперь вложены в блок `transport`). При обновлении
> расширение мигрирует конфиг ОДИН раз; после этого правки старого файла в
> `globalStorage` **ни на что не влияют** — Cline их больше не читает.
> Симптом: в чате доступен старый набор инструментов (например,
> легаси `list_runs` вместо кампанийных `get_spec`/`preflight`), хотя в старом
> файле всё «правильно». Проверять — по фактически запущенному процессу:
> `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object CommandLine`.

Актуальное (авторитетное) хранилище — **Cline 4.1.x и новее**:

- Windows: `%USERPROFILE%\.cline\data\settings\cline_mcp_settings.json`
- macOS/Linux: `~/.cline/data/settings/cline_mcp_settings.json`

Легаси-хранилище — **Cline 3.x** (оставлено для истории; новыми версиями не читается):

- Windows: `%APPDATA%\Code\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
- macOS: `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
- Linux: `~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

Не уверен, какой файл живой — правь оба (они не конфликтуют) и сверься
скриптом из §7. После сохранения нужен **реконнект MCP** (кнопка в панели
Cline) или перезапуск VS Code: набор инструментов в уже открытой сессии сам
не обновится.

### Регистрация `doe-introspect` — ЛЕГАСИ (по умолчанию выключен)

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
и L1-факта, фиксация предложенной записи журнала (`apply_note` / `reject_note`,
iter96) — акт человека кнопкой в приложении (разовый токен). Класс `propose`
(патч, пакет, правка полей формы, предложенная запись журнала —
`propose_decision` / `propose_fact` — в стейдж сессии) и `sandbox` тоже не
выдаются: предложение обязано попасть в панель приложения, где у человека есть
кнопки. Попытка позвать их возвращает объяснение, а не выполнение.

Самопроверка (пакет `mcp` не нужен):
```powershell
$env:DOE_CAMPAIGN_ROOT = "<REPO>\project_campaigns"
.venv\Scripts\python.exe src/mcp/campaign_server.py --selftest
```
Регистрация в `cline_mcp_settings.json` (путь к файлу — см. §5, он зависит от
версии Cline!). Это и есть рабочая конфигурация текущей машины; легаси-сервер
лежит рядом с `"disabled": true`.

**Формат Cline 4.1.x** (транспорт вынесен в блок `transport`) — актуальный:

```json
{
  "mcpServers": {
    "doe-campaign": {
      "transport": {
        "type": "stdio",
        "command": "C:\\Users\\anton\\Documents\\DOE\\.venv\\Scripts\\python.exe",
        "args": ["C:\\Users\\anton\\Documents\\DOE\\src\\mcp\\campaign_server.py"],
        "env": {
          "DOE_CAMPAIGN_ROOT": "C:\\Users\\anton\\Documents\\DOE\\project_campaigns",
          "PYTHONPATH": "C:\\Users\\anton\\Documents\\DOE",
          "PYTHONIOENCODING": "utf-8"
        }
      },
      "disabled": false,
      "autoApprove": [],
      "timeout": 60
    },
    "doe-introspect": {
      "transport": {
        "type": "stdio",
        "command": "C:\\Users\\anton\\Documents\\DOE\\.venv\\Scripts\\python.exe",
        "args": ["C:\\Users\\anton\\Documents\\DOE\\src\\mcp\\introspect_server.py"],
        "env": {
          "DOE_TRACE_ROOT": "C:\\Users\\anton\\Documents\\DOE\\project_demo\\trace",
          "PYTHONPATH": "C:\\Users\\anton\\Documents\\DOE",
          "PYTHONIOENCODING": "utf-8"
        }
      },
      "disabled": true,
      "autoApprove": []
    }
  }
}
```

`PYTHONIOENCODING=utf-8` обязателен на Windows: диагностика серверов пишется
по-русски, без него stdio-канал ломается на cp1251.

**Формат Cline 3.x** (легаси, плоские поля — если у тебя старое расширение):

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

Сверить, что Cline видит именно то, что нужно (печатает оба хранилища и
состояние `disabled` по каждому серверу):
```powershell
.venv\Scripts\python.exe tools\check_mcp_settings.py
```
Ожидаемый вывод: `doe-campaign` — вкл, `doe-introspect` — ВЫКЛ. Если в
НОВОМ хранилище `doe-campaign` отсутствует, а легаси включён — ты правил
не тот файл (см. врезку в §5).
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
- `tools/check_mcp_settings.py` — сверка настроек Cline с каноном §0
  (оба хранилища, обе схемы записи).
