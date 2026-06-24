"""assistant.py — встроенный ИИ-ассистент приложения + мост в trace (M9/§11).

Гибрид (вариант A + мост):
  * **Контекст страниц.** :func:`build_context` собирает JSON-сериализуемый
    снимок текущего состояния `PipelineRunner` (конфиг, пройденные стадии,
    ключевые метрики M1…M8, ветки, benchmark) — то, что видит пользователь
    на страницах приложения.
  * **LLM-ответ.** :func:`assistant_reply` отдаёт этот контекст + историю
    диалога модели через OpenRouter (тот же стек, что у Cline) и возвращает
    текстовый ответ. Сетевой вызов — на stdlib ``urllib`` (без новых зависимостей).
  * **Мост в trace.** :func:`write_live_snapshot` пишет тот же снимок в каталог
    ``DOE_TRACE_ROOT`` как обычный прогон `PipelineTrace`. Благодаря этому
    Cline в VS Code наблюдает ровно те же данные через MCP-сервер
    ``doe-introspect`` (list_runs / run_overview / get_stage / get_metrics).

Модуль НЕ зависит от Streamlit — чистая логика, тестируемая напрямую.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from src.observability.trace import PipelineTrace, _jsonable

# ----------------------------------------------------------------------
# Конфигурация backend (через переменные окружения)
# ----------------------------------------------------------------------
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
LIVE_RUN_PREFIX = "ui"


def default_trace_root() -> str:
    """Каталог trace: ``DOE_TRACE_ROOT`` или ``<repo>/project_demo/trace``.

    Должен совпадать с тем, что задан MCP-серверу ``doe-introspect`` —
    тогда снапшот приложения виден Cline в VS Code без доп. настройки.
    """
    env = os.environ.get("DOE_TRACE_ROOT")
    if env:
        return env
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "project_demo", "trace")


def model_name() -> str:
    return os.environ.get("DOE_ASSISTANT_MODEL", DEFAULT_MODEL)


def api_key() -> Optional[str]:
    """Ключ OpenRouter из окружения (``OPENROUTER_API_KEY``)."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    return key.strip() if key else None


def llm_available() -> bool:
    return bool(api_key())


# ----------------------------------------------------------------------
# Локальный .env: сохранение ключа между запусками (НЕ коммитится в git)
# ----------------------------------------------------------------------
# Файл .env лежит в корне репозитория и занесён в .gitignore — ключ остаётся
# только на машине пользователя и не попадает на GitHub.
_ENV_KEYS = ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "DOE_ASSISTANT_MODEL",
             "DOE_TRACE_ROOT")


def env_file_path() -> str:
    """Путь к локальному ``.env`` в корне репозитория."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, ".env")


def _parse_env_text(text: str) -> Dict[str, str]:
    """Разобрать содержимое .env в словарь (KEY=VALUE, # — комментарии)."""
    out: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            out[k] = v
    return out


def load_env_file(path: Optional[str] = None, *, override: bool = False) -> Dict[str, str]:
    """Загрузить переменные из локального ``.env`` в ``os.environ``.

    По умолчанию НЕ перетирает уже заданные переменные окружения
    (``override=False``) — внешнее окружение приоритетнее файла. Возвращает
    словарь применённых значений. Отсутствие файла — не ошибка (пустой dict).
    """
    path = path or env_file_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            parsed = _parse_env_text(fh.read())
    except OSError:
        return {}
    applied: Dict[str, str] = {}
    for k, v in parsed.items():
        if override or not os.environ.get(k):
            os.environ[k] = v
            applied[k] = v
    return applied


def save_api_key(key: str, *, model: Optional[str] = None,
                 path: Optional[str] = None) -> str:
    """Сохранить ключ OpenRouter (и опц. модель) в локальный ``.env``.

    Обновляет/добавляет только наши ключи, прочие строки файла сохраняет.
    Также сразу прокидывает значения в ``os.environ`` текущего процесса.
    Возвращает путь к файлу. Бросает ValueError при пустом ключе.
    """
    key = (key or "").strip()
    if not key:
        raise ValueError("Пустой ключ — нечего сохранять.")
    path = path or env_file_path()

    existing: Dict[str, str] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = _parse_env_text(fh.read())
        except OSError:
            existing = {}

    existing["OPENROUTER_API_KEY"] = key
    if model and model.strip():
        existing["DOE_ASSISTANT_MODEL"] = model.strip()

    lines = ["# Локальные секреты DOE — НЕ коммитить (файл в .gitignore).",
             f"# Обновлено: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    for k, v in existing.items():
        lines.append(f"{k}={v}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # сразу применяем в текущем процессе
    os.environ["OPENROUTER_API_KEY"] = key
    if model and model.strip():
        os.environ["DOE_ASSISTANT_MODEL"] = model.strip()
    return path


def api_key_persisted(path: Optional[str] = None) -> bool:
    """Есть ли сохранённый ключ в локальном ``.env``."""
    path = path or env_file_path()
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return bool(_parse_env_text(fh.read()).get("OPENROUTER_API_KEY"))
    except OSError:
        return False



def _safe_run_id(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", str(name)).strip("_") or "project"
    return f"{LIVE_RUN_PREFIX}_{slug}"


# ----------------------------------------------------------------------
# Сборка контекста страниц из runner
# ----------------------------------------------------------------------
def _stage_metrics(runner) -> Dict[str, Any]:
    """Ключевые (скалярные/мелкие) метрики пройденных стадий.

    Делегирует :meth:`PipelineRunner.stage_metrics` — это объединение кэша с
    диска (пережившего перезагрузку проекта) и свежих результатов текущей
    сессии. Полные матрицы дизайна сюда НЕ входят (велики); они попадают только
    в trace-снапшот (см. :func:`_trace_stage_payloads`).
    """
    if hasattr(runner, "stage_metrics"):
        return dict(runner.stage_metrics())
    # запасной путь для дак-объектов без метода (только in-memory results)
    if hasattr(runner, "stage_metrics_compact"):
        return dict(runner.stage_metrics_compact())
    return {}



def _branches_summary(runner) -> List[Dict[str, Any]]:
    out = []
    for bid, b in (getattr(runner, "branches", {}) or {}).items():
        out.append({
            "id": bid,
            "name": getattr(b, "name", bid),
            "goal": {p: getattr(s, "kind", str(s)) for p, s in
                     (getattr(b, "goal", {}) or {}).items()},
            "budget": getattr(b, "budget", None),
            "spent": getattr(b, "spent", None),
            "remaining": b.remaining() if hasattr(b, "remaining") else None,
            "status": getattr(b, "status", None),
            "d_best": getattr(b, "d_best", None),
            "stagnating": b.is_stagnating() if hasattr(b, "is_stagnating") else None,
        })
    return out


def completed_stages(runner) -> List[str]:
    """Пройденные стадии по ДОЛГОВЕЧНОМУ состоянию проекта, а не только по
    in-memory ``results``.

    Важно после загрузки проекта (``from_project``): восстанавливаются данные и
    модели (``state.models``, design), но словарь ``runner.results`` остаётся
    пустым до повторного запуска стадий в текущей сессии. Без этого ассистент
    ошибочно считал, что «предыдущие стадии не выполнены». Здесь объединяем то,
    что выполнено в сессии (``results``), с тем, что восстановлено из проекта
    (наличие соответствующих моделей/данных).
    """
    done = set(getattr(runner, "results", {}) or {})
    # кэш метрик с диска тоже подтверждает пройденность стадии
    done |= set(getattr(runner, "cached_metrics", {}) or {})
    state = getattr(runner, "state", None)
    models = getattr(state, "models", {}) or {}
    design = getattr(runner, "design", None)


    def has_model(prefix: str) -> bool:
        return any(k == prefix or k.startswith(prefix) for k in models)

    if design is not None:               # есть план/данные ⇒ геометрия+M2 были
        done.update({"M1", "M2"})
    if has_model("m3_scheffe"):
        done.add("M3_fit")
    if has_model("m3_ard_screening"):
        done.add("M3_ard")
    if has_model("m4_regimes"):
        done.add("M4")
    if has_model("m5_i_optimal"):
        done.add("M5")
    if has_model("m6_moe"):
        done.add("M6")
    if has_model("m7_final_moe"):
        done.add("M7")
    try:
        if state is not None and state.get("m8_recipe") is not None:
            done.add("M8")
    except Exception:  # noqa: BLE001 — get может бросить на пустом state
        pass
    return sorted(done)


def build_context(runner, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """JSON-сериализуемый снимок состояния приложения для ассистента."""
    stages_done = completed_stages(runner)
    metrics = _stage_metrics(runner)
    # стадии, пройденные ранее (восстановлены из проекта), но без детальных
    # метрик в текущей сессии — чтобы ассистент не выдумывал отсутствующие числа
    stages_without_metrics = [s for s in stages_done if s not in metrics]

    n_points = (0 if getattr(runner, "design", None) is None
                else int(len(runner.design)))
    responses_filled = False
    y = getattr(runner, "y", None)
    if y is not None:
        try:
            responses_filled = not bool(np.any(np.isnan(np.asarray(y, float))))
        except Exception:  # noqa: BLE001
            responses_filled = False

    ctx: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": getattr(getattr(runner, "cfg", None), "name", None),
        "config": _config_brief(runner),
        "current_stage": getattr(getattr(runner, "state", None), "stage", None),
        "stages_done": stages_done,
        # стадии, формально пройденные (модели восстановлены из проекта), но без
        # детальных метрик в текущей сессии — НЕ выдумывать по ним числа
        "stages_without_metrics": stages_without_metrics,
        "n_points": n_points,
        "responses_filled": responses_filled,
        "origin_counts": (runner.origin_counts()
                          if hasattr(runner, "origin_counts") and n_points else {}),
        "metrics": metrics,
        "branches": _branches_summary(runner),
    }

    if extra:
        ctx["ui"] = extra
    return _jsonable(ctx)


def _config_brief(runner) -> Dict[str, Any]:
    cfg = getattr(runner, "cfg", None)
    return {
        "q": getattr(runner, "q", None),
        "names": list(getattr(runner, "names", []) or []),
        "property_names": list(getattr(runner, "property_names", []) or []),
        "model": getattr(cfg, "model", None),
        "noise_sd": getattr(cfg, "noise_sd", None),
        "seed": getattr(cfg, "seed", None),
        "n_runs": getattr(runner, "n_runs", None),
    }


# ----------------------------------------------------------------------
# Мост в trace: запись живого снапшота как прогона PipelineTrace
# ----------------------------------------------------------------------
def _trace_stage_payloads(runner) -> List[Dict[str, Any]]:
    """Стадии для trace-снапшота: (stage, inputs, outputs, metrics).

    Сюда кладём и метрики, и компактные outputs (включая дизайн M2), чтобы
    MCP-инструменты ``get_stage`` / ``get_design`` отдавали полезные данные.
    """
    r = getattr(runner, "results", {}) or {}
    metrics = _stage_metrics(runner)
    payloads: List[Dict[str, Any]] = []

    def design_of(key: str):
        d = r.get(key, {}).get("design")
        return None if d is None else np.asarray(d).tolist()

    for stage in ("M1", "M2", "M3_fit", "M3_ard", "M4", "M5", "M6", "M7", "M8"):
        if stage not in r:
            continue
        outputs: Dict[str, Any] = {}
        if stage == "M2":
            outputs["design"] = design_of("M2")
        elif stage == "M5":
            outputs["design"] = design_of("M5")
        elif stage == "M8":
            outputs["recipe"] = r["M8"].get("recipe")
        elif stage == "M7":
            outputs["x_best"] = r["M7"].get("x_best")
        payloads.append({
            "stage": stage,
            "inputs": {},
            "outputs": outputs,
            "metrics": metrics.get(stage, {}),
        })

    if "benchmark" in r:
        b = r["benchmark"]
        payloads.append({
            "stage": "benchmark",
            "inputs": {},
            "outputs": {"recipe_pipeline": b.get("x_pipeline"),
                        "recipe_analytical": b.get("x_true")},
            "metrics": metrics.get("benchmark", {}),
        })
    return payloads


def _normalize_chat(chat_history) -> List[Dict[str, str]]:
    """Привести историю чата к списку ``{"role", "content"}`` (только диалог)."""
    out: List[Dict[str, str]] = []
    for m in (chat_history or []):
        role = (m.get("role") if isinstance(m, dict) else None)
        if role in ("user", "assistant"):
            out.append({"role": role,
                        "content": str(m.get("content", ""))})
    return out


def write_live_snapshot(runner, root: Optional[str] = None,
                        run_id: Optional[str] = None,
                        chat_history=None) -> Dict[str, str]:
    """Записать живой снимок приложения в trace (мост к Cline в VS Code).

    Если передана ``chat_history`` (список ``{"role", "content"}`` из чата
    ассистента), она пишется отдельной стадией ``assistant_chat`` — тогда Cline
    в VS Code видит саму переписку через MCP ``get_stage(run_id,
    "assistant_chat")`` и не требует повторных объяснений.

    Возвращает ``{"run_id", "path", "root", "n_messages"}``. Cline читает это
    через MCP-сервер ``doe-introspect`` теми же инструментами, что и обычные
    прогоны.
    """
    root = root or default_trace_root()
    name = getattr(getattr(runner, "cfg", None), "name", "project")
    rid = run_id or _safe_run_id(name)
    chat = _normalize_chat(chat_history)
    meta = {
        "source": "streamlit_live",
        "project": name,
        "config": _config_brief(runner),
        "current_stage": getattr(getattr(runner, "state", None), "stage", None),
        # полный список пройденных стадий (включая восстановленные из проекта) —
        # чтобы Cline видел прогресс даже до повторного запуска стадий в сессии
        "stages_completed": completed_stages(runner),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_chat_messages": len(chat),
    }

    tr = PipelineTrace(run_id=rid, root=root, meta=meta)
    for p in _trace_stage_payloads(runner):
        tr.log(p["stage"], inputs=p["inputs"], outputs=p["outputs"],
               metrics=p["metrics"])
    if chat:
        tr.log("assistant_chat", inputs={},
               outputs={"history": chat},
               metrics={"n_messages": len(chat)})
    path = tr.save()
    return {"run_id": rid, "path": path, "root": root,
            "n_messages": str(len(chat))}



# ----------------------------------------------------------------------
# LLM (OpenRouter через urllib)
# ----------------------------------------------------------------------
def system_prompt() -> str:
    return (
        "Ты — встроенный ИИ-ассистент инженерного приложения для планирования "
        "экспериментов со смесями (mixture DOE). Приложение прогоняет конвейер "
        "стадий M1…M8: M1 — геометрия области; M2 — D-оптимальный план и сбор "
        "откликов; M3 — анализ Шеффе + ARD-скрининг значимых компонентов; "
        "M4 — кластеризация режимов (GMM/BIC); M5 — I-оптимальный план; "
        "M6 — суррогатная модель MoE на каждое свойство; M7 — активное обучение; "
        "M8 — оптимизация рецептуры (desirability + стоимость). Есть «ветки» — "
        "цели по свойствам с бюджетом точек, и benchmark против известного "
        "оптимума синтетического полигона.\n\n"
        "Тебе передаётся JSON с актуальным состоянием страниц (метрики стадий, "
        "конфиг, ветки). Опирайся на эти ЧИСЛА, а не на догадки.\n\n"
        "Правила ответа:\n"
        "• отвечай по-русски, кратко и по делу, техническим языком;\n"
        "• чётко отделяй факт (из контекста) от интерпретации и предположения;\n"
        "• если данных в контексте не хватает — прямо скажи об этом и предложи, "
        "какую стадию выполнить или какие отклики заполнить;\n"
        "• стадии из списка `stages_done` СЧИТАЙ пройденными (для загруженного "
        "проекта модели восстановлены из файла); если стадия указана и в "
        "`stages_without_metrics`, значит её детальные метрики в этой сессии не "
        "пересчитаны — так и скажи и предложи перезапустить стадию ради цифр, "
        "но НЕ утверждай, что стадия «не выполнена»;\n"
        "• подсказывай следующий разумный шаг по конвейеру;\n"
        "• не выдумывай метрики, которых нет в контексте."

    )


def build_messages(context: Dict[str, Any], history: List[Dict[str, str]],
                   user_msg: str) -> List[Dict[str, str]]:
    ctx_json = json.dumps(context, ensure_ascii=False, indent=2)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt()},
        {"role": "system",
         "content": "Актуальный контекст приложения (JSON):\n```json\n"
                    + ctx_json + "\n```"},
    ]
    for m in history:
        role = m.get("role")
        if role in ("user", "assistant") and m.get("content"):
            messages.append({"role": role, "content": str(m["content"])})
    messages.append({"role": "user", "content": str(user_msg)})
    return messages


def call_llm(messages: List[Dict[str, str]], *, model: Optional[str] = None,
             key: Optional[str] = None, timeout: int = 120,
             temperature: float = 0.2) -> str:
    """Вызов OpenRouter chat/completions через stdlib urllib.

    Бросает RuntimeError с человекочитаемым текстом при отсутствии ключа или
    сетевой/HTTP-ошибке.
    """
    key = key or api_key()
    if not key:
        raise RuntimeError(
            "Не задан OPENROUTER_API_KEY. Укажите ключ в переменной окружения "
            "OPENROUTER_API_KEY перед запуском приложения.")
    payload = {
        "model": model or model_name(),
        "messages": messages,
        "temperature": float(temperature),
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL, data=data, method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AntonZhiharev/mixture-design-doe",
            "X-Title": "DOE Pipeline Assistant",
        })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Сетевая ошибка обращения к OpenRouter: {exc}") from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Неожиданный ответ OpenRouter: {body}") from exc


def assistant_reply(runner, history: List[Dict[str, str]], user_msg: str, *,
                    extra_context: Optional[Dict[str, Any]] = None,
                    model: Optional[str] = None) -> str:
    """Собрать контекст страниц и получить ответ модели на сообщение."""
    context = build_context(runner, extra=extra_context)
    messages = build_messages(context, history, user_msg)
    return call_llm(messages, model=model)
