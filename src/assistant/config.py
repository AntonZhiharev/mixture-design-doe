"""assistant/config.py — подключение к OpenRouter: ключ, модель, локальный .env.

Единственное место, где живут настройки доступа к модели. Раньше это было
внутри `src/apps/assistant.py` (UI-слой); вынесено сюда, чтобы движок
ассистента (:mod:`.llm`, инструменты, MCP-сервер) не зависел от Streamlit-кода,
а ключ не читался двумя разными способами.

Приоритет: переменные окружения ВЫШЕ файла ``.env`` (внешнее окружение —
осознанный выбор оператора). Файл ``.env`` лежит в корне репозитория и внесён
в ``.gitignore`` — ключ остаётся на машине пользователя.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

#: Модель по умолчанию. Ассистенту-архитектору нужны длинный контекст (спека
#: на 19 узлов + паспорта) и надёжный tool-calling, поэтому дефолт держим
#: свежим; переопределяется полем в UI или ``DOE_ASSISTANT_MODEL``.
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"

#: Ключи, которые пишем/читаем в локальном ``.env``.
ENV_KEYS = ("OPENROUTER_API_KEY", "OPENROUTER_KEY", "DOE_ASSISTANT_MODEL",
            "DOE_TRACE_ROOT", "DOE_SANDBOX_BACKEND",
            "DOE_ASSISTANT_TIME_BUDGET_S", "DOE_ASSISTANT_MAX_ITERATIONS")

#: Бюджет времени на ОДИН ход помощника (сек) и предел обращений к
#: инструментам за ход. Живут здесь, а не в :mod:`.llm`, потому что это
#: настройка ОПЕРАТОРА: на кампании из 23 узлов модель генерирует аргументы
#: пакета минутами (замер 14.08.2026 — ход 234 с при 4 вызовах инструментов
#: общей длительностью 0,2 с), и поднять предел человек должен без правки кода.
DEFAULT_TIME_BUDGET_S = 180.0
DEFAULT_MAX_ITERATIONS = 8

#: Ниже этого бюджет не опускаем: один запрос к модели с пакетом проекта
#: заведомо дольше, и «бюджет 5 с» означал бы ход, который не может
#: завершиться никогда.
MIN_TIME_BUDGET_S = 30.0

#: Сколько секунд должно остаться, чтобы ИМЕЛО СМЫСЛ начинать новый запрос к
#: модели. Без этого порога ход тратил деньги на обращение, которое обрывалось
#: проверкой бюджета сразу после возврата.
REQUEST_HEADROOM_S = 20.0


def _env_float(name: str, default: float, *, minimum: float) -> float:
    """Число из окружения с полом; мусор в переменной — не повод падать.

    Настройка, заданная с опечаткой, не должна ронять ассистента: берём
    дефолт. Молча принимать значение ниже пола тоже нельзя — см.
    :data:`MIN_TIME_BUDGET_S`.
    """
    raw = str(os.environ.get(name, "") or "").strip().replace(",", ".")
    if not raw:
        return float(default)
    try:
        val = float(raw)
    except ValueError:
        return float(default)
    return val if val >= minimum else float(minimum)


def time_budget_s() -> float:
    """Бюджет времени хода: ``DOE_ASSISTANT_TIME_BUDGET_S`` или дефолт."""
    return _env_float("DOE_ASSISTANT_TIME_BUDGET_S", DEFAULT_TIME_BUDGET_S,
                      minimum=MIN_TIME_BUDGET_S)


def max_iterations() -> int:
    """Предел обращений к инструментам: ``DOE_ASSISTANT_MAX_ITERATIONS``."""
    return int(_env_float("DOE_ASSISTANT_MAX_ITERATIONS",
                          float(DEFAULT_MAX_ITERATIONS), minimum=1.0))


def save_limits(*, budget_s: Optional[float] = None,
                iterations: Optional[int] = None,
                path: Optional[str] = None) -> str:
    """Сохранить лимиты хода в локальный ``.env`` и в текущий процесс.

    Отдельно от :func:`save_api_key`, чтобы менять бюджет можно было не вводя
    ключ заново (он в поле скрыт звёздочками, и «сохранить» затирало бы его
    пустым значением).
    """
    if budget_s is None and iterations is None:
        raise ValueError("Нечего сохранять: не задан ни бюджет, ни лимит шагов.")
    path = path or env_file_path()
    existing: Dict[str, str] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = parse_env_text(fh.read())
        except OSError:
            existing = {}
    if budget_s is not None:
        val = max(float(budget_s), MIN_TIME_BUDGET_S)
        existing["DOE_ASSISTANT_TIME_BUDGET_S"] = f"{val:g}"
        os.environ["DOE_ASSISTANT_TIME_BUDGET_S"] = f"{val:g}"
    if iterations is not None:
        val_i = max(int(iterations), 1)
        existing["DOE_ASSISTANT_MAX_ITERATIONS"] = str(val_i)
        os.environ["DOE_ASSISTANT_MAX_ITERATIONS"] = str(val_i)

    lines = ["# Локальные секреты DOE — НЕ коммитить (файл в .gitignore).",
             f"# Обновлено: {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    for k, v in existing.items():
        lines.append(f"{k}={v}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def env_file_path() -> str:
    """Путь к локальному ``.env`` в корне репозитория."""
    return os.path.join(repo_root(), ".env")


def parse_env_text(text: str) -> Dict[str, str]:
    """Разобрать содержимое .env в словарь (``KEY=VALUE``, ``#`` — комментарий)."""
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


def load_env_file(path: Optional[str] = None, *, override: bool = False
                  ) -> Dict[str, str]:
    """Загрузить переменные из ``.env`` в ``os.environ``.

    По умолчанию НЕ перетирает уже заданные переменные окружения. Отсутствие
    файла — не ошибка (пустой словарь).
    """
    path = path or env_file_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            parsed = parse_env_text(fh.read())
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
    """Сохранить ключ (и опц. модель) в локальный ``.env`` + текущий процесс."""
    key = (key or "").strip()
    if not key:
        raise ValueError("Пустой ключ — нечего сохранять.")
    path = path or env_file_path()

    existing: Dict[str, str] = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                existing = parse_env_text(fh.read())
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
            return bool(parse_env_text(fh.read()).get("OPENROUTER_API_KEY"))
    except OSError:
        return False


def api_key() -> Optional[str]:
    """Ключ OpenRouter из окружения (``OPENROUTER_API_KEY``/``OPENROUTER_KEY``)."""
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    return key.strip() if key else None


def model_name() -> str:
    return os.environ.get("DOE_ASSISTANT_MODEL", DEFAULT_MODEL)


def llm_available() -> bool:
    return bool(api_key())
