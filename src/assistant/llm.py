"""assistant/llm.py — OpenRouter: цикл вызова инструментов + интернет (iter60).

Ассистент-архитектор отличается от чата тем, что ОТВЕЧАЕТ ЧИСЛАМИ ИЗ ЯДРА, а не
по памяти: модель просит инструмент (`get_spec`, `simulate_bounds`, …), мы
исполняем его локально и возвращаем результат обратно в диалог. Здесь — сам
цикл, транспорт (stdlib ``urllib``, без новых зависимостей) и правила
безопасности разговора.

Ключевые решения:

* **`:online` — тумблер, а не модель.** Интернет включается суффиксом OpenRouter
  к имени модели (:func:`online_model`), поэтому отдельный поисковый ключ не
  нужен, а факт использования веба виден в сессии (``Message.web``) — иначе
  утверждение из сети неотличимо от утверждения из контекста проекта.
* **Ошибка инструмента возвращается МОДЕЛИ**, а не рушит ответ: `validate_spec`,
  отказавший на неверной роли узла, — это ПОЛЕЗНЫЙ сигнал, из которого модель
  делает вывод (иначе пользователь увидит стектрейс вместо объяснения).
* **Лимиты цикла**: число итераций и общее время. Без них модель может ходить
  по инструментам бесконечно; при исчерпании — честная пометка
  ``stopped_reason``, а не молчаливая обрезка.
* **Прогресс наружу** (``on_event``): UI обязан показывать, что идёт долгий
  вызов (`run_pytest`), а не выглядеть зависшим.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import DEFAULT_MODEL, OPENROUTER_URL, api_key, model_name

#: Суффикс веб-плагина OpenRouter.
ONLINE_SUFFIX = ":online"

#: Максимум итераций «модель → инструменты → модель» на один вопрос.
MAX_ITERATIONS = 8

#: Общий бюджет времени на цикл (сек). Длинные инструменты (`run_pytest`)
#: учитываются здесь же.
TIME_BUDGET_S = 180.0

#: Сколько символов результата инструмента отдаём модели. Хвост усечём с
#: ЯВНОЙ пометкой: молча обрезанная таблица выглядит как «данных нет».
MAX_TOOL_RESULT_CHARS = 20_000

DEFAULT_TIMEOUT = 180
DEFAULT_TEMPERATURE = 0.2

#: Эндпоинт распознавания речи OpenRouter (iter68). Отдельный от чата: голос
#: превращается в ТЕКСТ до входа в диалог, поэтому дальше работает весь
#: существующий конвейер (маршрутизация, инструменты, аудит), а в сессии
#: остаётся проверяемая фраза, а не непрослушиваемый блоб.
TRANSCRIBE_URL = "https://openrouter.ai/api/v1/audio/transcriptions"

#: Модель распознавания по умолчанию (переопределяется ``DOE_ASSISTANT_STT``).
DEFAULT_STT_MODEL = "openai/whisper-1"

#: Предел на аудио: у провайдера тайм-аут 60 с, длинную запись он не успеет.
MAX_AUDIO_BYTES = 20 * 1024 * 1024

_HEADERS_EXTRA = {
    "HTTP-Referer": "https://github.com/AntonZhiharev/mixture-design-doe",
    "X-Title": "DOE Campaign Architect",
}


class LLMError(RuntimeError):
    """Ошибка обращения к модели — с человекочитаемым текстом."""


# ----------------------------------------------------------------------
# Модель и веб
# ----------------------------------------------------------------------
def online_model(model: Optional[str] = None, *, web: bool = False) -> str:
    """Имя модели с учётом тумблера интернета (идемпотентно по суффиксу).

    Повторное включение не даёт ``model:online:online``; выключение снимает
    суффикс, если он был задан вручную в поле модели.
    """
    name = (model or model_name() or DEFAULT_MODEL).strip()
    base = name[: -len(ONLINE_SUFFIX)] if name.endswith(ONLINE_SUFFIX) else name
    return base + ONLINE_SUFFIX if web else base


def is_online(model: str) -> bool:
    return str(model or "").endswith(ONLINE_SUFFIX)


# ----------------------------------------------------------------------
# Транспорт
# ----------------------------------------------------------------------
#: Обрывки текста ответа провайдера, по которым видно, что модель не умеет
#: принимать картинку/звук (формулировки у провайдеров разные).
_MODALITY_MARKS = ("image", "vision", "multimodal", "input_audio", "audio",
                   "modality")


def _looks_like_modality_error(detail: str) -> bool:
    """Похоже ли HTTP 400 на «модель не поддерживает такой вход»."""
    s = str(detail or "").lower()
    return any(m in s for m in _MODALITY_MARKS)


def _http_transport(payload: Dict[str, Any], *, key: str, timeout: int,
                    url: str = "") -> Dict[str, Any]:
    """POST в OpenRouter на stdlib urllib (новых зависимостей нет).

    ``url`` позволяет тому же транспорту обслуживать и распознавание речи
    (:data:`TRANSCRIBE_URL`): заголовки, разбор ошибок и подсказки там ровно
    те же, дублировать их значило бы разъехаться в текстах отказов.
    """
    data = json.dumps(payload).encode("utf-8")
    headers = {"Authorization": f"Bearer {key}",
               "Content-Type": "application/json", **_HEADERS_EXTRA}
    req = urllib.request.Request(url or OPENROUTER_URL, data=data,
                                 method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        hint = {
            401: " Проверьте OPENROUTER_API_KEY.",
            402: " На счёте OpenRouter закончились средства.",
            404: " Такой модели нет — проверьте имя (и суффикс ':online').",
            429: " Слишком часто: подождите и повторите.",
        }.get(exc.code, "")
        if exc.code == 400 and _looks_like_modality_error(detail):
            # Иначе человек видит сырой JSON провайдера и не понимает, что
            # дело в выбранной модели, а не в его картинке.
            hint = (" Похоже, выбранная модель не принимает изображения "
                    "(vision) или аудио: смените модель в панели «Модель и "
                    "ключ» на мультимодальную.")
        raise LLMError(f"OpenRouter HTTP {exc.code}.{hint} {detail[:400]}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Сетевая ошибка обращения к OpenRouter: {exc}") from exc
    except ValueError as exc:
        raise LLMError(f"Ответ OpenRouter не разобран как JSON: {exc}") from exc


def chat_once(messages: Sequence[Dict[str, Any]], *,
              tools: Optional[Sequence[Dict[str, Any]]] = None,
              model: Optional[str] = None, key: Optional[str] = None,
              web: bool = False, temperature: float = DEFAULT_TEMPERATURE,
              timeout: int = DEFAULT_TIMEOUT,
              transport: Optional[Callable[..., Dict[str, Any]]] = None
              ) -> Dict[str, Any]:
    """Один запрос к модели. ``transport`` подменяется в тестах (без сети)."""
    key = key or api_key()
    if not key and transport is None:
        raise LLMError(
            "Не задан OPENROUTER_API_KEY: укажите ключ в панели ассистента "
            "(он сохранится в локальный .env) или в переменной окружения.")
    payload: Dict[str, Any] = {
        "model": online_model(model, web=web),
        "messages": list(messages),
        "temperature": float(temperature),
    }
    if tools:
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    fn = transport or _http_transport
    body = fn(payload, key=key or "", timeout=timeout)
    if not isinstance(body, dict) or not body.get("choices"):
        raise LLMError(f"Неожиданный ответ OpenRouter: {str(body)[:400]}")
    return body


# ----------------------------------------------------------------------
# Мультимодальный вход: картинки (iter68)
# ----------------------------------------------------------------------
def user_content(text: str, image_urls: Optional[Sequence[str]] = None) -> Any:
    """Содержимое user-сообщения: строка или мультимодальный список частей.

    Без картинок возвращается ПРОСТАЯ СТРОКА, а не список из одной части:
    так текстовый путь остаётся ровно тем, что был до iter68, и ни один
    провайдер не получает неожиданную форму запроса.

    Порядок частей — сначала текст, потом изображения: так рекомендует
    OpenRouter (иначе часть провайдеров хуже разбирает промпт).
    """
    text = str(text or "")
    urls = [str(u) for u in (image_urls or []) if str(u or "").strip()]
    if not urls:
        return text
    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for u in urls:
        parts.append({"type": "image_url", "image_url": {"url": u}})
    return parts


def has_images(message: Any) -> bool:
    """Есть ли в сообщении часть-изображение (для пометок в UI и тестов)."""
    content = (message or {}).get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return False
    return any(isinstance(p, dict) and p.get("type") == "image_url"
               for p in content)


# ----------------------------------------------------------------------
# Голос → текст (iter68)
# ----------------------------------------------------------------------
def stt_model() -> str:
    """Модель распознавания речи (``DOE_ASSISTANT_STT`` или дефолт)."""
    return os.environ.get("DOE_ASSISTANT_STT", "").strip() or DEFAULT_STT_MODEL


def transcribe(audio: bytes, *, fmt: str = "wav", model: Optional[str] = None,
               key: Optional[str] = None, language: str = "",
               timeout: int = 120,
               transport: Optional[Callable[..., Dict[str, Any]]] = None
               ) -> Dict[str, Any]:
    """Распознать речь через OpenRouter и вернуть ``{text, model, usage}``.

    Почему распознавание ОТДЕЛЬНО, а не аудио прямо в чат: наш рабочий ассистент
    держится на tool-calling и длинном контексте (Claude), а аудио на вход
    принимают другие модели. Превращая голос в текст заранее, мы не меняем
    модель разговора и оставляем в сессии проверяемую фразу — её видно,
    можно исправить и переспросить.

    Пустой результат — ЯВНАЯ ошибка: молча отправить в диалог пустой вопрос
    значит показать человеку ответ «не понял», не сказав, что запись не
    распознана (A0.6).
    """
    if not isinstance(audio, (bytes, bytearray)) or not audio:
        raise LLMError("Пустая аудиозапись: распознавать нечего.")
    audio = bytes(audio)
    if len(audio) > MAX_AUDIO_BYTES:
        raise LLMError(
            f"Запись слишком длинная: {len(audio) / 1048576:.1f} МБ при лимите "
            f"{MAX_AUDIO_BYTES / 1048576:.0f} МБ. У провайдера тайм-аут 60 с — "
            f"скажите короче или разбейте на части.")
    key = key or api_key()
    if not key and transport is None:
        raise LLMError(
            "Не задан OPENROUTER_API_KEY: распознавание речи идёт через тот же "
            "ключ, что и чат — укажите его в панели ассистента.")

    payload: Dict[str, Any] = {
        "model": str(model or stt_model()),
        "input_audio": {
            # Провайдер ждёт «сырой» base64 БЕЗ префикса data:… (в отличие от
            # картинок в чате) — на этом легко ошибиться, поэтому кодируем тут.
            "data": base64.b64encode(audio).decode("ascii"),
            "format": str(fmt or "wav").lower().lstrip("."),
        },
    }
    if language.strip():
        payload["language"] = language.strip()

    fn = transport or _http_transport
    body = fn(payload, key=key or "", timeout=timeout, url=TRANSCRIBE_URL)
    if not isinstance(body, dict):
        raise LLMError(f"Неожиданный ответ распознавания: {str(body)[:400]}")
    text = str(body.get("text", "") or "").strip()
    if not text:
        raise LLMError(
            "Речь не распознана (пустой текст). Проверьте, что микрофон "
            "записал звук, и попробуйте сказать ещё раз ближе к микрофону.")
    return {"text": text, "model": payload["model"],
            "usage": dict(body.get("usage", {}) or {})}


# ----------------------------------------------------------------------
# Результат цикла
# ----------------------------------------------------------------------
@dataclass
class LLMResult:
    """Итог одного «хода» ассистента.

    ``new_messages`` — всё, что нужно дописать в сессию (ответы модели и
    результаты инструментов) В ПОРЯДКЕ появления: без них следующий вопрос
    потеряет контекст вызовов.
    """
    text: str = ""
    new_messages: List[Dict[str, Any]] = field(default_factory=list)
    calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    web: bool = False
    iterations: int = 0
    stopped_reason: str = "final"

    @property
    def n_tool_calls(self) -> int:
        return len(self.calls)


def _accumulate_usage(total: Dict[str, int], usage: Any) -> Dict[str, int]:
    for k, v in (usage or {}).items():
        try:
            total[str(k)] = int(total.get(str(k), 0)) + int(v)
        except (TypeError, ValueError):
            continue
    return total


def _tool_payload(result: Any) -> str:
    """Результат инструмента → строка для модели (с честным усечением)."""
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(result)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = (text[:MAX_TOOL_RESULT_CHARS] +
                f"\n…[результат усечён: {len(text)} символов; "
                f"запроси конкретный фрагмент отдельным вызовом]")
    return text


def _parse_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw or "{}")
    except ValueError as exc:
        raise ValueError(f"аргументы не разобраны как JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("аргументы инструмента должны быть JSON-объектом")
    return parsed


# ----------------------------------------------------------------------
# Цикл «модель → инструменты → модель»
# ----------------------------------------------------------------------
def run_tool_loop(messages: Sequence[Dict[str, Any]], *,
                  dispatch: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
                  tools: Optional[Sequence[Dict[str, Any]]] = None,
                  model: Optional[str] = None, key: Optional[str] = None,
                  web: bool = False, temperature: float = DEFAULT_TEMPERATURE,
                  timeout: int = DEFAULT_TIMEOUT,
                  max_iterations: int = MAX_ITERATIONS,
                  time_budget_s: float = TIME_BUDGET_S,
                  transport: Optional[Callable[..., Dict[str, Any]]] = None,
                  on_event: Optional[Callable[[Dict[str, Any]], None]] = None
                  ) -> LLMResult:
    """Провести ход ассистента: модель ↔ инструменты, пока не будет ответа.

    ``dispatch(name, args)`` исполняет инструмент и возвращает JSON-совместимый
    результат; исключение внутри — НЕ ошибка хода: текст ошибки уходит модели
    как результат инструмента (A0.6 — отказ объясняет себя, и модель может его
    учесть). ``on_event`` получает события прогресса для UI:
    ``llm_request`` / ``tool_start`` / ``tool_end`` / ``done``.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations должен быть ≥ 1.")
    convo: List[Dict[str, Any]] = list(messages)
    res = LLMResult(model=online_model(model, web=web), web=bool(web))
    started = time.monotonic()

    def emit(kind: str, **kw: Any) -> None:
        if on_event is not None:
            try:
                on_event({"kind": kind, "elapsed_s": time.monotonic() - started,
                          **kw})
            except Exception:  # noqa: BLE001 — показ не должен рушить ход
                pass

    for it in range(1, int(max_iterations) + 1):
        res.iterations = it
        emit("llm_request", iteration=it, model=res.model)
        body = chat_once(convo, tools=tools, model=model, key=key, web=web,
                         temperature=temperature, timeout=timeout,
                         transport=transport)
        _accumulate_usage(res.usage, body.get("usage"))

        msg = (body.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        tool_calls = list(msg.get("tool_calls") or [])

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        convo.append(assistant_msg)
        res.new_messages.append(assistant_msg)

        if not tool_calls:
            res.text = content
            res.stopped_reason = "final"
            emit("done", iteration=it, reason="final")
            return res

        if dispatch is None:
            # Модель просит инструмент, а исполнителя нет — честно говорим об
            # этом, а не делаем вид, что ответ получен.
            res.text = content
            res.stopped_reason = "no_dispatch"
            emit("done", iteration=it, reason="no_dispatch")
            return res

        for call in tool_calls:
            fn = (call or {}).get("function") or {}
            name = str(fn.get("name", ""))
            t0 = time.monotonic()
            emit("tool_start", iteration=it, tool=name, args=fn.get("arguments"))
            try:
                args = _parse_args(fn.get("arguments"))
                out = dispatch(name, args)
                ok, payload = True, _tool_payload(out)
                error = ""
            except Exception as exc:  # noqa: BLE001 — отказ уходит МОДЕЛИ
                ok, error = False, f"{type(exc).__name__}: {exc}"
                payload = f"ОШИБКА ИНСТРУМЕНТА {name}: {error}"
                args = {"_raw": fn.get("arguments")}
            dt = time.monotonic() - t0

            tool_msg = {"role": "tool", "tool_call_id": str(call.get("id", "")),
                        "name": name, "content": payload}
            convo.append(tool_msg)
            res.new_messages.append(tool_msg)
            res.calls.append({"tool": name, "args": args, "ok": ok,
                              "error": error, "duration_s": round(dt, 3),
                              "summary": payload[:200]})
            emit("tool_end", iteration=it, tool=name, ok=ok, duration_s=dt,
                 error=error)

        if time.monotonic() - started > float(time_budget_s):
            res.stopped_reason = "time_budget"
            res.text = (content or "") + (
                "\n\n⏱ Бюджет времени на этот ход исчерпан "
                f"({time_budget_s:.0f} с): часть инструментов выполнена, "
                "ответ не завершён. Повторите вопрос — контекст вызовов "
                "сохранён в сессии.")
            emit("done", iteration=it, reason="time_budget")
            return res

    res.stopped_reason = "max_iterations"
    res.text = (
        f"⚠️ Достигнут предел в {max_iterations} обращений к инструментам за "
        f"один ход. Выполнено вызовов: {len(res.calls)}. Сузьте вопрос "
        f"(например, спросите про один узел) — все результаты уже в сессии.")
    emit("done", iteration=res.iterations, reason="max_iterations")
    return res


def progress_caption(event: Dict[str, Any]) -> str:
    """Событие цикла → строка для показа пользователю (чистая, без UI).

    Нужна, чтобы долгий вызов (`run_pytest`, `preflight`) не выглядел
    зависанием: в доке эта строка идёт рядом с прогресс-баром.
    """
    kind = str((event or {}).get("kind", ""))
    tool = event.get("tool", "")
    if kind == "llm_request":
        return f"🧠 запрос к модели (шаг {event.get('iteration', 1)})…"
    if kind == "tool_start":
        return f"🔧 выполняется `{tool}`…"
    if kind == "tool_end":
        mark = "✅" if event.get("ok", True) else "⛔"
        return (f"{mark} `{tool}` — {float(event.get('duration_s', 0.0)):.1f} с"
                + (f": {event.get('error')}" if not event.get("ok", True) else ""))
    if kind == "done":
        reason = {"final": "готово", "max_iterations": "предел вызовов",
                  "time_budget": "исчерпан бюджет времени",
                  "no_dispatch": "инструменты недоступны"
                  }.get(str(event.get("reason", "")), str(event.get("reason", "")))
        return f"🏁 {reason} ({float(event.get('elapsed_s', 0.0)):.1f} с)"
    return kind
