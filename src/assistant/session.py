"""assistant/session.py — сессия ИИ-ассистента, привязанная к проекту (iter58).

Сессия — ЕДИНИЦА ПАМЯТИ помощника-архитектора: переписка, приложенные файлы,
артефакты песочницы, предложенные (но не применённые) патчи спеки, аудит
вызовов инструментов и расход токенов. Она живёт рядом с проектом
(``project_campaigns/<проект>/assistant/session.json``, см. :mod:`.store`),
поэтому переезжает вместе с ним при сохранении/загрузке и исчезает при
удалении проекта — отдельной синхронизации нет.

Модуль JSON-native: без numpy, без pandas, без Streamlit. Всё, что нужно для
показа, строит :mod:`.views`; всё, что нужно для диска — :mod:`.store`.

Инварианты (ASSISTANT_SPEC):
  * **A0.6** — усечение контекста НЕ теряет историю: в модель уходит хвост
    диалога, на диске остаётся всё, факт усечения виден явной пометкой;
  * дедуп вложений по ``sha256`` — один файл = одна запись;
  * staged-патч НЕ применяется сам: статусы ``staged`` → ``applied`` |
    ``rejected``, переход из терминального статуса — явная ошибка.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

FORMAT_VERSION = "assistant-v1"

#: Роли сообщений. ``tool`` — результат вызова инструмента (iter60+).
ROLES = ("user", "assistant", "system", "tool")

#: Статусы предложенного патча спеки.
PATCH_STAGED = "staged"
PATCH_APPLIED = "applied"
PATCH_REJECTED = "rejected"
PATCH_STATUSES = (PATCH_STAGED, PATCH_APPLIED, PATCH_REJECTED)

#: Тип границы (правила диапазона системного промпта архитектора).
BOUND_TYPES = ("PHYSICAL", "CONVENTIONAL", "")

#: Уровень знания: L1 локальные факты > L2 литература > L3 проверяемое.
KNOWLEDGE_LEVELS = ("L1", "L2", "L3", "")

#: Грубая оценка «символов на токен» для бюджета контекста. Точный счёт
#: даёт только токенизатор модели, а он зависит от провайдера; для решения
#: «сколько сообщений влезет» этой точности достаточно, а зависимость
#: (tiktoken) не нужна.
CHARS_PER_TOKEN = 4


def _now() -> str:
    """Отметка времени UTC (секундная точность) — общая для всех записей."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def estimate_tokens(text: Any) -> int:
    """Грубая оценка числа токенов в тексте (см. :data:`CHARS_PER_TOKEN`)."""
    return max(1, len(str(text or "")) // CHARS_PER_TOKEN)


# ----------------------------------------------------------------------
# Записи сессии
# ----------------------------------------------------------------------
@dataclass
class Message:
    """Одно сообщение диалога.

    ``web`` — ответ получен с включённым интернет-каналом (OpenRouter
    ``:online``): без этой пометки нельзя отличить утверждение из веба от
    утверждения из контекста проекта. ``tool_calls``/``tool_call_id`` — сырые
    поля протокола инструментов (iter60), хранятся как есть.

    ``images`` (iter68) — ССЫЛКИ на приложенные изображения (``sha256``
    вложений), а НЕ base64. Причина: ``content`` остаётся строкой, поэтому
    оценка бюджета (:func:`estimate_tokens`) и ``session.json`` не распухают на
    мегабайт скриншота; сам data-URL собирается на момент отправки
    (:func:`assistant.files.attachment_data_url`).
    """
    role: str
    content: str
    ts: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _new_id("msg"))
    model: str = ""
    web: bool = False
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    images: List[str] = field(default_factory=list)

    def to_state(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"id": self.id, "role": self.role,
                               "content": self.content, "ts": self.ts}
        if self.model:
            out["model"] = self.model
        if self.web:
            out["web"] = True
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        if self.usage:
            out["usage"] = dict(self.usage)
        if self.images:
            out["images"] = list(self.images)
        return out

    @classmethod
    def from_state(cls, d: Dict[str, Any]) -> "Message":
        return cls(role=str(d.get("role", "user")),
                   content=str(d.get("content", "")),
                   ts=str(d.get("ts", "")) or _now(),
                   id=str(d.get("id", "")) or _new_id("msg"),
                   model=str(d.get("model", "")),
                   web=bool(d.get("web", False)),
                   tool_calls=list(d.get("tool_calls", []) or []),
                   tool_call_id=str(d.get("tool_call_id", "")),
                   name=str(d.get("name", "")),
                   usage=dict(d.get("usage", {}) or {}),
                   images=[str(x) for x in (d.get("images", []) or [])])

    def chat_message(self) -> Dict[str, Any]:
        """Вид сообщения для API модели (без внутренних полей сессии).

        ``images`` здесь НЕ разворачивается: собрать мультимодальный
        ``content`` может только тот, у кого есть доступ к файлам проекта
        (см. :func:`assistant.llm.user_content`). Сессия остаётся JSON-native.
        """
        out: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


@dataclass
class Attachment:
    """Приложенный файл: метаданные + извлечённый текст (усечён для контекста).

    Сам файл лежит в ``assistant/files/<sha256>__<имя>`` (см. :mod:`.files`,
    iter59); здесь — то, что видно ассистенту без чтения диска.
    """
    name: str
    sha256: str
    size: int = 0
    mime: str = ""
    stored_name: str = ""
    text: str = ""
    n_chars: int = 0
    truncated: bool = False
    note: str = ""
    ts: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _new_id("file"))

    def to_state(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "sha256": self.sha256,
                "size": int(self.size), "mime": self.mime,
                "stored_name": self.stored_name, "text": self.text,
                "n_chars": int(self.n_chars), "truncated": bool(self.truncated),
                "note": self.note, "ts": self.ts}

    @classmethod
    def from_state(cls, d: Dict[str, Any]) -> "Attachment":
        return cls(name=str(d.get("name", "")), sha256=str(d.get("sha256", "")),
                   size=int(d.get("size", 0) or 0), mime=str(d.get("mime", "")),
                   stored_name=str(d.get("stored_name", "")),
                   text=str(d.get("text", "")),
                   n_chars=int(d.get("n_chars", 0) or 0),
                   truncated=bool(d.get("truncated", False)),
                   note=str(d.get("note", "")),
                   ts=str(d.get("ts", "")) or _now(),
                   id=str(d.get("id", "")) or _new_id("file"))


@dataclass
class Artifact:
    """Выхлоп песочницы (png/txt/json), сохранённый в кампанию (iter62)."""
    name: str
    kind: str = "text"
    path: str = ""
    tool: str = ""
    caption: str = ""
    ts: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _new_id("art"))

    def to_state(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "path": self.path, "tool": self.tool,
                "caption": self.caption, "ts": self.ts}

    @classmethod
    def from_state(cls, d: Dict[str, Any]) -> "Artifact":
        return cls(name=str(d.get("name", "")), kind=str(d.get("kind", "text")),
                   path=str(d.get("path", "")), tool=str(d.get("tool", "")),
                   caption=str(d.get("caption", "")),
                   ts=str(d.get("ts", "")) or _now(),
                   id=str(d.get("id", "")) or _new_id("art"))


@dataclass
class StagedPatch:
    """Предложенный патч спеки — СТЕЙДЖ, а не применение (ASSISTANT_SPEC §2).

    Поля повторяют формат ответа архитектора (`## PATCH`): один узел = один
    пункт, тип границы PHYSICAL|CONVENTIONAL, уровень знания L1|L2|L3 с
    источником, уверенность, пометка «двигает ``spec_hash``». Таблица патчей в
    UI строится по этим полям и ничего не домысливает.
    """
    node: str
    field_name: str
    from_value: Any = None
    to_value: Any = None
    bound_type: str = ""
    level: str = ""
    source: str = ""
    rationale: str = ""
    confidence: str = ""
    affects_hash: bool = False
    status: str = PATCH_STAGED
    applied_ts: str = ""
    reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _new_id("patch"))

    def to_state(self) -> Dict[str, Any]:
        return {"id": self.id, "node": self.node, "field": self.field_name,
                "from": self.from_value, "to": self.to_value,
                "bound_type": self.bound_type, "level": self.level,
                "source": self.source, "rationale": self.rationale,
                "confidence": self.confidence,
                "affects_hash": bool(self.affects_hash),
                "status": self.status, "applied_ts": self.applied_ts,
                "reason": self.reason, "raw": dict(self.raw), "ts": self.ts}

    @classmethod
    def from_state(cls, d: Dict[str, Any]) -> "StagedPatch":
        return cls(node=str(d.get("node", "")),
                   field_name=str(d.get("field", d.get("field_name", ""))),
                   from_value=d.get("from", d.get("from_value")),
                   to_value=d.get("to", d.get("to_value")),
                   bound_type=str(d.get("bound_type", "")),
                   level=str(d.get("level", "")),
                   source=str(d.get("source", "")),
                   rationale=str(d.get("rationale", "")),
                   confidence=str(d.get("confidence", "")),
                   affects_hash=bool(d.get("affects_hash", False)),
                   status=str(d.get("status", PATCH_STAGED)),
                   applied_ts=str(d.get("applied_ts", "")),
                   reason=str(d.get("reason", "")),
                   raw=dict(d.get("raw", {}) or {}),
                   ts=str(d.get("ts", "")) or _now(),
                   id=str(d.get("id", "")) or _new_id("patch"))


@dataclass
class StagedSpec:
    """Предложенный ПАКЕТ phr-спеки целиком — стейдж, а не применение (iter71).

    Зачем отдельная запись, а не список :class:`StagedPatch`: патч правит ПОЛЕ
    существующего узла, а здесь предлагается ГЕОМЕТРИЯ — первичный ввод спеки
    (узлов ещё нет вовсе) и её эволюция: добавить узел, удалить узел, сменить
    роль. Такое изменение нельзя ни собрать из пофайловых правок, ни принять
    по частям: спека валидна только целиком (инварианты k=2/k≥3, ``members``,
    ``group_order``), поэтому и решение по ней одно — принять пакет или нет.

    ``nodes``/``group_order``/``spec_version`` — сам пакет в формате
    :meth:`PhrSpec.to_dicts`. ``summary`` — вычисленный ядром разбор
    (``q``, ``dim_z``, ``spec_hash``, состав добавленных/удалённых узлов,
    смена ролей): человек в UI видит, ЧТО он утверждает, не читая JSON глазами.
    """
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    group_order: List[str] = field(default_factory=list)
    spec_version: int = 2
    label: str = ""
    rationale: str = ""
    level: str = ""
    source: str = ""
    confidence: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    status: str = PATCH_STAGED
    applied_ts: str = ""
    reason: str = ""
    ts: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _new_id("spec"))

    def payload(self) -> Any:
        """Пакет в том виде, который принимает ``PhrSpec.from_dicts``.

        ``group_order`` пуст → плоский список узлов: спека без порядка групп
        сериализуется байт-в-байт как до iter48, и её отпечаток не должен
        зависеть от того, через какой канал она пришла.
        """
        if self.group_order:
            return {"spec_version": int(self.spec_version or 2),
                    "group_order": list(self.group_order),
                    "nodes": [dict(d) for d in self.nodes]}
        return [dict(d) for d in self.nodes]

    def to_state(self) -> Dict[str, Any]:
        return {"id": self.id, "nodes": [dict(d) for d in self.nodes],
                "group_order": list(self.group_order),
                "spec_version": int(self.spec_version or 2),
                "label": self.label, "rationale": self.rationale,
                "level": self.level, "source": self.source,
                "confidence": self.confidence, "summary": dict(self.summary),
                "status": self.status, "applied_ts": self.applied_ts,
                "reason": self.reason, "ts": self.ts}

    @classmethod
    def from_state(cls, d: Dict[str, Any]) -> "StagedSpec":
        d = dict(d or {})
        return cls(nodes=[dict(x) for x in (d.get("nodes") or [])],
                   group_order=[str(x) for x in (d.get("group_order") or [])],
                   spec_version=int(d.get("spec_version", 2) or 2),
                   label=str(d.get("label", "")),
                   rationale=str(d.get("rationale", "")),
                   level=str(d.get("level", "")),
                   source=str(d.get("source", "")),
                   confidence=str(d.get("confidence", "")),
                   summary=dict(d.get("summary", {}) or {}),
                   status=str(d.get("status", PATCH_STAGED)),
                   applied_ts=str(d.get("applied_ts", "")),
                   reason=str(d.get("reason", "")),
                   ts=str(d.get("ts", "")) or _now(),
                   id=str(d.get("id", "")) or _new_id("spec"))


@dataclass
class StagedProject:
    """Предложенный ПАКЕТ ПРОЕКТА: состав + отклики + процесс-оси (iter73).

    Зачем отдельно от :class:`StagedSpec`. Пакет спеки описывает ГЕОМЕТРИЮ и
    применим лишь к СУЩЕСТВУЮЩЕМУ проекту (``runner.set_phr_spec``). Но проект
    рождается из ТРЁХ блоков: состава (phr-спека), откликов и процесс-осей —
    откликов и осей в спеке нет по схеме. Из-за этого «Применить спеку» в пустой
    сессии отрабатывало «успешно» и не давало результата: писать было некуда.
    Здесь предлагается проект ЦЕЛИКОМ, и применение имеет смысл.

    ``package`` — самодокументируемый JSON (блоки ``spec``/``responses``/
    ``process``/``covariates``/``passport``), ``summary`` — разбор ядром
    (:func:`design.project_package.package_manifest`): человек видит по блокам,
    ЧТО именно приедет, не читая JSON глазами. Ввод идёт в несколько подходов,
    поэтому пакетов в сессии может быть много — статус у каждого свой.
    """
    package: Dict[str, Any] = field(default_factory=dict)
    label: str = ""
    rationale: str = ""
    level: str = ""
    source: str = ""
    confidence: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    status: str = PATCH_STAGED
    applied_ts: str = ""
    reason: str = ""
    ts: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _new_id("proj"))

    def payload(self) -> Dict[str, Any]:
        """Пакет в том виде, который принимает ``parse_project_package``."""
        return {k: v for k, v in dict(self.package).items()}

    def to_state(self) -> Dict[str, Any]:
        return {"id": self.id, "package": dict(self.package),
                "label": self.label, "rationale": self.rationale,
                "level": self.level, "source": self.source,
                "confidence": self.confidence, "summary": dict(self.summary),
                "status": self.status, "applied_ts": self.applied_ts,
                "reason": self.reason, "ts": self.ts}

    @classmethod
    def from_state(cls, d: Dict[str, Any]) -> "StagedProject":
        d = dict(d or {})
        return cls(package=dict(d.get("package", {}) or {}),
                   label=str(d.get("label", "")),
                   rationale=str(d.get("rationale", "")),
                   level=str(d.get("level", "")),
                   source=str(d.get("source", "")),
                   confidence=str(d.get("confidence", "")),
                   summary=dict(d.get("summary", {}) or {}),
                   status=str(d.get("status", PATCH_STAGED)),
                   applied_ts=str(d.get("applied_ts", "")),
                   reason=str(d.get("reason", "")),
                   ts=str(d.get("ts", "")) or _now(),
                   id=str(d.get("id", "")) or _new_id("proj"))


@dataclass
class StagedSetup:
    """Предложенная ТОЧЕЧНАЯ ПРАВКА ПОЛЕЙ ФОРМЫ сетапа (iter76).

    Закрывает замкнутый круг несобранного проекта: до сборки данные живут в
    ПОЛЯХ формы «🆕 Новый проект», и ассистент их не видел и не мог править —
    единственным его инструментом был пакет ПРОЕКТА целиком, который к тому же
    отклонялся, пока в полях уже что-то стоит. Здесь предлагается правка
    ОТДЕЛЬНЫХ полей (``fields`` — ``{ключ_виджета: значение}``); применяет
    человек кнопкой, результат — ``setup_prefill_pending`` (тот же механизм,
    что у загрузки проекта). Раннера правка НЕ касается: собранный проект
    правится пакетами спеки/патчами.
    """
    fields: Dict[str, Any] = field(default_factory=dict)
    label: str = ""
    rationale: str = ""
    level: str = ""
    source: str = ""
    confidence: str = ""
    status: str = PATCH_STAGED
    applied_ts: str = ""
    reason: str = ""
    ts: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _new_id("setup"))

    def to_state(self) -> Dict[str, Any]:
        return {"id": self.id, "fields": dict(self.fields),
                "label": self.label, "rationale": self.rationale,
                "level": self.level, "source": self.source,
                "confidence": self.confidence, "status": self.status,
                "applied_ts": self.applied_ts, "reason": self.reason,
                "ts": self.ts}

    @classmethod
    def from_state(cls, d: Dict[str, Any]) -> "StagedSetup":
        d = dict(d or {})
        return cls(fields=dict(d.get("fields", {}) or {}),
                   label=str(d.get("label", "")),
                   rationale=str(d.get("rationale", "")),
                   level=str(d.get("level", "")),
                   source=str(d.get("source", "")),
                   confidence=str(d.get("confidence", "")),
                   status=str(d.get("status", PATCH_STAGED)),
                   applied_ts=str(d.get("applied_ts", "")),
                   reason=str(d.get("reason", "")),
                   ts=str(d.get("ts", "")) or _now(),
                   id=str(d.get("id", "")) or _new_id("setup"))


@dataclass
class ToolCall:
    """Запись аудита вызова инструмента (дублируется в ``tool_calls.jsonl``)."""
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""
    duration_s: float = 0.0
    summary: str = ""
    ts: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _new_id("call"))

    def to_state(self) -> Dict[str, Any]:
        return {"id": self.id, "tool": self.tool, "args": dict(self.args),
                "ok": bool(self.ok), "error": self.error,
                "duration_s": float(self.duration_s),
                "summary": self.summary, "ts": self.ts}

    @classmethod
    def from_state(cls, d: Dict[str, Any]) -> "ToolCall":
        return cls(tool=str(d.get("tool", "")), args=dict(d.get("args", {}) or {}),
                   ok=bool(d.get("ok", True)), error=str(d.get("error", "")),
                   duration_s=float(d.get("duration_s", 0.0) or 0.0),
                   summary=str(d.get("summary", "")),
                   ts=str(d.get("ts", "")) or _now(),
                   id=str(d.get("id", "")) or _new_id("call"))


# ----------------------------------------------------------------------
# Сессия
# ----------------------------------------------------------------------
@dataclass
class AssistantSession:
    """Диалог ассистента с памятью, привязанный к проекту.

    ``ref`` (iter77) — ССЫЛКА на проект (``core.project_ref``), истинный ключ
    связи «проект ↔ переписка». До iter77 связь держалась на ``project``, то
    есть на ИМЕНИ: правка имени в поле формы переключала переписку на другой
    каталог, а переименовать проект было нельзя вовсе. Теперь имя — подпись,
    а адресация идёт по ``ref``.

    ``project`` — имя КАТАЛОГА проекта в ``project_campaigns`` (остаётся: по
    нему строятся пути, и старые сессии без ``ref`` читаются как раньше);
    ``model`` / ``web_enabled`` — политика подключения, живущая вместе с
    проектом (после перезапуска приложения не надо вспоминать, была ли
    включена сеть при этом разборе).
    """
    project: str = ""
    ref: str = ""
    model: str = ""
    web_enabled: bool = False
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    messages: List[Message] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)
    patches: List[StagedPatch] = field(default_factory=list)
    specs: List[StagedSpec] = field(default_factory=list)
    projects: List[StagedProject] = field(default_factory=list)
    setups: List[StagedSetup] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)

    # -- сообщения ------------------------------------------------------
    def add_message(self, role: str, content: str, **kw: Any) -> Message:
        """Добавить сообщение. Неизвестная роль — явная ошибка (A0.6)."""
        if role not in ROLES:
            raise ValueError(
                f"Неизвестная роль сообщения {role!r}: допустимы {ROLES}.")
        msg = Message(role=role, content=str(content), **kw)
        self.messages.append(msg)
        self.updated_at = _now()
        if msg.usage:
            self.add_usage(msg.usage)
        return msg

    def add_usage(self, usage: Dict[str, Any]) -> Dict[str, int]:
        """Накопить расход токенов (prompt/completion/total) по ответам."""
        for k, v in (usage or {}).items():
            try:
                self.usage[str(k)] = int(self.usage.get(str(k), 0)) + int(v)
            except (TypeError, ValueError):
                continue
        return dict(self.usage)

    def context_messages(self, *, max_tokens: int = 24000
                         ) -> List[Dict[str, Any]]:
        """Хвост диалога для модели в рамках бюджета + ЯВНАЯ пометка усечения.

        A0.6: молча урезанный диалог выглядит как амнезия ассистента, поэтому
        вместо отрезанной части возвращается одно системное сообщение «опущено
        N ранних сообщений (полный лог — в session.json проекта)». Полная
        история на диске не трогается.
        """
        if max_tokens <= 0:
            raise ValueError("Бюджет контекста max_tokens должен быть > 0.")
        kept: List[Message] = []
        budget = int(max_tokens)
        for msg in reversed(self.messages):
            cost = estimate_tokens(msg.content) + estimate_tokens(msg.tool_calls)
            if kept and cost > budget:
                break
            budget -= cost
            kept.append(msg)
        kept.reverse()
        out = [m.chat_message() for m in kept]
        n_omitted = len(self.messages) - len(kept)
        if n_omitted > 0:
            out.insert(0, {
                "role": "system",
                "content": (f"[сессия] Ранние сообщения опущены из контекста по "
                            f"бюджету: {n_omitted}. Полная переписка сохранена "
                            f"в проекте (assistant/session.json) — если нужен "
                            f"более ранний фрагмент, попроси пользователя "
                            f"напомнить, не выдумывай.")})
        return out

    # -- вложения -------------------------------------------------------
    def attachment_by_hash(self, sha256: str) -> Optional[Attachment]:
        for a in self.attachments:
            if a.sha256 == sha256:
                return a
        return None

    def add_attachment(self, att: Attachment) -> Attachment:
        """Добавить вложение с дедупом по ``sha256``.

        Повторное приложение того же файла возвращает СУЩЕСТВУЮЩУЮ запись:
        один файл — один экземпляр в контексте (иначе модель видит документ
        трижды и тратит на него бюджет).
        """
        found = self.attachment_by_hash(att.sha256)
        if found is not None:
            return found
        self.attachments.append(att)
        self.updated_at = _now()
        return att

    def remove_attachment(self, att_id: str) -> bool:
        n = len(self.attachments)
        self.attachments = [a for a in self.attachments if a.id != att_id]
        changed = len(self.attachments) != n
        if changed:
            self.updated_at = _now()
        return changed

    # -- артефакты ------------------------------------------------------
    def add_artifact(self, art: Artifact) -> Artifact:
        self.artifacts.append(art)
        self.updated_at = _now()
        return art

    # -- патчи ----------------------------------------------------------
    def stage_patch(self, patch: StagedPatch) -> StagedPatch:
        """Положить патч в стейдж (НЕ применять — это делает write-инструмент)."""
        if not patch.node:
            raise ValueError("Патч без узла (node) не принимается: один узел = "
                             "один пункт патча (формат ответа архитектора).")
        patch.status = PATCH_STAGED
        self.patches.append(patch)
        self.updated_at = _now()
        return patch

    def patch_by_id(self, patch_id: str) -> Optional[StagedPatch]:
        for p in self.patches:
            if p.id == patch_id:
                return p
        return None

    def set_patch_status(self, patch_id: str, status: str, *,
                         reason: str = "") -> StagedPatch:
        """Перевести патч в терминальный статус (``applied``/``rejected``).

        Повторный перевод уже применённого/отклонённого патча — явная ошибка:
        иначе «применено дважды» выглядело бы как норма.
        """
        if status not in PATCH_STATUSES:
            raise ValueError(f"Неизвестный статус патча {status!r}: "
                             f"допустимы {PATCH_STATUSES}.")
        p = self.patch_by_id(patch_id)
        if p is None:
            raise KeyError(f"Патч '{patch_id}' не найден в сессии.")
        if p.status != PATCH_STAGED:
            raise ValueError(
                f"Патч '{patch_id}' уже в статусе '{p.status}' — повторный "
                f"переход запрещён (предложите новый патч).")
        p.status = status
        p.reason = reason
        p.applied_ts = _now()
        self.updated_at = _now()
        return p

    def staged_patches(self) -> List[StagedPatch]:
        return [p for p in self.patches if p.status == PATCH_STAGED]

    # -- пакеты спеки (iter71) ------------------------------------------
    def stage_spec(self, spec: StagedSpec) -> StagedSpec:
        """Положить ПАКЕТ спеки в стейдж (применяет человек кнопкой).

        Пустой пакет не принимается: «спека без узлов» — не предложение, а
        потеря геометрии, и предлагать её молча нельзя (A0.6).
        """
        if not spec.nodes:
            raise ValueError(
                "Пакет спеки без узлов не принимается: предлагать пустую "
                "геометрию нельзя (это не правка, а потеря спеки).")
        spec.status = PATCH_STAGED
        self.specs.append(spec)
        self.updated_at = _now()
        return spec

    def spec_by_id(self, spec_id: str) -> Optional[StagedSpec]:
        for s in self.specs:
            if s.id == spec_id:
                return s
        return None

    def set_spec_status(self, spec_id: str, status: str, *,
                        reason: str = "") -> StagedSpec:
        """Перевести пакет спеки в терминальный статус (тот же протокол, что
        у патчей: повторный переход — явная ошибка)."""
        if status not in PATCH_STATUSES:
            raise ValueError(f"Неизвестный статус пакета спеки {status!r}: "
                             f"допустимы {PATCH_STATUSES}.")
        s = self.spec_by_id(spec_id)
        if s is None:
            raise KeyError(f"Пакет спеки '{spec_id}' не найден в сессии.")
        if s.status != PATCH_STAGED:
            raise ValueError(
                f"Пакет спеки '{spec_id}' уже в статусе '{s.status}' — "
                f"повторный переход запрещён (предложите новый пакет).")
        s.status = status
        s.reason = reason
        s.applied_ts = _now()
        self.updated_at = _now()
        return s

    def staged_specs(self) -> List[StagedSpec]:
        return [s for s in self.specs if s.status == PATCH_STAGED]

    # -- пакеты проекта (iter73) ----------------------------------------
    def stage_project(self, proj: StagedProject) -> StagedProject:
        """Положить ПАКЕТ ПРОЕКТА в стейдж (применяет человек кнопкой).

        Пустой пакет не принимается: «проект без блоков» — не предложение, а
        видимость работы. Полноту блоков проверяет ядро
        (:func:`design.project_package.parse_project_package`) ДО стейджа.
        """
        if not proj.package:
            raise ValueError(
                "Пакет проекта пуст: предлагать нечего. Проект рождается из "
                "блоков 'spec' (состав), 'responses' (отклики) и 'process' "
                "(оси процесса с границами).")
        proj.status = PATCH_STAGED
        self.projects.append(proj)
        self.updated_at = _now()
        return proj

    def project_by_id(self, project_id: str) -> Optional[StagedProject]:
        for p in self.projects:
            if p.id == project_id:
                return p
        return None

    def set_project_status(self, project_id: str, status: str, *,
                           reason: str = "") -> StagedProject:
        """Перевести пакет проекта в терминальный статус (протокол патчей)."""
        if status not in PATCH_STATUSES:
            raise ValueError(f"Неизвестный статус пакета проекта {status!r}: "
                             f"допустимы {PATCH_STATUSES}.")
        p = self.project_by_id(project_id)
        if p is None:
            raise KeyError(f"Пакет проекта '{project_id}' не найден в сессии.")
        if p.status != PATCH_STAGED:
            raise ValueError(
                f"Пакет проекта '{project_id}' уже в статусе '{p.status}' — "
                f"повторный переход запрещён (предложите новый пакет).")
        p.status = status
        p.reason = reason
        p.applied_ts = _now()
        self.updated_at = _now()
        return p

    def staged_projects(self) -> List[StagedProject]:
        return [p for p in self.projects if p.status == PATCH_STAGED]

    # -- правки полей формы сетапа (iter76) -------------------------------
    def stage_setup(self, setup: StagedSetup) -> StagedSetup:
        """Положить ПРАВКУ ПОЛЕЙ ФОРМЫ сетапа в стейдж (применяет человек).

        Пустая правка не принимается: «поправить ничего» — не предложение.
        """
        if not setup.fields:
            raise ValueError(
                "Правка полей сетапа пуста: нечего предлагать. Укажите "
                "{ключ_поля: значение} — ключи см. в снимке setup_snapshot.")
        setup.status = PATCH_STAGED
        self.setups.append(setup)
        self.updated_at = _now()
        return setup

    def setup_by_id(self, setup_id: str) -> Optional[StagedSetup]:
        for s in self.setups:
            if s.id == setup_id:
                return s
        return None

    def set_setup_status(self, setup_id: str, status: str, *,
                         reason: str = "") -> StagedSetup:
        """Перевести правку полей в терминальный статус (протокол патчей)."""
        if status not in PATCH_STATUSES:
            raise ValueError(f"Неизвестный статус правки сетапа {status!r}: "
                             f"допустимы {PATCH_STATUSES}.")
        s = self.setup_by_id(setup_id)
        if s is None:
            raise KeyError(f"Правка сетапа '{setup_id}' не найдена в сессии.")
        if s.status != PATCH_STAGED:
            raise ValueError(
                f"Правка сетапа '{setup_id}' уже в статусе '{s.status}' — "
                f"повторный переход запрещён (предложите новую правку).")
        s.status = status
        s.reason = reason
        s.applied_ts = _now()
        self.updated_at = _now()
        return s

    def staged_setups(self) -> List[StagedSetup]:
        return [s for s in self.setups if s.status == PATCH_STAGED]

    # -- аудит вызовов --------------------------------------------------
    def add_tool_call(self, call: ToolCall) -> ToolCall:
        self.tool_calls.append(call)
        self.updated_at = _now()
        return call

    # -- прочее ---------------------------------------------------------
    def clear_messages(self) -> None:
        """Очистить ПЕРЕПИСКУ, сохранив файлы, патчи, артефакты и аудит.

        «Очистить чат» не должно стирать приложенные паспорта сырья и историю
        решений — это разные вещи (A0.6).
        """
        self.messages = []
        self.updated_at = _now()

    def is_empty(self) -> bool:
        return not (self.messages or self.attachments or self.patches
                    or self.specs or self.projects or self.setups
                    or self.artifacts or self.tool_calls)

    def to_state(self) -> Dict[str, Any]:
        return {
            "format": FORMAT_VERSION,
            "project": self.project,
            "model": self.model,
            "web_enabled": bool(self.web_enabled),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage": {str(k): int(v) for k, v in (self.usage or {}).items()},
            "messages": [m.to_state() for m in self.messages],
            "attachments": [a.to_state() for a in self.attachments],
            "artifacts": [a.to_state() for a in self.artifacts],
            "patches": [p.to_state() for p in self.patches],
            "specs": [s.to_state() for s in self.specs],
            "projects": [p.to_state() for p in self.projects],
            "setups": [s.to_state() for s in self.setups],
            # iter77: ССЫЛКА на проект пишется, только когда она есть, —
            # иначе сессии, записанные до перехода на ссылку, изменили бы
            # состояние на диске от одного лишь чтения (round-trip должен
            # быть побайтно стабилен, см. ASSISTANT_SPEC §3.1).
            **({"ref": self.ref} if self.ref else {}),
            "tool_calls": [c.to_state() for c in self.tool_calls],
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "AssistantSession":
        state = dict(state or {})
        fmt = str(state.get("format", FORMAT_VERSION))
        if fmt != FORMAT_VERSION:
            raise ValueError(
                f"Неизвестный формат сессии ассистента: {fmt!r} "
                f"(ожидался {FORMAT_VERSION!r}).")
        s = cls(project=str(state.get("project", "")),
                # iter77: сессия до перехода на ссылку — ref пуст, и это не
                # ошибка: ссылку досоздаст store.load_session по каталогу.
                ref=str(state.get("ref", "")),
                model=str(state.get("model", "")),
                web_enabled=bool(state.get("web_enabled", False)),
                created_at=str(state.get("created_at", "")) or _now(),
                updated_at=str(state.get("updated_at", "")) or _now())
        s.usage = {str(k): int(v) for k, v in
                   (state.get("usage", {}) or {}).items()}
        s.messages = [Message.from_state(d) for d in
                      (state.get("messages", []) or [])]
        s.attachments = [Attachment.from_state(d) for d in
                         (state.get("attachments", []) or [])]
        s.artifacts = [Artifact.from_state(d) for d in
                       (state.get("artifacts", []) or [])]
        s.patches = [StagedPatch.from_state(d) for d in
                     (state.get("patches", []) or [])]
        # iter71: сессии, записанные до пакетов спеки, ключа 'specs' не имеют —
        # это НЕ повод отказать в загрузке (старые проекты открываются как были).
        s.specs = [StagedSpec.from_state(d) for d in
                   (state.get("specs", []) or [])]
        # iter73: то же правило совместимости для пакетов ПРОЕКТА — сессии,
        # записанные до них, ключа 'projects' не имеют и должны открываться.
        s.projects = [StagedProject.from_state(d) for d in
                      (state.get("projects", []) or [])]
        # iter76: правки полей формы сетапа — тот же принцип совместимости.
        s.setups = [StagedSetup.from_state(d) for d in
                    (state.get("setups", []) or [])]
        s.tool_calls = [ToolCall.from_state(d) for d in
                        (state.get("tool_calls", []) or [])]
        return s


def new_session(project: str, *, model: str = "", web_enabled: bool = False,
                ref: str = "") -> AssistantSession:
    """Пустая сессия проекта (используется, когда на диске её ещё нет).

    ``ref`` (iter77) — ссылка на проект: её проставляет :mod:`.store`, который
    знает каталог. Аргумент необязателен, чтобы вызовы из ядра и тестов, где
    каталога проекта нет вовсе, работали как прежде.
    """
    return AssistantSession(project=str(project or ""), model=str(model or ""),
                            web_enabled=bool(web_enabled),
                            ref=str(ref or ""))


def messages_from_pairs(pairs: Iterable[Dict[str, str]]) -> List[Message]:
    """Старая история чата (``{"role", "content"}``) → сообщения сессии.

    Нужна для переноса переписки из ``st.session_state['ai_history']`` старой
    вкладки ассистента: диалог не должен пропасть при переходе на сессии.
    """
    out: List[Message] = []
    for p in (pairs or []):
        role = str((p or {}).get("role", ""))
        if role in ("user", "assistant"):
            out.append(Message(role=role, content=str(p.get("content", ""))))
    return out
