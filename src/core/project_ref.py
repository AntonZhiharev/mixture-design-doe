"""core/project_ref.py — ИДЕНТИЧНОСТЬ проекта: ссылка (ref) отдельно от имени.

Архитектурный багфикс (iter77). До этого шага проект и переписка ассистента
были связаны ЧЕРЕЗ ИМЯ: каталог назывался как имя из поля «Имя проекта», а
``assistant/`` лежал внутри этого каталога. Следствия:

  * правка одной буквы в поле имени = ДРУГАЯ переписка, другие вложения,
    другие журналы (``tool_calls.jsonl``, ``decision_log.jsonl``,
    ``local_facts.jsonl``) — iter76 умел лишь ПРЕДУПРЕДИТЬ об этом
    (``assistant_dock.K_SWITCH_MSG``), но не устранить;
  * переименования проекта как операции не существовало вовсе;
  * не собранный проект (дефолтный ``my_project``) не имел идентичности:
    подвязать к нему переписку и вложения ДО старта было нельзя — каталог
    возникал лишь при первой записи и опознавался только по имени.

Раскладка (вариант C): **каталог остаётся человеко-читаемым**, идентичность
лежит ВНУТРИ него, истинный ключ — ``ref``:

    project_campaigns/<каталог>/
        project.json        # {"ref": "prj_…", "label": "…", …}  ← ЭТОТ файл
        campaign.json       # состояние движка (campaign_state)
        setup_draft.json    # черновик формы несобранного проекта (iter76)
        assistant/          # переписка, вложения, журналы (assistant.store)

Инварианты:
  * ``ref`` НЕизменяем и выдаётся при ПОЯВЛЕНИИ проекта (в том числе
    несобранного) — к нему можно вязать переписку до старта;
  * ``label`` (человеческое имя) меняется свободно и НИ НА ЧТО не влияет:
    переименование не переключает переписку и не двигает каталог;
  * ``dirname`` (каталог) стабилен: выбирается один раз при создании и далее
    не трогается — иначе пути во вложениях и артефактах поехали бы;
  * ОДИН источник истины — файлы на диске (реестра-дубликата нет), поэтому
    ручное копирование каталога проекта не рассинхронизирует состояние;
  * ЧТЕНИЕ не создаёт каталогов и не пишет на диск (тот же инвариант, что у
    :mod:`src.assistant.store`): ``read_identity``/``list_identities`` —
    чистое чтение, выдача ссылки — только явный :func:`ensure_identity`;
  * СТАРЫЙ каталог без ``project.json`` — не ошибка: он виден как ``legacy``
    (ссылки ещё нет), ссылка досоздаётся явно (A0.6 — молча чужое состояние
    не мутируем).

Модуль намеренно на голом stdlib (без numpy/pandas/Streamlit): его импортируют
и :mod:`src.assistant.store` (персистентность переписки), и
:mod:`src.apps.campaign_state` (движок), и MCP-сервер ``doe-campaign``.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

FORMAT_VERSION = "project-ref-v1"

#: Файл идентичности внутри каталога проекта.
IDENTITY_FILE = "project.json"

#: Префикс ссылки. Читаемый префикс важен: ``prj_ab12…`` в логе и в подписи UI
#: сразу опознаётся как ссылка на проект, а не как хэш спеки.
REF_PREFIX = "prj_"

#: Длина случайной части ссылки (hex). 12 hex = 48 бит — как у идентификаторов
#: сессии (``assistant.session._new_id``); коллизия на масштабе каталога
#: проектов практически исключена, а строка остаётся короткой.
REF_HEX = 12

_REF_RE = re.compile(r"^" + re.escape(REF_PREFIX) + r"[0-9a-f]{%d}$" % REF_HEX)

#: Признаки того, что каталог — ПРОЕКТ, а не случайная папка рядом.
#: ``campaign.json`` — собранный движок, ``setup_draft.json`` — черновик формы
#: (iter76), ``assistant/session.json`` — одна лишь переписка (проект, к
#: которому ещё ничего не подвязано, кроме диалога).
CAMPAIGN_FILE = "campaign.json"
SETUP_DRAFT_FILE = "setup_draft.json"
SESSION_REL = ("assistant", "session.json")

#: Каталог и имя дефолтного проекта «пока не стартовали».
DEFAULT_DIRNAME = "my_project"
DEFAULT_LABEL = "my_project"


def _now() -> str:
    """Отметка времени UTC (секундная точность) — как в assistant.session."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_ref() -> str:
    """Новая ссылка на проект (неизменяемый идентификатор)."""
    return f"{REF_PREFIX}{uuid.uuid4().hex[:REF_HEX]}"


def is_ref(value: Any) -> bool:
    """Похоже ли значение на ссылку проекта (без обращения к диску)."""
    return bool(_REF_RE.match(str(value or "")))


def validate_ref(ref: str) -> str:
    """Проверить формат ссылки или поднять :class:`ValueError`.

    Формат проверяется строго: ссылка — ключ связи проекта и переписки, и
    «почти похожая» строка (обрезанная, с иным префиксом) должна отвергаться
    сразу, а не приводить к молчаливому созданию второго проекта.
    """
    ref = str(ref or "").strip()
    if not is_ref(ref):
        raise ValueError(
            f"Недопустимая ссылка проекта: {ref!r}. Ожидается "
            f"'{REF_PREFIX}' + {REF_HEX} hex-символов "
            f"(например, '{REF_PREFIX}ab12cd34ef56').")
    return ref


def validate_dirname(name: str) -> str:
    """Имя КАТАЛОГА проекта: правила ``campaign_state._validate_name``.

    Анти-traversal: пустое имя, ``.``/``..`` и разделители путей запрещены.
    """
    name = (name or "").strip()
    if not name or name in (".", "..") or any(s in name for s in ("/", "\\")):
        raise ValueError(f"Недопустимое имя каталога проекта: {name!r}")
    return name


#: Символы, недопустимые в имени каталога на Windows (плюс разделители путей).
_BAD_DIR_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: Имена устройств Windows: каталог с таким именем создать нельзя.
_RESERVED_DIRS = ({"con", "prn", "aux", "nul"}
                  | {f"com{i}" for i in range(1, 10)}
                  | {f"lpt{i}" for i in range(1, 10)})


def slugify_label(label: str, *, fallback: str = "project") -> str:
    """Имя каталога из человеческого имени (для НОВОГО проекта).

    Каталог человеко-читаем (вариант C), поэтому имя берётся из первого
    ``label``: кириллица сохраняется как есть (файловая система её держит),
    убираются лишь запрещённые символы, точки по краям и повторные пробелы.
    Пустой результат заменяется ``fallback`` — безымянных каталогов не бывает.

    Внимание: это ОДНОРАЗОВОЕ преобразование при создании. Переименование
    проекта (:func:`rename_label`) каталог НЕ трогает.
    """
    text = _BAD_DIR_CHARS.sub(" ", str(label or ""))
    text = re.sub(r"\s+", " ", text).strip().strip(".").strip()
    if not text or text in (".", "..") or text.lower() in _RESERVED_DIRS:
        return fallback
    return text[:120]


# ----------------------------------------------------------------------
# Идентичность
# ----------------------------------------------------------------------
@dataclass
class ProjectIdentity:
    """Кто такой этот проект: ссылка, человеческое имя, каталог.

    ``ref`` — ключ связи (проект ↔ переписка ↔ вложения ↔ журналы);
    ``label`` — то, что человек читает и правит; ``dirname`` — каталог на
    диске (стабилен). ``label_history`` хранит прежние имена: переименование
    проекта — событие, о котором потом спрашивают «а как он раньше назывался».

    ``legacy=True`` означает каталог БЕЗ ``project.json`` (проект,
    существовавший до iter77): ссылки у него ещё нет, и выдаётся она только
    явным :func:`ensure_identity` — молча чужое состояние не мутируем (A0.6).
    """
    ref: str
    label: str
    dirname: str
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)
    label_history: List[str] = field(default_factory=list)
    legacy: bool = False

    def to_state(self) -> Dict[str, Any]:
        """JSON-состояние для ``project.json``.

        ``legacy`` вычисляется по наличию файла, поэтому в файле ему места
        нет: записанная идентичность по определению не legacy. ``dirname``
        тоже не пишем — это имя каталога, в котором файл лежит; хранить его
        внутри значило бы завести второй источник истины, расходящийся при
        переносе каталога.
        """
        out: Dict[str, Any] = {
            "format": FORMAT_VERSION,
            "ref": self.ref,
            "label": self.label,
            "created": self.created,
            "updated": self.updated,
        }
        if self.label_history:
            out["label_history"] = list(self.label_history)
        return out

    @classmethod
    def from_state(cls, state: Dict[str, Any], *, dirname: str
                   ) -> "ProjectIdentity":
        ref = validate_ref(str((state or {}).get("ref", "")))
        label = str((state or {}).get("label", "") or dirname)
        hist = [str(v) for v in ((state or {}).get("label_history") or [])]
        return cls(ref=ref, label=label, dirname=validate_dirname(dirname),
                   created=str((state or {}).get("created", "")) or _now(),
                   updated=str((state or {}).get("updated", "")) or _now(),
                   label_history=hist)

    @property
    def has_ref(self) -> bool:
        return is_ref(self.ref)

    def short_ref(self, n: int = 8) -> str:
        """Короткая форма ссылки для подписей UI (``prj_ab12cd34``)."""
        return self.ref[:len(REF_PREFIX) + max(0, int(n))]


# ----------------------------------------------------------------------
# Пути и признаки проекта
# ----------------------------------------------------------------------
def project_dir(root: str | Path, dirname: str) -> Path:
    return Path(root) / validate_dirname(dirname)


def identity_path(root: str | Path, dirname: str) -> Path:
    return project_dir(root, dirname) / IDENTITY_FILE


def has_project_content(root: str | Path, dirname: str) -> bool:
    """Похож ли каталог на проект (ссылка, движок, черновик или переписка).

    Нужно, чтобы скан каталога проектов не выдавал за проекты посторонние
    папки, но и не прятал проект, у которого пока есть ОДНА переписка.
    """
    base = project_dir(root, dirname)
    return (base / IDENTITY_FILE).exists() \
        or (base / CAMPAIGN_FILE).exists() \
        or (base / SETUP_DRAFT_FILE).exists() \
        or (base.joinpath(*SESSION_REL)).exists()


# ----------------------------------------------------------------------
# Чтение / запись идентичности
# ----------------------------------------------------------------------
def read_identity(root: str | Path, dirname: str) -> Optional[ProjectIdentity]:
    """Идентичность каталога или ``None``, если ссылки нет (legacy/не проект).

    ЧТЕНИЕ: каталогов не создаёт, на диск не пишет. Битый ``project.json`` —
    тоже ``None`` (а не исключение): проект должен открываться, а ссылку
    досоздаст :func:`ensure_identity`.
    """
    path = identity_path(root, dirname)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return ProjectIdentity.from_state(state, dirname=dirname)
    except (ValueError, OSError):
        return None


def write_identity(root: str | Path, identity: ProjectIdentity) -> str:
    """Записать ``project.json`` атомарно (tmp + replace); вернуть путь.

    Атомарность по той же причине, что у сессии ассистента: прерванная
    запись не должна оставить вместо ссылки полусериализованный JSON — иначе
    проект потеряет связь с перепиской.
    """
    base = project_dir(root, identity.dirname)
    base.mkdir(parents=True, exist_ok=True)
    path = base / IDENTITY_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(identity.to_state(), ensure_ascii=False,
                              indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return str(path)


def ensure_identity(root: str | Path, dirname: str, *, label: str = "",
                    ref: str = "") -> ProjectIdentity:
    """Гарантировать ссылку у каталога: прочитать или ВЫДАТЬ и записать.

    Это единственный путь появления ссылки. Вызывается там, где человек
    начинает работу с проектом (открытие приложения, сохранение, первое
    обращение к переписке), поэтому запись здесь — ожидаемое следствие
    действия, а не сюрприз при чтении.

    Идемпотентность: существующая ссылка НЕ переписывается (даже если передан
    другой ``ref``) — иначе проект «сменил бы личность» и потерял переписку.
    ``label`` существующего проекта тоже не затирается: имя меняет только
    явный :func:`rename_label`.
    """
    dirname = validate_dirname(dirname)
    found = read_identity(root, dirname)
    if found is not None:
        return found
    ident = ProjectIdentity(
        ref=validate_ref(ref) if ref else new_ref(),
        label=str(label or "").strip() or dirname,
        dirname=dirname)
    write_identity(root, ident)
    return ident


def create_project(root: str | Path, label: str, *, ref: str = "",
                   dirname: str = "") -> ProjectIdentity:
    """Создать НОВЫЙ проект: каталог + ссылка (каталог — из ``label``).

    Если каталог с таким именем занят ДРУГИМ проектом, к имени добавляется
    числовой суффикс (``имя (2)``): каталог человеко-читаем, но идентичность
    определяется ссылкой, поэтому одноимённые проекты допустимы.
    """
    label = str(label or "").strip() or DEFAULT_LABEL
    base = validate_dirname(dirname or slugify_label(label))
    target, n = base, 2
    while (Path(root) / target).exists():
        found = read_identity(root, target)
        if found is not None and ref and found.ref == ref:
            return found                    # тот же проект — не плодим копию
        if found is None and not has_project_content(root, target):
            break                           # пустой каталог можно занять
        target = f"{base} ({n})"
        n += 1
    return ensure_identity(root, target, label=label, ref=ref)


def rename_label(root: str | Path, ref: str, label: str) -> ProjectIdentity:
    """Переименовать проект: меняется ТОЛЬКО ``label``.

    Каталог, ссылка, переписка, вложения и журналы не двигаются — именно это
    и есть смысл перехода на ссылку. Прежнее имя уходит в ``label_history``:
    молча забывать, как проект назывался, нельзя.
    """
    ident = require_identity(root, ref)
    new_label = str(label or "").strip()
    if not new_label:
        raise ValueError("Имя проекта не может быть пустым: это то, по чему "
                         "человек узнаёт проект в списке.")
    if new_label == ident.label:
        return ident
    ident.label_history = list(ident.label_history) + [ident.label]
    ident.label = new_label
    ident.updated = _now()
    write_identity(root, ident)
    return ident


# ----------------------------------------------------------------------
# Скан каталога проектов и резолв
# ----------------------------------------------------------------------
def list_identities(root: str | Path, *, include_legacy: bool = True
                    ) -> List[ProjectIdentity]:
    """Проекты каталога (чистое чтение, сортировка по имени каталога).

    Каталог без ``project.json`` попадает в список как ``legacy`` (ссылки
    нет, ``label`` = имя каталога): спрятать существующий проект из-за
    отсутствия нового файла было бы потерей данных на глазах пользователя.
    """
    base = Path(root)
    if not base.exists():
        return []
    out: List[ProjectIdentity] = []
    for p in sorted(base.iterdir(), key=lambda q: q.name):
        if not p.is_dir():
            continue
        try:
            if not has_project_content(base, p.name):
                continue
        except ValueError:
            continue                        # имя, которое мы не адресуем
        found = read_identity(base, p.name)
        if found is not None:
            out.append(found)
        elif include_legacy:
            out.append(ProjectIdentity(ref="", label=p.name, dirname=p.name,
                                       legacy=True))
    return out


def find_by_ref(root: str | Path, ref: str) -> Optional[ProjectIdentity]:
    """Проект по ссылке или ``None``. Резолв — сканом (файлы = истина)."""
    ref = str(ref or "").strip()
    if not is_ref(ref):
        return None
    for ident in list_identities(root, include_legacy=False):
        if ident.ref == ref:
            return ident
    return None


def require_identity(root: str | Path, ref: str) -> ProjectIdentity:
    """Проект по ссылке или явная ошибка со списком доступных ссылок."""
    ident = find_by_ref(root, ref)
    if ident is None:
        known = [f"{i.short_ref()} «{i.label}»"
                 for i in list_identities(root, include_legacy=False)]
        raise ValueError(
            f"Проекта по ссылке {str(ref)!r} нет в каталоге {root}. "
            f"Доступны: {', '.join(known) if known else '—'}.")
    return ident


def resolve(root: str | Path, token: str) -> Optional[ProjectIdentity]:
    """Найти проект по ссылке, имени каталога или ``label`` (в этом порядке).

    Нужен для внешних входов (MCP-сервер, CLI), где человек называет проект
    как привык. Неоднозначный ``label`` (два проекта с одинаковым именем) НЕ
    разрешается: угадывать, про какой из них речь, нельзя — это правило уже
    действует в ``mcp.campaign_tools.resolve_project``.
    """
    token = str(token or "").strip()
    if not token:
        return None
    if is_ref(token):
        return find_by_ref(root, token)
    idents = list_identities(root)
    for ident in idents:                    # точное имя каталога
        if ident.dirname == token:
            return ident
    by_label = [i for i in idents if i.label == token]
    if len(by_label) == 1:
        return by_label[0]
    if len(by_label) > 1:
        raise ValueError(
            f"Имя «{token}» носят {len(by_label)} проекта "
            f"({', '.join(i.dirname for i in by_label)}). Имя — не ключ: "
            f"назовите проект ссылкой ("
            + ", ".join(i.short_ref() for i in by_label) + ").")
    return None


def migrate_root(root: str | Path) -> List[ProjectIdentity]:
    """Выдать ссылки ВСЕМ проектам каталога, у которых их ещё нет.

    Разовая миграция для проектов, созданных до iter77. Безопасна: ничего не
    переносит и не переименовывает — только дописывает ``project.json`` с
    новой ссылкой и ``label`` = имя каталога. Повторный вызов ничего не
    меняет (ссылки уже на месте), поэтому её можно звать при каждом старте
    приложения. Возвращает список ТОЛЬКО тех проектов, кому ссылку выдали.
    """
    out: List[ProjectIdentity] = []
    for ident in list_identities(root):
        if ident.legacy:
            out.append(ensure_identity(root, ident.dirname,
                                       label=ident.label))
    return out


def ensure_default_project(root: str | Path, *,
                           dirname: str = DEFAULT_DIRNAME,
                           label: str = DEFAULT_LABEL) -> ProjectIdentity:
    """Ссылка ДЕФОЛТНОГО (ещё не стартовавшего) проекта.

    Требование пользователя: не запущенный проект с базовым именем должен
    появляться СРАЗУ с назначенной ссылкой — чтобы к нему можно было
    подвязать переписку, вложения и черновик формы ещё до сборки движка.
    Если такой проект уже существует (в том числе не собранный), открывается
    он же с прежней ссылкой: жизнь дефолтного проекта прекращает только
    админское удаление.
    """
    return ensure_identity(root, dirname, label=label)
