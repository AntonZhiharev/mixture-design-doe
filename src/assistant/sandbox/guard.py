"""assistant/sandbox/guard.py — «сторож» подпроцесса песочницы (iter62).

Изоляция здесь делается ВНУТРИ дочернего интерпретатора: в каталог, который
кладётся первым в ``PYTHONPATH``, пишется ``sitecustomize.py`` — Python
импортирует его сам при старте, поэтому запрет действует одинаково и для
``run_python``, и для ``run_pytest``, и для любого кода, который они запустят.

Что запрещает сторож (ASSISTANT_SPEC, инварианты 3 и 4):

* **сеть** — ``socket``/``urllib`` подменяются на явный отказ. Интернет у
  ассистента есть, но ДРУГИМ каналом (OpenRouter ``:online``): подпроцесс,
  который сам ходит в сеть, невозможно отличить от подпроцесса, который
  выкачивает секреты;
* **запись вне рабочего каталога** — репозиторий доступен только на чтение,
  а ``tests/`` не правится вообще (там golden-числа iter45–57, они и есть
  контракт). Писать можно лишь в scratch-каталог прогона и во временный.

Отказ помечается маркером (``SANDBOX-DENIED[network]`` /
``SANDBOX-DENIED[write]``), чтобы бэкенд узнал ПРИЧИНУ в выводе и объяснил её
человеку словами, а не кодом возврата (A0.6).

Байт-код (``__pycache__``) и импорт пишет C-уровень интерпретатора мимо
``builtins.open``; чтобы это не считалось нарушением, подпроцесс запускается с
``PYTHONDONTWRITEBYTECODE=1``.
"""
from __future__ import annotations

#: Маркеры отказов — общий словарь сторожа и бэкенда.
DENY_MARK = "SANDBOX-DENIED"
NETWORK_MARK = f"{DENY_MARK}[network]"
WRITE_MARK = f"{DENY_MARK}[write]"

#: Имя файла, который Python импортирует сам при старте.
GUARD_FILENAME = "sitecustomize.py"

#: Переменная окружения с политикой (JSON) для сторожа.
POLICY_ENV = "DOE_SANDBOX_POLICY"


#: Исходник сторожа. Строка СЫРАЯ (r'''…'''): внутри есть пути с обратными
#: слэшами (устройство ``\\.\nul`` под Windows), и обычная строка съедала бы
#: их при генерации файла — сторож падал бы с SyntaxError, а подпроцесс
#: остался бы БЕЗ защиты.
GUARD_SOURCE = r'''"""АВТОГЕНЕРИРУЕТСЯ песочницей ассистента DOE — не редактировать.

Импортируется интерпретатором дочернего процесса (sitecustomize) и включает

запрет сети и запрет записи вне разрешённых каталогов.
"""
import builtins
import io
import json
import os

NET_MARK = "SANDBOX-DENIED[network]"
WRITE_MARK = "SANDBOX-DENIED[write]"

try:
    _P = json.loads(os.environ.get("DOE_SANDBOX_POLICY", "") or "{}")
except Exception:            # политика не разобрана — считаем максимально строгой
    _P = {}

_ALLOW_NET = bool(_P.get("allow_network", False))


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


_WRITE = [_norm(p) for p in (_P.get("write_roots") or [])]
_DENY = [_norm(p) for p in (_P.get("deny_write") or [])]


class SandboxDenied(PermissionError):
    """Действие запрещено политикой песочницы."""


def _under(path, root):
    return path == root or path.startswith(root + os.sep)


def _is_null_device(p):
    """«Пустое устройство» — не файловая система, запрещать его бессмысленно.

    В него пишут pytest (logging), subprocess и половина стандартной
    библиотеки; отказ здесь означал бы, что песочница не запускает тесты —
    то есть защита ломает ровно ту работу, ради которой существует.
    """
    if p.startswith("\\\\.\\") or p.startswith("//./"):
        return True
    return os.path.basename(p) in ("nul", "null") or p in (
        "/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty")


def _writable(path):
    try:
        p = _norm(os.fspath(path))
    except TypeError:        # файловый дескриптор (int) — уже открытый объект
        return True
    if _is_null_device(p):
        return True
    for d in _DENY:
        if _under(p, d):
            return False

    for r in _WRITE:
        if _under(p, r):
            return True
    return False


def _deny_write(path, action="Запись"):
    allowed = _WRITE[0] if _WRITE else "(нет)"
    raise SandboxDenied(
        "%s %s запрещена по пути: %s. Песочница видит репозиторий ТОЛЬКО НА "
        "ЧТЕНИЕ; писать можно лишь в рабочий каталог прогона (%s). Правка "
        "tests/ запрещена вообще: тесты — контракт (golden-числа iter45-57). "
        "Если нужно изменить спеку — предлагай патч (propose_patch), его "
        "применяет человек." % (WRITE_MARK, action, path, allowed))


def _deny_net(what="Сетевой вызов"):
    raise SandboxDenied(
        "%s %s запрещён: сети в песочнице нет. Интернет у ассистента есть "
        "ОТДЕЛЬНЫМ каналом (OpenRouter ':online'), а не из подпроцесса; "
        "попроси включить веб в панели, но помни: веб — знание уровня L2, "
        "локальные факты цеха (L1) его отменяют." % (NET_MARK, what))


# ---------------------------------------------------------------- запись
_real_open = builtins.open
_WRITE_FLAGS = ("w", "a", "x", "+")


def _is_write_mode(mode):
    return any(ch in str(mode) for ch in _WRITE_FLAGS)


def _guarded_open(file, mode="r", *args, **kwargs):
    if _is_write_mode(mode) and not _writable(file):
        _deny_write(file, "Открытие на запись")
    return _real_open(file, mode, *args, **kwargs)


builtins.open = _guarded_open
io.open = _guarded_open

_real_os_open = os.open
_WRITE_OFLAGS = (getattr(os, "O_WRONLY", 0) | getattr(os, "O_RDWR", 0)
                 | getattr(os, "O_APPEND", 0) | getattr(os, "O_CREAT", 0)
                 | getattr(os, "O_TRUNC", 0))


def _guarded_os_open(path, flags, *args, **kwargs):
    if (flags & _WRITE_OFLAGS) and not _writable(path):
        _deny_write(path, "os.open на запись")
    return _real_os_open(path, flags, *args, **kwargs)


os.open = _guarded_os_open


def _wrap_path_op(name, action, argno=0):
    fn = getattr(os, name, None)
    if fn is None:
        return

    def guarded(*args, **kwargs):
        if len(args) > argno and not _writable(args[argno]):
            _deny_write(args[argno], action)
        return fn(*args, **kwargs)

    setattr(os, name, guarded)


for _name, _action in (("remove", "Удаление"), ("unlink", "Удаление"),
                       ("rmdir", "Удаление каталога"),
                       ("mkdir", "Создание каталога"),
                       ("makedirs", "Создание каталогов"),
                       ("truncate", "Усечение файла"),
                       ("chmod", "Смена прав")):
    _wrap_path_op(_name, _action)

for _name in ("rename", "replace", "link", "symlink"):
    _wrap_path_op(_name, "Перемещение/связывание", argno=1)


# ------------------------------------------------------------------ сеть
if not _ALLOW_NET:
    import socket

    _real_socket = socket.socket

    class _BlockedSocket(_real_socket):
        """Сокет, который нельзя создать.

        Именно КЛАСС, а не функция: `ssl` при импорте объявляет
        ``class SSLSocket(socket)``, и подмена socket.socket функцией роняет
        половину стандартной библиотеки ещё до попытки выйти в сеть — то есть
        запрет ломал бы то, что должен всего лишь ограничивать.
        """

        def __init__(self, *args, **kwargs):
            _deny_net("socket")

    def _no_net(*args, **kwargs):
        _deny_net("сетевой вызов")

    socket.socket = _BlockedSocket
    socket.create_connection = _no_net
    socket.create_server = _no_net
    socket.getaddrinfo = _no_net
    socket.gethostbyname = _no_net

    try:
        import urllib.request as _urlreq

        def _no_url(*args, **kwargs):
            _deny_net("urlopen")

        _urlreq.urlopen = _no_url
    except Exception:
        pass
'''


def guard_source() -> str:
    """Исходник сторожа (чистая функция — удобна для теста «а что внутри»)."""
    return GUARD_SOURCE


