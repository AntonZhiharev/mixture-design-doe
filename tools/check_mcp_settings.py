"""tools/check_mcp_settings.py — какие MCP-серверы РЕАЛЬНО видит Cline.

Зачем это нужно (грабли, пойманные 11.08.2026). Cline **4.1.x** перенёс
хранилище настроек MCP из `globalStorage` расширения в домашний каталог и
сменил схему записи: `command`/`args`/`env` теперь вложены в блок
``transport``. Миграция происходит ОДИН раз при обновлении расширения, после
чего правки старого файла ни на что не влияют — Cline их больше не читает.

Симптом: в чате доступен старый набор инструментов (например, легаси
``list_runs`` вместо кампанийных ``get_spec``/``preflight``), хотя в старом
файле настроек «всё правильно». Скрипт печатает ОБА хранилища сразу, поэтому
расхождение видно мгновенно.

Ожидаемое состояние (`docs/MCP_SETUP.md` §0): ``doe-campaign`` — включён,
``doe-introspect`` — выключен (легаси).

Запуск:
    .venv\\Scripts\\python.exe tools\\check_mcp_settings.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

# ----------------------------------------------------------------------
# Пути к хранилищам настроек
# ----------------------------------------------------------------------
_REL_NEW = Path(".cline") / "data" / "settings" / "cline_mcp_settings.json"
_REL_OLD_TAIL = (Path("User") / "globalStorage" / "saoudrizwan.claude-dev"
                 / "settings" / "cline_mcp_settings.json")


def _new_store() -> Path:
    """Актуальное хранилище Cline 4.1.x — единое для всех ОС (``~/.cline``)."""
    return Path.home() / _REL_NEW


def _old_stores() -> list[Path]:
    """Легаси-хранилища Cline 3.x (`globalStorage` расширения) по всем ОС."""
    home = Path.home()
    candidates = [
        home / "AppData" / "Roaming" / "Code" / _REL_OLD_TAIL,          # Windows
        home / "Library" / "Application Support" / "Code" / _REL_OLD_TAIL,  # macOS
        home / ".config" / "Code" / _REL_OLD_TAIL,                     # Linux
    ]
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.insert(0, Path(appdata) / "Code" / _REL_OLD_TAIL)
    # уникальные, с сохранением порядка
    seen: set[Path] = set()
    out: list[Path] = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


# ----------------------------------------------------------------------
# Разбор и печать
# ----------------------------------------------------------------------
def _entrypoint(cfg: dict) -> str:
    """Имя запускаемого файла — по нему и опознаётся сервер.

    Поддержаны обе схемы: вложенный ``transport`` (Cline 4.1.x) и плоские
    поля (Cline 3.x).
    """
    transport = cfg.get("transport") or cfg
    args = transport.get("args") or []
    if args:
        return Path(str(args[0])).name
    command = transport.get("command")
    return Path(str(command)).name if command else "<не задан>"


def report(path: Path, label: str) -> dict[str, bool]:
    """Напечатать состояние одного файла настроек.

    Возвращает ``{имя сервера: включён}``; пустой словарь — файла нет или он
    нечитаем.
    """
    print(f"\n=== {label} ===")
    print(f"  {path}")
    if not path.exists():
        print("  НЕ НАЙДЕН")
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ОШИБКА ЧТЕНИЯ: {exc}")
        return {}

    servers = data.get("mcpServers") or {}
    print(f"  JSON корректен, серверов: {len(servers)}")
    enabled: dict[str, bool] = {}
    for name, cfg in servers.items():
        on = not bool(cfg.get("disabled"))
        enabled[name] = on
        state = "вкл " if on else "ВЫКЛ"
        print(f"    {name:16} {state}  {_entrypoint(cfg)}")
    return enabled


def verdict(enabled: dict[str, bool]) -> int:
    """Сверить фактическое состояние с каноном `docs/MCP_SETUP.md` §0."""
    print("\n--- проверка канона (MCP_SETUP §0) ---")
    problems: list[str] = []
    if not enabled.get("doe-campaign", False):
        problems.append("`doe-campaign` не включён — числа кампании считает не ядро, "
                        "а пересказ исходников (ASSISTANT_SPEC iter66)")
    if enabled.get("doe-introspect", False):
        problems.append("`doe-introspect` включён — это легаси; два похожих сервера "
                        "рядом дают путаницу, его держим выключенным")
    if problems:
        for text in problems:
            print(f"  [!] {text}")
        print("\n  Что делать: править НОВОЕ хранилище (см. §5/§7), затем реконнект "
              "MCP в панели Cline или перезапуск VS Code.")
        return 1
    print("  OK: doe-campaign включён, doe-introspect выключен.")
    return 0


def main() -> int:
    new = _new_store()
    enabled = report(new, "НОВОЕ хранилище (Cline 4.1.x, авторитетное)")

    found_old = False
    for path in _old_stores():
        if path.exists():
            found_old = True
            report(path, "СТАРОЕ хранилище (globalStorage, Cline 3.x — легаси)")
    if not found_old:
        print("\n=== СТАРОЕ хранилище (globalStorage, Cline 3.x) ===\n  не найдено")

    if not new.exists():
        print("\n[!] Нового хранилища нет: либо установлен Cline 3.x, либо расширение "
              "ещё не запускалось. Тогда авторитетен файл в globalStorage.")
    return verdict(enabled)


if __name__ == "__main__":
    raise SystemExit(main())
