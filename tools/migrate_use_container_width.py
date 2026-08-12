#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Миграция устаревшего kwarg Streamlit ``use_container_width`` → ``width``.

Streamlit объявил `use_container_width` устаревшим (удаление после
2025-12-31) и просит писать `width='stretch'` / `width='content'`.

Скрипт заменяет ТОЛЬКО литеральное ``use_container_width=True`` на
``width="stretch"`` — это поведенчески эквивалентно для всех виджетов,
где значение по умолчанию `width` иное (`content`), и является no-op там,
где по умолчанию уже `stretch`.

Безопасность правки обеспечивается тремя проверками:

1. до правки — в файле нет вызовов, где одновременно заданы `width` и
   `use_container_width` (иначе возник бы дубль ключевого слова);
2. до правки — все значения `use_container_width` являются литералом
   ``True`` (значения-выражения скрипт не трогает и файл пропускает);
3. после правки — новый текст парсится в AST, и число вызовов с `width`
   совпадает с прежним числом вызовов с `use_container_width`.

Запуск (из корня репозитория)::

    .venv\\Scripts\\python.exe tools/migrate_use_container_width.py --check
    .venv\\Scripts\\python.exe tools/migrate_use_container_width.py --apply

Без флагов действует как ``--check`` (ничего не пишет).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Tuple

OLD_KWARG = "use_container_width"
OLD_TEXT = f"{OLD_KWARG}=True"
NEW_TEXT = 'width="stretch"'

ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("src",)


# ----------------------------------------------------------------------
def _read(path: Path) -> str:
    """Прочитать файл БЕЗ преобразования переводов строк.

    ``Path.read_text`` работает в режиме universal newlines и схлопывает
    CRLF в LF; при обратной записи это молча переводит весь файл на LF.
    Здесь переводы строк сохраняются как есть.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def _calls_with_kwarg(tree: ast.AST, name: str) -> List[ast.Call]:
    """Все вызовы, у которых есть ключевое слово ``name``."""
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and name in [k.arg for k in n.keywords]]


def _audit(path: Path) -> Tuple[int, List[str]]:
    """Сколько вхождений и какие препятствия у файла.

    Возвращает (число вызовов с устаревшим kwarg, список проблем).
    """
    problems: List[str] = []
    src = _read(path)
    tree = ast.parse(src)
    calls = _calls_with_kwarg(tree, OLD_KWARG)
    for call in calls:
        args = {k.arg: k.value for k in call.keywords}
        if "width" in args:
            problems.append(f"{path}:{call.lineno}: заданы и width, и "
                            f"{OLD_KWARG} — ручная правка")
        val = args[OLD_KWARG]
        if not (isinstance(val, ast.Constant) and val.value is True):
            problems.append(f"{path}:{call.lineno}: {OLD_KWARG} задан "
                            f"выражением, а не литералом True")
    # Текстовых вхождений должно быть ровно столько же, сколько вызовов:
    # иначе есть написание с пробелами, которое подстановка не покроет.
    if src.count(OLD_TEXT) != len(calls):
        problems.append(f"{path}: текстовых '{OLD_TEXT}' "
                        f"{src.count(OLD_TEXT)}, а вызовов {len(calls)} — "
                        f"есть нестандартное написание")
    return len(calls), problems


def _migrate(path: Path) -> int:
    """Заменить вхождения в файле. Возвращает число замен."""
    src = _read(path)
    expected = len(_calls_with_kwarg(ast.parse(src), OLD_KWARG))
    if not expected:
        return 0

    before_width = len(_calls_with_kwarg(ast.parse(src), "width"))
    out = src.replace(OLD_TEXT, NEW_TEXT)

    new_tree = ast.parse(out)             # синтаксис не сломан
    if len(_calls_with_kwarg(new_tree, OLD_KWARG)):
        raise RuntimeError(f"{path}: остались вызовы с {OLD_KWARG}")
    after_width = len(_calls_with_kwarg(new_tree, "width"))
    if after_width != before_width + expected:
        raise RuntimeError(f"{path}: ожидали +{expected} вызовов с width, "
                           f"получили +{after_width - before_width}")

    # newline='' — сохраняем исходные CRLF, чтобы не рождать шумный дифф.
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    return expected


# ----------------------------------------------------------------------
def main(argv: List[str]) -> int:
    apply = "--apply" in argv
    files = sorted(p for d in SCAN_DIRS
                   for p in (ROOT / d).rglob("*.py")
                   if OLD_KWARG in _read(p))
    if not files:
        print("Вхождений не найдено — миграция не нужна.")
        return 0

    total, blocked = 0, []
    for path in files:
        count, problems = _audit(path)
        total += count
        rel = path.relative_to(ROOT)
        if problems:
            blocked.append(rel)
            for p in problems:
                print("  ! " + p)
        print(f"{count:4d}  {rel}")
    print(f"ИТОГО вхождений: {total}, файлов: {len(files)}")

    if blocked:
        print(f"ОСТАНОВ: файлов с препятствиями {len(blocked)} — "
              f"замена не выполнена.")
        return 2
    if not apply:
        print("Режим проверки (--check). Для правки запустить с --apply.")
        return 0

    done = sum(_migrate(p) for p in files)
    print(f"Заменено вхождений: {done}")
    return 0 if done == total else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
