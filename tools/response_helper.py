"""tools/response_helper.py — ручной хелпер откликов для battle-прогона §17.

Назначение (по запросу пользователя): при РУЧНОМ прохождении единого UI кампании
(§17.4 сетап → seed → ветки → рабочий стол) на каждом шаге, где форма просит
внести измеренные отклики Y, нужно откуда-то взять «правду лаборатории». Этот
хелпер её и считает: вводишь координаты точки (доли компонентов + режимы),
получаешь значения откликов синтетической истины battle-теста — и вручную
переносишь их в форму (столбцы «свойство (lab)»).

Истина — та же, что в ``tests/unit/test_iteration13_battle.py`` (единый источник
``src.verification.battle_truth``): никакого расхождения между тем, что считает
хелпер, и тем, к чему сходится пайплайн.

Два «мира» (см. ``--world``):
  * ``econ``  (по умолчанию) — STEP 6: 4 компонента {A,B,C,D} × процесс {T,P},
    5 откликов strength/gloss/dry_time/whiteStrength/rho + цена изделия
    (price_изд = price_состав·ρ). Аналитические оптимумы веток интерьерные.
  * ``3comp`` — фазовый мир: 3 компонента {A,B,C} × {T,P}, отклики
    strength/gloss/dry_time/price/rho.

Координаты:
  * доли компонентов смеси — как в форме (Σ=1; хелпер предупредит, если сумма
    заметно отличается от 1, но посчитает как введено);
  * процесс-оси T,P — истина живёт в КОДЕ [0,1]. UI кампании сохраняет процесс
    в таблицах дизайна АБСОЛЮТНЫМИ (реальными) значениями (сетап §17.4, дефолты
    T=150…200, P=1…5). Хелпер нормирует их в код сам:
    ``code = (real − lo) / (hi − lo)``. Границы берутся из ``--proc-bounds``;
    без него значение в [0,1] трактуется как уже кодовое, а значение ВНЕ [0,1]
    автоматически нормируется ДЕФОЛТНЫМИ границами сетапа (T=150…200, P=1…5)
    с предупреждением. Если границы в вашем сетапе другие — передайте их
    явно через ``--proc-bounds``.

Использование (одна точка, ОДНОЙ строкой — cmd/PowerShell не понимают '\\'):
    python tools/response_helper.py --world econ A=0.25 B=0.25 C=0.25 D=0.25 T=0.5 P=0.5
    python tools/response_helper.py A=0.3 B=0.3 C=0.2 D=0.2 T=0.5 P=0.5   # world=econ

Использование (процесс в РЕАЛЬНЫХ единицах, как в таблицах UI):
    python tools/response_helper.py --proc-bounds T=150:200,P=1:5 A=0.25 B=0.25 C=0.25 D=0.25 T=175 P=3

Пропущенные координаты: mixture-компоненты обязательны; процесс-оси по умолчанию
середина интервала (код 0.5). Значения округляются до 4 знаков (как «(lab)»).

Использование (много точек из CSV — колонки = имена координат; лишние колонки
таблицы UI («№ опыта», «источник», «* (lab)», расход сырья) игнорируются):
    python tools/response_helper.py --world econ --csv proposed_points.csv
    python tools/response_helper.py --proc-bounds T=150:200,P=1:5 --csv seed.csv
"""
from __future__ import annotations

import csv
import os
import sys
import warnings
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np

# repo root в sys.path (скрипт запускают напрямую: python tools/response_helper.py)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.verification import battle_truth as bt  # noqa: E402


_PROC_DEFAULT = 0.5   # середина куба [0,1] для не заданных процесс-осей (код)
_ROUND = 4            # знаков после запятой (как столбцы «свойство (lab)»)
_CODE_TOL = 1e-6      # допуск на выход кода за [0,1] (огрехи округления таблиц)

# Дефолтные РЕАЛЬНЫЕ границы процесс-осей формы сетапа кампании (§17.4,
# campaign_ui.render_process_bounds): ими автонормируем абсолютные значения из
# таблиц дизайна, если пользователь не передал --proc-bounds явно.
_UI_DEFAULT_PROC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "T": (150.0, 200.0),
    "P": (1.0, 5.0),
}

ProcBounds = Mapping[str, Tuple[float, float]]


def _world(world_key: str) -> Dict[str, object]:
    worlds = bt.worlds()
    if world_key not in worlds:
        raise KeyError(f"Неизвестный мир '{world_key}'. Доступно: "
                       f"{', '.join(worlds)}.")
    return worlds[world_key]


def coord_order(world_key: str = "econ") -> List[str]:
    """Порядок составных координат мира: компоненты смеси, затем процесс-оси."""
    w = _world(world_key)
    return list(w["mixture"]) + list(w["process"])


def _proc_to_code(name: str, value: float,
                  proc_bounds: Optional[ProcBounds]) -> float:
    """Значение процесс-оси → код [0,1].

    Приоритет: (1) явные ``proc_bounds[name] = (lo, hi)`` — значение РЕАЛЬНОЕ,
    нормируем ``(value − lo)/(hi − lo)``; (2) значение уже в [0,1] — кодовое,
    берём как есть; (3) значение вне [0,1], а ось есть в дефолтах формы сетапа
    (``_UI_DEFAULT_PROC_BOUNDS``: T=150…200, P=1…5) — АВТОнормируем дефолтами
    с предупреждением (таблицы дизайна UI хранят абсолютные значения);
    (4) иначе — явная ошибка с подсказкой про ``--proc-bounds``.
    """
    v = float(value)
    if proc_bounds and name in proc_bounds:
        lo, hi = float(proc_bounds[name][0]), float(proc_bounds[name][1])
        if not hi > lo:
            raise ValueError(f"Границы процесс-оси {name} некорректны: "
                             f"lo={lo:g} ≥ hi={hi:g}.")
        code = (v - lo) / (hi - lo)
        if code < -_CODE_TOL or code > 1.0 + _CODE_TOL:
            raise ValueError(
                f"Процесс-ось {name}={v:g} вне границ --proc-bounds "
                f"[{lo:g};{hi:g}] (код {code:g} ∉ [0,1]) — проверьте границы.")
        return min(max(code, 0.0), 1.0)

    if -_CODE_TOL <= v <= 1.0 + _CODE_TOL:
        return min(max(v, 0.0), 1.0)

    if name in _UI_DEFAULT_PROC_BOUNDS:
        lo, hi = _UI_DEFAULT_PROC_BOUNDS[name]
        code = (v - lo) / (hi - lo)
        if -_CODE_TOL <= code <= 1.0 + _CODE_TOL:
            warnings.warn(
                f"{name}={v:g} вне [0,1] — принято за АБСОЛЮТНОЕ значение и "
                f"нормировано дефолтными границами сетапа {name}∈[{lo:g};{hi:g}]"
                f" → код {code:.4f}. Если границы в сетапе другие — передайте "
                "их явно: --proc-bounds T=lo:hi,P=lo:hi.", stacklevel=2)
            return min(max(code, 0.0), 1.0)

    raise ValueError(
        f"Процесс-ось {name}={v:g} вне области: не код [0,1] и не попадает в "
        "дефолтные границы сетапа "
        f"{_UI_DEFAULT_PROC_BOUNDS.get(name, '—')}. Передайте реальные границы "
        "вашего сетапа через --proc-bounds, напр. --proc-bounds T=150:200,P=1:5.")


def evaluate_point(coords: Mapping[str, float], *, world_key: str = "econ",
                   proc_bounds: Optional[ProcBounds] = None) -> Dict[str, float]:
    """Отклики синтетической истины battle-теста в точке ``coords`` (БЕЗ шума).

    ``coords`` — словарь ``{имя координаты: значение}``: доли компонентов смеси
    (обязательны) и процесс-оси T,P (по умолчанию середина интервала). Процесс:
    при заданном ``proc_bounds`` (``{имя: (lo, hi)}``, реальные границы сетапа
    §17.4) значения T,P считаются РЕАЛЬНЫМИ и нормируются в код; без него
    значение в [0,1] — кодовое, вне [0,1] — автонормировка дефолтами формы
    сетапа (T=150…200, P=1…5) с предупреждением (см. :func:`_proc_to_code`).
    Лишние ключи ``coords`` игнорируются. Возвращает словарь
    ``{свойство: значение, ..., "price_состав": ..., "price_изд": ...,
    "Σmixture": ...}`` — все числа округлены до 4 знаков (как столбцы «(lab)»).
    ``price_изд = price_состав·ρ`` (§15.6 §3) считается по ρ той же истины.
    """
    w = _world(world_key)
    mix_names = list(w["mixture"])
    proc_names = list(w["process"])

    missing = [nm for nm in mix_names if nm not in coords]
    if missing:
        raise KeyError(f"Не заданы доли компонентов смеси: {', '.join(missing)} "
                       f"(мир '{world_key}', компоненты {mix_names}).")

    vec: List[float] = [float(coords[nm]) for nm in mix_names]
    for nm in proc_names:
        if nm in coords:
            vec.append(_proc_to_code(nm, float(coords[nm]), proc_bounds))
        else:
            vec.append(_PROC_DEFAULT)
    Xc = np.asarray(vec, float).reshape(1, -1)

    truth = w["build_truth"]()              # noise_sd=0 → чистая истина
    comp_price_fn = w["comp_price_fn"]

    y = np.asarray(truth.true(Xc), float).ravel()
    out: Dict[str, float] = {name: round(float(v), _ROUND)
                             for name, v in zip(truth.property_names, y)}

    pc = float(np.asarray(comp_price_fn(Xc), float).ravel()[0])
    out["price_состав"] = round(pc, _ROUND)
    if "rho" in truth.property_names:
        rho = float(np.asarray(truth.truths["rho"].true(Xc), float).ravel()[0])
        out["price_изд"] = round(pc * rho, _ROUND)
    out["Σmixture"] = round(float(sum(float(coords[nm]) for nm in mix_names)),
                            _ROUND)
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _clean_token(tok: str) -> str:
    """Счистить артефакты копипаста из доки/шелла: '\\', '^', кавычки, пробелы.

    cmd/PowerShell НЕ понимают bash-переносы '\\' — при копировании многострочной
    команды токены приходят как '\\A=0.25'. Терпим это молча, чтобы хелпер
    работал с командой, скопированной из плана как есть.
    """
    return tok.strip().strip('"').strip("'").lstrip("\\^").strip()


def _parse_kv(tokens: List[str]) -> Dict[str, float]:
    coords: Dict[str, float] = {}
    for raw in tokens:
        tok = _clean_token(raw)
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(
                f"Ожидалось имя=значение, дано '{raw}'. Подсказка: команду "
                "вводите ОДНОЙ строкой (cmd/PowerShell не понимают '\\'-переносы).")
        k, v = tok.split("=", 1)
        coords[k.strip()] = float(v.strip().replace(",", "."))
    return coords


def _parse_proc_bounds(specs: List[str]) -> Dict[str, Tuple[float, float]]:
    """Разобрать ``--proc-bounds``: 'T=150:200,P=1:5' (разделители ':' или '..')."""
    out: Dict[str, Tuple[float, float]] = {}
    for spec in specs:
        for part in _clean_token(spec).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"--proc-bounds: ожидалось ИМЯ=lo:hi, дано "
                                 f"'{part}'.")
            name, rng = part.split("=", 1)
            rng = rng.replace("..", ":")
            if ":" not in rng:
                raise ValueError(f"--proc-bounds {part}: ожидалось lo:hi "
                                 "(или lo..hi).")
            lo_s, hi_s = rng.split(":", 1)
            out[name.strip()] = (float(lo_s.replace(",", ".")),
                                 float(hi_s.replace(",", ".")))
    return out


def _fmt(out: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:g}" for k, v in out.items())


def _print_point(coords: Dict[str, float], world_key: str,
                 proc_bounds: Optional[ProcBounds]) -> None:
    # Предупреждения автонормировки показываем ЧИТАЕМО в stdout (stderr Windows
    # PowerShell уродует unicode-эскейпами), не глуша их для библиотечных
    # пользователей evaluate_point.
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        out = evaluate_point(coords, world_key=world_key,
                             proc_bounds=proc_bounds)
    order = coord_order(world_key)
    pt = "  ".join(f"{nm}={coords[nm]:g}" for nm in order if nm in coords)
    print(f"  точка: {pt}")
    for wm in wlist:
        print(f"  ℹ️  {wm.message}")
    print(f"  отклики: {_fmt(out)}")
    sm = out.get("Σmixture", 0.0)
    if abs(sm - 1.0) > 1e-3:
        print(f"  ⚠️  Σ долей смеси = {sm:g} ≠ 1 — проверьте состав "
              "(истина считается как введено).")
    print()


def _run_csv(path: str, world_key: str,
             proc_bounds: Optional[ProcBounds]) -> None:
    """Прогнать пачку точек из CSV. Берём ТОЛЬКО колонки-координаты мира;
    остальные колонки таблицы UI («№ опыта», «источник», «* (lab)», расход
    сырья и т.п.) молча игнорируются — можно скармливать выгрузку формы как есть.
    """
    order = set(coord_order(world_key))
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"CSV '{path}' пуст.")
        return
    for i, row in enumerate(rows, start=1):
        coords = {k.strip(): float(str(v).replace(",", "."))
                  for k, v in row.items()
                  if k and k.strip() in order and str(v).strip() != ""}
        print(f"[{i}]")
        _print_point(coords, world_key, proc_bounds)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    world_key = "econ"
    csv_path: Optional[str] = None
    pb_specs: List[str] = []
    kv: List[str] = []
    it = iter(argv)
    for tok in it:
        if tok == "--world":
            world_key = next(it)
        elif tok.startswith("--world="):
            world_key = tok.split("=", 1)[1]
        elif tok == "--csv":
            csv_path = next(it)
        elif tok.startswith("--csv="):
            csv_path = tok.split("=", 1)[1]
        elif tok == "--proc-bounds":
            pb_specs.append(next(it))
        elif tok.startswith("--proc-bounds="):
            pb_specs.append(tok.split("=", 1)[1])
        elif tok in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            kv.append(tok)

    proc_bounds = _parse_proc_bounds(pb_specs) if pb_specs else None

    w = _world(world_key)
    print(f"Мир: {world_key} — {w['label']}")
    if proc_bounds:
        pb = ", ".join(f"{k}∈[{lo:g};{hi:g}]"
                       for k, (lo, hi) in proc_bounds.items())
        print(f"Порядок координат: {', '.join(coord_order(world_key))} "
              f"(процесс в РЕАЛЬНЫХ единицах: {pb} → код [0,1]).")
    else:
        print(f"Порядок координат: {', '.join(coord_order(world_key))} "
              f"(процесс T,P: [0,1] — код; вне [0,1] — автонормировка "
              f"дефолтами сетапа T=150:200, P=1:5; иные границы → "
              f"--proc-bounds).")
    print(f"Отклики: {', '.join(w['responses'])} (+ price_состав/price_изд).")
    print()

    if csv_path:
        _run_csv(csv_path, world_key, proc_bounds)
        return 0
    if not kv:
        print("Не заданы координаты. Пример (одной строкой):\n"
              "  python tools/response_helper.py --world econ "
              "A=0.25 B=0.25 C=0.25 D=0.25 T=0.5 P=0.5\n"
              "Процесс в реальных единицах (как в таблицах UI):\n"
              "  python tools/response_helper.py --proc-bounds T=150:200,P=1:5 "
              "A=0.25 B=0.25 C=0.25 D=0.25 T=175 P=3")
        return 2
    _print_point(_parse_kv(kv), world_key, proc_bounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())