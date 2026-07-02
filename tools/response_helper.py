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
  * процесс-оси T,P — в КОДЕ [0,1] (как в схеме battle-истины и как задаются
    границы процесса [0,1] в сетапе кампании §17.4).

Использование (одна точка):
    python tools/response_helper.py --world econ A=0.25 B=0.25 C=0.25 D=0.25 T=0.5 P=0.5
    python tools/response_helper.py A=0.3 B=0.3 C=0.2 D=0.2 T=0.5 P=0.5   # world=econ

Пропущенные координаты: mixture-компоненты обязательны; процесс-оси по умолчанию
0.5 (середина куба). Значения округляются до 4 знаков (как столбцы «(lab)»).

Использование (много точек из CSV — колонки = имена координат):
    python tools/response_helper.py --world econ --csv proposed_points.csv
"""
from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Mapping, Optional

import numpy as np

# repo root в sys.path (скрипт запускают напрямую: python tools/response_helper.py)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.verification import battle_truth as bt  # noqa: E402


_PROC_DEFAULT = 0.5   # середина куба [0,1] для не заданных процесс-осей
_ROUND = 4            # знаков после запятой (как столбцы «свойство (lab)»)


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


def evaluate_point(coords: Mapping[str, float], *, world_key: str = "econ"
                   ) -> Dict[str, float]:
    """Отклики синтетической истины battle-теста в точке ``coords`` (БЕЗ шума).

    ``coords`` — словарь ``{имя координаты: значение}``: доли компонентов смеси
    (обязательны) и процесс-оси T,P в коде [0,1] (по умолчанию 0.5). Возвращает
    словарь ``{свойство: значение, ..., "price_состав": ..., "price_изд": ...,
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
    vec += [float(coords.get(nm, _PROC_DEFAULT)) for nm in proc_names]
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
    out["Σmixture"] = round(float(sum(coords[nm] for nm in mix_names)), _ROUND)
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_kv(tokens: List[str]) -> Dict[str, float]:
    coords: Dict[str, float] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Ожидалось имя=значение, дано '{tok}'.")
        k, v = tok.split("=", 1)
        coords[k.strip()] = float(v.strip().replace(",", "."))
    return coords


def _fmt(out: Dict[str, float]) -> str:
    return "  ".join(f"{k}={v:g}" for k, v in out.items())


def _print_point(coords: Dict[str, float], world_key: str) -> None:
    out = evaluate_point(coords, world_key=world_key)
    order = coord_order(world_key)
    pt = "  ".join(f"{nm}={coords.get(nm, _PROC_DEFAULT):g}" for nm in order)
    print(f"  точка: {pt}")
    print(f"  отклики: {_fmt(out)}")
    sm = out.get("Σmixture", 0.0)
    if abs(sm - 1.0) > 1e-3:
        print(f"  ⚠️  Σ долей смеси = {sm:g} ≠ 1 — проверьте состав "
              "(истина считается как введено).")
    print()


def _run_csv(path: str, world_key: str) -> None:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"CSV '{path}' пуст.")
        return
    for i, row in enumerate(rows, start=1):
        coords = {k: float(str(v).replace(",", "."))
                  for k, v in row.items() if str(v).strip() != ""}
        print(f"[{i}]")
        _print_point(coords, world_key)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    world_key = "econ"
    csv_path: Optional[str] = None
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
        elif tok in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            kv.append(tok)

    w = _world(world_key)
    print(f"Мир: {world_key} — {w['label']}")
    print(f"Порядок координат: {', '.join(coord_order(world_key))} "
          f"(процесс T,P — код [0,1]).")
    print(f"Отклики: {', '.join(w['responses'])} (+ price_состав/price_изд).")
    print()

    if csv_path:
        _run_csv(csv_path, world_key)
        return 0
    if not kv:
        print("Не заданы координаты. Пример:\n"
              "  python tools/response_helper.py --world econ "
              "A=0.25 B=0.25 C=0.25 D=0.25 T=0.5 P=0.5")
        return 2
    _print_point(_parse_kv(kv), world_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
