# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Iteration 20 — ручной хелпер откликов battle-прогона (§17, tools/response_helper).

Проверяем, что хелпер, которым пользователь ВРУЧНУЮ снимает «правду лаборатории»
для ручного прохождения UI кампании, считает РОВНО ту же истину, что и battle-тест
(единый источник ``src.verification.battle_truth``): значения откликов совпадают с
``truth.true``, цена изделия = price_состав·ρ, а сумма долей контролируется.
"""
import numpy as np

from tools.response_helper import evaluate_point, coord_order, main
from src.verification import battle_truth as bt


def test_helper_matches_econ_truth():
    """Отклики хелпера в econ-мире = truth.true(Xc) (никакого расхождения)."""
    coords = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25, "T": 0.5, "P": 0.5}
    out = evaluate_point(coords, world_key="econ")

    truth = bt.build_truth_econ()
    Xc = np.array([[coords[n] for n in coord_order("econ")]], float)
    for i, name in enumerate(truth.property_names):
        assert abs(out[name] - round(float(truth.true(Xc)[0, i]), 4)) < 1e-9, name

    # price_изд = price_состав·ρ (§15.6 §3)
    pc = float(bt.comp_price_econ(Xc)[0])
    rho = float(truth.truths["rho"].true(Xc)[0])
    assert abs(out["price_состав"] - round(pc, 4)) < 1e-9
    assert abs(out["price_изд"] - round(pc * rho, 4)) < 1e-9
    assert abs(out["Σmixture"] - 1.0) < 1e-9


def test_helper_matches_3comp_truth():
    """Отклики хелпера в 3comp-мире = truth.true(Xc)."""
    coords = {"A": 0.3, "B": 0.3, "C": 0.4, "T": 0.5, "P": 0.5}
    out = evaluate_point(coords, world_key="3comp")

    truth = bt.build_truth_3comp()
    Xc = np.array([[coords[n] for n in coord_order("3comp")]], float)
    for i, name in enumerate(truth.property_names):
        assert abs(out[name] - round(float(truth.true(Xc)[0, i]), 4)) < 1e-9, name


def test_helper_process_default_is_cube_center():
    """Пропущенные процесс-оси достраиваются серединой куба (T=P=0.5)."""
    full = evaluate_point({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25,
                           "T": 0.5, "P": 0.5}, world_key="econ")
    part = evaluate_point({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                          world_key="econ")
    assert full == part


def test_helper_missing_mixture_raises():
    """Не заданный компонент смеси — явная ошибка (не молчаливый дефолт)."""
    import pytest
    with pytest.raises(KeyError):
        evaluate_point({"A": 0.5, "B": 0.5, "T": 0.5, "P": 0.5}, world_key="econ")


def test_helper_deterministic():
    """Истина без шума → повтор даёт идентичные значения (воспроизводимо)."""
    c = {"A": 0.4, "B": 0.2, "C": 0.2, "D": 0.2}
    assert evaluate_point(c, world_key="econ") == evaluate_point(c, world_key="econ")


def test_cli_single_point_runs(capsys):
    """CLI одной точкой отрабатывает (rc=0) и печатает отклики."""
    rc = main(["--world", "econ", "A=0.25", "B=0.25", "C=0.25", "D=0.25",
               "T=0.5", "P=0.5"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "strength" in captured and "price_изд" in captured
