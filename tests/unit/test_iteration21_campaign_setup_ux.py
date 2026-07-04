# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 21 — C3/UX-правки вкладки «Кампания» (§17.4/§17.6.1).

Проверяем ЧИСТУЮ логику под замечания пользователя по UI (без Streamlit):

  * замечание 4 — универсальная формула стартового N скрининга
    :func:`recommended_seed_size` (N = q·(1+d) + ⌈q·(1+d)/2⌉, ≥ q+d+1);
  * замечание 1 — состав в ЧАСТЯХ ↔ ДОЛЯХ: :func:`mixture_amounts_to_fractions`
    и :func:`resolve_mixture_bounds` (доли / части / пусто);
  * замечание 7 / C3 — выгрузка общей базы кампании в таблицу/Excel с расходом
    сырья (:func:`campaign_base_dataframe` / :func:`campaign_base_excel_bytes`);
  * замечания 6/10 — научный операционный текст денежного канала (без жаргона
    «ЖИВОЙ/ALIVE/ценность раунда»).
"""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_ui as ui
from src.apps import campaign as cv

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
# Замечание 4 — формула стартового N (скрининг)
# ======================================================================
def test_recommended_seed_size_formula():
    # q=3, d=2 → P=9 → N = 9 + ceil(9/2)=5 = 14 (совпадает с демо-дефолтом)
    assert ui.recommended_seed_size(3, 2) == 14
    # q=2, d=1 → P=4 → N = 4 + 2 = 6
    assert ui.recommended_seed_size(2, 1) == 6
    # q=4, d=0 → P=4 → N = 4 + 2 = 6, но не меньше q+d+1=5 → 6
    assert ui.recommended_seed_size(4, 0) == 6


def test_recommended_seed_size_floor_qd1():
    # маленькая смесь: формула не должна опускаться ниже q+d+1
    assert ui.recommended_seed_size(1, 0) >= 1 + 0 + 1


def test_recommended_seed_size_validates():
    with pytest.raises(ValueError):
        ui.recommended_seed_size(0, 2)
    with pytest.raises(ValueError):
        ui.recommended_seed_size(3, -1)


# ======================================================================
# Замечание 1 — части ↔ доли
# ======================================================================
def test_mixture_amounts_to_fractions_normalizes():
    fr = ui.mixture_amounts_to_fractions([1, 1, 2])
    assert np.allclose(fr, [0.25, 0.25, 0.5])
    assert np.isclose(fr.sum(), 1.0)


def test_mixture_amounts_to_fractions_rejects_bad():
    with pytest.raises(ValueError):
        ui.mixture_amounts_to_fractions([])
    with pytest.raises(ValueError):
        ui.mixture_amounts_to_fractions([0, 0, 0])
    with pytest.raises(ValueError):
        ui.mixture_amounts_to_fractions([-1, 2, 3])


def test_resolve_mixture_bounds_fractions_passthrough():
    lo, hi = ui.resolve_mixture_bounds(3, "0, 0, 0", "0.5, 0.5, 1",
                                       mode="fractions")
    assert lo == [0.0, 0.0, 0.0]
    assert hi == [0.5, 0.5, 1.0]


def test_resolve_mixture_bounds_parts_converted():
    # части → доли по сумме верхних границ (mixture_utils): верх = 10 → делим на 10
    lo, hi = ui.resolve_mixture_bounds(2, "1, 2", "4, 6", mode="parts")
    assert np.allclose(hi, [0.4, 0.6])
    assert np.allclose(lo, [0.1, 0.2])


def test_resolve_mixture_bounds_empty_is_full_simplex():
    assert ui.resolve_mixture_bounds(3, "", "") == (None, None)


def test_resolve_mixture_bounds_size_mismatch():
    with pytest.raises(ValueError):
        ui.resolve_mixture_bounds(3, "0, 0", "1, 1", mode="fractions")


# ======================================================================
# Замечание 7 / C3 — выгрузка общей базы + расход сырья
# ======================================================================
def _demo_runner():
    return ui.build_demo_campaign_runner(seed=7, n_seed=14)


def test_campaign_base_dataframe_columns_and_rows():
    r = _demo_runner()
    df = ui.campaign_base_dataframe(r)
    assert len(df) == len(r.points) == 14
    # сквозной номер опыта + источник + составные координаты + отклики
    assert "№ опыта" in df.columns and "источник" in df.columns
    for cn in ui.setup_coord_names(r):
        assert cn in df.columns
    for p in r.property_names:
        assert f"{p} (изм.)" in df.columns
    assert list(df["№ опыта"]) == list(range(1, 15))


def test_campaign_base_dataframe_batch_adds_kg():
    r = _demo_runner()
    df = ui.campaign_base_dataframe(r, batch_kg=100.0)
    mix = list(r.current_schema.mixture_names)
    for cn in mix:
        col = f"{cn} ({ui.MASS_UNIT})"
        assert col in df.columns
    # расход = доля × размер партии; сумма по компонентам ≈ размер партии
    total = sum(float(df[f"{cn} ({ui.MASS_UNIT})"].iloc[0]) for cn in mix)
    assert np.isclose(total, 100.0, atol=1e-2)


def test_campaign_base_excel_bytes_nonempty_xlsx():
    r = _demo_runner()
    data = ui.campaign_base_excel_bytes(r, batch_kg=50.0)
    assert isinstance(data, (bytes, bytearray)) and len(data) > 0
    # .xlsx — это zip-контейнер, начинается с сигнатуры PK
    assert bytes(data[:2]) == b"PK"


def test_campaign_base_dataframe_empty_when_no_points():
    r = ui.build_setup_runner(
        mixture_names=["A", "B"], process_names=["T"],
        process_lower=[0.0], process_upper=[1.0],
        response_names=["y1"], seed=0)
    df = ui.campaign_base_dataframe(r)
    assert df.empty


# ======================================================================
# Замечания 6/10 — научный операционный текст денежного канала
# ======================================================================
def test_money_text_price_input_is_operational_no_jargon():
    r = _demo_runner()
    ex = cv.branch_money_explanation(r, "premium", n_candidates=120, n_mc=64,
                                     seed=0)
    assert ex["reason_code"] == "price_input_alive"
    text = ex["text"]
    # научный операционный язык
    assert "себестоимост" in text.lower()
    assert "УЧИТЫВАЕТСЯ" in text
    # без внутреннего жаргона
    assert "ЖИВОЙ" not in text and "ALIVE" not in text
    assert "Денежная ценность раунда" not in text


def test_money_text_rho_optimized_zeroed_operational():
    r = _demo_runner()
    ex = cv.branch_money_explanation(r, "rho_focus", n_candidates=120, n_mc=64,
                                     seed=0)
    assert ex["reason_code"] == "rho_optimized_zeroed"
    assert ex["economic_value"] == 0.0
    assert "дважды" in ex["text"].lower()


# ======================================================================
# Замечание 1 (форма ограничений состава) — доли/массовые части, L·/U·
# ======================================================================
def test_render_composition_bounds_empty_names_no_streamlit():
    # q == 0 — ранний выход БЕЗ обращения к Streamlit (чистая ветка)
    assert ui.render_composition_bounds([]) == (None, None)


def test_parts_form_math_matches_core_tightest_box():
    """Форма массовых частей делегирует в каноничную parts_ranges_to_fraction_bounds
    (tightest box), а не в приблизительную нормировку — фиксируем математику."""
    from src.core.simplex import parts_ranges_to_fraction_bounds
    # база A=100, B/C по 0..50 частей
    lo, hi = parts_ranges_to_fraction_bounds([100, 0, 0], [100, 50, 50])
    # A: min=100/(100+50+50)=0.5; max=100/(100+0+0)=1.0
    assert np.isclose(lo[0], 0.5) and np.isclose(hi[0], 1.0)
    # B: min=0; max=50/(50+100+0)=1/3
    assert np.isclose(lo[1], 0.0) and np.isclose(hi[1], 1.0 / 3.0)


# ======================================================================
# headless AppTest — форма ограничений состава рендерится и сетап строится
# ======================================================================
try:  # AppTest доступен только при установленном streamlit
    import os
    from streamlit.testing.v1 import AppTest

    _REPO = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    _APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")
    _HAS_APPTEST = True
except Exception:  # noqa: BLE001
    _HAS_APPTEST = False


@pytest.mark.skipif(not _HAS_APPTEST, reason="streamlit AppTest недоступен")
def test_setup_composition_form_renders_and_builds():
    at = AppTest.from_file(_APP, default_timeout=240).run()
    assert not at.exception
    # радио «Способ ввода» из формы ограничений состава присутствует
    radio_labels = {w.label for w in at.radio}
    assert "Способ ввода" in radio_labels
    # сборка проекта в режиме «Доли» по умолчанию (границы 0…1 → полный симплекс)
    b = [w for w in at.button if w.key == "setup_build"]
    assert b, "кнопка setup_build не найдена"
    b[0].click().run()
    assert not at.exception
    r = at.session_state["campaign_ctrl"].runner
    assert r.q == 3 and r.dim == 5


