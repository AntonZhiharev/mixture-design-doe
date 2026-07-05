# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 25 — C2 (§17.6.1): загрузка проекта ПОДТЯГИВАЕТ настройки в UI.

Багрепорт: проект сохраняется на этапе стартового дизайна (компоненты, границы
долей, процесс-параметры, точки), но после загрузки форма сетапа показывала
ДЕФОЛТЫ («A, B, C», 0…1, 150…200, seed=1) — пользователь видел «настройки не
подтянулись», хотя движок (runner) загружался верно.

Фикс (канон «логика+тест, потом UI»):
  * :func:`campaign_ui.setup_prefill_from_runner` — чистая проекция «раннер →
    ключи виджетов формы сетапа» (компоненты/отклики/процесс, границы долей,
    реальные границы процесс-осей, seed); применяется отложенно через
    ``setup_prefill_pending`` (нельзя менять ключ созданного виджета);
  * :func:`campaign_ui.project_settings_dataframe` — read-only сводка настроек
    ДЕЙСТВУЮЩЕГО проекта (читает движок, не форму);
  * headless AppTest: save (draft-этап) → свежая сессия → load → форма сетапа
    показывает настройки проекта, seed-точки восстановлены.
"""
import os
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_state as cst
from src.apps.campaign_ui import (build_setup_runner,
                                  project_settings_dataframe,
                                  setup_prefill_from_runner)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _custom_runner():
    """Раннер с НЕтривиальными настройками (границы долей + реальные T/P)."""
    return build_setup_runner(
        mixture_names=["Смола", "Отвердитель", "Пигмент"],
        process_names=["T", "P"],
        process_lower=[120.0, 2.0], process_upper=[180.0, 6.0],
        response_names=["вязкость", "прочность"],
        mixture_lower=[0.5, 0.1, 0.0], mixture_upper=[0.8, 0.3, 0.2],
        seed=5)


# ======================================================================
# Чистая логика: prefill формы сетапа из раннера
# ======================================================================
def test_setup_prefill_maps_runner_to_form_keys():
    pre = setup_prefill_from_runner(_custom_runner())
    assert pre["setup_mix"] == "Смола, Отвердитель, Пигмент"
    assert pre["setup_resp"] == "вязкость, прочность"
    assert pre["setup_proc"] == "T, P"
    assert pre["setup_seed"] == 5
    assert pre["setup_comp_mode"].startswith("Доли")
    # границы долей (ключи формы включают q=3)
    assert pre["setup_lo_3_0"] == pytest.approx(0.5)
    assert pre["setup_hi_3_0"] == pytest.approx(0.8)
    assert pre["setup_lo_3_1"] == pytest.approx(0.1)
    assert pre["setup_hi_3_2"] == pytest.approx(0.2)
    # реальные границы процесс-осей (ключи включают d=2)
    assert pre["setup_plo_2_0"] == pytest.approx(120.0)
    assert pre["setup_phi_2_0"] == pytest.approx(180.0)
    assert pre["setup_plo_2_1"] == pytest.approx(2.0)
    assert pre["setup_phi_2_1"] == pytest.approx(6.0)


def test_setup_prefill_roundtrip_through_save_load(tmp_path):
    """prefill из ЗАГРУЖЕННОГО раннера == prefill из исходного (C2)."""
    r0 = _custom_runner()
    cst.save_campaign(r0, str(tmp_path), "p")
    r1 = cst.load_campaign(str(tmp_path), "p")
    assert setup_prefill_from_runner(r1) == setup_prefill_from_runner(r0)


def test_project_settings_dataframe_reads_engine_bounds():
    df = project_settings_dataframe(_custom_runner())
    assert list(df["переменная"]) == ["Смола", "Отвердитель", "Пигмент", "T", "P"]
    row = df[df["переменная"] == "Смола"].iloc[0]
    assert row["нижняя"] == pytest.approx(0.5)
    assert row["верхняя"] == pytest.approx(0.8)
    assert "доля" in row["тип"]
    row_t = df[df["переменная"] == "T"].iloc[0]
    assert row_t["нижняя"] == pytest.approx(120.0)
    assert row_t["верхняя"] == pytest.approx(180.0)
    assert "процесс" in row_t["тип"]


# ======================================================================
# headless AppTest: load → форма показывает настройки, точки восстановлены
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.apps.streamlit_app import CAMPAIGN_ROOT  # noqa: E402
from src.apps import campaign as cv  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")


def test_load_prefills_setup_form_and_restores_seed_points():
    name = "regr_it25_prefill"
    try:
        cst.delete_campaign(CAMPAIGN_ROOT, name)
    except Exception:  # noqa: BLE001
        pass
    try:
        runner = _custom_runner()
        ctrl = cv.CampaignController(runner)
        X = np.asarray(ctrl.propose_seed(6, seed=2), float)
        draft = {"seed_X": [[float(v) for v in row] for row in X]}
        cst.save_campaign(runner, CAMPAIGN_ROOT, name, draft=draft)

        # свежая сессия: загрузка через сайдбар
        at = AppTest.from_file(APP, default_timeout=300).run()
        at.selectbox(key="campaign_select").set_value(name).run()
        btn = [w for w in at.button if w.key == "load_campaign"]
        assert btn, "кнопка load_campaign не найдена"
        btn[0].click().run()
        assert not at.exception
        assert not [e for e in at.error
                    if "Не удалось загрузить" in str(e.value)]

        ss = at.session_state
        # регресс: форма сетапа показывает НАСТРОЙКИ проекта, а не дефолты
        assert ss["setup_mix"] == "Смола, Отвердитель, Пигмент"
        assert ss["setup_resp"] == "вязкость, прочность"
        assert ss["setup_seed"] == 5
        assert float(ss["setup_lo_3_0"]) == pytest.approx(0.5)
        assert float(ss["setup_hi_3_0"]) == pytest.approx(0.8)
        assert float(ss["setup_plo_2_0"]) == pytest.approx(120.0)
        assert float(ss["setup_phi_2_0"]) == pytest.approx(180.0)
        # регресс: черновик стартового дизайна (точки) восстановлен
        assert np.allclose(np.asarray(ss["setup_seed_X"], float), X)
        # движок загружен с верными границами
        lr = ss["campaign_ctrl"].runner
        mb = lr.current_schema.mixture_block()
        assert list(mb.lower) == pytest.approx([0.5, 0.1, 0.0])
        assert list(mb.upper) == pytest.approx([0.8, 0.3, 0.2])
    finally:
        try:
            cst.delete_campaign(CAMPAIGN_ROOT, name)
        except Exception:  # noqa: BLE001
            pass