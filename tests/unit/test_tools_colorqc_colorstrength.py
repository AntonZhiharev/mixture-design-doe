# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Хелпер ColorStrength из журнала Color QC2 (``tools/colorqc_colorstrength.py``).

Синтетический ``*.colors`` (base64(JSON) + ``\\r\\n1``): декодирование,
Кубелка–Мунк, пропуск неизмеренных (нулевых) длин волн, эталон = 100 %.
"""
import base64
import json

import numpy as np
import pytest

from tools.colorqc_colorstrength import (
    color_strength_table,
    decode_colors_file,
    kubelka_munk,
    run_number,
)


def _record(uid, name, r_pct, target=None):
    rec = {
        "uid": uid, "name": name, "created_at": "2026-09-30T19:00:00+03:00",
        "lab_data_l": 50.0, "lab_data_a": 0.0, "lab_data_b": 0.0,
        "spectral_info": json.dumps({
            "wave_number": len(r_pct), "wave_interval": 10, "wave_start": 380,
            "spectral_data": list(r_pct),
        }),
    }
    if target:
        rec["target_uid"] = target
    return rec


def test_kubelka_munk_and_zero_is_nan():
    ks = kubelka_munk([50.0, 0.0])
    assert ks[0] == pytest.approx(0.25)  # (1-0.5)^2/(2*0.5)
    assert np.isnan(ks[1])


def test_decode_and_color_strength(tmp_path):
    n = 42  # 380..790 нм
    ref = _record("ref", "Образец 561-34 эталон_1", [0.0, 0.0] + [50.0] * (n - 2))
    smp = _record("s1", "Образец 561-8_1_2026-09-30", [0.0, 0.0] + [25.0] * (n - 2),
                  target="ref")
    payload = base64.b64encode(json.dumps([ref, smp]).encode("utf-8"))
    path = tmp_path / "x.colors"
    path.write_bytes(payload + b"\r\n1")

    records = decode_colors_file(path)
    summary, r_df, ks_df = color_strength_table(records)

    cs = summary["ColorStrength 400–700, %"].tolist()
    ks_ref = 0.25
    ks_smp = 0.75 ** 2 / 0.5
    assert cs[0] == pytest.approx(100.0)
    assert cs[1] == pytest.approx(100.0 * ks_smp / ks_ref)
    assert summary["№ опыта"].tolist() == [34, 8]
    assert np.isnan(ks_df.loc[380].iloc[0])


def test_run_number_absent():
    assert run_number("эталон") is None
