# Copyright 2026 AZH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Итерация 9, шаг 2b — режим «массовые части (база=100)» в UI (AppTest).

Проверяет, что переключение способа ввода ограничений на «Массовые части»
формирует нетривиальные границы долей и корректно строит регион в runner.
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")

PARTS_MODE = "Массовые части (база = 100)"


def test_parts_mode_builds_constrained_region():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception

    radio = [w for w in at.radio if w.key == "bmode_0"]
    assert radio, "переключатель режима ограничений не найден"
    radio[0].set_value(PARTS_MODE).run()
    assert not at.exception

    runner = at.session_state["runner"]
    lower = runner.region.lower
    upper = runner.region.upper
    # база (comp0) фиксирована 100, прочие 0..10 → доля базы имеет L>0.5
    assert lower[0] > 0.5
    # ограничения нетривиальны (не весь симплекс 0..1)
    assert lower.sum() > 0.0
    # регион валиден: ΣL ≤ 1 ≤ ΣU
    assert lower.sum() <= 1.0 + 1e-9
    assert upper.sum() >= 1.0 - 1e-9
