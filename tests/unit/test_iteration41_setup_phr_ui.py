# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 41 / UI_REVISION_SPEC iter41 — сетап: ввод phr-спеки + паспорт
кампании (чистые UI-хелперы, канон «логика+тест, потом UI»).

Проверяем:

  * ``parse_phr_spec_json`` — валидный JSON референсной спеки iter35 даёт
    спеку с golden-хешем; ошибки JSON человекочитаемы; ошибки КОНСТРУКТОРА
    (циклы/ссылки) уходят наружу как есть (A0.6);
  * ``phr_spec_summary_dataframe`` / ``phr_spec_fraction_dataframe`` —
    структура и согласованность с phr_intervals()/fraction_bounds();
  * ``parse_preflight_pairs`` / ``preflight_pairs_to_text`` — round-trip
    (включая оси-суммы) и явные ошибки формата;
  * ``setup_mixture_names`` — при активной спеке имена компонентов берутся
    из ``spec.component_names`` (поле формы игнорируется);
  * ``setup_prefill_from_runner`` — раннер с активной политикой (спека,
    метка, пары) даёт префилл формы с режимом «phr-спека (JSON)» и
    round-trip'ящимся JSON;
  * ``campaign_passport_dataframe`` — строки паспорта (hash-префикс 12,
    метка, пары; «—» когда политика не задана).
"""
import json
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import campaign_ui as ui
from src.apps.campaign_ui import build_setup_runner
from src.design.phr_sampler import PhrSpec

# Единый источник референсной 19-компонентной спеки PVC (iter35) и её
# golden-хеша (iter40): дублировать нельзя.
from tests.unit.test_iteration35_phr_spec_campaign import (COMPONENTS,
                                                           RECIPE_DICTS)
from tests.unit.test_iteration40_campaign_policy_persistence import \
    REFERENCE_HASH

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PAIRS_TEXT = ("UV_CSFCP | TiO2_BLR895\n"
              "\n"
              "T | PMPlus_8, DL_531\n"
              "DINP|TiO2_BLR895\n")

PAIRS_NORM = [(["UV_CSFCP"], ["TiO2_BLR895"]),
              (["T"], ["PMPlus_8", "DL_531"]),
              (["DINP"], ["TiO2_BLR895"])]


def _pvc_runner():
    """Раннер PVC-кампании с полной политикой (как в iter40)."""
    spec = PhrSpec.from_dicts(RECIPE_DICTS)
    lo, hi = spec.fraction_bounds()
    runner = build_setup_runner(
        mixture_names=list(COMPONENTS), process_names=["T"],
        process_lower=[150.0], process_upper=[200.0],
        response_names=["gloss"],
        mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=1)
    runner.set_phr_spec(spec)
    runner.set_campaign_label("PVC-профиль-2026")
    runner.set_preflight_pairs(ui.parse_preflight_pairs(PAIRS_TEXT))
    return runner


def _plain_runner():
    """Раннер без политики кампании (спеки/метки/пар нет)."""
    return build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T"],
        process_lower=[0.0], process_upper=[1.0],
        response_names=["strength"], seed=1)


# ======================================================================
# 1. parse_phr_spec_json — валидный ввод и человекочитаемые ошибки
# ======================================================================
class TestParsePhrSpecJson:

    def test_reference_spec_roundtrips_hash(self):
        """JSON референсной спеки → спека с golden-хешем и теми же именами."""
        spec = ui.parse_phr_spec_json(json.dumps(RECIPE_DICTS,
                                                 ensure_ascii=False))
        assert spec.spec_hash() == REFERENCE_HASH
        assert list(spec.component_names) == list(COMPONENTS)

    def test_pretty_json_same_hash(self):
        """indent/пробелы не влияют: hash считается по to_dicts, не по тексту."""
        spec = ui.parse_phr_spec_json(json.dumps(RECIPE_DICTS, indent=2,
                                                 ensure_ascii=False))
        assert spec.spec_hash() == REFERENCE_HASH

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="Пустой ввод"):
            ui.parse_phr_spec_json("   \n ")

    def test_broken_json_raises_readable(self):
        with pytest.raises(ValueError, match="Некорректный JSON"):
            ui.parse_phr_spec_json('[{"name": "x", ')

    def test_non_list_raises(self):
        with pytest.raises(ValueError, match="СПИСОК"):
            ui.parse_phr_spec_json('{"name": "x", "mode": "fixed"}')

    def test_non_dict_item_raises(self):
        with pytest.raises(ValueError, match="JSON-объектом"):
            ui.parse_phr_spec_json('[1, 2, 3]')

    def test_constructor_error_propagates(self):
        """Ошибка конструктора PhrSpec (битая ссылка) уходит наружу как есть."""
        bad = [{"name": "resin", "mode": "fixed", "value": 100.0},
               {"name": "a", "mode": "share_of", "of": "nope",
                "lo": 0.0, "hi": 1.0}]
        with pytest.raises(ValueError, match="референс 'nope' не найден"):
            ui.parse_phr_spec_json(json.dumps(bad))


# ======================================================================
# 2. Таблицы спеки: summary по узлам + phr-интервалы/доли компонентов
# ======================================================================
class TestSpecDataframes:

    def setup_method(self):
        self.spec = PhrSpec.from_dicts(RECIPE_DICTS)

    def test_summary_structure_and_leaves(self):
        df = ui.phr_spec_summary_dataframe(self.spec)
        assert list(df.columns) == ["узел", "режим", "lo", "hi", "ref",
                                    "cap_to", "cap_ratio", "компонент смеси"]
        assert len(df) == len(RECIPE_DICTS)                  # все узлы (20)
        # resin — внутренний узел share-группы, НЕ компонент смеси
        resin = df[df["узел"] == "resin"].iloc[0]
        assert not bool(resin["компонент смеси"])
        assert int(df["компонент смеси"].sum()) == len(COMPONENTS)

    def test_summary_cap_and_fixed(self):
        df = ui.phr_spec_summary_dataframe(self.spec).set_index("узел")
        uv = df.loc["UV_CSFCP"]
        assert uv["cap_to"] == "DINP, ESO"
        assert uv["cap_ratio"] == pytest.approx(0.03)
        eso = df.loc["ESO"]                                  # fixed: lo=hi=value
        assert eso["lo"] == eso["hi"] == pytest.approx(2.50)

    def test_fraction_dataframe_matches_spec(self):
        df = ui.phr_spec_fraction_dataframe(self.spec)
        assert list(df.index) == list(COMPONENTS)
        iv = self.spec.phr_intervals()
        lo, hi = self.spec.fraction_bounds()
        np.testing.assert_allclose(df["phr lo"].to_numpy(),
                                   [iv[nm][0] for nm in COMPONENTS])
        np.testing.assert_allclose(df["phr hi"].to_numpy(),
                                   [iv[nm][1] for nm in COMPONENTS])
        np.testing.assert_allclose(df["доля L"].to_numpy(), lo, atol=1e-6)
        np.testing.assert_allclose(df["доля U"].to_numpy(), hi, atol=1e-6)


# ======================================================================
# 3. parse_preflight_pairs / preflight_pairs_to_text — round-trip
# ======================================================================
class TestPreflightPairsText:

    def test_parse_basic_with_sum_axis(self):
        pairs = ui.parse_preflight_pairs(PAIRS_TEXT)
        assert pairs == PAIRS_NORM

    def test_roundtrip_text(self):
        txt = ui.preflight_pairs_to_text(PAIRS_NORM)
        assert ui.parse_preflight_pairs(txt) == PAIRS_NORM

    def test_roundtrip_from_runner_normalized(self):
        """Нормализованные пары раннера → текст → парсер → те же пары."""
        r = _pvc_runner()
        txt = ui.preflight_pairs_to_text(r.preflight_pairs)
        assert ui.parse_preflight_pairs(txt) == [
            (list(a), list(b)) for a, b in r.preflight_pairs]

    def test_empty_text_gives_empty_list(self):
        assert ui.parse_preflight_pairs("") == []
        assert ui.parse_preflight_pairs("\n  \n") == []
        assert ui.preflight_pairs_to_text([]) == ""

    def test_line_without_separator_raises(self):
        with pytest.raises(ValueError, match="Строка 1"):
            ui.parse_preflight_pairs("UV_CSFCP TiO2_BLR895")

    def test_two_separators_raise(self):
        with pytest.raises(ValueError, match="ровно один"):
            ui.parse_preflight_pairs("A | B | C")

    def test_empty_side_raises(self):
        with pytest.raises(ValueError, match="пустая сторона"):
            ui.parse_preflight_pairs("A | ")


# ======================================================================
# 4. Имена из спеки при сборке + префилл формы из раннера
# ======================================================================
class TestNamesAndPrefill:

    def test_setup_mixture_names_prefers_spec(self):
        spec = PhrSpec.from_dicts(RECIPE_DICTS)
        assert ui.setup_mixture_names(["X", "Y"], spec) == list(COMPONENTS)
        assert ui.setup_mixture_names(["X", "Y"], None) == ["X", "Y"]

    def test_prefill_with_active_spec(self):
        out = ui.setup_prefill_from_runner(_pvc_runner())
        assert out["setup_comp_mode"] == "phr-спека (JSON)"
        # JSON формы round-trip'ится в ту же геометрию (hash бит-в-бит)
        spec2 = ui.parse_phr_spec_json(out["setup_phr_json"])
        assert spec2.spec_hash() == REFERENCE_HASH
        assert out["setup_campaign_label"] == "PVC-профиль-2026"
        assert ui.parse_preflight_pairs(out["setup_preflight_pairs"]) == \
            PAIRS_NORM
        assert out["setup_mix"] == ", ".join(COMPONENTS)

    def test_prefill_without_spec_stays_fraction_mode(self):
        out = ui.setup_prefill_from_runner(_plain_runner())
        assert out["setup_comp_mode"] == "Доли (0…1)"
        assert "setup_phr_json" not in out
        assert out["setup_campaign_label"] == ""
        assert out["setup_preflight_pairs"] == ""

    def test_build_with_spec_names_accepts_set_phr_spec(self):
        """Сквозная сборка «как кнопка»: имена из спеки → set_phr_spec ОК."""
        spec = ui.parse_phr_spec_json(json.dumps(RECIPE_DICTS))
        lo, hi = spec.fraction_bounds()
        mix = ui.setup_mixture_names(["A", "B", "C"], spec)  # поле игнорируется
        runner = build_setup_runner(
            mixture_names=mix, process_names=["T"],
            process_lower=[150.0], process_upper=[200.0],
            response_names=["gloss"],
            mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=1)
        runner.set_phr_spec(spec)                            # не бросает
        assert runner.phr_spec is spec
        assert list(runner.current_schema.mixture_names) == list(COMPONENTS)


# ======================================================================
# 5. Паспорт кампании (строки для панели настроек)
# ======================================================================
class TestPassportDataframe:

    def test_full_policy_rows(self):
        df = ui.campaign_passport_dataframe(_pvc_runner()).set_index("параметр")
        spec_val = str(df.loc["phr-спека (decode-слой)", "значение"])
        assert REFERENCE_HASH[:12] in spec_val
        assert "q=19" in spec_val
        assert df.loc["метка кампании", "значение"] == "PVC-профиль-2026"
        pairs_val = str(df.loc["обязательные 2D-пары", "значение"])
        assert "PMPlus_8, DL_531" in pairs_val

    def test_empty_policy_shows_dashes(self):
        df = ui.campaign_passport_dataframe(_plain_runner())
        assert list(df["значение"]) == ["—", "—", "—"]