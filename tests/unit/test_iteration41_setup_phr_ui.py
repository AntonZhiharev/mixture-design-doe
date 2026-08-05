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
    метка, пары; «—» когда политика не задана);
  * iter41.4 — ИЕРАРХИЧЕСКИЙ ручной ввод: ``phr_tree_from_spec`` /
    ``phr_tree_to_dicts`` (round-trip референсной спеки бит-в-бит по
    ``spec_hash``), ``validate_phr_tree`` (группа без детей, ratio_to без
    ссылки, дубли имён, cap без ratio), ``phr_tree_move`` (порядок узлов
    ВХОДИТ в hash) и таблица детей группы.
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


# ======================================================================
# 6. iter41.4 — иерархический ручной ввод (дерево «группа → компоненты»)
# ======================================================================
def _tree_hash(tree) -> str:
    """Дерево ввода → spec_hash собранной из него спеки."""
    return PhrSpec.from_dicts(ui.phr_tree_to_dicts(tree)).spec_hash()


class TestPhrTreeRoundTrip:
    """Ручной ввод обязан давать ТУ ЖЕ спеку, что JSON-канал."""

    def test_reference_spec_round_trip_keeps_hash(self):
        """spec → дерево → dicts → spec: hash бит-в-бит (порядок сохранён)."""
        spec = PhrSpec.from_dicts(RECIPE_DICTS)
        tree = ui.phr_tree_from_spec(spec)
        assert _tree_hash(tree) == REFERENCE_HASH
        assert ui.phr_tree_to_dicts(tree) == spec.to_dicts()

    def test_reference_tree_has_resin_group_with_two_children(self):
        """Узел со share_of-детьми становится ГРУППОЙ, дети — внутри неё."""
        tree = ui.phr_tree_from_spec(PhrSpec.from_dicts(RECIPE_DICTS))
        resin = next(b for b in tree if b["name"] == "resin")
        assert resin["kind"] == "group"
        assert resin["total_mode"] == "fixed"
        assert resin["value"] == 100.0
        assert [c["name"] for c in resin["children"]] == ["PVC_67", "PVC_71"]
        # UV с динамическим потолком остаётся ОДИНОЧНЫМ узлом
        uv = next(b for b in tree if b["name"] == "UV_CSFCP")
        assert uv["kind"] == "single" and uv["mode"] == "absolute"
        assert list(uv["cap_to"]) == ["DINP", "ESO"]

    def test_group_expands_to_total_then_children(self):
        """Группа разворачивается в узел-тотал + share_of-детей по порядку."""
        tree = [ui.phr_group_block(
            "FILLER.total", total_mode="absolute", lo=5.0, hi=25.0,
            children=[{"name": "Chalk_95T", "lo": 0.3, "hi": 1.0},
                      {"name": "Chalk_1T", "lo": 0.0, "hi": 0.7}])]
        assert ui.phr_tree_to_dicts(tree) == [
            {"name": "FILLER.total", "mode": "absolute", "lo": 5.0, "hi": 25.0},
            {"name": "Chalk_95T", "mode": "share_of", "of": "FILLER.total",
             "lo": 0.3, "hi": 1.0},
            {"name": "Chalk_1T", "mode": "share_of", "of": "FILLER.total",
             "lo": 0.0, "hi": 0.7},
        ]

    def test_cap_to_text_equals_cap_to_list(self):
        """cap_to строкой «DINP, ESO» и списком — одна и та же спека."""
        base = ui.phr_tree_from_spec(PhrSpec.from_dicts(RECIPE_DICTS))
        as_text = [dict(b) for b in base]
        uv = next(b for b in as_text if b["name"] == "UV_CSFCP")
        uv["cap_to"] = "DINP, ESO"
        assert _tree_hash(as_text) == _tree_hash(base) == REFERENCE_HASH


class TestPhrTreeValidation:
    """Ошибки ловятся ДО конструктора и указывают на блок (A0.6)."""

    def test_group_without_children_is_rejected(self):
        """Пустая группа молча стала бы компонентом смеси — это отказ."""
        with pytest.raises(ValueError, match="без компонентов"):
            ui.phr_tree_to_dicts([ui.phr_group_block("STAB.total", lo=3.5,
                                                     hi=5.0)])

    def test_ratio_to_without_reference_is_rejected(self):
        with pytest.raises(ValueError, match="ratio_to"):
            ui.phr_tree_to_dicts([ui.phr_single_block(
                "SBM_55", mode="ratio_to", lo=0.02, hi=0.09)])

    def test_duplicate_names_rejected_across_levels(self):
        """Тотал группы и компонент делят ОДНО пространство имён."""
        tree = [ui.phr_group_block("X", lo=1.0, hi=2.0,
                                   children=[{"name": "c", "lo": 0.0, "hi": 1.0}]),
                ui.phr_single_block("c", mode="fixed", value=1.0)]
        with pytest.raises(ValueError, match="уже занято"):
            ui.phr_tree_to_dicts(tree)

    def test_cap_ratio_without_cap_to_rejected(self):
        with pytest.raises(ValueError, match="cap_to"):
            ui.phr_tree_to_dicts([ui.phr_single_block(
                "UV", mode="absolute", lo=0.05, hi=0.3, cap_ratio=0.03)])

    def test_empty_tree_rejected(self):
        with pytest.raises(ValueError, match="пуста"):
            ui.phr_tree_to_dicts([])

    def test_inverted_bounds_rejected(self):
        with pytest.raises(ValueError, match="больше верхней"):
            ui.phr_tree_to_dicts([ui.phr_single_block(
                "DINP", mode="absolute", lo=14.0, hi=4.0)])


class TestPhrTreeOrder:
    """Порядок узлов — часть спеки: кнопки ▲/▼ меняют spec_hash."""

    def test_move_changes_hash(self):
        tree = ui.phr_tree_from_spec(PhrSpec.from_dicts(RECIPE_DICTS))
        moved = ui.phr_tree_move(tree, 1, +1)          # DINP ↔ ESO
        assert [b["name"] for b in moved][:3] != [b["name"] for b in tree][:3]
        assert _tree_hash(moved) != REFERENCE_HASH

    def test_move_is_pure_and_reversible(self):
        tree = ui.phr_tree_from_spec(PhrSpec.from_dicts(RECIPE_DICTS))
        names = [b["name"] for b in tree]
        back = ui.phr_tree_move(ui.phr_tree_move(tree, 2, +1), 3, -1)
        assert [b["name"] for b in back] == names      # исходный не мутирован
        assert _tree_hash(back) == REFERENCE_HASH

    def test_move_out_of_range_is_noop(self):
        tree = [ui.phr_single_block("a", mode="fixed", value=1.0),
                ui.phr_single_block("b", mode="fixed", value=2.0)]
        assert [b["name"] for b in ui.phr_tree_move(tree, 0, -1)] == ["a", "b"]
        assert [b["name"] for b in ui.phr_tree_move(tree, 1, +1)] == ["a", "b"]


class TestPhrChildrenTable:
    """Таблица-редактор детей группы (порядок строк = порядок узлов)."""

    def test_dataframe_round_trip(self):
        kids = [{"name": "PF711", "lo": 0.6, "hi": 1.0},
                {"name": "PF711LB", "lo": 0.0, "hi": 0.4}]
        blk = ui.phr_group_block("STAB.total", lo=3.5, hi=5.0, children=kids)
        df = ui.phr_children_dataframe(blk)
        assert list(df.columns) == ["компонент", "доля L", "доля U"]
        assert ui.phr_children_from_dataframe(df) == kids

    def test_empty_rows_are_dropped(self):
        """Пустой хвост динамического редактора — не ошибка ввода."""
        df = ui.phr_children_dataframe({"children": []})
        assert ui.phr_children_from_dataframe(df) == []


