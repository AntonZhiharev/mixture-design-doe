# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 54 / UI_REVISION_SPEC P2.3 — паспорт кампании: лоты сырья,
anchor-рецепты (phr), разрешение весов (δ), group_order — в state и форму
сетапа (CAMPAIGN_SPEC_PVC §3: «записать ДО первого замера, задним числом
не восстанавливается»).

Проверяем (канон «сначала логика + тест, потом UI»):

  * ЯДРО — ``set_material_lots`` / ``set_anchor_recipes`` /
    ``set_weighing_resolution``: валидация имён против ПОЛНОЙ схемы,
    явные отказы (пустой лот, отрицательная доза, половина пары весов),
    очистка ``None``/пусто;
  * ПЕРСИСТЕНТНОСТЬ — round-trip через ``runner_to_state``/
    ``runner_from_state`` и файловый save/load; старый сейв без ключей
    грузится с пустым паспортом;
  * UI-хелперы (чистые) — ``parse_material_lots``/``material_lots_to_text``
    и ``parse_anchor_recipes``/``anchor_recipes_to_text``: round-trip,
    ошибки СИНТАКСИСА с номером строки (правила имён живут в сеттерах —
    в UI не дублируются, канон iter52);
  * ПАСПОРТ/ПРЕФИЛЛ — ``campaign_passport_dataframe`` несёт строки
    «порядок групп (group_order)» / «лоты сырья» / «anchor-рецепты (phr)» /
    «разрешение весов (δ)» («—» когда не задано; δ = шаг/гpp);
    ``setup_prefill_from_runner`` отдаёт ключи формы паспорта.
"""
from types import SimpleNamespace

import pytest

from src.apps import campaign_state as cst
from src.apps import campaign_ui as ui
from src.apps.campaign_ui import build_setup_runner

LOTS = {"A": "L-2408-17", "B": "B-77", "C": "партия 3"}
ANCHORS = {"anchor_main": {"A": 70.0, "B": 30.0, "C": 10.0},
           "edge": {"A": 55.0, "C": 2.5}}

LOTS_TEXT = "A: L-2408-17\nB: B-77\nC: партия 3"
ANCHORS_TEXT = ("anchor_main: A=70, B=30, C=10\n"
                "edge: A=55, C=2.5")


def _runner():
    """Раннер {A,B,C}×{T} без точек (паспорт задаётся ДО первого замера)."""
    return build_setup_runner(
        mixture_names=["A", "B", "C"], process_names=["T"],
        process_lower=[0.0], process_upper=[1.0],
        response_names=["strength"], seed=1)


def _passport_runner():
    """Раннер с ПОЛНЫМ паспортом P2.3."""
    r = _runner()
    r.set_material_lots(LOTS)
    r.set_anchor_recipes(ANCHORS)
    r.set_weighing_resolution(0.1, 5.0)
    return r


# ======================================================================
# 1. ЯДРО: set_material_lots
# ======================================================================
class TestSetMaterialLots:

    def test_valid_lots_stored_normalized(self):
        r = _runner()
        r.set_material_lots({"A": "  L-1  ", "B": "B-77"})
        assert r.material_lots == {"A": "L-1", "B": "B-77"}

    def test_unknown_component_raises(self):
        with pytest.raises(KeyError, match="не найден"):
            _runner().set_material_lots({"X": "L-1"})

    def test_empty_lot_raises(self):
        """Тихое «без лота» недопустимо (A0.6) — пустая строка = ошибка."""
        with pytest.raises(ValueError, match="Пустое обозначение лота"):
            _runner().set_material_lots({"A": "   "})

    def test_none_clears(self):
        r = _passport_runner()
        r.set_material_lots(None)
        assert r.material_lots == {}


# ======================================================================
# 2. ЯДРО: set_anchor_recipes
# ======================================================================
class TestSetAnchorRecipes:

    def test_valid_recipes_stored(self):
        r = _runner()
        r.set_anchor_recipes(ANCHORS)
        assert r.anchor_recipes == ANCHORS

    def test_subset_of_components_allowed(self):
        """Anchor может нести ПОДмножество компонентов (edge-рецепт)."""
        r = _runner()
        r.set_anchor_recipes({"edge": {"A": 55.0}})
        assert r.anchor_recipes["edge"] == {"A": 55.0}

    def test_unknown_component_raises(self):
        with pytest.raises(KeyError, match="не найден"):
            _runner().set_anchor_recipes({"m": {"X": 1.0}})

    def test_negative_dose_raises(self):
        with pytest.raises(ValueError, match="≥ 0"):
            _runner().set_anchor_recipes({"m": {"A": -1.0}})

    def test_nan_dose_raises(self):
        with pytest.raises(ValueError, match="конечной"):
            _runner().set_anchor_recipes({"m": {"A": float("nan")}})

    def test_non_numeric_dose_raises(self):
        with pytest.raises(ValueError, match="не число"):
            _runner().set_anchor_recipes({"m": {"A": "abc"}})

    def test_empty_recipe_raises(self):
        with pytest.raises(ValueError, match="пуст"):
            _runner().set_anchor_recipes({"m": {}})

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="Пустое имя"):
            _runner().set_anchor_recipes({"  ": {"A": 1.0}})

    def test_none_clears(self):
        r = _passport_runner()
        r.set_anchor_recipes(None)
        assert r.anchor_recipes == {}


# ======================================================================
# 3. ЯДРО: set_weighing_resolution
# ======================================================================
class TestSetWeighingResolution:

    def test_valid_pair_stored(self):
        r = _runner()
        r.set_weighing_resolution(0.1, 5.0)
        assert r.weighing_step_g == pytest.approx(0.1)
        assert r.grams_per_phr == pytest.approx(5.0)

    def test_both_zero_means_unset(self):
        r = _passport_runner()
        r.set_weighing_resolution(0.0, 0.0)
        assert r.weighing_step_g == 0.0 and r.grams_per_phr == 0.0

    def test_half_pair_raises(self):
        """Одно поле без другого δ не определяет — явная ошибка (A0.6),
        а не тихое «слой выключен» с потерей половины паспорта."""
        with pytest.raises(ValueError, match="ПАРОЙ"):
            _runner().set_weighing_resolution(0.1, 0.0)
        with pytest.raises(ValueError, match="ПАРОЙ"):
            _runner().set_weighing_resolution(0.0, 5.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="отрицательными"):
            _runner().set_weighing_resolution(-0.1, 5.0)

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="конечными"):
            _runner().set_weighing_resolution(float("inf"), 5.0)


# ======================================================================
# 4. Персистентность (campaign_state)
# ======================================================================
class TestPassportPersistence:

    def test_state_roundtrip(self):
        r1 = cst.runner_from_state(cst.runner_to_state(_passport_runner()))
        assert r1.material_lots == LOTS
        assert r1.anchor_recipes == ANCHORS
        assert r1.weighing_step_g == pytest.approx(0.1)
        assert r1.grams_per_phr == pytest.approx(5.0)

    def test_file_save_load_roundtrip(self, tmp_path):
        cst.save_campaign(_passport_runner(), str(tmp_path), "pass")
        r1 = cst.load_campaign(str(tmp_path), "pass")
        assert r1.material_lots == LOTS
        assert r1.anchor_recipes == ANCHORS
        assert r1.weighing_step_g == pytest.approx(0.1)
        assert r1.grams_per_phr == pytest.approx(5.0)

    def test_old_save_without_keys_loads_empty(self):
        state = cst.runner_to_state(_passport_runner())
        for key in ("material_lots", "anchor_recipes",
                    "weighing_step_g", "grams_per_phr"):
            state["runner"].pop(key)                    # сейв «до P2.3»
        r1 = cst.runner_from_state(state)
        assert r1.material_lots == {}
        assert r1.anchor_recipes == {}
        assert r1.weighing_step_g == 0.0
        assert r1.grams_per_phr == 0.0

    def test_empty_passport_roundtrips_empty(self):
        r1 = cst.runner_from_state(cst.runner_to_state(_runner()))
        assert r1.material_lots == {}
        assert r1.anchor_recipes == {}
        assert r1.weighing_step_g == 0.0 and r1.grams_per_phr == 0.0

    def test_state_keys_are_json_native(self):
        st = cst.runner_to_state(_passport_runner())["runner"]
        assert st["material_lots"] == LOTS
        assert st["anchor_recipes"] == ANCHORS
        assert isinstance(st["weighing_step_g"], float)
        assert isinstance(st["grams_per_phr"], float)


# ======================================================================
# 5. UI-хелперы: parse_material_lots / material_lots_to_text
# ======================================================================
class TestMaterialLotsText:

    def test_parse_basic(self):
        assert ui.parse_material_lots(LOTS_TEXT) == LOTS

    def test_roundtrip(self):
        txt = ui.material_lots_to_text(LOTS)
        assert ui.parse_material_lots(txt) == LOTS

    def test_lot_may_contain_colon(self):
        """Разделитель — ПЕРВОЕ «:»; лот может содержать двоеточия."""
        out = ui.parse_material_lots("A: лот: спец")
        assert out == {"A": "лот: спец"}

    def test_empty_text_gives_empty(self):
        assert ui.parse_material_lots("") == {}
        assert ui.parse_material_lots("\n  \n") == {}
        assert ui.material_lots_to_text({}) == ""

    def test_no_colon_raises_with_line(self):
        with pytest.raises(ValueError, match="Строка 2"):
            ui.parse_material_lots("A: L-1\nB L-2")

    def test_empty_lot_raises(self):
        with pytest.raises(ValueError, match="пустое имя компонента или лота"):
            ui.parse_material_lots("A:   ")

    def test_duplicate_component_raises(self):
        with pytest.raises(ValueError, match="дважды"):
            ui.parse_material_lots("A: L-1\nA: L-2")


# ======================================================================
# 6. UI-хелперы: parse_anchor_recipes / anchor_recipes_to_text
# ======================================================================
class TestAnchorRecipesText:

    def test_parse_basic(self):
        assert ui.parse_anchor_recipes(ANCHORS_TEXT) == ANCHORS

    def test_roundtrip(self):
        txt = ui.anchor_recipes_to_text(ANCHORS)
        assert ui.parse_anchor_recipes(txt) == ANCHORS

    def test_reference_anchor_format_parses(self):
        """Формат референсного anchor'а CAMPAIGN_SPEC_PVC §3 разбирается."""
        out = ui.parse_anchor_recipes(
            "anchor_main: PVC_67=70, PVC_71=30, DINP=10, ESO=2.5")
        assert out["anchor_main"]["ESO"] == pytest.approx(2.5)

    def test_empty_text_gives_empty(self):
        assert ui.parse_anchor_recipes("") == {}
        assert ui.anchor_recipes_to_text({}) == ""

    def test_no_colon_raises_with_line(self):
        with pytest.raises(ValueError, match="Строка 1"):
            ui.parse_anchor_recipes("anchor_main A=70")

    def test_pair_without_equals_raises(self):
        with pytest.raises(ValueError, match="комп=phr"):
            ui.parse_anchor_recipes("m: A70")

    def test_bad_float_raises(self):
        with pytest.raises(ValueError, match="не число"):
            ui.parse_anchor_recipes("m: A=x")

    def test_duplicate_recipe_raises(self):
        with pytest.raises(ValueError, match="дважды"):
            ui.parse_anchor_recipes("m: A=1\nm: B=2")

    def test_duplicate_component_raises(self):
        with pytest.raises(ValueError, match="дважды"):
            ui.parse_anchor_recipes("m: A=1, A=2")

    def test_empty_recipe_raises(self):
        with pytest.raises(ValueError, match="пуст"):
            ui.parse_anchor_recipes("m: ")


# ======================================================================
# 7. Паспорт кампании (таблица) + префилл формы
# ======================================================================
class TestPassportAndPrefill:

    def test_passport_rows_present_and_dash_when_unset(self):
        df = ui.campaign_passport_dataframe(_runner()).set_index("параметр")
        for row in ("порядок групп (group_order)", "лоты сырья",
                    "anchor-рецепты (phr)", "разрешение весов (δ)"):
            assert df.loc[row, "значение"] == "—"

    def test_passport_shows_lots_anchors_and_delta(self):
        df = ui.campaign_passport_dataframe(
            _passport_runner()).set_index("параметр")
        lots_val = str(df.loc["лоты сырья", "значение"])
        assert "A: L-2408-17" in lots_val and " ; " in lots_val
        anchors_val = str(df.loc["anchor-рецепты (phr)", "значение"])
        assert "anchor_main (3 комп.)" in anchors_val
        assert "edge (2 комп.)" in anchors_val
        weigh_val = str(df.loc["разрешение весов (δ)", "значение"])
        # golden CAMPAIGN_SPEC_PVC §5: 0.1 г / 5 г·phr⁻¹ → δ = 0.02 phr
        assert "0.02" in weigh_val and "0.1" in weigh_val and "5" in weigh_val

    def test_passport_group_order_from_spec(self):
        """group_order читается ИЗ АКТИВНОЙ спеки (единый источник, iter48) —
        отдельного поля в раннере нет: дубль разошёлся бы с отпечатком."""
        spec_stub = SimpleNamespace(
            q=19, dim_z=16, group_order=["FILLER.total", "SOFT.total"],
            spec_hash=lambda: "ab" * 32)
        stub = SimpleNamespace(
            phr_spec=spec_stub, campaign_label="", preflight_pairs=[],
            process_levels={}, material_lots={}, anchor_recipes={},
            weighing_step_g=0.0, grams_per_phr=0.0)
        df = ui.campaign_passport_dataframe(stub).set_index("параметр")
        assert (df.loc["порядок групп (group_order)", "значение"]
                == "FILLER.total → SOFT.total")

    def test_prefill_contains_passport_keys(self):
        out = ui.setup_prefill_from_runner(_passport_runner())
        assert ui.parse_material_lots(out["setup_material_lots"]) == LOTS
        assert ui.parse_anchor_recipes(out["setup_anchor_recipes"]) == ANCHORS
        assert out["setup_pass_weigh_step"] == pytest.approx(0.1)
        assert out["setup_pass_weigh_gpp"] == pytest.approx(5.0)

    def test_prefill_empty_without_passport(self):
        out = ui.setup_prefill_from_runner(_runner())
        assert out["setup_material_lots"] == ""
        assert out["setup_anchor_recipes"] == ""
        assert out["setup_pass_weigh_step"] == 0.0
        assert out["setup_pass_weigh_gpp"] == 0.0

    def test_prefill_matches_after_file_roundtrip(self, tmp_path):
        r0 = _passport_runner()
        cst.save_campaign(r0, str(tmp_path), "pp")
        r1 = cst.load_campaign(str(tmp_path), "pp")
        assert (ui.setup_prefill_from_runner(r1)
                == ui.setup_prefill_from_runner(r0))