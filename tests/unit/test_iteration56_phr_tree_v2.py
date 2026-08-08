# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 56 — P3.2 (UI_REVISION_SPEC): v2-редактор дерева phr-спеки.

До iter56 иерархический редактор («дерево») умел только legacy-схему v1
(``share_of``), и ЕДИНСТВЕННЫМ каналом ввода спеки кампании «pvc_edge_v1»
оставался JSON. Этот шаг учит чистый слой дерева схеме v2 (роли iter46/B6):

  * ``phr_tree_to_dicts(tree, schema_version=2, group_order=…)`` — эмиссия
    role-словарей: k=2 → SHARE_FREE + SHARE_CLOSURE (кто closure — пометка
    ребёнка), k≥3 → SHARE_SIMPLEX; техлимиты ``min_phr``/``max_phr``;
    ``scale='log'``; обёртка ``{"spec_version": 2, "group_order": …}``
    при непустом порядке групп (iter48/B4 — входит в spec_hash);
  * ``phr_tree_from_spec`` проецирует v2-спеку в дерево; НЕПРОЕЦИРУЕМЫЙ
    порядок узлов (члены группы не сразу за тоталом) — явная ошибка,
    а не тихая смена ``spec_hash`` на round-trip (A0.6);
  * ``validate_phr_tree(schema_version=…)``: v1 отвергает v2-фичи явной
    ошибкой (молча выброшенное поле — потерянное намерение), v2 ловит
    доли L/U у closure и min > max; состав ролей k=2/k≥3 НЕ дублируется —
    его валидирует конструктор PhrSpec (канон iter52);
  * таблица детей группы: в v2 добавляются колонки min phr / max phr /
    closure (NaN = лимита нет); колонки v1 не тронуты (контракт iter41).
"""
import numpy as np
import pandas as pd
import pytest

from src.apps import campaign_ui as ui
from src.apps.campaign_ui import build_setup_runner
from src.design.phr_sampler import PhrSpec

from tests.unit.test_iteration46_phr_roles import (PVC_V2, PVC_V2_NODES,
                                                   SMALL_V2)

GO_PVC = ["FILLER.total", "SOFT.total", "ACR.total", "LUB.total",
          "STAB.total"]


def _pvc(group_order=None):
    if group_order:
        return PhrSpec.from_dicts({"spec_version": 2,
                                   "group_order": list(group_order),
                                   "nodes": PVC_V2_NODES})
    return PhrSpec.from_dicts(PVC_V2)


def _small_spec(**node_patches):
    out = []
    for d in SMALL_V2:
        d2 = dict(d)
        if d2["name"] in node_patches:
            d2.update(node_patches[d2["name"]])
        out.append(d2)
    return PhrSpec.from_dicts(out)


# ======================================================================
# 1. Проекция спеки v2 в дерево и round-trip (hash бит-в-бит)
# ======================================================================
class TestTreeFromSpecV2:

    def test_pvc_roundtrip_preserves_hash_and_dicts(self):
        spec = _pvc()
        tree = ui.phr_tree_from_spec(spec)
        dicts = ui.phr_tree_to_dicts(tree, schema_version=2)
        # содержимое совпадает с канонической сериализацией _to_role_dicts
        assert dicts == spec.to_dicts()
        rebuilt = PhrSpec.from_dicts(dicts)
        assert rebuilt.spec_hash() == spec.spec_hash()
        assert rebuilt.dim_z == 16

    def test_closure_flag_and_limits_projected(self):
        spec = _small_spec(Y={"min_phr": 5.0})
        tree = ui.phr_tree_from_spec(spec)
        grp = next(b for b in tree if b["name"] == "G")
        kids = {c["name"]: c for c in grp["children"]}
        assert kids["Y"]["closure"] is True
        assert kids["X"]["closure"] is False
        assert kids["Y"]["min_phr"] == pytest.approx(5.0)
        assert kids["Y"]["max_phr"] is None
        # closure проецируется с сентинелем 0/0 (диапазон производный)
        assert kids["Y"]["lo"] == 0.0 and kids["Y"]["hi"] == 0.0
        # round-trip лимитов сохраняет hash
        rebuilt = PhrSpec.from_dicts(
            ui.phr_tree_to_dicts(tree, schema_version=2))
        assert rebuilt.spec_hash() == spec.spec_hash()

    def test_scale_log_projected_and_roundtrips(self):
        spec = PhrSpec.from_dicts([
            {"name": "base", "role": "FIXED", "value": 100.0},
            {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0],
             "scale": "log"},
        ])
        tree = ui.phr_tree_from_spec(spec)
        tio2 = next(b for b in tree if b["name"] == "TiO2")
        assert tio2["scale"] == "log"
        rebuilt = PhrSpec.from_dicts(
            ui.phr_tree_to_dicts(tree, schema_version=2))
        assert rebuilt.spec_hash() == spec.spec_hash()

    def test_capped_node_projected(self):
        spec = _pvc()
        tree = ui.phr_tree_from_spec(spec)
        uv = next(b for b in tree if b["name"] == "UV_CSFCP")
        assert list(uv["cap_to"]) == ["DINP", "ESO"]
        assert uv["cap_ratio"] == pytest.approx(0.03)

    def test_non_contiguous_members_rejected(self):
        # члены группы разделены посторонним узлом — проекция изменила бы
        # порядок узлов (и spec_hash) молча; вместо этого явная ошибка
        nodes = [
            {"name": "base", "role": "FIXED", "value": 100.0},
            {"name": "G", "role": "GROUP_TOTAL", "range": [10.0, 20.0],
             "members": ["X", "Y"]},
            {"name": "X", "role": "SHARE_FREE", "group": "G",
             "share_range": [0.2, 0.8]},
            {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
            {"name": "Y", "role": "SHARE_CLOSURE", "group": "G"},
        ]
        spec = PhrSpec.from_dicts(nodes)
        with pytest.raises(ValueError, match="не сразу за тоталом"):
            ui.phr_tree_from_spec(spec)

    def test_group_order_roundtrip(self):
        spec = _pvc(GO_PVC)
        tree = ui.phr_tree_from_spec(spec)
        d = ui.phr_tree_to_dicts(tree, schema_version=2,
                                 group_order=spec.group_order)
        assert isinstance(d, dict)
        assert d["group_order"] == GO_PVC
        rebuilt = PhrSpec.from_dicts(d)
        assert rebuilt.spec_hash() == spec.spec_hash()
        # без group_order — другой отпечаток (iter48/B4)
        plain = PhrSpec.from_dicts(
            ui.phr_tree_to_dicts(tree, schema_version=2))
        assert plain.spec_hash() != spec.spec_hash()


# ======================================================================
# 2. Эмиссия дерева в схему v2 (роли из структуры + пометки closure)
# ======================================================================
class TestTreeToDictsV2:

    def _tree_k2(self):
        return [
            ui.phr_single_block("base", mode="fixed", value=100.0),
            ui.phr_group_block("G", lo=10.0, hi=20.0, children=[
                {"name": "X", "lo": 0.2, "hi": 0.8},
                {"name": "Y", "lo": 0.0, "hi": 0.0, "closure": True},
            ]),
        ]

    def test_k2_roles_from_closure_flag(self):
        nodes = ui.phr_tree_to_dicts(self._tree_k2(), schema_version=2)
        by = {d["name"]: d for d in nodes}
        assert by["base"]["role"] == "FIXED"
        assert by["G"]["role"] == "GROUP_TOTAL"
        assert by["G"]["members"] == ["X", "Y"]
        assert by["X"]["role"] == "SHARE_FREE"
        assert by["X"]["share_range"] == [0.2, 0.8]
        assert by["Y"]["role"] == "SHARE_CLOSURE"
        assert "share_range" not in by["Y"]
        # собирается ядром
        assert PhrSpec.from_dicts(nodes).schema_version == 2

    def test_k3_all_simplex(self):
        tree = [
            ui.phr_single_block("base", mode="fixed", value=100.0),
            ui.phr_group_block("L", lo=1.0, hi=2.0, children=[
                {"name": "a", "lo": 0.3, "hi": 0.7},
                {"name": "b", "lo": 0.1, "hi": 0.6},
                {"name": "c", "lo": 0.1, "hi": 0.6},
            ]),
        ]
        nodes = ui.phr_tree_to_dicts(tree, schema_version=2)
        roles = {d["name"]: d["role"] for d in nodes}
        assert roles["a"] == roles["b"] == roles["c"] == "SHARE_SIMPLEX"
        assert PhrSpec.from_dicts(nodes).dim_z == 3   # тотал + 2 своб. доли

    def test_k2_without_closure_surfaces_core_error(self):
        # состав ролей НЕ дублируется в дереве: две SHARE_FREE эмитятся
        # как есть, а конструктор ядра даёт внятную ошибку (канон iter52)
        tree = [
            ui.phr_single_block("base", mode="fixed", value=100.0),
            ui.phr_group_block("G", lo=10.0, hi=20.0, children=[
                {"name": "X", "lo": 0.2, "hi": 0.8},
                {"name": "Y", "lo": 0.2, "hi": 0.8},
            ]),
        ]
        nodes = ui.phr_tree_to_dicts(tree, schema_version=2)
        with pytest.raises(ValueError, match="РОВНО один"):
            PhrSpec.from_dicts(nodes)

    def test_min_max_phr_carried(self):
        tree = self._tree_k2()
        tree[1]["children"][1]["min_phr"] = 5.0
        tree[1]["children"][0]["max_phr"] = 12.0
        nodes = ui.phr_tree_to_dicts(tree, schema_version=2)
        by = {d["name"]: d for d in nodes}
        assert by["Y"]["min_phr"] == 5.0
        assert by["X"]["max_phr"] == 12.0
        assert "min_phr" not in by["X"]      # не задан — ключа нет

    def test_capped_single_emits_list_and_scale(self):
        tree = [
            ui.phr_single_block("base", mode="fixed", value=100.0),
            ui.phr_single_block("DINP", mode="absolute", lo=4.0, hi=14.0),
            ui.phr_single_block("UV", mode="absolute", lo=0.05, hi=0.30,
                                cap_to="DINP", cap_ratio=0.03),
            ui.phr_single_block("TiO2", mode="absolute", lo=0.3, hi=8.0,
                                scale="log"),
        ]
        nodes = ui.phr_tree_to_dicts(tree, schema_version=2)
        by = {d["name"]: d for d in nodes}
        assert by["UV"]["role"] == "ABSOLUTE_CAPPED"
        assert by["UV"]["cap_to"] == ["DINP"]      # v2: всегда СПИСОК
        assert by["TiO2"]["scale"] == "log"
        assert "scale" not in by["DINP"]           # linear — ключа нет
        assert PhrSpec.from_dicts(nodes).schema_version == 2

    def test_v1_group_order_rejected(self):
        tree = [ui.phr_single_block("a", mode="fixed", value=1.0),
                ui.phr_single_block("b", mode="absolute", lo=1.0, hi=2.0)]
        with pytest.raises(ValueError, match="только схемой v2"):
            ui.phr_tree_to_dicts(tree, group_order=["G"])

    def test_v1_default_output_unchanged(self):
        # контракт iter41: дефолтный вызов эмитит legacy-словарь бит-в-бит
        tree = [ui.phr_group_block(
            "FILLER.total", lo=5.0, hi=25.0,
            children=[{"name": "Chalk_95T", "lo": 0.3, "hi": 1.0},
                      {"name": "Chalk_1T", "lo": 0.0, "hi": 0.7}])]
        assert ui.phr_tree_to_dicts(tree) == [
            {"name": "FILLER.total", "mode": "absolute", "lo": 5.0,
             "hi": 25.0},
            {"name": "Chalk_95T", "mode": "share_of", "of": "FILLER.total",
             "lo": 0.3, "hi": 1.0},
            {"name": "Chalk_1T", "mode": "share_of", "of": "FILLER.total",
             "lo": 0.0, "hi": 0.7},
        ]


# ======================================================================
# 3. Валидация дерева по схемам (A0.6 — ничего не выбрасываем молча)
# ======================================================================
class TestValidateBySchema:

    def _tree_with_child(self, **child_extra):
        ch = {"name": "X", "lo": 0.2, "hi": 0.8}
        ch.update(child_extra)
        return [ui.phr_group_block("G", lo=10.0, hi=20.0, children=[
            ch, {"name": "Y", "lo": 0.0, "hi": 0.0, "closure": True}])]

    def test_v1_rejects_closure_flag(self):
        tree = [ui.phr_group_block("G", lo=10.0, hi=20.0, children=[
            {"name": "X", "lo": 0.2, "hi": 0.8},
            {"name": "Y", "lo": 0.0, "hi": 0.0, "closure": True}])]
        with pytest.raises(ValueError, match="только в схеме v2"):
            ui.phr_tree_to_dicts(tree)                # schema_version=1

    def test_v1_rejects_min_phr(self):
        tree = [ui.phr_group_block("G", lo=10.0, hi=20.0, children=[
            {"name": "X", "lo": 0.2, "hi": 0.8, "min_phr": 3.0},
            {"name": "Y", "lo": 0.2, "hi": 0.8}])]
        with pytest.raises(ValueError, match="только в схеме v2"):
            ui.phr_tree_to_dicts(tree)

    def test_v1_rejects_scale_log(self):
        tree = [ui.phr_single_block("TiO2", mode="absolute", lo=0.3,
                                    hi=8.0, scale="log")]
        with pytest.raises(ValueError, match="только в схеме v2"):
            ui.phr_tree_to_dicts(tree)

    def test_v2_closure_with_nonzero_bounds_rejected(self):
        tree = [ui.phr_group_block("G", lo=10.0, hi=20.0, children=[
            {"name": "X", "lo": 0.2, "hi": 0.8},
            {"name": "Y", "lo": 0.1, "hi": 0.9, "closure": True}])]
        with pytest.raises(ValueError, match="ПРОИЗВОДНЫЙ"):
            ui.phr_tree_to_dicts(tree, schema_version=2)

    def test_v2_min_gt_max_rejected(self):
        tree = self._tree_with_child(min_phr=10.0, max_phr=5.0)
        with pytest.raises(ValueError, match="min phr больше max phr"):
            ui.phr_tree_to_dicts(tree, schema_version=2)

    def test_v2_scale_log_on_ratio_rejected(self):
        tree = [ui.phr_single_block("base", mode="fixed", value=100.0),
                ui.phr_single_block("r", mode="ratio_to", lo=0.1, hi=0.2,
                                    ref="base", scale="log")]
        with pytest.raises(ValueError, match="только для"):
            ui.phr_tree_to_dicts(tree, schema_version=2)

    def test_unknown_scale_rejected(self):
        tree = [ui.phr_single_block("a", mode="absolute", lo=1.0, hi=2.0,
                                    scale="log10")]
        with pytest.raises(ValueError, match="неизвестная шкала"):
            ui.phr_tree_to_dicts(tree, schema_version=2)


# ======================================================================
# 4. Таблица детей группы (v2-колонки; v1-контракт не тронут)
# ======================================================================
class TestChildrenDataframeV2:

    KIDS = [{"name": "X", "lo": 0.2, "hi": 0.8, "min_phr": None,
             "max_phr": 12.0, "closure": False},
            {"name": "Y", "lo": 0.0, "hi": 0.0, "min_phr": 5.0,
             "max_phr": None, "closure": True}]

    def test_v2_columns_and_roundtrip(self):
        blk = ui.phr_group_block("G", lo=10.0, hi=20.0, children=self.KIDS)
        df = ui.phr_children_dataframe(blk, schema_version=2)
        assert list(df.columns) == ["компонент", "доля L", "доля U",
                                    "min phr", "max phr", "closure"]
        assert np.isnan(df.iloc[0]["min phr"])       # NaN, не 0 (канон iter50)
        assert ui.phr_children_from_dataframe(df, schema_version=2) \
            == self.KIDS

    def test_v1_columns_unchanged(self):
        blk = ui.phr_group_block("G", lo=1.0, hi=2.0, children=[
            {"name": "a", "lo": 0.1, "hi": 0.9}])
        df = ui.phr_children_dataframe(blk)
        assert list(df.columns) == ["компонент", "доля L", "доля U"]
        assert ui.phr_children_from_dataframe(df) == [
            {"name": "a", "lo": 0.1, "hi": 0.9}]

    def test_v2_empty_template_row(self):
        df = ui.phr_children_dataframe({"children": []}, schema_version=2)
        assert list(df.columns) == ["компонент", "доля L", "доля U",
                                    "min phr", "max phr", "closure"]
        assert ui.phr_children_from_dataframe(df, schema_version=2) == []

    def test_v2_bad_limit_value_rejected(self):
        df = pd.DataFrame([{"компонент": "X", "доля L": 0.2, "доля U": 0.8,
                            "min phr": "abc", "max phr": np.nan,
                            "closure": False}])
        with pytest.raises(ValueError, match="min/max phr"):
            ui.phr_children_from_dataframe(df, schema_version=2)


# ======================================================================
# 5. Префилл формы сетапа из раннера с v2-спекой (+ group_order)
# ======================================================================
class TestPrefillV2:

    def _runner(self):
        spec = PhrSpec.from_dicts({"spec_version": 2, "group_order": ["G"],
                                   "nodes": SMALL_V2})
        lo, hi = spec.fraction_bounds()
        runner = build_setup_runner(
            mixture_names=list(spec.component_names), process_names=["T"],
            process_lower=[0.0], process_upper=[1.0],
            response_names=["strength"],
            mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=3)
        runner.set_phr_spec(spec)
        return runner

    def test_prefill_tree_schema_and_group_order(self):
        runner = self._runner()
        out = ui.setup_prefill_from_runner(runner)
        assert out["setup_phr_schema"] == ui._PHR_SCHEMA_V2
        assert out["setup_phr_group_order"] == "G"
        # дерево + порядок групп восстанавливают ТОТ ЖЕ отпечаток
        go = ui._parse_names(out["setup_phr_group_order"])
        rebuilt = PhrSpec.from_dicts(ui.phr_tree_to_dicts(
            out["setup_phr_tree"], schema_version=2, group_order=go))
        assert rebuilt.spec_hash() == runner.phr_spec.spec_hash()

    def test_prefill_v1_schema_label(self):
        legacy = PhrSpec.from_dicts([
            {"name": "base", "mode": "fixed", "value": 100.0},
            {"name": "DINP", "mode": "absolute", "lo": 4.0, "hi": 14.0},
        ])
        lo, hi = legacy.fraction_bounds()
        runner = build_setup_runner(
            mixture_names=list(legacy.component_names), process_names=["T"],
            process_lower=[0.0], process_upper=[1.0],
            response_names=["strength"],
            mixture_lower=lo.tolist(), mixture_upper=hi.tolist(), seed=3)
        runner.set_phr_spec(legacy)
        out = ui.setup_prefill_from_runner(runner)
        assert out["setup_phr_schema"] == ui._PHR_SCHEMA_V1
        assert "setup_phr_group_order" not in out
        rebuilt = PhrSpec.from_dicts(
            ui.phr_tree_to_dicts(out["setup_phr_tree"]))
        assert rebuilt.spec_hash() == legacy.spec_hash()