# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 50 — шаг P1.3 ревизии UI: ВИДИМОСТЬ спеки и границ точки.

Матрица покрытия (UI_REVISION_SPEC, «Приоритеты доработки») фиксировала две
дыры интерфейса:

  * ``role`` / ``min_phr`` / ``max_phr`` / ``scale`` / ``group_order`` — часть
    ГЕОМЕТРИИ и часть ``spec_hash`` (iter45–48), но в UI не показывались:
    сверить «CPE ≥ 3 phr», «TiO₂ по логу» или приоритет групп было негде;
  * условные границы §4 (немонотонная ``hi_φ(T)``) — ``point_report``
    (iter49/B7) в UI не звался НИ РАЗУ: «почему план не даёт такую точку»
    оставалось без ответа.

Проверяем:

  * ядро — ``PhrSpec.role_of``: роль выводится из СТРУКТУРЫ и совпадает с
    ролью в сериализации ``to_dicts`` (единый источник; хеши не «уехали»);
  * ``phr_spec_summary_dataframe`` — новые колонки и их значения
    (роль/min_phr/max_phr/scale), NaN у незаданных лимитов;
  * ``phr_spec_policy_caption`` — group_order / лог-оси / техлимиты / хеш;
  * ``point_bounds_dataframe`` — согласованность с ``point_report``, метки
    active (golden iter45: partners @T=5, range на полке, max_phr @T=15),
    пометка «✗» для точки вне геометрии (A0.6 — не исключение);
  * ``point_bounds_caption`` — глоссарий ТОЛЬКО встретившихся меток + счётчик.
"""
import numpy as np
import pytest

from src.apps import campaign_ui as ui
from src.design.phr_sampler import PhrSpec

# Референсная v2-спека (та же геометрия, что golden iter49/B7): все роли,
# лог-оси, cap, техлимиты SOFT-группы (golden iter45).
NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "ESO", "role": "FIXED", "value": 2.5},
    {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["PBNK", "CPE"]},
    {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
     "share_range": [0.0, 0.70], "max_phr": 8.0},
    {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT",
     "min_phr": 3.0},
    {"name": "STAB", "role": "GROUP_TOTAL", "range": [3.5, 5.0],
     "members": ["PF_LB", "PF"]},
    {"name": "PF_LB", "role": "SHARE_FREE", "group": "STAB",
     "share_range": [0.0, 0.40]},
    {"name": "PF", "role": "SHARE_CLOSURE", "group": "STAB"},
    {"name": "SBM", "role": "RATIO_TO", "reference": "STAB",
     "range": [0.02, 0.09]},
    {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0],
     "scale": "log"},
    {"name": "UV", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
     "scale": "log", "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
]

ROLES = {d["name"]: d["role"] for d in NODES}


def _spec(group_order=None) -> PhrSpec:
    if group_order is None:
        return PhrSpec.from_dicts(NODES)
    return PhrSpec.from_dicts({"spec_version": 2, "group_order": group_order,
                               "nodes": NODES})


def _point(dinp=6.0, t_soft=5.0, phi_pbnk=0.2, t_stab=4.0, phi_pflb=0.25,
           r_sbm=0.05, tio2=1.0, uv=0.10):
    """Рецепт в phr, порядок = component_names."""
    return [100.0, dinp, 2.5,
            phi_pbnk * t_soft, (1.0 - phi_pbnk) * t_soft,
            phi_pflb * t_stab, (1.0 - phi_pflb) * t_stab,
            r_sbm * t_stab, tio2, uv]


def _fractions(**kw):
    """Тот же рецепт в ДОЛЯХ — вход UI-хелперов (движок отдаёт доли)."""
    spec = _spec()
    return spec.to_fractions(np.asarray(_point(**kw), float))


# ======================================================================
# 1. Ядро: role_of — единый источник ролей
# ======================================================================
class TestRoleOf:

    def test_roles_match_declared_schema(self):
        """Роль выводится из СТРУКТУРЫ и совпадает с объявленной в JSON."""
        spec = _spec()
        for name, role in ROLES.items():
            assert spec.role_of(name) == role

    def test_serialization_uses_same_source(self):
        """``to_dicts`` и ``role_of`` не могут разойтись (единый источник)."""
        spec = _spec()
        for d in spec.to_dicts():
            assert d["role"] == spec.role_of(d["name"])

    def test_hash_unchanged_by_refactor(self):
        """Порядок ключей сериализации не менялся ⇒ round-trip сохраняет hash."""
        spec = _spec()
        assert PhrSpec.from_dicts(spec.to_dicts()).spec_hash() \
            == spec.spec_hash()

    def test_legacy_share_of_reported_as_is(self):
        """v1-спека ролей не имеет: share_of отдаётся как SHARE_OF, а не
        подменяется одной из новых ролей (A0.6)."""
        legacy = PhrSpec.from_dicts([
            {"name": "resin", "mode": "fixed", "value": 100.0},
            {"name": "stab", "mode": "absolute", "lo": 2.0, "hi": 5.0},
            {"name": "Ca", "mode": "share_of", "of": "stab",
             "lo": 0.2, "hi": 0.7},
            {"name": "Zn", "mode": "share_of", "of": "stab",
             "lo": 0.3, "hi": 0.8},
        ])
        assert legacy.role_of("Ca") == "SHARE_OF"
        assert legacy.role_of("resin") == "FIXED"
        assert legacy.role_of("stab") == "GROUP_TOTAL"

    def test_unknown_node_raises(self):
        with pytest.raises(ValueError, match="не найден"):
            _spec().role_of("нет-такого")


# ======================================================================
# 2. Сводка спеки: роль / техлимиты / шкала (P1.3)
# ======================================================================
class TestSummaryDataframe:

    def test_columns_and_roles(self):
        df = ui.phr_spec_summary_dataframe(_spec())
        assert list(df.columns) == [
            "узел", "роль", "режим", "lo", "hi", "ref", "cap_to", "cap_ratio",
            "min_phr", "max_phr", "scale", "компонент смеси"]
        assert dict(zip(df["узел"], df["роль"])) == ROLES

    def test_limits_and_scale_visible(self):
        df = ui.phr_spec_summary_dataframe(_spec()).set_index("узел")
        assert df.loc["PBNK", "max_phr"] == pytest.approx(8.0)
        assert np.isnan(df.loc["PBNK", "min_phr"])
        assert df.loc["CPE", "min_phr"] == pytest.approx(3.0)
        assert df.loc["TiO2", "scale"] == "log"
        assert df.loc["DINP", "scale"] == "linear"
        # у узлов без лимитов — NaN, а не 0.0 (нуль выглядел бы значением)
        assert np.isnan(df.loc["DINP", "min_phr"])
        assert np.isnan(df.loc["DINP", "max_phr"])

    def test_closure_shows_derived_range(self):
        """Диапазон closure ПРОИЗВОДНЫЙ (iter46/B2), не сентинель (0, 0)."""
        df = ui.phr_spec_summary_dataframe(_spec()).set_index("узел")
        assert (df.loc["PF", "lo"], df.loc["PF", "hi"]) == \
            pytest.approx((0.60, 1.0))


# ======================================================================
# 3. Подпись политики спеки (group_order / log / лимиты / hash)
# ======================================================================
class TestPolicyCaption:

    def test_mentions_geometry_policy(self):
        spec = _spec(group_order=["SOFT", "STAB"])
        txt = ui.phr_spec_policy_caption(spec)
        assert "SOFT → STAB" in txt              # приоритет групп
        assert "TiO2" in txt and "UV" in txt     # лог-оси
        assert "PBNK" in txt and "CPE" in txt    # техлимиты
        assert spec.spec_hash()[:12] in txt
        assert "схема v2" in txt

    def test_absent_policy_said_explicitly(self):
        """Без group_order/лог-осей/лимитов — «не задан»/«нет», а не пусто."""
        spec = PhrSpec.from_dicts([
            {"name": "resin", "mode": "fixed", "value": 100.0},
            {"name": "DINP", "mode": "absolute", "lo": 4.0, "hi": 14.0},
        ])
        txt = ui.phr_spec_policy_caption(spec)
        assert "не задан" in txt
        assert "лог-оси (сэмплинг по ln phr): нет" in txt
        assert "техлимиты min/max_phr: нет" in txt


# ======================================================================
# 4. Эффективные границы точки (поверх контракта iter49/B7)
# ======================================================================
class TestPointBounds:

    def test_matches_point_report(self):
        """UI ничего не пересчитывает: значения = контракт ядра."""
        spec = _spec()
        x = _fractions()
        df = ui.point_bounds_dataframe(spec, x).set_index("узел")
        rep = spec.point_report(spec.fractions_to_phr(np.asarray(x, float)))
        assert list(df.index) == [d["name"] for d in NODES]
        for nm, b in rep.effective_bounds.items():
            assert df.loc[nm, "lo"] == pytest.approx(b.lo, abs=1e-6)
            assert df.loc[nm, "hi"] == pytest.approx(b.hi, abs=1e-6)
            assert df.loc[nm, "активна lo"] == b.active_lo
            assert df.loc[nm, "активна hi"] == b.active_hi
        assert set(df["в границах"]) == {"✓"}

    def test_coordinate_kind_is_human_readable(self):
        df = ui.point_bounds_dataframe(_spec(), _fractions()).set_index("узел")
        assert df.loc["DINP", "координата"] == "phr"
        assert df.loc["PBNK", "координата"] == "доля"
        assert df.loc["SBM", "координата"] == "коэффициент"

    def test_active_labels_golden_nonmonotonic(self):
        """Golden iter45 ``hi(T) = min(0.70, 8/T, 1 − 3/T)``: активная метка
        верхней границы PBNK зависит от тотала НЕМОНОТОННО."""
        spec = _spec()
        lab = {}
        for t_soft, phi in ((5.0, 0.2), (10.5, 0.5), (15.0, 0.4)):
            df = ui.point_bounds_dataframe(
                spec, _fractions(t_soft=t_soft, phi_pbnk=phi)).set_index("узел")
            lab[t_soft] = df.loc["PBNK", "активна hi"]
        assert lab[5.0] == "partners"     # 1 − 3/5 = 0.40 — давят партнёры
        assert lab[10.5] == "range"       # полка 0.70 — собственный share_range
        assert lab[15.0] == "max_phr"     # 8/15 = 0.5333 — складской лимит

    def test_cap_label_depends_on_point(self):
        """Потолок УФ считается ПО ТОЧКЕ: при малой фазе он активен."""
        df = ui.point_bounds_dataframe(
            _spec(), _fractions(dinp=6.0)).set_index("узел")
        assert df.loc["UV", "активна hi"] == "cap"
        assert df.loc["UV", "hi"] == pytest.approx(0.03 * (6.0 + 2.5))

    def test_out_of_geometry_marked_not_raised(self):
        """Точка вне геометрии — «✗» в таблице (A0.6), не исключение."""
        spec = _spec()
        # CPE = 0.3·5 = 1.5 phr < min_phr = 3.0 → нарушение техминимума
        df = ui.point_bounds_dataframe(
            spec, _fractions(t_soft=5.0, phi_pbnk=0.7)).set_index("узел")
        assert df.loc["CPE", "в границах"] == "✗"

    def test_delta_phr_optional(self):
        """δ не обязателен: границы — про геометрию, не про весы."""
        spec = _spec()
        x = _fractions()
        a = ui.point_bounds_dataframe(spec, x)
        b = ui.point_bounds_dataframe(spec, x, delta_phr=0.02)
        assert list(a["lo"]) == list(b["lo"])
        assert list(a["активна hi"]) == list(b["активна hi"])


# ======================================================================
# 5. Подпись под таблицей границ
# ======================================================================
class TestPointBoundsCaption:

    def test_glossary_only_for_present_labels(self):
        df = ui.point_bounds_dataframe(_spec(), _fractions())
        txt = ui.point_bounds_caption(df)
        present = set(df["активна lo"]) | set(df["активна hi"])
        for lab in present:
            assert ui.ACTIVE_LABEL_RU[lab] in txt
        for lab in set(ui.ACTIVE_LABEL_RU) - present:
            assert ui.ACTIVE_LABEL_RU[lab] not in txt
        assert "все узлы внутри эффективных границ" in txt

    def test_counts_violations(self):
        df = ui.point_bounds_dataframe(
            _spec(), _fractions(t_soft=5.0, phi_pbnk=0.7))
        txt = ui.point_bounds_caption(df)
        assert "вне границ" in txt
        assert "не блокируется" in txt

    def test_empty_frame_is_explicit(self):
        import pandas as pd
        assert "пуста" in ui.point_bounds_caption(pd.DataFrame())
