# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 83 — ДОСТИЖИМОСТЬ ОТВЕСА от ВЕСА ЗАМЕСА (UI_REVISION_SPEC iter83).

Что было не так до iter83 (наблюдение сессии 12.08.2026):
  * паспорт кампании просил ``г на 1 phr`` — величину, которой в цехе НЕ
    оперируют. Чтобы её назвать, технолог должен сначала перебрать варианты
    рецептов и выбрать, при каком вводе базового компонента получается нужный
    масштаб; реально так никто не делает;
  * прикладной вопрос звучит иначе: «замес N кг, весы с такой ценой деления —
    возьмётся ли навеска самого малого компонента, где нужен премикс»;
  * маршрут технолога: массовые части → массовые ДОЛИ → граммы через вес
    замеса. ``г/phr`` в нём не нужен вообще — он остаётся ВНУТРЕННИМ
    масштабом ядра для ``δ_phr``.

Проверяемый канон:
  * :func:`batch_sigma_phr` суммирует ЛИСТЬЯ, а не все узлы: узел-тотал
    группы = сумма детей, и сложение всех узлов считало бы группы дважды
    (ровно эта ошибка живёт в ``readonly.get_spec.sigma_phr_static``);
  * :func:`batch_grams_per_phr` переводит вес замеса в масштаб ядра по ВЕРХУ
    Σphr — самая осторожная оценка разрешения (δ крупнее, а не мельче);
  * :func:`batch_weighing_report` считает навеску ЧЕРЕЗ ДОЛИ
    (``fraction_bounds``), и это НЕ то же самое, что через ``g/phr``: для
    fixed-узлов совпадает точно, для варьируемых — расходится, потому что
    Σphr_max недостижим;
  * бокс долей консервативен: расчётная навеска ≤ фактической (проверяется
    сэмплированием спеки) — ошибка в безопасную сторону;
  * вердикты: ``невозможно`` / ``грубо`` / ``премикс`` / ``низ 0`` / ``ок``;
    компонент с нулевым низом — не отказ весов (ложная тревога), а отдельная
    метка;
  * A0.6: хелпер ДИАГНОСТИЧЕН — исключения только на ошибках ДАННЫХ
    (неположительный замес / цена деления), проблемные навески возвращаются
    таблицей и называются ИМЕНАМИ в подписи.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.apps.campaign_ui import (BATCH_MIN_STEPS, BATCH_VERDICT_COARSE,
                                  BATCH_VERDICT_IMPOSSIBLE,
                                  BATCH_VERDICT_OK, BATCH_VERDICT_PREMIX,
                                  BATCH_VERDICT_ZERO, batch_grams_per_phr,
                                  batch_kg_from_grams_per_phr,
                                  batch_sigma_phr, batch_weighing_caption,
                                  batch_weighing_problems,
                                  batch_weighing_report)
from src.design.phr_sampler import PhrSpec, premix_required

#: Референсная v2-геометрия (та же, что в iter42/61/63): fixed-якорь RESIN,
#: группа с техлимитами, log-ось, ABSOLUTE_CAPPED с потолком по (DINP+ESO).
NODES = [
    {"name": "RESIN", "role": "FIXED", "value": 100.0},
    {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
    {"name": "ESO", "role": "FIXED", "value": 2.5},
    {"name": "SOFT", "role": "GROUP_TOTAL", "range": [5.0, 15.0],
     "members": ["PBNK", "CPE"]},
    {"name": "PBNK", "role": "SHARE_FREE", "group": "SOFT",
     "share_range": [0.0, 0.70], "max_phr": 8.0},
    {"name": "CPE", "role": "SHARE_CLOSURE", "group": "SOFT", "min_phr": 3.0},
    {"name": "TiO2", "role": "ABSOLUTE", "range": [0.3, 8.0], "scale": "log"},
    {"name": "UV", "role": "ABSOLUTE_CAPPED", "range": [0.05, 0.30],
     "scale": "log", "cap_to": ["DINP", "ESO"], "cap_ratio": 0.03},
]


def _spec() -> PhrSpec:
    return PhrSpec.from_dicts(NODES)


def _row(df: pd.DataFrame, name: str) -> pd.Series:
    return df.loc[df["компонент"] == name].iloc[0]


# ----------------------------------------------------------------------
# 83.1 Σphr: по ЛИСТЬЯМ, без двойного счёта групп
# ----------------------------------------------------------------------
class TestSigmaPhr:

    def test_golden_leaf_sum(self):
        """Golden (прогон 12.08.2026): 109.85 … 147.80 по листьям."""
        lo, hi = batch_sigma_phr(_spec())
        assert lo == pytest.approx(109.85)
        assert hi == pytest.approx(147.80)

    def test_group_total_is_not_counted_twice(self):
        """Сумма ВСЕХ узлов больше суммы листьев ровно на тотал группы SOFT.

        Это и есть ошибка, которую хелпер не повторяет: SOFT = PBNK + CPE,
        поэтому складывать его вместе с детьми нельзя.
        """
        spec = _spec()
        iv = spec.phr_intervals()
        leaf_lo, leaf_hi = batch_sigma_phr(spec)
        all_lo = sum(v[0] for v in iv.values())
        all_hi = sum(v[1] for v in iv.values())
        assert all_lo - leaf_lo == pytest.approx(iv["SOFT"][0])
        assert all_hi - leaf_hi == pytest.approx(iv["SOFT"][1])
        assert all_hi > leaf_hi          # завышение верха — не безобидно

    def test_sigma_covers_every_sampled_recipe(self):
        """Статический интервал СОДЕРЖИТ Σphr фактических точек спеки."""
        spec = _spec()
        lo, hi = batch_sigma_phr(spec)
        P = np.array([spec.decode(z) for z in spec.sample_z(500, seed=5)])
        S = P.sum(axis=1)
        assert float(S.min()) >= lo - 1e-9
        assert float(S.max()) <= hi + 1e-9


# ----------------------------------------------------------------------
# 83.2 вес замеса → масштаб ядра (г/phr)
# ----------------------------------------------------------------------
class TestGramsPerPhr:

    def test_golden_scale_by_upper_sigma(self):
        """25 кг / Σphr_max 147.8 → 169.1475 г на 1 phr (golden)."""
        assert batch_grams_per_phr(_spec(), 25.0) == pytest.approx(
            169.1475, rel=1e-6)

    def test_scale_is_linear_in_batch(self):
        spec = _spec()
        assert (batch_grams_per_phr(spec, 50.0)
                == pytest.approx(2 * batch_grams_per_phr(spec, 25.0)))

    def test_upper_sigma_gives_the_coarser_delta(self):
        """Выбор верха Σphr — осторожный: δ выходит КРУПНЕЕ, чем по низу.

        Занижать δ опаснее: это молча обещало бы разрешение, которого нет.
        """
        spec = _spec()
        lo, _hi = batch_sigma_phr(spec)
        step = 0.1
        delta_used = step / batch_grams_per_phr(spec, 25.0)
        delta_by_lo = step / (25.0 * 1000.0 / lo)
        assert delta_used > delta_by_lo

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_batch_is_explicit_error(self, bad):
        with pytest.raises(ValueError, match="замеса"):
            batch_grams_per_phr(_spec(), bad)

    @pytest.mark.parametrize("kg", [0.5, 1.0, 25.0, 300.0])
    def test_round_trip_kg_to_gpp_and_back(self, kg):
        """Круг «кг → г/phr → кг» точен: повторная сборка не сдвигает паспорт."""
        spec = _spec()
        gpp = batch_grams_per_phr(spec, kg)
        assert batch_kg_from_grams_per_phr(spec, gpp) == pytest.approx(kg)

    def test_reverse_without_spec_is_zero_not_a_guess(self):
        """Без спеки Σphr неизвестна ⇒ 0 («не задано»), а не выдуманное число."""
        assert batch_kg_from_grams_per_phr(None, 5.0) == 0.0

    @pytest.mark.parametrize("bad", [0.0, -1.0, None, "", "abc",
                                     float("nan"), float("inf")])
    def test_reverse_on_unset_or_broken_scale_is_zero(self, bad):
        """Префилл не падает на мусоре в паспорте — отдаёт «не задано»."""
        assert batch_kg_from_grams_per_phr(_spec(), bad) == 0.0


# ----------------------------------------------------------------------
# 83.3 таблица достижимости отвеса
# ----------------------------------------------------------------------
class TestBatchWeighingReport:

    def test_columns_and_rows(self):
        spec = _spec()
        df = batch_weighing_report(spec, 25.0, 0.1)
        assert list(df["компонент"]) == list(spec.component_names)
        for col in ("доля L", "доля U", "навеска min, г", "шагов весов",
                    "погрешность, %", "phr lo", "phr hi", "вердикт"):
            assert col in df.columns

    def test_grams_are_fraction_times_batch(self):
        """Арифметика ровно та, которой считает технолог: доля × вес замеса."""
        spec = _spec()
        batch_kg = 25.0
        df = batch_weighing_report(spec, batch_kg, 0.1)
        lo_x, _ = spec.fraction_bounds()
        assert np.allclose(np.asarray(df["навеска min, г"], float),
                           lo_x * batch_kg * 1000.0, atol=1e-3)

    def test_golden_min_weights(self):
        """Golden (12.08.2026), замес 25 кг: RESIN 16914.75 г, UV 8.47 г."""
        df = batch_weighing_report(_spec(), 25.0, 0.1)
        assert _row(df, "RESIN")["навеска min, г"] == pytest.approx(
            16914.7497, rel=1e-6)
        assert _row(df, "UV")["навеска min, г"] == pytest.approx(8.4717,
                                                                 rel=1e-4)
        assert _row(df, "UV")["шагов весов"] == pytest.approx(84.7, abs=0.1)
        assert _row(df, "UV")["погрешность, %"] == pytest.approx(1.18,
                                                                 abs=0.01)

    def test_fraction_route_differs_from_gpp_route(self):
        """Ключевой факт: через доли и через ``g/phr`` — РАЗНЫЕ числа.

        Для fixed-узлов (RESIN, ESO) совпадает точно, для варьируемых (DINP,
        CPE) — расходится: ``g/phr`` считает по Σphr_max, а такая точка
        недостижима (у группы Σφ = 1, у capped-оси работает потолок). Поэтому
        оценивать навеску через ``g/phr`` нельзя.
        """
        spec = _spec()
        batch_kg = 25.0
        df = batch_weighing_report(spec, batch_kg, 0.1)
        gpp = batch_grams_per_phr(spec, batch_kg)
        iv = spec.phr_intervals()
        for fixed in ("RESIN", "ESO"):
            assert (_row(df, fixed)["навеска min, г"]
                    == pytest.approx(iv[fixed][0] * gpp, rel=1e-6))
        for varying in ("DINP", "CPE"):
            via_frac = float(_row(df, varying)["навеска min, г"])
            via_gpp = iv[varying][0] * gpp
            assert via_frac > via_gpp + 1.0     # расхождение — десятки граммов

    def test_box_is_conservative_against_sampling(self):
        """Расчётная навеска ≤ фактической: ошибка в БЕЗОПАСНУЮ сторону."""
        spec = _spec()
        batch_g = 25.0 * 1000.0
        df = batch_weighing_report(spec, 25.0, 0.1)
        P = np.array([spec.decode(z) for z in spec.sample_z(800, seed=11)])
        X = P / P.sum(axis=1, keepdims=True)
        for j, nm in enumerate(spec.component_names):
            fact_min = float(X[:, j].min()) * batch_g
            assert float(_row(df, nm)["навеска min, г"]) <= fact_min + 1e-6


# ----------------------------------------------------------------------
# 83.4 вердикты
# ----------------------------------------------------------------------
class TestVerdicts:

    def test_good_lab_is_all_ok(self):
        """Замес 25 кг на весах 0.1 г — прямая навеска по всем осям."""
        df = batch_weighing_report(_spec(), 25.0, 0.1)
        assert set(df["вердикт"]) <= {BATCH_VERDICT_OK, BATCH_VERDICT_ZERO}
        assert batch_weighing_problems(df) == {}

    def test_small_batch_makes_uv_coarse(self):
        """Замес 1 кг: УФ 0.339 г = 3.4 шага весов → «грубо» (golden)."""
        df = batch_weighing_report(_spec(), 1.0, 0.1)
        uv = _row(df, "UV")
        assert uv["навеска min, г"] == pytest.approx(0.3389, rel=1e-3)
        assert uv["шагов весов"] == pytest.approx(3.4, abs=0.1)
        assert uv["вердикт"] == BATCH_VERDICT_COARSE
        assert batch_weighing_problems(df)[BATCH_VERDICT_COARSE] == ["UV"]

    def test_coarse_scales_flag_impossible(self):
        """Весы 100 г при замесе 25 кг: TiO2 (53.5 г) и UV (8.5 г) НЕ берутся."""
        df = batch_weighing_report(_spec(), 25.0, 100.0)
        assert _row(df, "TiO2")["вердикт"] == BATCH_VERDICT_IMPOSSIBLE
        assert _row(df, "UV")["вердикт"] == BATCH_VERDICT_IMPOSSIBLE
        probs = batch_weighing_problems(df)
        assert probs[BATCH_VERDICT_IMPOSSIBLE] == ["TiO2", "UV"]
        assert "DINP" in probs[BATCH_VERDICT_COARSE]

    def test_zero_low_bound_is_not_a_scale_failure(self):
        """PBNK с низом 0 phr: «низ 0», а не «невозможно» (ложная тревога).

        Компонент МОЖЕТ отсутствовать в рецепте; отвес по нулю не нормируется,
        поэтому шагов/погрешности нет, и в проблемы он не попадает.
        """
        df = batch_weighing_report(_spec(), 25.0, 0.1)
        pbnk = _row(df, "PBNK")
        assert pbnk["вердикт"] == BATCH_VERDICT_ZERO
        assert pbnk["навеска min, г"] == 0.0
        # pandas хранит пропуск как NaN (столбец числовой) — важно, что это
        # именно ПРОПУСК, а не 0 шагов и не 0% погрешности.
        assert pd.isna(pbnk["шагов весов"])
        assert pd.isna(pbnk["погрешность, %"])
        for names in batch_weighing_problems(df).values():
            assert "PBNK" not in names

    def test_premix_verdict_matches_core_rule(self):
        """Вердикт «премикс» — ровно правило ядра, не своя копия.

        Нужна ось УЗКАЯ, но с ВЫСОКИМ низом (STAB 4.0…4.3 phr): навеска
        крупная (847 г — весы берут с запасом), а рабочий диапазон всего
        0.3 phr, и δ съедает > 5% ⇒ прямой навеской план по этой оси
        нечитаем. Именно этот случай отличает премикс от «грубо»: на осях
        вида UV (низ 0.05 phr) весы упираются раньше, и честный вердикт там —
        «грубо», а не «премикс».
        """
        spec = PhrSpec.from_dicts([
            {"name": "RESIN", "role": "FIXED", "value": 100.0},
            {"name": "DINP", "role": "ABSOLUTE", "range": [4.0, 14.0]},
            {"name": "STAB", "role": "ABSOLUTE", "range": [4.0, 4.3]},
        ])
        batch_kg, step = 25.0, 5.0
        df = batch_weighing_report(spec, batch_kg, step)
        delta = step / batch_grams_per_phr(spec, batch_kg)
        iv = spec.phr_intervals()
        stab = _row(df, "STAB")
        assert stab["вердикт"] == BATCH_VERDICT_PREMIX
        assert stab["навеска min, г"] == pytest.approx(847.4576, rel=1e-4)
        assert premix_required(delta, *iv["STAB"])      # то же правило ядра
        # широкая ось при тех же весах премикса не требует
        assert _row(df, "DINP")["вердикт"] == BATCH_VERDICT_OK
        assert not premix_required(delta, *iv["DINP"])
        assert batch_weighing_problems(df)[BATCH_VERDICT_PREMIX] == ["STAB"]

    def test_min_steps_is_a_parameter(self):
        """Порог «грубо» настраиваем: при min_steps=1 UV перестаёт быть грубым."""
        spec = _spec()
        strict = batch_weighing_report(spec, 1.0, 0.1)
        loose = batch_weighing_report(spec, 1.0, 0.1, min_steps=1)
        assert _row(strict, "UV")["вердикт"] == BATCH_VERDICT_COARSE
        assert _row(loose, "UV")["вердикт"] != BATCH_VERDICT_COARSE
        assert BATCH_MIN_STEPS == 20


# ----------------------------------------------------------------------
# 83.5 подпись и явные отказы (A0.6)
# ----------------------------------------------------------------------
class TestCaptionAndErrors:

    def test_caption_carries_scale_sigma_and_names(self):
        txt = batch_weighing_caption(_spec(), 25.0, 100.0)
        assert "Σphr" in txt
        assert "г на 1 phr" in txt
        assert "TiO2" in txt and "UV" in txt      # проблемы — ИМЕНАМИ
        assert "не блокируется" in txt            # A0.6 сказано явно

    def test_caption_says_all_clear(self):
        txt = batch_weighing_caption(_spec(), 25.0, 0.1)
        assert "прямой навеской" in txt
        assert "⚠️" not in txt

    @pytest.mark.parametrize("kg,step", [(0.0, 0.1), (-5.0, 0.1),
                                         (25.0, 0.0), (25.0, -0.1)])
    def test_bad_data_is_explicit_error(self, kg, step):
        """Ошибки ДАННЫХ — исключения; недостижимый отвес — таблица (A0.6)."""
        with pytest.raises(ValueError):
            batch_weighing_report(_spec(), kg, step)

    def test_problems_on_empty_frame(self):
        assert batch_weighing_problems(pd.DataFrame()) == {}
        assert batch_weighing_problems(None) == {}
