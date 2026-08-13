# Copyright 2026 DOE contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 84 — хвосты iter83 + два отказа окна помощника.

Четыре независимых отказа, найденных в живой ПВХ-сессии (12.08.2026), и то,
что закрывает каждый:

**1. Σphr считался в трёх местах по-своему (хвост iter83 «б»).**
``readonly.get_spec.sigma_phr_static`` суммировал ВСЕ узлы ``phr_intervals()``
вместе с узлами-ТОТАЛАМИ групп; тотал группы = сумма своих детей ⇒ группы
входили в сумму ДВАЖДЫ. На референсной геометрии выходило ``114.85…162.80``
вместо верного ``109.85…147.80`` — расхождение ровно на интервал тотала
``SOFT``. Именно эти числа ассистент цитировал человеку, пока окно технолога
показывало другие. Лечение — один источник истины в ЯДРЕ
(:meth:`PhrSpec.sigma_phr_bounds`), к которому сведены оба слоя.

**2. График и таблица исчезали из разговора** («первичный вывод показывает, а
потом уходят»). Показ шёл только из ``TurnResult.new_artifacts`` — памяти
ОДНОГО прогона Streamlit. Любой rerun (нажали кнопку, раскрыли экспандер,
задали новый вопрос) перерисовывал ленту из истории, а история связи «ответ ↔
его файлы» не хранила вовсе. Файлы при этом лежали в проекте и были видны в
панели «🖼 Файлы расчётов» — терялись не данные, а КОНТЕКСТ показа. Лечение —
ссылки на артефакты в самом сообщении (``Message.artifacts``).

**3. В модель уходил весь чат целиком.** Замер сессии: 43 реплики,
100 862 символа ≈ 25 215 токенов при бюджете 24 000 — бюджет не срабатывал
НИКОГДА, и каждый вопрос тащил всю переписку. Вместе с системным промптом
(19,5 тыс. символов) и JSON-схемами 26 инструментов (16,6 тыс.) выходило
~29,8 тыс. токенов на ход. Лечение двойное: вырезание устаревших разделов
``ЧИСЛА`` из прошлых ответов + бюджет 12 000. Замер после — 20,4 тыс. (−31 %).

**4. Стоимость хода была невидима.** Подпись говорила «43 из 43», и это
читалось как «всё в порядке», хотя означало «уходит весь разговор».

Канон, который проверяется ниже:
  * Σphr — ЛИСТЬЯ, и ядро с инструментом обязаны давать одно число;
  * обрезка ``ЧИСЛА`` касается ТОЛЬКО контекста: сессия, диск и показ целы;
  * ПОСЛЕДНИЙ ответ не обрезается — на его числа опирается уточняющий вопрос;
  * вырезанное НЕ замалчивается (пометка вместо тела, A0.6);
  * артефакты живут в сообщении и переживают перезапуск приложения;
  * ссылки на артефакты НЕ уходят в модель (бюджет не жгут).
"""
from __future__ import annotations

import pytest

from src.apps import workspace as wsx
from src.assistant import store, views
from src.assistant.session import (NUMBERS_OMITTED_NOTE, Artifact, Message,
                                   estimate_tokens, new_session,
                                   strip_numbers_section)
from src.assistant.tools import dispatch
from src.assistant.tools.registry import ToolContext
from tests.unit.test_iteration83_batch_weighing import _spec

PROJECT = "iter84"

#: Ответ архитектора в формате iter64 — с разделом чисел и хвостом после него.
ANSWER = """## ОТВЕТ
Верх DINP упирается в техлимит, а не в заявленный диапазон.

## ЧИСЛА
- `explain_node(DINP)`: phr 4.0…14.0, active=max_phr
- `get_spec`: spec_hash=deadbeef12, Σphr 109.85…147.80

## OPEN_QUESTIONS
- Паспорт даёт 0.5 phr, технолог льёт 1.2 — что берём за границу?"""


def _session_with(n_answers: int = 3):
    """Сессия из ``n_answers`` пар «вопрос → ответ по формату»."""
    s = new_session(PROJECT)
    for i in range(n_answers):
        s.add_message("user", f"вопрос {i}")
        s.add_message("assistant", ANSWER)
    return s


# ======================================================================
# 1. Σphr: один источник истины (хвост iter83 «б»)
# ======================================================================
class TestSigmaPhrSingleSource:

    def test_core_sums_leaves_golden(self):
        """Golden прогона 12.08.2026: по листьям 109.85 … 147.80."""
        lo, hi = _spec().sigma_phr_bounds()
        assert lo == pytest.approx(109.85)
        assert hi == pytest.approx(147.80)

    def test_all_nodes_sum_would_double_count_groups(self):
        """Ровно та ошибка, что жила в инструменте: группы посчитаны дважды.

        Тест фиксирует не «правильное» поведение, а РАЗМЕР ошибки — иначе
        регресс вернул бы её незаметно: числа остались бы правдоподобными.
        """
        spec = _spec()
        iv = spec.phr_intervals()
        all_hi = sum(v[1] for v in iv.values())
        leaf_lo, leaf_hi = spec.sigma_phr_bounds()
        assert all_hi == pytest.approx(162.80)
        assert leaf_hi == pytest.approx(147.80)
        # расхождение — ровно интервал тотала группы SOFT, а не «шум»
        assert all_hi - leaf_hi == pytest.approx(iv["SOFT"][1])
        assert sum(v[0] for v in iv.values()) - leaf_lo == pytest.approx(
            iv["SOFT"][0])

    def test_tool_reports_leaf_sum_not_all_nodes(self):
        """`get_spec` отдаёт исправленные числа — те же, что видит технолог."""
        out = dispatch(ToolContext(spec=_spec()), "get_spec",
                       {"include_nodes": False})
        assert out["sigma_phr_static"] == pytest.approx([109.85, 147.80])

    def test_ui_helper_delegates_to_core(self):
        """Хелпер навески UI и ядро не могут разойтись: копии арифметики нет."""
        from src.apps.campaign_ui import batch_sigma_phr

        spec = _spec()
        assert batch_sigma_phr(spec) == spec.sigma_phr_bounds()

    def test_static_interval_contains_actual_recipes(self):
        """Смысловая проверка: интервал СОДЕРЖИТ Σphr фактических точек."""
        import numpy as np

        spec = _spec()
        lo, hi = spec.sigma_phr_bounds()
        P = np.array([spec.decode(z) for z in spec.sample_z(300, seed=11)])
        S = P.sum(axis=1)
        assert lo - 1e-9 <= S.min() and S.max() <= hi + 1e-9


# ======================================================================
# 2. Раздел ЧИСЛА: вырезается из КОНТЕКСТА, но не из истории
# ======================================================================
class TestStripNumbers:

    def test_body_removed_note_left(self):
        out = strip_numbers_section(ANSWER)
        assert "spec_hash=deadbeef12" not in out
        assert NUMBERS_OMITTED_NOTE in out
        # Заголовок остаётся: модель должна видеть, что числа БЫЛИ посчитаны,
        # и что их следует перезапросить — а не решить, что инструменты молчали.
        assert "## ЧИСЛА" in out

    def test_other_sections_survive(self):
        """Режется ровно раздел чисел, а не «всё после него»."""
        out = strip_numbers_section(ANSWER)
        assert "упирается в техлимит" in out
        assert "OPEN_QUESTIONS" in out
        assert "что берём за границу" in out

    def test_text_without_section_untouched(self):
        plain = "Обычный ответ без формата."
        assert strip_numbers_section(plain) == plain

    def test_empty_section_untouched(self):
        """Заголовок без тела резать нечего — пометка была бы шумом."""
        src = "## ОТВЕТ\nтекст\n\n## ЧИСЛА\n\n## PATCH\nнет"
        assert strip_numbers_section(src) == src

    def test_shorter_than_original(self):
        assert len(strip_numbers_section(ANSWER)) < len(ANSWER)

    def test_never_grows_the_text(self):
        """Инвариант: «оптимизация» не имеет права РАЗДУВАТЬ запрос.

        Поймано этим тестом на первой реализации: пометка была подробной и на
        коротком разделе («- `get_spec`: q=19») оказывалась длиннее вырезанного
        тела — экономия превращалась в перерасход, причём тем больший, чем
        лаконичнее отвечала модель.
        """
        for src in ("## ЧИСЛА\n- `get_spec`: q=19",
                    "## ОТВЕТ\nда\n\n## ЧИСЛА\n- x=1\n\n## PATCH\nнет",
                    ANSWER):
            assert len(strip_numbers_section(src)) <= len(src)


# ======================================================================
# 3. Контекст хода: экономия есть, история цела
# ======================================================================
class TestContextBudget:

    def test_last_answer_keeps_its_numbers(self):
        """На числа ПОСЛЕДНЕГО ответа опирается уточняющий вопрос человека."""
        ctx = _session_with(3).context_messages(max_tokens=100000)
        assert "spec_hash=deadbeef12" in str(ctx[-1]["content"])

    def test_earlier_answers_are_stripped(self):
        ctx = _session_with(3).context_messages(max_tokens=100000)
        older = [m for m in ctx if m["role"] == "assistant"][:-1]
        assert older, "нужны минимум два ответа"
        for m in older:
            assert NUMBERS_OMITTED_NOTE in str(m["content"])
            assert "spec_hash=deadbeef12" not in str(m["content"])

    def test_session_and_disk_not_mutated(self, tmp_path):
        """Обрезка — свойство СБОРКИ запроса, а не истории (A0.6)."""
        s = _session_with(3)
        s.context_messages(max_tokens=100000)
        assert all(m.content == ANSWER for m in s.messages
                   if m.role == "assistant")
        store.save_session(s, tmp_path, PROJECT)
        loaded = store.load_session(tmp_path, PROJECT)
        assert all(m.content == ANSWER for m in loaded.messages
                   if m.role == "assistant")

    def test_strip_can_be_switched_off(self):
        full = _session_with(3).context_messages(max_tokens=100000,
                                                 strip_numbers=False)
        assert all(NUMBERS_OMITTED_NOTE not in str(m["content"]) for m in full)

    def test_stripping_saves_budget(self):
        """Обрезка не декоративная: она РЕАЛЬНО уменьшает объём запроса."""
        s = _session_with(6)
        size = lambda ms: sum(len(str(m["content"])) for m in ms)  # noqa: E731
        big = size(s.context_messages(max_tokens=100000, strip_numbers=False))
        small = size(s.context_messages(max_tokens=100000))
        assert small < big

    def test_freed_budget_keeps_more_dialogue(self):
        """Сэкономленное идёт на РЕПЛИКИ: та же квота держит больше разговора."""
        s = _session_with(8)
        budget = estimate_tokens(ANSWER) * 6
        n_strip = len(s.context_messages(max_tokens=budget))
        n_full = len(s.context_messages(max_tokens=budget,
                                        strip_numbers=False))
        assert n_strip > n_full

    def test_truncation_still_announced(self):
        """Экономия не отменяет честности: усечение помечается как раньше."""
        s = _session_with(20)
        ctx = s.context_messages(max_tokens=200)
        assert ctx[0]["role"] == "system" and "опущены" in ctx[0]["content"]
        assert len(s.messages) == 40

    def test_default_budget_lowered(self):
        """Бюджет — константа, а не магическое число в теле функции."""
        from src.assistant.context import CONTEXT_TOKENS

        assert CONTEXT_TOKENS == 12000

    def test_caption_names_volume_and_uses_real_budget(self):
        """Подпись показывает ФАКТИЧЕСКИЙ бюджет хода, а не свой дефолт."""
        from src.assistant.context import CONTEXT_TOKENS

        txt = views.context_caption(_session_with(3))
        assert str(CONTEXT_TOKENS) in txt
        assert "токенов" in txt


# ======================================================================
# 4. Артефакты живут в сообщении, а не в памяти одного прогона
# ======================================================================
class TestMessageArtifacts:

    def test_message_carries_artifact_ids(self):
        s = new_session(PROJECT)
        m = s.add_message("assistant", "готово", artifacts=["art_1", "art_2"])
        assert m.artifacts == ["art_1", "art_2"]

    def test_artifacts_survive_save_load(self, tmp_path):
        """Главное свойство: график остаётся в разговоре после перезапуска."""
        s = new_session(PROJECT)
        s.add_message("assistant", "готово", artifacts=["art_1"])
        store.save_session(s, tmp_path, PROJECT)
        loaded = store.load_session(tmp_path, PROJECT)
        assert loaded.messages[-1].artifacts == ["art_1"]

    def test_artifacts_not_sent_to_model(self):
        """Ссылки — для показа: в контексте они жгли бы бюджет впустую."""
        msg = Message(role="assistant", content="готово", artifacts=["art_1"])
        assert "artifacts" not in msg.chat_message()

    def test_old_session_without_field_reads_fine(self):
        """Сессии до iter84 читаются как раньше — поля просто нет."""
        m = Message.from_state({"role": "assistant", "content": "старое"})
        assert m.artifacts == []

    def test_feed_item_exposes_artifacts(self):
        s = new_session(PROJECT)
        s.add_message("user", "построй график")
        s.add_message("assistant", "готово", artifacts=["art_1"])
        items = wsx.feed_items(s.messages)
        assert items[-1].artifacts == ("art_1",)
        assert items[0].artifacts == ()

    def test_message_outputs_resolves_only_existing_files(self, tmp_path):
        """Пропавший файл не показывается: заголовок без графика хуже пустоты."""
        png = tmp_path / "curve.png"
        png.write_bytes(b"\x89PNG\r\n")
        s = new_session(PROJECT)
        alive = s.add_artifact(Artifact(name="curve.png", kind="image",
                                       path=str(png), tool="run_python"))
        gone = s.add_artifact(Artifact(name="lost.png", kind="image",
                                      path=str(tmp_path / "lost.png")))
        out = views.message_outputs(s, [alive.id, gone.id])
        assert [o.name for o in out] == ["curve.png"]

    def test_message_outputs_ignores_foreign_artifacts(self):
        """Реплика показывает СВОИ файлы, а не всё, что есть в проекте."""
        s = new_session(PROJECT)
        s.add_artifact(Artifact(name="other.png", kind="image", path="nope"))
        assert views.message_outputs(s, []) == []
