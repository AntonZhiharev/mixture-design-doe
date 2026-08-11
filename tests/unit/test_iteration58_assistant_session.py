# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 58 / ASSISTANT_SPEC — СЕССИЯ ассистента, привязанная к проекту.

Помощник-архитектор без памяти бесполезен: технолог объясняет ему цех заново
каждый запуск, а принятые решения компании нигде не остаются. Слой сессии
даёт переписку, вложения, staged-патчи спеки, артефакты песочницы и аудит
вызовов — всё в каталоге ПРОЕКТА (`project_campaigns/<имя>/assistant/`),
поэтому память переезжает вместе с проектом и исчезает при его удалении.

Покрытие: round-trip состояния; поведение при отсутствующей/битой сессии
(A0.6 — пустая сессия, а не исключение); усечение контекста БЕЗ потери
истории; дедуп вложений по sha256; жизненный цикл патча (staged → applied,
повторный переход запрещён); append-only журналы (битая строка пропускается);
чистые таблицы показа `views` (их же печатает демо и будет рисовать док).
"""
import json
from pathlib import Path

import pytest

from src.assistant import store
from src.assistant.session import (Artifact, AssistantSession, Attachment,
                                    FORMAT_VERSION, Message, PATCH_APPLIED,
                                    PATCH_REJECTED, PATCH_STAGED, StagedPatch,
                                    ToolCall, estimate_tokens,
                                    messages_from_pairs, new_session)
from src.assistant import views


PROJECT = "pvc_edge_v1"


def _rich_session() -> AssistantSession:
    """Сессия с записями всех видов (образец разбора спеки ПВХ)."""
    s = new_session(PROJECT, model="anthropic/claude-sonnet-4.5",
                    web_enabled=True)
    s.add_message("user", "Три смазки хочу в группу с закрытием на последней.")
    s.add_message("assistant",
                  "При k≥3 closure запрещён: все члены — SHARE_SIMPLEX.",
                  model="anthropic/claude-sonnet-4.5", web=False,
                  usage={"prompt_tokens": 900, "completion_tokens": 120,
                         "total_tokens": 1020})
    s.add_attachment(Attachment(name="TDS_chalk.txt", sha256="a" * 64,
                                size=2048, mime="text/plain",
                                text="d50 не указан", n_chars=14))
    s.stage_patch(StagedPatch(
        node="LUB.OPE", field_name="role", from_value="SHARE_CLOSURE",
        to_value="SHARE_SIMPLEX", bound_type="PHYSICAL", level="L3",
        source="validate_spec", rationale="k=3 ⇒ closure запрещён",
        confidence="high", affects_hash=True))
    s.add_artifact(Artifact(name="narrowing.png", kind="image",
                            path="artifacts/narrowing.png", tool="plot",
                            caption="hi_phi(T) немонотонна"))
    s.add_tool_call(ToolCall(tool="get_spec", args={}, ok=True,
                             duration_s=0.02, summary="q=19, dim_z=16"))
    return s


# ----------------------------------------------------------------------
# 1. Round-trip состояния
# ----------------------------------------------------------------------
def test_session_state_round_trip_is_stable():
    """to_state → from_state → to_state: состояние не «дрейфует»."""
    s = _rich_session()
    st1 = s.to_state()
    s2 = AssistantSession.from_state(st1)
    assert s2.to_state() == st1
    assert st1["format"] == FORMAT_VERSION
    assert s2.project == PROJECT and s2.web_enabled is True
    assert len(s2.messages) == 2 and len(s2.attachments) == 1
    assert len(s2.patches) == 1 and len(s2.artifacts) == 1
    assert len(s2.tool_calls) == 1


def test_from_state_rejects_unknown_format():
    """Чужой формат — явная ошибка, а не «прочитаем как получится» (A0.6)."""
    with pytest.raises(ValueError, match="формат сессии"):
        AssistantSession.from_state({"format": "assistant-v99"})


def test_usage_accumulates_across_answers():
    s = new_session(PROJECT)
    s.add_message("assistant", "раз", usage={"total_tokens": 100})
    s.add_message("assistant", "два", usage={"total_tokens": 250})
    assert s.usage["total_tokens"] == 350


def test_unknown_role_rejected():
    s = new_session(PROJECT)
    with pytest.raises(ValueError, match="роль"):
        s.add_message("developer", "нет такой роли в сессии")


# ----------------------------------------------------------------------
# 2. Привязка к проекту / персистентность
# ----------------------------------------------------------------------
def test_save_load_round_trip_in_project_dir(tmp_path):
    """Сессия ложится в `<проект>/assistant/session.json` и читается назад."""
    s = _rich_session()
    path = store.save_session(s, tmp_path)
    assert Path(path) == tmp_path / PROJECT / "assistant" / "session.json"
    assert (tmp_path / PROJECT / "assistant" / "files").is_dir()
    assert (tmp_path / PROJECT / "assistant" / "artifacts").is_dir()

    loaded = store.load_session(tmp_path, PROJECT)
    assert loaded.to_state() == s.to_state()


def test_missing_session_loads_empty_without_error(tmp_path):
    """Старый проект без каталога assistant/ открывается как пустая сессия."""
    (tmp_path / "legacy").mkdir()
    (tmp_path / "legacy" / "campaign.json").write_text("{}", encoding="utf-8")

    s = store.load_session(tmp_path, "legacy")
    assert s.project == "legacy" and s.is_empty()
    # чтение НЕ создаёт каталогов — открытие проекта не мусорит на диске
    assert not (tmp_path / "legacy" / "assistant").exists()


def test_corrupt_session_is_preserved_not_dropped(tmp_path):
    """Битый session.json сохраняется рядом, работа продолжается (A0.6)."""
    store.ensure_dirs(tmp_path, PROJECT)
    store.session_path(tmp_path, PROJECT).write_text("{не json", encoding="utf-8")

    s = store.load_session(tmp_path, PROJECT)
    assert s.is_empty() is False          # есть системное предупреждение
    assert "corrupt" in s.messages[0].content
    assert (tmp_path / PROJECT / "assistant" / "session.corrupt.json").exists()


def test_sessions_of_two_projects_are_independent(tmp_path):
    a = new_session("proj_a")
    a.add_message("user", "вопрос про мел")
    b = new_session("proj_b")
    b.add_message("user", "вопрос про УФ")
    store.save_session(a, tmp_path)
    store.save_session(b, tmp_path)

    assert store.load_session(tmp_path, "proj_a").messages[0].content \
        == "вопрос про мел"
    assert store.load_session(tmp_path, "proj_b").messages[0].content \
        == "вопрос про УФ"


def test_invalid_project_name_rejected(tmp_path):
    for bad in ("", "..", "a/b", "a\\b"):
        with pytest.raises(ValueError, match="Недопустимое имя проекта"):
            store.assistant_dir(tmp_path, bad)


def test_save_is_atomic_no_tmp_left(tmp_path):
    store.save_session(_rich_session(), tmp_path)
    leftovers = list((tmp_path / PROJECT / "assistant").glob("*.tmp"))
    assert leftovers == []


# ----------------------------------------------------------------------
# 3. Контекст: усечение БЕЗ потери истории
# ----------------------------------------------------------------------
def test_context_truncates_tail_and_marks_omission():
    """В модель уходит хвост; факт усечения виден системной пометкой."""
    s = new_session(PROJECT)
    for i in range(40):
        s.add_message("user", f"сообщение {i}: " + "x" * 400)

    ctx = s.context_messages(max_tokens=500)
    assert ctx[0]["role"] == "system" and "опущены" in ctx[0]["content"]
    # последнее сообщение диалога — самое свежее
    assert ctx[-1]["content"].startswith("сообщение 39")
    assert len(ctx) < len(s.messages)
    # история на «диске» (в объекте сессии) НЕ урезана
    assert len(s.messages) == 40


def test_context_keeps_all_when_budget_is_enough():
    s = new_session(PROJECT)
    s.add_message("user", "коротко")
    s.add_message("assistant", "ответ")
    ctx = s.context_messages(max_tokens=10000)
    assert [m["role"] for m in ctx] == ["user", "assistant"]
    assert all(not str(m["content"]).startswith("[сессия]") for m in ctx)


def test_context_keeps_at_least_last_message_even_if_huge():
    """Одно гигантское сообщение не должно давать ПУСТОЙ контекст."""
    s = new_session(PROJECT)
    s.add_message("user", "x" * 100000)
    ctx = s.context_messages(max_tokens=10)
    assert [m for m in ctx if m["role"] == "user"], "последний вопрос потерян"


def test_context_budget_must_be_positive():
    with pytest.raises(ValueError, match="max_tokens"):
        new_session(PROJECT).context_messages(max_tokens=0)


def test_estimate_tokens_monotone():
    assert estimate_tokens("x" * 400) > estimate_tokens("x" * 40) > 0


# ----------------------------------------------------------------------
# 4. Вложения
# ----------------------------------------------------------------------
def test_attachment_dedup_by_sha256():
    """Один файл — одна запись: повтор возвращает существующую."""
    s = new_session(PROJECT)
    first = s.add_attachment(Attachment(name="tds.pdf", sha256="b" * 64))
    again = s.add_attachment(Attachment(name="tds_copy.pdf", sha256="b" * 64))
    assert again is first and len(s.attachments) == 1


def test_remove_attachment():
    s = new_session(PROJECT)
    a = s.add_attachment(Attachment(name="tds.pdf", sha256="c" * 64))
    assert s.remove_attachment(a.id) is True
    assert s.remove_attachment(a.id) is False
    assert s.attachments == []


def test_clear_messages_keeps_files_and_patches():
    """«Очистить чат» не стирает паспорта сырья и предложенные патчи."""
    s = _rich_session()
    s.clear_messages()
    assert s.messages == []
    assert len(s.attachments) == 1 and len(s.patches) == 1


# ----------------------------------------------------------------------
# 5. Патчи: стейдж и терминальные статусы
# ----------------------------------------------------------------------
def test_staged_patch_lifecycle():
    s = new_session(PROJECT)
    p = s.stage_patch(StagedPatch(node="UV_CSFCP", field_name="range",
                                  from_value=[0.05, 0.30],
                                  to_value=[0.05, 0.195]))
    assert p.status == PATCH_STAGED and s.staged_patches() == [p]

    s.set_patch_status(p.id, PATCH_APPLIED, reason="подтверждено технологом")
    assert p.status == PATCH_APPLIED and p.applied_ts
    assert s.staged_patches() == []


def test_patch_cannot_be_applied_twice():
    s = new_session(PROJECT)
    p = s.stage_patch(StagedPatch(node="DINP", field_name="range"))
    s.set_patch_status(p.id, PATCH_REJECTED)
    with pytest.raises(ValueError, match="повторный переход"):
        s.set_patch_status(p.id, PATCH_APPLIED)


def test_patch_without_node_rejected():
    with pytest.raises(ValueError, match="один узел"):
        new_session(PROJECT).stage_patch(StagedPatch(node="", field_name="range"))


def test_unknown_patch_id_and_status():
    s = new_session(PROJECT)
    with pytest.raises(KeyError):
        s.set_patch_status("patch_missing", PATCH_APPLIED)
    p = s.stage_patch(StagedPatch(node="ESO", field_name="value"))
    with pytest.raises(ValueError, match="статус"):
        s.set_patch_status(p.id, "maybe")


def test_patch_ids_are_unique():
    s = new_session(PROJECT)
    ids = {s.stage_patch(StagedPatch(node=f"n{i}", field_name="range")).id
           for i in range(50)}
    assert len(ids) == 50


# ----------------------------------------------------------------------
# 6. Журналы (append-only jsonl)
# ----------------------------------------------------------------------
def test_logs_append_and_read(tmp_path):
    store.append_log(tmp_path, PROJECT, "tool_calls",
                     {"tool": "get_spec", "ok": True})
    store.append_log(tmp_path, PROJECT, "tool_calls",
                     {"tool": "preflight", "ok": False, "error": "gate failed"})
    recs = store.read_log(tmp_path, PROJECT, "tool_calls")
    assert [r["tool"] for r in recs] == ["get_spec", "preflight"]
    assert store.read_log(tmp_path, PROJECT, "tool_calls", limit=1)[0]["tool"] \
        == "preflight"


def test_log_broken_line_is_skipped_not_fatal(tmp_path):
    """Одна битая строка аудита не должна ронять открытие проекта."""
    store.append_log(tmp_path, PROJECT, "decisions", {"title": "решение 1"})
    with open(store.log_path(tmp_path, PROJECT, "decisions"), "a",
              encoding="utf-8") as fh:
        fh.write("{битая строка\n")
    store.append_log(tmp_path, PROJECT, "decisions", {"title": "решение 2"})

    recs = store.read_log(tmp_path, PROJECT, "decisions")
    assert [r["title"] for r in recs] == ["решение 1", "решение 2"]


def test_missing_log_reads_empty(tmp_path):
    assert store.read_log(tmp_path, PROJECT, "local_facts") == []


def test_unknown_log_kind_rejected(tmp_path):
    with pytest.raises(ValueError, match="журнал"):
        store.log_path(tmp_path, PROJECT, "secrets")


def test_logs_survive_alongside_session(tmp_path):
    """Журналы и сессия — разные файлы: сохранение сессии их не перетирает."""
    store.append_log(tmp_path, PROJECT, "decisions", {"title": "мел до 100 phr"})
    store.save_session(_rich_session(), tmp_path)
    assert len(store.read_log(tmp_path, PROJECT, "decisions")) == 1


# ----------------------------------------------------------------------
# 7. Перенос старой истории чата
# ----------------------------------------------------------------------
def test_messages_from_pairs_keeps_dialog_only():
    msgs = messages_from_pairs([{"role": "user", "content": "привет"},
                                {"role": "assistant", "content": "здравствуйте"},
                                {"role": "system", "content": "служебное"}])
    assert [m.role for m in msgs] == ["user", "assistant"]


# ----------------------------------------------------------------------
# 8. Чистые таблицы показа (их печатает демо и будет рисовать док)
# ----------------------------------------------------------------------
def test_messages_dataframe_structure():
    df = views.messages_dataframe(_rich_session())
    assert list(df.columns) == ["время", "роль", "🌐", "модель", "сообщение",
                                "инструментов", "~токенов"]
    assert len(df) == 2
    assert df.iloc[0]["роль"].endswith("пользователь")


def test_attachments_dataframe_structure():
    df = views.attachments_dataframe(_rich_session())
    assert list(df.columns) == ["файл", "тип", "размер, КБ", "символов",
                                "усечён", "sha256", "примечание"]
    assert df.iloc[0]["файл"] == "TDS_chalk.txt"
    assert df.iloc[0]["sha256"] == "a" * 12


def test_patches_dataframe_marks_hash_change():
    """Главный вопрос ревизора — поедет ли spec_hash — виден в таблице."""
    df = views.staged_patches_dataframe(_rich_session())
    assert list(df.columns) == ["id", "узел", "поле", "было", "стало",
                                "граница", "знание", "уверенность", "хеш",
                                "статус", "обоснование"]
    assert "меняется" in df.iloc[0]["хеш"]
    assert df.iloc[0]["статус"].endswith("предложен")


def test_patches_dataframe_only_staged_filter():
    s = _rich_session()
    s.set_patch_status(s.patches[0].id, PATCH_APPLIED)
    assert len(views.staged_patches_dataframe(s, only_staged=True)) == 0
    assert len(views.staged_patches_dataframe(s)) == 1


def test_tool_calls_dataframe_accepts_objects_and_dicts():
    """Сессия хранит объекты, журнал — словари; таблица одна и та же."""
    s = _rich_session()
    from_objects = views.tool_calls_dataframe(s.tool_calls)
    from_dicts = views.tool_calls_dataframe([c.to_state() for c in s.tool_calls])
    assert list(from_objects.columns) == list(from_dicts.columns)
    assert from_objects.iloc[0]["инструмент"] == "get_spec"
    assert from_dicts.iloc[0]["итог"] == "ok"


def test_artifacts_and_decisions_dataframes():
    # iter74: столбец таблицы называется «файл» (слово «артефакт» — внутреннее).
    df = views.artifacts_dataframe(_rich_session())
    assert df.iloc[0]["файл"] == "narrowing.png"

    dec = views.decisions_dataframe([{
        "ts": "2026-08-08T10:00:00+00:00", "title": "мел до 100 phr",
        "nodes": ["FILLER.total"], "author": "технолог",
        "spec_hash": "c63b7e1696e1c449", "rationale": "L1-факт цеха"}])
    assert dec.iloc[0]["дата"] == "2026-08-08"
    assert dec.iloc[0]["spec_hash"] == "c63b7e1696e1c4"[:12]


def test_session_caption_reports_web_and_counts():
    # iter74: «в стейдже» → «ждёт применения» (подпись читает технолог).
    txt = views.session_caption(_rich_session())
    assert PROJECT in txt and "файлов: 1" in txt
    assert "патчей ждёт применения: 1" in txt and "интернет: включён" in txt


def test_context_caption_reports_truncation():
    s = new_session(PROJECT)
    for i in range(40):
        s.add_message("user", f"{i}: " + "x" * 400)
    txt = views.context_caption(s, max_tokens=500)
    assert "из 40" in txt and "опущены" in txt


def test_attachment_digest_clips_long_text():
    s = new_session(PROJECT)
    s.add_attachment(Attachment(name="big.txt", sha256="d" * 64,
                                text="y" * 9000, n_chars=9000))
    dig = views.attachment_digest(s, per_file_chars=100)
    assert len(dig[0]["text"]) == 100 and dig[0]["clipped"] is True


# ----------------------------------------------------------------------
# 9. Совместимость с движком кампании (сессия НЕ трогает campaign.json)
# ----------------------------------------------------------------------
def test_session_does_not_touch_campaign_json(tmp_path):
    (tmp_path / PROJECT).mkdir()
    camp = tmp_path / PROJECT / "campaign.json"
    camp.write_text(json.dumps({"format": "campaign-v1"}), encoding="utf-8")
    before = camp.read_text(encoding="utf-8")

    store.save_session(_rich_session(), tmp_path)
    store.append_log(tmp_path, PROJECT, "tool_calls", {"tool": "get_spec"})

    assert camp.read_text(encoding="utf-8") == before
