# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Iteration 77 — ИДЕНТИЧНОСТЬ проекта: ссылка вместо имени.

Архитектурный баг (замечание пользователя 12.08.2026): проект и переписка
ассистента были связаны ЧЕРЕЗ ИМЯ. Правка буквы в поле «Имя проекта»
переключала переписку, вложения и журналы на другой каталог; переименования
как операции не существовало; не стартовавший проект не имел идентичности,
и подвязать к нему что-либо до сборки было нельзя.

Здесь фиксируется ядро (`src/core/project_ref.py`, вариант C): каталог остаётся
человеко-читаемым, идентичность лежит в `project.json` внутри него, истинный
ключ — `ref`. Проверяется ровно то, что было сломано:

  1. ссылка выдаётся ДО старта (дефолтный проект) и не меняется;
  2. переименование меняет только `label` — каталог и ссылка стоят;
  3. чтение не мутирует диск, битый файл не роняет открытие проекта;
  4. legacy-каталоги (до iter77) видны и мигрируют без переносов;
  5. имя — не ключ: одноимённые проекты живут, резолв по имени отказывает
     явно, а не угадывает.
"""
import json
import os

import pytest

from src.apps import assistant_dock as dock
from src.apps import campaign_state as cst
from src.assistant import store
from src.assistant.session import new_session
from src.assistant.tools import ToolError
from src.core import project_ref as pr
from src.mcp import campaign_tools as ct


# ======================================================================
# 1. Формат ссылки
# ======================================================================
class TestRefFormat:
    def test_new_ref_has_prefix_and_length(self):
        ref = pr.new_ref()
        assert ref.startswith(pr.REF_PREFIX)
        assert len(ref) == len(pr.REF_PREFIX) + pr.REF_HEX
        assert pr.is_ref(ref)

    def test_refs_are_unique(self):
        assert len({pr.new_ref() for _ in range(200)}) == 200

    @pytest.mark.parametrize("bad", [
        "", "prj_", "prj_xyz", "prj_AB12CD34EF56",       # регистр не hex
        "proj_ab12cd34ef56", "ab12cd34ef56",
        "prj_ab12cd34ef5",                                # на символ короче
        "prj_ab12cd34ef567",                              # на символ длиннее
    ])
    def test_bad_ref_rejected(self, bad):
        assert pr.is_ref(bad) is False
        with pytest.raises(ValueError, match="Недопустимая ссылка"):
            pr.validate_ref(bad)

    def test_short_ref_is_readable(self):
        ident = pr.ProjectIdentity(ref="prj_ab12cd34ef56", label="x",
                                   dirname="x")
        assert ident.short_ref() == "prj_ab12cd34"

    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b"])
    def test_dirname_traversal_rejected(self, bad):
        with pytest.raises(ValueError, match="Недопустимое имя каталога"):
            pr.validate_dirname(bad)


# ======================================================================
# 2. Каталог из человеческого имени (одноразово, при создании)
# ======================================================================
class TestSlugify:
    def test_cyrillic_label_kept_as_is(self):
        assert pr.slugify_label("Разработка рецептов ПВХ") \
            == "Разработка рецептов ПВХ"

    def test_forbidden_chars_and_spaces_collapsed(self):
        assert pr.slugify_label('кромка: ПВХ / "жёсткая"') \
            == "кромка ПВХ жёсткая"

    @pytest.mark.parametrize("bad", ["", "   ", "...", "///", "CON", "nul"])
    def test_degenerate_label_falls_back(self, bad):
        assert pr.slugify_label(bad) == "project"

    def test_result_is_valid_dirname(self):
        assert pr.validate_dirname(pr.slugify_label("a/b\\c")) == "a b c"


# ======================================================================
# 3. Ссылка выдаётся ДО старта и НЕ меняется (требование пользователя)
# ======================================================================
class TestEnsureIdentity:
    def test_default_project_gets_ref_before_any_build(self, tmp_path):
        """Не стартовавший `my_project` сразу опознаваем по ссылке."""
        ident = pr.ensure_default_project(tmp_path)
        assert pr.is_ref(ident.ref)
        assert ident.dirname == pr.DEFAULT_DIRNAME
        assert ident.label == pr.DEFAULT_LABEL
        # ни движка, ни черновика — только идентичность
        assert (tmp_path / pr.DEFAULT_DIRNAME / pr.IDENTITY_FILE).exists()
        assert not (tmp_path / pr.DEFAULT_DIRNAME / pr.CAMPAIGN_FILE).exists()

    def test_reopened_default_keeps_same_ref(self, tmp_path):
        """«Открыть снова как дефолтный, ссылку оставить» — до удаления админом."""
        first = pr.ensure_default_project(tmp_path)
        again = pr.ensure_default_project(tmp_path)
        assert again.ref == first.ref

    def test_existing_ref_is_never_overwritten(self, tmp_path):
        first = pr.ensure_identity(tmp_path, "p", label="Проект")
        same = pr.ensure_identity(tmp_path, "p", label="Другое имя",
                                  ref=pr.new_ref())
        assert same.ref == first.ref, "ссылка проекта неизменяема"
        assert same.label == "Проект", "label затирать нельзя"

    def test_written_state_round_trips(self, tmp_path):
        ident = pr.ensure_identity(tmp_path, "p", label="Кромка")
        state = json.loads(
            (tmp_path / "p" / pr.IDENTITY_FILE).read_text(encoding="utf-8"))
        assert state["format"] == pr.FORMAT_VERSION
        assert state["ref"] == ident.ref and state["label"] == "Кромка"
        # dirname внутрь файла не пишем: имя каталога — это сам каталог
        assert "dirname" not in state
        assert pr.read_identity(tmp_path, "p").to_state() == ident.to_state()

    def test_write_is_atomic_no_tmp_left(self, tmp_path):
        pr.ensure_identity(tmp_path, "p", label="Кромка")
        assert list((tmp_path / "p").glob("*.tmp")) == []


# ======================================================================
# 4. ЧТЕНИЕ не мутирует диск; битый файл не роняет открытие (A0.6)
# ======================================================================
class TestReadIsPure:
    def test_read_absent_returns_none_and_creates_nothing(self, tmp_path):
        assert pr.read_identity(tmp_path, "нет-такого") is None
        assert list(tmp_path.iterdir()) == []

    def test_list_on_missing_root_is_empty(self, tmp_path):
        assert pr.list_identities(tmp_path / "нет") == []

    def test_corrupt_identity_does_not_raise(self, tmp_path):
        (tmp_path / "p").mkdir()
        (tmp_path / "p" / pr.IDENTITY_FILE).write_text("{не json",
                                                       encoding="utf-8")
        assert pr.read_identity(tmp_path, "p") is None
        # каталог всё равно опознан как проект (файл идентичности есть)
        assert pr.has_project_content(tmp_path, "p") is True
        # и ссылка досоздаётся явным вызовом
        assert pr.is_ref(pr.ensure_identity(tmp_path, "p").ref)

    def test_identity_without_ref_is_not_silently_accepted(self, tmp_path):
        (tmp_path / "p").mkdir()
        (tmp_path / "p" / pr.IDENTITY_FILE).write_text(
            json.dumps({"label": "без ссылки"}), encoding="utf-8")
        assert pr.read_identity(tmp_path, "p") is None


# ======================================================================
# 5. ПЕРЕИМЕНОВАНИЕ: меняется только label (суть багфикса)
# ======================================================================
class TestRename:
    def test_rename_keeps_ref_and_dir(self, tmp_path):
        ident = pr.create_project(tmp_path, "Кромка ПВХ")
        renamed = pr.rename_label(tmp_path, ident.ref, "Кромка ПВХ v2")
        assert renamed.ref == ident.ref
        assert renamed.dirname == ident.dirname
        assert renamed.label == "Кромка ПВХ v2"
        assert (tmp_path / ident.dirname).is_dir(), "каталог не двигается"

    def test_previous_label_is_remembered(self, tmp_path):
        ident = pr.create_project(tmp_path, "старое")
        pr.rename_label(tmp_path, ident.ref, "новое")
        again = pr.rename_label(tmp_path, ident.ref, "новейшее")
        assert again.label_history == ["старое", "новое"]

    def test_rename_to_same_label_is_noop(self, tmp_path):
        ident = pr.create_project(tmp_path, "имя")
        again = pr.rename_label(tmp_path, ident.ref, "имя")
        assert again.label_history == []

    def test_empty_label_refused(self, tmp_path):
        ident = pr.create_project(tmp_path, "имя")
        with pytest.raises(ValueError, match="не может быть пустым"):
            pr.rename_label(tmp_path, ident.ref, "   ")

    def test_rename_by_unknown_ref_explains(self, tmp_path):
        pr.create_project(tmp_path, "имя")
        with pytest.raises(ValueError, match="нет в каталоге"):
            pr.rename_label(tmp_path, pr.new_ref(), "другое")

    def test_renamed_project_keeps_its_session_dir(self, tmp_path):
        """Ровно тот баг: переписка НЕ переезжает при смене имени."""
        ident = pr.create_project(tmp_path, "Кромка")
        conv = tmp_path / ident.dirname / "assistant"
        conv.mkdir(parents=True)
        (conv / "session.json").write_text('{"messages": []}',
                                           encoding="utf-8")
        pr.rename_label(tmp_path, ident.ref, "Кромка (жёсткая)")
        found = pr.find_by_ref(tmp_path, ident.ref)
        assert found.dirname == ident.dirname
        assert (tmp_path / found.dirname / "assistant"
                / "session.json").exists()


# ======================================================================
# 6. СОЗДАНИЕ: каталог читаемый, одноимённые проекты допустимы
# ======================================================================
class TestCreate:
    def test_dir_is_human_readable(self, tmp_path):
        ident = pr.create_project(tmp_path, "Кромка ПВХ")
        assert ident.dirname == "Кромка ПВХ"
        assert (tmp_path / "Кромка ПВХ" / pr.IDENTITY_FILE).exists()

    def test_same_label_twice_gives_two_projects(self, tmp_path):
        a = pr.create_project(tmp_path, "Кромка")
        b = pr.create_project(tmp_path, "Кромка")
        assert a.ref != b.ref
        assert {a.dirname, b.dirname} == {"Кромка", "Кромка (2)"}
        assert a.label == b.label == "Кромка"

    def test_create_with_known_ref_returns_same_project(self, tmp_path):
        a = pr.create_project(tmp_path, "Кромка")
        again = pr.create_project(tmp_path, "Кромка", ref=a.ref)
        assert again.dirname == a.dirname, "копия проекта не создаётся"

    def test_empty_dir_is_reused_not_suffixed(self, tmp_path):
        (tmp_path / "Кромка").mkdir()          # пустая папка без признаков
        ident = pr.create_project(tmp_path, "Кромка")
        assert ident.dirname == "Кромка"

    def test_explicit_dirname_wins(self, tmp_path):
        ident = pr.create_project(tmp_path, "Кромка ПВХ", dirname="pvc_edge")
        assert ident.dirname == "pvc_edge" and ident.label == "Кромка ПВХ"


# ======================================================================
# 7. СКАН каталога и legacy (проекты до iter77)
# ======================================================================
class TestScan:
    def _legacy(self, tmp_path, name, *, kind="campaign"):
        base = tmp_path / name
        if kind == "campaign":
            base.mkdir(parents=True)
            (base / pr.CAMPAIGN_FILE).write_text("{}", encoding="utf-8")
        elif kind == "draft":
            base.mkdir(parents=True)
            (base / pr.SETUP_DRAFT_FILE).write_text("{}", encoding="utf-8")
        else:                                   # только переписка
            (base / "assistant").mkdir(parents=True)
            (base / "assistant" / "session.json").write_text(
                "{}", encoding="utf-8")
        return base

    def test_legacy_project_is_visible_without_ref(self, tmp_path):
        """Каталог до iter77 не прячем: это данные пользователя (A0.6)."""
        self._legacy(tmp_path, "старый")
        [ident] = pr.list_identities(tmp_path)
        assert ident.legacy is True and ident.ref == ""
        assert ident.label == "старый" and ident.has_ref is False

    def test_project_with_only_conversation_is_a_project(self, tmp_path):
        """`my_project` с одной перепиской (реальный случай) — проект."""
        self._legacy(tmp_path, "my_project", kind="session")
        assert [i.dirname for i in pr.list_identities(tmp_path)] \
            == ["my_project"]

    def test_draft_only_project_is_a_project(self, tmp_path):
        self._legacy(tmp_path, "черновик", kind="draft")
        assert len(pr.list_identities(tmp_path)) == 1

    def test_foreign_dirs_and_files_ignored(self, tmp_path):
        (tmp_path / "просто_папка").mkdir()
        (tmp_path / "файл.txt").write_text("x", encoding="utf-8")
        assert pr.list_identities(tmp_path) == []

    def test_include_legacy_false_hides_unreferenced(self, tmp_path):
        self._legacy(tmp_path, "старый")
        pr.create_project(tmp_path, "новый")
        assert [i.dirname for i in
                pr.list_identities(tmp_path, include_legacy=False)] == ["новый"]

    def test_migrate_assigns_refs_without_moving_anything(self, tmp_path):
        self._legacy(tmp_path, "старый")
        self._legacy(tmp_path, "my_project", kind="session")
        migrated = pr.migrate_root(tmp_path)
        assert {i.dirname for i in migrated} == {"старый", "my_project"}
        assert all(pr.is_ref(i.ref) for i in migrated)
        # каталоги на месте, содержимое не тронуто
        assert (tmp_path / "старый" / pr.CAMPAIGN_FILE).exists()
        assert (tmp_path / "my_project" / "assistant"
                / "session.json").exists()

    def test_migrate_is_idempotent(self, tmp_path):
        self._legacy(tmp_path, "старый")
        first = pr.migrate_root(tmp_path)[0]
        assert pr.migrate_root(tmp_path) == []
        assert pr.find_by_ref(tmp_path, first.ref).dirname == "старый"


# ======================================================================
# 8. РЕЗОЛВ: ссылка — ключ, имя — лишь подсказка
# ======================================================================
class TestResolve:
    def test_by_ref(self, tmp_path):
        ident = pr.create_project(tmp_path, "Кромка")
        assert pr.resolve(tmp_path, ident.ref).ref == ident.ref

    def test_by_dirname_and_by_label(self, tmp_path):
        ident = pr.create_project(tmp_path, "Кромка ПВХ", dirname="pvc")
        assert pr.resolve(tmp_path, "pvc").ref == ident.ref
        assert pr.resolve(tmp_path, "Кромка ПВХ").ref == ident.ref

    def test_ambiguous_label_refuses_to_guess(self, tmp_path):
        """Два проекта с ОДНИМ именем: угадывать, про какой речь, нельзя."""
        pr.create_project(tmp_path, "Кромка", dirname="pvc_a")
        pr.create_project(tmp_path, "Кромка", dirname="pvc_b")
        with pytest.raises(ValueError, match="Имя — не ключ"):
            pr.resolve(tmp_path, "Кромка")

    def test_exact_dirname_wins_over_ambiguous_label(self, tmp_path):
        """Имя каталога уникально в каталоге проектов — это точный адрес.

        Поэтому совпадение с ``dirname`` разрешается ДО проверки на
        неоднозначность ``label`` и не считается догадкой: у второго проекта
        каталог другой («Кромка (2)»), и спутать их нельзя.
        """
        first = pr.create_project(tmp_path, "Кромка")
        second = pr.create_project(tmp_path, "Кромка")
        assert first.dirname == "Кромка" and second.dirname == "Кромка (2)"
        assert pr.resolve(tmp_path, "Кромка").ref == first.ref
        assert pr.resolve(tmp_path, "Кромка (2)").ref == second.ref

    def test_unknown_and_empty_token(self, tmp_path):
        pr.create_project(tmp_path, "Кромка")
        assert pr.resolve(tmp_path, "нет такого") is None
        assert pr.resolve(tmp_path, "") is None

    def test_find_by_ref_ignores_malformed_token(self, tmp_path):
        pr.create_project(tmp_path, "Кромка")
        assert pr.find_by_ref(tmp_path, "Кромка") is None

    def test_require_identity_lists_available(self, tmp_path):
        ident = pr.create_project(tmp_path, "Кромка")
        with pytest.raises(ValueError, match=ident.short_ref()):
            pr.require_identity(tmp_path, pr.new_ref())


# ======================================================================
# 9. ПЕРЕПИСКА адресуется ссылкой (сам багфикс, ради которого всё)
# ======================================================================
class TestConversationFollowsRef:
    def test_conversation_survives_rename(self, tmp_path):
        """Главный сценарий: переименовали проект — диалог ТОТ ЖЕ."""
        ident = pr.create_project(tmp_path, "Кромка")
        s = store.load_session_by_ref(tmp_path, ident.ref)
        s.add_message("user", "почему верх мела 100 phr")
        store.save_session_by_ref(s, tmp_path, ident.ref)

        pr.rename_label(tmp_path, ident.ref, "Кромка ПВХ (жёсткая)")

        again = store.load_session_by_ref(tmp_path, ident.ref)
        assert [m.content for m in again.messages] \
            == ["почему верх мела 100 phr"]

    def test_session_carries_ref(self, tmp_path):
        ident = pr.create_project(tmp_path, "Кромка")
        s = store.load_session_by_ref(tmp_path, ident.ref)
        assert s.ref == ident.ref
        store.save_session_by_ref(s, tmp_path, ident.ref)
        state = json.loads(store.session_path_by_ref(
            tmp_path, ident.ref).read_text(encoding="utf-8"))
        assert state["ref"] == ident.ref

    def test_two_projects_with_same_label_keep_separate_conversations(
            self, tmp_path):
        """Одноимённые проекты не сливают переписку — ключ ссылка, не имя."""
        a = pr.create_project(tmp_path, "Кромка")
        b = pr.create_project(tmp_path, "Кромка")
        sa = store.load_session_by_ref(tmp_path, a.ref)
        sa.add_message("user", "вопрос A")
        store.save_session_by_ref(sa, tmp_path, a.ref)
        sb = store.load_session_by_ref(tmp_path, b.ref)
        sb.add_message("user", "вопрос B")
        store.save_session_by_ref(sb, tmp_path, b.ref)

        assert store.load_session_by_ref(
            tmp_path, a.ref).messages[0].content == "вопрос A"
        assert store.load_session_by_ref(
            tmp_path, b.ref).messages[0].content == "вопрос B"

    def test_logs_follow_ref_too(self, tmp_path):
        """Журналы (решения/аудит/L1) живут по той же ссылке."""
        ident = pr.create_project(tmp_path, "Кромка")
        store.append_log_by_ref(tmp_path, ident.ref, "decisions",
                                {"title": "мел до 100 phr"})
        pr.rename_label(tmp_path, ident.ref, "Кромка v2")
        recs = store.read_log_by_ref(tmp_path, ident.ref, "decisions")
        assert [r["title"] for r in recs] == ["мел до 100 phr"]

    def test_unknown_ref_refuses_with_explanation(self, tmp_path):
        pr.create_project(tmp_path, "Кромка")
        with pytest.raises(ValueError, match="нет в каталоге"):
            store.load_session_by_ref(tmp_path, pr.new_ref())

    def test_legacy_session_reads_with_empty_ref(self, tmp_path):
        """Проект до iter77: переписка читается, ссылка НЕ выдаётся молча."""
        conv = tmp_path / "старый" / "assistant"
        conv.mkdir(parents=True)
        s = new_session("старый")
        s.add_message("user", "старый вопрос")
        store.save_session(s, tmp_path, "старый")

        loaded = store.load_session(tmp_path, "старый")
        assert loaded.ref == "" and loaded.messages[0].content == "старый вопрос"
        state = json.loads((conv / "session.json").read_text(encoding="utf-8"))
        assert "ref" not in state, "чтение/запись не выдумывает ссылку"

    def test_after_migration_legacy_session_is_reachable_by_ref(self, tmp_path):
        (tmp_path / "старый" / "assistant").mkdir(parents=True)
        s = new_session("старый")
        s.add_message("user", "старый вопрос")
        store.save_session(s, tmp_path, "старый")

        [ident] = pr.migrate_root(tmp_path)
        again = store.load_session_by_ref(tmp_path, ident.ref)
        assert again.messages[0].content == "старый вопрос"
        assert again.ref == ident.ref, "ссылка подхвачена из каталога"

    def test_ref_of_dir_reports_absence_honestly(self, tmp_path):
        (tmp_path / "старый").mkdir()
        (tmp_path / "старый" / pr.CAMPAIGN_FILE).write_text("{}",
                                                           encoding="utf-8")
        assert store.ref_of_dir(tmp_path, "старый") == ""
        ident = pr.ensure_identity(tmp_path, "старый")
        assert store.ref_of_dir(tmp_path, "старый") == ident.ref


# ======================================================================
# 10. ДВИЖОК проекта тоже адресуется ссылкой (campaign_state)
# ======================================================================
class TestCampaignStateByRef:
    def test_not_started_project_is_listed_and_deletable(self, tmp_path):
        """Требование пользователя: проект живёт до удаления АДМИНОМ.

        Не стартовавший проект (есть только ссылка) виден в списке и убирается
        штатным удалением — иначе его нельзя ни подвязать, ни убрать.
        """
        ident = pr.ensure_default_project(tmp_path)
        assert cst.list_campaigns(tmp_path) == [ident.dirname]
        assert cst.campaign_exists_by_ref(tmp_path, ident.ref) is False
        assert cst.delete_campaign_by_ref(tmp_path, ident.ref) is True
        assert cst.list_campaigns(tmp_path) == []

    def test_setup_draft_by_ref_round_trip(self, tmp_path):
        """Данные можно вязать к проекту ДО сборки движка."""
        ident = pr.ensure_default_project(tmp_path)
        cst.save_setup_draft_by_ref(tmp_path, ident.ref,
                                    {"setup_components": "A, B, C"})
        assert cst.load_setup_draft_by_ref(tmp_path, ident.ref) \
            == {"setup_components": "A, B, C"}

    def test_draft_survives_rename(self, tmp_path):
        ident = pr.ensure_default_project(tmp_path)
        cst.save_setup_draft_by_ref(tmp_path, ident.ref,
                                    {"setup_components": "A, B"})
        pr.rename_label(tmp_path, ident.ref, "Кромка ПВХ")
        assert cst.load_setup_draft_by_ref(tmp_path, ident.ref) \
            == {"setup_components": "A, B"}

    def test_save_campaign_grants_ref(self, tmp_path):
        """Сохранение собранного проекта — тоже точка появления ссылки."""
        (tmp_path / "п").mkdir()
        cst.save_setup_draft(tmp_path, "п", {"setup_components": "A"})
        ident = pr.read_identity(tmp_path, "п")
        assert ident is not None and pr.is_ref(ident.ref)

    def test_list_projects_returns_identities(self, tmp_path):
        a = pr.create_project(tmp_path, "Кромка")
        idents = cst.list_projects(tmp_path)
        assert [i.ref for i in idents] == [a.ref]
        assert [i.label for i in idents] == ["Кромка"]

    def test_unknown_ref_is_refused_not_guessed(self, tmp_path):
        pr.create_project(tmp_path, "Кромка")
        with pytest.raises(ValueError, match="нет в каталоге"):
            cst.load_setup_draft_by_ref(tmp_path, pr.new_ref())

    def test_delete_refuses_foreign_dir(self, tmp_path):
        """Каталог без признаков проекта удалять нельзя (защита от промаха)."""
        (tmp_path / "просто_папка").mkdir()
        with pytest.raises(ValueError, match="не похож на проект"):
            cst.delete_campaign(tmp_path, "просто_папка")


# ======================================================================
# 11. MCP-сервер: ссылка как ключ, имя как подсказка
# ======================================================================
class TestMcpResolve:
    def test_resolve_accepts_ref_label_and_dirname(self, tmp_path):
        ident = pr.create_project(tmp_path, "Кромка ПВХ", dirname="pvc_edge")
        (tmp_path / "pvc_edge" / ct.STATE_FILE).write_text("{}",
                                                          encoding="utf-8")
        root = str(tmp_path)
        assert ct.resolve_project(ident.ref, root) == "pvc_edge"
        assert ct.resolve_project("Кромка ПВХ", root) == "pvc_edge"
        assert ct.resolve_project("pvc_edge", root) == "pvc_edge"

    def test_list_projects_reports_ref_and_label(self, tmp_path):
        ident = pr.create_project(tmp_path, "Кромка ПВХ", dirname="pvc_edge")
        (tmp_path / "pvc_edge" / ct.STATE_FILE).write_text("{}",
                                                          encoding="utf-8")
        [rec] = ct.list_projects(str(tmp_path))
        assert rec["project"] == "pvc_edge"
        assert rec["ref"] == ident.ref and rec["label"] == "Кромка ПВХ"

    def test_legacy_project_has_empty_ref_in_listing(self, tmp_path):
        (tmp_path / "старый").mkdir()
        (tmp_path / "старый" / ct.STATE_FILE).write_text("{}", encoding="utf-8")
        [rec] = ct.list_projects(str(tmp_path))
        assert rec["ref"] == "" and rec["label"] == "старый"

    def test_unknown_project_error_mentions_ref(self, tmp_path):
        (tmp_path / "п").mkdir()
        (tmp_path / "п" / ct.STATE_FILE).write_text("{}", encoding="utf-8")
        with pytest.raises(ToolError, match="prj_"):
            ct.resolve_project("нет такого", str(tmp_path))

    def test_ambiguous_label_is_refused(self, tmp_path):
        for d in ("a", "b"):
            pr.create_project(tmp_path, "Кромка", dirname=d)
            (tmp_path / d / ct.STATE_FILE).write_text("{}", encoding="utf-8")
        with pytest.raises(ToolError, match="Имя — не ключ"):
            ct.resolve_project("Кромка", str(tmp_path))


# ======================================================================
# 12. ДОК: сессия следует за ссылкой, а не за полем ввода
# ======================================================================
class TestDockUsesRef:
    def test_current_project_prefers_ref_over_name_field(self, tmp_path,
                                                         monkeypatch):
        """Сердце багфикса на уровне UI-логики (без Streamlit-прогона).

        В состоянии стоит ссылка проекта «pvc_edge» и ЧУЖОЕ имя в поле
        «campaign_name». Каталог должен быть выбран по ссылке.
        """
        ident = pr.create_project(tmp_path, "Кромка ПВХ", dirname="pvc_edge")
        monkeypatch.setattr(dock.st, "session_state",
                            {dock.K_REF: ident.ref,
                             "campaign_name": "совсем другое имя"},
                            raising=False)
        assert dock.current_project(str(tmp_path)) == "pvc_edge"
        assert dock.current_project_ref() == ident.ref

    def test_falls_back_to_name_when_no_ref(self, tmp_path, monkeypatch):
        """Совместимость: без ссылки в состоянии — прежнее поведение."""
        monkeypatch.setattr(dock.st, "session_state",
                            {"campaign_name": "старый"}, raising=False)
        assert dock.current_project(str(tmp_path)) == "старый"

    def test_unknown_ref_does_not_hijack_dir(self, tmp_path, monkeypatch):
        """Ссылка на удалённый проект не должна «съедать» имя из поля."""
        monkeypatch.setattr(dock.st, "session_state",
                            {dock.K_REF: pr.new_ref(),
                             "campaign_name": "старый"}, raising=False)
        assert dock.current_project(str(tmp_path)) == "старый"


# ======================================================================
# 13. ЖИВОЕ приложение (AppTest): переписка и правка имени
# ======================================================================
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
APP = os.path.join(_REPO, "src", "apps", "streamlit_app.py")
CAMPAIGN_ROOT = os.path.join(_REPO, "project_campaigns")


def test_default_project_has_ref_on_first_run():
    """Требование пользователя: дефолтный проект СРАЗУ со ссылкой."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    ref = at.session_state[dock.K_REF]
    assert pr.is_ref(ref), f"ссылка проекта не назначена: {ref!r}"
    ident = pr.find_by_ref(CAMPAIGN_ROOT, ref)
    assert ident is not None and ident.dirname == pr.DEFAULT_DIRNAME


def test_editing_name_field_does_not_switch_conversation():
    """САМ БАГ: правка имени в поле больше не переключает переписку."""
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    ref_before = at.session_state[dock.K_REF]
    project_before = at.session_state[dock.K_PROJECT]

    at.text_input(key="campaign_name").set_value("совсем другое имя").run()
    assert not at.exception
    assert at.session_state[dock.K_REF] == ref_before, "ссылка не менялась"
    assert at.session_state[dock.K_PROJECT] == project_before, \
        "переписка переключилась на другой каталог — это и был баг"
    # и предупреждения о «переключении переписки» тоже быть не должно
    assert dock.K_SWITCH_MSG not in at.session_state


def test_rename_button_changes_label_only():
    """Переименование меняет подпись, но не папку и не переписку.

    Работаем на СВОЁМ временном проекте, а не на дефолтном: тест не должен
    оставлять следов (в том числе в истории имён) в реальных проектах
    пользователя.
    """
    ident = pr.create_project(CAMPAIGN_ROOT, "iter77_rename_probe")
    try:
        at = AppTest.from_file(APP, default_timeout=300)
        at.session_state[dock.K_REF] = ident.ref
        at.session_state["campaign_name"] = ident.label
        at.run()
        assert not at.exception
        assert at.session_state[dock.K_PROJECT] == ident.dirname

        new_label = "Кромка ПВХ (проба iter77)"
        at.text_input(key="campaign_name").set_value(new_label).run()
        [b for b in at.button if b.key == "rename_campaign"][0].click().run()
        assert not at.exception

        after = pr.find_by_ref(CAMPAIGN_ROOT, ident.ref)
        assert after.label == new_label
        assert after.dirname == ident.dirname, "папка не двигается"
        assert after.label_history == [ident.label]
        # переписка осталась на том же каталоге
        assert at.session_state[dock.K_PROJECT] == ident.dirname
    finally:
        cst.delete_campaign_by_ref(CAMPAIGN_ROOT, ident.ref)
