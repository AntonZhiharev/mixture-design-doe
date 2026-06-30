# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 18 — удаление проекта под паролём (danger zone в UI).

Логика (без Streamlit):
  * delete_project удаляет валидный проект целиком и возвращает True;
  * для отсутствующего проекта — False (без исключения);
  * анти-traversal: пустое имя / '..' / разделители пути → ValueError;
  * каталог без state.json не удаляется (ValueError) — защита от сноса чужой папки;
  * admin-пароль: дефолт-константа, переопределение через DOE_ADMIN_PASSWORD,
    пустой ввод всегда неверен.
"""
import json

import pytest

from src.apps.pipeline_runner import delete_project, list_projects
from src.apps import admin


def _make_project(root, name):
    """Создать минимальный «проект»: каталог root/name со state.json."""
    pdir = root / name
    pdir.mkdir(parents=True)
    (pdir / "state.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    # немного содержимого, чтобы rmtree сносил непустой каталог
    (pdir / "data").mkdir()
    (pdir / "data" / "design.npy").write_bytes(b"\x00\x01")
    return pdir


def test_delete_existing_project(tmp_path):
    _make_project(tmp_path, "proj_a")
    _make_project(tmp_path, "proj_b")
    assert set(list_projects(tmp_path)) == {"proj_a", "proj_b"}

    assert delete_project(tmp_path, "proj_a") is True
    assert not (tmp_path / "proj_a").exists()
    assert list_projects(tmp_path) == ["proj_b"]


def test_delete_missing_project_returns_false(tmp_path):
    _make_project(tmp_path, "proj_a")
    assert delete_project(tmp_path, "nope") is False
    # существующий проект не тронут
    assert (tmp_path / "proj_a").exists()


@pytest.mark.parametrize("bad", ["", "  ", ".", "..", "a/b", "a\\b", "../x"])
def test_delete_rejects_bad_names(tmp_path, bad):
    with pytest.raises(ValueError):
        delete_project(tmp_path, bad)


def test_delete_refuses_non_project_dir(tmp_path):
    # каталог есть, но это НЕ проект (нет state.json) — удаление отклоняется
    (tmp_path / "not_a_project").mkdir()
    (tmp_path / "not_a_project" / "important.txt").write_text("keep me")
    with pytest.raises(ValueError):
        delete_project(tmp_path, "not_a_project")
    assert (tmp_path / "not_a_project" / "important.txt").exists()


def test_admin_password_default(monkeypatch):
    monkeypatch.delenv(admin.ADMIN_PASSWORD_ENV, raising=False)
    assert admin.admin_password() == admin.DEFAULT_ADMIN_PASSWORD
    assert admin.check_admin_password(admin.DEFAULT_ADMIN_PASSWORD) is True
    assert admin.check_admin_password("wrong") is False
    assert admin.check_admin_password("") is False
    assert admin.check_admin_password(None) is False


def test_admin_password_env_override(monkeypatch):
    monkeypatch.setenv(admin.ADMIN_PASSWORD_ENV, "s3cret")
    assert admin.admin_password() == "s3cret"
    assert admin.check_admin_password("s3cret") is True
    # дефолт больше не подходит при заданном env
    assert admin.check_admin_password(admin.DEFAULT_ADMIN_PASSWORD) is False


def test_admin_password_empty_env_falls_back(monkeypatch):
    # пустая env трактуется как «не задано» → дефолт
    monkeypatch.setenv(admin.ADMIN_PASSWORD_ENV, "")
    assert admin.admin_password() == admin.DEFAULT_ADMIN_PASSWORD
