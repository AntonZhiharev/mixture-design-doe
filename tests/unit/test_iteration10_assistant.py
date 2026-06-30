"""Iteration 10 / M9 — встроенный ИИ-ассистент приложения + мост в trace.

Проверяемый канон:
  * build_context отдаёт JSON-сериализуемый снимок страниц (метрики стадий,
    конфиг, флаги готовности) из «живого» PipelineRunner;
  * write_live_snapshot пишет тот же снимок в trace как обычный прогон, который
    читается MCP-логикой (queries.list_runs / run_overview / get_stage /
    get_design) — это и есть мост к Cline в VS Code;
  * build_messages формирует корректную структуру для LLM (system+контекст+
    история+вопрос);
  * call_llm честно падает с RuntimeError при отсутствии ключа (без сети).
"""
import json
import warnings

import numpy as np
import pytest

from src.apps.pipeline_runner import PipelineConfig, PipelineRunner
from src.apps import assistant as ai
from src.mcp import queries

warnings.filterwarnings("ignore")


def _runner(tmp_path, name="ai_proj"):
    cfg = PipelineConfig(q=3, model="linear", property_names=["A", "B"],
                         seed=1, n_restarts=2, noise_sd=0.1)
    r = PipelineRunner(cfg, tmp_path / name)
    r.run_m1()
    r.run_m2(simulate=True)
    r.run_m3_fit()
    return r


# ----------------------------------------------------------------------
def test_build_context_is_json_serializable_and_has_metrics(tmp_path):
    r = _runner(tmp_path)
    ctx = ai.build_context(r)
    # сериализуемость без падений (numpy → стандартные типы)
    json.dumps(ctx)
    assert ctx["config"]["q"] == 3
    assert ctx["config"]["property_names"] == ["A", "B"]
    assert "M1" in ctx["stages_done"] and "M2" in ctx["stages_done"]
    assert ctx["n_points"] > 0
    assert ctx["responses_filled"] is True
    # ключевые метрики стадий присутствуют
    assert "d_efficiency" in ctx["metrics"]["M2"]
    assert "A" in ctx["metrics"]["M3_fit"] and "r2" in ctx["metrics"]["M3_fit"]["A"]


def test_context_responses_not_filled_when_no_simulation(tmp_path):
    cfg = PipelineConfig(q=3, model="linear", property_names=["A"],
                         seed=1, n_restarts=2)
    r = PipelineRunner(cfg, tmp_path / "empty")
    r.run_m2(simulate=False)
    ctx = ai.build_context(r)
    assert ctx["n_points"] > 0
    assert ctx["responses_filled"] is False


def test_write_live_snapshot_readable_via_mcp_queries(tmp_path):
    r = _runner(tmp_path, name="bridge_proj")
    root = str(tmp_path / "trace")
    info = ai.write_live_snapshot(r, root=root)

    # прогон виден списком и карточкой
    runs = queries.list_runs(root)
    assert info["run_id"] in runs

    overview = queries.run_overview(root, info["run_id"])
    assert "M2" in overview["stages"]
    assert overview["meta"]["source"] == "streamlit_live"

    # дизайн M2 доступен через get_design (мост отдаёт точки)
    design = queries.get_design(root, info["run_id"], stage="M2")
    assert design["n_points"] == int(len(r.design))

    # метрики стадии читаются
    stage = queries.get_stage(root, info["run_id"], "M2")
    assert "d_efficiency" in stage["metrics"]


def test_build_messages_structure(tmp_path):
    r = _runner(tmp_path, name="msg_proj")
    ctx = ai.build_context(r)
    history = [{"role": "user", "content": "привет"},
               {"role": "assistant", "content": "здравствуйте"}]
    msgs = ai.build_messages(ctx, history, "что дальше?")
    assert msgs[0]["role"] == "system"            # системный промпт
    assert "контекст" in msgs[1]["content"].lower()  # контекст страниц
    assert msgs[-1] == {"role": "user", "content": "что дальше?"}
    # история сохранена между системными и последним вопросом
    roles = [m["role"] for m in msgs]
    assert roles.count("user") == 2 and roles.count("assistant") == 1


def test_call_llm_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_KEY", raising=False)
    assert ai.llm_available() is False
    with pytest.raises(RuntimeError):
        ai.call_llm([{"role": "user", "content": "x"}])


# ----------------------------------------------------------------------
# Контекст ЖИВОЙ формы сайдбара (ui.form) — ассистент «видит» интерфейс
# ----------------------------------------------------------------------
def test_form_context_captures_sidebar_fields():
    cfg = PipelineConfig(
        name="form_proj", q=3, model="quadratic", noise_sd=0.15, seed=7,
        names=["Вода", "Соль", "Сахар"], property_names=["A", "B"],
        lower=[0.1, 0.0, 0.0], upper=[0.8, 0.5, 0.5],
        comp_mode="parts", base_index=0, parts_min=[100.0, 0.0, 0.0],
        parts_max=[100.0, 10.0, 5.0], cost_coeffs=[1.0, 2.0, 3.0],
        cost_unit="₽/кг", batch_size=250.0, batch_unit="кг", n_blocks=2)
    fc = ai.form_context(cfg)
    form = fc["form"]
    assert form["name"] == "form_proj" and form["q"] == 3
    assert form["names"] == ["Вода", "Соль", "Сахар"]
    assert form["property_names"] == ["A", "B"]
    assert form["composition"]["mode"] == "parts"
    assert form["composition"]["base_index"] == 0
    assert form["composition"]["parts_max"] == [100.0, 10.0, 5.0]
    assert form["composition"]["lower"] == [0.1, 0.0, 0.0]
    assert form["cost_unit"] == "₽/кг" and form["cost_coeffs"] == [1.0, 2.0, 3.0]
    assert form["batch_size"] == 250.0 and form["batch_unit"] == "кг"
    assert form["n_blocks"] == 2
    # None-cfg → пустой словарь (ассистент просто не получит блок формы)
    assert ai.form_context(None) == {}


def test_build_context_nests_form_under_ui(tmp_path):
    r = _runner(tmp_path, name="ui_form_proj")
    fc = ai.form_context(r.cfg)
    ctx = ai.build_context(r, extra=fc)
    json.dumps(ctx)                       # сериализуемость не ломается
    assert ctx["ui"]["form"]["q"] == r.cfg.q
    assert ctx["ui"]["form"]["property_names"] == list(r.cfg.property_names)


def test_live_snapshot_carries_form_in_meta(tmp_path):
    r = _runner(tmp_path, name="snap_form_proj")
    fc = ai.form_context(r.cfg)
    root = str(tmp_path / "trace")
    info = ai.write_live_snapshot(r, root=root, form=fc)
    overview = queries.run_overview(root, info["run_id"])
    assert overview["meta"]["form"]["q"] == r.cfg.q
    assert overview["meta"]["form"]["property_names"] == list(r.cfg.property_names)


def test_context_includes_ui_guide_with_delete_button(tmp_path):
    """ui_guide описывает интерфейс, чтобы ассистент отвечал «где нажать».

    В частности — блок удаления проекта в сайдбаре (типовой вопрос).
    """
    r = _runner(tmp_path, name="guide_proj")
    ctx = ai.build_context(r)
    guide = ctx["ui_guide"]
    assert "sidebar" in guide and "tabs" in guide
    blob = json.dumps(guide, ensure_ascii=False).lower()
    assert "удалить проект" in blob and "пароль" in blob
    assert "benchmark" in blob and "ветки" in blob




