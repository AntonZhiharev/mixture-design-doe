# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Iteration 18 — ШАГ 5: мост кампании в ассистента и MCP (ТЗ v1.1, §16.1).

Проверяем, что per-branch роли и денежный канал ρ (И-5/Гр-1) реально доходят
до двух потребителей, а не остаются «в UI»:

  * :func:`campaign_assistant_overview` отдаёт дешёвую сводку (без econ-MC),
    но с честным занулённым/живым каналом ρ по веткам;
  * :func:`assistant.build_context` кладёт её в блок ``campaign`` контекста LLM;
  * :func:`assistant.write_live_snapshot` пишет стадию ``campaign`` в trace —
    её Cline в VS Code читает через MCP ``get_stage(run_id, "campaign")``.

Стартовый контраст демо-кампании: premium ρ=PRICE_INPUT (канал ALIVE,
``price_channel_suppressed=False``) vs rho_focus ρ=OPTIMIZED (канал ZEROED,
``price_channel_suppressed=True``).
"""
import warnings
from types import SimpleNamespace

import pytest
from sklearn.exceptions import ConvergenceWarning

from src.apps import assistant as ai
from src.apps import campaign as cv
from src.apps.campaign_ui import (build_demo_campaign_runner,
                                   campaign_assistant_overview)
from src.observability.trace import PipelineTrace

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _stub_runner():
    """Минимальный runner: build_context/write_live_snapshot не лезут в pipeline."""
    return SimpleNamespace(
        cfg=SimpleNamespace(name="campaign_mcp_test"),
        results={},
        state=SimpleNamespace(stage="campaign", models={}),
        design=None, y=None, Y=None, branches={})


@pytest.fixture(scope="module")
def overview():
    runner = build_demo_campaign_runner(seed=7, n_seed=12)
    ctrl = cv.CampaignController(runner)
    ov = campaign_assistant_overview(ctrl)
    return ov


def _by_id(ov):
    return {b["id"]: b for b in ov["branches"]}


# ======================================================================
# 1. Сводка для ассистента: дешёвая, но честная по каналу ρ (И-5)
# ======================================================================
def test_overview_carries_per_branch_money_channel(overview):
    assert overview is not None
    byid = _by_id(overview)
    assert {"premium", "rho_focus"} <= set(byid)
    # premium: ρ=PRICE_INPUT → канал ЖИВОЙ (не занулён)
    assert byid["premium"]["price_channel_suppressed"] is False
    # rho_focus: ρ=OPTIMIZED → канал ЗАНУЛЁН (двойной счёт δρ убран)
    assert byid["rho_focus"]["price_channel_suppressed"] is True


def test_overview_none_without_controller():
    # ctrl=None и пустой session_state → честный None, мост просто не добавит блок
    import src.apps.campaign_ui as cui
    cui.st.session_state.clear()
    assert campaign_assistant_overview() is None


# ======================================================================
# 2. build_context: сводка попадает в блок campaign контекста LLM
# ======================================================================
def test_build_context_includes_campaign(overview):
    ctx = ai.build_context(_stub_runner(), campaign=overview)
    assert "campaign" in ctx
    byid = _by_id(ctx["campaign"])
    assert byid["rho_focus"]["price_channel_suppressed"] is True
    assert byid["premium"]["price_channel_suppressed"] is False


def test_build_context_without_campaign_has_no_block():
    ctx = ai.build_context(_stub_runner(), campaign=None)
    assert "campaign" not in ctx


# ======================================================================
# 3. write_live_snapshot: стадия campaign видна через MCP-loader
# ======================================================================
def test_snapshot_logs_campaign_stage_for_mcp(tmp_path, overview):
    info = ai.write_live_snapshot(_stub_runner(), root=str(tmp_path),
                                  campaign=overview)
    tr = PipelineTrace.load(str(tmp_path), info["run_id"])
    assert "campaign" in tr.stages()
    ev = tr.get("campaign")
    # метрики стадии: 2 ветки, ровно одна с занулённым каналом (rho_focus)
    assert ev.metrics["n_branches"] == 2
    assert ev.metrics["n_price_suppressed"] == 1
    # сырьё атрибуции тоже доступно Cline (outputs == сводка кампании)
    byid = {b["id"]: b for b in ev.outputs["branches"]}
    assert byid["rho_focus"]["price_channel_suppressed"] is True
