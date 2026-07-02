# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Iteration 20 — campaign-native контекст ассистента (§17.6 salvage).

Первый шаг переноса ИИ-помощника на единый движок кампании (перед сносом M1…M8):
контекст строится ПРЯМО из сводки кампании (campaign_overview), без PipelineRunner
и стадий M1…M8. Проверяем: JSON-сериализуемо, несёт свойства/ветки/карту UI, и
подхватывается общим ``build_messages`` (тот же LLM-путь).
"""
import json

from src.apps import assistant as ai
from src.apps import campaign as cv
from src.apps.campaign_ui import build_demo_campaign_runner


def _overview(n_seed=8):
    runner = build_demo_campaign_runner(n_seed=n_seed)
    return runner, cv.CampaignController(runner).overview()


def test_campaign_context_is_jsonable_and_campaign_native():
    runner, ov = _overview()
    ctx = ai.build_campaign_context(ov)
    json.dumps(ctx)                              # сериализуемо (без падения)
    assert ctx["mode"] == "campaign"
    assert "strength" in ctx["property_names"]
    assert ctx["n_points"] == len(runner.points)
    assert {b["id"] for b in ctx["branches"]} == {"premium", "rho_focus"}
    # карта UI — campaign-native (поток §17), а НЕ M1…M8
    assert "flow" in ctx["ui_guide"]
    assert "stages_done" not in ctx           # стадий M1…M8 в режиме кампании нет


def test_campaign_context_builds_llm_messages():
    _, ov = _overview(n_seed=6)
    ctx = ai.build_campaign_context(ov)
    msgs = ai.build_messages(ctx, [], "что делать дальше?")
    assert msgs[0]["role"] == "system"
    assert "strength" in msgs[1]["content"]      # контекст вшит в system-JSON
    assert msgs[-1] == {"role": "user", "content": "что делать дальше?"}


def test_campaign_context_extra_ui_passthrough():
    _, ov = _overview(n_seed=6)
    ctx = ai.build_campaign_context(ov, extra={"note": "hi"})
    assert ctx["ui"] == {"note": "hi"}
