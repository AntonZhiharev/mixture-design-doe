"""M9 — чистая логика интроспекции trace (без зависимости от пакета `mcp`).

Все функции принимают `root` (каталог trace) и возвращают JSON-сериализуемые
структуры, читая сохранённые прогоны через PipelineTrace.load. Используется как
MCP-сервером (introspect_server.py), так и в тестах напрямую.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.observability.trace import PipelineTrace, list_runs as _list_runs

DEFAULT_ROOT = "project_demo/trace"


# ----------------------------------------------------------------------
def list_runs(root: str = DEFAULT_ROOT) -> List[str]:
    """Список доступных run_id."""
    return _list_runs(root)


def run_overview(root: str, run_id: str) -> Dict[str, Any]:
    """Краткая карточка прогона: meta, список стадий, итог benchmark."""
    tr = PipelineTrace.load(root, run_id)
    stages = tr.stages()
    overview: Dict[str, Any] = {"run_id": tr.run_id, "meta": tr.meta,
                                "stages": stages, "n_stages": len(stages)}
    if "benchmark" in stages:
        overview["benchmark"] = tr.get("benchmark").metrics
    al = [s for s in stages if s.startswith("AL_round_")]
    overview["n_al_rounds"] = len(al)
    return overview


def get_stage(root: str, run_id: str, stage: str) -> Dict[str, Any]:
    """Полное событие стадии (inputs/outputs/metrics/diagnostics)."""
    tr = PipelineTrace.load(root, run_id)
    if stage not in tr.stages():
        raise KeyError(f"stage '{stage}' not in run '{run_id}'; "
                       f"available: {tr.stages()}")
    return tr.get(stage).to_dict()


def get_metrics(root: str, run_id: str) -> Dict[str, Dict[str, Any]]:
    """Сводка метрик по всем стадиям: {stage: metrics}."""
    return PipelineTrace.load(root, run_id).metrics_summary()


def get_design(root: str, run_id: str, stage: str = "M2") -> Dict[str, Any]:
    """Точки дизайна со стадии (по умолчанию стартовый D-оптимальный M2)."""
    ev = PipelineTrace.load(root, run_id).get(stage)
    design = ev.outputs.get("design")
    return {"run_id": run_id, "stage": stage,
            "n_points": (len(design) if design is not None else 0),
            "design": design, "metrics": ev.metrics}


def al_progression(root: str, run_id: str) -> List[Dict[str, Any]]:
    """Прогресс active-learning раундов: ключевые метрики по порядку."""
    tr = PipelineTrace.load(root, run_id)
    rounds = []
    i = 0
    while f"AL_round_{i}" in tr.stages():
        m = tr.get(f"AL_round_{i}").metrics
        rounds.append({"round": i,
                       "n_exp": m.get("n_exp"),
                       "d_overall": m.get("d_overall"),
                       "cost_star": m.get("cost_star"),
                       "sigma_mean": m.get("sigma_mean"),
                       "accepted": m.get("accepted"),
                       "incumbent_cost": m.get("incumbent_cost")})
        i += 1
    return rounds


def diff_rounds(root: str, run_id: str, a: int, b: int) -> Dict[str, Any]:
    """Сравнение двух AL-раундов: дельты метрик и сдвиг рецептуры x*."""
    import numpy as np

    tr = PipelineTrace.load(root, run_id)
    ea, eb = tr.get(f"AL_round_{a}"), tr.get(f"AL_round_{b}")
    ma, mb = ea.metrics, eb.metrics

    def delta(key):
        va, vb = ma.get(key), mb.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            return vb - va
        return None

    out: Dict[str, Any] = {
        "run_id": run_id, "a": a, "b": b,
        "delta": {k: delta(k) for k in
                  ("n_exp", "d_overall", "cost_star", "sigma_mean")},
    }
    xa, xb = ea.outputs.get("x_star"), eb.outputs.get("x_star")
    if xa is not None and xb is not None:
        out["recipe_shift_l2"] = float(np.linalg.norm(np.array(xb) - np.array(xa)))
        out["x_star_a"], out["x_star_b"] = xa, xb
    return out


def get_benchmark(root: str, run_id: str) -> Dict[str, Any]:
    """Итог benchmark: метрики + спецификации (требование vs pipeline)."""
    ev = PipelineTrace.load(root, run_id).get("benchmark")
    return {"run_id": run_id, "metrics": ev.metrics,
            "specs": ev.diagnostics.get("specs", []),
            "recipes": {"analytical": ev.outputs.get("recipe_analytical"),
                        "pipeline": ev.outputs.get("recipe_pipeline")}}
