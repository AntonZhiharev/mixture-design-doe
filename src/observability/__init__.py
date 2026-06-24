"""M9 — слой наблюдаемости (observability).

PipelineTrace собирает структурные события стадий pipeline (M1..M8 / раунды
active learning / финальная оптимизация / benchmark) и сериализует их в JSON.
Не инвазивен и read-only снаружи: только логирует уже посчитанные величины,
не влияя на решения pipeline (источник истины алгоритма — ProjectState).
"""
from src.observability.trace import StageEvent, PipelineTrace, list_runs

__all__ = ["StageEvent", "PipelineTrace", "list_runs"]
