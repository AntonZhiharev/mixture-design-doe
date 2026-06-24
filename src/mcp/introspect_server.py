"""M9 — MCP-сервер `doe-introspect`.

Read-only интроспекция сохранённых прогонов pipeline (trace, см. src/observability).
Тонкая обёртка FastMCP поверх чистой логики из `src/mcp/queries.py`.

Запуск (stdio, как MCP-сервер):
    python src/mcp/introspect_server.py

Самопроверка без MCP-транспорта (для CI/отладки):
    python src/mcp/introspect_server.py --selftest

Каталог trace берётся из переменной окружения DOE_TRACE_ROOT,
по умолчанию <repo>/project_demo/trace.
"""
from __future__ import annotations

import os
import sys

# --- repo root в sys.path (сервер запускается извне рабочего каталога) ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.mcp import queries  # noqa: E402

TRACE_ROOT = os.environ.get("DOE_TRACE_ROOT",
                            os.path.join(_REPO_ROOT, "project_demo", "trace"))


# ======================================================================
# Самопроверка (не требует пакета mcp)
# ======================================================================
def _selftest() -> int:
    print(f"[doe-introspect] TRACE_ROOT = {TRACE_ROOT}")
    runs = queries.list_runs(TRACE_ROOT)
    print(f"[doe-introspect] runs: {runs}")
    if not runs:
        print("[doe-introspect] нет сохранённых прогонов — запустите "
              "run_iteration7_demo.py")
        return 0
    rid = runs[0]
    ov = queries.run_overview(TRACE_ROOT, rid)
    print(f"[doe-introspect] overview[{rid}]: stages={ov['n_stages']}, "
          f"al_rounds={ov['n_al_rounds']}, benchmark={ov.get('benchmark')}")
    prog = queries.al_progression(TRACE_ROOT, rid)
    print(f"[doe-introspect] al_progression: {len(prog)} rounds")
    print("[doe-introspect] selftest OK")
    return 0


# ======================================================================
# MCP-сервер
# ======================================================================
def build_server():
    """Создаёт FastMCP-сервер с tools/resources. Импорт mcp — ленивый."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("doe-introspect")

    # ---- resources -----------------------------------------------------
    @mcp.resource("doe://runs")
    def runs_resource() -> str:
        import json
        return json.dumps(queries.list_runs(TRACE_ROOT), ensure_ascii=False)

    @mcp.resource("doe://{run_id}/overview")
    def overview_resource(run_id: str) -> str:
        import json
        return json.dumps(queries.run_overview(TRACE_ROOT, run_id),
                          ensure_ascii=False, indent=2)

    # ---- tools ---------------------------------------------------------
    @mcp.tool()
    def list_runs() -> list:
        """Список сохранённых прогонов pipeline (run_id)."""
        return queries.list_runs(TRACE_ROOT)

    @mcp.tool()
    def run_overview(run_id: str) -> dict:
        """Карточка прогона: meta, список стадий, итог benchmark."""
        return queries.run_overview(TRACE_ROOT, run_id)

    @mcp.tool()
    def get_stage(run_id: str, stage: str) -> dict:
        """Полное событие стадии (inputs/outputs/metrics/diagnostics).

        stage: 'setup' | 'M2' | 'AL_round_{i}' | 'opt' | 'benchmark'.
        """
        return queries.get_stage(TRACE_ROOT, run_id, stage)

    @mcp.tool()
    def get_metrics(run_id: str) -> dict:
        """Сводка метрик по всем стадиям прогона."""
        return queries.get_metrics(TRACE_ROOT, run_id)

    @mcp.tool()
    def get_design(run_id: str, stage: str = "M2") -> dict:
        """Точки дизайна со стадии (по умолчанию стартовый D-оптимальный M2)."""
        return queries.get_design(TRACE_ROOT, run_id, stage)

    @mcp.tool()
    def al_progression(run_id: str) -> list:
        """Прогресс active-learning раундов (d_overall, cost*, sigma, ...)."""
        return queries.al_progression(TRACE_ROOT, run_id)

    @mcp.tool()
    def diff_rounds(run_id: str, a: int, b: int) -> dict:
        """Сравнить два AL-раунда: дельты метрик и сдвиг рецептуры x*."""
        return queries.diff_rounds(TRACE_ROOT, run_id, a, b)

    @mcp.tool()
    def get_benchmark(run_id: str) -> dict:
        """Итог benchmark: аналитический оптимум vs pipeline + спецификации."""
        return queries.get_benchmark(TRACE_ROOT, run_id)

    return mcp


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()
    server = build_server()
    server.run()          # stdio transport
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
