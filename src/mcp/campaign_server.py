"""iter66 — MCP-сервер `doe-campaign` (ASSISTANT_SPEC).

Даёт Cline в VS Code ТЕ ЖЕ read-only инструменты кампании, что видит технолог
в доке ассистента: `get_spec`, `explain_node`, `validate_spec` (dry-run),
`simulate_bounds`, `preflight`, `point_report`, `encode_recipe`, `get_runs`,
`campaign_overview`, `get_local_facts`, `get_decisions`, вложения. Числа
считает одна и та же реализация ядра (`src/assistant/tools`), поэтому «в
приложении одно, у Cline другое» невозможно по построению.

Класс `write` НЕ экспортируется: применение патча, запись решения и L1-факта —
акт ЧЕЛОВЕКА кнопкой в интерфейсе (разовый токен, iter63). Класс `propose`
(патч в стейдж чужой сессии) и `sandbox` (исполнение кода) тоже не выдаются.

Запуск (stdio, как MCP-сервер):
    python src/mcp/campaign_server.py

Самопроверка без MCP-транспорта (для CI/отладки):
    python src/mcp/campaign_server.py --selftest [проект]

Каталог кампаний берётся из переменной окружения DOE_CAMPAIGN_ROOT,
по умолчанию <repo>/project_campaigns.
"""
from __future__ import annotations

import json
import os
import sys

# --- repo root в sys.path (сервер запускается извне рабочего каталога) ---
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.mcp import campaign_tools as ct  # noqa: E402

SERVER_NAME = "doe-campaign"


# ======================================================================
# Самопроверка (не требует пакета mcp)
# ======================================================================
def _selftest(project: str = "") -> int:
    root = ct.campaign_root()
    print(f"[{SERVER_NAME}] CAMPAIGN_ROOT = {root}")
    projects = ct.list_projects()
    print(f"[{SERVER_NAME}] проектов: {len(projects)}")
    for d in projects:
        print(f"    • {d['project']}: campaign={'да' if d['has_campaign'] else 'нет'}"
              f", сессия={'да' if d['has_session'] else 'нет'}")
    print(f"[{SERVER_NAME}] экспортируется инструментов: "
          f"{len(ct.exported_names())} → {ct.exported_names()}")
    print(f"[{SERVER_NAME}] НЕ экспортируется (write/propose/sandbox): "
          f"{ct.hidden_names()}")
    if not projects:
        print(f"[{SERVER_NAME}] нет сохранённых кампаний — сохраните проект "
              f"в приложении или задайте DOE_CAMPAIGN_ROOT")
        return 0
    name = project or projects[0]["project"]
    print(f"[{SERVER_NAME}] project_status({name}):")
    print("    " + json.dumps(ct.project_status(name), ensure_ascii=False)[:600])
    # Класс write наружу не выходит — показываем отказ словами (A0.6).
    refused = ct.call_tool(name, "apply_patch", {"patch_id": "p_1"})
    print(f"[{SERVER_NAME}] apply_patch → ok={refused['ok']}: "
          f"{refused['error'][:120]}…")
    print(f"[{SERVER_NAME}] selftest OK")
    return 0


# ======================================================================
# MCP-сервер
# ======================================================================
def build_server():
    """FastMCP-сервер с read-only инструментами кампании. Импорт mcp — ленивый."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(SERVER_NAME)

    # ---- resources -----------------------------------------------------
    @mcp.resource("campaign://projects")
    def projects_resource() -> str:
        return json.dumps(ct.list_projects(), ensure_ascii=False, indent=2)

    @mcp.resource("campaign://{project}/status")
    def status_resource(project: str) -> str:
        return json.dumps(ct.project_status(project), ensure_ascii=False,
                          indent=2)

    # ---- служебные инструменты сервера ---------------------------------
    @mcp.tool()
    def list_projects() -> list:
        """Проекты-кампании каталога: есть ли движок и переписка ассистента."""
        return ct.list_projects()

    @mcp.tool()
    def project_status(project: str = "") -> dict:
        """Состояние проекта: метка, свойства, точки, ветки, spec_hash, сессия.

        Проект можно не указывать, если он в каталоге один.
        """
        return ct.project_status(project)

    @mcp.tool()
    def list_tools() -> list:
        """Каталог read-only инструментов кампании (класс write не выдаётся)."""
        return ct.tool_catalog()

    # ---- инструменты ядра: обёртки ГЕНЕРИРУЮТСЯ из реестра --------------
    for name, fn in ct.build_wrappers().items():
        try:
            mcp.add_tool(fn, name=name, description=fn.__doc__ or "")
        except TypeError:                       # старые версии FastMCP
            mcp.tool()(fn)

    return mcp


def main() -> int:
    if "--selftest" in sys.argv:
        rest = [a for a in sys.argv[1:] if not a.startswith("-")]
        return _selftest(rest[0] if rest else "")
    server = build_server()
    server.run()          # stdio transport
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
