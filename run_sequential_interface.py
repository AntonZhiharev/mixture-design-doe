"""
Sequential DOE Interface Launcher
==================================

Provides a CLI menu to launch:

  [1] Efficient Sequential Workflow (NEW — RECOMMENDED)
      ────────────────────────────────────────────────────
      Adaptive 3-phase strategy using Smart Simplex Centroid point generation.
      Saves 30-53% of experiments vs fixed JMP designs by stopping
      as soon as the model meets your R2 quality target.

        Phase 1  – Vertices x2 + all binary blends x1 + centroid   (~21 runs for q=5)
        Phase 2  – Ternary blends guided by Phase 1 significance    (+12 runs if needed)
        Phase 3  – Quaternary blends guided by Phase 2 significance (+8 runs if needed)
        Final    – Full diagnostics, model equation, export

      vs JMP fixed design: 45 runs (q=5)  |  saves 24+ runs = 53%
      vs old sequential:   27 runs (quadratic only, no 3-way+)

  [2] DOE Sequential Workflow (Classic)
      ─────────────────────────────────
      Full end-to-end workflow with D-optimal screening + fold-over:
        Stage 1 - Problem Setup (factors, bounds, model type)
        Stage 2 - Initial D-optimal screening design
        Stage 3 - Response data entry / CSV upload
        Stage 4 - Screening analysis with interactive factor selection
        Stage 5 - Fold-over / D-optimal augmentation for de-aliasing
        Stage 6 - Final competitive model

  [3] Sequential Regression Reconstruction (Original)
      ──────────────────────────────────────────────────
      Automated sequential reconstruction using synthetic responses.
      Useful for algorithm testing and convergence studies.

Usage
-----
  python run_sequential_interface.py            --> interactive menu
  python run_sequential_interface.py --efficient --> launch new efficient workflow
  python run_sequential_interface.py --new       --> launch classic sequential workflow
  python run_sequential_interface.py --old       --> launch original reconstruction
  python run_sequential_interface.py --port 8502 --efficient

Benchmark
---------
  python run_efficient_workflow.py  --> standalone benchmark vs true model
                                        (no Streamlit needed)
"""

import subprocess
import sys
import os


# -----------------------------------------------------------------------------
# App paths
# -----------------------------------------------------------------------------

EFFICIENT_APP_PATH = os.path.join("src", "apps", "efficient_sequential_workflow_app.py")
NEW_APP_PATH       = os.path.join("src", "apps", "doe_sequential_workflow_app.py")
OLD_APP_PATH       = os.path.join("src", "apps", "sequential_reconstruction_app.py")

APP_INFO = {
    "efficient": {
        "path": EFFICIENT_APP_PATH,
        "title": "Efficient Sequential Workflow (NEW -- RECOMMENDED)",
        "description": (
            "Adaptive 3-phase strategy using Smart Simplex Centroid point generation:\n"
            "  Phase 1: Vertices x2 + all binary blends x1 + centroid   (~21 runs q=5)\n"
            "  Phase 2: Ternary blends (guided by Phase 1 significance)  (+12 runs)\n"
            "  Phase 3: Quaternary blends (guided by Phase 2)            (+8  runs)\n"
            "  Stops as soon as model achieves your R2 target\n"
            "  Saves 30-53% runs vs JMP fixed design | Full Scheffe model support"
        ),
        "default_port": 8501,
    },
    "new": {
        "path": NEW_APP_PATH,
        "title": "DOE Sequential Workflow (Classic)",
        "description": (
            "Guides you step-by-step with D-optimal + fold-over:\n"
            "  Stage 1-2: D-optimal initial screening design (factorial or mixture)\n"
            "  Stage 3:   Enter real experimental responses\n"
            "  Stage 4:   ANOVA, half-normal, Pareto, interactive factor selection\n"
            "  Stage 5:   Fold-over / D-optimal augmentation for de-aliasing\n"
            "  Stage 6:   Final model (R2, LOF, residuals, response surface, export)"
        ),
        "default_port": 8502,
    },
    "old": {
        "path": OLD_APP_PATH,
        "title": "Sequential Regression Reconstruction (Original)",
        "description": (
            "Automated sequential reconstruction using synthetic responses:\n"
            "  Generates design + synthetic responses automatically\n"
            "  Iteratively adds points until convergence\n"
            "  Stage-by-stage coefficient recovery analysis\n"
            "  Useful for algorithm benchmarking"
        ),
        "default_port": 8503,
    },
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _print_banner():
    print()
    print("=" * 68)
    print("  DOE SEQUENTIAL INTERFACE LAUNCHER")
    print("=" * 68)


def _print_app_info(key: str, number: int):
    info = APP_INFO[key]
    print(f"\n  [{number}]  {info['title']}")
    for line in info["description"].splitlines():
        print(f"        {line}")
    print(f"        Default port : {info['default_port']}")


def _launch(app_key: str, port: int = None):
    info = APP_INFO[app_key]
    resolved_port = port or info["default_port"]
    app_path = info["path"]

    if not os.path.exists(app_path):
        print(f"\n  App not found at: {app_path}")
        print("  Make sure you are running this script from the project root.")
        sys.exit(1)

    print(f"\n  Launching: {info['title']}")
    print(f"  File    : {app_path}")
    print(f"  URL     : http://localhost:{resolved_port}")
    print("-" * 68)
    print("Press Ctrl+C to stop the server.\n")

    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(resolved_port),
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n  Server stopped by user.")
    except FileNotFoundError:
        print("\n  streamlit not found.  Install it with:")
        print("  pip install streamlit")
    except Exception as exc:
        print(f"\n  Error: {exc}")


def _interactive_menu(port: int = None):
    _print_banner()
    print()
    print("  Choose which interface to launch:\n")
    _print_app_info("efficient", 1)
    _print_app_info("new",       2)
    _print_app_info("old",       3)
    print()
    print("  [b]  Run benchmark (no Streamlit) -- python run_efficient_workflow.py")
    print("  [q]  Quit")
    print()

    choices = {
        "1": "efficient", "efficient": "efficient",
        "2": "new",       "new": "new",
        "3": "old",       "old": "old",
    }

    while True:
        choice = input("  Enter selection (1/2/3/b/q): ").strip().lower()
        if choice in choices:
            _launch(choices[choice], port)
            break
        elif choice == "b":
            print("\n  Running standalone benchmark...\n")
            try:
                subprocess.run([sys.executable, "run_efficient_workflow.py"])
            except Exception as e:
                print(f"  Error: {e}")
            break
        elif choice in ("q", "quit", "exit"):
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice. Enter 1, 2, 3, b or q.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    # Parse --port N
    port = None
    if "--port" in args:
        idx = args.index("--port")
        try:
            port = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        except (IndexError, ValueError):
            print("  --port requires an integer argument. Using default port.")

    # Flag-based direct launch
    if "--efficient" in args:
        _launch("efficient", port)
    elif "--new" in args:
        _launch("new", port)
    elif "--old" in args:
        _launch("old", port)
    else:
        _interactive_menu(port)


if __name__ == "__main__":
    main()
