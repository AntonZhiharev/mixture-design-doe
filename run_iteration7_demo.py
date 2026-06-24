"""
run_iteration7_demo.py — M9 (наблюдаемость): прогон одного мира benchmark'а
с включённым PipelineTrace, запись trace на диск и краткий разбор стадий.

Запуск:
    python run_iteration7_demo.py            # seed по умолчанию
    python run_iteration7_demo.py 123        # свой seed
"""
from __future__ import annotations

import sys

from src.observability.trace import PipelineTrace, list_runs
from run_pipeline_benchmark import run_trial


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    root = "project_demo/trace"
    run_id = f"bench_seed{seed}"

    trace = PipelineTrace(run_id=run_id, root=root,
                          meta={"source": "run_pipeline_benchmark", "seed": seed})
    print("=" * 72)
    print(f"ITERATION 7 — M9 PipelineTrace demo (run_id={run_id})")
    print("=" * 72)

    res = run_trial(seed, verbose=False, trace=trace)
    run_dir = trace.save()

    print(f"\n  trace сохранён в: {run_dir}")
    print(f"  стадии: {trace.stages()}")
    print("\n  метрики по стадиям:")
    for stage, metrics in trace.metrics_summary().items():
        short = {k: metrics[k] for k in list(metrics)[:4]}
        print(f"    {stage:14s} {short}")

    # round-trip: загрузим обратно с диска
    reloaded = PipelineTrace.load(root, run_id)
    bench = reloaded.get("benchmark").metrics if "benchmark" in reloaded.stages() else {}
    print("\n  round-trip load OK; benchmark:")
    print(f"    meets={bench.get('meets')}  price_gap={bench.get('price_gap_pct'):+.1f}%  "
          f"n_exp={bench.get('n_exp')}")
    print(f"\n  прогоны в {root}: {list_runs(root)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
