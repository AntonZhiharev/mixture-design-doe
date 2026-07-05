"""Быстрая репродукция: seed-таблица UI при n_blocks_start, выставленном ПОСЛЕ сборки."""
import numpy as np
from src.apps.campaign_ui import build_setup_runner, seed_design_dataframe

r = build_setup_runner(
    mixture_names=["A", "B", "C"], process_names=["T", "P"],
    process_lower=[150.0, 1.0], process_upper=[200.0, 5.0],
    response_names=["strength", "gloss", "dry_time", "whiteStrength", "rho"],
    seed=1)
X = np.asarray(r.propose_seed(14, seed=1), float)

# как в render_seed_entry: значение виджета присваивается атрибуту раннера
r.n_blocks_start = 2
try:
    lab = r.seed_block_labels(X)
    print("seed_block_labels OK:", lab)
except Exception as exc:
    print("seed_block_labels FAIL:", type(exc).__name__, exc)

df = seed_design_dataframe(r, X)
print("columns:", list(df.columns))
print("Блок in columns:", "Блок" in df.columns)