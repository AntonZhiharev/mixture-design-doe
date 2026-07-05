"""Headless-репродукция UI: меняем «Партий (блоков)» и смотрим таблицу seed."""
import numpy as np
from streamlit.testing.v1 import AppTest


def _script():
    import streamlit as st
    import numpy as np
    from src.apps import campaign_ui as ui
    from src.apps import campaign as cv

    if "campaign_ctrl" not in st.session_state:
        r = ui.build_setup_runner(
            mixture_names=["A", "B", "C"], process_names=["T", "P"],
            process_lower=[150.0, 1.0], process_upper=[200.0, 5.0],
            response_names=["strength", "gloss", "rho"], seed=1)
        st.session_state["campaign_ctrl"] = cv.CampaignController(r)
    ui.render_seed_entry(st.session_state["campaign_ctrl"])


at = AppTest.from_function(_script)
at.run(timeout=120)
print("exception:", at.exception)

# 1) предложить seed-дизайн
btn = [b for b in at.button if b.key == "setup_propose_seed"][0]
btn.click()
at.run(timeout=300)
print("after propose, exception:", at.exception)

df = at.session_state["setup_seed_df"] if "setup_seed_df" in at.session_state else None
print("columns nb=1:", list(df.columns) if df is not None else None)

# 2) поставить «Партий (блоков)» = 2
nb = [n for n in at.number_input if n.key == "setup_seed_blocks"][0]
nb.set_value(2)
at.run(timeout=300)
print("after nb=2, exception:", at.exception)

df = at.session_state["setup_seed_df"] if "setup_seed_df" in at.session_state else None
print("columns nb=2:", list(df.columns) if df is not None else None)
print("Блок in columns:", df is not None and "Блок" in df.columns)
print("runner.n_blocks_start:",
      at.session_state["campaign_ctrl"].runner.n_blocks_start)
