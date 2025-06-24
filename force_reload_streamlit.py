"""
Force reload script to clear all Python module caches
Run this before starting Streamlit to ensure fresh code is loaded
"""

import sys
import os

# Clear all relevant modules from cache
modules_to_clear = [
    'enhanced_mixture_designs',
    'mixture_designs',
    'sequential_mixture_doe',
    'streamlit_app'
]

print("Clearing Python module cache...")
for module_name in modules_to_clear:
    if module_name in sys.modules:
        del sys.modules[module_name]
        print(f"Cleared {module_name}")

print("Cache cleared. Now run: streamlit run streamlit_app.py")
