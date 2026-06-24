"""Лаунчер Streamlit-приложения pipeline M1–M8.

Запускает `streamlit run src/apps/streamlit_app.py` из корня репозитория.

Использование:
    python run_streamlit_app.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "src", "apps", "streamlit_app.py")


def main() -> int:
    cmd = [sys.executable, "-m", "streamlit", "run", APP]
    print("Запуск:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
