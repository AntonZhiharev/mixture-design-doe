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
    # cwd=HERE обязателен: проектные настройки Streamlit читаются из
    # `$CWD/.streamlit/config.toml` (`file_util.get_project_streamlit_file_path`
    # = `Path.cwd()/.streamlit`). Запуск лаунчера из другого каталога иначе
    # молча терял бы `client.toolbarMode` — и Ctrl+C снова открывал бы диалог
    # очистки кеша (iter93).
    return subprocess.call(cmd, cwd=HERE)


if __name__ == "__main__":
    raise SystemExit(main())
