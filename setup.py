"""
Setup configuration for Mixture Design DOE package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mixture-design-doe",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-efficiency mixture design of experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mixture-design-doe",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "matplotlib>=3.3",
        "seaborn>=0.11",
        "scipy>=1.7",
        "scikit-learn>=0.24",
        "streamlit>=1.0",
        "openpyxl>=3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.9",
        ],
        "viz": [
            "python-ternary>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mixture-design=src.apps.cli_app:main",
        ],
    },
)
