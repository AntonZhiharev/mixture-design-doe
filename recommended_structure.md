# Recommended Project Structure for Mixture Design DOE

## Folder Organization

```
mixture-design-doe/
│
├── .venv/                    # Virtual environment (excluded from search)
├── .vscode/                  # VS Code settings
│   └── settings.json         # Workspace-specific settings
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── core/                 # Core functionality
│   │   ├── __init__.py
│   │   ├── base_doe.py
│   │   ├── mixture_base.py
│   │   └── refactored_mixture_design.py
│   │
│   ├── algorithms/           # Optimization algorithms
│   │   ├── __init__.py
│   │   ├── mixture_algorithms.py
│   │   └── optimization_strategies.py
│   │
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   └── mixture_utils.py
│   │
│   └── apps/                 # Applications
│       ├── __init__.py
│       ├── streamlit_app.py
│       └── cli_app.py
│
├── tests/                    # Test files
│   ├── __init__.py
│   ├── test_refactored_design.py
│   ├── test_d_efficiency.py
│   └── test_integration.py
│
├── examples/                 # Example scripts
│   ├── basic_usage.py
│   ├── advanced_features.py
│   └── comparison_study.py
│
├── docs/                     # Documentation
│   ├── getting_started.md
│   ├── api_reference.md
│   └── theory.md
│
├── output/                   # Generated files (gitignored)
├── data/                     # Data files (gitignored)
│
├── requirements.txt          # Python dependencies
├── requirements-dev.txt      # Development dependencies
├── setup.py                  # Package setup
├── README.md                 # Project documentation
├── .gitignore               # Git ignore file
└── pyproject.toml           # Modern Python project config
```

## Benefits of This Structure:

1. **Clean separation**: Source code, tests, and examples are separated
2. **Virtual environment isolation**: `.venv/` is at the root but excluded from searches
3. **Modular design**: Code is organized by functionality
4. **Easy imports**: With proper `__init__.py` files, imports are clean
5. **Professional structure**: Follows Python packaging best practices

## VS Code Configuration

Create `.vscode/settings.json` to exclude venv from searches and optimize the workspace.

## Python Path Configuration

With this structure, you can:
- Run tests from the project root: `python -m pytest tests/`
- Import modules cleanly: `from src.core.refactored_mixture_design import MixtureDesign`
- Keep virtual environment separate from code
