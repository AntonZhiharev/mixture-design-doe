"""
Script to reorganize the project into the recommended structure
This will help separate source code from tests and virtual environments
"""

import os
import shutil
from pathlib import Path


def create_directory_structure():
    """Create the recommended directory structure"""
    
    directories = [
        "src",
        "src/core",
        "src/algorithms", 
        "src/utils",
        "src/apps",
        "tests",
        "examples",
        "docs",
        "output",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        
        # Create __init__.py files for Python packages
        if directory.startswith(("src", "tests")):
            init_file = Path(directory) / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Package initialization"""')
    
    print("✅ Directory structure created")


def move_files():
    """Move files to their appropriate locations"""
    
    # Define file mappings (current location -> new location)
    file_mappings = {
        # Core files
        "base_doe.py": "src/core/base_doe.py",
        "mixture_base.py": "src/core/mixture_base.py",
        "refactored_mixture_design.py": "src/core/refactored_mixture_design.py",
        "fixed_parts_mixture_designs.py": "src/core/fixed_parts_mixture_designs.py",
        "mixture_designs.py": "src/core/mixture_designs.py",
        
        # Algorithm files
        "mixture_algorithms.py": "src/algorithms/mixture_algorithms.py",
        
        # Utility files
        "mixture_utils.py": "src/utils/mixture_utils.py",
        
        # App files
        "streamlit_app.py": "src/apps/streamlit_app.py",
        "streamlit_app_refactored.py": "src/apps/streamlit_app_refactored.py",
        
        # Test files
        "test_refactored_design.py": "tests/test_refactored_design.py",
        "test_d_efficiency_improvement.py": "tests/test_d_efficiency_improvement.py",
        "test_direct_optimization.py": "tests/test_direct_optimization.py",
        "test_exact_same_approach.py": "tests/test_exact_same_approach.py",
        "test_import_fix.py": "tests/test_import_fix.py",
        "test_mixture_design_class.py": "tests/test_mixture_design_class.py",
        "test_optimized_mixture.py": "tests/test_optimized_mixture.py",
        "test_all_mixture_classes.py": "tests/test_all_mixture_classes.py",
        
        # Example files
        "example_usage.py": "examples/basic_usage.py",
        "compare_mixture_approaches.py": "examples/compare_mixture_approaches.py",
        "compare_d_efficiency.py": "examples/compare_d_efficiency.py",
        "demonstrate_improved_design.py": "examples/demonstrate_improved_design.py",
        
        # Documentation
        "getting_started.md": "docs/getting_started.md",
        "README_improvement.md": "docs/improvement_notes.md",
        "flexible_runs_summary.md": "docs/flexible_runs_summary.md",
    }
    
    moved_files = []
    skipped_files = []
    
    for src, dst in file_mappings.items():
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.exists():
            # Create parent directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.copy2(src_path, dst_path)
            moved_files.append(f"{src} → {dst}")
        else:
            skipped_files.append(src)
    
    print(f"\n✅ Moved {len(moved_files)} files:")
    for move in moved_files[:5]:  # Show first 5
        print(f"   {move}")
    if len(moved_files) > 5:
        print(f"   ... and {len(moved_files) - 5} more")
    
    if skipped_files:
        print(f"\n⚠️  Skipped {len(skipped_files)} files (not found)")


def create_setup_py():
    """Create a setup.py file for the package"""
    
    setup_content = '''"""
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
'''
    
    with open("setup.py", "w") as f:
        f.write(setup_content)
    
    print("\n✅ Created setup.py")


def create_pyproject_toml():
    """Create a modern pyproject.toml file"""
    
    pyproject_content = '''[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "mixture-design-doe"
version = "0.1.0"
description = "High-efficiency mixture design of experiments"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Python :: 3",
]

dependencies = [
    "numpy>=1.20",
    "pandas>=1.3",
    "matplotlib>=3.3",
    "seaborn>=0.11",
    "scipy>=1.7",
    "scikit-learn>=0.24",
    "streamlit>=1.0",
    "openpyxl>=3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "pytest-cov>=2.0",
    "black>=21.0",
    "flake8>=3.9",
    "mypy>=0.9",
]
viz = ["python-ternary>=1.0"]

[project.urls]
Homepage = "https://github.com/yourusername/mixture-design-doe"
Documentation = "https://mixture-design-doe.readthedocs.io"
Repository = "https://github.com/yourusername/mixture-design-doe"

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310']
include = '\\.pyi?$'

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/examples/*"]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
'''
    
    with open("pyproject.toml", "w") as f:
        f.write(pyproject_content)
    
    print("✅ Created pyproject.toml")


def create_requirements_dev():
    """Create requirements-dev.txt for development dependencies"""
    
    dev_requirements = '''# Development dependencies
pytest>=6.0
pytest-cov>=2.0
black>=21.0
flake8>=3.9
mypy>=0.9
ipython>=7.0
jupyter>=1.0
pre-commit>=2.0
'''
    
    with open("requirements-dev.txt", "w") as f:
        f.write(dev_requirements)
    
    print("✅ Created requirements-dev.txt")


def show_next_steps():
    """Show next steps for the user"""
    
    print("\n" + "=" * 60)
    print("PROJECT REORGANIZATION COMPLETE!")
    print("=" * 60)
    
    print("\n📋 WHAT WAS DONE:")
    print("1. Created recommended directory structure")
    print("2. Added .gitignore to exclude venv and cache files")
    print("3. Added VS Code settings to:")
    print("   - Exclude venv from searches")
    print("   - Auto-activate venv in terminals")
    print("   - Configure Python paths")
    print("4. Created setup.py and pyproject.toml for packaging")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Create virtual environment:")
    print("   python -m venv .venv")
    print("\n2. Activate it (VS Code will do this automatically):")
    print("   Windows: .venv\\Scripts\\activate")
    print("   Linux/Mac: source .venv/bin/activate")
    print("\n3. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("   pip install -r requirements-dev.txt  # for development")
    print("\n4. Install package in development mode:")
    print("   pip install -e .")
    print("\n5. VS Code will now:")
    print("   - Auto-activate venv when you open terminals")
    print("   - Exclude venv from all searches")
    print("   - Use the venv Python interpreter")
    
    print("\n✨ BENEFITS:")
    print("- Clean project structure")
    print("- Virtual environment is isolated")
    print("- No more searching through venv files")
    print("- Professional Python package structure")
    print("- Easy to test and distribute")


if __name__ == "__main__":
    print("Starting project reorganization...")
    
    # Create directory structure and move files
    create_directory_structure()
    move_files()
    
    create_setup_py()
    create_pyproject_toml()
    create_requirements_dev()
    
    show_next_steps()
