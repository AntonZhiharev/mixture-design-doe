"""
Script to clean up the project by moving remaining files and removing duplicates
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up the project structure"""
    
    # Files to move to specific directories
    files_to_move = {
        # Additional test files
        "test_doptimal_improvement.py": "tests/test_doptimal_improvement.py",
        "test_simple_doptimal.py": "tests/test_simple_doptimal.py",
        
        # Additional example/demo files
        "analyze_simplex_lattice_zeros.py": "examples/analyze_simplex_lattice_zeros.py",
        "compare_approaches.py": "examples/compare_approaches.py",
        "compare_candidate_distribution.py": "examples/compare_candidate_distribution.py",
        "debug_same_approach.py": "examples/debug_same_approach.py",
        "final_demonstration.py": "examples/final_demonstration.py",
        "final_test_complete_solution.py": "examples/final_test_complete_solution.py",
        "final_verification_fixed_components.py": "examples/final_verification_fixed_components.py",
        "parts_mode_demonstration.py": "examples/parts_mode_demonstration.py",
        "regular_vs_mixture_comparison.py": "examples/regular_vs_mixture_comparison.py",
        "force_reload_streamlit.py": "examples/force_reload_streamlit.py",
        "fix_streamlit_bounds.py": "examples/fix_streamlit_bounds.py",
        
        # Data files
        "simplex_lattice_design.csv": "data/simplex_lattice_design.csv",
        
        # Image files
        "design_comparison_2d.png": "output/design_comparison_2d.png",
        "direct_optimization_comparison.png": "output/direct_optimization_comparison.png",
        "fixed_parts_comparison.png": "output/fixed_parts_comparison.png",
        "mixture_design_comparison_2d.png": "output/mixture_design_comparison_2d.png",
        "parts_mode_comparison_parts.png": "output/parts_mode_comparison_parts.png",
        "parts_mode_comparison_proportions.png": "output/parts_mode_comparison_proportions.png",
        "regular_vs_mixture_comparison_3d.png": "output/regular_vs_mixture_comparison_3d.png",
        "regular_vs_mixture_comparison.png": "output/regular_vs_mixture_comparison.png",
        "scatter_distribution_comparison.png": "output/scatter_distribution_comparison.png",
        
        # Sequential DOE files (new utils)
        "sequential_doe.py": "src/utils/sequential_doe.py",
        "sequential_mixture_doe.py": "src/utils/sequential_mixture_doe.py",
        "response_analysis.py": "src/utils/response_analysis.py",
    }
    
    # Files that have already been moved and should be deleted from root
    files_to_delete = [
        # Core files already in src/core/
        "base_doe.py",
        "mixture_base.py",
        "refactored_mixture_design.py",
        "fixed_parts_mixture_designs.py",
        "mixture_designs.py",
        
        # Algorithm files already in src/algorithms/
        "mixture_algorithms.py",
        
        # Utils already in src/utils/
        "mixture_utils.py",
        
        # App files already in src/apps/
        "streamlit_app.py",
        "streamlit_app_refactored.py",
        
        # Test files already in tests/
        "test_refactored_design.py",
        "test_d_efficiency_improvement.py",
        "test_direct_optimization.py",
        "test_exact_same_approach.py",
        "test_import_fix.py",
        "test_mixture_design_class.py",
        "test_optimized_mixture.py",
        "test_all_mixture_classes.py",
        
        # Example files already in examples/
        "example_usage.py",
        "compare_mixture_approaches.py",
        "compare_d_efficiency.py",
        "demonstrate_improved_design.py",
        
        # Docs already moved
        "getting_started.md",
        "README_improvement.md",
        "flexible_runs_summary.md",
        
        # Other files
        "optimal_doe_python.py",  # Old version
        "optimal_doe_r.R",  # R version, not needed
    ]
    
    # Move files
    moved_count = 0
    for src, dst in files_to_move.items():
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dst_path))
            print(f"✅ Moved: {src} → {dst}")
            moved_count += 1
    
    # Delete duplicate files
    deleted_count = 0
    for file in files_to_delete:
        file_path = Path(file)
        if file_path.exists():
            file_path.unlink()
            print(f"🗑️  Deleted: {file}")
            deleted_count += 1
    
    print(f"\n📊 Summary:")
    print(f"   Moved: {moved_count} files")
    print(f"   Deleted: {deleted_count} duplicate files")
    
    # List remaining files in root
    remaining_files = []
    for file in os.listdir("."):
        if os.path.isfile(file) and not file.startswith(".") and file not in [
            "__init__.py", "setup.py", "pyproject.toml", "requirements.txt", 
            "requirements-dev.txt", "README.md", "reorganize_project.py", 
            "cleanup_project.py", "recommended_structure.md"
        ]:
            remaining_files.append(file)
    
    if remaining_files:
        print(f"\n⚠️  Files still in root directory:")
        for file in remaining_files:
            print(f"   - {file}")
    else:
        print(f"\n✨ Root directory is clean!")

if __name__ == "__main__":
    print("Starting project cleanup...")
    cleanup_project()
    print("\nCleanup complete!")
