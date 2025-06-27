"""
Basic test of D-optimal design issue
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.simplified_mixture_design import create_mixture_design

# Test basic functionality
print("Testing D-Optimal Design...")

# Create D-optimal design without interior points (should give only corners)
design1 = create_mixture_design('d-optimal', 3, n_runs=6, include_interior=False)
print("\nD-Optimal WITHOUT interior points:")
print(design1)

# Create D-optimal design with interior points
design2 = create_mixture_design('d-optimal', 3, n_runs=10, include_interior=True)
print("\nD-Optimal WITH interior points:")
print(design2)

# Analyze the difference
print("\nAnalysis:")
print(f"Design 1 has {len(design1)} points")
print(f"Design 2 has {len(design2)} points")

# Check for interior points
def has_interior_points(design):
    for _, row in design.iterrows():
        if all(row > 0.01):  # All components > 0
            return True
    return False

print(f"\nDesign 1 has interior points: {has_interior_points(design1)}")
print(f"Design 2 has interior points: {has_interior_points(design2)}")
