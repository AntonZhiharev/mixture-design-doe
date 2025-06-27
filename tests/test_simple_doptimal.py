"""
Simple test to check D-optimal design issue
"""

import sys
import os
import numpy as np
import pandas as pd

# Try different import methods
print("Testing imports...")

try:
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, parent_dir)
    
    # Try importing from current directory first
    from refactored_mixture_design import MixtureDesign
    print("✅ Imported from current directory")
except ImportError:
    try:
        # Try importing from src
        from src.core.refactored_mixture_design import MixtureDesign
        print("✅ Imported from src.core")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        exit(1)

# Test D-optimal design
print("\n" + "="*60)
print("Testing D-Optimal Design for Mixture")
print("="*60)

# Create mixture design instance
md = MixtureDesign(n_components=3)

# Generate D-optimal design
print("\nGenerating D-optimal design with 10 runs...")
design_df = md.create_design(n_runs=10, design_type='d-optimal')

print("\nDesign points:")
print(design_df)

# Analyze the design
X = design_df.values
n_components = X.shape[1]

# Check if points are only at corners
corner_points = 0
edge_points = 0
interior_points = 0

tolerance = 1e-6

for point in X:
    # Count non-zero components
    non_zero = np.sum(point > tolerance)
    
    if non_zero == 1:
        corner_points += 1
    elif non_zero == 2:
        edge_points += 1
    else:
        interior_points += 1

print(f"\nDesign Analysis:")
print(f"Total points: {len(X)}")
print(f"Corner points (pure components): {corner_points}")
print(f"Edge points (binary mixtures): {edge_points}")
print(f"Interior points (ternary mixtures): {interior_points}")

# Calculate D-efficiency
d_eff = md.calculate_d_efficiency(X)
print(f"\nD-efficiency: {d_eff:.4f}")

if corner_points == len(X):
    print("\n⚠️  WARNING: All points are at corners!")
    print("This confirms the issue - D-optimal is only selecting pure components.")
    print("This results in lower D-efficiency compared to designs with interior points.")
