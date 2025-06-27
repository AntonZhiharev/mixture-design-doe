"""
Fixed Parts Mixture Design Implementation
Specialized for mixture experiments with fixed components
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from mixture_base import MixtureBase
from mixture_utils import check_bounds

class FixedPartsMixtureDesign(MixtureBase):
    """
    Class for generating mixture designs with fixed components
    """
    
    def __init__(self, n_components: int, component_names: List[str] = None, 
                 component_bounds: List[Tuple[float, float]] = None,
                 use_parts_mode: bool = True, fixed_components: Dict[str, float] = None):
        """
        Initialize fixed parts mixture design generator
        
        Parameters:
        n_components: Number of mixture components
        component_names: Names of components (default: Comp1, Comp2, ...)
        component_bounds: Lower and upper bounds for each component in parts
        use_parts_mode: Should be True for fixed parts mode (default: True)
        fixed_components: Dict of component names and their fixed values in parts
        """
        # Force parts mode for fixed parts design
        if not use_parts_mode:
            print("Warning: Fixed parts mixture design requires parts mode. Setting use_parts_mode=True.")
            use_parts_mode = True
        
        super().__init__(n_components, component_names, component_bounds, 
                        use_parts_mode, fixed_components)
        
        # Store original parts for later use
        self.original_parts_bounds = self.original_bounds
        
        # Print information about fixed components
        if fixed_components:
            print("\nFixed Components (Parts):")
            for name, value in fixed_components.items():
                print(f"  {name}: {value}")
    
    def generate_fixed_parts_design(self, n_runs: int, design_type: str = "d-optimal", 
                                  model_type: str = "quadratic", max_iter: int = 1000, 
                                  random_seed: int = None) -> np.ndarray:
        """
        Generate mixture design with fixed components
        
        Parameters:
        n_runs: Number of runs in the design
        design_type: "d-optimal", "i-optimal", "simplex-lattice", "simplex-centroid", or "extreme-vertices"
        model_type: "linear", "quadratic", or "cubic"
        max_iter: Maximum number of iterations for coordinate exchange
        random_seed: Random seed for reproducibility
        
        Returns:
        np.ndarray: Design matrix
        """
        print(f"\nGenerating {design_type} design with {n_runs} runs")
        print(f"Model type: {model_type}")
        
        if design_type.lower() == "d-optimal":
            design = self.generate_d_optimal(n_runs, model_type, max_iter, random_seed)
        elif design_type.lower() == "i-optimal":
            design = self.generate_i_optimal(n_runs, model_type, max_iter, random_seed)
        else:
            raise ValueError(f"Design type '{design_type}' not supported for fixed parts design. "
                           f"Use 'd-optimal' or 'i-optimal'.")
        
        # Convert design to parts for reporting
        if hasattr(self, 'parts_design'):
            parts_design = self.parts_design
            
            print("\nDesign in Parts:")
            for i in range(min(5, len(design))):
                print(f"Run {i+1}:")
                for j, name in enumerate(self.component_names):
                    print(f"  {name}: {parts_design[i, j]:.4f}")
                print(f"  Total: {parts_design[i].sum():.4f}")
            
            if len(design) > 5:
                print("  ...")
        
        return design
    
    def convert_design_to_parts(self, design: np.ndarray) -> np.ndarray:
        """
        Convert normalized design to parts
        
        Parameters:
        design: Design matrix (normalized proportions)
        
        Returns:
        np.ndarray: Design matrix in parts
        """
        if not hasattr(self, 'original_fixed_components'):
            print("Warning: No original fixed components found. Cannot convert to parts.")
            return design
        
        # Get the original fixed component values
        original_fixed_values = self.original_fixed_components
        
        # Calculate total parts for each run
        total_parts = np.zeros(len(design))
        
        # For each run, calculate total parts based on fixed components
        for row_idx in range(len(design)):
            # Get current proportions for fixed components
            fixed_proportions = {}
            for comp_name in original_fixed_values.keys():
                comp_idx = self.component_names.index(comp_name)
                fixed_proportions[comp_name] = design[row_idx, comp_idx]
            
            # Calculate total parts based on fixed components
            run_total_parts = []
            for comp_name, fixed_prop in fixed_proportions.items():
                if fixed_prop > 0:  # Avoid division by zero
                    parts = original_fixed_values[comp_name] / fixed_prop
                    run_total_parts.append(parts)
            
            # Use average if we have multiple fixed components
            if run_total_parts:
                total_parts[row_idx] = np.mean(run_total_parts)
            else:
                # Fallback if no valid fixed components
                total_parts[row_idx] = 100.0  # Default batch size
        
        # Convert proportions to parts
        parts_design = np.zeros_like(design)
        
        for row_idx in range(len(design)):
            for comp_idx, comp_name in enumerate(self.component_names):
                if comp_name in original_fixed_values:
                    # Fixed components have constant parts
                    parts_design[row_idx, comp_idx] = original_fixed_values[comp_name]
                else:
                    # Variable components are scaled by total parts
                    parts_design[row_idx, comp_idx] = design[row_idx, comp_idx] * total_parts[row_idx]
        
        return parts_design
    
    def convert_parts_to_design(self, parts: np.ndarray) -> np.ndarray:
        """
        Convert parts to normalized design
        
        Parameters:
        parts: Design matrix in parts
        
        Returns:
        np.ndarray: Normalized design matrix
        """
        # Normalize each row to sum to 1
        row_sums = parts.sum(axis=1, keepdims=True)
        normalized_design = parts / row_sums
        
        return normalized_design
    
    def export_design_to_csv(self, design: np.ndarray, filename: str, 
                           include_parts: bool = True, batch_size: float = None) -> None:
        """
        Export design to CSV file
        
        Parameters:
        design: Design matrix
        filename: Output filename
        include_parts: Whether to include parts in the output
        batch_size: Batch size for scaling (if None, use calculated batch size)
        """
        # Create DataFrame with proportions
        df = pd.DataFrame(design, columns=self.component_names)
        df.index = [f"Run_{i+1}" for i in range(len(design))]
        df.index.name = "Run"
        
        # Add sum of proportions
        df["Sum"] = df.sum(axis=1)
        
        # Convert to parts if requested
        if include_parts:
            if hasattr(self, 'parts_design'):
                parts_design = self.parts_design
            else:
                parts_design = self.convert_design_to_parts(design)
            
            # Add parts columns
            for i, name in enumerate(self.component_names):
                df[f"{name}_Parts"] = parts_design[:, i]
            
            # Add sum of parts
            df["Total_Parts"] = parts_design.sum(axis=1)
            
            # Add batch-scaled quantities if batch size provided
            if batch_size is not None:
                for i, name in enumerate(self.component_names):
                    df[f"{name}_Batch"] = parts_design[:, i] * (batch_size / df["Total_Parts"])
                
                # Add sum of batch quantities
                df["Batch_Total"] = batch_size
        
        # Save to CSV
        df.to_csv(filename)
        print(f"Design exported to {filename}")
