"""
Fixed Parts Mixture Design Implementation
Based on TRUE understanding of Fixed Components

CORRECT UNDERSTANDING of Fixed Components:
- Fixed components have CONSTANT amounts in PARTS (e.g., always 10 parts)
- Variable components have VARIABLE amounts in PARTS (e.g., 0-20 parts)  
- Fixed components have VARIABLE PROPORTIONS (because total batch changes)
- They reduce design space by consuming fixed material amounts

This module provides a user-friendly interface to the TrueFixedComponentsMixture class.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import warnings

from .true_fixed_components_mixture import TrueFixedComponentsMixture

class FixedPartsMixtureDesign:
    """
    User-friendly interface for generating mixture designs with fixed components.
    
    This class provides a simplified API for the TrueFixedComponentsMixture implementation
    with backward compatibility and enhanced usability.
    """
    
    def __init__(self, component_names: List[str], 
                 fixed_parts: Dict[str, float] = None,
                 variable_bounds: Dict[str, Tuple[float, float]] = None,
                 **kwargs):
        """
        Initialize fixed parts mixture design generator.
        
        Parameters:
        -----------
        component_names : List[str]
            Names of all components in the mixture
        fixed_parts : Dict[str, float], optional
            Dictionary mapping fixed component names to their constant parts amount
            Example: {"Polymer_A": 10.0, "Catalyst": 2.5}
        variable_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for variable components in parts
            Example: {"Solvent": (0, 50), "Additive": (0, 20)}
        **kwargs : dict
            Additional arguments for backward compatibility (ignored with warning)
        """
        
        # Handle backward compatibility arguments
        if kwargs:
            deprecated_args = list(kwargs.keys())
            warnings.warn(
                f"Arguments {deprecated_args} are deprecated and ignored. "
                f"Use 'fixed_parts' and 'variable_bounds' instead.",
                DeprecationWarning,
                stacklevel=2
            )
        
        self.component_names = component_names
        self.fixed_parts = fixed_parts or {}
        self.variable_bounds = variable_bounds or {}
        
        # Validate inputs
        self._validate_inputs()
        
        # Create the core mixture designer
        self.designer = TrueFixedComponentsMixture(
            component_names=component_names,
            fixed_parts=fixed_parts,
            variable_bounds=variable_bounds
        )
        
        # Store last generated design for backward compatibility
        self.last_design = None
        self.last_parts_design = None
        self.last_batch_sizes = None
    
    def _validate_inputs(self):
        """Validate the input parameters."""
        if not self.component_names:
            raise ValueError("component_names cannot be empty")
        
        if not self.fixed_parts:
            raise ValueError("At least one fixed component must be specified")
        
        # Check for unknown fixed components
        unknown_fixed = set(self.fixed_parts.keys()) - set(self.component_names)
        if unknown_fixed:
            raise ValueError(f"Unknown fixed components: {unknown_fixed}")
        
        # Set default bounds for variable components
        variable_names = [name for name in self.component_names if name not in self.fixed_parts]
        for name in variable_names:
            if name not in self.variable_bounds:
                self.variable_bounds[name] = (0.0, 100.0)  # Default bounds
        
        print(f"\n✅ Fixed Parts Mixture Design Initialized:")
        print(f"   Total components: {len(self.component_names)}")
        print(f"   Fixed components: {len(self.fixed_parts)} {list(self.fixed_parts.keys())}")
        print(f"   Variable components: {len(variable_names)} {variable_names}")
        print(f"   Fixed parts consumption: {sum(self.fixed_parts.values())}")
    
    def generate_design(self, n_runs: int, design_type: str = "d-optimal", 
                       model_type: str = "quadratic", max_iter: int = 1000, 
                       random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate mixture design with fixed components.
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        design_type : str, default "d-optimal"
            Type of design to generate ("d-optimal", "i-optimal")
        model_type : str, default "quadratic"
            Model type ("linear", "quadratic", "cubic")
        max_iter : int, default 1000
            Maximum iterations for optimization algorithms
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame
            Design matrix with both parts and proportions
        """
        print(f"\n🎯 Generating {design_type} design with {n_runs} runs")
        print(f"   Model type: {model_type}")
        print(f"   Fixed components: {list(self.fixed_parts.keys())}")
        
        if design_type.lower() == "d-optimal":
            parts_design, prop_design, batch_sizes = self.designer.generate_d_optimal_design(
                n_runs=n_runs,
                model_type=model_type,
                max_iter=max_iter,
                n_candidates=2000,
                random_seed=random_seed
            )
        elif design_type.lower() == "i-optimal":
            # For I-optimal, we can use the same D-optimal algorithm for now
            # TODO: Implement proper I-optimal algorithm
            warnings.warn("I-optimal design not yet implemented. Using D-optimal instead.")
            parts_design, prop_design, batch_sizes = self.designer.generate_d_optimal_design(
                n_runs=n_runs,
                model_type=model_type,
                max_iter=max_iter,
                n_candidates=2000,
                random_seed=random_seed
            )
        else:
            raise ValueError(f"Unsupported design type: {design_type}. Use 'd-optimal' or 'i-optimal'.")
        
        # Store results for backward compatibility
        self.last_design = prop_design
        self.last_parts_design = parts_design
        self.last_batch_sizes = batch_sizes
        
        # Create comprehensive results DataFrame
        results_df = self.designer.create_results_dataframe(parts_design, prop_design, batch_sizes)
        
        print(f"\n✅ Design Generated Successfully!")
        print(f"   Runs: {len(results_df)}")
        print(f"   Fixed components maintain constant parts")
        print(f"   Variable components follow specified bounds")
        print(f"   Batch sizes: {batch_sizes.min():.1f} to {batch_sizes.max():.1f} parts")
        
        return results_df
    
    def generate_fixed_parts_design(self, n_runs: int, design_type: str = "d-optimal", 
                                  model_type: str = "quadratic", max_iter: int = 1000, 
                                  random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate fixed parts design (backward compatibility method).
        
        This method maintains backward compatibility with the old API.
        Returns proportions design matrix (normalized to sum to 1).
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        design_type : str, default "d-optimal"
            Type of design ("d-optimal", "i-optimal")  
        model_type : str, default "quadratic"
            Model type ("linear", "quadratic", "cubic")
        max_iter : int, default 1000
            Maximum iterations for optimization
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Design matrix (proportions, normalized to sum to 1)
        """
        warnings.warn(
            "generate_fixed_parts_design() is deprecated. Use generate_design() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Generate the design
        results_df = self.generate_design(
            n_runs=n_runs,
            design_type=design_type,
            model_type=model_type,
            max_iter=max_iter,
            random_seed=random_seed
        )
        
        # Extract proportions design matrix
        prop_cols = [f"{name}_Prop" for name in self.component_names]
        prop_design = results_df[prop_cols].values
        
        return prop_design
    
    def get_parts_design(self) -> Optional[np.ndarray]:
        """
        Get the last generated design in parts.
        
        Returns:
        --------
        np.ndarray or None
            Parts design matrix from last generation, or None if no design generated
        """
        if self.last_parts_design is None:
            warnings.warn("No design has been generated yet. Call generate_design() first.")
        
        return self.last_parts_design
    
    def get_proportions_design(self) -> Optional[np.ndarray]:
        """
        Get the last generated design in proportions.
        
        Returns:
        --------
        np.ndarray or None
            Proportions design matrix from last generation, or None if no design generated
        """
        if self.last_design is None:
            warnings.warn("No design has been generated yet. Call generate_design() first.")
        
        return self.last_design
    
    def get_batch_sizes(self) -> Optional[np.ndarray]:
        """
        Get the batch sizes for the last generated design.
        
        Returns:
        --------
        np.ndarray or None
            Batch sizes from last generation, or None if no design generated
        """
        if self.last_batch_sizes is None:
            warnings.warn("No design has been generated yet. Call generate_design() first.")
        
        return self.last_batch_sizes
    
    def convert_design_to_parts(self, design: np.ndarray, 
                               target_batch_size: Optional[float] = None) -> np.ndarray:
        """
        Convert a proportions design to parts.
        
        Parameters:
        -----------
        design : np.ndarray
            Design matrix in proportions (rows sum to 1)
        target_batch_size : float, optional
            Target batch size for scaling. If None, uses natural batch sizes.
            
        Returns:
        --------
        np.ndarray
            Design matrix in parts
        """
        if target_batch_size is not None:
            # Scale to target batch size
            parts_design = design * target_batch_size
            
            # Ensure fixed components have their correct constant values
            for i, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts_design[:, i] = self.fixed_parts[name]
            
            return parts_design
        else:
            # Use the designer's conversion method
            return self.designer.proportions_to_parts(design, np.sum(design, axis=1))
    
    def convert_parts_to_design(self, parts: np.ndarray) -> np.ndarray:
        """
        Convert a parts design to proportions.
        
        Parameters:
        -----------
        parts : np.ndarray
            Design matrix in parts
            
        Returns:
        --------
        np.ndarray
            Design matrix in proportions (normalized to sum to 1)
        """
        return self.designer.parts_to_proportions(parts)
    
    def export_design_to_csv(self, filename: str, design_df: Optional[pd.DataFrame] = None, 
                           include_verification: bool = True) -> None:
        """
        Export design to CSV file.
        
        Parameters:
        -----------
        filename : str
            Output filename
        design_df : pd.DataFrame, optional
            Design DataFrame to export. If None, uses last generated design.
        include_verification : bool, default True
            Whether to include verification columns
        """
        if design_df is None:
            if self.last_design is None:
                raise ValueError("No design to export. Generate a design first or provide design_df.")
            
            # Create DataFrame from last design
            design_df = self.designer.create_results_dataframe(
                self.last_parts_design, self.last_design, self.last_batch_sizes
            )
        
        # Add verification columns if requested
        if include_verification:
            # Add fixed component verification
            for name in self.fixed_parts:
                expected = self.fixed_parts[name]
                parts_col = f"{name}_Parts"
                if parts_col in design_df.columns:
                    design_df[f'{name}_FixedOK'] = np.abs(design_df[parts_col] - expected) < 1e-6
        
        # Save to CSV
        design_df.to_csv(filename)
        print(f"✅ Design exported to {filename}")
        print(f"   Rows: {len(design_df)}")
        print(f"   Columns: {len(design_df.columns)}")
    
    def print_design_summary(self, design_df: Optional[pd.DataFrame] = None) -> None:
        """
        Print a summary of the design.
        
        Parameters:
        -----------
        design_df : pd.DataFrame, optional
            Design DataFrame to summarize. If None, uses last generated design.
        """
        if design_df is None:
            if self.last_design is None:
                print("❌ No design to summarize. Generate a design first.")
                return
            
            design_df = self.designer.create_results_dataframe(
                self.last_parts_design, self.last_design, self.last_batch_sizes
            )
        
        print(f"\n📊 DESIGN SUMMARY")
        print(f"=" * 50)
        print(f"Runs: {len(design_df)}")
        print(f"Components: {len(self.component_names)}")
        print(f"Fixed components: {len(self.fixed_parts)}")
        
        print(f"\n🔧 FIXED COMPONENTS (Constant Parts):")
        for name, parts in self.fixed_parts.items():
            parts_col = f"{name}_Parts"
            prop_col = f"{name}_Prop"
            
            if parts_col in design_df.columns and prop_col in design_df.columns:
                parts_values = design_df[parts_col]
                prop_values = design_df[prop_col]
                
                print(f"  {name}:")
                print(f"    Parts: {parts_values.min():.3f} to {parts_values.max():.3f} (constant: {parts})")
                print(f"    Proportions: {prop_values.min():.3f} to {prop_values.max():.3f} (varies)")
        
        print(f"\n🔄 VARIABLE COMPONENTS (Variable Parts):")
        variable_names = [name for name in self.component_names if name not in self.fixed_parts]
        for name in variable_names:
            parts_col = f"{name}_Parts"
            
            if parts_col in design_df.columns:
                parts_values = design_df[parts_col]
                bounds = self.variable_bounds.get(name, (0, 100))
                
                print(f"  {name}:")
                print(f"    Parts: {parts_values.min():.3f} to {parts_values.max():.3f} (bounds: {bounds})")
        
        if 'Batch_Size' in design_df.columns:
            batch_sizes = design_df['Batch_Size']
            print(f"\n📏 BATCH SIZES:")
            print(f"  Range: {batch_sizes.min():.1f} to {batch_sizes.max():.1f} parts")
            print(f"  Fixed consumption: {sum(self.fixed_parts.values()):.1f} parts per batch")
            print(f"  Variable consumption: {batch_sizes.min() - sum(self.fixed_parts.values()):.1f} to {batch_sizes.max() - sum(self.fixed_parts.values()):.1f} parts")
    
    def validate_design(self, design_df: Optional[pd.DataFrame] = None) -> Dict[str, bool]:
        """
        Validate the design meets all constraints.
        
        Parameters:
        -----------
        design_df : pd.DataFrame, optional
            Design DataFrame to validate. If None, uses last generated design.
            
        Returns:
        --------
        Dict[str, bool]
            Validation results
        """
        if design_df is None:
            if self.last_design is None:
                raise ValueError("No design to validate. Generate a design first.")
            
            design_df = self.designer.create_results_dataframe(
                self.last_parts_design, self.last_design, self.last_batch_sizes
            )
        
        validation_results = {}
        
        # Check proportions sum to 1
        prop_cols = [f"{name}_Prop" for name in self.component_names]
        if all(col in design_df.columns for col in prop_cols):
            prop_sums = design_df[prop_cols].sum(axis=1)
            validation_results['proportions_sum_to_1'] = np.allclose(prop_sums, 1.0, atol=1e-6)
        
        # Check fixed components have constant parts
        for name in self.fixed_parts:
            parts_col = f"{name}_Parts"
            if parts_col in design_df.columns:
                expected = self.fixed_parts[name]
                parts_values = design_df[parts_col]
                validation_results[f'{name}_parts_constant'] = np.allclose(parts_values, expected, atol=1e-6)
        
        # Check variable components are within bounds
        variable_names = [name for name in self.component_names if name not in self.fixed_parts]
        for name in variable_names:
            parts_col = f"{name}_Parts"
            if parts_col in design_df.columns and name in self.variable_bounds:
                min_bound, max_bound = self.variable_bounds[name]
                parts_values = design_df[parts_col]
                within_bounds = np.all((parts_values >= min_bound - 1e-6) & (parts_values <= max_bound + 1e-6))
                validation_results[f'{name}_within_bounds'] = within_bounds
        
        # Overall validation
        validation_results['overall_valid'] = all(validation_results.values())
        
        return validation_results


# Convenience functions for backward compatibility
def create_fixed_parts_design(component_names: List[str], 
                            fixed_parts: Dict[str, float],
                            variable_bounds: Dict[str, Tuple[float, float]] = None) -> FixedPartsMixtureDesign:
    """
    Create a fixed parts mixture design generator.
    
    Convenience function for creating FixedPartsMixtureDesign instances.
    
    Parameters:
    -----------
    component_names : List[str]
        Names of all components
    fixed_parts : Dict[str, float]
        Fixed component parts (constant amounts)
    variable_bounds : Dict[str, Tuple[float, float]], optional
        Variable component bounds in parts
        
    Returns:
    --------
    FixedPartsMixtureDesign
        Configured design generator
    """
    return FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )


# Example usage
if __name__ == "__main__":
    # Example: 4-component polymer formulation with 2 fixed components
    component_names = ["Base_Polymer", "Catalyst", "Solvent", "Additive"]
    
    # Fixed components (constant parts)
    fixed_parts = {
        "Base_Polymer": 50.0,  # Always 50 parts
        "Catalyst": 2.5        # Always 2.5 parts
    }
    
    # Variable components bounds (in parts)
    variable_bounds = {
        "Solvent": (0.0, 40.0),    # 0-40 parts
        "Additive": (0.0, 15.0)    # 0-15 parts
    }
    
    print("=== Fixed Parts Mixture Design Example ===")
    
    # Create design generator
    designer = FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate D-optimal design
    design_df = designer.generate_design(
        n_runs=10,
        design_type="d-optimal",
        model_type="quadratic",
        random_seed=42
    )
    
    # Print summary
    designer.print_design_summary(design_df)
    
    # Validate design
    validation = designer.validate_design(design_df)
    print(f"\n✅ Design validation: {validation['overall_valid']}")
    
    # Export to CSV
    designer.export_design_to_csv("example_fixed_parts_design.csv", design_df)
    
    print(f"\n🎯 Fixed Parts Design Complete!")
    print(f"   Fixed components maintain constant parts")
    print(f"   Variable components follow specified bounds")
    print(f"   Design space reduced by fixed material consumption")
