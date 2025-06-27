"""
Base Mixture Design of Experiments Implementation
Specialized for mixture experiments where components sum to 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from mixture_utils import (
    validate_proportion_bounds, 
    validate_parts_bounds, 
    convert_parts_to_proportions,
    check_bounds
)

class MixtureBase:
    """
    Base class for generating mixture experimental designs
    """
    
    def __init__(self, n_components: int, component_names: List[str] = None, 
                 component_bounds: List[Tuple[float, float]] = None,
                 use_parts_mode: bool = False, fixed_components: Dict[str, float] = None):
        """
        Initialize mixture design generator
        
        Parameters:
        n_components: Number of mixture components
        component_names: Names of components (default: Comp1, Comp2, ...)
        component_bounds: Lower and upper bounds for each component
            - If use_parts_mode=False: bounds must be between 0 and 1 (proportions)
            - If use_parts_mode=True: bounds are in parts (any positive values)
        use_parts_mode: If True, work with parts that get normalized to proportions
        fixed_components: Dict of component names and their fixed values
        """
        self.n_components = n_components
        self.use_parts_mode = use_parts_mode
        self.fixed_components = fixed_components or {}
        
        if component_names is None:
            self.component_names = [f'Comp_{i+1}' for i in range(n_components)]
        else:
            self.component_names = component_names
            
        if component_bounds is None:
            if use_parts_mode:
                # Default parts bounds - give reasonable ranges
                self.component_bounds = [(0.1, 10.0)] * n_components
            else:
                self.component_bounds = [(0.0, 1.0)] * n_components
        else:
            # Validate each bounds element
            for i, bounds_element in enumerate(component_bounds):
                if not isinstance(bounds_element, (tuple, list)):
                    raise ValueError(f"bounds[{i}] is not tuple/list: {bounds_element}")
                
                if len(bounds_element) != 2:
                    raise ValueError(f"bounds[{i}] length is {len(bounds_element)}, expected 2: {bounds_element}")
                
                try:
                    lower, upper = bounds_element
                except ValueError as unpack_error:
                    raise ValueError(f"Cannot unpack bounds[{i}]: {bounds_element}") from unpack_error
            
            self.component_bounds = component_bounds
            
        # Store original bounds and fixed components for parts mode
        self.original_bounds = self.component_bounds.copy() if use_parts_mode else None
        self.original_fixed_components = self.fixed_components.copy()
        
        # Store which components are truly fixed for model matrix creation
        self.truly_fixed_components = set()
        if self.original_fixed_components:
            self.truly_fixed_components = set(self.original_fixed_components.keys())
        elif self.fixed_components:
            self.truly_fixed_components = set(self.fixed_components.keys())
        
        # Convert from parts to proportions if needed
        if self.use_parts_mode:
            self._convert_parts_to_proportions()
            
            # After parts conversion, also check for the proportions dict
            if hasattr(self, 'original_fixed_components_proportions'):
                self.truly_fixed_components.update(self.original_fixed_components_proportions.keys())
        
        # Validate bounds
        if not use_parts_mode:
            self.component_bounds = validate_proportion_bounds(self.component_bounds)
        else:
            self.component_bounds = validate_parts_bounds(self.component_bounds)
    
    def _convert_parts_to_proportions(self):
        """
        Convert parts to proportions using simple normalization
        """
        if not self.use_parts_mode:
            return
        
        print("\nConverting from parts to proportions...")
        
        # Simple normalization approach
        self.component_bounds = convert_parts_to_proportions(self.component_bounds)
        
        # Convert fixed components to proportions
        if self.fixed_components:
            total_parts = sum(self.fixed_components.values())
            fixed_components_props = {}
            
            for name, parts in self.fixed_components.items():
                fixed_components_props[name] = parts / total_parts
            
            # Store original fixed components proportions
            self.original_fixed_components_proportions = fixed_components_props
            
            # Update fixed components
            self.fixed_components = fixed_components_props
        
        # Switch to proportion mode since we've converted from parts
        self.use_parts_mode = False
        
        print("✅ Conversion complete")
    
    def _adjust_for_fixed_components(self, design: np.ndarray) -> np.ndarray:
        """
        Adjust design matrix to account for fixed components
        """
        if not self.fixed_components:
            return design
        
        # Get indices of fixed and variable components
        fixed_indices = []
        variable_indices = []
        
        for i, name in enumerate(self.component_names):
            if name in self.fixed_components:
                fixed_indices.append(i)
            else:
                variable_indices.append(i)
        
        adjusted_design = design.copy()
        
        # Calculate fixed sum
        fixed_sum = sum(self.fixed_components.values())
        remaining_sum = 1.0 - fixed_sum
        
        if remaining_sum <= 0:
            raise ValueError(f"Fixed components sum to {fixed_sum}, leaving no room for variable components")
        
        # For each row, adjust variable components
        for row_idx in range(adjusted_design.shape[0]):
            # Get current variable component values for this row
            variable_values = adjusted_design[row_idx, variable_indices]
            variable_sum = np.sum(variable_values)
            
            # If all variable components are zero, set them to equal proportions
            if variable_sum == 0:
                equal_share = remaining_sum / len(variable_indices)
                for i, idx in enumerate(variable_indices):
                    adjusted_design[row_idx, idx] = equal_share
            else:
                # Rescale variable components proportionally
                for i, idx in enumerate(variable_indices):
                    adjusted_design[row_idx, idx] = (variable_values[i] / variable_sum) * remaining_sum
            
            # Set fixed component values
            for idx, name in enumerate(self.component_names):
                if name in self.fixed_components:
                    adjusted_design[row_idx, idx] = self.fixed_components[name]
        
        return adjusted_design
    
    def _post_process_design_fixed_components(self, design: np.ndarray) -> np.ndarray:
        """
        Post-process design according to the Regular Optimal Design approach:
        1. Keep fixed components at constant parts across all runs
        2. Only normalize at the end when calculating final quantities
        """
        # Check if we have any original fixed components stored
        if not hasattr(self, 'original_fixed_components_proportions') and not self.original_fixed_components:
            return design
        
        if design.size == 0:
            return design
        
        print("\nPost-processing design with Regular Optimal Design approach")
        design_corrected = design.copy()
        
        # Get the original fixed component values
        if hasattr(self, 'original_fixed_components_proportions'):
            # From parts mode conversion - use the calculated proportions that were stored
            original_fixed_values = self.original_fixed_components_proportions
            print("Using original fixed components from parts mode conversion:")
            for comp_name, value in original_fixed_values.items():
                print(f"  {comp_name}: {value:.6f} (calculated proportion)")
        else:
            # Direct proportion input - use as-is
            original_fixed_values = self.original_fixed_components
            print("Using original fixed components from direct input:")
            for comp_name, value in original_fixed_values.items():
                print(f"  {comp_name}: {value:.6f}")
        
        # Step 1: Calculate the total parts for each run based on fixed components
        print("\nStep 1: Calculate total parts for each run")
        total_parts = np.zeros(len(design_corrected))
        
        # For each run, calculate total parts based on fixed components
        for row_idx in range(len(design_corrected)):
            # Get current proportions for fixed components
            fixed_proportions = {}
            for comp_name in original_fixed_values.keys():
                comp_idx = self.component_names.index(comp_name)
                fixed_proportions[comp_name] = design_corrected[row_idx, comp_idx]
            
            # Calculate total parts based on fixed components
            # Formula: total_parts = fixed_parts / fixed_proportion
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
        
        print(f"Total parts range: {np.min(total_parts):.2f} - {np.max(total_parts):.2f}")
        
        # Step 2: Convert proportions to parts for all components
        print("\nStep 2: Convert proportions to parts")
        parts_design = np.zeros_like(design_corrected)
        
        for row_idx in range(len(design_corrected)):
            for comp_idx, comp_name in enumerate(self.component_names):
                if comp_name in original_fixed_values:
                    # Fixed components have constant parts
                    parts_design[row_idx, comp_idx] = original_fixed_values[comp_name]
                else:
                    # Variable components are scaled by total parts
                    parts_design[row_idx, comp_idx] = design_corrected[row_idx, comp_idx] * total_parts[row_idx]
        
        # Step 3: Convert parts back to proportions (final normalization)
        print("\nStep 3: Final normalization to ensure sum = 1.0")
        design_normalized = parts_design / parts_design.sum(axis=1)[:, np.newaxis]
        
        print("\nFinal normalized design:")
        for i in range(min(3, len(design_normalized))):
            print(f"Mix {i+1}:")
            for j, comp_name in enumerate(self.component_names):
                if comp_name in original_fixed_values:
                    print(f"  {comp_name}: {design_normalized[i, j]:.6f} (FIXED PARTS: {parts_design[i, j]:.4f})")
                else:
                    print(f"  {comp_name}: {design_normalized[i, j]:.6f} (PARTS: {parts_design[i, j]:.4f})")
            print(f"  SUM: {sum(design_normalized[i]):.6f}, TOTAL PARTS: {parts_design[i].sum():.2f}")
        
        # Store parts design for later use
        self.parts_design = parts_design
        
        return design_normalized
    
    def convert_to_batch_quantities(self, design: np.ndarray, batch_size: float = 100.0) -> np.ndarray:
        """
        Convert normalized design to batch quantities
        """
        return design * batch_size
    
    def _generate_model_matrix(self, design: np.ndarray, model_type: str) -> np.ndarray:
        """
        Generate model matrix for given design and model type
        """
        if design.size == 0:
            return np.array([])
        
        # Get variable component indices (exclude fixed components)
        variable_indices = []
        for i, name in enumerate(self.component_names):
            if name not in self.truly_fixed_components:
                variable_indices.append(i)
        
        # Extract variable components from design
        X_var = design[:, variable_indices]
        
        if model_type == "linear":
            # Linear model: X
            X = X_var
        
        elif model_type == "quadratic":
            # Quadratic model: X + X*X
            X = X_var.copy()
            
            # Add quadratic terms (X_i * X_j)
            for i, j in itertools.combinations_with_replacement(range(X_var.shape[1]), 2):
                X = np.column_stack((X, X_var[:, i] * X_var[:, j]))
        
        elif model_type == "cubic":
            # Cubic model: X + X*X + X*X*X
            X = X_var.copy()
            
            # Add quadratic terms (X_i * X_j)
            for i, j in itertools.combinations_with_replacement(range(X_var.shape[1]), 2):
                X = np.column_stack((X, X_var[:, i] * X_var[:, j]))
            
            # Add cubic terms (X_i * X_j * X_k)
            for i, j, k in itertools.combinations_with_replacement(range(X_var.shape[1]), 3):
                X = np.column_stack((X, X_var[:, i] * X_var[:, j] * X_var[:, k]))
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return X
    
    def _calculate_d_efficiency(self, design: np.ndarray, model_type: str) -> float:
        """
        Calculate D-efficiency of design
        """
        if design.size == 0:
            return 0.0
        
        try:
            # Generate model matrix
            X = self._generate_model_matrix(design, model_type)
            
            # Check if X has full rank
            if np.linalg.matrix_rank(X) < X.shape[1]:
                print(f"Warning: Model matrix does not have full rank. D-efficiency may be unreliable.")
                return 0.0
            
            # Calculate D-efficiency
            XtX = X.T @ X
            det_XtX = np.linalg.det(XtX)
            
            if det_XtX <= 0:
                print(f"Warning: Determinant of X'X is non-positive: {det_XtX}. D-efficiency set to 0.")
                return 0.0
            
            # D-efficiency = (det(X'X))^(1/p) / n
            p = X.shape[1]  # Number of model parameters
            n = X.shape[0]  # Number of design points
            
            d_eff = (det_XtX ** (1/p)) / n
            
            return d_eff
        
        except Exception as e:
            print(f"Error calculating D-efficiency: {e}")
            return 0.0
    
    def _calculate_i_efficiency(self, design: np.ndarray, model_type: str) -> float:
        """
        Calculate I-efficiency (integrated prediction variance) of design
        """
        if design.size == 0:
            return 0.0
        
        try:
            # Generate model matrix
            X = self._generate_model_matrix(design, model_type)
            
            # Check if X has full rank
            if np.linalg.matrix_rank(X) < X.shape[1]:
                print(f"Warning: Model matrix does not have full rank. I-efficiency may be unreliable.")
                return 0.0
            
            # Calculate (X'X)^-1
            XtX = X.T @ X
            XtX_inv = np.linalg.inv(XtX)
            
            # Generate Monte Carlo samples from design space
            n_samples = 1000
            samples = self._generate_candidate_points(n_samples)
            
            # Generate model matrix for samples
            X_samples = self._generate_model_matrix(samples, model_type)
            
            # Calculate prediction variance for each sample
            pred_var = np.zeros(n_samples)
            for i in range(n_samples):
                x_i = X_samples[i]
                pred_var[i] = x_i @ XtX_inv @ x_i
            
            # I-efficiency = 1 / average prediction variance
            avg_pred_var = np.mean(pred_var)
            
            if avg_pred_var <= 0:
                print(f"Warning: Average prediction variance is non-positive: {avg_pred_var}. I-efficiency set to 0.")
                return 0.0
            
            i_eff = 1.0 / avg_pred_var
            
            return i_eff
        
        except Exception as e:
            print(f"Error calculating I-efficiency: {e}")
            return 0.0
    
    def _generate_candidate_points(self, n_points: int) -> np.ndarray:
        """
        Generate candidate points for optimal design
        """
        # This is a placeholder method to be implemented by subclasses
        # For the base class, just return random points
        candidates = np.random.random((n_points, self.n_components))
        candidates = candidates / candidates.sum(axis=1)[:, np.newaxis]
        
        # Filter to ensure all points satisfy bounds
        valid_candidates = []
        for point in candidates:
            if check_bounds(point, self.component_bounds):
                valid_candidates.append(point)
        
        # If we don't have enough valid candidates, try different strategies
        if len(valid_candidates) < n_points:
            print(f"Warning: Only {len(valid_candidates)} valid candidates found. Trying alternative strategies...")
            
            # Strategy 1: Try with relaxed bounds
            relaxed_bounds = []
            for lower, upper in self.component_bounds:
                # Relax bounds by 10%
                range_val = upper - lower
                relaxed_lower = max(0, lower - 0.1 * range_val)
                relaxed_upper = min(1.0, upper + 0.1 * range_val)
                relaxed_bounds.append((relaxed_lower, relaxed_upper))
            
            # Generate more points with relaxed bounds
            more_candidates = np.random.random((n_points * 2, self.n_components))
            more_candidates = more_candidates / more_candidates.sum(axis=1)[:, np.newaxis]
            
            for point in more_candidates:
                # Check with relaxed bounds
                if check_bounds(point, relaxed_bounds):
                    # Normalize to ensure sum is exactly 1.0
                    normalized_point = point / np.sum(point)
                    valid_candidates.append(normalized_point)
            
            # Strategy 2: Generate vertices and centroids
            # Vertices (pure components)
            for i in range(self.n_components):
                vertex = np.zeros(self.n_components)
                vertex[i] = 1.0
                if check_bounds(vertex, self.component_bounds):
                    valid_candidates.append(vertex)
            
            # Centroid
            centroid = np.ones(self.n_components) / self.n_components
            if check_bounds(centroid, self.component_bounds):
                valid_candidates.append(centroid)
            
            # Strategy 3: Generate binary mixtures
            for i in range(self.n_components):
                for j in range(i+1, self.n_components):
                    for ratio in [0.2, 0.5, 0.8]:
                        binary = np.zeros(self.n_components)
                        binary[i] = ratio
                        binary[j] = 1.0 - ratio
                        if check_bounds(binary, self.component_bounds):
                            valid_candidates.append(binary)
            
            # Deduplicate
            unique_candidates = []
            for point in valid_candidates:
                if not any(np.allclose(point, p, atol=1e-4) for p in unique_candidates):
                    unique_candidates.append(point)
                    if len(unique_candidates) >= n_points:
                        break
            
            valid_candidates = unique_candidates
        
        # If we still don't have enough, use centroid with small perturbations
        if len(valid_candidates) < n_points:
            print(f"Warning: Still only {len(valid_candidates)} valid candidates. Adding perturbed centroids...")
            centroid = np.ones(self.n_components) / self.n_components
            
            while len(valid_candidates) < n_points:
                # Create small random perturbations around centroid
                perturbation = np.random.uniform(-0.05, 0.05, self.n_components)
                point = centroid + perturbation
                
                # Ensure non-negative
                point = np.maximum(point, 0)
                
                # Normalize
                point = point / np.sum(point)
                
                if check_bounds(point, self.component_bounds):
                    valid_candidates.append(point)
                else:
                    # If still not valid, just use the centroid
                    valid_candidates.append(centroid.copy())
        
        # Ensure we have exactly n_points candidates
        if len(valid_candidates) > n_points:
            valid_candidates = valid_candidates[:n_points]
        elif len(valid_candidates) < n_points:
            # If we still don't have enough, duplicate existing points
            while len(valid_candidates) < n_points:
                valid_candidates.append(valid_candidates[0])
        
        print(f"Generated {len(valid_candidates)} candidate points")
        return np.array(valid_candidates)
    
    def evaluate_mixture_design(self, design: np.ndarray, model_type: str = "quadratic") -> Dict:
        """
        Evaluate mixture design using various criteria
        
        Parameters:
        design: Design matrix
        model_type: "linear", "quadratic", or "cubic"
        
        Returns:
        Dict: Dictionary of evaluation metrics
        """
        if design.size == 0:
            return {
                "d_efficiency": 0.0,
                "i_efficiency": 0.0,
                "condition_number": float('inf'),
                "average_variance": float('inf'),
                "max_variance": float('inf')
            }
        
        # Calculate D-efficiency
        d_eff = self._calculate_d_efficiency(design, model_type)
        
        # Calculate I-efficiency
        i_eff = self._calculate_i_efficiency(design, model_type)
        
        # Generate model matrix
        X = self._generate_model_matrix(design, model_type)
        
        # Calculate condition number
        try:
            XtX = X.T @ X
            cond_num = np.linalg.cond(XtX)
        except:
            cond_num = float('inf')
        
        # Calculate prediction variance
        try:
            XtX_inv = np.linalg.inv(XtX)
            
            # Generate test points
            test_points = self._generate_candidate_points(1000)
            X_test = self._generate_model_matrix(test_points, model_type)
            
            # Calculate prediction variance for each test point
            pred_var = np.zeros(len(test_points))
            for i in range(len(test_points)):
                x_i = X_test[i]
                pred_var[i] = x_i @ XtX_inv @ x_i
            
            avg_var = np.mean(pred_var)
            max_var = np.max(pred_var)
        except:
            avg_var = float('inf')
            max_var = float('inf')
        
        return {
            "d_efficiency": d_eff,
            "i_efficiency": i_eff,
            "condition_number": cond_num,
            "average_variance": avg_var,
            "max_variance": max_var
        }
    
    def plot_mixture_design(self, design: np.ndarray, title: str = "Mixture Design", 
                           show_labels: bool = True, save_path: str = None):
        """
        Plot mixture design using ternary plot for 3 components or scatter matrix for >3 components
        
        Parameters:
        design: Design matrix
        title: Plot title
        show_labels: Whether to show point labels
        save_path: Path to save plot (if None, plot is displayed)
        """
        if design.size == 0:
            print("Empty design, nothing to plot.")
            return
        
        if self.n_components == 2:
            # For 2 components, use simple line plot
            plt.figure(figsize=(8, 6))
            
            # Sort points by first component
            sorted_indices = np.argsort(design[:, 0])
            sorted_design = design[sorted_indices]
            
            plt.plot(sorted_design[:, 0], sorted_design[:, 1], 'o-')
            
            # Add point labels
            if show_labels:
                for i, point in enumerate(sorted_design):
                    plt.annotate(f"{i+1}", (point[0], point[1]), 
                                xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel(self.component_names[0])
            plt.ylabel(self.component_names[1])
            plt.title(title)
            plt.grid(True)
            
            # Add constraint line
            plt.plot([0, 1], [1, 0], 'k--', alpha=0.3)
            
            # Save or show plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
        
        elif self.n_components == 3:
            # For 3 components, use ternary plot if available
            try:
                import ternary
                
                # Create figure and ternary axis
                fig, ax = plt.subplots(figsize=(8, 7))
                tax = ternary.TernaryAxesSubplot(ax=ax)
                tax.boundary(linewidth=1.0)
                tax.gridlines(color="gray", multiple=0.1, linewidth=0.5)
                
                # Scale and set labels
                tax.set_title(title)
                tax.right_corner_label(self.component_names[0], fontsize=12)
                tax.top_corner_label(self.component_names[1], fontsize=12)
                tax.left_corner_label(self.component_names[2], fontsize=12)
                
                # Convert to ternary coordinates
                points = design[:, :3]
                
                # Plot points
                tax.scatter(points, marker='o', color='blue', s=50, zorder=5)
                
                # Add labels
                if show_labels:
                    for i, point in enumerate(points):
                        tax.annotate(f"{i+1}", point, xytext=(5, 5), 
                                    textcoords='offset points', zorder=6)
                
                # Save or show plot
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                else:
                    plt.show()
                    
            except ImportError:
                print("Ternary plot requires the 'python-ternary' package.")
                print("Install with: pip install python-ternary")
                
                # Fallback: use scatter matrix
                self._plot_scatter_matrix(design, title, show_labels, save_path)
        
        else:
            # For >3 components, use scatter matrix
            self._plot_scatter_matrix(design, title, show_labels, save_path)
    
    def _plot_scatter_matrix(self, design: np.ndarray, title: str, 
                            show_labels: bool, save_path: str = None):
        """
        Plot scatter matrix for designs with >3 components
        """
        # Create figure
        n_plots = min(self.n_components, 4)  # Limit to 4 components for readability
        fig, axes = plt.subplots(n_plots, n_plots, figsize=(10, 10))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        # Add title
        fig.suptitle(title, fontsize=16)
        
        # Plot scatter matrix
        for i in range(n_plots):
            for j in range(n_plots):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(design[:, i], bins=10, color='skyblue', alpha=0.7)
                    ax.set_xlabel(self.component_names[i])
                    ax.set_ylabel('Frequency')
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(design[:, j], design[:, i], color='blue', alpha=0.7)
                    
                    # Add labels
                    if show_labels:
                        for k, point in enumerate(design):
                            ax.annotate(f"{k+1}", (point[j], point[i]), 
                                      xytext=(5, 5), textcoords='offset points')
                    
                    ax.set_xlabel(self.component_names[j])
                    ax.set_ylabel(self.component_names[i])
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def generate_simplex_lattice(self, degree: int) -> np.ndarray:
        """
        Generate a simplex lattice design
        
        Parameters:
        degree: Degree of the simplex lattice design
        
        Returns:
        np.ndarray: Design matrix
        """
        # Calculate all possible combinations of proportions
        proportions = [i/degree for i in range(degree+1)]
        
        # Generate all combinations that sum to 1
        points = []
        for combo in itertools.product(proportions, repeat=self.n_components):
            if abs(sum(combo) - 1.0) < 1e-10:  # Check if sum is approximately 1
                points.append(combo)
        
        # Convert to numpy array
        design = np.array(points)
        
        # Filter points that satisfy bounds
        valid_points = []
        for point in design:
            if check_bounds(point, self.component_bounds):
                valid_points.append(point)
        
        if not valid_points:
            print("Warning: No valid points found for simplex lattice design with given bounds.")
            return np.array([])
        
        design = np.array(valid_points)
        
        # Adjust for fixed components if needed
        design = self._adjust_for_fixed_components(design)
        
        # Post-process design for fixed components if needed
        design = self._post_process_design_fixed_components(design)
        
        return design
