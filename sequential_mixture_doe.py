"""
Sequential Design of Experiments for Mixture Experiments
Handles mixture constraints where components must sum to 1
"""

import numpy as np
import pandas as pd
from mixture_designs import MixtureDesign
from typing import List, Tuple, Optional, Dict
import json
from datetime import datetime

class SequentialMixtureDOE(MixtureDesign):
    """
    Extended Mixture DOE class for sequential experimentation
    """
    
    def __init__(self, n_components: int, component_names: List[str] = None, 
                 component_bounds: List[Tuple[float, float]] = None,
                 fixed_components: Dict[str, float] = None,
                 use_parts_mode: bool = False):
        """
        Initialize sequential mixture DOE generator
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        component_names : List[str]
            Names of components
        component_bounds : List[Tuple[float, float]]
            Min/max bounds for each component
            - If use_parts_mode=False: bounds must be between 0 and 1 (proportions)
            - If use_parts_mode=True: bounds are in parts (any positive values)
        fixed_components : Dict[str, float]
            Components with fixed values
            - If use_parts_mode=False: values are proportions (must sum to < 1)
            - If use_parts_mode=True: values are parts (will be normalized)
        use_parts_mode : bool
            If True, work with parts that get normalized to proportions
        """
        # Store original bounds and fixed values if in parts mode
        self.use_parts_mode = use_parts_mode
        self.original_bounds = []
        self.original_fixed = {}
        
        if use_parts_mode:
            # Store original parts values
            self.original_bounds = component_bounds.copy() if component_bounds else []
            self.original_fixed = fixed_components.copy() if fixed_components else {}
            
            # For parent class, we'll use proportion bounds (0-1)
            if component_bounds:
                component_bounds = [(0.0, 1.0) for _ in component_bounds]
        
        # Initialize parent class
        super().__init__(n_components, component_names, component_bounds)
        self.experiment_history = []
        self.fixed_components = fixed_components or {}
        
        # Validate fixed components for proportion mode
        if not use_parts_mode and self.fixed_components:
            self._validate_fixed_components()
    
    def _validate_fixed_components(self):
        """Validate that fixed components are feasible"""
        fixed_sum = sum(self.fixed_components.values())
        if fixed_sum >= 1.0:
            raise ValueError(f"Fixed components sum to {fixed_sum}, leaving no room for variable components")
        
        # Check that fixed component names are valid
        for comp_name in self.fixed_components:
            if comp_name not in self.component_names:
                raise ValueError(f"Fixed component '{comp_name}' not in component names")
    
    def generate_d_optimal_mixture(self, n_runs: int, model_type: str = "quadratic", 
                                   random_seed: int = None) -> np.ndarray:
        """
        Generate D-optimal mixture design with proper parts handling
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        model_type : str
            Type of model to fit ("linear", "quadratic", "cubic")
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Design matrix (n_runs x n_components) with proportions
        """
        # Generate base design using parent method
        design = super().generate_d_optimal_mixture(n_runs, model_type, random_seed)
        
        # If in parts mode, we need to handle the conversion properly
        if self.use_parts_mode and self.original_bounds:
            # Convert design to parts space for optimization
            design_parts = self._proportions_to_parts_design(design)
            
            # Apply bounds in parts space
            for i in range(self.n_components):
                min_parts, max_parts = self.original_bounds[i]
                design_parts[:, i] = np.clip(design_parts[:, i], min_parts, max_parts)
            
            # Convert back to proportions
            design = self._parts_to_proportions_design(design_parts)
        
        # Apply fixed components
        design = self._adjust_for_fixed_components(design)
        
        return design
    
    def _proportions_to_parts_design(self, design_proportions: np.ndarray) -> np.ndarray:
        """
        Convert a design from proportions to parts based on bounds
        
        Parameters:
        -----------
        design_proportions : np.ndarray
            Design matrix with proportions
            
        Returns:
        --------
        np.ndarray
            Design matrix in parts space
        """
        design_parts = np.zeros_like(design_proportions)
        
        # Scale each component based on its bounds range
        for i in range(self.n_components):
            min_parts, max_parts = self.original_bounds[i]
            # Linear scaling from [0,1] to [min_parts, max_parts]
            design_parts[:, i] = design_proportions[:, i] * (max_parts - min_parts) + min_parts
        
        return design_parts
    
    def _parts_to_proportions_design(self, design_parts: np.ndarray) -> np.ndarray:
        """
        Convert a design matrix from parts to proportions
        
        Parameters:
        -----------
        design_parts : np.ndarray
            Design matrix with values in parts
            
        Returns:
        --------
        np.ndarray
            Design matrix with values as proportions (each row sums to 1)
        """
        # Calculate row sums
        row_sums = design_parts.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        # Normalize each row
        return design_parts / row_sums[:, np.newaxis]
    
    def parts_to_proportions(self, parts_dict: Dict[str, float], 
                           variable_parts: Dict[str, float] = None) -> Dict[str, float]:
        """
        Convert parts to proportions
        
        Parameters:
        -----------
        parts_dict : Dict[str, float]
            Component names and their parts
        variable_parts : Dict[str, float]
            Additional variable components and their parts
            
        Returns:
        --------
        Dict[str, float]
            Component names and their proportions (sum to 1)
        """
        all_parts = parts_dict.copy()
        if variable_parts:
            all_parts.update(variable_parts)
        
        total = sum(all_parts.values())
        if total == 0:
            raise ValueError("Total parts cannot be zero")
            
        return {name: value/total for name, value in all_parts.items()}
    
    def calculate_batch_quantities(self, design_proportions: np.ndarray, 
                                 batch_size: float = 100.0) -> pd.DataFrame:
        """
        Calculate actual quantities for a given batch size
        
        Parameters:
        -----------
        design_proportions : np.ndarray
            Design matrix with proportions (each row sums to 1)
        batch_size : float
            Total batch size (in whatever units you're using)
            
        Returns:
        --------
        pd.DataFrame
            Quantities for each component in each mixture
        """
        quantities = design_proportions * batch_size
        df = pd.DataFrame(quantities, columns=self.component_names)
        df.insert(0, 'Mixture', range(1, len(df) + 1))
        df['Total'] = df[self.component_names].sum(axis=1)
        return df
    
    def _adjust_for_fixed_components(self, design: np.ndarray) -> np.ndarray:
        """
        Adjust design matrix to account for fixed components
        """
        if not self.fixed_components:
            return design
        
        if self.use_parts_mode:
            # In parts mode, we need to work differently
            # First, convert design to parts
            design_parts = np.zeros_like(design)
            
            # For each mixture (row)
            for row_idx in range(design.shape[0]):
                # Start with fixed components in parts
                total_fixed_parts = 0
                for comp_name, fixed_parts in self.original_fixed.items():
                    comp_idx = self.component_names.index(comp_name)
                    design_parts[row_idx, comp_idx] = fixed_parts
                    total_fixed_parts += fixed_parts
                
                # For variable components, distribute the remaining proportion
                # First, get the target sum for variable components
                variable_indices = []
                for i, name in enumerate(self.component_names):
                    if name not in self.fixed_components:
                        variable_indices.append(i)
                
                if variable_indices:
                    # Get the original proportions for variable components
                    var_props = design[row_idx, variable_indices]
                    var_props = var_props / var_props.sum() if var_props.sum() > 0 else np.ones(len(variable_indices)) / len(variable_indices)
                    
                    # Assign parts to variable components based on their bounds
                    # Try to maintain relative proportions while respecting bounds
                    for i, idx in enumerate(variable_indices):
                        min_parts, max_parts = self.original_bounds[idx]
                        # Start with proportional allocation
                        target_parts = var_props[i] * (max_parts - min_parts) + min_parts
                        design_parts[row_idx, idx] = np.clip(target_parts, min_parts, max_parts)
            
            # Convert back to proportions
            design = self._parts_to_proportions_design(design_parts)
        else:
            # Original proportion mode logic
            # Get indices of fixed and variable components
            fixed_indices = []
            variable_indices = []
            
            for i, name in enumerate(self.component_names):
                if name in self.fixed_components:
                    fixed_indices.append(i)
                else:
                    variable_indices.append(i)
            
            # Set fixed component values
            for idx, name in enumerate(self.component_names):
                if name in self.fixed_components:
                    design[:, idx] = self.fixed_components[name]
            
            # Rescale variable components to sum to (1 - fixed_sum)
            fixed_sum = sum(self.fixed_components.values())
            remaining_sum = 1.0 - fixed_sum
            
            if variable_indices:
                variable_sum = design[:, variable_indices].sum(axis=1)
                for idx in variable_indices:
                    design[:, idx] = design[:, idx] / variable_sum * remaining_sum
        
        return design
    
    def augment_mixture_design(self, existing_design: np.ndarray, 
                              n_additional_runs: int, 
                              model_type: str = "quadratic",
                              focus_components: List[str] = None,
                              avoid_region: Optional[np.ndarray] = None,
                              random_seed: int = None) -> np.ndarray:
        """
        Augment an existing mixture design with additional experiments
        
        Parameters:
        -----------
        existing_design : np.ndarray
            Current mixture design matrix (n_existing_runs x n_components)
        n_additional_runs : int
            Number of new mixture experiments to add
        model_type : str
            "linear", "quadratic", or "cubic"
        focus_components : List[str]
            Components to focus on based on screening results
        avoid_region : np.ndarray, optional
            Mixture compositions to avoid
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            New mixture experiments only (n_additional_runs x n_components)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate candidate mixtures
        n_candidates = max(1000, n_additional_runs * 50)
        candidates = self._generate_mixture_candidates(n_candidates, focus_components)
        
        # Apply fixed components if any
        candidates = self._adjust_for_fixed_components(candidates)
        
        # Remove points too close to existing design
        candidates = self._filter_mixture_candidates(candidates, existing_design)
        
        # Remove avoided regions
        if avoid_region is not None:
            candidates = self._remove_avoided_mixtures(candidates, avoid_region)
        
        # Select additional points using D-optimal criterion
        augmented_design = existing_design.copy()
        new_points = []
        
        for _ in range(n_additional_runs):
            best_candidate = None
            best_d_eff = -np.inf
            
            # Test each candidate
            for candidate in candidates[:200]:  # Test subset for efficiency
                # Create temporary augmented design
                temp_design = np.vstack([augmented_design, 
                                       np.array(new_points + [candidate])])
                
                # Calculate D-efficiency
                d_eff = self._calculate_mixture_d_efficiency(temp_design, model_type)
                
                if d_eff > best_d_eff:
                    best_d_eff = d_eff
                    best_candidate = candidate
            
            if best_candidate is not None:
                new_points.append(best_candidate)
                # Remove selected point and nearby points
                distances = np.sum((candidates - best_candidate)**2, axis=1)
                candidates = candidates[distances > 0.001]
            else:
                print(f"Warning: Could only add {len(new_points)} mixture points")
                break
        
        return np.array(new_points)
    
    def _generate_mixture_candidates(self, n_candidates: int, 
                                   focus_components: List[str] = None) -> np.ndarray:
        """Generate candidate mixture compositions"""
        if self.use_parts_mode:
            # Generate candidates in parts space
            candidates_parts = np.zeros((n_candidates, self.n_components))
            
            # For each candidate
            for i in range(n_candidates):
                # Randomly assign parts within bounds
                for j in range(self.n_components):
                    if self.component_names[j] in self.fixed_components:
                        # Use fixed value
                        candidates_parts[i, j] = self.original_fixed[self.component_names[j]]
                    else:
                        # Random value within bounds
                        min_parts, max_parts = self.original_bounds[j]
                        candidates_parts[i, j] = np.random.uniform(min_parts, max_parts)
                
                # If focusing on certain components, bias towards them
                if focus_components:
                    for j, name in enumerate(self.component_names):
                        if name in focus_components and name not in self.fixed_components:
                            min_parts, max_parts = self.original_bounds[j]
                            # Bias towards higher values for focus components
                            candidates_parts[i, j] = np.random.uniform((min_parts + max_parts) / 2, max_parts)
            
            # Convert to proportions
            candidates = self._parts_to_proportions_design(candidates_parts)
        else:
            # Original proportion-based generation
            # Get variable components only
            variable_indices = []
            for i, name in enumerate(self.component_names):
                if name not in self.fixed_components:
                    variable_indices.append(i)
            
            n_variable = len(variable_indices)
            
            if n_variable == 0:
                raise ValueError("No variable components to optimize")
            
            # Generate random mixtures for variable components
            candidates = np.zeros((n_candidates, self.n_components))
            
            # Use Dirichlet distribution for mixture generation
            if focus_components:
                # Check if all variable components are focused
                all_focused = all(self.component_names[idx] in focus_components 
                                 for idx in variable_indices)
                
                if all_focused:
                    # If all components are focused, use diverse sampling strategies
                    strategies = ['uniform', 'vertices', 'edges', 'center']
                    variable_mixtures = []
                    
                    for i in range(n_candidates):
                        strategy = strategies[i % len(strategies)]
                        
                        if strategy == 'uniform':
                            # Uniform sampling
                            alpha = np.ones(n_variable)
                            mixture = np.random.dirichlet(alpha, 1)[0]
                        elif strategy == 'vertices':
                            # Near vertices
                            vertex_idx = i % n_variable
                            mixture = np.ones(n_variable) * 0.1
                            mixture[vertex_idx] = 1.0 - (n_variable - 1) * 0.1
                            # Add some noise
                            mixture += np.random.uniform(-0.05, 0.05, n_variable)
                            mixture = np.maximum(mixture, 0)
                            mixture = mixture / mixture.sum()
                        elif strategy == 'edges':
                            # Along edges
                            edge_idx = i % (n_variable * (n_variable - 1) // 2)
                            # Find which edge
                            edge_count = 0
                            for j in range(n_variable):
                                for k in range(j + 1, n_variable):
                                    if edge_count == edge_idx:
                                        mixture = np.zeros(n_variable)
                                        t = np.random.uniform(0.2, 0.8)
                                        mixture[j] = t
                                        mixture[k] = 1 - t
                                        # Add small amounts to other components
                                        for m in range(n_variable):
                                            if m != j and m != k:
                                                mixture[m] = 0.05
                                        mixture = mixture / mixture.sum()
                                        break
                                    edge_count += 1
                                if edge_count > edge_idx:
                                    break
                        else:  # center
                            # Near center
                            mixture = np.ones(n_variable) / n_variable
                            mixture += np.random.uniform(-0.1, 0.1, n_variable)
                            mixture = np.maximum(mixture, 0)
                            mixture = mixture / mixture.sum()
                        
                        variable_mixtures.append(mixture)
                    
                    variable_mixtures = np.array(variable_mixtures)
                else:
                    # Original logic - focus on specific components
                    alpha = np.ones(n_variable)
                    for i, idx in enumerate(variable_indices):
                        if self.component_names[idx] in focus_components:
                            alpha[i] = 3.0  # Higher weight for focus components
                    variable_mixtures = np.random.dirichlet(alpha, n_candidates)
            else:
                alpha = np.ones(n_variable)
                variable_mixtures = np.random.dirichlet(alpha, n_candidates)
            
            # Assign to full mixture array
            for i, idx in enumerate(variable_indices):
                candidates[:, idx] = variable_mixtures[:, i]
            
            # Apply bounds to each component
            for i in range(self.n_components):
                min_val, max_val = self.component_bounds[i]
                candidates[:, i] = np.clip(candidates[:, i], min_val, max_val)
            
            # Renormalize to ensure sum to 1
            candidates = self._normalize_mixtures(candidates)
        
        return candidates
    
    def _normalize_mixtures(self, mixtures: np.ndarray) -> np.ndarray:
        """Normalize mixtures to sum to 1"""
        row_sums = mixtures.sum(axis=1)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        return mixtures / row_sums[:, np.newaxis]
    
    def _filter_mixture_candidates(self, candidates: np.ndarray, 
                                 existing_design: np.ndarray, 
                                 min_distance: float = 0.02) -> np.ndarray:
        """Remove candidates too close to existing mixture points"""
        filtered = []
        
        for candidate in candidates:
            # Calculate distances to all existing points
            distances = np.sqrt(np.sum((existing_design - candidate)**2, axis=1))
            
            # Keep if far enough from all existing points
            if np.min(distances) > min_distance:
                filtered.append(candidate)
        
        return np.array(filtered) if filtered else candidates
    
    def _remove_avoided_mixtures(self, candidates: np.ndarray, 
                                avoided_regions: np.ndarray,
                                buffer: float = 0.05) -> np.ndarray:
        """Remove candidates near avoided mixture regions"""
        filtered = []
        
        for candidate in candidates:
            distances = np.sqrt(np.sum((avoided_regions - candidate)**2, axis=1))
            if np.min(distances) > buffer:
                filtered.append(candidate)
        
        return np.array(filtered) if filtered else candidates
    
    def _calculate_mixture_d_efficiency(self, design: np.ndarray, 
                                      model_type: str) -> float:
        """Calculate D-efficiency for mixture design"""
        # Create model matrix for mixture
        if model_type == "linear":
            # Scheffé linear model (no intercept)
            model_matrix = design
        elif model_type == "quadratic":
            # Scheffé quadratic model
            n_runs = design.shape[0]
            n_terms = self.n_components + (self.n_components * (self.n_components - 1)) // 2
            model_matrix = np.zeros((n_runs, n_terms))
            
            # Linear terms
            model_matrix[:, :self.n_components] = design
            
            # Interaction terms
            col = self.n_components
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    model_matrix[:, col] = design[:, i] * design[:, j]
                    col += 1
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Calculate D-efficiency
        try:
            XtX = model_matrix.T @ model_matrix
            det_XtX = np.linalg.det(XtX)
            n_params = model_matrix.shape[1]
            d_eff = (det_XtX / len(design)) ** (1 / n_params)
            return d_eff
        except:
            return -1e6
    
    def get_mixture_recommendations(self, n_variable_components: int) -> dict:
        """
        Get recommendations for sequential mixture experimentation
        
        Parameters:
        -----------
        n_variable_components : int
            Number of components that can vary (not fixed)
            
        Returns:
        --------
        dict
            Recommendations for Stage 1 and Stage 2
        """
        # Stage 1: Linear mixture model
        stage1_params = n_variable_components  # No intercept in mixture models
        stage1_min = stage1_params
        stage1_rec = max(int(np.ceil(stage1_params * 1.5)), stage1_params + 3)
        stage1_exc = max(stage1_params * 2, stage1_params + 5)
        
        # Stage 2: Quadratic mixture model
        quad_params = n_variable_components + (n_variable_components * (n_variable_components - 1)) // 2
        total_rec = max(int(np.ceil(quad_params * 1.5)), quad_params + 5)
        stage2_rec = max(total_rec - stage1_rec, 5)
        
        return {
            'stage1': {
                'purpose': 'Screening (Linear Mixture Model)',
                'minimum': stage1_min,
                'recommended': stage1_rec,
                'excellent': stage1_exc,
                'can_fit': f'Linear mixture model with {stage1_params} parameters',
                'note': 'No intercept in mixture models'
            },
            'stage2': {
                'purpose': 'Optimization (Quadratic Mixture Model)',
                'recommended_additional': stage2_rec,
                'total_runs': stage1_rec + stage2_rec,
                'total_parameters': quad_params,
                'efficiency_gain': 'Focuses on promising mixture regions'
            },
            'mixture_specific': {
                'fixed_components': len(self.fixed_components),
                'variable_components': n_variable_components,
                'constraints': 'All components sum to 1',
                'bounds': 'All components between 0 and 1'
            }
        }
    
    def save_mixture_stage(self, design: np.ndarray, stage_name: str, 
                         metadata: dict = None) -> str:
        """Save mixture experiment stage to history"""
        stage_info = {
            'stage_name': stage_name,
            'timestamp': datetime.now().isoformat(),
            'design': design.tolist(),
            'n_runs': len(design),
            'n_components': self.n_components,
            'component_names': self.component_names,
            'fixed_components': self.fixed_components,
            'metadata': metadata or {}
        }
        
        self.experiment_history.append(stage_info)
        
        # Save to file
        filename = f"sequential_mixture_{stage_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(stage_info, f, indent=2)
        
        return filename


# Example usage
def sequential_mixture_example():
    """Example: Sequential mixture experimentation with parts mode"""
    print("=== Sequential Mixture DOE Example (Parts Mode) ===\n")
    
    # Define mixture with parts
    component_names = ['Polymer_A', 'Polymer_B', 'Filler', 'Additive_1', 'Additive_2']
    
    # Bounds in parts (not proportions)
    component_bounds_parts = [
        (1.0, 3.0),    # Polymer_A: 1-3 parts
        (1.0, 3.0),    # Polymer_B: 1-3 parts  
        (0.2, 1.0),    # Filler: 0.2-1 parts
        (0.02, 0.1),   # Additive_1: 0.02-0.1 parts
        (0.02, 0.1)    # Additive_2: 0.02-0.1 parts
    ]
    
    # Fix one component in parts
    fixed_components_parts = {'Additive_2': 0.05}  # Always 0.05 parts
    
    # Create sequential mixture DOE in parts mode
    seq_mix = SequentialMixtureDOE(
        n_components=5,
        component_names=component_names,
        component_bounds=component_bounds_parts,
        fixed_components=fixed_components_parts,
        use_parts_mode=True
    )
    
    # Get recommendations
    n_variable = 4  # 5 components - 1 fixed
    recommendations = seq_mix.get_mixture_recommendations(n_variable)
    
    print("Stage 1 Recommendations:")
    print(f"  Minimum: {recommendations['stage1']['minimum']} runs")
    print(f"  Recommended: {recommendations['stage1']['recommended']} runs")
    print(f"  Can fit: {recommendations['stage1']['can_fit']}")
    
    # Generate Stage 1 design
    stage1_design = seq_mix.generate_d_optimal_mixture(
        n_runs=recommendations['stage1']['recommended'],
        model_type="linear",
        random_seed=42
    )
    
    print(f"\nStage 1 Design Generated: {len(stage1_design)} mixtures")
    print("First 3 mixtures (proportions):")
    for i in range(min(3, len(stage1_design))):
        print(f"  Mix {i+1}: {stage1_design[i].round(3)}")
        print(f"    Sum: {stage1_design[i].sum():.3f}")
    
    # Show batch quantities for 100 unit batch
    batch_quantities = seq_mix.calculate_batch_quantities(stage1_design[:3], batch_size=100)
    print("\nBatch Quantities (100 unit batch):")
    print(batch_quantities.round(2))
    
    # Generate Stage 2 augmentation
    stage2_design = seq_mix.augment_mixture_design(
        stage1_design,
        n_additional_runs=recommendations['stage2']['recommended_additional'],
        model_type="quadratic",
        focus_components=['Polymer_A', 'Polymer_B'],  # Focus on main polymers
        random_seed=43
    )
    
    print(f"\nStage 2 Augmentation: {len(stage2_design)} additional mixtures")
    print("Focused on Polymer_A and Polymer_B based on Stage 1 results")
    
    # Verify all constraints
    all_designs = np.vstack([stage1_design, stage2_design])
    print(f"\nVerification:")
    print(f"  All sums close to 1: {np.allclose(all_designs.sum(axis=1), 1.0)}")
    print(f"  All values non-negative: {np.all(all_designs >= 0)}")
    
    return seq_mix, stage1_design, stage2_design


if __name__ == "__main__":
    seq_mix, stage1, stage2 = sequential_mixture_example()
