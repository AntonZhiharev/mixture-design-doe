"""
Fixed Parts Mixture Design Implementation
Based on TRUE understanding of Fixed Components

CORRECT UNDERSTANDING of Fixed Components:
- Fixed components have CONSTANT amounts in PARTS (e.g., always 10 parts)
- Variable components have VARIABLE amounts in PARTS (e.g., 0-20 parts)  
- Fixed components have VARIABLE PROPORTIONS (because total batch changes)
- They reduce design space by consuming fixed material amounts

This module provides a complete implementation with improved candidate generation
for better space-filling properties and distribution across the design space.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import warnings
from scipy.spatial.distance import pdist

class FixedPartsMixtureDesign:
    """
    Complete implementation for generating mixture designs with fixed components.
    
    Features improved candidate generation with:
    - Latin Hypercube Sampling for space-filling
    - Enhanced structured points for boundary coverage
    - Proportional candidate generation for constraint awareness
    - Multi-strategy candidate mix for optimal exploration
    """
    
    def __init__(self, component_names: List[str], 
                 fixed_parts: Dict[str, float] = None,
                 variable_bounds: Dict[str, Tuple[float, float]] = None,
                 enable_anti_clustering: bool = None,
                 min_distance_factor: float = 0.15,
                 space_filling_weight: float = 0.3,
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
        enable_anti_clustering : bool, optional
            Enable anti-clustering algorithm. If None, auto-detects based on constraint level
        min_distance_factor : float, default 0.15
            Minimum distance between points as fraction of space diagonal (anti-clustering)
        space_filling_weight : float, default 0.3
            Weight for space-filling objective vs D-efficiency (anti-clustering)
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
        
        # Anti-clustering parameters
        self.min_distance_factor = min_distance_factor
        self.space_filling_weight = space_filling_weight
        
        # Setup derived attributes
        self.variable_names = [name for name in self.component_names if name not in self.fixed_parts]
        self.fixed_names = list(self.fixed_parts.keys())
        self.n_components = len(self.component_names)
        self.n_fixed = len(self.fixed_parts)
        self.n_variable = len(self.variable_names)
        
        # Calculate design space properties
        self.total_fixed_parts = sum(self.fixed_parts.values())
        self.variable_total_range = self._calculate_variable_ranges()
        
        # Auto-detect clustering risk and setup anti-clustering
        if enable_anti_clustering is None:
            self.enable_anti_clustering = self._should_enable_anti_clustering()
        else:
            self.enable_anti_clustering = enable_anti_clustering
        
        # Setup anti-clustering metrics if enabled
        if self.enable_anti_clustering:
            self._setup_anti_clustering_metrics()
        
        # Validate inputs (after setting enable_anti_clustering)
        self._validate_inputs()
        
        # Store last generated design for backward compatibility
        self.last_design = None
        self.last_parts_design = None
        self.last_batch_sizes = None
    
    def _calculate_variable_ranges(self) -> Tuple[float, float]:
        """Calculate the range of total variable parts."""
        min_total = sum(bounds[0] for bounds in self.variable_bounds.values())
        max_total = sum(bounds[1] for bounds in self.variable_bounds.values())
        return min_total, max_total
    
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
        
        if self.enable_anti_clustering:
            print(f"   🎯 Anti-clustering ENABLED - Enhanced space-filling active")
            print(f"   Anti-clustering factor: {self.min_distance_factor}")
            print(f"   Space-filling weight: {self.space_filling_weight}")
        else:
            print(f"   📊 Standard algorithm - Enhanced candidate generation")
    
    def _should_enable_anti_clustering(self) -> bool:
        """Auto-detect if anti-clustering should be enabled based on constraint level."""
        if self.n_variable < 2:
            return False  # Not applicable for low dimensions
        
        # Calculate constraint level
        total_batch_range = self.total_fixed_parts + self.variable_total_range[1]
        fixed_ratio = self.total_fixed_parts / total_batch_range
        
        # Calculate variable space size (normalized)
        var_space_sizes = []
        for name in self.variable_names:
            min_val, max_val = self.variable_bounds[name]
            var_space_sizes.append(max_val - min_val)
        
        avg_var_range = np.mean(var_space_sizes) if var_space_sizes else 0
        
        # Enable anti-clustering if:
        # 1. High fixed ratio (>40%) OR
        # 2. Small variable ranges (<30) OR  
        # 3. High constraint scenarios
        enable = (fixed_ratio > 0.4) or (avg_var_range < 30) or (total_batch_range / avg_var_range > 3 if avg_var_range > 0 else True)
        
        print(f"   Auto-detection: Fixed ratio={fixed_ratio:.1%}, Avg var range={avg_var_range:.1f}")
        print(f"   Anti-clustering auto-enabled: {enable}")
        
        return enable
    
    def _setup_anti_clustering_metrics(self):
        """Setup anti-clustering space metrics."""
        if self.n_variable >= 2:
            # Calculate space diagonal using first two variable components
            var_ranges = []
            for name in self.variable_names[:2]:
                min_val, max_val = self.variable_bounds[name]
                var_ranges.append(max_val - min_val)
            
            self.space_diagonal = np.sqrt(sum(r**2 for r in var_ranges))
            self.min_distance_threshold = self.min_distance_factor * self.space_diagonal
        else:
            self.space_diagonal = max(self.variable_bounds[self.variable_names[0]]) if self.variable_names else 1.0
            self.min_distance_threshold = self.min_distance_factor * self.space_diagonal
        
        print(f"   Space diagonal: {self.space_diagonal:.3f}")
        print(f"   Minimum distance threshold: {self.min_distance_threshold:.3f}")
    
    def generate_design(self, n_runs: int, design_type: str = "d-optimal", 
                       model_type: str = "quadratic", max_iter: int = 1000, 
                       random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate mixture design with fixed components using improved candidate generation.
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        design_type : str, default "d-optimal"
            Type of design to generate ("d-optimal", "i-optimal", "space-filling")
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
        algorithm_type = "ANTI-CLUSTERING" if self.enable_anti_clustering else "IMPROVED"
        print(f"\n🚀 Generating {algorithm_type} {design_type} design with {n_runs} runs")
        print(f"   Model type: {model_type}")
        print(f"   Fixed components: {list(self.fixed_parts.keys())}")
        
        if self.enable_anti_clustering:
            print(f"   Using anti-clustering algorithm for better space-filling!")
            print(f"   Anti-clustering factor: {self.min_distance_factor}")
            print(f"   Space-filling weight: {self.space_filling_weight}")
        else:
            print(f"   Using enhanced candidate generation for better space-filling!")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Choose generation method based on anti-clustering setting
        if self.enable_anti_clustering:
            if design_type.lower() == "d-optimal":
                parts_design, prop_design, batch_sizes = self._generate_anti_clustering_d_optimal(
                    n_runs, model_type, max_iter, random_seed
                )
            elif design_type.lower() == "i-optimal":
                parts_design, prop_design, batch_sizes = self._generate_anti_clustering_i_optimal(
                    n_runs, model_type, max_iter, random_seed
                )
            elif design_type.lower() == "space-filling":
                parts_design, prop_design, batch_sizes = self._generate_space_filling_design(
                    n_runs, random_seed
                )
            else:
                raise ValueError(f"Unsupported design type: {design_type}. Use 'd-optimal', 'i-optimal', or 'space-filling'.")
        else:
            if design_type.lower() == "d-optimal":
                parts_design, prop_design, batch_sizes = self._generate_d_optimal_design(
                    n_runs=n_runs,
                    model_type=model_type,
                    max_iter=max_iter,
                    n_candidates=2000,
                    random_seed=random_seed
                )
            elif design_type.lower() == "i-optimal":
                parts_design, prop_design, batch_sizes = self._generate_i_optimal_design(
                    n_runs=n_runs,
                    model_type=model_type,
                    max_iter=max_iter,
                    n_candidates=2000,
                    random_seed=random_seed
                )
            else:
                raise ValueError(f"Unsupported design type: {design_type}. Use 'd-optimal' or 'i-optimal'. For space-filling, enable anti-clustering.")
        
        # Store results for backward compatibility
        self.last_design = prop_design
        self.last_parts_design = parts_design
        self.last_batch_sizes = batch_sizes
        
        # Create comprehensive results DataFrame
        results_df = self._create_results_dataframe(parts_design, prop_design, batch_sizes)
        
        print(f"\n✅ {algorithm_type} Design Generated Successfully!")
        print(f"   Runs: {len(results_df)}")
        print(f"   Fixed components maintain constant parts")
        print(f"   Variable components follow specified bounds")
        print(f"   Batch sizes: {batch_sizes.min():.1f} to {batch_sizes.max():.1f} parts")
        
        # Display appropriate space-filling analysis
        if self.enable_anti_clustering:
            self._analyze_anti_clustering_performance(parts_design)
        else:
            self._display_space_filling_analysis(parts_design)
        
        return results_df
    
    def _generate_d_optimal_design(self, n_runs: int, model_type: str = "quadratic", 
                                 max_iter: int = 1000, n_candidates: int = 2000,
                                 random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate D-optimal design using improved candidate generation.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (parts_design, proportions_design, batch_sizes)
        """
        print(f"\nGenerating improved D-optimal design for fixed components mixture:")
        print(f"  Runs: {n_runs}")
        print(f"  Model: {model_type}")
        print(f"  Fixed components: {list(self.fixed_parts.keys())}")
        
        # Generate improved candidate set
        candidate_parts, candidate_props, candidate_batches = self._generate_improved_candidate_set(n_candidates)
        
        print(f"\nImproved candidate set generated:")
        print(f"  Total candidates: {len(candidate_parts)}")
        
        # Initialize with best candidates using improved selection
        initial_indices = self._select_initial_design(candidate_props, n_runs, model_type)
        current_design_indices = list(initial_indices)
        
        # Calculate initial D-efficiency
        initial_design = candidate_props[current_design_indices]
        try:
            initial_d_efficiency = self._calculate_d_efficiency(initial_design, model_type)
            print(f"  Initial D-efficiency: {initial_d_efficiency:.6e}")
        except:
            initial_d_efficiency = -np.inf
            print(f"  Initial D-efficiency: calculation failed")
        
        # D-optimal coordinate exchange with improved algorithm
        best_d_efficiency = initial_d_efficiency
        best_design_indices = current_design_indices.copy()
        
        print(f"\nRunning improved coordinate exchange optimization:")
        
        for iteration in range(max_iter):
            improved = False
            
            # Try to replace each point in the design
            for i in range(n_runs):
                current_idx = current_design_indices[i]
                best_replacement_idx = current_idx
                best_replacement_efficiency = best_d_efficiency
                
                # Try replacing with each candidate
                for candidate_idx in range(len(candidate_props)):
                    if candidate_idx in current_design_indices:
                        continue
                    
                    # Create test design with replacement
                    test_indices = current_design_indices.copy()
                    test_indices[i] = candidate_idx
                    test_design = candidate_props[test_indices]
                    
                    # Calculate D-efficiency
                    try:
                        d_efficiency = self._calculate_d_efficiency(test_design, model_type)
                        
                        if d_efficiency > best_replacement_efficiency:
                            best_replacement_efficiency = d_efficiency
                            best_replacement_idx = candidate_idx
                            improved = True
                    except:
                        continue
                
                # Apply best replacement
                if best_replacement_idx != current_idx:
                    current_design_indices[i] = best_replacement_idx
                    best_d_efficiency = best_replacement_efficiency
            
            # Update best design if improved
            if improved:
                best_design_indices = current_design_indices.copy()
                if (iteration + 1) % 100 == 0:
                    print(f"  Iteration {iteration + 1}: D-efficiency = {best_d_efficiency:.6e}")
            else:
                break
        
        final_d_efficiency = best_d_efficiency
        print(f"  Final D-efficiency after {iteration + 1} iterations: {final_d_efficiency:.6e}")
        
        # Extract final design
        final_parts = candidate_parts[best_design_indices]
        final_props = candidate_props[best_design_indices]
        final_batches = candidate_batches[best_design_indices]
        
        # Verify design properties
        self._verify_design_properties(final_parts, final_props, final_batches)
        
        return final_parts, final_props, final_batches
    
    def _generate_i_optimal_design(self, n_runs: int, model_type: str = "quadratic", 
                                 max_iter: int = 1000, n_candidates: int = 2000,
                                 random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate I-optimal design using improved candidate generation.
        
        I-optimal designs minimize the average prediction variance over the design region.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (parts_design, proportions_design, batch_sizes)
        """
        print(f"\nGenerating improved I-optimal design for fixed components mixture:")
        print(f"  Runs: {n_runs}")
        print(f"  Model: {model_type}")
        print(f"  Fixed components: {list(self.fixed_parts.keys())}")
        
        # Generate improved candidate set
        candidate_parts, candidate_props, candidate_batches = self._generate_improved_candidate_set(n_candidates)
        
        print(f"\nImproved candidate set generated:")
        print(f"  Total candidates: {len(candidate_parts)}")
        
        # Initialize with best candidates using improved selection
        initial_indices = self._select_initial_design(candidate_props, n_runs, model_type)
        current_design_indices = list(initial_indices)
        
        # Calculate initial I-efficiency
        initial_design = candidate_props[current_design_indices]
        try:
            initial_i_efficiency = self._calculate_i_efficiency(initial_design, candidate_props, model_type)
            print(f"  Initial I-efficiency: {initial_i_efficiency:.6e}")
        except:
            initial_i_efficiency = -np.inf
            print(f"  Initial I-efficiency: calculation failed")
        
        # I-optimal coordinate exchange with improved algorithm
        best_i_efficiency = initial_i_efficiency
        best_design_indices = current_design_indices.copy()
        
        print(f"\nRunning improved I-optimal coordinate exchange optimization:")
        
        for iteration in range(max_iter):
            improved = False
            
            # Try to replace each point in the design
            for i in range(n_runs):
                current_idx = current_design_indices[i]
                best_replacement_idx = current_idx
                best_replacement_efficiency = best_i_efficiency
                
                # Try replacing with each candidate
                for candidate_idx in range(len(candidate_props)):
                    if candidate_idx in current_design_indices:
                        continue
                    
                    # Create test design with replacement
                    test_indices = current_design_indices.copy()
                    test_indices[i] = candidate_idx
                    test_design = candidate_props[test_indices]
                    
                    # Calculate I-efficiency
                    try:
                        i_efficiency = self._calculate_i_efficiency(test_design, candidate_props, model_type)
                        
                        if i_efficiency > best_replacement_efficiency:
                            best_replacement_efficiency = i_efficiency
                            best_replacement_idx = candidate_idx
                            improved = True
                    except:
                        continue
                
                # Apply best replacement
                if best_replacement_idx != current_idx:
                    current_design_indices[i] = best_replacement_idx
                    best_i_efficiency = best_replacement_efficiency
            
            # Update best design if improved
            if improved:
                best_design_indices = current_design_indices.copy()
                if (iteration + 1) % 100 == 0:
                    print(f"  Iteration {iteration + 1}: I-efficiency = {best_i_efficiency:.6e}")
            else:
                break
        
        final_i_efficiency = best_i_efficiency
        print(f"  Final I-efficiency after {iteration + 1} iterations: {final_i_efficiency:.6e}")
        
        # Extract final design
        final_parts = candidate_parts[best_design_indices]
        final_props = candidate_props[best_design_indices]
        final_batches = candidate_batches[best_design_indices]
        
        # Verify design properties
        self._verify_design_properties(final_parts, final_props, final_batches)
        
        return final_parts, final_props, final_batches
    
    def _calculate_i_efficiency(self, design: np.ndarray, candidates: np.ndarray, model_type: str) -> float:
        """
        Calculate I-efficiency of the design.
        
        I-efficiency is related to the average prediction variance over the design region.
        """
        try:
            # Build model matrix for design
            X_design = self._build_model_matrix(design, model_type)
            
            # Calculate information matrix
            XTX = X_design.T @ X_design
            
            # Check if matrix is invertible
            try:
                XTX_inv = np.linalg.inv(XTX)
            except np.linalg.LinAlgError:
                return -np.inf
            
            # For I-optimality, use a subset of candidate points for computational efficiency
            # Take every 10th candidate to reduce computation while maintaining coverage
            candidate_subset = candidates[::10] if len(candidates) > 100 else candidates
            
            # Build model matrix for candidates (prediction points)
            X_pred = self._build_model_matrix(candidate_subset, model_type)
            
            # Calculate average prediction variance
            avg_pred_var = 0.0
            n_pred = len(X_pred)
            
            for i in range(n_pred):
                x_pred = X_pred[i:i+1, :]  # Single row as matrix
                pred_var = x_pred @ XTX_inv @ x_pred.T
                avg_pred_var += pred_var[0, 0]
            
            avg_pred_var /= n_pred
            
            # I-efficiency is inverse of average prediction variance
            i_efficiency = 1.0 / avg_pred_var if avg_pred_var > 0 else 0.0
            
            return i_efficiency
        except Exception as e:
            # Print debug info for troubleshooting
            print(f"    I-efficiency calculation error: {e}")
            return -np.inf
    
    def _generate_improved_candidate_set(self, n_candidates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate improved candidate set using multiple strategies for better space-filling.
        
        Parameters:
        -----------
        n_candidates : int
            Total number of candidates to generate
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (parts_candidates, prop_candidates, batch_candidates)
        """
        print(f"  Variable component total range: [{self.variable_total_range[0]:.3f}, {self.variable_total_range[1]:.3f}] parts")
        
        # Calculate candidate distribution
        n_structured = min(50, max(2**self.n_variable + 5, 9))  # Enhanced structured points
        n_lhs = int(0.30 * n_candidates)  # 30% Latin Hypercube
        n_proportional = int(0.40 * n_candidates)  # 40% Proportional
        n_random = n_candidates - n_structured - n_lhs - n_proportional  # Remaining random
        
        all_parts = []
        all_props = []
        all_batches = []
        
        # 1. Enhanced Structured Points (corners, edges, centers)
        structured_parts = self._generate_enhanced_structured_points()
        print(f"  Generated {len(structured_parts)} enhanced structured points")
        
        for parts in structured_parts:
            props = self._parts_to_proportions(parts)
            batch_size = np.sum(parts)
            
            all_parts.append(parts)
            all_props.append(props)
            all_batches.append(batch_size)
        
        # 2. Latin Hypercube Sampling for space-filling
        if n_lhs > 0:
            lhs_parts = self._generate_lhs_candidates(n_lhs)
            print(f"  Generated {len(lhs_parts)} LHS candidates")
            
            for parts in lhs_parts:
                props = self._parts_to_proportions(parts)
                batch_size = np.sum(parts)
                
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(batch_size)
        
        # 3. Proportional candidates (constraint-aware)
        if n_proportional > 0:
            prop_parts = self._generate_proportional_candidates(n_proportional)
            print(f"  Generated {len(prop_parts)} proportional candidates")
            
            for parts in prop_parts:
                props = self._parts_to_proportions(parts)
                batch_size = np.sum(parts)
                
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(batch_size)
        
        # 4. Random candidates for diversity
        if n_random > 0:
            random_parts = self._generate_random_candidates(n_random)
            print(f"  Generated {len(random_parts)} random candidates")
            
            for parts in random_parts:
                props = self._parts_to_proportions(parts)
                batch_size = np.sum(parts)
                
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(batch_size)
        
        # Convert to arrays
        candidate_parts = np.array(all_parts)
        candidate_props = np.array(all_props)
        candidate_batches = np.array(all_batches)
        
        print(f"\nImproved candidate set generated:")
        print(f"  Total candidates: {len(candidate_parts)}")
        print(f"  Structured points: {n_structured}")
        print(f"  LHS candidates: {n_lhs}")
        print(f"  Proportional candidates: {n_proportional}")
        print(f"  Random candidates: {n_random}")
        
        return candidate_parts, candidate_props, candidate_batches
    
    def _generate_enhanced_structured_points(self) -> List[np.ndarray]:
        """Generate enhanced structured points including corners, edges, centers."""
        structured_points = []
        
        # All corner points of variable space
        if self.n_variable <= 4:  # Only for reasonable dimensions
            for i in range(2**self.n_variable):
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[j] = self.fixed_parts[name]
                
                # Set variable components to corners
                var_idx = 0
                for j, name in enumerate(self.component_names):
                    if name not in self.fixed_parts:
                        min_val, max_val = self.variable_bounds[name]
                        # Use binary representation to get corners
                        if (i >> var_idx) & 1:
                            parts[j] = max_val
                        else:
                            parts[j] = min_val
                        var_idx += 1
                
                structured_points.append(parts)
        
        # Edge midpoints for 2D variable space
        if self.n_variable == 2:
            var_names = self.variable_names
            bounds_0 = self.variable_bounds[var_names[0]]
            bounds_1 = self.variable_bounds[var_names[1]]
            
            # Edge midpoints
            edge_configs = [
                (bounds_0[0], (bounds_1[0] + bounds_1[1]) / 2),  # Left edge
                (bounds_0[1], (bounds_1[0] + bounds_1[1]) / 2),  # Right edge
                ((bounds_0[0] + bounds_0[1]) / 2, bounds_1[0]),  # Bottom edge
                ((bounds_0[0] + bounds_0[1]) / 2, bounds_1[1]),  # Top edge
            ]
            
            for var_0, var_1 in edge_configs:
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[j] = self.fixed_parts[name]
                
                # Set variable components
                parts[self.component_names.index(var_names[0])] = var_0
                parts[self.component_names.index(var_names[1])] = var_1
                
                structured_points.append(parts)
        
        # Overall centroid
        parts = np.zeros(self.n_components)
        for j, name in enumerate(self.component_names):
            if name in self.fixed_parts:
                parts[j] = self.fixed_parts[name]
            else:
                min_val, max_val = self.variable_bounds[name]
                parts[j] = (min_val + max_val) / 2
        structured_points.append(parts)
        
        return structured_points
    
    def _generate_lhs_candidates(self, n_samples: int) -> List[np.ndarray]:
        """Generate Latin Hypercube Sampling candidates for variable components."""
        if self.n_variable == 0:
            return []
        
        # Generate LHS in [0,1] space
        lhs_samples = self._latin_hypercube_sampling(n_samples, self.n_variable)
        
        candidates = []
        for sample in lhs_samples:
            parts = np.zeros(self.n_components)
            
            # Set fixed components
            for j, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts[j] = self.fixed_parts[name]
            
            # Map LHS samples to variable bounds
            var_idx = 0
            for j, name in enumerate(self.component_names):
                if name not in self.fixed_parts:
                    min_val, max_val = self.variable_bounds[name]
                    parts[j] = min_val + sample[var_idx] * (max_val - min_val)
                    var_idx += 1
            
            candidates.append(parts)
        
        return candidates
    
    def _latin_hypercube_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        """Generate Latin Hypercube Sampling points in [0,1]^n_dims."""
        samples = np.zeros((n_samples, n_dims))
        
        for i in range(n_dims):
            # Create stratified intervals
            intervals = np.arange(n_samples) / n_samples
            # Add random jitter within each interval
            jitter = np.random.uniform(0, 1/n_samples, n_samples)
            stratified_samples = intervals + jitter
            # Random permutation to break correlation between dimensions
            samples[:, i] = np.random.permutation(stratified_samples)
        
        return samples
    
    def _generate_proportional_candidates(self, n_samples: int) -> List[np.ndarray]:
        """Generate proportional candidates using constraint-aware sampling."""
        candidates = []
        
        for _ in range(n_samples):
            # Generate random proportions for variable components
            var_props = np.random.dirichlet(np.ones(self.n_variable))
            
            # Scale to fit within variable space
            min_total, max_total = self.variable_total_range
            total_var_parts = min_total + np.random.random() * (max_total - min_total)
            
            parts = np.zeros(self.n_components)
            
            # Set fixed components
            for j, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts[j] = self.fixed_parts[name]
            
            # Set variable components proportionally
            var_idx = 0
            for j, name in enumerate(self.component_names):
                if name not in self.fixed_parts:
                    min_val, max_val = self.variable_bounds[name]
                    # Use proportional allocation but respect bounds
                    proposed_val = var_props[var_idx] * total_var_parts
                    parts[j] = max(min_val, min(max_val, proposed_val))
                    var_idx += 1
            
            candidates.append(parts)
        
        return candidates
    
    def _generate_random_candidates(self, n_samples: int) -> List[np.ndarray]:
        """Generate random candidates using uniform sampling."""
        candidates = []
        
        for _ in range(n_samples):
            parts = np.zeros(self.n_components)
            
            # Set fixed components
            for j, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts[j] = self.fixed_parts[name]
                else:
                    min_val, max_val = self.variable_bounds[name]
                    parts[j] = np.random.uniform(min_val, max_val)
            
            candidates.append(parts)
        
        return candidates
    
    def _parts_to_proportions(self, parts: np.ndarray) -> np.ndarray:
        """Convert parts to proportions (normalized to sum to 1)."""
        total = np.sum(parts)
        if total <= 0:
            raise ValueError("Total parts must be positive")
        return parts / total
    
    def _select_initial_design(self, candidates: np.ndarray, n_runs: int, model_type: str) -> List[int]:
        """Select initial design using space-filling criteria."""
        n_candidates = len(candidates)
        
        if n_candidates <= n_runs:
            return list(range(n_candidates))
        
        # Use determinant maximization for initial selection
        best_determinant = -np.inf
        best_indices = None
        
        # Try multiple random starts
        for _ in range(min(50, n_candidates // n_runs)):
            indices = np.random.choice(n_candidates, n_runs, replace=False)
            design = candidates[indices]
            
            try:
                det = self._calculate_d_efficiency(design, model_type)
                if det > best_determinant:
                    best_determinant = det
                    best_indices = list(indices)
            except:
                continue
        
        if best_indices is None:
            # Fallback to random selection
            best_indices = list(np.random.choice(n_candidates, n_runs, replace=False))
        
        return best_indices
    
    def _calculate_d_efficiency(self, design: np.ndarray, model_type: str) -> float:
        """Calculate D-efficiency of the design."""
        try:
            # Build model matrix
            X = self._build_model_matrix(design, model_type)
            
            # Calculate information matrix
            XTX = X.T @ X
            
            # Calculate determinant
            det = np.linalg.det(XTX)
            
            # Calculate D-efficiency
            n_runs, n_params = X.shape
            d_efficiency = (det / n_runs) ** (1 / n_params) if det > 0 and n_params > 0 else 0.0
            
            return d_efficiency
        except:
            return -np.inf
    
    def _build_model_matrix(self, design: np.ndarray, model_type: str) -> np.ndarray:
        """Build model matrix for the given design and model type."""
        n_runs, n_components = design.shape
        
        if model_type == "linear":
            # For mixture designs, use only linear terms (no intercept due to sum constraint)
            return design
        
        elif model_type == "quadratic":
            # Mixture quadratic: linear terms + two-way interactions
            X = []
            
            for i in range(n_runs):
                row = []
                
                # Linear terms
                for j in range(n_components):
                    row.append(design[i, j])
                
                # Two-way interactions
                for j in range(n_components):
                    for k in range(j + 1, n_components):
                        row.append(design[i, j] * design[i, k])
                
                X.append(row)
            
            return np.array(X)
        
        elif model_type == "cubic":
            # Mixture cubic: linear + quadratic interactions + three-way interactions
            X = []
            
            for i in range(n_runs):
                row = []
                
                # Linear terms
                for j in range(n_components):
                    row.append(design[i, j])
                
                # Two-way interactions
                for j in range(n_components):
                    for k in range(j + 1, n_components):
                        row.append(design[i, j] * design[i, k])
                
                # Three-way interactions
                for j in range(n_components):
                    for k in range(j + 1, n_components):
                        for l in range(k + 1, n_components):
                            row.append(design[i, j] * design[i, k] * design[i, l])
                
                X.append(row)
            
            return np.array(X)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _verify_design_properties(self, parts_design: np.ndarray, prop_design: np.ndarray, batch_sizes: np.ndarray):
        """Verify that the design meets all constraints."""
        print(f"\nImproved Design Verification:")
        
        # Check proportions sum to 1
        prop_sums = prop_design.sum(axis=1)
        props_valid = np.allclose(prop_sums, 1.0, atol=1e-6)
        print(f"  Proportion sums correct: {props_valid} (range: {prop_sums.min():.6f} to {prop_sums.max():.6f})")
        
        # Check fixed components
        for name in self.fixed_parts:
            idx = self.component_names.index(name)
            expected = self.fixed_parts[name]
            actual_parts = parts_design[:, idx]
            is_constant = np.allclose(actual_parts, expected, atol=1e-6)
            print(f"  {name} parts constant: {is_constant} (expected: {expected}, actual: [{actual_parts.min():.6f}, {actual_parts.max():.6f}])")
        
        # Check variable component bounds
        for name in self.variable_names:
            idx = self.component_names.index(name)
            min_bound, max_bound = self.variable_bounds[name]
            actual_parts = parts_design[:, idx]
            within_bounds = np.all((actual_parts >= min_bound - 1e-6) & (actual_parts <= max_bound + 1e-6))
            print(f"  {name} bounds [{min_bound}, {max_bound}]: {within_bounds} (actual: [{actual_parts.min():.6f}, {actual_parts.max():.6f}])")
        
        # Check batch sizes
        expected_batches = parts_design.sum(axis=1)
        batch_match = np.allclose(batch_sizes, expected_batches, atol=1e-6)
        print(f"  Batch sizes: {batch_sizes.min():.1f} to {batch_sizes.max():.1f}")
        print(f"  Theoretical range: {self.total_fixed_parts + self.variable_total_range[0]:.1f} to {self.total_fixed_parts + self.variable_total_range[1]:.1f}")
    
    def _create_results_dataframe(self, parts_design: np.ndarray, prop_design: np.ndarray, batch_sizes: np.ndarray) -> pd.DataFrame:
        """Create comprehensive results DataFrame."""
        results = {}
        
        # Add run numbers
        results['Run'] = range(1, len(parts_design) + 1)
        
        # Add parts columns
        for i, name in enumerate(self.component_names):
            results[f'{name}_Parts'] = parts_design[:, i]
        
        # Add proportion columns
        for i, name in enumerate(self.component_names):
            results[f'{name}_Prop'] = prop_design[:, i]
        
        # Add batch sizes
        results['Batch_Size'] = batch_sizes
        
        return pd.DataFrame(results)
    
    def _display_space_filling_analysis(self, parts_design: np.ndarray):
        """Display space-filling analysis for the variable components."""
        if self.n_variable < 2:
            print(f"  Space-filling analysis: Not applicable for {self.n_variable} variable components")
            return
        
        # Extract variable components
        var_indices = [self.component_names.index(name) for name in self.variable_names[:2]]  # Use first 2 for analysis
        var_parts = parts_design[:, var_indices]
        
        # Calculate minimum distances
        min_distances = []
        for i in range(len(var_parts)):
            distances = []
            for j in range(len(var_parts)):
                if i != j:
                    dist = np.linalg.norm(var_parts[i] - var_parts[j])
                    distances.append(dist)
            if distances:
                min_distances.append(min(distances))
        
        if min_distances:
            min_min_dist = min(min_distances)
            avg_min_dist = np.mean(min_distances)
            clustered_points = sum(1 for d in min_distances if d < 1.0)
            
            print(f"  Space-filling metrics:")
            print(f"    Minimum distance between points: {min_min_dist:.3f}")
            print(f"    Average minimum distance: {avg_min_dist:.3f}")
            print(f"    Points with distance < 1.0: {clustered_points}/{len(min_distances)}")
            
            if clustered_points == 0:
                print(f"    ✅ No clustering detected!")
            else:
                print(f"    ⚠️ Some clustering detected")
    
    # Backward compatibility methods
    def get_parts_design(self) -> pd.DataFrame:
        """Get the parts design from the last generated design."""
        if self.last_parts_design is None:
            raise ValueError("No design has been generated yet. Call generate_design() first.")
        
        # Use "_Parts" suffix for consistency with generate_design() output
        parts_columns = [f"{name}_Parts" for name in self.component_names]
        parts_df = pd.DataFrame(self.last_parts_design, columns=parts_columns)
        parts_df.index = [f"Run_{i+1}" for i in range(len(parts_df))]
        return parts_df
    
    def get_proportions_design(self) -> pd.DataFrame:
        """Get the proportions design from the last generated design."""
        if self.last_design is None:
            raise ValueError("No design has been generated yet. Call generate_design() first.")
        
        props_df = pd.DataFrame(self.last_design, columns=self.component_names)
        props_df.index = [f"Run_{i+1}" for i in range(len(props_df))]
        return props_df
    
    def create_results_dataframe(self, parts_design: np.ndarray, prop_design: np.ndarray, batch_sizes: np.ndarray) -> pd.DataFrame:
        """Create results dataframe - public interface for backward compatibility."""
        return self._create_results_dataframe(parts_design, prop_design, batch_sizes)
    
    @property
    def parts_design(self) -> np.ndarray:
        """Get parts design as numpy array for backward compatibility."""
        return self.last_parts_design
    
    @property
    def prop_design(self) -> np.ndarray:
        """Get proportions design as numpy array for backward compatibility."""
        return self.last_design
    
    @property
    def batch_sizes(self) -> np.ndarray:
        """Get batch sizes as numpy array for backward compatibility."""
        return self.last_batch_sizes

    # Anti-clustering methods
    def _generate_anti_clustering_d_optimal(self, n_runs: int, model_type: str, max_iter: int, random_seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate anti-clustering D-optimal design."""
        print(f"\n🔧 Anti-Clustering D-Optimal Generation:")
        
        # Generate anti-clustering candidate set
        candidate_parts, candidate_props, candidate_batches = self._generate_anti_clustering_candidates(2000)
        
        # Select initial design
        initial_indices = self._select_initial_design(candidate_props, n_runs, model_type)
        
        # Run anti-clustering coordinate exchange
        final_indices = self._anti_clustering_coordinate_exchange(
            candidate_props, initial_indices, model_type, max_iter
        )
        
        return candidate_parts[final_indices], candidate_props[final_indices], candidate_batches[final_indices]
    
    def _generate_anti_clustering_i_optimal(self, n_runs: int, model_type: str, max_iter: int, random_seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate anti-clustering I-optimal design."""
        print(f"\n🔧 Anti-Clustering I-Optimal Generation:")
        
        # Generate anti-clustering candidate set
        candidate_parts, candidate_props, candidate_batches = self._generate_anti_clustering_candidates(2000)
        
        # Select initial design
        initial_indices = self._select_initial_design(candidate_props, n_runs, model_type)
        
        # Run anti-clustering coordinate exchange with I-optimal objective
        final_indices = self._anti_clustering_coordinate_exchange(
            candidate_props, initial_indices, model_type, max_iter, objective="i-optimal"
        )
        
        return candidate_parts[final_indices], candidate_props[final_indices], candidate_batches[final_indices]
    
    def _generate_space_filling_design(self, n_runs: int, random_seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate pure space-filling design."""
        print(f"\n🔧 Pure Space-Filling Generation:")
        
        # Generate candidates focused on space-filling
        candidate_parts, candidate_props, candidate_batches = self._generate_anti_clustering_candidates(3000)
        
        # Select points to maximize minimum distance
        selected_indices = self._select_space_filling_design(candidate_props, n_runs)
        
        return candidate_parts[selected_indices], candidate_props[selected_indices], candidate_batches[selected_indices]
    
    def _generate_anti_clustering_candidates(self, n_candidates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate anti-clustering candidate set."""
        print(f"  🎯 Generating anti-clustering candidates:")
        
        # Calculate candidate distribution
        n_structured = min(20, max(5, 2**self.n_variable))
        n_grid = int(0.35 * n_candidates)  # 35% grid
        n_lhs = int(0.25 * n_candidates)   # 25% LHS
        n_random = n_candidates - n_structured - n_grid - n_lhs
        
        all_parts, all_props, all_batches = [], [], []
        
        # 1. Enhanced structured points
        structured_parts = self._generate_enhanced_structured_points()[:n_structured]
        for parts in structured_parts:
            props = self._parts_to_proportions(parts)
            all_parts.append(parts)
            all_props.append(props)
            all_batches.append(np.sum(parts))
        
        # 2. Grid-based candidates for even distribution
        if n_grid > 0:
            grid_parts = self._generate_grid_candidates(n_grid)
            for parts in grid_parts:
                props = self._parts_to_proportions(parts)
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(np.sum(parts))
        
        # 3. LHS candidates
        if n_lhs > 0:
            lhs_parts = self._generate_lhs_candidates(n_lhs)
            for parts in lhs_parts:
                props = self._parts_to_proportions(parts)
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(np.sum(parts))
        
        # 4. Random candidates
        if n_random > 0:
            random_parts = self._generate_random_candidates(n_random)
            for parts in random_parts:
                props = self._parts_to_proportions(parts)
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(np.sum(parts))
        
        candidate_parts = np.array(all_parts)
        candidate_props = np.array(all_props)
        candidate_batches = np.array(all_batches)
        
        print(f"    Structured points: {n_structured}")
        print(f"    Grid candidates: {n_grid}")
        print(f"    LHS candidates: {n_lhs}")
        print(f"    Random candidates: {n_random}")
        print(f"    Total anti-clustering candidates: {len(candidate_parts)}")
        
        return candidate_parts, candidate_props, candidate_batches
    
    def _generate_grid_candidates(self, n_candidates: int) -> List[np.ndarray]:
        """Generate grid-based candidates for even space coverage."""
        if self.n_variable < 2:
            return self._generate_random_candidates(n_candidates)
        
        # Create grid in first two variable dimensions
        grid_size = int(np.ceil(np.sqrt(n_candidates)))
        candidates = []
        
        var_names = self.variable_names[:2]
        bounds_0 = self.variable_bounds[var_names[0]]
        bounds_1 = self.variable_bounds[var_names[1]]
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(candidates) >= n_candidates:
                    break
                
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                for k, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[k] = self.fixed_parts[name]
                
                # Set first two variable components on grid
                val_0 = bounds_0[0] + (i / (grid_size - 1)) * (bounds_0[1] - bounds_0[0])
                val_1 = bounds_1[0] + (j / (grid_size - 1)) * (bounds_1[1] - bounds_1[0])
                
                parts[self.component_names.index(var_names[0])] = val_0
                parts[self.component_names.index(var_names[1])] = val_1
                
                # Set remaining variable components randomly
                for name in self.variable_names[2:]:
                    min_val, max_val = self.variable_bounds[name]
                    parts[self.component_names.index(name)] = np.random.uniform(min_val, max_val)
                
                candidates.append(parts)
        
        return candidates[:n_candidates]
    
    def _anti_clustering_coordinate_exchange(self, candidates: np.ndarray, initial_indices: List[int], 
                                           model_type: str, max_iter: int, objective: str = "d-optimal") -> List[int]:
        """Run anti-clustering coordinate exchange optimization."""
        current_indices = initial_indices.copy()
        current_design = candidates[current_indices]
        
        # Calculate initial scores
        if objective == "d-optimal":
            current_d_eff = self._calculate_d_efficiency(current_design, model_type)
        else:
            current_d_eff = self._calculate_i_efficiency(current_design, candidates, model_type)
        
        current_space_score = self._calculate_space_filling_score(current_design)
        current_combined = (1 - self.space_filling_weight) * current_d_eff + self.space_filling_weight * current_space_score
        
        print(f"  Initial D-efficiency: {current_d_eff:.6e}")
        print(f"  Initial space-filling score: {current_space_score:.6f}")
        
        print(f"\n🔄 Running anti-clustering coordinate exchange:")
        
        for iteration in range(max_iter):
            improved = False
            
            for i in range(len(current_indices)):
                best_replacement = current_indices[i]
                best_combined_score = current_combined
                
                for candidate_idx in range(len(candidates)):
                    if candidate_idx in current_indices:
                        continue
                    
                    # Test replacement
                    test_indices = current_indices.copy()
                    test_indices[i] = candidate_idx
                    test_design = candidates[test_indices]
                    
                    # Check minimum distance constraint
                    if not self._meets_distance_constraint(test_design):
                        continue
                    
                    # Calculate combined score
                    try:
                        if objective == "d-optimal":
                            d_eff = self._calculate_d_efficiency(test_design, model_type)
                        else:
                            d_eff = self._calculate_i_efficiency(test_design, candidates, model_type)
                        
                        space_score = self._calculate_space_filling_score(test_design)
                        combined_score = (1 - self.space_filling_weight) * d_eff + self.space_filling_weight * space_score
                        
                        if combined_score > best_combined_score:
                            best_combined_score = combined_score
                            best_replacement = candidate_idx
                            improved = True
                    except:
                        continue
                
                current_indices[i] = best_replacement
                current_combined = best_combined_score
            
            if not improved:
                break
        
        print(f"  Converged after {iteration + 1} iterations")
        print(f"  Final combined score: {current_combined:.6f}")
        
        return current_indices
    
    def _meets_distance_constraint(self, design: np.ndarray) -> bool:
        """Check if design meets minimum distance constraints."""
        if self.n_variable < 2:
            return True
        
        # Extract variable components
        var_indices = [self.component_names.index(name) for name in self.variable_names[:2]]
        var_design = design[:, var_indices]
        
        # Check all pairwise distances
        for i in range(len(var_design)):
            for j in range(i + 1, len(var_design)):
                dist = np.linalg.norm(var_design[i] - var_design[j])
                if dist < self.min_distance_threshold:
                    return False
        
        return True
    
    def _calculate_space_filling_score(self, design: np.ndarray) -> float:
        """Calculate space-filling score based on minimum distances."""
        if self.n_variable < 2:
            return 1.0
        
        # Extract variable components  
        var_indices = [self.component_names.index(name) for name in self.variable_names[:2]]
        var_design = design[:, var_indices]
        
        # Calculate minimum distances
        distances = pdist(var_design)
        if len(distances) == 0:
            return 0.0
        
        min_distance = np.min(distances)
        avg_distance = np.mean(distances)
        
        # Normalize by space diagonal
        space_fill_score = min_distance / self.space_diagonal
        
        return min(1.0, space_fill_score)
    
    def _select_space_filling_design(self, candidates: np.ndarray, n_runs: int) -> List[int]:
        """Select design points to maximize space-filling."""
        if len(candidates) <= n_runs:
            return list(range(len(candidates)))
        
        # Start with random point
        selected_indices = [np.random.randint(len(candidates))]
        
        # Greedily add points that maximize minimum distance
        while len(selected_indices) < n_runs:
            best_candidate = -1
            best_min_distance = -1
            
            for candidate_idx in range(len(candidates)):
                if candidate_idx in selected_indices:
                    continue
                
                # Calculate minimum distance to selected points
                min_dist = float('inf')
                for selected_idx in selected_indices:
                    dist = np.linalg.norm(candidates[candidate_idx] - candidates[selected_idx])
                    min_dist = min(min_dist, dist)
                
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_candidate = candidate_idx
            
            if best_candidate >= 0:
                selected_indices.append(best_candidate)
            else:
                # Fallback to random selection
                remaining = [i for i in range(len(candidates)) if i not in selected_indices]
                selected_indices.append(np.random.choice(remaining))
        
        return selected_indices
    
    def _analyze_anti_clustering_performance(self, parts_design: np.ndarray):
        """Analyze anti-clustering performance."""
        if self.n_variable < 2:
            print(f"  🎯 Anti-clustering analysis: Not applicable for {self.n_variable} variable components")
            return
        
        # Extract variable components
        var_indices = [self.component_names.index(name) for name in self.variable_names[:2]]
        var_parts = parts_design[:, var_indices]
        
        # Calculate distance metrics
        distances = pdist(var_parts) if len(var_parts) > 1 else [0]
        
        if len(distances) > 0:
            min_distance = np.min(distances)
            avg_distance = np.mean(distances)
            max_distance = np.max(distances)
            
            # Count clustered pairs
            clustered_pairs = sum(1 for d in distances if d < 2.0)
            total_pairs = len(distances)
            
            # Calculate space utilization
            space_utilization = min_distance / self.space_diagonal * 100
            
            print(f"\n🎯 Anti-Clustering Performance Analysis:")
            print(f"  Minimum distance: {min_distance:.3f} (threshold: {self.min_distance_threshold:.3f})")
            print(f"  Average distance: {avg_distance:.3f}")
            print(f"  Maximum distance: {max_distance:.3f}")
            print(f"  Clustered pairs: {clustered_pairs}/{total_pairs} ({clustered_pairs/total_pairs:.1%})")
            
            if min_distance >= self.min_distance_threshold:
                print(f"  ✅ ANTI-CLUSTERING SUCCESS: All points meet distance requirements")
            else:
                print(f"  ⚠️ Minor clustering: {clustered_pairs/total_pairs:.1%} of point pairs clustered")
            
            print(f"  Space utilization: {space_utilization:.1f}%")

# Backward compatibility alias
FixedComponentsMixtureDesign = FixedPartsMixtureDesign
