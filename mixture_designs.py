"""
Mixture Design of Experiments Implementation
Specialized for mixture experiments where components sum to 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Tuple, Dict, Optional
import itertools
import warnings
warnings.filterwarnings('ignore')

class MixtureDesign:
    """
    Class for generating optimal mixture experimental designs
    """
    
    def __init__(self, n_components: int, component_names: List[str] = None, 
                 component_bounds: List[Tuple[float, float]] = None):
        """
        Initialize mixture design generator
        
        Parameters:
        n_components: Number of mixture components
        component_names: Names of components (default: Comp1, Comp2, ...)
        component_bounds: Lower and upper bounds for each component (default: (0, 1))
        """
        self.n_components = n_components
        
        if component_names is None:
            self.component_names = [f'Comp_{i+1}' for i in range(n_components)]
        else:
            self.component_names = component_names
            
        if component_bounds is None:
            self.component_bounds = [(0.0, 1.0)] * n_components
        else:
            self.component_bounds = component_bounds
            
        # Validate bounds
        for i, (lower, upper) in enumerate(self.component_bounds):
            if lower < 0 or upper > 1 or lower >= upper:
                raise ValueError(f"Invalid bounds for component {i+1}: ({lower}, {upper})")
        
        # Check if bounds are feasible (sum of lower bounds <= 1, sum of upper bounds >= 1)
        sum_lower = sum(bound[0] for bound in self.component_bounds)
        sum_upper = sum(bound[1] for bound in self.component_bounds)
        
        if sum_lower > 1:
            raise ValueError("Sum of lower bounds exceeds 1 - infeasible mixture space")
        if sum_upper < 1:
            raise ValueError("Sum of upper bounds less than 1 - infeasible mixture space")
    
    def generate_simplex_lattice(self, degree: int = 2) -> np.ndarray:
        """
        Generate simplex lattice design
        
        Parameters:
        degree: Degree of lattice (2 for {0, 1/2, 1}, 3 for {0, 1/3, 2/3, 1}, etc.)
        """
        # Generate lattice points
        points = []
        levels = np.linspace(0, 1, degree + 1)
        
        # Generate all combinations that sum to 1
        for combination in itertools.combinations_with_replacement(range(degree + 1), self.n_components):
            if sum(combination) == degree:
                point = [x / degree for x in combination]
                # Check bounds
                if self._check_bounds(point):
                    points.append(point)
        
        # Add permutations for non-symmetric points
        unique_points = []
        for point in points:
            perms = list(set(itertools.permutations(point)))
            for perm in perms:
                if self._check_bounds(list(perm)) and list(perm) not in unique_points:
                    unique_points.append(list(perm))
        
        return np.array(unique_points)
    
    def generate_simplex_centroid(self) -> np.ndarray:
        """
        Generate simplex centroid design
        """
        points = []
        
        # Single component vertices (if bounds allow)
        for i in range(self.n_components):
            point = [0.0] * self.n_components
            point[i] = 1.0
            if self._check_bounds(point):
                points.append(point)
        
        # Binary blends (centroids of edges)
        for i, j in itertools.combinations(range(self.n_components), 2):
            point = [0.0] * self.n_components
            point[i] = 0.5
            point[j] = 0.5
            if self._check_bounds(point):
                points.append(point)
        
        # Ternary blends and higher
        for r in range(3, self.n_components + 1):
            for combo in itertools.combinations(range(self.n_components), r):
                point = [0.0] * self.n_components
                for idx in combo:
                    point[idx] = 1.0 / r
                if self._check_bounds(point):
                    points.append(point)
        
        # Overall centroid
        centroid = [1.0 / self.n_components] * self.n_components
        if self._check_bounds(centroid):
            points.append(centroid)
        
        return np.array(points)
    
    def generate_extreme_vertices(self) -> np.ndarray:
        """
        Generate extreme vertices design based on component bounds
        """
        vertices = []
        
        # Generate vertices by systematically setting components to bounds
        def generate_vertices_recursive(current_point, remaining_sum, component_idx):
            if component_idx == self.n_components - 1:
                # Last component gets the remaining sum
                current_point[component_idx] = remaining_sum
                lower, upper = self.component_bounds[component_idx]
                if lower <= remaining_sum <= upper:
                    vertices.append(current_point.copy())
                return
            
            lower, upper = self.component_bounds[component_idx]
            # Try both bounds for current component
            for bound_val in [lower, upper]:
                if bound_val <= remaining_sum:
                    current_point[component_idx] = bound_val
                    generate_vertices_recursive(current_point, remaining_sum - bound_val, component_idx + 1)
        
        initial_point = [0.0] * self.n_components
        generate_vertices_recursive(initial_point, 1.0, 0)
        
        # Remove duplicates
        unique_vertices = []
        for vertex in vertices:
            if not any(np.allclose(vertex, existing) for existing in unique_vertices):
                unique_vertices.append(vertex)
        
        return np.array(unique_vertices)
    
    def generate_d_optimal_mixture(self, n_runs: int, model_type: str = "quadratic", 
                                  max_iter: int = 1000, random_seed: int = None) -> np.ndarray:
        """
        Generate D-optimal mixture design using coordinate exchange
        
        Parameters:
        n_runs: Number of experimental runs
        model_type: "linear", "quadratic", or "cubic"
        max_iter: Maximum iterations for optimization
        random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate candidate points
        candidates = self._generate_candidate_points(n_runs * 20)
        
        # Initialize with random feasible design
        current_design = candidates[:n_runs].copy()
        best_d_eff = self._calculate_d_efficiency(current_design, model_type)
        
        for iteration in range(max_iter):
            improved = False
            
            for i in range(n_runs):
                best_point = current_design[i].copy()
                best_eff = best_d_eff
                
                # Try replacing point i with candidates
                for candidate in candidates[n_runs:n_runs+50]:
                    temp_design = current_design.copy()
                    temp_design[i] = candidate
                    
                    temp_eff = self._calculate_d_efficiency(temp_design, model_type)
                    
                    if temp_eff > best_eff:
                        best_eff = temp_eff
                        best_point = candidate.copy()
                        improved = True
                
                if improved:
                    current_design[i] = best_point
                    best_d_eff = best_eff
            
            if not improved:
                break
        
        return current_design
    
    def generate_i_optimal_mixture(self, n_runs: int, model_type: str = "quadratic",
                                  max_iter: int = 1000, random_seed: int = None) -> np.ndarray:
        """
        Generate I-optimal mixture design
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        candidates = self._generate_candidate_points(n_runs * 20)
        current_design = candidates[:n_runs].copy()
        
        # Generate integration points for I-efficiency calculation
        integration_points = self._generate_candidate_points(200)
        
        best_i_eff = self._calculate_i_efficiency(current_design, model_type, integration_points)
        
        for iteration in range(max_iter):
            improved = False
            
            for i in range(n_runs):
                best_point = current_design[i].copy()
                best_eff = best_i_eff
                
                for candidate in candidates[n_runs:n_runs+50]:
                    temp_design = current_design.copy()
                    temp_design[i] = candidate
                    
                    temp_eff = self._calculate_i_efficiency(temp_design, model_type, integration_points)
                    
                    if temp_eff < best_eff:  # Lower is better for I-optimal
                        best_eff = temp_eff
                        best_point = candidate.copy()
                        improved = True
                
                if improved:
                    current_design[i] = best_point
                    best_i_eff = best_eff
            
            if not improved:
                break
        
        return current_design
    
    def create_mixture_model_matrix(self, X: np.ndarray, model_type: str = "quadratic") -> np.ndarray:
        """
        Create model matrix for mixture models (Scheffe polynomials)
        
        Parameters:
        X: Design matrix (n_runs x n_components)
        model_type: "linear", "quadratic", or "cubic"
        """
        n_runs, n_components = X.shape
        
        if model_type == "linear":
            # Linear mixture model: β₁x₁ + β₂x₂ + ... + βₙxₙ (no intercept)
            return X
        
        elif model_type == "quadratic":
            # Quadratic mixture model: Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ
            terms = []
            
            # Linear terms
            for i in range(n_components):
                terms.append(X[:, i])
            
            # Interaction terms
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    terms.append(X[:, i] * X[:, j])
            
            return np.column_stack(terms)
        
        elif model_type == "cubic":
            # Cubic mixture model: includes linear, quadratic, and cubic terms
            terms = []
            
            # Linear terms
            for i in range(n_components):
                terms.append(X[:, i])
            
            # Quadratic interaction terms
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    terms.append(X[:, i] * X[:, j])
            
            # Cubic terms (three-way interactions)
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    for k in range(j + 1, n_components):
                        terms.append(X[:, i] * X[:, j] * X[:, k])
            
            return np.column_stack(terms)
        
        else:
            raise ValueError("model_type must be 'linear', 'quadratic', or 'cubic'")
    
    def evaluate_mixture_design(self, X: np.ndarray, model_type: str = "quadratic") -> Dict:
        """
        Evaluate mixture design properties
        """
        d_eff = self._calculate_d_efficiency(X, model_type)
        i_eff = self._calculate_i_efficiency(X, model_type)
        
        return {
            'n_runs': X.shape[0],
            'n_components': X.shape[1],
            'model_type': model_type,
            'd_efficiency': d_eff,
            'i_efficiency': i_eff,
            'design_matrix': X
        }
    
    def plot_ternary_design(self, X: np.ndarray, title: str = "Mixture Design"):
        """
        Plot ternary diagram for 3-component mixtures
        """
        if self.n_components != 3:
            print("Ternary plots only available for 3-component mixtures")
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create ternary plot triangle
            triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
            
            # Convert mixture coordinates to ternary plot coordinates
            plot_points = []
            for point in X:
                x1, x2, x3 = point
                # Convert to barycentric coordinates
                x = x2 + 0.5 * x3
                y = np.sqrt(3)/2 * x3
                plot_points.append([x, y])
            
            plot_points = np.array(plot_points)
            
            # Draw triangle
            triangle_patch = Polygon(triangle, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(triangle_patch)
            
            # Plot design points
            ax.scatter(plot_points[:, 0], plot_points[:, 1], c='red', s=100, alpha=0.7, 
                      edgecolors='black', zorder=5)
            
            # Add point labels
            for i, (x, y) in enumerate(plot_points):
                ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
            
            # Add component labels
            ax.text(-0.05, -0.05, self.component_names[0], fontsize=12, ha='center')
            ax.text(1.05, -0.05, self.component_names[1], fontsize=12, ha='center')
            ax.text(0.5, np.sqrt(3)/2 + 0.05, self.component_names[2], fontsize=12, ha='center')
            
            # Add grid lines
            for i in range(1, 10):
                val = i / 10
                # Lines parallel to side 1-2 (bottom)
                x_start = val * 0.5
                y_start = val * np.sqrt(3)/2
                x_end = val * 0.5 + (1 - val)
                y_end = val * np.sqrt(3)/2
                ax.plot([x_start, x_end], [y_start, y_end], 'lightgray', alpha=0.5)
                
                # Lines parallel to side 1-3 (left)
                x_start = (1 - val) * 0.5
                y_start = (1 - val) * np.sqrt(3)/2
                x_end = val + (1 - val) * 0.5
                y_end = (1 - val) * np.sqrt(3)/2
                ax.plot([x_start, x_end], [y_start, y_end], 'lightgray', alpha=0.5)
                
                # Lines parallel to side 2-3 (right)
                x_start = val
                y_start = 0
                x_end = (1 - val) * 0.5
                y_end = (1 - val) * np.sqrt(3)/2
                ax.plot([x_start, x_end], [y_start, y_end], 'lightgray', alpha=0.5)
            
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib required for plotting")
    
    def _check_bounds(self, point: List[float]) -> bool:
        """Check if point satisfies component bounds"""
        for i, val in enumerate(point):
            lower, upper = self.component_bounds[i]
            if val < lower - 1e-10 or val > upper + 1e-10:
                return False
        return True
    
    def _generate_candidate_points(self, n_points: int) -> np.ndarray:
        """Generate random feasible mixture points"""
        points = []
        max_attempts = n_points * 10
        attempts = 0
        
        while len(points) < n_points and attempts < max_attempts:
            attempts += 1
            
            # Generate random point using Dirichlet distribution
            alpha = np.ones(self.n_components)
            point = np.random.dirichlet(alpha)
            
            # Check bounds
            if self._check_bounds(point):
                points.append(point)
        
        # If we don't have enough points, use systematic generation
        while len(points) < n_points:
            # Generate point by adjusting a feasible point
            if len(points) > 0:
                base_point = points[np.random.randint(len(points))].copy()
                # Add small random perturbation
                perturbation = np.random.normal(0, 0.01, self.n_components)
                new_point = base_point + perturbation
                # Normalize to sum to 1
                new_point = new_point / np.sum(new_point)
                if self._check_bounds(new_point):
                    points.append(new_point)
            else:
                # Use centroid as fallback
                centroid = [1.0 / self.n_components] * self.n_components
                if self._check_bounds(centroid):
                    points.append(centroid)
        
        return np.array(points[:n_points])
    
    def _calculate_d_efficiency(self, X: np.ndarray, model_type: str) -> float:
        """Calculate D-efficiency for mixture design"""
        try:
            model_matrix = self.create_mixture_model_matrix(X, model_type)
            XtX = np.dot(model_matrix.T, model_matrix)
            
            if np.linalg.cond(XtX) > 1e12:
                return -1e6
            
            det_XtX = np.linalg.det(XtX)
            n_runs = X.shape[0]
            n_params = model_matrix.shape[1]
            
            d_eff = (det_XtX / n_runs) ** (1/n_params)
            return d_eff
        except:
            return -1e6
    
    def _calculate_i_efficiency(self, X: np.ndarray, model_type: str, 
                               integration_points: np.ndarray = None) -> float:
        """Calculate I-efficiency for mixture design"""
        try:
            model_matrix = self.create_mixture_model_matrix(X, model_type)
            XtX = np.dot(model_matrix.T, model_matrix)
            
            if np.linalg.cond(XtX) > 1e12:
                return 1e6
            
            XtX_inv = np.linalg.inv(XtX)
            
            if integration_points is None:
                integration_points = self._generate_candidate_points(200)
            
            total_pred_var = 0
            for point in integration_points:
                x_vec = self.create_mixture_model_matrix(point.reshape(1, -1), model_type)
                pred_var = np.dot(np.dot(x_vec, XtX_inv), x_vec.T)[0, 0]
                total_pred_var += pred_var
            
            avg_pred_var = total_pred_var / len(integration_points)
            return avg_pred_var
        except:
            return 1e6


# Example usage and utility functions
def mixture_response_analysis(designs: List[np.ndarray], 
                            response_functions: List[callable],
                            design_names: List[str],
                            component_names: List[str]) -> pd.DataFrame:
    """
    Analyze multiple response functions for different mixture designs
    """
    results = []
    
    for design, name in zip(designs, design_names):
        result = {'Design': name, 'n_runs': design.shape[0]}
        
        # Calculate responses for each function
        for i, response_func in enumerate(response_functions):
            responses = response_func(design)
            result[f'Response_{i+1}_mean'] = np.mean(responses)
            result[f'Response_{i+1}_std'] = np.std(responses)
            result[f'Response_{i+1}_range'] = np.ptp(responses)
        
        results.append(result)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== Mixture Design Demo ===\n")
    
    # Example: 3-component mixture (e.g., chemical formulation)
    component_names = ['Polymer_A', 'Polymer_B', 'Additive']
    component_bounds = [(0.2, 0.7), (0.2, 0.7), (0.1, 0.6)]  # More realistic bounds
    
    mixture = MixtureDesign(3, component_names, component_bounds)
    
    # Generate different types of mixture designs
    print("1. Simplex Lattice Design (degree 3):")
    lattice_design = mixture.generate_simplex_lattice(degree=3)
    print(f"Generated {len(lattice_design)} points")
    if len(lattice_design) > 0:
        print(pd.DataFrame(lattice_design, columns=component_names).round(3))
    else:
        print("No valid lattice points found with current bounds. Trying D-optimal instead...")
        lattice_design = mixture.generate_d_optimal_mixture(10, "quadratic", random_seed=42)
        print(pd.DataFrame(lattice_design, columns=component_names).round(3))
    
    print("\n2. Simplex Centroid Design:")
    centroid_design = mixture.generate_simplex_centroid()
    print(f"Generated {len(centroid_design)} points")
    print(pd.DataFrame(centroid_design, columns=component_names).round(3))
    
    print("\n3. D-optimal Mixture Design:")
    d_optimal_mixture = mixture.generate_d_optimal_mixture(n_runs=12, random_seed=42)
    d_results = mixture.evaluate_mixture_design(d_optimal_mixture, "quadratic")
    print(f"D-efficiency: {d_results['d_efficiency']:.4f}")
    print(f"I-efficiency: {d_results['i_efficiency']:.4f}")
    print(pd.DataFrame(d_optimal_mixture, columns=component_names).round(3))
    
    print("\n4. I-optimal Mixture Design:")
    i_optimal_mixture = mixture.generate_i_optimal_mixture(n_runs=12, random_seed=42)
    i_results = mixture.evaluate_mixture_design(i_optimal_mixture, "quadratic")
    print(f"D-efficiency: {i_results['d_efficiency']:.4f}")
    print(f"I-efficiency: {i_results['i_efficiency']:.4f}")
    
    # Example response functions for mixture
    def strength_response(X):
        """Material strength as function of mixture components"""
        return (50 * X[:, 0] + 60 * X[:, 1] + 30 * X[:, 2] + 
                20 * X[:, 0] * X[:, 1] - 10 * X[:, 0] * X[:, 2] + 
                np.random.normal(0, 2, len(X)))
    
    def flexibility_response(X):
        """Material flexibility as function of mixture components"""
        return (30 * X[:, 0] + 80 * X[:, 1] + 40 * X[:, 2] + 
                15 * X[:, 1] * X[:, 2] + 
                np.random.normal(0, 1.5, len(X)))
    
    print("\n5. Multiple Response Analysis:")
    designs = [d_optimal_mixture, i_optimal_mixture]
    design_names = ['D-optimal', 'I-optimal']
    response_functions = [strength_response, flexibility_response]
    
    np.random.seed(42)
    comparison = mixture_response_analysis(designs, response_functions, design_names, component_names)
    print(comparison.round(3))
    
    print("\n=== Mixture Design Demo Complete ===")
