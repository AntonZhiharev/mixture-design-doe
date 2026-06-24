"""
JMP-Style Sparsity-Focused Mixture Design Generator
===================================================

BREAKTHROUGH DISCOVERY from analyzing actual JMP designs:
The secret to estimating higher-order interactions is SPARSITY!

JMP uses 82% SPARSE points (2+ zeros) to:
- Reduce collinearity between interaction terms
- Provide clearer signals for each interaction
- Enable detection of 3-way and 4-way interactions

Key insight from MixtureDesigh45Runs1Order.xlsx analysis:
- 22% Vertices (1 active, 4 zeros)
- 33% Edges (2 active, 3 zeros)  
- 27% Ternary (3 active, 2 zeros)
- 11% Quaternary (4 active, 1 zero)
- Only 7% Interior (all 5 active)

This module implements this SPARSITY-FIRST strategy.
"""

import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations, product
import random


class JMPStyleSparseDesignGenerator:
    """
    Generate JMP-style sparse mixture designs optimized for higher-order interactions.
    
    The key innovation: SPARSITY reduces collinearity and enables reliable
    estimation of 3-way and 4-way interactions from limited data.
    """
    
    def __init__(self, n_components: int = 5, 
                 component_names: Optional[List[str]] = None,
                 random_seed: int = 42):
        """
        Initialize sparse design generator.
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components (default: 5)
        component_names : List[str], optional
            Names of components (default: x1, x2, ..., x5)
        random_seed : int
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.component_names = component_names or [f"x{i+1}" for i in range(n_components)]
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"JMP-Style Sparse Design Generator initialized:")
        print(f"  Components: {n_components}")
        print(f"  Strategy: SPARSITY-FIRST (82% sparse points)")
    
    def generate_sparse_design(self, n_runs: int, use_replication: bool = True) -> np.ndarray:
        """
        Generate sparse mixture design matching JMP's strategy.
        
        Point allocation based on actual JMP design analysis (MixtureDesigh45Runs1Order.xlsx):
        - ~22% Vertices (pure components) with 2x replication of key vertices
        - ~33% Edges (binary mixtures) with 2x replication of mid-edges
        - ~27% Ternary (3-component mixtures)
        - ~11% Quaternary (4-component mixtures)
        - ~7% Interior (~3 points) with 3x centroid replication
        
        JMP REPLICATION STRATEGY (13 replicates in 45-run design):
        - Centroid replicated 3 times (1 base + 2 extra)
        - Several vertex points replicated 2x
        - Mid-edge points (0.5/0.5) replicated 2x
        - Some ternary points replicated
        
        This replication is CRITICAL for:
        - Pure error estimation
        - Increased statistical power for higher-order detection
        - Better detection of 3-way and 4-way interactions
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        use_replication : bool
            Whether to include replicated points (default: True)
            
        Returns:
        --------
        np.ndarray
            Design matrix (n_runs × n_components)
        """
        print(f"\n{'='*80}")
        print(f"GENERATING JMP-STYLE SPARSE DESIGN: {n_runs} runs")
        if use_replication:
            print(f"Strategy: ENHANCED SPARSITY + STRATEGIC REPLICATION (JMP Match)")
        else:
            print(f"Strategy: NO REPLICATION (Unique points only)")
        print(f"{'='*80}")
        
        design_points = []
        
        # JMP-style replication strategy: ~29% of points are replicated (13 of 45)
        n_replicates = 0
        n_centroid_replicates = 0
        n_vertex_replicates = 0
        n_edge_replicates = 0
        n_ternary_replicates = 0
        
        if use_replication and n_runs >= 20:
            # Match JMP's ~29% replication rate
            replication_rate = 0.29
            n_replicates = max(5, int(n_runs * replication_rate))
            
            # Distribute replicates strategically (JMP style)
            # Centroid: 2 extra (3 total) - crucial for estimating pure error
            n_centroid_replicates = 2
            # Vertices: 2 replicates - for pure component estimates
            n_vertex_replicates = min(2, n_replicates - n_centroid_replicates)
            # Edges: 4 replicates (mid-edges 0.5/0.5) - key for 2-way interactions
            remaining = n_replicates - n_centroid_replicates - n_vertex_replicates
            n_edge_replicates = min(4, remaining)
            # Ternary: rest - key for 3-way interactions  
            n_ternary_replicates = n_replicates - n_centroid_replicates - n_vertex_replicates - n_edge_replicates
            
            print(f"\nStrategic Replication Plan ({n_replicates} total replicates):")
            print(f"  Centroid: +{n_centroid_replicates} (3 total for pure error)")
            print(f"  Vertices: +{n_vertex_replicates} (for main effects)")
            print(f"  Edges:    +{n_edge_replicates} (for 2-way interactions)")
            print(f"  Ternary:  +{n_ternary_replicates} (for 3-way interactions)")
            
        n_unique_budget = n_runs - n_replicates
        
        # JMP proportions matched more precisely (from 45-run analysis):
        # Vertices: 10 unique (22%), Edges: 15 (33%), Ternary: 12 (27%), 
        # Quaternary: 5 (11%), Interior: 3 (7%)
        # But we want MORE sparsity, so reduce interior further
        
        # JMP proportions matched precisely (from 45-run analysis):
        # JMP has: 10 vertices (22%), 15 edges (33%), 12 ternary (27%), 5 quaternary (11%), 3 interior (7%)
        # KEY INSIGHT: JMP has quaternary covering ALL 5 C(5,4)=5 combinations!
        
        n_vertices = max(self.n_components, int(n_unique_budget * 0.22))
        n_edges = int(n_unique_budget * 0.33)
        # CRITICAL: Ensure we have at least 5 quaternary points (one per 4-component combo)
        n_quaternary = max(5, int(n_unique_budget * 0.11))  # At least one per combination
        n_ternary = int(n_unique_budget * 0.27)
        # Minimal interior (just 1-2 centroids base, rest will be replicates)
        n_interior = max(1, n_unique_budget - n_vertices - n_edges - n_ternary - n_quaternary)
        
        # Ensure low interior: cap at 3, redistribute excess to ternary
        if n_interior > 3:
            excess = n_interior - 3
            n_interior = 3
            # Redistribute to ternary (most helpful for 3-way)
            n_ternary += excess
        
        print(f"\nTarget distribution (Unique points for {n_unique_budget} budget):")
        print(f"  Vertices (1 active, 4 zeros): {n_vertices}")
        print(f"  Edges (2 active, 3 zeros): {n_edges}")
        print(f"  Ternary (3 active, 2 zeros): {n_ternary}")
        print(f"  Quaternary (4 active, 1 zero): {n_quaternary}")
        print(f"  Interior (5 active, 0 zeros): {n_interior}")
        sparse_count = n_vertices + n_edges + n_ternary
        sparse_pct = sparse_count / n_unique_budget * 100
        print(f"  Target sparsity: {sparse_pct:.1f}% (2+ zeros)")
        
        # Phase 1: VERTICES (Pure components)
        # NOTE: Only n_components unique vertices exist, redistribute excess to edges
        actual_vertices = min(n_vertices, self.n_components)
        extra_for_edges = n_vertices - actual_vertices
        if extra_for_edges > 0:
            n_edges += extra_for_edges  # Redistribute to edges
        
        print(f"\n📍 Phase 1: Generating {actual_vertices} vertex points...")
        vertices = self._generate_vertices(actual_vertices)
        design_points.extend(vertices)
        
        # Phase 2: EDGES (Binary mixtures - CRITICAL for 2-way interactions)
        print(f"\n📍 Phase 2: Generating {n_edges} edge points...")
        edges = self._generate_edges(n_edges)
        design_points.extend(edges)
        
        # Phase 3: TERNARY (3-component - KEY for 3-way interactions!)
        print(f"\n📍 Phase 3: Generating {n_ternary} ternary points...")
        ternary = self._generate_ternary(n_ternary)
        design_points.extend(ternary)
        
        # Phase 4: QUATERNARY (4-component - for 4-way interactions)
        print(f"\n📍 Phase 4: Generating {n_quaternary} quaternary points...")
        quaternary = self._generate_quaternary(n_quaternary)
        design_points.extend(quaternary)
        
        # Phase 5: INTERIOR (Minimal - only for coverage)
        print(f"\n📍 Phase 5: Generating {n_interior} interior points...")
        interior = self._generate_interior(n_interior)
        design_points.extend(interior)
        
        # Phase 6: STRATEGIC REPLICATION (JMP-style)
        # KEY INSIGHT: JMP replicates DIFFERENT points, not the same one!
        # Each replicated point provides error estimate at THAT location.
        if n_replicates > 0:
            print(f"\n📍 Phase 6: Adding {n_replicates} strategic replications...")
            
            # Track which points we've already replicated to avoid duplicating
            replicated_keys = set()
            
            # 6a. Replicate Centroid (CRITICAL for pure error)
            centroid = np.ones(self.n_components) / self.n_components
            centroid_key = tuple(centroid.round(6))
            for _ in range(n_centroid_replicates):
                design_points.append(centroid.copy())
            replicated_keys.add(centroid_key)
            print(f"   ✓ Added {n_centroid_replicates} centroid replicates (pure error estimation)")
            
            # 6b. Replicate DIFFERENT Vertices (each vertex once, unique only)
            if n_vertex_replicates > 0:
                vertex_dict = {}  # Use dict to get unique vertices only
                for p in design_points:
                    if np.sum(p < 0.01) == (self.n_components - 1):
                        key = tuple(p.round(6))
                        if key not in replicated_keys and key not in vertex_dict:
                            vertex_dict[key] = p
                
                vertex_points = list(vertex_dict.items())  # [(key, point), ...]
                n_to_add = min(n_vertex_replicates, len(vertex_points))
                if n_to_add > 0:
                    selected = random.sample(vertex_points, n_to_add)
                    for key, p in selected:
                        design_points.append(p.copy())
                        replicated_keys.add(key)
                    print(f"   ✓ Added {n_to_add} DIFFERENT vertex replicates")
            
            # 6c. Replicate DIFFERENT Mid-Edges (crucial for 2-way detection)
            if n_edge_replicates > 0:
                edge_points = []
                for p in design_points:
                    if np.sum(p < 0.01) == (self.n_components - 2):
                        key = tuple(p.round(6))
                        if key not in replicated_keys:
                            # Prefer mid-edges (0.5/0.5)
                            is_mid_edge = int(np.sum(np.isclose(p, 0.5)) >= 1)
                            edge_points.append((p, key, is_mid_edge))
                
                # Sort to prioritize mid-edges (is_mid_edge=1 first)
                edge_points.sort(key=lambda x: -x[2])
                
                n_to_add = min(n_edge_replicates, len(edge_points))
                if n_to_add > 0:
                    for i in range(n_to_add):
                        p, key, _ = edge_points[i]
                        design_points.append(p.copy())
                        replicated_keys.add(key)
                    print(f"   ✓ Added {n_to_add} DIFFERENT edge replicates (2-way power)")
            
            # 6d. Replicate DIFFERENT Ternary points - KEY for 3-way detection
            if n_ternary_replicates > 0:
                ternary_points = []
                for p in design_points:
                    if np.sum(p < 0.01) == (self.n_components - 3):
                        key = tuple(p.round(6))
                        if key not in replicated_keys:
                            ternary_points.append((p, key))
                
                n_to_add = min(n_ternary_replicates, len(ternary_points))
                if n_to_add > 0:
                    selected = random.sample(ternary_points, n_to_add)
                    for p, key in selected:
                        design_points.append(p.copy())
                        replicated_keys.add(key)
                    print(f"   ✓ Added {n_to_add} DIFFERENT ternary replicates (3-way power)")
        
        # Convert to array
        design = np.array(design_points[:n_runs])
        
        # Verify sparsity
        sparsity_check = self._analyze_sparsity(design)
        
        print(f"\n{'='*80}")
        print(f"DESIGN GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"  Total points: {len(design)}")
        print(f"  Unique points: {len(np.unique(design.round(6), axis=0))}")
        print(f"  Replicated points: {n_replicates}")
        print(f"  Sparsity: {sparsity_check['sparse_percentage']:.1f}% (Target: 82%+)")
        print(f"  Ready for higher-order interaction estimation!")
        
        return design
    
    def _generate_vertices(self, n_target: int) -> List[np.ndarray]:
        """Generate pure component vertices - always exactly n_components unique vertices"""
        vertices = []
        
        # Add each pure component (exactly once)
        for i in range(self.n_components):
            point = np.zeros(self.n_components)
            point[i] = 1.0
            vertices.append(point)
        
        # NOTE: We only generate n_components unique vertices
        # Any additional "vertex budget" should be redistributed to edges
        # This prevents duplicate vertices in the base design
        return vertices[:min(n_target, self.n_components)]
    
    def _generate_edges(self, n_target: int) -> List[np.ndarray]:
        """
        Generate binary mixture points (edges).
        
        CRITICAL: Include ALL mid-edges (0.5/0.5) FIRST!
        JMP includes all C(5,2)=10 mid-edges for complete 2-way coverage.
        
        Strategy:
        1. First add ALL 10 mid-edges (0.5/0.5 for each pair)
        2. Then fill remaining slots with other ratios (0.3/0.7, 0.25/0.75)
        """
        edges = []
        all_pairs = list(combinations(range(self.n_components), 2))
        
        # PHASE 1: Add ALL mid-edges (0.5/0.5) FIRST - this is critical!
        # JMP has all 10 mid-edges for complete 2-way interaction coverage
        for i, j in all_pairs:
            point = np.zeros(self.n_components)
            point[i] = 0.5
            point[j] = 0.5
            edges.append(point)
        
        print(f"      Added {len(edges)} mid-edges (0.5/0.5) for ALL pairs")
        
        if len(edges) >= n_target:
            return edges[:n_target]
        
        # PHASE 2: Add other ratios to fill remaining slots
        other_ratios = [0.3, 0.7, 0.25, 0.75]
        additional_edges = []
        
        for i, j in all_pairs:
            for ratio in other_ratios:
                point = np.zeros(self.n_components)
                point[i] = ratio
                point[j] = 1.0 - ratio
                additional_edges.append(point)
        
        # Shuffle additional edges and add what we need
        random.shuffle(additional_edges)
        remaining_needed = n_target - len(edges)
        edges.extend(additional_edges[:remaining_needed])
        
        return edges[:n_target]
    
    def _generate_ternary(self, n_target: int) -> List[np.ndarray]:
        """
        Generate ternary mixture points (3 active components).
        
        CRITICAL for 3-way interactions!
        
        Strategy: Multiple ratio patterns
        - Balanced: [0.33, 0.33, 0.34]
        - One dominant: [0.50, 0.25, 0.25]
        - Two similar: [0.35, 0.35, 0.30]
        - Varied: [0.20, 0.30, 0.50]
        """
        ternary = []
        
        # Ratio patterns mimicking JMP
        ratio_patterns = [
            [0.333, 0.333, 0.334],  # Balanced (good for symmetric 3-way)
            [0.500, 0.250, 0.250],  # One dominant
            [0.350, 0.350, 0.300],  # Two similar
            [0.200, 0.300, 0.500],  # Varied
            [0.400, 0.300, 0.300],  # Slight dominant
            [0.250, 0.375, 0.375],  # Two dominant
        ]
        
        # Generate for all 3-component combinations
        for combo in combinations(range(self.n_components), 3):
            for pattern in ratio_patterns:
                if len(ternary) >= n_target:
                    break
                
                point = np.zeros(self.n_components)
                for idx, comp_idx in enumerate(combo):
                    point[comp_idx] = pattern[idx]
                
                # Normalize
                point = point / np.sum(point)
                ternary.append(point)
            
            if len(ternary) >= n_target:
                break
        
        # Shuffle for diversity
        random.shuffle(ternary)
        
        return ternary[:n_target]
    
    def _generate_quaternary(self, n_target: int) -> List[np.ndarray]:
        """
        Generate quaternary mixture points (4 active components).
        
        KEY for 4-way interactions!
        
        Strategy: Leave one component at zero, distribute among other 4
        - Equal: [0.25, 0.25, 0.25, 0.25, 0] - Maximum quartic leverage
        - Slight variation: [0.23, 0.25, 0.27, 0.25, 0]
        - JMP-style: [0.20, 0.25, 0.28, 0.27, 0]
        """
        quaternary = []
        
        # Ratio patterns for 4 components
        ratio_patterns_4 = [
            [0.250, 0.250, 0.250, 0.250],  # Equal (optimal quartic)
            [0.230, 0.250, 0.270, 0.250],  # Slight asymmetry
            [0.200, 0.250, 0.280, 0.270],  # JMP-style variation
            [0.210, 0.240, 0.260, 0.290],  # Gradual increase
            [0.220, 0.260, 0.260, 0.260],  # One smaller
        ]
        
        # For each component that will be zero
        for zero_idx in range(self.n_components):
            for pattern in ratio_patterns_4:
                if len(quaternary) >= n_target:
                    break
                
                point = np.zeros(self.n_components)
                non_zero_indices = [i for i in range(self.n_components) if i != zero_idx]
                
                for idx, comp_idx in enumerate(non_zero_indices):
                    point[comp_idx] = pattern[idx]
                
                # Normalize
                point = point / np.sum(point)
                quaternary.append(point)
            
            if len(quaternary) >= n_target:
                break
        
        # Shuffle
        random.shuffle(quaternary)
        
        return quaternary[:n_target]
    
    def _generate_interior(self, n_target: int) -> List[np.ndarray]:
        """
        Generate interior points (all components active).
        
        Keep minimal - only for overall coverage and centroid.
        """
        interior = []
        
        # Always include overall centroid
        centroid = np.ones(self.n_components) / self.n_components
        interior.append(centroid)
        
        # Add a few near-centroid points with variation
        for _ in range(n_target - 1):
            if len(interior) >= n_target:
                break
            
            # Generate random point near centroid
            point = np.random.dirichlet(np.ones(self.n_components) * 2.0)  # Alpha=2 concentrates near center
            interior.append(point)
        
        return interior[:n_target]
    
    def _analyze_sparsity(self, design: np.ndarray) -> dict:
        """Analyze sparsity of the design"""
        n_points = len(design)
        
        zeros_per_point = []
        for point in design:
            n_zeros = np.sum(point < 0.01)
            zeros_per_point.append(n_zeros)
        
        # Count by sparsity level
        sparse_points = sum(1 for z in zeros_per_point if z >= 2)  # 2+ zeros
        
        return {
            'total_points': n_points,
            'sparse_points': sparse_points,
            'sparse_percentage': (sparse_points / n_points * 100) if n_points > 0 else 0,
            'zeros_distribution': {
                '0_zeros': sum(1 for z in zeros_per_point if z == 0),
                '1_zero': sum(1 for z in zeros_per_point if z == 1),
                '2_zeros': sum(1 for z in zeros_per_point if z == 2),
                '3_zeros': sum(1 for z in zeros_per_point if z == 3),
                '4_zeros': sum(1 for z in zeros_per_point if z == 4),
            }
        }


# Convenience function
def generate_jmp_sparse_design(n_runs: int, n_components: int = 5, 
                               component_names: Optional[List[str]] = None,
                               use_replication: bool = True) -> np.ndarray:
    """
    Generate JMP-style sparse mixture design.
    
    Parameters:
    -----------
    n_runs : int
        Number of experimental runs
    n_components : int
        Number of components (default: 5)
    component_names : List[str], optional
        Component names
    use_replication : bool
        Whether to include replicated points (default: True)
        
    Returns:
    --------
    np.ndarray
        Sparse design matrix optimized for higher-order interactions
    """
    generator = JMPStyleSparseDesignGenerator(n_components, component_names)
    return generator.generate_sparse_design(n_runs, use_replication=use_replication)


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("JMP-STYLE SPARSE DESIGN GENERATOR")
    print("="*80)
    print("\nBased on breakthrough discovery: SPARSITY is the key!")
    print("Actual JMP designs use 82% sparse points to enable higher-order detection.")
    
    # Generate 45-run design (matching JMP example)
    design = generate_jmp_sparse_design(n_runs=45, n_components=5)
    
    print(f"\n{'='*80}")
    print("DESIGN PREVIEW")
    print(f"{'='*80}")
    print("\nFirst 10 points:")
    for i, point in enumerate(design[:10]):
        n_zeros = np.sum(point < 0.01)
        print(f"  Run {i+1:2d}: {[f'{x:.3f}' for x in point]} ({n_zeros} zeros)")
    
    print(f"\nDesign shape: {design.shape}")
    print(f"All points sum to 1.0: {np.allclose(design.sum(axis=1), 1.0)}")
    
    print(f"\n{'='*80}")
    print("Ready to use for higher-order interaction estimation!")
    print(f"{'='*80}\n")
