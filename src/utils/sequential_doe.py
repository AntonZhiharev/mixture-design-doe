"""
Sequential/Augmented Design of Experiments
Allows adding experiments to an existing design while maintaining optimality
"""

import numpy as np
import pandas as pd
from optimal_doe_python import OptimalDOE
from typing import List, Tuple, Optional, Union
import json
from datetime import datetime

class SequentialDOE(OptimalDOE):
    """
    Extended DOE class for sequential experimentation
    """
    
    def __init__(self, n_factors: int, factor_ranges: List[Tuple[float, float]] = None):
        """Initialize sequential DOE generator"""
        super().__init__(n_factors, factor_ranges)
        self.experiment_history = []
        
    def augment_design(self, existing_design: np.ndarray, 
                      n_additional_runs: int, 
                      model_order: int = 2,
                      criterion: str = "D-optimal",
                      avoided_region: Optional[np.ndarray] = None,
                      focus_region: Optional[dict] = None,
                      random_seed: int = None) -> np.ndarray:
        """
        Augment an existing design with additional experiments
        
        Parameters:
        -----------
        existing_design : np.ndarray
            Current design matrix (n_existing_runs x n_factors)
        n_additional_runs : int
            Number of new experiments to add
        model_order : int
            Model order (1=linear, 2=quadratic)
        criterion : str
            "D-optimal" or "I-optimal"
        avoided_region : np.ndarray, optional
            Regions to avoid (e.g., where previous experiments failed)
        focus_region : dict, optional
            Region to focus on based on screening results
            Example: {'factor_indices': [0, 2], 'ranges': [(0.5, 1.0), (-0.5, 0.5)]}
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            New experiments only (n_additional_runs x n_factors)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Generate candidate points
        n_candidates = max(1000, n_additional_runs * 50)
        candidates = self.generate_candidate_points(n_candidates)
        
        # Apply focus region if specified
        if focus_region:
            candidates = self._apply_focus_region(candidates, focus_region)
            
        # Remove points too close to existing design
        candidates = self._filter_candidates(candidates, existing_design)
        
        # Remove avoided regions
        if avoided_region is not None:
            candidates = self._remove_avoided_regions(candidates, avoided_region)
        
        # Initialize augmented design with existing points
        augmented_design = existing_design.copy()
        
        # Select additional points using coordinate exchange
        new_points = []
        
        for _ in range(n_additional_runs):
            best_candidate = None
            best_efficiency = -np.inf if criterion == "D-optimal" else np.inf
            
            # Test each candidate
            for candidate in candidates[:200]:  # Test subset for efficiency
                # Create temporary augmented design
                temp_design = np.vstack([augmented_design, 
                                       np.array(new_points + [candidate])])
                
                # Calculate efficiency
                if criterion == "D-optimal":
                    efficiency = self.d_efficiency(temp_design, model_order)
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_candidate = candidate
                else:  # I-optimal
                    efficiency = self.i_efficiency(temp_design, model_order)
                    if efficiency < best_efficiency:
                        best_efficiency = efficiency
                        best_candidate = candidate
            
            if best_candidate is not None:
                new_points.append(best_candidate)
                # Remove selected point and nearby points from candidates
                distances = np.sum((candidates - best_candidate)**2, axis=1)
                candidates = candidates[distances > 0.01]
            else:
                print(f"Warning: Could only add {len(new_points)} points")
                break
                
        return np.array(new_points)
    
    def _apply_focus_region(self, candidates: np.ndarray, focus_region: dict) -> np.ndarray:
        """Apply focus region constraints to candidates"""
        if 'factor_indices' in focus_region and 'ranges' in focus_region:
            mask = np.ones(len(candidates), dtype=bool)
            for idx, (min_val, max_val) in zip(focus_region['factor_indices'], 
                                              focus_region['ranges']):
                mask &= (candidates[:, idx] >= min_val) & (candidates[:, idx] <= max_val)
            
            focused_candidates = candidates[mask]
            
            # If too few candidates remain, mix with original
            if len(focused_candidates) < 100:
                n_original = min(100 - len(focused_candidates), len(candidates))
                mixed = np.vstack([focused_candidates, 
                                 candidates[np.random.choice(len(candidates), 
                                                           n_original, replace=False)]])
                return mixed
            
            return focused_candidates
        
        return candidates
    
    def _filter_candidates(self, candidates: np.ndarray, 
                          existing_design: np.ndarray, 
                          min_distance: float = 0.05) -> np.ndarray:
        """Remove candidates too close to existing design points"""
        filtered = []
        
        for candidate in candidates:
            # Calculate distances to all existing points
            distances = np.sqrt(np.sum((existing_design - candidate)**2, axis=1))
            
            # Keep if far enough from all existing points
            if np.min(distances) > min_distance:
                filtered.append(candidate)
                
        return np.array(filtered) if filtered else candidates
    
    def _remove_avoided_regions(self, candidates: np.ndarray, 
                               avoided_regions: np.ndarray,
                               buffer: float = 0.1) -> np.ndarray:
        """Remove candidates near avoided regions"""
        filtered = []
        
        for candidate in candidates:
            distances = np.sqrt(np.sum((avoided_regions - candidate)**2, axis=1))
            if np.min(distances) > buffer:
                filtered.append(candidate)
                
        return np.array(filtered) if filtered else candidates
    
    def save_experiment_stage(self, design: np.ndarray, stage_name: str, 
                            metadata: dict = None) -> str:
        """
        Save experiment stage to history
        
        Parameters:
        -----------
        design : np.ndarray
            Design matrix for this stage
        stage_name : str
            Name of the stage (e.g., "Screening", "Augmentation 1")
        metadata : dict
            Additional information about the stage
            
        Returns:
        --------
        str
            Filename where the stage was saved
        """
        stage_info = {
            'stage_name': stage_name,
            'timestamp': datetime.now().isoformat(),
            'design': design.tolist(),
            'n_runs': len(design),
            'n_factors': design.shape[1],
            'metadata': metadata or {}
        }
        
        self.experiment_history.append(stage_info)
        
        # Save to file
        filename = f"sequential_doe_{stage_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(stage_info, f, indent=2)
            
        return filename
    
    def analyze_sequential_efficiency(self, stages: List[np.ndarray], 
                                    model_order: int = 2) -> pd.DataFrame:
        """
        Analyze how efficiency improves with each sequential stage
        
        Parameters:
        -----------
        stages : List[np.ndarray]
            List of design matrices for each stage
        model_order : int
            Model order for efficiency calculation
            
        Returns:
        --------
        pd.DataFrame
            Efficiency metrics for each stage
        """
        results = []
        cumulative_design = np.empty((0, self.n_factors))
        
        for i, stage_design in enumerate(stages):
            cumulative_design = np.vstack([cumulative_design, stage_design])
            
            d_eff = self.d_efficiency(cumulative_design, model_order)
            i_eff = self.i_efficiency(cumulative_design, model_order)
            
            results.append({
                'Stage': i + 1,
                'New_Runs': len(stage_design),
                'Total_Runs': len(cumulative_design),
                'D_Efficiency': d_eff,
                'I_Efficiency': i_eff,
                'Parameters': self._count_parameters(model_order)
            })
            
        return pd.DataFrame(results)
    
    def _count_parameters(self, model_order: int) -> int:
        """Count number of parameters in the model"""
        if model_order == 1:
            return 1 + self.n_factors  # intercept + main effects
        elif model_order == 2:
            # intercept + main + squared + interactions
            return 1 + self.n_factors + self.n_factors + \
                   (self.n_factors * (self.n_factors - 1)) // 2
        return 0
    
    def get_recommended_runs(self, model_order: int = 2, 
                           quality_level: str = "recommended") -> dict:
        """
        Get recommended number of runs based on rules of thumb
        
        Parameters:
        -----------
        model_order : int
            1 for linear, 2 for quadratic
        quality_level : str
            "minimum", "recommended", or "excellent"
            
        Returns:
        --------
        dict
            Dictionary with run recommendations
        """
        n_params = self._count_parameters(model_order)
        
        # Calculate based on quality level
        if quality_level == "minimum":
            multiplier = 1.0
        elif quality_level == "recommended":
            multiplier = 1.5
        elif quality_level == "excellent":
            multiplier = 2.0
        else:
            multiplier = 1.5  # default to recommended
        
        n_runs = int(np.ceil(n_params * multiplier))
        
        return {
            'model_order': model_order,
            'model_type': 'Linear' if model_order == 1 else 'Quadratic',
            'n_parameters': n_params,
            'quality_level': quality_level,
            'multiplier': multiplier,
            'minimum_runs': n_params,
            'recommended_runs': n_runs,
            'efficiency_expected': {
                'minimum': '70-80%',
                'recommended': '85-95%', 
                'excellent': '95-99%'
            }.get(quality_level, '85-95%')
        }
    
    def get_sequential_recommendations(self) -> dict:
        """
        Get recommendations for sequential experimentation
        
        Returns:
        --------
        dict
            Recommendations for Stage 1 and Stage 2
        """
        # Stage 1: Linear model recommendations
        stage1_min = self._count_parameters(1)
        stage1_rec = int(np.ceil(stage1_min * 1.5))
        stage1_exc = stage1_min * 2
        
        # Stage 2: Additional runs for quadratic
        quad_params = self._count_parameters(2)
        total_rec = int(np.ceil(quad_params * 1.5))
        stage2_rec = max(total_rec - stage1_rec, 5)  # At least 5 additional
        
        return {
            'stage1': {
                'purpose': 'Screening (Linear Model)',
                'minimum': stage1_min,
                'recommended': stage1_rec,
                'excellent': stage1_exc,
                'can_fit': f'Linear model with {stage1_min} parameters'
            },
            'stage2': {
                'purpose': 'Augmentation (Quadratic Model)',
                'recommended_additional': stage2_rec,
                'total_runs': stage1_rec + stage2_rec,
                'total_parameters': quad_params,
                'efficiency_gain': 'Focuses on important factors identified in Stage 1'
            },
            'benefits': [
                f'Total {stage1_rec + stage2_rec} runs vs {quad_params} minimum',
                'Can stop after Stage 1 if linear model sufficient',
                'Stage 2 design informed by Stage 1 results',
                'Better than {stage1_rec + stage2_rec} random experiments'
            ]
        }
    
    def recommend_next_stage(self, current_design: np.ndarray, 
                           responses: np.ndarray = None,
                           target_efficiency: float = 0.9,
                           model_order: int = 2) -> dict:
        """
        Recommend next experimental stage based on current results
        
        Parameters:
        -----------
        current_design : np.ndarray
            Current design matrix
        responses : np.ndarray, optional
            Response values from current experiments
        target_efficiency : float
            Target D-efficiency to achieve
        model_order : int
            Model order
            
        Returns:
        --------
        dict
            Recommendations for next stage
        """
        current_d_eff = self.d_efficiency(current_design, model_order)
        n_params = self._count_parameters(model_order)
        
        recommendations = {
            'current_efficiency': current_d_eff,
            'target_efficiency': target_efficiency,
            'efficiency_gap': target_efficiency - current_d_eff
        }
        
        # Estimate runs needed
        if current_d_eff < target_efficiency:
            # Rough estimate: each additional run improves efficiency
            estimated_runs = int(np.ceil(
                (target_efficiency - current_d_eff) * len(current_design) / current_d_eff
            ))
            recommendations['estimated_additional_runs'] = max(5, estimated_runs)
        else:
            recommendations['estimated_additional_runs'] = 0
            
        # Analyze responses if provided
        if responses is not None:
            # Simple analysis: identify high-variance regions
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(current_design, responses)
            
            # Feature importance
            importances = rf.feature_importances_
            important_factors = np.argsort(importances)[-3:]  # Top 3
            
            recommendations['important_factors'] = important_factors.tolist()
            recommendations['focus_suggestion'] = (
                f"Consider focusing on factors {important_factors.tolist()} "
                "which show highest impact on responses"
            )
            
        return recommendations


# Example usage functions
def sequential_experiment_example():
    """
    Example: Sequential experimentation workflow
    """
    print("=== Sequential DOE Example ===\n")
    
    # Initialize sequential DOE
    n_factors = 6
    factor_names = ['Temp', 'Pressure', 'pH', 'Time', 'Conc_A', 'Conc_B']
    factor_ranges = [(80, 120), (1, 5), (6, 8), (30, 90), (0.1, 1.0), (0.5, 2.0)]
    
    seq_doe = SequentialDOE(n_factors, factor_ranges)
    
    # Stage 1: Screening (Linear model)
    print("Stage 1: Screening Design")
    print("-" * 40)
    initial_design = seq_doe.generate_d_optimal(
        n_runs=15, model_order=1, random_seed=42
    )
    
    initial_results = seq_doe.evaluate_design(initial_design, model_order=1)
    print(f"Initial design: {initial_results['n_runs']} runs")
    print(f"D-efficiency: {initial_results['d_efficiency']:.4f}")
    
    # Save stage 1
    seq_doe.save_experiment_stage(
        initial_design, 
        "Screening",
        metadata={'objective': 'Identify important factors', 'model': 'linear'}
    )
    
    # Simulate responses (in practice, these come from actual experiments)
    np.random.seed(42)
    true_coeffs = [80, 5, -3, 2, 8, -4, 1]  # intercept + main effects
    X_model = seq_doe.create_model_matrix(initial_design, model_order=1)
    responses = X_model @ true_coeffs + np.random.normal(0, 2, len(initial_design))
    
    # Get recommendations
    recommendations = seq_doe.recommend_next_stage(
        initial_design, responses, target_efficiency=0.9
    )
    print(f"\nRecommendations after Stage 1:")
    print(f"Current efficiency: {recommendations['current_efficiency']:.4f}")
    print(f"Suggested additional runs: {recommendations['estimated_additional_runs']}")
    
    # Stage 2: Augmentation for quadratic model
    print("\n\nStage 2: Augmentation for Optimization")
    print("-" * 40)
    
    # Focus on important factors (from screening)
    focus_region = {
        'factor_indices': [0, 4],  # Temperature and Conc_A
        'ranges': [(90, 110), (0.5, 0.8)]  # Narrowed ranges
    }
    
    additional_design = seq_doe.augment_design(
        initial_design,
        n_additional_runs=15,
        model_order=2,  # Now fitting quadratic
        criterion="D-optimal",
        focus_region=focus_region,
        random_seed=43
    )
    
    print(f"Added {len(additional_design)} experiments")
    
    # Combine designs
    combined_design = np.vstack([initial_design, additional_design])
    combined_results = seq_doe.evaluate_design(combined_design, model_order=2)
    
    print(f"Combined design: {combined_results['n_runs']} runs")
    print(f"D-efficiency (quadratic): {combined_results['d_efficiency']:.4f}")
    
    # Save stage 2
    seq_doe.save_experiment_stage(
        additional_design,
        "Augmentation",
        metadata={
            'objective': 'Fit quadratic model for optimization',
            'focus': 'Temperature and Conc_A', 
            'model': 'quadratic'
        }
    )
    
    # Analyze sequential efficiency
    print("\n\nSequential Efficiency Analysis")
    print("-" * 40)
    efficiency_df = seq_doe.analyze_sequential_efficiency(
        [initial_design, additional_design], 
        model_order=2
    )
    print(efficiency_df.to_string(index=False))
    
    # Create summary DataFrame
    summary_data = []
    for i, (name, design) in enumerate([
        ("Initial (15 runs)", initial_design),
        ("Combined (30 runs)", combined_design)
    ]):
        # Evaluate for both linear and quadratic
        linear_eval = seq_doe.evaluate_design(design, model_order=1)
        quad_eval = seq_doe.evaluate_design(design, model_order=2)
        
        summary_data.append({
            'Design': name,
            'Runs': len(design),
            'D-eff (Linear)': f"{linear_eval['d_efficiency']:.4f}",
            'D-eff (Quadratic)': f"{quad_eval['d_efficiency']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n\nDesign Comparison Summary")
    print("-" * 40)
    print(summary_df.to_string(index=False))
    
    return seq_doe, initial_design, additional_design, combined_design


# Practical workflow function
def create_sequential_plan(factor_names: List[str], 
                         factor_ranges: List[Tuple[float, float]],
                         stage1_runs: int = 15,
                         stage2_runs: int = 15,
                         output_file: str = "sequential_doe_plan.xlsx") -> str:
    """
    Create a complete sequential experimentation plan
    
    Parameters:
    -----------
    factor_names : List[str]
        Names of factors
    factor_ranges : List[Tuple[float, float]]
        Min/max ranges for each factor
    stage1_runs : int
        Number of runs for screening
    stage2_runs : int
        Number of runs for augmentation
    output_file : str
        Excel file to save the plan
        
    Returns:
    --------
    str
        Path to saved file
    """
    n_factors = len(factor_names)
    seq_doe = SequentialDOE(n_factors, factor_ranges)
    
    # Generate stage 1
    stage1_design = seq_doe.generate_d_optimal(
        n_runs=stage1_runs, 
        model_order=1,  # Linear for screening
        random_seed=42
    )
    
    # Generate stage 2 (augmentation)
    stage2_design = seq_doe.augment_design(
        stage1_design,
        n_additional_runs=stage2_runs,
        model_order=2,  # Quadratic for optimization
        random_seed=43
    )
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Stage 1 sheet
        stage1_df = pd.DataFrame(stage1_design, columns=factor_names)
        stage1_df.insert(0, 'Run_Order', range(1, len(stage1_df) + 1))
        stage1_df.insert(1, 'Stage', 'Screening')
        
        # Add response columns
        for i in range(3):  # 3 example responses
            stage1_df[f'Response_{i+1}'] = np.nan
            
        stage1_df.to_excel(writer, sheet_name='Stage_1_Screening', index=False)
        
        # Stage 2 sheet  
        stage2_df = pd.DataFrame(stage2_design, columns=factor_names)
        stage2_df.insert(0, 'Run_Order', range(stage1_runs + 1, 
                                              stage1_runs + len(stage2_df) + 1))
        stage2_df.insert(1, 'Stage', 'Augmentation')
        
        # Add response columns
        for i in range(3):
            stage2_df[f'Response_{i+1}'] = np.nan
            
        stage2_df.to_excel(writer, sheet_name='Stage_2_Augmentation', index=False)
        
        # Combined sheet
        combined_df = pd.concat([stage1_df, stage2_df], ignore_index=True)
        combined_df.to_excel(writer, sheet_name='All_Experiments', index=False)
        
        # Efficiency summary
        efficiency_data = []
        
        # Stage 1 efficiency
        stage1_eval_linear = seq_doe.evaluate_design(stage1_design, model_order=1)
        stage1_eval_quad = seq_doe.evaluate_design(stage1_design, model_order=2)
        
        efficiency_data.append({
            'Stage': 'Screening Only',
            'Total Runs': stage1_runs,
            'Linear D-Efficiency': stage1_eval_linear['d_efficiency'],
            'Quadratic D-Efficiency': stage1_eval_quad['d_efficiency'],
            'Can Fit': 'Linear model'
        })
        
        # Combined efficiency
        combined_design = np.vstack([stage1_design, stage2_design])
        combined_eval_linear = seq_doe.evaluate_design(combined_design, model_order=1)
        combined_eval_quad = seq_doe.evaluate_design(combined_design, model_order=2)
        
        efficiency_data.append({
            'Stage': 'Screening + Augmentation',
            'Total Runs': stage1_runs + stage2_runs,
            'Linear D-Efficiency': combined_eval_linear['d_efficiency'],
            'Quadratic D-Efficiency': combined_eval_quad['d_efficiency'],
            'Can Fit': 'Quadratic model with all interactions'
        })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        efficiency_df.to_excel(writer, sheet_name='Efficiency_Analysis', index=False)
        
        # Instructions sheet
        instructions = pd.DataFrame({
            'Instructions': [
                '1. Start with Stage 1 (Screening) experiments',
                '2. Run all 15 experiments and record responses',
                '3. Analyze results to identify important factors',
                '4. Use results to potentially adjust Stage 2 experiments',
                '5. Run Stage 2 (Augmentation) experiments',
                '6. Combine all data for final analysis',
                '',
                'Benefits of Sequential Approach:',
                '- Stage 1 identifies important factors',
                '- Stage 2 can focus on important region',
                '- Total of 30 experiments vs 28 minimum for full quadratic',
                '- Can stop after Stage 1 if linear model is sufficient',
                '- Stage 2 design is optimized given Stage 1 results'
            ]
        })
        instructions.to_excel(writer, sheet_name='Instructions', index=False)
    
    print(f"Sequential DOE plan saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    # Run example
    seq_doe, stage1, stage2, combined = sequential_experiment_example()
    
    # Create practical plan
    print("\n\n=== Creating Practical Sequential Plan ===")
    factor_names = ['Temperature', 'Pressure', 'pH', 'Time', 'Catalyst', 'Flow_Rate']
    factor_ranges = [(80, 120), (1, 5), (6, 8), (30, 90), (0.1, 1.0), (10, 50)]
    
    output_file = create_sequential_plan(
        factor_names, 
        factor_ranges,
        stage1_runs=15,
        stage2_runs=15
    )
    
    print("\nSequential experimentation allows you to:")
    print("1. Start with screening (15 experiments)")
    print("2. Identify important factors")
    print("3. Add targeted experiments (15 more)")
    print("4. Build better models with same total effort")
