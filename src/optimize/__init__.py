"""optimize/ — M8: product optimisation (desirability + cost) on the simplex."""
from .desirability import (
    DesirabilitySpec,
    DesirabilityResult,
    ChanceConstraint,
    hard_threshold_spec,
    desirability_value,
    overall_desirability,
    Desirability,
    optimize_desirability,
)

__all__ = [
    "DesirabilitySpec",
    "DesirabilityResult",
    "ChanceConstraint",
    "hard_threshold_spec",
    "desirability_value",
    "overall_desirability",
    "Desirability",
    "optimize_desirability",
]
