"""
Mixture Design of Experiments Package (legacy root namespace).

NOTE: The original top-level convenience imports referenced modules that no
longer exist at the repo root (they now live under ``src/``). Importing this
package must not crash test collection or the new ``src`` pipeline, so the
legacy imports are guarded and exposed only when available.
"""

__all__ = []

try:  # legacy convenience exports — optional, never fatal
    from mixture_designs import (  # type: ignore
        MixtureDesignGenerator, MixtureDesign, EnhancedMixtureDesign,
    )
    from mixture_base import MixtureBase  # type: ignore
    from mixture_designs import MixtureDesign as OptimizedMixtureDesign  # type: ignore
    from fixed_parts_mixture_designs import FixedPartsMixtureDesign  # type: ignore

    __all__ = [
        "MixtureDesignGenerator",
        "MixtureBase",
        "OptimizedMixtureDesign",
        "FixedPartsMixtureDesign",
        "MixtureDesign",
        "EnhancedMixtureDesign",
    ]
except Exception:  # pragma: no cover - legacy modules absent in new layout
    pass
