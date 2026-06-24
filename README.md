# 🔬 Mixture & Factorial Design of Experiments (DOE)

A Python toolkit for creating, analyzing, and fitting **mixture** and **factorial** DOE models,
with interactive Streamlit web-apps and a full statistical suite — ANOVA, R², Lenth's half-normal,
Pareto chart, LOF test, Shapiro-Wilk, residual diagnostics, fold-over, augmentation, and more.

---

## 🚀 Quick Start — Launch an App

> **All commands must be run from the project root directory (`d:\DOE`).**

### ▶ Option A — **Efficient Sequential Workflow** ⭐ NEW, RECOMMENDED

Adaptive 3-phase strategy using **Smart Simplex Centroid** point generation.
Stops adding experiments as soon as the model meets your R² target.
**Saves 16–53% of runs** vs fixed JMP-style designs.

```bash
# Interactive menu (choose from all available apps)
python run_sequential_interface.py

# Launch directly — efficient workflow on port 8501
python run_sequential_interface.py --efficient

# Launch on a custom port
python run_sequential_interface.py --efficient --port 8502

# Run standalone benchmark (no Streamlit needed)
.venv\Scripts\python.exe run_efficient_workflow.py
```

Then open your browser at **http://localhost:8501**

---

### ▶ Option B — Classic Sequential DOE Workflow

D-optimal screening + fold-over/augmentation, for factorial and mixture designs.

```bash
python run_sequential_interface.py --new
```

---

### ▶ Option C — Main DOE Design App (Design generation + Data Analysis)

```bash
streamlit run src/apps/streamlit_app.py
```

---

### ▶ All Available Apps

| App | Launch command | Port |
|-----|---------------|------|
| ⭐ **Efficient Sequential** (recommended) | `python run_sequential_interface.py --efficient` | 8501 |
| 🔬 **Classic Sequential** (D-opt + fold-over) | `python run_sequential_interface.py --new` | 8502 |
| 🎯 **Main Design App** (design + ANOVA) | `streamlit run src/apps/streamlit_app.py` | 8501 |
| 🔄 **Sequential Reconstruction** (legacy) | `python run_sequential_interface.py --old` | 8503 |
| 🧪 **Staged Parameter Recovery** | `streamlit run src/apps/staged_parameter_recovery_app.py` | 8503 |

---

## 📦 Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt
```

### Requirements (key packages)

```
streamlit>=1.28.0
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0
scikit-learn>=1.1.0
plotly>=5.17.0
matplotlib>=3.5.0
openpyxl>=3.1.0        # Excel export
reportlab>=4.0.0       # PDF reports
kaleido>=0.2.1         # Plotly image export
Pillow>=9.0.0
python-docx>=1.1.0     # DOCX reports
```

---

## ⭐ Efficient Sequential Workflow — Algorithm Guide

### Why It's More Efficient

Traditional approaches generate a **fixed-size design upfront** (e.g., 45 runs for JMP Smart Simplex
Centroid for 5 components) regardless of whether the model is already good enough.
The new workflow **stops as soon as R² reaches your target**, saving 16–53% of experiments.

### 3-Phase Adaptive Strategy (q = 5 components)

| Phase | New Points Added | Cumulative Runs | Model Fitted | Typical Stop? |
|-------|-----------------|-----------------|--------------|---------------|
| **Phase 1** | Vertices×2 + all binary blends×1 + centroid×1 | **21** | Linear + 2-way Scheffé | If R² ≥ target |
| **Phase 2** | Ternary blends (guided by Phase 1 significance) + 2 LOF centroids | **33** | + 3-way Scheffé | If R² ≥ target |
| **Phase 3** | Quaternary blends (guided by Phase 2) + binary/ternary replicates | **43** | Full 5-way Scheffé | Always |

Point generation uses **Smart Simplex Centroid** (from `smart_simplex_centroid.py`) with
**JMP-validated per-order replication counts**:

| Order k | Replicates (q=5) | Rationale |
|---------|------------------|-----------|
| 1 (pure vertices) | 2 | Pure-error estimation |
| 2–3 (middle) | 1 | One observation per blend type |
| 4 (near-centroid) | 1 | Near-redundant with centroid |
| 5 (overall centroid) | 3 | LOF detection + highest-order interaction |

### Benchmark Results (5 components, 16-term true model)

True model:
```
5A + 4B + 3C + 2D + 0.5E
+ 5AB + 4AC + 3BC + 2CD + 0.5DE
+ 5ABC + 4ACD + 3BCD
+ 5ABCD + 4BCDE
+ 5ABCDE
```

| Algorithm | Runs | CV-R² | Terms Detected | FP | 4-way | 5-way |
|-----------|------|-------|---------------|-----|-------|-------|
| Old quadratic (2-way) | 27 | 0.9963 | 10/16 | 0 | ❌ | ❌ |
| Old cubic (3-way) | 42 | 1.0000 | 13/16 | 0 | ❌ | ❌ |
| Old quartic (4-way) | 49 | 1.0000 | 15/16 | 0 | ✅ | ❌ |
| Old quintic (5-way) | **51** | **1.0000** | **16/16** | 0 | ✅ | ✅ |
| **New (Phase 1 only, R²≥0.97)** | **21** | **0.9951** | **10/16** | 0 | — | — |
| **New (all 3 phases)** | **43** | **1.0000** | **16/16** | 0 | ✅ | ✅ |

**New algorithm saves 8 runs (16%) vs old quintic** while detecting exactly the same 16/16 terms.

### Minimum Runs to Detect Each Interaction Order

| Interaction Order | Terms | Minimum Runs Needed | Phase |
|---|---|---|---|
| 2-way | AB, AC, BC, CD, DE | **Run 20** | Phase 1 |
| 3-way | ABC, ACD, BCD | **Run 30** | Phase 2 |
| 4-way | ABCD, BCDE | **Run 37** | Phase 3 |
| **5-way** | **ABCDE** | **Run 37** | **Phase 3** |

> **Note**: 4-way and 5-way terms both become detectable at run 37 — once quaternary blends
> are added and the 31-parameter model matrix becomes estimable, the overall centroid (present
> since Phase 1) immediately provides 5-way ABCDE information.

### Benchmarking Scripts

```bash
# Full standalone benchmark (no Streamlit required):
.venv\Scripts\python.exe run_efficient_workflow.py

# Old vs New algorithm comparison (all model orders):
.venv\Scripts\python.exe run_algorithm_comparison.py

# High-order interaction detection study (4-way, 5-way):
.venv\Scripts\python.exe run_high_order_detection.py
```

---

## 🗂️ Project Structure

```
DOE/
├── run_sequential_interface.py          ← CLI launcher (menu or --efficient/--new/--old)
├── run_efficient_workflow.py            ← Standalone benchmark for new algorithm
├── run_algorithm_comparison.py          ← Old vs New algorithm comparison
├── run_high_order_detection.py          ← High-order interaction detection study
│
├── src/
│   ├── apps/
│   │   ├── efficient_sequential_workflow_app.py ← NEW Efficient 3-phase app ⭐
│   │   ├── streamlit_app.py                     ← Main DOE design + ANOVA app
│   │   ├── doe_sequential_workflow_app.py        ← Classic D-opt + fold-over workflow
│   │   ├── sequential_reconstruction_app.py      ← Legacy reconstruction app
│   │   └── staged_parameter_recovery_app.py      ← Staged parameter recovery
│   │
│   ├── algorithms/
│   │   ├── smart_simplex_centroid.py            ← Smart Simplex Centroid ⭐
│   │   │                                           (JMP-validated replication + augmentation)
│   │   ├── d_optimal_algorithm.py               ← D-optimal exchange algorithm
│   │   ├── candidate_generation.py              ← LHS & simplex candidate pools
│   │   ├── jmp_style_mixture_design.py          ← JMP-style mixture design
│   │   ├── jmp_style_screening.py               ← JMP-style screening
│   │   ├── jmp_full_model_screening.py          ← Full model screening
│   │   ├── hierarchical_d_optimal.py            ← Hierarchical D-optimal
│   │   ├── hierarchical_screening.py            ← Hierarchical screening
│   │   ├── adaptive_sequential_doe.py           ← Adaptive sequential DOE
│   │   ├── sequential_adaptive_mixture_design.py
│   │   ├── sequential_regression_reconstruction.py
│   │   └── mixture_algorithms.py
│   │
│   ├── core/
│   │   └── optimal_design_generator.py          ← Core design generator
│   │
│   └── utils/
│       ├── detailed_anova.py                    ← ANOVA calculations
│       ├── math_utils.py                        ← LHS, simplex normalization
│       ├── mixture_utils.py                     ← Mixture design utilities
│       ├── d_efficiency_calculator.py           ← D-efficiency metrics
│       ├── response_analysis.py                 ← Response surface analysis
│       ├── sequential_doe.py                    ← Sequential DOE utilities
│       ├── sequential_mixture_doe.py            ← Sequential mixture utilities
│       ├── pdf_report_generator.py              ← PDF report export
│       └── docx_report_generator.py             ← DOCX report export
│
├── docs/                                        ← Analysis notes & findings
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
└── examples/
    ├── basic_usage.py
    └── parts_mode_demonstration.py
```

---

## 🔬 Efficient Sequential Workflow — Stage Guide (10 Stages)

`efficient_sequential_workflow_app.py` guides you through **10 interactive stages**:

| Stage | What happens |
|-------|-------------|
| **1 — Setup** | Choose number of components, names, response variable, R² target (default 0.97) |
| **2 — Phase 1 Design** | Generates vertices×2 + all binary blends×1 + centroid×1; download Excel/CSV |
| **3 — Phase 1 Data Entry** | Enter responses manually or upload file |
| **4 — Phase 1 Analysis** | Fits linear+2-way Scheffé; ANOVA, Pareto, residuals; **stops if R² ≥ target** |
| **5 — Phase 2 Design** | Ternary blends (guided by Phase 1 significant pairs) + 2 LOF centroids |
| **6 — Phase 2 Data Entry** | Enter Phase 2 responses |
| **7 — Phase 2 Analysis** | Fits +3-way Scheffé; **stops if R² ≥ target** |
| **8 — Phase 3 Design** | Quaternary blends (guided by Phase 2 significant triples) + replicates |
| **9 — Phase 3 Data Entry** | Enter Phase 3 responses |
| **10 — Final Model** | Full Scheffé up to 5-way; coefficients, ANOVA, residuals, equation, Excel export |

---

## 🔬 Classic Sequential DOE Workflow — Stage Guide (6 Stages)

`doe_sequential_workflow_app.py` guides you through **6 interactive stages**:

| Stage | What happens |
|-------|-------------|
| **1 — Setup** | Choose experiment type (factorial or mixture), enter factor/component names & ranges, select model order (linear / quadratic / cubic). Mixture designs support **Parts mode** (see below) |
| **2 — Initial Design** | Generates D-optimal screening design; annotates point types; download as **Excel or CSV** |
| **3 — Data Entry** | Enter responses in editable table **or** upload a completed Excel/CSV file |
| **4 — Screening Analysis** | ANOVA · Half-Normal plot (Lenth ME/SME) · Pareto chart · Aliasing matrix · **Interactive term selection** with live R² preview |
| **5 — Fold-Over / Augment** | **Factorial**: full or partial fold-over · **Mixture**: greedy D-optimal augmentation |
| **6 — Final Model** | Reduced model · R²/Adj-R²/RMSE · F-test · LOF test · Shapiro-Wilk · Residuals · Response surface · Model equation · **Full Excel export** |

---

### 🧮 Parts Mode for Mixture Components

Toggle **"Parts mode"** in Stage 1 to work with absolute amounts (grams, mL, phr) rather than
fractions. The proportion of each component is defined as:

$$x_i = \frac{\text{amount}_i}{\sum_j \text{amount}_j}$$

Achievable proportion bounds are auto-derived:

$$x_{i,\min} = \frac{a_i}{a_i + \sum_{j \neq i} b_j} \qquad
x_{i,\max} = \frac{b_i}{b_i + \sum_{j \neq i} a_j}$$

where $a_i$ = min amount, $b_i$ = max amount for component $i$.

---

### ♻️ Constrained Mixture Designs (Proportion Mode)

For components with lower bounds (e.g., 0.1 ≤ x₁ ≤ 0.6), the
**pseudocomponent transformation** is applied automatically:

$$w_i = \frac{x_i - L_i}{1 - \sum_j L_j} \qquad \Longleftrightarrow \qquad x_i = L_i + w_i \cdot \left(1 - \sum_j L_j\right)$$

---

### Mixture Design — Scheffé Model Reference

For a **q-component Scheffé model** (no pure quadratic terms because Σxᵢ = 1):

| Order | Model form | # terms (q=5) | Min runs needed |
|-------|-----------|--------------|-----------------|
| Linear (1-way) | $\hat{y} = \sum_i \beta_i x_i$ | 5 | 7 |
| Quadratic (2-way) | + $\sum_{i<j} \beta_{ij} x_i x_j$ | 15 | 17 |
| Cubic (3-way) | + $\sum_{i<j<k} \beta_{ijk} x_i x_j x_k$ | 25 | 27 |
| Quartic (4-way) | + $\sum_{i<j<k<l} \beta_{ijkl} x_i x_j x_k x_l$ | 30 | 32 |
| Quintic (5-way) | + $\beta_{12345} x_1 x_2 x_3 x_4 x_5$ | 31 | 33 |

> **Why no xᵢ² terms?** The simplex constraint Σxᵢ = 1 makes pure quadratic terms linearly
> dependent — they cannot be estimated separately from the linear terms.

---

## 📊 Features

### Design Generation
- **Smart Simplex Centroid** with JMP-validated per-order replication (extracted from
  `smart_simplex_centroid.py`) — structured centroid blends: vertices, binary, ternary,
  quaternary, overall centroid
- **Adaptive 3-phase staging** — stops early when R² target reached, saving 16–53% runs
- **D-optimal** factorial designs (greedy max-min exchange, LHS candidate pool)
- **Simplex lattice / centroid** base for mixture designs
- **Constrained mixture** design via pseudocomponent transformation
- **Parts mode** — enter component ranges in real units; proportion bounds auto-derived
- **D-optimal augmentation** for both factorial (fold-over) and mixture designs
- **Full fold-over** (all factors negated) and **partial fold-over** (selected factors negated)

### Statistical Analysis
- ANOVA table (Regression / Residual / Total) with F-statistic and p-value
- **Lenth's Half-Normal plot** with ME / SME significance thresholds
- **Pareto chart** of |t-statistics| with α=0.05 reference line
- **Aliasing / correlation matrix** with severity labels
- Lack-of-Fit (LOF) test using replicated runs
- **Shapiro-Wilk** residuals normality test
- 95% confidence intervals on all coefficients
- Interactive checkbox-based term selection with **live Adj-R² preview**
- **True positive / false positive / false negative** term recovery metrics

### Visualization
- 3D response surface (factorial designs, first two factors)
- Ternary contour plot (3-component mixture designs)
- 2×2 residual diagnostic panel
- Ternary scatter plot of initial design points
- Pareto and half-normal effect significance plots

### Export
- **Excel workbook** with sheets: Coefficients, ANOVA, Predictions, Budget
- Per-phase design downloads (Phase 1, 2, 3 separately)
- Parts-mode design in actual units
- Individual CSV/Excel downloads
- DOCX and PDF report generation

---

## 🧪 Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific suites
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/performance/ -v

# Functional test for sequential workflow helpers
python tests/_test_workflow_functions.py
```

---

## 📖 Documentation

See the `docs/` folder for detailed notes:

| File | Topic |
|------|-------|
| `docs/getting_started.md` | Basic usage guide |
| `docs/pseudo_statistics_explanation.md` | Why Scheffé models have no xᵢ² terms |
| `docs/post_selection_inference.md` | p-value inflation after factor selection |
| `docs/jmp_comparison_findings.md` | Comparison with JMP software output |
| `docs/point_generation_comparison.md` | Comparison of point generation methods |
| `docs/anova_statistical_issues_analysis.md` | ANOVA statistical considerations |
| `docs/false_positive_analysis.md` | False-positive diagnosis and prevention |
| `docs/hierarchical_function_findings.md` | Hierarchical function recovery findings |
| `docs/hierarchical_screening_approach.md` | Hierarchical screening methodology |
| `docs/high_order_interaction_limitation.md` | Limits of high-order interaction estimation |
| `docs/sequential_regression_reconstruction.md` | Sequential regression reconstruction |
| `docs/flexible_runs_summary.md` | Flexible run count strategies |
| `docs/jmp_advanced_screening_methods.md` | Advanced JMP-style screening methods |
| `docs/improvement_notes.md` | Development notes and improvement log |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see `LICENSE` for details.
