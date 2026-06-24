"""Построение golden-фикстур (REBUILD_SPEC §6), БОЕВОЙ вариант (q=5).

Эталон считается **штатным R** (base R: lm/qr/det/solve + явные формулы
Деррингера–Сюича и GP-постериора) через `r/compute_reference.R`. Обмен Python↔R —
через CSV. Если R не найден, используется независимый Python-эталон
(`reference.py`) — числа совпадают (тот же мат-контракт).

Запуск:
    python -m src.verification.generate_golden            # авто: R, иначе python
    python -m src.verification.generate_golden --engine python
    DOE_RSCRIPT="d:\\Program Files\\R\\R-4.6.0\\bin\\Rscript.exe" python -m ...

Масштаб: Scheffé quadratic q=5 → p=15 параметров; GP Matérn5/2 с 5 ARD-длинами.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from . import reference as ref
from .golden_io import save_fixture

_HERE = os.path.dirname(os.path.abspath(__file__))
_R_SCRIPT = os.path.join(_HERE, "r", "compute_reference.R")

Q = 5
ORDER = 2
P = Q + Q * (Q - 1) // 2          # 15
NAMES = [chr(ord("A") + i) for i in range(Q)]
TERM_NAMES = NAMES + [f"{NAMES[i]}*{NAMES[j]}"
                      for i in range(Q) for j in range(i + 1, Q)]


# ----------------------------------------------------------------------
# Поиск Rscript
# ----------------------------------------------------------------------
def find_rscript() -> str | None:
    env = os.environ.get("DOE_RSCRIPT")
    if env and os.path.exists(env):
        return env
    onpath = shutil.which("Rscript")
    if onpath:
        return onpath
    for root in (r"C:\Program Files\R", r"D:\Program Files\R",
                 r"d:\Program Files\R", r"C:\Program Files (x86)\R"):
        hits = sorted(glob.glob(os.path.join(root, "*", "bin", "Rscript.exe")))
        if hits:
            return hits[-1]
    return None


# ----------------------------------------------------------------------
# Боевые входы (детерминированные)
# ----------------------------------------------------------------------
def _design_q5() -> np.ndarray:
    rng = np.random.default_rng(2024)
    verts = np.eye(Q)
    edges = []
    for i in range(Q):
        for j in range(i + 1, Q):
            v = np.zeros(Q); v[i] = v[j] = 0.5
            edges.append(v)
    edges = np.array(edges)
    centroid = np.full((1, Q), 1.0 / Q)
    interior = rng.dirichlet(np.ones(Q), size=14)
    return np.vstack([verts, edges, centroid, interior])     # 5+10+1+14 = 30


def _scheffe_inputs():
    X = _design_q5()
    M = ref.scheffe_design_matrix(X, ORDER)
    beta_true = np.array([10, 12, 8, 9, 11,                   # linear (5)
                          4, -3, 2, 1, -2, 3, -1, 2, -4, 1],  # pairwise (10)
                         dtype=float)
    rng = np.random.default_rng(20240601)
    y = M @ beta_true + rng.normal(0.0, 0.2, size=len(X))
    return X, y


def _moments_q5(n=40000, seed=11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.dirichlet(np.ones(Q), size=n)
    G = ref.scheffe_design_matrix(pts, ORDER)
    return (G.T @ G) / n


def _desirability_inputs():
    y_max = np.array([0., 20., 35., 50., 65., 80., 100.])
    y_min = np.array([1., 1.8, 2.5, 3.0, 3.5, 4.2, 5.0])
    y_tgt = np.array([40., 48., 55., 60., 66., 73., 80.])
    return y_max, y_min, y_tgt


def _gp_inputs():
    rng = np.random.default_rng(5)
    X_train = rng.random((8, Q))
    y_train = np.array([1.0, 0.5, -0.5, 0.2, -1.0, 0.8, -0.3, 0.1])
    X_test = rng.random((4, Q))
    const, ls, noise = 1.5, [0.4, 0.7, 0.5, 0.9, 0.6], 0.05
    return X_train, y_train, X_test, const, ls, noise


# ----------------------------------------------------------------------
# R-движок: запись CSV, запуск Rscript, чтение выходов
# ----------------------------------------------------------------------
def _run_r_engine(rscript: str) -> dict:
    io = tempfile.mkdtemp(prefix="doe_golden_")
    sav = lambda name, a: np.savetxt(os.path.join(io, name),
                                     np.atleast_2d(a), delimiter=",")
    savv = lambda name, a: np.savetxt(os.path.join(io, name),
                                      np.asarray(a, float).ravel(), delimiter=",")

    X, y = _scheffe_inputs()
    sav("scheffe_X.csv", X); savv("scheffe_y.csv", y)
    savv("iopt_W.csv", None) if False else np.savetxt(
        os.path.join(io, "iopt_W.csv"), _moments_q5(), delimiter=",")

    y_max, y_min, y_tgt = _desirability_inputs()
    savv("desir_y_max.csv", y_max); savv("desir_y_min.csv", y_min)
    savv("desir_y_tgt.csv", y_tgt)
    with open(os.path.join(io, "desir_params.csv"), "w", encoding="utf-8") as fh:
        fh.write("low,high,target,s,s2,weight\n")
        fh.write("20,80,0,1,1,1\n")       # max
        fh.write("1.5,4.5,0,2,2,2\n")     # min
        fh.write("40,80,60,1,2,1\n")      # target

    Xtr, ytr, Xte, const, ls, noise = _gp_inputs()
    sav("gp_Xtrain.csv", Xtr); savv("gp_ytrain.csv", ytr); sav("gp_Xtest.csv", Xte)
    savv("gp_params.csv", [const, noise]); savv("gp_ls.csv", ls)

    res = subprocess.run([rscript, _R_SCRIPT, io], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Rscript failed:\n{res.stdout}\n{res.stderr}")
    print(res.stdout.strip())

    rd = lambda name: np.atleast_1d(np.loadtxt(os.path.join(io, name), delimiter=","))
    out = {
        "X": X, "y": y,
        "scheffe_coef": rd("out_scheffe_coef.csv"),
        "scheffe_fitted": rd("out_scheffe_fitted.csv"),
        "scheffe_scalars": rd("out_scheffe_scalars.csv"),
        "dopt": rd("out_dopt_scalars.csv"),
        "iopt": float(rd("out_iopt_scalar.csv")[0]),
        "moments": np.loadtxt(os.path.join(io, "iopt_W.csv"), delimiter=","),
        "d_max": rd("out_desir_dmax.csv"), "d_min": rd("out_desir_dmin.csv"),
        "d_tgt": rd("out_desir_dtgt.csv"), "d_overall": rd("out_desir_overall.csv"),
        "gp_mean": rd("out_gp_mean.csv"), "gp_sd": rd("out_gp_sd.csv"),
        "y_max": y_max, "y_min": y_min, "y_tgt": y_tgt,
        "gp": (Xtr, ytr, Xte, const, ls, noise),
    }
    shutil.rmtree(io, ignore_errors=True)
    return out


def _run_python_engine() -> dict:
    """Fallback: те же величины независимым Python-эталоном (reference.py)."""
    X, y = _scheffe_inputs()
    M = ref.scheffe_design_matrix(X, ORDER)
    g = ref.ref_ols(M, y)
    moments = _moments_q5()
    y_max, y_min, y_tgt = _desirability_inputs()
    d_max = ref.ref_desirability(y_max, "max", 20, 80, s=1.0)
    d_min = ref.ref_desirability(y_min, "min", 1.5, 4.5, s=2.0)
    d_tgt = ref.ref_desirability(y_tgt, "target", 40, 80, target=60, s=1.0, s2=2.0)
    overall = ref.ref_overall([d_max, d_min, d_tgt], [1.0, 2.0, 1.0])
    Xtr, ytr, Xte, const, ls, noise = _gp_inputs()
    gp = ref.gp_posterior(Xtr, ytr, Xte, const=const, length_scale=ls,
                          noise=noise, kernel="matern52")
    return {
        "X": X, "y": y,
        "scheffe_coef": g["coefficients"], "scheffe_fitted": g["fitted"],
        "scheffe_scalars": np.array([g["r2"], g["adj_r2"], g["rmse"]]),
        "dopt": np.array([ref.ref_d_criterion(M), ref.ref_d_efficiency(M)]),
        "iopt": ref.ref_i_criterion(M, moments), "moments": moments,
        "d_max": d_max, "d_min": d_min, "d_tgt": d_tgt, "d_overall": overall,
        "gp_mean": gp["mean"], "gp_sd": gp["std"],
        "y_max": y_max, "y_min": y_min, "y_tgt": y_tgt,
        "gp": (Xtr, ytr, Xte, const, ls, noise),
    }


# ----------------------------------------------------------------------
# Сборка фикстур
# ----------------------------------------------------------------------
def _gp_state(Xtr, ytr, const, ls, noise) -> dict:
    from sklearn.gaussian_process.kernels import (
        ConstantKernel, Matern, WhiteKernel)
    kern = (ConstantKernel(const) * Matern(length_scale=ls, nu=2.5)
            + WhiteKernel(noise))
    return {
        "mean_model": "quadratic", "kernel": "matern52", "noise_floor": 1e-6,
        "names": None,
        "scheffe": {"model": "quadratic", "names": None, "q": Q,
                    "term_names": TERM_NAMES, "coefficients": np.zeros(P),
                    "r2": 0.0, "adj_r2": 0.0, "rmse": 0.0},
        "kernel_theta": kern.theta, "X_train": Xtr, "resid_train": ytr,
    }


def build_fixtures(engine_tag: str, data: dict) -> dict:
    X, y = data["X"], data["y"]
    sc = data["scheffe_scalars"]
    Xtr, ytr, Xte, const, ls, noise = data["gp"]

    fixtures = {}
    fixtures["scheffe_ols"] = {
        "description": f"Scheffe quadratic OLS (q={Q}, p={P}) vs R lm.",
        "engine": engine_tag,
        "r_reference": ("df <- data.frame(y, M)  # M = 15 Scheffe terms\n"
                        "fit <- lm(y ~ . - 1, data=df); coef(fit)"),
        "tol": "scheffe_coef",
        "inputs": {"X": X, "y": y, "model": "quadratic"},
        "expected": {"coefficients": data["scheffe_coef"],
                     "fitted": data["scheffe_fitted"],
                     "r2": float(sc[0]), "adj_r2": float(sc[1]),
                     "rmse": float(sc[2])},
    }
    fixtures["d_optimality"] = {
        "description": f"D-criterion det(MᵀM) & D-efficiency (q={Q}, p={P}).",
        "engine": engine_tag,
        "r_reference": "det(t(M)%*%M); (det(t(M)%*%M))^(1/p)/n",
        "tol": "d_optimality",
        "inputs": {"X": X, "model": "quadratic"},
        "expected": {"d_criterion": float(data["dopt"][0]),
                     "d_efficiency": float(data["dopt"][1])},
    }
    fixtures["i_optimality"] = {
        "description": f"I-criterion trace((MᵀM)⁻¹·W) over simplex (q={Q}).",
        "engine": engine_tag,
        "r_reference": "sum(diag(solve(t(M)%*%M, W)))",
        "tol": "i_optimality",
        "inputs": {"X": X, "model": "quadratic", "moments": data["moments"]},
        "expected": {"i_criterion": float(data["iopt"])},
    }
    fixtures["desirability"] = {
        "description": "Derringer-Suich d_i (max/min/target) + weighted dOverall.",
        "engine": engine_tag,
        "r_reference": ("dMax/dMin/dTarget формулы; "
                        "exp(sum(w*log(d)))  # weighted geometric mean"),
        "tol": "desirability",
        "inputs": {
            "max": {"y": data["y_max"], "low": 20, "high": 80, "s": 1.0, "weight": 1.0},
            "min": {"y": data["y_min"], "low": 1.5, "high": 4.5, "s": 2.0, "weight": 2.0},
            "target": {"y": data["y_tgt"], "low": 40, "high": 80, "target": 60,
                       "s": 1.0, "s2": 2.0, "weight": 1.0},
        },
        "expected": {"d_max": data["d_max"], "d_min": data["d_min"],
                     "d_target": data["d_tgt"], "d_overall": data["d_overall"]},
    }
    fixtures["gp_fixed_hyper"] = {
        "description": f"GP Matern5/2 ARD posterior at fixed θ (q={Q}, 5 ARD).",
        "engine": engine_tag,
        "r_reference": ("K<-const*matern52(Xtr,Xtr,ls)+noise*I; "
                        "mu<-t(Ks)%*%solve(K,ytr); var<-const+noise-..."),
        "tol": "gp",
        "inputs": {"X_test": Xte,
                   "state": _gp_state(Xtr, ytr, const, ls, noise)},
        "expected": {"mean": data["gp_mean"], "std": data["gp_sd"]},
    }
    return fixtures


def build_gmm_fixture() -> dict:
    """GMM в пространстве свойств (2D), K=3 по BIC — структурная проверка."""
    rng = np.random.default_rng(7)
    centers = np.array([[0.0, 0.0], [14.0, 0.0], [7.0, 12.0]])
    m = 30
    Y = np.vstack([c + rng.normal(0, 0.25, size=(m, 2)) for c in centers])
    groups = np.array([0] * m + [1] * m + [2] * m)

    return {
        "description": "GMM regimes (2D property space): BIC selects K=3.",
        "engine": "python",
        "r_reference": "library(mclust); Mclust(Y)$G; Mclust(Y)$classification",
        "tol": None,
        "inputs": {"Y": Y, "k_range": [1, 2, 3, 4, 5], "seed": 0},
        "expected": {"n_regimes": 3, "groups": groups},
    }


# ----------------------------------------------------------------------
def main() -> int:
    engine = "auto"
    if "--engine" in sys.argv:
        engine = sys.argv[sys.argv.index("--engine") + 1]

    rscript = find_rscript()
    if engine == "python" or (engine == "auto" and rscript is None):
        tag = "python-reference"
        print(f"[golden] engine = {tag} (R not used)")
        data = _run_python_engine()
    else:
        tag = "R-base"
        print(f"[golden] engine = {tag}: {rscript}")
        data = _run_r_engine(rscript)

    fixtures = build_fixtures(tag, data)
    fixtures["gmm_regimes"] = build_gmm_fixture()
    for name, fx in fixtures.items():
        path = save_fixture(name, fx)
        print(f"[golden] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
