"""
Algorithm Comparison: Old Sequential vs New Efficient
======================================================
Compares two sequential mixture DOE strategies on the benchmark model:

  True model (5 components A,B,C,D,E):
    5A + 4B + 3C + 2D + 0.5E
  + 5AB + 4AC + 3BC + 2CD + 0.5DE
  + 5ABC + 4ACD + 3BCD
  + 5ABCD + 4BCDE
  + 5ABCDE
  (16 active terms out of possible 31)

OLD ALGORITHM  (doe_sequential_workflow_app.py)
  - D-optimal initial design (n_params+3 runs, quadratic Scheffé)
  - Greedy D-optimal augmentation (~half of initial runs)
  - Fits quadratic (2-way) model only
  - Cannot detect 3-way+ interactions at all

NEW ALGORITHM  (efficient_sequential_workflow_app.py / run_efficient_workflow.py)
  - Smart Simplex Centroid point generation (from streamlit_app.py)
  - JMP-validated per-order replication  (from smart_simplex_centroid.py)
  - Adaptive 3-phase staging: stops when R² target reached
  - Phase 1: vertices×2 + binary×1 + centroid×1
  - Phase 2: ternary blends (guided)
  - Phase 3: quaternary blends (guided)
  - Fits linear through 5th-order Scheffé

Usage:
  python run_algorithm_comparison.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from itertools import combinations
from math import comb
from scipy import stats as scipy_stats
from typing import Dict, List, Optional, Tuple
import io as _io

# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK MODEL
# ─────────────────────────────────────────────────────────────────────────────

COMP  = ["A", "B", "C", "D", "E"]
Q     = 5
TRUE_TERMS = {
    "A":5.0,"B":4.0,"C":3.0,"D":2.0,"E":0.5,
    "AB":5.0,"AC":4.0,"BC":3.0,"CD":2.0,"DE":0.5,
    "ABC":5.0,"ACD":4.0,"BCD":3.0,
    "ABCD":5.0,"BCDE":4.0,
    "ABCDE":5.0,
}

def true_y(pt: np.ndarray, noise: float = 0.0,
           rng: Optional[np.random.Generator] = None) -> float:
    A,B,C,D,E = pt
    v = (5*A+4*B+3*C+2*D+0.5*E
         +5*A*B+4*A*C+3*B*C+2*C*D+0.5*D*E
         +5*A*B*C+4*A*C*D+3*B*C*D
         +5*A*B*C*D+4*B*C*D*E
         +5*A*B*C*D*E)
    if noise > 0 and rng is not None:
        v += rng.normal(0, noise)
    return float(v)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def build_X(design: np.ndarray, max_order: int) -> Tuple[np.ndarray, List[str]]:
    """Build Scheffé model matrix (no intercept)."""
    q   = design.shape[1]
    cols, names = [], []
    for i in range(q):
        cols.append(design[:, i])
        names.append(COMP[i])
    for order in range(2, min(max_order, q)+1):
        for combo in combinations(range(q), order):
            col = np.ones(len(design))
            for idx in combo:
                col *= design[:, idx]
            cols.append(col)
            names.append("".join(COMP[j] for j in combo))
    return np.column_stack(cols), names


def fit(X: np.ndarray, y: np.ndarray) -> Dict:
    """OLS fit returning statistics."""
    n, p = X.shape
    df_r = max(n - p, 1)
    try:
        b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception as e:
        return {"error": str(e)}
    yp = X @ b
    res = y - yp
    ss_t = np.sum((y - y.mean())**2)
    ss_r = np.sum(res**2)
    r2   = 1 - ss_r/ss_t if ss_t > 0 else 0.0
    mse  = ss_r / df_r
    rmse = mse**0.5
    adj  = 1 - (ss_r/df_r)/(ss_t/max(n-1,1)) if ss_t > 0 else 0.0
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = (np.abs(mse * np.diag(XtX_inv)))**0.5
        t  = b / (se + 1e-300)
        pv = 2*(1 - scipy_stats.t.cdf(np.abs(t), df_r))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan); t = np.full(p, np.nan); pv = np.ones(p)
    return dict(b=b, se=se, t=t, pv=pv, yp=yp, res=res,
                r2=r2, adj=adj, rmse=rmse, mse=mse, n=n, p=p, df_r=df_r)


def cv_r2(b: np.ndarray, max_order: int, n_test: int = 1000, seed: int = 99) -> float:
    """Cross-validated R² on new Dirichlet test points."""
    rng  = np.random.default_rng(seed)
    pts  = rng.dirichlet(np.ones(Q), n_test)
    yt   = np.array([true_y(pt) for pt in pts])
    Xt,_ = build_X(pts, max_order)
    nc   = min(len(b), Xt.shape[1])
    yp   = Xt[:, :nc] @ b[:nc]
    ss_t = np.sum((yt - yt.mean())**2)
    ss_r = np.sum((yt - yp)**2)
    return float(1 - ss_r/ss_t)


def term_recovery(b: np.ndarray, pv: np.ndarray, names: List[str], alpha=0.05):
    """
    Return (true_positive, false_positive, false_negative) counts.
    true_positive  = true non-zero terms detected as significant
    false_positive = zero terms detected as significant
    false_negative = true non-zero terms missed (not significant)
    """
    tp = fp = fn = 0
    for i, name in enumerate(names):
        true_nonzero = abs(TRUE_TERMS.get(name, 0.0)) > 0
        sig = pv[i] < alpha
        if true_nonzero and sig:   tp += 1
        if (not true_nonzero) and sig: fp += 1
        if true_nonzero and not sig:   fn += 1
    return tp, fp, fn


def _centroid(q: int, idx: Tuple) -> np.ndarray:
    pt = np.zeros(q)
    for i in idx: pt[i] = 1.0/len(idx)
    return pt


def smart_reps(q: int) -> Dict[int,int]:
    k_ref = 2; maxp = comb(q, min(k_ref,q)); reps = {}
    for k in range(1, q+1):
        if k==1:      reps[k]=2
        elif k==q:    reps[k]=3
        elif k>=q-1:  reps[k]=1
        else:         reps[k]=max(1,min(3,round(maxp/comb(q,k))))
    return reps


# ─────────────────────────────────────────────────────────────────────────────
# OLD ALGORITHM  (faithful replica of doe_sequential_workflow_app.py)
# ─────────────────────────────────────────────────────────────────────────────

class OldSequentialDoe:
    """
    Replicates the doe_sequential_workflow_app.py strategy:

      1. Simplex-lattice base (vertices + binary blends + centroid)
         augmented D-optimally to reach n_initial runs.
      2. Fit quadratic Scheffé (linear + 2-way).
      3. Greedy D-optimal augmentation of ~n_initial//2 new runs.
      4. Refit with combined dataset.

    Model is capped at quadratic — identical to the old app's default.
    We also test a 'cubic' variant to show its limit.
    """

    def __init__(self, noise: float = 0.0, model_type: str = "quadratic",
                 seed: int = 42):
        self.noise = noise
        self.model_type = model_type   # "quadratic" | "cubic"
        self.seed = seed
        self.rng  = np.random.default_rng(seed)
        self.max_order = 2 if model_type == "quadratic" else 3

        # Count minimum parameters for chosen model
        self.n_params     = Q + comb(Q, 2) + (comb(Q,3) if model_type=="cubic" else 0)
        self.n_initial    = self.n_params + 3  # initial D-opt runs
        self.n_aug        = self.n_initial // 2

        self.design : Optional[np.ndarray] = None
        self.responses   : Optional[np.ndarray] = None
        self.fit_result  : Optional[Dict]       = None
        self.stage_log   : List[Dict]           = []

    # ── Step 1: Initial design ────────────────────────────────────────────────

    def _gen_initial(self) -> np.ndarray:
        """
        Simplex-lattice base + D-optimal interior augmentation.
        Mirrors _generate_mixture_design() from doe_sequential_workflow_app.py.
        """
        # Structured base (all vertices, binary blends, centroid)
        base = []
        for k in range(1, Q+1):
            if k > self.max_order and k < Q: continue   # skip ternary+ for quadratic
            for combo in combinations(range(Q), k):
                base.append(_centroid(Q, combo))
        base = np.array(base)
        n_base = len(base)

        if self.n_initial <= n_base:
            # Greedy max-min selection from base
            sel = [0]
            rem = list(range(1, n_base))
            while len(sel) < self.n_initial and rem:
                sel_pts = base[sel]
                best = max(rem, key=lambda i: np.min(
                    np.sum((base[i]-sel_pts)**2, axis=1)))
                sel.append(best); rem.remove(best)
            return base[sel]

        # Augment with interior (Dirichlet) points
        extra = self.n_initial - n_base
        cands = self.rng.dirichlet(np.ones(Q)*2, extra*20)
        current = base.copy()
        added   = []
        rem     = list(range(len(cands)))
        while len(added) < extra and rem:
            best = max(rem, key=lambda i: np.min(
                np.sum((cands[i]-current)**2, axis=1)))
            added.append(best); current = np.vstack([current, cands[best]])
            rem.remove(best)
        aug = cands[added] if added else cands[:extra]
        return np.vstack([base, aug])

    # ── Step 2: Greedy D-optimal augmentation ────────────────────────────────

    def _augment(self, existing: np.ndarray) -> np.ndarray:
        """
        Greedy D-optimal augmentation — mirrors generate_augmentation_mixture().
        """
        cands = self.rng.dirichlet(np.ones(Q), 600)
        aug   = []
        cur   = existing.copy()
        for _ in range(self.n_aug):
            X_cur,_ = build_X(cur, self.max_order)
            base_m  = X_cur.T @ X_cur
            best_det = -np.inf; best_pt = None
            for cand in cands:
                xn,_ = build_X(cand.reshape(1,-1), self.max_order)
                m_new = base_m + xn.T @ xn
                try:   d = np.linalg.det(m_new)
                except: d = 0.0
                if d > best_det:
                    best_det = d; best_pt = cand
            if best_pt is not None:
                aug.append(best_pt)
                cur = np.vstack([cur, best_pt.reshape(1,-1)])
                cands = cands[~np.all(np.isclose(cands, best_pt), axis=1)]
        return np.array(aug) if aug else cands[:self.n_aug]

    # ── Public run ────────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"OLD ALGORITHM  (model_type={self.model_type})")
            print(f"{'─'*60}")

        # --- Stage 1: initial design ---
        d_init = self._gen_initial()
        y_init = np.array([true_y(pt, self.noise, self.rng) for pt in d_init])
        if verbose:
            print(f"  Stage 1 (initial D-optimal): {len(d_init)} runs")

        X1, names1 = build_X(d_init, self.max_order)
        f1 = fit(X1, y1 := y_init)

        self.stage_log.append({"stage": "initial",
                                "n": len(d_init), "r2": f1["r2"],
                                "adj": f1["adj"], "rmse": f1["rmse"]})
        if verbose:
            print(f"    R²={f1['r2']:.4f}  Adj-R²={f1['adj']:.4f}  "
                  f"RMSE={f1['rmse']:.4f}")

        # --- Stage 2: augmentation ---
        d_aug = self._augment(d_init)
        y_aug = np.array([true_y(pt, self.noise, self.rng) for pt in d_aug])
        if verbose:
            print(f"  Stage 2 (D-optimal augmentation): +{len(d_aug)} runs")

        # Combined
        d_all = np.vstack([d_init, d_aug])
        y_all = np.concatenate([y_init, y_aug])
        X2, names2 = build_X(d_all, self.max_order)
        f2 = fit(X2, y_all)

        self.design    = d_all
        self.responses = y_all
        self.fit_result = f2
        tp, fp, fn = term_recovery(f2["b"], f2["pv"], names2)
        cv = cv_r2(f2["b"], self.max_order)

        self.stage_log.append({"stage": "augmented",
                                "n": len(d_all), "r2": f2["r2"],
                                "adj": f2["adj"], "rmse": f2["rmse"],
                                "tp": tp, "fp": fp, "fn": fn, "cv_r2": cv})
        if verbose:
            print(f"    Combined {len(d_all)} runs:"
                  f"  R²={f2['r2']:.4f}  Adj-R²={f2['adj']:.4f}"
                  f"  RMSE={f2['rmse']:.4f}")
            print(f"    Term recovery: TP={tp}  FP={fp}  FN={fn}  "
                  f"CV-R²={cv:.4f}")

        return {"n_runs": len(d_all),
                "r2": f2["r2"], "adj": f2["adj"], "rmse": f2["rmse"],
                "cv_r2": cv, "tp": tp, "fp": fp, "fn": fn,
                "max_order": self.max_order, "names": names2,
                "b": f2["b"], "pv": f2["pv"]}


# ─────────────────────────────────────────────────────────────────────────────
# NEW ALGORITHM  (efficient_sequential_workflow / smart simplex centroid)
# ─────────────────────────────────────────────────────────────────────────────

class NewAdaptiveDoe:
    """
    Replicates the efficient_sequential_workflow_app.py strategy.
    Uses Smart Simplex Centroid point generation + JMP-validated replication.
    """

    def __init__(self, noise: float = 0.0, r2_target: float = 0.97,
                 alpha: float = 0.05, seed: int = 42):
        self.noise     = noise
        self.r2_target = r2_target
        self.alpha     = alpha
        self.rng       = np.random.default_rng(seed)

        self.design_pts : List[np.ndarray] = []
        self.resp       : List[float]      = []
        self.stage_log  : List[Dict]       = []

    def _add(self, pts: List[np.ndarray]):
        for pt in pts:
            self.design_pts.append(pt)
            self.resp.append(true_y(pt, self.noise, self.rng))

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    def _phase1(self):
        reps = smart_reps(Q)
        pts  = []
        # Vertices × 2
        for c in combinations(range(Q), 1):
            for _ in range(reps[1]): pts.append(_centroid(Q, c))
        # Binary × 1
        for c in combinations(range(Q), 2):
            pts.append(_centroid(Q, c))
        # Overall centroid × 1
        pts.append(_centroid(Q, tuple(range(Q))))
        self._add(pts)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    def _phase2(self, sig_binary_idx: List[int]):
        binary_combos  = list(combinations(range(Q), 2))
        ternary_all    = list(combinations(range(Q), 3))
        reps = smart_reps(Q)

        if sig_binary_idx:
            sig_set = set(sig_binary_idx)
            ternary_sel = [t for t in ternary_all
                           if any(binary_combos.index((i,j)) in sig_set
                                  for i,j in combinations(t,2)
                                  if (i,j) in binary_combos)]
            if not ternary_sel: ternary_sel = ternary_all
        else:
            ternary_sel = ternary_all

        pts = [_centroid(Q, c) for c in ternary_sel
               for _ in range(max(reps[3],1))]
        pts += [_centroid(Q, tuple(range(Q))) for _ in range(2)]
        self._add(pts)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    def _phase3(self, sig_ternary_idx: List[int], sig_binary_idx: List[int]):
        ternary_combos = list(combinations(range(Q), 3))
        quat_all       = list(combinations(range(Q), 4))
        binary_combos  = list(combinations(range(Q), 2))
        reps = smart_reps(Q)

        if sig_ternary_idx:
            sig_set = set(sig_ternary_idx)
            quat_sel = [qb for qb in quat_all
                        if any(ternary_combos.index(tr) in sig_set
                               for tr in combinations(qb,3)
                               if tr in ternary_combos)]
            if not quat_sel: quat_sel = quat_all
        else:
            quat_sel = quat_all

        pts = [_centroid(Q, c) for c in quat_sel
               for _ in range(max(reps[4],1))]
        # Replicate top 2 binary
        for bi in sig_binary_idx[:2]:
            if bi < len(binary_combos):
                pts.append(_centroid(Q, binary_combos[bi]))
        # Replicate top 3 ternary
        for ti in sig_ternary_idx[:3]:
            if ti < len(ternary_combos):
                pts.append(_centroid(Q, ternary_combos[ti]))
        self._add(pts)

    # ── fit helper ────────────────────────────────────────────────────────────
    def _fit(self, max_order: int) -> Dict:
        d = np.array(self.design_pts)
        y = np.array(self.resp)
        X, names = build_X(d, max_order)
        while X.shape[1] > X.shape[0] and max_order > 1:
            max_order -= 1
            X, names = build_X(d, max_order)
        f = fit(X, y)
        if "error" in f:
            return {}
        n_bin  = comb(Q,2)
        n_ter  = comb(Q,3) if max_order >= 3 else 0
        sig_b  = [i for i,p in enumerate(f["pv"][Q:Q+n_bin]) if p < self.alpha]
        sig_t  = [i for i,p in enumerate(f["pv"][Q+n_bin:Q+n_bin+n_ter]) if p < self.alpha]
        f.update(dict(sig_b=sig_b, sig_t=sig_t,
                      names=names, max_order=max_order))
        return f

    # ── Public run ────────────────────────────────────────────────────────────
    def run(self, verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"NEW ALGORITHM  (adaptive Smart Simplex Centroid)")
            print(f"{'─'*60}")

        # Phase 1
        self._phase1()
        f1 = self._fit(max_order=2)
        if verbose:
            print(f"  Phase 1 (vertices+binary+centroid): {len(self.design_pts)} runs  "
                  f"R²={f1.get('r2',0):.4f}")
        if f1.get("r2",0) >= self.r2_target:
            if verbose: print(f"    --> R² target reached. Stopping.")
            pass
        else:
            # Phase 2
            self._phase2(f1.get("sig_b",[]))
            f2 = self._fit(max_order=3)
            if verbose:
                print(f"  Phase 2 (+ternary+LOF): {len(self.design_pts)} runs  "
                      f"R²={f2.get('r2',0):.4f}")
            if f2.get("r2",0) >= self.r2_target:
                if verbose: print(f"    --> R² target reached. Stopping.")
                f1 = f2
            else:
                # Phase 3
                self._phase3(f2.get("sig_t",[]), f2.get("sig_b",[]))
                f3 = self._fit(max_order=Q)
                if verbose:
                    print(f"  Phase 3 (+quaternary): {len(self.design_pts)} runs  "
                          f"R²={f3.get('r2',0):.4f}")
                f1 = f3

        final = f1
        tp, fp, fn = term_recovery(final.get("b", np.zeros(1)),
                                   final.get("pv", np.ones(1)),
                                   final.get("names", []))
        cv = cv_r2(final.get("b", np.zeros(1)), final.get("max_order", 2))

        if verbose:
            print(f"    Term recovery: TP={tp}  FP={fp}  FN={fn}  "
                  f"CV-R²={cv:.4f}")

        return {"n_runs": len(self.design_pts),
                "r2": final.get("r2",0),
                "adj": final.get("adj",0), "rmse": final.get("rmse",0),
                "cv_r2": cv, "tp": tp, "fp": fp, "fn": fn,
                "max_order": final.get("max_order", 2),
                "names": final.get("names",[]),
                "b": final.get("b", np.zeros(1)),
                "pv": final.get("pv", np.ones(1))}


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON BATTERY
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison_battery(noise_levels=(0.0, 0.05, 0.10, 0.20),
                            n_reps: int = 5,
                            verbose_single: bool = True) -> pd.DataFrame:
    """
    Run both algorithms n_reps times per noise level.
    Return a summary DataFrame.
    """

    print("\n" + "="*74)
    print("FULL COMPARISON BATTERY")
    print("="*74)

    records = []

    for noise in noise_levels:
        old_q_runs  = []; old_q_r2s  = []; old_q_adj  = []; old_q_rmse = []
        old_q_cv    = []; old_q_tp   = []; old_q_fp   = []; old_q_fn   = []

        old_c_runs  = []; old_c_r2s  = []; old_c_adj  = []; old_c_rmse = []
        old_c_cv    = []; old_c_tp   = []; old_c_fp   = []; old_c_fn   = []

        new_runs    = []; new_r2s    = []; new_adj    = []; new_rmse   = []
        new_cv      = []; new_tp     = []; new_fp     = []; new_fn     = []

        for rep in range(n_reps):
            seed = 42 + rep * 17

            # --- Old quadratic ---
            old_stdout = sys.stdout; sys.stdout = _io.StringIO()
            oa = OldSequentialDoe(noise=noise, model_type="quadratic", seed=seed).run(verbose=False)
            sys.stdout = old_stdout
            old_q_runs.append(oa["n_runs"]); old_q_r2s.append(oa["r2"])
            old_q_adj.append(oa["adj"]);     old_q_rmse.append(oa["rmse"])
            old_q_cv.append(oa["cv_r2"]);    old_q_tp.append(oa["tp"])
            old_q_fp.append(oa["fp"]);       old_q_fn.append(oa["fn"])

            # --- Old cubic ---
            old_stdout = sys.stdout; sys.stdout = _io.StringIO()
            ob = OldSequentialDoe(noise=noise, model_type="cubic", seed=seed).run(verbose=False)
            sys.stdout = old_stdout
            old_c_runs.append(ob["n_runs"]); old_c_r2s.append(ob["r2"])
            old_c_adj.append(ob["adj"]);     old_c_rmse.append(ob["rmse"])
            old_c_cv.append(ob["cv_r2"]);    old_c_tp.append(ob["tp"])
            old_c_fp.append(ob["fp"]);       old_c_fn.append(ob["fn"])

            # --- New adaptive ---
            old_stdout = sys.stdout; sys.stdout = _io.StringIO()
            nr = NewAdaptiveDoe(noise=noise, r2_target=0.97, seed=seed).run(verbose=False)
            sys.stdout = old_stdout
            new_runs.append(nr["n_runs"]); new_r2s.append(nr["r2"])
            new_adj.append(nr["adj"]);     new_rmse.append(nr["rmse"])
            new_cv.append(nr["cv_r2"]);    new_tp.append(nr["tp"])
            new_fp.append(nr["fp"]);       new_fn.append(nr["fn"])

        records.append({
            "Noise σ": noise,
            # Old quadratic
            "OldQ Runs":    int(round(np.mean(old_q_runs))),
            "OldQ Train-R²": round(np.mean(old_q_r2s), 4),
            "OldQ CV-R²":   round(np.mean(old_q_cv),  4),
            "OldQ RMSE":    round(np.mean(old_q_rmse), 4),
            "OldQ TP":      round(np.mean(old_q_tp), 1),
            "OldQ FP":      round(np.mean(old_q_fp), 1),
            "OldQ FN":      round(np.mean(old_q_fn), 1),
            # Old cubic
            "OldC Runs":    int(round(np.mean(old_c_runs))),
            "OldC Train-R²": round(np.mean(old_c_r2s), 4),
            "OldC CV-R²":   round(np.mean(old_c_cv),  4),
            "OldC RMSE":    round(np.mean(old_c_rmse), 4),
            "OldC TP":      round(np.mean(old_c_tp), 1),
            "OldC FP":      round(np.mean(old_c_fp), 1),
            "OldC FN":      round(np.mean(old_c_fn), 1),
            # New adaptive
            "New Runs":     int(round(np.mean(new_runs))),
            "New Train-R²": round(np.mean(new_r2s), 4),
            "New CV-R²":    round(np.mean(new_cv),  4),
            "New RMSE":     round(np.mean(new_rmse), 4),
            "New TP":       round(np.mean(new_tp), 1),
            "New FP":       round(np.mean(new_fp), 1),
            "New FN":       round(np.mean(new_fn), 1),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN POINT DISTRIBUTION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def design_point_breakdown(verbose=True) -> pd.DataFrame:
    """Show how many blend points of each order each algorithm uses."""

    def count_orders(pts: np.ndarray) -> Dict[int,int]:
        cnt = {}
        for pt in pts:
            o = sum(1 for x in pt if x > 1e-9)
            cnt[o] = cnt.get(o,0) + 1
        return cnt

    rows = []
    for q in [3, 4, 5]:
        # New Phase 1 for q=q
        reps = smart_reps(q)
        pts_new = []
        for c in combinations(range(q),1):
            for _ in range(reps[1]): pts_new.append(_centroid(q,c))
        for c in combinations(range(q),2): pts_new.append(_centroid(q,c))
        pts_new.append(_centroid(q,tuple(range(q))))
        new_p1 = np.array(pts_new)

        # Old quadratic initial for q=q
        n_p = q + comb(q,2)
        n_init_old = n_p + 3
        old_base = []
        for k in [1,2,q]:
            for c in combinations(range(q),k):
                old_base.append(_centroid(q,c))
        old_arr = np.array(old_base)

        r = {"q": q,
             "New Phase1 runs": len(new_p1),
             "Old Initial runs": n_init_old,
             "New Phase1 has all binary": int(len(new_p1) >= q + comb(q,2)),
             "New Phase1 has all ternary": 0,  # Phase 1 does NOT include ternary
             "JMP total": sum(comb(q,k)*reps.get(k,1) for k in range(1,q+1)),
        }
        rows.append(r)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# DETAILED SINGLE-RUN COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def single_run_detailed():
    """Run both algorithms once (zero noise) and print full side-by-side."""

    print("\n" + "="*74)
    print("SINGLE-RUN DETAILED COMPARISON  (zero noise, seed=42)")
    print("="*74)

    print("\n--- OLD ALGORITHM (quadratic model) ---")
    old_q = OldSequentialDoe(noise=0.0, model_type="quadratic", seed=42)
    rq = old_q.run(verbose=True)

    print("\n--- OLD ALGORITHM (cubic model) ---")
    old_c = OldSequentialDoe(noise=0.0, model_type="cubic", seed=42)
    rc = old_c.run(verbose=True)

    print("\n--- NEW ALGORITHM (adaptive, up to 5-way) ---")
    new = NewAdaptiveDoe(noise=0.0, r2_target=0.97, seed=42)
    rn = new.run(verbose=True)

    # ── Side-by-side coefficient table ────────────────────────────────────────
    print(f"\n{'─'*74}")
    print(f"COEFFICIENT RECOVERY COMPARISON  (★ = active in true model)")
    print(f"{'─'*74}")
    print(f"  {'Term':<8}  {'True':>6}  "
          f"{'OldQ coef':>10} {'OldQ p':>7}  "
          f"{'OldC coef':>10} {'OldC p':>7}  "
          f"{'New coef':>10} {'New p':>7}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*10} {'─'*7}  "
          f"{'─'*10} {'─'*7}  {'─'*10} {'─'*7}")

    # All term names across all three models
    all_terms = list(dict.fromkeys(
        list(rq.get("names",[])) +
        list(rc.get("names",[])) +
        list(rn.get("names",[]))
    ))

    def _lookup(result, name):
        names = result.get("names",[])
        b     = result.get("b", np.zeros(1))
        pv    = result.get("pv", np.ones(1))
        if name in names:
            i = names.index(name)
            return float(b[i]), float(pv[i])
        return None, None

    tp_q = tp_c = tp_n = 0
    fp_q = fp_c = fp_n = 0
    fn_q = fn_c = fn_n = 0
    n_true = sum(1 for t in TRUE_TERMS.values() if abs(t)>0)

    for term in all_terms:
        true_val = TRUE_TERMS.get(term, 0.0)
        star     = "★" if abs(true_val)>0 else " "
        bq, pq = _lookup(rq, term)
        bc, pc = _lookup(rc, term)
        bn, pn = _lookup(rn, term)

        fmt_coef = lambda b: f"{b:10.3f}" if b is not None else f"{'---':>10}"
        fmt_p    = lambda p: (f"{p:7.4f}" if (p is not None and p < 0.05)
                              else (f"{p:7.4f}" if p is not None else f"{'---':>7}"))

        def sig_mark(p): return "*" if (p is not None and p < 0.05) else " "

        row = (f"  {star}{term:<7}  {true_val:6.2f}  "
               f"{fmt_coef(bq)}{sig_mark(pq)}{fmt_p(pq)}  "
               f"{fmt_coef(bc)}{sig_mark(pc)}{fmt_p(pc)}  "
               f"{fmt_coef(bn)}{sig_mark(pn)}{fmt_p(pn)}")
        print(row)

        for (b,p,tp_r,fp_r,fn_r) in [
                (bq,pq,tp_q,fp_q,fn_q),(bc,pc,tp_c,fp_c,fn_c),(bn,pn,tp_n,fp_n,fn_n)]:
            pass

    # Recompute properly
    for result, label in [(rq,"OldQ"),(rc,"OldC"),(rn,"New")]:
        names = result.get("names",[])
        b     = result.get("b", np.zeros(len(names)))
        pv    = result.get("pv", np.ones(len(names)))
        tp_l, fp_l, fn_l = term_recovery(b, pv, names)
        cv_l  = result.get("cv_r2",0)
        print(f"\n  {label}: TP={tp_l}  FP={fp_l}  FN={fn_l}  "
              f"Runs={result['n_runs']}  Train-R²={result['r2']:.4f}  "
              f"CV-R²={cv_l:.4f}  max_order={result['max_order']}")

    print(f"\n  True model has {n_true} non-zero terms.")
    print(f"  TP = correctly detected active terms")
    print(f"  FP = falsely flagged inactive terms")
    print(f"  FN = missed active terms (false negatives)")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(df: pd.DataFrame):
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\n{'═'*74}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'═'*74}")
    print(f"Columns:")
    print(f"  OldQ = Old algorithm, quadratic (2-way) Scheffé model")
    print(f"  OldC = Old algorithm, cubic (3-way) Scheffé model")
    print(f"  New  = New adaptive Smart Simplex Centroid (up to 5-way)")
    print(f"  TP/FP/FN = True Positive / False Positive / False Negative term detections")
    print(f"  CV-R² = Cross-validated R² on 1000 new Dirichlet test points")
    print()

    # Compact view
    view_cols = [
        "Noise σ",
        "OldQ Runs","OldQ Train-R²","OldQ CV-R²","OldQ RMSE","OldQ TP","OldQ FP","OldQ FN",
        "OldC Runs","OldC Train-R²","OldC CV-R²","OldC RMSE","OldC TP","OldC FP","OldC FN",
        "New Runs","New Train-R²","New CV-R²","New RMSE","New TP","New FP","New FN",
    ]
    view = df[view_cols].copy()

    # Pretty-print row by row
    header1 = f"  {'Noise':>7}  | {'OldQ (quadratic)':^40} | {'OldC (cubic)':^40} | {'New (adaptive)':^40}"
    header2 = (f"  {'σ':>7}  | "
               f"{'Runs':>5} {'R²-tr':>7} {'R²-cv':>7} {'RMSE':>7} {'TP':>3} {'FP':>3} {'FN':>3} | "
               f"{'Runs':>5} {'R²-tr':>7} {'R²-cv':>7} {'RMSE':>7} {'TP':>3} {'FP':>3} {'FN':>3} | "
               f"{'Runs':>5} {'R²-tr':>7} {'R²-cv':>7} {'RMSE':>7} {'TP':>3} {'FP':>3} {'FN':>3}")
    print(header1)
    print(f"  {'─'*7}  | {'─'*40} | {'─'*40} | {'─'*40}")
    print(header2)
    print(f"  {'─'*7}──|──{'─'*40}──|──{'─'*40}──|──{'─'*40}")

    for _, row in df.iterrows():
        print(f"  {row['Noise σ']:>7.2f}  | "
              f"{row['OldQ Runs']:>5} {row['OldQ Train-R²']:>7.4f} {row['OldQ CV-R²']:>7.4f} "
              f"{row['OldQ RMSE']:>7.4f} {row['OldQ TP']:>3.0f} {row['OldQ FP']:>3.0f} {row['OldQ FN']:>3.0f} | "
              f"{row['OldC Runs']:>5} {row['OldC Train-R²']:>7.4f} {row['OldC CV-R²']:>7.4f} "
              f"{row['OldC RMSE']:>7.4f} {row['OldC TP']:>3.0f} {row['OldC FP']:>3.0f} {row['OldC FN']:>3.0f} | "
              f"{row['New Runs']:>5} {row['New Train-R²']:>7.4f} {row['New CV-R²']:>7.4f} "
              f"{row['New RMSE']:>7.4f} {row['New TP']:>3.0f} {row['New FP']:>3.0f} {row['New FN']:>3.0f}")

    # ── Insight summary ────────────────────────────────────────────────────────
    row0 = df[df["Noise σ"]==0.0].iloc[0]
    print(f"\n{'─'*74}")
    print(f"KEY INSIGHTS (zero noise, q=5, 16-term true model):")
    print(f"{'─'*74}")
    print(f"  Run counts:")
    print(f"    OldQ: {row0['OldQ Runs']} runs (quadratic, 2-way only)")
    print(f"    OldC: {row0['OldC Runs']} runs (cubic, up to 3-way)")
    print(f"    New:  {row0['New Runs']} runs (adaptive, up to 5-way)")

    savings_vs_oldc = float(row0['OldC Runs']) - float(row0['New Runs'])
    savings_pct     = 100.0 * savings_vs_oldc / float(row0['OldC Runs'])
    print(f"\n  New vs OldC saves: {savings_vs_oldc:+.0f} runs ({savings_pct:.0f}%)")

    print(f"\n  True positive rate (out of {len(TRUE_TERMS)} active terms):")
    print(f"    OldQ TP: {row0['OldQ TP']:.0f}  (misses all 3-way+ terms → FN={row0['OldQ FN']:.0f})")
    print(f"    OldC TP: {row0['OldC TP']:.0f}  (misses 4-way+ terms → FN={row0['OldC FN']:.0f})")
    print(f"    New  TP: {row0['New TP']:.0f}  FN={row0['New FN']:.0f}")

    print(f"\n  Cross-validated R² (1000 test points):")
    print(f"    OldQ CV-R²: {row0['OldQ CV-R²']:.4f}")
    print(f"    OldC CV-R²: {row0['OldC CV-R²']:.4f}")
    print(f"    New  CV-R²: {row0['New CV-R²']:.4f}")

    print(f"\n  False positives (zero terms flagged significant):")
    print(f"    OldQ FP: {row0['OldQ FP']:.0f}")
    print(f"    OldC FP: {row0['OldC FP']:.0f}")
    print(f"    New  FP: {row0['New FP']:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("ALGORITHM COMPARISON: OLD SEQUENTIAL vs NEW EFFICIENT")
    print(f"Benchmark model: {len(TRUE_TERMS)}-term, 5-component Scheffé (linear through 5-way)")
    print(f"True model: 5A+4B+3C+2D+0.5E+5AB+4AC+3BC+2CD+0.5DE+5ABC+4ACD+3BCD+5ABCD+4BCDE+5ABCDE")

    # 1. Detailed single run
    single_run_detailed()

    # 2. Battery comparison
    print(f"\n(Running battery comparison — ~30 seconds...)")
    df = run_comparison_battery(
        noise_levels=[0.0, 0.05, 0.10, 0.20],
        n_reps=5,
    )

    # 3. Print table + insights
    print_comparison_table(df)

    # 4. Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "algorithm_comparison_results.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Full results saved to: {csv_path}")

    print(f"\n{'═'*74}")
    print("DONE")
    print(f"{'═'*74}")


if __name__ == "__main__":
    main()
