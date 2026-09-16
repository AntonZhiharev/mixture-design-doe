"""verification/torn_plots.py — графика battle-тестов с РВАНОЙ областью (iter100).

Две картинки, отвечающие на вопрос «куда GP звал и куда ставили точки»:

  * :func:`star_projection` — «цветик»: составные координаты ``Xc``
    (доли A,B,C + процесс T,P в коде [0,1]) проецируются на плоскость
    как взвешенная сумма единичных векторов-осей, разложенных по кругу
    (star coordinates / RadViz-подобная проекция). Одна и та же проекция
    даёт две СВЯЗАННЫЕ панели: слева — карта поля суррогата (σ или μ) с
    наложенной истинной «дырой», справа — точки: seed годные/негодные,
    предложения по раундам, ``x_best``, аналитический оптимум.
  * :func:`cliff_slice` — срез вдоль расстояния до кромки ``g − threshold``:
    истина (экспонента до обрыва), μ±2σ суррогата по кандидатам, точки
    раундов — видно, как ядро «подкрадывается» к обрыву и что оно
    экстраполирует за кромкой.

Только matplotlib (backend Agg, без окон). Функции чистые: принимают массивы,
возвращают путь к PNG; не зависят от раннера, чтобы тест мог рисовать
любую стадию.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# ----------------------------------------------------------------------
# Star-проекция составных координат
# ----------------------------------------------------------------------
def star_axes(n_axes: int) -> np.ndarray:
    """Единичные векторы ``n_axes`` осей, равномерно по кругу (n×2)."""
    ang = np.linspace(0.0, 2 * np.pi, int(n_axes), endpoint=False) + np.pi / 2
    return np.column_stack([np.cos(ang), np.sin(ang)])


def star_project(Xc: np.ndarray, weights: Optional[Sequence[float]] = None
                 ) -> np.ndarray:
    """Проекция ``Xc`` (n×k) на плоскость: ``Σ_j w_j·x_j·e_j`` (n×2).

    Все координаты ожидаются в [0,1] (доли и код процесса). ``weights`` —
    вес оси (по умолчанию 1).
    """
    Xc = np.atleast_2d(np.asarray(Xc, float))
    k = Xc.shape[1]
    E = star_axes(k)
    w = np.ones(k) if weights is None else np.asarray(weights, float)
    return (Xc * w) @ E


def _draw_star_frame(ax, names: Sequence[str]) -> None:
    E = star_axes(len(names))
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, lw=0.8, color="0.6",
                            ls="--"))
    for (ex, ey), nm in zip(E, names):
        ax.plot([0, ex], [0, ey], color="0.55", lw=0.8)
        ax.text(ex * 1.08, ey * 1.08, nm, ha="center", va="center",
                fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _round_color(k: int, K: int):
    return plt.get_cmap("autumn_r")(0.15 + 0.85 * k / max(K - 1, 1))


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def star_projection(*, names: Sequence[str],
                    grid_Xc: np.ndarray, grid_field: np.ndarray,
                    grid_feasible: np.ndarray,
                    seed_Xc: np.ndarray, seed_feasible: np.ndarray,
                    round_Xc: Sequence[np.ndarray],
                    round_feasible: Sequence[np.ndarray],
                    x_best: Optional[np.ndarray] = None,
                    x_opt: Optional[np.ndarray] = None,
                    field_label: str = "σ суррогата",
                    title: str = "", path: str | Path = "star.png") -> Path:
    """Две связанные панели одной star-проекции: поле суррогата + точки.

    ``grid_*`` — плотный пул кандидатов: поле рисуется как scatter по
    проекции (разные рецепты могут проецироваться рядом — это ЛИНИЯ ВЗГЛЯДА
    на 4-мерную область, а не карта плоскости); дыра — те же точки пула
    с ``grid_feasible=False``. ``round_Xc[k]`` — точки раунда k+1.
    """
    names = list(names)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.8))
    fig.suptitle(title, fontsize=12)

    # ---------- левая панель: поле суррогата + истинная дыра ----------
    g2 = star_project(grid_Xc)
    fld = np.asarray(grid_field, float).ravel()
    feas = np.asarray(grid_feasible, bool).ravel()
    order = np.argsort(fld)                      # большие значения сверху
    sc = a1.scatter(g2[order, 0], g2[order, 1], c=fld[order], s=9,
                    cmap="viridis", alpha=0.85, linewidths=0)
    a1.scatter(g2[~feas, 0], g2[~feas, 1], s=14, facecolors="none",
               edgecolors="red", linewidths=0.35, alpha=0.55,
               label="истинная дыра (образец не получен)")
    _draw_star_frame(a1, names)
    a1.set_title(f"{field_label} по пулу кандидатов (красный контур — дыра)",
                 fontsize=10)
    cb = fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.02)
    cb.set_label(field_label)
    a1.legend(loc="lower left", fontsize=8, frameon=True)

    # ---------- правая панель: точки ------------------------------------
    a2.scatter(g2[~feas, 0], g2[~feas, 1], s=6, color="mistyrose",
               linewidths=0, label="дыра (по истине)", zorder=0)
    s2 = star_project(seed_Xc)
    sf = np.asarray(seed_feasible, bool).ravel()
    a2.scatter(s2[sf, 0], s2[sf, 1], s=26, color="0.35", marker="o",
               label=f"seed измерен ({int(sf.sum())})", zorder=2)
    a2.scatter(s2[~sf, 0], s2[~sf, 1], s=30, color="0.35", marker="x",
               label=f"seed — образца нет ({int((~sf).sum())})", zorder=2)
    K = max(len(round_Xc), 1)
    for k, (Xk, fk) in enumerate(zip(round_Xc, round_feasible)):
        Xk = np.atleast_2d(np.asarray(Xk, float))
        fk = np.asarray(fk, bool).ravel()
        if Xk.shape[0] == 0:
            continue
        p2 = star_project(Xk)
        col = _round_color(k, K)
        a2.scatter(p2[fk, 0], p2[fk, 1], s=70, color=col, marker="o",
                   edgecolors="k", linewidths=0.6, zorder=4,
                   label=f"раунд {k + 1}" if k in (0, K - 1) else None)
        a2.scatter(p2[~fk, 0], p2[~fk, 1], s=90, color=col, marker="X",
                   edgecolors="k", linewidths=0.6, zorder=4)
        for (px, py) in p2:
            a2.annotate(str(k + 1), (px, py), fontsize=6, ha="center",
                        va="center", zorder=5)
    if x_best is not None:
        b2 = star_project(np.atleast_2d(x_best))
        a2.scatter(b2[:, 0], b2[:, 1], s=260, marker="*", color="gold",
                   edgecolors="k", linewidths=0.8, zorder=6,
                   label="x_best (измеренный рекорд)")
    if x_opt is not None:
        o2 = star_project(np.atleast_2d(x_opt))
        a2.scatter(o2[:, 0], o2[:, 1], s=200, marker="P", color="deepskyblue",
                   edgecolors="k", linewidths=0.8, zorder=6,
                   label="аналитический оптимум (по истине)")
    _draw_star_frame(a2, names)
    a2.set_title("где ставили точки: seed → раунды (цвет = номер, X = промах)",
                 fontsize=10)
    # легенда ПОД осью: внутри круга она закрывала бы точки раундов
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
              fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


# ----------------------------------------------------------------------
# Срез вдоль расстояния до кромки
# ----------------------------------------------------------------------
def cliff_slice(*, dist_grid: np.ndarray, mu_grid: np.ndarray,
                sd_grid: np.ndarray, truth_grid: np.ndarray,
                dist_seed: np.ndarray, y_seed: np.ndarray,
                dist_rounds: Sequence[np.ndarray],
                y_rounds: Sequence[np.ndarray],
                y_cliff: float, response: str = "yield",
                gate_label: str = "surface − порог",
                title: str = "", path: str | Path = "cliff.png") -> Path:
    """Срез «отклик vs расстояние до кромки» (кромка — dist = 0).

    ``dist_* = g_true − threshold`` (отрицательное — за обрывом: истина 0,
    измерения нет). Пул кандидатов даёт μ±2σ суррогата как функцию dist
    (медианы по бинам), истина — кривая, точки — что реально мерили.
    ``y_rounds[k]`` может содержать NaN — промах в дыру, рисуется крестом
    на уровне 0.
    """
    dist_grid = np.asarray(dist_grid, float).ravel()
    mu = np.asarray(mu_grid, float).ravel()
    sd = np.asarray(sd_grid, float).ravel()
    tr = np.asarray(truth_grid, float).ravel()
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(title, fontsize=12)

    bins = np.linspace(dist_grid.min(), dist_grid.max(), 41)
    ctr, m_mu, m_sd, m_tr = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dist_grid >= lo) & (dist_grid < hi)
        if m.sum() < 3:
            continue
        ctr.append(0.5 * (lo + hi))
        m_mu.append(np.median(mu[m]))
        m_sd.append(np.median(sd[m]))
        m_tr.append(np.median(tr[m]))
    ctr, m_mu, m_sd, m_tr = map(np.asarray, (ctr, m_mu, m_sd, m_tr))
    ax.fill_between(ctr, m_mu - 2 * m_sd, m_mu + 2 * m_sd, color="tab:blue",
                    alpha=0.18, label="суррогат μ ± 2σ (медианы по бинам)")
    ax.plot(ctr, m_mu, color="tab:blue", lw=1.8, label="суррогат μ")
    ax.plot(ctr, m_tr, color="k", lw=1.6, ls="--",
            label="истина (медиана по бину)")
    ax.scatter(dist_grid, tr, s=3, color="0.75", alpha=0.4, linewidths=0,
               label="истина по кандидатам", zorder=0)
    ax.axvline(0.0, color="red", lw=1.5)
    ax.axvspan(dist_grid.min(), 0.0, color="red", alpha=0.06)
    ax.text(0.0, y_cliff * 1.02, "← за кромкой образца нет (MISSING) ",
            color="red", fontsize=9, ha="right", va="bottom")

    ds = np.asarray(dist_seed, float).ravel()
    ys = np.asarray(y_seed, float).ravel()
    ok = np.isfinite(ys)
    ax.scatter(ds[ok], ys[ok], s=22, color="0.35", label="seed измерен",
               zorder=3)
    ax.scatter(ds[~ok], np.zeros(int((~ok).sum())), s=26, color="0.35",
               marker="x", label="seed — образца нет", zorder=3)
    K = max(len(dist_rounds), 1)
    for k, (dk, yk) in enumerate(zip(dist_rounds, y_rounds)):
        dk = np.asarray(dk, float).ravel()
        yk = np.asarray(yk, float).ravel()
        okk = np.isfinite(yk)
        col = _round_color(k, K)
        ax.scatter(dk[okk], yk[okk], s=64, color=col, edgecolors="k",
                   linewidths=0.6, zorder=4,
                   label=f"раунд {k + 1}" if k in (0, K - 1) else None)
        ax.scatter(dk[~okk], np.zeros(int((~okk).sum())), s=90, color=col,
                   marker="X", edgecolors="k", linewidths=0.6, zorder=4)
        for x_, y_ in zip(dk, np.where(okk, yk, 0.0)):
            ax.annotate(str(k + 1), (x_, y_), fontsize=6, ha="center",
                        va="center", zorder=5)
    ax.set_xlabel(f"расстояние до кромки: {gate_label} (по истине)")
    ax.set_ylabel(response)
    ax.set_ylim(-0.05 * y_cliff, 1.35 * y_cliff)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


def round_metrics(dist_rounds: Sequence[np.ndarray]) -> Dict[str, List[float]]:
    """Сводка по раундам: доля промахов, глубина промаха, близость к кромке.

    ``dist = g_true − threshold``; промах — ``dist < 0``; ``miss_depth`` —
    максимум ``−dist`` среди промахов (0, если их нет): мелкий промах у
    кромки ≠ «чёрная дыра» в глубине; ``nearest_abs`` — минимум ``|dist|``
    раунда (как близко подошли к обрыву с любой стороны).
    """
    frac, depth, near = [], [], []
    for dk in dist_rounds:
        dk = np.asarray(dk, float).ravel()
        miss = dk < 0.0
        frac.append(float(miss.mean()) if dk.size else 0.0)
        depth.append(float((-dk[miss]).max()) if miss.any() else 0.0)
        near.append(float(np.abs(dk).min()) if dk.size else float("nan"))
    return {"miss_frac": frac, "miss_depth": depth, "nearest_abs": near}

