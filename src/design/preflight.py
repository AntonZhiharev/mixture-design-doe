"""design/preflight.py — iter32: preflight-диагностика плана ДО прогона.

Контекст (docs/DECODE_LAYER_PROPOSAL.md, шаг 1 поэтапного плана): вырожденный
план (тесный phr-бокс после проективного преобразования, слипшиеся компоненты,
дыры покрытия оси суммы группы) должен ловиться ДЁШЕВО и ДО измерений,
независимо от параметризации входа. Диагностика read-only: ничего не меняет и
ничего не блокирует (A0.6 — решение за пользователем); ``passed``/``failures``
— сигнал для UI и логов.

Проверки предложенного дизайна ``X`` (n × (q+d), составные координаты текущей
схемы) относительно reference-пула ``X_ref`` из ТОЙ ЖЕ области (та же политика
кандидатов, что в ``_phase_candidates``):

  1. **rank** — модельная матрица дизайна натягивает всё, что идентифицируемо
     в ЭТОЙ области: ``rank(F) == rank(F_ref)`` (ловит n<p и точную
     коллинеарность вроде «DL всегда равен SBM»);
  2. **cond** — обусловленность масштабированной модельной матрицы не хуже
     reference более чем в ``cond_factor`` раз (ловит тесный бокс);
  3. **VIF** — максимальный uncentered-VIF термов не хуже reference более чем
     в ``vif_factor`` раз (масштабная коллинеарность термов);
  4. **corr** — нет пары свободных координат со |corr| ≥ ``corr_max``
     (почти функциональная связь осей; пара называется по именам);
  5. **blind** — нет «слепого направления»: минимальная дисперсия дизайна в
     подпространстве допустимых вариаций (ортогонально Σx=1 и запертым осям)
     не меньше ``blind_ratio`` × reference; направление провала именуется;
  6. **coverage** — покрытие оси суммы каждой функциональной группы (iter31)
     не уже ``coverage_min`` × reference-диапазона;
  7. **pair-coverage** (iter37, скрин-аудит п.4) — покрытие 2D-сетки
     ОБЯЗАТЕЛЬНЫХ пар осей кампании (УФ×TiO₂, T×УФ, ΔT×Σ_ACR и т.п.):
     доля занятых дизайном ячеек ``pair_grid × pair_grid`` относительно
     занятых reference-пулом не меньше ``pair_coverage_min``. Ось пары —
     сумма координат (одиночная координата = список из одного индекса),
     поэтому проверяются и пары с групповой осью (ΔT×Σ_ACR).

ПОЧЕМУ ГЕЙТЫ ОТНОСИТЕЛЬНЫЕ (сверка с классикой). Абсолютные пороги
регрессионной диагностики из внешнего обсуждения (cond<30, VIF<5,
max|corr|<0.30) в ДОЛЯХ Шеффе-квадратик неприменимы: эмпирика на q=6
(recheck3-сценарий iter31) даёт для ХОРОШЕГО плана n=24 cond≈1200–2000,
maxVIF≈4e4–1.4e5, max|corr|≈0.9 — Σx=1 наводит структурную коллинеарность,
узкая область (L/U-бокс) — масштабную. Поэтому null-модель гейтов —
равномерный reference-пул из той же области: «план не хуже, чем хороший
случайный план ЭТОЙ области». Исключение — corr-гейт: |corr| ≥ 0.98 означает
почти функциональную связь пары осей в любой области (абсолютный порог).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.schema import ProjectSchema
from .block_model import ModelTerms, build_model_terms, model_matrix

_EPS = 1e-12


# ----------------------------------------------------------------------
# Пороги гейтов (относительные к reference, кроме corr_max)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PreflightThresholds:
    """Пороги preflight-гейтов. Дефолты откалиброваны эмпирически (см. модуль).

    ``cond_factor``/``vif_factor``/``blind_ratio`` — множители к reference
    (хороший план n≈p+50% держится в ~5×/~20×/~0.8 от reference; вырожденный
    тесный бокс даёт ~200×/~1e4×/~0.02 — зазор на порядок в обе стороны).
    ``corr_max`` — абсолютный порог слипшейся пары осей.
    ``coverage_min`` — минимальная доля reference-диапазона суммы группы.
    ``pair_coverage_min``/``pair_grid`` (iter37) — гейт покрытия обязательных
    2D-пар: доля ячеек ``pair_grid²``-сетки пары, занятых дизайном, от занятых
    reference-пулом.
    """
    cond_factor: float = 20.0
    vif_factor: float = 100.0
    corr_max: float = 0.98
    blind_ratio: float = 0.10
    coverage_min: float = 0.80
    pair_coverage_min: float = 0.60
    pair_grid: int = 3


@dataclass
class GroupCoverage:
    """Покрытие оси суммы одной функциональной группы (iter31)."""
    names: List[str]
    lo: float
    hi: float
    ref_lo: float
    ref_hi: float
    coverage: float
    ok: bool


@dataclass
class PairCoverage:
    """Покрытие 2D-сетки одной обязательной пары осей (iter37, п.4).

    Ось = сумма координат ``names_*``; ``occupied``/``occupied_ref`` — число
    занятых ячеек сетки дизайном/reference; ``coverage`` — доля занятых
    дизайном среди занятых reference (учитываются только reference-ячейки:
    вне них область пары пуста и «непокрытость» не вменяется плану)."""
    names_a: List[str]
    names_b: List[str]
    occupied: int
    occupied_ref: int
    coverage: float
    ok: bool


@dataclass
class PreflightReport:
    """Итог preflight-диагностики плана (все числа — для интроспекции)."""
    n: int
    p: int
    rank: int
    rank_ref: int
    rank_ok: bool
    cond: float
    cond_ref: float
    cond_ok: bool
    vif_max: float
    vif_ref_max: float
    vif_term: str
    vif_ok: bool
    corr_max_abs: float
    corr_pair: Optional[Tuple[str, str]]
    corr_ok: bool
    eig_min: float
    eig_min_ref: float
    blind_ok: bool
    blind_direction: Optional[Dict[str, float]]
    group_coverage: List[GroupCoverage] = field(default_factory=list)
    coverage_ok: bool = True
    pair_coverage: List[PairCoverage] = field(default_factory=list)
    pair_ok: bool = True
    thresholds: PreflightThresholds = field(default_factory=PreflightThresholds)

    @property
    def passed(self) -> bool:
        return (self.rank_ok and self.cond_ok and self.vif_ok
                and self.corr_ok and self.blind_ok and self.coverage_ok
                and self.pair_ok)

    @property
    def failures(self) -> List[str]:
        """Человекочитаемые причины провала (пусто, если план прошёл)."""
        t = self.thresholds
        out: List[str] = []
        if not self.rank_ok:
            out.append(f"rank {self.rank} < {self.rank_ref} (идентифицируемо "
                       f"в области): часть термов модели не разрешима")
        if not self.cond_ok:
            out.append(f"cond {self.cond:.0f} > {t.cond_factor:g}×reference "
                       f"({self.cond_ref:.0f}): план плохо обусловлен")
        if not self.vif_ok:
            out.append(f"VIF {self.vif_max:.0f} (терм {self.vif_term}) > "
                       f"{t.vif_factor:g}×reference ({self.vif_ref_max:.0f})")
        if not self.corr_ok and self.corr_pair is not None:
            out.append(f"|corr({self.corr_pair[0]}, {self.corr_pair[1]})| = "
                       f"{self.corr_max_abs:.2f} ≥ {t.corr_max:g}: оси слиплись")
        if not self.blind_ok:
            d = ""
            if self.blind_direction:
                d = " вдоль " + " ".join(
                    f"{v:+.2f}·{k}" for k, v in self.blind_direction.items())
            out.append(f"слепое направление{d}: дисперсия плана "
                       f"{self.eig_min:.2e} < {t.blind_ratio:g}×reference "
                       f"({self.eig_min_ref:.2e})")
        for g in self.group_coverage:
            if not g.ok:
                out.append(f"покрытие Σ({', '.join(g.names)}) = "
                           f"{g.coverage:.0%} < {t.coverage_min:.0%}: план "
                           f"видит [{g.lo:.3f}, {g.hi:.3f}] из "
                           f"[{g.ref_lo:.3f}, {g.ref_hi:.3f}]")
        for pc in self.pair_coverage:
            if not pc.ok:
                out.append(f"покрытие пары Σ({', '.join(pc.names_a)}) × "
                           f"Σ({', '.join(pc.names_b)}) = {pc.coverage:.0%} "
                           f"< {t.pair_coverage_min:.0%}: занято "
                           f"{pc.occupied} из {pc.occupied_ref} ячеек "
                           f"{t.pair_grid}×{t.pair_grid}")
        return out

    def rows(self) -> List[Dict[str, Any]]:
        """Построчная таблица проверок для показа (чистая, без UI-зависимостей)."""
        t = self.thresholds
        rows = [
            {"Проверка": "rank (идентифицируемость)",
             "План": f"{self.rank}", "Допуск": f"= {self.rank_ref}",
             "ОК": self.rank_ok},
            {"Проверка": "cond (обусловленность)",
             "План": f"{self.cond:.0f}",
             "Допуск": f"≤ {t.cond_factor:g}×{self.cond_ref:.0f}",
             "ОК": self.cond_ok},
            {"Проверка": f"VIF max ({self.vif_term})",
             "План": f"{self.vif_max:.0f}",
             "Допуск": f"≤ {t.vif_factor:g}×{self.vif_ref_max:.0f}",
             "ОК": self.vif_ok},
            {"Проверка": ("|corr| пары "
                          + (f"({self.corr_pair[0]}, {self.corr_pair[1]})"
                             if self.corr_pair else "—")),
             "План": f"{self.corr_max_abs:.2f}",
             "Допуск": f"< {t.corr_max:g}", "ОК": self.corr_ok},
            {"Проверка": "blind (мин. дисперсия направления)",
             "План": f"{self.eig_min:.2e}",
             "Допуск": f"≥ {t.blind_ratio:g}×{self.eig_min_ref:.2e}",
             "ОК": self.blind_ok},
        ]
        for g in self.group_coverage:
            rows.append({"Проверка": f"покрытие Σ({', '.join(g.names)})",
                         "План": f"{g.coverage:.0%}",
                         "Допуск": f"≥ {t.coverage_min:.0%}", "ОК": g.ok})
        for pc in self.pair_coverage:
            rows.append({"Проверка": (f"пара Σ({', '.join(pc.names_a)}) × "
                                      f"Σ({', '.join(pc.names_b)})"),
                         "План": f"{pc.coverage:.0%}",
                         "Допуск": f"≥ {t.pair_coverage_min:.0%}", "ОК": pc.ok})
        return rows

    def summary(self) -> Dict[str, Any]:
        """Компактная сводка для логов/интроспекции."""
        return {
            "n": self.n, "p": self.p, "passed": self.passed,
            "rank": self.rank, "rank_ref": self.rank_ref,
            "cond": self.cond, "cond_ref": self.cond_ref,
            "vif_max": self.vif_max, "vif_ref_max": self.vif_ref_max,
            "corr_max_abs": self.corr_max_abs,
            "eig_min": self.eig_min, "eig_min_ref": self.eig_min_ref,
            "group_coverage": [g.coverage for g in self.group_coverage],
            "pair_coverage": [pc.coverage for pc in self.pair_coverage],
            "failures": self.failures,
        }


# ----------------------------------------------------------------------
# Числовые примитивы (чистые)
# ----------------------------------------------------------------------
def _scaled_svd_metrics(F: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """(rank, cond, VIF-вектор) масштабированной модельной матрицы.

    Столбцы нормируются на L2 (нулевые остаются нулём — их ловит rank);
    ``cond`` — отношение первого сингулярного числа к последнему ЗНАЧИМОМУ
    (числовой ранг с относительным допуском 1e-9); VIF — uncentered, через
    ``diag(pinv(FᵀF))`` (устойчив к вырождению; стандарт для моделей без
    intercept — Шеффе, §13.3).
    """
    F = np.atleast_2d(np.asarray(F, float))
    if F.size == 0:
        return 0, float("inf"), np.empty(0)
    norms = np.linalg.norm(F, axis=0)
    Fs = F / np.where(norms > _EPS, norms, 1.0)
    sv = np.linalg.svd(Fs, compute_uv=False)
    if sv.size == 0 or sv[0] <= _EPS:
        return 0, float("inf"), np.full(F.shape[1], np.inf)
    rank = int((sv > sv[0] * 1e-9).sum())
    cond = float(sv[0] / sv[rank - 1]) if rank > 0 else float("inf")
    vif = np.diag(np.linalg.pinv(Fs.T @ Fs))
    return rank, cond, np.asarray(vif, float)


def _max_abs_corr(X: np.ndarray, names: Sequence[str]
                  ) -> Tuple[float, Optional[Tuple[str, str]]]:
    """Максимум |corr| по парам координат с ненулевой дисперсией + имена пары."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] < 2:
        return 0.0, None
    stds = X.std(axis=0)
    idx = np.where(stds > _EPS)[0]
    if idx.size < 2:
        return 0.0, None
    C = np.corrcoef(X[:, idx], rowvar=False)
    np.fill_diagonal(C, 0.0)
    k = np.unravel_index(int(np.abs(C).argmax()), C.shape)
    return (float(abs(C[k])),
            (str(names[idx[k[0]]]), str(names[idx[k[1]]])))


def _variation_basis(dim: int, q: int, locked: Sequence[int]) -> np.ndarray:
    """Ортонормальный базис допустимых вариаций дизайна (dim × k).

    Дополнение к span{вектор единиц на mixture-осях (Σx=1), орты запертых
    координат}: вдоль этих направлений вариация НЕВОЗМОЖНА в области, и они
    не считаются «слепыми».
    """
    rows: List[np.ndarray] = []
    if q > 0:
        v = np.zeros(dim)
        v[:q] = 1.0
        rows.append(v)
    for j in locked:
        e = np.zeros(dim)
        e[int(j)] = 1.0
        rows.append(e)
    if not rows:
        return np.eye(dim)
    A = np.vstack(rows)
    _, s, Vt = np.linalg.svd(A)
    r = int((s > _EPS).sum())
    return Vt[r:].T


def _min_variance_direction(X: np.ndarray, B: np.ndarray
                            ) -> Tuple[float, Optional[np.ndarray]]:
    """(мин. собственное значение ковариации в базисе B, направление в исходных
    координатах). Меньше двух точек → (0, None): план заведомо слеп."""
    X = np.atleast_2d(np.asarray(X, float))
    if B.shape[1] == 0:
        return float("inf"), None            # вариаций нет вовсе — нечему слепнуть
    if X.shape[0] < 2:
        return 0.0, None
    Z = X @ B
    S = np.cov(Z, rowvar=False)
    S = np.atleast_2d(S)
    w, V = np.linalg.eigh(S)
    return float(w[0]), np.asarray(B @ V[:, 0], float)


# ----------------------------------------------------------------------
# Главная функция
# ----------------------------------------------------------------------
def preflight_design(schema: ProjectSchema, X: np.ndarray, X_ref: np.ndarray, *,
                     groups: Optional[Sequence[Sequence[int]]] = None,
                     pairs: Optional[Sequence[Tuple[Sequence[int],
                                                    Sequence[int]]]] = None,
                     terms: Optional[ModelTerms] = None,
                     thresholds: Optional[PreflightThresholds] = None
                     ) -> PreflightReport:
    """Preflight-диагностика дизайна ``X`` относительно reference-пула ``X_ref``.

    ``X``/``X_ref`` — составные координаты текущей схемы (n × (q+d));
    ``X_ref`` — «хороший случайный план этой области» (null-модель гейтов),
    в runner его строит та же политика кандидатов, что для seed/веток.
    ``groups`` — функциональные группы ИНДЕКСОВ mixture-координат текущей
    схемы (iter31) для проверки покрытия оси суммы; ``None`` — без проверки.
    ``pairs`` (iter37, п.4) — обязательные 2D-пары осей: элемент —
    ``(indices_a, indices_b)``, ось = сумма координат по индексам; ``None`` —
    без проверки пар.
    Чистая и read-only; исключения — только на несогласованные размерности.
    """
    thr = thresholds or PreflightThresholds()
    X = np.atleast_2d(np.asarray(X, float))
    X_ref = np.atleast_2d(np.asarray(X_ref, float))
    if X.shape[1] != X_ref.shape[1]:
        raise ValueError(f"X и X_ref разной размерности: "
                         f"{X.shape[1]} ≠ {X_ref.shape[1]}.")
    mt = terms if terms is not None else build_model_terms(schema)
    q = int(schema.n_mixture)
    dim = X.shape[1]
    coord_names = list(schema.mixture_names) + list(schema.process_names)
    if len(coord_names) != dim:
        raise ValueError(f"Дизайн имеет {dim} координат, схема — "
                         f"{len(coord_names)}.")

    # --- запертые координаты (lower==upper mixture-компонентов) -----------
    locked: List[int] = []
    mb = schema.mixture_block()
    if mb is not None:
        lo = np.asarray(mb.lower, float)
        hi = np.asarray(mb.upper, float)
        locked = [i for i in range(q) if (hi[i] - lo[i]) < 1e-12]

    # --- 1/2/3: rank / cond / VIF модельной матрицы (терм-базис p_eff) ----
    F_ref = model_matrix(schema, X_ref, terms=mt)
    F = model_matrix(schema, X, terms=mt)
    keep = np.linalg.norm(F_ref, axis=0) > _EPS   # термы, живые в области
    names_kept = [nm for nm, k in zip(mt.names, keep) if k]
    rank_ref, cond_ref, vif_ref = _scaled_svd_metrics(F_ref[:, keep])
    rank, cond, vif = _scaled_svd_metrics(F[:, keep])
    rank_ok = rank >= rank_ref
    cond_ok = bool(cond <= thr.cond_factor * cond_ref)
    vif_max = float(vif.max()) if vif.size else 0.0
    vif_ref_max = float(vif_ref.max()) if vif_ref.size else 0.0
    vif_term = (names_kept[int(vif.argmax())] if vif.size else "—")
    vif_ok = bool(vif_max <= thr.vif_factor * max(vif_ref_max, 1.0))

    # --- 4: слипшиеся пары координат --------------------------------------
    corr_max_abs, corr_pair = _max_abs_corr(X, coord_names)
    corr_ok = bool(corr_max_abs < thr.corr_max)

    # --- 5: слепое направление (в подпространстве допустимых вариаций) ----
    B = _variation_basis(dim, q, locked)
    eig_min, w_dir = _min_variance_direction(X, B)
    eig_min_ref, _ = _min_variance_direction(X_ref, B)
    if eig_min_ref <= _EPS:
        blind_ok = True                       # область сама не варьирует — не слепота плана
    else:
        blind_ok = bool(eig_min >= thr.blind_ratio * eig_min_ref)
    blind_direction: Optional[Dict[str, float]] = None
    if not blind_ok and w_dir is not None:
        blind_direction = {coord_names[i]: round(float(w_dir[i]), 3)
                           for i in range(dim) if abs(w_dir[i]) > 0.05}

    # --- 6: покрытие оси суммы функциональных групп (iter31) --------------
    gcov: List[GroupCoverage] = []
    for g in (groups or []):
        idx = [int(i) for i in g]
        s = X[:, idx].sum(axis=1)
        sr = X_ref[:, idx].sum(axis=1)
        span = float(s.max() - s.min()) if len(s) else 0.0
        ref_span = float(sr.max() - sr.min()) if len(sr) else 0.0
        coverage = (span / ref_span) if ref_span > _EPS else 1.0
        gcov.append(GroupCoverage(
            names=[coord_names[i] for i in idx],
            lo=float(s.min()) if len(s) else 0.0,
            hi=float(s.max()) if len(s) else 0.0,
            ref_lo=float(sr.min()) if len(sr) else 0.0,
            ref_hi=float(sr.max()) if len(sr) else 0.0,
            coverage=float(coverage),
            ok=bool(coverage >= thr.coverage_min)))
    coverage_ok = all(g.ok for g in gcov)

    # --- 7: покрытие 2D-сетки обязательных пар осей (iter37, п.4) ----------
    pcov: List[PairCoverage] = []
    for ia, ib in (pairs or []):
        ia = [int(i) for i in ia]
        ib = [int(i) for i in ib]
        pcov.append(_pair_coverage(
            X, X_ref, ia, ib,
            names_a=[coord_names[i] for i in ia],
            names_b=[coord_names[i] for i in ib],
            grid=int(thr.pair_grid), min_cov=float(thr.pair_coverage_min)))
    pair_ok = all(pc.ok for pc in pcov)

    return PreflightReport(
        n=int(X.shape[0]), p=int(mt.p),
        rank=rank, rank_ref=rank_ref, rank_ok=bool(rank_ok),
        cond=cond, cond_ref=cond_ref, cond_ok=cond_ok,
        vif_max=vif_max, vif_ref_max=vif_ref_max, vif_term=vif_term,
        vif_ok=vif_ok,
        corr_max_abs=corr_max_abs, corr_pair=corr_pair, corr_ok=corr_ok,
        eig_min=float(eig_min), eig_min_ref=float(eig_min_ref),
        blind_ok=blind_ok, blind_direction=blind_direction,
        group_coverage=gcov, coverage_ok=coverage_ok,
        pair_coverage=pcov, pair_ok=pair_ok, thresholds=thr)


def _pair_coverage(X: np.ndarray, X_ref: np.ndarray,
                   ia: Sequence[int], ib: Sequence[int], *,
                   names_a: List[str], names_b: List[str],
                   grid: int, min_cov: float) -> PairCoverage:
    """Покрытие 2D-сетки пары осей (iter37, п.4) — чистый примитив.

    Границы сетки — по reference-пулу (он определяет достижимую область
    пары, включая непрямоугольные формы вроде UV-трапеции); ячейка «занята»,
    если в неё попала хоть одна точка. Coverage = |ячейки(X) ∩ ячейки(ref)| /
    |ячейки(ref)|: план не штрафуется за ячейки, пустые и у reference
    (вне области пары). Вырожденная ось reference (нулевой размах) → пара
    не проверяема, coverage = 1.0 (не слепота плана — область не варьирует).
    """
    def _axis(M: np.ndarray, idx: Sequence[int]) -> np.ndarray:
        return M[:, list(idx)].sum(axis=1)

    u, v = _axis(X, ia), _axis(X, ib)
    ur, vr = _axis(X_ref, ia), _axis(X_ref, ib)
    span_u = float(ur.max() - ur.min())
    span_v = float(vr.max() - vr.min())
    if span_u <= _EPS or span_v <= _EPS:
        return PairCoverage(names_a=list(names_a), names_b=list(names_b),
                            occupied=0, occupied_ref=0, coverage=1.0, ok=True)

    def _cells(a: np.ndarray, b: np.ndarray) -> set:
        ka = np.clip(((a - ur.min()) / span_u * grid).astype(int), 0, grid - 1)
        kb = np.clip(((b - vr.min()) / span_v * grid).astype(int), 0, grid - 1)
        return set(zip(ka.tolist(), kb.tolist()))

    ref_cells = _cells(ur, vr)
    des_cells = _cells(u, v) & ref_cells
    cov = len(des_cells) / max(len(ref_cells), 1)
    return PairCoverage(names_a=list(names_a), names_b=list(names_b),
                        occupied=len(des_cells), occupied_ref=len(ref_cells),
                        coverage=float(cov), ok=bool(cov >= min_cov))
