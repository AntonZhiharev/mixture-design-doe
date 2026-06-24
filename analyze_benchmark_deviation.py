"""
analyze_benchmark_deviation.py — почему рецепты разошлись, хотя спеки выполнены.

Берём два рецепта из прогона run_pipeline_benchmark.py (аналитический оптимум и
решение pipeline) и разбираем КОЛИЧЕСТВЕННО:
  1) какие ограничения АКТИВНЫ в аналитическом оптимуме (slack каждого);
  2) насколько целевые свойства меняются ВДОЛЬ вектора отклонения Δ = x_pipe-x_opt
     (если ~0 — Δ лежит в «нейтральном к спекам» многообразии → equifinality);
  3) сколько цены реально завязано на этом Δ и на отпускании граничного компонента.
"""
import numpy as np
from src.core.synthetic import SyntheticScheffe

np.set_printoptions(precision=4, suppress=True)

Q = 5
PRICE = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
LOWER = np.array([0.05] * Q)
UPPER = np.array([0.5] * Q)
TRUE = {n: SyntheticScheffe(q=Q, model="quadratic", noise_sd=0.0, seed=s)
        for n, s in zip(["P1", "P2", "P3", "P4"], [11, 22, 33, 44])}

# рецепты и спеки — из прогона бенчмарка
x_opt = np.array([0.1534, 0.2259, 0.05, 0.3761, 0.1946])
x_pipe = np.array([0.1887, 0.1068, 0.2415, 0.3345, 0.1285])
SPEC = {  # (тип, значение)
    "P1": ("=", 7.050), "P2": ("≥", 7.060),
    "P3": ("≤", 7.054), "P4": ("=", 7.241),
}


def props(x):
    return {n: float(TRUE[n].true(x.reshape(1, -1))[0]) for n in TRUE}


def grad(f, x, eps=1e-5):
    g = np.zeros(Q)
    for i in range(Q):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps; xm[i] -= eps
        g[i] = (f.true(xp.reshape(1, -1))[0] - f.true(xm.reshape(1, -1))[0]) / (2 * eps)
    return g - g.mean()          # проекция на касательную симплекса (Σδ=0)


po, pp = props(x_opt), props(x_pipe)
delta = x_pipe - x_opt
u = delta / np.linalg.norm(delta)

# spread свойств по области (для нормировки)
S = np.random.default_rng(1).dirichlet(np.ones(Q), 4000)
S = LOWER + S * (1 - LOWER.sum())
spread = {n: float(np.std(TRUE[n].true(S))) for n in TRUE}

print("=" * 70)
print("1) АКТИВНОСТЬ ОГРАНИЧЕНИЙ в аналитическом оптимуме")
print("=" * 70)
for n, (t, v) in SPEC.items():
    if t == "=":
        print(f"  {n} = {v:.3f}: значение={po[n]:.3f}  |откл|={abs(po[n]-v):.3f}  -> АКТИВНО (равенство)")
    elif t == "≥":
        print(f"  {n} ≥ {v:.3f}: значение={po[n]:.3f}  slack=+{po[n]-v:.3f}  -> {'активно' if po[n]-v<0.02 else 'НЕ активно (запас)'}")
    else:
        print(f"  {n} ≤ {v:.3f}: значение={po[n]:.3f}  slack=+{v-po[n]:.3f}  -> {'активно' if v-po[n]<0.02 else 'НЕ активно (запас)'}")
print("  границы компонентов (lower=0.05):")
for i in range(Q):
    at_lo = abs(x_opt[i] - LOWER[i]) < 1e-3
    print(f"    c{i+1}={x_opt[i]:.3f}  {'<- на НИЖНЕЙ границе (АКТИВНА)' if at_lo else ''}")

print("\n" + "=" * 70)
print("2) ВЕКТОР ОТКЛОНЕНИЯ Δ = x_pipe - x_opt  и чувствительность свойств")
print("=" * 70)
for i in range(Q):
    print(f"  c{i+1}: {x_opt[i]:.3f} -> {x_pipe[i]:.3f}   Δ={delta[i]:+.3f}   цена/ед={PRICE[i]:.0f}")
print(f"  Σ|Δ| (без c3)        = {np.sum(np.abs(delta)) - abs(delta[2]):.3f}")
print(f"  |Δ c3|               = {abs(delta[2]):.3f}  (доминирует)")

print("\n  Чувствительность свойства вдоль Δ (как сильно свойство меняется,")
print("  если двигаться по направлению Δ), нормировано на spread свойства:")
for n in TRUE:
    g = grad(TRUE[n], x_opt)
    dP_lin = float(g @ delta)                 # линейный прогноз изменения вдоль Δ
    dP_act = pp[n] - po[n]                     # фактическое изменение
    rel = abs(dP_act) / spread[n]
    tag = "НЕЙТРАЛЬНО к спеку" if rel < 0.25 else "заметно меняется (в пределах запаса)"
    print(f"    {n}: ΔP_факт={dP_act:+.3f}  (|ΔP|/spread={rel:.2f})  {tag}")

print("\n" + "=" * 70)
print("3) ЦЕНА: откуда +4.5%")
print("=" * 70)
print(f"  Δцена вдоль Δ = PRICE·Δ = {PRICE @ delta:+.4f}  (= c_pipe - c_opt)")
print(f"  c_opt={PRICE@x_opt:.4f}   c_pipe={PRICE@x_pipe:.4f}")
# что даёт отпускание c3 с границы: верни c3 на 0.05, массу — в самый дешёвый c5
x_fix = x_pipe.copy()
freed = x_fix[2] - 0.05
x_fix[2] = 0.05
x_fix[4] += freed                      # в самый дешёвый компонент
print(f"\n  Если у pipeline-рецепта вернуть c3 на нижнюю границу 0.05,")
print(f"  а освободившуюся массу {freed:.3f} отдать самому дешёвому c5:")
print(f"    новая цена = {PRICE@x_fix:.4f}  (было {PRICE@x_pipe:.4f})")
pf = props(x_fix)
print(f"    свойства: " + ", ".join(f"{n}={pf[n]:.3f}" for n in TRUE))
print("  -> видно, держит ли спеки такой «дешёвый угол» (это и есть то,")
print("     что нашла аналитика, но soft-desirability не дожала).")
print("=" * 70)
