"""
Posterior "at least this good" curve: for each possible true win rate x,
what's the probability the model's real long-run win rate is at least x?
This is just 1 - CDF of the same posterior from bayesian_nfl_signal_v3.py
(Beta(alpha_post, beta_post) from the 294W-189L verified NFL record).
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

BREAKEVEN = 110 / 210
PRIOR_STRENGTH = 80
alpha_prior = BREAKEVEN * PRIOR_STRENGTH
beta_prior = (1 - BREAKEVEN) * PRIOR_STRENGTH
alpha_post = alpha_prior + 294
beta_post = beta_prior + (483 - 294)

x = np.linspace(0.45, 0.72, 500)
p_at_least = 1 - stats.beta.cdf(x, alpha_post, beta_post)

BLUE = "#1f6feb"
GRAY = "#6e7781"
RED = "#cf222e"

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(x, p_at_least * 100, color=BLUE, linewidth=2.5)
ax.fill_between(x, p_at_least * 100, color=BLUE, alpha=0.08)

# Mark the points discussed
markers = [
    (BREAKEVEN, "Breakeven\n(52.4%)", GRAY),
    (0.597, "Expected value\n(59.7%)", "#d97706"),
    (0.60, "60%", RED),
]
offsets = [(12, -22), (10, 12), (10, -28)]
for (xm, label, color), off in zip(markers, offsets):
    ym = (1 - stats.beta.cdf(xm, alpha_post, beta_post)) * 100
    ax.plot([xm, xm], [0, ym], color=color, linewidth=1, linestyle=":")
    ax.plot([x[0], xm], [ym, ym], color=color, linewidth=1, linestyle=":")
    ax.scatter([xm], [ym], color=color, s=50, zorder=5)
    ax.annotate(f"{label}\n{ym:.1f}% likely", (xm, ym),
                textcoords="offset points", xytext=off,
                fontsize=8.5, color=color, fontweight="bold")

ax.set_xlabel("Win rate (x)")
ax.set_ylabel("Probability true win rate ≥ x")
ax.set_title("How likely is the model to be AT LEAST this good?\n(NFL model, 294W-189L verified record)", pad=14)
ax.set_ylim(0, 108)
ax.set_ylim(0, 102)
ax.set_xlim(0.45, 0.72)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("win_rate_certainty_curve.png", dpi=150)
print("Saved win_rate_certainty_curve.png")
for xm, label, _ in markers:
    ym = (1 - stats.beta.cdf(xm, alpha_post, beta_post)) * 100
    print(f"P(theta >= {xm:.3f}) = {ym:.2f}%")
