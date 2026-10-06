"""
Simplified version of the Part 1 posterior chart -- just the single bell
curve for "how likely is each possible true win rate," dropping the
prior/MCMC-vs-closed-form overlay from the original (useful for showing
methodology, but busy if you just want to show the result).
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BREAKEVEN = 110 / 210
PRIOR_STRENGTH = 80
alpha_prior = BREAKEVEN * PRIOR_STRENGTH
beta_prior = (1 - BREAKEVEN) * PRIOR_STRENGTH
alpha_post = alpha_prior + 294
beta_post = beta_prior + (483 - 294)

x = np.linspace(0.45, 0.72, 500)
y = stats.beta.pdf(x, alpha_post, beta_post)
mean = alpha_post / (alpha_post + beta_post)

BLUE = "#1f6feb"
GRAY = "#6e7781"

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, color=BLUE, linewidth=2.5)
ax.fill_between(x, y, color=BLUE, alpha=0.15)

# Breakeven reference line
ax.axvline(BREAKEVEN, color=GRAY, linewidth=1.5, linestyle="--")
ax.text(BREAKEVEN, ax.get_ylim()[1] if False else max(y) * 1.05,
        f"Breakeven\n{BREAKEVEN:.1%}", color=GRAY, fontsize=9,
        ha="center", fontweight="bold")

# Expected value marker
y_at_mean = stats.beta.pdf(mean, alpha_post, beta_post)
ax.plot([mean, mean], [0, y_at_mean], color="#d97706", linewidth=1.5, linestyle=":")
ax.annotate(f"Expected win rate\n{mean:.1%}", (mean, y_at_mean),
            textcoords="offset points", xytext=(10, 10),
            fontsize=9.5, color="#d97706", fontweight="bold")

# 75% and 95% central credible intervals
lo75, hi75 = stats.beta.ppf([0.125, 0.875], alpha_post, beta_post)
lo95, hi95 = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

x95 = np.linspace(lo95, hi95, 300)
x75 = np.linspace(lo75, hi75, 300)
BAND_75_COLOR = "#1d4ed8"
BAND_95_COLOR = "#93c5fd"
ax.fill_between(x95, stats.beta.pdf(x95, alpha_post, beta_post),
                 color=BAND_95_COLOR, alpha=0.55, zorder=2)
ax.fill_between(x75, stats.beta.pdf(x75, alpha_post, beta_post),
                 color=BAND_75_COLOR, alpha=0.55, zorder=2)

for edge in (lo95, hi95):
    ax.plot([edge, edge], [0, stats.beta.pdf(edge, alpha_post, beta_post)],
            color=BAND_95_COLOR, linewidth=1.2)
for edge in (lo75, hi75):
    ax.plot([edge, edge], [0, stats.beta.pdf(edge, alpha_post, beta_post)],
            color=BAND_75_COLOR, linewidth=1.2)

legend_handles = [
    Patch(facecolor=BAND_75_COLOR, alpha=0.55,
          label=f"75% chance the true win rate is between {lo75:.1%} and {hi75:.1%}"),
    Patch(facecolor=BAND_95_COLOR, alpha=0.55,
          label=f"95% chance the true win rate is between {lo95:.1%} and {hi95:.1%}"),
]
ax.legend(handles=legend_handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=9.5)

ax.set_xlabel("True win rate")
ax.set_title("Likelihood of Win Rate")
ax.set_yticks([])
ax.set_xlim(0.45, 0.72)
ax.set_ylim(0, max(y) * 1.45)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig("win_rate_posterior_simple.png", dpi=150, bbox_inches="tight")
print("Saved win_rate_posterior_simple.png")
