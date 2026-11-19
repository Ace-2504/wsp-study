"""LinkedIn visual: the AND-gate edit's discrimination on normal held-out questions vs the
overlap-matched pool (answer sentence deleted, question words kept). Numbers from docs/14 Part B:
D on pool v2 = 0.720 [0.698, 0.742]; D matched = 0.122 [0.097, 0.150].

    python docs/figures/make_linkedin_overlap.py   -> docs/figures/linkedin_overlap_drop.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
vals = [0.720, 0.122]
cis = [(0.698, 0.742), (0.097, 0.150)]
cols = [BLUE, ORANGE]
labels = ["Normal questions", "Answer sentence deleted,\nquestion words kept"]

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
fig = plt.figure(figsize=(7.2, 9.0), dpi=150, facecolor=SURFACE)

fig.text(0.07, 0.935, "My one-neuron edit wasn't really reading.", fontsize=21, fontweight="bold", color=INK)
fig.text(0.07, 0.893, "It was mostly matching keywords.", fontsize=21, fontweight="bold", color=ORANGE)
fig.text(0.07, 0.835,
         "Separation score: how much more often the model says\n"
         "“not stated in the context” when the answer is missing than\n"
         "when it is there.  1 = perfect,  0 = can't tell them apart.",
         fontsize=11.5, color=INK2, linespacing=1.45, va="top")

ax = fig.add_axes([0.07, 0.17, 0.86, 0.55], facecolor=SURFACE)
x = [0.0, 1.0]
w = 0.52
for xi, v, c, (lo, hi) in zip(x, vals, cols, cis):
    ax.add_patch(FancyBboxPatch((xi - w / 2, 0), w, v, boxstyle="round,pad=0,rounding_size=0.012",
                                linewidth=0, facecolor=c, mutation_aspect=1 / 1.6))
    ax.plot([xi, xi], [lo, hi], color=INK, lw=1.2, solid_capstyle="butt")
    ax.plot([xi - 0.05, xi + 0.05], [lo, lo], color=INK, lw=1.2)
    ax.plot([xi - 0.05, xi + 0.05], [hi, hi], color=INK, lw=1.2)
    ax.text(xi, hi + 0.035, f"{v:.2f}", ha="center", va="bottom", fontsize=30, fontweight="bold", color=INK)

ax.annotate("", xy=(0.92, 0.30), xytext=(0.30, 0.70),
            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, connectionstyle="arc3,rad=-0.35"))
ax.text(1.02, 0.60, "only 17%\nsurvives", ha="left", va="center", fontsize=14, color=INK, fontweight="bold")

ax.set_xlim(-0.55, 1.55)
ax.set_ylim(0, 0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12.5, color=INK)
ax.tick_params(axis="x", length=0, pad=10)
ax.set_yticks([0, 0.25, 0.5, 0.75])
ax.tick_params(axis="y", labelsize=10, colors=INK2, length=0)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)
ax.axhline(0, color=INK2, lw=1)

fig.text(0.07, 0.065,
         "125M legal QA model · 1 neuron (3 weight vectors) edited · held-out questions, 95% CIs\n"
         "github.com/Ace-2504/wsp-study",
         fontsize=10, color=INK2, linespacing=1.5)

out = Path(__file__).with_name("linkedin_overlap_drop.png")
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(out)
