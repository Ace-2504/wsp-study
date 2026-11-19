"""Diagram of the edited neuron (docs/13): two detectors multiplied (AND), one write.

    python docs/figures/make_neuron_flow.py   -> docs/figures/neuron_1880_flow.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

BG, PANEL, PANEL2, LINE = "#0F1419", "#161D24", "#1C242C", "#2A3440"
INK, MUTED, CORAL, AMBER, GREEN, BLUE = "#E7ECF0", "#8E9CA9", "#FF8A65", "#F4B860", "#6FD39A", "#6CB4FF"
SANS = ["Segoe UI", "DejaVu Sans"]
MONO = ["Consolas", "DejaVu Sans Mono"]

fig = plt.figure(figsize=(7.2, 9.0), dpi=150, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 125); ax.axis("off")


def box(x, y, w, h, edge, fill=PANEL, lw=1.4, r=2.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fill, ec=edge, lw=lw))


def arrow(x1, y1, x2, y2, color=MUTED, rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def t(x, y, s, size=11, color=INK, weight="normal", ha="center", va="center", fam=SANS, **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, family=fam, **kw)


# title
t(7, 118.5, "How one neuron learned to say", 19, INK, "bold", ha="left")
t(7, 113.3, "“not stated in the context”", 19, CORAL, "bold", ha="left")
t(7, 108.6, "Inside a 125M legal QA model: worker 1880 in layer 7, rewired.", 10.5, MUTED, ha="left")

# 1. input
box(20, 96, 60, 7.5, LINE, PANEL2)
t(50, 101.0, "The model reads the question", 12, INK, "bold")
t(50, 98.1, "layers 1–6 · its “thought” = 768 numbers", 9.5, MUTED, fam=MONO)
arrow(50, 96, 50, 91.2)

# layer-7 container
box(6, 28.6, 88, 62.4, CORAL, BG, lw=1.2, r=3)
ax.add_patch(FancyBboxPatch((9, 88.6), 34, 4.6, boxstyle="round,pad=0,rounding_size=1.6", fc=CORAL, ec="none"))
t(26, 90.9, "LAYER 7 · WORKER 1880", 9.5, "#1A0D08", "bold", fam=MONO)

# eyes
for x0, col, tag, q, sub, note in (
        (10, BLUE, "EYE 1 · gate row", "Does the answer\nlook missing?",
         "768 numbers · fitted detector", "mostly reads keyword overlap"),
        (52, GREEN, "EYE 2 · up row", "Is the reply about\nto start?",
         "768 numbers · fitted detector", "quiet inside ordinary text")):
    box(x0, 64, 38, 21.5, col)
    t(x0 + 3, 82.4, tag, 9.5, col, "bold", ha="left", fam=MONO)
    t(x0 + 19, 75.6, q, 13, INK, "bold", linespacing=1.25)
    t(x0 + 19, 69.0, sub, 8.8, MUTED, fam=MONO)
    t(x0 + 19, 66.0, note, 8.8, col, style="italic")

# AND
ax.add_patch(Circle((50, 55.5), 5.0, fc=CORAL, ec="none"))
t(50, 55.7, "×", 22, "#1A0D08", "bold")
arrow(29, 64, 45.6, 57.6, BLUE, rad=0.15)
arrow(71, 64, 54.4, 57.6, GREEN, rad=-0.15)
t(60.5, 55.5, "multiply = AND", 10.5, CORAL, "bold", ha="left")
t(60.5, 51.7, "fires only if both say yes", 9.5, MUTED, ha="left")
arrow(50, 50.5, 50, 44.8, CORAL, lw=1.8)

# hand
box(14, 31.5, 72, 12.5, AMBER)
t(17, 41.2, "HAND · down column", 9.5, AMBER, "bold", ha="left", fam=MONO)
t(50, 37.3, "Adds a nudge toward “not stated in the context”", 12, INK, "bold")
t(50, 34.0, "768 numbers · found by a small optimisation, model frozen", 8.8, MUTED, fam=MONO)
arrow(50, 31.5, 50, 26.2)

# rest + output
box(20, 18.5, 60, 7.5, LINE, PANEL2)
t(50, 23.4, "Layers 8–12 carry it to the output", 12, INK, "bold")
t(50, 20.5, "unchanged", 9.5, MUTED, fam=MONO)
arrow(50, 18.5, 50, 14.4)
box(24, 7.5, 52, 6.8, CORAL, "#2A1A14")
t(50, 10.9, "“not stated in the context”", 13, CORAL, "bold", fam=MONO)

# footer
t(50, 3.4, "2,304 numbers changed out of 125.8 million (0.002%) · everything else untouched",
  8.8, MUTED)

out = Path(__file__).with_name("neuron_1880_flow.png")
fig.savefig(out, dpi=150, facecolor=BG)
print(out)
