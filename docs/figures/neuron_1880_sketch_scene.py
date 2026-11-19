"""Excalidraw sketch of how neuron 1880 (layer 7) was rewired. Numbers from docs/12, docs/13, docs/19.

    PYTHONUTF8=1 python docs/figures/neuron_1880_sketch_scene.py
    -> docs/figures/neuron_1880_sketch.excalidraw (+ .svg preview)
"""
import os
import sys
from pathlib import Path

SKILL = os.environ.get("SKETCH_BUILDER_DIR")
sys.path.insert(0, SKILL)
from sketch_builder import Scene  # noqa: E402

BLUE, GREEN, PINK, YELLOW, PURPLE, ORANGE, TEAL = "#a5d8ff", "#b2f2bb", "#ffc9c9", "#ffec99", "#d0bfff", "#ffd8a8", "#96f2d7"
A_BLUE, A_GREEN, A_RED, A_ORANGE, A_PURPLE, A_TEAL, INK = "#1971c2", "#2f9e44", "#e03131", "#f08c00", "#9c36b5", "#099268", "#1e1e1e"

s = Scene(bg="#faf7f0", seed=11)
s.title("How I rewired neuron 1880", 40, 20, font_size=48, hl="#ffec99")
s.note("one worker inside a 125M legal QA model, taught to say “not stated in the context”",
       44, 95, color=INK, font_size=23)

# ---- the main path (centre column) ----
q = s.box("question + documents", 580, 150, 340, 58, fill=BLUE)
l16 = s.box("layers 1–6  (unchanged)", 580, 245, 340, 58, fill=PURPLE)

# layer-7 frame, drawn first so its contents sit on top
frame = s.box("", 495, 340, 510, 510, fill="transparent", stroke=A_ORANGE)
s.note("LAYER 7 · neuron 1880 of 3,072", 515, 350, color=A_ORANGE, font_size=20)

x = s.box("the model’s “thought”\n768 numbers", 600, 395, 300, 64, fill=BLUE, font_size=20)
eye1 = s.box("EYE 1 · gate row\ndoes the answer\nlook missing?", 518, 505, 215, 96, fill=BLUE, font_size=20)
eye2 = s.box("EYE 2 · up row\nis the reply about\nto start?", 767, 505, 215, 96, fill=GREEN, font_size=20)
andb = s.box("×  =  AND", 680, 645, 140, 58, fill=ORANGE, font_size=22)
hand = s.box("HAND · down column\nadd a nudge toward\n“not stated in the context”", 560, 745, 380, 88,
             fill=YELLOW, font_size=20)

l812 = s.box("layers 8–12  (unchanged)", 580, 895, 340, 58, fill=PURPLE)
out = s.box("“not stated in the context”", 560, 995, 380, 62, fill=PINK, font_size=21)

s.arrow(q, l16, color=A_BLUE)
s.arrow(l16, x, color=A_BLUE)
s.arrow(x, eye1, color=A_BLUE)
s.arrow(x, eye2, color=A_GREEN)
s.arrow(eye1, andb, color=A_BLUE)
s.arrow(eye2, andb, color=A_GREEN)
s.arrow(andb, hand, color=A_ORANGE)
s.arrow(hand, l812, color=A_ORANGE)
s.arrow(l812, out, color=A_RED)

# ---- left: steps 1, 2, 4 ----
s.note("STEP 1 · pick the worker nobody needs\n"
       "measured all 3,072 workers on ordinary text\n"
       "1880 contributed the least → fired from its old job",
       40, 330, color=A_PURPLE, font_size=21)
s.freearrow(455, 362, 505, 372, color=A_PURPLE)

s.note("STEP 2 · build eye 1 (gate)\n"
       "record the thought on 1000s of training questions\n"
       "label each: answer missing / answer present\n"
       "logistic regression finds the separating direction\n"
       "× sharpness S → a switch, not a dimmer",
       40, 455, color=A_BLUE, font_size=21)
s.freearrow(455, 540, 512, 552, color=A_BLUE)
s.note("⚠ it ends up mostly reading keyword overlap,\n   not real understanding (0.72 → 0.12)",
       40, 605, color=A_RED, font_size=21)

s.note("STEP 4 · build the hand (down)\n"
       "freeze the whole model\n"
       "optimise ONLY these 768 numbers (300 steps)\n"
       "goal: max chance of “not stated in the context”\n"
       "size capped, then ÷ typical firing strength",
       40, 745, color="#b07d00", font_size=21)
s.freearrow(470, 800, 553, 790, color=A_ORANGE)

# ---- right: step 3, the AND, safety ----
s.note("STEP 3 · build eye 2 (up)\n"
       "version 1 had no eye 2: it fired mid-text\n"
       "and made SEC text 1.72% worse\n"
       "ridge detector: ≈ 1 at the reply start,\n"
       "≈ 0 in documents, replies and web text",
       1045, 455, color=A_GREEN, font_size=21)
s.freearrow(1040, 540, 988, 552, color=A_GREEN)
s.note("→ forgetting fell to 0.011%", 1045, 600, color=A_GREEN, font_size=22)
s.highlight(1045, 632, 245, color="#b2f2bb")

s.note("reaction = squash(gate score) × up score\n"
       "if either is ≈ 0, the worker stays quiet\n"
       "fires only on: answer missing AND reply starting",
       1045, 665, color=A_ORANGE, font_size=21)
s.freearrow(1040, 690, 826, 676, color=A_ORANGE)

safety = s.box("SAFETY GATE\n≤ 25% change per weight matrix\n≤ 5% change to the whole model\n"
               "then held-out tests + the final 2,000-question exam",
               1045, 800, 460, 130, fill=TEAL, font_size=20)
s.arrow(hand, safety, color=A_TEAL, dashed=True)

# ---- footer ----
s.note("changed: 3 × 768 = 2,304 numbers out of 125.8 million (0.002%) · every other number identical",
       40, 1095, color=INK, font_size=23)
s.highlight(40, 1128, 880, color="#ffec99")

here = Path(__file__).parent
paths = s.save(str(here / "neuron_1880_sketch"))
svg = Path(paths[1])                      # builder leaves a raw "&" in the font URL: invalid XML
svg.write_text(svg.read_text(encoding="utf-8").replace("&display=", "&amp;display="), encoding="utf-8")
print(paths)
