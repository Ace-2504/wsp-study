"""Record the study page's playground (story-site) and turn it into a GIF for the README.

    python -m http.server 8766 --directory story-site     # in another terminal
    python docs/figures/make_study_playground_gif.py
    -> docs/figures/study_playground.gif
"""
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8766/"
OUT = Path(__file__).parent / "study_playground.gif"
W, H = 1180, 1000
# (question, context, seconds to hold after asking)
SEQ = [("copland", "out", 5.5), ("sherlock", "in", 5.0), ("sherlock", "out", 5.0), ("stryker", "out", 6.5)]

tmp = Path(tempfile.mkdtemp())
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    ctx = b.new_context(viewport={"width": W, "height": H}, color_scheme="light",
                        record_video_dir=str(tmp), record_video_size={"width": W, "height": H})
    t0 = time.time()
    pg = ctx.new_page()
    pg.goto(URL, wait_until="networkidle")
    pg.evaluate("document.fonts.ready")
    pg.add_style_tag(content=".snav{display:none!important}")
    pg.evaluate("""() => { const b = document.querySelector('#play .pg');
                     window.scrollTo(0, b.getBoundingClientRect().top + window.scrollY - 12); }""")
    pg.wait_for_timeout(1200)
    box = pg.evaluate("""() => { const a = document.querySelector('#play .pg').getBoundingClientRect();
                          const n = document.getElementById('pgsay').getBoundingClientRect();
                          return {x: a.left, y: a.top, w: a.width, h: n.bottom - a.top}; }""")
    start = time.time() - t0
    for q, c, hold in SEQ:
        pg.click(f'#qbtns button[data-q="{q}"]')
        pg.click(f'#cbtns button[data-c="{c}"]')
        pg.wait_for_timeout(500)
        pg.click("#pgask")
        pg.wait_for_timeout(int(hold * 1000))
    dur = time.time() - t0 - start
    vid = pg.video.path()
    ctx.close(); b.close()

x, y, w, h = (int(round(v)) for v in (box["x"], box["y"], box["w"], box["h"]))
w -= w % 2; h -= h % 2
crop = f"crop={w}:{min(h, H - y) - (min(h, H - y) % 2)}:{x}:{y}"
common = ["-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(vid)]
pal = tmp / "pal.png"
scale = "scale=760:-1:flags=lanczos"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *common, "-vf",
                f"{crop},fps=8,{scale},palettegen=max_colors=64:stats_mode=diff", str(pal)], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *common, "-i", str(pal), "-lavfi",
                f"{crop},fps=8,{scale}[v];[v][1:v]paletteuse=dither=none:diff_mode=rectangle",
                "-loop", "0", str(OUT)], check=True)
shutil.rmtree(tmp, ignore_errors=True)
print(OUT, round(OUT.stat().st_size / 1e6, 2), "MB", box)
