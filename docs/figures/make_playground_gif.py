"""Record the live playground and turn it into a GIF + MP4 for LinkedIn.

    python docs/figures/make_playground_gif.py
    -> docs/figures/playground.gif, docs/figures/playground.mp4
"""
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://harman-weight-programming.vercel.app/"
OUT = Path(__file__).parent
W, H = 1100, 1160
SEQ = [("ex-missing", 8.0), ("ex-present", 7.5), ("ex-tricky", 9.0)]   # (button, seconds to hold)

tmp = Path(tempfile.mkdtemp())
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    ctx = b.new_context(viewport={"width": W, "height": H}, color_scheme="dark",
                        record_video_dir=str(tmp), record_video_size={"width": W, "height": H})
    t0 = time.time()
    pg = ctx.new_page()
    pg.goto(URL, wait_until="networkidle")
    pg.evaluate("document.fonts.ready")
    # hide the sticky bar (it would cover the panel); reserve room for the longest caption and document
    pg.add_style_tag(content=".nav{display:none!important}.narrate{min-height:132px!important}.qdoc{min-height:74px!important}")
    pg.evaluate("""() => { const b = document.getElementById('pgbox');
                     window.scrollTo(0, b.getBoundingClientRect().top + window.scrollY - 16); }""")
    pg.wait_for_timeout(7000)            # let the auto-run on scroll finish, so the clip starts clean
    pg.click("#reset")
    pg.wait_for_timeout(600)
    box = pg.evaluate("""() => { const a = document.getElementById('pgbox').getBoundingClientRect();
                          const n = document.getElementById('narrate').getBoundingClientRect();
                          return {x: a.left, y: a.top, w: a.width, h: n.bottom - a.top}; }""")
    start = time.time() - t0
    for btn, hold in SEQ:
        pg.click("#" + btn)
        pg.wait_for_timeout(int(hold * 1000))
    dur = time.time() - t0 - start
    vid = pg.video.path()
    ctx.close(); b.close()

print(json.dumps({"crop": box, "start": round(start, 2), "dur": round(dur, 2)}))
x, y, w, h = (int(round(v)) for v in (box["x"], box["y"], box["w"], box["h"]))
w -= w % 2; h -= h % 2
crop = f"crop={w}:{h}:{x}:{y}"
src = str(vid)
mp4 = OUT / "playground.mp4"
gif = OUT / "playground.gif"
pal = tmp / "pal.png"
common = ["-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", src]
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *common, "-vf", f"{crop},fps=30",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4)], check=True)
scale = "scale=720:-1:flags=lanczos"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *common, "-vf",
                f"{crop},fps=8,{scale},palettegen=max_colors=64:stats_mode=diff", str(pal)], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *common, "-i", str(pal), "-lavfi",
                f"{crop},fps=8,{scale}[v];[v][1:v]paletteuse=dither=none:diff_mode=rectangle",
                "-loop", "0", str(gif)], check=True)
shutil.rmtree(tmp, ignore_errors=True)
for f in (gif, mp4):
    print(f, round(f.stat().st_size / 1e6, 2), "MB")
