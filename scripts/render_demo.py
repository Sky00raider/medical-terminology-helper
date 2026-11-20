"""Run the real CLI on a few commands and draw the output as a terminal-style SVG for the README.
Run: python scripts/render_demo.py   (writes docs/demo.svg; needs models/)"""
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = [
    ["complete", "Inflammation of the", "-n", "3", "--words", "14", "--seed", "1"],
    ["define", "pericarditis", "-n", "2", "--seed", "3"],
    ["next", "The femur is the", "-k", "4"],
    ["roots", "gastroenteritis"],
]
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
lines = []
for args in COMMANDS:
    out = subprocess.run([sys.executable, "-m", "app.cli", *args], cwd=ROOT, capture_output=True, encoding="utf-8",
                         errors="replace", check=True, env=env).stdout
    quoted = " ".join(f'"{a}"' if " " in a else a for a in args)
    lines.append(("cmd", f"$ python -m app.cli {quoted}"))
    for row in out.rstrip("\n").splitlines():
        row = row if len(row) <= 112 else row[:109] + "..."
        lines.append(("opt" if row.lstrip()[:2] in ("1.", "2.", "3.") else "out", row))
    lines.append(("out", ""))
lines.pop()

COLORS = {"cmd": "#89b4fa", "opt": "#a6e3a1", "out": "#cdd6f4"}
CW, LH, PAD, TOP = 8.4, 19, 18, 44
width = int(PAD * 2 + CW * max(len(t) for _, t in lines)) + 10
height = TOP + LH * len(lines) + PAD
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       f'<rect width="{width}" height="{height}" rx="10" fill="#1e1e2e"/>',
       '<circle cx="20" cy="18" r="6" fill="#f38ba8"/><circle cx="40" cy="18" r="6" fill="#f9e2af"/>'
       '<circle cx="60" cy="18" r="6" fill="#a6e3a1"/>',
       '<g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="14" xml:space="preserve">']
for i, (kind, text) in enumerate(lines):
    svg.append(f'<text x="{PAD}" y="{TOP + i * LH}" fill="{COLORS[kind]}">{escape(text)}</text>')
svg += ["</g>", "</svg>"]
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "demo.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
print(f"docs/demo.svg: {len(lines)} lines, {width}x{height}")
