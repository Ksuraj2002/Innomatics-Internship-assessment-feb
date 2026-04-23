from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"

SOURCES = [
    (DOCS_DIR / "HLD.md", DOCS_DIR / "HLD.pdf"),
    (DOCS_DIR / "LLD.md", DOCS_DIR / "LLD.pdf"),
    (DOCS_DIR / "Technical_Documentation.md", DOCS_DIR / "Technical_Documentation.pdf"),
]


def _draw_wrapped_text(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, line_height: float) -> float:
    words = text.split(" ")
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(candidate, "Helvetica", 10) <= max_width:
            current = candidate
            continue

        c.drawString(x, y, current)
        y -= line_height
        current = word

    if current:
        c.drawString(x, y, current)
        y -= line_height

    return y


def markdown_to_pdf(src: Path, out: Path) -> None:
    c = canvas.Canvas(str(out), pagesize=A4)
    width, height = A4

    left = 2 * cm
    right = width - (2 * cm)
    top = height - (2 * cm)
    bottom = 2 * cm

    y = top
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, src.stem.replace("_", " "))
    y -= 1.0 * cm

    c.setFont("Helvetica", 10)
    content = src.read_text(encoding="utf-8").splitlines()

    for raw_line in content:
        line = raw_line.rstrip()

        if y < bottom:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = top

        if not line:
            y -= 0.5 * cm
            continue

        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            size = 13 if level == 1 else 11
            c.setFont("Helvetica-Bold", size)
            y = _draw_wrapped_text(c, title, left, y, right - left, 0.55 * cm)
            c.setFont("Helvetica", 10)
            continue

        cleaned = line.replace("```", "").replace("`", "")
        y = _draw_wrapped_text(c, cleaned, left, y, right - left, 0.5 * cm)

    c.save()


def main() -> None:
    for src, out in SOURCES:
        if not src.exists():
            print(f"Skipping missing source: {src}")
            continue
        markdown_to_pdf(src, out)
        print(f"Generated: {out}")


if __name__ == "__main__":
    main()
