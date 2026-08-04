"""Render the markdown reports to PDF (python-markdown -> styled HTML -> Chrome print).

    python render_reports.py                    # every *_REPORT.md
    python render_reports.py TORQUE_REPORT.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CHROME = "/usr/bin/google-chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font: 10.5pt/1.5 "DejaVu Sans", "Helvetica Neue", Arial, sans-serif;
       color: #1a1a1a; max-width: 100%; }
h1 { font-size: 19pt; margin: 0 0 4pt; line-height: 1.25; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; padding-top: 2pt;
     border-top: 1px solid #d8d8d8; }
h3 { font-size: 11.5pt; margin: 13pt 0 4pt; color: #333; }
h1 + p, h2 + p { margin-top: 4pt; }
p, li { margin: 5pt 0; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 9pt;
       background: #f2f2f4; padding: 1px 3px; border-radius: 2px; }
pre { background: #f7f7f9; border: 1px solid #e2e2e6; border-radius: 3px;
      padding: 7pt 9pt; overflow-x: auto; }
pre code { background: none; padding: 0; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 9.5pt; }
th, td { border: 1px solid #d0d0d4; padding: 4pt 7pt; text-align: left;
         vertical-align: top; }
th { background: #f0f0f3; font-weight: 600; }
tr:nth-child(even) td { background: #fafafb; }
blockquote { margin: 8pt 0; padding: 6pt 11pt; border-left: 3px solid #b8b8c0;
             background: #f7f7f9; }
blockquote p { margin: 3pt 0; }
hr { border: none; border-top: 1px solid #d8d8d8; margin: 14pt 0; }
a { color: #1a4d8f; text-decoration: none; word-break: break-word; }
strong { font-weight: 600; }
/* keep tables and headings from splitting across pages */
table, blockquote, pre { page-break-inside: avoid; }
tr { page-break-inside: avoid; }
h1, h2, h3 { page-break-after: avoid; }
"""


def render(md_path: Path) -> Path:
    html_body = markdown.markdown(
        md_path.read_text(),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{md_path.stem}</title><style>{CSS}</style></head>"
            f"<body>{html_body}</body></html>")
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html)
    pdf_path = md_path.with_suffix(".pdf")
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", "--virtual-time-budget=3000",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=True, capture_output=True, timeout=120)
    html_path.unlink()
    return pdf_path


def main() -> None:
    names = sys.argv[1:] or sorted(p.name for p in HERE.glob("*_REPORT.md"))
    for name in names:
        pdf = render(HERE / name)
        print(f"{name} -> {pdf.name}  ({pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
