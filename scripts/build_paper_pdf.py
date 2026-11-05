"""
build_paper_pdf.py -- render docs/paper_draft.md to a PDF with figures.

Handles the markdown subset the draft actually uses: headings, bold,
italic, inline code, bullets, numbered lists, tables, block quotes, rules, and
figure anchors of the form

    ![FIG:filename.png|caption text]

Figures are read from results/paper_figures/.

    python scripts/build_paper_pdf.py [-o out.pdf]
"""
import argparse
import html
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (HRFlowable, Image, KeepTogether,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "paper_figures")
ACCENT = colors.HexColor("#1f4e79")
MUTED = colors.HexColor("#5a5a5a")
RULE = colors.HexColor("#cccccc")


def styles():
    ss = getSampleStyleSheet()
    def mk(name, **kw):
        base = dict(fontName="Times-Roman", fontSize=9.5, leading=13.2,
                    spaceAfter=5, alignment=TA_JUSTIFY)
        base.update(kw)
        return ParagraphStyle(name, parent=ss["Normal"], **base)
    return {
        "title": mk("title", fontName="Times-Bold", fontSize=17, leading=21,
                    spaceAfter=10, alignment=TA_LEFT, textColor=ACCENT),
        "h2": mk("h2", fontName="Times-Bold", fontSize=12.5, leading=15,
                 spaceBefore=13, spaceAfter=5, alignment=TA_LEFT,
                 textColor=ACCENT, keepWithNext=1),
        "h3": mk("h3", fontName="Times-Bold", fontSize=10.5, leading=13,
                 spaceBefore=9, spaceAfter=3, alignment=TA_LEFT, keepWithNext=1),
        "body": mk("body"),
        "bullet": mk("bullet", leftIndent=11, bulletIndent=2, spaceAfter=2.5),
        "quote": mk("quote", leftIndent=11, rightIndent=8, fontName="Times-Italic",
                    textColor=ACCENT, spaceBefore=5, spaceAfter=6),
        "caption": mk("caption", fontSize=8, leading=10.5, spaceBefore=4,
                      spaceAfter=11, textColor=MUTED),
        "cell": mk("cell", fontSize=8, leading=10.2, spaceAfter=0,
                   alignment=TA_LEFT),
        "cellh": mk("cellh", fontName="Times-Bold", fontSize=8, leading=10.2,
                    spaceAfter=0, alignment=TA_LEFT),
        "meta": mk("meta", fontName="Times-Italic", fontSize=8.5, leading=11,
                   textColor=MUTED, spaceAfter=9, alignment=TA_LEFT),
        "code": mk("code", fontName="Courier", fontSize=8.2, leading=11,
                   alignment=TA_LEFT, leftIndent=9, spaceBefore=4,
                   spaceAfter=7, textColor=colors.HexColor("#24292e")),
    }


def inline(text):
    """Markdown inline -> reportlab's mini-HTML.  Escape first, then mark up."""
    t = html.escape(text, quote=False)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)",
               r'<link href="\2" color="#1f4e79">\1</link>', t)
    t = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8.3">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)
    t = t.replace("--", "–")
    return t


def figure(name, caption, st, width):
    path = os.path.join(FIG_DIR, name)
    if not os.path.exists(path):
        return [Paragraph(f"[missing figure: {name}]", st["caption"])]
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize()
    w = width * (0.98 if iw / ih > 1.9 else 0.78)
    img = Image(path, width=w, height=w * ih / iw)
    n = figure.counter = getattr(figure, "counter", 0) + 1
    cap = Paragraph(f"<b>Figure {n}.</b> {inline(caption)}", st["caption"])
    return [KeepTogether([Spacer(1, 4), img, cap])]


def table(rows, st, width):
    head, body = rows[0], rows[1:]
    ncol = len(head)
    data = [[Paragraph(inline(c), st["cellh"]) for c in head]]
    data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in body]
    longest = [max(len(r[j]) if j < len(r) else 0 for r in rows) + 4 for j in range(ncol)]
    widths = [width * n / sum(longest) for n in longest]
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, ACCENT),
        ("LINEABOVE", (0, 0), (-1, 0), 0.7, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white,
                                              colors.HexColor("#f6f8fa")]),
    ]))
    return [Spacer(1, 3), t, Spacer(1, 8)]


SEP = re.compile(r"^\|[\s:|-]+\|$")


def is_table_start(lines, i):
    """A pipe line is a table only when the next line is a separator row.

    Prose can legitimately begin with a pipe -- `|margin|` written inline --
    and without this check that line matches neither the table branch nor the
    paragraph branch, so the cursor never advances.
    """
    if not lines[i].strip().startswith("|"):
        return False
    return i + 1 < len(lines) and bool(SEP.match(lines[i + 1].strip()))


def parse(md, st, width):
    flow = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            i += 1
            continue

        m = re.match(r"^!\[FIG:([^|]+)\|(.+)\]$", s)
        if m:
            flow += figure(m.group(1).strip(), m.group(2).strip(), st, width)
            i += 1
            continue

        if is_table_start(lines, i):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = lines[i].strip().strip("|")
                if not SEP.match(lines[i].strip()):
                    rows.append([c.strip() for c in r.split("|")])
                i += 1
            if rows:
                flow += table(rows, st, width)
            continue

        if s in ("---", "***", "___"):
            flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                   spaceBefore=7, spaceAfter=7))
            i += 1
            continue

        if s.startswith("# "):
            flow.append(Paragraph(inline(s[2:]), st["title"]))
            i += 1
            continue
        if s.startswith("## "):
            flow.append(Paragraph(inline(s[3:]), st["h2"]))
            i += 1
            continue
        if s.startswith("### "):
            flow.append(Paragraph(inline(s[4:]), st["h3"]))
            i += 1
            continue

        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(html.escape(lines[i], quote=False))
                i += 1
            i += 1          # closing fence
            if buf:
                flow.append(Paragraph("<br/>".join(buf).replace(" ", "&nbsp;"),
                                      st["code"]))
            continue

        if s.startswith("> "):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            flow.append(Paragraph(inline(" ".join(buf)), st["quote"]))
            continue

        if re.match(r"^[-*] ", s) or re.match(r"^\d+\. ", s):
            while i < len(lines):
                c = lines[i].strip()
                mb = re.match(r"^[-*] (.*)", c)
                mn = re.match(r"^(\d+)\. (.*)", c)
                if not (mb or mn):
                    break
                i += 1
                txt = mb.group(1) if mb else mn.group(2)
                bullet = "•" if mb else f"{mn.group(1)}."
                # continuation lines of the same item
                while i < len(lines) and lines[i].startswith(("   ", "\t")) \
                        and lines[i].strip() \
                        and not re.match(r"^[-*] |^\d+\. ",
                                         lines[i].strip()):
                    txt += " " + lines[i].strip()
                    i += 1
                flow.append(Paragraph(inline(txt), st["bullet"],
                                      bulletText=bullet))
            flow.append(Spacer(1, 4))
            continue

        start = i
        buf = [lines[i].strip()]
        i += 1          # always consume the current line: the cursor cannot stall
        while i < len(lines) and lines[i].strip() \
                and not is_table_start(lines, i) \
                and not re.match(r"^(#{1,3} |> |[-*] |\d+\. |---$|!\[FIG:|```)",
                                 lines[i].strip()):
            buf.append(lines[i].strip())
            i += 1
        para = " ".join(buf)
        if para.startswith("*") and para.endswith("*") and len(para) < 400:
            flow.append(Paragraph(inline(para.strip("*")), st["meta"]))
        else:
            flow.append(Paragraph(inline(para), st["body"]))
        assert i > start
    return flow


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 7.8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(A4[0] / 2, 12 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default=os.path.join(ROOT, "docs",
                                                          "paper_draft.md"))
    ap.add_argument("-o", "--output", default=os.path.join(ROOT, "docs",
                                                           "paper_draft.pdf"))
    a = ap.parse_args()

    md = open(a.input).read()
    st = styles()
    doc = SimpleDocTemplate(a.output, pagesize=A4,
                            leftMargin=22 * mm, rightMargin=22 * mm,
                            topMargin=20 * mm, bottomMargin=20 * mm,
                            title="Buy the Better Guard",
                            author="Md. Zariful Huq")
    width = doc.width
    doc.build(parse(md, st, width), onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {a.output}  ({os.path.getsize(a.output)/1024:.0f} KB, "
          f"{getattr(figure, 'counter', 0)} figures)")


if __name__ == "__main__":
    main()
