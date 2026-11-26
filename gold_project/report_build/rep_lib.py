"""Helpers used to write the final report into the course template (.docx).
Every paragraph is cloned from a paragraph of the template, so the fonts / spacing / bullets stay the same."""
import copy
import json
import re

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph
from PIL import Image

import os

# everything is found relative to this file, so the package folder can be moved anywhere
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # Simple_Report_Package
PROJ = os.path.join(BASE, "gold_project")
TEMPLATE = os.path.join(BASE, "Final_project_template.docx")
FONT = "Times New Roman"


def n0(x):
    return f"{x:,.0f}"


def n1(x):
    return f"{x:,.1f}"


def n2(x):
    return f"{x:,.2f}"


def n3(x):
    return f"{x:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def sgn(x, d=2):
    """number with a real minus sign"""
    return f"{x:.{d}f}"


def set_font(run, size=None, bold=None, italic=None, code=False):
    name = "Courier New" if code else FONT
    rpr = run._r.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), name)
    if size:
        run.font.size = Pt(size)
    elif code:
        run.font.size = Pt(10)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def add_runs(par, text, size=None, bold=None, italic=None):
    """tiny markup:  **bold**   `code`"""
    for tok in re.split(r"(\*\*.+?\*\*|`.+?`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            r = par.add_run(tok[2:-2])
            set_font(r, size, True, italic)
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1])
            set_font(r, size, bold, italic, code=True)
        else:
            r = par.add_run(tok)
            set_font(r, size, bold, italic)


def tidy_spacing(p, before=0, after=6, line=None):
    """replace the template's 'auto spacing' by explicit spacing (used for captions / table cells)"""
    ppr = p._p.get_or_add_pPr()
    sp = ppr.find(qn("w:spacing"))
    if sp is None:
        sp = OxmlElement("w:spacing")
        ppr.append(sp)
    for a in ("w:beforeAutospacing", "w:afterAutospacing"):
        if sp.get(qn(a)) is not None:
            del sp.attrib[qn(a)]
    sp.set(qn("w:before"), str(int(before * 20)))
    sp.set(qn("w:after"), str(int(after * 20)))
    if line:
        sp.set(qn("w:line"), str(int(line * 240)))


class Ctx:
    def __init__(self):
        self.doc = Document(TEMPLATE)
        self.body = self.doc.element.body
        self.sect = self.body.find(qn("w:sectPr"))
        P = self.doc.paragraphs
        self.proto_body = copy.deepcopy(P[15]._p)
        self.proto_bullet = copy.deepcopy(P[18]._p)
        self.proto_indent = copy.deepcopy(P[19]._p)
        self.headings = {}
        for i, key in [(14, "I"), (16, "II"), (23, "III"), (31, "IV"), (37, "V"), (42, "VI"), (47, "VII"), (49, "VIII")]:
            h = copy.deepcopy(P[i]._p)
            for br in list(h.iter(qn("w:lastRenderedPageBreak"))):
                br.getparent().remove(br)
            self.headings[key] = h
        # remove everything after the title block (the first 14 paragraphs are kept)
        keep = {id(P[i]._p) for i in range(14)}
        for el in list(self.body):
            if el is self.sect or id(el) in keep:
                continue
            self.body.remove(el)
        self.R = json.load(open(PROJ + r"\results.json"))
        self.df = pd.read_csv(PROJ + r"\gold_clean.csv", parse_dates=["Date"], index_col="Date")
        self.fig_order = []
        self.fig_added = []
        self.app_added = 0
        self.tab_n = 0

    def plan_figures(self, keys):
        self.fig_order = list(keys)

    def fno(self, key):
        return self.fig_order.index(key) + 1

    def _append(self, el):
        self.sect.addprevious(el)

    def _make(self, proto, text="", **kw):
        el = copy.deepcopy(proto)
        for r in list(el.findall(qn("w:r"))):
            el.remove(r)
        p = Paragraph(el, self.doc._body)
        self._append(el)
        if text:
            add_runs(p, text, **kw)
        return p

    def heading(self, key):
        el = copy.deepcopy(self.headings[key])
        self._append(el)
        keep_next(Paragraph(el, self.doc._body))

    def para(self, text, align="justify", **kw):
        p = self._make(self.proto_body, text, **kw)
        if align == "justify":
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return p

    def bullet(self, lead, text=""):
        p = self._make(self.proto_bullet)
        r = p.add_run(lead)
        set_font(r, bold=True)
        if text:
            add_runs(p, " " + text)
        return p

    def sub(self, text):
        return self._make(self.proto_indent, text)

    def caption(self, text, above=False):
        p = self._make(self.proto_body, text, size=10, italic=True)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        tidy_spacing(p, before=10 if above else 2, after=3 if above else 10)
        if above:
            keep_next(p)
        return p

    def figure(self, key, caption, appendix=False, max_w=6.3, max_h=6.2):
        if appendix:
            self.app_added += 1
            label = f"Figure A{self.app_added}"
        else:
            self.fig_added.append(key)
            assert self.fig_order[len(self.fig_added) - 1] == key, f"figure order mismatch at {key}"
            label = f"Figure {len(self.fig_added)}"
        path = PROJ + rf"\figures\{key}.png"
        w, h = Image.open(path).size
        wi = max_w
        if wi * h / w > max_h:
            wi = max_h * w / h
        p = self._make(self.proto_body)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        tidy_spacing(p, before=6, after=0)
        keep_next(p)
        p.add_run().add_picture(path, width=Inches(wi))
        self.caption(f"{label}. {caption}")

    def table(self, caption, header, rows, widths=None, aligns=None, size=9.5):
        self.tab_n += 1
        self.caption(f"Table {self.tab_n}. {caption}", above=True)
        ncol = len(header)
        t = self.doc.add_table(rows=1 + len(rows), cols=ncol)
        self.sect.addprevious(t._tbl)
        t.alignment = 1
        t.autofit = False
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "4")
            e.set(qn("w:space"), "0")
            e.set(qn("w:color"), "808080")
            borders.append(e)
        t._tbl.tblPr.append(borders)
        if widths is None:
            widths = [6.5 / ncol] * ncol
        if aligns is None:
            aligns = ["l"] + ["r"] * (ncol - 1)
        amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "r": WD_ALIGN_PARAGRAPH.RIGHT, "c": WD_ALIGN_PARAGRAPH.CENTER}
        for ri, row in enumerate([header] + rows):
            tr = t.rows[ri]
            trPr = tr._tr.get_or_add_trPr()
            trPr.append(OxmlElement("w:cantSplit"))
            if ri == 0:
                trPr.append(OxmlElement("w:tblHeader"))
            for ci, val in enumerate(row):
                cell = tr.cells[ci]
                cell.width = Inches(widths[ci])
                par = cell.paragraphs[0]
                par.alignment = amap["c" if ri == 0 else aligns[ci]]
                tidy_spacing(par, before=1.5, after=1.5, line=1.0)
                if ri < len(rows):
                    keep_next(par)
                add_runs(par, str(val), size=size, bold=True if ri == 0 else None)
                if ri == 0:
                    shd = OxmlElement("w:shd")
                    shd.set(qn("w:val"), "clear")
                    shd.set(qn("w:color"), "auto")
                    shd.set(qn("w:fill"), "E7E6E6")
                    cell._tc.get_or_add_tcPr().append(shd)
        sp = self._make(self.proto_body)
        tidy_spacing(sp, before=0, after=4)
        return t

        sp.set(qn("w:lineRule"), "auto")


def keep_next(p):
    p.paragraph_format.keep_with_next = True
