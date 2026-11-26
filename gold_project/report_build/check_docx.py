import io, os, re, sys
from docx import Document
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(HERE))
d = Document(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "Final_Project_Report_Gold_Price_Simple.docx"))
out = io.open(os.path.join(BASE, "evidence", "check_output.txt"), "w", encoding="utf8")
texts = [p.text for p in d.paragraphs]
for t in d.tables:
    for row in t.rows:
        for cell in row.cells:
            texts.append(cell.text)
bad = {}
for t in texts:
    for ch in t:
        if ord(ch) > 126 or (ord(ch) < 32 and ch not in "\t\n"):
            bad[ch] = bad.get(ch, 0) + 1
out.write("non-keyboard chars: %s\n" % {("U+%04X" % ord(k)): v for k, v in bad.items()})
for i, t in enumerate(texts[:14]):
    out.write("%d %r\n" % (i, t))
words = sum(len(t.split()) for t in texts)
out.write("paragraphs %d words %d inline images %d tables %d\n" % (len(d.paragraphs), words, len(d.inline_shapes), len(d.tables)))
flag = ["delve", "crucial", "leverage", "utilize", "furthermore", "moreover", "robust", "comprehensive", "landscape", "tapestry", "pivotal", "seamless", "notably", "underscore", "showcase", "in conclusion"]
full = " ".join(texts).lower()
out.write("flag words: %s\n" % {w: full.count(w) for w in flag if w in full})
out.write("placeholders: %s\n" % re.findall(r"\[[^\]]+\]", " ".join(texts)))
out.close()
