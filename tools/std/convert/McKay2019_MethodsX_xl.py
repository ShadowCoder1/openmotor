import io, re, zipfile
import numpy as np

_COL = re.compile(r"^([A-Z]+)")


def col_idx(s):
    n = 0
    for ch in s:
        n = n * 26 + ord(ch) - 64
    return n - 1


def workbook_sheets(x):
    wbx = x.read("xl/workbook.xml").decode("utf8")
    rels = x.read("xl/_rels/workbook.xml.rels").decode("utf8")
    out = []
    for m in re.finditer(r"<sheet\b([^>]*)/>", wbx):
        a = m.group(1)
        nm = re.search(r'name="([^"]*)"', a).group(1)
        rid = re.search(r'r:id="([^"]*)"', a).group(1)
        t = None
        for r in re.finditer(r"<Relationship\b([^>]*)/>", rels):
            if 'Id="%s"' % rid in r.group(1):
                t = re.search(r'Target="([^"]*)"', r.group(1)).group(1)
        t = t.lstrip("/")
        out.append((nm, t if t.startswith("xl/") else "xl/" + t))
    return out


def shared_strings(x):
    if "xl/sharedStrings.xml" not in x.namelist():
        return []
    s = x.read("xl/sharedStrings.xml").decode("utf8")
    return [re.sub(r"<[^>]+>", "", m.group(1)) for m in re.finditer(r"<si>(.*?)</si>", s, re.S)]



import xml.etree.ElementTree as ET


def read_sheet(x, path, ss, ncols=16):
    """list of rows; each row a list of length ncols (float, str or None). Parsed with ElementTree iterparse."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rows = []
    with x.open(path) as fh:
        for ev, el in ET.iterparse(fh, events=("end",)):
            if el.tag == ns + "row":
                row = [None] * ncols
                for c in el:
                    r = c.get("r")
                    j = 0
                    while j < len(r) and r[j].isalpha():
                        j += 1
                    ci = col_idx(r[:j])
                    if ci >= ncols:
                        continue
                    t = c.get("t")
                    v = c.find(ns + "v")
                    if t == "inlineStr":
                        is_ = c.find(ns + "is")
                        row[ci] = "".join(is_.itertext()) if is_ is not None else None
                    elif v is None or v.text is None:
                        continue
                    elif t == "s":
                        row[ci] = ss[int(v.text)]
                    elif t in ("str", "e", "b"):
                        row[ci] = v.text
                    else:
                        try:
                            row[ci] = float(v.text)
                        except ValueError:
                            row[ci] = v.text
                rows.append(row)
                el.clear()
    return rows
