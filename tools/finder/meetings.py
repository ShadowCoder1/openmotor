"""Roster of motor-control researchers from public meeting programmes, for the people pipeline (people.py).

usage: python3 -I tools/finder/meetings.py [--out work/meetings]   ->  <out>/roster.csv (same columns as the NCM roster)
then:  python3 -I tools/finder/people.py resolve --out work/meetings   (and works, links, deposits, prep, build, report)

Sources (all public pages whose robots.txt allows reading them):
  - Advances in Computational Motor Control / Translational and Computational Motor Control, the motor-control satellite
    of the Society for Neuroscience meeting: proceedings 2002-2013 and programmes 2014-2015 (talk and poster authors).
  - XV Progress in Motor Control (International Society of Motor Control, 2025): poster index (presenting authors).
  - Society for Neuroscience annual meeting, sfn.org pages: chairs and co-chairs of Theme E (Motor Systems) symposia and
    minisymposia, 2019, 2021 and 2022 (the only years sfn.org lists). The meeting's poster and talk abstracts live in the
    Neuroscience Meeting Planner (abstractsonline.com), whose robots.txt disallows all automated access; not used.
People already on the NCM roster (work/ncm/roster.csv) are left out: their work was searched in the NCM run."""
import argparse
import csv
import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import people as P  # noqa: E402

MLMC = "https://sites.google.com/view/mlmc-proceedings/all"
TCMC = "https://sites.google.com/site/acmcconference/untitled/{}"
PMC = "https://web.uri.edu/chs/wp-content/uploads/sites/969/Poster-Sessions-Schedule-XV-PMC-2025.pdf"
SFN = "https://www.sfn.org/meetings/neuroscience-{}/sessions-and-events/scientific-program/{}"
PART = {"van", "von", "de", "der", "den", "da", "del", "di", "la", "le", "du", "dos", "ter"}
NOT_NAME = re.compile(r"\d|session|plenary|break|speaker|talks?\b|posters?\b|program|lecture|memorial|summer school|"
                      r"university|institute|committee|coffee|lunch|welcome|award", re.I)


def lines(raw):
    t = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    return [re.sub(r"\s+", " ", x).strip() for x in html.unescape(re.sub(r"<[^>]+>", "\n", t)).split("\n") if x.strip()]


def is_initials(s):
    return bool(re.fullmatch(r"(?:[A-Z]\.?-?\s?){1,3}", s.strip()))


def author_names(line):
    """Names in an author line ('Harel, N., Pena, S. and Spugen, A.' / 'Shanechi M, Orsborn A & Carmena, J.' /
    'J. Patton, F. Mussa-Ivaldi and Y. Wei' / 'Alkis Hadjiosif, Maurice Smith'), as 'Given Family'; None when the line
    is not an author line."""
    s = re.sub(r"\(.*?\)", "", line).strip(" .;")
    if not s or NOT_NAME.search(s) or len(s) > 400 or s.endswith("?"):
        return None
    s = re.sub(r",?\s+(?:and|&)\s+", ", ", s)
    s = re.sub(r"(?<=\b[A-Z]{2}) (?=[A-Z][a-z])", ", ", s)       # 'Hart BM Behbahani FM' -> 'Hart BM, Behbahani FM'
    parts = [p.strip(" .") for p in s.split(",") if p.strip(" .")]
    merged = []
    for p in parts:                                              # 'Harel', 'N.' -> 'N. Harel'
        if is_initials(p) and merged and len(merged[-1].split()) == 1:
            merged[-1] = f"{p} {merged[-1]}"
        else:
            merged.append(p)
    out = []
    for p in merged:
        toks = p.split()
        if not 2 <= len(toks) <= 5:
            return None
        if any(t[0].islower() and t.lower() not in PART for t in toks):
            return None
        if len(toks) == 2 and re.fullmatch(r"[A-Z]{1,3}", toks[1]) and not is_initials(toks[0]):
            toks = [toks[1], toks[0]]                            # 'Shanechi M' -> 'M Shanechi'
        out.append(" ".join(toks))
    return out or None


def proceedings(h, cache):
    rows = []
    pages = [(MLMC, None)] + [(TCMC.format(y), y) for y in (2014, 2015)]
    for url, year in pages:
        f = cache / (re.sub(r"\W+", "_", url.split("//")[1]) + ".html")
        if not f.exists():
            raw = h.get(url, raw=True)
            if raw:
                f.write_bytes(raw)
        L = lines(f.read_bytes())
        y = year
        for x in L:
            m = re.match(r"^Volume \d+, (\d{4})$", x)
            if m:
                y = int(m.group(1))
                continue
            if y is None:
                continue
            for n in author_names(x) or []:
                name = "ACMC" if y <= 2011 else "TCMC"
                rows.append({"name": n, "affiliation": "", "role": f"talk or poster author, {name} {y} (SfN satellite)",
                             "source": url})
    return rows


def pmc(h, cache):
    f = cache / "Poster-Sessions-Schedule-XV-PMC-2025.pdf"
    if not f.exists():
        f.write_bytes(h.get(PMC, raw=True))
    txt = subprocess.run(["pdftotext", "-layout", str(f), "-"], capture_output=True, text=True).stdout
    rows = []
    for m in re.finditer(r"P[12]-\d+\s+([^,\n]{2,40}),\s?([A-Z][^\n]*?)(?=\s{2,}|$)", txt, flags=re.M):
        rows.append({"name": f"{m.group(2).strip()} {m.group(1).strip()}", "affiliation": "",
                     "role": "poster presenter, XV Progress in Motor Control 2025", "source": PMC})
    return rows


def sfn(h, cache):
    rows = []
    for y in (2019, 2021, 2022):
        for kind in ("symposia", "minisymposia"):
            url = SFN.format(y, kind)
            f = cache / f"sfn_{y}_{kind}.html"
            if not f.exists():
                raw = h.get(url, raw=True)
                if raw:
                    f.write_bytes(raw)
            if not f.exists():
                continue
            L = lines(f.read_bytes())
            # 2019: each session block has 'Chair:' name 'Institution:' inst ... 'Theme:' theme.
            # 2021/2022: sessions under theme headings; 'Chair: Name, PhD' then the institution on the next line.
            theme, block = "", []
            def flush():
                for n, inst, role in block:
                    if theme.startswith("Theme E"):
                        rows.append({"name": n, "affiliation": inst, "role": f"{role}, SfN {y} {kind[:-1]}um (Theme E)",
                                     "source": url})
            start = next((i for i, x in enumerate(L) if x.startswith("Theme I")), 0) + 1
            for i, x in enumerate(L[start:], start):
                if re.match(r"^Theme [A-I] – ", x) and not (i > 0 and L[i - 1] == "Theme:"):
                    flush(); block = []; theme = x
                    continue
                m = re.match(r"^(Co-)?Chair:\s*(.*)$", x)
                if m:
                    nm = m.group(2) or (L[i + 1] if i + 1 < len(L) else "")
                    inst = ""
                    j = i + (1 if m.group(2) else 2)
                    if j < len(L):
                        inst = L[j + 1] if L[j] == "Institution:" and j + 1 < len(L) else L[j]
                    nm = re.sub(r",?\s*(PhD|MD|DPhil|MS|MSc|DVM)\b\.?", "", nm).strip(" ,")
                    if P.looks_like_name(nm):
                        block.append((nm, inst if not inst.endswith(":") else "", "co-chair" if m.group(1) else "chair"))
                if x == "Theme:" and i + 1 < len(L):            # 2019 style: theme closes the block
                    theme = L[i + 1]
                    flush(); block = []; theme = ""
            flush()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.HERE / "work" / "meetings"))
    a = ap.parse_args()
    out = Path(a.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    h = C.Http(rate=0.5)
    rows = proceedings(h, cache) + pmc(h, cache) + sfn(h, cache)
    ncm = {P.name_key(r["name"]) for r in csv.DictReader(open(C.HERE / "work/ncm/roster.csv", encoding="utf-8"))}
    ncm_init = {k.split("|")[0] + "|" + k.split("|")[1][:1] for k in ncm}
    full = {}                                                     # 'thoroughman|k' -> 'thoroughman|kurt' when one full name fits
    for r in rows:
        k = P.name_key(r["name"])
        if len(k.split("|")[1]) > 1:
            full.setdefault(k.split("|")[0] + "|" + k.split("|")[1][:1], set()).add(k)
    def key(n):
        k = P.name_key(n)
        if len(k.split("|")[1]) <= 1 and len(full.get(k, ())) == 1:
            return next(iter(full[k]))
        return k
    def on_ncm(n):
        k = P.name_key(n)
        return k in ncm or (len(k.split("|")[1]) <= 1 and k in ncm_init)
    by = {}
    for r in rows:
        k = key(r["name"])
        if on_ncm(r["name"]) or k in ncm:
            continue
        if k not in by:
            by[k] = dict(r)
        else:
            b = by[k]
            if len(r["name"]) > len(b["name"]):                  # keep the fullest spelling ('Maurice Smith' over 'M Smith')
                b["name"] = r["name"]
            for c in ("affiliation", "role", "source"):
                if r[c] and r[c] not in b[c].split(" | "):
                    b[c] = " | ".join(x for x in (b[c], r[c]) if x)
    with open(out / "roster.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "affiliation", "role", "source"])
        w.writeheader()
        w.writerows(sorted(by.values(), key=lambda r: key(r["name"])))
    src = {}
    for r in by.values():
        for s in {re.sub(r",.*", "", x.split(", ")[-1]) for x in r["role"].split(" | ")}:
            src[s] = src.get(s, 0) + 1
    print(f"meetings: {len(rows)} author mentions, {len(by)} people not on the NCM roster -> {out / 'roster.csv'}")
    print(f"  NCM overlap left out: {len({key(r['name']) for r in rows if on_ncm(r['name']) or key(r['name']) in ncm})}")


if __name__ == "__main__":
    main()
