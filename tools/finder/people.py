"""Authors mode for the finder: start from a list of people (the Society for the Neural Control of Movement roster), find
their published work that comes with a public participant- or trial-level data deposit, and hand the candidates to the
same screening and listing conventions as the weekly finder.

usage:
  python3 -I tools/finder/people.py roster  [--out work/ncm]          # NCM web pages + meeting books -> roster.csv
  python3 -I tools/finder/people.py resolve [--out work/ncm]          # roster -> OpenAlex author ids (resolved.json)
  python3 -I tools/finder/people.py works   [--out work/ncm]          # all works of resolved authors (works.jsonl)
  python3 -I tools/finder/people.py links   [--out work/ncm]          # data links: Europe PMC full texts, DataCite,
                                                                       # bioRxiv, the authors' own deposits -> candidates.jsonl
  python3 -I tools/finder/people.py deposits [--out work/ncm]         # repository API facts for each candidate deposit

Resolution is careful on purpose: a person is matched to an OpenAlex author only when the name agrees (family name and
first given name or initial) and either an affiliation from the roster appears in the author's affiliation history, or
(no affiliation known) exactly one same-name author works in motor control. Several OpenAlex ids are kept for one person
only when each of them passes the same test (OpenAlex sometimes splits one person); namesakes are never merged.

OpenAlex budget: without OPENALEX_API_KEY an IP gets 1,000 credits a day; autocomplete and single works by DOI cost
nothing, a filtered list costs 1 credit per page of 200, a search costs 10. This script uses only the first three.
Nothing here downloads data files: only metadata and repository APIs."""
import argparse
import csv
import html
import json
import re
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

NCM = "https://ncm-society.org"
PAGES = ["2027-meeting", "program", "keynote", "satellite-meeting", "2026-meeting", "meeting-history", "ncm-leadership",
         "ncm-elections", "scholarships", "about-the-society", "in-memoriam", "diversity-program"]
BOOKS = {"NCM 2026 meeting book (Kobe)": "/wp-content/uploads/2026/04/NCM2026DigitalProgramApr23.pdf",
         "NCM 2025 meeting book (Panama City)": "/wp-content/uploads/2025/05/NCM2025FinalProgram.pdf",
         "NCM 2024 meeting book (Dubrovnik)": "/wp-content/uploads/2024/05/NCM2024FinalApril20.pdf"}


# ------------------------------------------------------------------ names
def fold(s):
    s = (s or "").replace("ß", "ss").replace("ø", "o").replace("Ø", "O").replace("ł", "l").replace("Ł", "L").replace("æ", "ae")
    s = re.sub(r"([aou])̈", r"\1", unicodedata.normalize("NFD", s))  # ä -> a (Jörn == Jorn)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def split_name(full):
    """'J. Andrew Pruszynski' -> ('Pruszynski', ['J', 'Andrew']); keeps particles with the family name."""
    full = re.sub(r"\s+", " ", re.sub(r"\(.*?\)", " ", full or "")).strip(" ,.;*")
    if "," in full:
        fam, giv = [x.strip() for x in full.split(",", 1)]
        toks = giv.replace(".", ". ").split()
    else:
        toks = full.replace(".", ". ").split()
        if len(toks) < 2:
            return full, []
        i = len(toks) - 1
        while i > 1 and toks[i - 1].lower() in ("van", "von", "de", "der", "den", "da", "del", "di", "la", "le", "du", "dos", "ter"):
            i -= 1
        fam, toks = " ".join(toks[i:]), toks[:i]
    return fam, [t.strip(".") for t in toks if t.strip(".")]


def name_key(full):
    fam, giv = split_name(full)
    first = next((g for g in giv if len(g) > 1), giv[0] if giv else "")
    f = lambda x: re.sub(r"(?<=[aou])e", "", re.sub(r"[^a-z]", "", fold(x)))  # Joern == Jorn
    return f"{f(fam)}|{f(first)}"


def looks_like_name(s):
    toks = s.split()
    return 2 <= len(toks) <= 6 and all(t[0].isupper() or t.lower() in ("van", "von", "de", "der", "den", "da", "del", "di", "la", "le", "du", "dos", "ter")
                                      or t[0] in "'’(" for t in toks) and not re.search(r"\d|@|University|Institute|Center|Centre", s)


# ------------------------------------------------------------------ web pages
def page_text(raw):
    t = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    m = re.search(r"<main.*?</main>|<article.*?</article>", t, flags=re.S)
    body = m.group(0) if m else t
    body = re.sub(r"<script.*?</script>|<style.*?</style>|<nav.*?</nav>|<footer.*?</footer>|<header.*?</header>", "", body, flags=re.S)
    txt = html.unescape(re.sub(r"<br\s*/?>|</p>|</li>|</h\d>|</tr>|</div>", "\n", body))
    txt = re.sub(r"<[^>]+>", " ", txt).replace("\xa0", " ").replace("​", "")
    return [re.sub(r"\s+", " ", x).strip() for x in txt.split("\n") if x.strip()]


def from_pages(pages):
    """People named on the society's web pages, with the role each page gives them."""
    out = []
    add = lambda n, a, r, p: out.append({"name": n.strip(" ,"), "affiliation": (a or "").strip(" ,"), "role": r, "source": f"{NCM}/{p}/"})
    L = pages.get("ncm-leadership", [])
    roles = ("President", "Vice-President", "Secretary", "Development Officer", "Academic Development")
    sec = ""
    for i, x in enumerate(L):
        if x in ("NCM Officers", "NCM Committees", "NCM Directors"):
            sec = x
        elif re.match(r"^[A-Z].*Committee$", x):
            sec = x
        elif sec == "NCM Officers" and i + 1 < len(L) and L[i + 1].startswith(roles) and looks_like_name(x):
            add(x, "", "NCM officer: " + L[i + 1], "ncm-leadership")
        elif sec.endswith("Committee"):
            m = re.match(r"^(Chair:\s*)?([^,]+?)\s*,\s*(.+)$", x)
            if m and looks_like_name(m.group(2)):
                add(m.group(2), m.group(3), f"{sec}{' chair' if m.group(1) else ' member'}", "ncm-leadership")
        elif sec == "NCM Directors" and i + 1 < len(L):
            m = re.match(r"^(.+?)( – Trainee Board Member.*)?$", x)
            if looks_like_name(re.sub(r"\(.*?\)", "", m.group(1))) and re.search(r", [A-Z]{3}$", L[i + 1]):
                add(re.sub(r"\(.*?\)", "", m.group(1)), re.sub(r", [A-Z]{3}$", "", L[i + 1]),
                    "NCM board" + (" (trainee member)" if m.group(2) else " director"), "ncm-leadership")
    E = pages.get("ncm-elections", [])
    for i, x in enumerate(E[:-1]):
        if looks_like_name(x) and len(x) < 40 and len(E[i + 1]) < 200 and not looks_like_name(E[i + 1]) and \
                re.search(r"Univ|Inst|Lab|College|School|Center|Centre|Hospital|Foundation|Research|Polytechn|ETH|EPFL|CNRS|INSERM|RIKEN", E[i + 1]):
            add(x, E[i + 1], "candidate, NCM board election 2026-2029", "ncm-elections")
    for x in pages.get("scholarships", []):
        m = re.match(r"^([^,:]+?)\s*,\s*(.+?)\s*:\s*([^,:]+)$", x)
        if m and looks_like_name(m.group(1)) and looks_like_name(m.group(3)):
            add(m.group(1), m.group(2), "NCM scholarship recipient", "scholarships")
            add(m.group(3), "", "mentor of an NCM scholarship recipient", "scholarships")
    A = pages.get("about-the-society", [])
    for i, x in enumerate(A[:-2]):
        if looks_like_name(x) and re.match(r"^\d{4}\s*[–-]\s*(\d{4}|Present)$", A[i + 1]):
            add(x, A[i + 2], f"NCM president {A[i + 1]}", "about-the-society")
    M = pages.get("in-memoriam", [])
    for i, x in enumerate(M[:-1]):
        if looks_like_name(x) and len(x) < 40 and not M[i + 1].startswith(x.split()[0] + " " ) or (looks_like_name(x) and "passed away" in M[i + 1]):
            if looks_like_name(x) and len(x) < 40:
                aff = M[i + 1] if "passed away" not in M[i + 1] else ""
                add(x, aff, "in memoriam", "in-memoriam")
    K = pages.get("keynote", [])
    for i, x in enumerate(K[:-2]):
        m = re.match(r"^(\d{4}) (Distinguished Career|Early Career) Award Winner$", x)
        if m:
            j = i + 1 + (1 if re.match(r"^[A-Z][a-z]+day,", K[i + 1]) else 0)
            add(K[j], K[j + 1], f"{m.group(1)} {m.group(2)} Award (keynote)", "keynote")
    for x in pages.get("satellite-meeting", []):
        m = re.match(r"^([^,]+?)\s*,\s*(.+)$", x)
        if m and looks_like_name(m.group(1)) and len(x) < 120:
            add(m.group(1), m.group(2), "satellite meeting 2027 speaker or organizer", "satellite-meeting")
    seen, uniq = set(), []
    for r in out:
        k = (name_key(r["name"]), r["role"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


# ------------------------------------------------------------------ meeting books
def book_people(pdf_path, label):
    """Authors of every talk and poster in a meeting book (names with superscript affiliation numbers), and the
    'Organizer:', 'Discussant:', 'Chair:' and 'Presenting Author:' lines, and the poster author index."""
    import fitz  # PyMuPDF; only needed for this step
    doc = fitz.open(pdf_path)
    people = []
    code_re = re.compile(r"^(P\d-[A-Z]-\d+|O\d+\.\d+|[A-Z]?\d?-?[A-Z]-\d+|\d-[A-Z]-\d+)\b")
    in_index = False
    for pno, page in enumerate(doc):
        W = page.rect.width
        d = page.get_text("dict")
        lines = []
        for b in d["blocks"]:
            for ln in b.get("lines", []):
                sp = [s for s in ln["spans"] if s["text"].strip()]
                if sp:
                    lines.append((ln["bbox"][0], ln["bbox"][1], sp))
        if not lines:
            continue
        sizes = sorted(s["size"] for _, _, sp in lines for s in sp)
        body = sizes[len(sizes) // 2]
        text_all = " ".join(s["text"] for _, _, sp in lines for s in sp)
        if re.search(r"(Poster )?Author Index", text_all[:300], re.I):
            in_index = True
        if in_index:
            cols = {}
            for x, y, sp in sorted(lines, key=lambda t: (round(t[0] / 40), t[1])):
                for s in sp:
                    tx = s["text"].strip()
                    if "Medium" in s["font"] or "Bold" in s["font"] and s["size"] < 12:
                        if re.match(r"^[^\d,]+,\s*[^\d,]+$", tx) and not re.search(r"Poster|Name|Annual|Japan", tx):
                            people.append({"name": tx, "affiliation": "", "role": "poster author (index)", "source": label, "where": "author index"})
                            cols[round(x / 40)] = len(people) - 1
                        elif tx and "," not in tx and not re.search(r"\d|Name|Poster|Index|Annual|Japan", tx) and round(x / 40) in cols:
                            k = cols.pop(round(x / 40))  # second line of a wrapped name
                            people[k]["name"] = (people[k]["name"] + ("" if people[k]["name"].endswith("-") else " ") + tx).strip()
            continue
        lines.sort(key=lambda t: (0 if t[0] < W / 2 - 25 else 1, t[1]))
        ctx, state, auth, aff = "", "", [], []

        def flush():
            if not auth:
                return
            a = re.sub(r"\}\s*\{|\}\s*,\s*\{", ",", "".join(auth))
            a = re.sub(r"\s+", " ", a)
            f = re.sub(r"\s+", " ", "".join(aff))
            affs = {}
            for m in re.finditer(r"\{([^}]*)\}([^{]*)", f):
                for n in re.findall(r"\d+", m.group(1)):
                    affs[n] = m.group(2).strip(" ,;")
            for m in re.finditer(r"\s*,?\s*([^{},]+?)\s*\{([^}]*)\}", a):
                nm = re.sub(r"[*†‡§#]+", "", m.group(1)).strip()
                if looks_like_name(nm) or (len(nm.split()) >= 2 and len(nm) < 45 and not re.search(r"\d", nm)):
                    nums = re.findall(r"\d+", m.group(2))
                    people.append({"name": nm, "affiliation": "; ".join(dict.fromkeys(affs.get(n, "") for n in nums if affs.get(n))),
                                   "role": ("poster author" if ctx.startswith("P") or re.match(r"^\d-[A-Z]", ctx) else "talk or panel author"),
                                   "source": label, "where": f"{ctx or 'session'} (p. {pno + 1})"})
            auth.clear()
            aff.clear()

        for x, y, sp in lines:
            first = sp[0]
            tx = "".join(s["text"] for s in sp).strip()
            sup = lambda s: s["size"] < 0.72 * body and re.fullmatch(r"[\d,\s*†‡]+", s["text"].strip() or "x")
            m = re.match(r"^(Organi[sz]er|Discussant|Chair|Co-Chair|Moderator|Presenting Author)s?:\s*(.+)$", tx)
            if m:
                flush()
                state = ""
                for nm in re.split(r",| and |;", m.group(2)):
                    if looks_like_name(nm.strip()):
                        people.append({"name": nm.strip(), "affiliation": "", "role": m.group(1).lower(), "source": label,
                                       "where": f"{ctx or 'session'} (p. {pno + 1})"})
                continue
            mc = code_re.match(tx)
            if mc and first["size"] >= body * 0.98 and ("Medium" in first["font"] or "Bold" in first["font"]):
                flush()
                ctx, state = mc.group(1), "title"
                continue
            if sup(first) and state in ("authors", "affil"):
                state = "affil"
                aff.append("".join("{" + s["text"].strip() + "}" if sup(s) else s["text"] for s in sp) + " ")
                continue
            has_sup = any(sup(s) for s in sp[1:])
            if has_sup and not sup(first) and state in ("title", "authors", "") and "Book" in first["font"] or \
                    has_sup and not sup(first) and state in ("title", "authors") and first["size"] < body * 1.05:
                if state == "affil":
                    flush()
                state = "authors"
                auth.append("".join("{" + s["text"].strip() + "}" if sup(s) else s["text"] for s in sp) + " ")
                continue
            if state == "authors" and not has_sup and not sup(first) and first["size"] < body * 1.05:
                auth.append(tx + " ")  # an author line wrapped without a superscript on it
                continue
            if state == "affil" and not sup(first) and first["size"] < body * 1.05 and "Medium" not in first["font"]:
                aff.append(tx + " ")
                continue
            if state in ("affil", "authors"):
                flush()
                state = "title" if ("Medium" in first["font"] or "Bold" in first["font"]) else ""
        flush()
    return people


def cmd_roster(a):
    out = Path(a.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    h = C.Http(rate=0.5)
    pages = {}
    for p in PAGES:
        f = cache / f"{p}.html"
        if not f.exists():
            raw = h.get(f"{NCM}/{p}/", raw=True)
            if raw:
                f.write_bytes(raw)
        if f.exists():
            pages[p] = page_text(f.read_bytes())
    rows = from_pages(pages)
    # the 2027 programme at a glance and keynote page name the 2027 award lectures; the 2027 book is not out yet
    for label, path in BOOKS.items():
        f = cache / Path(path).name
        if not f.exists():
            raw = h.get(NCM + path, raw=True, timeout=180)
            if raw:
                f.write_bytes(raw)
        if f.exists():
            got = book_people(str(f), f"{label}, {NCM}{path}")
            C.log(f"{label}: {len(got)} name mentions")
            rows += got
    # merge mentions of one person (same family name and first given name)
    by = {}
    for r in rows:
        if not r["name"] or len(r["name"]) > 60:
            continue
        nm = re.sub(r"^(Keynote|Speaker|Chair|Organi[sz]er)s?:\s*", "", r["name"]).strip()
        if "," in nm and r["role"].endswith("(index)"):
            fam, giv = [x.strip() for x in nm.split(",", 1)]
            nm = f"{giv} {fam}"
        k = name_key(nm)
        if k.endswith("|") or k.startswith("|"):
            continue
        p = by.setdefault(k, {"name": nm, "affiliations": [], "roles": [], "sources": []})
        if len(nm) > len(p["name"]) and r["role"] != "poster author (index)":
            p["name"] = nm  # the fullest spelling (with accents and middle names) from the abstracts
        for aff in (r.get("affiliation") or "").split("; "):
            if aff and aff not in p["affiliations"]:
                p["affiliations"].append(aff)
        role = r["role"] + (f" {r['where']}" if r.get("where") and r["role"] != "poster author (index)" else "")
        if role not in p["roles"]:
            p["roles"].append(role)
        if r["source"] not in p["sources"]:
            p["sources"].append(r["source"])
    people = sorted(by.values(), key=lambda p: name_key(p["name"]))
    with open(out / "roster.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "affiliation", "role", "source"])
        for p in people:
            w.writerow([p["name"], " | ".join(p["affiliations"][:4]), " | ".join(p["roles"][:6]) + (" | ..." if len(p["roles"]) > 6 else ""),
                        " | ".join(p["sources"])])
    print(f"roster: {len(people)} people from {len(rows)} mentions -> {out / 'roster.csv'}")


import threading  # noqa: E402
NCBI_LOCK, NCBI_LAST = threading.Lock(), [0.0]


def epmc_rows(h, query, n=25):
    d = h.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", {"query": query, "format": "json", "pageSize": n, "resultType": "lite"})
    return ((d or {}).get("resultList") or {}).get("result") or []


# ------------------------------------------------------------------ resolve: roster -> OpenAlex author ids
OA = "https://api.openalex.org"
AFF_STOP = set("university universite universitat universita universidad universidade univ institute institut instituto department "
               "dept center centre school college hospital research sciences science medical medicine national technology health "
               "faculty laboratory lab labs foundation clinic clinical the and for of de la der des di du und et state graduate "
               "neuroscience neurosciences program programme division group unit campus".split())
MOTOR_TOPIC = re.compile(r"motor|movement|gait|balance|postur|saccad|eye|oculo|speech|voice|reach|grasp|limb|muscle|electromyog|"
                         r"kinemat|biomech|locomot|cerebell|basal ganglia|parkinson|stroke|tremor|dystonia|prosthe|haptic|rehabilit|"
                         r"sensorimotor|propriocep|vestibul|spinal|neuromuscular|brain-computer|transcranial|tactile|somatosens|sport|"
                         r"exercise|music|rhythm|timing|hand|finger|walking|ataxia|neural dynamics|robot", re.I)


def aff_tokens(s):
    return {t for t in re.findall(r"[a-z]{4,}|[A-Z]{3,5}", fold(s) + " " + " ".join(re.findall(r"\b[A-Z]{3,5}\b", s or "")))
            if t not in AFF_STOP}


def same_name(roster_name, oa_name):
    def parts(n):
        fam, giv = split_name(n)
        f = lambda x: re.sub(r"[^a-z]", "", fold(x))
        return f(fam), [f(g) for g in giv if f(g)]
    rf, rg = parts(roster_name)
    for of, og in (parts(oa_name), parts(" ".join(reversed(oa_name.split()))) if len(oa_name.split()) == 2 else (None, None)):
        if of is None or of != rf or not rg or not og:
            continue
        rfull, ofull = [g for g in rg if len(g) > 1], [g for g in og if len(g) > 1]
        if rfull and ofull:
            if rfull[0] == ofull[0] or rfull[0] in ofull or ofull[0] in rfull or \
                    rfull[0].replace("oe", "o").replace("ue", "u") == ofull[0].replace("oe", "o").replace("ue", "u"):
                return True
            if "".join(rfull) == "".join(ofull):  # Chih Yang == Chihyang
                return True
            continue
        if rg[0][0] == og[0][0]:
            return True
    return False


def cmd_resolve(a):
    out = Path(a.out)
    roster = list(csv.DictReader(open(out / "roster.csv", encoding="utf-8")))
    if a.limit:
        roster = roster[:a.limit]
    cache_p = out / "autocomplete.json"
    cache = C.load_json(cache_p, {})
    h = C.Http(rate=4)
    from concurrent.futures import ThreadPoolExecutor

    def ac(q):  # autocomplete is free; four connections at a time, each with its own polite gap
        hh = C.Http(rate=3)
        d = hh.get(f"{OA}/autocomplete/authors", C.openalex_params({"q": fold(q)}))
        return q, ([{"id": x["id"].rsplit("/", 1)[-1], "name": x.get("display_name"), "hint": x.get("hint"),
                     "works": x.get("works_count"), "orcid": x.get("external_id")} for x in (d or {}).get("results", [])] if d else None)
    todo = [r["name"] for r in roster if r["name"] not in cache or cache[r["name"]] is None]
    with ThreadPoolExecutor(4) as ex:
        for i, (q, v) in enumerate(ex.map(ac, todo)):
            cache[q] = v
            if i % 300 == 0:
                C.save_json(cache_p, cache)
                C.log(f"  autocomplete {i}/{len(todo)}")
    C.save_json(cache_p, cache)
    # profiles (affiliation history, topics) of every same-name candidate: 1 credit per 100 authors
    prof_p = out / "author_profiles.json"
    prof = C.load_json(prof_p, {})
    need = sorted({c["id"] for r in roster for c in (cache.get(r["name"]) or []) if same_name(r["name"], c["name"] or "")} - set(prof))
    for i in range(0, len(need), 100):
        d = h.get(f"{OA}/authors", C.openalex_params({"filter": "id:" + "|".join(need[i:i + 100]), "per_page": 100,
                  "select": "id,display_name,display_name_alternatives,works_count,affiliations,last_known_institutions,topics,orcid"}))
        for x in (d or {}).get("results", []):
            aid = x["id"].rsplit("/", 1)[-1]
            prof[aid] = {"name": x["display_name"], "alt": (x.get("display_name_alternatives") or [])[:12], "works": x.get("works_count"),
                         "insts": sorted({(f.get("institution") or {}).get("display_name", "") for f in x.get("affiliations") or []} |
                                         {i_.get("display_name", "") for i_ in x.get("last_known_institutions") or []}),
                         "topics": [t["display_name"] for t in (x.get("topics") or [])[:10]], "orcid": x.get("orcid")}
        C.save_json(prof_p, prof)
    res = {}
    for r in roster:
        cands = [c for c in (cache.get(r["name"]) or []) if same_name(r["name"], c["name"] or "") and c["id"] in prof]
        rt = aff_tokens(r["affiliation"])
        strong, motor = [], []
        for c in cands:
            p = prof[c["id"]]
            it = set().union(*[aff_tokens(x) for x in p["insts"]]) if p["insts"] else set()
            mot = sum(1 for t in p["topics"][:6] if MOTOR_TOPIC.search(t))
            if rt and rt & it:
                strong.append((c["id"], sorted(rt & it)[:4], mot))
            elif mot >= 2:
                motor.append((c["id"], mot))
        if strong:
            ids = [s[0] for s in strong if s[2] >= 1 or len(strong) == 1]
            res[r["name"]] = {"ids": ids or [strong[0][0]], "how": "name + affiliation (" + ", ".join(strong[0][1]) + ")" +
                              (f"; {len(ids)} OpenAlex profiles" if len(ids) > 1 else "")}
        elif len(motor) == 1 and not rt:
            res[r["name"]] = {"ids": [motor[0][0]], "how": "name + motor-control topics; only such author of that name (no affiliation in roster)"}
        elif len(motor) == 1 and rt:
            res[r["name"]] = {"ids": [motor[0][0]], "how": "name + motor-control topics; only such author of that name (roster affiliation not in OpenAlex history)",
                              "weak": True}
        else:
            res[r["name"]] = {"ids": [], "how": "not resolved: " + ("no same-name author" if not cands else
                              f"{len(motor)} same-name motor-control authors, none at the roster affiliation" if motor else
                              "same-name authors, none at the roster affiliation or in motor control")}
    # second pass for people with an affiliation but no match: a paper of theirs at that affiliation (Europe PMC author +
    # affiliation search), then that paper's OpenAlex record (free) gives the author id; the raw affiliation on the paper
    # must again share a distinctive word with the roster affiliation
    fb_p = out / "fallback.json"
    fb = C.load_json(fb_p, {})
    for k, r in enumerate(roster):
        if res[r["name"]]["ids"] or not aff_tokens(r["affiliation"]) or r["name"] in fb:
            continue
        fam, giv = split_name(r["name"])
        if not giv:
            continue
        toks = [t for t in aff_tokens(r["affiliation"].split(" | ")[0]) if t.islower()][:2]
        if not toks:
            fb[r["name"]] = []
            continue
        q = f'AUTH:"{fold(fam)} {giv[0][0]}" AND (' + " OR ".join(f'AFF:"{t}"' for t in toks) + ")"
        rows = [x for x in epmc_rows(h, q) if x.get("doi")][:4]
        found = []
        for x in rows:
            w = h.get(f"{OA}/works/doi:{C.norm_doi(x['doi'])}", C.openalex_params({"select": "id,authorships"}))
            for au in (w or {}).get("authorships") or []:
                nm = (au.get("author") or {}).get("display_name") or ""
                raw = " ".join(au.get("raw_affiliation_strings") or []) + " " + " ".join(i.get("display_name", "") for i in au.get("institutions") or [])
                if same_name(r["name"], nm) and aff_tokens(r["affiliation"]) & aff_tokens(raw) and (au.get("author") or {}).get("id"):
                    found.append(au["author"]["id"].rsplit("/", 1)[-1])
        fb[r["name"]] = sorted(set(found))
        if k % 50 == 0:
            C.save_json(fb_p, fb)
    C.save_json(fb_p, fb)
    for name, ids in fb.items():
        if ids and name in res and not res[name]["ids"]:
            res[name] = {"ids": ids[:3], "how": "name + a paper at the roster affiliation (Europe PMC, OpenAlex authorship)"}
    C.save_json(out / "resolved.json", res)
    n = sum(1 for v in res.values() if v["ids"])
    print(f"resolve: {n}/{len(res)} people resolved ({sum(1 for v in res.values() if v.get('weak'))} on topics only, affiliation differs); "
          f"works on their profiles: {sum((prof.get(i) or {}).get('works') or 0 for v in res.values() for i in v['ids'])}")


# ------------------------------------------------------------------ works: every work of the resolved authors
def cmd_works(a):
    out = Path(a.out)
    res = C.load_json(out / "resolved.json", {})
    prof = C.load_json(out / "author_profiles.json", {})
    owner, dropped = {}, []
    for name, v in res.items():
        for i in v["ids"]:
            pr = prof.get(i) or {}
            mot = sum(1 for t in (pr.get("topics") or [])[:6] if MOTOR_TOPIC.search(t))
            if (pr.get("works") or 0) > 2000 or ((pr.get("works") or 0) > 600 and mot < 2):
                dropped.append(f"{name} {i} ({pr.get('works')} works, {mot} motor topics): profile looks merged with namesakes")
                continue
            owner.setdefault(i, []).append(name)
    C.save_json(out / "dropped_profiles.json", dropped)
    ids = sorted(owner)
    done_p = out / "works_done.json"
    done = set(C.load_json(done_p, []))
    h = C.Http(rate=4)
    wp = out / "works.jsonl"
    seen = {w["id"] for w in C.read_jsonl(wp)} if wp.exists() else set()
    todo = [i for i in ids if i not in done]
    sel = "id,doi,title,publication_year,type,ids,primary_location,authorships,primary_topic"
    for b in range(0, len(todo), 50):
        chunk = todo[b:b + 50]
        page, rows = 1, []  # page numbers: cursor paging needs an API key
        while page:
            d = h.get(f"{OA}/works", C.openalex_params({"filter": "author.id:" + "|".join(chunk) + f",from_publication_date:{a.since},type:article|preprint|dataset", "per_page": 200, "page": page, "select": sel}), timeout=150)
            if not d:
                C.log("  works: request failed; stopping here (rerun to continue)")
                page = None
                chunk = []
                break
            for w in d.get("results", []):
                wid = w["id"].rsplit("/", 1)[-1]
                mine = [x for x in w.get("authorships") or [] if ((x.get("author") or {}).get("id") or "").rsplit("/", 1)[-1] in owner]
                people = sorted({n for x in mine for n in owner[x["author"]["id"].rsplit("/", 1)[-1]]})
                if wid in seen:
                    continue
                seen.add(wid)
                au = w.get("authorships") or []
                rows.append({"id": wid, "doi": C.norm_doi(w.get("doi")), "title": w.get("title") or "", "year": w.get("publication_year"),
                             "type": w.get("type"), "pmcid": ((w.get("ids") or {}).get("pmcid") or "").rsplit("/", 1)[-1],
                             "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                             "people": people, "n_authors": len(au),
                             "first": (au[0].get("author") or {}).get("display_name", "") if au else "",
                             "last": (au[-1].get("author") or {}).get("display_name", "") if au else "",
                             "topic": ((w.get("primary_topic") or {}).get("display_name") or ""),
                             "field": (((w.get("primary_topic") or {}).get("field") or {}).get("display_name") or ""),
                             })
            cnt = (d.get("meta") or {}).get("count") or 0
            page = page + 1 if len(d.get("results") or []) == 200 and page * 200 < min(cnt, 10000) else None
        if rows:
            with open(wp, "a", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        done |= set(chunk)
        C.save_json(done_p, sorted(done))
        C.log(f"  works: {len(done)}/{len(ids)} authors, {len(seen)} works")
    print(f"works: {len(seen)} works of {len(done)}/{len(ids)} OpenAlex authors -> {wp}")


# ------------------------------------------------------------------ links: which of their works have a public data deposit
DATA_TYPES = ("dataset",)
SKIP_TYPES = ("paratext", "erratum", "retraction", "editorial", "letter", "peer-review", "supplementary-materials", "libguides")
PAPER_REL_OK = ("issupplementto", "isreferencedby", "iscitedby", "isdescribedby", "isdocumentedby", "issourceof", "isderivedfrom",
                "references", "cites", "issupplementedby", "isrequiredby", "documents", "describes", "ispartof", "haspart")


def datacite_dataset_hits(h, query, size=100):
    d = h.get("https://api.datacite.org/dois", {"query": query, "page[size]": size,
                                                 "fields[dois]": "doi,url,titles,relatedIdentifiers,types,creators,publicationYear"})
    out = []
    for x in (d or {}).get("data", []):
        a = x.get("attributes", {})
        typ = ((a.get("types") or {}).get("resourceTypeGeneral") or "").lower()
        if typ not in ("dataset", "collection", "other", ""):
            continue
        rel = [(C.norm_doi(r.get("relatedIdentifier")), (r.get("relationType") or "").lower()) for r in a.get("relatedIdentifiers") or []
               if (r.get("relatedIdentifierType") or "").upper() == "DOI"]
        out.append({"doi": C.norm_doi(a.get("doi")), "url": a.get("url") or "", "title": ((a.get("titles") or [{}])[0] or {}).get("title", ""),
                    "papers": [d_ for d_, t in rel if d_ and not C.is_data_doi(d_) and t in PAPER_REL_OK],
                    "creators": [c.get("name", "") for c in a.get("creators") or []][:30], "year": a.get("publicationYear")})
    return out


def cmd_links(a):
    import find as F  # the weekly finder's search helpers
    out = Path(a.out)
    works = []
    for line in open(out / "works.jsonl", encoding="utf-8"):  # safe to run while the works step is still appending
        try:
            works.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    res = C.load_json(out / "resolved.json", {})
    h = C.Http(rate=3)
    papers = [w for w in works if w["doi"] and w["type"] not in SKIP_TYPES + DATA_TYPES and not C.is_data_doi(w["doi"])]
    by_doi = {}
    for w in papers:
        by_doi.setdefault(w["doi"], w)
    papers = list(by_doi.values())
    C.log(f"links: {len(papers)} papers with DOIs, {sum(1 for w in works if w['type'] in DATA_TYPES)} OpenAlex datasets")
    phases = ("epmc", "dc", "bx", "names")
    st = {}
    for ph in phases:
        st.update(C.load_json(out / f"links_{ph}.json", {}))
    for k, v in {"epmc_done": [], "ft": {}, "dc_done": [], "names_done": [], "deposits_dc": {}, "deposits_own": {}, "deposit_people": {},
                 "bx": {}}.items():
        st.setdefault(k, v)
    keys = {"epmc": ("epmc_done", "ft"), "dc": ("dc_done", "deposits_dc"), "bx": ("bx",), "names": ("names_done", "deposits_own", "deposit_people")}

    def save(ph):
        C.save_json(out / f"links_{ph}.json", {k: st[k] for k in keys[ph]})
    if a.phase in ("all", "epmc"):
        # 1) Europe PMC: which papers' open full texts mention a repository; then the data availability statement's links
        done = set(st["epmc_done"])
        todo = [d for d in by_doi if d not in done]
        for i in range(0, len(todo), 40):
            rows = F.epmc_by_dois(h, todo[i:i + 40], repo_filter=True)
            for r in rows:
                d = C.norm_doi(r.get("doi"))
                if d in by_doi and r.get("pmcid") and d not in st["ft"]:
                    st["ft"][d] = {"pmcid": r["pmcid"], "links": None}
            st["epmc_done"] += todo[i:i + 40]
            if i % 2000 == 0:
                save("epmc")
                C.log(f"  europepmc {i}/{len(todo)}; full texts to read {len(st['ft'])}")
        save("epmc")
        from concurrent.futures import ThreadPoolExecutor

        def ncbi(hh, pmcid):
            with NCBI_LOCK:
                time.sleep(max(0, 0.36 - (time.time() - NCBI_LAST[0])))
                NCBI_LAST[0] = time.time()
            x = hh.get("https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/", {"verb": "GetRecord", "metadataPrefix": "pmc",
                       "identifier": "oai:pubmedcentral.nih.gov:" + pmcid.replace("PMC", "")}, raw=True)
            return x if x and b"<article" in x else None

        def read_ft(item):  # Europe PMC serves full texts slowly (and 500 for some); PubMed Central's OAI has the same JATS
            i, (d, v) = item
            hh = C.Http(rate=2, tries=1, timeout=60)
            if i % 4 == 3:
                x = hh.get(f"{F.EPMC}/{v['pmcid']}/fullTextXML", raw=True) or ncbi(hh, v["pmcid"])
            else:
                x = ncbi(hh, v["pmcid"]) or hh.get(f"{F.EPMC}/{v['pmcid']}/fullTextXML", raw=True)
            links, text = F.das_links(x) if x else ([], "")
            return d, links, text[:700], x is not None
        todo_ft = [(d, v) for d, v in st["ft"].items() if v["links"] is None]
        with ThreadPoolExecutor(9) as ex:
            for n, (d, links, text, ok) in enumerate(ex.map(read_ft, enumerate(todo_ft)), 1):
                st["ft"][d]["links"], st["ft"][d]["statement"] = (links, text) if ok else ([], "full text not served")
                if n % 300 == 0:
                    save("epmc")
                    C.log(f"  full texts {n}/{len(todo_ft)}")
        save("epmc")
    if a.phase in ("all", "dc"):
        # 2) DataCite: datasets that name one of the papers
        done = set(st["dc_done"])
        todo = [d for d in by_doi if d not in done]
        for i in range(0, len(todo), 25):
            chunk = [{"doi": d} for d in todo[i:i + 25]]
            F.link_datacite(h, chunk)
            for p in chunk:
                if p.get("_links"):
                    st["deposits_dc"].setdefault(p["doi"], [])
                    for u in p["_links"]:
                        if u not in st["deposits_dc"][p["doi"]]:
                            st["deposits_dc"][p["doi"]].append(u)
            st["dc_done"] += todo[i:i + 25]
            if i % 2500 == 0:
                save("dc")
                C.log(f"  datacite {i}/{len(todo)}")
        save("dc")
    if a.phase in ("all", "bx"):
        # 3) bioRxiv/medRxiv preprints: data statements in the JATS full text (bioRxiv limits full texts hard: capped)
        bx = [d for d in by_doi if F.BIORXIV_DOI.match(d) and d not in st["bx"] and d not in st["ft"]]
        for d in bx[: a.limit or 400]:
            p = {"doi": d, "venue": by_doi[d].get("venue", "")}
            F.link_biorxiv(h, [p], [1], cap=1)
            st["bx"][d] = {"links": p.get("_links") or [], "statement": (p.get("_statement") or "")[:700], "published": p.get("published_doi", "")}
        save("bx")
    if a.phase in ("all", "names"):
        # 4) deposits the people made themselves: OpenAlex datasets on their profiles, and DataCite datasets under their names
        titles = [(" ".join(C.norm_title(w["title"])), w["doi"]) for w in papers if w.get("title")]
        titles_by = {}  # a deposit is matched by title only against papers of the same people
        for w in papers:
            for pp in w.get("people") or []:
                titles_by.setdefault(pp, []).append((" ".join(C.norm_title(w.get("title") or "")), w["doi"]))
        own = {}
        for w in works:
            if w["type"] in DATA_TYPES and w["doi"]:
                own.setdefault(w["doi"], {"title": w["title"], "people": w["people"], "via": "OpenAlex dataset on the person's profile"})
        names_done = set(st["names_done"])
        for name, v in res.items():
            if not v["ids"] or name in names_done:
                continue
            fam, giv = split_name(name)
            if not giv:
                continue
            q = f'creators.name:"{fam}, {giv[0]}*" AND types.resourceTypeGeneral:Dataset'
            for x in datacite_dataset_hits(h, q.replace('*"', '"')) + (datacite_dataset_hits(h, f'creators.name:"{fam}, {giv[0][0]}." AND types.resourceTypeGeneral:Dataset') if len(giv[0]) > 1 else []):
                if not any(same_name(name, c if "," not in c else (c.split(",", 1)[1] + " " + c.split(",", 1)[0])) for c in x["creators"]):
                    continue
                o = own.setdefault(x["doi"], {"title": x["title"], "people": [], "via": "DataCite dataset with the person as creator"})
                o["papers"] = x["papers"]
                o["url"] = x["url"]
                if name not in o["people"]:
                    o["people"].append(name)
            st["names_done"].append(name)
            if len(st["names_done"]) % 200 == 0:
                save("names")
                C.log(f"  datacite names {len(st['names_done'])}")
        save("names")
        # dataset -> paper: the deposit's own related identifiers, else a paper of theirs with (nearly) the same title
        unlinked = []
        for ddoi, o in own.items():
            if "papers" not in o:
                o.update({k: v for k, v in (datacite_dataset_hits(h, f'doi:"{ddoi}"', 1) or [{}])[0].items() if k in ("papers", "url")})
            pd = [p for p in o.get("papers") or [] if p]
            if o["via"].startswith("DataCite"):  # found by name only: the paper must be one of the person's own works
                pd = [p for p in pd if p in by_doi]
            if not pd:
                t = " ".join(C.norm_title(o["title"]))
                best = max(((C.title_sim(o["title"], wt) if abs(len(wt) - len(t)) < 0.4 * max(len(t), 1) else 0, wd)
                            for pp in o["people"] for wt, wd in titles_by.get(pp, [])), default=(0, ""))
                if best[0] >= 0.85:
                    pd = [best[1]]
            if not pd:
                unlinked.append({"data_doi": ddoi, "title": o["title"], "people": o["people"], "via": o["via"]})
                continue
            u = o.get("url") or ""
            url = u if C.data_key(u).split(":")[0] in ("zenodo", "figshare", "osf", "openneuro") else "https://doi.org/" + ddoi
            for p in pd:
                st["deposits_own"].setdefault(p, [])
                if url not in st["deposits_own"][p]:
                    st["deposits_own"][p].append(url)
                st["deposit_people"].setdefault(p, sorted(set(o["people"])))
        save("names")
        C.save_json(out / "unlinked_deposits.json", unlinked)
    if a.phase not in ("all", "assemble"):
        print(f"links: phase {a.phase} done")
        return
    st["deposits"] = {}
    for src in (st["deposits_dc"], st["deposits_own"]):
        for k_, v_ in src.items():
            st["deposits"].setdefault(k_, [])
            st["deposits"][k_] += [u for u in v_ if u not in st["deposits"][k_]]
    unlinked = C.load_json(out / "unlinked_deposits.json", [])
    # 5) one candidate per paper
    idx = C.listed_index()
    cands, listed = [], []
    for d in sorted(set(st["deposits"]) | {k for k, v in st["ft"].items() if v.get("links")} | {k for k, v in st["bx"].items() if v.get("links")}):
        w = by_doi.get(d) or {}
        links = []
        for u in (st["ft"].get(d) or {}).get("links") or []:
            links.append((u, "Europe PMC full text, data availability statement"))
        for u in (st["bx"].get(d) or {}).get("links") or []:
            links.append((u, "bioRxiv full text, data availability statement"))
        for u in st["deposits"].get(d) or []:
            links.append((u, "DataCite / the person's own deposit names this paper"))
        data, seenk = [], set()
        for u, via in links:
            k = C.data_key(u)
            if not k or k in seenk or k == "doi:" + d or (k.startswith("doi:") and not C.is_data_doi(k[4:])):
                continue
            seenk.add(k)
            data.append({"url": C.canonical_url(u) if not u.lower().startswith("https://doi.org/") else u, "key": k, "host": C.host_label(u), "via": via})
        if not data:
            continue
        people = sorted(set(w.get("people") or []) | set((st.get("deposit_people") or {}).get(d) or []))
        c = {"cid": d, "doi": d, "title": w.get("title", ""), "year": w.get("year"), "venue": w.get("venue", ""), "type": w.get("type", ""),
             "first": w.get("first", ""), "last": w.get("last", ""), "people": people, "topic": w.get("topic", ""), "data": data,
             "statement": ((st["ft"].get(d) or {}).get("statement") or (st["bx"].get(d) or {}).get("statement") or ""),
             "published_doi": (st["bx"].get(d) or {}).get("published", "")}
        why = ""
        if d in idx["dois"] or (c["published_doi"] and c["published_doi"] in idx["dois"]):
            why = "paper already listed"
        elif any(x["key"] in idx["keys"] for x in data):
            why = "data deposit already listed"
        elif C.listed_title_match(c["title"], idx):
            why = "title already listed (" + C.listed_title_match(c["title"], idx) + ")"
        if why:
            listed.append({"doi": d, "reason": why})
            continue
        cands.append(c)
    C.write_jsonl(out / "candidates.jsonl", cands)
    C.save_json(out / "already_listed.json", listed)
    print(f"links: {len(cands)} candidate papers with data links; {len(listed)} already listed; {len(unlinked)} deposits without a paper")


# ------------------------------------------------------------------ deposits: what is in each linked deposit
def deposit_meta(h, url):
    """Title, description, license and file list of a deposit from its repository API (metadata only)."""
    k = C.data_key(url)
    kind, _, v = k.partition(":")
    m = {"key": k, "title": "", "description": "", "license": "", "files": [], "status": "", "related": []}
    try:
        if kind == "zenodo":
            r = h.get(f"https://zenodo.org/api/records/{v}")
            if r:
                md = r.get("metadata") or {}
                m.update(title=md.get("title", ""), description=C.clean_text(md.get("description", ""), 900),
                         license=(md.get("license") or {}).get("id", ""), files=[(f.get("key"), f.get("size")) for f in r.get("files") or []],
                         status="ok", rtype=(md.get("resource_type") or {}).get("type", ""),
                         related=[x.get("identifier") for x in md.get("related_identifiers") or []][:8], access=md.get("access_right", ""))
        elif kind == "figshare":
            r = h.get(f"https://api.figshare.com/v2/articles/{v}")
            if r:
                m.update(title=r.get("title", ""), description=C.clean_text(r.get("description", ""), 900),
                         license=(r.get("license") or {}).get("name", ""), files=[(f.get("name"), f.get("size")) for f in r.get("files") or []],
                         status="ok", rtype=r.get("defined_type_name", ""), related=[r.get("resource_doi") or ""])
        elif kind == "osf":
            node = h.get(f"https://api.osf.io/v2/nodes/{v}/", {"embed": "license"})
            if node and node.get("data"):
                at = node["data"].get("attributes") or {}
                lic = ((((node["data"].get("embeds") or {}).get("license") or {}).get("data") or {}).get("attributes") or {}).get("name", "")
                m.update(title=at.get("title", ""), description=C.clean_text(at.get("description", ""), 900), license=lic, status="ok",
                         public=at.get("public"))
                summ = __import__("find").deposit_files(h, {"key": k, "files": ""})
                m["files_summary"] = summ
                kids = h.get(f"https://api.osf.io/v2/nodes/{v}/children/", {"page[size]": 20})
                m["components"] = [((c.get("attributes") or {}).get("title", "")) for c in (kids or {}).get("data", [])][:10]
            else:
                reg = h.get(f"https://api.osf.io/v2/registrations/{v}/")
                if reg and reg.get("data"):
                    at = reg["data"].get("attributes") or {}
                    m.update(title=at.get("title", ""), description=C.clean_text(at.get("description", ""), 900), status="ok (registration)")
                else:
                    m["status"] = "not public or not found"
        elif kind == "github":
            import os
            gh = {"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"]} if os.environ.get("GITHUB_TOKEN") else None
            r = h.get(f"https://api.github.com/repos/{v}", headers=gh)
            if r:
                t = h.get(f"https://api.github.com/repos/{v}/git/trees/{r.get('default_branch', 'main')}", {"recursive": 1}, headers=gh)
                m.update(title=r.get("full_name", ""), description=r.get("description") or "", license=(r.get("license") or {}).get("spdx_id") or "",
                         files=[(x.get("path"), x.get("size")) for x in (t or {}).get("tree", []) if x.get("type") == "blob"], status="ok")
        elif kind == "openneuro":
            m["status"] = "ok" if h.get(f"https://openneuro.org/crn/datasets/{v}/snapshots", raw=True) is not None else "not found"
            q = {"query": "query($id: ID!){dataset(id:$id){latestSnapshot{description{Name License} summary{subjects modalities tasks totalFiles size}}}}",
                 "variables": {"id": v}}
            r = h.request("https://openneuro.org/crn/graphql", data=q)
            s_ = ((((r or {}).get("data") or {}).get("dataset") or {}).get("latestSnapshot") or {})
            if s_:
                m.update(title=(s_.get("description") or {}).get("Name", ""), license=(s_.get("description") or {}).get("License", ""),
                         description=json.dumps(s_.get("summary") or {})[:600], status="ok")
        else:
            doi = v if kind == "doi" else ""
            if kind == "mendeley":
                doi = ""
                r = h.get(f"https://data.mendeley.com/public-api/datasets/{v}", tries=2)
                if r:
                    m.update(title=r.get("name", ""), description=C.clean_text(r.get("description", ""), 900),
                             license=((r.get("data_licence") or {}).get("short_name") or ""), status="ok",
                             files=[(f.get("filename"), f.get("size")) for f in r.get("files") or []])
            if doi:
                r = h.get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
                a = ((r or {}).get("data") or {}).get("attributes") or {}
                if a:
                    m.update(title=((a.get("titles") or [{}])[0] or {}).get("title", ""),
                             description=C.clean_text(((a.get("descriptions") or [{}])[0] or {}).get("description", ""), 900),
                             license=((a.get("rightsList") or [{}])[0] or {}).get("rights", "") if a.get("rightsList") else "",
                             status="ok", rtype=((a.get("types") or {}).get("resourceTypeGeneral") or ""), publisher=str(a.get("publisher") or "")[:80],
                             url=a.get("url", ""), related=[x.get("relatedIdentifier") for x in a.get("relatedIdentifiers") or []][:8],
                             sizes=a.get("sizes") or [], formats=a.get("formats") or [])
                    if doi.startswith("10.7910/dvn") or "dataverse" in (a.get("url") or ""):
                        host = urllib.parse.urlparse(a.get("url") or "https://dataverse.harvard.edu").hostname
                        dv = h.get(f"https://{host}/api/datasets/:persistentId/", {"persistentId": "doi:" + doi.upper()})
                        lv = ((dv or {}).get("data") or {}).get("latestVersion") or {}
                        m["files"] = [((f.get("dataFile") or {}).get("filename"), (f.get("dataFile") or {}).get("filesize")) for f in lv.get("files") or []]
                        if isinstance(lv.get("license"), dict):
                            m["license"] = m["license"] or lv["license"].get("name", "")
                    if doi.startswith("10.5061/dryad"):
                        dr = h.get(f"https://datadryad.org/api/v2/datasets/{urllib.parse.quote('doi:' + doi, safe='')}")
                        if dr:
                            m["license"] = m["license"] or dr.get("license", "")
                            m["files_summary"] = f"Dryad deposit, {int(dr.get('storageSize') or 0) / 1e6:.1f} MB"
                else:
                    m["status"] = "no DataCite record"
            elif kind == "physionet":
                r = h.get(f"https://physionet.org/content/{v}/", raw=True)
                m["status"] = "ok" if r else "not found"
    except Exception as e:  # noqa: BLE001
        m["status"] = f"error {type(e).__name__}"
    if m["files"] and not m.get("files_summary"):
        m["files_summary"] = __import__("find").files_summary(m["files"])
    m["file_names"] = [str(n) for n, _ in m["files"][:25]]
    m.pop("files", None)
    return m


def cmd_deposits(a):
    out = Path(a.out)
    cands = C.read_jsonl(out / "candidates.jsonl")
    cache_p = out / "deposits.json"
    cache = C.load_json(cache_p, {})
    from concurrent.futures import ThreadPoolExecutor
    groups = {}
    for c in cands:
        for d in c["data"][:4]:
            if d["key"] not in cache:
                groups.setdefault(d["key"].split(":")[0], {})[d["key"]] = d["url"]
    lock = threading.Lock()

    def run(kind):  # one worker per repository, each polite on its own host
        hh = C.Http(rate=3)
        for i, (k, u) in enumerate(groups[kind].items(), 1):
            m = deposit_meta(hh, u)
            with lock:
                cache[k] = m
                if i % 100 == 0:
                    C.save_json(cache_p, cache)
                    C.log(f"  deposits {kind} {i}/{len(groups[kind])}")
    with ThreadPoolExecutor(len(groups) or 1) as ex:
        list(ex.map(run, list(groups)))
    C.save_json(cache_p, cache)
    print(f"deposits: {len(cache)} deposits described")


# ------------------------------------------------------------------ prep: automatic rejections, then batches for screening
CODE_EXT = {"py", "m", "r", "ipynb", "js", "c", "cpp", "h", "java", "jl", "sh", "yml", "yaml", "toml", "cfg", "gitignore", "lock",
            "md", "txt", "rst", "html", "css", "json", "license", "in", "ini", "rmd", "qmd", "(none)"}
DATA_EXT = {"csv", "tsv", "mat", "xlsx", "xls", "dat", "h5", "hdf5", "nwb", "c3d", "edf", "bdf", "set", "fdt", "trc", "npy", "npz",
            "pkl", "rds", "rdata", "sav", "parquet", "feather", "zip", "gz", "tar", "7z", "rar", "wav", "mot", "sto", "xdf", "vhdr",
            "eeg", "nii", "tdms", "asc", "edf+", "pickle", "mvnx", "bvh", "tsv.gz", "dta", "jsonl", "acq", "smr", "mp4", "avi", "mov"}


def exts_of(meta):
    names = meta.get("file_names") or []
    ex = {Path(n.lower()).suffix.lstrip(".") or "(none)" for n in names}
    m = re.match(r"\d+ files: (.*?)(;|$)", meta.get("files_summary") or "")
    if m:
        ex |= set(re.findall(r"\d+ \.([a-z0-9()+]+)", m.group(1)))
    return ex


MOTOR_TEXT = re.compile(r"\b(motor|movement|moving|reach|grasp|grip|gait|walk|locomot|run(ning|ner)|saccad|eye[- ]movement|gaze|pursuit|"
                        r"oculomotor|speech|speak|articulat|vocal|voice|phonat|postur|balance|sway|stance|electromyo|EMG|kinemat|"
                        r"force|torque|tapping|tap|synchroni[sz]ation|rhythm|timing|sequence|skill|handwrit|drawing|typing|keystroke|"
                        r"pointing|aiming|tracking|adaptation|finger|hand|arm|leg|limb|muscle|tremor|parkinson|ataxia|dystonia|stroke|"
                        r"rehabilitat|exoskelet|prosthe|bimanual|throw|kick|jump|swim|cycling|pedal|dexterity|manipulat|lifting|"
                        r"action|response inhibition|stop-signal|joystick|cursor|brain-computer|steering|driving|sport|exercise|"
                        r"head|trunk|foot|ankle|knee|hip|wrist|elbow|shoulder)", re.I)


def auto_reject(c, deps):
    import find as F
    why = F.off_topic(c["title"], c.get("abstract", ""))
    if why == "animal":
        return "not human: animal study"
    if not MOTOR_TEXT.search(c["title"] + " " + c.get("abstract", "")):
        return "not motor: no movement terms in the title or abstract"
    ok = []
    for d in c["data"]:
        if d.get("problem"):
            continue
        m = deps.get(d["key"]) or {}
        st = m.get("status", "")
        ex = exts_of(m)
        if st and not st.startswith("ok"):
            d["problem"] = "data link does not resolve (" + st + ")"
        elif m.get("rtype") and m["rtype"].lower() in ("software", "publication", "presentation", "poster", "image", "video", "text", "journalarticle", "preprint"):
            d["problem"] = f"deposit is {m['rtype']}, not data"
        elif d["key"].startswith("github:") and ex and not (ex & DATA_EXT):
            d["problem"] = "code only (GitHub repository without data files)"
        elif ex and ex <= (F.DOC_EXT | {"(none)"}):
            d["problem"] = "documents only, no data files"
        elif re.search(r"restricted|closed access|embargo|metadata only|request", str(m.get("license") or ""), re.I):
            d["problem"] = f"not public: {m.get('license')}"
        elif m.get("access") and m["access"] != "open":
            d["problem"] = f"access {m['access']}"
        else:
            ok.append(d)
    if not ok:
        return "; ".join(sorted({d.get("problem", "") for d in c["data"]}))[:200] or "no usable data link"
    c["data"] = ok + [d for d in c["data"] if d not in ok]
    return ""


def cmd_prep(a):
    out = Path(a.out)
    cands = C.read_jsonl(out / "candidates.jsonl")
    deps = C.load_json(out / "deposits.json", {})
    h = C.Http(rate=4)
    abs_p = out / "abstracts_cache.json"
    abst = C.load_json(abs_p, {})
    keep, rej = [], []
    # a deposit linked to several papers (DataCite "references"/"cites" links, journal supplements of another article)
    # stays only with the paper whose title it carries
    users = {}
    for c in cands:
        for d in c["data"]:
            users.setdefault(d["key"], []).append(c)
    for k, cs in users.items():
        t = (deps.get(k) or {}).get("title") or ""
        t2 = re.sub(r"^(additional file|supplementary|supplemental|table|data[ _]sheet|figure|appendix)[^:]*?\bof\b\s*", "", t, flags=re.I)
        supp = t2 != t
        if len(cs) < 2 and not supp:
            continue
        best = max(cs, key=lambda c: C.title_sim(t2, c["title"]))
        for c in cs:
            if (c is not best and len(cs) > 1) or (supp and C.title_sim(t2, c["title"]) < 0.6):
                for d in c["data"]:
                    if d["key"] == k:
                        d["problem"] = "the deposit belongs to another paper"
    for i, c in enumerate(cands):
        if c["doi"] not in abst:
            w = h.get(f"{OA}/works/doi:{urllib.parse.quote(c['doi'])}", C.openalex_params({"select": "abstract_inverted_index,title"}))
            ai = (w or {}).get("abstract_inverted_index") or {}
            pos = sorted((p_, t) for t, ps in ai.items() for p_ in ps)
            abst[c["doi"]] = " ".join(t for _, t in pos)[:2500]
            if i % 200 == 0:
                C.save_json(abs_p, abst)
        c["abstract"] = abst[c["doi"]]
        why = auto_reject(c, deps)
        if why:
            rej.append({"doi_or_url": c["doi"], "reason": why})
            continue
        for d in c["data"][:4]:
            m = deps.get(d["key"]) or {}
            d.update({k: m.get(k) for k in ("title", "description", "license", "files_summary", "file_names", "components", "rtype") if m.get(k)})
            if d.get("description"):
                d["description"] = d["description"][:500]
            if d.get("file_names"):
                d["file_names"] = d["file_names"][:15]
        c["data"] = c["data"][:4]
        keep.append(c)
    C.save_json(abs_p, abst)
    sd = out / "screen"
    sd.mkdir(exist_ok=True)
    old = sorted(f for f in sd.glob("batch_*.jsonl") if ".decisions" not in f.name)
    have = {c["cid"] for f in old for c in C.read_jsonl(f)}
    keep = [c for c in keep if c["cid"] not in have]  # only new candidates get new batches
    size, start = 60, len(old)
    for b in range(0, len(keep), size):
        C.write_jsonl(sd / f"batch_{start + b // size + 1:03d}.jsonl", keep[b:b + size])
    prev = {r["doi_or_url"] for r in C.read_jsonl(out / "auto_rejected.jsonl")} if (out / "auto_rejected.jsonl").exists() else set()
    C.write_jsonl(out / "auto_rejected.jsonl", (C.read_jsonl(out / "auto_rejected.jsonl") if prev else []) +
                  [r for r in rej if r["doi_or_url"] not in prev])
    print(f"prep: {len(keep)} to screen in {(len(keep) + size - 1) // size} batches; {len(rej)} rejected automatically")


# ------------------------------------------------------------------ build: screened-in candidates -> source/extra_studies.json
ORDER = ["id", "label", "year", "papers", "data_url", "host", "license", "license_note", "formats", "n_participants", "task",
         "methods", "populations", "lab", "extra_tags"]


def vocab(extra):
    v = {"task": set(C.TASKS), "methods": set(), "populations": set(), "extra": set()}
    for e in extra:
        v["methods"] |= set(e.get("methods") or [])
        v["populations"] |= set(e.get("populations") or [])
        v["extra"] |= {tuple(t) for t in e.get("extra_tags") or []}
    return v


def label_accented(families, year):
    f = [x for x in families if x] or ["Anonymous"]
    return f"{f[0]} {year}" if len(f) == 1 else f"{f[0]} & {f[1]} {year}" if len(f) == 2 else f"{f[0]} et al. {year}"


def lab_accented(h, doi, families):
    w = h.get(f"{OA}/works/doi:{urllib.parse.quote(doi)}", C.openalex_params({"select": "authorships"}))
    au = (w or {}).get("authorships") or []
    last = next((x for x in reversed(au) if x.get("author_position") == "last"), au[-1] if au else None)
    fam = families[-1] if families else ""
    inst = ""
    if last:
        insts = last.get("institutions") or []
        inst = insts[0].get("display_name", "") if insts else ""
        if not fam:
            fam = ((last.get("author") or {}).get("display_name", "") or " ").split()[-1]
    return f"{fam} ({inst})" if inst else fam


ABBR = {"journal": "J", "biomechanics": "Biomech", "neuroscience": "Neurosci", "neurophysiology": "Neurophysiol", "experimental": "Exp",
        "research": "Res", "human": "Hum", "movement": "Mov", "science": "Sci", "sciences": "Sci", "behavior": "Behav",
        "behaviour": "Behav", "behavioral": "Behav", "psychology": "Psychol", "rehabilitation": "Rehabil", "engineering": "Eng",
        "medicine": "Med", "clinical": "Clin", "physiology": "Physiol", "biology": "Biol", "computational": "Comput",
        "communications": "Commun", "reports": "Rep", "frontiers": "Front", "applied": "Appl", "systems": "Syst",
        "international": "Int", "american": "Am", "society": "Soc", "transactions": "Trans", "proceedings": "Proc",
        "language": "Lang", "hearing": "Hear", "acoustical": "Acoust", "cognitive": "Cogn", "cognition": "Cogn", "motor": "Mot",
        "exercise": "Exerc", "neurology": "Neurol", "neurological": "Neurol", "disorders": "Disord", "annals": "Ann",
        "neuroengineering": "Neuroeng", "neuroimage": "NeuroImage", "perception": "Percept", "performance": "Perform",
        "physical": "Phys", "therapy": "Ther", "medical": "Med", "biomedical": "Biomed", "electromyography": "Electromyogr",
        "kinesiology": "Kinesiol", "orthopaedic": "Orthop", "orthopedic": "Orthop", "sport": "Sport", "sports": "Sports",
        "national": "Natl", "academy": "Acad", "united": "U", "states": "S", "royal": "R", "letters": "Lett", "reviews": "Rev", "integrative": "Integr", "comparative": "Comp", "annual": "Annu",
        "psychological": "Psychol", "neuropsychologia": "Neuropsychologia", "developmental": "Dev", "development": "Dev",
        "child": "Child", "speech": "Speech", "neural": "Neural", "vision": "Vis", "visual": "Vis", "ophthalmology": "Ophthalmol",
        "otolaryngology": "Otolaryngol", "attention": "Atten", "quarterly": "Q", "review": "Rev", "advances": "Adv",
        "technology": "Technol", "robotics": "Robot", "automation": "Autom", "intelligent": "Intell", "computing": "Comput",
        "computer": "Comput", "health": "Health", "aging": "Aging", "ageing": "Ageing", "gerontology": "Gerontol",
        "biological": "Biol", "cerebral": "Cereb", "cortex": "Cortex", "brain": "Brain", "stimulation": "Stimul"}


def abbreviate(full):
    words = [w for w in re.split(r"[\s:,&]+", full or "") if w and w.lower() not in ("of", "the", "and", "for", "in", "on", "&")]
    return " ".join(ABBR.get(w.lower(), w) for w in words)


def cmd_build(a):
    import listing as L
    out = Path(a.out)
    ep = C.HERE / "source/extra_studies.json"
    extra = json.load(open(ep, encoding="utf-8"))
    voc = vocab(extra)
    idx = C.listed_index()
    vmap = L.venue_map(idx["extra"])
    taken = set(idx["ids"])
    deps = C.load_json(out / "deposits.json", {})
    cands = {c["cid"]: c for f in sorted((out / "screen").glob("batch_*.jsonl")) if ".decisions" not in f.name for c in C.read_jsonl(f)}
    decided = [d for f in sorted((out / "screen").glob("batch_*.decisions.jsonl")) for d in C.read_jsonl(f)]
    built_p = out / "built.json"
    built = C.load_json(built_p, {"done": [], "rejected": []})
    h = C.Http(rate=3)
    new, added_rows = [], []
    keys_new = {}
    for d in decided:
        cid = d.get("cid")
        if cid in built["done"] or cid not in cands:
            continue
        built["done"].append(cid)
        c = cands[cid]
        if d.get("include") != "yes":
            built["rejected"].append({"doi_or_url": cid, "reason": d.get("reason", "screened out")[:250]})
            continue
        url = d.get("data_url") or c["data"][0]["url"]
        k = C.data_key(url)
        meta = L.paper_meta(h, c["doi"])
        if not meta:
            built["rejected"].append({"doi_or_url": cid, "reason": "paper DOI has no Crossref or DataCite record"})
            continue
        jd = meta["journal_doi"] or C.norm_doi(c.get("published_doi"))
        if meta["preprint"] and jd and jd != c["doi"]:
            jm = L.paper_meta(h, jd)
            if jm:
                meta = jm
        if not meta["preprint"] and meta["venue_full"] and (not meta["short"] or (meta["short"] == meta["venue_full"] and len(meta["short"]) > 14)):
            meta["short"] = abbreviate(meta["venue_full"])
            meta["venue"] = meta["short"] + meta["venue"][len(meta["venue_full"]):]
        if meta["doi"] in idx["dois"]:
            built["rejected"].append({"doi_or_url": cid, "reason": f"journal version {meta['doi']} already listed"})
            continue
        if k in idx["keys"]:
            built["rejected"].append({"doi_or_url": cid, "reason": "data deposit already listed"})
            continue
        if k in keys_new:  # the same deposit serves two papers: one listing with both papers
            e = keys_new[k]
            if meta["doi"] not in [p_["doi"] for p_ in e["papers"]]:
                e["papers"].append({"authors": meta["authors"], "year": meta["year"], "title": meta["title"], "venue": meta["venue"], "doi": meta["doi"]})
                for r in added_rows:
                    if r["id"] == e["id"]:
                        r["doi"] += " " + meta["doi"]
                        r["people"] = "; ".join(sorted(set(r["people"].split("; ")) | set(c["people"])))
            continue
        page = h.get(url, raw=True, tries=2, timeout=40)
        api_ok = (deps.get(k) or {}).get("status", "").startswith("ok")
        if page is None and not api_ok:
            built["rejected"].append({"doi_or_url": url, "reason": "data URL does not respond"})
            continue
        if re.fullmatch(r"\s*open access\s*", str((deps.get(k) or {}).get("license") or ""), re.I):
            (deps.get(k) or {})["license"] = ""  # "Open Access" is an access level, not a license
        lr = str((deps.get(k) or {}).get("license") or "") + " " + str((deps.get(k) or {}).get("access") or "")
        if re.search(r"restricted|closed|embargo|metadata only|on request", lr, re.I):
            built["rejected"].append({"doi_or_url": url, "reason": f"not public: the repository lists {lr.strip()}"})
            continue
        task = d.get("task")
        if task not in voc["task"]:
            built["rejected"].append({"doi_or_url": cid, "reason": f"no valid task tag ({task})"})
            continue
        methods = [m for m in d.get("methods") or [] if m in voc["methods"]]
        pops = [m for m in d.get("populations") or [] if m in voc["populations"]]
        xt = [list(t) for t in d.get("extra_tags") or [] if tuple(t) in voc["extra"]]
        dm = deps.get(k) or {}
        year = int(meta["year"]) if meta["year"] else c.get("year")
        e = {"id": L.make_id(meta, vmap, taken), "label": label_accented(meta["families"], year), "year": year,
             "papers": [{"authors": meta["authors"], "year": year, "title": meta["title"], "venue": meta["venue"], "doi": meta["doi"]}],
             "data_url": url, "host": C.host_label(url) if C.host_label(url) != "other" else (d.get("host") or "other"),
             "license": L.norm_license(dm.get("license") or ""), "license_note": None,
             "formats": L.formats_of(dm.get("file_names") or [x for x in re.findall(r"\d+ (\.[a-z0-9]+)", dm.get("files_summary") or "")]),
             "n_participants": d.get("n_participants") if isinstance(d.get("n_participants"), int) else None, "task": task,
             "methods": methods, "populations": pops, "lab": lab_accented(h, meta["doi"], meta["families"]), "extra_tags": xt}
        e = {kk: e[kk] for kk in ORDER}
        new.append(e)
        keys_new[k] = e
        idx["dois"].add(meta["doi"])
        added_rows.append({"id": e["id"], "people": "; ".join(c["people"]), "doi": meta["doi"], "data_url": url, "task": task})
    if new and not a.dry:
        extra += new
        ep.write_text(json.dumps(extra, indent=1, ensure_ascii=False), encoding="utf-8")
        ap_ = out / "added.csv"
        first = not ap_.exists()
        with open(ap_, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "people", "doi", "data_url", "task"])
            if first:
                w.writeheader()
            w.writerows(added_rows)
    if not a.dry:
        C.save_json(built_p, built)
    print(f"build: {len(new)} new entries{' (dry run)' if a.dry else ''}; {len(built['rejected'])} rejected so far")
    for e in new[:5]:
        print(f"  {e['id']}  {e['task']}  {e['data_url']}")


# ------------------------------------------------------------------ report: coverage.csv and rejected.csv
def cmd_report(a):
    out = Path(a.out)
    roster = list(csv.DictReader(open(out / "roster.csv", encoding="utf-8")))
    res = C.load_json(out / "resolved.json", {})
    works = C.read_jsonl(out / "works.jsonl") if (out / "works.jsonl").exists() else []
    cands = C.read_jsonl(out / "candidates.jsonl") if (out / "candidates.jsonl").exists() else []
    added = list(csv.DictReader(open(out / "added.csv", encoding="utf-8"))) if (out / "added.csv").exists() else []
    nw, nd, na = {}, {}, {}
    for w in works:
        for p in w["people"]:
            nw[p] = nw.get(p, 0) + 1
    for c in cands:
        for p in c["people"]:
            nd[p] = nd.get(p, 0) + 1
    for r in added:
        for p in r["people"].split("; "):
            na[p] = na.get(p, 0) + 1
    with open(out / "coverage.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["roster person", "openalex id", "how resolved", "works checked", "datasets found", "added"])
        for r in roster:
            v = res.get(r["name"]) or {"ids": [], "how": "not resolved"}
            w.writerow([r["name"], " ".join(v["ids"]) or "not resolved", v["how"], nw.get(r["name"], 0), nd.get(r["name"], 0), na.get(r["name"], 0)])
    rows = [(x["doi_or_url"], x["reason"]) for x in C.read_jsonl(out / "auto_rejected.jsonl")] if (out / "auto_rejected.jsonl").exists() else []
    rows += [(x["doi_or_url"], x["reason"]) for x in C.load_json(out / "built.json", {"rejected": []})["rejected"]]
    rows += [(x["doi"], "already listed on OpenMotor (" + x["reason"] + ")") for x in C.load_json(out / "already_listed.json", [])]
    rows += [("https://doi.org/" + x["data_doi"], "deposit with no paper DOI linked to it (" + x["via"] + ")")
             for x in C.load_json(out / "unlinked_deposits.json", []) if x.get("via", "").startswith("OpenAlex")]
    with open(out / "rejected.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["doi or url", "reason"])
        w.writerows(rows)
    print(f"report: coverage {len(roster)} people; rejected.csv {len(rows)} rows")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["roster", "resolve", "works", "links", "deposits", "prep", "build", "report"])
    ap.add_argument("--out", default=str(C.HERE / "work" / "ncm"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--phase", default="all", choices=["all", "epmc", "dc", "bx", "names", "assemble"])
    ap.add_argument("--since", default="2010-01-01", help="works step: earliest publication date (OpenAlex credit budget)")
    a = ap.parse_args()
    globals()["cmd_" + a.cmd](a)


if __name__ == "__main__":
    main()
