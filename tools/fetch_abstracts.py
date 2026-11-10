"""Fetch the abstract of every paper in site/catalog.json by DOI and save source/abstracts.json.

Sources, in order: Europe PMC (journals and bioRxiv/medRxiv preprints), Crossref (publisher-deposited JATS),
OpenAlex (inverted index), Semantic Scholar. The first source whose title matches the catalog title wins.
Text is cleaned to plain paragraphs; typographic quotes and dashes become plain ASCII, as on the rest of the site.
Re-running keeps abstracts already fetched unless --refresh is given.
"""
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "source/abstracts.json"
UA = {"User-Agent": "OpenMotor-site/1.0 (abstract lookup for a public data catalog)"}


def get(url, accept=None):
    h = dict(UA)
    if accept:
        h["Accept"] = accept
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (404, 400):
                return None
            time.sleep(2 + 3 * attempt)
        except Exception:
            time.sleep(2 + 3 * attempt)
    return None


def clean(t):
    t = re.sub(r"<(jats:)?title>\s*(abstract|summary)\s*</(jats:)?title>", " ", t, flags=re.I)
    t = re.sub(r"</(jats:)?(p|sec)>", "\n\n", t)
    t = re.sub(r"<(jats:)?title>(.*?)</(jats:)?title>", r"\n\n\2: ", t)
    t = re.sub(r"<h\d>(.*?)</h\d>", r"\n\n\1: ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = (t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
          .replace("—", " - ").replace("–", "-").replace("−", "-").replace(" ", " ")
          .replace(" ", " ").replace(" ", " "))
    t = re.sub(r"^\s*(abstract|summary)\s*[:.]?\s*", "", t, flags=re.I)
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", t)]
    paras = [re.sub(r"\s+([,.;:)])", r"\1", p).replace("( ", "(") for p in paras if p]
    return "\n\n".join(p.replace(" - -", " -") for p in paras)


def same_title(a, b):
    n = lambda x: re.sub(r"[^a-z0-9 ]", "", html.unescape(re.sub(r"<[^>]+>", "", x or "")).lower())
    return SequenceMatcher(None, n(a), n(b)).ratio() >= 0.8


def europepmc(doi):
    q = urllib.parse.quote(f'DOI:"{doi}"')
    j = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&resultType=core&format=json&pageSize=5")
    for r in (j or {}).get("resultList", {}).get("result", []):
        if (r.get("doi") or "").lower() == doi.lower() and r.get("abstractText"):
            return r.get("title"), r["abstractText"]
    return None


def crossref(doi):
    j = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
    m = (j or {}).get("message") or {}
    if m.get("abstract"):
        return (m.get("title") or [""])[0], m["abstract"]
    return None


def openalex(doi):
    j = get("https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi))
    inv = (j or {}).get("abstract_inverted_index")
    if not inv:
        return None
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return j.get("title"), " ".join(w for _, w in pos)


def s2(doi):
    j = get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields=title,abstract")
    if j and j.get("abstract"):
        return j.get("title"), j["abstract"]
    return None


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    have = {} if refresh or not OUT.exists() else json.load(open(OUT))
    catalog = json.load(open(HERE / "site/catalog.json"))
    papers = [p for s in catalog["studies"] for p in s["papers"] if (p.get("doi") or "").startswith("10.")]
    extra = HERE / "source/extra_studies.json"
    if extra.exists():
        papers += [p for s in json.load(open(extra)) for p in s["papers"] if (p.get("doi") or "").startswith("10.")]
    for p in papers:
        doi = p["doi"]
        if doi in have and have[doi].get("text"):
            continue
        got = None
        for name, fn in [("Europe PMC", europepmc), ("Crossref", crossref), ("OpenAlex", openalex), ("Semantic Scholar", s2)]:
            r = fn(doi)
            if r and r[1] and len(clean(r[1])) > 200:
                ok = same_title(r[0], p["title"])
                if ok:
                    got = {"text": clean(r[1]), "source": name}
                    break
                print(f"  title mismatch from {name} for {doi}: {r[0]!r}")
            time.sleep(0.3)
        have[doi] = got or {"text": "", "source": None}
        print(f"{'OK ' if got else '-- '} {doi}  {(got or {}).get('source')}  {p['title'][:60]}", flush=True)
        json.dump(have, open(OUT, "w"), indent=1, ensure_ascii=False)
    n = sum(1 for p in papers if have.get(p["doi"], {}).get("text"))
    print(f"{n} of {len(papers)} papers have an abstract")
