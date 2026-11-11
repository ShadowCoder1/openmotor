"""Weekly finder, step 1 (deterministic, no AI): find newly shared human motor control / motor learning datasets.

usage: python3 -I tools/finder/find.py [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--state PATH] [--out DIR]
                                       [--sources zenodo,figshare,osf,datacite,europepmc,openalex] [--max-fulltext N]

Sources: Zenodo, Figshare, OSF (projects and preprints), DataCite (datasets, and datasets linked to a paper DOI),
Europe PMC (papers and preprints; data availability statements of open full texts, bioRxiv/medRxiv JATS, OSF preprint
data links), OpenAlex (new works by the last authors of listed studies; works citing key papers).
A candidate needs a paper or preprint DOI AND a public data location. Data deposits without a paper are held back
(and rechecked on later runs). Everything already listed on the site, or seen by an earlier run, is dropped.

Writes <out>/candidates.jsonl, <out>/held.jsonl, <out>/find_summary.json and <out>/state.json (the next state; the
weekly workflow commits it to source/finder/state.json after publishing). Reads --state (default source/finder/state.json)."""
import argparse
import datetime as dt
import re
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from terms import ANIMAL, GENERIC, HUMAN, KEY_PAPERS, OFF_TOPIC, OSF_TERMS, PAPER_TERMS, TERMS  # noqa: E402

ALL_PHRASES = [p for ps in TERMS.values() for p in ps]
PAPER_REL = {"issupplementto", "isreferencedby", "iscitedby", "isdescribedby", "isdocumentedby", "issourceof", "isderivedfrom",
             "ispartof", "references", "cites", "isrequiredby", "documents", "describes"}
STRONG_REL = {"issupplementto", "isreferencedby", "iscitedby", "isdescribedby", "isdocumentedby", "documents", "describes"}
VERSION_REL = {"isversionof", "hasversion", "isidenticalto", "isnewversionof", "ispreviousversionof"}
HOST_RANK = ["osf", "zenodo", "figshare", "doi", "openneuro", "physionet", "mendeley", "gin", "dandi", "github", "url"]
DAS_TITLE = re.compile(r"data\s*(and|&)?\s*(code|materials?|software)?\s*(availability|accessibility|sharing|access|deposition|statement)|"
                       r"availability of (data|supporting data|materials)|open (data|practices|science)|code and data|"
                       r"data,? (code|materials)", re.I)
DATA_SENT = re.compile(r"\b(data|dataset|datasets|data set|raw data|code)\b[^.]{0,200}\b(available|deposited|shared|accessible|"
                       r"found|uploaded|archived|stored|provided|posted|downloaded|hosted)", re.I)
REPO_TERMS = '("osf.io" OR "zenodo" OR "figshare" OR "github.com" OR "dataverse" OR "dryad" OR "physionet" OR "openneuro" OR "mendeley data" OR "g-node" OR "dandiarchive")'
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
OSF_PREPRINT_DOI = re.compile(r"^10\.(31234|31219|31236|31235|51224|31222|31730|35542)/(osf\.io/)?([a-z0-9]{5})(_v\d+)?", re.I)
BIORXIV_DOI = re.compile(r"^10\.(1101|64898)/")


def phrases_in(text, phrases=ALL_PHRASES):
    """Phrases found in the text; a match on generic phrases only counts when the text also mentions people."""
    t = (text or "").lower()
    found = [p for p in phrases if p.lower() in t]
    if found and all(p in GENERIC for p in found) and not re.search(HUMAN, t, re.I):
        return []
    return found


def task_of(phrase):
    for tag, ps in TERMS.items():
        if phrase in ps:
            return tag
    return ""


def off_topic(title, text):
    if re.search(ANIMAL, title or "", re.I) and not re.search(r"\bhumans?\b", title or "", re.I):
        return "animal"
    if re.search(OFF_TOPIC, (title or "") + " " + (text or "")[:600], re.I):
        return "off topic"
    if re.search(ANIMAL, (text or "")[:1500], re.I) and not re.search(HUMAN, text or "", re.I):
        return "animal"
    return ""


def groups(size=12):
    for tag, ps in TERMS.items():
        for i in range(0, len(ps), size):
            yield tag, ps[i:i + size]


def hit(source, why, paper=None, data=None, statement=""):
    return {"source": source, "why": why, "paper": paper, "data": data or [], "statement": statement}


def data_rec(url, title="", description="", license="", files="", created="", extra_keys=(), meta=None):
    k = C.data_key(url)
    return {"url": C.canonical_url(url) if not url.lower().startswith("https://doi.org/") else url, "key": k,
            "keys": sorted({k, *[x for x in extra_keys if x]}), "title": C.clean_text(title, 300),
            "description": C.clean_text(description, 1500), "license": license or "", "files": files or "", "created": created,
            "host": C.host_label(url), "meta": meta or {}}


def paper_rec(doi, title="", abstract="", year=None, venue="", authors="", published="", pmcid="", via=""):
    return {"doi": C.norm_doi(doi), "title": C.clean_text(title, 400), "abstract": C.clean_text(abstract, 4000), "year": year,
            "venue": C.clean_text(venue, 200), "authors": C.clean_text(authors, 400), "published": published or "", "pmcid": pmcid or "",
            "via": via}


def files_summary(files):
    """files: list of (name, size or None)."""
    if not files:
        return ""
    ext = {}
    total = 0
    for name, size in files:
        e = Path(str(name)).suffix.lower().lstrip(".") or "(none)"
        ext[e] = ext.get(e, 0) + 1
        total += int(size or 0)
    parts = ", ".join(f"{n} .{e}" for e, n in sorted(ext.items(), key=lambda kv: -kv[1])[:8])
    size = f"; {total / 1e6:.1f} MB" if total else ""
    return f"{len(files)} files: {parts}{size}"


DOC_EXT = {"pdf", "doc", "docx", "odt", "rtf", "tif", "tiff", "png", "jpg", "jpeg", "gif", "svg", "eps", "ppt", "pptx", "mp4",
           "avi", "mov", "wmv", "webm", "html", "htm", "md"}


def documents_only(files):
    """True when the file summary lists only documents, images or videos (a journal's supplementary text, not data)."""
    m = re.match(r"\d+ files: (.*?)(;|$)", files or "")
    if not m:
        return False
    exts = re.findall(r"\d+ \.([a-z0-9()]+)", m.group(1))
    return bool(exts) and all(e in DOC_EXT for e in exts)


def paper_dois_from_related(rels, kind_field="relationType", id_field="relatedIdentifier", type_field="relatedIdentifierType"):
    strong, weak, versions = [], [], []
    for r in rels or []:
        rel = str(r.get(kind_field) or r.get("relation") or "").lower()
        ident = r.get(id_field) or r.get("identifier") or ""
        typ = str(r.get(type_field) or r.get("scheme") or "").lower()
        if typ not in ("doi", "") and "doi.org" not in str(ident).lower():
            if typ == "url" and C.repo_links(ident):
                versions.append(C.data_key(ident)) if rel in VERSION_REL else None
            continue
        d = C.norm_doi(ident)
        if not d:
            continue
        if C.is_data_doi(d):
            if rel in VERSION_REL:
                versions.append(C.data_key("https://doi.org/" + d))
            continue
        if rel in STRONG_REL:
            strong.append(d)
        elif rel in PAPER_REL:
            weak.append(d)
    return strong, weak, versions


# ------------------------------------------------------------------ sources: data first
def src_zenodo(h, since, until, ctx):
    out = {}
    for tag, ps in groups(10):
        q = "(" + " OR ".join(f'"{p}"' for p in ps) + f") AND created:[{since} TO {until}]"
        for page in range(1, 11):  # anonymous requests may ask for at most 25 records per page
            d = h.get("https://zenodo.org/api/records", {"q": q, "type": "dataset", "sort": "newest", "size": 25, "page": page})
            hits = (d or {}).get("hits", {}).get("hits", [])
            for r in hits:
                m = r.get("metadata", {})
                text = m.get("title", "") + " " + C.clean_text(m.get("description", "")) + " " + " ".join(m.get("keywords") or [])
                ph = phrases_in(text, ps)
                if not ph or str(r.get("id")) in out:
                    continue
                strong, weak, versions = paper_dois_from_related(m.get("related_identifiers"), "relation", "identifier", "scheme")
                desc_dois = [x for x in C.dois_in(C.clean_text(m.get("description", ""))) if not C.is_data_doi(x)]
                files = files_summary([(f.get("key"), f.get("size")) for f in r.get("files") or []])
                url = f"https://zenodo.org/records/{r['id']}"
                keys = versions + ([f"zenodo:{r['conceptrecid']}"] if r.get("conceptrecid") else [])
                dr = data_rec(url, m.get("title", ""), m.get("description", ""), (m.get("license") or {}).get("id", ""), files,
                              r.get("created", "")[:10], keys,
                              {"creators": [c.get("name", "") for c in m.get("creators") or []][:8], "paper_dois": strong,
                               "weak_dois": weak + desc_dois})
                out[str(r["id"])] = hit("zenodo", f"Zenodo dataset matching '{ph[0]}'", data=[dr])
            if len(hits) < 25:
                break
    return list(out.values())


def src_datacite(h, since, until, ctx):
    out = {}
    for tag, ps in groups(12):
        P = " OR ".join(f'"{p}"' for p in ps)
        q = f"(titles.title:({P}) OR descriptions.description:({P}) OR subjects.subject:({P})) AND registered:[{since} TO {until}]"
        url = "https://api.datacite.org/dois?" + urllib.parse.urlencode({"query": q, "resource-type-id": "dataset", "page[size]": 100,
                                                                         "page[cursor]": 1})
        for _ in range(5):
            d = h.get(url)
            if not d:
                break
            for x in d.get("data", []):
                a = x.get("attributes", {})
                doi = C.norm_doi(a.get("doi"))
                title = " ".join(t.get("title", "") for t in a.get("titles") or [])[:400]
                desc = " ".join(t.get("description", "") or "" for t in a.get("descriptions") or [])
                subj = " ".join(s.get("subject", "") for s in a.get("subjects") or [])
                ph = phrases_in(title + " " + desc + " " + subj, ps)
                if not ph or doi in out:
                    continue
                strong, weak, versions = paper_dois_from_related(a.get("relatedIdentifiers"))
                desc_dois = [y for y in C.dois_in(desc) if not C.is_data_doi(y)]
                lic = " ".join((r.get("rightsIdentifier") or r.get("rights") or "") for r in a.get("rightsList") or [])[:80]
                link = a.get("url") or ""
                url_ = link if C.data_key(link).split(":")[0] in ("zenodo", "figshare", "osf", "openneuro", "dandi") else "https://doi.org/" + doi
                keys = versions + [C.data_key("https://doi.org/" + doi), C.data_key(link)]
                pub = a.get("publisher")
                pub = pub.get("name") if isinstance(pub, dict) else pub
                dr = data_rec(url_, title, desc, lic, "", (a.get("registered") or "")[:10], keys,
                              {"creators": [c.get("name", "") for c in a.get("creators") or []][:8], "paper_dois": strong,
                               "weak_dois": weak + desc_dois, "publisher": pub or "",
                               "sizes": a.get("sizes") or [], "formats": a.get("formats") or []})
                out[doi] = hit("datacite", f"DataCite dataset ({pub}) matching '{ph[0]}'", data=[dr])
            nxt = (d.get("links") or {}).get("next")
            if not nxt or len(d.get("data", [])) < 100:
                break
            url = nxt
    return list(out.values())


def src_figshare(h, since, until, ctx):
    out = {}
    details_left = 80
    for tag, ps in TERMS.items():
        for p in ps:
            if len(p) < 8 or p in ("motor control", "motor performance", "hand movements", "arm movements"):
                continue  # Figshare's search ignores phrases; short or generic phrases return noise (DataCite covers them)
            d = h.request("https://api.figshare.com/v2/articles/search",
                          data={"search_for": p, "item_type": 3, "published_since": since, "page_size": 100})
            for a in d or []:
                aid = str(a.get("id"))
                if aid in out:
                    continue
                t = (a.get("title") or "") + " " + (a.get("resource_title") or "")
                if not phrases_in(t, [p]):
                    continue
                if details_left <= 0:
                    break
                details_left -= 1
                full = h.get(f"https://api.figshare.com/v2/articles/{aid}") or {}
                desc = full.get("description", "")
                strong = [C.norm_doi(full.get("resource_doi"))] if full.get("resource_doi") else []
                rel = [{"relation": r.get("relation"), "identifier": r.get("identifier"), "scheme": r.get("identifier_type")}
                       for r in full.get("related_materials") or []]
                s2, weak, versions = paper_dois_from_related(rel, "relation", "identifier", "scheme")
                files = files_summary([(f.get("name"), f.get("size")) for f in full.get("files") or []])
                dr = data_rec(a.get("url_public_html") or f"https://figshare.com/articles/dataset/x/{aid}", a.get("title", ""), desc,
                              (full.get("license") or {}).get("name", ""), files, (a.get("published_date") or "")[:10], versions,
                              {"creators": [x.get("full_name", "") for x in full.get("authors") or []][:8],
                               "paper_dois": [x for x in strong + s2 if x], "weak_dois": weak + [y for y in C.dois_in(C.clean_text(desc)) if not C.is_data_doi(y)]})
                out[aid] = hit("figshare", f"Figshare dataset matching '{p}'", data=[dr])
    return list(out.values())


def osf_license(h, node):
    lic = ((node.get("relationships") or {}).get("license") or {}).get("data") or {}
    return lic.get("id", "")


def src_osf(h, since, until, ctx):
    out = {}
    for p in OSF_TERMS:
        for field in ("title", "description"):
            d = h.get("https://api.osf.io/v2/nodes/", {f"filter[{field}][icontains]": p, "filter[date_modified][gte]": since,
                                                       "page[size]": 100})
            for n in (d or {}).get("data", []):
                a = n.get("attributes", {})
                if n["id"] in out or not a.get("public", True):
                    continue
                text = a.get("title", "") + " " + (a.get("description") or "") + " " + " ".join(a.get("tags") or [])
                ph = phrases_in(text)
                if not ph:
                    continue
                if (a.get("date_created") or "")[:10] > until:
                    continue
                desc_dois = [x for x in C.dois_in(a.get("description") or "") if not C.is_data_doi(x)]
                out[n["id"]] = hit("osf", f"OSF project matching '{ph[0]}'",
                                   data=[data_rec(f"https://osf.io/{n['id']}/", a.get("title", ""), a.get("description") or "", "",
                                                  "", (a.get("date_created") or "")[:10], (),
                                                  {"paper_dois": [], "weak_dois": desc_dois, "osf_node": n["id"],
                                                   "tags": (a.get("tags") or [])[:10]})])
    # papers or preprints linked to the projects that survive the topic filter
    lookups = 0
    for k, x in out.items():
        dr = x["data"][0]
        if off_topic(dr["title"], dr["description"]) or dr["meta"]["weak_dois"] or lookups >= 150:
            continue
        lookups += 1
        pp = h.get(f"https://api.osf.io/v2/nodes/{k}/preprints/") or {}
        for p in pp.get("data", []):
            pdoi = C.norm_doi((p.get("links") or {}).get("preprint_doi"))
            adoi = C.norm_doi((p.get("attributes") or {}).get("doi"))
            dr["meta"]["paper_dois"] += [d for d in (adoi, pdoi) if d]
    return list(out.values())


def osf_preprint_hit(h, p, why):
    a = p.get("attributes", {})
    links = a.get("data_links") or []
    node = ((p.get("relationships") or {}).get("node") or {}).get("data") or {}
    urls = [u for u in links if C.repo_links(u) or "osf.io" in u]
    if node.get("id") and not urls and a.get("has_data_links") in ("available", "yes"):
        urls = [f"https://osf.io/{node['id']}/"]
    pdoi = C.norm_doi((p.get("links") or {}).get("preprint_doi"))
    adoi = C.norm_doi(a.get("doi"))
    prov = (((p.get("relationships") or {}).get("provider") or {}).get("data") or {}).get("id", "osf")
    paper = paper_rec(adoi or pdoi, a.get("title", ""), a.get("description", ""), (a.get("date_published") or a.get("date_created") or "")[:4],
                      "" if adoi else {"psyarxiv": "PsyArXiv", "sportrxiv": "SportRxiv", "socarxiv": "SocArXiv"}.get(prov, prov),
                      "", (a.get("date_published") or a.get("date_created") or "")[:10], via=f"OSF preprint ({prov})")
    if adoi and pdoi:
        paper["preprint_doi"] = pdoi
    return hit("osf-preprints", why, paper=paper, data=[data_rec(u) for u in urls[:3]],
               statement="Data links given on the preprint: " + ", ".join(urls[:3]) if urls else "")


def src_osf_preprints(h, since, until, ctx):
    out = {}
    for p in OSF_TERMS:
        d = h.get("https://api.osf.io/v2/preprints/", {"filter[title][icontains]": p, "filter[date_created][gte]": since,
                                                       "page[size]": 100})
        for x in (d or {}).get("data", []):
            guid = x["id"].split("_")[0]
            if guid in out:
                continue
            hx = osf_preprint_hit(h, x, f"OSF preprint matching '{p}'")
            if hx["data"]:
                out[guid] = hx
    return list(out.values())


# ------------------------------------------------------------------ sources: papers first
def epmc_search(h, query, result="lite", cap=3000):
    rows, cursor = [], "*"
    while len(rows) < cap:
        d = h.get(f"{EPMC}/search", {"query": query, "format": "json", "pageSize": 1000 if result == "lite" else 100,
                                     "resultType": result, "cursorMark": cursor})
        if not d:
            break
        res = (d.get("resultList") or {}).get("result") or []
        rows += res
        nxt = d.get("nextCursorMark")
        if not res or not nxt or nxt == cursor:
            break
        cursor = nxt
    return rows


def epmc_paper(r, via):
    doi = C.norm_doi(r.get("doi"))
    venue = r.get("journalTitle") or ((r.get("journalInfo") or {}).get("journal") or {}).get("medlineAbbreviation") or \
        ((r.get("bookOrReportDetails") or {}).get("publisher") if r.get("source") == "PPR" else "")
    p = paper_rec(doi, r.get("title", ""), r.get("abstractText", ""), r.get("pubYear"), venue, r.get("authorString", ""),
                  r.get("firstPublicationDate", ""), r.get("pmcid", ""), via)
    p["epmc_id"] = f"{r.get('source')}/{r.get('id')}"
    p["preprint"] = r.get("source") == "PPR"
    return p


def das_links(xml):
    """Repository links in the data availability statement of a JATS full text, with the statement text."""
    if isinstance(xml, bytes):
        xml = xml.decode("utf-8", "replace")
    blocks = []
    for m in re.finditer(r"<(sec|notes)\b[^>]*(sec-type|notes-type)=\"[^\"]*data[- ]?availab[^\"]*\"[^>]*>(.*?)</\1>", xml, re.S | re.I):
        blocks.append(m.group(3))
    for m in re.finditer(r"<meta-name>\s*Data Availability\s*</meta-name>\s*<meta-value>(.*?)</meta-value>", xml, re.S | re.I):
        blocks.append(m.group(1))
    for m in re.finditer(r"<(sec|notes|fn|boxed-text)\b[^>]*>\s*(?:<label>.*?</label>\s*)?<title>(.*?)</title>(.*?)</\1>", xml, re.S | re.I):
        if DAS_TITLE.search(C.clean_text(m.group(2))):
            blocks.append(m.group(3))
    links, text = [], ""
    for b in blocks:
        hrefs = " ".join(re.findall(r'xlink:href="([^"]+)"', b))
        t = C.clean_text(b)
        for u in C.repo_links(hrefs + " " + t):
            if u not in links:
                links.append(u)
        if not text and t:
            text = t[:700]
    if not links:
        for m in re.finditer(r"<p\b[^>]*>(.*?)</p>", xml, re.S):
            raw = m.group(1)
            t = C.clean_text(raw)
            if not DATA_SENT.search(t) or re.search(r"(previously published|publicly available data(set)?s? (from|of)|"
                                                    r"obtained from|downloaded from) ", t, re.I):
                continue
            us = C.repo_links(" ".join(re.findall(r'xlink:href="([^"]+)"', raw)) + " " + t)
            if us:
                links += [u for u in us if u not in links]
                text = text or t[:700]
    return links, text


def link_epmc_fulltext(h, papers, budget, phase_end=None):
    n = 0
    for p in papers:
        if n >= budget[0] or (phase_end and time.time() > phase_end):
            break
        if not p.get("pmcid"):
            continue
        x = h.get(f"{EPMC}/{p['pmcid']}/fullTextXML", raw=True, key="europepmc full text", tries=2, timeout=30)
        n += 1
        if not x:
            continue
        links, text = das_links(x)
        if links:
            p["_links"] = links
            p["_statement"] = text
    budget[0] -= n
    return n


def link_biorxiv(h, papers, budget, cap=80):
    n = 0
    for p in papers:
        if n >= min(budget[0], cap):
            break
        if not BIORXIV_DOI.match(p["doi"] or "") or p.get("_links"):
            continue
        server = "medrxiv" if "medrxiv" in (p.get("venue") or "").lower() else "biorxiv"
        d = h.get(f"https://api.biorxiv.org/details/{server}/{p['doi']}", key="biorxiv api", tries=2)
        coll = (d or {}).get("collection") or []
        if not coll:
            continue
        last = coll[-1]
        if last.get("published") and last["published"] != "NA":
            p["published_doi"] = C.norm_doi(last["published"])
        if last.get("jatsxml"):
            x = h.get(last["jatsxml"], raw=True, key="biorxiv full text", tries=1)  # bioRxiv rate-limits full texts hard
            n += 1
            if x:
                p["_jats_read"] = True
                links, text = das_links(x)
                if links:
                    p["_links"], p["_statement"] = links, text
    budget[0] -= n
    return n


def link_osf_preprint(h, papers):
    for p in papers:
        m = OSF_PREPRINT_DOI.match(p["doi"] or "")
        if not m or p.get("_links"):
            continue
        d = h.get(f"https://api.osf.io/v2/preprints/{m.group(3)}/")
        if d and d.get("data"):
            hx = osf_preprint_hit(h, d["data"], "")
            if hx["data"]:
                p["_links"] = [x["url"] for x in hx["data"]]
                p["_statement"] = hx["statement"]


def link_datacite(h, papers):
    """Datasets (any DataCite repository) that name one of these paper DOIs as a related identifier."""
    by = {p["doi"]: p for p in papers if p.get("doi")}
    dois = list(by)
    for i in range(0, len(dois), 25):
        chunk = dois[i:i + 25]
        q = "relatedIdentifiers.relatedIdentifier:(" + " OR ".join(f'"{d}"' for d in chunk) + ")"
        d = h.get("https://api.datacite.org/dois", {"query": q, "page[size]": 100})
        for x in (d or {}).get("data", []):
            a = x.get("attributes", {})
            typ = ((a.get("types") or {}).get("resourceTypeGeneral") or "").lower()
            if typ in ("text", "journalarticle", "preprint", "software", "workflow", "image", "audiovisual", "sound") or C.data_key(a.get("url")).startswith("dandi:"):
                continue
            ddoi = C.norm_doi(a.get("doi"))
            for r in a.get("relatedIdentifiers") or []:
                rd = C.norm_doi(r.get("relatedIdentifier"))
                if rd in by:
                    link = a.get("url") or ""
                    u = link if C.data_key(link).split(":")[0] in ("zenodo", "figshare", "osf", "openneuro") else "https://doi.org/" + ddoi
                    by[rd].setdefault("_links", [])
                    if u not in by[rd]["_links"]:
                        by[rd]["_links"].append(u)
                        by[rd]["_statement"] = by[rd].get("_statement") or f"DataCite: dataset {ddoi} names this paper ({r.get('relationType')})."
                        by[rd]["_via_datacite"] = True


def epmc_by_dois(h, dois, repo_filter=True, result="lite"):
    out = []
    dois = [d for d in dois if d]
    for i in range(0, len(dois), 40):
        q = "(" + " OR ".join(f'DOI:"{d}"' for d in dois[i:i + 40]) + ")" + (f" AND {REPO_TERMS}" if repo_filter else "")
        out += epmc_search(h, q, result, cap=200)
    return out


def src_europepmc(h, since, until, ctx):
    y = int(until[:4])
    animals = "NOT TITLE:(mice OR mouse OR rat OR rats OR rodent OR monkey OR monkeys OR macaque OR zebrafish OR drosophila OR songbird OR molecular OR protein)"
    pool = {}
    repo_ids = set()
    qgroups = [(tag, ps, True) for tag, ps in groups(14)]
    qgroups += [("broad", PAPER_TERMS[i:i + 10], False) for i in range(0, len(PAPER_TERMS), 10)]
    for tag, ps, strong in qgroups:
        base = f'TITLE_ABS:({" OR ".join(chr(34) + p + chr(34) for p in ps)}) AND CREATION_DATE:[{since} TO {until}] AND PUB_YEAR:[{y - 1} TO {y}] {animals}'
        for r in epmc_search(h, base):
            p = epmc_paper(r, "Europe PMC")
            if p["doi"] and p["doi"] not in pool:
                ph = phrases_in(p["title"], ps)
                p["_why"] = (f"Europe PMC {'preprint' if p['preprint'] else 'paper'} on {tag} ('{ph[0] if ph else ps[0]}' in title/abstract)"
                             if strong else f"Europe PMC {'preprint' if p['preprint'] else 'paper'} with a broad motor term in title/abstract")
                p["_strong"] = strong
                pool[p["doi"]] = p
            elif p["doi"] and strong:
                pool[p["doi"]]["_strong"] = True
        for r in epmc_search(h, base + f" AND {REPO_TERMS}"):
            repo_ids.add(C.norm_doi(r.get("doi")))
    ctx["epmc_pool"] = len(pool)
    papers = list(pool.values())
    C.log(f"  europepmc: {len(pool)} papers, {len(repo_ids)} mention a repository, "
          f"{sum(1 for p in papers if p['preprint'])} preprints")
    budget = ctx["fulltext_budget"]
    # strong keyword matches first, so a time cap drops the broad ones
    ctx["fulltext_epmc"] = link_epmc_fulltext(h, sorted((p for p in papers if p["doi"] in repo_ids), key=lambda p: not p.get("_strong")),
                                              budget, ctx.get("phase_end"))
    C.log(f"  europepmc: {ctx['fulltext_epmc']} full texts read")
    pre = sorted((p for p in papers if p["preprint"]), key=lambda p: not p.get("_strong"))  # strong matches first
    ctx["fulltext_biorxiv"] = link_biorxiv(h, pre, budget)
    C.log(f"  europepmc: {ctx['fulltext_biorxiv']} bioRxiv/medRxiv full texts read")
    link_osf_preprint(h, [p for p in papers if p["preprint"]])
    link_datacite(h, papers)
    out = []
    watch = ctx.setdefault("watch_new", {})
    for p in papers:
        links = p.pop("_links", []) or C.repo_links(p.get("abstract"))
        strong = p.pop("_strong", False)
        if not links:
            # no data link yet: open-access full texts reach Europe PMC weeks after publication, and bioRxiv may have
            # refused the full text; strong matches are rechecked on later runs (see src_watch)
            if strong and (not p["preprint"] or (BIORXIV_DOI.match(p["doi"]) and not p.get("_jats_read"))):
                watch[p["doi"]] = {"first_seen": until, "preprint": p["preprint"], "title": p["title"][:300]}
            if strong:
                ctx.setdefault("unlinked", {})[p["doi"]] = p["title"]
            continue
        st = p.pop("_statement", "")
        why = p.pop("_why", "Europe PMC")
        if p.pop("_via_datacite", False):
            why += "; data found through DataCite"
        out.append(hit("europepmc", why, paper=p, data=[data_rec(u) for u in links[:4]], statement=st))
    return out


def openalex_abstract(w):
    inv = w.get("abstract_inverted_index") or {}
    pos = sorted((i, word) for word, idx in inv.items() for i in idx)
    return " ".join(word for _, word in pos)


def openalex_paper(w, via):
    loc = w.get("primary_location") or {}
    venue = ((loc.get("source") or {}).get("display_name")) or ""
    auth = ", ".join((a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or [])
    p = paper_rec(w.get("doi") or "", w.get("title") or "", openalex_abstract(w), w.get("publication_year"), venue, auth,
                  w.get("publication_date") or "", via=via)
    p["preprint"] = w.get("type") == "preprint"
    return p


def oa_id(x):
    """Short OpenAlex id (W123, A456) from a full id URL; '' when missing."""
    return str(x or "").rsplit("/", 1)[-1]


def openalex_cache(h, idx, authors_path):
    """OpenAlex ids of the last author of every listed paper and of the key papers (resolved once, cached)."""
    cache = C.load_json(authors_path, {"authors": {}, "key_papers": {}})
    todo = sorted(d for d in idx["dois"] if d not in cache["authors"])
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        d = h.get("https://api.openalex.org/works", C.openalex_params({"filter": "doi:" + "|".join(chunk), "per-page": 50,
                                                                       "select": "doi,authorships"}))
        if d is None:
            return cache, False
        found = {}
        for w in d.get("results", []):
            au = w.get("authorships") or []
            last = next((a for a in reversed(au) if a.get("author_position") == "last"), au[-1] if au else None)
            if last and (last.get("author") or {}).get("id"):
                found[C.norm_doi(w.get("doi"))] = {"id": oa_id(last["author"]["id"]),
                                                   "name": last["author"].get("display_name", "")}
        for doi in chunk:
            cache["authors"][doi] = found.get(doi) or {"id": None}
    kp = [d for d in KEY_PAPERS if d not in cache["key_papers"]]
    if kp:
        d = h.get("https://api.openalex.org/works", C.openalex_params({"filter": "doi:" + "|".join(kp), "per-page": 50,
                                                                       "select": "id,doi,title"}))
        for w in (d or {}).get("results", []):
            cache["key_papers"][C.norm_doi(w.get("doi"))] = {"id": oa_id(w.get("id")), "title": w.get("title", "")}
        for k in kp:
            cache["key_papers"].setdefault(k, {"id": None})
    C.save_json(authors_path, cache)
    return cache, True


def src_openalex(h, since, until, ctx):
    cache, ok = openalex_cache(h, ctx["idx"], ctx["authors_path"])
    if not ok:
        ctx["openalex_error"] = "OpenAlex did not answer (daily budget or outage); author and citation searches skipped"
        return []
    sel = "id,doi,title,publication_date,publication_year,type,primary_location,authorships,abstract_inverted_index"
    works = {}
    aids = sorted({a["id"] for a in cache["authors"].values() if a.get("id")})
    names = {a["id"]: a["name"] for a in cache["authors"].values() if a.get("id")}
    flt = f"from_publication_date:{since},to_publication_date:{until},type:article|preprint"
    for i in range(0, len(aids), 50):
        cursor = "*"
        for _ in range(5):
            d = h.get("https://api.openalex.org/works", C.openalex_params({
                "filter": f"authorships.author.id:{'|'.join(aids[i:i + 50])},{flt}", "per-page": 200, "select": sel, "cursor": cursor}))
            if not d:
                break
            for w in d.get("results", []):
                if not w.get("doi"):
                    continue
                last = [a for a in (w.get("authorships") or []) if oa_id((a.get("author") or {}).get("id")) in names]
                who = names.get(oa_id(last[0]["author"]["id"]), "") if last else ""
                works.setdefault(C.norm_doi(w["doi"]), (w, f"OpenAlex: new work by {who}, last author of a listed study"))
            cursor = (d.get("meta") or {}).get("next_cursor")
            if not cursor or len(d.get("results", [])) < 200:
                break
    kp = {v["id"]: k for k, v in cache["key_papers"].items() if v.get("id")}
    if kp:
        cursor = "*"
        for _ in range(10):
            d = h.get("https://api.openalex.org/works", C.openalex_params({
                "filter": f"cites:{'|'.join(kp)},{flt}", "per-page": 200, "select": sel + ",referenced_works", "cursor": cursor}))
            if not d:
                break
            for w in d.get("results", []):
                if not w.get("doi"):
                    continue
                cited = [KEY_PAPERS.get(kp[oa_id(r)]) or cache["key_papers"][kp[oa_id(r)]].get("title") or "a key paper"
                         for r in w.get("referenced_works") or [] if oa_id(r) in kp]
                works.setdefault(C.norm_doi(w["doi"]), (w, f"OpenAlex: cites {cited[0] if cited else 'a key paper'}"))
            cursor = (d.get("meta") or {}).get("next_cursor")
            if not cursor or len(d.get("results", [])) < 200:
                break
    ctx["openalex_works"] = len(works)
    papers = []
    for doi, (w, why) in works.items():
        p = openalex_paper(w, "OpenAlex")
        p["_why"] = why
        if off_topic(p["title"], p["abstract"]):
            continue
        papers.append(p)
    # data locations: open full texts in Europe PMC that mention a repository, preprint routes, DataCite
    ft = {C.norm_doi(r.get("doi")): r for r in epmc_by_dois(h, [p["doi"] for p in papers])}
    for p in papers:
        r = ft.get(p["doi"])
        if r and r.get("pmcid"):
            p["pmcid"] = r["pmcid"]
    ctx["fulltext_openalex"] = link_epmc_fulltext(h, [p for p in papers if p["doi"] in ft], ctx["fulltext_budget"], ctx.get("phase_end"))
    link_biorxiv(h, [p for p in papers if p["preprint"] or BIORXIV_DOI.match(p["doi"])], ctx["fulltext_budget"])
    link_osf_preprint(h, papers)
    link_datacite(h, papers)
    out = []
    for p in papers:
        links = p.pop("_links", []) or C.repo_links(p.get("abstract"))
        if not links:
            continue
        st = p.pop("_statement", "")
        why = p.pop("_why")
        p.pop("_via_datacite", None)
        out.append(hit("openalex", why, paper=p, data=[data_rec(u) for u in links[:4]], statement=st))
    return out


def src_watch(h, since, until, ctx):
    """Papers found on earlier runs without a data link (strong keyword match): look again for an open full text in
    Europe PMC (they arrive weeks after publication), the bioRxiv full text, and deposits that name them (DataCite)."""
    old = {d: v for d, v in (ctx.get("watch_old") or {}).items() if d not in ctx.get("watch_new", {})}
    if not old:
        return []
    papers = [paper_rec(d, via="Europe PMC (rechecked)") for d in old]
    for p in papers:
        p["preprint"] = bool(old[p["doi"]].get("preprint"))
        p["_why"] = f"paper first seen {old[p['doi']].get('first_seen')} without a data link; link found on a later run"
    journal = [p for p in papers if not p["preprint"]]
    ft = {C.norm_doi(r.get("doi")): r for r in epmc_by_dois(h, [p["doi"] for p in journal])}
    for p in journal:
        r = ft.get(p["doi"])
        if r and r.get("pmcid"):
            p["pmcid"] = r["pmcid"]
            p.update({k: v for k, v in epmc_paper(r, "Europe PMC (rechecked)").items() if v and k not in ("doi", "preprint")})
    ctx["fulltext_watch"] = link_epmc_fulltext(h, [p for p in journal if p.get("pmcid")], ctx["fulltext_budget"])
    link_biorxiv(h, [p for p in papers if p["preprint"]], ctx["fulltext_budget"], cap=40)
    link_datacite(h, papers)
    out = []
    for p in papers:
        links = p.pop("_links", [])
        p.pop("_jats_read", None)
        p.pop("_via_datacite", None)
        why = p.pop("_why")
        st = p.pop("_statement", "")
        if links:
            out.append(hit("watch", why, paper=p, data=[data_rec(u) for u in links[:4]], statement=st))
    ctx["watch_found"] = {p["paper"]["doi"] for p in out}
    return out


# minutes each source may use at most (so a slow source cannot starve the ones after it); weekly runs need far less
SOURCE_MINUTES = {"zenodo": 6, "datacite": 6, "figshare": 6, "osf": 12, "osf-preprints": 5, "europepmc": 30, "openalex": 15,
                  "watch": 10, "title-match": 15}


def title_words(t, n=12):
    """The first n plain words of a title, safe inside a quoted search phrase."""
    w = re.sub(r"[^A-Za-z0-9 \-]+", " ", C.clean_text(t)).split()
    return " ".join(w[:n])


def src_title_match(h, since, until, ctx):
    """Papers with no data link yet (this run's strong matches and the recheck list): look for a deposit whose title is
    the paper's title (authors often name their OSF project or Zenodo record after the paper). DataCite (Zenodo,
    Figshare, Dryad, registered OSF projects, ...) is searched ten titles at a time; OSF projects one title at a time."""
    papers = dict(ctx.get("unlinked") or {})
    for d, v in (ctx.get("watch_old") or {}).items():
        if v.get("title") and d not in papers and v.get("title_checks", 0) < 3:
            papers[d] = v["title"]
    papers = {d: t for d, t in papers.items() if len(title_words(t).split()) >= 5}
    found = {}
    items = list(papers.items())
    for i in range(0, len(items), 10):
        chunk = items[i:i + 10]
        q = "titles.title:(" + " OR ".join(f'"{title_words(t)}"' for _, t in chunk) + ")"
        d = h.get("https://api.datacite.org/dois", {"query": q, "page[size]": 50, "resource-type-id": "dataset"})
        for x in (d or {}).get("data", []):
            a = x.get("attributes", {})
            dt_ = " ".join(t.get("title", "") for t in a.get("titles") or [])
            for doi, t in chunk:
                if C.title_sim(dt_, t) >= 0.85 or title_words(t).lower() in re.sub(r"[^a-z0-9 \-]+", " ", dt_.lower()):
                    link = a.get("url") or ""
                    ddoi = C.norm_doi(a.get("doi"))
                    u = link if C.data_key(link).split(":")[0] in ("zenodo", "figshare", "osf", "openneuro") else "https://doi.org/" + ddoi
                    found.setdefault(doi, []).append((u, dt_))
    checked = 0
    for doi, t in items:
        if doi in found or h.deadline and time.time() > h.deadline - 30:
            continue
        # OSF matches the literal text, so search the longest stretch of the title without punctuation
        seg = max(re.split(r"[:;,.?!()\[\]\"“”]+", C.clean_text(t)), key=lambda x: len(x.split())).strip()
        r = h.get("https://api.osf.io/v2/nodes/", {"filter[title][icontains]": " ".join(seg.split()[:8]), "page[size]": 10})
        checked += 1
        for n in (r or {}).get("data", []):
            nt = (n.get("attributes") or {}).get("title", "")
            if C.title_sim(nt, t) >= 0.85:
                found.setdefault(doi, []).append((f"https://osf.io/{n['id']}/", nt))
    ctx["title_checked"] = set(papers)
    out = []
    for doi, links in found.items():
        p = paper_rec(doi, papers[doi], via="Europe PMC")
        out.append(hit("title-match", "a deposit is titled like the paper (" + C.host_label(links[0][0]) + ")", paper=p,
                       data=[data_rec(u, tt) for u, tt in links[:3]],
                       statement="Found by title: the deposit's title matches the paper's title."))
    C.log(f"  title-match: {len(papers)} papers checked ({checked} on OSF), {len(out)} with a deposit")
    return out

SOURCES = [("zenodo", src_zenodo), ("datacite", src_datacite), ("figshare", src_figshare), ("osf", src_osf),
           ("osf-preprints", src_osf_preprints), ("europepmc", src_europepmc), ("openalex", src_openalex), ("watch", src_watch),
           ("title-match", src_title_match)]


# ------------------------------------------------------------------ merging, linking data to papers
def crossref_title_search(h, title, creators, published):
    t = " ".join(C.norm_title(title))
    if len(t) < 30:
        return None
    y = int((published or "2026")[:4])
    d = h.get("https://api.crossref.org/works", {"query.bibliographic": title[:300], "rows": 4,
                                                  "filter": f"from-pub-date:{y - 2}-01-01", "select": "DOI,title,author,type,issued"})
    surnames = {re.split(r"[,\s]+", c.strip())[0].lower() for c in creators if c.strip()}
    for it in ((d or {}).get("message") or {}).get("items", []):
        if it.get("type") not in ("journal-article", "posted-content", "proceedings-article"):
            continue
        if C.title_sim(title, (it.get("title") or [""])[0]) < 0.85:
            continue
        fam = {(a.get("family") or "").lower() for a in it.get("author") or []}
        if surnames and not (surnames & fam):
            continue
        return C.norm_doi(it["DOI"])
    return None


PAPER_TYPES = {"journal-article", "posted-content", "proceedings-article", "book-chapter", "report", "dissertation"}


def paper_check(h, doi, via):
    """A paper record for this DOI if it is a paper or preprint (Crossref, or DataCite for arXiv-like preprints), else None."""
    doi = C.norm_doi(doi)
    if not doi or C.is_data_doi(doi):
        return None
    m = ((h.get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}") or {}).get("message")) or {}
    if m:
        if m.get("type") not in PAPER_TYPES:
            return None
        pre = m.get("type") == "posted-content"
        venue = (m.get("institution") or [{}])[0].get("name", "") if pre else (m.get("container-title") or [""])[0]
        p = paper_rec(doi, (m.get("title") or [""])[0], m.get("abstract", ""),
                      ((m.get("issued") or {}).get("date-parts") or [[None]])[0][0], venue,
                      ", ".join(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author") or []),
                      "-".join(str(x) for x in ((m.get("published") or m.get("issued") or {}).get("date-parts") or [[]])[0]), via=via)
        p["preprint"] = pre
        rel = (m.get("relation") or {}).get("is-preprint-of") or []
        p["published_doi"] = next((C.norm_doi(r.get("id")) for r in rel if r.get("id-type") == "doi"), "")
        return p
    a = (((h.get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}") or {}).get("data") or {}).get("attributes")) or {}
    if ((a.get("types") or {}).get("resourceTypeGeneral") or "") in ("Preprint", "JournalArticle", "ConferencePaper", "Text"):
        p = paper_rec(doi, ((a.get("titles") or [{}])[0]).get("title", ""), "", a.get("publicationYear"),
                      (a.get("publisher") or {}).get("name", "") if isinstance(a.get("publisher"), dict) else a.get("publisher", ""),
                      ", ".join(c.get("name", "") for c in a.get("creators") or []), via=via)
        p["preprint"] = True
        return p
    return None


def merge(hits):
    """Group hits that share a paper DOI or a data key."""
    groups_ = []
    by_doi, by_key = {}, {}
    for x in hits:
        p = x["paper"]
        keys = {k for d in x["data"] for k in d["keys"]}
        gi = None
        if p and p.get("doi") and p["doi"] in by_doi:
            gi = by_doi[p["doi"]]
        for k in keys:
            if gi is None and k in by_key:
                gi = by_key[k]
        if gi is None:
            groups_.append({"paper": p, "data": [], "sources": [], "why": [], "statement": ""})
            gi = len(groups_) - 1
        g = groups_[gi]
        if p and (not g["paper"] or (not g["paper"].get("abstract") and p.get("abstract"))):
            g["paper"] = {**(g["paper"] or {}), **{k: v for k, v in p.items() if v}}
        have = {k for d in g["data"] for k in d["keys"]}
        for d in x["data"]:
            if not set(d["keys"]) & have:
                g["data"].append(d)
                have |= set(d["keys"])
            else:
                old = next(o for o in g["data"] if set(o["keys"]) & set(d["keys"]))
                for f in ("title", "description", "license", "files"):
                    old[f] = old[f] or d[f]
                old["keys"] = sorted(set(old["keys"]) | set(d["keys"]))
                for f in ("paper_dois", "weak_dois", "creators"):
                    old["meta"][f] = list(dict.fromkeys((old["meta"].get(f) or []) + (d["meta"].get(f) or [])))
        if x["source"] not in g["sources"]:
            g["sources"].append(x["source"])
        if x["why"] and x["why"] not in g["why"]:
            g["why"].append(x["why"])
        g["statement"] = g["statement"] or x.get("statement", "")
        if g["paper"] and g["paper"].get("doi"):
            by_doi[g["paper"]["doi"]] = gi
        for d in g["data"]:
            for k in d["keys"]:
                by_key[k] = gi
    return groups_


def primary_data(g):
    def rank(d):
        kind = d["key"].split(":")[0]
        return (HOST_RANK.index(kind) if kind in HOST_RANK else 99, -len(d.get("files") or ""))
    return sorted(g["data"], key=rank)[0] if g["data"] else None


def deposit_files(h, d):
    """A short summary of the deposit's files (name extensions, count, size) from the repository API."""
    if d.get("files"):
        return d["files"]
    kind, _, v = d["key"].partition(":")
    try:
        if kind == "zenodo":
            r = h.get(f"https://zenodo.org/api/records/{v}")
            return files_summary([(f.get("key"), f.get("size")) for f in (r or {}).get("files") or []])
        if kind == "figshare":
            r = h.get(f"https://api.figshare.com/v2/articles/{v}")
            if r and not d.get("license"):
                d["license"] = (r.get("license") or {}).get("name", "")
            return files_summary([(f.get("name"), f.get("size")) for f in (r or {}).get("files") or []])
        if kind == "osf":
            r = h.get(f"https://api.osf.io/v2/nodes/{v}/files/osfstorage/", {"page[size]": 100})
            fs = []
            for f in (r or {}).get("data", [])[:100]:
                a = f.get("attributes", {})
                if a.get("kind") == "folder" and len(fs) < 200:
                    sub = h.get(f"https://api.osf.io/v2/nodes/{v}/files/osfstorage{a.get('path', '/')}", {"page[size]": 100})
                    fs += [(s["attributes"].get("name"), s["attributes"].get("size")) for s in (sub or {}).get("data", [])
                           if s["attributes"].get("kind") == "file"]
                elif a.get("kind") == "file":
                    fs.append((a.get("name"), a.get("size")))
            node = h.get(f"https://api.osf.io/v2/nodes/{v}/", {"embed": "license"}) or {}
            lic = ((((node.get("data") or {}).get("embeds") or {}).get("license") or {}).get("data") or {}).get("attributes", {})
            if lic.get("name") and not d.get("license"):
                d["license"] = lic["name"]
            if not d.get("title"):
                d["title"] = ((node.get("data") or {}).get("attributes") or {}).get("title", "")
            return files_summary(fs) or ("no files in OSF storage (may use linked storage or components)" if r is not None else "")
        if kind == "github":
            r = h.get(f"https://api.github.com/repos/{v}")
            if r and not d.get("license"):
                d["license"] = (r.get("license") or {}).get("spdx_id") or ""
            t = h.get(f"https://api.github.com/repos/{v}/git/trees/{(r or {}).get('default_branch', 'main')}", {"recursive": 1})
            return files_summary([(x.get("path"), x.get("size")) for x in (t or {}).get("tree", []) if x.get("type") == "blob"])
        if kind == "doi" and v.startswith("10.7910/dvn"):
            r = h.get("https://dataverse.harvard.edu/api/datasets/:persistentId/", {"persistentId": "doi:" + v.upper()})
            fs = (((r or {}).get("data") or {}).get("latestVersion") or {}).get("files") or []
            return files_summary([((f.get("dataFile") or {}).get("filename"), (f.get("dataFile") or {}).get("filesize")) for f in fs])
        if kind == "doi" and v.startswith("10.5061/dryad"):
            r = h.get(f"https://datadryad.org/api/v2/datasets/{urllib.parse.quote('doi:' + v, safe='')}")
            if r:
                d["license"] = d.get("license") or r.get("license", "")
                return f"Dryad deposit, {int(r.get('storageSize') or 0) / 1e6:.1f} MB"
    except Exception as e:  # noqa: BLE001  (a summary is optional)
        C.log(f"  files summary failed for {d['key']}: {type(e).__name__}")
    return ""


def enrich_papers(h, cands):
    """Abstract, venue, year and authors for papers that came from a data deposit (Europe PMC, then OpenAlex)."""
    need = [c for c in cands if not c.get("abstract") or not c.get("title")]
    rows = {C.norm_doi(r.get("doi")): r for r in epmc_by_dois(h, [c["doi"] for c in need], repo_filter=False, result="core")}
    for c in need:
        r = rows.get(c["doi"])
        if r:
            p = epmc_paper(r, "")
            for f in ("title", "abstract", "venue", "authors", "year", "published"):
                c[f] = c.get(f) or p[f]
            continue
        w = h.get(f"https://api.openalex.org/works/doi:{c['doi']}", C.openalex_params())
        if w:
            p = openalex_paper(w, "")
            for f in ("title", "abstract", "venue", "authors", "year", "published"):
                c[f] = c.get(f) or p[f]
            continue
        cr = h.get(f"https://api.crossref.org/works/{urllib.parse.quote(c['doi'])}")
        m = (cr or {}).get("message") or {}
        if m:
            c["title"] = c.get("title") or C.clean_text((m.get("title") or [""])[0])
            c["abstract"] = c.get("abstract") or C.clean_text(m.get("abstract", ""), 4000)
            c["venue"] = c.get("venue") or (m.get("container-title") or [m.get("institution", [{}])[0].get("name", "") if m.get("institution") else ""])[0]
            c["year"] = c.get("year") or ((m.get("issued") or {}).get("date-parts") or [[None]])[0][0]
            c["authors"] = c.get("authors") or ", ".join(f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author") or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--state", default=str(C.STATE))
    ap.add_argument("--authors", default=str(C.AUTHORS))
    ap.add_argument("--out", default=str(C.WORK))
    ap.add_argument("--sources", default=",".join(n for n, _ in SOURCES))
    ap.add_argument("--max-fulltext", type=int, default=500)
    ap.add_argument("--overlap-days", type=int, default=7)
    ap.add_argument("--time-budget", type=float, default=95, help="minutes for all sources; later requests are skipped")
    ap.add_argument("--time-scale", type=float, default=1.0, help="multiplies each source's own time cap (SOURCE_MINUTES)")
    ap.add_argument("--link-budget", type=float, default=25, help="minutes for linking deposits to papers and enrichment")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = C.load_json(a.state, C.empty_state())
    for k, v in C.empty_state().items():
        state.setdefault(k, v)
    today = dt.date.today()
    until = a.until or today.isoformat()
    if a.since:
        since = a.since
    elif state.get("last_run"):
        since = (dt.date.fromisoformat(state["last_run"]) - dt.timedelta(days=a.overlap_days)).isoformat()
    else:
        since = (today - dt.timedelta(days=14)).isoformat()
    C.log(f"finder: {since} to {until}")
    idx = C.listed_index()
    h = C.Http()
    h.deadline = time.time() + 60 * a.time_budget
    ctx = {"idx": idx, "authors_path": a.authors, "fulltext_budget": [a.max_fulltext], "watch_old": dict(state.get("watch") or {})}
    enabled = set(a.sources.split(","))
    hits, per_source = [], {}
    run_deadline = h.deadline
    for name, fn in SOURCES:
        if name not in enabled:
            continue
        t0, n0 = time.time(), sum(h.count.values())
        h.deadline = min(run_deadline, t0 + 60 * SOURCE_MINUTES.get(name, 10) * a.time_scale)
        ctx["phase_end"] = t0 + 60 * SOURCE_MINUTES.get(name, 10) * a.time_scale * 0.55  # full-text reading stops here
        try:
            got = fn(h, since, until, ctx)
        except Exception as e:  # noqa: BLE001  (one broken source must not stop the run)
            C.log(f"  source {name} failed: {type(e).__name__}: {str(e)[:200]}")
            got = []
            per_source.setdefault(name, {})["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        kept = [x for x in got if not off_topic((x["paper"] or {}).get("title") or (x["data"][0]["title"] if x["data"] else ""),
                                                 ((x["paper"] or {}).get("abstract") or "") + " " + (x["data"][0]["description"] if x["data"] else ""))]
        per_source.setdefault(name, {}).update(hits=len(got), on_topic=len(kept), requests=sum(h.count.values()) - n0,
                                               seconds=round(time.time() - t0))
        C.log(f"  {name}: {len(got)} hits, {len(kept)} on topic, {per_source[name]['requests']} requests, {per_source[name]['seconds']} s")
        hits += kept
    sources_out_of_time = h.out_of_time  # some source hit its time cap
    h.deadline = time.time() + 60 * a.link_budget  # the linking and enrichment below get their own time
    groups_ = merge(hits)
    # data-first groups: find their paper
    held, cands, dropped = [], [], {"listed": 0, "seen": 0, "no data": 0, "listed title": 0}
    for g in groups_:
        if not g["data"]:
            dropped["no data"] += 1
            continue
        if not (g["paper"] and g["paper"].get("doi")):
            d0 = g["data"][0]
            tries = [(x, "the deposit's related-paper link") for d in g["data"] for x in d["meta"].get("paper_dois") or []]
            tries += [(x, "a DOI in the deposit's description") for d in g["data"] for x in d["meta"].get("weak_dois") or []]
            paper = None
            for doi, via in list(dict.fromkeys(tries))[:4]:
                paper = paper_check(h, doi, via)
                if paper:
                    break
            if not paper:
                doi = crossref_title_search(h, d0["title"], d0["meta"].get("creators") or [], d0.get("created"))
                paper = paper_check(h, doi, "a Crossref search for the deposit's title") if doi else None
            if not paper:
                held.append({"data_url": d0["url"], "data_key": d0["key"], "title": d0["title"], "host": d0["host"],
                             "created": d0.get("created"), "source": g["sources"], "why_found": "; ".join(g["why"])})
                continue
            g["paper"] = paper
        cands.append(g)
    # dedupe against the site and earlier runs
    final, listed_found = [], []
    seen = state["seen"]
    for g in cands:
        p = g["paper"]
        keys = {k for d in g["data"] for k in d["keys"]}
        pdois = {p["doi"], C.norm_doi(p.get("published_doi")), C.norm_doi(p.get("preprint_doi"))} - {""}
        if pdois & idx["dois"] or keys & idx["keys"]:
            dropped["listed"] += 1
            listed_found.append(sorted(pdois & idx["dois"]) + sorted(keys & idx["keys"]))
            continue
        if any(("doi:" + d) in seen for d in pdois) or any(k in seen for k in keys):
            dropped["seen"] += 1
            continue
        if p.get("title") and C.listed_title_match(p["title"], idx):
            dropped["listed title"] += 1
            continue
        final.append(g)
    # held deposits from earlier runs: keep them for 180 days, drop the ones now listed or found with a paper
    now_keys = {k for g in final for d in g["data"] for k in d["keys"]}
    new_held = {}
    for k, v in state["held"].items():
        if k in idx["keys"] or k in now_keys:
            continue
        if v.get("first_seen", until) >= (dt.date.fromisoformat(until) - dt.timedelta(days=180)).isoformat():
            new_held[k] = v
    for x in held:
        if x["data_key"] in idx["keys"] or x["data_key"] in seen:
            continue
        new_held.setdefault(x["data_key"], {"title": x["title"], "url": x["data_url"], "first_seen": until})
    # recheck older held deposits: has a paper appeared?
    rechecked = 0
    for k, v in list(new_held.items()):
        if v.get("first_seen") == until or rechecked >= 40:
            continue
        rechecked += 1
        doi = crossref_title_search(h, v["title"], [], v.get("first_seen"))
        if doi and doi not in idx["dois"] and ("doi:" + doi) not in seen:
            final.append({"paper": paper_rec(doi, via="a Crossref search for the held deposit's title"),
                          "data": [data_rec(v["url"], v["title"])], "sources": ["held"], "why": [f"held deposit from {v['first_seen']}; its paper has appeared"],
                          "statement": ""})
            del new_held[k]
    # dedupe papers among themselves (a preprint and its journal version)
    enrich_papers(h, [g["paper"] for g in final])
    uniq, seen_t = [], []
    for g in final:
        t = " ".join(C.norm_title(g["paper"].get("title")))
        if t and any(C.title_sim(t, s) > 0.92 for s in seen_t):
            dropped["duplicate candidate"] = dropped.get("duplicate candidate", 0) + 1
            continue
        seen_t.append(t)
        uniq.append(g)
    rows = []
    for g in uniq:
        p = g["paper"]
        if not p.get("title"):
            dropped["paper DOI did not resolve"] = dropped.get("paper DOI did not resolve", 0) + 1
            continue
        d = primary_data(g)
        d["files"] = deposit_files(h, d)
        if documents_only(d["files"]):
            dropped["deposit holds documents or images only"] = dropped.get("deposit holds documents or images only", 0) + 1
            continue
        rows.append({
            "cid": f"c{len(rows) + 1:03d}", "doi": p["doi"], "title": p.get("title", ""), "abstract": p.get("abstract", ""), "year": p.get("year"),
            "venue": p.get("venue", ""), "authors": p.get("authors", ""), "published": p.get("published", ""),
            "preprint": bool(p.get("preprint")), "published_doi": p.get("published_doi", ""), "paper_found_via": p.get("via", ""),
            "data_url": d["url"], "data_key": d["key"], "data_keys": sorted({k for x in g["data"] for k in x["keys"]}),
            "host": d["host"], "other_data_urls": [x["url"] for x in g["data"] if x is not d][:4],
            "data_title": d.get("title", ""), "data_description": d.get("description", "")[:1200], "data_license": d.get("license", ""),
            "files": d.get("files", ""), "data_statement": g.get("statement", "")[:700],
            "source": g["sources"], "why_found": "; ".join(g["why"])[:500],
        })
    C.write_jsonl(out / "candidates.jsonl", rows)
    C.write_jsonl(out / "held.jsonl", held)
    state["held"] = new_held
    # papers waiting for a data link: keep 90 days; drop what was found, listed or seen
    found = {r["doi"] for r in rows} | (ctx.get("watch_found") or set())
    cutoff = (dt.date.fromisoformat(until) - dt.timedelta(days=90)).isoformat()
    watch = {d: v for d, v in (state.get("watch") or {}).items()
             if d not in found and v.get("first_seen", until) >= cutoff and d not in idx["dois"] and ("doi:" + d) not in seen}
    for d, v in (ctx.get("watch_new") or {}).items():
        if d not in found and d not in idx["dois"] and ("doi:" + d) not in seen:
            watch.setdefault(d, v)
    for d in ctx.get("title_checked") or ():
        if d in watch:
            watch[d]["title_checks"] = watch[d].get("title_checks", 0) + 1
    state["watch"] = watch
    state["last_run"] = until
    C.save_json(out / "state.json", state)
    summary = {"since": since, "until": until, "per_source": per_source, "hits": len(hits), "groups": len(groups_),
               "with_paper": len(cands), "held_back_new": len(held), "held_total": len(new_held), "dropped": dropped,
               "candidates": len(rows), "watching": len(watch), "already_listed_found": listed_found, "requests": h.count, "http_errors": h.errors,
               "fulltexts": {k: ctx.get(k) for k in ("fulltext_epmc", "fulltext_biorxiv", "fulltext_openalex")},
               "europepmc_pool": ctx.get("epmc_pool"), "openalex_works": ctx.get("openalex_works"),
               "notes": ([ctx["openalex_error"]] if ctx.get("openalex_error") else [])
               + (["a source reached its time cap; some results were not checked (see per_source seconds)"] if sources_out_of_time else [])
               + ([f"the {a.link_budget:g}-minute linking budget ran out; some deposits were not matched to papers"]
                  if h.out_of_time and not sources_out_of_time else [])
               + [f"{x} stopped answering; skipped for the rest of the run" for x in sorted(h.down)]}
    C.save_json(out / "find_summary.json", summary)
    print(f"finder: {len(rows)} candidates, {len(held)} deposits held back (no paper), {sum(h.count.values())} requests")


if __name__ == "__main__":
    main()
