"""Weekly finder, step 3 (deterministic): turn screened-in candidates into source/extra_studies.json entries.

usage: python3 -I tools/finder/listing.py [--screened work/finder/screened.jsonl] [--out work/finder/new_entries.json]
                                          [--state work/finder/state.json] [--limit N]

Paper metadata comes from Crossref (DataCite for DOIs Crossref does not know), the lab from OpenAlex (last author and
institution), and license, host and formats from the repository's API. Ids follow the site's convention
FirstAuthorYear_Venue and never collide with an existing id, converter or record. A preprint whose journal version is out
is listed under the journal paper. Entries are listed whether or not standardization succeeds later.
Also writes <out dir>/new_abstracts.json (abstracts for source/abstracts.json) and adds each new id to the state's
`pending` queue for standardization. This script does not modify source/extra_studies.json: publish.py does."""
import argparse
import datetime as dt
import re
import sys
import unicodedata
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

PREPRINT_VENUE = [("10.1101/", "bioRxiv"), ("10.64898/", "bioRxiv"), ("10.31234/", "PsyArXiv"), ("10.51224/", "SportRxiv"),
                  ("10.48550/", "arXiv"), ("10.21203/", "ResearchSquare"), ("10.20944/", "Preprints"), ("10.31219/", "OSFPreprints"),
                  ("10.2139/", "SSRN"), ("10.36227/", "TechRxiv"), ("10.31730/", "AfricArXiv")]
LICENSES = [(r"cc0|public domain|cc-zero|creative commons zero", "CC0"), (r"by-nc-nd|by nc nd|noncommercial-noderiv", "CC BY-NC-ND 4.0"),
            (r"by-nc-sa|by nc sa|noncommercial-sharealike", "CC BY-NC-SA 4.0"), (r"by-nd|by nd|noderiv", "CC BY-ND 4.0"),
            (r"by-nc|by nc|noncommercial", "CC BY-NC 4.0"), (r"by-sa|by sa|sharealike", "CC BY-SA 4.0"),
            (r"cc-by|cc by|attribution|creativecommons\.org/licenses/by/", "CC BY 4.0"), (r"\bmit\b", "MIT"),
            (r"gpl-?3|gnu general public license v?3|gpl v3", "GPL-3.0"), (r"apache", "Apache 2.0"), (r"odc-by", "ODC-By 1.0"),
            (r"\bbsd\b", "BSD")]
FORMAT = {"csv": "CSV", "tsv": "CSV", "mat": "MATLAB", "m": "MATLAB", "xlsx": "Excel", "xls": "Excel", "txt": "text", "dat": "text",
          "zip": "ZIP", "py": "Python", "ipynb": "Python", "npy": "Python", "npz": "Python", "pkl": "Python", "r": "R", "rds": "R",
          "rdata": "R", "json": "JSON", "h5": "HDF5", "hdf5": "HDF5", "nwb": "HDF5", "c3d": "C3D", "sav": "SPSS", "rar": "RAR",
          "edf": "EDF", "bdf": "EDF", "tar": "TAR", "gz": "ZIP", "7z": "ZIP", "xml": "XML", "wav": "audio", "mp4": "video",
          "avi": "video", "mov": "video", "set": "EEGLAB", "mid": "MIDI", "parquet": "Parquet", "feather": "Python"}


def ascii_fold(s):
    s = (s or "").replace("ß", "ss").replace("ø", "o").replace("Ø", "O").replace("ł", "l").replace("Ł", "L").replace("æ", "ae")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def initials(given):
    parts = re.split(r"[\s.]+", given or "")
    out = ""
    for p in parts:
        if not p:
            continue
        out += "".join(x[0] for x in p.split("-") if x)
    return out.upper()


def crossref(h, doi):
    r = h.get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    return (r or {}).get("message")


def datacite_meta(h, doi):
    r = h.get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
    a = ((r or {}).get("data") or {}).get("attributes")
    if not a or ((a.get("types") or {}).get("resourceTypeGeneral") or "") not in ("Preprint", "JournalArticle", "ConferencePaper", "Text"):
        return None  # a dataset or software DOI is not a paper
    au = [{"family": c.get("familyName") or (c.get("name") or "").split(",")[0], "given": c.get("givenName") or ""} for c in a.get("creators") or []]
    pub = a.get("publisher")
    return {"author": au, "title": [(a.get("titles") or [{}])[0].get("title", "")], "issued": {"date-parts": [[a.get("publicationYear")]]},
            "type": "posted-content", "institution": [{"name": pub.get("name") if isinstance(pub, dict) else pub}]}


def paper_meta(h, doi):
    m = crossref(h, doi)
    if m and m.get("type") not in ("journal-article", "posted-content", "proceedings-article", "book-chapter", "report", "dissertation"):
        return None
    m = m or datacite_meta(h, doi)
    if not m:
        return None
    au = m.get("author") or []
    authors = ", ".join(f"{a.get('family') or a.get('name', '')} {initials(a.get('given'))}".strip() for a in au)
    dp = ((m.get("published-print") or m.get("published-online") or m.get("issued") or m.get("posted") or {}).get("date-parts") or [[None]])[0]
    year = dp[0] if dp and dp[0] else None
    ptype = m.get("type", "")
    preprint = ptype == "posted-content" or any(doi.startswith(p) for p, _ in PREPRINT_VENUE)
    if preprint:
        venue_full = venue = ""
        short = next((v for p, v in PREPRINT_VENUE if doi.startswith(p)), "") or \
            re.sub(r"[^A-Za-z]", "", ((m.get("institution") or [{}])[0].get("name") or m.get("group-title") or "Preprint"))
    else:
        short = (m.get("short-container-title") or [""])[0].replace(".", "").strip()
        venue_full = (m.get("container-title") or [""])[0]
        venue = short or venue_full
        vol, iss = m.get("volume", ""), m.get("issue", "")
        page = m.get("article-number") or m.get("page", "")
        if vol:
            venue += f" {vol}" + (f"({iss})" if iss else "") + (f":{page}" if page else "")
    rel = (m.get("relation") or {}).get("is-preprint-of") or []
    journal = next((C.norm_doi(r.get("id")) for r in rel if r.get("id-type") == "doi"), "")
    return {"authors": authors, "families": [a.get("family") or a.get("name", "") for a in au], "year": year,
            "title": C.clean_text((m.get("title") or [""])[0]), "venue": venue, "venue_full": venue_full, "short": short,
            "doi": doi, "preprint": preprint, "journal_doi": journal, "abstract": C.clean_text(m.get("abstract", ""))}


def venue_map(extra):
    """Id suffixes the curators used, keyed by the normalized venue name of the paper."""
    out = {}
    for e in extra:
        v = re.split(r"\s+\d", (e["papers"][0].get("venue") or ""))[0]
        suf = e["id"].split("_")[1] if "_" in e["id"] else ""
        k = re.sub(r"[^a-z]", "", v.lower())
        if k and suf and k not in out:
            out[k] = suf
    return out


def venue_suffix(meta, vmap):
    if meta["preprint"]:
        return meta["short"] or "Preprint"
    for name in (meta["short"], meta["venue_full"]):
        k = re.sub(r"[^a-z]", "", (name or "").lower())
        if k in vmap:
            return vmap[k]
    name = meta["short"] or meta["venue_full"] or "Journal"
    words = [w for w in re.split(r"[^A-Za-z0-9]+", ascii_fold(name)) if w and w.lower() not in ("of", "the", "and", "for", "in", "on")]
    s = "".join(w if w.isupper() or len(w) <= 4 else w[0].upper() + w[1:] for w in words)
    return s[:28] or "Journal"


def make_id(meta, vmap, taken):
    fam = ascii_fold(meta["families"][0] if meta["families"] else "Anon")
    fam = re.sub(r"[^A-Za-z-]", "", fam.replace(" ", "")) or "Anon"
    base = f"{fam}{meta['year'] or dt.date.today().year}_{venue_suffix(meta, vmap)}"
    cand, n = base, 1
    while cand in taken:
        n += 1
        cand = f"{base}{'abcdefghij'[n - 1] if n <= 10 else n}"
    taken.add(cand)
    return cand


def label_of(meta):
    fams = [ascii_fold(f) for f in meta["families"]] or ["Anonymous"]
    y = meta["year"] or ""
    if len(fams) == 1:
        return f"{fams[0]} {y}"
    if len(fams) == 2:
        return f"{fams[0]} & {fams[1]} {y}"
    return f"{fams[0]} et al. {y}"


def lab_of(h, doi, meta):
    w = h.get(f"https://api.openalex.org/works/doi:{doi}", C.openalex_params({"select": "authorships"}))
    au = (w or {}).get("authorships") or []
    last = next((a for a in reversed(au) if a.get("author_position") == "last"), au[-1] if au else None)
    fam = ascii_fold(meta["families"][-1]) if meta["families"] else ""
    inst = ""
    if last:
        insts = last.get("institutions") or []
        inst = insts[0].get("display_name", "") if insts else ""
        if not fam:
            fam = ascii_fold((last.get("author") or {}).get("display_name", "").split()[-1])
    return f"{fam} ({ascii_fold(inst)})" if inst else fam


def norm_license(raw):
    s = (raw or "").lower()
    if not s or s in ("none", "no license", "other", "notspecified", "noassertion"):
        return None
    for pat, name in LICENSES:
        if re.search(pat, s):
            if name.startswith("CC") and name != "CC0":
                ver = re.search(r"([1-4])\.0", s)
                if ver and ver.group(1) != "4":
                    name = name.replace("4.0", ver.group(1) + ".0")
            return name
    return raw.strip()[:60]


def deposit_info(h, url, cand):
    """License (raw), file extensions and a corrected data URL from the repository API."""
    k = C.data_key(url)
    kind, _, v = k.partition(":")
    lic, exts = cand.get("data_license") or "", []
    try:
        if kind == "zenodo":
            r = h.get(f"https://zenodo.org/api/records/{v}") or {}
            lic = lic or ((r.get("metadata") or {}).get("license") or {}).get("id", "")
            exts = [f.get("key", "") for f in r.get("files") or []]
        elif kind == "figshare":
            r = h.get(f"https://api.figshare.com/v2/articles/{v}") or {}
            lic = lic or (r.get("license") or {}).get("name", "")
            exts = [f.get("name", "") for f in r.get("files") or []]
        elif kind == "osf":
            r = h.get(f"https://api.osf.io/v2/nodes/{v}/", {"embed": "license"}) or {}
            la = ((((r.get("data") or {}).get("embeds") or {}).get("license") or {}).get("data") or {}).get("attributes", {})
            lic = la.get("name") or ""
            fl = h.get(f"https://api.osf.io/v2/nodes/{v}/files/osfstorage/", {"page[size]": 100}) or {}
            exts = [f["attributes"].get("name", "") for f in fl.get("data", []) if f["attributes"].get("kind") == "file"]
        elif kind == "github":
            r = h.get(f"https://api.github.com/repos/{v}") or {}
            lic = lic or (r.get("license") or {}).get("spdx_id") or ""
            t = h.get(f"https://api.github.com/repos/{v}/git/trees/{r.get('default_branch', 'main')}", {"recursive": 1}) or {}
            exts = [x.get("path", "") for x in t.get("tree", []) if x.get("type") == "blob"]
        elif kind == "doi" and v.startswith("10.5061/dryad"):
            r = h.get(f"https://datadryad.org/api/v2/datasets/{urllib.parse.quote('doi:' + v, safe='')}") or {}
            lic = lic or r.get("license", "") or "CC0"
        elif kind == "doi" and v.startswith("10.7910/dvn"):
            r = h.get("https://dataverse.harvard.edu/api/datasets/:persistentId/", {"persistentId": "doi:" + v.upper()}) or {}
            lv = (r.get("data") or {}).get("latestVersion") or {}
            lic = lic or (lv.get("license") or {}).get("name", "") if isinstance(lv.get("license"), dict) else lic or lv.get("license", "")
            exts = [(f.get("dataFile") or {}).get("filename", "") for f in lv.get("files") or []]
    except Exception as e:  # noqa: BLE001
        C.log(f"  deposit info failed for {k}: {type(e).__name__}")
    if not exts and cand.get("files"):
        exts = ["x." + e for e in re.findall(r"\d+ \.([a-z0-9]+)", cand["files"])]
    return lic, kind, exts


def formats_of(names):
    count = {}
    for n in names:
        e = Path(str(n).lower()).suffix.lstrip(".")
        f = FORMAT.get(e)
        if f:
            count[f] = count.get(f, 0) + 1
    order = sorted(count, key=lambda f: (f in ("Python", "R", "ZIP"), -count[f]))
    return order[:3]


def build_entry(h, cand, idx, vmap, taken):
    s = cand["screen"]
    doi = cand["doi"]
    meta = paper_meta(h, doi)
    if not meta:
        return None, "the paper DOI has no Crossref or DataCite record"
    jdoi = meta["journal_doi"] or C.norm_doi(cand.get("published_doi"))
    if meta["preprint"] and jdoi and jdoi != doi:
        jm = paper_meta(h, jdoi)
        if jm:
            meta = jm
    if meta["doi"] in idx["dois"]:
        return None, f"the journal version {meta['doi']} is already listed"
    sid = make_id(meta, vmap, taken)
    lic_raw, kind, names = deposit_info(h, cand["data_url"], cand)
    lic = norm_license(lic_raw)
    note = None
    if not lic and kind == "osf":
        note = "No license is stated on the OSF project."
    elif not lic:
        note = "No license is stated on the deposit."
    url = cand["data_url"]
    entry = {"id": sid, "label": label_of(meta), "year": int(meta["year"]) if meta["year"] else dt.date.today().year,
             "lab": lab_of(h, meta["doi"], meta),
             "papers": [{"authors": meta["authors"], "year": int(meta["year"]) if meta["year"] else None, "title": meta["title"],
                         "venue": meta["venue"], "doi": meta["doi"]}],
             "data_url": url, "host": cand.get("host") or C.host_label(url), "formats": formats_of(names),
             "n_participants": s.get("n_participants"), "task": s["task"], "methods": s.get("methods") or [],
             "populations": s.get("populations") or [], "license": lic, "license_note": note, "extra_tags": [],
             "found_by": {"finder": dt.date.today().isoformat(), "how": cand.get("why_found", "")[:200]}}
    abstract = cand.get("abstract") or meta.get("abstract") or ""
    return entry, abstract


def ascii_text(t):
    return (t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("—", " - ").replace("–", "-")
             .replace("−", "-").replace(" ", " "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screened", default=str(C.WORK / "screened.jsonl"))
    ap.add_argument("--out", default=str(C.WORK / "new_entries.json"))
    ap.add_argument("--state", default=str(C.WORK / "state.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [r for r in C.read_jsonl(a.screened) if (r.get("screen") or {}).get("include") == "yes"]
    if a.limit:
        rows = rows[:a.limit]
    idx = C.listed_index()
    vmap = venue_map(idx["extra"])
    taken = set(idx["ids"])
    h = C.Http()
    state = C.load_json(a.state, C.empty_state())
    state.setdefault("pending", {})
    entries, abstracts, problems = [], {}, []
    for r in rows:
        try:
            e, extra = build_entry(h, r, idx, vmap, taken)
        except Exception as ex:  # noqa: BLE001
            e, extra = None, f"{type(ex).__name__}: {str(ex)[:150]}"
        if not e:
            problems.append({"doi": r["doi"], "title": r.get("title", "")[:150], "problem": extra})
            continue
        entries.append(e)
        if extra and len(extra) > 200:
            abstracts[e["papers"][0]["doi"]] = {"text": ascii_text(extra), "source": "finder"}
        state["pending"][e["id"]] = {"listed": dt.date.today().isoformat(), "doi": e["papers"][0]["doi"], "data_url": e["data_url"],
                                     "attempts": 0, "data_level": r["screen"].get("data_level"), "cid": r["cid"]}
        print(f"listing: {e['id']}  {e['label']}  [{e['task']}]  {e['data_url']}")
    C.save_json(a.out, entries)
    C.save_json(Path(a.out).with_name("new_abstracts.json"), abstracts)
    C.save_json(Path(a.out).with_name("listing_problems.json"), problems)
    C.save_json(a.state, state)
    print(f"listing: {len(entries)} new entries, {len(problems)} not listed")


if __name__ == "__main__":
    main()
