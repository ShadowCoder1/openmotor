"""Build the OpenMotor site into dist/.

Inputs
  site/template.html, site/catalog.json, site/columns.json      page, study catalog, data dictionary
  source/release/manifest.json                                   from tools/pc_export_release.py (PC)
  source/study_facts.json                                        from tools/study_facts.py (tags)
  source/release/*.zip                                           the per-dataset download files
  source/release_config.json                                     which license kinds are published
Outputs
  dist/index.html     complete document for a static host (GitHub Pages, Vercel)
  dist/artifact.html  page content only, for the private claude.ai preview
  dist/catalog.csv, dist/columns.csv
  dist/data/*.zip     only the datasets whose license kind is published
"""
import csv
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "dist"
REL = HERE / "source/release"
(OUT / "data").mkdir(parents=True, exist_ok=True)
shutil.rmtree(OUT / "img", ignore_errors=True)
(OUT / "img").mkdir()
shutil.copy2(HERE / "source/figures/openmotor_logo.png", OUT / "img/openmotor-logo.png")  # the OpenMotor wordmark
shutil.copy2(HERE / "source/figures/openmotor_insights.png", OUT / "img/openmotor-insights.png")  # Standardization page, Insights (the user, 2026-11-08)
# Home page: the Physical Intelligence Lab logo (from the user) and the Carnegie Mellon wordmark (the user's logo with the
# red square removed, set in Carnegie Red #C41230 like the lab logo's red), 2026-11-08. Both are 216 px tall (3x their 72 px display).
shutil.copy2(HERE / "source/figures/pil_logo.png", OUT / "img/pil-logo.png")
shutil.copy2(HERE / "source/figures/cmu_wordmark.png", OUT / "img/cmu-wordmark.png")
# tab icon: the "O" and "M" cut from the OpenMotor wordmark, on white (the user, 2026-11-08; was a white "OM" on red)
shutil.copy2(HERE / "source/figures/favicon_32.png", OUT / "img/favicon-32.png")
shutil.copy2(HERE / "source/figures/favicon_64.png", OUT / "img/favicon-64.png")
shutil.copy2(HERE / "source/figures/apple_touch_icon.png", OUT / "img/apple-touch-icon.png")

catalog = json.load(open(HERE / "site/catalog.json", encoding="utf-8"))
columns = json.load(open(HERE / "site/columns.json", encoding="utf-8"))
cfg = json.load(open(HERE / "source/release_config.json"))
manifest = json.load(open(REL / "manifest.json"))

# ---- download files
files, published = [], set(cfg["publish_kinds"])
for f in (OUT / "data").glob("*.zip"):
    f.unlink()
# 2026-11-10: the first release's ZIPs are GitHub release assets (tag cfg["bundle_tag"], next to the all-in-one bundle),
# no longer copied into the Pages site: Pages' file sync kept timing out on the 126 MB of ZIPs (5 failed deploys 11-09/10).
ORIG_URL = f"https://github.com/{cfg['repo']}/releases/download/{cfg['bundle_tag']}/"
for ds, m in manifest.items():
    status = "available" if m["kind"] in published else "pending"
    files.append({"name": m["name"], "file": m["file"], "bytes": m["bytes"], "rows": m["rows"], "persons": m["persons"],
                  "subjects": m["subjects"], "license": cfg["license_labels"][m["kind"]], "kind": m["kind"],
                  "status": status, "catalog_study": m["catalog_study"], "sha256": m["sha256"]})
    if status == "available":
        files[-1]["url"] = ORIG_URL + m["file"]
files.sort(key=lambda f: f["name"].lower())
all_ids = {s["id"] for s in catalog["studies"]}
assert {f["catalog_study"] for f in files} == all_ids, "every study needs its files"
(OUT / "SHA256SUMS.txt").write_text("".join(f"{f['sha256']}  data/{f['file']}\n" for f in files if f["status"] == "available"))

# ---- counts for every study from the export (formerly held-out studies had none in the catalog)
PNAME = {"abrupt_rotation": "Abrupt rotation", "gradual_rotation": "Gradual rotation", "clamp": "Error clamp",
         "cue_context": "Contextual cues", "reward_score": "Reward or score", "no_perturbation": "No perturbation",
         "mirror_gain": "Mirror or gain", "other_mapping": "Other mapping", "target_jump": "Target jump"}
DSI = json.load(open(HERE / "source/datasets.json"))
by_cs = {}
for ds, m in manifest.items():
    by_cs.setdefault(m["catalog_study"], []).append((ds, m))
exp_counts = {}
for ds, m in manifest.items():
    for e, ps in m["paradigms"].items():
        for p in ps:
            exp_counts[PNAME[p]] = exp_counts.get(PNAME[p], 0) + 1
n_exp = sum(len(m["experiments"]) for m in manifest.values())
assert n_exp == catalog["meta"]["total"]["experiments"], n_exp
for s in catalog["studies"]:
    if s["side"] != "lockbox":
        continue
    (ds, m), = by_cs[s["id"]]
    tags = {}
    for ps in m["paradigms"].values():
        for p in ps:
            tags[PNAME[p]] = tags.get(PNAME[p], 0) + 1
    label = cfg["license_labels"][m["kind"]]
    s.update(people=m["persons"], trials=m["rows"], experiments=len(m["experiments"]), lab=DSI[ds]["lab"],
             paradigms=sorted(tags, key=lambda k: -tags[k]), license=label, licenses=[[label, ""]])
for s in catalog["studies"]:
    s["side"] = "development"  # the public site no longer separates a held-out set

# ---- tags: how each study was run (from its released data) and what it is about (from its paper titles)
import re
FACTS = json.load(open(HERE / "source/study_facts.json"))
FACETS = [["task", "Task"], ["manipulation", "Manipulation"], ["population", "Population"], ["setting", "Setting"],
          ["device", "Device"], ["measure", "Measures"], ["design", "Design"], ["topic", "Topics"]]
MANIP_ORDER = ["Abrupt rotation", "Gradual rotation", "Error clamp", "Mirror or gain", "Target jump", "Other mapping",
               "Contextual cues", "Reward or score", "No perturbation"]
POP = {"parkinsons": "Parkinson's disease", "cerebellar_degeneration": "cerebellar degeneration",
       "low_vision": "low vision", "deafferented": "deafferentation"}
POP_FROM_TITLE = {  # named in the paper title but not coded in the population column of the data
    "Barzi2026_ExpBrainRes": "anxiety disorders", "Stroh2024_JDeafStudDeafEduc": "deaf adults",
    "Cipleu2026_NeurorehabilNeuralRepair": "low vision", "Cisneros2026_NatHumBehav": "older adults"}
DEVICE = {"mouse": "mouse", "trackpad": "trackpad", "mouse_or_trackpad": "mouse or trackpad", "touchscreen": "touchscreen",
          "tablet": "digitizing tablet", "kinarm": "KINARM", "robot": "robot"}
TOPICS = [  # (tag, pattern searched in the paper titles)
    ("implicit learning", r"\bimplicit"), ("explicit strategies", r"\bexplicit|strateg"),
    ("reward-based learning", r"reward|reinforcement"), ("savings and relearning", r"\bsavings|relearn|re-?exposure"),
    ("motor memory", r"(?<!working )memor|forgetting|retention"), ("proprioception", r"propriocept|hand position"),
    ("uncertainty", r"uncertaint"), ("attention", r"\battention"), ("working memory", r"working memory"),
    ("dual task", r"dual-?task"), ("decision making", r"decision"), ("individual differences", r"individual differences|predictors of"),
    ("aging", r"\bageing\b|\baging\b|age-related|lifespan|\belderly\b|with age\b|effect of age\b"),
    ("intermanual transfer", r"intermanual|interlimb|across hands|untrained arm|effector-independent"), ("prism adaptation", r"prism"),
    ("feedback timing", r"delayed feedback|feedback duration"), ("brain stimulation", r"\btdcs\b|direct-current stimulation"),
    ("virtual reality", r"virtual reality")]

def slug(f, v):
    return re.sub(r"[^a-z0-9]+", "-", ((v if f == "topic" else f + " " + v).replace("'", "")).lower()).strip("-")

for s in catalog["studies"]:
    fx = [(m["name"], FACTS[m["name"]]) for _, m in by_cs[s["id"]]]
    tags = [("task", "reaching")]
    ROT = {"Abrupt rotation": "visuomotor rotation", "Gradual rotation": "visuomotor rotation"}  # one method; gradual onset is a design tag
    tags += list(dict.fromkeys(("manipulation", ROT.get(p, p.lower())) for p in MANIP_ORDER if p in s.get("paradigms", [])))
    pops = {POP[p] for _, f in fx for p in f["population"] if p in POP} | {POP_FROM_TITLE[n] for n, _ in fx if n in POP_FROM_TITLE}
    for _, f in fx:
        ages = list(f["group_age_median"].values()) or ([f["age_median"]] if f["age_median"] is not None else [])
        if any(a < 18 for a in ages):
            pops.add("children")
        if any(a >= 60 for a in ages):
            pops.add("older adults")
    tags += [("population", p) for p in sorted(pops)]
    tags += [("setting", v) for v in ["lab", "online"] if any(v in f["setting"] for _, f in fx)]
    devs = {d for _, f in fx for d in f["device"] if d in DEVICE}
    if "mouse_or_trackpad" in devs and devs & {"mouse", "trackpad"}:
        devs.discard("mouse_or_trackpad")
    tags += [("device", DEVICE[d]) for d in DEVICE if d in devs]
    if any(f["rt"] for _, f in fx):
        tags.append(("measure", "reaction time"))
    if any(f["aim_reports"] for _, f in fx):
        tags.append(("measure", "aim reports"))
    if any(f["hand_reports"] for _, f in fx):
        tags.append(("measure", "proprioceptive reports"))
    if any(f["sessions_max"] >= 1 for _, f in fx):
        tags.append(("design", "multiple sessions"))
    if "Gradual rotation" in s.get("paradigms", []):
        tags.append(("design", "gradual perturbation"))
    if any("given_with_angle" in f["instr_strategy"] for _, f in fx):
        tags.append(("design", "instructed strategy"))
    titles = " ".join(p["title"] for p in s["papers"])
    tags += [("topic", t) for t, pat in TOPICS if re.search(pat, titles, re.I)]
    s["tags"] = [{"f": f, "v": v, "slug": slug(f, v)} for f, v in tags]
assert len({t["slug"] for s in catalog["studies"] for t in s["tags"]}) == len({(t["f"], t["v"]) for s in catalog["studies"] for t in s["tags"]})

# ---- abstracts (tools/fetch_abstracts.py, by DOI; matched on title)
ABS = json.load(open(HERE / "source/abstracts.json"))
for s in catalog["studies"]:
    for p in s["papers"]:
        text = ABS.get(p["doi"], {}).get("text", "")
        p["abstract"] = "\n\n".join(x for x in text.split("\n\n") if not x.startswith("Editorial note"))
missing_abs = [p["doi"] for s in catalog["studies"] for p in s["papers"] if not p["abstract"] and p["doi"].startswith("10.")]
assert not missing_abs, missing_abs

# ---- original data: where the authors shared it (links from build_catalog.py) and in what file formats
for s in catalog["studies"]:
    for l in s["links"]:
        if l["url"].startswith("https://doi.org/10.12751/g-node."):  # G-Node DOIs still resolve to a retired page (404)
            l["url"] = "https://doi.gin.g-node.org/" + l["url"].split("doi.org/", 1)[1] + "/"
ORIG_FMT = json.load(open(HERE / "source/original_formats.json"))
ORIG_FMT.update({  # checked by hand: Korka's files have no extension but are plain text; Li's 7-Zip archive holds
    "Korka2023_Cortex": {"text": 180},          # SPSS files; Todorov's data are text .log files
    "Li2021_FrontAgingNeurosci": {"SPSS": 4},
    "Todorov2025_bioRxiv": {"text": 52}})
FMT_LABEL = {"text": "Text", "R": "R data", "Python pickle": "Python pickle", "NumPy": "NumPy"}
for s in catalog["studies"]:
    count = {}
    for _, m in by_cs[s["id"]]:
        for k, n in ORIG_FMT[m["name"]].items():
            count[k] = count.get(k, 0) + n
    assert count, ("no original format for", s["id"])
    s["formats"] = [FMT_LABEL.get(k, k) for k, _ in sorted(count.items(), key=lambda kv: -kv[1])[:2]]
meta = dict(catalog["meta"])
fmt_studies = {}  # Beta page: how many standardized studies were shared in each kind of file
for s in catalog["studies"]:
    kinds = {FMT_LABEL.get(k, k) for _, m in by_cs[s["id"]] for k in ORIG_FMT[m["name"]]}
    for k in kinds:
        fmt_studies[k] = fmt_studies.get(k, 0) + 1
meta["beta"] = {"formats": [{"name": k, "studies": v} for k, v in sorted(fmt_studies.items(), key=lambda kv: (-kv[1], kv[0]))]}
meta["paradigms"] = sorted(({"name": k, "experiments": v} for k, v in exp_counts.items()), key=lambda r: -r["experiments"])
meta["development"] = {"studies": len(catalog["studies"]), "experiments": n_exp,
                       "people": sum(s["people"] for s in catalog["studies"]), "trials": sum(s["trials"] for s in catalog["studies"])}
assert meta["development"]["trials"] == meta["total"]["trials"], meta["development"]

# ---- studies listed with a link to the authors' data only (no standardized file yet): source/extra_studies.json
EXTRA_PATH = HERE / "source/extra_studies.json"
EXTRA = json.load(open(EXTRA_PATH)) if EXTRA_PATH.exists() else []
n_standardized = len(catalog["studies"])
for e in EXTRA:
    lic = e.get("license") or "Not stated"
    s = {"id": e["id"], "side": "development", "source": "linked", "lab": e.get("lab", ""), "papers": e["papers"], "year": e["year"],
         "label": e["label"], "people": e.get("n_participants"), "trials": None, "experiments": None, "paradigms": [],
         "license": lic, "licenses": [[lic, e.get("license_note", "")]], "links": [{"url": e["data_url"], "host": e["host"]}],
         "formats": e.get("formats") or []}
    for p in s["papers"]:
        p["abstract"] = "\n\n".join(x for x in ABS.get(p["doi"], {}).get("text", "").split("\n\n") if not x.startswith("Editorial note"))
    tags = [("task", e["task"])] + [("manipulation", m) for m in e.get("methods", [])] + [("population", q) for q in e.get("populations", [])]
    tags += [(f, v) for f, v in e.get("extra_tags", [])]  # measure and setting tags of the 2026-11-08 expansion
    titles = " ".join(p["title"] for p in s["papers"])
    tags += [("topic", tp) for tp, pat in TOPICS if re.search(pat, titles, re.I)]
    s["tags"] = []
    for f, v in dict.fromkeys(tags):
        s["tags"].append({"f": f, "v": v, "slug": slug(f, v)})
    catalog["studies"].append(s)
# the headline participant count leaves out two very large online datasets (an online game with 854,064 players and a typing
# test with 168,000 typists); both stay listed (owner's decision, 2026-11-08)
HEADLINE_EXCLUDE = {"Stafford2014_PsycholSci", "Dhakal2018_CHI"}
meta["total"] = dict(meta["total"], studies=len(catalog["studies"]),
                     people=sum(s["people"] or 0 for s in catalog["studies"] if s["id"] not in HEADLINE_EXCLUDE))

# ---- standardized datasets of other tasks (tools/std; not in the motor-centaur corpus): source/release_extra/<id>.zip
import hashlib as _hl
REX, RREC = HERE / "source/release_extra", HERE / "source/release_extra_records"
EXTRA_IDS = {e["id"]: e for e in EXTRA}
extra_files = []
n_skipped = 0
for rp in sorted(RREC.glob("*.json")):
    if rp.name.endswith(".skip.json"):
        n_skipped += 1
        continue
    r = json.load(open(rp))
    z = REX / f"{r['id']}.zip"
    assert r["id"] in EXTRA_IDS, f"standardized {r['id']} has no entry in source/extra_studies.json"
    e = EXTRA_IDS[r["id"]]
    entry = {"name": r["id"], "file": f"{r['id']}.zip", "rows": r["rows"], "persons": r.get("persons", r["subjects"]),
             "subjects": r["subjects"], "license": e.get("license") or "Not stated", "kind": "extra", "status": "available",
             "catalog_study": r["id"], "module": r["module"]}
    if r.get("url"):  # hosted as a GitHub release asset (tools/std/upload.py)
        url = r["url"].replace("https://github.com/ShadowCoder1/openmotor/", f"https://github.com/{cfg['repo']}/")  # repo moved 2026-11-08
        entry.update(url=url, bytes=r["zip_bytes"], sha256=r["zip_sha256"])
    elif z.exists():
        shutil.copy2(z, OUT / "data" / z.name)
        entry.update(bytes=z.stat().st_size, sha256=_hl.sha256(z.read_bytes()).hexdigest())
    else:
        continue
    extra_files.append(entry)
files += extra_files
(OUT / "SHA256SUMS.txt").write_text("".join(f"{f['sha256']}  {f.get('url') or 'data/' + f['file']}\n" for f in files if f["status"] == "available"))
meta["standardizedExtra"] = {"studies": len(extra_files), "skipped": n_skipped, "rows": sum(f["rows"] for f in extra_files),
                             "people": sum(f["persons"] for f in extra_files if f["name"] not in HEADLINE_EXCLUDE),
                             "reaching": sum(1 for f in extra_files if f["module"] == "reach90")}

# formats of the other tasks: the core columns and each task module (tools/std/stdlib.py, tools/std/modules/*.py)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("om_stdlib", HERE / "tools/std/stdlib.py")
_std = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_std)
MODULE_LABEL = {"speech": "Speech", "walking": "Walking", "saccade": "Saccades", "tapping": "Tapping", "keypress": "Key presses"}
task_formats = [{"name": "Core", "description": "Columns every dataset of another task has, before its task columns.",
                 "columns": [{"name": c[1], "type": c[2], "values": c[3], "description": c[4]} for c in _std.CORE]}]
for mf in sorted((HERE / "tools/std/modules").glob("*.py")):
    _ms = _ilu.spec_from_file_location(f"om_mod_{mf.stem}", mf)
    _m = _ilu.module_from_spec(_ms)
    _ms.loader.exec_module(_m)
    task_formats.append({"name": MODULE_LABEL.get(mf.stem, mf.stem.title()), "description": _m.DESCRIPTION,
                         "columns": [{"name": c[0], "type": c[1], "values": c[2], "description": c[3]} for c in _m.COLUMNS]})
meta["standardizedStudies"] = n_standardized
meta["repo"] = cfg.get("repo")
meta["facets"] = FACETS
meta["submitEndpoint"] = cfg.get("submit_endpoint") or None
_bundle = HERE / "source/bundle/openmotor-all-datasets.zip"
if cfg.get("bundle_tag") and (_bundle.exists() or cfg.get("bundle_bytes")):
    meta["bundle"] = {"url": f"https://github.com/{cfg['repo']}/releases/download/{cfg['bundle_tag']}/openmotor-all-datasets.zip",
                      "bytes": _bundle.stat().st_size if _bundle.exists() else cfg["bundle_bytes"]}

# ---- catalog.csv (one row per study) and columns.csv
by_study = {}
for f in files:
    by_study.setdefault(f["catalog_study"], []).append(f)
def first_author(s):
    a = s["papers"][0]["authors"]
    return (a[3:] if a.lower().startswith("'t ") else a).lower(), s["year"]
with open(OUT / "catalog.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)  # follows the Studies table; standardized downloads are left out for now (JT, 2026-11-07)
    w.writerow(["study", "year", "authors", "title", "venue", "doi", "lab", "participants", "license", "tags", "data", "format"])
    for s in sorted(catalog["studies"], key=first_author):
        P = s["papers"]
        w.writerow([s["label"], s["year"], " | ".join(p["authors"] for p in P), " | ".join(p["title"] for p in P),
                    " | ".join(p["venue"] for p in P), " | ".join(p["doi"] for p in P), s.get("lab", ""),
                    s.get("people") or "", s.get("license", ""),
                    "; ".join(t["v"] if t["f"] == "topic" else t["f"] + ": " + t["v"] for t in s["tags"]),
                    " | ".join(l["url"] for l in s["links"]) or "view-only link in the paper", ", ".join(s["formats"])])
shutil.copy2(HERE / "source/columns.csv", OUT / "columns.csv")

# ---- per-study plot files (tools/beta_build.py) are not published: JT took 'Check a study' off the site (2026-11-08).
# They stay in source/beta/; set SHIP_BETA_PLOTS to publish them again with the page in site/template_v10_beta_full.html.
SHIP_BETA_PLOTS = False
shutil.rmtree(OUT / "beta", ignore_errors=True)
if SHIP_BETA_PLOTS:
    (OUT / "beta").mkdir()
    for f in (HERE / "source/beta").glob("*.json"):
        if f.name != "index.json":
            shutil.copy2(f, OUT / "beta" / f.name)

# ---- sample rows for the Data Format page (real rows from a published file)
import io as _io, zipfile as _zf
SNAME = "Cipleu2026_NeurorehabilNeuralRepair"
SCOLS = ["subject", "trial", "phase", "target_angle", "rotation", "fb_visible", "fb_mode", "hand_angle", "cursor_error", "rt_ms"]
NUMERIC = ["trial", "target_angle", "rotation", "hand_angle", "cursor_error", "rt_ms"]
# The ZIP is only on the curator's computer; the rows are kept in source/sample_rows.json so the site builds anywhere.
SAMPLE_JSON = HERE / "source/sample_rows.json"
if (REL / f"{SNAME}.zip").exists():
    with _zf.ZipFile(REL / f"{SNAME}.zip") as z:
        rdr = csv.DictReader(_io.TextIOWrapper(z.open(f"{SNAME}/{SNAME}.csv"), encoding="utf-8"))
        srows = [r for r in rdr if r["subject"] == f"{SNAME}_S0001" and 7 <= int(r["trial"]) <= 13]
    SAMPLE_JSON.write_text(json.dumps(srows, indent=0))
else:
    srows = json.load(open(SAMPLE_JSON))
def _fv(c, v):
    if c in NUMERIC and v not in ("", None) and c != "trial":
        return f"{float(v):.1f}"
    return v
sample = {"dataset": SNAME, "columns": SCOLS, "numeric": NUMERIC, "from": int(srows[0]["trial"]), "to": int(srows[-1]["trial"]),
          "rows": [[_fv(c, r[c]) for c in SCOLS] for r in srows]}
assert len(srows) == 7

# ---- page
payload = {"meta": meta, "studies": catalog["studies"], "files": files, "columns": columns, "sample": sample, "taskFormats": task_formats}
LIC_NOTE = {  # plain hover notes; each README keeps the exact source of the license
    "None stated": "The authors posted these data without a license.",
    "CC0": "The authors released these data under CC0 (public domain).",
    "CC BY 4.0": "The authors released these data under CC BY 4.0.",
    "GPL-3.0": "The authors released these data under the GNU GPL version 3.",
    "MIT": "The authors released these data under the MIT license.",
    "CC BY-NC 4.0": "The authors released these data under CC BY-NC 4.0 (non-commercial use only).",
    "Not stated": "The authors posted these data without a license.",
    "CC BY-SA 4.0": "The authors released these data under CC BY-SA 4.0.",
    "CC BY-NC-SA 4.0": "The authors released these data under CC BY-NC-SA 4.0 (non-commercial use only).",
    "CC BY-NC-ND 4.0": "The authors released these data under CC BY-NC-ND 4.0 (non-commercial use, no derivatives).",
    "ODC-By 1.0": "The authors released these data under the Open Data Commons Attribution License.",
    "ODbL 1.0": "The authors released these data under the Open Database License.",
    "PDDL": "The authors released these data under the Open Data Commons Public Domain Dedication.",
    "Apache 2.0": "The authors released these data under the Apache 2.0 license.",
    "BSD": "The authors released these data under a BSD license.",
    "CC BY 3.0 US": "The authors released these data under CC BY 3.0 US.",
    "CC BY-ND 4.0": "The authors released these data under CC BY-ND 4.0 (no derivatives).",
    "AFL 3.0": "The authors released these data under the Academic Free License 3.0.",
    "Etalab 2.0": "The authors released these data under the Etalab Open License 2.0.",
    "Custom (non-commercial)": "The authors' own license allows non-commercial use; see the deposit for its terms.",
    "Custom (academic use)": "The authors' own license allows academic use; see the deposit for its terms.",
    "Creative Commons (version not stated)": "The authors chose a Creative Commons license without saying which one.",
    "Other open license": "The authors released these data under an open license; see the deposit for its terms.",
}
for s in payload["studies"]:
    # links stay: each study's Original data column points to the authors' own deposit (JT, 2026-11-06)
    if "licenses" in s:
        s["licenses"] = [[l[0], LIC_NOTE.get(l[0], l[1])] for l in s["licenses"]]
        if s["source"] == "openmotor":
            s["licenses"] = [["CC BY 4.0", "OpenMotor shares these data under CC BY 4.0."]]
        if s["id"] == "Clayton2024_PLoSONE":
            s["licenses"] = [["CC BY 4.0", "The authors' processed data are CC BY 4.0; the raw files this dataset was converted from have no license."]]
SHOW_ABSTRACTS = False  # JT 2026-11-07: each study is its title, tags and data; abstracts stay in source/abstracts.json
if not SHOW_ABSTRACTS:
    for s in payload["studies"]:
        for p in s["papers"]:
            p.pop("abstract", None)
tpl = (HERE / "site/template.html").read_text(encoding="utf-8")
# ---- page parts (site/parts/<name>.html|.css|.js, added 2026-11-10): extra views such as Explore and the upload flow.
# Each .js registers window.OM_PARTS[<name>] before the main script runs; site/parts/ORDER lists the parts to include.
PARTS_DIR = HERE / "site/parts"
_order = [l.strip() for l in (PARTS_DIR / "ORDER").read_text().splitlines() if l.strip() and not l.startswith("#")] if (PARTS_DIR / "ORDER").exists() else []
_ph, _pc, _pj = [], [], []
for _n in _order:
    for _ext, _acc in (("html", _ph), ("css", _pc), ("js", _pj)):
        _f = PARTS_DIR / f"{_n}.{_ext}"
        if _f.exists():
            _acc.append(_f.read_text(encoding="utf-8"))
tpl = (tpl.replace("<!--PARTS-HTML-->", "\n".join(_ph)).replace("/*PARTS-CSS*/", "\n".join(_pc))
          .replace("<!--PARTS-JS-->", "\n".join("<script>\n" + j.replace("</script", "<\\/script") + "\n</script>" for j in _pj)))
# public settings for the parts (the API base for counts and uploads, data files for Explore)
payload["meta"]["api"] = cfg.get("api_base") or None
data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
page = tpl.replace("/*CATALOG*/", data)
assert "/*CATALOG*/" not in page
(OUT / "artifact.html").write_text(page, encoding="utf-8")
head, sep, rest = page.partition('<a class="skip"')
# Visitor and download counts (2026-11-10, the user): our own counter at <api_base>/api/hit (openmotor-api on Vercel;
# no cookies, no stored IP addresses; a visitor is a daily-salted hash). Only on the public index.html. Counts every page
# of the hash-routed site (#studies, #study/<id>, ...) and every click on a dataset ZIP link as a download.
# Downloads are also counted by GitHub itself (release asset download counts, read by <api_base>/api/stats).
API = cfg.get("api_base")
counter = ""
if API:
    assert re.fullmatch(r"https://[a-z0-9.-]+", API), API
    counter = ('<script>\n(function () {\n'
               f'  var U = "{API}/api/hit", last = "";\n'
               '  function send(o) { var b = JSON.stringify(o); try { if (!(navigator.sendBeacon && navigator.sendBeacon(U, b))) fetch(U, {method: "POST", body: b, keepalive: true}); } catch (e) {} }\n'
               '  function view() { var p = location.pathname + (location.hash || "#home").split("?")[0]; if (p === last) return; last = p; send({p: p, e: "view", r: document.referrer}); }\n'
               '  view(); window.addEventListener("hashchange", function () { setTimeout(view, 0); });\n'
               '  document.addEventListener("click", function (e) {\n'
               '    var a = e.target.closest && e.target.closest(\'a[href*="/releases/download/"]\');\n'
               '    if (a) send({p: "download/" + a.getAttribute("href").split("/").pop(), e: "download", r: location.hash});\n'
               '  }, true);\n})();\n</script>\n')
(OUT / "index.html").write_text('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                                 '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                                 + head + '</head>\n<body>\n' + sep + rest + '\n' + counter + '</body>\n</html>\n', encoding="utf-8")
n_av = sum(f["status"] == "available" for f in files)
print(f"page {len(page) // 1024} KB; downloads available {n_av}/{len(files)} "
      f"({sum(f['bytes'] for f in files if f['status'] == 'available') / 1e6:.1f} MB)")
