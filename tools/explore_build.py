"""Data for the Explore page: behaviour pooled across standardized studies, by family and by term.

Run:  python3 -I tools/explore_build.py            (then tools/build_page.py copies source/explore/site/* to dist/explore/)
      python3 -I tools/explore_build.py --no-fetch (offline: no ZIP or abstract downloads; uses only what is cached)
Needs python 3.12+, numpy and pandas. A dataset ZIP is read from source/release/ when it is there (the curator's
computer) and otherwise downloaded from its public release URL to a temporary folder, read, and deleted, one at a time.
Each dataset's result is cached in source/explore/ds/<dataset>.json together with the ZIP's sha256 and EXTRACT_VERSION,
so a rerun only reads datasets that are new or changed (new studies arrive weekly from the finder and from contributors;
nothing has to be done by hand for them to appear). Abstracts missing from source/abstracts.json are looked up by DOI
(tools/fetch_abstracts.py) and kept in source/explore/abstracts.json.

Outputs (committed with the site; only group means, never a participant code):
  source/explore/ds/<dataset>.json   cache: the lines of one dataset, all families
  source/explore/site/index.json     families and build facts (the page loads this first)
  source/explore/site/<family>.json  one family: its lines, its studies and its mined terms (loaded when chosen)


DATA CONTRACT: how to add a family
==================================
A family is one kind of experiment whose lines can sit on one y-axis and be pooled: reaching under a visuomotor
rotation ("rotation"), reaching with error-clamped feedback ("clamp"); later split-belt walking, speech perturbation,
saccade adaptation, force-field reaching, sequence learning, tapping. The page (site/parts/explore.js) knows nothing
about tasks: everything it draws comes from the family entry below and from the lines.

1. Add an entry to FAMILIES (the page lists them in this order):
     id       short slug used in the URL (#explore/<id>/<term>)
     label    menu name ("Visuomotor rotation");  task: the study tag task it belongs to ("reaching")
     about    one plain sentence: what a line is and what y means, shown under the curve
     y        y-axis name ("Hand angle");  unit: y unit as printed ("deg", "Hz", "%", "ms")
     pct      true when y / size * 100 is meaningful ("percent of the rotation"); the page then offers both
     size     name of the perturbation size ("Rotation", "Clamp", "Belt speed ratio") and sizeUnit
     x        x-axis name ("Trials from rotation onset");  window: [first, last] x shown by default
     sizeBins size ranges [low, high) offered by the size filter, next to every exact size
     measures list of {key, label, unit: "y" or "trials", text}; keys used by the reaching families are
              early, late, ae (aftereffect) and half (trials to half of late); reuse them when they apply
     module   the standardized module whose datasets feed it ("reach90"; see source/release_extra_records/*.json)
2. Write an extractor for the module and register it in EXTRACTORS: extractor(zip_path, ds) -> list of lines,
   where ds is the dataset entry from datasets() (name, study, paradigms, ...). One extractor may feed several
   families (reach90 feeds rotation and clamp); each line names its family. Bump EXTRACT_VERSION when an extractor
   changes, so cached datasets are read again.
3. Every line is one cohort (participants of one group who had the same perturbation schedule), aligned at its first
   perturbed trial (x = 0), with the sign flipped per participant so that compensating the perturbation is positive:
     id      "<dataset>|<experiment>|<k>", unique and stable between builds
     family  a FAMILIES id;  ds: dataset name (= files[].name of the page data);  study: catalog study id (studies[].id)
     exp     experiment id;  el: experiment label ("Experiment 2");  label: the line's group and schedule
     n       participants analysed (people, not sequences): those whose own first perturbation began at the line's onset
             with the line's size and stayed the same through the block (lines where this is under 90% of the group are
             not pooled);  nall: people in the group when it differs;  seq: sequences when a person gave more than one
     size    perturbation size in y units (> 0 after the flip); clamps and rotations in degrees
     x0      x of y[0] (<= 0: the baseline trials kept before onset);  y: group mean per trial, null where there was no
             feedback or no data; the line ends with its first perturbation block
     blk     trials in the first perturbation block;  nfb: its feedback trials
     m       {measure key: [mean, se, n]} in y units (trials for "half"); se from participants; missing keys = not defined
     a       line attributes used by term rules: pop (population code), age (median years), setting, device,
             fb (online or endpoint), strategy, goal, prior (share with earlier exposure), ae_instr
     ok      1 when the line enters the pooled curve and the meta-analysis; 0 with why = the plain reason
   Families whose datasets are not trial tables (walking strides, speech trials) still produce these fields; only the
   extractor differs. Terms need nothing per family: tag terms come from the study tags in the page, and mined terms are
   computed below from titles and abstracts of the studies that have lines in the family.
"""
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import warnings
import zlib
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "source/explore"
DS_DIR, SITE_DIR = OUT / "ds", OUT / "site"
ABS_CACHE = OUT / "abstracts.json"
EXTRACT_VERSION = 3      # bump when an extractor changes: every cached dataset is read again

FAMILIES = [
    {"id": "rotation", "label": "Visuomotor rotation", "task": "reaching", "module": "reach90",
     "about": "Each line is one group of participants who reached to targets while the cursor was rotated by the same amount on the same schedule. "
              "Hand angles are flipped so that moving against the rotation is positive; 100% means the hand fully cancels the rotation.",
     "y": "Hand angle", "unit": "deg", "pct": True, "size": "Rotation", "sizeUnit": "deg",
     "x": "Trials from rotation onset", "window": [-20, 300], "sizeBins": [[0, 25], [25, 40], [40, 55], [55, 180]],
     "measures": [
         {"key": "early", "label": "Early adaptation", "unit": "y", "text": "mean of feedback trials 2 to 11 of the first rotation block"},
         {"key": "late", "label": "Late adaptation", "unit": "y", "text": "mean of the last 20 feedback trials of the first rotation block"},
         {"key": "ae", "label": "Aftereffect", "unit": "y", "text": "mean of the first 10 no-feedback trials right after that block, when such a block follows it"},
         {"key": "half", "label": "Trials to half", "unit": "trials", "text": "rotation trials until the group mean (5-trial moving average) first reaches half of its late value"}]},
    {"id": "clamp", "label": "Error clamp", "task": "reaching", "module": "reach90",
     "about": "Each line is one group of participants who saw a cursor that always moved at a fixed angle from the target, whatever their hand did, and were told to ignore it. "
              "Hand angles are flipped so that moving away from the clamp (implicit adaptation) is positive. Clamp sizes differ between studies; the size filter keeps one range.",
     "y": "Hand angle", "unit": "deg", "pct": False, "size": "Clamp", "sizeUnit": "deg",
     "x": "Trials from clamp onset", "window": [-20, 300], "sizeBins": [[0, 5], [5, 25], [25, 180]],
     "measures": [
         {"key": "early", "label": "Early adaptation", "unit": "y", "text": "mean of feedback trials 2 to 11 of the first clamp block"},
         {"key": "late", "label": "Late adaptation", "unit": "y", "text": "mean of the last 20 feedback trials of the first clamp block"},
         {"key": "ae", "label": "Aftereffect", "unit": "y", "text": "mean of the first 10 no-feedback trials right after that block, when such a block follows it"},
         {"key": "half", "label": "Trials to half", "unit": "trials", "text": "clamp trials until the group mean (5-trial moving average) first reaches half of its late value"}]},
]

# ---------------------------------------------------------------- datasets

def load_json(p, default=None):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else default


def datasets():
    """Every standardized dataset: the first release (source/release/manifest.json, module reach90) and the later ones
    (source/release_extra_records/<id>.json with a public url). Returns dicts: name, study, module, url, sha256, local, paradigms."""
    cfg = load_json(HERE / "source/release_config.json", {})
    out = []
    man = load_json(HERE / "source/release/manifest.json", {})
    base = f"https://github.com/{cfg.get('repo')}/releases/download/{cfg.get('bundle_tag')}/"
    for m in man.values():
        out.append({"name": m["name"], "study": m["catalog_study"], "module": "reach90", "url": base + m["file"],
                    "sha256": m["sha256"], "local": HERE / "source/release" / m["file"], "paradigms": m.get("paradigms") or {}})
    for rp in sorted((HERE / "source/release_extra_records").glob("*.json")):
        if rp.name.endswith(".skip.json"):
            continue
        r = load_json(rp)
        if not r.get("url") or not r.get("zip_sha256"):
            continue
        url = r["url"].replace("https://github.com/ShadowCoder1/openmotor/", f"https://github.com/{cfg.get('repo')}/")  # repo moved 2026-11-08
        out.append({"name": r["id"], "study": r["id"], "module": r["module"], "url": url, "sha256": r["zip_sha256"],
                    "local": HERE / "source/release_extra" / f"{r['id']}.zip", "paradigms": {}})
    return out


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get_zip(ds, tmp):
    """(path, downloaded?) for a dataset's ZIP, checked against its sha256."""
    if ds["local"].exists() and sha256_of(ds["local"]) == ds["sha256"]:
        return ds["local"], False
    dest = Path(tmp) / f"{ds['name']}.zip"
    for attempt in range(3):
        try:
            req = urllib.request.Request(ds["url"], headers={"User-Agent": "OpenMotor-explore-build/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if sha256_of(dest) == ds["sha256"]:
                return dest, True
            print(f"  sha256 mismatch for {ds['name']}", flush=True)
        except Exception as e:  # network trouble: try again, then skip this dataset for this run
            print(f"  download failed for {ds['name']}: {type(e).__name__}", flush=True)
        time.sleep(3 + 5 * attempt)
    dest.unlink(missing_ok=True)
    return None, False

# ---------------------------------------------------------------- reach90 extractor

PRE = 20          # baseline trials kept before onset
MAX_X = 400       # perturbation trials kept after onset (the page shows -20 to 300 by default and up to 400)
LATE_N = 20       # feedback trials averaged for late adaptation
EARLY = (1, 11)   # feedback trials 2 to 11 (the first perturbed trial was planned before any error was seen)
AE_N = 10         # no-feedback trials averaged for the aftereffect
MIN_FB = 30       # feedback trials the first block needs to give early and late values
MIN_PEOPLE = 3
SMOOTH = 5        # moving average for trials to half
BOOT = 1000       # bootstrap resamples of participants for the SE of trials to half (200 left the SE unstable by up to 50%)
HALF_SE_MIN = 0.29  # floor for that SE: the error of rounding to whole trials (1 / sqrt(12)), so a line never gets infinite weight
FAMILY_OF_MODE = {"rotated": "rotation", "clamped": "clamp"}


def _beta():
    spec = importlib.util.spec_from_file_location("om_beta_build", HERE / "tools/beta_build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mode(s):
    s = s.dropna()
    return None if s.empty else s.value_counts().index[0]


def _stat(v):
    """[mean, se, n] of per-person values; se from the participants (sd / sqrt(n))."""
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < MIN_PEOPLE:
        return None
    return [round(float(v.mean()), 2), round(float(v.std(ddof=1) / math.sqrt(len(v))), 3), int(len(v))]


def _movavg(y, w=SMOOTH):
    """centered moving average that skips missing values (needs 3 of the w trials)."""
    ok = np.isfinite(y)
    z = np.where(ok, y, 0.0)
    k = np.ones(w)
    s = np.convolve(z, k, "same")
    c = np.convolve(ok.astype(float), k, "same")
    return np.where(c >= 3, s / np.maximum(c, 1), np.nan)


def _half(curve, xs):
    """perturbation trials until the smoothed curve first reaches half of its late value (None if it never learned)."""
    fin = np.isfinite(curve)
    if fin.sum() < MIN_FB:
        return np.nan
    late = np.nanmean(curve[fin][-LATE_N:])
    if not late > 0:
        return np.nan
    sm = _movavg(curve)
    hit = np.nonzero(sm >= late / 2)[0]
    return float(xs[hit[0]] + 1) if len(hit) else np.nan


def reach90_lines(zip_path, ds):
    bb = _beta()
    lines = []
    count = {}

    def on_line(exp, exp_label, subs, varies, d, sched, modes, nofb, series, **_):
        k = count[exp] = count.get(exp, -1) + 1
        vals = sched.to_numpy(dtype=float)
        T = len(vals)
        start = next((t for t in range(T) if np.isfinite(vals[t]) and vals[t] != 0), None)
        if start is None:
            return
        mode0 = modes.iloc[start]
        fam = FAMILY_OF_MODE.get(mode0)
        if fam is None:
            return
        P = float(vals[start])
        nf = nofb.fillna(False).to_numpy(dtype=bool)
        md = modes.to_numpy(dtype=object)
        has = d[d["hand_n"].notna()].groupby("trial").size().reindex(range(T), fill_value=0).to_numpy()
        end = start  # the first block: same perturbation; no-feedback probes and empty trials inside it do not end it
        for t in range(start + 1, T):
            v = vals[t]
            if np.isfinite(v):
                if v == P:
                    end = t
                    continue
                break
            if nf[t] or has[t] == 0:
                continue
            break
        is_fb = [np.isfinite(vals[t]) and vals[t] == P and not nf[t] and md[t] == mode0 for t in range(T)]
        nfb = sum(is_fb[start:end + 1])

        # participants analysed: those whose own first perturbation starts at the line's onset with the same size and
        # stays the same through the block (later blocks may differ, for example when their order was counterbalanced)
        person = d["person"].astype(str) if "person" in d.columns else d["subject"].astype(str)
        p_of = dict(zip(d["subject"], person))
        pn = d[np.isfinite(d["pert_n"].to_numpy(dtype=float))]
        own_first = pn[pn["pert_n"] != 0].groupby("subject")["trial"].min()
        inblk = pn[(pn["trial"] >= start) & (pn["trial"] <= end)].groupby("subject")["pert_n"].apply(lambda v: bool((v == P).all()))
        cons = [s for s in subs if own_first.get(s) == start and bool(inblk.get(s, False))]
        dA = d[d["subject"].isin(cons)]
        hasA = dA[dA["hand_n"].notna()].groupby("trial").size().reindex(range(T), fill_value=0).to_numpy()

        # group mean per trial: baseline window, then the first block (feedback trials only)
        hand = dA.groupby("trial")["hand_n"].mean().reindex(range(T))
        x0 = -min(PRE, start)
        y = []
        for x in range(x0, min(end - start, MAX_X) + 1):
            t = start + x
            if nf[t] or (x >= 0 and not is_fb[t]) or (x < 0 and np.isfinite(vals[t]) and vals[t] != 0):
                y.append(None)
            else:
                y.append(bb.r1(hand.iloc[t]) if np.isfinite(hand.iloc[t]) else None)

        blk = dA[(dA["trial"] >= start) & (dA["trial"] <= end) & dA["fb_mode"].eq(mode0) & dA["pert_n"].eq(P) & dA["hand_n"].notna()]
        blk = blk.sort_values(["subject", "trial"])
        g = blk.groupby("subject", sort=False)["hand_n"]

        def per_person(v):
            v = v.dropna()
            return v.groupby(v.index.map(p_of)).mean().to_numpy() if len(v) else np.array([])

        early = per_person(g.apply(lambda v: v.iloc[EARLY[0]:EARLY[1]].mean() if len(v) >= EARLY[1] else np.nan))
        late = per_person(g.apply(lambda v: v.tail(LATE_N).mean() if len(v) >= LATE_N else np.nan))
        m = {}
        if nfb >= MIN_FB:
            for key, v in (("early", early), ("late", late)):
                st = _stat(v)
                if st:
                    m[key] = st

        # aftereffect: a run of no-feedback trials right after the block (up to 2 empty trials between)
        ae_instr = None
        t = end + 1
        while t < T and hasA[t] == 0 and t <= end + 2:
            t += 1
        if t < T and nf[t] and nfb >= MIN_FB:
            run = []
            while t < T and nf[t] and len(run) < AE_N:
                if hasA[t] > 0:
                    run.append(t)
                t += 1
            if len(run) >= 3:
                ae_rows = dA[dA["trial"].isin(run) & dA["hand_n"].notna() & dA["fb_visible"].eq("none")]
                st = _stat(per_person(ae_rows.groupby("subject")["hand_n"].mean()))
                if st:
                    m["ae"] = st
                    ae_instr = _mode(ae_rows["instr_strategy"])

        # trials to half of the late value, with a bootstrap SE over participants
        if "late" in m and m["late"][0] > 2 * m["late"][1]:
            piv = blk.assign(p=blk["subject"].map(p_of)).pivot_table(index="p", columns="trial", values="hand_n", aggfunc="mean")
            xs = piv.columns.to_numpy() - start
            M = piv.to_numpy(dtype=float)
            h0 = _half(np.nanmean(M, axis=0), xs)
            if np.isfinite(h0):
                rng = np.random.default_rng(zlib.crc32(f"{ds['name']}|{exp}|{k}".encode()))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # a resample can leave a trial with no data
                    hb = np.array([_half(np.nanmean(M[rng.integers(0, len(M), len(M))], axis=0), xs) for _ in range(BOOT)])
                fin = hb[np.isfinite(hb)]
                se = round(max(float(fin.std(ddof=1)), HALF_SE_MIN), 2) if len(fin) >= 0.8 * BOOT else None
                m["half"] = [int(h0), se, int(len(M))]

        first = (dA if len(dA) else d).groupby("subject").first()
        age = pd.to_numeric(first["age"], errors="coerce").dropna() if "age" in first else pd.Series(dtype=float)
        a = {"pop": _mode(first["population"]), "age": round(float(age.median()), 1) if len(age) >= max(1, 0.5 * len(first)) else None,
             "setting": _mode(first["setting"]), "device": _mode(first["device"]) if "device" in first else None,
             "fb": _mode(blk["fb_visible"]), "strategy": _mode(blk["instr_strategy"]), "goal": _mode(blk["instr_goal"]),
             "prior": round(float((first["prior_exposure"].astype(str).str.lower() == "true").mean()), 2), "ae_instr": ae_instr}
        n_all = len({p_of[s] for s in subs if s in p_of})
        n_people = len({p_of[s] for s in cons})
        word = "rotation" if fam == "rotation" else "clamp"
        why = None
        if len(cons) < 0.9 * len(subs):
            why = f"participants in this group followed different {word} schedules"
        elif nfb < MIN_FB:
            why = f"the first {word} changed after {nfb} feedback trial{'' if nfb == 1 else 's'} (fewer than {MIN_FB}: a gradual or changing schedule)"
        elif n_people < MIN_PEOPLE:
            why = f"fewer than {MIN_PEOPLE} participants"
        elif a["goal"] == "hand_to_angle" or a["strategy"] == "given_with_angle":
            why = "participants were told where to aim"
        elif "late" not in m:
            why = f"fewer than {MIN_PEOPLE} participants with {LATE_N} usable feedback trials"
        line = {"id": f"{ds['name']}|{exp}|{k}", "family": fam, "ds": ds["name"], "study": ds["study"], "exp": exp, "el": exp_label,
                "label": series["label"], "n": n_people, "size": round(P, 2), "x0": x0, "y": y, "blk": int(end - start + 1), "nfb": int(nfb),
                "m": m, "a": a, "ok": 0 if why else 1}
        if n_all != n_people:
            line["nall"] = n_all  # people in the group, when some had another schedule and are left out
        if len(cons) != n_people:
            line["seq"] = len(cons)
        if why:
            line["why"] = why
        lines.append(line)

    bb.build_dataset(ds["name"], ds.get("paradigms") or {}, zip_path=zip_path, extra_cols=("person", "age", "device"), on_line=on_line)
    return lines


EXTRACTORS = {"reach90": reach90_lines}

# ---------------------------------------------------------------- texts and mined terms

def studies_text():
    """{study id: {"title": ..., "abstract": ..., "dois": [...]}} from site/catalog.json, source/extra_studies.json and the abstracts."""
    cat = load_json(HERE / "site/catalog.json", {"studies": []})["studies"] + load_json(HERE / "source/extra_studies.json", [])
    abs1 = load_json(HERE / "source/abstracts.json", {})
    abs2 = load_json(ABS_CACHE, {})
    out = {}
    for s in cat:
        titles, abstracts, missing = [], [], []
        for p in s["papers"]:
            titles.append(p["title"])
            a = (abs1.get(p.get("doi") or "", {}) or {}).get("text") or (abs2.get(p.get("doi") or "", {}) or {}).get("text") or ""
            a = "\n\n".join(x for x in a.split("\n\n") if not x.startswith("Editorial note"))
            if a:
                abstracts.append(a)
            elif (p.get("doi") or "").startswith("10."):
                missing.append(p)
        out[s["id"]] = {"title": " ".join(titles), "abstract": "\n\n".join(abstracts), "missing": missing}
    return out


def fetch_missing_abstracts(texts, wanted):
    """Look up abstracts that source/abstracts.json does not have (tools/fetch_abstracts.py), for the studies in `wanted`.
    Papers with no abstract anywhere are tried again after 30 days."""
    spec = importlib.util.spec_from_file_location("om_fetch_abstracts", HERE / "tools/fetch_abstracts.py")
    fa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fa)
    cache = load_json(ABS_CACHE, {})
    today = date.today()
    changed = False
    for sid in sorted(wanted):
        for p in texts.get(sid, {}).get("missing", []):
            c = cache.get(p["doi"])
            if c and (c.get("text") or (today - date.fromisoformat(c.get("tried", "2000-01-01"))).days < 30):
                continue
            got = None
            for name, fn in [("Europe PMC", fa.europepmc), ("Crossref", fa.crossref), ("OpenAlex", fa.openalex), ("Semantic Scholar", fa.s2)]:
                try:
                    r = fn(p["doi"])
                except Exception:
                    r = None
                if r and r[1] and len(fa.clean(r[1])) > 200 and fa.same_title(r[0], p["title"]):
                    got = {"text": fa.clean(r[1]), "source": name}
                    break
                time.sleep(0.3)
            cache[p["doi"]] = got or {"text": "", "source": None, "tried": today.isoformat()}
            changed = True
            print(f"  abstract {'found' if got else 'not found'}: {p['doi']}", flush=True)
    if changed:
        ABS_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")


# A controlled vocabulary of motor-learning terms (label, regular expressions; case is ignored). Each pattern should name
# the concept, not a common verb: "exploration" but not "explore", which abstracts use for "we explored whether".
VOCAB = [
    # learning processes
    ("implicit adaptation", [r"\bimplicit\w*"]),
    ("explicit learning", [r"\bexplicit\w*"]),
    ("strategy use", [r"\bstrateg(y|ies|ic)\b"]),
    ("re-aiming", [r"\bre-?aim\w*", r"\baiming (strateg\w*|direction\w*|location\w*|report\w*)"]),
    ("aim reports", [r"\baim(ing)? reports?\b", r"\breport(ed|ing)? (their|the) (intended )?aim", r"\breported aiming"]),
    ("savings", [r"\bsavings\b", r"\bfaster re-?learning\b"]),
    ("relearning", [r"\bre-?learn\w*", r"\bre-?exposure\b", r"\bre-?adapt\w*"]),
    ("generalization", [r"\bgenerali[sz]\w*"]),
    ("transfer", [r"\btransfer\w*"]),
    ("intermanual transfer", [r"\binter-?manual\b", r"\binter-?limb\b", r"\b(untrained|opposite|other|non-?trained) (hand|arm|limb)\b"]),
    ("consolidation", [r"\bconsolidat\w*"]),
    ("interference", [r"\binterferen\w*"]),
    ("anterograde interference", [r"\banterograde\b"]),
    ("retention", [r"\bretention\b", r"\bretain(ed|s)?\b"]),
    ("forgetting and decay", [r"\bforget\w*", r"\bforgot\w*", r"\bdecay\w*"]),
    ("spontaneous recovery", [r"\bspontaneous recovery\b", r"\brebound\w*"]),
    ("aftereffect", [r"\bafter-?effects?\b"]),
    ("washout and de-adaptation", [r"\bwash-?out\b", r"\bde-?adapt\w*"]),
    ("use-dependent learning", [r"\buse-? ?dependent\b"]),
    ("motor memory", [r"\bmotor memor\w*"]),
    ("long-term memory", [r"\blong-?term\b", r"\b24 ?h(ours?)?\b", r"\bnext day\b", r"\bone day later\b", r"\bdays? later\b"]),
    ("multiple sessions", [r"\b(two|three|four|five|several|multiple|consecutive|separate) (sessions|days)\b", r"\bacross (sessions|days)\b"]),
    ("sleep", [r"\bsleep\w*", r"\bnaps?\b"]),
    ("spacing of practice", [r"\bspac(ed|ing) (practice|training|trials)\b", r"\bdistributed practice\b", r"\bmassed\b", r"\binter-?trial interval\w*"]),
    ("practice schedule", [r"\bpractice schedule\w*", r"\bblocked (and|or|vs\.?|versus) random\b", r"\brandom practice\b", r"\bblocked practice\b", r"\bcontextual interference\b"]),
    ("amount of practice", [r"\bextended (practice|training)\b", r"\bprolonged (practice|training|exposure)\b", r"\bamount of (practice|training)\b"]),
    ("meta-learning", [r"\bmeta-?learning\b", r"\blearning to learn\b", r"\blearning-to-learn\b"]),
    ("structural learning", [r"\bstructur(al|e) learning\b"]),
    ("skill learning", [r"\bskill\w*"]),
    ("de novo learning", [r"\bde novo\b"]),
    ("dual adaptation", [r"\bdual adaptation\b", r"\bopposing (rotations?|perturbations?|force fields?|mappings?)\b", r"\b(two|opposite) (opposing|opposite) (rotations?|perturbations?)\b"]),
    ("trial-by-trial learning", [r"\btrial-?by-?trial\b", r"\btrial-?to-?trial\b", r"\bsingle-?trial\b"]),
    ("learning rate", [r"\blearning rates?\b", r"\brate of (learning|adaptation)\b", r"\bspeed of (learning|adaptation)\b"]),
    ("asymptote", [r"\basymptot\w*", r"\bsaturat\w*", r"\bplateau\w*"]),
    ("habit", [r"\bhabit\w*"]),
    ("procedural memory", [r"\bprocedural\b"]),
    ("declarative memory", [r"\bdeclarative\b"]),
    # errors and models
    ("error-based learning", [r"\berror-? ?based\b"]),
    ("sensory prediction error", [r"\bsensory prediction errors?\b", r"\bprediction errors?\b"]),
    ("task error", [r"\btask errors?\b", r"\bperformance errors?\b", r"\btarget errors?\b"]),
    ("error sensitivity", [r"\berror sensitivity\b", r"\bsensitivity to errors?\b"]),
    ("error size", [r"\berror (size|magnitude)s?\b", r"\b(size|magnitude) of (the )?errors?\b", r"\b(large|small|big) errors\b"]),
    ("error history", [r"\berror history\b", r"\bhistory of errors\b", r"\bmemory of errors\b"]),
    ("perturbation size", [r"\bperturbation (size|magnitude)s?\b", r"\b(rotation|clamp) (size|magnitude|angle)s?\b",
                           r"\b(size|magnitude) of the (rotation|perturbation|clamp)\b", r"\b(small|large|larger|smaller) (rotations?|perturbations?|clamps?)\b"]),
    ("state-space model", [r"\bstate-? ?space\b"]),
    ("two-rate model", [r"\btwo-? ?state\b", r"\bdual-? ?rate\b", r"\btwo-? ?rate\b", r"\bfast and slow\b", r"\bslow and fast\b", r"\bmultiple timescales?\b"]),
    ("computational model", [r"\bcomputational model\w*", r"\bmodel-? ?based\b", r"\bmodel fit\w*"]),
    ("Bayesian inference", [r"\bbayes\w*"]),
    ("internal model", [r"\binternal models?\b"]),
    ("forward model", [r"\bforward models?\b", r"\befference cop\w*", r"\bcorollary discharge\b"]),
    ("credit assignment", [r"\bcredit assignment\b", r"\bsource of (the )?errors?\b", r"\berror attribution\b", r"\battribut\w* (the )?errors?\b"]),
    ("uncertainty", [r"\buncertain\w*"]),
    ("variability", [r"\bvariabilit\w*"]),
    ("motor noise", [r"\bmotor noise\b", r"\bsensory noise\b", r"\bnoise\b", r"\bnoisy\b"]),
    ("exploration", [r"\bexploration\b", r"\bexplore-?exploit\w*", r"\bmotor exploration\b"]),
    ("individual differences", [r"\bindividual differences\b", r"\binter-?individual\b", r"\bbetween-?subject variab\w*", r"\bindividual variab\w*"]),
    # reward and success
    ("reward", [r"\brewards?\b", r"\brewarded\b", r"\brewarding\b"]),
    ("reinforcement learning", [r"\breinforcement\b"]),
    ("punishment", [r"\bpunish\w*"]),
    ("task success", [r"\btask success\b", r"\bsuccess (signals?|feedback|rates?)\b", r"\btarget hits?\b", r"\bhit(ting)? the target\b",
                      r"\bmiss(ing|ed)? the target\b", r"\bbinary feedback\b", r"\bhit or miss\b", r"\bhitting\b"]),
    ("motivation", [r"\bmotivat\w*"]),
    ("effort", [r"\beffort\w*"]),
    ("decision making", [r"\bdecision\w*"]),
    # cognition
    ("working memory", [r"\bworking memory\b"]),
    ("spatial working memory", [r"\bspatial working memory\b", r"\bmental rotation\b"]),
    ("attention", [r"\battention\w*"]),
    ("dual task", [r"\bdual-? ?task\w*", r"\bsecondary task\b", r"\bcognitive load\b"]),
    ("awareness", [r"\baware\w*", r"\bunaware\w*"]),
    ("instructions", [r"\binstruct\w*"]),
    ("cognitive function", [r"\bcognitive (control|function\w*|abilit\w*|resources?|processes|decline|impairment)\b", r"\bexecutive function\w*"]),
    ("anxiety", [r"\banxi\w*"]),
    ("sense of agency", [r"\bagency\b"]),
    ("motor imagery", [r"\bimagery\b", r"\bimagined\b", r"\bimagining\b"]),
    ("action observation", [r"\baction observation\b", r"\bobservational learning\b", r"\bobserving (another|others|a model)\b"]),
    # feedback and perturbation
    ("visuomotor rotation", [r"\bvisuomotor rotations?\b", r"\b(cursor|visual|feedback) rotations?\b", r"\brotated (cursor|feedback|visual feedback)\b"]),
    ("error clamp", [r"\b(error|visual) clamps?\b", r"\bclamped (visual )?feedback\b", r"\btask-?irrelevant (visual )?(feedback|errors?)\b", r"\binvariant (visual )?(feedback|errors?)\b"]),
    ("gradual perturbation", [r"\bgradual\w*", r"\bincremental\w*", r"\bramp\w*"]),
    ("abrupt perturbation", [r"\babrupt\w*", r"\bsudden\w*"]),
    ("random perturbations", [r"\brandom (perturbations?|rotations?|clamps?|errors?)\b", r"\bpseudo-?random\w*", r"\bunpredictab\w*"]),
    ("mirror reversal", [r"\bmirror\w*"]),
    ("visual feedback", [r"\bvisual feedback\b"]),
    ("feedback delay", [r"\bdelay(ed)? (visual |cursor )?feedback\b", r"\bfeedback delays?\b", r"\btemporal delays?\b", r"\bdelay(s|ed)? (of|in) (the )?(visual )?feedback\b"]),
    ("endpoint feedback", [r"\bend-?point feedback\b", r"\bterminal feedback\b", r"\bknowledge of results\b", r"\bendpoint (cursor|error)\b"]),
    ("online feedback", [r"\bonline (visual )?feedback\b", r"\bcontinuous (visual )?feedback\b", r"\bconcurrent feedback\b"]),
    ("no-feedback trials", [r"\bno-?feedback\b", r"\bwithout (visual )?feedback\b", r"\bno visual feedback\b", r"\babsence of (visual )?feedback\b"]),
    ("feedback uncertainty", [r"\bblur\w*", r"\bcursor cloud\b", r"\b(feedback|visual|sensory) uncertainty\b", r"\bnoisy (visual )?feedback\b", r"\bfeedback reliability\b"]),
    ("target jumps", [r"\btarget jumps?\b", r"\bcursor jumps?\b", r"\bjump(ed|ing|s)?\b"]),
    ("target size", [r"\btarget (size|width)s?\b", r"\bsize of the target\b", r"\b(large|small|larger|smaller) targets?\b"]),
    ("target direction", [r"\btarget directions?\b", r"\bmovement directions?\b", r"\bnumber of targets\b", r"\bdirectional\w*", r"\banisotrop\w*"]),
    ("contextual cues", [r"\bcontextual\b", r"\bcontexts?\b", r"\bcues?\b"]),
    ("catch trials", [r"\bcatch trials?\b", r"\bprobe trials?\b"]),
    ("perturbation awareness", [r"\b(aware|awareness) of the (rotation|perturbation|clamp)\b"]),
    # proprioception and multisensory
    ("proprioception", [r"\bpropriocept\w*", r"\bfelt (hand|position|arm)\b", r"\bposition sense\b"]),
    ("proprioceptive recalibration", [r"\brecalibrat\w*", r"\bproprioceptive (shift|change)s?\b"]),
    ("hand localization", [r"\blocali[sz]\w*"]),
    ("multisensory integration", [r"\bmultisensory\b", r"\bcue combination\b", r"\bsensory integration\b", r"\bvisuo-?proprioceptive\b", r"\bintegrat\w* (visual|sensory|proprioceptive)\b"]),
    ("sensory weighting", [r"\b(sensory|cue|visual|proprioceptive) (weight\w*|reliabilit\w*)", r"\breliability-? ?weight\w*", r"\bweighting of\b"]),
    ("vision of the hand", [r"\bvision of the (hand|arm)\b", r"\b(hand|arm) (was |is )?visible\b", r"\bvisible hand\b"]),
    # movement
    ("reaction time", [r"\breaction times?\b", r"\bresponse times?\b"]),
    ("preparation time", [r"\bpreparation times?\b", r"\bplanning times?\b", r"\btime pressure\b", r"\btime constraints?\b", r"\b(restricted|limited|shortened|forced) (preparation|reaction|response) times?\b"]),
    ("movement speed", [r"\bmovement (time|speed|duration|velocity)\b", r"\bvigou?r\b", r"\bpeak velocity\b"]),
    ("speed-accuracy trade-off", [r"\bspeed-? ?accuracy\b", r"\btrade-?offs?\b"]),
    ("online corrections", [r"\bonline correct\w*", r"\bfeedback (correction|response)s?\b", r"\bcorrective\b", r"\bfeedback control\w*", r"\bvisuomotor feedback\b"]),
    ("feedforward control", [r"\bfeed-?forward\b"]),
    ("motor planning", [r"\bmotor planning\b", r"\bmovement planning\b", r"\bmotor plans?\b", r"\baction planning\b"]),
    ("kinematics", [r"\bkinematic\w*", r"\btrajector\w*", r"\bcurvature\b"]),
    ("limb impedance", [r"\bimpedance\b", r"\bstiffness\b", r"\bco-?contraction\b"]),
    ("muscle activity", [r"\bEMG\b", r"\belectromyograph\w*", r"\bmuscle activit\w*"]),
    ("eye movements", [r"\beye movements?\b", r"\bsaccad\w*", r"\bgaze\b", r"\beye-?track\w*", r"\boculomotor\b"]),
    ("arm posture and workspace", [r"\bposture\w*", r"\barm configuration\w*", r"\bworkspace\w*"]),
    ("bimanual movements", [r"\bbimanual\b", r"\bboth (hands|arms)\b", r"\btwo hands\b"]),
    ("handedness", [r"\bhanded\w*", r"\bdominant (hand|arm)\b", r"\bnon-?dominant\b"]),
    ("hemispheric differences", [r"\bhemispher\w*", r"\blaterali[sz]\w*"]),
    # brain
    ("cerebellum", [r"\bcerebell\w*"]),
    ("basal ganglia and dopamine", [r"\bbasal ganglia\b", r"\bstriat\w*", r"\bdopamin\w*"]),
    ("motor cortex", [r"\bmotor cortex\b", r"\bM1\b", r"\bprimary motor\b"]),
    ("prefrontal cortex", [r"\bprefrontal\b", r"\bDLPFC\b"]),
    ("parietal cortex", [r"\bparietal\b"]),
    ("brain stimulation", [r"\btDCS\b", r"\btranscranial\b", r"\bTMS\b", r"\bbrain stimulation\b"]),
    ("EEG", [r"\bEEG\b", r"\belectroenceph\w*", r"\bevent-?related potentials?\b", r"\bERPs?\b", r"\breadiness potentials?\b", r"\boscillat\w*"]),
    ("neuroimaging", [r"\bfMRI\b", r"\bfunctional magnetic\b", r"\bneuroimaging\b", r"\bBOLD\b", r"\bMRI\b"]),
    ("neural mechanisms", [r"\bneural (correlates?|activity|basis|mechanisms?|substrates?|circuits?)\b"]),
    # people
    ("aging", [r"\baging\b", r"\bageing\b", r"\bolder adults?\b", r"\belderly\b", r"\bage-?related\b", r"\blifespan\b", r"\bolder (participants|individuals|people|subjects|age)\b"]),
    ("children and development", [r"\bchild\w*", r"\badolescen\w*", r"\bdevelopmental\b"]),
    ("Parkinson's disease", [r"\bparkinson\w*"]),
    ("cerebellar ataxia", [r"\bataxi\w*", r"\bcerebellar (degeneration|patients?|damage|lesions?|ataxia)\b", r"\bspinocerebellar\b"]),
    ("stroke", [r"\bstroke\w*", r"\bhemipare\w*"]),
    ("neurological patients", [r"\bpatients?\b", r"\bclinical\b"]),
    ("medication", [r"\bmedicat\w*", r"\blevodopa\b", r"\bL-?dopa\b"]),
    ("deafferentation", [r"\bdeafferent\w*"]),
    ("low vision", [r"\blow vision\b", r"\bvisual impairment\b", r"\bvisually impaired\b"]),
    ("sex differences", [r"\bsex differences?\b", r"\bgender\b"]),
    ("expertise", [r"\bexpert\w*", r"\bmusicians?\b", r"\bathletes?\b", r"\bvideo games?\b", r"\bgamers?\b"]),
    ("personality and traits", [r"\btraits?\b", r"\bpersonality\b"]),
    ("fatigue", [r"\bfatigu\w*"]),
    # setting
    ("online experiments", [r"\bonline (experiments?|study|studies|samples?|participants|platforms?|testing|settings?|data collection|version)\b",
                            r"\bweb-?based\b", r"\bweb browsers?\b", r"\bcrowd-?sourc\w*", r"\bprolific\b", r"\bmechanical turk\b", r"\bmturk\b",
                            r"\bat home\b", r"\bhome-?based\b", r"\bremote(ly)? (test\w*|collect\w*|conduct\w*)\b"]),
    ("virtual reality", [r"\bvirtual reality\b", r"\bVR\b", r"\bimmersive\b"]),
    ("robotic manipulandum", [r"\brobot\w*", r"\bmanipulandum\b", r"\bKINARM\b", r"\bexoskeleton\w*"]),
    ("force field", [r"\bforce-? ?fields?\b"]),
    ("large samples", [r"\blarge (samples?|cohorts?|datasets?|scale)\b", r"\blarge-?scale\b", r"\bmega-?analys\w*", r"\bthousands? of participants\b"]),
    ("replication", [r"\breplicat\w*", r"\breproducib\w*"]),
]
TFIDF_MIN = 0.005  # a study is linked to a mined term when tf-idf >= this (tf = mentions per word of its titles and abstracts)
MIN_STUDIES = 3    # a term is listed for a family only when it links at least this many studies with pooled lines in that family


def term_slug(label):
    return re.sub(r"[^a-z0-9]+", "-", label.replace("'", "").lower()).strip("-")


def _compile(pats):
    # case is ignored, except in patterns for acronyms (BOLD, M1, VR, EMG), which match only in capitals
    return [re.compile(p, 0 if re.search(r"\\b[A-Z][A-Z0-9]+\\b", p) else re.I) for p in pats]


VOCAB_RE = [(lab, term_slug(lab), _compile(pats)) for lab, pats in VOCAB]
assert len({s for _, s, _ in VOCAB_RE}) == len(VOCAB_RE), "two vocabulary terms share a slug"


def mine_terms(study_ids, texts, pooled_studies):
    """Neurosynth-style links between the studies of one family and the vocabulary.
    Returns (terms, doc facts). Within the family's studies: tf = mentions / words of titles and abstracts,
    idf = ln(N / studies that mention the term), weight = tf * idf; a study is linked when weight >= TFIDF_MIN."""
    docs = {}
    for sid in study_ids:
        t = texts.get(sid, {"title": "", "abstract": ""})
        text = (t["title"] + "\n\n" + t["abstract"]).strip()
        docs[sid] = {"text": text, "words": max(1, len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))), "abstract": bool(t["abstract"])}
    N = len(docs)
    terms = []
    for lab, slug, regs in VOCAB_RE:
        hits = {}
        for sid, dct in docs.items():
            found = [m.group(0) for r in regs for m in r.finditer(dct["text"])]
            if found:
                hits[sid] = found
        if not hits:
            continue
        idf = math.log(N / len(hits))
        links = {}
        for sid, found in hits.items():
            w = len(found) / docs[sid]["words"] * idf
            if w >= TFIDF_MIN:
                words = sorted({f.lower() for f in found}, key=lambda f: (-sum(1 for g in found if g.lower() == f), f))[:4]  # most frequent first, then a-z
                links[sid] = [round(w, 4), len(found), words]
        if sum(1 for sid in links if sid in pooled_studies) >= MIN_STUDIES:
            terms.append({"slug": slug, "label": lab, "df": len(hits), "studies": links})
    return terms, {sid: {"words": d["words"], "abstract": d["abstract"]} for sid, d in docs.items()}

# ---------------------------------------------------------------- build

def build(fetch=True):
    DS_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    all_ds = [d for d in datasets() if d["module"] in EXTRACTORS]
    tmp = tempfile.mkdtemp(prefix="om-explore-", dir=os.environ.get("EXPLORE_TMP") or None)
    n_new = 0
    try:
        for ds in all_ds:
            cp = DS_DIR / f"{ds['name']}.json"
            old = load_json(cp)
            if old and old.get("sha256") == ds["sha256"] and old.get("version") == EXTRACT_VERSION:
                continue
            if not fetch and not ds["local"].exists():
                print(f"  skipped (offline): {ds['name']}", flush=True)
                continue
            zp, downloaded = get_zip(ds, tmp)
            if zp is None:
                continue
            try:
                lines = EXTRACTORS[ds["module"]](zp, ds)
            finally:
                if downloaded:
                    zp.unlink(missing_ok=True)
            cp.write_text(json.dumps({"dataset": ds["name"], "study": ds["study"], "module": ds["module"], "sha256": ds["sha256"],
                                      "version": EXTRACT_VERSION, "lines": lines}, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
            n_new += 1
            fams = {}
            for l in lines:
                fams[l["family"]] = fams.get(l["family"], 0) + 1
            print(f"{ds['name']:44s} {len(lines):3d} lines  {fams}", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    names = {d["name"] for d in all_ds}
    for cp in DS_DIR.glob("*.json"):  # datasets no longer listed
        if cp.stem not in names:
            cp.unlink()

    lines_by_fam = {f["id"]: [] for f in FAMILIES}
    for cp in sorted(DS_DIR.glob("*.json")):
        for l in load_json(cp)["lines"]:
            if l["family"] in lines_by_fam:
                lines_by_fam[l["family"]].append(l)
    texts = studies_text()
    if fetch:
        fetch_missing_abstracts(texts, {l["study"] for ls in lines_by_fam.values() for l in ls})
        texts = studies_text()

    index = {"built": datetime.now().strftime("%Y-%m-%d"), "families": [],
             "rules": {"tfidf_min": TFIDF_MIN, "min_studies": MIN_STUDIES, "vocabulary": len(VOCAB), "min_fb": MIN_FB, "min_people": MIN_PEOPLE,
                       "late_n": LATE_N, "early": [EARLY[0] + 1, EARLY[1]], "ae_n": AE_N, "smooth": SMOOTH, "boot": BOOT, "half_se_min": HALF_SE_MIN, "pre": PRE, "max_x": MAX_X}}
    for fam in FAMILIES:
        L = sorted(lines_by_fam[fam["id"]], key=lambda l: (l["study"].lower(), l["ds"], l["exp"], l["id"]))
        sids = sorted({l["study"] for l in L})
        pooled = {l["study"] for l in L if l["ok"]}
        terms, docs = mine_terms(sids, texts, pooled)
        studies = {sid: {"ds": sorted({l["ds"] for l in L if l["study"] == sid}), **docs[sid]} for sid in sids}
        body = {"family": fam["id"], "lines": L, "studies": studies, "terms": terms}
        (SITE_DIR / f"{fam['id']}.json").write_text(json.dumps(body, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        entry = dict(fam, file=f"{fam['id']}.json", lines=len(L), pooled=sum(l["ok"] for l in L), studies=len(sids),
                     pooledStudies=len(pooled), terms=len(terms), bytes=(SITE_DIR / f"{fam['id']}.json").stat().st_size)
        entry.pop("module")
        index["families"].append(entry)
        print(f"{fam['id']}: {len(L)} lines ({entry['pooled']} pooled) from {len(sids)} studies; {len(terms)} mined terms; "
              f"{entry['bytes'] / 1e3:.0f} kB; {sum(1 for s in studies.values() if not s['abstract'])} studies without an abstract", flush=True)
    # the build date changes only when the data do, so a rerun with nothing new leaves every file as it was
    digest = hashlib.sha256(json.dumps({k: v for k, v in index.items() if k != "built"}, sort_keys=True).encode()
                            + b"".join((SITE_DIR / f"{f['id']}.json").read_bytes() for f in FAMILIES)).hexdigest()[:16]
    old_index = load_json(SITE_DIR / "index.json", {})
    index["digest"] = digest
    if old_index.get("digest") == digest:
        index["built"] = old_index.get("built", index["built"])
    (SITE_DIR / "index.json").write_text(json.dumps(index, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(f"{n_new} datasets read this run; {len(all_ds)} datasets in Explore families")


if __name__ == "__main__":
    build(fetch="--no-fetch" not in sys.argv)
