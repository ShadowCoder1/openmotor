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
   Families whose datasets are not trial tables (walking strides, speech trials, saccades, force-field trials, key-press
   blocks) still produce these fields: each has a prepare_<family>() that turns a dataset into units (one row per stride,
   trial or block, with y signed and baseline-corrected as the family defines it) and series_lines() makes the lines and
   the measures from the windows ("w") in its FAMILIES entry (see "series families" below). Their caches are versioned
   per module in MODULE_VERSION, so changing one family's code rereads only that module's datasets. Terms need nothing per family: tag terms come from the study tags in the page, and mined terms are
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
    # Families made by series_lines() (see "series families" below). Extra keys the page reads: xname and xunit (what x
    # counts), dec (decimals of y), gap (longest gap bridged in the pooled mean), maxX, unitName (the y unit in words),
    # methods (one or two plain sentences for "How the numbers are computed"); each measure's "w" is its window.
    {"id": "splitbelt", "label": "Split-belt walking", "task": "walking", "module": "walking",
     "about": "Each line is one group who walked on a treadmill whose two belts were split to different speeds. "
              "y is step length asymmetry minus the person's own baseline: it drops when the belts split and climbs back toward 0 as people adapt.",
     "y": "Step length asymmetry", "unit": "", "unitName": "step length asymmetry", "pct": False, "size": "Belt speed ratio", "sizeLabel": "Belt speed ratio", "sizeUnit": ":1",
     "x": "Strides from split onset", "xname": "Strides", "xunit": "stride", "dec": 3, "gap": 5, "maxX": 1000,
     "window": [-50, 600], "sizeBins": [[0, 2.5], [2.5, 10]],
     "series": {"pre": 50, "max_x": 1000, "min_block": 100, "round": 3, "mround": 4, "unit_word": "stride", "need": "late"},
     "cmp": {"what": "step length asymmetry", "value": "Step length asymmetry column, (fast - slow) / (fast + slow)", "guess": ["step_length_asymmetry", "sla", "step_asymmetry"],
             "base": True, "baseN": 50, "scale": 1, "note": "y is your asymmetry minus each person's mean over their last 50 baseline strides."},
     "measures": [
         {"key": "init", "label": "Initial perturbation", "unit": "y", "w": ["x", 0, 5, 3], "text": "mean of strides 1 to 5 of the first split block"},
         {"key": "early", "label": "Early adaptation", "unit": "y", "w": ["x", 5, 30, 13], "text": "mean of strides 6 to 30 of the split block"},
         {"key": "late", "label": "Late adaptation", "unit": "y", "w": ["last", 30, 15], "text": "mean of the last 30 strides of the split block"},
         {"key": "ae", "label": "Aftereffect", "unit": "y", "w": ["post", 5, 3], "text": "mean of the first 5 tied-belt treadmill strides right after the split block (a catch counts)"}],
     "methods": "Split-belt walking: step length asymmetry, (fast - slow) / (fast + slow) step length, per stride, minus each person's mean over 50 tied treadmill strides on the same slope before the split, leaving out the last 5. Splits that ramp up, are walked on a slope, or show step-length targets on screen are listed but not pooled."},
    {"id": "saccade", "label": "Saccade adaptation", "task": "eye movements", "module": "saccade",
     "about": "Each line is one group who made saccades to a target that stepped during the saccade, by the same amount and in the same direction on every trial. "
              "y is the change in saccade gain from baseline, signed so that adapting toward the step is positive (a gain decrease for a backward step).",
     "y": "Gain change", "unit": "%", "unitName": "% of target distance", "pct": True, "size": "Step", "sizeUnit": "%",
     "x": "Trials from step onset", "xname": "Trials", "xunit": "trial", "dec": 1, "gap": 5, "maxX": 600,
     "window": [-50, 300], "sizeBins": [[0, 20], [20, 35], [35, 100]],
     "series": {"pre": 50, "max_x": 600, "min_block": 30, "round": 1, "mround": 2, "unit_word": "trial", "need": "late"},
     "cmp": {"what": "saccade gain", "value": "Saccade gain column (amplitude / target distance)", "guess": ["saccade_gain", "gain"],
             "base": True, "baseN": 0, "scale": 100, "ask": "dir", "note": "y is the gain change from each person's baseline mean, times 100, positive toward the step."},
     "measures": [
         {"key": "early", "label": "Early adaptation", "unit": "y", "w": ["x", 1, 11, 5], "text": "mean of trials 2 to 11 of the first step block"},
         {"key": "late", "label": "Late adaptation", "unit": "y", "w": ["last", 20, 10], "text": "mean of the last 20 trials of the step block"},
         {"key": "ae", "label": "Aftereffect", "unit": "y", "w": ["post", 10, 5], "text": "mean of the first 10 trials right after the block, when the target no longer stepped"}],
     "methods": "Saccade adaptation: gain is the primary saccade amplitude along the target direction divided by the target distance; y is its change from the person's baseline mean, in % of the target distance, positive toward the step. % of the step divides that by the step size."},
    {"id": "forcefield", "label": "Force field reaching", "task": "reaching", "module": "force_field_reaching",
     "about": "Each line is one group who reached while a robot pushed the hand sideways in proportion to its speed (a velocity-dependent curl field). "
              "y is force compensation on channel trials, the share of the field's force the person pushed against the channel wall; 100% means full compensation.",
     "y": "Force compensation", "unit": "%", "unitName": "% of the field", "pct": False, "size": "Field", "sizeUnit": "N s/m",
     "x": "Trials from field onset", "xname": "Trials", "xunit": "trial", "dec": 1, "gap": 12, "maxX": 1000,
     "window": [-50, 400], "sizeBins": [[0, 12], [12, 100]],
     "series": {"pre": 50, "max_x": 1000, "min_block": 40, "round": 1, "mround": 2, "unit_word": "trial", "need": "late"},
     "cmp": {"what": "force compensation", "value": "Force compensation column (%, empty on non-channel trials)", "guess": ["force_compensation_pct", "force_compensation", "fc", "adaptation_index"],
             "base": False, "baseN": 0, "scale": 1, "note": "Values are used as they are, in % of the field."},
     "measures": [
         {"key": "early", "label": "Early adaptation", "unit": "y", "w": ["x", 0, 40, 2], "text": "mean of the channel trials among the first 40 trials of the first field block"},
         {"key": "late", "label": "Late adaptation", "unit": "y", "w": ["full_last", 5, 3], "text": "mean of the last 5 channel trials of the field block"},
         {"key": "ae", "label": "Aftereffect", "unit": "y", "w": ["post", 5, 3], "text": "mean of the first 5 channel trials right after the field block, when channel trials follow it"}],
     "methods": "Force field: force compensation on channel trials, the force pushed into the channel wall as a % of the force the field would have applied (no baseline subtraction). Field trials have no value, so gaps of up to 12 trials are bridged in the pooled mean. Ramped fields, opposing fields switched by context, and studies with only kinematic errors are listed but not pooled."},
    {"id": "sequence", "label": "Sequence learning (SRT)", "task": "sequence learning", "module": "keypress", "noSize": True,
     "blockWord": "run of sequence blocks", "lineWord": "practised the same sequence on the same schedule", "onsetWord": "block of the sequence",
     "about": "Each line is one group who pressed keys in answer to cues that followed one repeating sequence, block after block. "
              "y is how much faster they answered than in their first sequence block; a random or new-sequence block right after training shows how much of the speed-up belonged to the sequence.",
     "y": "Speed-up", "unit": "ms", "unitName": "ms faster than the first sequence block", "pct": False, "size": "Sequence", "sizeUnit": "",
     "x": "Blocks from the first sequence block", "xname": "Blocks", "xunit": "block", "dec": 0, "gap": 1, "maxX": 50,
     "window": [-4, 25], "sizeBins": [],
     "series": {"pre": 4, "max_x": 50, "min_block": 4, "round": 1, "mround": 1, "unit_word": "block", "need": "late"},
     "measures": [
         {"key": "early", "label": "Early speed-up", "unit": "y", "w": ["x", 1, 3, 1], "text": "mean speed-up in sequence blocks 2 and 3"},
         {"key": "late", "label": "Overall speed-up", "unit": "y", "w": ["last", 1, 1], "text": "speed-up in the last sequence block (general and sequence-specific learning together)"},
         {"key": "seq", "label": "Sequence-specific learning", "unit": "y", "w": ["minus", ["last", 1, 1], ["post", 1, 1]], "text": "RT in the random or new-sequence block right after training minus RT in the last sequence block"}],
     "methods": "Sequence learning: per block, the mean RT of correct key presses the authors kept (a 3-SD rule per person when they marked none, warm-up presses left out), and y is the first sequence block's mean minus this block's, in ms. Alternating (ASRT) and probabilistic sequences, and designs with no random block after training, are listed but not pooled.",
     "cmp": {"what": "mean reaction time", "value": "Mean reaction time column (ms), one row per block", "guess": ["rt_ms", "rt", "mean_rt", "reaction_time"],
             "base": False, "baseN": 0, "ref": "first", "scale": -1, "note": "One row per block: y is the first sequence block's RT minus each block's RT."}},
    {"id": "speech", "label": "Speech formant adaptation", "task": "speech", "module": "speech",
     "about": "Each line is one group who heard their own voice through headphones with the vowel formants shifted on every trial of a block. "
              "y is how much they changed their produced formants against the shift, in % of the full shift; 100% means the shift is fully cancelled.",
     "y": "Compensation", "unit": "%", "unitName": "% of the shift", "pct": False, "size": "Shift", "sizeUnit": "cents",
     "x": "Trials from shift onset", "xname": "Trials", "xunit": "trial", "dec": 1, "gap": 5, "maxX": 600,
     "window": [-30, 300], "sizeBins": [[0, 200], [200, 300], [300, 3000]],
     "series": {"pre": 30, "max_x": 600, "min_block": 30, "round": 1, "mround": 2, "unit_word": "trial", "need": "late"},
     "cmp": {"what": "produced formant", "value": "Produced formant column (the shifted formant, in the unit of the shift)", "guess": ["f1_mel", "f1_hz", "f1", "f1_cents"],
             "base": True, "baseN": 0, "scale": -100, "ask": "shift", "note": "y is minus the change from each person's baseline mean, in % of the shift (one formant, one baseline for all words)."},
     "measures": [
         {"key": "early", "label": "Early adaptation", "unit": "y", "w": ["full_first", 10, 5], "text": "mean of the first 10 trials at the full shift"},
         {"key": "late", "label": "Late adaptation", "unit": "y", "w": ["full_last", 30, 10], "text": "mean of the last 30 trials at the full shift"},
         {"key": "ae", "label": "Aftereffect", "unit": "y", "w": ["post", 10, 5], "text": "mean of the first 10 trials with normal feedback right after the shift ends"}],
     "methods": "Speech: the produced formant change from each person's baseline for the same word, projected on the direction of the shift and flipped, in % of the full shift. Ramp trials are drawn but early and late use only trials at the full shift."},
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


# ---------------------------------------------------------------- series families (walking, saccades, force fields, speech)
# These datasets are not reach90 trial tables, so each family has a small "prepare" function that turns one dataset into
# units (strides, trials, utterances), one row each, with the family's y already signed (learning positive) and, where the
# family says so, baseline-corrected and normalized. series_lines() then makes the lines the same way for every family.
# A unit table has the columns:
#   exp, el      experiment id and label;  glabel: group label (one line per exp x glabel x block size)
#   subject      one sequence;  person: one participant;  order: position of the unit in the subject's sequence
#   y            the family's y on this unit, NaN when the unit has none (for example a field trial in the force-field family)
#   stage        "base" (baseline units that may be drawn before x = 0), "pert" (the perturbation), "probe" (units inside
#                the perturbation block that do not end it and carry no y: catch, no-feedback or test trials), "post" (units
#                right after the block that can give the aftereffect), anything else ("other")
#   size         size of the perturbation on "pert" units, > 0, in the family's size unit
#   why          optional: the reason the line is not pooled (the same for the whole group), else empty
#   prior        optional: True when the person already had a perturbation earlier in the dataset
#   age, population, setting, device   optional participant columns, as in the standardized files

MODULE_VERSION = {"walking": 1, "saccade": 1, "force_field_reaching": 1, "speech": 1, "keypress": 1}  # per-module cache version (reach90 uses EXTRACT_VERSION)


def module_version(module):
    return MODULE_VERSION.get(module, EXTRACT_VERSION)


def read_main(zip_path, ds, cols, numeric=(), need_any=()):
    """The dataset's main CSV (<id>/<id>.csv inside its ZIP), only the columns in `cols` that it has, as strings except
    `numeric`, which are converted to floats. Read in chunks so that large files stay small in memory; with `need_any`,
    only rows where at least one of those columns has a value are kept (a 40-million-row typing file then keeps none)."""
    import zipfile
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    main = next((n for n in names if n.endswith(f"{ds['name']}/{ds['name']}.csv") or n == f"{ds['name']}.csv"), None)
    if main is None:
        main = max((n for n in names if not n.lower().endswith("columns.csv")), key=lambda n: z.getinfo(n).file_size)
    head = pd.read_csv(z.open(main), nrows=0).columns.tolist()
    use = [c for c in cols if c in head]
    need = [c for c in need_any if c in use]
    parts = []
    for ch in pd.read_csv(z.open(main), usecols=use, dtype=str, keep_default_na=False, na_values=[""], chunksize=200_000):
        if need_any:
            ch = ch[ch[need].notna().any(axis=1)] if need else ch.iloc[:0]
        parts.append(ch)
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=use)
    for c in numeric:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in cols:
        if c not in d.columns:
            d[c] = np.nan
    return d


def _truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _window(y, x, size, full, post, spec, block_len):
    """mean of one subject's y over a measure window (see FAMILIES 'w'); NaN when the window has too few values.
    y, x: the subject's block units (y NaN where none); post: the y of its post units, in order."""
    kind = spec[0]
    if kind == "minus":                 # one window minus another, for the same subject
        return _window(y, x, size, full, post, spec[1], block_len) - _window(y, x, size, full, post, spec[2], block_len)
    if kind == "x":                     # block positions a <= x < b
        a, b = spec[1], spec[2]
        v = y[(x >= a) & (x < b)]
        need = spec[3] if len(spec) > 3 else max(1, (b - a) // 2)
    elif kind == "last":                # the last n positions of the block
        v = y[x >= block_len - spec[1]]
        need = spec[2] if len(spec) > 2 else max(1, spec[1] // 2)
    elif kind in ("full_first", "full_last"):  # the first or last n units with a y at the block's full size
        if kind == "full_first" and len(full) and not full[0]:
            return np.nan                   # a ramp came first: "early" at full size would not be early
        v = y[full & np.isfinite(y)]
        v = v[:spec[1]] if kind == "full_first" else v[-spec[1]:]
        need = spec[2] if len(spec) > 2 else max(1, spec[1] // 2)
    elif kind == "post":                # the first n post units with a y
        v = post[np.isfinite(post)][:spec[1]]
        need = spec[2] if len(spec) > 2 else max(1, spec[1] // 2)
    else:
        raise ValueError(kind)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) >= need else np.nan


def _share(v):
    """prior exposure of one sequence: True/False, or a share (the expected share when the order of sessions is unknown)"""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return 0.0
    if isinstance(v, str):
        return 1.0 if v.strip().lower() in ("true", "1", "yes") else 0.0
    return float(v)


def _pop_code(t):
    """the reach90 population codes that the page's tag rules test, from the free text of the other modules"""
    if t is None:
        return None
    t = str(t).lower()
    for code, words in (("parkinsons", ("parkinson",)), ("cerebellar_degeneration", ("cerebell", "ataxia")), ("low_vision", ("low vision",)),
                        ("deafferented", ("deafferent",)), ("stroke", ("stroke", "hemipare")), ("healthy", ("healthy", "control", "typical"))):
        if any(w in t for w in words):
            return code
    return t


def _device_code(t):
    """a device code the page's tag rules know, or None (then the rules treat the group as not coded)"""
    if t is None:
        return None
    t = str(t).lower()
    for code, words in (("kinarm", ("kinarm",)), ("robot", ("robot", "manipulandum", "vbot")), ("tablet", ("digitizing tablet", "digitising tablet")),
                        ("trackpad", ("trackpad",)), ("mouse", ("mouse",)), ("touchscreen", ("touchscreen", "touch screen"))):
        if any(w in t for w in words):
            return code
    return None


def series_lines(ds, fam, u):
    """Lines of one family from a unit table (see above). One line per experiment x group x block size."""
    F = next(f for f in FAMILIES if f["id"] == fam)
    S = F["series"]
    pre, max_x, min_blk, nd = S["pre"], S["max_x"], S["min_block"], S.get("round", 1)
    if u is None or not len(u):
        return []
    u = u.copy()
    for c in ("why", "prior", "age", "population", "setting", "device"):
        if c not in u.columns:
            u[c] = np.nan
    u["y"] = pd.to_numeric(u["y"], errors="coerce").astype(float)
    u["size"] = pd.to_numeric(u["size"], errors="coerce").astype(float)
    u = u.sort_values(["subject", "order"], kind="stable")
    per = {}  # subject -> its first perturbation block
    for s, g in u.groupby("subject", sort=False):
        st = g["stage"].to_numpy(dtype=object)
        idx = np.nonzero(st == "pert")[0]
        if not len(idx):
            continue
        on = idx[0]
        end = on
        while end + 1 < len(st) and st[end + 1] in ("pert", "probe"):
            end += 1
        blk = g.iloc[on:end + 1]
        sz = blk["size"].to_numpy(dtype=float)
        size = float(np.nanmax(sz)) if np.isfinite(sz).any() else np.nan
        if np.isfinite(size) and size <= 0:
            continue
        if not np.isfinite(size) and not (blk["why"].notna().any() if "why" in blk else False):
            continue                        # an unknown size is kept only for a line that is listed, not pooled
        size = round(size, 2) if np.isfinite(size) else -1.0
        k = end + 1
        while k < len(st) and st[k] not in ("post", "pert", "base") and k <= end + 3:  # up to 3 units between the block and its post units
            k += 1
        p = k
        while p < len(st) and st[p] == "post":
            p += 1
        base_y = g.iloc[:on]
        base_y = base_y[base_y["stage"] == "base"]["y"].to_numpy(dtype=float)[-pre:] if pre else np.array([])
        by = blk["y"].to_numpy(dtype=float)
        by = np.where(blk["stage"].to_numpy(dtype=object) == "pert", by, np.nan)
        per[s] = {"exp": g["exp"].iloc[0], "el": g["el"].iloc[0], "glabel": g["glabel"].iloc[0], "person": str(g["person"].iloc[0]),
                  "size": size, "y": by, "x": np.arange(len(blk)), "full": np.isclose(blk["size"].to_numpy(dtype=float), size, atol=0.006),
                  "post": g.iloc[k:p]["y"].to_numpy(dtype=float), "base": base_y, "row": g.iloc[0], "prior": g["prior"].iloc[0]}
    lines = []
    groups = {}
    for s, r in per.items():
        groups.setdefault((r["exp"], r["glabel"]), []).append(s)
    for (exp, glabel), subs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        sizes = pd.Series([per[s]["size"] for s in subs]).value_counts()
        all_people = {per[s]["person"] for s in subs}
        for k_size, (P, _) in enumerate(sorted(sizes.items(), key=lambda kv: -kv[1])):
            cons = [s for s in subs if per[s]["size"] == P]
            people = {per[s]["person"] for s in cons}
            # a group with more than one block size gives one line per size; the main size (most people) is first
            share_ok = len(sizes) == 1 or len(cons) >= 0.9 * len(subs)
            lens = np.array([len(per[s]["y"]) for s in cons])
            L = int(np.median(lens))
            end = min(L, max_x + 1)
            M = np.full((len(cons), end), np.nan)
            B = np.full((len(cons), pre), np.nan) if pre else None
            for i, s in enumerate(cons):
                yy = per[s]["y"][:end]
                M[i, :len(yy)] = yy
                if pre and len(per[s]["base"]):
                    bb = per[s]["base"]
                    B[i, pre - len(bb):] = bb
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                cm = np.nanmean(M, axis=0) if len(cons) else np.array([])
                cb = np.nanmean(B, axis=0) if pre else np.array([])
            fin = np.nonzero(np.isfinite(cb))[0] if pre else []
            first_b = int(fin[0]) if len(fin) else pre
            npre = pre - first_b
            # keep the baseline stretch that has data, ending at x = -1
            ycurve = [None if not np.isfinite(v) else round(float(v), nd) for v in (list(cb[first_b:]) if pre else []) + list(cm)]
            p_of = {s: per[s]["person"] for s in cons}
            m = {}
            for ms in F["measures"]:
                vals = {}
                for s in cons:
                    r = per[s]
                    v = _window(r["y"], r["x"], r["size"], r["full"], r["post"], ms["w"], len(r["y"]))
                    if np.isfinite(v):
                        vals.setdefault(p_of[s], []).append(v)
                v = np.array([float(np.mean(v)) for v in vals.values()])
                if len(v) >= MIN_PEOPLE:  # as _stat, rounded to the family's digits
                    mr = S.get("mround", 2)
                    m[ms["key"]] = [round(float(v.mean()), mr), round(float(v.std(ddof=1) / math.sqrt(len(v))), mr + 1), int(len(v))]
            rows = pd.DataFrame([per[s]["row"] for s in cons])
            prs = rows.drop_duplicates("person")
            age = pd.to_numeric(prs["age"], errors="coerce").dropna()
            a = {"pop": _pop_code(_mode(prs["population"])), "age": round(float(age.median()), 1) if len(age) >= max(1, 0.5 * len(prs)) else None,
                 "setting": _mode(prs["setting"]), "device": _device_code(_mode(prs["device"])),
                 "prior": round(float(np.mean([_share(per[s]["prior"]) for s in cons])), 2)}
            whys = pd.Series([per[s]["row"]["why"] for s in cons]).dropna()
            why = None
            unit = S["unit_word"]
            if P < 0:
                why = str(whys.iloc[0]) if len(whys) else "the perturbation size is not in the data"
            elif not share_ok:
                why = "participants in this group had different perturbation sizes"
            elif len(whys):
                why = str(whys.iloc[0])
            elif L < min_blk:
                why = f"the first perturbation block had {L} {unit}s (fewer than {min_blk})"
            elif len(people) < MIN_PEOPLE:
                why = f"fewer than {MIN_PEOPLE} participants"
            elif S["need"] not in m:
                why = f"fewer than {MIN_PEOPLE} participants with enough usable {unit}s for the {next(x['label'] for x in F['measures'] if x['key'] == S['need']).lower()}"
            label = str(glabel) if glabel == glabel and str(glabel) else "All participants"
            if len(sizes) > 1 and P >= 0:
                label += f", {F['size'].lower()} {P:g}{F['sizeUnit'] if F['sizeUnit'].startswith(':') else ' ' + F['sizeUnit']}"
            gi = sorted({str(per[t]["glabel"]) for t in per if per[t]["exp"] == exp}).index(str(glabel))
            line = {"id": f"{ds['name']}|{exp}|{gi}" + (f".{k_size}" if k_size else ""), "family": fam, "ds": ds["name"], "study": ds["study"], "exp": str(exp),
                    "el": str(per[cons[0]]["el"]), "label": label, "n": len(people), "size": None if P < 0 else P, "x0": -npre, "y": ycurve,
                    "blk": L, "nfb": int(np.median([np.sum(np.isfinite(per[s]["y"])) for s in cons])), "m": m, "a": a, "ok": 0 if why else 1}
            if len(all_people) != len(people):
                line["nall"] = len(all_people)
            if len(cons) != len(people):
                line["seq"] = len(cons)
            if why:
                line["why"] = why
            lines.append(line)
    return lines



# ---------------------------------------------------------------- split-belt walking (module walking)

# prepare() for the "splitbelt" family (module walking): one unit per stride.
#
# y = step_length_asymmetry (SLA, (fast - slow) / (fast + slow) step length; the module already names legs by their belt,
# so no per-person flip; rebuilt from step_length_fast_m/slow_m when only those are shared) minus the person's baseline
# SLA. Baseline = mean of the last 50 usable (not excluded) tied-belt treadmill strides before the first real split, on the
# same slope as the split, from phase "baseline" when there is any, never with step-length feedback or targets on screen,
# and from the tied segment (same block and condition) just before the split when that segment has at least 20 usable
# strides. A person with fewer than 10 such strides gets no y; a group where nobody has any keeps raw SLA and is not pooled.
# size = fast/slow belt ratio (pert_value, else the belt speeds).
#
# Stages: "pert" = the first split run of at least 50 strides that is not a practice exposure (shorter or practice splits
# before it are "other"); it ends where the authors start a block they call re-adaptation. A tied treadmill run of at most
# 30 strides between two split runs is a catch inside the block ("probe") unless the authors call it an aftereffect or
# washout, in which case it ends the block and is "post". "post" = the tied treadmill strides (no feedback) right after the
# block, so overground walking right after the block gives no aftereffect. Strides with no SLA at the very end of a block
# are dropped (padding). A deposit that shares SLA per step (Brinkerhoff 2023) is turned into strides by averaging
# consecutive pairs of steps. The baseline is the 50 strides before the last 5 tied strides (as labTools does).
SB_COLS = ["experiment", "group", "condition", "subject", "person", "subject_excluded_by_authors", "age", "population", "setting",
        "device", "session", "block", "trial", "phase", "unit", "pert_type", "pert_value", "feedback", "excluded_by_authors",
        "walking_context", "step_length_asymmetry", "step_length_fast_m", "step_length_slow_m", "belt_speed_fast_mps",
        "belt_speed_slow_mps", "target_step_length_asymmetry", "target_step_length", "target_step_length_fast_m",
        "target_step_length_slow_m", "surface_incline_deg"]
SB_NUMERIC = ["age", "session", "surface_incline_deg", "pert_value", "step_length_asymmetry", "step_length_fast_m", "step_length_slow_m",
           "belt_speed_fast_mps", "belt_speed_slow_mps", "target_step_length_asymmetry", "target_step_length",
           "target_step_length_fast_m", "target_step_length_slow_m"]

SB_MIN_BLOCK = 50   # split runs shorter than this before the real block are practice exposures ("other")
SB_MAX_CATCH = 30   # a tied run this short between two split runs is a catch inside the block
SB_BASE_N, SB_BASE_PREF, SB_BASE_MIN = 50, 20, 10  # baseline strides used; needed in the preferred segment; needed at all
SB_BASE_SKIP = 5    # tied strides left out just before the split (as labTools does: transition strides can carry artefacts)
# Documented special cases (dataset, group) -> extra fields. Sombric2021's YoungNaive2017 group is the same 11 young
# adults and recordings as Sombric2017 visit 1 (identical ages, baseline SLA series correlate r = 0.86-0.98), and 6 of
# the 8 OldNaive sessions match Sombric2017's older adults the same way: pooling them would count people twice.
# OldExpert had walked on a split-belt treadmill before (README); their earlier sessions are not linked in the deposit.
SB_SPECIAL = {
    ("Sombric2021_FrontAgingNeurosci", "YoungNaive2017"): {"why": "same participants and recordings as Sombric et al. 2017, pooled from that dataset"},
    ("Sombric2021_FrontAgingNeurosci", "OldNaive"): {"why": "6 of these 8 sessions are recordings already pooled from Sombric et al. 2017"},
    ("Sombric2021_FrontAgingNeurosci", "OldExpert"): {"prior": True},
}


def _sb_runs(mask):
    """(start, end) index pairs (end exclusive) of the True runs of a boolean array"""
    m = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(m == 1)[0], np.nonzero(m == -1)[0]))


def _sb_sequence(g):
    """stages, baseline and flags of one subject's strides (in walking order); None when it has no real split block"""
    n = len(g)
    split = g["pert_type"].eq("split_belt").to_numpy()
    tm = ~g["walking_context"].eq("overground").to_numpy()   # missing context = treadmill (stride tables of split-belt studies)
    phase = g["phase"].fillna("").to_numpy(dtype=object)
    fb = (g["feedback"].eq("altered") | g["condition"].fillna("").str.lower().str.contains("feedback on")
          | g[["target_step_length_asymmetry", "target_step_length", "target_step_length_fast_m", "target_step_length_slow_m"]].notna().any(axis=1)).to_numpy()
    ratio = g["pert_value"].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        br = g["belt_speed_fast_mps"].to_numpy(dtype=float) / g["belt_speed_slow_mps"].to_numpy(dtype=float)
    ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, br)
    raw = g["raw"].to_numpy(dtype=float)
    runs = _sb_runs(split)
    real = next((r for r in runs if r[1] - r[0] >= SB_MIN_BLOCK and pd.Series(phase[r[0]:r[1]]).mode().iloc[0] != "practice"), None)
    if real is None:
        return None
    on, k = real
    # a run that goes straight into a block the authors call re-adaptation (its catch is missing) ends there
    lab = g["block"].fillna("").str.lower().str.replace(r"[^a-z]", "", regex=True).to_numpy(dtype=object)
    cut = next((c for c in range(on + 1, k) if lab[c] != lab[c - 1] and "readapt" in lab[c]), None)
    k = cut or k
    stage = np.full(n, "other", dtype=object)
    stage[on:k] = "pert"
    for a, b in runs:                                          # catches inside the block
        if a < k:
            continue
        if cut:
            break
        gap = slice(k, a)
        if a - k <= SB_MAX_CATCH and tm[gap].all() and not np.isin(phase[gap], ["aftereffect", "washout"]).any():
            stage[gap], stage[a:b], k = "probe", "pert", b
        else:
            break
    p = k
    while p < n and not split[p] and tm[p] and not fb[p]:     # tied treadmill strides right after the block
        stage[p] = "post"
        p += 1
    # baseline: usable tied treadmill strides before the split, preferring phase "baseline" and the segment just before
    cand = (~split) & tm & (~fb) & np.isfinite(raw) & (np.arange(n) < on)
    inc = g["surface_incline_deg"].to_numpy(dtype=float)       # tied walking on another slope is not this split's baseline
    if np.isfinite(inc[on]):
        cand &= ~(np.isfinite(inc) & ~np.isclose(inc, inc[on]))
    if (cand & (phase == "baseline")).any():
        cand &= phase == "baseline"
    base, nbase = np.nan, int(cand.sum())
    if nbase:
        seg = (g["block"].fillna("") + "|" + g["condition"].fillna("")).to_numpy(dtype=object)
        last = np.nonzero(cand)[0][-1]
        first = last
        while first > 0 and seg[first - 1] == seg[last]:
            first -= 1
        pref = cand & (np.arange(n) >= first) & (np.arange(n) <= last)
        use_idx = np.nonzero(pref if pref.sum() >= SB_BASE_PREF else cand)[0]
        use_idx = use_idx[:len(use_idx) - SB_BASE_SKIP][-SB_BASE_N:]
        if len(use_idx) >= SB_BASE_MIN:
            base = float(np.mean(raw[use_idx]))
            stage[use_idx] = "base"
    # why the block cannot be pooled (feedback first: it is the stronger reason)
    why = None
    first_pert = np.nonzero(stage == "pert")[0][:30]
    if fb[first_pert].any():
        nf = k - on if fb[on:k].all() else int(np.argmin(fb[on:k])) if fb[on] else int(fb[on:k].sum())
        why = ("visual feedback on step lengths or a step-length target was on screen " +
               ("during the split" if nf == k - on else f"for the first {nf} strides of the split"))
    r0 = ratio[on:min(k, on + 100)]
    ramp = np.isfinite(r0).mean() < 0.5 or np.nanmin(r0) < 0.95 * np.nanmax(r0)
    if why is None and ramp:
        why = "the belts were not split abruptly to a constant ratio (gradual ramp)"
    slope = float(np.nanmedian(inc[on:k])) if np.isfinite(inc[on:k]).any() else 0.0
    if why is None and abs(slope) > 0.5:     # incline and decline change the adaptation a lot; the family is level walking
        why = f"the split was walked on {'an incline' if slope > 0 else 'a decline'} of {abs(slope):g} degrees, not on level ground"
    size = np.where(stage == "pert", ratio, np.nan)
    if ramp:                                                    # list the line at the ratio the split ramped to
        size = np.where(stage == "pert", np.nanmax(np.where(split, ratio, np.nan)) if np.isfinite(ratio[split]).any() else np.nan, np.nan)
    cond = pd.Series(g["condition"].to_numpy()[stage == "pert"]).dropna().unique()   # kept only when one condition spans the block
    return {"stage": stage, "base": base, "nbase": nbase, "why": why, "size": size,
            "cond": str(cond[0]) if len(cond) == 1 else ""}


def prepare_splitbelt(d, ds):
    d = d[~d["subject_excluded_by_authors"].astype(str).eq("True")].copy()
    sla = d["step_length_asymmetry"].astype(float)
    f, s = d["step_length_fast_m"].astype(float), d["step_length_slow_m"].astype(float)
    d["raw"] = sla.where(sla.notna(), (f - s) / (f + s))      # rebuilt only from real per-leg step lengths
    if not (d["pert_type"].eq("split_belt").any() and d["raw"].notna().any()):
        return pd.DataFrame()
    d.loc[d["excluded_by_authors"].astype(str).eq("True"), "raw"] = np.nan
    # keep the unit that carries SLA (Brinkerhoff: SLA on "step" rows, work rates on "stride" rows)
    unit = d.loc[d["raw"].notna(), "unit"].mode()
    unit = unit.iloc[0] if len(unit) else None
    if unit is not None and d["unit"].nunique() > 1:
        d = d[d["unit"].eq(unit)]
    if unit == "step":                                          # pairs of consecutive steps -> strides
        d = d.assign(_k=d.groupby("subject", sort=False).cumcount() // 2)
        agg = {c: "first" for c in d.columns if c not in ("subject", "_k", "raw")}
        agg["raw"] = "mean"
        d = d.groupby(["subject", "_k"], sort=False).agg(agg).reset_index().drop(columns="_k")
    out, info = [], []
    for subj, g in d.groupby("subject", sort=False):           # rows are in walking order within a subject
        # strides without SLA at the end of a block (Wood 2024 pads some adaptation tables with empty rows) are dropped,
        # so that "the last 30 strides" are the last measured ones; blocks with no SLA at all are kept as they are
        blk = (g["block"].fillna("") != g["block"].fillna("").shift()).cumsum().to_numpy()
        fin = g["raw"].notna().to_numpy()
        last_fin = pd.Series(np.where(fin, np.arange(len(g)), -1)).groupby(blk).transform("max").to_numpy()
        g = g[(np.arange(len(g)) <= last_fin) | (last_fin < 0)].reset_index(drop=True)
        r = _sb_sequence(g)
        if r is None:
            continue
        info.append({"subject": subj, "person": str(g["person"].iloc[0]), "exp": g["experiment"].fillna("").iloc[0],
                     "group": "" if pd.isna(g["group"].iloc[0]) else str(g["group"].iloc[0]), "cond": r["cond"],
                     "session": g["session"].iloc[0], "nbase": r["nbase"], "base": r["base"], "why": r["why"]})
        out.append(pd.DataFrame({"subject": subj, "order": np.arange(len(g)), "raw": g["raw"].to_numpy(dtype=float),
                                 "stage": r["stage"], "size": r["size"], "age": g["age"], "population": g["population"],
                                 "setting": g["setting"], "device": g["device"]}))
    if not out:
        return pd.DataFrame()
    I = pd.DataFrame(info)
    # line label: the group, plus the condition of the split when it differs between sequences of the group
    I["glabel"] = I["group"]
    multi = (I.groupby(["exp", "group"])["cond"].transform("nunique") > 1) & I["cond"].ne("")
    I.loc[multi, "glabel"] = I.loc[multi, "group"] + ", " + I.loc[multi, "cond"]
    # prior: the person already had a split in an earlier sequence (by session, then file order)
    I["_o"] = np.arange(len(I))
    I = I.sort_values(["person", "session", "_o"], kind="stable")
    I["prior"] = I.groupby("person").cumcount() > 0
    I = I.sort_values("_o")
    mixed = I.groupby(["exp", "glabel"])["prior"].transform("nunique") > 1
    I["glabel"] = [g + (f", session {s:g}" if m and s == s else "") for g, s, m in zip(I["glabel"], I["session"], mixed)]
    # baseline: groups where nobody has one keep raw SLA and are not pooled; elsewhere a missing baseline drops the person
    nob = I.groupby(["exp", "glabel"])["nbase"].transform("max") == 0
    I.loc[nob & I["why"].isna(), "why"] = "no tied-belt baseline was shared, so y is raw step length asymmetry, not baseline-corrected"
    I["shift"] = np.where(nob, 0.0, I["base"])
    for (dsn, grp), extra in SB_SPECIAL.items():
        if dsn == ds["name"]:
            sel = I["group"].eq(grp)
            for key, val in extra.items():
                I.loc[sel, key] = val
    I["why"] = I.groupby(["exp", "glabel"])["why"].transform(lambda w: w.dropna().iloc[0] if w.notna().any() else None)
    u = pd.concat(out, ignore_index=True).merge(I[["subject", "person", "exp", "glabel", "prior", "why", "shift"]], on="subject")
    u["y"] = u["raw"] - u["shift"]
    u["el"] = u["exp"]
    return u[["exp", "el", "glabel", "subject", "person", "order", "y", "stage", "size", "why", "prior",
              "age", "population", "setting", "device"]]


# ---------------------------------------------------------------- saccade adaptation (module saccade)

# Saccade adaptation family: prepare(d, ds) -> unit table for series_lines (one unit = one trial, one primary saccade).
#
# gain  = primary saccade amplitude projected on the target direction / target amplitude. The amplitude vector is
#         saccade_amplitude_x/y, else end - start position, else the end position (saccade started at fixation).
#         saccade_gain is not used: some deposits divide by the stepped target's position (Schutz2014).
# y     = (gain - the sequence's mean baseline gain for that target) x 100 x sign of the step along the target axis,
#         so adapting toward the step is positive. Baseline = the no-step trials before the first step trial
#         (practice excluded), per target (amplitude, direction); trials to targets that never stepped, excluded or blink
#         trials and saccades made in localization trials have no y.
# size  = |step along the target axis| / target amplitude x 100 (% of the target distance).
# stage = "pert" for step trials of the first block (same step on every trial for that target), "probe" for no-step
#         trials inside it (tests, localizations), "base" before it, "post" for the no-step trials right after its last step
#         trial (up to the next perturbation of any kind), "other" for the rest.
# Only steps along the saccade are this family: steps with no along-axis component (cross-axis designs: Huang2017 Exp 2,
# Schutz2014 Exp 4-5, Schutz2015, Wolf2019, Wagner2021, Wagner2023) are not gain perturbations and give no units.
# Sequences: a subject that returns to baseline after a step block starts a new episode (one sequence each); a sequence
# whose targets stepped in opposite directions (forward for rightward, backward for leftward saccades) is split by target.
# A sequence whose step block repeats an earlier one value for value is a copy in the deposit and is dropped.
# Lines: group, the condition when it changes the block (not when it only changes what follows an identical block), and
# the step direction ("backward step" = gain-down). prior: 1 after an earlier perturbation in a known order; when a
# person's sequences have no known order, (k - 1) / k for each of k sequences (the expected share).
# "trial" is the dataset's trial number, kept for checks.
SC_COLS = ["experiment", "group", "condition", "subject", "person", "subject_excluded_by_authors", "age", "population",
        "setting", "device", "session", "trial", "phase", "pert_type", "pert_value", "pert_unit", "excluded_by_authors",
        "blink", "saccade_index_in_trial", "target_amplitude_deg", "target_direction_deg", "step_x_deg", "step_y_deg",
        "saccade_start_x_deg", "saccade_start_y_deg", "saccade_end_x_deg", "saccade_end_y_deg",
        "saccade_amplitude_deg", "saccade_amplitude_x_deg", "saccade_amplitude_y_deg", "localization_kind"]
SC_NUMERIC = ["age", "session", "trial", "pert_value", "saccade_index_in_trial", "target_amplitude_deg", "target_direction_deg",
           "step_x_deg", "step_y_deg", "saccade_start_x_deg", "saccade_start_y_deg", "saccade_end_x_deg", "saccade_end_y_deg",
           "saccade_amplitude_deg", "saccade_amplitude_x_deg", "saccade_amplitude_y_deg"]

# displacements along the saccade that make a line; only the intrasaccadic target step is pooled
SC_STEP_WHY = {"intrasaccadic_step": None,
            "task_relevant_offset": "no target step: the target stayed in place and a perceptual task was shown at an offset location",
            "postsaccadic_step": "the target moved after the saccade had ended, not during it",
            "postsaccadic_target_shift": "the target moved after the saccade had ended, not during it"}
SC_PROBE_WHY = "half or more of the trials in the step block were test trials without the step"
SC_RANDOM_WHY = "the step direction or size changed from trial to trial"


def _sc_s(x):
    return x.astype(str).str.strip()


def prepare_saccade(d, ds):
    d = d[_sc_s(d["subject_excluded_by_authors"]) != "True"].copy()
    d = d[~(d["saccade_index_in_trial"] > 1)]                       # primary saccade only, where several are listed
    if not len(d):
        return pd.DataFrame()
    d["_row"] = np.arange(len(d))
    d = d.sort_values(["subject", "trial", "_row"], kind="stable").reset_index(drop=True)
    pt = _sc_s(d["pert_type"]).where(d["pert_type"].notna(), "none")
    phase = _sc_s(d["phase"]).where(d["phase"].notna(), "")

    # ---- gain along the target axis
    th = np.radians(d["target_direction_deg"].to_numpy(float))
    ux, uy = np.cos(th), np.sin(th)
    ax, ay = d["saccade_amplitude_x_deg"].to_numpy(float), d["saccade_amplitude_y_deg"].to_numpy(float)
    ex, ey = d["saccade_end_x_deg"].to_numpy(float), d["saccade_end_y_deg"].to_numpy(float)
    sx0, sy0 = d["saccade_start_x_deg"].to_numpy(float), d["saccade_start_y_deg"].to_numpy(float)
    ax = np.where(np.isfinite(ax), ax, ex - np.where(np.isfinite(sx0), sx0, 0.0))
    ay = np.where(np.isfinite(ay), ay, ey - np.where(np.isfinite(sy0), sy0, 0.0))
    ay = np.where(np.isfinite(ax) & ~np.isfinite(ay), 0.0, ay)      # horizontal-only datasets
    proj = np.where(np.isfinite(ax), ax * ux + ay * uy, d["saccade_amplitude_deg"].to_numpy(float))
    amp = d["target_amplitude_deg"].to_numpy(float)
    g_raw = proj / amp
    # some deposits give the amplitude positive in the trial's direction for leftward saccades too: saccades never go
    # away from the target on average, so a target direction whose median gain is negative is flipped
    dkey = np.round(d["target_direction_deg"].to_numpy(float)) % 360
    for k in np.unique(dkey[np.isfinite(dkey)]):
        m = (dkey == k) & np.isfinite(g_raw)
        if m.sum() >= 10 and np.median(g_raw[m]) < 0:
            g_raw = np.where(dkey == k, -g_raw, g_raw)
    # excluded and blink trials have no gain; neither have saccades made in localization trials (a second task, often
    # with the target removed), so the baseline and the aftereffect come from the same kind of trial as the steps
    bad = (_sc_s(d["excluded_by_authors"]) == "True") | (_sc_s(d["blink"]) == "True") | d["localization_kind"].notna()
    gain = np.where(bad.to_numpy(), np.nan, g_raw)

    # ---- the step along the target axis
    stx, sty = d["step_x_deg"].to_numpy(float), d["step_y_deg"].to_numpy(float)
    none_xy = ~np.isfinite(stx) & ~np.isfinite(sty)
    stx, sty = np.nan_to_num(stx), np.nan_to_num(sty)
    pv_deg = np.where(_sc_s(d["pert_unit"]) == "deg", d["pert_value"].to_numpy(float), np.nan)
    along = np.where(none_xy, pv_deg, stx * ux + sty * uy)            # no x/y given: pert_value, positive outward
    cross = np.where(none_xy, 0.0, -stx * uy + sty * ux)
    is_pert = (pt != "none").to_numpy() & (phase != "test").to_numpy()   # "test": perception trials, not exposure
    step = is_pert & pt.isin(list(SC_STEP_WHY)).to_numpy() & (np.abs(along) > 1e-6) & (np.abs(along) >= 0.5 * np.hypot(along, cross))
    if not step.any():
        return pd.DataFrame()
    tkey = np.char.add(np.round(amp, 1).astype(str), np.char.add("@", dkey.astype(str)))

    # ---- episodes: a sequence that goes back to baseline/practice after a perturbation starts a new episode (Souto2016)
    prev = phase.groupby(d["subject"]).shift(1).fillna("")
    new_ep = phase.isin(["baseline", "practice"]) & ~prev.isin(["baseline", "practice", "test", ""])
    d["_ep"] = new_ep.astype(int).groupby(d["subject"]).cumsum()
    d["_pert"], d["_step"], d["_along"], d["_gain"], d["_graw"], d["_key"] = is_pert, step, along, gain, g_raw, tkey
    d["_pt"], d["_phase"] = pt, phase

    # ---- prior exposure: order known within a subject (episodes) and across subjects with different sessions
    expo = d[d["_pert"]].groupby(["person", "subject"]).agg(ep0=("_ep", "min"), ses=("session", "first")).reset_index()
    prior_of = {}
    for p, e in expo.groupby("person"):
        for _, r in e.iterrows():
            others = e[e["subject"] != r["subject"]]
            known_before = (np.isfinite(r["ses"]) & (others["ses"] < r["ses"])).any()
            unknown = others[~(np.isfinite(others["ses"]) & np.isfinite(r["ses"]) & (others["ses"] != r["ses"]))]
            prior_of[r["subject"]] = (r["ep0"], 1.0 if known_before else len(unknown) / (len(unknown) + 1))

    seqs, seen = [], set()
    for (subj, ep), g in d.groupby(["subject", "_ep"], sort=False):
        st = g["_step"].to_numpy()
        if not st.any():
            continue
        keys = g.loc[st, "_key"]
        sign_of = {k: np.sign(g.loc[st & (g["_key"] == k).to_numpy(), "_along"].iloc[0]) for k in keys.unique()}
        split = len(set(sign_of.values())) > 1                       # forward for one target, backward for another
        for k_sub in (list(sign_of) if split else [None]):
            h = g[g["_key"] == k_sub] if split else g
            s = _sc_seq_units(h, sign_of, subj, ep, prior_of)
            if s is None:
                continue
            fp = tuple(np.round(h.loc[h["_step"], "_graw"].dropna().to_numpy()[:40], 4))
            if len(fp) >= 20 and fp in seen:                          # the deposit repeats a block (Souto2016 exp2 folder)
                continue
            seen.add(fp)
            seqs.append(s)
    if not seqs:
        return pd.DataFrame()
    u = pd.concat(seqs, ignore_index=True)

    # ---- group labels: group, the condition when it changes what happens in the block, and the step direction
    info = u.groupby("subject").agg(exp=("exp", "first"), grp=("_grp", "first"), cond=("_cond", "first"),
                                    blk=("_blk", "first"), after=("_after", "first"), word=("_word", "first"))
    use_cond = {}
    for (e, gr), q in info.groupby(["exp", "grp"]):
        c = q.groupby("cond").agg(blk=("blk", "median"), after=("after", "median"))
        # a condition that only changes what comes after an identical block (Tyralla2023: no-error block length) is merged
        use_cond[(e, gr)] = len(c) > 1 and not (c["blk"].nunique() == 1 and c["after"].nunique() > 1)
    lab = {}
    for sub, r in info.iterrows():
        parts = [r["grp"]] if r["grp"] else []
        if use_cond[(r["exp"], r["grp"])] and r["cond"]:
            parts.append(r["cond"])
        if not any(w in " ".join(parts).lower() for w in ("inward", "outward", "backward", "forward")):
            parts.append(r["word"])
        lab[sub] = ", ".join(parts)
    u["glabel"] = u["subject"].map(lab)
    return u.drop(columns=["_grp", "_cond", "_blk", "_after", "_word"])


def _sc_seq_units(h, sign_of, subj, ep, prior_of):
    """units of one sequence (one subject episode, or one target of it): stages, y and size; None when it has no data."""
    st = h["_step"].to_numpy()
    al = h["_along"].to_numpy()
    key = h["_key"].to_numpy()
    pert = h["_pert"].to_numpy()
    n = len(h)
    i0 = int(np.nonzero(st)[0][0])
    ref, prv = {}, {}                                                # first and previous step value of each target
    changes = 0
    last = i0
    stop = n
    for i in range(i0, n):
        if st[i]:
            r = ref.setdefault(key[i], al[i])
            p = prv.get(key[i], al[i])
            changes += abs(al[i] - p) > max(0.05, 0.03 * abs(p))
            prv[key[i]] = al[i]
            if abs(al[i] - r) > max(0.05, 0.03 * abs(r)):
                stop = min(stop, i)                                   # the step changed: the first block ends
            elif stop == n:
                last = i
        elif pert[i]:                                                 # another kind of perturbation ends the block
            stop = min(stop, i)
    random = changes > 2
    if random:                                                       # keep the whole step run so the line is listed
        last = int(np.nonzero(st)[0][-1])
    stage = np.full(n, "other", dtype=object)
    pre = np.arange(n) < i0
    stage[pre & ~pert & (h["_phase"].to_numpy() != "practice")] = "base"
    blk = (np.arange(n) >= i0) & (np.arange(n) <= last)
    stage[blk] = np.where(st[blk], "pert", "probe")
    j = last + 1
    while j < n and not pert[j]:
        stage[j] = "post"
        j += 1
    gain = h["_gain"].to_numpy()
    y = np.full(n, np.nan)
    for k, sg in sign_of.items():
        mk = key == k
        b = gain[mk & (stage == "base")]
        b = b[np.isfinite(b)]
        if len(b):
            sgn = np.where(random & st, np.sign(al), sg)              # random steps: each trial signed by its own step
            y[mk] = (sgn * (gain - b.mean()) * 100.0)[mk]
    y[~np.isin(stage, ["base", "pert", "post"])] = np.nan
    if not np.isfinite(y[stage == "pert"]).any():
        return None
    amp = h["target_amplitude_deg"].to_numpy(float)
    size = np.where(stage == "pert", np.round(np.abs(al) / amp * 100.0, 1), np.nan)
    kinds = h.loc[(stage == "pert"), "_pt"].unique()
    why = next((SC_STEP_WHY[t] for t in kinds if SC_STEP_WHY.get(t)), None)
    if why is None and random:
        why = SC_RANDOM_WHY
    if why is None and (stage == "probe").sum() >= (stage == "pert").sum():
        why = SC_PROBE_WHY
    blk_rows = h[stage == "pert"]
    cond = blk_rows["condition"].dropna().astype(str)
    word = "backward step" if np.sign(al[i0]) < 0 else "forward step"
    ep0, pr = prior_of.get(subj, (ep, 0.0))
    prior = 1.0 if ep > ep0 else pr
    sid = f"{subj}#{ep}" + (f"|{key[i0]}" if len(set(key[st])) == 1 and len(sign_of) > 1 else "")
    grp = h["group"].dropna().astype(str)
    return pd.DataFrame({
        "exp": str(h["experiment"].iloc[0]) if h["experiment"].notna().any() else "",
        "el": str(h["experiment"].iloc[0]) if h["experiment"].notna().any() else "",
        "subject": sid, "person": h["person"].astype(str).to_numpy(), "order": np.arange(n), "trial": h["trial"].to_numpy(),
        "y": y, "stage": stage,
        "size": size, "why": why, "prior": prior, "age": h["age"].to_numpy(), "population": h["population"].to_numpy(),
        "setting": h["setting"].to_numpy(), "device": h["device"].to_numpy(),
        "_grp": grp.iloc[0] if len(grp) else "", "_cond": cond.mode().iloc[0] if cond.nunique() == 1 else "",
        "_blk": last - i0 + 1, "_after": n - last - 1, "_word": word})


# ---------------------------------------------------------------- sequence learning (module keypress)

# Family "sequence": serial reaction time (SRT) tasks, module keypress.
#
# One unit is one block. y is the speed-up of the block's mean correct reaction time from the first block of the practised
# sequence (RT of that first block minus RT of this block, ms; positive = faster). Stages: "pert" for the blocks of the
# practised sequence from the first one on, "post" for the random or new-sequence block(s) right after them (they give the
# sequence-specific learning: RT there minus RT in the last practised block), "base" for random blocks before the first
# practised block. Block RT = mean of correct presses the authors kept (3-SD rule per participant when they marked none).
# Only cued tasks with one row per key press and a reaction time on the training blocks qualify; self-paced tapping
# (inter-key intervals only), sequence-level rows and typing return nothing.
KP_COLS = ["experiment", "group", "subject", "person", "subject_excluded_by_authors", "age", "population", "setting", "device",
        "session", "block", "trial", "unit", "condition", "excluded_by_authors", "block_number", "block_type", "element_type",
        "triplet_type", "transition_probability", "sequence", "correct", "rt_ms"]
KP_NUMERIC = ["age", "trial", "rt_ms", "transition_probability"]

KP_WHY_PROB = ("probabilistic or alternating sequence (ASRT, probabilistic SRT): its sequence learning is the RT difference "
            "between likely and unlikely elements, not the speed-up on one repeating sequence")
KP_WHY_NORAND = ("no random or new-sequence block right after the practised blocks, so sequence-specific learning cannot be "
              "told apart from general speed-up")


def _kp_txt(s):
    return s.astype(object).where(s.notna(), "").astype(str).str.strip()


def prepare_sequence(d, ds):
    d = d[~_kp_txt(d["subject_excluded_by_authors"]).eq("True")]
    d = d[_kp_txt(d["unit"]).eq("keypress") & _kp_txt(d["block_type"]).ne("")].copy()
    if not len(d) or not d["rt_ms"].notna().any():
        return pd.DataFrame()
    for c in ("experiment", "group", "subject", "person", "session", "block", "condition", "block_number", "block_type",
              "element_type", "triplet_type", "sequence", "correct", "excluded_by_authors"):
        d[c] = _kp_txt(d[c])
    d["_row"] = np.arange(len(d))
    d = d.sort_values(["subject", "trial", "_row"], kind="stable")
    # a valid RT: correct (when the dataset says), not excluded by the authors, not a warm-up element, positive
    ok = d["rt_ms"].gt(0) & ~d["excluded_by_authors"].eq("True") & ~d["element_type"].eq("warm-up")
    if d["correct"].isin(["True", "False"]).any():
        ok &= d["correct"].eq("True")
    d["_rt"] = d["rt_ms"].where(ok)
    if not d["excluded_by_authors"].isin(["True", "False"]).any():
        # the authors marked no RT outliers: drop RTs beyond 3 SD of the participant's mean, the usual SRT rule
        m, sd = d.groupby("subject")["_rt"].transform("mean"), d.groupby("subject")["_rt"].transform("std")
        d["_rt"] = d["_rt"].where((d["_rt"] - m).abs() <= 3 * sd)
    # blocks: consecutive rows of one subject with the same session, block number, block label and block type
    key = d["subject"] + "|" + d["session"] + "|" + d["block_number"] + "|" + d["block"] + "|" + d["block_type"]
    d["_blk"] = (key != key.shift()).cumsum()
    prob = d["triplet_type"].isin(["high", "low"]) | d["transition_probability"].between(0.3, 0.99)
    d["_prob"] = prob
    d["_rand"] = d["element_type"].eq("random")
    g = d.groupby("_blk", sort=False)
    B = g.agg(subject=("subject", "first"), person=("person", "first"), experiment=("experiment", "first"),
              group=("group", "first"), session=("session", "first"), block=("block", "first"), btype=("block_type", "first"),
              rt=("_rt", "mean"), nrt=("_rt", "count"), prob=("_prob", "any"), rand=("_rand", "mean"),
              age=("age", "first"), population=("population", "first"), setting=("setting", "first"), device=("device", "first"))
    B["seqlab"] = g["sequence"].agg(lambda s: s[s.ne("")].mode().iloc[0] if s.ne("").any() else "")
    B["cond"] = g["condition"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
    B = B.reset_index(drop=True)

    units, firsts = [], {}
    for s, b in B.groupby("subject", sort=False):
        b = b.reset_index(drop=True)
        train_seq = b.loc[b.btype.eq("training") & b.seqlab.ne(""), "seqlab"]
        train_seq = train_seq.mode().iloc[0] if len(train_seq) else ""
        is_rand = b.btype.eq("random") | (b.rand >= 0.9)
        is_seq = b.btype.isin(["training", "test"]) & ~is_rand & (b.seqlab.eq("") | b.seqlab.eq(train_seq))
        is_new = b.btype.eq("test") & ~is_rand & b.seqlab.ne("") & b.seqlab.ne(train_seq)
        has_rt = b.nrt > 0
        kind = np.where(~has_rt, "none", np.where(is_rand | is_new, "rand", np.where(is_seq, "seq", "other")))
        start = np.nonzero(kind == "seq")[0]
        if not len(start) or not b.btype.eq("training").any():
            continue
        i0 = start[0]
        stage = np.where(kind[:i0] == "rand", "base", "other").tolist()
        i = i0
        same = b.session.eq(b.session.iloc[i0]).to_numpy()   # the practised run and its test stay within one session
        while i < len(b) and kind[i] in ("seq", "none") and same[i]:
            stage.append("pert" if kind[i] == "seq" else "probe")
            i += 1
        while i < len(b) and kind[i] == "rand" and same[i]:
            stage.append("post")
            i += 1
        stage += ["other"] * (len(b) - i)
        b["stage"] = stage
        b["y"] = b.rt.iloc[i0] - b.rt
        b.loc[~b.stage.isin(["base", "pert", "post"]), "y"] = np.nan
        pert = b[b.stage.eq("pert")]
        b["has_post"] = bool(b.stage.eq("post").any())
        b["is_prob"] = bool(pert.prob.any())
        b["glab_cond"] = b.cond.iloc[i0]
        b["exp0"] = b.experiment.iloc[i0]
        b["order"] = np.arange(len(b))
        firsts[s] = tuple(np.round(pert.rt.to_numpy(dtype=float), 3))
        units.append(b)
    if not units:
        return pd.DataFrame()
    u = pd.concat(units, ignore_index=True)

    # a person can appear under several subject codes: drop a later code whose practised blocks repeat the same RTs
    # (the same key presses analysed in two experiments), mark other later codes as prior exposure
    first_of = u.drop_duplicates("subject").sort_values(["exp0", "subject"], kind="stable")
    seen, drop, prior = {}, set(), set()
    for s, p in zip(first_of.subject, first_of.person):
        if p in seen:
            (drop if firsts[s] in seen[p] else prior).add(s)
        seen.setdefault(p, set()).add(firsts[s])
    u = u[~u.subject.isin(drop)].copy()
    u["prior"] = u.subject.isin(prior)

    # experiment: the subject's experiment, unless a subject's rows span several (then it is a task label, not an experiment)
    multi = u.groupby("subject").experiment.nunique().gt(1).any()
    u["exp"] = "" if multi else u["exp0"]
    # group label: the group, plus the condition of the first practised block when it is constant there and differs between people
    sub = u.drop_duplicates("subject")
    cond_varies = sub.glab_cond.notna().all() and sub.groupby("exp").glab_cond.nunique().gt(1).any()
    u["glabel"] = u.group + ((", " + u.glab_cond.astype(str)) if cond_varies else "")
    u["glabel"] = u.glabel.str.strip(", ")
    # why: decided per group (exp x glabel), from what most of its people did
    gs = u.drop_duplicates("subject").groupby(["exp", "glabel"]).agg(prob=("is_prob", "mean"), post=("has_post", "mean"))
    why = {k: (KP_WHY_PROB if r.prob >= 0.5 else KP_WHY_NORAND if r.post < 0.5 else np.nan) for k, r in gs.iterrows()}
    u["why"] = [why[k] for k in zip(u.exp, u.glabel)]
    return pd.DataFrame({"exp": u.exp, "el": u.exp, "glabel": u.glabel, "subject": u.subject, "person": u.person,
                         "order": u.order, "y": u.y, "stage": u.stage, "size": 1.0, "why": u.why, "prior": u.prior,
                         "age": u.age, "population": u.population, "setting": u.setting, "device": u.device})


# ---------------------------------------------------------------- speech formant adaptation (module speech)

# Explore family "speech": sustained adaptation to formant-shifted auditory feedback (module speech).
#
# One unit is one utterance (or one trial position of a 3-trial epoch mean, see below). y is the produced formant change from
# the person's own baseline for the same word, projected on the shift direction and flipped, in % of the full shift:
#     y = -(Delta . s_hat) / |S| * 100,   Delta = (F1, F2) - baseline mean of the same word, in the unit of the shift
# so 100 means the shift is fully cancelled. size is the full shift in cents at the baseline formant (the formant with the
# larger cents change), one value per line: the mean over the line's people.
#
# Decided from the data, per participant sequence (subject):
#   - a row is shifted when pert_type is formant_shift and the feedback is altered; the shift vector is (shift_f1, shift_f2)
#     in the unit named by pert_unit (mel, cents, Hz or percent of F1). Other units (bite blocks, pitch, centralizing fields
#     given as a scale factor or % of the distance to the vowel-space centre) are not this family.
#   - rows with masked-only feedback (feedback masked or none: no speech was heard) are left out, unless their phase is the
#     authors' after-effect (aftereffect or washout), where they are post trials; feedback mixed with masking noise is normal.
#   - a word is the key stimulus + segment + text colour. If a word gets shifts in different directions within one person
#     (random up/down trials, inward/outward pushing fields) the person is left out: that is trial-to-trial learning, not
#     sustained adaptation. If the shift changes size from trial to trial other than a ramp up to a hold, likewise.
#   - if different words of one person carry different shift directions at the same time (opposite shifts by word, tone,
#     language or text colour), each direction is its own sequence and the line is not pooled (dual adaptation).
#     Words that were never shifted are dropped from the sequences.
#   - the first shift block runs from the first shifted trial to the last shifted trial before a trial of another phase;
#     unshifted trials inside it are probes. If they are half of the block or more the shift was on a random subset of
#     trials (within-trial compensation design) and the person is left out.
#   - post = unshifted trials of phase aftereffect or washout after the block (masked-only trials skipped).
#   - baseline = unshifted trials of phase baseline before the block; each word's baseline mean is taken over its trials
#     among the last 30 baseline trials of the sequence (all its baseline trials if none are there).
#   - y comes from the absolute formants when they are shared on shift trials (mel from Hz with mel = 2595 log10(1 + f/700)
#     when only Hz is shared), else from the authors' change from baseline, else from their compensation_mel (checked to be
#     -(Delta . s_hat) in Bradshaw2024/2025), which is then used as shared (the authors' own baseline).
#   - a condition that is constant per sequence joins the group label only when it changes the schedule (ramp length or
#     shift size), e.g. gradual vs sudden; otherwise (word, dual task) its sequences share the line.
#   - prior = the sequence is from a later session than the person's first perturbed one (session column).
# Special cases (documented): rows that are 3-trial epoch means (measure_window says so, Kim2025_CommunPsychol) are spread
# over their 3 trial positions so that x counts trials; synchronous speech with a recorded voice and noise-vocoded feedback
# are listed but not pooled; Lametti2025 Experiment 5 has no after-effect (SP_NO_POST). prepare.skipped holds, after a
# call, the reason each left-out subject was left out.


SP_COLS = ["experiment", "group", "condition", "subject", "person", "subject_excluded_by_authors", "age", "population", "setting",
        "device", "session", "trial", "phase", "unit", "pert_type", "pert_unit", "feedback", "excluded_by_authors",
        "stimulus", "segment", "measure_window", "text_colour", "shift_f1", "shift_f2", "f1_hz", "f2_hz", "f1_mel", "f2_mel",
        "f1_cents", "f2_cents", "f1_change_hz", "f2_change_hz", "f1_change_mel", "f2_change_mel", "compensation_mel"]
SP_NUMERIC = ["age", "session", "trial", "shift_f1", "shift_f2", "f1_hz", "f2_hz", "f1_mel", "f2_mel", "f1_cents", "f2_cents",
           "f1_change_hz", "f2_change_hz", "f1_change_mel", "f2_change_mel", "compensation_mel"]

SP_BASE_N = 30          # baseline window: the last 30 baseline trials before the shift
SP_DUAL = "other words or contexts of the same people had a different shift at the same time (dual adaptation)"
# documented special case: in Lametti2025 Experiment 5 the after-effect block (S7) holds new sentences that reuse the sentence
# numbers of the training sentences (README), so a per-word baseline does not exist for them: no after-effect there.
SP_NO_POST = {("Lametti2025_Cognition", "Experiment 5")}


def _sp_mel(f):
    return 2595 * np.log10(1 + f / 700)


def _sp_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


def _sp_unit(u):
    u = str(u).lower()
    for unit, pat in (("pct", r"^(percent|%) of f1\b"), ("mel", r"^mels?\b"), ("cents", r"^cents?\b"), ("hz", r"^hz\b")):
        if re.search(pat, u):
            return unit
    return None


def _sp_values(g, unit, k):
    """produced formant k (1 or 2) of the rows of g in the shift's unit: (absolute-like values, authors' change values).
    Absolute-like values may carry a constant per-word offset (cents from the authors' reference), which the baseline removes."""
    hz, mel, ct = g[f"f{k}_hz"].to_numpy(float), g[f"f{k}_mel"].to_numpy(float), g[f"f{k}_cents"].to_numpy(float)
    if unit == "mel":
        return np.where(np.isfinite(mel), mel, _sp_mel(hz)), g[f"f{k}_change_mel"].to_numpy(float)
    if unit == "cents":
        return np.where(np.isfinite(ct), ct, 1200 * np.log2(np.where(hz > 0, hz, np.nan))), np.full(len(g), np.nan)
    if unit == "hz":
        return hz, g[f"f{k}_change_hz"].to_numpy(float)
    return hz, np.full(len(g), np.nan)          # pct: Hz now, % of the word's baseline later


def _sp_cents(unit, s, fb):
    """|full shift| in cents at baseline formant fb (Hz) for one formant's shift s in the shift's unit"""
    if not s:
        return np.nan
    if unit == "cents":
        return abs(s)
    if unit == "pct":
        return abs(1200 * np.log2(1 + s / 100))
    if not np.isfinite(fb) or fb <= 0:
        return np.nan
    fs = _sp_hz(_sp_mel(fb) + s) if unit == "mel" else fb + s
    return abs(1200 * np.log2(fs / fb)) if fs > 0 else np.nan


def _sp_dirs(v, tol=0.05):
    """cluster unit vectors (rows of v) into distinct directions; returns (labels, directions)"""
    dirs, lab = [], []
    for x in v:
        j = next((i for i, dd in enumerate(dirs) if np.hypot(*(x - dd)) < tol), None)
        if j is None:
            dirs.append(x)
            j = len(dirs) - 1
        lab.append(j)
    return np.array(lab), dirs


def _sp_shift_text(s1, s2, unit):
    parts = [f"F{k} {v:+.4g}" for k, v in ((1, s1), (2, s2)) if abs(v) > 1e-9]
    return ", ".join(parts) + " " + {"pct": "% of F1"}.get(unit, unit)


def _sp_sequence(g, unit):
    """stages, full shift and ramp length of one sequence (rows of one direction of one subject, in order);
    (None, reason) when it is not a sustained shift."""
    sh = g["_shifted"].to_numpy(bool)
    mag = g["_mag"].to_numpy(float)
    ph = g["phase"].astype(str).to_numpy()
    n = len(g)
    on = int(np.nonzero(sh)[0][0])
    end, i = on, on + 1
    while i < n and (sh[i] or ph[i] == ph[on]):  # the block ends at the last shifted trial before a trial of another phase
        if sh[i]:
            end = i
        i += 1
    blk = slice(on, end + 1)
    probe = ~sh[blk]
    if probe.mean() >= 0.5:
        return None, "shift on a random subset of trials"
    m = mag[blk][sh[blk] & np.isfinite(mag[blk])]
    if not len(m):
        return None, "shift size not shared"
    M = m.max()
    full_at = int(np.argmax(m >= M * 0.99))
    if np.any(np.diff(m[:full_at + 1]) < -0.01 * M) or np.any(m[full_at:] < M * 0.99):
        return None, "shift size changed from trial to trial"
    if len(m) - full_at < 10:
        return None, "the shift never held one size for 10 trials"
    stage = np.full(n, "other", dtype=object)
    stage[:on][(ph[:on] == "baseline") & ~sh[:on]] = "base"
    stage[blk] = np.where(sh[blk], "pert", "probe")
    nxt = np.nonzero(sh[end + 1:])[0]
    stop = end + 1 + (int(nxt[0]) if len(nxt) else n)
    post = np.isin(ph[end + 1:stop], ["aftereffect", "washout"])
    stage[end + 1:stop][post] = "post"
    i_full = np.nonzero(sh & np.isclose(mag, M, rtol=0.01))[0][0]
    return {"stage": stage, "S": (g["_s1"].iloc[i_full], g["_s2"].iloc[i_full]), "M": M,
            "ramp": int(np.sum(sh[blk] & ~(mag[blk] >= M * 0.99)))}, None


def prepare_speech(d, ds):
    prepare_speech.skipped = {}
    d = d[~d["subject_excluded_by_authors"].eq("True") & ~d["unit"].eq("participant summary")].copy()
    for c in ("experiment", "group", "condition"):
        d[c] = d[c].fillna("").astype(str)
    d["_u"] = d["pert_unit"].map(_sp_unit)
    d["_shifted"] = d["pert_type"].eq("formant_shift") & d["feedback"].eq("altered") & d["_u"].notna()
    if not d["_shifted"].any():
        return pd.DataFrame()
    # masked-only feedback (no speech heard) is not baseline, shift or after-effect, unless the authors call it their after-effect phase
    d = d[~d["feedback"].isin(["masked", "none"]) | d["phase"].isin(["aftereffect", "washout"])].copy()
    s1, s2 = d["shift_f1"].astype(float), d["shift_f2"].astype(float)
    d["_s1"] = s1.fillna(0).where(s1.notna() | s2.notna())
    d["_s2"] = s2.fillna(0).where(s1.notna() | s2.notna())
    d["_mag"] = np.where(d["_shifted"], np.hypot(d["_s1"], d["_s2"]), 0.0)
    d["_key"] = d["stimulus"].fillna("").astype(str) + "|" + d["segment"].fillna("").astype(str) + "|" + d["text_colour"].fillna("").astype(str)
    d["_row"] = np.arange(len(d))
    text = (d["experiment"].fillna("") + " " + d["condition"].fillna("") + " " + d["device"].fillna("")).str.lower()
    d["_why"] = np.where(text.str.contains("synchron"), "participants read in synchrony with a recorded voice, whose formants also pulled theirs",
                         np.where(text.str.contains("vocod"), "the feedback was noise-vocoded (degraded) throughout", ""))
    seqs, skipped = [], {}
    for subj, g in d.sort_values(["subject", "trial", "_row"], kind="stable").groupby("subject", sort=False):
        if not g["_shifted"].any():
            continue
        unit = g.loc[g["_shifted"], "_u"].mode().iloc[0]
        known = g["_shifted"].to_numpy() & np.isfinite(g["_mag"].to_numpy(float)) & (g["_mag"].to_numpy(float) > 0)
        if not known.any():
            skipped[subj] = "shift size not shared"
            continue
        uv = np.c_[g["_s1"], g["_s2"]][known] / g["_mag"].to_numpy(float)[known, None]
        lab, dirs = _sp_dirs(uv)
        kd = pd.DataFrame({"key": g["_key"].to_numpy()[known], "dir": lab}).groupby("key")["dir"].agg(lambda x: set(x))
        if (kd.map(len) > 1).any():
            skipped[subj] = "shift direction changed from trial to trial within a word"
            continue
        kdir = kd.map(lambda x: next(iter(x)))
        for j in sorted(set(kdir)):
            keys = set(kdir.index[kdir == j])
            h = g[g["_key"].isin(keys)]
            info, why_not = _sp_sequence(h, unit)
            if info is None:
                skipped[subj] = why_not
                continue
            conds = h.loc[h["_shifted"], "condition"].unique()
            seqs.append({"subject": subj, "rows": h, "info": info, "unit": unit, "ndir": len(set(kdir)), "dir": j,
                         "dlabel": conds[0] if len(conds) == 1 and conds[0] else _sp_shift_text(*info["S"], unit),
                         "cond": g["condition"].iloc[0] if g["condition"].nunique() == 1 else None})
    prepare_speech.skipped = skipped
    if not seqs:
        return pd.DataFrame()
    # a condition that is constant per sequence joins the group label when it changes the schedule (ramp length, shift size)
    sig = {}
    for q in seqs:
        r0 = q["rows"].iloc[0]
        sig.setdefault((r0["experiment"], r0["group"]), {}).setdefault(q["cond"], []).append((q["info"]["ramp"], round(q["info"]["M"], 1)))
    split = {k: len(v) > 1 and None not in v and len({max(set(x), key=x.count) for x in v.values()}) > 1 for k, v in sig.items()}
    out = []
    for q in seqs:
        h, info, unit = q["rows"], q["info"], q["unit"]
        r0 = h.iloc[0]
        glabel = r0["group"]
        if split[(r0["experiment"], r0["group"])]:
            glabel = (glabel + ", " if glabel else "") + str(q["cond"])
        if q["ndir"] > 1:
            glabel = (glabel + ", " if glabel else "") + str(q["dlabel"])
        stage = info["stage"]
        if (ds["name"], r0["experiment"]) in SP_NO_POST:
            stage = np.where(stage == "post", "other", stage)
        S1, S2 = info["S"]
        M = info["M"]
        u1, u2 = S1 / M, S2 / M
        # produced values in the shift unit: absolute (or with a constant per-word offset) if shared on shift trials, else
        # the authors' change from their baseline, else their compensation (already projected and opposing-positive)
        a1, c1 = _sp_values(h, unit, 1)
        a2, c2 = _sp_values(h, unit, 2)
        pert = stage == "pert"
        need2 = abs(u2) > 1e-9
        def ok(x1, x2):
            return np.isfinite(x1[pert]).any() and (not need2 or np.isfinite(x2[pert]).any())
        comp = h["compensation_mel"].to_numpy(float)
        src = "abs" if ok(a1, a2) else "chg" if ok(c1, c2) else "comp" if unit == "mel" and np.isfinite(comp[pert]).any() else None
        if src is None:
            skipped[q["subject"]] = "no produced formants on shift trials"
            continue
        keys = h["_key"].to_numpy()
        rep = 3 if h["measure_window"].fillna("").str.contains("then over the epoch's three words").all() else 1
        base_idx = np.nonzero(stage == "base")[0]
        win = base_idx[-(SP_BASE_N // rep):]
        hzb = [h[f"f{k}_hz"].to_numpy(float) if h[f"f{k}_hz"].notna().any() else _sp_hz(h[f"f{k}_mel"].to_numpy(float)) for k in (1, 2)]
        fb1, fb2 = (np.nanmean(x[base_idx]) if len(base_idx) and np.isfinite(x[base_idx]).any() else np.nan for x in hzb)
        if src == "comp":
            r = comp / M * 100
            b = np.zeros(len(h))
        else:
            v1, v2 = (a1, a2) if src == "abs" else (c1, c2)
            if not need2:
                v2 = np.zeros(len(h))
            b1, b2 = np.full(len(h), np.nan), np.full(len(h), np.nan)
            for k in set(keys):
                for v, b in ((v1, b1), (v2, b2)):
                    w = [i for i in win if keys[i] == k and np.isfinite(v[i])] or [i for i in base_idx if keys[i] == k and np.isfinite(v[i])] \
                        or [i for i in win if np.isfinite(v[i])]
                    b[keys == k] = np.mean(v[w]) if w else (0.0 if src == "chg" else np.nan)
            D1, D2 = v1 - b1, v2 - b2
            if unit == "pct":   # % change from the word's baseline formant
                D1 = 100 * D1 / b1
                D2 = 100 * D2 / b2 if need2 else D2
            y = -(D1 * u1 + (D2 * u2 if need2 else 0)) / M * 100
            r = -(v1 * u1 + (v2 * u2 if need2 else 0)) / M * 100   # before the baseline is removed (for checks against papers)
            b = r - y
        y = np.where(h["excluded_by_authors"].eq("True").to_numpy(), np.nan, r - b)
        if not np.isfinite(y[pert]).any():
            skipped[q["subject"]] = "no produced formants on shift trials"
            continue
        cents = np.nanmax([_sp_cents(unit, S1, fb1), _sp_cents(unit, S2, fb2), -1])
        frac = np.where(pert, h["_mag"].to_numpy(float) / M, np.nan)
        keep = stage != "other"
        order = h["trial"].to_numpy(float)
        sess = h["session"].to_numpy(float)
        u = pd.DataFrame({"exp": r0["experiment"], "el": r0["experiment"],
                          "glabel": glabel, "subject": f"{q['subject']}|{q['dir']}" if q["ndir"] > 1 else q["subject"],
                          "person": h["person"].to_numpy(), "order": order, "y": y, "stage": stage, "frac": frac,
                          "why": SP_DUAL if q["ndir"] > 1 else (r0["_why"] or np.nan), "cents": cents if cents > 0 else np.nan,
                          "session": np.nanmin(sess) if np.isfinite(sess).any() else np.nan, "trial": order, "r": r, "key": keys,
                          "age": h["age"].to_numpy(), "population": h["population"].to_numpy(), "setting": h["setting"].to_numpy(),
                          "device": h["device"].to_numpy()})[keep]
        if rep > 1:  # one row = mean of a 3-trial epoch: give it each of its 3 trial positions
            u = u.loc[u.index.repeat(rep)].assign(order=lambda x: x["order"] * rep + np.tile(np.arange(rep), len(x) // rep) - rep + 1)
        out.append(u)
    prepare_speech.skipped = skipped
    if not out:
        return pd.DataFrame()
    u = pd.concat(out, ignore_index=True)
    u = u[u.groupby("subject")["cents"].transform(lambda c: np.isfinite(c).all())]  # size in cents unknown (no baseline formant)
    if not len(u):
        return pd.DataFrame()
    first = u.groupby("person")["session"].transform("min")
    u["prior"] = u["session"] > first
    # one size per line: the mean over the line's sequences of each one's full shift in cents
    G = u.drop_duplicates("subject").groupby(["exp", "glabel"])["cents"].mean().round(0)
    u["size"] = u.set_index(["exp", "glabel"]).index.map(G).to_numpy(float) * u["frac"]
    return u.drop(columns=["frac", "cents", "session"]).reset_index(drop=True)


# ---------------------------------------------------------------- force-field reaching (module force_field_reaching)

# Force-field family: prepare(d, ds) for module force_field_reaching (see FAMILIES 'forcefield' and series_lines).
#
# One unit = one trial of one participant sequence. y = force compensation (FC, %) on channel trials of the trained hand and
# direction, signed so that compensating the first field is positive; NaN on field, null and other trials. The first field
# block runs from the first field trial of the first run of at least 3 field trials, over the channel trials inside it, to the
# last field trial of the same field; a run of 3 or more channel trials right after it is "post" (an error-clamp decay block).
FF_COLS = ["experiment", "group", "condition", "subject", "person", "subject_excluded_by_authors", "age", "population", "setting",
        "device", "session", "trial", "phase", "pert_type", "pert_value", "pert_unit", "excluded_by_authors", "trial_type", "hand",
        "target_direction_deg", "force_compensation_pct", "force_over_velocity"]
FF_NUMERIC = ["trial", "pert_value", "age", "target_direction_deg", "force_compensation_pct", "force_over_velocity"]

FF_MIN_FIELD_RUN = 3     # a block starts with a run of at least 3 field trials (single exposures, e.g. Heald 2021 Exp 2 triplets, are not a block)
FF_MAX_CATCH = 2         # null runs of up to 2 trials between field trials are catch trials inside the block
FF_MIN_CHANNEL_BLOCK = 10  # a run of 10 or more channel trials is a channel block, not channel trials scattered in the field block
FF_MIN_POST = 3          # channel trials right after the block's last field trial count as "post" when there are at least 3 in a row
FF_DUAL_RUN = 5          # opposing fields whose same-sign runs have a median length under 5 trials are switched by context (dual adaptation)
# Documented special cases (the data cannot show them):
FF_DUAL_IDS = {"Yeo2015_PLoSOne": "two opposing fields, each tied to an arm posture (dual adaptation)"}  # field sign not in the deposit
FF_SIZE_FROM_PAPER = {"Foray2025_eNeuro": 15.0}  # k = 15 N s/m (Methods, Eq. 1); the deposit has no field trials
FF_NO_FC = "the deposit has no force measure on channel trials (kinematic errors only)"


def _ff_scale(unit, val):
    """factor from the deposit's field coefficient to N s/m; NaN when the field is not velocity-dependent (N/m) or unknown"""
    u = str(unit).lower().replace(" ", "")
    if "ns/cm" in u:
        return 100.0
    if "ns/m" in u:
        return 1.0
    if "curl" in u and abs(val) < 1:  # Yokoi2022: unit not given; B and force_over_velocity share units (N per cm/s), so 0.15 = 15 N s/m
        return 100.0
    return np.nan


def _ff_segment(ff, ch, start):
    """end index (inclusive, a field trial) of the stretch of field trials that begins at `start`: gaps between field trials
    stay inside when they are channel trials only (fewer than FF_MIN_CHANNEL_BLOCK) or null trials only (at most FF_MAX_CATCH)"""
    idx = np.flatnonzero(ff)
    idx = idx[idx >= start]
    end = idx[0]
    for b in idx[1:]:
        gap = slice(end + 1, b)
        n = b - end - 1
        if n and not ((ch[gap].all() and n < FF_MIN_CHANNEL_BLOCK) or ((~ch[gap]).all() and n <= FF_MAX_CATCH)):
            break
        end = b
    return end


def _ff_first_block(ff, ch):
    """(onset, end) of the first stretch with at least FF_MIN_FIELD_RUN field trials, or None"""
    k = 0
    while True:
        rest = np.flatnonzero(ff[k:])
        if not len(rest):
            return None
        a = k + rest[0]
        e = _ff_segment(ff, ch, a)
        if ff[a:e + 1].sum() >= FF_MIN_FIELD_RUN:
            return a, e
        k = e + 1


def _ff_channel(g):
    return (g["trial_type"].astype(str).eq("channel") | g["pert_type"].isin(["channel", "error_clamp"])).to_numpy()


def _ff_field(g):
    """field trials (Albert2020 marks its channel trials in a field block with the field's pert_type too)"""
    return g["pert_type"].astype(str).str.startswith("force_field").to_numpy() & ~_ff_channel(g)


def _ff_runs(v):
    """lengths of runs of equal values"""
    v = np.asarray(v)
    if not len(v):
        return np.array([], int)
    cut = np.flatnonzero(v[1:] != v[:-1]) + 1
    return np.diff(np.r_[0, cut, len(v)])


def _ff_sequence(g, ds, fc_col):
    """(stage, size, y, fc, why) arrays for one participant sequence (rows in order), or None when it has no field block"""
    n = len(g)
    ff = _ff_field(g)
    ch = _ff_channel(g)
    B = g["pert_value"].to_numpy(dtype=float)
    stage = np.array(["other"] * n, dtype=object)
    size = np.full(n, np.nan)
    why = []
    blk = _ff_first_block(ff, ch)
    if blk is None:
        # no field trials shared at all, only the channel trials of training (Foray2025): those stand for the block, unpooled
        lrn = ch & g["phase"].eq("learning").to_numpy()
        if not fc_col or not lrn.any():
            return None
        a = e = int(np.flatnonzero(lrn)[0])
        while e + 1 < n and lrn[e + 1]:
            e += 1
        stage[:a] = "base"
        stage[a:e + 1] = "pert"
        size[a:e + 1] = FF_SIZE_FROM_PAPER.get(ds["name"], np.nan)
        fc = g[fc_col].to_numpy(dtype=float)
        return stage, size, np.full(n, np.nan), fc, ("the deposit has only the channel trials of training, not the field trials "
                                                     "around them, so their trial positions are unknown")
    a, e = blk
    seg = np.arange(a, e + 1)
    fseg = seg[ff[seg]]
    sgn = np.sign(B[fseg])
    sgn = sgn[np.isfinite(sgn) & (sgn != 0)]
    cond = g["condition"].astype(str).to_numpy()
    dual = FF_DUAL_IDS.get(ds["name"])
    if not dual and len(set(sgn)) > 1 and np.median(_ff_runs(sgn)) < FF_DUAL_RUN:
        dual = "two opposing fields alternate within the block (dual adaptation)"
    if not dual and g["condition"].notna().all() and len(_ff_runs(cond[seg])) > 4 and len(set(cond[fseg])) > 1:
        dual = "two contexts alternate within the block, each with its own field (dual adaptation)"
    if not dual and len(set(sgn)) > 1:  # a reversal: the block ends at the last trial of the first field
        flip = fseg[np.flatnonzero(np.sign(B[fseg]) == -sgn[0])[0]]
        e = fseg[fseg < flip][-1]
        fseg = fseg[fseg <= e]
    # channel trials right after the last field trial: post when 3 or more in a row, else part of the block
    # (when the phase changes inside that run, as in Franklin2025, the channel trials before the change stay in the block)
    ph = g["phase"].to_numpy(dtype=object)
    r = 0
    while e + 1 + r < n and ch[e + 1 + r]:
        r += 1
    j = next((i for i in range(r) if ph[e + 1 + i] != ph[e]), r)
    post = r - j >= FF_MIN_POST if j < r else r >= FF_MIN_POST
    if j < r:      # phase change: the channel trials before it belong to the block, the run after it is post if long enough
        e, r = e + j, r - j
    elif not post:  # a short run in the block's phase: channel trials scattered at the end of the block
        e += r
    s0 = np.sign(np.median(B[fseg][B[fseg] != 0])) if (B[fseg] != 0).any() else 1.0
    absB = np.abs(B[fseg])
    full = np.nanmax(absB) if np.isfinite(absB).any() else np.nan
    scale = _ff_scale(g["pert_unit"].iloc[fseg[0]], full)
    stage[:a] = np.where(ff[:a], "other", "base")
    stage[a:e + 1] = np.where(ff[a:e + 1] | ch[a:e + 1], "pert", "probe")
    if post:
        stage[e + 1:e + 1 + r] = "post"
    size[fseg] = absB * scale
    size[a:e + 1] = pd.Series(size[a:e + 1]).ffill().to_numpy()  # channel trials carry the field level around them
    # FC on channel trials of the trained hand and target direction, signed with the first field
    if fc_col == "force_compensation_pct":
        fc = g[fc_col].to_numpy(dtype=float) * s0
    elif fc_col == "force_over_velocity":  # Yokoi2022: lateral force / speed at peak velocity; field trials show the field's sign
        v = g[fc_col].to_numpy(dtype=float)
        sv = np.sign(np.nanmedian(v[fseg])) or 1.0
        fc = 100.0 * v * sv / full
    else:
        fc = np.full(n, np.nan)
        why.append(FF_NO_FC)
    fc = np.where(ch, fc, np.nan)
    for col in ("hand", "target_direction_deg"):
        x = g[col].to_numpy(dtype=object)
        trained = {t for t in x[fseg] if t == t}
        if trained:
            fc = np.where(np.array([t != t or t in trained for t in x]), fc, np.nan)
    if dual:
        why.append(dual)
    if not np.isfinite(scale):
        unit = str(g["pert_unit"].iloc[fseg[0]]).replace(" ", "")
        why.append("the field strength is not in the data" if "curl" in str(g["pert_type"].iloc[fseg[0]]) + unit else
                   "a position-dependent field, not a velocity-dependent one" if unit.endswith("N/m") else
                   "not a velocity-dependent curl field with one field constant")
    if absB.min() < 0.95 * full:
        why.append("the field was ramped up gradually" if absB[0] < 0.5 * full else "the field strength changed within the block")
    y = fc if not why else np.full(n, np.nan)  # lines that are not pooled keep FC only in the fc column
    return stage, size, y, fc, "; ".join(dict.fromkeys(why))


def prepare_forcefield(d, ds):
    d = d[~d["subject_excluded_by_authors"].eq("True")].copy()
    fc_col = next((c for c in ("force_compensation_pct", "force_over_velocity") if d[c].notna().any()), None)
    if not d["pert_type"].astype(str).str.startswith("force_field").any() and not (
            fc_col and (d["trial_type"].eq("channel") & d["phase"].eq("learning")).any()):
        return pd.DataFrame()  # no curl field here (feedback-response and step-load studies of the module)
    d["sess"] = pd.to_numeric(d["session"], errors="coerce")
    d = d.sort_values(["subject", "sess", "trial"], kind="stable")
    out = []
    for s, g in d.groupby("subject", sort=False):
        g = g.reset_index(drop=True)
        res = _ff_sequence(g, ds, fc_col)
        if res is None:
            continue
        stage = res[0]
        # a condition whose trials inside the block are all channel trials (Sadeghi2018 Return reaches) is another movement
        # whose channel trials measure generalization: its rows are left out (catch-trial conditions are null trials, kept)
        ff = _ff_field(g)
        isch = _ff_channel(g)
        inblk = np.isin(stage, ["pert", "probe"])
        cond = g["condition"].astype(str).to_numpy()
        met = set(cond[inblk & ff])
        other = {c for c in set(cond[inblk]) - met if isch[inblk & (cond == c)].all()}
        if met and g["condition"].notna().all() and other:
            g = g[~np.isin(cond, list(other))].reset_index(drop=True)
            res = _ff_sequence(g, ds, fc_col)
        stage, size, y, fc, why = res
        y = np.where(g["excluded_by_authors"].eq("True"), np.nan, y)
        fc = np.where(g["excluded_by_authors"].eq("True"), np.nan, fc)
        out.append(pd.DataFrame({"exp": g["experiment"].fillna("").astype(str), "group": g["group"].fillna("").astype(str),
                                 "cond1": str(g["condition"].iloc[0]) if g["condition"].nunique() == 1 else "",
                                 "subject": s, "person": g["person"].astype(str), "order": np.arange(len(g)), "y": y, "fc": fc,
                                 "stage": stage, "size": size, "why": why or np.nan, "session": g["session"],
                                 "sess": g["sess"], "age": g["age"], "population": g["population"], "setting": g["setting"],
                                 "device": g["device"]}))
    if not out:
        return pd.DataFrame()
    u = pd.concat(out, ignore_index=True)
    # group label: the group, plus the condition when each sequence has one condition and a group has several (abrupt, gradual)
    per = u.groupby("subject", sort=False).agg(exp=("exp", "first"), group=("group", "first"), cond1=("cond1", "first"),
                                               person=("person", "first"), sess=("sess", "min"))
    many = per[per.cond1.ne("")].groupby(["exp", "group"]).cond1.nunique()
    per["glabel"] = [f"{g}, {c}" if c and many.get((x, g), 0) > 1 else g for x, g, c in zip(per.exp, per.group, per.cond1)]
    # prior exposure: a person with several sequences had one earlier when the sessions order them; when the order is not in
    # the data (Hadjiosif2024, counterbalanced) each sequence gets the expected share (k - 1) / k
    k = per.groupby("person").sess.transform("count")
    nseq = per.groupby("person").group.transform("size")
    first = per.groupby("person").sess.transform("min")
    per["prior"] = np.where(nseq == 1, False, np.where(k == nseq, per.sess > first, (nseq - 1) / nseq)).astype(object)
    u["glabel"] = u["subject"].map(per["glabel"])
    u["prior"] = u["subject"].map(per["prior"])
    u["el"] = u["exp"]
    return u.drop(columns=["cond1", "group", "sess"])


def _series_extractor(fam, prepare, cols, numeric, need_any=()):
    def run(zip_path, ds):
        return series_lines(ds, fam, prepare(read_main(zip_path, ds, cols, numeric, need_any), ds))
    return run


EXTRACTORS = {"reach90": reach90_lines,
              "walking": _series_extractor("splitbelt", prepare_splitbelt, SB_COLS, SB_NUMERIC),
              "saccade": _series_extractor("saccade", prepare_saccade, SC_COLS, SC_NUMERIC),
              "keypress": _series_extractor("sequence", prepare_sequence, KP_COLS, KP_NUMERIC, need_any=("block_type",)),
              "speech": _series_extractor("speech", prepare_speech, SP_COLS, SP_NUMERIC),
              "force_field_reaching": _series_extractor("forcefield", prepare_forcefield, FF_COLS, FF_NUMERIC)}

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
    """Term links (tf-idf) between the studies of one family and the vocabulary.
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
            if old and old.get("sha256") == ds["sha256"] and old.get("version") == module_version(ds["module"]):
                continue
            if not fetch and not ds["local"].exists():
                print(f"  skipped (offline): {ds['name']}", flush=True)
                continue
            zp, downloaded = get_zip(ds, tmp)
            if zp is None:
                continue
            try:
                lines = EXTRACTORS[ds["module"]](zp, ds)
            except Exception as e:  # one dataset the code cannot read must not stop the others; it is tried again next run
                import traceback
                print(f"  EXTRACTOR FAILED for {ds['name']}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue
            finally:
                if downloaded:
                    zp.unlink(missing_ok=True)
            cp.write_text(json.dumps({"dataset": ds["name"], "study": ds["study"], "module": ds["module"], "sha256": ds["sha256"],
                                      "version": module_version(ds["module"]), "lines": lines}, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
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
        ser = entry.pop("series", None)
        if ser:
            entry["pre"], entry["minBlock"], entry["yround"] = ser["pre"], ser["min_block"], ser.get("round", 1)
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
