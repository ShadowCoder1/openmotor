"""Shared code for standardizing datasets that are not in the motor-centaur corpus (any motor task).

Every standardized file has the CORE columns (identification, participant, order, task, perturbation, feedback),
followed by the columns of one task MODULE (walking, speech, saccade, ...), or is in the 90-column reaching format of the
97 standardized studies (module 'reach90'). Modules are added with the first real dataset of that kind, never ahead of
the data. Rules:
  - the authors' trial-level values are re-encoded, not re-analysed: no new measures beyond unit or sign changes
    that the README states;
  - participant identifiers are replaced by codes unique within the dataset (subject = <id>_S0001..., person =
    <id>_P0001..., assigned here in order of first appearance), and every text cell is scanned for identifier patterns;
  - a ZIP is written only after at least one check against a number the authors published (or an explicit
    statement that the paper reports none, with the internal checks that were run instead); the checks are listed in
    the README.
Output matches the 97 standardized files: CSV with CRLF line ends, True/False, integers without a decimal point,
other numbers to 8 significant digits, empty cells for missing values.

Raw downloads live in raw_extra/<id>/ (untrusted data: read only, never executed; fetch with tools/std/fetch.py).
Converters live in tools/std/convert/<id>.py and are run with python3 -I.
"""
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
RAW = HERE / "raw_extra"
OUT = HERE / "source/release_extra"
REC = HERE / "source/release_extra_records"

# short plain numbers cannot match any ID pattern (all need 16+ hex chars, an "@", "A" + 11 capitals, or hex-hex-), so the scan skips them
_PLAIN_NUMBER = re.compile(r"[-+0-9.eE]{1,15}")
ID_PATTERNS = {
    "hex16": re.compile(r"\b[0-9a-f]{16}\b"),
    "hex24": re.compile(r"\b[0-9a-f]{24}\b"),
    "mturk": re.compile(r"\bA[0-9A-Z]{11,14}\b"),
    "uuid": re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
}

# (section, column, type, values or units, description)
CORE = [
    ("Identification", "dataset", "text", "", "Dataset name."),
    ("Identification", "study", "text", "", "Study name: first author, year and journal of the paper."),
    ("Identification", "experiment", "text", "", "Experiment within the dataset, in the authors' naming where they give one."),
    ("Identification", "group", "text", "", "Between-participant condition, in the authors' naming. Empty for single-group experiments."),
    ("Identification", "subject", "text", "", "One sequence of rows from one participant, coded within this dataset."),
    ("Identification", "person", "text", "",
     "One participant, coded within this dataset. A person can have more than one subject sequence, for example after "
     "taking part in two experiments."),
    ("Identification", "subject_excluded_by_authors", "true/false", "", "The original analysis excluded this participant."),
    ("Participant", "age", "number", "years", "Age in years, if the authors shared it per participant."),
    ("Participant", "sex", "category", "F, M, other", "Sex or gender as reported."),
    ("Participant", "handedness", "category", "right, left, ambi", "Handedness, if reported."),
    ("Participant", "population", "text", "", "healthy, or the clinical or developmental group named by the authors."),
    ("Participant", "setting", "category", "lab, online, clinic, home", "Where the data were collected."),
    ("Participant", "device", "text", "", "Apparatus, for example treadmill, microphone and headphones, keyboard."),
    ("Trial order", "session", "integer", "", "Session number, from 1."),
    ("Trial order", "block", "text", "", "Block or phase label in the authors' naming."),
    ("Trial order", "trial", "integer", "", "Order of the row within the participant's sequence, from 1."),
    ("Trial order", "phase", "category", "practice, baseline, learning, aftereffect, washout, test, transfer, other",
     "Standard phase of the row."),
    ("Trial order", "time_s", "number", "seconds", "Time of the row from the start of the session, where recorded."),
    ("Task", "task", "text", "", "Kind of movement, as on the OpenMotor Studies page (for example walking, speech, eye movements, "
     "finger tapping and timing, sequence learning, grasping and lifting, balance and posture, tracking)."),
    ("Task", "unit", "category", "trial, stride, step, utterance, keypress, sequence, tap, saccade, participant summary, recording",
     "What one row is."),
    ("Task", "condition", "text", "", "Within-participant condition of the row, in the authors' naming."),
    ("Perturbation", "pert_type", "text", "", "Perturbation in force on this row: none, or for example split_belt, "
     "formant_shift, pitch_shift, force_field, intrasaccadic_step, delay."),
    ("Perturbation", "pert_value", "number", "", "Size of the perturbation, in pert_unit. 0 when none."),
    ("Perturbation", "pert_unit", "text", "", "Unit of pert_value, for example ratio, Hz, cents, deg, ms."),
    ("Feedback", "feedback", "category", "normal, altered, none, masked", "Sensory feedback the participant received."),
    ("Outcome", "excluded_by_authors", "true/false", "", "The original analysis excluded this row."),
]

MODULES = {}  # name -> list of (section, column, type, values or units, description)


def register_module(name, cols):
    MODULES[name] = [("Task-specific", *c) if len(c) == 4 else c for c in cols]


def load_module(name):
    """Task modules live in tools/std/modules/<name>.py, each defining COLUMNS = [(column, type, values or units, description), ...]
    and a DESCRIPTION string. The first dataset of a kind defines its module; later datasets of that kind reuse it."""
    if name not in MODULES:
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"om_module_{name}", Path(__file__).resolve().parent / "modules" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        register_module(name, mod.COLUMNS)
    return MODULES[name]


def reach90_columns():
    """The 90-column OpenMotor reaching format of the 97 standardized studies (source/columns.csv), for new reaching datasets."""
    with open(HERE / "source/columns.csv", newline="", encoding="utf-8") as f:
        return [(r["section"], r["column"], r["type"], r["values_or_units"], r["description"]) for r in csv.DictReader(f)]


def columns_for(module):
    return reach90_columns() if module == "reach90" else CORE + load_module(module)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def raw_dir(ds_id):
    return RAW / ds_id


class Checks:
    def __init__(self):
        self.items = []

    def published(self, what, computed, reported, tol, source):
        ok = abs(float(computed) - float(reported)) <= tol
        self.items.append(dict(kind="published", what=what, computed=float(computed), reported=float(reported), tol=tol,
                               source=source, ok=bool(ok)))
        return ok

    def internal(self, what, ok, detail=""):
        self.items.append(dict(kind="internal", what=what, ok=bool(ok), detail=detail))
        return ok

    def passed(self):
        return all(c["ok"] for c in self.items) and any(c["kind"] == "published" and c["ok"] for c in self.items)


_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


def _is_missing(v):
    if v is None:
        return True
    try:
        import pandas as pd
        if v is pd.NA or v is pd.NaT:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return isinstance(v, float) and math.isnan(v) or (hasattr(v, "dtype") and v != v)
    except Exception:  # noqa: BLE001
        return False


def _fmt(v, ctype):
    if _is_missing(v):
        return ""
    if ctype == "true/false":
        s = str(v).strip().lower()
        if s in _TRUE:
            return "True"
        if s in _FALSE:
            return "False"
        if s == "":
            return ""
        raise ValueError(f"not a true/false value: {v!r}")
    tn = type(v).__name__
    if isinstance(v, bool) or tn in ("bool_", "bool"):
        return "True" if v else "False"
    if isinstance(v, int) or tn.startswith(("int", "uint")):
        return str(int(v))
    if isinstance(v, float) or tn.startswith("float"):
        f = float(v)
        if math.isinf(f):
            return ""
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return format(f, ".8g")
    return str(v)


def _iter_rows(rows, names):
    if hasattr(rows, "to_dict"):
        missing = [c for c in names if c not in rows.columns and c != "person"]
        extra = [c for c in rows.columns if c not in names]
        assert not missing and not extra, (missing, extra)
        cols = [c for c in names if c in rows.columns]
        for tup in rows[cols].itertuples(index=False, name=None):
            yield dict(zip(cols, tup))
    else:
        for r in rows:
            yield r


def skip(ds_id, reason, details=""):
    """Record why a listed dataset was not standardized (it stays a link to the authors' data on the site).
    reason: one of 'no trial-level data', 'license does not allow a converted copy', 'too large', 'not documented enough',
    'no number to check against', 'files unreadable', 'not motor behaviour', 'other'."""
    REC.mkdir(parents=True, exist_ok=True)
    rec = dict(id=ds_id, skipped=True, reason=reason, details=details,
               sources=json.load(open(raw_dir(ds_id) / "SOURCES.json")) if (raw_dir(ds_id) / "SOURCES.json").exists() else [])
    json.dump(rec, open(REC / f"{ds_id}.skip.json", "w"), indent=1)
    print(f"{ds_id}: SKIPPED ({reason}) {details[:200]}")
    return rec


def release(ds_id, module, rows, meta, checks, no_published_reason=None, participants=None, participant_columns=None,
            extra_files=None):
    """Write source/release_extra/<ds_id>.zip and its record.

    rows: pandas DataFrame, list of dicts, or any re-iterable of dicts, with exactly the columns of the format ('person' may
          be left out: it is then the same as 'subject'). Subject and person codes are re-assigned here.
    meta: paper (authors, year, title, venue, doi), lab, data_url, license, license_url, license_text (optional), changes,
          how (list of sentences), unit_note, differences (optional list: published numbers that did not reproduce and why).
    participants (optional): one row per person with key 'person' (the converter's person code) and the columns described by
          participant_columns = [(column, type, values or units, description), ...]; written as participants.csv.
    module 'reach90' writes the 90-column reaching format (source/columns.csv).
    Task-module columns a row leaves out are written empty (so a module can gain columns later); CORE columns are required.
    extra_files: optional list of (path inside <ds_id>/, bytes or local path) for formats that ship more than one table, such
    as module 'timeseries' (samples/...csv)."""
    cols = columns_for(module)
    names = [c[1] for c in cols]
    ctype = {c[1]: c[2] for c in cols}
    if no_published_reason is None:
        assert checks.passed(), [c for c in checks.items if not c["ok"]] or "no published check"
    else:
        assert all(c["ok"] for c in checks.items) and checks.items, "internal checks failed or missing"
    OUT.mkdir(parents=True, exist_ok=True)
    REC.mkdir(parents=True, exist_ok=True)
    zpath = OUT / f"{ds_id}.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    subj_map, pers_map = {}, {}
    hits = {}
    n_rows = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        with z.open(f"{ds_id}/{ds_id}.csv", "w", force_zip64=True) as raw:
            fh = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            w = csv.writer(fh, lineterminator="\r\n")
            w.writerow(names)
            core_names = {c[1] for c in (cols if module == "reach90" else CORE)}
            for r in _iter_rows(rows, names):
                unknown = set(r) - set(names)
                missing_core = core_names - set(r) - {"person"}
                assert not unknown and not missing_core, (sorted(unknown), sorted(missing_core))
                if len(r) < len(names):
                    r = {**{c: "" for c in names}, **r}
                s_old = r["subject"]
                p_old = r.get("person", s_old)
                if _is_missing(p_old) or p_old == "":
                    p_old = s_old
                s_new = subj_map.setdefault(s_old, f"{ds_id}_S{len(subj_map) + 1:04d}")
                p_new = pers_map.setdefault(p_old, f"{ds_id}_P{len(pers_map) + 1:04d}")
                vals = []
                for c in names:
                    if c == "subject":
                        v = s_new
                    elif c == "person":
                        v = p_new
                    else:
                        v = _fmt(r[c], ctype[c])
                        if c not in ("dataset", "study") and v and v not in ("True", "False") and not _PLAIN_NUMBER.fullmatch(v):
                            for pn, pat in ID_PATTERNS.items():
                                if pat.search(v):
                                    hits.setdefault(pn, set()).add(c)
                    vals.append(v)
                w.writerow(vals)
                n_rows += 1
            fh.flush()
            fh.detach()
        assert not hits, f"identifier-like text in {hits}"
        assert n_rows > 0
        cbuf = io.StringIO()
        cw = csv.writer(cbuf, lineterminator="\r\n")
        cw.writerow(["section", "column", "type", "values_or_units", "description"])
        for c in cols:
            cw.writerow(c)
        z.writestr(f"{ds_id}/columns.csv", cbuf.getvalue())
        if participants is not None:
            pc = [("person", "text", "", "Participant code, as in the main table.")] + list(participant_columns or [])
            pb = io.StringIO()
            pw = csv.writer(pb, lineterminator="\r\n")
            pw.writerow([c[0] for c in pc])
            for pr in (participants.to_dict("records") if hasattr(participants, "to_dict") else participants):
                assert pr["person"] in pers_map, "participants row for a person with no data rows"
                vals = [pers_map[pr["person"]]] + [_fmt(pr.get(c[0]), c[1]) for c in pc[1:]]
                for v in vals[1:]:
                    for pn, pat in ID_PATTERNS.items():
                        assert not pat.search(v), f"identifier-like text in participants.csv ({pn})"
                pw.writerow(vals)
            z.writestr(f"{ds_id}/participants.csv", pb.getvalue())
            qb = io.StringIO()
            qw = csv.writer(qb, lineterminator="\r\n")
            qw.writerow(["column", "type", "values_or_units", "description"])
            for c in pc:
                qw.writerow(c)
            z.writestr(f"{ds_id}/participants_columns.csv", qb.getvalue())
        p = meta["paper"]
        readme = [f"OpenMotor dataset {ds_id}", "", "Paper",
                  f"  {p['authors']} ({p['year']}). {p['title']}. {p['venue']}.",
                  f"  https://doi.org/{p['doi']}" if p.get("doi") else "  (no DOI)", "",
                  f"Lab: {meta.get('lab', '')}", f"Original data deposit: {meta['data_url']}", "", "Files",
                  f"  {ds_id}.csv     one row per {meta['unit_note']}, in the OpenMotor format ({len(names)} columns)",
                  "  columns.csv   what each column means"]
        if participants is not None:
            readme += ["  participants.csv   one row per participant (participants_columns.csv says what each column means)"]
        readme += ["", "License"]
        if str(meta["license"]).lower() in ("none stated", "not stated", ""):
            readme += ["  The authors posted these data without a license. This converted version is shared with credit to the",
                       "  authors and will be removed on request (open an issue at the OpenMotor GitHub repository)."]
        else:
            readme += [f"  The original data are licensed {meta['license']} by their authors ({meta.get('license_url', '')}).",
                       "  This converted version is shared under the same license."]
        readme += [f"  Changes made: {meta['changes']}", "", "How this file was made"]
        readme += ["  " + s for s in meta["how"]]
        readme += ["  Participant codes were assigned in order of first appearance: subject is one sequence of rows, person",
                   "  is one participant."]
        pub = [c for c in checks.items if c["kind"] == "published"]
        if no_published_reason:
            readme += ["  " + no_published_reason]
        else:
            readme += ["", "Checks against the paper",
                       f"  The converted rows reproduce {len(pub)} number{'s' if len(pub) != 1 else ''} reported in the paper, for example:"]
            for c in pub[:8]:
                readme += [f"  - {c['what']}: {c['computed']:.6g} here, {c['reported']:.6g} in the paper ({c['source']})"]
            if len(pub) > 8:
                readme += [f"  ... and {len(pub) - 8} more."]
        if meta.get("differences"):
            readme += ["", "Known differences from the paper"] + ["  - " + d for d in meta["differences"]]
        readme += ["", "Please cite the paper above when you use these data."]
        z.writestr(f"{ds_id}/README.txt", "\r\n".join(readme) + "\r\n")
        if meta.get("license_text"):
            z.writestr(f"{ds_id}/LICENSE.txt", meta["license_text"])
        for arc, content in (extra_files or []):
            assert not arc.startswith("/") and ".." not in arc, arc
            if isinstance(content, (bytes, str)) and not (isinstance(content, str) and Path(content).exists()):
                z.writestr(f"{ds_id}/{arc}", content)
            else:
                z.write(content, f"{ds_id}/{arc}")
    tmp.replace(zpath)
    old_skip = REC / f"{ds_id}.skip.json"
    if old_skip.exists():
        old_skip.unlink()
    rec = dict(id=ds_id, module=module, rows=n_rows, subjects=len(subj_map), persons=len(pers_map), zip_bytes=zpath.stat().st_size,
               zip_sha256=sha256(zpath), checks=checks.items, no_published_reason=no_published_reason,
               differences=meta.get("differences", []),
               sources=json.load(open(raw_dir(ds_id) / "SOURCES.json")) if (raw_dir(ds_id) / "SOURCES.json").exists() else [])
    json.dump(rec, open(REC / f"{ds_id}.json", "w"), indent=1)
    print(f"{ds_id}: {n_rows} rows, {len(subj_map)} subjects, {len(pers_map)} persons, {zpath.stat().st_size} bytes; "
          f"published checks {len(pub)}, all checks ok {all(c['ok'] for c in checks.items)}")
    return rec
