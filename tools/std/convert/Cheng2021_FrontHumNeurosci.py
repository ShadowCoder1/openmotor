"""Cheng, Niziolek, Buchwald & McAllister (2021), Front Hum Neurosci 15:660948: speech perception, production distinctness
and production variability (centering) in 37 female speakers.

Raw files, raw_extra/Cheng2021_FrontHumNeurosci/ (OSF t7ry2, fetched with tools/std/fetch.py; no license stated):
  centering_table_all_data_eh_{9,10}.csv, ..._ae_{9,10,11,12}.csv   one row per production of 'head' (EH) or 'had' (AE) in the
      baseline probe: subj, vowel, stage, trialdists, inidists, middists, loginidists, logmiddists, f0s, f0s_rev, f1s_mid, f2s_mid
  VAP_production.csv           productions of 'head' in the authors' formant-adaptation task (stages Pre, Intro, Ramp, Hold, Post)
  VAP_acuity_results_all.csv   perceptual acuity and bias per participant and sound (perception)
  stat_analysis.Rmd            the authors' analysis code (read as text)
What the values are (paper, Methods "Baseline Probe", "Formant Tracking", "Centering Ratio"; the Rmd):
  each participant said 'head' and 'had' 45 times in the baseline probe; six /e/ and 11 /ae/ tokens were removed (tracking);
  f1s_mid, f2s_mid  formants over the middle 50% of the vowel, in mels (the Rmd computes ellipse areas from them in mels);
  inidists, middists  Euclidean distance (mel, F1-F2 plane) from the participant's median production of that vowel, over the
      first 50 ms and over the middle 50% of the vowel; trialdists = inidists - middists (the centering of the production).
Run from the site folder: python3 -I tools/std/convert/Cheng2021_FrontHumNeurosci.py
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cheng2021_FrontHumNeurosci"
SRC = "Cheng, Niziolek, Buchwald & McAllister 2021, Front Hum Neurosci"
WORD = {"EH": ("head", "vowel /ɛ/"), "AE": ("had", "vowel /æ/")}
WINDOW = ("f1_mel, f2_mel and distance_mid_mel: middle 50% of the vowel duration; distance_initial_mel: first 50 ms of the vowel")
PER_WORD = 45  # productions of each word per participant (paper, Baseline Probe)


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def table_csv(df, cols):
    """The authors' table as shared (numbers formatted like the main table), for the extra files."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow([c for c, _ in cols])
    for rec in df.to_dict("records"):
        w.writerow([stdlib._fmt(rec[src], "number") if isinstance(rec[src], (float, np.floating, int, np.integer)) else
                    ("" if pd.isna(rec[src]) else str(rec[src])) for _, src in cols])
    return buf.getvalue().encode()


def main():
    raw = stdlib.raw_dir(ID)
    ck = stdlib.Checks()
    parts = []
    for vowel, files in (("EH", (9, 10)), ("AE", (9, 10, 11, 12))):
        for i in files:
            d = pd.read_csv(raw / f"centering_table_all_data_{vowel.lower()}_{i}.csv")
            parts.append(d.assign(src_file=f"{vowel}_{i}"))
    c = pd.concat(parts, ignore_index=True)
    c["trial"] = c.groupby(["subj", "vowel"]).cumcount() + 1  # the authors' trialnum: running count within participant and vowel
    pcode = {}
    rows = []
    for r in c.itertuples(index=False):
        word, seg = WORD[r.vowel]
        person = pcode.setdefault(r.subj, f"{ID}_P{len(pcode) + 1:04d}")  # the codes release() assigns (first appearance)
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Baseline probe", "group": "", "subject": f"{r.subj}|{word}",
            "person": r.subj, "subject_excluded_by_authors": False, "age": "", "sex": "F", "handedness": "",
            "population": "healthy", "setting": "", "device": "", "session": 1, "block": r.stage, "trial": int(r.trial),
            "phase": "baseline" if r.stage == "Pre" else "other", "time_s": "", "task": "speech", "unit": "utterance",
            "condition": r.vowel, "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal",
            "excluded_by_authors": False,
            "stimulus": word, "segment": seg, "measure_window": WINDOW,
            "f1_mel": float(r.f1s_mid), "f2_mel": float(r.f2s_mid),
            "distance_initial_mel": float(r.inidists), "distance_mid_mel": float(r.middists), "centering_mel": float(r.trialdists),
            "shift_f1": 0.0, "shift_f2": 0.0,
        })

    # ---- the adaptation-task and acuity tables, as shared, with participant codes
    v = pd.read_csv(raw / "VAP_production.csv", na_values=["--undefined--"])
    a = pd.read_csv(raw / "VAP_acuity_results_all.csv")
    v["person"] = v.subj.map(pcode)
    for s_ in a.SUBJ:  # 6 participants of the acuity file have no production data: codes after the 37 of the main table
        pcode.setdefault(s_, f"{ID}_P{len(pcode) + 1:04d}")
    a["person"] = a.SUBJ.map(pcode)
    vcols = [("person", "person"), ("stage", "stage"), ("phase_label", "Phase"), ("count", "Count"), ("condition", "Condition"),
             ("word", "Word"), ("textgrid_label", "TextGridLabel"), ("duration_s", "duration"), ("f0_midpoint", "f0_midpoint"),
             ("F1_midpoint", "F1_midpoint"), ("F2_midpoint", "F2_midpoint"), ("F3_midpoint", "F3_midpoint"),
             ("pre_mean", "pre_mean"), ("pre_sd", "pre_sd"), ("F1_norm", "F1_norm"), ("F1_target", "F1_target"),
             ("F1_diff", "F1_diff"), ("mean_diff", "mean_diff")]
    acols = [("person", "person"), ("sound", "Sound"), ("PT", "PT"), ("acuity", "ACUITY"), ("bias", "BIAS")]
    for df_, cols in ((v, vcols), (a, acols)):
        for _, src in cols:
            if df_[src].dtype == object and src not in ("person",):
                df_[src] = df_[src].where(~df_[src].astype(str).str.contains("undefined"), np.nan)
    for src in ("f0_midpoint", "F1_midpoint", "F2_midpoint", "F3_midpoint"):
        v[src] = pd.to_numeric(v[src], errors="coerce")
    extra = [("adaptation_task_tokens.csv", table_csv(v, vcols)), ("acuity_results.csv", table_csv(a, acols))]

    # ---- internal checks
    eh, ae = c[c.vowel == "EH"], c[c.vowel == "AE"]
    ck.internal("both vowels come from the same 37 participants", eh.subj.nunique() == ae.subj.nunique() == 37
                and set(eh.subj) == set(ae.subj), f"{eh.subj.nunique()} / {ae.subj.nunique()}")
    ck.internal("every row is a baseline-probe production (stage Pre)", bool((c.stage == "Pre").all()), str(c.stage.unique()))
    ck.internal("trialdists = inidists - middists on every row (the authors' centering)",
                bool(np.allclose(c.trialdists, c.inidists - c.middists, atol=1e-5)))
    per = c.groupby(["subj", "vowel"]).size()
    ck.internal(f"at most {PER_WORD} productions per participant and word", bool((per <= PER_WORD).all()), f"max {per.max()}")
    ck.internal("no missing values in the formant, distance and centering columns",
                bool(c[["f1s_mid", "f2s_mid", "inidists", "middists", "trialdists"]].notna().all().all()))
    ck.internal("formants within mel ranges for /e/ and /ae/ (F1 400-1200, F2 1000-2000)",
                bool(c.f1s_mid.between(400, 1200).all() and c.f2s_mid.between(1000, 2000).all()),
                f"F1 {c.f1s_mid.min():.0f}-{c.f1s_mid.max():.0f}, F2 {c.f2s_mid.min():.0f}-{c.f2s_mid.max():.0f}")
    ck.internal("adaptation-task table holds only participants of the main table; the acuity table has 43 participants, 6 with no "
                "production data", bool(v.person.notna().all() and a.SUBJ.nunique() == 43 and len(set(a.SUBJ) - set(c.subj)) == 6),
                f"{len(v)} / {len(a)} rows")
    ck.internal("adaptation-task table: 37 participants, stages Pre, Intro, Ramp, Hold, Post",
                v.subj.nunique() == 37 and set(v.stage) == {"Pre", "Intro", "Ramp", "Hold", "Post"}, str(len(v)))

    # ---- published numbers (Methods: Participants; Formant Tracking)
    ck.published("participants with baseline-probe productions", c.subj.nunique(), 37, 0, f"{SRC}, Methods, Participants")
    ck.published("/e/ ('head') tokens removed by the authors (45 x 37 minus rows shared)", PER_WORD * 37 - len(eh), 6, 0,
                 f"{SRC}, Methods, Formant Tracking")
    ck.published("/ae/ ('had') tokens removed by the authors (45 x 37 minus rows shared)", PER_WORD * 37 - len(ae), 11, 0,
                 f"{SRC}, Methods, Formant Tracking")

    for x in ck.items:
        if not x["ok"]:
            print("FAILED:", x)
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Re-encoded the authors' baseline-probe centering tables (centering_table_all_data_eh_*.csv, _ae_*.csv) with one "
                 "row per row, added the authors' adaptation-task tokens (VAP_production.csv) and acuity results "
                 "(VAP_acuity_results_all.csv) as extra tables, and replaced participant identifiers with codes."),
        how=[
            "One row is one production of 'head' (vowel /ɛ/) or 'had' (/æ/) in the baseline probe before the authors' "
            "adaptation task, in the order of the authors' centering tables (all 'head' rows, then all 'had' rows); each of the "
            "37 female participants said each word 45 times and the authors removed 6 'head' and 11 'had' tokens for tracking "
            "problems. trial counts the rows within one participant and word (the order across the two words is not shared), so "
            "subject is one participant's sequence for one word and person is the participant.",
            "f1_mel and f2_mel are the formants over the middle 50% of the vowel, in mels, as shared (f1s_mid, f2s_mid). "
            "distance_initial_mel, distance_mid_mel and centering_mel are the authors' inidists, middists and trialdists: the "
            "distance in mels from the participant's median production of that vowel in the F1-F2 plane, over the first 50 ms "
            "and over the middle 50% of the vowel, and their difference (the centering of that production).",
            "The authors' f0 columns (f0s, f0s_rev) are not described in the deposit or the paper and were left out; no feedback "
            "was altered in this probe.",
            "adaptation_task_tokens.csv is the authors' VAP_production.csv as shared (productions of 'head' in stages Pre, "
            "Intro, Ramp, Hold and Post of their formant-adaptation task; participant column replaced by the codes, row "
            "numbers and recording-file names dropped); that task is described in the authors' earlier paper (Klaus et al. "
            "2019), not in this deposit, so it is not in the main table. acuity_results.csv is VAP_acuity_results_all.csv "
            "(perceptual acuity and bias per participant and presentation; 43 participants, the 6 who are not in the main table have codes after "
            "the 37 of the main table).",
        ],
        unit_note="spoken word (utterance) of one participant in the baseline probe",
        differences=[
            "The paper's Table 1 (n = 31) leaves out participants flagged as outliers on several measures; the shared tables hold "
            "all 37 participants, so the centering ratio and the other Table 1 values were not recomputed here (the token and "
            "participant counts of the Methods were)."],
    )
    return stdlib.release(ID, "speech", rows, meta, ck, extra_files=extra)


if __name__ == "__main__":
    main()
