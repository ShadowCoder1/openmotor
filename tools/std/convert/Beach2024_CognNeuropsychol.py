"""Beach & Niziolek (2024), Cognitive Neuropsychology 41(1-2):51-69: a vowel-modified Stroop task (original sample, 17 adults).

Raw files, raw_extra/Beach2024_CognNeuropsychol/ (OSF 489pf, statistics/; fetched with tools/std/fetch.py; no license stated):
  stroop_durations_BosVoice.csv   subj, color, word, cond, dur     one row per kept trial, in the authors' order; dur in seconds
  stroop_rts_BosVoice.csv         subj, color, word, cond, rt      the same rows in the same order; rt in seconds
  stroop_f0_mean_byColor_mid50p.csv, stroop_f0_var_byColor_mid50p.csv, stroop_fmt_var_byColor_mid50p.csv
                                  subj, cond, color, value         one row per kept trial, grouped by condition and colour (their
                                  order within a group is not the trial order of the files above, so they are not merged)
  stroop_ntrials_BosVoice.csv     subj, cond, ntrials, perctrials  trials kept per participant and condition
  stroop_stats.R                  the authors' analysis code (read as text)
What the codes are (paper, Methods "Stimuli and Procedure" and "Analysis"; checked against the design in the tables):
  cond   1 congruent (red/green/blue in its own ink; 40 trials per word), 2 vowel-modified (rid, rad in red; grin, grain in
         green; bleed, blow in blue; 40 per word), 3 colour-incongruent (the other two colour words in each ink; 20 per word)
  color  ink colour 1 red, 2 green, 3 blue; word 1 red, 4 green, 7 blue (the colour words); 2, 3 / 5, 6 / 8, 9 the vowel-modified
         words of red / green / blue ink, in the paper's order within each pair (which of the two is which is not stated)
  rt     voicing onset relative to stimulus onset; dur  voicing onset to offset of the spoken colour name
Run from the site folder: python3 -I tools/std/convert/Beach2024_CognNeuropsychol.py
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

ID = "Beach2024_CognNeuropsychol"
SRC = "Beach & Niziolek 2024, Cognitive Neuropsychology"
COND = {1: "congruent", 2: "vowel-modified", 3: "color-incongruent"}
INK = {1: "red", 2: "green", 3: "blue"}
WORD = {1: "red", 4: "green", 7: "blue", 2: "rid or rad (word code 2)", 3: "rid or rad (word code 3)",
        5: "grin or grain (word code 5)", 6: "grin or grain (word code 6)", 8: "bleed or blow (word code 8)",
        9: "bleed or blow (word code 9)"}
DEVICE = "AKG Pro Audio C520 head-worn condenser microphone (11025 Hz), MATLAB Psychtoolbox"
WINDOW = "duration_ms: voicing onset to offset of the spoken colour name; reaction_time_ms: stimulus onset to voicing onset"
DESIGN = {1: 120, 2: 240, 3: 120}  # trials per participant and condition (paper, Stimuli and Procedure)


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def to_csv(header, recs):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(header)
    for r in recs:
        w.writerow(r)
    return buf.getvalue().encode()


def main():
    raw = stdlib.raw_dir(ID)
    ck = stdlib.Checks()
    d = pd.read_csv(raw / "stroop_durations_BosVoice.csv")
    r = pd.read_csv(raw / "stroop_rts_BosVoice.csv")
    f0 = pd.read_csv(raw / "stroop_f0_mean_byColor_mid50p.csv")
    f0v = pd.read_csv(raw / "stroop_f0_var_byColor_mid50p.csv")
    fv = pd.read_csv(raw / "stroop_fmt_var_byColor_mid50p.csv")
    nt = pd.read_csv(raw / "stroop_ntrials_BosVoice.csv")
    key = ["subj", "color", "word", "cond"]
    aligned = len(d) == len(r) and bool((d[key].values == r[key].values).all())
    ck.internal("duration and reaction-time tables hold the same rows in the same order", aligned, f"{len(d)} rows")
    assert aligned
    d["trial"] = d.groupby("subj").cumcount() + 1  # running count of the kept trials within a participant
    pcode, rows = {}, []
    for x, rt in zip(d.itertuples(index=False), r.rt):
        pcode.setdefault(x.subj, f"{ID}_P{len(pcode) + 1:04d}")
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Original sample", "group": "", "subject": f"S{x.subj}", "person": f"S{x.subj}",
            "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEVICE, "session": 1, "block": "", "trial": int(x.trial), "phase": "test", "time_s": "",
            "task": "speech", "unit": "utterance", "condition": COND[x.cond], "pert_type": "none", "pert_value": 0.0,
            "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
            "stimulus": f"{WORD[x.word]} in {INK[x.color]} ink", "segment": "spoken colour name", "measure_window": WINDOW,
            "duration_ms": float(x.dur) * 1000.0, "reaction_time_ms": float(rt) * 1000.0, "shift_f1": 0.0, "shift_f2": 0.0,
        })
    # extra tables, as shared (participant column replaced by the codes)
    cells = f0[["subj", "cond", "color"]].values
    extra_acoustic = to_csv(
        ["person", "condition", "ink_color", "f0_mean", "f0_variability", "formant_variability"],
        [[pcode[s], COND[c], INK[k], stdlib._fmt(float(a), "number"), stdlib._fmt(float(b), "number"), stdlib._fmt(float(e), "number")]
         for (s, c, k), a, b, e in zip(cells, f0.value, f0v.value, fv.value)])
    extra_trials = to_csv(["person", "condition", "trials_kept", "share_kept"],
                          [[pcode[s], COND[c], int(n), stdlib._fmt(float(p), "number")]
                           for s, c, n, p in zip(nt.subj, nt.cond, nt.ntrials, nt.perctrials)])

    # ---- internal checks
    ck.internal("17 participants in every table", all(t.subj.nunique() == 17 for t in (d, f0, f0v, fv, nt)))
    ck.internal("the three acoustic tables hold the same participant, condition and colour on every row, in the same order",
                bool((f0[["subj", "cond", "color"]].values == f0v[["subj", "cond", "color"]].values).all()
                     and (f0[["subj", "cond", "color"]].values == fv[["subj", "cond", "color"]].values).all()))
    k1 = d.groupby(["subj", "cond", "color"]).size()
    k2 = f0.groupby(["subj", "cond", "color"]).size()
    ck.internal("the acoustic tables hold as many rows per participant, condition and colour as the trial table", k1.equals(k2),
                f"{len(k1)} cells")
    n_rows = d.groupby(["subj", "cond"]).size().rename("n").reset_index().merge(nt, on=["subj", "cond"], validate="one_to_one")
    ck.internal("trials per participant and condition in the trial table = ntrials of the authors' table",
                bool((n_rows.n == n_rows.ntrials).all()), f"{int(n_rows.ntrials.sum())} trials")
    ck.internal("share kept = ntrials / designed trials (120, 240, 120)",
                bool(np.allclose(nt.perctrials, nt.ntrials / nt.cond.map(DESIGN), atol=1e-6)))
    ink_words = d.groupby(["cond", "color"]).word.apply(lambda s: tuple(sorted(set(s)))).to_dict()
    exp_words = {(1, 1): (1,), (1, 2): (4,), (1, 3): (7,), (2, 1): (2, 3), (2, 2): (5, 6), (2, 3): (8, 9), (3, 1): (4, 7),
                 (3, 2): (1, 7), (3, 3): (1, 4)}
    ck.internal("words per condition and ink colour as in the design (own colour word; rid/rad, grin/grain, bleed/blow; the other "
                "two colour words)", ink_words == exp_words)
    ck.internal("durations 70-600 ms and reaction times 320-1500 ms",
                bool(d.dur.between(0.07, 0.6).all() and r.rt.between(0.32, 1.5).all()),
                f"dur {d.dur.min():.3f}-{d.dur.max():.3f}, rt {r.rt.min():.3f}-{r.rt.max():.3f}")

    # ---- published numbers (Results; original sample)
    ck.published("participants in the original sample", d.subj.nunique(), 17, 0, f"{SRC}, Methods, Participants")
    P = f"{SRC}, Results, The color-word interference effect"
    for c, (m, sd, lo, hi, pct) in {1: (119, 2, 113, 120, 99.26), 2: (238, 3, 228, 240, 99.12), 3: (110, 9, 87, 120, 91.52)}.items():
        x = nt[nt.cond == c]
        ck.published(f"trials kept per participant, {COND[c]}: mean (of {DESIGN[c]})", x.ntrials.mean(), m, 0.5, P)
        ck.published(f"trials kept per participant, {COND[c]}: SD", x.ntrials.std(ddof=1), sd, 0.5, P)
        ck.published(f"trials kept per participant, {COND[c]}: minimum", x.ntrials.min(), lo, 0, P)
        ck.published(f"trials kept per participant, {COND[c]}: maximum", x.ntrials.max(), hi, 0, P)
        ck.published(f"error-free trials, {COND[c]}: mean percentage", 100 * x.perctrials.mean(), pct, 0.01, P)
    Q = f"{SRC}, Methods, Analysis (Duration)"
    for k, (m, sd, share) in {1: (229, 63, 65), 2: (264, 76, 79), 3: (279, 84, 80)}.items():
        x = d.dur[d.color == k] * 1000
        ck.published(f"syllable duration, {INK[k]} ink: mean (ms)", x.mean(), m, 0.5, Q)
        ck.published(f"syllable duration, {INK[k]} ink: SD (ms)", x.std(ddof=1), sd, 0.5, Q)
        ck.published(f"trials with duration of at least 200 ms, {INK[k]} ink (%)", 100 * (x >= 200).mean(), share, 0.5, Q)
    rt = r.assign(rt_ms=r.rt * 1000)
    diff = rt.rt_ms[rt.cond == 3].mean() - rt.rt_ms[rt.cond == 1].mean()
    near = dict(rt_diff=diff)  # the paper's 121 ms is a model estimate; the raw difference of means agrees
    ck.published("reaction time, colour-incongruent minus congruent: raw difference of means (ms; the paper's is a model estimate)",
                 diff, 121, 3, f"{SRC}, Results, The color-word interference effect")

    for x in ck.items:
        if not x["ok"]:
            print("FAILED:", x)
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Merged the authors' trial-level duration and reaction-time tables into one table with condition and stimulus "
                 "labels from the paper's design, added their acoustic tables (f0 mean, f0 variability, formant variability) and "
                 "trials-kept table as extra files, and replaced participant numbers with codes."),
        how=[
            "One row is one kept trial of the colour-naming task in the order of the authors' table (spoken colour name of one "
            "of 17 adults; trials with errors are absent, so trial counts the rows kept per participant): the participant read "
            "a word shown in red, green or blue ink and said the ink colour; condition is congruent (own colour word), "
            "vowel-modified (rid, rad, grin, grain, bleed, blow) or colour-incongruent (another colour word).",
            "reaction_time_ms is the time from stimulus onset to voicing onset and duration_ms the voicing onset to offset of "
            "the response, both the authors' values converted from seconds. The shared tables give the vowel-modified word only "
            "as a number (2/3 rid or rad, 5/6 grin or grain, 8/9 bleed or blow, in the paper's order within each pair), which "
            "stimulus keeps as 'word code'.",
            "acoustic_by_condition_and_color.csv holds the authors' f0 mean, f0 variability and formant variability for the same "
            "17 participants (middle 50% of the response; one row per kept trial, grouped by participant, condition and ink "
            "colour; the order within a group is not the trial order of the main table, so they could not be joined row by row; "
            "units are not stated in the deposit). trials_kept.csv is their trials-kept table.",
            "Only the paper's original sample (17 adults) is in the shared statistics tables; the replication sample (18 adults) "
            "is not.",
        ],
        unit_note="kept trial (spoken colour name) of one participant in the Stroop task",
    )
    rec = stdlib.release(ID, "speech", rows, meta, ck, extra_files=[
        ("acoustic_by_condition_and_color.csv", extra_acoustic), ("trials_kept.csv", extra_trials)])
    print(near)
    return rec


if __name__ == "__main__":
    main()
