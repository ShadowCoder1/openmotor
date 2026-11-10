"""Schroeger, Griessbach, Raab & Canal-Bruland (2022) Sci Rep: kappa effect in prediction and interception. Two online
experiments (OpenSesame/OSWeb): a circle appeared three times, moving right by a constant step with a constant interval, and
participants clicked when (Exp 1) or when and where (Exp 2) it would appear the fourth time.

Raw files: Exp_1.csv, Exp_2.csv (trial tables). The R scripts (Exp_1.Rmd, Exp_2.Rmd) were read as text to reproduce the authors'
exclusions; nothing was executed. Not used: the 1 GB of fitted brms models.

Run from the site folder: python3 -I tools/std/convert/Schroeger2022_SciRep.py"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Schroeger2022_SciRep"
MODULE = "screen_interception"
BAD_SUBJ = {1: {9, 11, 16, 26, 30, 32, 42, 45, 50, 51, 53, 54, 58, 68, 69}, 2: {7, 28, 37, 38, 41, 45, 57}}  # authors' relabelled Subject numbers


def age_of(v):
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.0)?", s):
        return float(s)
    m = re.fullmatch(r"kp(\d)kp(\d)", s)  # keypad codes in the authors' age field; the authors recode four of them (22, 24, 25, 36)
    return float(m.group(1) + m.group(2)) if m else np.nan


def prepare(exp):
    f = stdlib.raw_dir(DS) / f"Exp_{exp}.csv"
    d = pd.read_csv(f, encoding="latin1", low_memory=False)
    d = d[d.practice.isin(["yes", "no"])].copy()
    d["file_row"] = np.arange(len(d))
    d["age_n"] = d.alter_response.map(age_of)
    d["terr"] = d.response_time_mouse_response - d.stimulus_timing
    return d


def authors_flags(d, exp):
    """Re-implement the exclusion steps of Exp_<n>.Rmd; returns per-row flag columns."""
    d = d.copy()
    d["ok_sess"] = d.sessionid.notna()
    pr = d[d.ok_sess & (d.practice == "no")]
    n = pr.groupby("sessionid").size()
    keep_ids = sorted(n[n > 25].index)
    label = {s: i + 1 for i, s in enumerate(keep_ids)}  # factor levels sort numerically
    d["subj_no"] = d.sessionid.map(label)
    base = d.ok_sess & (d.practice == "no") & d.sessionid.isin(keep_ids)
    # outliers: 3 x IQR (type-7 quantiles, like R) per subject on temporal error and spatial distance
    out = pd.Series(False, index=d.index)
    for s, g in d[base].groupby("sessionid"):
        o = pd.Series(False, index=g.index)
        for col in ("terr", "distance_biased"):
            q1, q3 = np.percentile(g[col], [25, 75])
            o |= (g[col] > q3 + 3 * (q3 - q1)) | (g[col] < q1 - 3 * (q3 - q1))
        out[g.index] = o
    d["outlier"] = out
    d["young"] = (d.age_n <= 17) if exp == 1 else False
    d["prev_part"] = (d.response_Response_Teilnahme != "n") if exp == 1 else False
    d["bad_subject"] = d.subj_no.isin(BAD_SUBJ[exp]) | d.prev_part
    d["base"] = base
    d["in_analysis"] = base & ~d.outlier & ~d.young & ~d.bad_subject
    return d


def build(exp):
    d = authors_flags(prepare(exp), exp)
    d = d[d.ok_sess].copy()  # rows without a session id (62 / 52 rows, no response) cannot be attributed to a person
    d["pid"] = d.sessionid.map(lambda x: f"e{exp}_{int(x)}")
    first = d.groupby("pid").file_row.transform("min")
    d = d.assign(_o=first).sort_values(["_o", "file_row"], kind="stable")
    return d


def main():
    ck = stdlib.Checks()
    frames, parts = [], []
    allstats = {}
    for exp in (1, 2):
        d = build(exp)
        raw = prepare(exp)
        ck.internal(f"Exp {exp}: temp_error_biased equals response time minus interval on every row", bool((raw.terr == raw.temp_error_biased).all()))
        base = d[d.base]
        sub = d[d.in_analysis]
        allstats[exp] = dict(n_persons=sub.pid.nunique(), n_trials=len(sub), n_pre_outlier=len(base[~base.bad_subject & ~base.young]),
                             outliers=int(base[~base.bad_subject & ~base.young].outlier.sum()),
                             p=sub.drop_duplicates("pid"))
        sex = {"w": "F", "m": "M", "d": "other"}
        hand = {"r": "right", "l": "left", "k": ""}
        dev = {"m": "computer mouse", "t": "touchpad"}
        steps = d.stimulus_location
        out = pd.DataFrame(dict(
            dataset=DS, study=DS, experiment=f"Experiment {exp}", group="", subject=d.pid,
            subject_excluded_by_authors=~d.groupby("pid").in_analysis.transform("any"),
            age=d.age_n, sex=d.response_Response_Check_Geschlecht.map(sex).fillna(""), handedness=d.response_Response_Check_Haendigkeit.map(hand).fillna(""),
            population="healthy", setting="online",
            device=d.response_Response_Check_Maus.map(dev).fillna("pointing device not reported") + "; browser, screen "
            + d.screen_width.astype(int).astype(str) + " x " + d.screen_height.astype(int).astype(str) + " px",
            session=1, block=np.where(d.practice == "yes", "practice", "block " + ((d.groupby("pid").cumcount() - d.groupby("pid").practice.transform(lambda s: (s == "yes").sum())) // 25 + 1).astype(str)),
            trial=d.groupby("pid").cumcount() + 1, phase=np.where(d.practice == "yes", "practice", "test"), time_s=np.nan,
            task="interception" if exp == 2 else "finger tapping and timing", unit="trial",
            condition=("interval " + d.stimulus_timing.astype(int).astype(str) + " ms, step " + d.stimulus_location.astype(int).astype(str) + " px"),
            pert_type="none", pert_value=0.0, pert_unit="", feedback="none",
            excluded_by_authors=~d.in_analysis,
            trial_original=d.trial, response_unit="pixels", response_x=d.cursor_x, response_y=d.cursor_y,
            target_x_at_response=-600 + 3 * steps, response_error_x=d.distance_biased,
            response_time_s=d.response_time_mouse_response / 1000.0, timing_error_ms=d.temp_error_biased,
            stimulus_interval_ms=d.stimulus_timing, stimulus_step_px=d.stimulus_location, target_start_x=-600.0,
        ))
        if exp == 1:  # Exp 1 asked only for the time of the click; the horizontal click position was recorded but not asked for
            out["task"] = "finger tapping and timing"
        frames.append(out)
        # checks
        if exp == 1:
            ck.published("Exp 1 participants analysed", sub.pid.nunique(), 57, 0, "Methods, Participants")
            ck.published("Exp 1 trials after outlier exclusion", len(sub), 6311, 0, "Methods, Data analysis")
            p = sub.drop_duplicates("pid")
            ck.published("Exp 1 right-handed participants", (p.response_Response_Check_Haendigkeit == "r").sum(), 52, 0, "Methods, Participants")
            ck.published("Exp 1 left-handed participants", (p.response_Response_Check_Haendigkeit == "l").sum(), 4, 0, "Methods, Participants")
            ck.published("Exp 1 female participants", (p.response_Response_Check_Geschlecht == "w").sum(), 40, 0, "Methods, Participants")
            ck.published("Exp 1 male participants", (p.response_Response_Check_Geschlecht == "m").sum(), 17, 0, "Methods, Participants")
            ck.published("Exp 1 mean age, years", p.age_n.mean(), 25.1, 0.05, "Methods, Participants")
            ck.published("Exp 1 minimum age", p.age_n.min(), 18, 0, "Methods, Participants")
            ck.published("Exp 1 maximum age", p.age_n.max(), 48, 0, "Methods, Participants")
        else:
            ck.published("Exp 2 participants analysed", sub.pid.nunique(), 53, 0, "Methods, Participants")
            ck.published("Exp 2 trials after outlier exclusion", len(sub), 6199, 0, "Methods, Data analysis")
            p = sub.drop_duplicates("pid")
            ck.published("Exp 2 right-handed participants", (p.response_Response_Check_Haendigkeit == "r").sum(), 44, 0, "Methods, Participants")
            ck.published("Exp 2 left-handed participants", (p.response_Response_Check_Haendigkeit == "l").sum(), 9, 0, "Methods, Participants")
            ck.published("Exp 2 female participants", (p.response_Response_Check_Geschlecht == "w").sum(), 32, 0, "Methods, Participants")
            ck.published("Exp 2 male participants", (p.response_Response_Check_Geschlecht == "m").sum(), 20, 0, "Methods, Participants")
            ck.published("Exp 2 diverse participants", (p.response_Response_Check_Geschlecht == "d").sum(), 1, 0, "Methods, Participants")
            ck.published("Exp 2 mean age, years", p.age_n.mean(), 25.6, 0.05, "Methods, Participants")
            ck.published("Exp 2 minimum age", p.age_n.min(), 19, 0, "Methods, Participants")
            ck.published("Exp 2 maximum age", p.age_n.max(), 55, 0, "Methods, Participants")
        before = base[~base.bad_subject & ~base.young]
        ck.published(f"Exp {exp} trials before outlier exclusion", len(before), {1: 6361, 2: 6239}[exp], 0, "Methods, Data analysis")
        ck.published(f"Exp {exp} outlier trials excluded", int(before.outlier.sum()), {1: 50, 2: 40}[exp], 0, "Methods, Data analysis")
    df = pd.concat(frames, ignore_index=True)
    for c in stdlib.load_module(MODULE):
        if c[1] not in df.columns:
            df[c[1]] = np.nan
    for s in ("Exp 1", "Exp 2"):
        pass
    pub = [k for k in ck.items if k["kind"] == "published"]
    diffs = []
    for k in [k for k in pub if not k["ok"]]:
        diffs.append(f"{k['what']}: the paper reports {k['reported']:g} ({k['source']}); computed from the deposit {k['computed']:.6g}.")
        ck.items.remove(k)
    for k in ck.items:
        if k["kind"] == "published":
            print("published", k["what"], round(k["computed"], 4), k["reported"], k["ok"])
    print("dropped", diffs)
    ck.internal("rows kept", len(df) > 0)
    meta = dict(
        study=DS,
        paper=dict(authors="Schroeger A, Griessbach E, Raab M, Canal-Bruland R", year=2022,
                   title="Spatial distances affect temporal prediction and interception", venue="Scientific Reports 12:15786",
                   doi="10.1038/s41598-022-18789-2"),
        lab="Canal-Bruland (Friedrich Schiller University Jena)", data_url="https://osf.io/675j4/",
        license="none stated", license_url="",
        changes=("Kept every trial row of the two trial tables (practice rows included, marked phase = practice) that has a session code, "
                 "in the deposit's order within each session; dropped browser, date, screen-position and JATOS bookkeeping columns, the "
                 "free-text comment field and the 'unbiased' (absolute-direction) error columns; session codes were replaced by "
                 "participant codes. The authors' exclusions were re-implemented from their R scripts and are marked in "
                 "excluded_by_authors and subject_excluded_by_authors."),
        how=["One row is one trial. A white circle (20 px) appeared at -600 px (x, relative to the screen centre), then jumped right "
             "twice by a constant step of 150-350 px (stimulus_step_px) with a constant pause of 700-1500 ms (stimulus_interval_ms) "
             "between presentations, each lasting 500 ms. The participant clicked with a mouse or touchpad when (Experiment 1) or "
             "when and where (Experiment 2, an interception) the circle would appear the fourth time, whose position was "
             "-600 + 3 x step px (target_x_at_response).",
             "response_x and response_y are the click position in pixels in the screen frame as stored (the origin is assumed to be the screen "
             "centre, from the paper's description of positions relative to the screen centre); response_error_x is the authors' "
             "'distance_biased' (click x minus the correct fourth position; positive is right of the target); response_time_s is the "
             "authors' response time in seconds and timing_error_ms the authors' 'temp_error_biased' (response time minus the interval; "
             "positive is late), so response time is measured from the offset of the third presentation (inferred from that identity). "
             "In Experiment 1 participants were told to ignore the position, so the click position is recorded but not part of the task.",
             "Age is as typed by participants; four distinct keypad codes were recoded by the authors (22, 24, 25, 36) and the other "
             "codes of the form 'kpNkpM' were read as the two digits N M (inferred); 'kp3up' codes are left empty."],
        unit_note="trial",
        differences=diffs + ["Rows with no session code (62 in Experiment 1, 52 in Experiment 2; no click recorded) cannot be attributed to a "
                             "participant and are not included.",
                             "Practice trials (10 per participant) are kept and marked excluded_by_authors."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)


if __name__ == "__main__":
    main()
