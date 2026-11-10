"""Hickman, Fraser, Galea, Happe & Cook (2026), J Neurodev Disord 18:8: shapes tracing with a stylus on a tablet at home, by 31 autistic
adults, 32 adults with Parkinson's disease (OFF and ON medication on separate days) and 31 control adults.

Raw files, raw_extra/Hickman2026_JNeurodevDisord/ (OSF 5g7ft, no license stated):
  TabletStudyData.zip   188 PsychoPy/Pavlovia result files (ASD_Day1, ASD_Day2, CTRL_day1, CTRL_day2, PD_OFF, PD_ON; one file per
                        participant and session). One row per attempt at tracing a shape; mouse.x, mouse.y, mouse.leftButton and
                        mouse.time are lists of samples; escapePositive is the attempt outcome (the authors' script Script1func.m comments
                        "0 fail, 1 success, 2 timeout but also success?" and rejects attempts with 0); a row without trial_shape_ID closes
                        a block (blocks.thisN, shape_id). Positions are PsychoPy height units; the authors' scale is 1200 (pixels).
  MetaData_ALL.csv      questionnaire and demographic values per participant (Group 1 = PD, 2 = control, 3 = autism: from the ID ranges of
                        the tablet files; Gender 1 = M, 2 = F, 3 = other: from Table 1 counts)
Not converted: the 2 GB .mat files (the authors' processed kinematics), Reaction Time Task data (not in the tablet zip).
Run from the site folder: python3 -I tools/std/convert/Hickman2026_JNeurodevDisord.py
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hickman2026_JNeurodevDisord"
SRC = "Hickman et al. 2026, J Neurodev Disord 18:8"
RAW = stdlib.raw_dir(ID)
FOLDERS = {"ASD_Day1": ("autism", 1, ""), "ASD_Day2": ("autism", 2, ""), "CTRL_day1": ("control", 1, ""), "CTRL_day2": ("control", 2, ""),
           "PD_OFF": ("Parkinson's disease", None, "OFF"), "PD_ON": ("Parkinson's disease", None, "ON")}
LAB = {"ASD_Day1": "d1", "ASD_Day2": "d2", "CTRL_day1": "d1", "CTRL_day2": "d2", "PD_OFF": "off", "PD_ON": "on"}
POP = {"autism": "autism", "control": "healthy", "Parkinson's disease": "Parkinson's disease"}
SHAPE = {"images/angfreq_4.png": "rounded square (angular frequency 4)", "images/angfreq_4to3.png": "petals (angular frequency 4/3)",
         "images/angfreq_2.png": "ellipse (angular frequency 2)", "images/angfreq_4to5.png": "clover (angular frequency 4/5)"}
GROUPNUM = {1: "Parkinson's disease", 2: "control", 3: "autism"}
SEX = {1: "M", 2: "F", 3: "other"}


def lst(s):
    s = str(s).strip()
    if s in ("", "nan"):
        return np.array([])
    return np.fromstring(s.strip("[]"), sep=",")


def fmt(a):
    return ["" if v != v else format(float(v), ".8g") for v in a]


PCOLS = [
    ("non_verbal_reasoning", "number", "proportion correct", "Matrix Reasoning Item Bank score: proportion of correct responses within 8 min (column NVR)."),
    ("autism_quotient", "number", "points", "Autism Quotient total (column AQ)."),
    ("raads", "number", "points", "Ritvo Autism Asperger Diagnostic Scale total (column RAADS)."),
    ("phq_depression", "number", "points", "Patient Health Questionnaire score (column PHQ)."),
    ("gad_anxiety", "number", "points", "Generalised Anxiety Disorder assessment score (column GAD)."),
    ("updrs_part2", "number", "points", "Parkinsonian traits: MDS-UPDRS Part II (motor experiences of daily living) questionnaire score (column UPDRS)."),
    ("age_at_diagnosis", "number", "years", "Age at clinical diagnosis (column Age_of_diagnosis); empty for controls."),
    ("years_since_diagnosis", "number", "years", "Years since diagnosis (column Years_since_diagnosis); empty for controls."),
    ("levodopa_equivalent_mg", "number", "mg", "Levodopa equivalent daily dose (column ldopa_equiv); empty where not given."),
]


def main():
    checks = stdlib.Checks()
    meta_df = pd.read_csv(RAW / "MetaData_ALL.csv", encoding="utf-8-sig").set_index("Ppt_ID")
    z = zipfile.ZipFile(RAW / "TabletStudyData.zip")
    files = {}
    for n in z.namelist():
        m = re.match(r"Tablet Study Data/(\w+)/(\d+)_shapes_.*\.csv$", n)
        if m:
            files.setdefault(int(m.group(2)), {})[m.group(1)] = n
    pids = sorted(files)
    pidx = {p: i + 1 for i, p in enumerate(pids)}
    rows, out_files = [], []
    mism = []
    nblocks, att_per_block, succ_per_block, sess_consistent = [], [], [], True
    for p in pids:
        md = meta_df.loc[p]
        grp = GROUPNUM[int(md.Group)]
        for folder in sorted(files[p], key=lambda k: list(FOLDERS).index(k)):
            gname, ses, med = FOLDERS[folder]
            assert gname == grp, (p, folder, grp)
            d = pd.read_csv(io.BytesIO(z.read(files[p][folder])))
            assert d.participant.nunique() == 1
            if int(d.participant.iloc[0]) != p:  # one participant's two PD files carry a different number typed into the task; the file name and metadata agree
                mism.append(p)
            if med:
                ses = int(md["Day_Shapes_OFF"] if med == "OFF" else md["Day_Shapes_ON"])
            if not med:
                sess_consistent &= bool((d.session.dropna().astype(int) == ses).all())
            code = f"S{pidx[p]:02d}{LAB[folder]}"
            fr = float(d.frameRate.dropna().iloc[0]) if d.frameRate.notna().any() else None
            att = d[d["trial_shape_ID"].notna()]
            blk_rows = d[d["blocks.thisN"].notna()]
            # block of each attempt = the next block-closing row
            block_of, cur = {}, []
            for idx in d.index:
                if pd.notna(d.at[idx, "trial_shape_ID"]):
                    cur.append(idx)
                elif pd.notna(d.at[idx, "blocks.thisN"]):
                    for c in cur:
                        block_of[c] = int(d.at[idx, "blocks.thisN"]) + 1
                    cur = []
            for c in cur:
                block_of[c] = None
            nblocks.append(len(blk_rows))
            k = 0
            counts = {}
            for idx in att.index:
                r = d.loc[idx]
                x, y, tm, lb = lst(r["mouse.x"]), lst(r["mouse.y"]), lst(r["mouse.time"]), lst(r["mouse.leftButton"])
                assert len(x) == len(y) == len(tm) == len(lb), (p, folder, idx, len(x), len(y), len(tm), len(lb))
                k += 1
                esc = int(r["escapePositive"]) if pd.notna(r["escapePositive"]) else None
                b = block_of[idx]
                counts.setdefault(b, []).append(esc)
                fn = f"samples/{code}_a{k:02d}.csv"
                body = "time_s,x_height_units,y_height_units,left_button_down\r\n" + "".join(
                    f"{a},{bb},{c},{dd}\r\n" for a, bb, c, dd in zip(fmt(tm), fmt(x), fmt(y), fmt(lb)))
                out_files.append((fn, body.encode()))
                shape = SHAPE[r["trial_shape_ID"]]
                rows.append(dict(
                    dataset=ID, study=ID, experiment="", group=gname, subject=code, subject_excluded_by_authors=False,
                    age=float(md.Age), sex=SEX[int(md.Gender)], handedness="", population=POP[gname], setting="home",
                    device="stylus on a Samsung Galaxy Tab A7 touch screen (PsychoPy on Pavlovia)",
                    session=ses, block=f"block {b}" if b else "", trial=k, phase="test", time_s="", task="tracking", unit="recording",
                    condition=shape + (f", {med} medication" if med else ""),
                    pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=(esc == 0),
                    recording=k, samples_file=fn, signal_kind="stylus position on a tablet while tracing a shape counter-clockwise for 10 cycles",
                    channels="x_height_units;y_height_units;left_button_down",
                    channel_units="PsychoPy height units (1 = screen height, x1200 gives the authors' pixels);PsychoPy height units;0 or 1",
                    sample_rate_hz="", n_samples=len(x), duration_s=float(tm[-1]) if len(tm) else "", events="",
                    processing="none (raw samples as logged by PsychoJS)",
                    trial_info=f"shape_image={r['trial_shape_ID'].split('/')[-1]};outcome_code={esc};attempt_in_block={int(r['shape_reps.thisN']) + 1}"
                               + (f";medication_state={med}" if med else "")))
            att_per_block += [len(v) for v in counts.values()]
            succ_per_block += [sum(1 for e in v if e in (1, 2)) for v in counts.values()]
    # participants.csv: one row per person
    parts = []
    for p in pids:
        md = meta_df.loc[p]
        parts.append({"person": f"S{pidx[p]:02d}", "non_verbal_reasoning": md.NVR, "autism_quotient": md.AQ, "raads": md.RAADS,
                      "phq_depression": md.PHQ, "gad_anxiety": md.GAD, "updrs_part2": md.UPDRS, "age_at_diagnosis": md.Age_of_diagnosis,
                      "years_since_diagnosis": md.Years_since_diagnosis, "levodopa_equivalent_mg": md.ldopa_equiv})
    # fix person codes: rows carry subject S01d1 etc.; stdlib needs a person per row
    for r in rows:
        r["person"] = r["subject"][:3]
    # participants rows must use the converter's person code (= S01...)
    T = meta_df.loc[pids]
    grp_names = T.Group.map(GROUPNUM)
    P = f"{SRC}, Methods and Table 1"
    for g, n in [("autism", 31), ("Parkinson's disease", 32), ("control", 31)]:
        checks.published(f"participants, {g}", (grp_names == g).sum(), n, 0, P)
    checks.published("participants in all", len(pids), 94, 0, f"{SRC}, Table 1 (F(2, 91) tests: 94 participants)")
    for g, mn, sd in [("autism", 55.52, 8.01), ("Parkinson's disease", 63.16, 7.60), ("control", 59.00, 9.96)]:
        s = T[grp_names == g]
        checks.published(f"mean age, {g}", s.Age.mean(), mn, 0.005, f"{SRC}, Table 1")
        checks.published(f"SD of age, {g}", s.Age.std(ddof=1), sd, 0.005, f"{SRC}, Table 1")
    for g, mm_, ff, oo in [("autism", 14, 16, 1), ("Parkinson's disease", 19, 13, 0), ("control", 15, 16, 0)]:
        s = T[grp_names == g]
        checks.published(f"males, {g}", (s.Gender == 1).sum(), mm_, 0, f"{SRC}, Table 1")
        checks.published(f"females, {g}", (s.Gender == 2).sum(), ff, 0, f"{SRC}, Table 1")
        checks.published(f"other gender, {g}", (s.Gender == 3).sum(), oo, 0, f"{SRC}, Table 1")
    for col, nm, vals, tol in [("NVR", "non-verbal reasoning", {"autism": 0.57, "Parkinson's disease": 0.55, "control": 0.58}, 0.015),
                               ("PHQ", "depression", {"autism": 8.2, "Parkinson's disease": 7.87, "control": 3.0}, 0.05),
                               ("GAD", "anxiety", {"autism": 9.9, "Parkinson's disease": 5.1, "control": 3.6}, 0.05),
                               ("UPDRS", "UPDRS", {"autism": 3.7, "Parkinson's disease": 11.91, "control": 0.94}, 0.05)]:
        for g, v in vals.items():
            if col == "NVR":
                continue
            checks.published(f"mean {nm}, {g}", T[grp_names == g][col].mean(), v, 0.005 if round(v, 2) != round(v, 1) else tol, f"{SRC}, Table 1")
    checks.published("mean years since diagnosis, autism", T[grp_names == "autism"].Years_since_diagnosis.mean(), 4.23, 0.005, f"{SRC}, Table 1")
    checks.published("mean years since diagnosis, PD", T[grp_names == "Parkinson's disease"].Years_since_diagnosis.mean(), 3.69, 0.005, f"{SRC}, Table 1")
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    print("blocks per file", set(nblocks), "attempts per block max", max(att_per_block), "successes per block max", max(succ_per_block), "session consistent", sess_consistent)
    checks.internal("188 sessions of 94 participants, 8 blocks each", len(set(r["subject"] for r in rows)) == 188 and set(nblocks) == {8}, str(set(nblocks)))
    checks.internal("at most 7 attempts and 4 successful attempts in a block", max(att_per_block) <= 7 and max(succ_per_block) <= 4)
    print("participant number inside the file differs from file name for", len(mism), "files")
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    pp = s["papers"][0]
    paper = {k: pp[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("One row per attempt at tracing a shape; the attempt's lists of x, y, button and time samples were written to "
                 "samples/<subject>_a<N>.csv unchanged, the shape, block, outcome code and (for Parkinson's disease) medication state were "
                 "read from the same result file, and age, gender and questionnaire scores came from MetaData_ALL.csv. Participant "
                 "numbers and the dates in the file names were not carried. The authors' processed kinematic .mat files and the "
                 "reaction time task were not converted."),
        how=[
            "One row is one attempt at the Shapes Tracing Task: the participant traced a shape on a tablet with a stylus, counter-clockwise, for 10 full "
            "cycles (90 s timeout), at home. Four shapes (rounded square, petals, ellipse, clover) were each traced in two blocks, eight blocks in random "
            "order; a block ended after four successful trials or seven attempts. Trial counts the attempts within a session.",
            "The sample files hold the logged samples: time_s (from the start of the attempt), x and y in PsychoPy height units (1 = screen height; the "
            "authors' scripts multiply by 1200 to get pixels in the central square of the 2000 x 1200 screen) and the left-button flag (1 while the stylus is "
            "on the screen). The sampling is irregular (the device's maximum is 60 Hz), so sample_rate_hz is empty.",
            "Group is autism, Parkinson's disease or control. People with Parkinson's disease did the task on two days, OFF and ON dopaminergic medication "
            "(order counterbalanced); the paper analyses only OFF. Autistic and control participants did it on two days (session 1 and 2, from the folder names; the session field inside the files is not used). The outcome code "
            "(trial_info outcome_code) is escapePositive; the authors' script reads 0 as a failed attempt and rejects it (excluded_by_authors is True for "
            "these), 1 as success and 2 as a timeout that is also kept. One participant's two Parkinson's disease files carry a different participant number inside the file than in the file name and metadata; the file name and metadata are followed. The authors also dropped trials with fewer than 2.5 traces of the shape; that rule "
            "is not applied here. Age, gender and questionnaire scores are in the standard columns and participants.csv (the deposit gives no "
            "handedness code).",
        ],
        unit_note="attempt at tracing a shape (samples in samples/)",
        differences=[
            "The group coding of MetaData_ALL.csv (1 = Parkinson's disease, 2 = control, 3 = autism; gender 1 = male, 2 = female, 3 = other) is not "
            "documented in the deposit; it was inferred from the participant number ranges of the tablet files and from the group counts in Table 1, "
            "and the age, gender and questionnaire means in Table 1 are reproduced with it.",
        ],
    )
    return stdlib.release(ID, "timeseries", rows, meta, checks, participants=parts, participant_columns=PCOLS, extra_files=out_files)


if __name__ == "__main__":
    main()
