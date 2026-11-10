"""Karekal et al. (2025), eNeuro 12(10):ENEURO.0286-25.2025: continuous movement stop task (CMST) in people with Parkinson's disease
(OFF then ON medication) and healthy older adults (two sessions). Participants moved a computer mouse clockwise around an on-screen circle
while a countdown ran, and stopped when a STOP cue appeared (planned: after the countdown reached 1; unplanned: earlier). One row per trial.

Run from the site folder: python3 -I tools/std/convert/Karekal2025_eNeuro.py
Raw files (untrusted data, read only) in raw_extra/Karekal2025_eNeuro/ from https://osf.io/8vqu6/ (component h3yga, Behavioral_Data_All_Groups):
94 CSV files named <folder>__<HC|PD><number>.csv (the folder name is the authors' folder), and the authors' read-me PDF. The EEG (.bdf) files of
the other components are not used. The CSV layouts and their reader are described in Karekal2025_eNeuro_parse.py.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Karekal2025_eNeuro_parse import read_file  # noqa: E402

ID = "Karekal2025_eNeuro"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
STUDY = "Karekal et al. 2025, eNeuro"
DEVICE = "computer mouse (dominant hand), browser task (jsPsych) on a computer screen, EEG recorded in parallel"
FOLDERS = {  # folder -> (group, session, state)
    "HC_Session_1_OFF": ("healthy older adults", 1, "session 1 (no medication)"),
    "HC_Session_2_ON": ("healthy older adults", 2, "session 2 (no medication)"),
    "PD_OFF": ("Parkinson's disease", 1, "OFF medication"),
    "PD_ON": ("Parkinson's disease", 2, "ON medication"),
}


# the authors' read-me: trials can be excluded on the exclude column saying these three; the other texts (too far from the circle, remember: try to stop, no)
# are the training feedback messages and do not exclude a trial
EXCL = {"too slow", "wrong way", "remember: don't stop too early"}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


files = sorted(RAW.glob("*__*.csv"), key=lambda p: (list(FOLDERS).index(p.name.split("__")[0]), p.name))
assert len(files) == 94, len(files)
persons, tcount = {}, {}
rows, extra = [], []
status_n, mismatch = {}, 0
acc = {}
sex_of = {}
n_planned = {}
for p in files:
    folder, rest = p.name[:-4].split("__")
    grp, ses, state = FOLDERS[folder]
    key = rest  # HC01 / PD07: the number is the person within the group (the folder, not the label inside the file, decides the group)
    persons.setdefault(key, len(persons) + 1)
    pi = persons[key]
    code = f"{ID}_S{pi:04d}"
    for x in read_file(p):
        st = x["status"].split(":")[0]
        status_n[st] = status_n.get(st, 0) + 1
        if not x["pid"].startswith(rest[:2]):
            mismatch += 1
        tcount[key] = tcount.get(key, 0) + 1
        tn = tcount[key]
        fn = f"samples/{code}_trial{tn:04d}.csv"
        L = x["lists"]
        if L is None:
            df = pd.DataFrame({"time_s": [], "segment": [], "x": [], "y": []})
        else:
            gx, gy, gt, sx, sy, stt = L
            assert len(gx) == len(gy) == len(gt) and len(sx) == len(sy) == len(stt)
            df = pd.DataFrame({
                "time_s": np.array(list(gt) + list(stt), dtype=float) / 1000.0,
                "segment": [0] * len(gt) + [1] * len(stt),
                "x": list(gx) + list(sx), "y": list(gy) + list(sy)})
        extra.append((fn, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
        block = int(num(x["block"]))
        excl = x["exclude"].strip()
        analysed = (excl not in EXCL) and block > 0
        planned = x["signal"] == "go"
        if block > 0:
            a = acc.setdefault((grp, ses, key), [0, 0])
            a[1] += 1
            a[0] += int(excl not in EXCL)
            n_planned[(key, ses, x["signal"])] = n_planned.get((key, ses, x["signal"]), 0) + 1
        sex_of.setdefault(key, set()).add(x["gender"])
        info = [f"signal={x['signal']}", f"countdown_start={x['count']}"]
        if x["stop_time"] not in ("null", ""):
            info.append(f"stop_cue_time_s={float(x['stop_time']):.6g}")
        if x["goRT"] not in ("null", "", "?"):
            info.append(f"start_RT_ms={x['goRT']}")
        if x["RT"] not in ("null", "", "?"):
            info.append(f"stop_completion_time_ms={x['RT']}")
        if x.get("one_value"):
            info.append("note=only one of start_RT and stop_completion_time was recorded in this row")
        info.append(f"exclude={excl}")
        info.append(f"browser={x['fixed'][3]} {x['fixed'][4]};os={x['fixed'][5]};screen={x['fixed'][8] if len(x['fixed']) > 8 else ''}")
        ft = num(x["fix_time"])
        rows.append(dict(
            dataset=ID, study=STUDY, experiment="Continuous movement stop task", group=grp, subject=str(pi), person=str(pi),
            subject_excluded_by_authors=False, age="", sex={"Female": "F", "Male": "M"}.get(x["gender"], ""), handedness="",
            population=grp, setting="lab", device=DEVICE, session=ses, block=f"block {block}", trial=tn,
            phase="practice" if block == 0 else "test", time_s=ft / 1000.0 if not np.isnan(ft) else "", task="tracking", unit="recording",
            condition=("planned stop; " if planned else "unplanned stop; ") + state,
            pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=not analysed,
            recording=tn, samples_file=fn,
            signal_kind="mouse cursor position on the screen (px): the go segment from the START cue and the stop segment from the STOP cue",
            channels="segment;x;y", channel_units="0 go, 1 stop;px;px", sample_rate_hz="", n_samples=len(df),
            duration_s=float(df.time_s.max()) if len(df) else "",
            events=("stop_cue_in_trial=" + f"{float(x['stop_time']):.6g}") if x["stop_time"] not in ("null", "") else "",
            processing="none (as shared; time_s is milliseconds / 1000, measured from the START cue in the go segment and from the STOP cue in the stop segment)" if L is not None else "traces not available for this row",
            trial_info=";".join(info)))
D = pd.DataFrame(rows)

P = checks.published
pd_p = {k for k in persons if k.startswith("PD")}
hc_p = {k for k in persons if k.startswith("HC")}
sx = {k: ("F" if v == {"Female"} else "M" if v == {"Male"} else "?") for k, v in sex_of.items()}
P("participants with Parkinson's disease", len(pd_p), 23, 0, "Methods, Participants (23 individuals with PD)")
P("female participants with Parkinson's disease", sum(sx[k] == "F" for k in pd_p), 8, 0, "Table 1 (PD: 8 F, 15 M)")
P("male participants with Parkinson's disease", sum(sx[k] == "M" for k in pd_p), 15, 0, "Table 1 (PD: 8 F, 15 M)")
P("planned stop trials per recording session (after practice)", float(np.median([v for (k, s, g), v in n_planned.items() if g == "go"])), 80, 0, "Methods, Task (80 planned and 80 unplanned stop trials)")
P("unplanned stop trials per recording session (after practice)", float(np.median([v for (k, s, g), v in n_planned.items() if g == "stop"])), 80, 0, "Methods, Task (80 planned and 80 unplanned stop trials)")
acc_txt = []
for (grp, ses, lab, pct) in [("healthy older adults", 1, "HC S1", 99), ("healthy older adults", 2, "HC S2", 98), ("Parkinson's disease", 1, "PD OFF", 93), ("Parkinson's disease", 2, "PD ON", 93)]:
    v = [100 * a[0] / a[1] for (g, s, k), a in acc.items() if g == grp and s == ses]
    acc_txt.append(f"{lab} {np.mean(v):.1f} (paper {pct})")
checks.internal("94 files read, 168 rows each", len(D) == 94 * 168, f"{len(D)} rows")
n_tr = sum(v for k, v in status_n.items() if k != "no trace")
checks.internal("traces rebuilt for at least 98% of the rows that should have a trace", status_n.get("unparsed", 0) <= 0.02 * n_tr, str(status_n))
meta = dict(
    paper=dict(authors="Karekal A, Schultz K, Rockhill AP, Hawong HY, Weston S, Miocinovic S, Swann NC", year=2025,
               title="Electrophysiological Signatures of Planned and Unplanned Continuous Movement Termination in Parkinson's Disease",
               venue="eNeuro 12(10):ENEURO.0286-25.2025", doi="10.1523/eneuro.0286-25.2025"),
    lab="Swann (University of Oregon)",
    data_url="https://osf.io/8vqu6/", license="none stated", license_url="",
    changes=("Each trial row of the 94 behavioural CSV files (one per participant and session) became one row, with the mouse trace as a samples CSV (segment 0 = go lists, 1 = stop lists; "
             "time_s = the file's milliseconds / 1000). The files had been through a spreadsheet and exist in three layouts (see Known differences); the six trace lists were rebuilt "
             "from them. Participant labels were replaced by codes. The STOP-cue time (stop_time) is given to 6 significant digits (the files hold 16). Sex is the gender column of the files; age is empty (the age column is blank or a placeholder). "
             "Clinical scores of Table 1 and the EEG recordings are not included."),
    how=["One row is one trial of the continuous movement stop task. After a fixation cue and a START cue the participant moved the mouse clockwise around an on-screen circle at a steady pace while a countdown "
         "from a random number (count) ran down by one per second; the STOP cue came after the countdown reached 1 (planned stop, signal go) or earlier at a random time (unplanned stop, signal stop; stop_cue_time_s). "
         "start_RT_ms is the time from the START cue to the start of the cursor movement, stop_completion_time_ms the time from the STOP cue until the cursor came to full arrest. block 0 is practice. "
         "excluded_by_authors is true for practice trials and for trials whose exclude column says too slow, wrong way or remember: don't stop too early (the three exclusions named in the authors' read-me; the other texts of the column, such as too far from the circle and remember: try to stop, are training feedback messages and are kept in trial_info).",
         "Samples: the cursor position in screen pixels at each recorded mouse sample. Segment 0 (go) has its times counted from the START cue and segment 1 (stop) from the STOP cue, so the two segments have different time origins; "
         "times are not evenly spaced (about 17 ms apart when the mouse moved). x and y are screen pixels of a 1920 x 1080 window, origin and orientation not stated by the authors.",
         "Persons: the number in the file name (HC01...) is one person across the two sessions; HC session 1 and 2 are two sessions on one morning about an hour apart, PD OFF was recorded first, PD ON about an hour after taking medication."],
    differences=[
        "The files were exported from a spreadsheet and exist in three layouts. In the layout with one value per cell the comma between two lists lies inside one cell; where the spreadsheet had turned that cell into a number, the digits were "
        "joined, and they were split again where the neighbouring values continue best (a rule of ours, not the authors'). Rows whose trace could not be rebuilt keep their scalars and have an empty trace: " + ", ".join(f"{v} {k}" for k, v in sorted(status_n.items())) + " rows by status.",
        "The participant label inside %d files starts with PD although the file is in a healthy-control folder (or the reverse); the folder and file name were used for the group." % mismatch,
        "The paper's accuracies (93% PD OFF, 93% PD ON, 99% HC S1, 98% HC S2) did not reproduce from the exclude column under the read-me rule (percent of non-practice trials not marked too slow, wrong way or stopped early, mean over participants): " + "; ".join(acc_txt) + ". The paper does not define accuracy (Results cite 93/93/99/98% and Methods list four trial exclusions: RT over 1.5 s, counter-clockwise movement, stopping before the STOP cue, no movement initiated). Checked 11-09: the gap is not a parse problem (the exclude column is read cleanly in every row; the rebuilt traces have no backward time and jumps above 300 px in only 15 of about 15,700 segments, mostly at segment starts). Variants tried (add RT over 1500 ms: no change; add missing RT or start RT: PD 93.5/93.4, HC 98.0/97.2; wrong-way plus early only: PD 95.1/94.7, HC 98.8/98.1; pooled instead of per-participant means: same) reproduce neither pair, so accuracy is not used as a check.",
        "23 rows have a stop segment of a single sample (x, y, time) right before the exclude text; the first release read those three numbers as part of the exclude text (so 13 wrong-way or far-from-circle labels were not recognised) and dropped the sample; both are now fixed.",
        "A few rows have a single value where start_RT and stop_completion_time should both be; it is listed as stop_completion_time_ms with a note in trial_info (in most of them it equals the last stop-segment time, which is how the stop completion time is defined), and start_RT is left out.",
    ],
    unit_note="trial (one recording)",
)
stdlib.release(ID, "timeseries", D, meta, checks, extra_files=extra)
