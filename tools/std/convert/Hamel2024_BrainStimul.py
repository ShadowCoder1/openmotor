"""Hamel et al. (2024), Brain Stimulation 17(2):349-361: 4-element finger-press sequences with TMS over left or right M1.

Run from the site folder: python3 -I tools/std/convert/Hamel2024_BrainStimul.py
Raw file (untrusted data, read only): raw_extra/Hamel2024_BrainStimul/Hamel_data.xlsx from https://osf.io/tc3dx/ (no license stated).
Four sheets = four groups of 20 different people (Experiment 1 motor preparation, Experiment 2 sequence learning; each with a
Left M1 and a Right M1 group); participant numbers 1-20 repeat in every sheet and are NOT the same people.
Expected values of published checks come from the published paper (Methods 2.1, 2.4, 2.7, 2.9), read in the
UCLouvain repository copy https://research.dial.uclouvain.be/bitstreams/765adc03-d8a2-4aa4-8e72-3aa609484f48/download
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hamel2024_BrainStimul"
F = stdlib.raw_dir(ID) / "Hamel_data.xlsx"
checks = stdlib.Checks()
SHEETS = [("Left M1 - Motor Preparation", 1, "Left M1"), ("Right M1 - Motor Preparation", 1, "Right M1"),
          ("Left M1 - Sequence Learning", 2, "Left M1"), ("Right M1 - Sequence Learning", 2, "Right M1")]
MODCOLS = [c[1] for c in stdlib.columns_for("keypress")]
frames = []
raw = {}
for sheet, exp, grp in SHEETS:
    d = pd.read_excel(F, sheet_name=sheet)
    raw[sheet] = d
    n = len(d)
    pid = d["Participants_ID"].to_numpy()
    person = np.array([f"E{exp}{grp[0]}_{p:02d}" for p in pid])
    trial = d.groupby("Participants_ID").cumcount().to_numpy() + 1
    if exp == 1:
        pulse = d["TMS Pulse Types"].fillna("").astype(str).str.strip()
        rt_ratio, mt_ratio = d["RT (% of NoTMS)"], d["MT (% of NoTMS)"]
        rejected = pulse.isin(["CSE", "SICI"]) & d["RT (% of NoTMS)"].isna()
        t = pd.DataFrame({
            "block": "motor preparation (random sequences)", "block_type": "random", "phase": "other", "unit": "trial",
            "epoch": "", "condition": (pulse + " " + d["Time-Points"].fillna("")).str.strip(),
            "element_type": "random", "rt_ms": "", "rt_correct_ms": "", "movement_time_ms": "",
            "rt_ratio_notms": rt_ratio, "mt_ratio_notms": mt_ratio, "tms_timepoint": d["Time-Points"].fillna(""),
            "excluded_by_authors": rejected.to_numpy(),
        })
        t["cse_col"] = d["CSE (mV)"].to_numpy()
    else:
        pulse = d["TMS Pulse Types"].fillna("").astype(str).str.strip()
        short = d["Phases"].isna()
        t = pd.DataFrame({
            "block": np.where(short, d["Blocks"] + " block (short pseudorandom block)", d["Blocks"]),
            "block_type": np.where(d["Blocks"] == "Learning", np.where(short, "random", "training"), "random"),
            "phase": np.where(d["Blocks"] == "Learning", "learning", "other"), "unit": "trial",
            "epoch": d["Phases"].map({"Early": 1, "Late": 2}).astype(object).where(d["Phases"].notna(), "").to_numpy(),
            "condition": (d["Blocks"] + " " + d["Phases"].fillna("") + " " + pulse + " " + d["Time-Points"].fillna("")).str.split().str.join(" ").to_numpy(),
            "element_type": np.where((d["Blocks"] == "Learning") & ~short, "sequence", "random"),
            "rt_ms": d["RT (ms)"], "rt_correct_ms": "", "movement_time_ms": d["MT (ms)"],
            "rt_ratio_notms": "", "mt_ratio_notms": "", "tms_timepoint": d["Time-Points"].fillna(""),
            "excluded_by_authors": d["RT (ms)"].isna().to_numpy(),
        })
        t["cse_col"] = d["CSE (mV)"].to_numpy()
    acc = d["Accuracy (1 or 0)"]
    t["correct"] = acc.map({1: True, 0: False}).astype(object).where(acc.notna(), "").to_numpy()
    t["initiating_finger"] = d["Initiating Fingers"].fillna("").to_numpy()
    t["tms_pulse_type"] = pulse.replace({"": "none" if False else ""}).to_numpy()
    t["mep_mv"] = d["CSE (mV)"].to_numpy()
    t["sici_ratio"] = d["SICI (%)"].to_numpy()
    t = t.drop(columns=["cse_col"])
    t["dataset"], t["study"], t["experiment"] = ID, ID, f"{ID}_Exp{exp}"
    t["group"] = f"{grp} TMS"
    t["subject"], t["person"] = person, person
    t["trial"] = trial
    t["_exp"], t["_grp"], t["_pid"] = exp, grp, pid
    frames.append(t)
t = pd.concat(frames, ignore_index=True)
for c, v in {"subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "right", "population": "healthy", "setting": "lab",
             "device": "four-key keyboard, right hand; TMS over M1 with EMG of the first dorsal interosseous", "session": 1,
             "time_s": "", "task": "sequence learning", "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal",
             "block_number": "", "stimulus": "", "correct_key": "", "key_pressed": "", "sequence": "", "sequence_position": "",
             "repetition": "", "triplet_type": "none"}.items():
    t[c] = v
for col in MODCOLS:
    if col not in t.columns:
        t[col] = ""
t["tms_pulse_type"] = np.where(t.tms_pulse_type == "", "", t.tms_pulse_type)

# ---------------------------------------------------------------- checks
checks.internal("each sheet has 20 participants, numbered 1-20", all(raw[s]["Participants_ID"].nunique() == 20 for s, _, _ in SHEETS))
checks.internal("pulse labels cleaned of stray spaces (' SICI ', 'CSE  ', '  NoTMS' are the authors' typing; same four labels after strip)",
                set(t.tms_pulse_type.unique()) <= {"CSE", "SICI", "NoTMS", ""})
# Methods 2.7: 24 TMS trials per variable and time point (first experiment, time points before and at the GoCue)
e1 = t[t._exp == 1]
cnt = e1[e1.tms_pulse_type.isin(["CSE", "SICI"]) & e1.tms_timepoint.isin(["-500ms", "-250ms", "GoCue"])].groupby(
    ["subject", "tms_pulse_type", "tms_timepoint"]).size()
checks.published("Experiment 1: TMS trials per participant, variable (CSE, SICI) and time point -500 ms, -250 ms and GoCue (all 240 combinations of 40 participants)",
                 cnt.min() if (cnt == 24).all() and len(cnt) == 40 * 2 * 3 else -1, 24, 0, "Methods 2.7 (24 TMS trials for each TMS variable and time point)")
e1tms = e1[e1.tms_pulse_type.isin(["CSE", "SICI"])]
checks.published("Experiment 1: TMS trials per participant (CSE + SICI)", e1tms.groupby("subject").size().unique()[0]
                 if e1tms.groupby("subject").size().nunique() == 1 else -1, 240, 0, "Methods 2.4 (288 trials, 240 TMS and 48 NoTMS)")
e2 = t[t._exp == 2]
bl = e2.groupby(["subject", e2.block.str.startswith("Learning")]).size().unstack()
# right group's participant 20 has one extra Random row in the authors' sheet
checks.published("Experiment 2: Random trials per participant (median over 40)", float(bl[False].median()), 264, 0,
                 "Methods 2.4 (264 trials in the 2 Random blocks)")
checks.published("Experiment 2: Learning trials per participant (median over 40)", float(bl[True].median()), 336, 0,
                 "Methods 2.4 (336 trials in the 2 Learning blocks)")
sh = e2[e2.block.str.contains("short")].groupby("subject").size()
checks.published("Experiment 2: trials in the short pseudorandom blocks per participant (4 blocks x 18 trials)", float(sh.median()), 72, 0,
                 "Methods 2.4 (each Learning block starts and ends with a short block of 18 trials)")
checks.published("participants (4 groups of 20)", t.person.nunique(), 80, 0, "Methods 2.1 (four groups of 20, 80 participants)")
checks.published("Experiment 2: shortest reaction time kept (ms; the authors rejected RTs below 150 ms)", 1.0 if e2.rt_ms.replace("", np.nan).astype(float).min() >= 150 else 0.0, 1.0, 0,
                 "Methods 2.9 (RTs below 150 ms were rejected)")
checks.internal("every sheet row kept, in the authors' order", len(t) == sum(len(raw[s]) for s, _, _ in SHEETS))
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))
if not all(c["ok"] for c in checks.items):
    sys.exit(1)

t = t.drop(columns=["_exp", "_grp", "_pid"])
meta = dict(
    study="Hamel et al. 2024",
    paper=dict(authors="Hamel R, Waltzing BM, Hinder MR, McAllister CJ, Jenkinson N, Galea JM", year=2024,
               title="Bilateral intracortical inhibition during unilateral motor preparation and sequence learning",
               venue="Brain Stimul 17(2):349-361", doi="10.1016/j.brs.2024.03.009"),
    lab="Galea (University of Birmingham)",
    data_url="https://osf.io/tc3dx/",
    license="none stated", license_url="",
    changes=("Converted to the OpenMotor format from the four sheets of the authors' workbook (behaviour and the per-trial TMS "
             "measures the authors share; no EMG traces); participant codes replaced, stray spaces removed from the pulse-type labels. "
             "The sheets carry no ages or sexes."),
    how=[
        "One row is one trial of a task in which the right hand pressed a 4-element finger sequence after a go cue, with the sequence "
        "started by the index or the little finger (initiating_finger). Four groups of 20 different people: Experiment 1 (motor "
        "preparation, random sequences) and Experiment 2 (sequence learning), each with TMS over the left M1 (group 'Left M1 TMS') "
        "or the right M1 ('Right M1 TMS'). Participant numbers 1-20 repeat in every sheet; person codes are separate for the four groups.",
        "tms_pulse_type is CSE (single pulse), SICI (paired pulse) or NoTMS; tms_timepoint is when the pulse came relative to the go "
        "cue; mep_mv and sici_ratio are the authors' per-trial TMS measures (SICI as a proportion of CSE). Empty pulse type means the "
        "authors' sheet has no label (short pseudorandom blocks, and TMS trials the authors rejected, which keep their time point).",
        "Experiment 2: rt_ms is the time from go cue to the first press, movement_time_ms from first to last press, correct whether the "
        "sequence was executed correctly within 1750 ms. Rows are in the authors' order, listing the two Learning blocks and then the "
        "two Random blocks of each participant; the block order that was counterbalanced between participants is not in the sheet. "
        "epoch is 1 for the Early and 2 for the Late half of a block (first and last 66 trials); the short pseudorandom blocks at the "
        "start and end of each Learning block have no phase.",
        "Experiment 1: the authors share reaction time and movement time only as ratios to the participant's NoTMS average "
        "(rt_ratio_notms, mt_ratio_notms); rows labelled NoTMS are 1 by construction.",
        "excluded_by_authors is true for trials whose reaction time the authors removed (empty RT: below 150 ms, or a pulse outside the "
        "allowed window), and, in Experiment 1, for CSE and SICI rows without a ratio.",
    ],
    unit_note="trial (one 4-element finger-press sequence)",
    differences=[
        "Experiment 1 has 408 rows per participant against 288 trials in the paper (240 TMS, 48 NoTMS): 120 rows labelled NoTMS carry "
        "RT = MT = 1 and 48 rows have no pulse label; the sheet does not say how they relate to the 48 NoTMS trials, so they are kept as shared.",
        "In the Right M1 sequence-learning sheet participant 20 has one more Random row (265) than the 264 in the paper, and one participant has "
        "no accuracy for 132 trials and a few trials have no initiating finger; these are left empty as in the sheet.",
    ],
)
stdlib.release(ID, "keypress", t, meta, checks)
