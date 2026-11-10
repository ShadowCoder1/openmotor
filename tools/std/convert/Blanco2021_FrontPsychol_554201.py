"""Blanco, Tassani & Ramirez (2021), Front Psychol 12:648479: real-time sound and motion feedback for violin bow technique learning
in novices (feedback group FG, control group CG) and expert violinists (EG). One row per full-bow exercise: the authors' bow-motion
descriptors (Kinect) and tone-quality descriptors (audio), plus the bow-motion series of the trial as a CSV in the ZIP.

Raw file (raw_extra/Blanco2021_FrontPsychol_554201/Sound_and_Motion_Data.rar from Zenodo 4554201, extracted on the PC with 7-Zip;
only data files read, nothing run): Sound_and_Motion_Data/participants_raw_Sound_Motion.mat holds three MATLAB cell arrays of
'participant' class objects (participants_FG, _CG, _EG), read without MATLAB by the MCOS reader (Blanco2021_FrontPsychol_554201_mcos.py,
a copy of Liu2024_RSOS_mcos.py). Properties are documented in the authors' Readme.txt, participant.m and participant_results.m (read as text).
The audio is not in the file ('audio_raw_data' is empty); the f0/power/aperiodicity series (10k+ samples per trial) are not carried, only the
per-trial descriptors the authors computed from them.
Run from the site folder: python3 -I tools/std/convert/Blanco2021_FrontPsychol_554201.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Blanco2021_FrontPsychol_554201_mcos import MatObjects  # noqa: E402

ID = "Blanco2021_FrontPsychol_554201"
RAW = stdlib.raw_dir(ID)
SRC = "Blanco et al. 2021 (bow technique)"
FS = 86.13
checks = stdlib.Checks()
o = MatObjects(RAW / "ex/Sound_and_Motion_Data/participants_raw_Sound_Motion.mat")
TMP = RAW / "samples_tmp"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir()

GROUP = {"participants_FG": "feedback", "participants_CG": "control", "participants_EG": "expert violinists"}
# readme: left-handed participants and one who had played the violin are excluded from all the analysis
EXCL_VIOLIN = {"Participant17_FG"}
# (results object, MC key, GS key, block, phase, condition, feedback)
CONDS = [("Results_pre", "Violin_Trials_MC", "Violin_Trials_GS", "Baseline", "baseline", "Baseline", None),
         ("Results_feed", "Violin_Trials_MC", "Violin_Trials_GS", "Acquisition", "learning", "Acquisition-NIC", None),
         ("Results_feed", "Violin_Trials_MC_online_MC_Task", "Violin_Trials_GS_online_MC_Task", "Acquisition", "learning", "Acquisition-KIC", "kinematic"),
         ("Results_feed", "Violin_Trials_MC_online_GS_Task", "Violin_Trials_GS_online_GS_Task", "Acquisition", "learning", "Acquisition-SQIC", "sound"),
         ("Results_post", "Violin_Trials_MC", "Violin_Trials_GS", "Transfer", "transfer", "Transfer", None)]
CH = [("bowSkew", "bow_skewness_deg"), ("tilt", "bow_tilt_deg"), ("inclination", "bow_inclination_deg"),
      ("bowBridge", "bow_bridge_distance"), ("Speed", "bow_speed"), ("ViolinDist", "bow_violin_distance")]


def arr(x):
    return np.asarray(x, float).ravel()


def sc(x):
    a = arr(x)
    return float(a[0]) if a.size else np.nan


people = []
for var in ("participants_FG", "participants_CG", "participants_EG"):
    v = o.variable(o.top[var])
    for p in (v if isinstance(v, list) else [v]):
        people.append((var, re.sub(r"[\[\]']", "", str(p["ID"])), p))
checks.internal("72 participants (29 FG, 28 CG, 15 EG)", len(people) == 72, str(len(people)))

rows, parts, files, order = [], [], [], {}
n_mc_gs_mismatch = 0
for var, pid, p in people:
    grp = GROUP[var]
    ex = sc(p["Lefthanded"]) == 1 or pid in EXCL_VIOLIN
    order[pid] = len(order) + 1
    scode = f"{ID}_S{order[pid]:04d}"
    gen = sc(p["Gender"])
    parts.append(dict(person=pid, musician=sc(p["Musician"]), years_playing_instrument=sc(p["Years"]), lefthanded=sc(p["Lefthanded"]),
                      amusia_code=sc(p["Amusia"]), voice_comfortable_pitch_hz=sc(p["Freq_ref"]), sweep_failed=sc(p["Sweep"]),
                      pitch_discrimination_pre_hz=sc(p["Test_pre"]), pitch_discrimination_post_hz=sc(p["Test_post"]),
                      good_at_singing=sc(p["Good_at_singing"]), kinect_score=sc(p["Kinect_score"]), timbre_score=sc(p["Timbre_score"]),
                      pitch_score=sc(p["Pitch_score"]), more_improvement=sc(p["More_improvement"]), satisfaction=sc(p["Satisfaction"]),
                      feedback_difficult_to_understand=sc(p["Feed_diff"]), cannot_play_while_watching=sc(p["Cannot_play"]),
                      feedback_too_fast=sc(p["Feed_fast"]), too_much_feedback=sc(p["Too_much_feedback"])))
    base = dict(dataset=ID, study="Blanco 2021 Front Psychol (bow technique)", experiment="violin bow technique with real-time feedback",
                group=grp, subject=pid, person=pid, subject_excluded_by_authors=bool(ex), age=sc(p["Age"]),
                sex={1.0: "F", 0.0: "M"}.get(gen, ""), handedness="left" if sc(p["Lefthanded"]) == 1 else "right",
                population="expert violinists" if grp == "expert violinists" else "novices (no violin experience)", setting="lab",
                device="Kinect motion tracking, microphone, SkyNote", session=1, task="violin bowing", unit="trial",
                pert_type="none", pert_value=0, pert_unit="", excluded_by_authors=bool(ex))
    n = 0
    for res, mck, gsk, block, phase, cond, fb in CONDS:
        r = p[res]
        mc, gs = r[mck], r[gsk]
        if mc.size == 0:
            continue
        mc, gs = mc[0, 0], gs[0, 0]
        k = arr(mc["bowSkew"]).size
        kg = arr(gs["pitch_stab"]).size if gs.size else 0
        if kg != k:
            n_mc_gs_mismatch += 1
        for i in range(k):
            n += 1
            ser = [arr(mc[f + "_raw_data"].ravel()[i]) for f, _ in CH]
            ns = len(ser[0])
            assert all(len(s) == ns for s in ser), (pid, cond, i)
            fn = f"samples/{ID}_S{order[pid]:04d}_trial{n:03d}.csv"
            df = pd.DataFrame({"time_s": np.arange(ns) / FS, **{nm: s for (_, nm), s in zip(CH, ser)}})
            df.to_csv(TMP / f"{scode}_{n:03d}.csv", index=False, lineterminator="\r\n", float_format="%.8g")
            files.append((fn, str(TMP / f"{scode}_{n:03d}.csv")))
            has_gs = i < kg
            rows.append({**base, "block": block, "trial": n, "phase": phase, "condition": cond,
                         "feedback": "altered" if (fb and grp != "control") else "none",
                         "bow_skewness": arr(mc["bowSkew"])[i], "bow_tilt_sd": arr(mc["tilt"])[i], "bow_inclination_sd": arr(mc["inclination"])[i],
                         "bow_bridge_distance_sd": arr(mc["bowBridge"])[i], "bow_speed_sd": arr(mc["Speed"])[i],
                         "bow_violin_distance_sd": arr(mc["ViolinDist"])[i],
                         "pitch_instability": arr(gs["pitch_stab"])[i] if has_gs else np.nan,
                         "dynamic_instability": arr(gs["dyn_stab"])[i] if has_gs else np.nan,
                         "aperiodicity_sd": arr(gs["aperiodicity"])[i] if has_gs else np.nan,
                         "n_motion_samples": ns, "motion_sample_rate_hz": FS, "samples_file": fn})
names = [c[1] for c in stdlib.columns_for("violin_bowing")]
a = pd.DataFrame(rows).reindex(columns=names)
pp = pd.DataFrame(parts)
print("conditions where sound descriptors are fewer/more than motion descriptors:", n_mc_gs_mismatch)

# ---- published checks (Participants 2.1, 2.2, 3.3, 4.5.1)
by = {g: [x for x in people if x[0] == g] for g in GROUP}
ps = lambda g: [x[2] for x in by[g]]  # noqa: E731
DIFF = []


def pub(what, got, rep, tol, src):
    if abs(got - rep) <= tol:
        checks.published(what, got, rep, tol, SRC + ", " + src)
    else:
        DIFF.append(f"{what}: {got:.4g} here, {rep} printed in the paper (not used as a check)")


checks.published("beginners (no violin experience)", len(by["participants_FG"]) + len(by["participants_CG"]), 57, 0, SRC + ", 2.1 Participants")
checks.published("expert violinists", len(by["participants_EG"]), 15, 0, SRC + ", 2.1 Participants")
checks.published("women among the experts", sum(sc(p["Gender"]) == 1 for p in ps("participants_EG")), 8, 0, SRC + ", 2.1 Participants")
checks.published("men among the experts", sum(sc(p["Gender"]) == 0 for p in ps("participants_EG")), 7, 0, SRC + ", 2.1 Participants")
checks.published("women in the feedback group", sum(sc(p["Gender"]) == 1 for p in ps("participants_FG")), 15, 0, SRC + ", 2.1 Participants")
checks.published("men in the feedback group", sum(sc(p["Gender"]) == 0 for p in ps("participants_FG")), 14, 0, SRC + ", 2.1 Participants")
checks.published("women in the control group", sum(sc(p["Gender"]) == 1 for p in ps("participants_CG")), 19, 0, SRC + ", 2.1 Participants")
checks.published("men in the control group", sum(sc(p["Gender"]) == 0 for p in ps("participants_CG")), 9, 0, SRC + ", 2.1 Participants")
for g, nm, m, s in (("participants_EG", "expert", 32.4, 10.06), ("participants_FG", "feedback", 29.915, 4.88), ("participants_CG", "control", 28.91, 7.5)):
    ages = np.array([sc(p["Age"]) for p in ps(g)])
    pub(f"mean age, {nm} group", np.nanmean(ages), m, 0.005 if m != 32.4 else 0.05, "2.1 Participants")
    pub(f"SD of age, {nm} group", np.nanstd(ages, ddof=1), s, 0.005 if s != 7.5 else 0.05, "2.1 Participants")
yrs = np.array([sc(p["Years"]) for p in ps("participants_EG")])
pub("experts' mean years of experience", np.nanmean(yrs), 18.6, 0.05, "2.1 Participants")
pub("experts' SD of years of experience", np.nanstd(yrs, ddof=1), 5.53, 0.005, "2.1 Participants")
ny = np.array([sc(p["Years"]) for g in ("participants_FG", "participants_CG") for p in ps(g)])
pub("beginners' mean years of training", np.nanmean(ny), 0.06, 0.005, "2.1 Participants")
pub("beginners' SD of years of training", np.nanstd(ny, ddof=1), 0.23, 0.005, "2.1 Participants")
lh = lambda g: sum(sc(p["Lefthanded"]) == 1 for p in ps(g))  # noqa: E731
checks.published("left-handed beginners, control group", lh("participants_CG"), 2, 0, SRC + ", 3.3 Statistical Analysis")
checks.published("left-handed beginners, feedback group", lh("participants_FG"), 4, 0, SRC + ", 3.3 Statistical Analysis")
for g, nm, m in (("participants_CG", "control", 34.611), ("participants_FG", "feedback", 33.517), ("participants_EG", "expert", 33.769)):
    sat = np.array([sc(p["Satisfaction"]) for p in ps(g)])
    pub(f"mean satisfaction score, {nm} group", np.nanmean(sat), m, 0.0005, "4.5.1 Satisfaction")
for k, rep in (("Baseline", 10), ("Transfer", 10)):
    cnt = a[a.condition == k].groupby("subject").size()
    pub(f"trials per participant in the {k} block (median)", float(cnt.median()), rep, 0, "2.2.1 Experimental Procedure")
# improvement over Baseline in bow skewness, feedback group (analysis sample: left-handed and violin-trained excluded; outliers could not be removed)
kin = a[(~a.excluded_by_authors.astype(bool)) & a.group.eq("feedback")]
pm = kin.groupby(["subject", "condition"]).bow_skewness.mean().unstack()
for c, rep in (("Acquisition-NIC", 14.5), ("Acquisition-KIC", 31.0), ("Acquisition-SQIC", 27.8)):
    imp = 100 * (1 - pm[c].mean() / pm["Baseline"].mean())
    pub(f"feedback group bow skewness improvement over Baseline, {c} (%)", imp, rep, 0.05, "4.2 Kinematic Analysis")

checks.internal("every participant has all five conditions", bool(a.groupby("subject").condition.nunique().min() >= 5), str(a.groupby("subject").condition.nunique().min()))
checks.internal("no participant lost", a.subject.nunique() == 72)
checks.internal("bow skewness is within +-90 deg", bool(a.bow_skewness.dropna().abs().max() < 90), str(a.bow_skewness.max()))
for c in checks.items:
    if not c["ok"]:
        print("FAILED", c)
print(len(a), "rows;", len(files), "sample files;", DIFF)
if not checks.passed():
    sys.exit(1)
s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
pa = s["papers"][0]
meta = dict(
    paper={k: pa[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Unpacked the authors' MATLAB participant objects into one row per bowing trial (the authors' per-trial descriptors of bow motion and tone quality) with the trial's bow-motion series as a CSV file, added block, phase, condition and group labels, and replaced participant identifiers with codes. The audio-derived f0, power and aperiodicity series (10,000+ samples per trial), the Kinect position and rotation fields the authors mark as not used, the pitch-matching trials and the pitch-instrument-condition (PIC) trials are not included; questionnaire scores are in participants.csv.",
    how=["One row is one full-bow exercise on the violin (alternating up and down bows over the whole bow). The bow-motion descriptors are the authors' per-trial values from Kinect tracking at 86.13 samples/s: bow_skewness is the mean absolute yaw angle of the bow (0 = bow perpendicular to the strings; checked against the series: it equals the mean of the absolute series for every trial), the other five are standard deviations over the trial. Angles are Euler angles in the violin's coordinate system and are taken to be degrees (values within +-90); the bow-bridge distance, bow speed and bow-violin distance units are not stated in the deposit. pitch_instability and dynamic_instability are the standard deviations of f0 (Hz) and power over the trial: lower means more stable (the authors' fields pitch_stab and dyn_stab are called stability in the paper but are instabilities).",
         "Blocks follow the paper: Baseline, Acquisition and Transfer. Within Acquisition, condition is NIC (normal instrument, no feedback), KIC (kinematic condition: real-time visual feedback on bow motion for the feedback and expert groups, attention to bow motion only for the control group) or SQIC (sound quality condition: real-time feedback on tone for the feedback and expert groups, attention to sound only for the control group); feedback is altered where real-time feedback was shown. The deposit holds ten NIC, five KIC and five SQIC trials per participant; the paper describes the acquisition sub-blocks differently, and the deposit does not say how trials map to sub-blocks. The deposit lists trials by condition, not in the order performed, so trial is the row's position in this table, block by block. A few participants have fewer trials.",
         "group: feedback (FG and AG in the deposit's ID labels), control (CG and AG2) or expert violinists (EG), as the authors' Readme states. subject_excluded_by_authors marks the 7 beginners the Readme and paper exclude from all analyses (6 left-handed, 1 who had had violin lessons as a child). The analysis-specific outliers (7 in bow skewness, 2 in sound quality in the paper; the Readme names only some of them) are not marked. Two experts who belonged to the project were excluded from the questionnaires only. participants.csv holds the authors' per-participant scores.",
         "samples/*.csv: one file per trial with time_s (from 0, at 86.13 Hz) and the six bow-motion channels exactly as shared (no filtering or resampling); tracking glitches from the Kinect are left in (the authors note bad tracking in some trials)."],
    unit_note="bowing trial", differences=DIFF + ["The paper's mixed-design ANOVAs and post-hoc tests were not recomputed. The paper's outlier removal could not be reproduced, because the Readme does not name every removed participant."])
pcols = [("musician", "number", "0/1", "1 if the participant is a musician (authors' Musician)."),
         ("years_playing_instrument", "number", "years", "Years playing a musical instrument (authors' Years)."),
         ("lefthanded", "number", "0/1", "1 if left-handed (Lefthanded); left-handed beginners are excluded by the authors."),
         ("amusia_code", "number", "binary code as a number", "Brams amusia test result (authors' Amusia, shown as the number of the code)."),
         ("voice_comfortable_pitch_hz", "number", "Hz", "Frequency of the most comfortable pitch on the syllable da (Freq_ref)."),
         ("sweep_failed", "number", "0/1", "1 if the participant could not cover the required range in the vocal sweep (Sweep)."),
         ("pitch_discrimination_pre_hz", "number", "Hz", "Pitch discrimination threshold before the experiment (Test_pre)."),
         ("pitch_discrimination_post_hz", "number", "Hz", "Pitch discrimination threshold after the experiment (Test_post)."),
         ("good_at_singing", "number", "0 no, 1 neutral, 2 yes", "Self-rating: do you consider yourself good at singing."),
         ("kinect_score", "number", "1-5", "Rated effectiveness of the Kinect and motion detection technology (Kinect_score)."),
         ("timbre_score", "number", "1-5", "Rated effectiveness of the sound quality technology (Timbre_score)."),
         ("pitch_score", "number", "1-5", "Rated effectiveness of the pitch detection technology (Pitch_score)."),
         ("more_improvement", "number", "0 pitch, 1 timbre stability, 2 motion and kinematics, 3 other", "What the participant thinks improved more during the session."),
         ("satisfaction", "number", "8-40 (the Readme says 1-40)", "Satisfaction score from the post-experiment questionnaire."),
         ("feedback_difficult_to_understand", "number", "1-5", "Agreement: feedback difficult to understand (Feed_diff)."),
         ("cannot_play_while_watching", "number", "1-5", "Agreement: cannot play while watching the feedback (Cannot_play)."),
         ("feedback_too_fast", "number", "1-5", "Agreement: feedback too fast to follow (Feed_fast)."),
         ("too_much_feedback", "number", "1-5", "Agreement: too much feedback information (Too_much_feedback).")]
stdlib.release(ID, "violin_bowing", a, meta, checks, participants=pp, participant_columns=pcols, extra_files=files)
shutil.rmtree(TMP)
