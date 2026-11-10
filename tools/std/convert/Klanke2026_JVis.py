"""Klanke, Ohl, Esen & Rolfs (2026) J Vis: awareness of catch-up saccades during smooth pursuit (three experiments).

Raw files: the per-participant-session CSVs of the three OSF projects cited by the paper: SPCS_*.csv (Experiment 1, OSF mhfke,
the listed deposit), SPC2_*.csv (Experiment 2, OSF hcf9a) and SPVS_*.csv (Experiment 3, OSF qbpkz). The shared analysis script
(analysis_catchUp-saccade_awareness.Rmd, identical in all three projects) and the projects' wiki pages were read as text, never run.

Run from the site folder: python3 -I tools/std/convert/Klanke2026_JVis.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Klanke2026_JVis"
MODULE = "saccade"
EXPS = [("SPCS", "Experiment 1 (SPCS)"), ("SPC2", "Experiment 2 (SPC2)"), ("SPVS", "Experiment 3 (SPVS)")]
INFO_COLS = ["localizationResponse", "phavel", "stiori", "dispSacIdx", "dispSacAmp", "dispSacVPk", "dispSacDur", "dispSacAng",
             "tedf_StimCf", "tedf_StimMS", "tedf_StimMT", "tedf_StimCd", "tedf_StimOf", "refTimeStamp", "fixpox",
             "sacDelay", "sacHorzVPeak", "sacVertVPeak", "sacVPeakTP", "sacDist", "sacAngle1", "sacAngle2", "cmbAng", "cmbPkVel",
             "saccDir", "dirPeakVel", "dirPhasVel", "dirPhasVelUp", "dirPhasVelDown", "retVelVals", "retinalVelos", "jumpDir",
             "lernLevel", "saccadeTypes", "dotNum", "dotMod", "dotDist", "dotAngle", "tSacGo", "sacTime", "sacrelmo",
             "mean_gainR", "mean_gainL"]


def info_str(r):
    parts = []
    for c in INFO_COLS:
        v = r[c]
        if pd.isna(v):
            continue
        parts.append(f"{c}={stdlib._fmt(v, 'text') if not isinstance(v, str) else v}")
    return "; ".join(parts)


def ci_half(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def main():
    files = sorted(glob.glob(str(stdlib.raw_dir(DS) / "*.csv")))
    parts = []
    for prefix, _ in EXPS:
        for f in [x for x in files if Path(x).name.startswith(prefix + "_")]:
            parts.append(pd.read_csv(f, low_memory=False))
    d = pd.concat(parts, ignore_index=True)
    ck = stdlib.Checks()
    ck.internal("104 participant-session files read (SPCS 24, SPC2 40, SPVS 40)", len(parts) == 104, str(len(parts)))
    d["subj"] = d["Experiment"] + "_" + d["vpno"].astype(str)
    # trial order in the participant's sequence (rows of one trial share tID and are contiguous)
    d["trial_seq"] = d.groupby("subj")["tID"].transform(lambda s: (s != s.shift()).cumsum())
    d["n_sac"] = d.groupby(["Experiment", "tID"])["sacNumber"].transform("max")
    ck.internal("rows of a trial are contiguous and sacNumber counts up within a trial",
                bool(d.groupby(["Experiment", "tID"])["sacNumber"].apply(lambda s: (s.diff().dropna() >= 0).all()).all()
                     and d.groupby("subj")["tID"].apply(lambda s: s.nunique() == (s != s.shift()).sum()).all()))
    has = d["sacNumber"] > 0
    exp_name = dict(EXPS)
    out = pd.DataFrame({
        "dataset": DS, "study": DS, "experiment": d["Experiment"].map(exp_name), "group": "", "subject": d["subj"],
        "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
        "device": "EyeLink 2 head-mounted eye tracker (500 Hz), projection screen at 340 cm",
        "session": d["seno"], "block": d["block"], "trial": d["trial_seq"], "phase": "test", "time_s": "",
        "task": "eye movements", "unit": np.where(has, "saccade", "trial"), "condition": d["condition"],
        "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
        "pursuit_target_speed_deg_s": d["tarvel"], "pursuit_target_direction": d["tardir"],
        "saccade_index_in_trial": np.where(has, d["sacNumber"], np.nan),
        "n_saccades": d["n_sac"],
        "saccade_onset_ms": d["sacOnset"].where(has), "saccade_offset_ms": d["sacOffset"].where(has),
        "saccade_duration_ms": d["sacDur"].where(has), "saccade_peak_velocity_deg_s": d["sacVPeak"].where(has),
        "saccade_amplitude_deg": d["sacAmp"].where(has),
        "eye": d["domEye"].map({"R": "right", "L": "left"}),
        "response": d["visResp"],
        "flash_seen_report": d["perceptualResponse"].map({0: False, 1: True}),
        "saccade_made_report": d["saccadeResponse"].map({0: False, 1: True}),
        "report_confidence": d["certaintyResponse"],
        "trial_info": [info_str(r) for _, r in d[INFO_COLS].iterrows()],
    })
    ck.internal("every row of every file kept in order", len(out) == len(d) == sum(len(p) for p in parts))
    ck.internal("report coding is 0/1 everywhere",
                bool(d["perceptualResponse"].isin([0, 1]).all() and d["saccadeResponse"].isin([0, 1]).all()))
    ck.internal("visResp text agrees with perceptualResponse",
                bool(((d["visResp"] == "have_seen") == (d["perceptualResponse"] == 1)).all()))
    ck.internal("at most 90 recorded trials per block in Experiment 1 (6 blocks x 90 per session; the deposit's trial counter also counts repeated trials)",
                bool(d[d.Experiment == "SPCS"].groupby(["vpno", "seno", "block"])["tID"].nunique().max() <= 90))

    # ---- published numbers
    ntr = d.groupby(["Experiment", "vpno"])["tID"].nunique()
    sec = "Results, 'Trial counts' / Methods, Participants"
    rep_n = {"SPCS": (8, 1566, 81), "SPC2": (10, 2038, 98), "SPVS": (10, 1946, 92)}
    for e, (n, m, c) in rep_n.items():
        x = ntr[e]
        ck.published(f"{exp_name[e]}: participants", len(x), n, 0, "Methods, Participants")
        ck.published(f"{exp_name[e]}: mean valid trials per participant", x.mean(), m, 0.5, "Results, trial counts")
        ck.published(f"{exp_name[e]}: 95% CI half-width of trials per participant", ci_half(x), c, 0.5, "Results, trial counts")
    # saccade rate per participant, (session or saccade type) and target speed, as in the authors' script
    dd = d.copy()
    dd["cmb_dv"] = dd["cmb_dv"].fillna("").astype(str)
    last = dd.groupby(["Experiment", "tID"]).tail(1)
    r = last.groupby(["Experiment", "vpno", "cmb_dv", "tarvel"]).apply(
        lambda s: pd.Series(dict(rate=s["sacNumber"].sum() / len(s) / ((s["tedf_StimOf"] - s["tedf_StimOn"]) / 1000).mean()))
    ).reset_index()
    rep_rate = {("SPCS", 6): (0.98, 0.33), ("SPCS", 9): (1.30, 0.42), ("SPCS", 12): (1.55, 0.53),
                ("SPC2", 3): (1.00, 0.18), ("SPC2", 6): (1.50, 0.23), ("SPC2", 9): (1.94, 0.28),
                ("SPVS", 3): (0.69, 0.19), ("SPVS", 6): (0.92, 0.21), ("SPVS", 9): (1.12, 0.27)}
    for (e, v), (m, c) in rep_rate.items():
        x = r[(r.Experiment == e) & (r.tarvel == v)]["rate"]
        ck.published(f"{exp_name[e]}: saccade rate at {v} deg/s (per s)", x.mean(), m, 0.0051, "Results, 'Saccade rate'")
        ck.published(f"{exp_name[e]}: 95% CI half-width of saccade rate at {v} deg/s", ci_half(x), c, 0.0051, "Results, 'Saccade rate'")

    meta = dict(
        study=DS,
        paper=dict(authors="Klanke JN, Ohl S, Esen AN, Rolfs M",
                   year=2026, title="In pursuit of saccade awareness: Limited volitional control and minimal conscious access to catch-up saccades during smooth pursuit eye movements",
                   venue="J Vis 26(6):10", doi="10.1167/jov.26.6.10"),
        lab="Rolfs (Humboldt-Universitat zu Berlin)",
        data_url="https://osf.io/mhfke/ (Experiment 1); Experiments 2 and 3: https://osf.io/hcf9a/ and https://osf.io/qbpkz/",
        license="none stated", license_url="",
        changes=("Combined the per-participant-session CSVs of the three OSF projects into one table with participant codes; "
                 "dropped the trial identifier (it contains participant letters) and a few columns that repeat others "
                 "(cmb_dv, tardi, visResp-as-number, StimOn); values the format has no column for are kept as name=value pairs "
                 "in trial_info under their deposit names."),
        how=["A trial of the experiment is a pursuit of a dot moving toward a fixation point at 3-12 deg/s for 1 s while a stimulus "
             "flash was triggered by the participant's catch-up saccade (active), replayed independent of eye movements (replay) or "
             "absent (noStim); afterwards the participant reported whether a flash was seen and whether a catch-up saccade was made "
             "(Experiment 1 also rated confidence 1-4, Experiments 2 and 3 also localized the flash). One row is one saccade the authors "
             "detected in a trial (unit saccade, saccade_index_in_trial counts them); a trial without a detected saccade has one row "
             "with empty saccade columns (unit trial). n_saccades is the trial total and trial counts trials in the participant's sequence.",
             "Saccade times are ms from stimulus onset (fixation point onset is at -500 ms), amplitude in deg and peak velocity in "
             "deg/s as in the deposit; the deposit names no units, they are taken from the Methods of the paper and its script. "
             "pursuit_target_speed_deg_s and direction come from tarvel and tardir. eye is the deposit's domEye, which varies within "
             "some participants. Experiments are coded by the deposit letters SPCS, SPC2 and SPVS; the participant letter in the "
             "deposit is only unique within an experiment, so persons are not linked across experiments.",
             "trial_info holds the remaining deposit columns under their deposit names, unchanged: the displayed (replayed) saccade "
             "dispSac*, stimulus phase-shift speed phavel and orientation stiori, trial event times tedf_* (ms from stimulus onset), "
             "refTimeStamp, saccade details (sacHorzVPeak, sacVertVPeak, sacVPeakTP, sacDist, sacAngle*, sacDelay, cmbAng, cmbPkVel, "
             "saccDir, dirPeakVel), retinal velocity values, Experiment 3 saccade-target dots (dot*, jumpDir, tSacGo, sacTime), "
             "sacrelmo, lernLevel and the mean pursuit gain of each eye. The deposit defines none of these in a readme or codebook: "
             "their meaning is read from the names, the paper and the shared R script only, so use them with care.",
             "The authors' exclusions (trials with more than one saccade, saccades outside the stimulus window, participants "
             "who did not finish) are applied inside their analysis script and are not flagged in the files, so excluded_by_authors "
             "is False for every row."],
        unit_note="detected saccade (or trial, when no saccade was detected)",
        differences=["The paper's Experiment 3 intended-versus-unintended saccade rates (0.77 and 1.05 per s, with a +-0.13/0.12 interval) "
                     "were not reproduced: the same calculation gives 0.78 and 1.05 with +-0.18 intervals, so these are not used as checks. The "
                     "deposit's listed project (mhfke) holds only Experiment 1; Experiments 2 and 3 come from the two further OSF projects cited by the paper."],
    )
    stdlib.release(DS, MODULE, out.to_dict("records"), meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
