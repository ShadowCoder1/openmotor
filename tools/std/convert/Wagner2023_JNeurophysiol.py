"""Wagner & Schutz (2023) J Neurophysiol: saccade adaptation driven by two post-saccadic candidate stimuli.

Raw file: raw_extra/Wagner2023_JNeurophysiol/data.zip (Zenodo 7128739), member _forZenodo/eyeMovementsAndBehavior.csv
(one row per trial, 65 unnamed columns described in _forZenodo/README.rtf) and _forZenodo/adapDir.csv.
The deposit's analysis scripts (analysis.zip) were read as text only, to confirm the reference frames and the
baseline correction; nothing from the deposit is executed.

Run from the site folder: python3 -I tools/std/convert/Wagner2023_JNeurophysiol.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Wagner2023_JNeurophysiol"
MODULE = "saccade"

# Column names for the 65 columns of eyeMovementsAndBehavior.csv, in order (README.rtf).
RAW = (["experiment", "condition", "participant",
        "ts_target_on", "ts_detect", "ts_first_off", "ts_second_on", "ts_second_off", "ts_resp_on", "ts_resp_off",
        "n_sacc_before_change", "trial_no", "trial_type",
        "target_x", "target_y", "first_x", "first_y", "second_x", "second_y",
        "gap_first", "gap_second", "gap_reported"]
       + [f"ps_{n}" for n in ("on_sample", "off_sample", "blink", "on_ts", "on_x", "on_y", "off_ts", "off_x", "off_y",
                              "dur", "amp_x", "amp_y", "amp", "lat", "dir_end_rad", "dir_end_deg", "dir_amp_rad",
                              "dir_amp_deg", "pv", "pv_hor")]
       + [f"ss_{n}" for n in ("on_sample", "off_sample", "blink", "on_ts", "on_x", "on_y", "off_ts", "off_x", "off_y",
                              "dur", "amp_x", "amp_y", "amp", "lat", "dir_end_rad", "dir_end_deg", "dir_amp_rad",
                              "dir_amp_deg", "pv", "pv_hor")]
       + ["reterr_primary", "reterr_second", "excluded"])
assert len(RAW) == 65

EXPERIMENT = {1: "small-target", 2: "large-target"}
CONDITION = {1: "discriminate-1st", 2: "discriminate-2nd", 3: "discriminate-1st-no-2nd", 4: "discriminate-2nd-no-1st"}
TRIAL_TYPE = {1: "baseline", 2: "adaptation", 3: "retention"}           # README: 3 = post-adaptation; paper: retention
PHASE = {1: "baseline", 2: "learning", 3: "aftereffect"}
GAP = {1: "right", 2: "up", 3: "left", 4: "bottom"}                      # README coding of gap location and report
FIX_Y_SCREEN = -2.0      # fixation cross 2 deg below screen centre (Methods, Stimuli); target and candidate positions
                         # are logged relative to screen centre, gaze relative to the fixation cross (analysis code)
N_TRIALS, N_PART = 350, 48


def design_trial_type(pos):
    return np.where(pos <= 50, 1, np.where(pos <= 250, 2, 3))


def main():
    zpath = stdlib.raw_dir(DS) / "data.zip"
    with zipfile.ZipFile(zpath) as z:
        df = pd.read_csv(io.BytesIO(z.read("_forZenodo/eyeMovementsAndBehavior.csv")), header=None, names=RAW,
                         na_values=["NaN"])
        adapdir = pd.read_csv(io.BytesIO(z.read("_forZenodo/adapDir.csv")), header=None, na_values=["NaN"]).to_numpy()
    ck = stdlib.Checks()

    # ---- structure of the authors' file -------------------------------------------------------------------------
    ck.internal("file is 2 experiments x 4 conditions x 48 participant numbers x 350 trials",
                len(df) == 2 * 4 * N_PART * N_TRIALS and df.groupby(["experiment", "condition", "participant"]).size().eq(N_TRIALS).all(),
                f"{len(df)} rows")
    df["pos"] = df.groupby(["experiment", "condition", "participant"]).cumcount() + 1
    value_cols = RAW[3:]
    key = ["experiment", "condition", "participant"]
    has_data = df.groupby(key)["ts_target_on"].transform(lambda s: s.notna().any())
    dropped = df[~has_data]
    n_dropped_blocks = dropped.groupby(key).ngroups
    ck.internal("participant-by-condition blocks the README calls never recorded are empty and are left out",
                bool(dropped[value_cols].isna().all().all()) and n_dropped_blocks == 183,
                f"{n_dropped_blocks} empty blocks, {len(dropped)} rows")
    d = df[has_data].copy()
    n_sessions = d.groupby(key).ngroups
    ck.internal("recorded sessions: 201 participant-by-condition blocks of 350 trials", n_sessions == 201 and len(d) == 201 * N_TRIALS,
                f"{n_sessions} sessions, {len(d)} rows")
    present = d["trial_no"].notna()
    ck.internal("the authors' trial number equals the row position within each session",
                bool((d.loc[present, "trial_no"] == d.loc[present, "pos"]).all()))
    empty = d["ts_target_on"].isna()
    ck.internal("trials with no recorded data inside recorded sessions are all flagged excluded by the authors",
                bool((d.loc[empty, value_cols[:-1]].isna().all().all()) and (d.loc[empty, "excluded"] == 1).all()),
                f"{int(empty.sum())} such trials kept as placeholder rows")
    tt = design_trial_type(d["pos"].to_numpy())
    ck.internal("trial type follows position: 1-50 baseline, 51-250 adaptation, 251-350 retention",
                bool((d.loc[~empty, "trial_type"].to_numpy() == tt[~empty.to_numpy()]).all()))

    # ---- participants ------------------------------------------------------------------------------------------------
    n_cond = d.groupby(["experiment", "participant"])["condition"].nunique()
    excluded_sub = {k for k, v in n_cond.items() if v < 4}   # authors' rule (adaptTargSel_excludeSubjects.m)
    per_exp = {e: int((n_cond.loc[e] > 0).sum()) for e in (1, 2)}
    included = {e: int((n_cond.loc[e] == 4).sum()) for e in (1, 2)}
    ck.internal("participants with data: 26 small-target (paper: 26 recorded), 28 large-target (paper: 29 recorded; "
                "the one excluded for data quality has no data in the deposit)", per_exp == {1: 26, 2: 28}, str(per_exp))
    ck.internal("participants in the analysis: 24 per experiment (Methods, Participants)", included == {1: 24, 2: 24}, str(included))
    ck.internal("participants excluded for missing sessions: 2 small-target, 4 large-target (Methods: opted out)",
                sorted(e for e, _ in excluded_sub) == [1, 1, 2, 2, 2, 2])
    subjects = sorted(n_cond.index)
    code = {k: f"{DS}_S{i + 1:03d}" for i, k in enumerate(subjects)}

    # ---- design checks -----------------------------------------------------------------------------------------------
    cond = d["condition"].to_numpy()
    first_relevant = np.isin(cond, (1, 3))
    rel_loc = np.where(first_relevant, 2.0, -2.0)            # task-relevant candidate, authors' recoded horizontal axis
    ok = True
    for (e, c, p), _ in d.groupby(key):
        v = adapdir[int(p) - 1, (int(e) - 1) * 4 + int(c) - 1]
        ok &= bool(v == (2.0 if c in (1, 3) else -2.0))
    ck.internal("adapDir.csv gives +2 deg (discriminate-1st conditions) or -2 deg (discriminate-2nd conditions) "
                "in every recorded session", ok)
    w = ~empty
    tdx, tdy = d["target_x"] - 0.0, d["target_y"] - FIX_Y_SCREEN
    ck.internal("eye movement target 10 deg straight up from the fixation cross in every trial with data",
                bool((np.hypot(tdx[w], tdy[w]) == 10).all() and (tdx[w] == 0).all()))
    adap = w & (d["trial_type"] == 2)
    ck.internal("candidates 2 deg left and right of the target in every adaptation trial (first at +2 on the recoded axis), "
                "second candidate 349-351 ms after saccade detection",
                bool((d.loc[adap, "first_x"] == 2).all() and (d.loc[adap, "second_x"] == -2).all()
                     and (d.loc[adap, "first_y"] == d.loc[adap, "target_y"]).all()
                     and (d.loc[w, "ts_second_on"] - d.loc[w, "ts_detect"]).between(349, 351).all()))
    sac = d["ps_amp"].notna()
    ident = (np.allclose(d.loc[sac, "ps_amp_x"], d.loc[sac, "ps_off_x"] - d.loc[sac, "ps_on_x"], atol=1e-9)
             and np.allclose(d.loc[sac, "ps_amp_y"], d.loc[sac, "ps_off_y"] - d.loc[sac, "ps_on_y"], atol=1e-9)
             and np.allclose(d.loc[sac, "ps_amp"], np.hypot(d.loc[sac, "ps_amp_x"], d.loc[sac, "ps_amp_y"]), atol=1e-9)
             and (d.loc[sac, "ps_lat"] == d.loc[sac, "ps_on_ts"] - d.loc[sac, "ts_target_on"]).all()
             and (d.loc[sac, "ps_dur"] == d.loc[sac, "ps_off_ts"] - d.loc[sac, "ps_on_ts"]).all())
    ck.internal("amplitudes equal end minus start positions; latency and duration equal timestamp differences", bool(ident))
    re_ok = np.allclose(d.loc[sac, "reterr_primary"], d.loc[sac, "ps_off_x"] - rel_loc[sac.to_numpy()], atol=1e-9)
    ck.internal("retinal error equals landing x minus the task-relevant candidate's x, in every trial with a saccade",
                bool(re_ok and d.loc[~sac, "reterr_primary"].isna().all()))
    no_resp = adap & d["gap_reported"].isna()
    ck.internal("blink, no-primary-saccade and no-response adaptation trials are all flagged excluded",
                bool((d.loc[d["ps_blink"] == 1, "excluded"] == 1).all() and (d.loc[~sac, "excluded"] == 1).all()
                     and (d.loc[no_resp, "excluded"] == 1).all()
                     and d.loc[w & (d["trial_type"] != 2), "gap_reported"].isna().all()))
    ck.internal("ranges: latency 1-1500 ms, duration > 0, peak velocity > 0, |horizontal amplitude| < 13 deg",
                bool(d.loc[sac, "ps_lat"].between(1, 1500).all() and (d.loc[sac, "ps_dur"] > 0).all()
                     and (d.loc[sac, "ps_pv"] > 0).all() and (d.loc[sac, "ps_amp_x"].abs() < 13).all()))

    # ---- standardized rows (authors' row order: experiment, condition, participant, trial) --------------------------
    n = len(d)
    exp = d["experiment"].to_numpy().astype(int)
    pos = d["pos"].to_numpy().astype(int)
    ttype = design_trial_type(pos)
    is_adap = ttype == 2
    second_delay = (d["ts_second_on"] - d["ts_detect"]).to_numpy()
    shown = w.to_numpy() & is_adap                            # a candidate was logged as shown on this row
    t0 = d.groupby(key)["ts_target_on"].transform("min")
    rel_gap = np.where(first_relevant, d["gap_first"], d["gap_second"])
    resp = d["gap_reported"].to_numpy()

    def col(name):
        return d[name].to_numpy(dtype=float)

    def bool_or_empty(mask_true, mask_known):
        return [bool(t) if k else "" for t, k in zip(mask_true, mask_known)]

    # task-relevant (step) and task-irrelevant (distractor) candidate on adaptation rows
    step_x = np.where(shown, np.where(first_relevant, col("first_x"), col("second_x")) - col("target_x"), np.nan)
    step_y = np.where(shown, np.where(first_relevant, col("first_y"), col("second_y")) - col("target_y"), np.nan)
    step_delay = np.where(shown, np.where(first_relevant, 0.0, second_delay), np.nan)
    has_distr = shown & np.isin(cond, (1, 2))                # both candidates shown only in conditions 1 and 2
    distr_x = np.where(has_distr, np.where(first_relevant, col("second_x"), col("first_x")) - col("target_x"), np.nan)
    distr_y = np.where(has_distr, np.where(first_relevant, col("second_y"), col("first_y")) - col("target_y"), np.nan)
    distr_delay = np.where(has_distr, np.where(first_relevant, second_delay, 0.0), np.nan)

    out = pd.DataFrame({
        "dataset": DS,
        "study": DS,
        "experiment": [EXPERIMENT[e] for e in exp],
        "group": "",
        "subject": [code[(e, p)] for e, p in zip(d["experiment"], d["participant"])],
        "subject_excluded_by_authors": [(e, p) in excluded_sub for e, p in zip(d["experiment"], d["participant"])],
        "age": "", "sex": "", "handedness": "",
        "population": "healthy",
        "setting": "lab",
        "device": "EyeLink 1000+ eye tracker (right eye, 1000 Hz), VIEWPixx monitor at 60 cm, keyboard",
        "session": "",
        "block": [TRIAL_TYPE[t] for t in ttype],
        "trial": pos,
        "phase": [PHASE[t] for t in ttype],
        "time_s": ((d["ts_target_on"] - t0) / 1000.0).to_numpy(),
        "task": "eye movements",
        "unit": "saccade",
        "condition": [CONDITION[c] for c in cond],
        "pert_type": np.where(~is_adap, "none", np.where(first_relevant, "intrasaccadic_step", "postsaccadic_step")),
        "pert_value": np.where(is_adap, rel_loc, 0.0),
        "pert_unit": "deg",
        "feedback": np.where(is_adap, "altered", np.where(exp == 1, "none", "normal")),
        "excluded_by_authors": (d["excluded"] == 1).to_numpy(),
        "target_amplitude_deg": np.where(w, np.hypot(tdx, tdy), np.nan),
        "target_direction_deg": np.where(w, np.degrees(np.arctan2(tdy, tdx)), np.nan),
        "step_x_deg": step_x,
        "step_y_deg": step_y,
        "step_delay_ms": step_delay,
        "distractor_x_deg": distr_x,
        "distractor_y_deg": distr_y,
        "distractor_delay_ms": distr_delay,
        "saccade_start_x_deg": col("ps_on_x"),
        "saccade_start_y_deg": col("ps_on_y"),
        "saccade_end_x_deg": col("ps_off_x"),
        "saccade_end_y_deg": col("ps_off_y"),
        "saccade_amplitude_deg": col("ps_amp"),
        "saccade_amplitude_x_deg": col("ps_amp_x"),
        "saccade_amplitude_y_deg": col("ps_amp_y"),
        "saccade_gain": np.nan,
        "saccade_latency_ms": col("ps_lat"),
        "saccade_duration_ms": col("ps_dur"),
        "saccade_peak_velocity_deg_s": col("ps_pv"),
        "postsaccadic_error_deg": col("reterr_primary"),
        "blink": bool_or_empty(d["ps_blink"].to_numpy() == 1, d["ps_blink"].notna().to_numpy()),
        "secondary_saccade_end_x_deg": col("ss_off_x"),
        "secondary_saccade_end_y_deg": col("ss_off_y"),
        "secondary_saccade_amplitude_x_deg": col("ss_amp_x"),
        "secondary_saccade_amplitude_y_deg": col("ss_amp_y"),
        "secondary_saccade_latency_ms": col("ss_lat"),
        "eye": "right",
        "response": [GAP[int(r)] if r == r else "" for r in resp],
        "response_correct": bool_or_empty(resp == rel_gap, ~np.isnan(resp)),
    })
    ck.internal("every row the authors shared for a recorded session is kept, in order", len(out) == n == 70350)
    ck.internal("pert_value matches the logged task-relevant candidate offset on every adaptation row with data",
                bool(np.allclose(out.loc[shown, "pert_value"], step_x[shown])))
    ck.internal("time_s increases within every session", bool(out.groupby(
        [d["experiment"].to_numpy(), cond, d["participant"].to_numpy()])["time_s"].apply(
        lambda s: (s.dropna().diff().dropna() > 0).all()).all()))

    # ---- published numbers, from the standardized rows ---------------------------------------------------------------
    inc = out[~out["subject_excluded_by_authors"]]
    amp_x = inc["saccade_amplitude_x_deg"].where(~inc["excluded_by_authors"])
    reported = {  # (experiment, condition): (last adaptation block, first retention block), deg
        ("small-target", "discriminate-1st"): (1.08, 0.72), ("small-target", "discriminate-2nd"): (0.51, 0.57),
        ("small-target", "discriminate-1st-no-2nd"): (1.05, 0.78), ("small-target", "discriminate-2nd-no-1st"): (-0.71, -0.28),
        ("large-target", "discriminate-1st"): (1.46, 0.50), ("large-target", "discriminate-2nd"): (-0.38, -0.09),
        ("large-target", "discriminate-1st-no-2nd"): (1.45, 0.57), ("large-target", "discriminate-2nd-no-1st"): (-1.04, -0.49),
    }
    excl_pct = {  # Methods, Data exclusion: mean % of trials excluded per condition
        ("small-target", "discriminate-1st"): 11.95, ("small-target", "discriminate-2nd"): 10.76,
        ("small-target", "discriminate-1st-no-2nd"): 9.81, ("small-target", "discriminate-2nd-no-1st"): 13.63,
        ("large-target", "discriminate-1st"): 10.35, ("large-target", "discriminate-2nd"): 11.07,
        ("large-target", "discriminate-1st-no-2nd"): 12.51, ("large-target", "discriminate-2nd-no-1st"): 10.05,
    }
    sec = {"small-target": "Results, 'Obligatory Use of Early Postsaccadic Error Signals in the Small-Target Experiment'",
           "large-target": "Results, 'Eye Movement Target Properties Modulate Prioritization of Error Signals'"}
    for (e, c), (rep_adap, rep_ret) in reported.items():
        g = inc[(inc["experiment"] == e) & (inc["condition"] == c)]
        ax = amp_x.loc[g.index]
        adap_m, ret_m = [], []
        for _, s in g.groupby("subject"):
            corrected = ax.loc[s.index] - ax.loc[s.index[s["phase"] == "baseline"]].mean()   # baseline correction
            adap_m.append(corrected[s["trial"].between(201, 250)].mean())                   # last adaptation block
            ret_m.append(corrected[s["trial"].between(251, 300)].mean())                    # first retention block
        ck.published(f"{e}, {c}: mean baseline-corrected horizontal primary saccade amplitude, last adaptation block (deg)",
                     np.mean(adap_m), rep_adap, 0.005, sec[e] + ", Fig. 2E-H")
        ck.published(f"{e}, {c}: mean baseline-corrected horizontal primary saccade amplitude, first retention block (deg)",
                     np.mean(ret_m), rep_ret, 0.005, sec[e] + ", Fig. 2E-H")
        pct = g.groupby("subject")["excluded_by_authors"].apply(lambda s: 100.0 * s.sum() / N_TRIALS)
        ck.published(f"{e}, {c}: mean % of trials excluded per participant", pct.mean(), excl_pct[(e, c)], 0.005,
                     "Methods, 'Data exclusion'")

    meta = dict(
        study=DS,
        paper=dict(authors="Wagner I, Schutz AC", year=2023,
                   title="Interaction of dynamic error signals in saccade adaptation",
                   venue="J Neurophysiol 129(3):717-732", doi="10.1152/jn.00419.2022"),
        lab="Schutz (University of Marburg)",
        data_url="https://doi.org/10.5281/zenodo.7128739",
        license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Converted the trial table (eyeMovementsAndBehavior.csv) to the OpenMotor format with participant codes, "
                 "left out the empty blocks the authors kept for sessions that were never recorded, and gave target and "
                 "candidate positions relative to the fixation cross and the target instead of the screen centre; the "
                 "saccade traces and some per-trial values (raw timestamps, sample numbers, saccade direction angles, a "
                 "second retinal error) stay only in the original deposit."),
        how=["One row is one trial of one participant's session: the upward primary saccade to a target 10 deg above "
             "the fixation cross and the first secondary saccade, with the authors' exclusion flags; each of the four "
             "conditions was one session of 350 trials on its own day, in an order the deposit does not record, so "
             "session is empty and trial counts 1-350 within each condition.",
             "Eye positions are in deg relative to the fixation cross, y positive upward, and the horizontal axis is "
             "the authors' recoded one: positive toward the side where the first candidate appeared for that "
             "participant (left for half of the participants, right for the other half).",
             "pert_value and step_x_deg give the task-relevant candidate's horizontal offset from the target (+2 or "
             "-2 deg), shown at online saccade detection in the discriminate-1st conditions (intrasaccadic_step) and "
             "350 ms later in the discriminate-2nd conditions (postsaccadic_step); distractor columns give the other, "
             "task-irrelevant candidate, and feedback is none in baseline and retention trials of the small-target "
             "experiment, where the target disappeared at saccade detection.",
             "postsaccadic_error_deg is the authors' retinal error (landing x minus the task-relevant candidate's x), "
             "which they also computed in baseline and retention trials where no candidate was shown; the paper's "
             "adaptation measure is saccade_amplitude_x_deg minus its mean over the participant's baseline trials."],
        unit_note="trial (one primary saccade, with the first secondary saccade if any)",
        differences=["The large-target experiment has 28 recorded participants here against 29 in the paper: one participant the authors excluded has no data in the deposit. The 24 + 24 analysed participants match."],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
