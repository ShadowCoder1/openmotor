"""Lopez-Moliner, Vullings, Madelain & van Beers (2019) Sci Rep: trial-by-trial corrections when intercepting a moving target
with a stylus movement (arm experiment, 15 participants x 360 trials).

Raw file: raw_extra/LopezMoliner2019_SciRep/arm_movt_data.RData (R data frame 'x', read with pyreadr, never executed as R). The
button-press file (button_data.RData), the saccade file (eye_KR_KP.RData) and the 1.4 GB simulation file are not used: the
timing definitions of the first two are not documented well enough, and the paper's participant counts for the eye
experiments do not match the deposit. The deposit's 'subject' column holds participant initials and is dropped (the 'id'
column S1-S15 is used instead).

Run from the site folder: python3 -I tools/std/convert/LopezMoliner2019_SciRep.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "LopezMoliner2019_SciRep"
MODULE = "interception_timing"
DEVICE = ("hand-held stylus on a Calcomp DrawingTablet III (60 x 34 cm, 125 Hz); stimuli projected at 72 Hz on a screen 40 cm above "
          "the tablet")


def main():
    x = list(pyreadr.read_r(str(stdlib.raw_dir(DS) / "arm_movt_data.RData")).values())[0].reset_index(drop=True)
    ck = stdlib.Checks()
    ids = x["id"].astype(str)
    ck.internal("rows are in order within each participant (trial number counts 1..360)",
                bool((x.groupby(ids).trialnum.apply(lambda s: (s.to_numpy() == np.arange(1, len(s) + 1)).all())).all()))
    ck.internal("target starting distance = target speed x 0.8 s on every row",
                bool(np.allclose(x.xinit.abs(), x.velx.abs() * 0.8)))
    sp = x.velx.abs()
    df = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment="arm movement", group="", subject=ids, subject_excluded_by_authors=False,
        age=np.nan, sex="", handedness="", population="healthy", setting="lab", device=DEVICE, session=x.block + 1,
        block=[f"session {int(b) + 1}" for b in x.block], trial=x.trialnum, phase="test", time_s=np.nan,
        task="interception", unit="trial", condition="", pert_type="none", pert_value=0.0, pert_unit="",
        feedback="normal", excluded_by_authors=x.tErr.isna(),
        trial_in_block=x.trial, target_speed_cm_s=sp, target_direction=np.where(x.velx < 0, "leftward", "rightward"),
        target_start_x_cm=x.xinit.abs(), movement_onset_s=x["tStart.f"], crossing_time_s=x.timeEnd, movement_time_s=x.mt,
        temporal_error_s=x.tErr, spatial_error_cm=x.sErr, hit=x.hit.astype(bool), peak_velocity_cm_s=x.peakV,
        crossing_velocity_cm_s=x.crossV, time_to_contact_s=x.ttc, hand_end_x_cm=x.xhand_end, hand_end_y_cm=x.yhand_end,
        target_end_cm=x.targetEnd,
    ))
    ck.internal("every row of the arm-movement data frame is kept, in order (5,400)", len(df) == 5400)
    per = x.groupby(ids).size()
    ck.published("participants in the arm-movement experiment", ids.nunique(), 15, 0, "Methods, Participants")
    ck.published("trials per participant (360)", per.mean(), 360, 0, "Methods, Arm-movement experiment")
    ck.published("sessions per participant (4 sessions of 90 trials)", x.groupby(ids).block.nunique().mean(), 4, 0, "Methods, Arm-movement experiment")
    ck.published("trials per session (90)", x.groupby([ids, x.block]).size().mean(), 90, 0, "Methods, Arm-movement experiment")
    ck.published("number of target speeds (20, 25 and 30 cm/s)", sp.nunique(), 3, 0, "Methods, Arm-movement experiment")
    ck.published("slowest target speed, cm/s", sp.min(), 20, 0, "Methods, Arm-movement experiment")
    ck.published("fastest target speed, cm/s", sp.max(), 30, 0, "Methods, Arm-movement experiment")
    ck.published("time for the target to reach the interception zone, s (xinit / speed)", (x.xinit.abs() / sp).mean(), 0.8, 1e-9, "Methods, Arm-movement experiment")
    ck.internal("trials with no temporal error are under 2% of rows (the paper: under 2% missed the zone)", bool(x.tErr.isna().mean() < 0.02))

    meta = dict(
        study=DS,
        paper=dict(authors="Lopez-Moliner J, Vullings C, Madelain L, van Beers RJ", year=2019,
                   title="Prediction and final temporal errors are used for trial-to-trial motor corrections",
                   venue="Sci Rep 9:19230", doi="10.1038/s41598-019-55560-6"),
        lab="Lopez-Moliner (University of Barcelona)",
        data_url="https://osf.io/u273g/",
        license="none stated",
        license_url="",
        changes=("Kept the 5,400 rows of the arm-movement data frame in the authors' order, dropped the lag columns and model "
                 "columns computed by the authors from neighbouring trials, dropped the participant-initials column, and gave "
                 "participants codes; target speed is the absolute value of the deposit's signed velocity (the sign is kept in "
                 "target_direction)."),
        how=["One row is one trial in which a target dot moved horizontally at 20, 25 or 30 cm/s and reached the interception zone "
             "0.8 s after it started; the participant moved a stylus from a home position and had to cross the target's path at "
             "that moment. 15 participants did 360 trials in four sessions of 90 (block and session give the session).",
             "Times are in seconds and distances in centimetres, as in the paper; the deposit's temporal error, spatial error, hit flag, "
             "time to contact, hand end position and target end columns are given as shared (see columns.csv for what is and is not "
             "defined). excluded_by_authors marks the 3 trials with no temporal error. The data are not reach90 because the "
             "task is timed interception of a moving target, not reaching to a fixed target.",
             "Not included: the button-press file, the eye-movement file and the simulations."],
        unit_note="interception trial",
        differences=["Whether the sign of tErr in the deposit follows the paper's definition (crossing time minus target time) was not "
                     "verified; the hit flag does not equal 'absolute temporal error under 20 ms' on 26% of rows.",
                     "The paper reports 3 left-handed participants of 15; the deposit's target-direction signs split 13 and 2 and are "
                     "not linked to handedness, so handedness is empty."],
    )
    stdlib.release(DS, MODULE, df, meta, ck)
    for k in ck.items:
        if k["kind"] == "published":
            print("published", k["what"], round(k["computed"], 4), k["reported"], k["ok"])
        elif not k["ok"]:
            print("INTERNAL FAIL", k["what"])


if __name__ == "__main__":
    main()
