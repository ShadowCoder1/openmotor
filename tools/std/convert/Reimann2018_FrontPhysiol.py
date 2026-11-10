"""Reimann, Fettrow, Thompson & Jeka (2018) Front Physiol 9:1271: walking in a virtual-reality cave with brief visual
fall perturbations; 20 young adults. Raw file: raw_extra/Reimann2018_FrontPhysiol/VisionData.zip (Zenodo 4986147, a copy of
Dryad 10.5061/dryad.bn960cq, CC0). Only VisionData/results.csv (the authors' table for their statistics, one row per
perturbation, first post-stimulus step) is read; results.mat (453 MB, full variables) is not converted.
Run from the site folder: python3 -I tools/std/convert/Reimann2018_FrontPhysiol.py"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Reimann2018_FrontPhysiol"
DEVICE = "Split-belt treadmill (self-paced, metronome 90 bpm) in a virtual-reality dome showing a corridor; Vicon (250 Hz), force plates (1000 Hz), EMG"


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "VisionData.zip")
    d = pd.read_csv(io.BytesIO(z.read("VisionData/results.csv")))
    assert len(d) == 4445 and d["index"].eq("ONE").all() and d.subject.nunique() == 20
    side = lambda s: s.split("_")[1].lower()  # noqa: E731
    ck = stdlib.Checks()
    rows = []
    seq = {}
    for r in d.itertuples(index=False):
        seq[r.subject] = k = seq.get(r.subject, 0) + 1
        rows.append(dict(
            dataset=DS, study=DS, experiment="", group="", subject=r.subject, person=r.subject, subject_excluded_by_authors=False,
            age=float("nan"), sex="", handedness="", population="healthy", setting="lab", device=DEVICE, session=1, block="trial %d" % r._8, trial=k, phase="test",
            time_s=float("nan"), task="walking", unit="step", condition=r.direction.lower().replace("towards", "toward"),
            pert_type="visual_scene_roll", pert_value=10.8, pert_unit="deg", feedback="altered", excluded_by_authors=False,
            walking_trial=r._8, stretch_start_s=r._9, stretch_end_s=r._10, stance_foot=side(r.stance_foot),
            trigger_foot=side(r.trigger_foot), perturbation_side=side(r.perturbation), step_after=1,
            direction=r.direction.lower().replace("towards", "toward"),
            step_placement_x_m=r.step_placement_x_inverted, stimulus_response_x_m=r.stimulus_response_x_inverted,
            cop_from_com_x_integrated_m_s=r.cop_from_com_x_integrated_inverted,
            trigger_leg_ankle_eversion_rad=r.trigger_leg_ankle_eversion_step_end_inverted,
            contra_leg_hip_abduction_rad=r.contra_leg_hip_abduction_step_end_inverted,
            trigger_leg_peroneus_longus_emg=r.trigger_leg_pero_lng_integrated_inverted,
            contra_leg_gluteus_medius_emg=r.contra_leg_glut_med_integrated_inverted))
    df = pd.DataFrame(rows)
    toward, away = df[df.direction == "toward"], df[df.direction == "away"]
    ck.published("participants", df.subject.nunique(), 20, 0, "Methods, Participants (20 healthy young adults)")
    ck.published("steps, toward condition", len(toward), 2225, 4, "Results, Table 3 (N = 2,225)")
    ck.published("steps, away condition", len(away), 2224, 4, "Results, Table 3 (N = 2,224)")
    ck.published("foot placement change, toward (mm)", toward.step_placement_x_m.mean() * 1000, 4.3, 0.6,
                 "Results / Table 3 (4.3 mm; the paper reports mixed-model estimates, here raw means)")
    ck.published("foot placement change, away (mm)", away.step_placement_x_m.mean() * 1000, 4.0, 0.6,
                 "Results / Table 3 (4.0 mm; mixed-model estimate vs raw mean)")
    ck.internal("no missing values in the response columns", not df.filter(like="_m").isna().any().any()
                and df.trigger_leg_peroneus_longus_emg.notna().all())
    ck.internal("trigger foot equals stance foot in every row", (df.trigger_foot == df.stance_foot).all())
    ck.internal("direction labels split the rows near evenly", abs(len(toward) - len(away)) < 10)
    meta = dict(
        paper=dict(authors="Reimann H, Fettrow T, Thompson ED, Jeka JJ", year=2018, title="Neural Control of Balance During Walking",
                   venue="Front Physiol 9:1271", doi="10.3389/fphys.2018.01271"),
        lab="Jeka / Reimann (University of Delaware)", data_url="https://doi.org/10.5281/zenodo.4986147 (copy of https://doi.org/10.5061/dryad.bn960cq)",
        license="CC0 1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read results.csv of the deposit (the table the authors fed to their statistical models) row by row, in order, with the "
                 "authors' columns renamed with units in the name and labels lower-cased; participant initials were replaced by codes. "
                 "The deposit's full-time-series file (results.mat, 453 MB), figure code and R output were not included."),
        how=["One row is the response of one participant to one visual fall perturbation (20 participants, 4,445 perturbations). The scene "
             "was rolled about the line of sight by 10.8 deg at heel strike of a randomly chosen foot; the row holds the authors' summary of "
             "the first step after it, each measure minus the mean of the two control steps before the trigger.",
             "Positive values mean toward the side of the triggering foot (the authors mirrored left-trigger trials). Distances are in m, "
             "angles in rad and integrated EMG in fractions of the participant's control activation times seconds, as in the deposit "
             "(the paper reports mm, degrees and percent*s). The deposit does not give age or sex per participant.",
             "block gives the authors' trial number within the participant; trial counts perturbations in the order of the authors' file "
             "(by participant, trial and time)."],
        unit_note="perturbation (the first step after it) of one participant",
        differences=["The table holds 4,445 first-step rows (2,223 toward, 2,222 away); the paper's Table 3 states N = 2,225 and 2,224 for "
                     "the same step, 4 rows more. The paper's 4.3 and 4.0 mm are mixed-model estimates; the raw means here are within 0.5 mm. "
                     "The 'group' column of the deposit says 'to be determined' for every row and is not used."])
    stdlib.release(DS, "balance_perturbation", df, meta, ck)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))


main()
