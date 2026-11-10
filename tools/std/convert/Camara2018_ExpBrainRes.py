"""Camara, de la Malla, Lopez-Moliner & Brenner (2018) Exp Brain Res: eye movements in interception with delayed visual
feedback (OSF zxmpa). Two experiments, RawData_Exp1.zip (13 participants, 271 trials) and RawData_Exp2.zip (9 participants,
260 trials). Per trial: .eye (eye recording), .opt (finger position, Optotrak), .txt (trial settings); per participant the
summary table allIT<nn>.log. Files are read as text data only; the .txt trial settings (which hold a recording file path and
date) are not copied.

Run from the site folder: python3 -I tools/std/convert/Camara2018_ExpBrainRes.py
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Camara2018_ExpBrainRes"
MODULE = "timeseries"
NOVALUE = 12345      # code in allIT*.log for "no temporal error" (finger did not cross the target path)

EYE_HEADER = ["Time", "LeftPupil", "LeftGazeX", "LeftGazeY", "RightPupil", "RightGazeX", "RightGazeY", "HeadX", "HeadY",
              "HeadZ", "HeadHor", "HeadVer", "LeftEyeHor", "LeftEyeVer", "RightEyeHor", "RightEyeVer"]
EYE_CH = EYE_HEADER[1:]
EYE_UNITS = ["a.u.", "cm", "cm", "a.u.", "cm", "cm", "unit not stated", "unit not stated", "unit not stated", "deg", "deg",
             "deg", "deg", "deg", "deg"]
FINGER_CH = ["finger_x", "finger_y", "finger_z"]
EXPS = {
    1: dict(zip="RawData_Exp1.zip", name="Experiment 1: delayed cursor ramp", n_part=13, n_trials=271,
            cond={"light": ("hand visible", "baseline"), "off": ("no feedback", None), "delay": ("delayed cursor", "learning")}),
    2: dict(zip="RawData_Exp2.zip", name="Experiment 2: lights off, delayed cursor", n_part=9, n_trials=260,
            cond={"minimal_delay": ("minimal delay", "baseline"), "fixed_delay_200": ("fixed 200 ms delay", "learning"),
                  "random_delay": ("random delays", "test")}),
}


def read_log(z, path):
    d = pd.read_csv(io.BytesIO(z.read(path)), sep=r"\s+", engine="python")
    return d.rename(columns={"cond": "condition"})


def csv_text(header, lines):
    return header + "\n" + "\n".join(lines) + "\n"


def main():
    ck = stdlib.Checks()
    rows, extra, summary = [], [], []
    seen = {}

    for e, cfg in EXPS.items():
        z = zipfile.ZipFile(stdlib.raw_dir(DS) / cfg["zip"])
        parts = sorted({int(m.group(1)) for n in z.namelist() if (m := re.search(r"/participant(\d+)/", n))})
        ck.internal(f"Experiment {e}: {cfg['n_part']} participant folders", len(parts) == cfg["n_part"], str(len(parts)))
        n_trials_all = 0
        excl_n = 0
        for p in parts:
            base = f"RawData_Exp{e}/participant{p}/"
            logname = [n for n in z.namelist() if n.startswith(base) and n.endswith(".log")]
            assert len(logname) == 1
            lg = read_log(z, logname[0])
            assert lg.trial.tolist() == list(range(1, cfg["n_trials"] + 1)), (e, p)
            label = f"E{e}_{p}"
            seen.setdefault(label, len(seen) + 1)
            code = f"{DS}_S{seen[label]:04d}"
            first_off = lg.condition.eq("off").idxmax() if e == 1 else None
            rec_no = 0
            for _, r in lg.iterrows():
                t = int(r.trial)
                stem = f"{base}participant{p}_{t}"
                eye = z.read(stem + ".eye").decode("latin1").splitlines()
                assert eye[0].split() == EYE_HEADER, (stem, eye[0])
                eye_lines = [",".join(l.split()) for l in eye[1:]]
                assert all(len(l.split(",")) == 16 for l in eye_lines)
                opt = z.read(stem + ".opt").decode("latin1").splitlines()
                opt_lines = [",".join(l.split()) for l in opt]
                assert all(len(l.split(",")) == 4 for l in opt_lines)
                n_trials_all += 1
                cname, ph = cfg["cond"][r.condition]
                if e == 1 and r.condition == "off":
                    ph = "baseline" if t < 100 else "aftereffect"
                    cname = "no feedback (before delay)" if t < 100 else "no feedback (after delay)"
                delayed = r.feedback_delay > 0
                no_fb = r.feedback_delay < 0
                excl = r.error_respect_to_hand == NOVALUE
                excl_n += bool(excl)
                tm_eye = [float(l.split(",")[0]) for l in eye_lines]
                tm_opt = [float(l.split(",")[0]) for l in opt_lines]
                common = dict(
                    dataset=DS, study=DS, experiment=cfg["name"], group="", subject=label, person=label,
                    subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                    device="Optotrak 3020 (finger, 250 Hz), Eyelink II (500 Hz), projector at 60 Hz with half-silvered mirror",
                    session=1, block=cname, trial=t, phase=ph, time_s="", task="eye movements", unit="recording",
                    condition=cname, pert_type="delay" if delayed else "none",
                    pert_value=round(float(r.feedback_delay) * 1000.0, 3) if delayed else 0.0, pert_unit="ms",
                    feedback="none" if no_fb else ("altered" if delayed else "normal"), excluded_by_authors=bool(excl),
                )
                rec_no += 1
                rows.append({**common, "recording": rec_no, "samples_file": f"samples/{code}_t{t:03d}_eye.csv",
                             "signal_kind": "eye recording: pupil, gaze position and eye and head orientation",
                             "channels": ";".join(EYE_CH), "channel_units": ";".join(EYE_UNITS), "sample_rate_hz": 500,
                             "n_samples": len(eye_lines), "duration_s": max(tm_eye) - min(tm_eye), "events": "",
                             "processing": "gaze and eye angles as saved by the experiment program (calibrated against head "
                                           "markers); no further processing stated in the deposit"})
                extra.append((f"samples/{code}_t{t:03d}_eye.csv", csv_text("time_s," + ",".join(EYE_CH), eye_lines)))
                rec_no += 1
                rows.append({**common, "recording": rec_no, "samples_file": f"samples/{code}_t{t:03d}_finger.csv",
                             "signal_kind": "index finger position (Optotrak marker on the nail)",
                             "channels": ";".join(FINGER_CH), "channel_units": ";".join(["mm"] * 3), "sample_rate_hz": 250,
                             "n_samples": len(opt_lines), "duration_s": max(tm_opt) - min(tm_opt), "events": "",
                             "processing": "none stated (raw Optotrak positions)"})
                extra.append((f"samples/{code}_t{t:03d}_finger.csv", csv_text("time_s," + ",".join(FINGER_CH), opt_lines)))
                summary.append(dict(
                    subject=code, experiment=cfg["name"], trial=t, condition_in_deposit=r.condition, success=r.success,
                    feedback_delay_s=r.feedback_delay,
                    temporal_error_cursor_s="" if r.error_respect_to_cursor == NOVALUE else r.error_respect_to_cursor,
                    temporal_error_hand_s="" if r.error_respect_to_hand == NOVALUE else r.error_respect_to_hand,
                    starting_point_radius_cm=r.starting_point_radius, target_radius_cm=r.target_radius,
                    target_velocity_x_cm_s=r.target_velocity, target_x_cm=r.target_x, target_y_cm=r.target_y,
                    image_delay_s=r.image_delay))
        ck.published(f"Experiment {e}: trials per participant x participants ({cfg['n_trials']} x {cfg['n_part']})",
                     n_trials_all, cfg["n_trials"] * cfg["n_part"], 0, "README: 271 trials in Experiment 1, 260 in Experiment 2")
        ck.published(f"Experiment {e}: participants", len(parts), cfg["n_part"], 0, "README / Methods, Participants")
        ck.internal(f"Experiment {e}: share of trials without a temporal error (finger lifted or off the target path) below 2.1%",
                    excl_n / n_trials_all < 0.021, f"{100 * excl_n / n_trials_all:.2f}%")

    sm = pd.DataFrame(summary)
    s1, s2 = sm[sm.experiment.str.startswith("Experiment 1")], sm[sm.experiment.str.startswith("Experiment 2")]
    c1 = s1[s1.subject == s1.subject.iloc[0]].condition_in_deposit.value_counts().to_dict()
    ck.published("Experiment 1: hand-visible trials per participant", c1["light"], 40, 0, "Methods, Experiment 1 (40 trials)")
    ck.published("Experiment 1: adaptation (delayed cursor) trials per participant", c1["delay"], 151, 0, "Methods, Experiment 1 (151 trials)")
    ck.published("Experiment 1: no-feedback trials per participant (40 before and 40 after the delay)", c1["off"], 80, 0, "Methods, Experiment 1")
    ck.published("Experiment 1: largest imposed delay (s)", s1.feedback_delay_s.max(), 0.200, 0.001, "Methods, Experiment 1 (delay rises to 200 ms)")
    c2 = s2[s2.subject == s2.subject.iloc[0]].condition_in_deposit.value_counts().to_dict()
    ck.published("Experiment 2: first trials at the minimal delay", c2["minimal_delay"], 20, 0, "Methods, Experiment 2 (20 trials)")
    ck.published("Experiment 2: trials at the fixed 200 ms delay", c2["fixed_delay_200"], 80, 0, "Methods, Experiment 2 (80 trials)")
    rnd = s2[s2.condition_in_deposit == "random_delay"]
    ck.published("Experiment 2: random-delay trials per participant", rnd.groupby("subject").size().iloc[0], 160, 0, "Methods, Experiment 2 (160 trials)")
    ck.published("Experiment 2: random-delay trials per participant and delay value (59, 100, 150, 200 ms)",
                 rnd.groupby(["subject", "feedback_delay_s"]).size().median(), 40, 0, "Methods, Experiment 2 (40 trials each)")
    ck.internal("Experiment 2 random delays take exactly four values 0.05, 0.10, 0.15, 0.20 s (nominal 59, 100, 150, 200 ms)",
                sorted(rnd.feedback_delay_s.unique().round(3)) == [0.05, 0.1, 0.15, 0.2])
    ck.internal("two recordings (eye, finger) per trial: 2 x (3523 + 2340) rows", len(rows) == 2 * (3523 + 2340))
    ck.internal("sample counts: every recording has samples", all(r["n_samples"] > 10 for r in rows))

    extra.append(("trial_summary.csv", sm.to_csv(index=False, lineterminator="\r\n")))
    extra.append(("trial_summary_columns.csv", "\r\n".join([
        "column,description",
        "subject,Participant sequence code as in the main table",
        "experiment,Experiment",
        "trial,Trial number",
        "condition_in_deposit,Condition label in the deposit's allIT log (light = hand visible; off = no cursor; delay / minimal_delay / fixed_delay_200 / random_delay = delayed cursor)",
        "success,hit or miss of the target (deposit's allIT log)",
        "feedback_delay_s,Delay of the cursor logged by the experiment program in seconds (0 = hand visible; -1 = no cursor); the paper gives a minimal system delay of about 59 ms",
        "temporal_error_cursor_s,Temporal error of the cursor: lateral offset at the crossing of the target path divided by target speed (s); empty where the log has the code 12345",
        "temporal_error_hand_s,The same for the finger; empty where the log has the code 12345",
        "starting_point_radius_cm,Radius of the starting point (cm)",
        "target_radius_cm,Radius of the target (cm)",
        "target_velocity_x_cm_s,Horizontal target velocity (cm/s); sign gives the direction",
        "target_x_cm,Horizontal start position of the target (cm)",
        "target_y_cm,Vertical position of the target path (cm)",
        "image_delay_s,Measured image delay logged for the trial (s)", ""])))

    meta = dict(
        study=DS,
        paper=dict(authors="Camara C, de la Malla C, Lopez-Moliner J, Brenner E", year=2018,
                   title="Eye movements in interception with delayed visual feedback",
                   venue="Exp Brain Res 236(7):1837-1847", doi="10.1007/s00221-018-5257-8"),
        lab="Brenner (VU Amsterdam) / Lopez-Moliner (Universitat de Barcelona)",
        data_url="https://osf.io/zxmpa/",
        license="none stated", license_url="",
        changes=("Each trial's eye file and finger file became a CSV of samples (time_s first, then one column per channel, "
                 "values copied as written); the per-trial summary log became trial_summary.csv with the sentinel 12345 "
                 "replaced by an empty cell; the trial settings text files and calibration files were not copied (they hold a "
                 "recording file path and date); participant folders became codes."),
        how=["One row is one recording: for every trial two rows, the eye recording (16 columns of the authors' eye file: pupil "
             "sizes, gaze position, head position and orientation, eye angles) and the index finger position (x, y, z). "
             "Samples are the authors' own; the eye file carries each timestamp twice or three times (500 Hz samples, time "
             "stamps in 8 ms steps) and these time stamps are kept as they are.",
             "The task is interception of a moving target with the unseen finger while a cursor shows the finger with a delay. "
             "Experiment 1: hand visible (40 trials), no feedback (40), delayed cursor with the delay ramping up to 200 ms (151), "
             "no feedback (40). Experiment 2: lights off, delayed cursor, 59 ms (20 trials), 200 ms (80), then 59, 100, 150, 200 ms "
             "randomly (160). pert_value is the delay logged by the experiment program (ms); trial_summary.csv holds hits, "
             "temporal errors and target settings per trial. excluded_by_authors marks trials where the log has no temporal "
             "error (the paper excludes trials where the finger lifted or did not reach the target path, under 2%).",
             "Units of the eye file are not stated in the deposit: gaze x and y are read as screen positions in cm and eye and "
             "head angles as degrees from the paper's description (gaze on the screen, 1 cm/s about 1 deg/s) and the value ranges; "
             "finger positions as millimetres from the starting point at y = -10 cm (the finger rests near y = -97 in the data). "
             "Participant numbers repeat between the two experiments (7 of the 9 in Experiment 2 also took part in Experiment 1, "
             "not stated which), so they are coded as separate people."],
        unit_note="recording (eye or finger samples of one trial)",
        differences=["The paper's license for the article is CC BY 4.0 but the OSF project states no license for the data. "
                     "The repeated-measures ANOVA of finger temporal error in Experiment 2 (F[3,21] = 57) was not reproduced "
                     "from the log (F[3,24] = 12.4 for 9 participants, F[3,21] = 9.0 without participant 6); the paper does not "
                     "say how the error was averaged. The reproduced numbers are counts and design values."],
    )
    stdlib.release(DS, MODULE, rows, meta, ck, extra_files=extra)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}  ok={c['ok']}")


if __name__ == "__main__":
    main()
