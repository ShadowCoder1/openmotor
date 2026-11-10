"""Schutz, Lossin & Gegenfurtner (2015) Vision Res 113:169-178: dynamic integration of information about salience and value for
smooth pursuit eye movements.

Raw file: raw_extra/Schutz2015_VisRes/S2015.zip (Zenodo 579843, SchutzLossinGegenfurtner_VisRes_2015.zip). Layout:
exp<1|2>_<salience|value>/subj_<n>/{data.csv, eye<trial>.csv} and readme.txt, which documents the columns (read as text). data.csv has one
row of parameters and analysed values per trial (302 columns); eyeN.csv holds that trial's gaze samples (1 ms steps from target onset).
Nothing in the deposit is run. Module: timeseries (one row per trial, its gaze samples as CSV inside the ZIP).

Run from the site folder: python3 -I tools/std/convert/Schutz2015_VisRes.py
"""
import io
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Schutz2015_VisRes"
RAW = stdlib.raw_dir(DS)
TMP = RAW / "_samples"
TMP.mkdir(exist_ok=True)
NAMES = {2: "dot_density_1", 3: "dot_density_2", 4: "motion_speed_1_deg_s", 5: "motion_speed_2_deg_s", 6: "motion_direction_1",
         7: "motion_direction_2", 8: "coherence_1_pct", 9: "coherence_2_pct", 10: "luminance_contrast_1", 11: "red_green_contrast_1",
         12: "blue_yellow_contrast_1", 13: "luminance_contrast_2", 14: "red_green_contrast_2", 15: "blue_yellow_contrast_2",
         16: "feedback", 17: "value_1", 18: "value_2", 19: "area_1", 20: "area_2", 21: "horizontal_target_direction_2",
         22: "vertical_target_direction_2", 23: "pursuit_latency_s", 24: "vertical_eye_speed_200_400ms",
         25: "eye_direction_200_400ms", 26: "decision_200_400ms", 27: "vertical_eye_speed_800_1000ms",
         28: "eye_direction_800_1000ms", 29: "decision_800_1000ms"}
EXP = {1: "Experiment 1 (coherence)", 2: "Experiment 2 (dot number)"}


def g(x):
    return "" if x != x else f"{x:.8g}"


def main():
    ck = stdlib.Checks()
    z = zipfile.ZipFile(RAW / "S2015.zip")
    names = [n for n in z.namelist() if not n.startswith("__MACOSX")]
    pat = re.compile(r"exp(\d)_(salience|value)/subj_(\d+)/data\.csv$")
    sets = sorted((int(m[1]), m[2], int(m[3]), n) for n in names if (m := pat.search(n)))
    rows, files = [], []
    ncols = set()
    dts = []
    ntr = defaultdict(int)
    n_eye_missing = 0
    persons = defaultdict(set)
    for e, cond, s, dn in sets:
        a = np.genfromtxt(io.BytesIO(z.read(dn)), delimiter=",")
        ncols.add(a.shape[1])
        d = dn.rsplit("/", 1)[0]
        persons[e].add(s)
        for r in a:
            t = int(r[0])
            en = f"{d}/eye{t}.csv"
            if en not in names:
                n_eye_missing += 1
                continue
            b = np.genfromtxt(io.BytesIO(z.read(en)), delimiter=",")
            dts.append(float(np.median(np.diff(b[:, 5]))))
            fname = f"e{e}_{cond}_s{s:02d}_t{t:03d}.csv"
            with open(TMP / fname, "w", newline="") as fh:
                fh.write("time_s,gaze_x_deg,gaze_y_deg,velocity_x_deg_s,velocity_y_deg_s,direction,eyelink_flags\n")
                for q in b:
                    fh.write(f"{q[5]:.3f},{g(q[0])},{g(q[1])},{g(q[2])},{g(q[3])},{g(q[4])},{int(q[6])}\n")
            files.append((f"samples/{fname}", str(TMP / fname)))
            ntr[(e, cond, s)] += 1
            info = ";".join(f"{NAMES[c]}={g(r[c - 1])}" for c in range(2, 30))
            rows.append(dict(
                dataset=DS, study=DS, experiment=EXP[e], group="", subject=f"E{e}_{cond}_S{s}", person=f"E{e}_S{s}",
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                device="EyeLink eye tracker (1000 Hz as the samples are 1 ms apart); two random-dot motion directions shown in one patch",
                session=1, block=f"{cond} manipulation", trial=t, phase="test", time_s="",
                task="eye movements", unit="recording", condition=("salience varied (no reward)" if cond == "salience" else "value varied (win or lose money by direction)"),
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=t, samples_file=f"samples/{fname}",
                signal_kind="gaze position and velocity of the tracked eye while pursuing a random-dot kinematogram with two coherent motion directions",
                channels="gaze_x_deg;gaze_y_deg;velocity_x_deg_s;velocity_y_deg_s;direction;eyelink_flags",
                channel_units="deg (assumed);deg (assumed);deg/s (assumed);deg/s (assumed);unit not stated;bit 1 saccade, bit 2 blink, bit 3 marker",
                sample_rate_hz=1000, n_samples=len(b), duration_s=float(b[-1, 5] - b[0, 5]), events="",
                processing="none stated (eye position and velocity as saved by the authors)", trial_info=info))

    ck.published("columns per trial in data.csv (readme.txt lists 1-29 and three blocks of 91 values: 30-120, 121-211, 212-302)",
                 float(max(ncols)), 302, 0, "Deposit documentation, readme.txt (data.csv columns)")
    ck.published("values per 100 ms window grid (0 to 900 ms in 10 ms steps)", len(range(0, 901, 10)), 91, 0,
                 "Deposit documentation, readme.txt (columns 30-120, 121-211, 212-302)")
    ck.published("sampling interval of the eye files (s)", float(np.median(dts)), 0.001, 1e-6, "Deposit documentation, readme.txt (column 6, time from target onset) with 1 ms steps")
    ck.internal("all data.csv files have 302 columns", ncols == {302})
    ck.internal("every trial has an eye file", n_eye_missing == 0, f"{n_eye_missing} missing")
    ck.internal("one row and one sample file per trial", len(rows) == len(files) and len(rows) > 0, f"{len(rows)} trials")
    ck.internal("each experiment has the same participants in the salience and value manipulations",
                all({s for (ee, c, s) in ntr if ee == e and c == "salience"} == {s for (ee, c, s) in ntr if ee == e and c == "value"} for e in (1, 2)))

    meta = dict(
        study=DS,
        paper=dict(authors="Schutz AC, Lossin F, Gegenfurtner KR", year=2015,
                   title="Dynamic integration of information about salience and value for smooth pursuit eye movements", venue="Vision Research 113:169-178",
                   doi="10.1016/j.visres.2014.08.009"),
        lab="Schutz / Gegenfurtner (University of Giessen)",
        data_url="https://doi.org/10.5281/zenodo.579843",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Each trial's eye file (eyeN.csv) became one CSV with the authors' seven columns (time from target onset, gaze position, velocity, "
                 "direction, EyeLink flags); the first 29 columns of the trial's row in data.csv (parameters and the authors' analysed values) are in trial_info. "
                 "Columns 30-302 of data.csv (direction, vertical eye speed and decision on a 10 ms grid) are derived from the same samples and are not included. "
                 "Participant numbers were replaced by codes."),
        unit_note="trial (recording) of one participant",
        how=["One row is one trial: the participant tracked a random-dot kinematogram that contained two coherent motion directions. In the salience "
             "manipulation the coherence (Experiment 1) or the dot number (Experiment 2) of one signal was varied; in the value manipulation one of the "
             "two directions was rewarded and the other penalised (value_1 -100 and value_2 100 in the authors' columns). The deposit holds one folder "
             "per experiment and manipulation. subject is one participant in one experiment and manipulation; person is the participant number within the "
             "experiment (the same set of numbers occurs in both manipulations; numbers are not assumed to match between experiments).",
             "Samples: time_s from target onset (1 ms steps, 1.2 s), gaze position and velocity as saved by the authors; the readme gives units as '? of "
             "visual angle', read here as degrees; direction has no stated unit. trial_info names the readme's columns 2-29 (the readme lists 'luminance "
             "contrast 1', 'red-green contrast 1', 'blue-yellow contrast 1' twice; the second set is read as target 2's, inferred). Per the readme, the data are coded so that "
             "target 2 always moved to the lower right."],
        differences=["The paper could not be read when this release was made (publisher access refused), so no number from the paper was reproduced: "
                     "the published checks are facts documented in the deposit's own readme (column layout and sampling grid). The paper's participant and "
                     "trial counts and pursuit results are not rebuilt here.",
                     "The readme says all data are coded so that target 2 moved to the lower right, but horizontal_target_direction_2 is 1 (rightward) only in the value "
                     "manipulation; in the salience manipulation it takes both values -1 and 1 (and vertical_target_direction_2 takes both values in every file), so the "
                     "coding statement is not true of every row; values are left as saved.",
                     "The deposit has no exclusion flags; excluded_by_authors is False in every row."],
    )
    stdlib.release(DS, "timeseries", rows, meta, ck, extra_files=files)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.6f} vs {c['reported']}")
    print({e: len(v) for e, v in persons.items()}, len(rows))


if __name__ == "__main__":
    main()
