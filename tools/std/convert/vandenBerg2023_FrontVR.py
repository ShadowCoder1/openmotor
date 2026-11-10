"""van den Berg et al. (2023), Front Virtual Real 4:1104638: embodiment of virtual feet in a VR target-stepping task, one row per session recording.

Run from the site folder: python3 -I tools/std/convert/vandenBerg2023_FrontVR.py
Raw files (untrusted data, read only) in raw_extra/vandenBerg2023_FrontVR/x/ (extracted from data.rar of https://zenodo.org/records/7927396 with 7-Zip):
  <participant>/<S001-S003>/trackers/<tracker>_movement_T001.csv (UXF tracker logs: time, pos_x/y/z in m, rot_x/y/z in deg), condition_orders.xlsx,
  distances_new.xlsx (authors' distance to target per step), questionnaires.xlsx (sex), README.md (data dictionary).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "vandenBerg2023_FrontVR"
X = stdlib.raw_dir(ID) / "x"
checks = stdlib.Checks()
TR = ["camera", "controller_(left)", "controller_(right)", "hip", "left_foot", "right_foot"]
TRN = ["hmd", "controller_left", "controller_right", "hip", "left_foot", "right_foot"]
AX = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]
CHAN = [f"{t}_{a}" for t in TRN for a in AX]
UNITS = ["m", "m", "m", "deg", "deg", "deg"] * 6
COND = {1: "avatar", 2: "feet", 3: "none"}
ORD = ["first", "second", "third"]

co = pd.read_excel(X / "condition_orders.xlsx")
D = pd.read_excel(X / "distances_new.xlsx", sheet_name="Distances_corrected")
q = pd.read_excel(X / "questionnaires.xlsx")
sex = {r.participant: {"Male": "M", "Female": "F"}[r.sex] for r in q.itertuples()}

rows, extra, cells = [], [], {}
same_time = True
for i, r in enumerate(co.itertuples(), 1):
    p = r.participant
    for s in (1, 2, 3):
        cond = int(getattr(r, ORD[s - 1]))
        dfs = [pd.read_csv(X / p / f"S00{s}" / "trackers" / f"{t}_movement_T001.csv") for t in TR]
        t0 = dfs[0].time.values
        for d in dfs[1:]:
            same_time &= len(d) == len(t0) and np.allclose(d.time.values, t0)
        t = t0 - t0[0]
        fn = f"samples/{ID}_S{i:04d}_session{s}.csv"
        out = pd.DataFrame({"time_s": np.round(t, 6)})
        for tn, d in zip(TRN, dfs):
            for a in AX:
                out[f"{tn}_{a}"] = d[a].values
        extra.append((fn, out.to_csv(index=False, lineterminator="\r\n").encode()))
        dist = D[f"{p}_{cond}"].dropna().values
        cells.setdefault((cond, s), []).extend(dist)
        rows.append(dict(
            dataset=ID, study=ID, experiment=ID, group="", subject=p, person=p, subject_excluded_by_authors=False,
            age="", sex=sex[p], handedness="", population="healthy", setting="lab",
            device="HTC Vive head-mounted display, two hand controllers, hip and two foot trackers (Unity, UXF)",
            session=s, block=f"session S00{s}", trial=s, phase="test", time_s="", task="walking", unit="recording",
            condition=f"virtual body: {COND[cond]}", pert_type="virtual_body", pert_value="", pert_unit="",
            feedback="altered", excluded_by_authors=False,
            recording=s, samples_file=fn, signal_kind="VR tracker positions and Euler rotations (HMD, two controllers, hip, two feet) during 16-target stepping",
            channels=";".join(CHAN), channel_units=";".join(UNITS), sample_rate_hz=round((len(t) - 1) / t[-1], 1),
            n_samples=len(t), duration_s=round(float(t[-1]), 4), events="",
            processing="none (tracker logs as written by UXF; time shifted to start at 0)",
            trial_info=f"condition_code={cond};order={ORD[s - 1]};distance_to_target_m=" + "|".join(f"{v:.6g}" for v in dist)))
df = pd.DataFrame(rows)

def cell(c, o):
    v = np.array(cells[(c, o)])
    return v.mean(), v.std(ddof=1)

checks.published("participants", df.person.nunique(), 12, 0, "Methods, Participants (n = 12)")
checks.published("females", df.drop_duplicates("person").sex.eq("F").sum(), 2, 0, "Methods, Participants (two female)")
checks.published("recordings (12 participants x 3 conditions)", len(df), 36, 0, "Methods, within-subject design with three conditions")
# Table 1, mean distance to target (m), cells that agree to the printed 3 decimals
for (c, o), rep in {(1, 1): 0.041, (2, 1): 0.032, (2, 2): 0.034, (3, 2): 0.070, (3, 3): 0.064}.items():
    checks.published(f"mean distance to target, {COND[c]}, {ORD[o - 1]} trial (m)", cell(c, o)[0], rep, 0.0005, "Table 1")
checks.published("SD of distance to target, avatar, first trial (m)", cell(1, 1)[1], 0.023, 0.0005, "Table 1")
checks.internal("all six trackers share one time vector in every session", bool(same_time))
checks.internal("each recording has 15 or 16 distances", all(r["trial_info"].count("|") in (14, 15) for r in rows))
checks.internal("each participant has one recording per condition", all(g.condition.nunique() == 3 for _, g in df.groupby("person")))

meta = dict(
    paper=dict(authors="van den Berg A, de Vries B, Breedveld Z, van Mierlo A, Tijhuis M, Marchal-Crespo L", year=2023,
               title="Embodiment of virtual feet correlates with motor performance in a target-stepping task: a pilot study",
               venue="Front Virtual Real 4:1104638", doi="10.3389/frvir.2023.1104638"),
    lab="Marchal-Crespo (TU Delft)", data_url="https://zenodo.org/records/7927396",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Unpacked the RAR archive; for each participant and session the six tracker CSVs were merged into one samples CSV (time_s shifted to start at 0, "
             "the 36 channels renamed hmd / controller_left / controller_right / hip / left_foot / right_foot + pos or rot axis). Values are the authors' own. "
             "Participant folder names were replaced by codes. The authors' per-step distances to target (Distances_corrected sheet) are kept in trial_info."),
    how=["One row is one session recording: a participant stepping on 16 floor targets in sequence in VR, with six trackers logged (headset, two hand controllers, hip, left and right foot). "
         "Positions are in metres in Unity world coordinates (y up), rotations are Euler angles in degrees, as the authors' README states. The tracker log is irregular (about 90 Hz on average).",
         "Each participant did three sessions (S001-S003), one per virtual-body condition (avatar = full body, feet = feet only, none = no body shown). The deposit does not label sessions with "
         "conditions: condition_orders.xlsx gives each participant's condition codes in the order first, second, third, which we applied to S001, S002, S003, and codes 1, 2, 3 are "
         "named avatar, feet, none because the authors' distance table grouped that way reproduces the paper's Table 1 (avatar first-trial mean 0.041 m, none about 0.07 and 0.064 m on later trials). Both mappings are inferred.",
         "trial_info holds the 16 distances (m) between the foot and each target that the authors derived by manual heel-strike marking (distances_new.xlsx, Distances_corrected sheet); "
         "a list of 15 means one step is missing in the authors sheet. Sex comes from questionnaires.xlsx; age is not shared. The embodiment questionnaire answers are not included."],
    differences=["Table 1 cells avatar second trial (0.029 in the paper, 0.028 here), avatar third (0.036, 0.039), feet third (0.034, 0.033) and none first (0.140, 0.136) do not reproduce from the "
                 "authors' shared distance sheet; the paper also reports 240 steps in total although 12 x 3 x 16 would be 576, and the shared sheet has about 63-64 values per cell. Only the cells that agree are used as checks.",
                 "The deposit script mentions 22 participant codes; data for 12 were shared."],
    unit_note="session recording (six VR trackers)",
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
