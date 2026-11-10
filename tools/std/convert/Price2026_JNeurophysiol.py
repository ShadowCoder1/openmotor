"""Price et al. (2026), J Neurophysiol 135(3):579-589: weight-bearing symmetry after asymmetric surface stiffness walking (Figshare 31290160, CC BY 4.0).

Raw files used, raw_extra/<id>/:
  allParticicipantsOutcomeMeasures.mat   MATLAB 5 struct 'results' (read with scipy): per participant (12) and condition (3) the spatio-temporal
                                         measures of 30 strides: temporal.step / temporal.stance (30 x 2: perturbed, unperturbed), temporal.stride (30),
                                         spatial.step (30 x 2), spatial.stride (30)
  participant info.csv                   mass, sex, age, height, perturbed side, treadmill speed
  readme.txt                             description (it describes 5 conditions, the file holds 3 per participant; see README of the release)
Not used: the 3 GB of raw force plate, marker, EMG and insole files, and the GRF profiles, EMG and insole arrays of the .mat (not stride tables).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Price2026_JNeurophysiol"
RAW = stdlib.raw_dir(ID)
r = sio.loadmat(RAW / "allParticicipantsOutcomeMeasures.mat", squeeze_me=True, struct_as_record=False)["results"]
info = pd.read_csv(RAW / "participant info.csv", encoding="utf-8-sig")
info.columns = ["Participant", "mass", "sex", "age", "height", "side", "speed"]
subj = [str(s) for s in r.subj]
side = [str(s) for s in r.perturbSide]
assert subj == list(info.Participant) and side == [{"L": "left", "R": "right"}[x] for x in info.side]
CONDS = [("Baseline", "baseline", "baseline"), ("Post-early", "aftereffect", "after asymmetric stiffness walking, early"),
         ("Post-late", "aftereffect", "after asymmetric stiffness walking, late")]
rows = []
tr = 0
for i, s in enumerate(subj):
    ip = info.iloc[i]
    tr = 0
    for j, (cname, phase, cdesc) in enumerate(CONDS):
        stt, sta, stp = r.temporal.step[i, j], r.temporal.stance[i, j], r.temporal.stride[i, j]
        sls, slg = r.spatial.step[i, j], r.spatial.stride[i, j]
        n = len(stp)
        assert stt.shape == (n, 2) and sta.shape == (n, 2) and sls.shape == (n, 2) and len(slg) == n
        for k in range(n):
            tr += 1
            rows.append(dict(
                dataset=ID, study="Price 2026 J Neurophysiol", experiment="asymmetric surface stiffness walking", group="",
                subject=s, person=s, subject_excluded_by_authors=False, age=int(ip.age), sex=ip.sex, handedness="",
                population="healthy young adults", setting="lab", device="force-instrumented dual-belt treadmill (Bertec); AdjuSST adjustable-stiffness treadmill for the intervention",
                session=1, block=cname, trial=tr, phase=phase, time_s="", task="walking", unit="stride", condition=cdesc,
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                walking_context="treadmill", walking_speed_mps=float(ip.speed), perturbed_side=side[i],
                step_time_perturbed_s=stt[k, 0], step_time_unperturbed_s=stt[k, 1],
                stance_time_perturbed_s=sta[k, 0], stance_time_unperturbed_s=sta[k, 1],
                step_length_perturbed_m=sls[k, 0], step_length_unperturbed_m=sls[k, 1],
                stride_time_s=stp[k], stride_length_m=slg[k]))
D = pd.DataFrame(rows)
part = pd.DataFrame([dict(person=s, mass_kg=float(info.mass[i]), height_cm=float(info.height[i]), perturbed_side=side[i]) for i, s in enumerate(subj)])
pcols = [("mass_kg", "number", "kg", "Body mass (participant info.csv)."), ("height_cm", "number", "cm", "Height (participant info.csv)."),
         ("perturbed_side", "category", "left, right", "Side of the leg on the low-stiffness belt in the asymmetric-stiffness phase.")]

c = stdlib.Checks()
c.published("participants", D.person.nunique(), 12, 0, "Methods, Participants")
c.published("female participants", int((info.sex == "F").sum()), 11, 0, "Methods, Participants (11F/1M)")
c.published("age mean (years)", info.age.mean(), 21, 0.5, "Methods, Participants (21 +/- 1)")
c.published("age SD (years)", info.age.std(ddof=1), 1, 0.5, "Methods, Participants (21 +/- 1)")
c.published("mass mean (kg)", info.mass.mean(), 69.3, 0.05, "Methods, Participants")
c.published("mass SD (kg)", info.mass.std(ddof=1), 8.1, 0.05, "Methods, Participants")
c.published("height mean (cm)", info.height.mean(), 164, 0.5, "Methods, Participants")
c.published("height SD (cm)", info.height.std(ddof=1), 7, 0.5, "Methods, Participants")
c.published("participants with left-side perturbation", sum(1 for x in side if x == "left"), 8, 0, "Methods, AdjuSST (first eight left)")
c.published("participants with right-side perturbation", sum(1 for x in side if x == "right"), 4, 0, "Methods, AdjuSST (final four right)")
c.published("treadmill speed (m/s)", info.speed.mean(), 1.25, 0.005, "Methods, Protocol (1.25 m/s)")
for j, (cname, _, _) in enumerate(CONDS):
    m = D[D.block == cname]
    c.published(f"stride time {cname} (s, mean of participant means)", m.groupby("person").stride_time_s.mean().mean(), 1.1, 0.05, "Table 1 (1.1 +/- 0.0-0.1)")
    c.published(f"stride length {cname} (m, mean of participant means)", m.groupby("person").stride_length_m.mean().mean(), 1.3, 0.05, "Table 1 (1.3 +/- 0.1)")
c.internal("12 participants x 3 conditions x 30 strides", len(D) == 12 * 3 * 30, str(len(D)))
c.internal("stride time = sum of the two step times", bool(np.nanmax(np.abs(D.stride_time_s - D.step_time_perturbed_s - D.step_time_unperturbed_s)) < 0.01))
c.internal("stride length = sum of the two step lengths", bool(np.nanmax(np.abs(D.stride_length_m - D.step_length_perturbed_m - D.step_length_unperturbed_m)) < 0.01))
c.internal("no missing stride values", int(D[["step_time_perturbed_s", "stance_time_perturbed_s", "step_length_perturbed_m", "stride_time_s"]].isna().sum().sum()) == 0)

asym = lambda a, b: 100 * (a - b) / (a + b)
def win(df, name, col, which):
    out = []
    for p, g in df.groupby("person", sort=False):
        out.append(g[col].values)
    return out
txt = []
for lab, cond, sl in (("Baseline", "Baseline", slice(-10, None)), ("Post-early", "Post-early", slice(0, 10)), ("Post-late", "Post-late", slice(-10, None))):
    v = [asym(g.step_time_perturbed_s.values[sl], g.step_time_unperturbed_s.values[sl]).mean() for _, g in D[D.block == cond].groupby("person", sort=False)]
    txt.append(f"{lab} {np.mean(v):.2f}")
step_asym_note = ("step time asymmetry (perturbed minus unperturbed, % of their sum) over the last 10 / first 10 / last 10 strides of the "
                  "three conditions is " + ", ".join(txt) + " here; Table 1 prints 0.3, -0.4 and 0.1")

meta = dict(
    paper=dict(authors="Price M, Schell EG, Jaramillo JM, Chiasson JM, Metsker LC, Huber ME, Hoogkamer W", year=2026,
               title="Weight-bearing symmetry changes after asymmetric surface stiffness walking", venue="J Neurophysiol 135(3):579-589", doi="10.1152/jn.00240.2025"),
    lab="Hoogkamer & Huber (UMass Amherst)", data_url="https://doi.org/10.6084/m9.figshare.31290160",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Only allParticicipantsOutcomeMeasures.mat (the authors' stride-by-stride spatio-temporal measures) and participant info.csv were converted. Each of the 30 strides per "
             "participant and condition became one row; the two-column arrays (perturbed, unperturbed) became _perturbed and _unperturbed columns. The raw force plate, marker, EMG and insole "
             "files (3 GB) and the .mat's GRF, EMG and insole profile arrays were not converted."),
    how=["One row is one stride of one participant walking on the force-instrumented treadmill at 1.25 m/s. The three conditions are the authors' three column groups; the .mat does "
         "not name them, so Baseline, Post-early and Post-late are taken from the paper (Baseline: before the 12-minute bout on the adjustable-stiffness treadmill; Post-early and "
         "Post-late: the initial and final part of the 5-minute walk after it). Each has 30 strides in the file; the paper analyses the last 10 strides of Baseline, the first 10 of "
         "Post-early and the last 10 of Post-late, and the file does not say which strides of the trials the 30 are. Rows are in the file's order.",
         "step_time_*: contralateral heel strike to ipsilateral heel strike (s); stance_time_*: heel strike to toe-off (s); step_length_*: anterior-posterior distance between the heel "
         "markers at the leg's heel strike (m); stride_time_s and stride_length_m are the sums of the two steps (checked in the rows). Column order in the .mat is perturbed side first, as the readme says.",
         "perturbed_side is the leg on the low-stiffness belt (15 kN/m) during the 10-minute adaptation bout, which is not part of these rows; pert_type is therefore none on every row and "
         "the Post rows have phase aftereffect."],
    differences=[
        "The authors' readme describes 12 x 5 arrays (including asymmetric-walking conditions) for the temporal and spatial measures, but the .mat holds 12 x 3, so only the three conditions "
        "with force-plate treadmill data (Baseline, Post-early, Post-late) are released; the order is inferred from the paper.",
        "The asymmetry means of Table 1 were not reproduced from these rows and are not used as checks: " + step_asym_note + ". The deposit does not say which 30 strides were saved nor how the "
        "readme's perturbed/unperturbed column order relates to the paper's sign convention, so the difference could not be traced."],
    unit_note="stride of one participant on the force-instrumented treadmill",
)
stdlib.release(ID, "walking", D.to_dict("records"), meta, c, participants=part, participant_columns=pcols)
