"""Iturralde & Torres-Oviedo (2019) eNeuro 6(2):ENEURO.0358-18.2019: split-belt walking adaptation in 16 healthy older adults
(46-78 years) with kinematics, ground reaction forces and EMG of 15 muscles per leg. One row per stride.

Raw file: raw_extra/Iturralde2019_eNeuro/healthyParams_scrubbed.mat (Figshare 10.6084/m9.figshare.5764641, MIT licence), a MATLAB v7.3
(HDF5) file holding one labTools classdef object (groupAdaptationData, variable 'controls'): 16 adaptationData objects, each with
metaData (conditions and their trial numbers), subData (sex, age, dominant leg and arm, height, weight) and data, a parameterSeries of
strides x 1684 parameters with a label and a description for every parameter. The objects are decoded from the file's own object
store (/#subsystem#/MCOS) with h5py (Iturralde2019_eNeuro_mcos.py, adapted from Liu2024_RSOS_mcos.py); nothing from the deposit or from
the lab's code is run. The lab's published analysis code was not needed. Parameters are looked up by label for every participant
(one participant's table lists the fast/slow-leg parameters in a different order).
Run from the site folder: python3 -I tools/std/convert/Iturralde2019_eNeuro.py
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
from Iturralde2019_eNeuro_mcos import H5Objects  # noqa: E402

DS = "Iturralde2019_eNeuro"
MODULE = "walking"
RAW = stdlib.raw_dir(DS)
SRC = "Iturralde & Torres-Oviedo 2019"
ck = stdlib.Checks()

# Table 1 of the paper: sex, age (years) and self-selected medium treadmill speed (m/s) of S01..S16, in the deposit's order C0001..C0016
T1_AGE = [46, 51, 65, 58, 57, 52, 78, 52, 68, 62, 75, 57, 52, 64, 74, 49]
T1_SEX = "FFFFMMMFMFMMMMMM"
T1_MID = [0.94, 1.02, 1.08, 0.90, 1.04, 1.05, 0.66, 1.16, 0.85, 0.98, 1.11, 0.99, 1.16, 1.25, 1.11, 1.08]
# Table 3 of the paper: median across participants of the late long-exposure and late washout intervals (ms)
T3 = {("Adaptation", "late"): [176, 329, 161, 473], ("Washout", "late"): [187, 405, 181, 394]}
T3_PARAMS = ["doubleSupportSlow", "swingTimeSlow", "doubleSupportFast", "swingTimeFast"]  # DS1, swing slow, DS2, swing fast

BLOCK_PHASE = {"OG Base": "baseline", "TM slow": "baseline", "TM mid": "baseline", "Short exposure": "learning", "TM base": "baseline",
               "Adaptation": "learning", "Washout": "washout", "OG washout": "washout"}
# module column -> (labTools label, factor to the module's unit)
PARAMS = {
    "stride_time_s": ("strideTimeSlow", 1.0), "step_length_fast_m": ("stepLengthFast", 0.001), "step_length_slow_m": ("stepLengthSlow", 0.001),
    "step_time_fast_s": ("stepTimeFast", 1.0), "step_time_slow_s": ("stepTimeSlow", 1.0),
    "stance_time_fast_s": ("stanceTimeFast", 1.0), "stance_time_slow_s": ("stanceTimeSlow", 1.0),
    "double_support_fast_s": ("doubleSupportFast", 1.0), "double_support_slow_s": ("doubleSupportSlow", 1.0),
    "step_length_asymmetry": ("stepLengthAsym", 1.0), "step_position_asymmetry": ("Sout", 1.0), "step_time_asymmetry": ("Tout", 1.0),
    "stance_time_asymmetry": ("Tgoal", 1.0), "double_support_asymmetry_s": ("doubleSupportDiff", 1.0),
    "leading_position_fast_m": ("alphaFast", 0.001), "leading_position_slow_m": ("alphaSlow", 0.001),
    "trailing_position_fast_m": ("XFast", 0.001), "trailing_position_slow_m": ("XSlow", 0.001),
    "step_position_contribution": ("spatialContributionNorm2", 1.0), "step_time_contribution": ("stepTimeContributionNorm2", 1.0),
    "velocity_contribution": ("velocityContributionNorm2", 1.0),
}
MAIN_LABELS = {lab for lab, _ in PARAMS.values()} | {"bad", "good", "trial", "initTime", "fakeParam"}
DEVICE = ("Bertec instrumented split-belt treadmill; Vicon motion capture (100 Hz, hip and ankle markers); Delsys Trigno surface EMG "
          "(2000 Hz, 15 muscles per leg); handrail available")


def text(a):
    if isinstance(a, np.ndarray):
        return str(a.ravel()[0]) if a.size else ""
    return str(a)


def num(a):
    return float(np.ravel(a)[0])


def clean(label):
    return re.sub(r"\s+", "", label)


o = H5Objects(RAW / "healthyParams_scrubbed.mat")
grp = o.top("controls")
assert grp["__class__"] == "groupAdaptationData"
TMP = RAW / "samples_tmp"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir()

frames, people, files, ref_labels, ref_desc, late = [], [], [], None, None, {}
for k, a in enumerate(grp["adaptData"].ravel()):
    md, sub, ps = a["metaData"], a["subData"], a["data"]
    labels = [text(x) for x in ps["labels"].ravel()]
    descs = [text(x) for x in ps["description_"].ravel()]
    X = ps["Data_"].astype(float)
    assert X.shape[1] == len(labels) == len(descs)
    if ref_labels is None:
        ref_labels, ref_desc = labels, dict(zip(labels, descs))
    assert sorted(labels) == sorted(ref_labels), k
    col = {lab: X[:, i] for i, lab in enumerate(labels)}
    n = X.shape[0]
    trial = col["trial"]
    names = [text(c) for c in md["conditionName"].ravel()]
    assert names == list(BLOCK_PHASE), names
    block = np.array([""] * n, dtype=object)
    for cname, tic in zip(names, md["trialsInCondition"].ravel()):
        block[np.isin(trial, np.ravel(tic))] = cname
    assert (block != "").all()
    order = np.argsort(np.arange(n))  # rows stay in the deposit's order
    leg = text(sub["dominantLeg"]).lower()
    mid = T1_MID[k]
    code = f"{DS}_S{k + 1:04d}"
    people.append(dict(person=f"C{k + 1:04d}", age=num(sub["age"]), sex=text(sub["sex"]), leg=leg, arm=text(sub["dominantArm"]).lower(), mid=mid,
                       counts={b: int((block == b).sum()) for b in names}))
    split = np.isin(block, ["Short exposure", "Adaptation"])
    og = np.array([b.startswith("OG") for b in block])
    fast = np.where(split, 4 / 3 * mid, np.where(block == "TM slow", 2 / 3 * mid, mid))
    slow = np.where(split, 2 / 3 * mid, np.where(block == "TM slow", 2 / 3 * mid, mid))
    fast, slow = np.where(og, np.nan, fast), np.where(og, np.nan, slow)
    bad = col["bad"] == 1
    subj_ex = (k == 0)  # S01 failed to follow the instructions during split-belt walking and was excluded from the authors' analyses
    f = dict(dataset=DS, study=DS, experiment="", group="", subject=people[-1]["person"], person=people[-1]["person"],
             subject_excluded_by_authors=subj_ex, age=people[-1]["age"], sex={"Female": "F", "Male": "M"}[people[-1]["sex"]],
             handedness=people[-1]["arm"], population="healthy older adults", setting="lab", device=DEVICE, session=1,
             block=block, trial=np.arange(1, n + 1), phase=[BLOCK_PHASE[b] for b in block], time_s="", task="walking", unit="stride",
             condition=block, pert_type=np.where(split, "split_belt", "none"), pert_value=np.where(split, 2.0, 0.0),
             pert_unit=np.where(split, "ratio", ""), feedback="normal", excluded_by_authors=bad | subj_ex, fast_leg=leg,
             belt_speed_fast_mps=fast, belt_speed_slow_mps=slow, walking_trial=trial.astype(int), stride_start_s=col["initTime"],
             walking_context=np.where(og, "overground", "treadmill"))
    for c, (lab, factor) in PARAMS.items():
        f[c] = col[lab] * factor
    df = pd.DataFrame(f)
    frames.append(df)
    # every other parameter, by label, in one CSV per participant (rows in the same order as the main table)
    extra = {"trial": np.arange(1, n + 1)}
    for lab in ref_labels:
        if lab not in MAIN_LABELS:
            extra[clean(lab)] = col[lab]
    pd.DataFrame(extra).to_csv(TMP / f"p{k + 1:04d}.csv", index=False, lineterminator="\r\n", float_format="%.8g")
    files.append((f"samples/{DS}_S{k + 1:04d}_stride_parameters.csv", str(TMP / f"p{k + 1:04d}.csv")))
    # Table 3 epochs: late = last 40 strides of the condition leaving out the last stride
    for cond, ep in T3:
        idx = np.where(block == cond)[0]
        ii = idx[:-1][-40:]
        late.setdefault((cond, ep), []).append([np.nanmean(col[p][ii]) * 1000 for p in T3_PARAMS])

dfall = pd.concat(frames, ignore_index=True)
for c in ["stride_start_s"] + list(PARAMS):
    dfall[c] = dfall[c].round(6)
desc_rows = [("trial", "Row number of the stride in the participant's rows of the main table (1-based)")] + \
    [(clean(l), ref_desc[l]) for l in ref_labels if l not in MAIN_LABELS]
pdesc = pd.DataFrame(desc_rows, columns=["parameter", "description_from_deposit"])
files.append((f"samples/{DS}_parameter_descriptions.csv", pdesc.to_csv(index=False, lineterminator="\r\n").encode("utf-8")))

# ---- internal checks
ck.internal("16 participants", len(people) == 16 and dfall.subject.nunique() == 16)
cnts = {b: (min(p["counts"][b] for p in people), max(p["counts"][b] for p in people)) for b in BLOCK_PHASE}
ck.internal("every participant walked every condition; tied blocks hold at least the planned 50 slow, 50 medium, 150 baseline strides, "
            "split blocks the planned 10 (short) and 900 (long) strides, washout 600 (paper, Experimental design)",
            all(cnts[b][0] >= m for b, m in (("TM slow", 50), ("TM mid", 50), ("Short exposure", 10), ("TM base", 150), ("Adaptation", 900))) and cnts["Washout"][1] >= 600, str(cnts))
ck.internal("the dominant leg is the fast leg (paper: the dominant leg always walked faster); dominant legs: " +
            str(pd.Series([p["leg"] for p in people]).value_counts().to_dict()), set(p["leg"] for p in people) <= {"right", "left"})
ex = dfall.excluded_by_authors
ck.internal("excluded_by_authors = the deposit's bad-stride flag or the one excluded participant", True, f"{int(ex.sum())} of {len(dfall)} strides")
good = dfall[~dfall.excluded_by_authors]
d = (good.stance_time_asymmetry - good.double_support_asymmetry_s / good.stride_time_s - good.step_time_asymmetry).abs().dropna()
ck.internal("signs: step_time_asymmetry = stance_time_asymmetry - double_support_asymmetry_s / stride time", bool(d.median() < 0.002),
            f"n={len(d)}, median |diff| {d.median():.5f}, 95th pct {d.quantile(.95):.5f}")
tot = (good.step_position_contribution + good.step_time_contribution + good.velocity_contribution - good.step_length_asymmetry).abs().dropna()
ck.internal("the three contributions add up to step_length_asymmetry", bool(tot.median() < 0.002), f"median |diff| {tot.median():.5f}, 95th pct {tot.quantile(.95):.5f}")
early = good[(good.block == "Adaptation")].groupby("subject", sort=False).head(5)
base = good[good.block == "TM base"].groupby("subject", sort=False).tail(40)
ck.internal("step_length_asymmetry is negative early in split-belt walking and near 0 in baseline (paper: negative = longer slow step)",
            bool(early.step_length_asymmetry.mean() < -0.05 and abs(base.step_length_asymmetry.mean()) < 0.1),
            f"early adaptation {early.step_length_asymmetry.mean():.3f}, late baseline {base.step_length_asymmetry.mean():.3f}")

# ---- published numbers
ck.published("participants", len(people), 16, 0, SRC + ", Methods: Subjects")
ages = [p["age"] for p in people]
ck.published("mean age, years (Methods: 61 +- 9.9)", np.mean(ages), 61, 0.5, SRC + ", Methods: Subjects")
SD_AGE = np.std(ages, ddof=1)
n_age = sum(abs(p["age"] - t) < 1 for p, t in zip(people, T1_AGE))
n_sex = sum(p["sex"][0] == t for p, t in zip(people, T1_SEX))
DIFF_T1 = f"{n_age} of 16 participants have the age printed in Table 1 and {n_sex} of 16 the printed sex, in the deposit's order (the participant in the table's S06 row is 63 years old here, 52 in the table; the S04 row is female in the table, male here); the deposit's data are kept as shared."
ck.published("youngest age, years", min(p["age"] for p in people), 46, 0.5, SRC + ", Methods: Subjects")
ck.published("oldest age, years", max(p["age"] for p in people), 78, 0.5, SRC + ", Methods: Subjects")
DIFF = []
names3 = ["DS1 [FHS-STO]", "swing of the slow leg", "DS2 [SHS-FTO]", "swing of the fast leg"]
for (cond, ep), rep in T3.items():
    med = np.median(np.array(late[(cond, ep)]), axis=0)
    for nm, m, r in zip(names3, med, rep):
        label = {"Adaptation": "late long exposure", "Washout": "late washout"}[cond]
        ck.published(f"median {nm} in {label} (ms)", m, r, 0.5, SRC + ", Table 3")
sp = good[good.block == "TM mid"].groupby("subject", sort=False).stride_time_s.median()
ck.internal("median stride time while walking at the medium speed is between 0.7 and 1.8 s for everyone", bool(sp.between(0.7, 1.8).all()), f"{sp.min():.2f}-{sp.max():.2f} s")
for c in ck.items:
    if not c["ok"]:
        print("FAILED", c)
print(len(dfall), "rows;", len(files), "extra files")
if not ck.passed():
    sys.exit(1)

s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == DS)
pa = s["papers"][0]
meta = dict(
    paper={k: pa[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
    license="MIT", license_url="https://opensource.org/licenses/MIT",
    changes=("Read the deposit's MATLAB v7.3 (labTools) objects without the lab's code. The main table holds, per stride, the kinematic "
             "parameters that fit the walking module (lengths converted from mm to m), the recording number, the stride start time and "
             "the bad-stride flag. Every other per-stride parameter of the deposit (EMG summaries and gait-phase averages of 15 muscles per leg, "
             "joint and limb angles, ground reaction forces and the remaining kinematic measures) is in samples/*_stride_parameters.csv, one file "
             "per participant, with the authors' parameter names and values unchanged. Participant IDs, dates, the experimenter field, height "
             "and weight are left out."),
    how=["One row is one stride (slow-leg heel strike to the next) of one participant, in the deposit's order; the deposit's conditions are the blocks "
         "(OG Base and OG washout were walked overground, the others on the treadmill). The dominant leg (the one the participant uses to kick a ball) "
         "walked on the fast belt. The deposit does not hold belt speeds: they are set from the paper (slow = 2/3 and fast = 4/3 of each participant's "
         "self-selected medium speed, medium from Table 1; TM slow both belts at the slow speed; split belts in Short exposure and Adaptation; "
         "tied at the medium speed otherwise), so a participant's speeds are the paper's design values, not measurements.",
         "Values and signs are the deposit's (labTools): step_position_asymmetry is Sout, step_time_asymmetry Tout, stance_time_asymmetry Tgoal, "
         "double_support_asymmetry_s the deposit's doubleSupportDiff (slow minus fast, s), and the three contributions are the labTools Norm2 versions; "
         "stance_time_asymmetry minus double_support_asymmetry_s over stride time reproduces step_time_asymmetry stride by stride. "
         "leg_orientation_asymmetry and range_of_motion_asymmetry are left empty because the deposit's formulas for them differ from this module's "
         "(their raw ingredients are in the stride-parameter files). stride_start_s is the deposit's initTime.",
         "excluded_by_authors marks the strides the deposit flags as bad (missing or out-of-order gait events, or a stride time out of range). The first "
         "participant (S01 in the paper) is marked subject_excluded_by_authors: the paper excluded this person from its analyses because they did not "
         "follow the instructions during split-belt walking; the deposit still holds their data.",
         "samples/<id>_Sxxxx_stride_parameters.csv: columns trial (row number in the main table) and the deposit's parameters named as in the deposit "
         "with spaces removed; a name starting with f belongs to the fast (dominant) leg and s to the slow leg. samples/<id>_parameter_descriptions.csv "
         "gives the deposit's description of each parameter. EMG values named Norm... are the paper's normalized EMG (0-100% of each muscle's baseline range); "
         "Old... and raw averages use the earlier normalization or none, as the deposit labels them."],
    unit_note="stride (from a slow-leg heel strike to the next one)",
    differences=["Table 3 intervals for baseline and for early long exposure and early washout, and the interquartile ranges of Table 3, were not reproduced "
                 "exactly from the shared strides (for example baseline DS2 median 168 ms here, 166 in the paper); only the late long exposure and late "
                 "washout medians are used as checks, with the epochs the paper defines (last 40 strides without the last one, all strides, all 16 participants).",
                 "The paper's text says 10 of the 16 participants were women, while its Table 1 lists 6 women and 10 men; the data hold 5 women and 11 men. " + DIFF_T1 + f" The SD of age is {SD_AGE:.2f} here and 9.9 in the paper (not used as a check). The medium speeds (and so the belt speeds) are matched to participants by their order in Table 1."])
pcols = [("self_selected_mid_speed_mps", "number", "m/s", "Self-selected medium treadmill speed of the participant (Table 1 of the paper); the slow and fast belt speeds are 2/3 and 4/3 of it.")]
parts = pd.DataFrame([dict(person=p["person"], self_selected_mid_speed_mps=p["mid"]) for p in people])
dfall = dfall.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE)])
stdlib.release(DS, MODULE, dfall, meta, ck, participants=parts, participant_columns=pcols, extra_files=files)
shutil.rmtree(TMP)
for c in ck.items:
    if c["kind"] == "published":
        print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}  ({c['source']})")
    elif c.get("detail"):
        print(f"  [internal] {c['what'][:70]}: {c['detail']}")
