"""Zhou, Kruse, Brower, North & Joiner (2022), J Neurophysiol 128(4):854-871: motion state-dependent learning from physical
velocity-dependent force-field perturbations (FF group) versus explicit visual feedback of the same force (EVF group), 20
participants per group, Kinarm end-point robot. The authors shared processed per-participant arrays (adaptation coefficients
on error-clamp trials, force profiles, hand paths on a few force-field trials), not trial tables: released as participant
summaries in long form (one row per participant and value), with the array data as extra CSV files.

Run from the site folder: python3 -I tools/std/convert/Zhou2022_JNeurophysiol.py
Raw files (untrusted data, read only) in raw_extra/Zhou2022_JNeurophysiol/ from https://doi.org/10.6084/m9.figshare.17096630:
  Savings_Control_Group.mat   struct Savings_Control = the force-field (FF) group of the paper (20 participants)
  Savings_Explicit_Group.mat  struct Savings_Explicit = the explicit visual feedback (EVF) group (20 participants)
The deposit has no readme; the meaning of the arrays is taken from the paper (Methods) and from identities between arrays
checked below.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zhou2022_JNeurophysiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
GROUPS = [("FF", "Savings_Control_Group.mat", "Savings_Control"), ("EVF", "Savings_Explicit_Group.mat", "Savings_Explicit")]
AC = "adaptation coefficient: slope of the lateral force on an error-clamp trial regressed on the ideal velocity-dependent force (dimensionless; 1 = full compensation), after subtracting the participant's mean baseline force profile"

# 2-D (participant x point) adaptation coefficient arrays: field name suffix -> (phase, description of the points)
ARR2 = {
    "baseline": ("baseline", "baseline error-clamp trial, in order (8 per participant, scattered in 120 null trials)"),
    "pretrain": ("baseline", "the 2 error-clamp trials just before the force field / visual feedback was introduced (the second is the pre-trial of the first EC-FF-EC triplet)"),
    "train": ("learning", "time course of training: point 1 is the error-clamp trial after the first perturbation trial (equals post_FF_train); point 11 equals the mean of the last 10 training trials (late_train); points 2-10 are not described by the authors"),
    "washout": ("washout", "decay and washout: point 1 equals the late-training value, point 27 equals end_Null_washout (mean of the error-clamp trials at the end of the washout); the intermediate points are not described by the authors"),
    "reexposure": ("learning", "reexposure: point 1 is the pre-trial and point 2 the post-trial of the EC-FF-EC triplet at the start of reexposure; later points are not described by the authors"),
}
# 1-D (participant) arrays
ARR1 = {
    "early_train": ("learning", "mean adaptation coefficient over the first 10 training trials (early training)"),
    "middle_train": ("learning", "mean adaptation coefficient over the middle 10 training trials"),
    "late_train": ("learning", "mean adaptation coefficient over the last 10 training trials"),
    "end_EC_washout": ("washout", "adaptation coefficient at the end of the 60 consecutive error-clamp decay trials (as named by the authors)"),
    "end_Null_washout": ("washout", "mean adaptation coefficient of the error-clamp trials at the end of the washout"),
    "pre_FF_train": ("learning", "adaptation coefficient on the error-clamp trial just before the first perturbation trial of training"),
    "post_FF_train": ("learning", "adaptation coefficient on the error-clamp trial just after the first perturbation trial of training"),
    "pre_FF_reexposure": ("learning", "adaptation coefficient on the error-clamp trial just before the first perturbation trial of reexposure"),
    "post_FF_reexposure": ("learning", "adaptation coefficient on the error-clamp trial just after the first perturbation trial of reexposure"),
    "middle_reexposure": ("learning", "mean adaptation coefficient over the middle 10 reexposure trials"),
    "late_reexposure": ("learning", "mean adaptation coefficient over the last 10 reexposure trials"),
    "single_trial_adaptation_train": ("learning", "single-trial change in adaptation coefficient across the first training perturbation trial"),
    "single_trial_adaptation_reexposure": ("learning", "single-trial change in adaptation coefficient across the first reexposure perturbation trial"),
}
DISP = {  # EVF group only: hand displacement on the two physical FF trials, units not stated (hand coordinates look like cm)
    "first_FF_familiarization_displacement_peak_vel": "lateral hand displacement at peak velocity on the first physical force-field trial among the familiarization trials",
    "second_FF_familiarization_displacement_peak_vel": "lateral hand displacement at peak velocity on the second physical force-field trial among the familiarization trials",
    "first_FF_reexposure_displacement_150_after_peak_vel": "lateral hand displacement 150 ms after peak velocity on the first physical force-field trial after reexposure",
    "second_FF_reexposure_displacement_150_after_peak_vel": "lateral hand displacement 150 ms after peak velocity on the second physical force-field trial after reexposure",
    "first_FF_reexposure_displacement_peak_vel": "lateral hand displacement at peak velocity on the first physical force-field trial after reexposure",
    "second_FF_reexposure_displacement_peak_vel": "lateral hand displacement at peak velocity on the second physical force-field trial after reexposure",
}

MEAN10 = {"early_train", "middle_train", "late_train", "middle_reexposure", "late_reexposure"}
rows, extra = [], {}
n_missing = 0
store = {}
for gi, (grp, fn, key) in enumerate(GROUPS):
    S = sio.loadmat(RAW / fn, squeeze_me=True, struct_as_record=False)[key]
    store[grp] = S
    checks.internal(f"{grp}: 20 participants in every array", S.adaptation_coefficients_baseline.shape[0] == 20)

for grp, fn, key in GROUPS:
    S = store[grp]
    ac = lambda n: getattr(S, n if n.startswith("single_trial") else "adaptation_coefficients_" + n)  # noqa: E731
    checks.internal(f"{grp}: train point 1 = post_FF_train, train point 11 = late_train, washout point 1 = late_train, "
                    "washout point 27 = end_Null_washout, reexposure points 1-2 = pre/post FF reexposure, pretrain point 2 = pre_FF_train",
                    bool(np.allclose(ac("train")[:, 0], ac("post_FF_train"), equal_nan=True)
                         and np.allclose(ac("train")[:, 10], ac("late_train"), equal_nan=True)
                         and np.allclose(ac("washout")[:, 0], ac("late_train"), equal_nan=True)
                         and np.allclose(ac("washout")[:, 26], ac("end_Null_washout"), equal_nan=True)
                         and np.allclose(ac("reexposure")[:, 0], ac("pre_FF_reexposure"), equal_nan=True)
                         and np.allclose(ac("reexposure")[:, 1], ac("post_FF_reexposure"), equal_nan=True)
                         and np.allclose(ac("pretrain")[:, 1], ac("pre_FF_train"), equal_nan=True)))
    pert = "force_field" if grp == "FF" else "explicit_visual_force_feedback"
    pu = grp == "EVF"
    for p in range(20):
        subj = f"{grp}{p + 1}"
        order = []
        for name, (phase, desc) in ARR2.items():
            a = ac(name)
            for j in range(a.shape[1]):
                order.append((f"adaptation_coefficient_{name}_{j + 1:02d}", f"{AC}. {desc}", phase, name, a[p, j], 1 if name in ("baseline", "pretrain") else ""))
        for name, (phase, desc) in ARR1.items():
            order.append((f"adaptation_coefficient_{name}", f"{AC}. {desc}", phase, name, ac(name)[p], 10 if name in MEAN10 else ""))
        if grp == "EVF":
            for name, desc in DISP.items():
                order.append((name, f"{desc} (units not stated in the deposit; hand coordinates in the same file are of the size of cm)", "other", name, getattr(S, name)[p], 1))
        for t, (meas, mdesc, phase, name, v, nt) in enumerate(order):
            if v != v:
                n_missing += 1
                continue
            on = name in ("train", "reexposure") or (("train" in name or "reexposure" in name) and not name.startswith("pre_")
                                                      and name != "pretrain")
            rows.append(dict(
                dataset=ID, study=ID, experiment=ID, group=grp, subject=subj, person=subj, subject_excluded_by_authors=False,
                age="", sex="", time_s="", handedness="right", population="healthy", setting="lab", device="Kinarm End-Point Lab", session=1,
                block=name, trial=0, phase=phase, task="reaching", unit="participant summary",
                condition=grp, pert_type=pert if on else "none", pert_value=15 if on else 0, pert_unit="N s/m" if on else "",
                feedback="altered" if (pu and on) else "normal", excluded_by_authors=False,
                measure=meas, measure_description=mdesc, statistic="mean" if name in MEAN10 else "value",
                value=float(v), value_unit="dimensionless" if not meas.endswith("_peak_vel") and "displacement" not in meas else "unknown (cm?)",
                n_trials=nt, session_context=""))
    # order of the row within the participant's sequence
cnt = {}
for r in rows:
    cnt[r["subject"]] = cnt.get(r["subject"], 0) + 1
    r["trial"] = cnt[r["subject"]]

# ---- extra files: force profiles and hand paths, columns named by the final person codes (order of first appearance)
codes = {}
for r in rows:
    codes.setdefault(r["subject"], f"{ID}_P{len(codes) + 1:04d}")


def csv_bytes(header, mat):
    b = io.StringIO()
    b.write(",".join(header) + "\r\n")
    for i in range(mat.shape[0]):
        b.write(",".join("" if x != x else "%.8g" % x for x in mat[i]) + "\r\n")
    return b.getvalue().encode()


files = []
for grp, fn, key in GROUPS:
    S = store[grp]
    hdr = ["sample"] + [codes[f"{grp}{p + 1}"] for p in range(20)]
    for f in S._fieldnames:
        a = getattr(S, f)
        if f.startswith(("force_profile_", "ideal_force_")) and f != "force_profile_pretrain":
            files.append((f"samples/{grp}_{f}.csv", csv_bytes(hdr, np.column_stack([np.arange(a.shape[0]), a.astype(float)]))))
        elif f.endswith(("handX", "handY")):
            files.append((f"samples/{grp}_{f}.csv", csv_bytes(hdr, np.column_stack([np.arange(a.shape[0]), a.astype(float)]))))
    fp = S.force_profile_pretrain
    cols = [fp[p][:, k].astype(float) for p in range(20) for k in range(fp[p].shape[1])]
    hdr2 = ["sample"] + [f"{codes[f'{grp}{p + 1}']}_pre{k + 1}" for p in range(20) for k in range(fp[p].shape[1])]
    files.append((f"samples/{grp}_force_profile_pretrain.csv", csv_bytes(hdr2, np.column_stack([np.arange(1201)] + cols))))

# ---- checks against the paper
def grp_stats(grp, f):
    m = [r["value"] for r in rows if r["group"] == grp and r["measure"] == f]
    return np.mean(m), np.std(m, ddof=1) / np.sqrt(len(m)), len(m)


def part_means(grp, name, n):
    out = []
    for p in range(20):
        v = [r["value"] for r in rows if r["subject"] == f"{grp}{p + 1}" and r["measure"] == name]
        out.append(np.mean(v))
    return out


SRC = "Results, 'Motor Learning: Force-Field Perturbations versus Explicit Visual Feedback'"
rep = {"FF": dict(base=(0.0012, 0.0053), pre=(-0.015, 0.0066), early=(0.12, 0.019), mid=(0.51, 0.041), late=(0.63, 0.033)),
       "EVF": dict(base=(0.0056, 0.0051), pre=(0.0013, 0.0070), early=(-0.027, None), mid=(0.67, 0.14), late=(0.86, None))}
for g in ("FF", "EVF"):
    for nm, blk, key in (("baseline (mean of 8 error-clamp trials)", "adaptation_coefficient_baseline_", "base"),
                         ("pretraining (mean of 2 error-clamp trials)", "adaptation_coefficient_pretrain_", "pre")):
        pm = []
        for p in range(20):
            v = [r["value"] for r in rows if r["subject"] == f"{g}{p + 1}" and r["measure"].startswith(blk)]
            pm.append(np.mean(v))
        checks.published(f"{g}: mean adaptation coefficient, {nm}", np.mean(pm), rep[g][key][0], 6e-5 if key == "base" else 5e-4, SRC)
        checks.published(f"{g}: SEM of that mean", np.std(pm, ddof=1) / np.sqrt(20), rep[g][key][1], 6e-5, SRC)
    for key, blk in (("early", "adaptation_coefficient_early_train"), ("mid", "adaptation_coefficient_middle_train"),
                     ("late", "adaptation_coefficient_late_train")):
        m, s, n = grp_stats(g, blk)
        checks.published(f"{g}: mean adaptation coefficient, {key} training", m, rep[g][key][0], 0.005, SRC)
        if rep[g][key][1] is not None:
            checks.published(f"{g}: SEM, {key} training", s, rep[g][key][1], 0.0015, SRC)
checks.published("participants (20 per group, 40 in all)", len({r["subject"] for r in rows}), 40, 0, "Methods, Experimental Paradigm")
checks.internal("20 participants per group", len({r["subject"] for r in rows if r["group"] == "FF"}) == 20
                and len({r["subject"] for r in rows if r["group"] == "EVF"}) == 20)
_ac = [r["value"] for r in rows if r["measure"].startswith("adaptation")]
checks.internal("every value finite, adaptation coefficients within (-5, 5)", all(np.isfinite(r["value"]) for r in rows)
                and all(-5 < v < 5 for v in _ac), f"range {min(_ac):.2f} to {max(_ac):.2f}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Joiner (UC Davis)",
    data_url="https://doi.org/10.6084/m9.figshare.17096630", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor summary format in long form (one row per participant and value), participant codes "
            "replaced, missing (NaN) values left out, force profiles and hand paths written to samples/*.csv.",
    unit_note="participant and value (participant summary)",
    how=[
        "The authors shared processed arrays, not trial tables. Each row is one participant's adaptation coefficient (the "
        "slope of the lateral force on an error-clamp trial against the ideal velocity-dependent force; 1 means full "
        "compensation) at one point of the protocol: baseline error-clamp trials, the two pretraining trials, 11 points of the "
        "training time course, 27 points of decay and washout, 12 points of reexposure, and the authors' early, middle and late "
        "means. The meaning of the points was read from the paper and from identities between arrays (listed in the checks); "
        "points the authors do not describe are labelled as such and keep their array position (block, measure).",
        "group is FF (physical velocity-dependent force field, 15 N s/m) or EVF (explicit visual feedback of the same force, no "
        "physical field). The deposit names the FF group 'Control'; the paper's Methods say the FF group is the one with physical perturbations.",
        f"{n_missing} array entries are NaN in the deposit (trials excluded by the authors' speed and force criteria) and have no row.",
        "samples/ holds the participants' mean force profiles (1201 samples centred on peak velocity, 1000 Hz, N, plus the ideal "
        "force) and, for the EVF group, hand x, y (5000 samples) on the physical force-field trials; columns are the person "
        "codes of the main table. The deposit gives no units for the hand paths and displacements; the target distance (about 30 "
        "in y) and the paper's 20-30 cm targets suggest cm.",
        "Direction of the field or feedback (clockwise or counter-clockwise, counterbalanced) and the order of participants are "
        "not stated per participant in the deposit; participant k of a group is the k-th person of that group in the main table.",
    ],
    differences=[
        "The paper gives the EVF early-training SE as 0.093 and the late-training SE as 0.077; from the deposit they are 0.0093 "
        "(the paper's value is probably a typing slip) and 0.083 (17 of 20 participants have a late value), so only the means were checked for those."],
)
stdlib.release(ID, "summary", rows, meta, checks, extra_files=files)
