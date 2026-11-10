"""Mathew, Lefevre & Crevecoeur (2021), eNeuro: savings in force field learning. 16 right-handed adults in two groups of 8 (G1
counter-clockwise curl field, G2 clockwise), 415 forward reaches each (15 baseline, 180 adaptation, 40 washout, 180
re-adaptation), one row per reach in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Mathew2021_eNeuro.py
Raw file (untrusted data, read only): raw_extra/Mathew2021_eNeuro/PublicationData_v3.zip (Zenodo 4570064), the .mat files in
Matfiles/ and its ReadMe.txt. The authors' ScriptFig2-4.m were read as text only (never run) to learn what the matrices are.
ThetaG1/ThetaG2 = initial reach angle (G1 / G2); PL/PLb = path length; R4/R4b = correlation of measured and commanded force;
XminCCW/XmaxCW = maximum lateral deviation; XoverCCW/XoverCW = lateral target overshoot; Fmin/Fminb = peak terminal force.
Suffix b = the clockwise-field group (G2). Sign of the angle: the paper says positive is to the left of the start-target line
(counter-clockwise); checked here against the stored hand paths (Hx*, Hy*): leftward paths have positive ThetaG.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Mathew2021_eNeuro"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
z = zipfile.ZipFile(RAW / "PublicationData_v3.zip")


def L(n):
    d = sio.loadmat(io.BytesIO(z.read(f"PublicationData_v3/Matfiles/{n}.mat")), squeeze_me=True)
    return np.asarray([v for k, v in d.items() if not k.startswith("__")][0], dtype=float)


G = {  # group -> (label, field, matrices)
    "G1": ("CCW", dict(theta="ThetaG1", pl="PL", r="R4", xdev="XminCCW", xov="XoverCCW", f="Fmin")),
    "G2": ("CW", dict(theta="ThetaG2", pl="PLb", r="R4b", xdev="XmaxCW", xov="XoverCW", f="Fminb")),
}
BLK = [(0, 15, "Baseline", "baseline"), (15, 195, "Adaptation", "learning"), (195, 235, "Washout", "washout"),
       (235, 415, "Readaptation", "relearning")]
rows, supp = [], []
sub = 0
data = {}
for g, (lab, mm) in G.items():
    M = {k: L(v) for k, v in mm.items()}
    data[g] = M
    assert all(a.shape == (8, 415) for a in M.values())
    for p in range(8):
        sub += 1
        for lo, hi, bname, phase in BLK:
            for t in range(lo, hi):
                th = M["theta"][p, t]
                anyv = any(np.isfinite(M[k][p, t]) for k in M)
                row = dict.fromkeys(names, "")
                row.update(
                    dataset=ID, experiment="force_field_savings", study=ID, subject=f"{g}_{p+1}", person=f"{g}_{p+1}",
                    id_type="lab", person_link="assumed_unique", group=f"{g}_{lab}_field", prior_exposure="unknown",
                    subject_excluded_by_authors="False", handedness="right", population="healthy", device="kinarm",
                    setting="lab", trial=t, trial_orig=t + 1, session=0, block=bname, phase=phase,
                    trial_kind="reach" if anyv else "missing", break_before="True" if t == 0 else "False",
                    target_angle=90, target_distance_norm=1, target_event="none", target_jump=0, probe_kind="none",
                    instr_goal="cursor_to_target", hand_used="right", hand_measure="initial_direction",
                    hand_angle=th if np.isfinite(th) else "", excluded_by_authors="False" if anyv else "True",
                    fb_visible="online", fb_mode="veridical", fb_form="cursor", online_cursor="veridical",
                    fb_endpoint_transform_only="False", rotation=0)
                rows.append(row)
                supp.append((f"{ID}_S{sub:04d}", g, t + 1, M["pl"][p, t], M["r"][p, t], M["xdev"][p, t], M["xov"][p, t], M["f"][p, t]))

n_missing = sum(r["trial_kind"] == "missing" for r in rows)
checks.internal("16 sequences x 415 reaches = 6640 rows", len(rows) == 6640)
checks.internal("hand angles within (-90, 90)", all(-90 < float(r["hand_angle"]) < 90 for r in rows if r["hand_angle"] != ""))
checks.internal("missing reach rows are only those with every measure empty", n_missing == sum(
    not any(np.isfinite(data[g][k][p, t]) for k in data[g]) for g in data for p in range(8) for t in range(415)))


def col(g, k, t):
    return data[g][k][:, t - 1]


checks.published("participants (two groups of 8)", len({r["person"] for r in rows}), 16, 0, "Methods, Participants ('sixteen')")
checks.published("trials per participant", 415, 415, 0, "Methods, Task (15 + 180 + 40 + 180 trials)")
checks.published("initial reach angle G1, first adaptation trial (deg)", np.nanmean(col("G1", "theta", 16)), 20.9, 0.1, "Results, initial reach angle")
checks.published("initial reach angle G1, first readaptation trial (deg)", np.nanmean(col("G1", "theta", 236)), 19.2, 0.1, "Results, initial reach angle")
checks.published("initial reach angle G2, first adaptation trial (deg)", np.nanmean(col("G2", "theta", 16)), -11.47, 0.1, "Results, initial reach angle")
checks.published("initial reach angle G2, first readaptation trial (deg)", np.nanmean(col("G2", "theta", 236)), -11.5, 0.1, "Results, initial reach angle")
pl = lambda t: np.nanmean(np.r_[col("G1", "pl", t), col("G2", "pl", t)])
checks.published("path length, first adaptation trial, both groups (cm)", pl(16), 31.8, 0.15, "Results, first trial of adaptation vs readaptation")
checks.published("path length, first readaptation trial, both groups (cm)", pl(236), 25.2, 0.15, "Results, first trial of adaptation vs readaptation")
xd = lambda t: np.nanmean(np.r_[-(col("G1", "xdev", t) - 5), col("G2", "xdev", t) - 5])
checks.published("maximum lateral deviation, first adaptation trial (cm)", xd(16), 7.9, 0.1, "Results (authors' script: sign flipped for G1, minus 5)")
checks.published("maximum lateral deviation, first readaptation trial (cm)", xd(236), 6.3, 0.1, "Results")
xo = lambda t: np.nanmean(np.r_[col("G1", "xov", t) - 5, -(col("G2", "xov", t) - 5)])
checks.published("lateral target overshoot, first adaptation trial (cm)", xo(16), 2.18, 0.02, "Results")
checks.published("lateral target overshoot, first readaptation trial (cm)", xo(236), 0.67, 0.02, "Results")
pf = lambda t: np.nanmean(np.r_[-col("G1", "f", t), -col("G2", "f", t)])
checks.published("peak terminal force, first adaptation trial (N)", pf(16), 2.67, 0.02, "Results (stored value times -1)")
checks.published("peak terminal force, first readaptation trial (N)", pf(236), 1.58, 0.02, "Results (stored value times -1)")
rr = lambda t: np.nanmean(np.r_[col("G1", "r", t), col("G2", "r", t)] ** 2)
checks.published("force correlation (squared), first adaptation trial", rr(16), 0.42, 0.01, "Results (R squared)")
checks.published("force correlation (squared), first readaptation trial", rr(236), 0.60, 0.01, "Results (R squared)")

sb = io.StringIO()
sb.write("subject,group,trial_orig,path_length_cm,force_correlation_r,lateral_deviation_raw,lateral_overshoot_raw,peak_terminal_force_raw\r\n")
for s in supp:
    sb.write(",".join([s[0], s[1], str(s[2])] + ["" if not np.isfinite(v) else format(float(v), ".8g") for v in s[3:]]) + "\r\n")
sup_txt = sb.getvalue()

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Crevecoeur & Lefevre (UCLouvain)", data_url="https://doi.org/10.5281/zenodo.4570064",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="the authors' matrices (participants x trials) were turned into one table with one row per reach; hand_angle is their "
            "initial reach angle; the other per-trial measures are in trial_measures.csv; participant codes replaced.",
    unit_note="reach of one participant",
    how=[
        "One row is one forward reach (15 cm, KINARM) of one of 16 participants: G1 (8 people) met only a counter-clockwise curl "
        "force field, G2 (8 people) only a clockwise one (13 N s/m, velocity dependent, per the paper). Each did 415 reaches: "
        "15 baseline, 180 adaptation, 40 washout, 180 re-adaptation (the authors' trial ranges 1-15, 16-195, 196-235, 236-415). "
        "The field is off in baseline and washout and on in the other blocks; the reach90 format has no force-field column, so the "
        "field direction is in group (G1_CCW_field / G2_CW_field) and the field state in block.",
        "hand_angle is the authors' initial reach angle (matrices ThetaG1 / ThetaG2) relative to the straight line from start to "
        "target; target_angle is 90 (straight ahead = up on the screen, an assumption from the forward reach). The paper defines "
        "positive angles as to the left of that line, which is counter-clockwise for a reach straight ahead; this was also checked "
        "against the stored first-trial hand paths (paths that curve left have positive angles). Trials without an angle but "
        "with another measure stay as reach rows with an empty angle; trials with no value at all are trial_kind missing.",
        "trial_measures.csv holds the other per-trial values, unchanged: path length (cm), the correlation (r) between commanded "
        "and measured force, and the raw maximum lateral deviation, lateral overshoot (the authors' scripts subtract 5 from both, "
        "and flip the sign of the deviation for G1 and of the overshoot for G2) and peak terminal force (stored negative; the paper "
        "reports its magnitude). The deposit does not give units for the deviation and overshoot; the paper reports them in cm.",
        "The paper describes two groups of 8 (16 people), so rows 1-8 of the authors' counter-clockwise matrices and rows 1-8 of "
        "their clockwise matrices (suffix b) are coded as 16 different people. The authors' own scripts analyse the two as paired "
        "columns of the same 8 rows; the deposit does not say which reading is right, and the paper's 16 participants was followed.",
    ],
    differences=["The authors' stored x-deviation and overshoot values include an offset of 5 (their scripts subtract it); the "
                 "raw numbers are kept in trial_measures.csv.",
                 "Hand paths, EMG and force traces in the deposit (first and last trials of each block, 8 people per group) are not "
                 "converted; they are not trial-by-trial."],
)
stdlib.release(ID, "reach90", rows, meta, checks, extra_files=[("trial_measures.csv", sup_txt)])
