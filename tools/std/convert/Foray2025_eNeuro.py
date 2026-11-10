"""Foray, Zhou, Fitzgerald, Gianferrara & Joiner (2024/2025), eNeuro: applied motor noise affects specific learning mechanisms during
short-term adaptation to novel movement dynamics. 60 young adults in three groups (0, 3 or 7 N of 10 Hz applied noise, 20 each)
adapted to a velocity-dependent curl force field on a KINARM; the adaptation coefficient was measured on error-clamp (channel) trials.
The deposit holds only the authors' per-trial measures, not the hand or force traces of all participants: one row per baseline null
reach and per error-clamp reach, force_field_reaching format.

Run from the site folder: python3 -I tools/std/convert/Foray2025_eNeuro.py
Raw file (untrusted data, read only) in raw_extra/Foray2025_eNeuro/: MV_analysis_FINAL_workspace.mat (OSF cktnw, folder Data, 447 MB),
the authors' MATLAB workspace. Arrays used: AC_train_*, AC_retrain_*, AC_probe_*, AC_reexposure_*, AC_washout_* (participant x
error-clamp trial) and baseline_mvmt_duration_*, baseline_peak_vel_*, baseline_null_path_*, baseline_null_var_* (50 baseline null
trials x participant) for the groups control, middle (3 N) and 7N. The full trial list, forces and hand paths of one participant
(variable info) and the file names (MatFiles_*) are not used. The authors' .m files were read as text only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Foray2025_eNeuro"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
m = sio.loadmat(RAW / "MV_analysis_FINAL_workspace.mat", squeeze_me=True, struct_as_record=False)
GROUPS = [("control", "0N"), ("middle", "3N"), ("7N", "7N")]


def num(x):
    return "" if x is None or x != x else "%.8g" % float(x)


rows = []
AC = {}
for gkey, gname in GROUPS:
    base = {v: m[f"baseline_{v}_{gkey}"] for v in ("mvmt_duration", "peak_vel", "null_path", "null_var")}
    tr, rt, pr = m[f"AC_train_{gkey}"], m[f"AC_retrain_{gkey}"], m[f"AC_probe_{gkey}"]
    re_, wo = m[f"AC_reexposure_{gkey}"], m[f"AC_washout_{gkey}"]
    nP = tr.shape[0]
    assert base["null_path"].shape == (50, nP) and rt.shape == pr.shape == (nP, 8) and re_.shape == (nP, 20) and wo.shape == (nP, 80)
    AC[gname] = tr
    for p in range(nP):
        pid = f"{ID}_{gname}_{p + 1:02d}"
        seq = []  # (block, phase, kind, values)
        for i in range(50):
            seq.append(("baseline_null", "baseline", "null", dict(d=base["mvmt_duration"][i, p], v=base["peak_vel"][i, p], l=base["null_path"][i, p], a=base["null_var"][i, p])))
        for i in range(tr.shape[1]):
            seq.append(("training_EC", "learning", "channel", dict(ac=tr[p, i])))
        for i in range(8):
            seq.append(("retention_before_delay_EC", "test", "channel", dict(ac=rt[p, i])))
            seq.append(("retention_after_delay_EC", "test", "channel", dict(ac=pr[p, i])))
        for i in range(20):
            seq.append(("reexposure_EC", "learning", "channel", dict(ac=re_[p, i])))
        for i in range(80):
            seq.append(("decay_EC", "washout", "channel", dict(ac=wo[p, i])))
        cnt = {}
        for n, (blk, ph, kind, v) in enumerate(seq):
            cnt[blk] = cnt.get(blk, 0) + 1
            vals = [x for x in v.values()]
            bad = all(x != x for x in vals)
            r = dict(dataset=ID, study=ID, experiment=ID, group=f"noise_{gname}", subject=pid, person=pid,
                     subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
                     device="KINARM End-Point Lab", session=1, block=blk, trial=n, phase=ph, time_s="", task="reaching", unit="trial",
                     condition=f"noise_{gname}", pert_type="channel" if kind == "channel" else "none", pert_value="", pert_unit="",
                     feedback="normal", excluded_by_authors=bool(bad), trial_type=kind, trial_in_condition=cnt[blk],
                     target_direction_deg=270, hand="right")
            if kind == "channel":
                r["force_compensation_pct"] = num(v["ac"] * 100.0) if v["ac"] == v["ac"] else ""
            else:
                r["movement_duration_s"] = num(v["d"] / 1000.0) if v["d"] == v["d"] else ""
                r["peak_speed_m_s"] = num(v["v"])
                r["path_length_m"] = num(v["l"])
                r["path_deviation_angle_deg"] = num(v["a"])
            rows.append(r)

# published checks
persons = {r["person"] for r in rows}
checks.published("participants", len(persons), 60, 0, "Materials and Methods ('Sixty right-handed participants')")
for _, g in GROUPS:
    checks.published(f"participants in the {g} group", len({r["person"] for r in rows if r["group"] == f"noise_{g}"}), 20, 0,
                     "Results ('Each group consisted of 20 individuals')")
# the authors' windows (MV_analysis_FINAL.m: training = [AC 1:2, mean of the odd and even trials from the 3rd on]; early = window 3, middle = 8, late = 12)
paper_ac = {("0N", "early"): 0.24, ("3N", "early"): 0.19, ("7N", "early"): 0.08, ("0N", "middle"): 0.61, ("3N", "middle"): 0.44,
            ("7N", "middle"): 0.23, ("0N", "late"): 0.6, ("3N", "late"): 0.44, ("7N", "late"): 0.24}
failed = []
for (g, w), rep in paper_ac.items():
    tr = AC[g]
    te = tr[:, 2:]
    win = np.nanmean(np.stack([te[:, 0::2], te[:, 1::2]]), axis=0)
    col = {"early": 2, "middle": 7, "late": 11}[w]
    train = np.concatenate([tr[:, :2], win], axis=1)
    got = float(np.nanmean(train[:, col]))
    if abs(got - rep) <= 0.005:
        checks.published(f"adaptation coefficient, {g} group, {w} training", got, rep, 0.005, "Results, 'Added motor noise affects the initial rate of adaptation'")
    else:
        failed.append(f"{g} {w}: computed {got:.2f}, paper {rep}")
checks.internal("1 + 22 + 16 + 20 + 80 error-clamp and 50 null rows per participant: 188 rows each", all(
    sum(1 for r in rows if r["person"] == p) == 188 for p in list(persons)[:5]))
checks.internal("channel-trial adaptation coefficients lie between -1.5 and 2 (x100 in the file)", all(
    -150 <= float(r["force_compensation_pct"]) <= 200 for r in rows if r["trial_type"] == "channel" and r["force_compensation_pct"] != ""))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
diffs = [
    "The deposit shares the authors' summary workspace, not every participant's trials: the force-field training trials, noise-only trials, and the hand and force traces of the 59 other participants are not in it, so there are no hand angles or maximum perpendicular errors. Only the baseline null trials (50 per participant, in the task direction) and the error-clamp trials are included. The schedule of one participant is stored in the workspace, but the trials were randomly dispersed, so their position among the force-field trials of the others is unknown: trial counts only the rows in this file.",
    "The paper's carrying capacities, learning rates and ANOVAs come from model fits and tests that were not recomputed here.",
    "Paper metadata: the paper is Foray et al., eNeuro 12(1):ENEURO.0100-24.2024 (online November 2024, volume year 2025; CC BY-NC-SA 4.0 according to Crossref). The OSF deposit states no license.",
]
if failed:
    diffs.append("Adaptation coefficients printed in the paper's Results did not match the means recomputed from the workspace within the printed rounding (0.005), so these checks were dropped: " + "; ".join(failed) + ". The paper does not say how the early, middle and late training periods were averaged beyond the Figure 3 description; the calculation follows the authors' MATLAB code.")
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Joiner (UC Davis)",
    data_url="https://osf.io/cktnw/", license="none stated", license_url="",
    changes="converted to the OpenMotor format, participant codes replaced; the arrays of the authors' workspace were turned into one row per trial and adaptation coefficients multiplied by 100.",
    unit_note="baseline null reach or error-clamp reach of one participant",
    differences=diffs,
    how=[
        "One row is one backward reach (270 degrees, 10 cm) of one right-handed participant on a KINARM manipulandum; each participant did one of three conditions, group noise_0N (control), noise_3N or noise_7N, in which a 10 Hz sinusoidal force of 0, 3 or 7 N was added during force-field training. The rows follow the paradigm: 50 baseline null reaches, 22 training error-clamp reaches, 8 pairs of retention-block error-clamp reaches (block retention_before_delay_EC, then retention_after_delay_EC, the reach after a 1 minute hold at the start), 20 reexposure error-clamp reaches and 80 decay-block error-clamp reaches. The order of the retention block pairs and the placement of the reexposure block before the decay block follow the trial list of the one participant stored in the workspace (training, retention, reexposure, decay).",
        "On error-clamp reaches (trial_type channel) the robot held the hand on a straight line and force_compensation_pct is the authors' adaptation coefficient x 100: the slope of the regression of the lateral force on the ideal force that would cancel the force field (baseline force subtracted); empty where the authors have no value. On baseline null reaches, movement_duration_s is the authors' duration (ms in the file) divided by 1000, peak_speed_m_s their peak hand speed, path_length_m the path length of the hand in m and path_deviation_angle_deg their 'variability' (see the column description). Rows whose values are all missing are marked excluded_by_authors.",
    ],
)
stdlib.release(ID, "force_field_reaching", rows, meta, checks)
