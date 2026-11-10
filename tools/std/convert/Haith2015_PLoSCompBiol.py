"""Haith, Huberdeau & Krakauer (2015), PLoS Comput Biol: forced-response reaching with target jumps (three experiments,
24 participant sessions), one row per reach in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Haith2015_PLoSCompBiol.py
Raw files (untrusted data, read only) in raw_extra/Haith2015_PLoSCompBiol/, from the Zenodo copy (record 4936553, CC0) of
the Dryad deposit https://doi.org/10.5061/dryad.63k6q: Expt1_Data.mat, Expt2_Data.mat, Expt3_Data.mat and the readmes.

Per trial the authors share the unfiltered finger path (X, Y in m, 130 Hz), the pre- and post-jump target (0 = East, 1 =
North-East, 2 = North ... in steps of 45 deg) and the time of the target jump. hand_angle is not in the files: it is
recomputed here with the paper's definition of initial reach direction (direction of tangential velocity 100 ms after
movement onset; onset = latest time before peak speed with speed below 0.02 m/s; 2nd-order Savitzky-Golay smoothing with a
half-width of 54 ms = 7 samples), in the screen frame, counter-clockwise positive, relative to the initial target.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Haith2015_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
FS = 130.0
START = (0.6, 0.23)
names = [c[1] for c in stdlib.reach90_columns()]


def num(x):
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


def wrap(a):
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def load(e):
    m = sio.loadmat(RAW / f"Expt{e}_Data.mat", squeeze_me=True, struct_as_record=False)
    d = m["data"]
    d = d.reshape(-1, 1) if d.ndim == 1 else d
    return d


def reach_measures(x, y):
    """onset time (s), initial direction (deg, screen frame) and extent, or None when the finger did not move."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    bad = np.where(np.isnan(x) | np.isnan(y))[0]
    if bad.size:  # Experiment 1's paths are padded with NaN after the end of the movement record
        x, y = x[:bad[0]], y[:bad[0]]
    if x.size < 20:
        return None
    xs = savgol_filter(x, 15, 2)
    ys = savgol_filter(y, 15, 2)
    vx = savgol_filter(x, 15, 2, deriv=1, delta=1 / FS)
    vy = savgol_filter(y, 15, 2, deriv=1, delta=1 / FS)
    sp = np.hypot(vx, vy)
    p = int(np.argmax(sp))
    below = np.where(sp[:p] < 0.02)[0]
    if below.size == 0:
        return None
    o = int(below[-1])
    k = min(o + int(round(0.1 * FS)), x.size - 1)
    extent = float(np.max(np.hypot(x - x[0], y - y[0])))
    return o / FS, math.degrees(math.atan2(vy[k], vx[k])), extent


rows, person_n, subj_n = [], {}, {}
stats = {}
GROUPS = {1: ["single condition"], 2: ["45deg jump", "135deg jump"], 3: ["No Barrier", "Barrier"]}
for e in (1, 2, 3):
    d = load(e)
    for i in range(d.shape[0]):
        pk = (e, i)
        for j in range(d.shape[1]):
            c = d[i, j]
            n = int(c.Ntrials)
            assert len(c.X) == n and len(c.Y) == n, (e, i, j)
            pre = np.asarray(c.target_pre_jump, int)
            post = np.asarray(c.target_post_jump, int)
            jt = np.asarray(c.jump_time, float)
            if np.nanmax(jt) > 50:  # Experiment 2's jump times are stored in ms, the others in s
                jt = jt / 1000.0
            cond = GROUPS[e][j]
            sub = f"{ID}_E{e}_S{i:02d}_C{j}"
            for k in range(n):
                jump = jt[k] > 0
                if jump != (pre[k] != post[k]):
                    stats["jump_flag_mismatch"] = stats.get("jump_flag_mismatch", 0) + 1
                ta = 45.0 * pre[k]
                meas = reach_measures(c.X[k], c.Y[k])
                row = dict.fromkeys(names, "")
                excl = meas is None or meas[2] < 0.05
                tj = wrap(45.0 * (post[k] - pre[k])) if jump else 0.0
                row.update(
                    dataset=ID, experiment=f"{ID}_Expt{e}", study=ID, subject=sub, person=f"{ID}_E{e}_P{i:02d}",
                    id_type="lab", person_link="unknown", prior_exposure="unknown",
                    subject_excluded_by_authors="False", handedness="right", device="other", setting="lab",
                    population="healthy", trial=str(k), trial_orig=str(k + 1), session="0", block=cond, phase="other",
                    trial_kind="reach", break_before="True" if k == 0 else "False",
                    target_angle=num(ta), target_id=str(pre[k]), target_size_deg=num(math.degrees(math.atan2(12.5, 80))),
                    target_distance_norm="1", target_visible_during_move="True",
                    target_event="jump" if jump else "none", target_jump=num(tj),
                    jump_time_ms=num((jt[k] - meas[0]) * 1000.0) if (jump and meas) else "",
                    probe_kind="none", instr_goal="hand_to_target", instr_strategy="unspecified", instr_disclosure="none",
                    instr_report="none", target_to_go_ms="1500", fb_cued="False", hand_used="right",
                    hand_measure="initial_direction", excluded_by_authors="True" if excl else "False",
                    fb_visible="online", fb_mode="veridical", fb_form="cursor", online_cursor="veridical",
                    fb_endpoint_transform_only="False", rotation="0", outcome_channels="",
                )
                if meas:
                    row["rt_ms"] = num(meas[0] * 1000.0)
                    row["hand_angle"] = num(wrap(meas[1] - ta))
                    row["hand_extent"] = num(meas[2] / 0.08)
                rows.append(row)
                stats.setdefault((e, i, j), []).append(
                    (jump, excl, None if not meas else wrap(meas[1] - ta), None if not meas else meas[0], int(pre[k]), tj))
            subj_n[sub] = n
            person_n[pk] = 1

# ---------------------------------------------------------------- checks
def per_exp(e):
    return {k: v for k, v in stats.items() if isinstance(k, tuple) and k[0] == e}

checks.internal("jump flag (jump_time > 0) agrees with pre != post jump target on every trial", stats.get("jump_flag_mismatch", 0) == 0,
                str(stats.get("jump_flag_mismatch", 0)))
checks.internal("rows = sum of trials in the three .mat files", len(rows) == sum(subj_n.values()), str(len(rows)))
for e, rep in ((1, 10), (2, 6), (3, 8)):
    checks.published(f"Experiment {e}: participants", len({k[1] for k in per_exp(e)}), rep, 0, "Methods, Participants / Experiment %d" % e)
checks.published("Experiment 3: trials per session (8 blocks of 100)", np.mean([len(v) for v in per_exp(3).values()]), 800, 0,
                 "Methods, Experiment 3 (two sessions, each 8 blocks of 100 trials)")
jump3 = [t[5] for v in per_exp(3).values() for t in v if t[0]]
checks.published("Experiment 3: largest absolute jump on any trial (deg)", max(abs(x) for x in jump3), 45, 0, "Methods, Experiment 3 ('always +-45')")
jump1 = sorted({abs(t[5]) for v in per_exp(1).values() for t in v if t[0]})
checks.published("Experiment 1: number of distinct jump sizes (45, 90, 135)", len(jump1), 3, 0, "Methods, Experiment 1")
allj = [t[0] for v in stats.values() if isinstance(v, list) for t in v]
checks.published("fraction of jump trials", np.mean(allj), 0.30, 0.03, "Methods, Experiment 1 ('About 30% of trials')")
tg2 = {}
for (e, i, j), v in per_exp(2).items():
    tg2.setdefault(j, set()).update(45 * t[4] for t in v)
sep = {j: max(s) - min(s) for j, s in tg2.items()}
checks.published("Experiment 2: target separation in the 45 deg condition", sep[0], 45, 0, "Methods, Experiment 2")
checks.published("Experiment 2: target separation in the 135 deg condition", sep[1], 135, 0, "Methods, Experiment 2")

def sd_nojump(e, j, k=None):
    out = []
    for (ee, i, jj), v in per_exp(e).items():
        if jj != j:
            continue
        by = {}
        for jump, excl, ha, on, tid, tj in v:
            if not jump and not excl and ha is not None:
                by.setdefault(tid, []).append(ha)
        def trim(a):
            a = np.asarray(a)
            if k is None:
                return a
            md = np.median(a)
            sc = 1.4826 * np.median(np.abs(a - md))
            return a[np.abs(a - md) <= k * sc]
        sds = [np.std(trim(a), ddof=1) for a in by.values() if len(a) > 5]
        out.append(np.mean(sds))
    return float(np.mean(out))
sd_nb, sd_b = sd_nojump(3, 0, 5), sd_nojump(3, 1, 5)
for kk in (2, 2.5, 3, 4):
    print("trim", kk, "SD no barrier", sd_nojump(3, 0, kk), "barrier", sd_nojump(3, 1, kk))
print("SD no barrier", sd_nb, "barrier", sd_b)
checks.published("Experiment 3: SD of initial reach direction, no-jump trials, no barrier, outliers beyond 5 robust SD removed (deg)", sd_nb, 4.8, 1.0, "Methods, Experiment 3, Variability")
checks.published("Experiment 3: SD of initial reach direction, no-jump trials, barrier, outliers beyond 5 robust SD removed (deg)", sd_b, 4.4, 1.0, "Methods, Experiment 3, Variability")
for e, rep in ((1, 4.5), (2, 4.7), (3, 2.75)):
    ex = [np.mean([t[1] for t in v]) * len(v) for v in per_exp(e).values()]
    print("excluded per sequence/participant exp", e, np.mean(ex))
ex1 = np.mean([sum(t[1] for t in v) for v in per_exp(1).values()])
print("exp1 excl", ex1)
mid = []
for (e, i, j), v in per_exp(1).items():
    on = [t[3] for t in v if not t[0] and not t[1] and t[3] is not None]
    mid.append(np.std(on, ddof=1) * 1000)
print("onset SD exp1", np.mean(mid))
checks.published("Experiment 1: mean of per-participant SD of movement onset time, no-jump trials (ms)", np.mean(mid), 79, 25, "Methods, Experiment 1, Movement onset")
checks.internal("all hand angles within (-180, 180]", all(-180 < float(r["hand_angle"]) <= 180 for r in rows if r["hand_angle"]))
# direction sanity: no-jump trials should point at the target (mean hand angle near 0 means the screen-frame sign is right)
mh = np.nanmean([abs(t[2]) for v in stats.values() if isinstance(v, list) for t in v if not t[0] and not t[1] and t[2] is not None])
print("mean abs hand angle on no-jump trials", mh)
checks.internal("no-jump, non-excluded trials: mean |hand_angle| < 15 deg (X, Y axes and the 0=East, counter-clockwise target coding agree)", mh < 15, f"{mh:.2f}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Haith & Krakauer (Johns Hopkins)", data_url="https://doi.org/10.5061/dryad.63k6q (Zenodo copy: https://doi.org/10.5281/zenodo.4936553)",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes="converted to the OpenMotor format, participant codes replaced; hand angle, reaction time and extent were computed from the shared finger paths (see How this file was made).",
    unit_note="trial of one participant and condition",
    how=[
        "One row is one reach of one participant in one condition: Experiment 1 (10 participants, one condition), Experiment 2 (6 participants, two target-separation conditions, 45 and 135 deg) and Experiment 3 (8 participants, no-barrier and barrier conditions); the condition is in block, and person links the two sequences of one participant in Experiments 2 and 3. Two Experiment 3 participants also took part in Experiment 1 (paper), but the files do not say which, so they have separate person codes.",
        "The authors shared the finger path of every trial, not a hand angle. hand_angle is the direction of tangential velocity 100 ms after movement onset (the paper's initial reach direction), computed from the unfiltered X and Y with a 2nd-order Savitzky-Golay filter (half-width 54 ms) and the onset rule of the paper (latest time before peak speed with speed under 0.02 m/s). It is in the screen frame, counter-clockwise positive, relative to the initial target (target_angle). Targets are coded 0 = East, 1 = North-East, 2 = North ... in the readme, so target_angle = 45 deg times the code; the sign of the Y axis was not stated, and was confirmed from the data (reaches on trials without a jump point at the coded target, mean |hand_angle| given in the checks).",
        "On the 30% of jump trials the target jumped by 45, 90 or 135 deg (Experiments 1 and 2: whichever the two locations allow) before movement onset; target_jump is the jump (post minus pre, counter-clockwise positive) and hand_angle stays relative to the initial target. jump_time_ms is the jump time minus the movement onset (negative = the jump came first), so the authors' re-preparation time is its negative. Experiment 2's jump times are stored in ms in the deposit and the others in s; they were put in the same unit.",
        "rt_ms is the movement onset time in ms from the start of the trial record; the fourth tone (the go signal) sounded at 1500 ms (readme), stored as target_to_go_ms. excluded_by_authors marks trials where the finger moved less than 5 cm (one of the authors' two exclusion rules; the other, excessive curvature, cannot be applied from the shared data and the trials excluded for it are not flagged). Path samples (130 Hz) are not included.",
    ],
    differences=["The SD of initial reach direction (Experiment 3) is 11 deg without outlier removal and about 4.5 deg after removing, per participant and target, trials more than 5 robust SDs (1.4826 x MAD) from the median; the paper says it rejected outlier points but not by which rule, so the published 4.8 and 4.4 deg are reproduced only with such a rule (a tolerance of 1 deg is used). Mean movement-onset SD and the number of excluded trials per participant are recomputed by us from the unfiltered paths with the paper's rules, and may differ slightly from the published values (see the checks)."],
)
stdlib.release(ID, "reach90", rows, meta, checks)
