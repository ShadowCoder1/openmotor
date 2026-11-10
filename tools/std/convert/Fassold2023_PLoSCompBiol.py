"""Fassold, Locke & Landy (2023), PLoS Comput Biol: sensorimotor confidence, stylus reaches to a target on a Wacom tablet seen through a mirror.
A control task (session 1: 300 reaches to one target, with the felt endpoint reported) and the main task (sessions 2 to 4: 3 x 300 reaches
to targets on an arc, triplets of two feedback reaches and one confidence reach). 14 participants, one row per reach, module endpoint_reaching.

Run from the site folder: python3 -I tools/std/convert/Fassold2023_PLoSCompBiol.py
Raw files (untrusted data, read only) in raw_extra/Fassold2023_PLoSCompBiol/, from https://osf.io/xrv65/ (folder data): the 14 files
<participant>_output.mat (named by initials, never reproduced). Each holds, among others, rawTarg and rawEndpoints (900 x 2: target and
endpoint of every main-task reach in tablet millimetres, in trial order), contTar (1 x 2), contReach and contInd (300 x 2: the control
target, the reach endpoints and the endpoints the participant indicated), endPoints (300 x 2), confCirc, pointsEarned, pointsPoss.
The authors' MATLAB files (analysis_code, experiment_code) were read as text, never run.
Not converted: endPoints (the 300 confidence-trial errors rotated into a frame the deposit does not describe: their distances to the
target equal those of rawEndpoints - rawTarg on every third reach, which is checked below), endPtsFB, confCirc, pointsEarned and pointsPoss
(the deposit does not say which reaches they belong to: they do not line up with endPoints), the model simulations and fits, and the
two participants' files that the deposit does not have.
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Fassold2023_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
CORE_NAMES = [c[1] for c in stdlib.CORE]
MOD_NAMES = [c[1] for c in stdlib.load_module("endpoint_reaching")]
T_COMMON = np.array([238.0, 134.0])  # tablet centre used by the authors' analysis (participantDataFit.m); only for the check below

files = sorted(RAW.glob("*_output.mat"))
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("14 participant files, all with the downloaded hash", len(files) == 14 and all(stdlib.sha256(p) == src[p.name]["sha256"] for p in files))


def fv(x):
    return "%.8g" % float(x)


rows = []
ok_shape = ok_triplet = ok_bounds = True
n_control = []
for p in files:
    m = sio.loadmat(p)
    sub = f"{ID}_{len(set(r['subject'] for r in rows)) + 1:02d}"
    raw_e, raw_t = m["rawEndpoints"], m["rawTarg"]
    cr, ci, ct = m["contReach"], m["contInd"], m["contTar"].ravel()
    nc = len(cr)
    n_control.append(nc)
    ok_shape &= raw_e.shape == (900, 2) and raw_t.shape == (900, 2) and cr.shape == (nc, 2) and ci.shape == (nc, 2) and ct.shape == (2,)
    # the endPoints errors sit on every third reach (positions 3, 6, ... of the 900): their distances to the common target equal the
    # raw error distances there, so the third reach of each triplet is the confidence reach
    d_ep = np.sort(np.hypot(*(m["endPoints"] - T_COMMON).T))
    d_raw = np.sort(np.hypot(*(raw_e - raw_t)[2::3].T))
    ok_triplet &= bool(np.abs(d_ep - d_raw).max() < 1e-6)
    allxy = np.vstack([raw_e, raw_t])
    ok_bounds &= bool(allxy.min() >= -10 and allxy[:, 0].max() <= 486 and allxy[:, 1].max() <= 278)
    base = dict(dataset=ID, study=ID, group="", subject=sub, person=sub, subject_excluded_by_authors="False", age="", sex="",
                handedness="right", population="healthy", setting="lab", device="Wacom Cintiq 22 tablet, mirror and projector",
                phase="test", task="reaching", unit="trial", pert_type="none", pert_value="0", pert_unit="", excluded_by_authors="False")
    for k in range(nc):
        rows.append({**{c: "" for c in CORE_NAMES + MOD_NAMES}, **base, "experiment": f"{ID}_control", "session": "1",
                     "block": f"block{k // 60 + 1}", "trial": str(k + 1), "condition": "control_report", "feedback": "none",
                     "trial_role": "control_test", "target_x_mm": fv(ct[0]), "target_y_mm": fv(ct[1]),
                     "endpoint_x_mm": fv(cr[k, 0]), "endpoint_y_mm": fv(cr[k, 1]),
                     "reported_x_mm": fv(ci[k, 0]), "reported_y_mm": fv(ci[k, 1])})
    for k in range(900):
        pos = k % 3
        rows.append({**{c: "" for c in CORE_NAMES + MOD_NAMES}, **base, "experiment": f"{ID}_main", "session": str(2 + k // 300),
                     "block": f"block{(k % 300) // 60 + 1}", "trial": str(k % 300 + 1), "condition": "confidence" if pos == 2 else "feedback",
                     "feedback": "none" if pos == 2 else "normal", "trial_role": "confidence_trial" if pos == 2 else "feedback_trial",
                     "triplet_position": str(pos + 1), "target_x_mm": fv(raw_t[k, 0]), "target_y_mm": fv(raw_t[k, 1]),
                     "endpoint_x_mm": fv(raw_e[k, 0]), "endpoint_y_mm": fv(raw_e[k, 1])})

n_p = len(set(r["subject"] for r in rows))
checks.internal("every file has 900 main-task reaches and equally many control reaches and reported endpoints (300 in 13 files, 240 in one)", ok_shape and sorted(n_control) == [240] + [300] * 13)
checks.internal("on every third reach the raw error distance equals the distance of the deposit's endPoints (confidence trials), "
                "all 14 participants", ok_triplet)
checks.internal("all targets and endpoints lie within 10 mm of the 476 x 268 mm tablet", ok_bounds)
checks.internal("rows = 14 x 900 + control reaches", len(rows) == 14 * 900 + sum(n_control), f"{len(rows)} rows")
checks.published("control-task test reaches, most in any participant", max(n_control), 300, 0,
                 "Methods, Control motor-awareness task (five blocks of 60 test trials)")
checks.published("main-task reaches per participant (sessions 2 to 4, 300 trials each)",
                 sum(r["experiment"].endswith("main") for r in rows) / n_p, 900, 0, "Methods, Main task (300 experimental trials per session, sessions 2 to 4)")
checks.published("main-task sessions per participant", len({r["session"] for r in rows if r["experiment"].endswith("main")}), 3, 0,
                 "Methods, Main task (sessions 2 to 4)")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Landy (NYU)",
    data_url="https://osf.io/xrv65/",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format; participant codes replaced; only the target and endpoint coordinates (and the reported "
            "endpoints of the control task) were kept.",
    unit_note="reach of one participant",
    how=[
        "One row is one stylus reach on a tablet, in the tablet's own millimetre coordinates: 300 control-task reaches (240 for one participant) (session 1; a single target; the "
        "participant also reported where the reach ended) and 900 main-task reaches (sessions 2 to 4; targets on an arc around the start, in "
        "triplets: two reaches with endpoint feedback and a third without feedback, followed by a confidence report).",
        "The deposit stores the main task as rawTarg and rawEndpoints in trial order. Sessions are the three consecutive blocks of 300 and "
        "triplets are the consecutive groups of three (positions 1 and 2 feedback, position 3 confidence); both follow the paper's design and "
        "were confirmed from the data: the distances of the deposit's endPoints (the confidence-trial errors) equal the raw error distances on every "
        "third reach. Blocks of 60 are counted from the position within the session (five per session, as the Methods say).",
        "The confidence circle sizes and points the deposit also holds are not included: the deposit does not say which reaches they belong to "
        "and they do not line up with the confidence-trial endpoints. The coordinate units (mm) follow from the tablet size (476 x 268 mm) "
        "given in the paper; the deposit does not state them. Age and sex were not shared per participant.",
    ],
    differences=[
        "One participant's control task has 240 reaches (the paper says 300 test trials); rows are kept as shared.",
        "The paper reports 16 participants; the OSF deposit holds files for 14, so counts, means and fitted parameters computed here cannot "
        "match the paper's (for example the mean reach error of the confidence trials is 20.2 mm here against 20.57 mm in the paper, and the mean "
        "confidence circle 27.0 mm against 26.93 mm).",
    ],
)
stdlib.release(ID, "endpoint_reaching", rows, meta, checks)
