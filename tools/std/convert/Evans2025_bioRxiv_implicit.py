"""Evans, Taylor & Landy (2025), bioRxiv: implicit adaptation's effect on sensorimotor and motor confidence. 20
participants, two sessions each of slicing reaches to one target with a sinusoidal error clamp (+-10 deg, 12 cycles per
240 trials), one row per reach, in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Evans2025_bioRxiv_implicit.py
Raw file (untrusted data, read only) in raw_extra/Evans2025_bioRxiv_implicit/: implicitAdaptation_repo.zip, the GitHub
repository marissahevans/implicitAdaptation at commit 3cb6dfd (one archive, so that no URL or file name with participant
initials or session dates enters the record). Read from it: data/<folder>/*_errresults.mat (one file per run; a session
is sometimes split over two or three runs) and the authors' processed *_sensorimotor.mat / *_motoraware.mat, which tell
which run each analysed trial came from. analysisFiles/dataProcessingSaveFiles.m and experimentalFiles/errorClampTrialSeq.m
were read as text, never run. Folder names (initials) and dates are used only to order runs and never written.

Signs. endPtAngle is the direction of the endpoint from the start, from the +x axis of the screen with y up (it equals
atan2 of the stored screen endpoint on every trial), target straight ahead at 90 deg: hand_angle = endPtAngle - 90,
counter-clockwise positive, and the clamp is errorClamp - 90 in the same frame (the cursor is drawn there). The stored
reportAngle is already in that frame: the experiment code saves it as '-reportAngle; %negative to match reach angles', the
stored marker positions on the screen give the same angle, and in the no-feedback pre-test it rises with the hand in most
runs. hand_report = reportAngle. (The authors' dataProcessingSaveFiles.m negates it once more, see differences.)
"""
import io
import json
import math
import re
import statistics as st
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Evans2025_bioRxiv_implicit"
RAW = stdlib.raw_dir(ID)
ARCHIVE = "implicitAdaptation_repo.zip"
checks = stdlib.Checks()

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal(f"{ARCHIVE} matches the downloaded hash", stdlib.sha256(RAW / ARCHIVE) == src[ARCHIVE]["sha256"])
Z = zipfile.ZipFile(RAW / ARCHIVE)
mats = [n for n in Z.namelist() if "/data/" in n and n.endswith(".mat")]
folders = sorted({n.split("/data/")[1].split("/")[0] for n in mats})
checks.published("participants", len(folders), 20, 0, "Methods, Participants ('Twenty individuals'; also the deposit README)")


def load(n):
    return sio.loadmat(io.BytesIO(Z.read(n)), squeeze_me=True, struct_as_record=False)


def run_key(n):
    m = re.search(r"_S(\d)_(\d{4}-\d\d-\d\d)_(\d\d-\d\d)_errresults\.mat$", n)
    return m.group(2), m.group(3), int(m.group(1))


def wrap(a):
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def r8(x):
    return float(format(x, ".8g"))


names = [c[1] for c in stdlib.reach90_columns()]
rows = []
frame_err = 0.0
proc_dev = 0.0
pre_r = []
clamp_ok = True
for pi, f in enumerate(folders, start=1):
    runs = []
    for n in sorted((x for x in mats if f"/data/{f}/" in x and x.endswith("_errresults.mat")), key=run_key):
        r = load(n)["errResultsMat"]
        tn = np.asarray(r.trialNum).astype(int)
        rep = np.atleast_1d(np.asarray(r.reportAngle, float))
        arc = np.asarray(r.reportArc, float).reshape(-1, 2) if np.asarray(r.reportArc).size else np.zeros((0, 2))
        runs.append(dict(s=run_key(n)[2], tn=tn, ep=np.asarray(r.endPtAngle, float), ec=np.asarray(r.errorClamp, float), rep=rep,
                         arc=arc, rt=np.asarray(r.RTs, float), mt=np.asarray(r.MTs, float),
                         pts=np.asarray(r.pointsEarned, float).ravel(), xy=np.asarray(r.wacScreenEnd, float).reshape(-1, 2),
                         start=np.asarray(r.startPos, float), tgt=tuple(np.asarray(r.targetLoc).tolist()), dist=int(r.targetDist)))
    # which run the authors took each analysed trial from (their processed files), and so which session is which task
    task_of_session, used = {}, set()
    for task in ("sensorimotor", "motoraware"):
        m = load(next(x for x in mats if f"/data/{f}/" in x and x.endswith(f"_{task}.mat")))
        trial = np.asarray(m["trial"]).astype(int).ravel()
        endpt = np.asarray(m["endPt"], float).ravel()
        cand = [[i for i, x in enumerate(runs) if t <= len(x["tn"]) and x["tn"][t - 1] == t] for t in trial]
        off = np.median([runs[i]["ep"][t - 1] - 90 - e for c, t, e in zip(cand, trial, endpt) for i in c])
        pick = [min(c, key=lambda i: abs(runs[i]["ep"][t - 1] - 90 - e - off)) for c, t, e in zip(cand, trial, endpt)]
        proc_dev = max(proc_dev, max(abs(runs[i]["ep"][t - 1] - 90 - e - off) for i, t, e in zip(pick, trial, endpt)))
        clamp_ok &= trial.tolist() == list(range(61, 301))
        for i, t in zip(pick, trial):
            used.add((i, t))
        sess = {runs[i]["s"] for i in pick}
        assert len(sess) == 1, (pi, task, sess)
        task_of_session[sess.pop()] = "sensorimotor" if task == "sensorimotor" else "motor_awareness"
    assert len(task_of_session) == 2
    later = {}
    for i, x in enumerate(runs):
        for t in x["tn"][x["tn"] != 0]:
            later.setdefault((x["s"], int(t)), []).append(i)
    for i, x in enumerate(runs):
        task = task_of_session.get(x["s"], "extra_session")
        first = True
        for k in np.nonzero(x["tn"])[0]:
            t = int(x["tn"][k])
            superseded = later[(x["s"], t)][-1] != i
            if t > 60:
                assert ((i, t) in used) == (not superseded and task != "extra_session"), (pi, i, t)
            part = "practice" if t <= 20 else "pretest" if t <= 60 else "clamp"
            hand = r8(wrap(x["ep"][k] - 90.0))
            if x["xy"].shape[0] > k:
                a = math.degrees(math.atan2(x["start"][1] - x["xy"][k, 1], x["xy"][k, 0] - x["start"][0]))
                frame_err = max(frame_err, abs(wrap(a - x["ep"][k])))
            has_rep = t > 20 and k < len(x["rep"]) and not (x["rep"][k] == 0 and k < len(x["arc"]) and not x["arc"][k].any())
            clamp = r8(x["ec"][k] - 90.0) if part == "clamp" else ""
            if part == "clamp":
                clamp_ok &= abs(clamp - 10 * math.sin(2 * math.pi * (t - 61) / 20)) < 1e-6
            row = dict.fromkeys(names, "")
            row.update(
                dataset=ID, experiment=f"{ID}_ErrorClamp", study=ID, subject=f"p{pi}", person=f"p{pi}", id_type="lab",
                person_link="assumed_unique", group="", prior_exposure="unknown", subject_excluded_by_authors=False,
                hand_used="right", device="tablet", setting="lab", population="healthy",
                trial=0, trial_orig=t, session=x["s"] - 1, block=f"{task}_{part}", trial_kind="reach",
                phase={"practice": "practice", "pretest": "baseline", "clamp": "learning"}[part], break_before=first,
                target_angle=90, target_id=0, target_distance_norm=1, target_visible_during_move=True, target_event="none",
                target_jump=0, probe_kind="none", instr_goal="hand_to_target",
                instr_strategy="forbidden" if part == "clamp" else "unspecified",
                instr_disclosure="exact" if part == "clamp" else "none",
                instr_report="hand_post" if has_rep else "none", fb_cued=True,
                hand_angle=hand, hand_angle_fb=hand, hand_measure="fixed_distance",
                rt_ms=1000.0 * x["rt"][k], mt_ms=1000.0 * x["mt"][k],
                hand_report=r8(wrap(x["rep"][k])) if has_rep else "",
                excluded_by_authors=superseded or task == "extra_session",
                fb_visible="none" if part == "pretest" else "online",
                fb_mode={"practice": "veridical", "pretest": "none", "clamp": "clamped"}[part],
                fb_form="none" if part == "pretest" else "cursor",
                online_cursor={"practice": "veridical", "pretest": "none", "clamp": "perturbed"}[part],
                fb_endpoint_transform_only=False, rotation=0 if part == "practice" else "", clamp=clamp,
                cursor_error=hand if part == "practice" else clamp, fb_delay_ms="" if part == "pretest" else 0,
                outcome_channels={"practice": "cursor", "pretest": "", "clamp": "cursor|points"}[part],
                score=float(x["pts"][k]) if part == "clamp" and k < len(x["pts"]) else "")
            rows.append(row)
            first = False
            if part == "pretest" and has_rep:
                pre_r.append((pi, i, hand, row["hand_report"]))
    assert all(x["tgt"] == (512, 259) and x["dist"] == 334 and tuple(x["start"]) == (512, 593) for x in runs)
# trial: order within each participant's sequence, from 0
seq = {}
for r in rows:
    r["trial"] = seq.get(r["subject"], 0)
    seq[r["subject"]] = r["trial"] + 1

checks.internal("endPtAngle equals the direction of the stored screen endpoint from the start (y up) on every trial",
                frame_err < 1e-9, f"max {frame_err:.1e} deg")
checks.internal("the authors' processed sensorimotor and motor-awareness files (240 clamp trials each) are reproduced from "
                "the runs (endPtAngle - 90 minus its mean)", proc_dev < 1e-9 and clamp_ok, f"max {proc_dev:.1e} deg")
checks.internal("clamp = 10 sin(2 pi (t - 61) / 20) deg on trials 61-300 (0 on trial 61)", clamp_ok)
runs_r = {}
for pi, i, h, rp in pre_r:
    runs_r.setdefault((pi, i), []).append((h, rp))
cors = [np.corrcoef(*zip(*v))[0, 1] for v in runs_r.values() if len(v) >= 20]
checks.internal("no-feedback pre-test: the report rises with the hand in most runs (report in the hand's frame)",
                sum(c > 0 for c in cors) >= 0.75 * len(cors), f"{sum(c > 0 for c in cors)} of {len(cors)} runs, median r {np.median(cors):.2f}")
n_by = {}
for r in rows:
    if r["block"].endswith("_clamp") and not r["excluded_by_authors"]:
        n_by[(r["subject"], r["block"])] = n_by.get((r["subject"], r["block"]), 0) + 1
checks.internal("every participant has 240 analysed clamp trials in each task", sorted(set(n_by.values())) == [240] and len(n_by) == 40)
checks.internal("hand_angle and hand_report within (-180, 180]", all(-180 < r["hand_angle"] <= 180 for r in rows)
                and all(-180 < r["hand_report"] <= 180 for r in rows if r["hand_report"] != ""))


def lag(task):
    """The authors' reach lag (dataProcessingSaveFiles.m): phase of the 12-cycle Fourier component of the centred reach
    over the 240 clamp trials, lag = (-theta - 90) / 18 trials, theta wrapped below 0."""
    out = []
    for s in sorted({r["subject"] for r in rows}, key=lambda x: int(x[1:])):
        v = sorted((r["trial_orig"], r["hand_angle"]) for r in rows if r["subject"] == s and r["block"] == f"{task}_clamp"
                   and not r["excluded_by_authors"])
        x = np.array([h for t, h in v])
        th = math.degrees(np.angle(np.fft.fft(x - x.mean())[12]))
        out.append((-(th - 360 if th > 0 else th) - 90) / 18)
    return out


sm, ma = lag("sensorimotor"), lag("motor_awareness")
checks.published("reach lag behind the clamp at 12 cycles/session, sensorimotor task, mean (trials)", st.mean(sm), 11.89, 0.005,
                 "Results, sensorimotor task (Fig 8 text)")
checks.published("same, SD across participants", st.stdev(sm), 0.61, 0.005, "Results, sensorimotor task (Fig 8 text)")
checks.published("reach lag behind the clamp, motor-awareness task, mean (trials)", st.mean(ma), 11.84, 0.005, "Fig 11 legend")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")} | {"venue": paper["venue"] or "bioRxiv"},
    lab="Landy (NYU) with Taylor (Princeton)",
    data_url="https://github.com/marissahevans/implicitAdaptation",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format and participant codes replaced.",
    unit_note="trial of one participant sequence",
    how=[
        "One row is one reach, every trial in the deposit's run files in time order: per session 20 practice reaches with "
        "a veridical cursor, 40 pre-test reaches without feedback followed by a report of the felt hand direction, and 240 "
        "reaches with a sinusoidal error clamp (10 sin(2 pi (t - 61)/20) deg); session 0 or 1 is the sensorimotor or the "
        "motor-awareness task (block says which).",
        "Angles are counter-clockwise positive from the target straight ahead: hand_angle is where the hand crossed the "
        "target circle, hand_report the reported hand direction (pre-test trials, and every clamp trial of the "
        "motor-awareness task), score the points earned for the confidence arc; the arcs themselves have no column here.",
        "Sessions split over several run files are joined in time order; empty array slots (trial number 0) are not rows; "
        "excluded_by_authors marks a first attempt at trials 1-60 that a later run repeated (one participant) and a "
        "third session the paper does not use (one participant).",
    ],
    differences=[
        "The authors' dataProcessingSaveFiles.m sets report = -reportAngle, although the experiment code already saved "
        "reportAngle 'negative to match reach angles'; hand_report here keeps the saved sign, so report phases and lags "
        "computed from the authors' processed files are those of the mirrored series.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
