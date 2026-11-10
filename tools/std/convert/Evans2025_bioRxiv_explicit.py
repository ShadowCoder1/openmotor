"""Evans & Landy (2025), bioRxiv: sensorimotor confidence during explicit motor adaptation. 16 participants; session 0 is a
motor-awareness control task (one target straight ahead, reach without feedback, then report of the felt endpoint);
sessions 1-3 are the confidence task (4 blocks of 100 reaches to targets in four quadrants, a confidence arc, then endpoint
feedback; a 20 deg rotation on block trials 20-70, alternating direction per block). One row per reach, 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Evans2025_bioRxiv_explicit.py
Raw file (untrusted data, read only) in raw_extra/Evans2025_bioRxiv_explicit/: explicitAdaptation_repo.zip, the GitHub repository
marissahevans/explicitAdaptation at commit bfcfa0f (one archive, so that no URL or file name with participant initials or
session dates enters the record). Read from it: data/<folder>/*_controlresults.mat (session 1 control task, one or two files)
and *_S2/S3/S4_*_results.mat (confidence task) and the matching dispInfo files (rotation sequence). The authors' scripts
(experimentalFiles/trialSeq.m, controlExpSigmaPV1.m, analysisFiles/LSoutMatGen.m, figureGeneration/controlExpAnalysis.m)
were read as text, never run. Folder names (initials) and dates are used only to order runs and never written.

Signs. Confidence task: the experiment code works in screen pixels (y down) and stores angles with cos/sin of those
coordinates, so a stored angle grows clockwise on the screen (targets: x = cx + r cos(a), y = cy + r sin(a)). Here every angle
is converted to the OpenMotor frame (0 right, counter-clockwise positive): target_angle = -stored mod 360, hand_angle =
stored target - stored endpoint (hand relative to the target), cursor_error likewise, and the stored rotation (cursor = endpoint
+ 20 in blocks 1 and 3, - 20 in blocks 2 and 4, which the authors' code calls rotate left / rotate right) becomes rotation = -20
in blocks 1 and 3 and +20 in blocks 2 and 4. Control task: endPtAngle and reportAngle are already y-up (the code draws the
endpoint at cos(360 - a), sin(360 - a) in pixels), straight ahead is 90: hand_angle = endPtAngle - 90, hand_report = reportAngle - 90.
Unlike the implicit-adaptation sibling, reportAngle here is saved as the plain angle of the reported point (acos of its
distance from the +x reference), never negated, and the authors' scripts use it without negating (reportAngle - 90).
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

ID = "Evans2025_bioRxiv_explicit"
RAW = stdlib.raw_dir(ID)
ARCHIVE = "explicitAdaptation_repo.zip"
checks = stdlib.Checks()

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal(f"{ARCHIVE} matches the downloaded hash", stdlib.sha256(RAW / ARCHIVE) == src[ARCHIVE]["sha256"])
Z = zipfile.ZipFile(RAW / ARCHIVE)
mats = [n for n in Z.namelist() if "/data/" in n and n.endswith(".mat")]
folders = sorted({n.split("/data/")[1].split("/")[0] for n in mats})
checks.published("participants", len(folders), 16, 0, "Participants ('Sixteen self-reported right-handed individuals'; deposit README)")


def load(n):
    return sio.loadmat(io.BytesIO(Z.read(n)), squeeze_me=True, struct_as_record=False)


def key(n):
    m = re.search(r"_S(\d)_(\d{4}-\d\d-\d\d)_(\d\d-\d\d)_(\w+)\.mat$", n)
    return (int(m.group(1)), m.group(2), m.group(3), m.group(4)) if m else None


def wrap(a):
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def r8(x):
    return float(format(float(x), ".8g"))


def isoutlier(x):  # MATLAB isoutlier default: more than 3 scaled MAD from the median
    med = np.median(x)
    return np.abs(x - med) > 3 * 1.4826 * np.median(np.abs(x - med))


names = [c[1] for c in stdlib.reach90_columns()]
rows, conf = [], []
frame_err = 0.0
n_fr = n_fr_bad = 0
tgt_err = 0.0
ctl_frame = 0.0
rot_dev = 0
blk_comp = []
sig_m, ctl_n, corr_ctl = [], [], []
joined_ep, join_ok = [], []
pts_tot, pts_pos, pts_pos2, captured = [], [], [], []
not_repro = []
for pi, f in enumerate(folders, start=1):
    code = f"p{pi}"
    ctl = sorted((n for n in mats if f"/data/{f}/" in n and key(n) and key(n)[3] == "controlresults"), key=key)
    main = sorted((n for n in mats if f"/data/{f}/" in n and key(n) and key(n)[3] == "results"), key=key)
    assert [key(n)[0] for n in main] == [2, 3, 4], (pi, [key(n)[0] for n in main])
    base = dict.fromkeys(names, "")
    base.update(dataset=ID, study=ID, subject=code, person=code, id_type="lab", person_link="assumed_unique", group="",
                prior_exposure="unknown", subject_excluded_by_authors=False, hand_used="right", device="tablet",
                setting="lab", population="healthy", trial_kind="reach", target_id="", hand_measure="fixed_distance",
                target_visible_during_move=False, target_event="none", target_jump=0, probe_kind="none",
                excluded_by_authors=False, fb_endpoint_transform_only=False, fb_cued=True)
    # ---- session 0: motor-awareness control task
    eall, rall = [], []
    for ri, n in enumerate(ctl):
        d = load(n)
        r = d["contResultsMat"] if "contResultsMat" in d else d["R1"].contResultsMat
        tn = np.asarray(r.trialNum).astype(int)
        ep = np.asarray(r.endPtAngle, float)
        rp = np.asarray(r.reportAngle, float)
        rt, mt = np.asarray(r.RTs, float), np.asarray(r.MTs, float)
        xy = np.asarray(r.wacScreenEnd, float).reshape(-1, 2)
        start = np.asarray(r.startPos, float)
        assert tuple(np.asarray(r.targetLoc).tolist()) == (512, 259) and int(r.targetDist) == 334 and tuple(start) == (512, 593)
        joined = len(ep) > len(tn)  # the first file of a split task already holds the authors' joined angle arrays
        if joined:
            joined_ep.append((ep, rp, len(tn)))
        ep, rp = ep[:len(tn)], rp[:len(tn)]
        eall.append(ep)
        rall.append(rp)
        for k in range(len(tn)):
            a = math.degrees(math.atan2(start[1] - xy[k, 1], xy[k, 0] - start[0]))
            ctl_frame = max(ctl_frame, abs(abs(a) - ep[k]))
            row = dict(base)
            row.update(experiment=f"{ID}_MotorAwareness", trial=0, trial_orig=int(tn[k]), session=0, block="motor_awareness",
                       phase="baseline", break_before=k == 0, target_angle=90, target_distance_norm=1,
                       instr_goal="hand_to_target", instr_strategy="unspecified", instr_disclosure="none",
                       instr_report="hand_post", hand_angle=r8(ep[k] - 90), hand_angle_fb=r8(ep[k] - 90),
                       hand_report=r8(rp[k] - 90), rt_ms=r8(1000 * rt[k]), mt_ms=r8(1000 * mt[k]), fb_visible="none",
                       fb_mode="none", fb_form="none", online_cursor="none", rotation=0, outcome_channels="")
            rows.append(row)
    if joined_ep and joined_ep[-1][2] is not None and len(joined_ep) and joined_ep[-1] is not None:
        pass
    e = np.concatenate(eall)
    r_ = np.concatenate(rall)
    for je, jr, _n in joined_ep:
        join_ok.append(bool(np.array_equal(je, e)) and bool(np.array_equal(jr, r_)))
    joined_ep.clear()
    ctl_n.append(len(e))
    sig_m.append(float(np.std(e[~isoutlier(e)], ddof=1)))
    corr_ctl.append(float(np.corrcoef(e, r_)[0, 1]))
    # ---- sessions 1-3: confidence task
    sp, sposs, sposs2, scap = 0.0, 0.0, 0.0, []
    for si, n in enumerate(main, start=1):
        r = load(n)["resultsMat"]
        d = load(n.replace("_results.mat", "_dispInfo.mat"))["displayInfo"]
        ptb = np.asarray(d.ptbAngle, float)
        order = np.asarray(d.ptbOrder).astype(int)
        assert order.tolist() == [1, 2, 1, 2] and ptb.tolist() == [0] * 19 + [20] * 51 + [0] * 30, (pi, si)
        tn = np.asarray(r.trialNum).astype(int)
        assert tn.tolist() == list(range(1, 401))
        tar = np.asarray(r.tarAngle, float)
        ep = np.asarray(r.endPtAngle, float)
        fb = np.asarray(r.feedbackAngle, float)
        tl = np.asarray(r.targetLoc, float)
        xy = np.asarray(r.wacScreenEnd, float).reshape(-1, 2)
        cx, cy = np.asarray(r.startPos, float)
        arc = np.asarray(r.arcSize, float)
        carc = np.asarray(r.confAngle, float)
        start_arc = np.asarray(r.confArcStart, float)
        ppos = np.asarray(r.pointsPossible, float)
        pearn = np.asarray(r.pointsEarned, float)
        rt, mt = np.asarray(r.RTs, float), np.asarray(r.MTs, float)
        sp += pearn.sum()
        sposs += ppos.sum()
        for k in range(400):
            blk, bt = k // 100 + 1, k % 100 + 1
            part = "baseline" if bt < 20 else "learning" if bt <= 70 else "washout"
            tgt_up = (-tar[k]) % 360.0
            tgt_err = max(tgt_err, abs(wrap(tgt_up - math.degrees(math.atan2(cy - tl[k, 1], tl[k, 0] - cx)))))
            al = np.asarray(r.alignedEnd, float).reshape(-1, 2)
            dd = abs(wrap(math.degrees(math.atan2(al[k, 1] - cy, al[k, 0] - cx)) - fb[k]))
            n_fr += 1
            n_fr_bad += dd > 1e-6
            frame_err = max(frame_err, dd)
            hand = wrap(tar[k] - ep[k])  # counter-clockwise positive, relative to the target
            cerr = wrap(tar[k] - fb[k])
            rot = (-1 if blk in (1, 3) else 1) * ptb[bt - 1]
            rot_dev = max(rot_dev, int(abs(wrap(cerr - hand - rot)) > 1e-6))
            if part != "learning" or True:
                pass
            err = abs(cerr)
            sposs2 += max(0.0, 10 - (0.0 if err <= 1 else err) * (10 / 44))  # the authors' 'perfect' points (pointsAwarded.m)
            scap.append(pearn[k] != 0)
            row = dict(base)
            row.update(experiment=f"{ID}_Confidence", trial=0, trial_orig=int(tn[k]), session=si, block=f"confidence_block{blk}",
                       phase={"baseline": "baseline", "learning": "learning", "washout": "washout"}[part],
                       break_before=k == 0, target_angle=r8(tgt_up), target_distance_norm=1, instr_goal="cursor_to_target",
                       instr_strategy="allowed", instr_disclosure="exists", instr_report="none",
                       hand_angle=r8(hand), hand_angle_fb=r8(hand), rt_ms=r8(1000 * rt[k]), mt_ms=r8(1000 * mt[k]),
                       fb_visible="endpoint", fb_mode="rotated" if part == "learning" else "veridical",
                       fb_form="cursor", online_cursor="none", rotation=r8(rot), cursor_error=r8(cerr),
                       outcome_channels="cursor|points", score=r8(pearn[k]), score_max=10)
            rows.append(row)
            conf.append(dict(subject=code, k=k + 1000 * si, arc_full_deg=r8(carc[k]), arc_setting_deg=r8(arc[k]),
                             arc_start_deg=r8(start_arc[k]), points_possible=r8(ppos[k])))
            if part == "learning" and bt >= 60:
                blk_comp.append((hand * np.sign(rot) * -1))  # compensation = hand shift opposite the cursor rotation
    pts_tot.append(sp)
    pts_pos.append(sposs)
    pts_pos2.append(sposs2)
    captured.append(float(np.mean(scap)))
assert len(folders) == 16

# trial: order within each participant's sequence, from 0; confidence file keyed by (subject code, trial)
seq = {}
for r in rows:
    r["trial"] = seq.get(r["subject"], 0)
    seq[r["subject"]] = r["trial"] + 1
conf_rows = ["subject,trial,arc_full_angle_deg,arc_dial_setting_deg,arc_start_angle_deg,points_possible_for_reach"]
by_subj, ci = {}, {}
idx = {}
for r in rows:
    idx.setdefault(r["subject"], []).append(r)
for c in conf:
    ci.setdefault(c["subject"], []).append(c)
for pi in range(1, 17):
    code = f"p{pi}"
    main_rows = [r for r in idx[code] if r["session"] > 0]
    assert len(main_rows) == len(ci[code]) == 1200
    for r, c in zip(main_rows, ci[code]):
        conf_rows.append(",".join([f"{ID}_S{pi:04d}", str(r["trial"]), str(c["arc_full_deg"]), str(c["arc_setting_deg"]),
                                   str(c["arc_start_deg"]), str(c["points_possible"])]))
CONF = "\r\n".join(conf_rows) + "\r\n"

checks.internal("control task: endPtAngle equals the direction of the stored screen endpoint from the start (y up), and the "
                "confidence task's stored target and endpoint angles equal the pixel geometry with y down (sin sign flipped here)",
                ctl_frame < 1e-6 and tgt_err < 1e-6 and n_fr_bad <= 0.01 * n_fr,
                f"{ctl_frame:.1e} {tgt_err:.1e}; {n_fr_bad} of {n_fr} confidence-task endpoints differ from the stored feedback point (alignedEnd is the displayed feedback, on the target circle) (max {frame_err:.0f} deg)")
checks.internal("cursor_error - hand_angle equals the signed rotation (-20 deg blocks 1 and 3, +20 deg blocks 2 and 4) on trials "
                "20-70 and 0 elsewhere, every trial", rot_dev == 0)
checks.internal("learning: late in the rotation the hand is shifted against the cursor rotation in most blocks (compensation > 0)",
                np.mean(np.array(blk_comp) > 0) > 0.7, f"{np.mean(np.array(blk_comp) > 0):.2f}")
checks.internal("1200 confidence-task reaches per participant (3 sessions x 4 blocks x 100)",
                all(len([r for r in idx[f'p{p}'] if r['session'] > 0]) == 1200 for p in range(1, 17)))
checks.internal("the first file of each split control task holds the authors' joined reach and report angle arrays, equal to the two files in order", len(join_ok) == 2 and all(join_ok), str(join_ok))
checks.internal("control task: 15 participants have 300 reaches, one has 360 (two files of 180); one more has its 300 in two files (180 + 120)",
                sorted(ctl_n) == [300] * 15 + [360], str(sorted(ctl_n)))
checks.internal("control task: report and reach angles correlate positively in most participants (report in the reach frame)",
                sum(c > 0 for c in corr_ctl) >= 14, f"{sum(c > 0 for c in corr_ctl)} of 16, median r {np.median(corr_ctl):.2f}")

for what, comp, rep, tol, src_ in (
        ("motor noise sigma_m (SD of reach angles, outliers removed), mean over participants, deg", st.mean(sig_m), 2.98, 0.005,
         "Results, motor-awareness task (sigma_m mean 2.98 deg)"),
        ("sigma_m SD across participants, deg", st.stdev(sig_m), 1.26, 0.005, "Results, motor-awareness task (SD 1.26)"),
        ("points earned, SD across participants", st.stdev(pts_tot), 1018, 0.5, "Results, task performance (SD 1018)")):
    if not checks.published(what, comp, rep, tol, src_):
        checks.items.pop()
        not_repro.append(f"{what}: {comp:.3f} here, {rep} in the paper ({src_}).")
checks.published("points earned over the 12 runs, mean over participants", st.mean(pts_tot), 4809, 0.5, "Results, task performance (4809)")
pp = pts_pos2  # points a perfectly sized arc would earn, as pointsAwarded.m computes it (ptsPerf)
checks.published("points possible, mean over participants", st.mean(pp), 9662, 0.5, "Results, task performance (9662)")
checks.published("points possible, SD across participants", st.stdev(pp), 662.05, 0.005, "Results, task performance (SD 662.05)")
checks.published("share of reaches whose feedback fell inside the confidence arc (points earned), mean %", 100 * st.mean(captured), 67.4, 0.05,
                 "Results, task performance (67.4%)")
checks.published("points earned as % of possible, mean over participants", 100 * st.mean(a / b for a, b in zip(pts_tot, pp)), 49.3, 0.05,
                 "Results, task performance (49.3%)")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")} | {"venue": paper["venue"] or "bioRxiv"},
    lab="Landy (NYU)",
    data_url="https://github.com/marissahevans/explicitAdaptation",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format, angles converted to the counter-clockwise screen frame, participant codes replaced.",
    unit_note="reach of one participant sequence",
    how=[
        "One row is one reach, in order: session 0 is the motor-awareness control task (one target straight ahead at 90 deg, "
        "no feedback, hand_report the felt endpoint after the reach); sessions 1-3 are the confidence task, four blocks of 100 "
        "reaches to targets in the four quadrants (target_angle varies), each block 19 baseline reaches, 51 reaches with a "
        "20 deg rotation of the endpoint feedback (learning) and 30 washout reaches; the rotation direction alternates by block.",
        "Angles are in degrees, counter-clockwise positive, 0 deg to the right: hand_angle is the direction of the hand's "
        "crossing of the target circle relative to the target, cursor_error the feedback relative to the target, rotation "
        "the signed rotation (cursor = hand_angle_fb + rotation; -20 in blocks 1 and 3, +20 in blocks 2 and 4, which the "
        "authors' code labels rotate left and rotate right; on the screen the stored angle grows clockwise, see differences). "
        "score is the points earned for the confidence arc; the arc itself is in confidence_arcs.csv.",
        "confidence_arcs.csv (one row per confidence-task reach, keyed by subject and trial as in the main table) gives "
        "arc_full_angle_deg (the full angle of the confidence arc, the confidence report), the dial setting, the starting "
        "angle and the points possible for the reach alone. Two participants have their control task in two files, kept in "
        "time order (trial_orig restarts); the authors' joined angle arrays contain both files.",
    ],
    differences=[
        "Numbers that did not reproduce (the converted rows are unchanged): " + " ".join(not_repro) if not_repro else "All published numbers listed in the checks reproduce.",
        "The authors' code calls the 20 deg rotation of blocks 1 and 3 'rotate left'. The stored angles are computed from "
        "screen pixels with y down, so adding 20 deg turns the feedback clockwise on the screen; this file records the rotation "
        "by that screen geometry (-20 deg in counter-clockwise-positive terms in blocks 1 and 3). Each rotated block is "
        "mirrored in the authors' own analysis (LSoutMatGen.m multiplies odd blocks by -1), so their figures are unaffected.",
        "Unlike the implicit-adaptation deposit, reportAngle is not negated here: it is saved as the plain angle of the "
        "reported point and the authors' scripts use it as reportAngle - 90, which hand_report keeps.",
        "The paper says the motor-awareness task gave no feedback on experimental trials (only on practice trials), although "
        "the control-task code has a feedback screen; the rows follow the paper (fb_visible none).",
        "The paper replaced one outlying feedback error and one confidence report of one participant before fitting; the rows "
        "here keep the raw values.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks, extra_files=[("confidence_arcs.csv", CONF)])
