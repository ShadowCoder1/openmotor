"""Sadaphal, Kumar & Mutha (2022), eNeuro: learning from task performance errors (target shifts, no cursor rotation).
Three experiments, 75 analysed adults, one row per trial in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Sadaphal2022_eNeuro.py
Raw file (untrusted data, read only): raw_extra/Sadaphal2022_eNeuro/Sadaphal_Kumar_Mutha_eNeuro2022_data.xlsx (Figshare 19154378),
sheets experiment1-3 plus the ReadMe sheet. Sign: the paper (Methods, Data analysis) says counter-clockwise deviations are
positive, and the target shift was counter-clockwise, so the late-learning deviation is positive (about +45 deg in Exp 1).
"""
import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Sadaphal2022_eNeuro"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
wb = openpyxl.load_workbook(RAW / "Sadaphal_Kumar_Mutha_eNeuro2022_data.xlsx", read_only=True)
PHASE = {"Baseline": "baseline", "Learning": "learning", "Washout": "washout", "Savings": "relearning"}
SHIFT = {1: 45.0, 2: 45.0}
rows = []
per = collections.OrderedDict()  # (exp, group, subj) -> list of authors' rows
for e, sh in (("experiment1", 1), ("experiment2", 2), ("experiment3", 3)):
    data = [r for r in list(wb[e].iter_rows(values_only=True))[1:] if r[0] is not None]
    for r in data:
        per.setdefault((int(r[0]), str(r[1]), int(r[2])), []).append(r)
checks.internal("every sheet row belongs to a participant sequence with increasing trial numbers",
                all([r[4] for r in v] == sorted(r[4] for r in v) and len(set(r[4] for r in v)) == len(v) for v in per.values()))


def num(x):
    return "" if x is None else "%.8g" % float(x)


nsub = 0
pers = {}
for (ex, grp, sid), v in per.items():
    nsub += 1
    total = 304 if ex == 2 else 416
    byn = {int(r[4]): r for r in v}
    shift_deg = {1: 45.0, 2: 45.0}.get(ex) or float(grp.split("_")[0])
    blocks = {}
    for n, r in byn.items():
        blocks[n] = r[3]
    # block of a missing trial = block of the neighbouring present trials (they are contiguous within a block)
    prev_b = None
    for n in range(1, total + 1):
        r = byn.get(n)
        row = dict.fromkeys(names, "")
        if r is None:
            nxt = next((byn[m][3] for m in range(n, total + 1) if m in byn), prev_b)
            bname = nxt if prev_b == nxt else nxt
            row.update(trial_kind="missing", excluded_by_authors="True", block=bname, phase=PHASE[bname])
        else:
            bname = r[3]
            prev_b = bname
            sj = r[5] == "target_shift"
            ha = float(r[6])
            has_aim = ex == 2
            row.update(
                trial_kind="reach", excluded_by_authors="False", block=bname, phase=PHASE[bname],
                target_event="jump" if sj else "none", target_jump=num(shift_deg if sj else 0),
                hand_angle=num(ha), hand_angle_fb=num(ha), rt_ms=num(1000 * float(r[7])),
                aim_report=num(r[8]) if has_aim else "",
                instr_report="aim_pre" if has_aim else "none", aim_report_timing="pre" if has_aim else "",
                fb_visible="online", fb_mode="veridical", fb_form="cursor", online_cursor="veridical",
                fb_endpoint_transform_only="False", rotation="0", cursor_error=num(ha), fb_delay_ms="0",
                outcome_channels="cursor|points")
        row.update(
            dataset=ID, experiment=f"{ID}_Exp{ex}", study=ID, subject=f"{ID}_E{ex}_{nsub:04d}",
            person=f"{ID}_P{nsub:04d}", id_type="lab", person_link="assumed_unique",
            group="" if grp == "NA" else grp, prior_exposure="unknown", subject_excluded_by_authors="False",
            handedness="right", device="tablet", setting="lab", population="healthy",
            trial=str(n - 1), trial_orig=str(n), session="0", break_before="True" if n == 1 else "False",
            target_visible_during_move="True", target_distance_norm="1", probe_kind="none",
            instr_goal="cursor_to_target", instr_strategy="unspecified", instr_disclosure="exists",
            hand_used="right", hand_measure="peak_velocity", fb_cued="False")
        rows.append(row)

checks.internal("rows = authors' rows + placeholders for trials removed as outliers",
                len(rows) == 416 * 65 + 304 * 10 and sum(r["trial_kind"] == "reach" for r in rows) == 12373 + 2988 + 14315)
checks.internal("placeholders carry no data", all(r["hand_angle"] == "" for r in rows if r["trial_kind"] == "missing"))
checks.internal("hand_angle within +-180", all(abs(float(r["hand_angle"])) <= 180 for r in rows if r["hand_angle"]))

# ---- published checks
by = collections.OrderedDict()
for r in rows:
    by.setdefault(r["subject"], []).append(r)
cnt = collections.Counter((v[0]["experiment"][-4:], v[0]["group"]) for v in by.values())
checks.published("Exp 1 participants", cnt[("Exp1", "")], 30, 0, "Methods, Experiment 1 (n = 30)")
checks.published("Exp 2 participants", cnt[("Exp2", "")], 10, 0, "Methods, Experiment 2 (n = 10)")
checks.published("Exp 3 participants analysed (36 recruited, 1 excluded)", sum(n for (e, g), n in cnt.items() if e == "Exp3"), 35, 0,
                 "Methods, Experiment 3")
for g, n in (("15_jump", 11), ("30_jump", 12), ("60_jump", 12)):
    checks.published(f"Exp 3 group {g} participants", cnt[("Exp3", g)], n, 0, "Methods, Experiment 3")


def late(v):
    xs = [float(r["hand_angle"]) for r in v if r["block"] == "Learning" and r["hand_angle"] != "" and r["target_jump"] != "0"]
    return st.mean(xs[-4:])


def sem(x):
    return st.stdev(x) / math.sqrt(len(x))


S = "Results, late-learning hand deviation"
for exn, g, m, s in (("Exp1", "", 44.65, 1.13), ("Exp2", "", 46.194, 0.913), ("Exp3", "15_jump", 12.032, 2.076),
                     ("Exp3", "30_jump", 29.458, 1.426), ("Exp3", "60_jump", 54.239, 2.261)):
    x = [late(v) for v in by.values() if v[0]["experiment"].endswith(exn) and v[0]["group"] == g]
    checks.published(f"{exn} {g}: late-learning hand deviation, mean (last 4 shift trials)", st.mean(x), m, 1.5, S)
    checks.published(f"{exn} {g}: late-learning hand deviation, SEM", sem(x), s, 0.6, S)

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Mutha (IIT Gandhinagar)", data_url="https://doi.org/10.6084/m9.figshare.19154378",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format, participant codes replaced, and one placeholder row added for every trial number "
            "the authors removed as an outlier.",
    differences=["Late-learning means and SEMs (mean of the last 4 target-shift learning trials per participant) agree with the paper within about 1.3 deg and 0.4 deg, not exactly: the paper averages the last reach to each of the four targets, but the file does not say which target each trial used, so the last four trials are used instead (a tolerance of 1.5 deg and 0.6 deg is used)."],
    unit_note="trial of one participant sequence; trial_orig is the authors' trial number",
    how=[
        "One row is one reach to a target 10 cm away on a tablet. The cursor was veridical; the target jumped 45 deg (Exp 1 and 2) "
        "or 15, 30, 60 deg (Exp 3 groups) counter-clockwise when the hand left the start circle on 'target_shift' trials "
        "(target_jump), and stayed on 'no_shift' trials. Blocks: Baseline, Learning, Washout, Savings (phase relearning).",
        "hand_angle is the authors' baseline-subtracted hand deviation from the original target at peak speed, in degrees. The "
        "paper says counter-clockwise is positive, so no sign change was made; target_jump is positive for the counter-clockwise shift.",
        "The deposit does not say which of the four target directions (45, 135, 225, 315 deg) each trial used, so target_angle "
        "is empty. Trials the authors removed as outliers are absent from their file; they are kept here as 'missing' placeholder "
        "rows (excluded_by_authors True) so that trial_orig runs 1..416 (Exp 2: 1..304).",
        "In Exp 2, aim_report is the reported aiming direction in degrees (the file holds degrees, 5 deg landmark spacing; 45 = the shifted target).",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
