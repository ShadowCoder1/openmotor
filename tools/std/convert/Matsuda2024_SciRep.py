"""Matsuda & Abe (2024), Sci Rep: implicit sensorimotor learning in a ballistic stick flick (projection task). 24 right-handed
adults flick a gamepad stick to send a cursor 15 cm to a target at 70, 90 or 110 deg; error-clamped cursor feedback of -12, 0 or +12 deg
on perturbation trials, each followed by a no-feedback trial. 45 baseline trials + 378 main trials per participant, one row per
trial in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Matsuda2024_SciRep.py
Raw files (untrusted data, read only) in raw_extra/Matsuda2024_SciRep/ (Figshare 25572135): ba_1-24.txt (baseline), main_1-24.txt,
'Description of individual data file.txt'. AFC_25-48.txt (a separate control group of 24 doing a two-alternative forced choice
perception task) is not converted: not motor behaviour. The authors' .m scripts were read as text, never run.
Sign: the paper does not state the sign of SMD (stick movement direction, relative to the target here); the clamp is
counter-clockwise positive. The mean slope of the change in SMD from a perturbation trial to the next no-feedback trial on the clamp
is negative (about -0.31), i.e. people move opposite to the cursor error, so SMD is counter-clockwise positive like the clamp.
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Matsuda2024_SciRep"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
TARGET = {"1": 70.0, "2": 90.0, "3": 110.0}
SPEED = {"1": "fast", "2": "medium", "3": "slow"}  # the authors' script comments (the description file lists them in another order)
CLAMP = {"1": -12.0, "2": 0.0, "3": 12.0}


def num(x):
    return "" if x is None else "%.8g" % float(x)


def read(f):
    return list(csv.DictReader(open(RAW / f, newline="")))


rows = []
subj = {}
sd = {"1": [], "2": [], "3": []}
miss, fa = [], []
slopes = []
n_out = n_main = 0
mean_smd = {"1": [], "2": [], "3": []}
for p in range(1, 25):
    ba, ma = read(f"ba_{p}.txt"), read(f"main_{p}.txt")
    checks.internal(f"participant {p}: 45 baseline and 378 main trials", len(ba) == 45 and len(ma) == 378)
    seq = [("baseline", r) for r in ba] + [("main", r) for r in ma]
    # trigger flags: each perturbation trial is followed by a no-feedback trial that carries the trigger response
    flag = [False] * len(ma)
    pert = [k for k, r in enumerate(ma) if r["feedback"] in CLAMP]
    nxt = [ma[k + 1]["trigger"] == "0" for k in pert]
    miss.append(100 * sum(nxt) / len(nxt))
    for k, m_ in zip(pert, nxt):
        if m_:
            flag[k] = flag[k + 1] = True
    others = [k for k in range(1, len(ma)) if ma[k - 1]["feedback"] not in CLAMP]
    fa.append(100 * sum(ma[k]["trigger"] == "1" for k in others) / len(others))
    for k in others:
        if ma[k]["trigger"] == "1":
            flag[k] = True
    xs, ys = [], []
    for k in range(1, len(ma)):
        a, b = ma[k - 1], ma[k]
        if a["feedback"] in CLAMP and b["feedback"] == "10" and abs(float(a["SMD"])) <= 90 and abs(float(b["SMD"])) <= 90:
            xs.append(CLAMP[a["feedback"]]); ys.append(float(b["SMD"]) - float(a["SMD"]))
    mx, my = st.mean(xs), st.mean(ys)
    slopes.append(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs))
    for t in "123":
        v = [float(r["SMD"]) for r in ba if r["target"] == t and abs(float(r["SMD"])) <= 90]
        sd[t].append(st.stdev(v))
        mean_smd[t].append(st.mean(float(r["SMD"]) for r in ma if r["target"] == t and r["feedback"] == "0" and abs(float(r["SMD"])) <= 90))
    for i, (blk, r) in enumerate(seq):
        k = i - 45
        raw_smd = float(r["SMD"])
        smd = (raw_smd + 180.0) % 360.0 - 180.0  # two raw values lie beyond +-180 deg: wrapped
        big = abs(raw_smd) > 90
        n_out += big
        n_main += blk == "main"
        fbk = r["feedback"]
        row = dict.fromkeys(names, "")
        clamped = fbk in CLAMP
        veridical = fbk == "0"
        nofb = fbk == "10"
        row.update(
            dataset=ID, experiment=f"{ID}_main", study=ID, subject=f"{ID}_S{p:04d}", person=f"{ID}_P{p:04d}",
            id_type="lab", person_link="assumed_unique", prior_exposure="unknown", subject_excluded_by_authors="False",
            handedness="right", device="other", setting="lab", population="healthy",
            trial=str(i), trial_orig=str(i + 1 if blk == "baseline" else k + 1), session="0", block=blk,
            phase="baseline" if blk == "baseline" else ("probe" if nofb else "learning"),
            trial_kind="reach", break_before="True" if i == 0 or i == 45 else "False",
            target_angle=num(TARGET[r["target"]]), target_id=str(int(r["target"]) - 1), target_distance_norm="1",
            target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
            instr_goal="cursor_to_target", hand_used="right",
            cue_elements=f"speed_condition:{SPEED[r['speed']]}", fb_cued="False",
            hand_angle=num(smd), hand_angle_fb=num(smd), hand_measure="other",
            excluded_by_authors="True" if (big or (blk == "main" and flag[k])) else "False",
            fb_visible="none" if nofb else "endpoint",
            fb_mode="none" if nofb else ("clamped" if clamped else "veridical"),
            fb_form="none" if nofb else "cursor", online_cursor="none", fb_endpoint_transform_only="False",
            clamp=num(CLAMP[fbk]) if clamped else "",
            cursor_error="" if nofb else (num(CLAMP[fbk]) if clamped else num(smd)),
            outcome_channels="" if nofb else "cursor")
        rows.append(row)

checks.internal("rows = 24 x (45 + 378)", len(rows) == 24 * 423)
checks.internal("no hand angle beyond +-180", all(abs(float(r["hand_angle"])) <= 180 for r in rows))
checks.internal("mean SMD on null trials is near 0 for every target (SMD is relative to the target)",
                all(abs(st.mean(v)) < 6 for v in mean_smd.values()), str({k: round(st.mean(v), 2) for k, v in mean_smd.items()}))
checks.internal("the change in SMD on the next no-feedback trial has a negative slope on the clamp for 22 or more of 24 participants "
                "(sign check: SMD is counter-clockwise positive)", sum(s < 0 for s in slopes) >= 22,
                f"{sum(s < 0 for s in slopes)} of 24, mean slope {st.mean(slopes):.3f}")
checks.published("participants", len(set(r["subject"] for r in rows)), 24, 0, "Methods, Participants (24 young adults)")
checks.published("baseline trials per participant", sum(r["block"] == "baseline" for r in rows) / 24, 45, 0, "Methods (45 null trials)")
checks.published("main-phase trials per participant", n_main / 24, 378, 0, "Methods (378 trials)")
checks.published("main-phase no-feedback trials per participant", sum(r["fb_mode"] == "none" for r in rows) / 24, 81, 0,
                 "Methods (81 no-feedback trials)")
checks.published("main-phase perturbation trials per participant", sum(r["fb_mode"] == "clamped" for r in rows) / 24, 81, 0,
                 "Methods (81 perturbation trials)")
checks.published("main-phase null trials per participant", sum(r["fb_mode"] == "veridical" and r["block"] == "main" for r in rows) / 24, 216,
                 0, "Methods (216 null trials)")
for t, lab, (m, s) in (("3", "110", (11.9, 3.7)), ("2", "90", (6.0, 2.4)), ("1", "70", (10.8, 3.4))):
    checks.published(f"baseline SD of stick direction, {lab} deg target: mean over participants", st.mean(sd[t]), m, 0.1,
                     "Results, movement variability in the baseline phase")
    checks.published(f"baseline SD of stick direction, {lab} deg target: SD over participants", st.stdev(sd[t]), s, 0.1,
                     "Results, movement variability in the baseline phase")
checks.published("misses, % of perturbation trials: mean", st.mean(miss), 1.95, 0.005, "Methods, secondary task")
checks.published("misses, % of perturbation trials: SD", st.stdev(miss), 1.78, 0.005, "Methods, secondary task")
checks.published("false alarms, % of null and no-feedback trials: mean", st.mean(fa), 0.15, 0.005, "Methods, secondary task")
checks.published("false alarms, % of null and no-feedback trials: SD", st.stdev(fa), 0.26, 0.005, "Methods, secondary task")
checks.published("trials with stick direction beyond 90 deg, % of main-phase trials", 100 * n_out / n_main, 0.03, 0.005, "Methods, exclusions (0.03%)")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Abe (Hokkaido University)", data_url="https://doi.org/10.6084/m9.figshare.25572135",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format and participant codes replaced; two SMD values below -180 deg (-285.7 and -288.1) wrapped into (-180, 180] (to 74.3 and 71.9); the baseline and main files of a participant were put in one sequence.",
    unit_note="trial of one participant sequence (45 baseline trials, then 378 main trials)",
    how=[
        "One row is one flick of a gamepad stick that sends a cursor 15 cm toward a target at 70, 90 or 110 deg (0 deg = right). hand_angle "
        "is the authors' stick movement direction (SMD) in degrees, measured where the stick reached 90% of its tilt, relative to the "
        "target. The deposit's main files hold the 378 main trials, the ba files the 45 baseline trials (null trials, veridical cursor).",
        "Main trials: null (veridical cursor), perturbation (cursor clamped at -12, 0 or +12 deg from the target; clamp column) and "
        "no-feedback trials (no cursor); every perturbation trial is followed by a no-feedback trial (phase probe). The cursor appears after the flick, "
        "so fb_visible is endpoint and online_cursor is none. cue_elements gives the speed condition (fast, medium or slow target speed, "
        "labelled as in the authors' script).",
        "Sign: the paper says positive clamp values are counter-clockwise but does not state the SMD sign. It was inferred: the change in SMD "
        "from a perturbation trial to the next no-feedback trial has a negative slope on the clamp (about -0.31 across participants, "
        "negative for nearly all), so people move opposite to the cursor error and SMD is counter-clockwise positive like the clamp.",
        "excluded_by_authors marks trials with |SMD| over 90 deg, and the perturbation / no-feedback pair of each missed trigger response "
        "and each false-alarm trial (the authors drop flagged trials; the trigger pull for a perturbation is stored on the following "
        "trial). The trigger column itself has no place in this format and is not included. The separate 2AFC perception control group "
        "(AFC files) is not converted.",
    ],
    differences=["The baseline SD of stick direction for the 90 deg target is 5.94 deg here (and in the authors' sd_baseline.csv) against 6.0 in the paper text, so a tolerance of 0.1 deg is used for the 70/90/110 deg baseline SDs.", "Trials beyond 90 deg are counted before wrapping (3 raw values, 0.03% of main-phase trials as in the paper).", "Misses and false alarms were reproduced as: a miss is a perturbation trial whose next trial has no trigger pull, a false alarm "
                 "a trigger pull on any other trial after the first; the paper does not define them beyond the percentages."],
)
stdlib.release(ID, "reach90", rows, meta, checks)
