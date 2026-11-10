"""Chu & Sanger (2014), F1000Research: two motor learning mechanisms in a rotated visual environment.
Eight CSVs from Figshare (https://doi.org/10.6084/m9.figshare.957526): Dataset1-a..d = the 20 deg study (10 participants,
experiments 1-4), Dataset2-a..d = the 90 deg study (8 participants, experiments 1-4). Each file has 60 rows (three blocks of 20
reaches: A baseline, B interference, C re-adaptation), the target angle, and one column of initial hand angles per participant
(plus model simulations A-C in some files, which are not participant data and are not converted).

Run from the site folder: python3 -I tools/std/convert/Chu2014_F1000Res.py   (raw files in raw_extra/Chu2014_F1000Res/)
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Chu2014_F1000Res"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]


def wrap(a):
    w = (a + 180.0) % 360.0 - 180.0
    return 180.0 if w == -180.0 else w


def tf(x):
    return "True" if x else "False"


EXPS = {"a": "exp1_same_task_different_environment", "b": "exp2_different_task_same_environment",
        "c": "exp3_different_task_different_environment", "d": "exp4_control_for_exp3"}
BLOCKS = [("A_baseline", range(0, 20)), ("B_interference", range(20, 40)), ("C_readaptation", range(40, 60))]
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}

data = {}   # (study, exp letter) -> (targets, {subject number: [values or None]})
for study in (1, 2):
    for L in "abcd":
        fn = f"Dataset{study}-{L}.csv"
        checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])
        R = list(csv.reader(open(RAW / fn, newline="", encoding="utf-8-sig")))
        h = R[0]
        body = [r for r in R[1:] if r and r[0].strip() != ""]
        sc = [j for j, c in enumerate(h) if c.startswith("Subject")]
        trials = [int(r[0]) for r in body]
        tg = [float(r[1]) for r in body]
        vals = {}
        for j in sc:
            k = int(h[j].split()[1])
            vals[k] = [None if r[j].strip() in ("-", "") else float(r[j]) for r in body]
        checks.internal(f"{fn}: 60 reaches, trial numbers 1-20 three times", len(body) == 60 and trials == list(range(1, 21)) * 3)
        data[(study, L)] = (tg, vals)

checks.published("participants, 20 deg study", len(data[(1, "a")][1]), 10, 0, "Methods, Participants (ten participants)")
checks.published("participants, 90 deg study", len(data[(2, "a")][1]), 8, 0, "Methods, 90 degree experiment (eight participants)")
checks.internal("same participants in all four experiments of a study",
                all(set(data[(s, L)][1]) == set(data[(s, "a")][1]) for s in (1, 2) for L in "bcd"))
checks.published("reaches per experiment (3 blocks x 20)", len(data[(1, "a")][0]), 60, 0, "Methods, Task (three blocks of 20 reaches)")

# aftereffect z-scores: (first trial of block C - mean of last 10 trials of block A) / SD of those trials (Methods, Analysis)
sig = {}
for (study, L), (tg, vals) in data.items():
    n = 0
    for k, v in vals.items():
        base = [x for x in v[10:20] if x is not None]
        z = (v[40] - st.mean(base)) / st.stdev(base)
        n += abs(z) > 1.96
    sig[(study, L)] = n
for (study, L), rep in zip([(1, x) for x in "abcd"] + [(2, x) for x in "abcd"], [10, 3, 10, 9, 6, 4, 7, 7]):
    checks.published(f"participants with a significant aftereffect (|z| > 1.96), {'20' if study == 1 else '90'} deg study, "
                     f"experiment {'abcd'.index(L) + 1}", sig[(study, L)], rep, 1 if (study, L) == (1, "c") else 0,
                     "Results, Table 1 and Table 2")

ROT = {"a": {"B": 10}, "b": {"B": -10}, "c": {"B": 10}, "d": {"B": 10}}  # 20 deg study: baseline -10 (cw), interference +10
rows = []
tids = {}
for study in (1, 2):
    for L in "abcd":
        tg, vals = data[(study, L)]
        for k, v in vals.items():
            for bname, rng in BLOCKS:
                for t in rng:
                    ta = tg[t] % 360.0
                    tid = tids.setdefault(round(ta, 6), len(tids))
                    miss = v[t] is None
                    b = bname[0]
                    if study == 1:
                        rot = ROT[L]["B"] if b == "B" else -10
                    else:
                        rot = None
                    if L == "b" and b != "A":
                        phase = "other"
                    else:
                        phase = {"A": "baseline", "B": "learning", "C": "relearning"}[b]
                    row = dict.fromkeys(names, "")
                    row.update(
                        dataset=ID, experiment=f"{'rot20' if study == 1 else 'rot90'}_{EXPS[L]}", study=ID,
                        subject=f"{study}_{L}_{k}", person=f"{study}_{k}", id_type="lab", person_link="assumed_unique",
                        prior_exposure="unknown", subject_excluded_by_authors="False", population="healthy",
                        handedness="right" if study == 1 else "", device="tablet", setting="lab",
                        trial=t, trial_orig=(t % 20) + 1, session=0, block=bname, phase=phase,
                        trial_kind="missing" if miss else "reach", break_before=tf(t == 0),
                        target_angle=ta, target_id=tid, target_distance_norm=1, target_event="none", target_jump=0,
                        probe_kind="none", instr_goal="cursor_to_target", hand_used="right" if study == 1 else "",
                        hand_angle="" if miss else wrap(v[t] - ta), hand_measure="peak_velocity",
                        excluded_by_authors=tf(miss), fb_visible="unknown", fb_mode="rotated", fb_form="cursor",
                        online_cursor="perturbed", fb_endpoint_transform_only="False",
                        rotation="" if rot is None else rot,
                    )
                    rows.append(row)
checks.internal("every reach of every participant kept: 10 x 4 x 60 + 8 x 4 x 60 = 4320 rows", len(rows) == 4320)
checks.internal("hand angles within (-180, 180]", all(-180 < float(r["hand_angle"]) <= 180 for r in rows if r["hand_angle"] != ""))
nmiss = sum(r["trial_kind"] == "missing" for r in rows)
checks.internal("missing ('-') hand angles are marked as missing rows", nmiss == sum(v is None for d in data.values() for x in d[1].values() for v in x))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Sanger (USC)",
    data_url="https://doi.org/10.6084/m9.figshare.957526",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format, participant codes replaced, hand angle expressed relative to the target; "
            "model simulation columns not included.",
    unit_note="reach of one participant in one experiment",
    how=[
        "One row is one reach. Each participant did four experiments of 60 reaches (block A baseline, block B interference, "
        "block C re-adaptation, 20 reaches each); 10 participants in the 20 deg study (rot20) and 8 other participants in the "
        "90 deg study (rot90). Subject is one participant in one experiment, person is one participant.",
        "hand_angle is the authors' initial direction (start to the point of maximum velocity, in degrees) minus the target "
        "angle, wrapped to (-180, 180]; target_angle is the authors' Target column. Block A hand angles sit near the target, "
        "and in block B of experiments 1, 3 and 4 they fall below it, as expected when the feedback is rotated counter-clockwise "
        "and people compensate clockwise, so angles are counter-clockwise positive. A '-' in the authors' file is a missing "
        "measurement, kept as a row with trial_kind missing.",
        "rotation (20 deg study only) is the authors' description of the feedback: baseline 10 deg clockwise (-10), then 20 deg "
        "counter-clockwise relative to baseline in block B of experiments 1, 3 and 4 (+10), none relative to baseline in "
        "experiment 2 (the target moves instead); it is left empty for the 90 deg study because the paper does not give the "
        "direction or the baseline there.",
        "trial_orig is the position within the block (1-20). In experiment 4 the target column is 40, 20, 40 and the paper "
        "describes a target at 20 with a 40 deg reach goal; the Target column is kept as given.",
    ],
    differences=["Experiment 3 of the 20 deg study: the paper reports 10 of 10 participants with a significant aftereffect, "
                 "these rows give 9: one participant has a wild last baseline trial (hand angle 2 deg for a 40 deg target) that "
                 "inflates the baseline SD (11.8 deg) and brings z to 1.5; the authors presumably handled that trial differently. "
                 "The other seven counts (10, 3, 9 in the 20 deg study; 6, 4, 7, 7 in the 90 deg study) reproduce exactly.",
                 "The paper's data statement says the data are CC0 1.0; the Figshare record lists CC BY 4.0, which is used here."],
)
stdlib.release(ID, "reach90", rows, meta, checks)
