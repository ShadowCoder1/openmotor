"""Selftest2026_Raw (Chu & Sanger, 2014, F1000Research 3:72): the published visuomotor rotation data re-written by the
contributors as a lab-style file (radians, own column names), one row per reach, in the reach90 format.

Run from the site folder: python3 -I tools/std/convert/Selftest2026_Raw.py
Raw files (untrusted data, read only) in raw_extra/Selftest2026_Raw/:
  lab_data.csv  participant, trial_num, target_deg, rot, hand_dir_rad, rt_s, feedback (rt_s empty, feedback 'unknown')
  README.txt    column descriptions (units and signs as used below)
"""
import csv
import math
import sys
from collections import OrderedDict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Selftest2026_Raw"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()


def num(x):
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


raw = list(csv.DictReader(open(RAW / "lab_data.csv", newline="", encoding="utf-8")))
subs = OrderedDict()
for r in raw:
    subs.setdefault(r["participant"], []).append(r)


def design(v):
    # The file's rot column has the paper's wording, which is the mirror of the angle frame (contributor, round 3):
    # the rotation in the angle frame is -rot.
    rots = [float(r["rot"]) for r in v]
    if all(x == 0 for x in rots):
        return "rotation_not_recorded"
    if all(x == -10 for x in rots):
        return "rotation_plus10"
    return "rotation_plus10_minus10_plus10"


group = {s: design(v) for s, v in subs.items()}
names = [c[1] for c in stdlib.reach90_columns()]
target_ids = {}
rows = []
for i, (s, v) in enumerate(subs.items(), start=1):
    seg = 0
    prev = None
    for j, r in enumerate(v):
        rot = -float(r["rot"]) if float(r["rot"]) != 0 else 0.0  # sign corrected to the angle frame (contributor)
        if rot != prev:
            seg += 1
            prev = rot
        ta = float(r["target_deg"]) % 360.0
        tid = target_ids.setdefault(ta, len(target_ids))
        unrec = group[s] == "rotation_not_recorded"  # rot = 0 there means unknown, not "no rotation"
        ha = math.degrees(float(r["hand_dir_rad"]))
        row = dict.fromkeys(names, "")
        row.update(
            dataset=ID, experiment=f"{ID}_Main", study=ID, subject=f"{ID}_S{i:04d}", person=f"{ID}_P{i:04d}",
            id_type="lab", person_link="unknown", group=group[s], prior_exposure="unknown",
            subject_excluded_by_authors="False", device="Wacom Intuos 2 pen tablet, 50 Hz, hand hidden", setting="lab", population="healthy",
            trial=str(j), trial_orig=r["trial_num"], session="0",
            block=f"segment{seg}_rot{rot:g}" if not unrec else f"segment{seg}_rotation_not_recorded",
            phase="other" if unrec else "learning", trial_kind="reach",
            break_before="True" if j == 0 else "False",
            target_angle=num(ta), target_id=str(tid),
            cursor_error=num(ha),
            hand_angle="" if unrec else num(ha - rot), hand_angle_fb="" if unrec else num(ha - rot), hand_measure="peak_velocity",
            excluded_by_authors="False",
            fb_visible="unknown", fb_mode="unknown",
            rotation="" if unrec else num(rot),
        )
        rows.append(row)

n_rows = len(raw)
checks.internal("every row of lab_data.csv is kept, in order", len(rows) == n_rows == 4317)
checks.internal("72 participants, 59 or 60 rows each, rows grouped by participant",
                len(subs) == 72 and all(59 <= len(v) <= 60 for v in subs.values())
                and [r["participant"] for r in raw] == [s for s, v in subs.items() for _ in v])
checks.internal("trial numbers increase within each participant; 3 participants lack one trial number (24, 34, 41)",
                all(all(int(b["trial_num"]) > int(a["trial_num"]) for a, b in zip(v, v[1:])) for v in subs.values())
                and sorted(s for s, v in subs.items() if len(v) == 59) == ["S0024", "S0034", "S0041"])
checks.internal("rotation is 0, -10 or 10 deg; three designs: 32 with rot = 0 throughout (90 deg study, rotation not recorded), 10 constant (-10 in the file, +10 after the sign flip), 30 with -10/+10/-10 segments in the file",
                {float(r["rot"]) for r in raw} <= {0.0, -10.0, 10.0}
                and Counter(group.values()) == Counter({"rotation_not_recorded": 32, "rotation_plus10": 10,
                                                        "rotation_plus10_minus10_plus10": 30}))
first_after = [math.degrees(float(v[0]["hand_dir_rad"])) for s_, v in subs.items() if group[s_] != "rotation_not_recorded"]
checks.internal("hand_angle is the cursor error (cursor_error): mean of the first reach is positive (about +7 deg) under the +10 deg rotation of the angle frame",
                sum(first_after) / len(first_after) > 4, "mean %.1f deg" % (sum(first_after) / len(first_after)))
checks.internal("hand_angle = cursor_error - rotation where the rotation is known, empty where it is not recorded",
                all((r["hand_angle"] == "") == (r["rotation"] == "") for r in rows)
                and all(abs(float(r["hand_angle"]) - (float(r["cursor_error"]) - float(r["rotation"]))) < 1e-4
                        for r in rows if r["rotation"] != ""))
checks.internal("hand direction within (-pi, pi] rad and finite",
                all(-math.pi < float(r["hand_dir_rad"]) <= math.pi for r in raw))
checks.internal("target_angle within [0, 360)", all(0 <= float(r["target_angle"]) < 360 for r in rows))

n_grp = Counter(group.values())
# Each person did 4 experiments, so codes / 4 = people (contributor's description; Methods, Tables 1 and 2 of the paper).
checks.published("people in the 20 deg study (codes with a recorded rotation / 4)",
                 (n_grp["rotation_plus10"] + n_grp["rotation_plus10_minus10_plus10"]) / 4, 10, 0,
                 "Chu & Sanger 2014, Methods and Tables 1-2: 10 participants in the 20 deg study")
checks.published("people in the 90 deg study (codes with rotation 0 / 4)", n_grp["rotation_not_recorded"] / 4, 8, 0,
                 "Chu & Sanger 2014, Methods and Tables 1-2: 8 participants in the 90 deg study")



def aftereffect(v):
    """Paper's criterion (contributor's description, Tables 1-2): z = (first trial of block C - mean of last 10 trials of
    block A) / SD of those 10 trials; blocks are trials 1-20 (A), 21-40 (B), 41-60 (C)."""
    A = [math.degrees(float(r["hand_dir_rad"])) for r in v if 11 <= int(r["trial_num"]) <= 20]
    C = [math.degrees(float(r["hand_dir_rad"])) for r in v if int(r["trial_num"]) >= 41]
    m = sum(A) / len(A)
    sd = math.sqrt(sum((x - m) ** 2 for x in A) / (len(A) - 1))
    return abs((C[0] - m) / sd) > 1.96


sig = {g: sum(aftereffect(v) for s_, v in subs.items() if group[s_] == g) for g in n_grp}
checks.published("90 deg study: participants with a significant aftereffect, experiments 1-4 summed (6+4+7+7)",
                 sig["rotation_not_recorded"], 24, 0, "Chu & Sanger 2014, Table 2 (|z| > 1.96, as given by the contributor)")
checks.published("20 deg study, the experiment without the +10/-10 switch: significant aftereffects (3 of 10)",
                 sig["rotation_plus10"], 3, 0, "Chu & Sanger 2014, Table 1, experiment 2 (the 10 codes with one constant rotation)")
checks.internal("20 deg study, the other three experiments: Table 1 reports 10+10+9 = 29 significant aftereffects; this file gives 28 (not reproduced exactly)",
                sig["rotation_plus10_minus10_plus10"] == 28, "computed %d of 30, reported 29" % sig["rotation_plus10_minus10_plus10"])

meta = dict(
    study=ID,
    paper=dict(authors="Chu VWT, Sanger TD", year=2014,
               title="Two different motor learning mechanisms contribute to learning reaching movements in a rotated visual environment",
               venue="F1000Research 3:72", doi="10.12688/f1000research.3676.2"),
    lab="Sanger lab",
    data_url="https://doi.org/10.6084/m9.figshare.957526",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="rotation sign flipped to the angle frame (contributor); converted to the OpenMotor format, participant codes replaced, and hand direction converted from radians to degrees.",
    unit_note="reach",
    differences=["the recorded angle is the cursor direction (peak-velocity direction of the on-screen movement), stored in cursor_error; hand_angle is derived as cursor_error minus rotation and is empty where the rotation was not recorded; rotation sign flipped from the contributor's rot column to match the angle frame; the 28 vs 29 aftereffect count in the 20 deg study is not reproduced exactly."],
    how=[
        "One row is one reach by one participant (59 or 60 per participant, 72 participants, 4317 rows); every row of the "
        "uploaded file is kept in order.",
        "The recorded angle is the direction of the cursor on the screen relative to the target (a cursor error), from the "
        "start to the point of maximum velocity, converted from radians to degrees, counter-clockwise positive; it is stored in "
        "cursor_error. It jumps when the rotation changes and then returns toward 0 as people adapt. rotation is the cursor "
        "rotation in degrees in the same angle frame (the contributor's rot column has its sign flipped: the file's rot follows "
        "the paper's wording, the mirror of the angle frame). hand_angle = cursor_error - rotation (cursor = hand + rotation), "
        "and is empty for the 90 deg study, where the rotation was not recorded.",
        "Reaction time and cursor visibility were not recorded in the file (empty and 'unknown'), so rt_ms is empty and "
        "feedback visibility is 'unknown'.",
        "Participants are healthy adult volunteers (10 in the 20 deg study, 8 in a separate 90 deg study); each participant code is one person in one of 4 experiments, and the same person appears under 4 codes that cannot be linked. The 32 codes with rotation 0 are the 90 deg study, whose rotation was not recorded in the file (rotation is left empty there: it is unknown, not a zero rotation). Device: Wacom Intuos 2 pen tablet at 50 Hz, hand hidden below the screen; 20 deg study 10 right-handed adults aged 23-27 (right hand), 90 deg study 8 adults (mean age 25.4, dominant hand).",
        "group names the rotation schedule seen in the file (rotation not recorded; +10 deg throughout; +10, then -10, then +10 deg in "
        "segments of about 20 trials); block numbers those segments.",
    ],
    unit_note_extra=None,
)
meta.pop("unit_note_extra")
stdlib.release(ID, "reach90", rows, meta, checks)
