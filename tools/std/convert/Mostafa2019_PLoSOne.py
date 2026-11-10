"""Mostafa, 't Hart & Henriques (2019), PLoS ONE: hand localization and no-cursor reaches after passive "exposure"
training (the robot moved the hand while the cursor went straight to the target; 24 participants in the deposit, 21 analysed)
or after classic active reach training with a rotated cursor (21 participants from an earlier study), in the 90-column
reaching format.

Run from the site folder: python3 -I tools/std/convert/Mostafa2019_PLoSOne.py
Raw files (untrusted data, read only) in raw_extra/Mostafa2019_PLoSOne/, from https://osf.io/zfdth/:
  localization_exposure.csv, localization_classic.csv, localization_online.csv, nocursor_exposure.csv, nocursor_classic.csv,
  participants_exposure.csv (age, sex), blinkdetect_exposure.csv (attention task, proportion correct per session).
The authors' analysis code on GitHub was not run. The training trials themselves are not in the deposit.

Signs. Angles are in the screen frame from the home position, 0 = right, counter-clockwise positive: handangle_deg is atan2(y, x)
of the hand position (checked on every row), and endpoint_angle = atan2(y, x) - target (checked on every row, unwrapped).
taperror_deg = tap angle - hand angle (checked on every row). The paper does not state a sign; the direction of the effects
agrees with a clockwise-rotated cursor (see README).
"""
import csv
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Mostafa2019_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
LOW = ["exp015", "exp007", "exp018"]  # the three lowest performers on the attention task (proportion correct near 0.5 in the rotated session)
TARGETS = [15, 25, 35, 45, 55, 65, 75]


def num(x):
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


rd = lambda f: pd.read_csv(RAW / f)  # noqa: E731
Lx, Lc, Lo = rd("localization_exposure.csv"), rd("localization_classic.csv"), rd("localization_online.csv")
Nx, Nc = rd("nocursor_exposure.csv"), rd("nocursor_classic.csv")
P, B = rd("participants_exposure.csv"), rd("blinkdetect_exposure.csv")

# ---------------------------------------------------------------- internal checks on the sign conventions
for nm, d in (("Lx", Lx), ("Lc", Lc), ("Lo", Lo)):
    checks.internal(f"{nm}: hand angle = atan2(y, x) of the hand position, tap angle = atan2 of the tap, taperror = tap - hand on every row",
                    float(np.max(np.abs(np.degrees(np.arctan2(d.handy_cm, d.handx_cm)) - d.handangle_deg))) < 0.01
                    and float(np.max(np.abs(np.degrees(np.arctan2(d.tapy_cm, d.tapx_cm)) - d.tapangle_deg))) < 0.01
                    and float(np.max(np.abs(d.tapangle_deg - d.handangle_deg - d.taperror_deg))) < 1e-4)
for nm, d in (("Nx", Nx), ("Nc", Nc)):
    dd = np.degrees(np.arctan2(d.handendy_cm, d.handendx_cm)) - d.target - d.endpoint_angle
    checks.internal(f"{nm}: endpoint_angle = atan2(y, x) of the endpoint minus the target angle (mod 360) on every row",
                    float(np.max(np.abs(wrap(dd)))) < 1e-6)
checks.internal("localization hand positions lie on the 10 cm arc (9.9 to 10.1 cm)",
                all(np.hypot(d.handx_cm, d.handy_cm).between(9.85, 10.15).all() for d in (Lx, Lc, Lo)))
ana = ~Nx.participant.isin(LOW)
aft_x = Nx[ana].groupby("rotated").endpoint_angle.median()
aft_c = Nc.groupby("rotated").endpoint_angle.median()
checks.internal("no-cursor reaches after rotation end counter-clockwise of the aligned ones (median shift > 0 deg) in both groups, "
                "as expected after a clockwise cursor rotation",
                aft_x[1] - aft_x[0] > 3 and aft_c[1] - aft_c[0] > 3, f"exposure {aft_x[1] - aft_x[0]:.1f}, classic {aft_c[1] - aft_c[0]:.1f} deg")

# ---------------------------------------------------------------- published checks
perf_rot = B[B.rotated_b == 1].set_index("participant").performance
checks.internal("the 3 lowest attention-task performers in the rotated session are exp015, exp007, exp018 (0.51, 0.52, 0.54); the next is 0.68",
                sorted(perf_rot.sort_values().index[:3]) == sorted(LOW) and perf_rot.sort_values().iloc[3] > 0.65)
n_exp = Lx.participant.nunique()
analysed = Lx[~Lx.participant.isin(LOW)]
S = "Methods, Participants"
checks.published("exposure group, participants analysed (25 recruited, 1 not following instructions, 3 low attention performance)",
                 n_exp - len(LOW), 21, 0, S)
checks.published("classic group, participants", Lc.participant.nunique(), 21, 0, S)
checks.published("exposure group, mean age of the analysed participants (years)", P[~P.participant.isin(LOW)].age.mean(), 20.1, 0.1, S)
checks.published("exposure group, SD of age (years)", P[~P.participant.isin(LOW)].age.std(), 2.3, 0.1, S)
checks.published("exposure group, number of women among the analysed participants", (P[~P.participant.isin(LOW)].sex == "F").sum(), 13, 0, S)
checks.published("exposure group, localization trials left after removal (16800 - 1024 = 15776; the paper's counts are 4 times the rows)",
                 4 * len(analysed), 15776, 0, "Methods, Data exclusions (1024 / 16800 removed)")
checks.published("classic group, localization trials left after removal (8400 - 220 = 8180; 4 times the delayed-localization rows)",
                 4 * len(Lc), 8180, 0, "Methods, Data exclusions (220 / 8400 removed)")
checks.published("classic group, no-cursor reaches left after removal (3969 - 1)", len(Nc), 3968, 0,
                 "Methods, Data exclusions (1 / 3969 removed); 21 participants x 9 blocks x 21 reaches = 3969")

# ---------------------------------------------------------------- rows
low_reason = "low performance on the attention (cursor blink) task during exposure training"
age = P.set_index("participant").age.to_dict()
sex = P.set_index("participant").sex.to_dict()
rows, loc_extra, nc_extra = [], [], []
order = []  # participants in order of first appearance, for the person table


def base(pid, group, seq, experiment):
    r = dict.fromkeys(names, "")
    low = pid in LOW
    r.update(dataset=ID, experiment=experiment, study=ID, subject=f"{pid}_{seq}", person=pid, id_type="lab",
             person_link="assumed_unique", group=group, prior_exposure="unknown",
             subject_excluded_by_authors="True" if low else "False", subject_exclusion_reason=low_reason if low else "",
             age=num(age[pid]) if pid in age else "", sex=sex.get(pid, ""), device="robot", setting="lab", population="healthy",
             session="0", trial_kind="reach", target_distance_norm="1", hand_used="right", fb_visible="none", fb_mode="none",
             fb_form="none", online_cursor="none", fb_endpoint_transform_only="False", excluded_by_authors="False")
    return r


def localization(d, group, experiment, seq, source):
    out = []
    for pid in d.participant.drop_duplicates():
        g = d[d.participant == pid].copy()
        g["_o"] = np.arange(len(g))
        g = g.sort_values(["rotated_b", "iteration", "_o"], kind="stable")
        for j, (_, x) in enumerate(g.iterrows()):
            passive = int(x.passive_b) == 1
            rot = int(x.rotated_b) == 1
            r = base(pid, group, seq, experiment)
            r.update(trial=str(j), block=f"{'rotated' if rot else 'aligned'}_{'passive' if passive else 'active'}_{int(x.iteration)}",
                     phase="probe" if rot else "baseline", trial_kind="proprio_probe" if passive else "reach",
                     break_before="True" if j == 0 else "False",
                     probe_kind="passive_robot" if passive else "active_post_reach", probe_location_deg=num(x.handangle_deg),
                     instr_goal="" if passive else "hand_to_target", hand_extent=num(math.hypot(x.handx_cm, x.handy_cm) / 10.0),
                     instr_report="hand_passive" if passive else "hand_post", hand_report=num(x.taperror_deg),
                     target_event="none")
            if not passive:
                r["instr_goal"] = ""  # free choice of direction: no instructed target
            out.append(r)
            loc_extra.append([r["subject"], j, source, int(x.rotated_b), int(x.iteration), int(x.passive_b), x.handx_cm, x.handy_cm,
                              x.handangle_deg, x.tapx_cm, x.tapy_cm, x.tapangle_deg, x.taperror_deg])
    return out


def nocursor(d, group, experiment, seq):
    out = []
    for pid in d.participant.drop_duplicates():
        g = d[d.participant == pid].copy()
        g["_o"] = np.arange(len(g))
        g = g.sort_values(["rotated", "repetition", "trial", "_o"], kind="stable")
        for j, (_, x) in enumerate(g.iterrows()):
            rot = int(x.rotated) == 1
            r = base(pid, group, seq, experiment)
            r.update(trial=str(j), trial_orig=str(int(x.trial)), block=f"{'rotated' if rot else 'aligned'}_noCursor_{int(x.repetition)}",
                     phase="aftereffect" if rot else "baseline", break_before="True" if (j == 0 or int(x.trial) == 0) else "False",
                     target_angle=num(x.target), target_id=str(TARGETS.index(int(x.target))), target_visible_during_move="True",
                     target_event="none", target_jump="0", probe_kind="none", instr_goal="hand_to_target", instr_strategy="unspecified",
                     instr_report="none", hand_angle=num(wrap(x.endpoint_angle)), hand_measure="endpoint",
                     hand_extent=num(math.hypot(x.handendx_cm, x.handendy_cm) / 10.0), rotation="")
            out.append(r)
            nc_extra.append([r["subject"], j, int(x.rotated), int(x.repetition), int(x.trial), int(x.target), x.handendx_cm, x.handendy_cm,
                             x.endpoint_angle])
    return out


rows += localization(Lx, "exposure", f"{ID}_ExposureLocalization", "loc", "localization_exposure")
rows += nocursor(Nx, "exposure", f"{ID}_ExposureNoCursor", "nc")
rows += localization(Lc, "classic", f"{ID}_ClassicLocalization", "loc", "localization_classic")
rows += localization(Lo, "classic", f"{ID}_ClassicOnlineLocalization", "locon", "localization_online")
rows += nocursor(Nc, "classic", f"{ID}_ClassicNoCursor", "nc")
checks.internal("every row the authors shared is kept (17078 = 4491 + 4512 + 2045 + 2062 + 3968)",
                len(rows) == 17078 == len(Lx) + len(Nx) + len(Lc) + len(Lo) + len(Nc))
checks.internal("hand_angle within (-180, 180]; 6 exposure reaches ended on the far side of home (|angle| > 90 deg) and are kept",
                all(-180 < float(r["hand_angle"]) <= 180 for r in rows if r["hand_angle"]))


def csv_text(head, data):
    b = io.StringIO()
    w = csv.writer(b, lineterminator="\r\n")
    w.writerow(head)
    for r in data:
        w.writerow([r[0]] + [num(v) if not isinstance(v, str) else v for v in r[1:]])
    return b.getvalue()


# the supplement keys are the raw sequence names; replace them with the codes the release assigns (order of first appearance)
code = {}
for r in rows:
    code.setdefault(r["subject"], f"{ID}_S{len(code) + 1:04d}")
for lst in (loc_extra, nc_extra):
    for r in lst:
        r[0] = code[r[0]]
loc_txt = csv_text(["subject", "trial", "source_file", "rotated_session", "iteration", "passive", "hand_x_cm", "hand_y_cm", "hand_angle_deg",
                    "tap_x_cm", "tap_y_cm", "tap_angle_deg", "tap_error_deg"], loc_extra)
nc_txt = csv_text(["subject", "trial", "rotated_session", "repetition", "trial_in_repetition", "target_deg", "endpoint_x_cm", "endpoint_y_cm",
                   "endpoint_angle_deg"], nc_extra)

ppl = []
bp = B.pivot(index="participant", columns="rotated_b", values="performance")
for pid in dict.fromkeys(r["person"] for r in rows):
    ppl.append(dict(person=pid, group="exposure" if pid.startswith("exp") else "classic",
                    analysed=("False" if pid in LOW else "True") if pid.startswith("exp") else "True",
                    attention_task_correct_aligned=bp.loc[pid, 0] if pid in bp.index else None,
                    attention_task_correct_rotated=bp.loc[pid, 1] if pid in bp.index else None))
pcols = [("group", "text", "", "exposure or classic training group."),
         ("analysed", "true/false", "", "False for the three exposure participants the authors left out for low performance on the attention task."),
         ("attention_task_correct_aligned", "number", "proportion", "Proportion correct on the cursor-blink attention task during the aligned session (exposure group only)."),
         ("attention_task_correct_rotated", "number", "proportion", "Same, rotated session.")]

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Henriques & 't Hart (York University)",
    data_url="https://osf.io/zfdth/",
    license="none stated",
    license_url="",
    changes="converted to the OpenMotor format, participant codes replaced, no-cursor endpoint angles wrapped into (-180, 180]; "
            "positions, tap angles and the attention-task scores are in extra tables.",
    unit_note="localization response or no-cursor reach",
    how=[
        "One row is one trial of a test task after training on a 30 degree visuomotor rotation, in a two-joint robot manipulandum "
        "with the right hand: either a no-cursor reach to one of 7 targets (15 to 75 deg, 10 cm from home; hand_angle = endpoint "
        "direction relative to the target), or a hand localization trial (the hand was stopped on a 10 cm arc, and the person tapped "
        "where it felt with the left index finger on a touch screen). Training trials are not in the deposit.",
        "Active localization (probe_kind active_post_reach, trial_kind reach): the person chose the direction of a reach; passive "
        "(passive_robot, proprio_probe): the robot moved the hand to the same places. hand_report is the tap angle minus the actual hand "
        "angle (the authors' taperror_deg), counter-clockwise positive; probe_location_deg is where the hand was, from the home "
        "position (0 deg right). hand_angle is empty on localization rows (no target). localization_details.csv has the positions.",
        "Groups: exposure (the robot moved the hand while the cursor went straight to the target; 24 participants in the deposit, 3 "
        "flagged as excluded by the authors for low attention-task performance), and classic (active reaches with a rotated cursor; "
        "21 participants of an earlier study). Each participant has up to three sequences (experiment column): localization, no-cursor "
        "reaches and, for the classic group, the authors' 'online' localization tasks, which this paper does not analyse and whose exact "
        "definition is not in the deposit. The order of tasks within a session is not in the files, so each task is its own sequence; "
        "within a sequence the aligned session comes first and localization is ordered by the authors' iteration number (inferred).",
        "Signs are counter-clockwise positive in the screen frame. The paper does not state them; the checked facts are that every "
        "angle equals atan2(y, x) of the stored coordinates, that no-cursor reaches after rotation end counter-clockwise of the aligned "
        "ones (about +6 deg in both groups), and that taps shift clockwise after rotation, as expected after a clockwise cursor rotation.",
    ],
    differences=[
        "The deposit already leaves out the trials the authors removed from localization (about 6 % exposure, 2 % classic). Their counts, "
        "4 times the rows, agree with the paper. For the no-cursor reaches of the exposure group the deposit has 3946 rows for the 21 "
        "analysed participants, fewer than the 3982 the paper's counts imply (3984 - 2); one participant is missing three blocks. "
        "The classic no-cursor count matches (3969 - 1 = 3968).",
        "Six exposure no-cursor reaches (all of analysed participants) end on the far side of home (|angle| greater than 90 deg after "
        "wrapping); the paper does not say whether they were removed, so they are kept and not flagged.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks, participants=ppl, participant_columns=pcols,
               extra_files=[("localization_details.csv", loc_txt), ("nocursor_details.csv", nc_txt)])
