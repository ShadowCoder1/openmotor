"""Leow, Bernheine, Carroll, Dux & Filmer (2024), eNeuro: dopamine and explicit motor skill learning (OSF wnfxt, no license stated).
Raw (untrusted, read only): mentalrotS1.mat (session 1, drug or placebo) and mentalrotS2.mat (session 2, placebo for everyone),
each (38 people x 560 trials) arrays; Error.sav (drug group, sex of each person; the .sav files peakvelocity/RT are per-bin summaries).
Rows are in the authors' arrays order, which is sorted by instructed angle (0, 30, 60, 120, 135, 150, 165; 80 trials each), not by the
randomized order in which people did the blocks.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Leow2024_eNeuro"
RAW = stdlib.raw_dir(ID)
names = [c[1] for c in stdlib.reach90_columns()]
checks = stdlib.Checks()
sav, _ = pyreadstat.read_sav(RAW / "Error.sav")
sav = sav[sav.Subject.astype(str).str.startswith("Y")].set_index("Subject")
M = [sio.loadmat(RAW / f, struct_as_record=False, squeeze_me=True) for f in ("mentalrotS1.mat", "mentalrotS2.mat")]
rows, ppl = [], []
n_missing = {}
for s in range(38):
    num = int(re.search(r"\\Y(\d+)\\S", str(M[0]["Paths"][s])).group(1))
    assert int(re.search(r"\\Y(\d+)\\S", str(M[1]["Paths"][s])).group(1)) == num
    key = "Y%02d" % num
    drug, sex = sav.DrugCond[key], {"m": "M", "f": "F"}.get(str(sav.gender[key]).strip().lower(), "")
    ppl.append(dict(person=key, drug_group=drug, sex=sex))
    for ses, m in enumerate(M):
        J = m["TargetJumpAngs"][s].astype(int)
        for t in range(560):
            ang = float(m["angrew"][s, t])
            miss = m["MovementTime"][s, t] == 0 and m["ReactionTime"][s, t] == 0
            n_missing[(key, ses)] = n_missing.get((key, ses), 0) + int(miss)
            j = int(J[t])
            if miss:
                j = 165  # placeholder rows of the unfinished block (zeros in the deposit)
            r = dict.fromkeys(names, "")
            r.update(dataset=ID, experiment="explicit_aiming", study=ID, subject=f"{key}_s{ses + 1}", person=key, id_type="lab",
                     person_link="assumed_unique", group=drug, prior_exposure="unknown", subject_excluded_by_authors="False",
                     sex=sex, device="robot", setting="lab", population="healthy",
                     trial=ses * 560 + t, session=ses, block=f"aim_{j}deg", phase="baseline" if j == 0 else "learning",
                     trial_kind="missing" if miss else "reach", break_before=str(t == 0 or J[t] != J[t - 1]),
                     target_angle=float(m["TargetAngs"][s, t]), target_distance_norm="1", target_visible_during_move="True",
                     target_event="jump" if j else "none", target_jump=-j, instr_goal="hand_to_angle", goal_angle_deg=-j,
                     instr_strategy="given_with_angle", strategy_angle_deg=-j, instr_disclosure="exact", instr_report="none",
                     hand_used="right" if False else "", hand_angle="" if miss else -ang, hand_measure="fixed_distance",
                     rt_ms=float(m["ReactionTime"][s, t]) * 1000, mt_ms=float(m["MovementTime"][s, t]) * 1000,
                     excluded_by_authors=str(bool(miss)), fb_visible="online", fb_mode="veridical", fb_form="cursor",
                     online_cursor="veridical", rotation="0", outcome_channels="reward_tone", reward=int(m["Feedback"][s, t]),
                     reward_center_deg=-j, reward_halfwidth_deg=10)
            if miss:
                for k in ("rt_ms", "mt_ms", "reward", "target_angle", "target_event", "target_jump", "goal_angle_deg", "strategy_angle_deg",
                          "reward_center_deg", "reward_halfwidth_deg", "instr_goal", "instr_strategy", "outcome_channels", "fb_visible",
                          "fb_mode", "fb_form", "online_cursor", "rotation"):
                    r[k] = ""
            rows.append(r)

P = pd.DataFrame(ppl)
checks.published("participants", len(P), 38, 0, "Methods, Participants")
checks.published("levodopa group", (P.drug_group == "levodopa").sum(), 19, 0, "Methods, Design")
checks.published("placebo group", (P.drug_group == "placebo").sum(), 19, 0, "Methods, Design")
checks.published("women", (P.sex == "F").sum(), 27, 0, "Methods, Participants")
checks.published("aiming blocks per session (0, 30, 60, 120, 135, 150, 165 deg)", 7, 7, 0, "Methods, Task")
checks.published("trials per aiming block (10 bins x 8)", 560 / 7, 80, 0, "Methods, Task")
miss_people = {k for (k, ses), n in n_missing.items() if ses == 1 and n > 0}
checks.published("participants missing the 165 deg block (two levodopa people; the deposit has them in session 2)", len(miss_people), 2, 0, "Methods, Participants")
checks.internal("the missing 165 deg block belongs to levodopa people", all(P.drug_group[P.person == k].iloc[0] == "levodopa" for k in miss_people))
checks.internal("rows", len(rows) == 38 * 1120, str(len(rows)))
J1 = M[0]["TargetJumpAngs"].astype(int)
checks.internal("late-block hand angle follows instruction: median |angle| for 135 deg blocks within 10 deg of 135",
                abs(np.nanmedian(np.abs(M[0]["angrew"][J1 == 135])) - 135) < 10)
paper = dict(authors="Leow LA, Bernheine L, Carroll TJ, Dux PE, Filmer HL", year=2024,
             title="Dopamine Increases Accuracy and Lengthens Deliberation Time in Explicit Motor Skill Learning",
             venue="eNeuro 11(1):ENEURO.0360-23.2023", doi="10.1523/ENEURO.0360-23.2023")
meta = dict(
    study=ID, paper=paper, lab="Carroll & Dux (University of Queensland)", data_url="https://osf.io/wnfxt/", license="none stated",
    license_url="",
    changes="converted to the OpenMotor format; participant numbers replaced by codes; the authors' reach angle (angrew) multiplied by -1 "
            "to the counter-clockwise-positive convention; reaction and movement times converted from s to ms; trial order as in the authors' arrays.",
    unit_note="trial (one reach) of one participant",
    how=[
        "One row is one reach on a vBot planar robot to one of 8 targets 9 cm from the start (0, 45, ... 315 deg). In each of 7 blocks people had to "
        "aim a set angle clockwise of the target (0, 30, 60, 120, 135, 150 or 165 deg, told before the block; 80 trials per block); the target jumped "
        "to the instructed location when the hand passed 4 cm, a red cursor was shown from 4 cm on, and a coin sound played when the reach angle at "
        "4 cm was within 10 deg of the instructed angle. Session 1: 19 people got levodopa and 19 placebo before the session (group column); session 2 (at least 18 h later) was placebo for all.",
        "hand_angle is the authors' 'angrew' (the angle used for the sound), multiplied by -1. In the authors' arrays the median angle follows the instruction as a "
        "positive number (about +31 for the 30 deg block, +134 for 135), so their frame is clockwise-positive; the instruction says clockwise, so hand_angle, goal_angle_deg, "
        "strategy_angle_deg and target_jump are negative here (inferred from the data; the direction of the target jump is not stated, we assume it follows the instruction). "
        "reward is 1 when the sound played (in the data this is exactly |errorrew| < 10 deg). target_angle is the authors' TargetAngs, presumed in the screen frame.",
        "The arrays are sorted by instructed angle, not by the randomized order in which people did the blocks, and the 20-trial washout blocks are not in the deposit, so trial "
        "is the position in the authors' array (session 2 continues from 560). The two levodopa people who skipped the 165 deg block have 79 of its 80 rows (session 2) as placeholder rows (trial_kind missing; zeros in the deposit).",
        "Not in the reach90 columns and not copied: peak velocity, the other angle measures (angEnd, ang20perc, ang80perc, ang100ms), movement-reaction time, and the per-bin summary files "
        "(peakvelocity.sav, RT.sav, Error.sav); age is not in the deposit (paper: median 21.58 years). The paper says accuracy is measured at 1.8 cm; the beep rule at 4 cm. "
        "The deposit does not say at what distance angrew is taken; we describe it as a fixed-distance angle.",
    ],
    differences=[],
)
stdlib.release(ID, "reach90", rows, meta, checks, participants=ppl,
               participant_columns=[("drug_group", "text", "", "Session 1 drug: levodopa (Madopar 125) or placebo."),
                                    ("sex", "category", "F, M", "Sex from the deposit's Error.sav.")])
