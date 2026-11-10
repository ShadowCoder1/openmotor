"""Roth et al. (2023), Proc R Soc B: reinforcement-based regulation of motor exploration along redundant solution manifolds.
Three experiments x 18 participants, 500 reaches each (50 baseline, 200 first reinforcement condition, 50 washout, 200 second
condition), one row per reach in the 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Roth2023_ProcRSocB.py
Raw files (untrusted data, read only) in raw_extra/Roth2023_ProcRSocB/: Aim1_Exp<e>_Subject_<n>.pkl from
https://github.com/CashabackLab/Exploration-Along-Solution-Manifolds-Data (dill pickles of the authors' Exploration_Subject
class). They are read with the restricted reader in Roth2023_ProcRSocB_loader.py (no code from the files is run). The authors'
Exploration_Subject.py was read as text to learn the attributes: Target_Table (start, target and target rotation), Condition_Dict
(per condition, a list of one DataFrame per trial with hand x, y in m and event codes), Condition_Order, u_pos / v_pos.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
import Roth2023_ProcRSocB_loader as ld  # noqa: E402

ID = "Roth2023_ProcRSocB"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
EXPN = {1: "exp1_visible_target_size", 2: "exp2_hidden_reward_zone", 3: "exp3_reinforcement_probability"}


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def key(k):
    return str(k).replace(" ", "_")


rows, supp = [], []
acf = {}  # (exp, condition) -> list over persons
sub_idx = 0
for e in (1, 2, 3):
    for s in range(1, 19):
        o = ld.load(RAW / f"Aim1_Exp{e}_Subject_{s}.pkl")
        tt = o.Target_Table
        sx, sy, tx, ty = tt["X"][0] / 100, tt["Y"][0] / 100, tt["X"][1] / 100, tt["Y"][1] / 100
        td = np.array([tx - sx, ty - sy])
        nd = float(np.hypot(*td))
        tang = math.degrees(math.atan2(td[1], td[0])) % 360.0
        order = {key(k): int(v) for k, v in o.Condition_Order.items()}  # {condition code or name: block position}
        cds = [c for c in o.condition_list if c not in ("Baseline", "Washout")]
        # position 2 = first experimental block, 4 = second; the keys of Condition_Order are the authors' condition codes
        # (names in experiment 3), in the same order as the condition names for names, by code otherwise
        if all(not k.isdigit() for k in order):
            pos = {key(c): order[key(c)] for c in cds}
        else:
            raise_codes = order
            code_of = {"Reinforcement": "1", "Square": "7", "Small_Rect": "5"}
            pos = {c: order[code_of[c]] for c in cds}
        seq = sorted(cds, key=lambda c: pos[c])
        assert [pos[c] for c in seq] == [2, 4], pos
        group = f"{key(seq[0])}_then_{key(seq[1])}"
        blocks = [("Baseline", "Baseline", "baseline", 50), (seq[0], seq[0], "learning", 200), ("Washout", "Washout", "washout", 50),
                  (seq[1], seq[1], "learning", 200)]
        sub_idx += 1
        trial = 0
        for cname, label, phase, n in blocks:
            if cname == "Washout":
                # the deposit's Washout list is a copy of Baseline (identical in every participant): placeholders only
                assert all(np.array_equal(o.u_pos["Baseline"], o.u_pos["Washout"]) for _ in [0])
            for i in range(n):
                row = dict.fromkeys(names, "")
                hand = None
                if cname != "Washout":
                    df = o.Condition_Dict[cname][i]
                    hand = "Right_Hand" if "Right_HandX" in df.columns else "Left_Hand"
                row.update(
                    dataset=ID, experiment=EXPN[e], study=ID, subject=f"E{e}_{s}", person=f"E{e}_{s}", id_type="lab",
                    person_link="assumed_unique", group=group, prior_exposure="unknown", subject_excluded_by_authors="False",
                    population="healthy", device="kinarm", setting="lab", trial=trial, trial_orig=i + 1, session=0,
                    block=label, phase=phase, break_before="True" if trial == 0 else "False", target_angle=tang,
                    target_distance_norm=1, target_event="none", target_jump=0, probe_kind="none", instr_goal="hand_to_target",
                    hand_measure="endpoint", target_id=0, fb_form="none")
                if cname == "Washout":
                    row.update(trial_kind="missing", excluded_by_authors="True")
                else:
                    ev = df["Event_Codes"]
                    hs = df[ev == "HAND_STEADY"]
                    vx = hs[hand + "X"].iloc[0] - sx
                    vy = hs[hand + "Y"].iloc[0] - sy
                    vec = np.array([vx, vy])
                    ha = wrap(math.degrees(math.atan2(vy, vx)) - tang)
                    rel = vec - td
                    along = float(rel @ td / nd)
                    lat_right = -float((rel[1] * td[0] - rel[0] * td[1]) / nd)  # the authors' u: positive to the right of the reach
                    gi, ls, st = (df.index[ev == c][0] for c in ("GREENLIGHT", "LEFT_START", "HAND_STEADY"))
                    base = cname == "Baseline"
                    row.update(
                        trial_kind="reach", hand_used="right" if hand == "Right_Hand" else "left", hand_angle=ha,
                        hand_extent=float(np.hypot(vx, vy)) / nd, rt_ms=int(ls - gi), mt_ms=int(st - ls),
                        excluded_by_authors="False", hand_measure="endpoint",
                        fb_visible=("endpoint" if i < 40 else "none") if base else "none",
                        fb_mode="veridical" if base and i < 40 else "none", online_cursor="none",
                        fb_endpoint_transform_only="False",
                        outcome_channels="cursor" if base and i < 40 else ("reward_tone" if not base else ""),
                        reward="" if base else int(len(df.index[ev == "SUB_REWARD"]) > 0), reward_space="" if base else "hand")
                    k = "Baseline" if base else cname
                    ua, va = float(o.u_pos[k][i]), float(o.v_pos[k][i])
                    assert abs(ua - lat_right) < 1e-9 and abs(va - along) < 1e-9, (e, s, cname, i, ua, lat_right, va, along)
                    supp.append((f"{ID}_S{sub_idx:04d}", trial + 1, ua, va))
                rows.append(row)
                trial += 1
        for cname in seq:
            v = np.asarray(o.v_pos[cname])
            acf.setdefault((e, key(cname)), []).append(float(np.corrcoef(v[1:], v[:-1])[0, 1]))

n = len(rows)
checks.internal("54 sequences x 500 trials = 27000 rows", n == 27000)
checks.internal("washout placeholders: 50 per sequence", sum(r["trial_kind"] == "missing" for r in rows) == 54 * 50)
checks.published("participants per experiment", len({r["person"] for r in rows if r["experiment"] == EXPN[1]}), 18, 0, "Methods (18 per experiment)")
checks.published("participants total", len({r["person"] for r in rows}), 54, 0, "Methods (54 total)")
checks.published("trials per participant", 500, 500, 0, "Methods (50 + 200 + 50 + 200)")
checks.published("trials per participant (rows)", n / 54, 500, 0, "Methods (50 + 200 + 50 + 200)")
for ex, (a, b), exp_pct in ((1, ("Reinforcement", "Square"), 77.78), (2, ("Small_Rect", "Square"), 77.78), (3, ("High_Frequency", "Low_Frequency"), 66.67)):
    pa, pb = np.array(acf[(ex, a)]), np.array(acf[(ex, b)])
    print(f"exp{ex}: lag-1 ACF of extent, {a} > {b} in {int((pa > pb).sum())}/18; {b} > {a} in {int((pb > pa).sum())}/18")
    acf[("n", ex)] = (a, b, int((pa > pb).sum()), int((pb > pa).sum()), exp_pct)
for ex in (1, 2, 3):
    a, b, na, nb, exp_pct = acf[("n", ex)]
    best = na
    checks.published(f"experiment {ex}: participants with the larger lag-1 autocorrelation of extent in the condition pair of the paper (%)",
                     100 * best / 18, exp_pct, 0.01, "Results, serial dependence of movement extent (theta-hat)")

import io  # noqa: E402
sb = io.StringIO()
sb.write("subject,trial,aim_lateral_m,extent_m\r\n")
for r in supp:
    sb.write(f"{r[0]},{r[1]},{r[2]:.8g},{r[3]:.8g}\r\n")

paper = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Cashaback (University of Delaware)", data_url="https://github.com/CashabackLab/Exploration-Along-Solution-Manifolds-Data",
    license="none stated", license_url="",
    changes="the authors' per-trial hand recordings were reduced to the endpoint of each reach (hand position when it came to rest) "
            "and expressed as a direction and an extent relative to the target; participant codes replaced.",
    unit_note="reach of one participant",
    how=[
        "One row is one reach of one participant (18 per experiment, 3 experiments, 54 people), toward a target 17 cm away at "
        "135 deg in the robot's frame (start and target positions are from the authors' Target_Table, in cm). Each sequence is "
        "50 baseline reaches, 200 reaches in one reinforcement condition, 50 washout reaches and 200 in the other condition; "
        "group gives the order of the two conditions (the authors' Condition_Order). Condition names are the authors' own.",
        "hand_angle is the direction of the final hand position (the sample at the HAND_STEADY event) from the start, minus the "
        "target direction, wrapped to (-180, 180]; counter-clockwise positive. hand_extent is the distance from the start in units "
        "of the target distance. trial_kind is reach. The authors' own coordinates (aim = u, positive to the right of the reach, "
        "and extent = v, positive beyond the target, both in m relative to the target, as in their code) are in "
        "aim_extent.csv; they match the values recomputed here exactly.",
        "rt_ms and mt_ms use the authors' event times (index of each trial's data frame, in ms): reaction time from GREENLIGHT to "
        "LEFT_START and movement time from LEFT_START to HAND_STEADY (the definitions in their code). reward is 1 when the trial "
        "has the authors' SUB_REWARD event (pleasant sound, brief target expansion and a small payment) and 0 otherwise; empty in "
        "baseline, where there was none. Which condition is the task-redundant one is not labelled in the deposit; the paper's "
        "serial-dependence result is reproduced when it is Reinforcement in experiment 1, Small_Rect in experiment 2 and "
        "High_Frequency (80% reinforcement) in experiment 3, which is how the names are read. "
        "Cursor feedback of the final hand position was given on the first 40 baseline trials "
        "(fb_visible endpoint) and no feedback on the last 10 (paper, Methods). The target and reward-zone sizes are not in the "
        "deposit, so hit and the reward zone columns are empty.",
        "The deposit's Washout block is a copy of its Baseline block in every participant (the same values), so it is not "
        "converted: the 50 washout positions of each sequence are placeholder rows (trial_kind missing) that keep the trial "
        "numbers of the second condition at 300-499. Which participants used the left hand is from the authors' column names "
        "(Left_Hand: hand_used left).",
    ],
    differences=["The deposit's Washout lists repeat the Baseline lists (authors' u_pos is identical for the two in all 54 "
                 "participants), so washout reaches could not be recovered; they are placeholder rows.",
                 "The GitHub repository has no license file; the same data are on Dryad (doi 10.5061/dryad.ngf1vhj10) under CC0, "
                 "which would allow sharing. The Zenodo copy of that deposit holds only 4 kB placeholders for the three data folders, "
                 "so the GitHub files were used. 'none stated' is recorded here as the brief allows."],
)
stdlib.release(ID, "reach90", rows, meta, checks, extra_files=[("aim_extent.csv", sb.getvalue())])
