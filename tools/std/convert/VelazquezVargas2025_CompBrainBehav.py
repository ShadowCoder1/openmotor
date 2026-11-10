"""Velazquez-Vargas & Taylor (2025), Comput Brain Behav: learning a mapping of eight keys (A, S, D, F, H, J, K, L) to the eight
chess-knight moves on a 15 x 15 grid; Experiment 1 (Full group and Mapping group, 25 each) and Experiment 2 (horizon, 25),
one row per trial.

Raw files (untrusted data, read only) in raw_extra/<id>/, from OSF project wm5fu (Data/Experiment 1/Raw data, Data/Experiment 2),
named E<experiment>_<original name>: E1_data_full.csv, E1_data_mapping.csv, E1_subjectInfo_full.csv, E1_subjectInfo_mapping.csv,
E2_data_horizon.csv, E2_subjectsInfo_horizon.csv (E1_Clicks_per_subject.csv is not used).
Run from the site folder: python3 -I tools/std/convert/VelazquezVargas2025_CompBrainBehav.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "VelazquezVargas2025_CompBrainBehav"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
W = 15
OFF = {dx - W * dy: (dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if sorted((abs(dx), abs(dy))) == [1, 2]}  # knight moves
FILES = [("Experiment 1", "Full", "E1_data_full.csv", "E1_subjectInfo_full.csv"),
         ("Experiment 1", "Mapping", "E1_data_mapping.csv", "E1_subjectInfo_mapping.csv"),
         ("Experiment 2", "", "E2_data_horizon.csv", "E2_subjectsInfo_horizon.csv")]
SEX = {1: "M", 2: "F", 3: "other", 4: ""}
DEVICE = "laboratory computer keyboard, keys A, S, D, F, H, J, K and L"

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for _, _, df_, if_ in FILES:
    for fn in (df_, if_):
        checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])


def read(fn):
    d = pd.read_csv(RAW / fn, low_memory=False, encoding="utf-8-sig")
    d.columns = [c.replace(".", " ") for c in d.columns]
    return d.rename(columns={"Old or New S T": "Old or New S-T"})


def nn(v):
    return None if pd.isna(v) else v


rows, S = [], []
bad = {}
gaps_empty = []
for exp, grp, dfile, ifile in FILES:
    d, info = read(dfile), read(ifile)
    d["sid"] = d["Subject ID"].str.extract(r"subject:\s*(\d+)\s+trial")[0]
    info["sid"] = info["Subject ID"].astype(str)
    checks.internal(f"{dfile}: every participant of the data file is in the subject info, once, and 25 of them",
                    set(d.sid) == set(info.sid) and not info.sid.duplicated().any() and d.sid.nunique() == 25)
    info = info.set_index("sid")
    for sid, g in d.groupby("sid", sort=False):
        ir = info.loc[sid]
        offs = {}
        for _, r in g.iterrows():
            if pd.isna(r["Key presses"]):
                continue
            keys, cells, hit = json.loads(r["Key presses"]), json.loads(r["State locations"]), json.loads(r["Hit the wall"])
            for k, a, b, w_ in zip(keys, cells[:-1], cells[1:], hit):
                if w_ == 0:
                    offs.setdefault(k, []).append(b - a)
        bad["mapping not constant"] = bad.get("mapping not constant", 0) + (not all(len(set(v)) == 1 for v in offs.values()))
        bad["offset not a knight move"] = bad.get("offset not a knight move", 0) + sum(1 for v in offs.values() if v[0] not in OFF)
        km = ";".join(f"{chr(k)}={OFF[offs[k][0]][0]},{OFF[offs[k][0]][1]}" for k in sorted(offs, key=chr))
        person = f"{exp}|{dfile}|{sid}"
        one_move = 0
        for _, r in g.iterrows():
            tnum = int(r["Trial number"])
            base = dict(
                dataset=ID, study=ID, experiment=exp, group=grp, subject=person, person=person, subject_excluded_by_authors=False,
                age=float(ir["Age"]), sex=SEX[int(ir["Gender"])], handedness="", population="healthy", setting="lab", device=DEVICE,
                session=1, trial=tnum, phase="learning", time_s="", task="sequence learning", unit="trial", pert_type="none",
                pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False, grid_size=W, key_moves=km,
                world_code=nn(r["World codes"]) or "", navigation_code=str(ir["Navigation codes"]))
            if pd.isna(r["Key presses"]):          # trial 500 of four Experiment 2 participants: the authors' row is empty
                rows.append({**base, "block": "main", "condition": "", "min_moves": int(r["Min moves"]), "start_cell": "", "target_cell": ""})
                bad["empty rows"] = bad.get("empty rows", 0) + 1
                continue
            keys, cells, iki, hit = (json.loads(r[c]) for c in ("Key presses", "State locations", "Inter key time", "Hit the wall"))
            n = int(r["Number of moves"])
            assert len(keys) == n == len(iki) == len(hit) and len(cells) == n + 1 and cells[0] == r["Start location"], (dfile, sid)
            assert all(1 <= c <= W * W for c in cells) and all(chr(k) in "ASDFHJKL" for k in keys)
            bad["wall press moved"] = bad.get("wall press moved", 0) + sum(1 for a, b, w_ in zip(cells[:-1], cells[1:], hit) if w_ and a != b)
            bad["feedback 0 but reached"] = bad.get("feedback 0 but reached", 0) + (r["Feedback type"] == 0 and cells[-1] == r["Target location"] and int(r["Trial time"]) <= 10000)
            bad["feedback >0 but not reached"] = bad.get("feedback >0 but not reached", 0) + (r["Feedback type"] > 0 and cells[-1] != r["Target location"] and int(r["Trial time"]) != 10000)
            bad["late arrival coded 0"] = bad.get("late arrival coded 0", 0) + (r["Feedback type"] == 0 and cells[-1] == r["Target location"])
            bad["points at the 10 s limit without arrival"] = bad.get("points at the 10 s limit without arrival", 0) + (r["Feedback type"] > 0 and cells[-1] != r["Target location"])
            is1 = (int(r["Target location"]) - int(r["Start location"])) in OFF       # target one knight move from the start
            if exp == "Experiment 2":
                block, cond = "main", f"target {int(r['Min moves'])} move{'s' if int(r['Min moves']) > 1 else ''} away"
            else:
                block = "mapping (targets 1 move away)" if is1 else "3-move targets"
                cond = "target 1 move away" if is1 else "target 3 moves away"
            rows.append({**base, "block": block, "condition": cond, "start_cell": int(r["Start location"]),
                         "target_cell": int(r["Target location"]), "n_moves": n, "keys_pressed": ";".join(chr(k) for k in keys),
                         "cells_visited": ";".join(map(str, cells)), "inter_key_ms": ";".join(map(str, iki)),
                         "wall_hits": ";".join(map(str, hit)), "trial_time_ms": int(r["Trial time"]), "feedback_type": int(r["Feedback type"]),
                         "min_moves": int(r["Min moves"]) if "Min moves" in r and not pd.isna(r["Min moves"]) else "",
                         "points": float(r["Points On Trial"]), "points_total": float(r["Cumulative points"]),
                         "no_moves_flag": bool(r["Trial with no moves"]), "feedback_shown": bool(r["Is there feedback"]),
                         "new_pair": bool(r["Old or New S-T"])})
        pts = g["Cumulative points"].dropna()
        gap = float(ir["Game points"]) - float(pts.iloc[-1])
        if g["Key presses"].isna().any():
            gaps_empty.append(gap)
        else:
            bad["game points != last cumulative"] = bad.get("game points != last cumulative", 0) + (gap != 0)
        S.append(dict(person=person, exp=exp, group=grp, age=float(ir["Age"]), gender=int(ir["Gender"]), n_trials=len(g), finished=bool(ir["Experiment finished"]),
                      first=bool(ir["First attempt"]), time=ir["Experiment time"], game_points=ir["Game points"],
                      trial_nav=g["Navigation codes"].dropna().iloc[0], assignment=str(ir["Navigation codes"])))
t = pd.DataFrame(rows)
del rows
S = pd.DataFrame(S)
for k, lab in (("mapping not constant", "every participant's keys make one fixed move (presses that did not hit a wall)"),
               ("offset not a knight move", "every key's move is one of the eight chess-knight moves"),
               ("wall press moved", "a press that hit a wall never moved the cursor"),
               ("feedback 0 but reached", "feedback type 0 whenever the cursor did not end on the target, and when it did only after the 10 s limit (late arrivals score 0)"),
               ("feedback >0 but not reached", "feedback type 1 or 2 only when the cursor ended on the target, apart from trials logged at exactly 10000 ms"),
               ("game points != last cumulative", "the subject info's game points equal the last cumulative points of the trial file, for every participant whose trial 500 is present")):
    checks.internal(lab, bad.get(k, 0) == 0, f"{bad.get(k, 0)} exceptions")
checks.internal("four Experiment 2 participants have an empty row for trial 500 (kept, values empty)", bad.get("empty rows", 0) == 4)

# ---------------------------------------------------------------- the paper's numbers
M = "Methods, Participants"
checks.published("Experiment 1: participants in the Full group", ((S.exp == "Experiment 1") & (S.group == "Full")).sum(), 25, 0, M)
checks.published("Experiment 1: participants in the Mapping group", ((S.exp == "Experiment 1") & (S.group == "Mapping")).sum(), 25, 0, M)
checks.published("Experiment 2: participants", (S.exp == "Experiment 2").sum(), 25, 0, M)
checks.published("all experiments: participants", len(S), 75, 0, M)
gc = S.gender.value_counts()
for code, name, rep in ((1, "male", 34), (2, "female", 39), (3, "non-binary", 2)):
    checks.published(f"all experiments: {name} participants", gc.get(code, 0), rep, 0, M)
AGE_M, AGE_SD = S.age.mean(), S.age.std(ddof=1)
checks.published("all experiments: mean age (years; the paper's decimals look truncated, so one unit of the last digit is allowed)", AGE_M, 19.6, 0.1, M)
checks.published("all experiments: SD of age (years; same allowance)", AGE_SD, 1.5, 0.1, M)
tp = S.groupby("exp").n_trials.agg(["min", "max"])
checks.published("Experiment 1: trials per participant (all 500)", tp.loc["Experiment 1", "max"] if tp.loc["Experiment 1", "min"] == 500 else -1, 500, 0, "Methods, Experiment 1")
checks.published("Experiment 2: trials per participant (all 500)", tp.loc["Experiment 2", "max"] if tp.loc["Experiment 2", "min"] == 500 else -1, 500, 0, "Methods, Experiment 2")
e1 = t[t.experiment == "Experiment 1"]
b1 = e1.groupby(["group", "subject"]).block.value_counts().unstack(fill_value=0)
print("  [probe] Experiment 1 block counts per participant:", {g: b1.loc[g].drop_duplicates().to_dict("records") for g in ("Full", "Mapping")})
mp = b1.loc["Mapping"]
checks.published("Experiment 1, Mapping group: trials with targets 1 move away (every participant)", mp["mapping (targets 1 move away)"].iloc[0] if mp["mapping (targets 1 move away)"].nunique() == 1 else -1, 200, 0, "Methods, Experiment 1")
checks.published("Experiment 1, Full group: trials with targets 3 moves away (every participant)", b1.loc["Full"]["3-move targets"].iloc[0] if b1.loc["Full"]["3-move targets"].nunique() == 1 else -1, 500, 0, "Methods, Experiment 1")
t2 = t[(t.experiment == "Experiment 2") & (t.target_cell != "")]
checks.internal("Experiment 2: every participant has 125 trials of each horizon (1-4 moves, the authors' min moves), 124 for the "
                "four whose trial 500 is empty", t[t.experiment == "Experiment 2"].groupby(["subject", "min_moves"]).size().isin([124, 125]).all()
                and set(t[t.experiment == "Experiment 2"].min_moves) == {1, 2, 3, 4})
tg = t2.groupby("min_moves").target_cell.nunique()
checks.published("Experiment 2: unique target locations per horizon (8 for each of the four horizons)", tg.min() if tg.min() == tg.max() else -1, 8, 0,
                 "Methods, Experiment 2")
checks.published("Experiment 2: unique target locations in total", t2.target_cell.nunique(), 32, 0, "Methods, Experiment 2")
checks.published("Experiment 1, Full group: unique target locations (20 clover targets)", e1[e1.group == "Full"].target_cell.nunique(), 20, 0,
                 "Methods, Experiment 1")


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return float(r.statistic), float(r.pvalue), float((va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)))


e1 = e1.assign(opt=(e1.feedback_type == 2).astype(float), arr=(e1.feedback_type >= 1).astype(float))
PUB = {(1, 200): (8.06, 47.98, None), (201, 500): (0.37, 41.56, 0.71)}
for (lo, hi), (tr, dfr, pr) in PUB.items():
    pm = e1[e1.trial.between(lo, hi)].groupby(["group", "subject"]).points.mean().unstack(0)
    tt, pp, df = welch(pm["Mapping"].dropna(), pm["Full"].dropna())
    R = "Results, Experiment 1 (mean points per participant, Mapping vs Full, Welch t-test; the paper's decimals look truncated)"
    checks.published(f"Experiment 1, trials {lo}-{hi}: Welch t of mean points per participant, Mapping vs Full", tt, tr, 0.01, R)
    checks.published(f"Experiment 1, trials {lo}-{hi}: Welch degrees of freedom", df, dfr, 0.02, R)
    if pr:
        checks.published(f"Experiment 1, trials {lo}-{hi}: p value", pp, pr, 0.01, R)
f200 = e1[(e1.group == "Full") & e1.trial.between(1, 200)]
checks.published("Experiment 1, Full group, first 200 trials: median number of key presses per trial", f200.n_moves.median(), 7, 0,
                 "Results, Experiment 1")

for c in checks.items:
    if c["kind"] == "published":
        print(f"  {'ok ' if c['ok'] else 'BAD'} {c['what']}: computed {c['computed']:.4f}, reported {c['reported']}")
    elif not c["ok"] or c.get("detail"):
        print(f"  {'ok ' if c['ok'] else 'FAILED'} [internal] {c['what']} {c.get('detail', '')}")

# ---------------------------------------------------------------- release
names = [c[1] for c in stdlib.columns_for("navigation")]
for c in names:
    if c not in t.columns:
        t[c] = ""
t = t[names]
entry = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
partrows = [dict(person=r.person, experiment=r.exp, group=r.group, experiment_finished=r.finished, first_attempt=r.first,
                 experiment_time_ms=r.time, game_points=r.game_points, trial_file_navigation_code=r.trial_nav,
                 key_assignment_code=r.assignment) for r in S.itertuples()]
pcols = [("experiment", "text", "", "Experiment of the participant."),
         ("group", "text", "", "Full or Mapping group (Experiment 1); empty in Experiment 2."),
         ("experiment_finished", "true/false", "", "The authors' 'experiment finished' flag in the subject info."),
         ("first_attempt", "true/false", "", "The authors' 'first attempt' flag in the subject info."),
         ("experiment_time_ms", "number", "ms", "Total time of the experiment as the authors recorded it."),
         ("game_points", "number", "points", "The authors' 'game points' of the participant (their total score)."),
         ("trial_file_navigation_code", "text", "", "The 'navigation codes' value in the participant's trial file (the same on every row)."),
         ("key_assignment_code", "text", "", "The 'navigation codes' value in the subject info: the authors' list of eight knight-move codes, "
          "one per key, as in their file.")]
meta = dict(
    study=ID,
    paper={k: entry["papers"][0][k] for k in ("authors", "year", "title", "venue", "doi")},
    lab=entry["lab"],
    data_url="https://osf.io/wm5fu/",
    license="none stated",
    license_url="",
    changes=("Converted the per-trial CSV files of the three data sets into one table, decoded the key codes to letters, wrote the "
             "lists of each trial as semicolon-separated text, added each participant's age and gender from the subject-info files, "
             "labelled trials by the target's distance, and replaced participant identifiers (session start times) with codes."),
    how=[
        "One row is one trial of the knight-move task: a frog (cursor) started in the centre cell (start_cell 113) of a 15 x 15 grid "
        "(cells numbered 1-225 row by row from the top left) and the participant pressed the keys A, S, D, F, H, J, K and L, each "
        "moving the frog one chess-knight move, to reach a target (clover) within 10 s. The key-to-move assignment differs between "
        "participants and is listed in key_moves, read off the cursor's moves (steps right, steps up).",
        "keys_pressed, cells_visited (start cell plus the cell after every press), inter_key_ms (time from trial start to the first "
        "press, then between presses) and wall_hits are the authors' per-press lists; feedback_type is their outcome code (0 target not "
        "reached, 1 reached with extra presses, 2 reached in the fewest presses); points and points_total are their trial and "
        "cumulative points. Four Experiment 2 participants have an empty row for trial 500, kept with empty values.",
        "Experiment 1 groups: Full (500 trials with targets 3 moves away) and Mapping (200 trials with targets 1 move away, block "
        "'mapping (targets 1 move away)', then 300 with targets 3 moves away); a target is 1 move away when it is one knight move "
        "from the start cell. Experiment 2 (horizon): targets 1 to 4 moves away (the authors' min_moves, condition). The deposit does "
        "not flag the paper's no-feedback planning trials, and 'is there feedback' is 1 on every row.",
        "No participant is marked as excluded: the paper reports no exclusions and gives 25 participants per group, as in the files.",
    ],
    unit_note="trial of the knight-move navigation task",
    differences=[
        "Mean age of the 75 participants is {:.3f} years (SD {:.3f}) here, printed as 19.6 (1.5) in the paper; the paper's decimals look "
        "truncated rather than rounded, so the checks allow one unit of the last digit.".format(AGE_M, AGE_SD),
        "For the four Experiment 2 participants whose trial 500 is empty, the subject info's game points are lower than the trial file's last "
        "cumulative points, by {:.0f} to {:.0f} points; for the other 71 participants they are equal.".format(-max(gaps_empty), -min(gaps_empty)),
        "The paper's Welch t-test of key presses per trial in the first 200 trials (Mapping vs Full, t(35.42) = -14.75) is not reproduced "
        "by the participants' mean number of key presses (t = -9.475, 24 degrees of freedom); the measure behind it is not stated.",
        "{} trials are logged as arrivals after the 10 s limit with feedback type 0 and 0 points, and {} trials logged at exactly 10000 "
        "ms have points although the logged cells do not end on the target; both are kept as logged.".format(
            bad.get("late arrival coded 0", 0), bad.get("points at the 10 s limit without arrival", 0)),
    ],
)
stdlib.release(ID, "navigation", t, meta, checks, participants=partrows, participant_columns=pcols)
