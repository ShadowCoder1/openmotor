"""Velazquez-Vargas, Daw & Taylor (2024), PLoS Comput Biol: learning a 3-key de novo mapping (J, K, L) that steers a cursor over a
9 x 9 grid; Experiments 1-4 (Single vs Multiple start-target pairs; stochastic mapping in Experiment 4), one row per trial.

Raw files (untrusted data, read only) in raw_extra/<id>/, from OSF project trjsg, named E<experiment>_<original name>:
  E1_data_Generalization_SONA.csv, E1_subjectsInfo_Generalization_SONA.csv
  E2_data_Generalization_single.csv / _multiple.csv and E2_subjectsInfo_Generalization_single.csv / _multiple.csv
  E3_data_Generalization_forget_1STD_2h.csv, E3_subjectsInfo_Generalization_forget_1STD_2h.csv
  E4_subjects_data_stochastic.csv, E4_subjects_info_stochastic.csv
  small files (directions, map, condition vectors) used only as documentation.
Run from the site folder: python3 -I tools/std/convert/VelazquezVargas2024_PLoSCompBiol.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "VelazquezVargas2024_PLoSCompBiol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
KEY = {74: "J", 75: "K", 76: "L"}
OFF = {1: (1, 0), -1: (-1, 0), 9: (0, -1), -9: (0, 1), 8: (-1, -1), -8: (1, 1), 10: (1, -1), -10: (-1, 1)}  # cell offset -> (right, up)
GROUP = {1: "Single", 4: "Multiple"}
FILES = [("Experiment 1", "E1_data_Generalization_SONA.csv", "E1_subjectsInfo_Generalization_SONA.csv"),
         ("Experiment 2", "E2_data_Generalization_single.csv", "E2_subjectsInfo_Generalization_single.csv"),
         ("Experiment 2", "E2_data_Generalization_multiple.csv", "E2_subjectsInfo_Generalization_multiple.csv"),
         ("Experiment 3", "E3_data_Generalization_forget_1STD_2h.csv", "E3_subjectsInfo_Generalization_forget_1STD_2h.csv"),
         ("Experiment 4", "E4_subjects_data_stochastic.csv", "E4_subjects_info_stochastic.csv")]
SEX = {1: "M", 2: "F", 3: "other", 4: ""}
DEVICE = "laboratory computer keyboard, keys J, K and L"

src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
for _, df_, if_ in FILES:
    for fn in (df_, if_):
        checks.internal(f"{fn} matches the downloaded hash", stdlib.sha256(RAW / fn) == src[fn]["sha256"])


def read(fn):
    d = pd.read_csv(RAW / fn, low_memory=False, encoding="utf-8-sig")
    d.columns = [c.replace(".", " ") for c in d.columns]
    return d.rename(columns={"Old or New S T": "Old or New S-T"})


rows, parts = [], []
sub_info = {}          # (experiment, sid) -> dict
dev = {"learning": [0, 0], "transfer": [0, 0]}   # stochastic deviations in Experiment 4: [deviating presses, presses]
bad = {}
explicit = {}
for exp, dfile, ifile in FILES:
    d, info = read(dfile), read(ifile)
    d["sid"] = d["Subject ID"].str.extract(r"subject:\s*(\d+)\s+trial")[0]
    info["sid"] = info["Subject ID"].astype(str)
    checks.internal(f"{dfile}: every participant of the data file is in the subject info, once", set(d.sid) <= set(info.sid)
                    and not info.sid.duplicated().any())
    info = info.set_index("sid")
    ex_flag = {}
    for sid, ir in info.iterrows():
        fin = bool(ir["Experiment finished"])
        first = bool(ir["First attempt"])
        ex_flag[sid] = (not fin) or (exp == "Experiment 1" and not first)
        sub_info[(exp, dfile, sid)] = dict(finished=fin, first=first, time=ir["Experiment time"], age=ir["Age"], gender=int(ir["Gender"]))
        if exp == "Experiment 4":
            explicit[sid] = (ir["explicitTest_ans"], ir["Navigation codes"], int(ir["Experimental group"]))
    for sid, g in d.groupby("sid", sort=False):
        ir = info.loc[sid]
        grp = "" if exp == "Experiment 3" else GROUP[int(g["Experimental group"].iloc[0])]
        # key -> cell offset of this participant, from the presses that did not hit a wall
        offs = {}
        for _, r in g.iterrows():
            keys, cells, hit = json.loads(r["Key presses"]), json.loads(r["State locations"]), json.loads(r["Hit the wall"])
            for k, a, b, w in zip(keys, cells[:-1], cells[1:], hit):
                if w == 0:
                    offs.setdefault(k, []).append(b - a)
        modal = {k: max(set(v), key=v.count) for k, v in offs.items()}
        if exp != "Experiment 4":
            ok = all(len(set(v)) == 1 for v in offs.values())
            bad["mapping not constant"] = bad.get("mapping not constant", 0) + (not ok)
        km = ";".join(f"{KEY[k]}={OFF[modal[k]][0]},{OFF[modal[k]][1]}" for k in sorted(modal))
        for _, r in g.iterrows():
            keys, cells, iki, hit = (json.loads(r[c]) for c in ("Key presses", "State locations", "Inter key time", "Hit the wall"))
            n = int(r["Number of moves"])
            assert len(keys) == n == len(iki) == len(hit) and len(cells) == n + 1 and cells[0] == r["Start location"], (dfile, sid)
            assert set(keys) <= set(KEY) and all(1 <= c <= 81 for c in cells)
            bad["wall press moved"] = bad.get("wall press moved", 0) + sum(1 for a, b, w in zip(cells[:-1], cells[1:], hit) if w and a != b)
            bad["feedback 0 but reached"] = bad.get("feedback 0 but reached", 0) + (r["Feedback type"] == 0 and cells[-1] == r["Target location"])
            bad["feedback >0 but not reached"] = bad.get("feedback >0 but not reached", 0) + (r["Feedback type"] > 0 and cells[-1] != r["Target location"])
            new = bool(r["Old or New S-T"])
            t = int(r["Trial number"])
            if exp == "Experiment 3":
                block = ("generalization 1" if t <= 100 else "generalization 2") if new else ("multiple training" if t <= 80 else "single training")
            else:
                block = "generalization" if new else "training"
            phase = "transfer" if new else "learning"
            if exp == "Experiment 4":
                for k, a, b, w in zip(keys, cells[:-1], cells[1:], hit):
                    if w == 0:
                        dev[phase][1] += 1
                        dev[phase][0] += (b - a) != modal[k]
            rows.append(dict(
                dataset=ID, study=ID, experiment=exp, group=grp, subject=f"{exp}|{dfile}|{sid}", person=f"{exp}|{dfile}|{sid}",
                subject_excluded_by_authors=ex_flag[sid], age=float(ir["Age"]), sex=SEX[int(ir["Gender"])], handedness="",
                population="healthy", setting="lab", device=DEVICE, session=1, block=block, trial=t, phase=phase, time_s="",
                task="sequence learning", unit="trial", condition="new start-target pair" if new else "trained start-target pair",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                grid_size=9, start_cell=int(r["Start location"]), target_cell=int(r["Target location"]), key_moves=km,
                world_code=ir["World codes"], navigation_code=r["Navigation codes"] if "Navigation codes" in r.index else ir["Navigation codes"],
                n_moves=n, keys_pressed=";".join(KEY[k] for k in keys), cells_visited=";".join(map(str, cells)),
                inter_key_ms=";".join(map(str, iki)), wall_hits=";".join(map(str, hit)), trial_time_ms=int(r["Trial time"]),
                feedback_type=int(r["Feedback type"]), no_moves_flag=bool(r["Trial with no moves"]),
                feedback_shown=bool(r["Is there feedback"]), new_pair=new))
        sub_info[(exp, dfile, sid)].update(group=grp, n_trials=len(g), excluded=ex_flag[sid],
                                           person=f"{exp}|{dfile}|{sid}", km=km)
t = pd.DataFrame(rows)
del rows
checks.internal("every participant's moves follow one key-to-move mapping (Experiments 1-3, presses that did not hit a wall)",
                bad.get("mapping not constant", 0) == 0)
checks.internal("a press that hit a wall never moved the cursor", bad.get("wall press moved", 0) == 0)
checks.internal("feedback type 0 exactly when the cursor did not end on the target", bad.get("feedback 0 but reached", 0) == 0
                and bad.get("feedback >0 but not reached", 0) == 0)
print("Experiment 4 stochastic deviations (presses that did not hit a wall): training %d/%d = %.3f, generalization %d/%d = %.3f" % (
    dev["learning"][0], dev["learning"][1], dev["learning"][0] / max(dev["learning"][1], 1),
    dev["transfer"][0], dev["transfer"][1], dev["transfer"][0] / max(dev["transfer"][1], 1)))
# Experiment 4: 20% of presses in training move the cursor elsewhere (paper, Methods)
is4 = t.experiment == "Experiment 4"
t[["pert_type", "pert_value", "pert_unit"]] = t[["pert_type", "pert_value", "pert_unit"]].astype(object)
rate_train = dev["learning"][0] / dev["learning"][1]
rate_gen = dev["transfer"][0] / dev["transfer"][1]
checks.internal("Experiment 4: about a fifth of the training presses that did not hit a wall moved the cursor away from the key's "
                "usual move (stochastic mapping, probability 0.2; the share is a little above 0.2 because intended wall hits are "
                "left out), and none did in the generalization trials", 0.15 < rate_train < 0.30 and rate_gen == 0,
                f"training {rate_train:.3f}, generalization {rate_gen:.3f}")
if rate_gen > 0.1:
    t.loc[is4, ["pert_type", "pert_value", "pert_unit"]] = ["stochastic_mapping", 0.2, "probability"]
else:
    t.loc[is4 & (t.phase == "learning"), ["pert_type", "pert_value", "pert_unit"]] = ["stochastic_mapping", 0.2, "probability"]

# ---------------------------------------------------------------- the analysed participants and the paper's numbers
S = pd.DataFrame([dict(exp=k[0], file=k[1], sid=k[2], **v) for k, v in sub_info.items() if "n_trials" in v])
A = S[~S.excluded]
cnt = A.groupby(["exp", "group"]).size()
src_ = "Methods, Participants and experimental design"
for e, g, rep in (("Experiment 1", "Single", 16), ("Experiment 1", "Multiple", 16), ("Experiment 2", "Single", 16),
                  ("Experiment 2", "Multiple", 16), ("Experiment 4", "Single", 16), ("Experiment 4", "Multiple", 16)):
    checks.published(f"{e}, {g} group: participants", cnt.get((e, g), 0), rep, 0, src_)
checks.published("Experiment 3: participants", (A.exp == "Experiment 3").sum(), 16, 0, src_)
checks.published("all experiments: participants", len(A), 112, 0, "Methods, Participants")
gc = A.gender.value_counts()
for code, name, rep in ((1, "male", 49), (2, "female", 58), (3, "non-binary", 4), (4, "undisclosed", 1)):
    checks.published(f"all experiments: {name} participants", gc.get(code, 0), rep, 0, "Methods, Participants")
AGE_M, AGE_SD = A.age.mean(), A.age.std(ddof=1)
checks.published("all experiments: mean age (years; the paper's values look truncated, so one unit of the last digit is allowed)",
                 AGE_M, 19.9, 0.1, "Methods, Participants")
checks.published("all experiments: SD of age (years; same allowance)", AGE_SD, 1.4, 0.1, "Methods, Participants")
ntr = A.groupby("exp").n_trials.agg(["min", "max"])
for e, rep in (("Experiment 1", 300), ("Experiment 2", 280), ("Experiment 3", 1120), ("Experiment 4", 280)):
    checks.published(f"{e}: trials per analysed participant (minimum and maximum equal)", ntr.loc[e, "max"] if ntr.loc[e, "min"] == ntr.loc[e, "max"] else -1,
                     rep, 0, "Methods, Experimental design (Exp 3: 80 + 20 + 1000 + 20)")
b3 = t[(t.experiment == "Experiment 3") & ~t.subject_excluded_by_authors].groupby("subject").block.value_counts().unstack()
same_b3 = bool((b3.nunique() == 1).all())
cnt3 = {k: int(b3[k].iloc[0]) for k in b3.columns}
sp3 = t[(t.experiment == "Experiment 3") & ~t.subject_excluded_by_authors & (t.block == "generalization 2")]
n_single_pair = int(((sp3.start_cell == 52) & (sp3.target_cell == 47)).groupby(sp3.subject).sum().iloc[0])
checks.internal("Experiment 3: all participants have the same number of trials in each block (80 multiple-pair training, then "
                "generalization 1 of 20 trials)", same_b3 and cnt3["multiple training"] == 80 and cnt3["generalization 1"] == 20,
                f"blocks {cnt3}; generalization-2 trials with the Single training pair: {n_single_pair}")
pairs = T0 = t[~t.subject_excluded_by_authors & ~t.new_pair].groupby(["experiment", "group", "subject"]).apply(
    lambda x: len(set(zip(x.start_cell, x.target_cell))), include_groups=False)
newc = t[~t.subject_excluded_by_authors & t.new_pair & t.experiment.isin(["Experiment 1", "Experiment 2"])].groupby("subject").size()
checks.internal("Experiments 1, 2 and 4: the Single group trained on 1 start-target pair and the Multiple group on 4; Experiments 1 "
                "and 2 have 20 new-pair trials per participant",
                (pairs.xs("Single", level="group") == 1).all() and (pairs.xs("Multiple", level="group") == 4).all()
                and (newc == 20).all(), f"pairs {sorted(set(pairs.values))}, new-pair trials {sorted(set(newc.values))}")
# explicit knowledge test of Experiment 4 (all three keys named correctly)
ok_exp = {}
for sid, (ans, code, g) in explicit.items():
    answers = [int(x) for x in json.loads(ans.replace("'", '"'))]
    codes = json.loads(code)
    inf = sub_info[("Experiment 4", "E4_subjects_data_stochastic.csv", sid)]
    ok_exp[sid] = (answers == codes, GROUP[g])
for g, rep in (("Single", 11), ("Multiple", 10)):
    checks.published(f"Experiment 4, {g} group: participants who named all three keys' moves correctly", sum(v[0] for v in ok_exp.values() if v[1] == g), rep, 0,
                     "Results, Experiment 4 (explicit test of the mapping)")

# optimal arrivals (feedback type 2): Welch t of Single vs Multiple
def prop_opt(df):
    return df.groupby("subject").feedback_type.apply(lambda s: float((s == 2).mean()))


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    r = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(r.statistic), float(r.pvalue), float(df)


T = t[~t.subject_excluded_by_authors]
WELCH = {  # experiment -> (last training bin: t, p, df), (first generalization bin: t, df)
    "Experiment 1": ((-0.5, 0.61, 29.06), (-5.1, 29.99)), "Experiment 2": ((-0.19, 0.84, 29.99), (-4.81, 22.68))}
for e, (last_bin, gen_bin) in WELCH.items():
    te = T[T.experiment == e]
    grp = te.drop_duplicates("subject").set_index("subject").group
    for name, dd, rep in (("last training bin (trials 251-260)", te[te.trial.between(251, 260)], last_bin),
                          ("first generalization bin (first 10 generalization trials)", te[te.new_pair].groupby("subject").head(10), gen_bin)):
        p = prop_opt(dd)
        tt, pp, df = welch(p[grp.reindex(p.index) == "Single"], p[grp.reindex(p.index) == "Multiple"])
        R = f"Results, {e} (proportion of optimal arrivals, Single vs Multiple, Welch t-test; the paper's decimals look truncated)"
        unit = 0.01 if abs(rep[0]) < 1 or len(str(rep[0])) > 4 else 0.1
        checks.published(f"{e}, {name}: Welch t", tt, rep[0], unit, R)
        checks.published(f"{e}, {name}: Welch degrees of freedom", df, rep[-1], 0.01, R)
        if len(rep) == 3:
            checks.published(f"{e}, {name}: p value", pp, rep[1], 0.01, R)

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
partrows = [dict(person=v["person"], experiment=k[0], experiment_finished=v["finished"], first_attempt=v["first"],
                 experiment_time_ms=v["time"], key_moves=v["km"]) for k, v in sub_info.items() if "n_trials" in v]
pcols = [("experiment", "text", "", "Experiment of the participant."),
         ("experiment_finished", "true/false", "", "The authors' 'experiment finished' flag in the subject info."),
         ("first_attempt", "true/false", "", "The authors' 'first attempt' flag in the subject info."),
         ("experiment_time_ms", "number", "ms", "Total time of the experiment as the authors recorded it (0 when not finished).")]
partrows = [{k: v for k, v in p.items() if k != "key_moves"} for p in partrows]
meta = dict(
    study=ID,
    paper={k: entry["papers"][0][k] for k in ("authors", "year", "title", "venue", "doi")},
    lab=entry["lab"],
    data_url="https://osf.io/trjsg/",
    license="none stated",
    license_url="",
    changes=("Converted the per-trial CSV files of the four experiments into one table, decoded the key codes 74, 75, 76 to "
             "J, K, L, wrote the lists of each trial as semicolon-separated text, added each participant's age and gender from the "
             "subject-info files, marked the participants the paper's counts imply as excluded, and replaced participant "
             "identifiers (session start times) with codes."),
    how=[
        "One row is one trial of the grid task: a ship (cursor) started in start_cell on a 9 x 9 grid and the participant pressed "
        "J, K or L to move it towards the anchor (target_cell) within 10 s; cells are numbered 1-81 row by row from the top left. "
        "Each key moves the cursor one cell in a fixed direction (for example K one step right, J down-left, L up-left), the "
        "mapping differs between participants in Experiments 2-4 and is listed in key_moves, read off the cursor's moves; "
        "in Experiment 4 each press instead moved the cursor in a random other direction with probability 0.2 during training.",
        "keys_pressed, cells_visited (start cell plus the cell after every press), inter_key_ms (time from trial start to the first "
        "press, then between presses) and wall_hits are the authors' per-press lists. feedback_type is their outcome code "
        "(0 target not reached, 1 reached with extra presses, 2 reached in the fewest presses).",
        "group is Single (one start-target pair, practised in all training trials) or Multiple (four pairs); Experiment 3 has "
        "one group (Multiple training, then Single training). Generalization trials are those with new_pair set (new start-target "
        "pairs, phase transfer); Experiment 3 has two generalization blocks (trials 81-100 and the new-pair trials after trial 100), "
        "and its second block holds 30 trials flagged as new, 10 of which use the Single training pair, so the paper's 1000 Single-pair "
        "trials are 990 here plus those 10.",
        "Participants marked excluded by the authors are those their subject-info files mark as not finished, plus one "
        "participant of Experiment 1 who was not on their first attempt; with them excluded, the group sizes are those of the paper. "
        "Their trials are kept.",
    ],
    unit_note="trial of the grid navigation task",
    differences=[
        "Mean age of the 112 analysed participants is {:.3f} years (SD {:.3f}) here, printed as 19.9 (1.4) in the paper; the paper's "
        "decimals look truncated rather than rounded (for example t = -0.199 is printed -0.19 for Experiment 2), so the checks allow "
        "one unit of the last digit.".format(AGE_M, AGE_SD),
        "The paper's text gives no exclusion criteria. Participants marked as excluded here are those with 'experiment finished' = 0 "
        "in the subject info, plus one Experiment 1 participant whose 'first attempt' = 0; this reproduces the paper's group sizes "
        "(16 and 16 in Experiments 1, 2 and 4; 16 in Experiment 3) and every statistic checked.",
    ],
)
stdlib.release(ID, "navigation", t, meta, checks, participants=partrows, participant_columns=pcols)
