"""Varghese, Gordon, Sainburg, Winstein & Schweighofer (2023), PNAS: adaptive control is reversed between hands after left
hemisphere stroke and lost following right hemisphere stroke. 20 young adults and 30 chronic stroke survivors (15 left, 15 right
hemisphere) made bimanual reaches straight ahead with both hands on air sleds, either with one cursor at the average of the two
hands (BmC) or with a cursor for each hand (BmS). One row per reach of one hand, 90-column reaching format.

Run from the site folder: python3 -I tools/std/convert/Varghese2023_PNAS.py
Raw files (untrusted data, read only) in raw_extra/Varghese2023_PNAS/, from the OSF GitHub add-on (github.com/rinivarg/AdaptCtrlStroke):
  YNC_allData_120321_full.csv   the main table: one row per trial and "Hand" (left, right, cursor), 32 columns
  analysis_rev.ipynb            the authors' R notebook (read as JSON text, never run): the printed list of the 55 outlier trials
README.md, forTable1_v2.csv and fmchron_subj.csv (group demographics, not linked to the trial table) are not converted.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Varghese2023_PNAS"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]

d = pd.read_csv(RAW / "YNC_allData_120321_full.csv")
# the notebook prints the trials it removed (2.75% of the 2,000 one-cursor trials)
nb = json.load(open(RAW / "analysis_rev.ipynb", encoding="utf8"))
outs = []
for o in nb["cells"][15]["outputs"]:
    if "text" in o:
        outs += re.findall(r'"([A-Z]+_Subj\d+_BmC\d+_\d+)"', "".join(o["text"]))
outs = set(outs)

# only the one-cursor (BmC) and two-cursor (BmS) conditions are described in the paper
KEEP = ["BmS05", "BmS15", "BmC05", "BmC15"]
dk = d[d.Cond.isin(KEEP)].copy()
dropped_conds = sorted(set(d.Cond) - set(KEEP))
GROUPS = {"YNC": "young adults", "LHP": "left hemisphere stroke (right hand paretic)", "RHP": "right hemisphere stroke (left hand paretic)"}


def num(x):
    if x is None or x != x:
        return ""
    return "%.8g" % (0.0 if x == 0 else float(x))


rows, extra = [], []
cols_extra = ["row", "group", "block", "trial_orig", "hand_used"] + [c for c in d.columns if c not in ("Grp", "Subj", "SubNo", "Cond", "CondG", "Dist", "Trial", "Hand")]
n_excl = 0
for (g, s), ds in dk.groupby(["Grp", "Subj"], sort=False):
    pid = f"{ID}_{g}_{s}"
    wide = {}
    for (cond, hand), x in ds.groupby(["Cond", "Hand"]):
        wide[(cond, hand)] = x.set_index("Trial")
    for hand in ("left", "right"):
        sid = f"{pid}_{hand}"
        k = 0
        for cond in KEEP:
            if (cond, hand) not in wide:
                continue
            h = wide[(cond, hand)]
            cur = wide[(cond, "cursor")]
            dist = 5 if cond.endswith("05") else 15
            one_cursor = cond.startswith("BmC")
            for t in sorted(h.index):
                r = h.loc[t]
                c = cur.loc[t]
                ex = f"{g}_{s}_{cond}_{t}" in outs
                n_excl += ex and hand == "left"
                fde_cursor = c.FDE if one_cursor else r.FDE
                row = dict.fromkeys(names, "")
                row.update(
                    dataset=ID, experiment=f"{ID}_{'one_cursor' if one_cursor else 'two_cursors'}", study=ID, subject=sid, person=pid,
                    id_type="lab", person_link="assumed_unique", group=g, prior_exposure="unknown", subject_excluded_by_authors="False",
                    handedness="right", device="other", setting="lab", population="healthy" if g == "YNC" else "stroke",
                    trial=str(k), trial_orig=str(int(t)), session="0", block=cond, phase="other", trial_kind="reach",
                    break_before="True" if k == 0 else "False",
                    target_angle="90", target_distance_norm=num(dist / 15.0), target_visible_during_move="True", target_event="none",
                    target_jump="0", probe_kind="none", instr_goal="cursor_to_target", instr_strategy="unspecified",
                    instr_disclosure="none", instr_report="none", fb_cued="False", hand_used=hand,
                    hand_angle=num(-r.IDE), hand_angle_fb=num(-r.FDE), hand_measure="initial_direction",
                    mt_ms=num(r.MT * 1000.0) if r.MT == r.MT else "", excluded_by_authors=str(bool(ex)),
                    fb_visible="endpoint", fb_mode="other" if one_cursor else "veridical", fb_form="cursor", online_cursor="none",
                    fb_endpoint_transform_only="False", rotation="0", cursor_error=num(-fde_cursor), outcome_channels="cursor|points")
                rows.append(row)
                ex_row = {"row": len(rows), "group": g, "block": cond, "trial_orig": int(t), "hand_used": hand}
                for cc in cols_extra[5:]:
                    ex_row[cc] = r[cc]
                extra.append(ex_row)
                k += 1

ex_df = pd.DataFrame(extra, columns=cols_extra)
n_persons = len({r["person"] for r in rows})
byg = {g: len({r["person"] for r in rows if r["group"] == g}) for g in GROUPS}
bmc_trials = dk[(dk.Cond.isin(["BmC05", "BmC15"])) & (dk.Hand == "cursor")].shape[0]
checks.published("participants", n_persons, 50, 0, "Results, first paragraph (fifty adults)")
checks.published("young adults", byg["YNC"], 20, 0, "Results, first paragraph (20 neurotypical young adults)")
checks.published("left hemisphere stroke", byg["LHP"], 15, 0, "Results, first paragraph (15 left hemiparesis; the file's LHP group has 15 participants)")
checks.published("right hemisphere stroke", byg["RHP"], 15, 0, "Results, first paragraph (15 right hemiparesis; the file's RHP group has 15 participants)")
checks.published("one-cursor trials analysed for the within-trial gain (50 x 2 distances x 20)", bmc_trials, 2000, 0,
                 "Methods, Kinematic data processing (55 out of 2,000 trials excluded)")
checks.published("trials removed as outliers", n_excl, 55, 0, "Methods, Kinematic data processing (55 out of 2,000 trials, 2.75% were excluded)")
checks.published("percentage of one-cursor trials removed", 100.0 * n_excl / bmc_trials, 2.75, 0.005, "Methods, Kinematic data processing (2.75%)")
checks.internal("20 trials per participant, condition and hand", all(len(x) == 20 for x in dk.groupby(["Grp", "Subj", "Cond", "Hand"]).Trial.agg(list)))
checks.internal("every participant has all four documented conditions", all(x.Cond.nunique() == 4 for _, x in dk.groupby(["Grp", "Subj"])))
checks.internal("rows = 2 hands x trials of the documented conditions", len(rows) == 2 * dk[dk.Hand == "left"].shape[0], str(len(rows)))
checks.internal("all 55 listed outlier trials found", len({o for o in outs}) == 55 and all(
    ((dk.Grp == o.split('_')[0]) & (dk.Subj == o.split('_')[1]) & (dk.Cond == o.split('_')[2]) & (dk.Trial == int(o.split('_')[3]))).any() for o in outs))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Schweighofer & Winstein (USC)",
    data_url="https://osf.io/dk5sn/", license="none stated", license_url="",
    changes="converted to the OpenMotor format, one row per hand and reach, participant codes replaced, angles negated to counter-clockwise positive "
            "(sign inferred), the unpublished conditions of the file left out; the authors' full set of per-trial measures is kept in trial_measures.csv.",
    unit_note="reach of one hand",
    differences=[
        f"The file also holds conditions the paper does not describe ({', '.join(dropped_conds)}; 43 of the 50 participants did them). Their meaning is not documented in the deposit, so they are not converted.",
        "The within-trial gains (g) and trial-to-trial learning rates of the paper come from robust mixed-effects models in R and were not recomputed here.",
        "The order of the two target distances (5 and 15 cm) within a condition was counterbalanced across participants, but the file does not say which came first. trial counts the two-cursor condition first, then the one-cursor condition (as the paper states), and the 5 cm block before the 15 cm block; trial_orig is the authors' 1 to 20 within each block.",
        "Demographics (age, sex, handedness score, Fugl-Meyer score, chronicity) are in the deposit's forTable1_v2.csv and fmchron_subj.csv, which use other participant labels than the trial table and cannot be linked to it with certainty, so they are not included.",
        "The authors' outlier trials (55, listed in their notebook output, one-cursor condition only) are marked in excluded_by_authors on both hands' rows of that trial."],
    how=[
        "One row is one reach of one hand (left or right) in a bimanual task: the participant held the two hands on air sleds, 12 cm either side of the midline, and reached straight ahead to a target 5 or 15 cm away without seeing the cursor during the movement; the cursor was shown again when the movement ended. In the one-cursor condition (experiment one_cursor, block BmC) a single cursor was drawn at the average position of the two hands; in the two-cursor condition (two_cursors, block BmS) each hand controlled its own cursor and had its own target. 20 reaches per condition and distance. Each participant gives two subject sequences (one per hand) that share one person code. group is YNC (young adults), LHP (left hemisphere stroke, right hand paretic) or RHP (right hemisphere stroke, left hand paretic).",
        "hand_angle is the authors' IDE, the direction of the hand at peak acceleration relative to the target direction (degrees), and hand_angle_fb is their FDE, the direction at the end of the movement; the direction at peak velocity (PVDE) and the other kinematic measures are in trial_measures.csv. Target direction is straight ahead, so target_angle is 90. In the authors' file positive values mean rightward: the paper says the cursor moves rightward when its initial directional error is positive, and the lateral displacement of the hand at peak velocity rises with PVDE. They were negated here so that, as in the other studies, positive is counter-clockwise (leftward). The Methods call the system a standard right-hand coordinate system, so this sign is inferred, not stated in the deposit.",
        "cursor_error is the negated final direction error of the cursor seen at the end of the reach: for the one-cursor condition the file's cursor row (the average of the two hands), for the two-cursor condition the hand's own error. mt_ms is the authors' MT times 1000 (the deposit does not define MT; values of 0.4 s are of the size of the movement time). trial_measures.csv has one row per row of the main file (column row) with the authors' own columns and signs: CT, RT, PPT, TPV, MT, PV, EE, VE, IDE, PRDE, PVDE, FDE, FPE, IDIST, PVDIST, FDIST, CURV, OnAxErr, OffAxErr, XPosPV, YPosPV, XPosOff, YPosOff, XDistPV (not all are defined in the deposit; the paper defines IDE, PVDE and the peak velocity PV).",
        "Hand paths were sampled at 116 Hz with electromagnetic sensors; the paper filtered them at 10 Hz. device is 'other' (Kinereach air-sled setup), phase is 'other' because there is no learning schedule, and break_before is True at the first reach of each hand's sequence."],
)
stdlib.release(ID, "reach90", rows, meta, checks, extra_files=[("trial_measures.csv", ex_df.to_csv(index=False, lineterminator="\r\n").encode())])
