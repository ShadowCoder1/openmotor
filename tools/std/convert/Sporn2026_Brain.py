"""Sporn, Herrojo Ruiz, Fathana, Zich, Bestmann & Ward (2026), Brain: impaired reward-based learning but preserved motor
invigoration in chronic stroke. 40 chronic stroke survivors (SK) and 30 age-matched healthy controls (HC) did a probabilistic
reversal learning task, reaching to one of two targets whose reward probabilities changed in four blocks (78 trials per limb),
with the strong and the weak limb (KINARM Exoskeleton). One row per trial, module reward_reaching.

Run from the site folder: python3 -I tools/std/convert/Sporn2026_Brain.py
Raw file (untrusted data, read only) in raw_extra/Sporn2026_Brain/, from https://osf.io/gky2r/ (Data folder, CC BY 4.0):
  data4brms.csv   the authors' trial-level file for their movement-time models (columns group, subject, arm, block, trial, logMT,
                  muhat2). The OSF Code folder (README.md, run_allmodels_shifted_lognormal_MT.R, Figure3.R) was read as text,
                  never run: group 1 = SK, 0 = HC; arm 2 is recoded to 0 = the strong limb, so arm 1 is the weak limb;
                  MT = exp(logMT) in ms; prediction = muhat2.
"""
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Sporn2026_Brain"
RAW = stdlib.raw_dir(ID)
FN = "data4brms.csv"
OSF_MD5 = "1cddbe53f690b1ba65b6f3f26159c8ff"  # file checksum on OSF
checks = stdlib.Checks()
LIMB = {1: "weak", 2: "strong"}
GROUP = {1: "SK", 0: "HC"}
POP = {1: "chronic stroke", 0: "healthy"}


def tf(x):
    return "True" if x else "False"


src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("data4brms.csv matches the downloaded hash and the MD5 and size listed on OSF",
                stdlib.sha256(RAW / FN) == src[FN]["sha256"] and hashlib.md5(open(RAW / FN, "rb").read()).hexdigest() == OSF_MD5
                and (RAW / FN).stat().st_size == 491014)
d = pd.read_csv(RAW / FN)
checks.internal("10850 rows = 70 participants x 155 trials, each participant's rows contiguous and numbered 1-70",
                len(d) == 70 * 155 and d.subject.nunique() == 70 and d.subject.ne(d.subject.shift()).sum() == 70
                and sorted(d.subject.unique()) == list(range(1, 71)) and (d.groupby("subject").size() == 155).all())
checks.internal("group is constant within participant (40 with group 1 = SK, 30 with group 0 = HC)",
                (d.groupby("subject").group.nunique() == 1).all()
                and Counter(d.drop_duplicates("subject").group) == Counter({1: 40, 0: 30}))
ok_struct = True
n_short_weak = 0
for s_, g in d.groupby("subject", sort=False):
    a1, a2 = g[g.arm == 1], g[g.arm == 2]
    n1, n2 = len(a1), len(a2)
    n_short_weak += n1 == 77
    ok_struct &= (g.arm.tolist() == [1] * n1 + [2] * n2 and {n1, n2} == {77, 78}
                  and a1.trial.tolist() == list(range(1, n1 + 1)) and a2.trial.tolist() == list(range(1, n2 + 1))
                  and a1.block.tolist() == sorted(a1.block) and a2.block.tolist() == sorted(a2.block, reverse=True)
                  and all(18 <= v <= 21 for v in Counter(a1.block).values()) and all(18 <= v <= 21 for v in Counter(a2.block).values())
                  and set(a1.block) == set(a2.block) == {1, 2, 3, 4})
checks.internal("every participant: the file lists the weak limb (arm 1) first, with blocks 1 to 4, then the strong limb (arm 2), with "
                "blocks 4 to 1 (the reward-probability order was reversed); one limb has 78 trials and the other 77 (66 participants "
                f"78 weak and 77 strong, {n_short_weak} the reverse); blocks have 18 to 21 trials", ok_struct and n_short_weak == 4)
mt = np.exp(d.logMT)
checks.internal("movement time exp(logMT) lies between 200 ms and 8 s (213 to 7848 ms); 433 rows have no logMT, 81 no prediction",
                mt.min() > 200 and mt.max() < 8000 and d.logMT.isna().sum() == 433 and d.muhat2.isna().sum() == 81)
checks.internal("the prediction strength is 0 on trial 1 of each limb's sequence (140 rows), never negative, and has a value on every "
                "other row with a movement time except 78",
                (d[d.trial == 1].muhat2 == 0).all() and (d.muhat2.dropna() >= 0).all()
                and int((d.muhat2.isna() & d.logMT.notna()).sum()) == 78)

# ---------------------------------------------------------------- rows
names = [c[1] for c in stdlib.columns_for("reward_reaching")]
rows = []
for k, (s, g) in enumerate(d.groupby("subject", sort=False)):
    for j, r in enumerate(g.itertuples(index=False)):
        limb = LIMB[int(r.arm)]
        row = dict.fromkeys(names, "")
        has_mt = not math.isnan(r.logMT)
        has_pred = not math.isnan(r.muhat2)
        row.update(
            dataset=ID, study=ID, experiment=f"{ID}_Main", group=GROUP[int(r.group)], subject=f"{ID}_{int(s)}", person=f"{ID}_{int(s)}",
            subject_excluded_by_authors="False", population=POP[int(r.group)], setting="lab", device="KINARM Exoskeleton",
            session=1, block=f"block{int(r.block)}", trial=j + 1, phase="learning", task="reaching", unit="trial",
            condition=f"{limb} limb", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=tf(not (has_mt and has_pred)),
            limb=limb, contingency_block=int(r.block), trial_in_limb=int(r.trial),
            movement_time_ms=math.exp(r.logMT) if has_mt else "", log_movement_time=r.logMT if has_mt else "",
            prediction_strength=r.muhat2 if has_pred else "",
        )
        rows.append(row)
checks.internal("every row the authors shared is kept, in order (10850)",
                len(rows) == len(d) and [r["trial_in_limb"] for r in rows] == d.trial.tolist()
                and [r["contingency_block"] for r in rows] == d.block.tolist())
checks.internal("rows without a movement time or a prediction (the rows the authors' models drop) are flagged excluded_by_authors",
                sum(r["excluded_by_authors"] == "True" for r in rows) == int((d.logMT.isna() | d.muhat2.isna()).sum()))

# ---------------------------------------------------------------- published numbers, from the converted rows
persons = {}
for r in rows:
    persons[r["person"]] = r["group"]
reach = pd.DataFrame(rows)
checks.published("participants", len(persons), 70, 0, "Methods, Experimental Subjects (40 stroke survivors and 30 controls)")
checks.published("chronic stroke survivors (SK)", sum(g == "SK" for g in persons.values()), 40, 0, "Methods, Experimental Subjects")
checks.published("age-matched healthy controls (HC)", sum(g == "HC" for g in persons.values()), 30, 0, "Methods, Experimental Subjects")
checks.published("trials per limb", reach.groupby(["person", "limb"]).size().max(), 78, 0,
                 "Methods, task ('limiting the total number of trials to 78')")
checks.published("blocks of reward probabilities per limb", reach.contingency_block.nunique(), 4, 0, "Methods, task (four blocks)")
bl = reach.groupby(["person", "limb", "contingency_block"]).size()
checks.published("shortest block (reversals 'every 18-21 trials')", bl.min(), 18, 0, "Methods, task")
checks.published("longest block (reversals 'every 18-21 trials')", bl.max(), 21, 0, "Methods, task")

# the 2 x 2 (group x limb) analysis of log movement time: reported for comparison
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

m = d.dropna(subset=["logMT"]).groupby(["subject", "group", "arm"]).logMT.mean().reset_index()
m["g"] = m.group.map(GROUP)
m["limb"] = m.arm.map(LIMB)
an = sm.stats.anova_lm(smf.ols("logMT ~ C(g, Sum) * C(limb, Sum)", data=m).fit(), typ=3)
F_g, F_i, p_l = an.loc["C(g, Sum)", "F"], an.loc["C(g, Sum):C(limb, Sum)", "F"], an.loc["C(limb, Sum)", "PR(>F)"]
w = m[m.g == "SK"].pivot(index="subject", columns="limb", values="logMT")
checks.internal("stroke survivors are slower than controls and, in stroke survivors, slower with the weak limb (direction of the paper's "
                "group effect and group x limb interaction)",
                m[m.g == "SK"].logMT.mean() > m[m.g == "HC"].logMT.mean() and (w["weak"] - w["strong"]).mean() > 0
                and (m[m.g == "HC"].pivot(index="subject", columns="limb", values="logMT").pipe(lambda x: (x["weak"] - x["strong"]).mean())) < (w["weak"] - w["strong"]).mean())

# ---------------------------------------------------------------- release
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Ward / Bestmann (UCL Queen Square)",
    data_url="https://osf.io/gky2r/",
    license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="converted to the OpenMotor format, participant codes replaced, movement time back-transformed with exp() as in the "
            "authors' script, and the authors' arm code written as limb (arm 1 weak, arm 2 strong).",
    unit_note="trial of one participant sequence (one reach)",
    how=[
        "One row is one trial of a probabilistic reversal learning task in which people reach to one of two targets whose reward "
        "probabilities change over four blocks of 18 to 21 trials (78 trials per limb), with the strong limb (non-paretic in "
        "stroke survivors, dominant in controls) and the weak limb (paretic or non-dominant); group is SK (40 chronic stroke "
        "survivors) or HC (30 age-matched controls).",
        "The deposit (data4brms.csv, the authors' file for their movement-time models) holds only the movement time, as log ms "
        "(log_movement_time) and back-transformed with exp() as in the authors' script (movement_time_ms), and the trial-wise "
        "prediction strength of the authors' hierarchical Gaussian filter model of each person's choices (prediction_strength, a "
        "model output, 0 on the first trial of a limb's sequence); choices, outcomes, reaction times and reach directions are "
        "not in the deposit, and neither are age or sex.",
        "Rows keep the authors' order: the weak limb's trials (blocks 1 to 4), then the strong limb's (blocks 4 to 1, because the "
        "order of reward probabilities was reversed for the other limb); the order of testing was counterbalanced across "
        "participants and is not recorded. contingency_block names the reward probabilities (1: 0.9/0.1, 2: 0.3/0.7, "
        "3: 0.7/0.3, 4: 0.1/0.9 for targets 1/2) and trial_in_limb is the authors' trial number.",
        "Rows without a movement time (433) or without a prediction (81) are kept and flagged excluded_by_authors, because the "
        "authors' models drop them; the strong limb's sequences have 77 trials, one fewer than the 78 of the Methods.",
    ],
    differences=[
        f"The paper's 2 x 2 analysis of log-transformed movement time (group x limb, F(1, 136)) reports a group effect of F = 69.84, "
        f"an interaction of F = 5.91 and p = 0.090 for the limb effect; a Type III ANOVA of each participant's mean logMT per limb "
        f"(140 values) gives F = {F_g:.2f}, F = {F_i:.2f} and p = {p_l:.3f}. The direction reproduces (stroke survivors slower, "
        "most with the weak limb) but the authors' participant-level summary is not stated, so the F values are not recovered.",
        "The Methods give 78 trials per limb; in the shared file one limb has 78 trials and the other 77 for every participant (the weak limb has 78 for 66 participants and the strong limb for the other 4), so one trial per participant is absent.",
        "The shared file has no choices or outcomes, so the paper's accuracy, win-stay and model-parameter results cannot be checked.",
    ],
)
stdlib.release(ID, "reward_reaching", rows, meta, checks)
