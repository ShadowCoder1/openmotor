"""Zhang, Wang, Zhang, Nie & Wei (2024), eLife 13:RP94608: perceptual error based on Bayesian cue combination drives implicit
motor adaptation. Experiment 2 of the paper (84 participants in 7 groups of 12, one error-clamp size per group: 2, 4, 8, 16, 32,
64 or 95 deg; 120 cycles of 4 reaches) in the 90-column reaching format; one row per CYCLE (the deposit shares the mean hand angle
of each 4-reach cycle, not single reaches). The single-trial-learning table of Experiment 4 and the proprioception table of
Experiment 3 are shipped unchanged as supplementary CSV files.

Run from the site folder: python3 -I tools/std/convert/Zhang2024_eLife.py
Raw files (untrusted data, read only) in raw_extra/Zhang2024_eLife/, from https://doi.org/10.6084/m9.figshare.24503926 (CC BY 4.0):
  Exp2.csv, Exp3.csv, Exp4data.csv, README.txt  (own experiments);
  Fig1B.csv (group means of Kim 2018 and Morehead 2017), Tsay_report.csv and Tsay_STL.csv (re-analysed data of Tsay et al.,
  which are separate studies listed on OpenMotor) are NOT converted. Exp1.csv is named in the README but is not in the deposit.
"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zhang2024_eLife"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]

d = pd.read_csv(RAW / "Exp2.csv")
e3 = pd.read_csv(RAW / "Exp3.csv")
e4 = pd.read_csv(RAW / "Exp4data.csv")

rows = []
for (sub), g in d.groupby("sub", sort=True):
    g = g.sort_values("cycleNum")
    assert (g.cycleNum.values == np.arange(1, 121)).all()
    size = float(g.perturbSize.iloc[0])
    for r in g.itertuples(index=False):
        c = int(r.cycleNum)
        tt = int(r.trialType)
        fb_veridical = c <= 15
        if tt == 0:
            block, phase = ("baseline_veridical", "baseline") if fb_veridical else ("baseline_no_feedback", "baseline")
        elif tt == 1:
            block, phase = "clamp", "learning"
        else:  # trialType 2 in the file (the README says 3)
            block, phase = "no_feedback_aftereffect", "aftereffect"
        row = dict.fromkeys(names, "")
        row.update(
            dataset=ID, experiment="Exp2", study=ID, subject=f"Exp2_s{int(sub)}", person=f"Exp2_s{int(sub)}",
            id_type="lab", person_link="assumed_unique", group=f"clamp_{int(size)}deg", prior_exposure="unknown",
            subject_excluded_by_authors="False", handedness="right", device="tablet", setting="lab", population="healthy",
            trial=str(c - 1), trial_orig=str(c), session="0", block=block, phase=phase, trial_kind="reach",
            break_before="True" if c == 1 else "False",
            instr_goal="hand_to_target", instr_strategy="unspecified",
            hand_used="right", hand_angle="%.8g" % r.handangle, hand_measure="fixed_distance",
            excluded_by_authors="False",
            fb_visible="unknown" if tt != 2 and (fb_veridical or tt == 1) else "none",
            fb_mode="veridical" if (tt == 0 and fb_veridical) else "clamped" if tt == 1 else "none",
            fb_form="cursor" if (tt == 1 or fb_veridical) else "none",
        )
        if tt == 1:
            row.update(clamp="%.8g" % (-size), cursor_error="%.8g" % (-size), online_cursor="unknown")
        rows.append(row)

# ---------------------------------------------------------------- published checks (paper, Methods and Results of Exp 2)
cnt = d.groupby("group")["sub"].nunique()
checks.published("Experiment 2 participants", d["sub"].nunique(), 84, 0, "Methods, Experiment 2 ('Eighty-four participants')")
checks.published("participants per clamp-size group", float(cnt.min()), 12, 0, "Methods, Experiment 2 (seven groups of 12)")
checks.published("participants per clamp-size group (largest)", float(cnt.max()), 12, 0, "Methods, Experiment 2 (seven groups of 12)")
checks.published("number of clamp sizes", d.perturbSize.nunique(), 7, 0, "Methods, Experiment 2 (2, 4, 8, 16, 32, 64, 95 deg)")
checks.published("sum of the seven clamp sizes (2+4+8+16+32+64+95)", float(sum(d.perturbSize.unique())), 221, 0,
                 "Methods, Experiment 2 (clamp sizes 2, 4, 8, 16, 32, 64, 95 deg)")
one = d[d["sub"] == 1]
checks.published("cycles per participant (480 reaches / 4)", float(len(one)), 120, 0, "Methods, Experiment 2 (120 cycles)")
checks.published("baseline cycles per participant (15 veridical + 15 without feedback)", float((one.trialType == 0).sum()), 30, 0,
                 "Methods, Experiment 2 (15 + 15 baseline cycles)")
checks.published("clamp cycles per participant", float((one.trialType == 1).sum()), 80, 0, "Methods, Experiment 2 (80 cycles of perturbation)")
checks.published("no-feedback aftereffect cycles per participant", float((one.trialType == 2).sum()), 10, 0,
                 "Methods, Experiment 2 (10 cycles without feedback)")
late = d[d.cycleNum.between(101, 110)].groupby(["group", "sub"]).handangle.mean().groupby("group").mean()
size_by_group = d.groupby("group").perturbSize.first()
over = int((late > size_by_group).sum())
checks.published("groups whose adaptation extent exceeds the clamp size (2, 4, 8, 16 deg)", over, 4, 0,
                 "Results, Experiment 2 ('groups with 2-16 deg ... overcompensated', cycles 101-110)")
checks.published("groups of 32, 64 and 95 deg that overcompensate", float((late > size_by_group)[size_by_group >= 32].sum()), 0, 0,
                 "Results, Experiment 2 (only the 2-16 deg groups overcompensate)")
ext = d[d.cycleNum.between(101, 110)].groupby(["group", "sub"]).handangle.mean()
F, p = stats.f_oneway(*[ext[g].values for g in sorted(ext.index.get_level_values(0).unique())])
checks.published("participants in Experiment 4's table of single-trial learning", e4["sub"].nunique(), 19, 0,
                 "Methods, Experiment 4 ('Nineteen participants')")
checks.published("single-trial-learning values per participant in Experiment 4 (3 days x 15 blocks x 12 clamp trials)",
                 float(len(e4) / e4["sub"].nunique()), 540, 0, "Methods, Experiment 4 (540 trials overall, 12 clamp trials per block)")
checks.published("participants in Experiment 3's proprioception table", e3["sub"].nunique(), 11, 0, "Methods, Experiment 3 (11 participants)")
checks.published("test trials per participant and perturbation size in Experiment 3", float(e3.groupby(["sub", "perturbSize"]).size().max()), 3, 0,
                 "Methods, Experiment 3 (3-trial proprioception test block)")
checks.internal("every participant has the same group and clamp size on all 120 cycles; sub ids unique to a group",
                bool(d.groupby("sub").group.nunique().max() == 1))
checks.internal("rows kept: 84 x 120 = 10080", len(rows) == 10080 == len(d))
checks.internal("no missing hand angles", int(d.handangle.isna().sum()) == 0)
checks.internal("Experiment 4 table has 19 x 540 = 10260 rows and all 6 clamp sizes x 2 cursor conditions",
                len(e4) == 10260 and e4.perturb.nunique() == 6 and e4.blur.nunique() == 2)

buf3, buf4 = io.StringIO(), io.StringIO()
e3.to_csv(buf3, index=False, lineterminator="\r\n")
e4.to_csv(buf4, index=False, lineterminator="\r\n")
extra = [("supplementary/Exp3_proprioception_bias.csv", buf3.getvalue().encode()),
         ("supplementary/Exp4_single_trial_learning.csv", buf4.getvalue().encode())]

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Wei (Peking University)", data_url="https://doi.org/10.6084/m9.figshare.24503926",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Experiment 2 converted to the OpenMotor reaching format with participant codes replaced; the authors' "
            "Experiment 3 and 4 tables copied unchanged (columns as in their README) into supplementary/.",
    unit_note="cycle of 4 reaches of one participant (mean hand angle)",
    how=[
        "One row is one CYCLE of one participant, not one reach: the authors share the hand angle averaged over the four "
        "reaches (to targets at 45, 135, 225 and 315 deg) that make up a cycle. trial counts cycles from 0 and trial_orig is "
        "the authors' cycleNum. Each of the 84 participants has 120 cycles: 15 with veridical cursor feedback and 15 without "
        "feedback (baseline), 80 with an error-clamped cursor, and 10 without feedback (aftereffect). Each group of 12 had one "
        "clamp size (2, 4, 8, 16, 32, 64 or 95 deg). Because rows are cycle means, target_angle is empty and "
        "trial_kind reach means a cycle of reaches.",
        "hand_angle is the authors' handangle in deg, measured where the hand crossed half the target distance (Methods). The "
        "paper does not state the sign convention for Experiment 2. In the data the angle becomes positive during the clamp, so "
        "positive is the direction opposite to the cursor's offset (adaptation). The Experiment 1 figure caption defines positive "
        "perturbations as clockwise, so the clamp is taken to be clockwise (clamp and cursor_error = minus the size) and "
        "hand_angle is therefore counter-clockwise positive as in the other OpenMotor reaching studies. This is an inference, not stated for Experiment 2; if the clamp was in fact counter-clockwise, clamp and cursor_error would be positive and hand_angle negated.",
        "fb_visible, online_cursor and the cursor timing are unknown for Experiment 2 (the paper does not say whether feedback "
        "was endpoint or online there). Age, sex and handedness are reported only for all 115 participants of the paper "
        "(22 +/- 3 years, all right-handed) and are not given per participant.",
        "supplementary/Exp4_single_trial_learning.csv (19 participants, sub numbered as in the authors' file): sub, day, perturb "
        "(clamp angle in deg, +-4, +-16, +-64), blur (1 clear cursor, 2 blurred cursor), stl (single-trial learning in deg: the "
        "hand-direction change between the no-feedback reaches before and after a clamp trial). "
        "supplementary/Exp3_proprioception_bias.csv (11 participants): perturbSize (deg), trialNum (test trial 1 to 3 after a "
        "reaching block), propBias (proprioceptive bias in deg). Experiment 1, Fig1B and the re-analysed data of Tsay et al. in "
        "the deposit are not included (Tsay et al.'s studies are separate OpenMotor entries).",
    ],
    differences=["The deposit's README says trialType 3 marks the washout cycles; the file codes them 2 (cycles 111-120).",
                 f"The paper reports a one-way ANOVA of adaptation extent (mean of cycles 101-110) over the seven groups of F(6,83) = 12.108. "
                 f"From the 84 participants of this file the same analysis gives F(6,77) = {F:.3f} (p = {p:.1e}); 83 cannot be "
                 "the error degrees of freedom of seven groups of 12, so the reported degrees of freedom or the exact measure "
                 "(for example a baseline correction or a different window) differ from the file's."],
)
stdlib.release(ID, "reach90", rows, meta, checks, extra_files=extra)
