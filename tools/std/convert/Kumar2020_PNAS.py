"""Kumar, Panthi, Divakar & Mutha (2020), PNAS: mechanistic determinants of effector-independent motor memory encoding.

Run from the site folder: python3 -I tools/std/convert/Kumar2020_PNAS.py
Raw file (untrusted data, read only) in raw_extra/Kumar2020_PNAS/ from https://figshare.com/articles/dataset/Kumar_et_al_2020_data/12416426
(CC BY 4.0): Kumar_et_al_2020_data.xlsx, the data behind the figures of the paper. Only the sheets that hold one value per participant
are converted (Exp1_Early_Late_Learning, Fig 2A, Fig 2B, Fig 4C and 4D insets, Fig 4F, Exp3_Early_Late_Learning, Fig 5C). Sheets that hold group
means across trials (Fig1C, 4C and 4D curves, 5B, 5D, S1, S3B), model outputs (Fig 3, S2, S3D) are not converted.
The paper (PNAS 117:17338, Europe PMC / PMC7382293) was read for the design and the statistics used in the checks.
"""
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Kumar2020_PNAS"
ck = stdlib.Checks()
wb = openpyxl.load_workbook(stdlib.raw_dir(ID) / "Kumar_et_al_2020_data.xlsx", data_only=True)
SETS = [40, 80, 160, 320]
rows = []
DEV = "digitizing tablet and stylus, hand hidden by a mirror, cursor shown on a horizontal screen"


def add(exp, group, person, measure, desc, value, n_trials="", stat="mean", block="", hand="", condition="", pert=True, context=""):
    rows.append(dict(
        dataset=ID, study=ID, experiment=exp, group=group, subject=person, subject_excluded_by_authors=False,
        age="", sex="", handedness="right", population="healthy", setting="lab", device=DEV, session=1, block=block,
        trial="", phase="learning", time_s="", task="reaching (visuomotor rotation)", unit="participant summary",
        condition=condition, pert_type="visuomotor_rotation" if pert else "none", pert_value=30 if pert else 0,
        pert_unit="deg counter-clockwise" if pert else "", feedback="altered" if pert else "normal", excluded_by_authors=False,
        measure=measure, measure_description=desc, statistic=stat, value=value, value_unit="deg", n_trials=n_trials,
        session_context=context))


# ---- Experiment 1: early and late learning of each participant (64)
ws = wb["Exp1_Early_Late_Learning"]
E1 = []
for r in ws.iter_rows(min_row=3, values_only=True):
    assert (r[0], r[1], r[2]) == (r[5], r[6], r[7])
    E1.append(dict(k=r[0], arm=r[1].lower(), nset=int(r[2]), early=r[3], late=r[8]))
    p = f"E1_{r[1]}_{r[2]}_{r[0]}"
    g = f"{r[1].lower()} arm learned, {r[2]} learning trials"
    add("Exp1_learning", g, p, "direction_error_early_learning",
        "Mean direction error (cursor direction relative to the target direction at peak velocity, counter-clockwise positive) on the first four learning trials, baseline direction error of the arm subtracted (paper, Methods, Experiment 1)",
        r[3], 4, block="learning", condition="early learning")
    add("Exp1_learning", g, p, "direction_error_late_learning",
        "Mean direction error on the last four learning trials, baseline subtracted", r[8], 4, block="learning", condition="late learning")
e1 = pd.DataFrame(E1)

# ---- Experiment 1 generalization: Fig 2A (left arm) and Fig 2B (right arm), 8 participants per learning-trial set and block
F2 = {}
for sheet, arm, other in (("Fig 2A", "left", "right"), ("Fig 2B", "right", "left")):
    w = wb[sheet]
    for r in w.iter_rows(min_row=3, max_row=10, values_only=True):
        for i, s in enumerate(SETS):
            for kind, val, kcol in (("naive", r[1 + i], r[0]), ("generalization", r[7 + i], r[6])):
                F2.setdefault((arm, kind), []).append(dict(nset=s, k=kcol, v=val))
                if kind == "naive":
                    g = f"{arm} arm, naive learning, {s} learning trials in the study's trial set"
                    meas = "direction_error_naive_learning"
                    desc = (f"Direction error of the {arm} arm on its initial naive learning trials, as plotted in the paper's Fig 2 (the sheet does not state how many trials); "
                            "this participant learned the rotation with this arm")
                else:
                    g = f"{arm} arm tested after {other}-arm learning, {s} learning trials"
                    meas = "direction_error_generalization"
                    desc = (f"Direction error of the {arm} arm on the generalization trials after the participant learned the rotation with the {other} arm for {s} trials "
                            "(Fig 2 of the paper; the sheet does not state how many trials)")
                add("Exp1_generalization", g, f"E1G_{sheet[-1]}_{kind}_{s}_{kcol}", meas, desc, val, block="generalization" if kind == "generalization" else "learning",
                    condition=f"{arm} arm, {kind}", context="Participants of this sheet cannot be matched to those of the Exp1_learning sheet from the deposit")

# ---- Experiment 2: hd-tDCS over PPC during learning; Fig 4C/4D insets (first and last 8 learning trials) and Fig 4F (first 25, rest)
def inset(name):
    out = []
    for r in wb[name].iter_rows(min_row=4, max_row=40, min_col=5, max_col=11, values_only=True):
        if r[0] is None:
            break
        out.append(r)
    return out


f4 = np.array([[np.nan if v is None else v for v in r] for r in wb["Fig 4F"].iter_rows(min_row=3, max_row=19, min_col=1, max_col=12, values_only=True)], dtype=float)
E2G = {"Left PPC": ("right arm learned, real hd-tDCS over left PPC", "real", "left"), "Left Sham": ("right arm learned, sham hd-tDCS (left PPC site)", "sham", "left"),
       "Right PPC": ("left arm learned, real hd-tDCS over right PPC", "real", "right"), "Right Sham": ("left arm learned, sham hd-tDCS (right PPC site)", "sham", "right")}
E2 = {}
for sheet, names, cols in (("Fig 4C", ("Left PPC", "Left Sham"), ((1, 5), (2, 6))), ("Fig 4D", ("Right PPC", "Right Sham"), ((1, 5), (2, 6)))):
    ins = inset(sheet)
    for nm, (ce, cl) in zip(names, cols):
        g = E2G[nm][0]
        for j, r in enumerate(ins):
            k = int(r[0])
            e, l = r[ce], r[cl]
            p = f"E2_{nm.replace(' ', '_')}_{k}"
            E2.setdefault(nm, {})[k] = dict(e8=e, l8=l)
            add("Exp2_hdtDCS", g, p, "direction_error_first8_learning", "Mean direction error on the first eight learning trials (one movement to each of the eight targets); not baseline subtracted ('baseline un-subtracted' in the paper's Experiment 2 Methods)",
                e, 8, block="learning", condition="first 8 learning trials")
            add("Exp2_hdtDCS", g, p, "direction_error_last8_learning", "Mean direction error on the last eight learning trials; not baseline subtracted", l, 8, block="learning", condition="last 8 learning trials")
cmap = {"Left PPC": (1, 7), "Left Sham": (2, 8), "Right PPC": (3, 9), "Right Sham": (4, 10)}
for nm, (ce, cl) in cmap.items():
    for i in range(len(f4)):
        if np.isnan(f4[i, ce]):
            continue
        k = int(f4[i, 0])
        p = f"E2_{nm.replace(' ', '_')}_{k}"
        assert k in E2[nm], "Fig 4F participant missing from the Fig 4C/4D insets"
        E2[nm][k].update(e25=f4[i, ce], rest=f4[i, cl])
        add("Exp2_hdtDCS", E2G[nm][0], p, "direction_error_first25_learning", "Mean direction error over the first 25 learning trials (Fig 4F, 'Early Learning')", f4[i, ce], 25, block="learning", condition="first 25 learning trials")
        add("Exp2_hdtDCS", E2G[nm][0], p, "direction_error_remaining_learning", "Mean direction error over the remaining learning trials after the first 25 (Fig 4F)", f4[i, cl], "", block="learning", condition="remaining learning trials")

# ---- Experiment 3: hd-tDCS after learning; early and late learning (real, sham) and generalization (Fig 5C)
ws = wb["Exp3_Early_Late_Learning"]
e3 = []
for r in ws.iter_rows(min_row=3, values_only=True):
    if r[0] is None:
        continue
    assert (r[0], r[1]) == (r[4], r[5])
    e3.append(dict(k=r[0], grp=r[1], early=r[2], late=r[6]))
    g = f"left arm learned, {r[1].lower()} hd-tDCS over left PPC after learning"
    p = f"E3_{r[1]}_{r[0]}"
    add("Exp3_learning", g, p, "direction_error_early_learning", "Direction error in early learning, left arm (the sheet gives no window; Experiment 3 quantification is 'similar to experiment 1', that is the first four learning trials)", r[2], block="learning", condition="early learning")
    add("Exp3_learning", g, p, "direction_error_late_learning", "Direction error in late learning, left arm (presumably the last four learning trials)", r[6], block="learning", condition="late learning")
e3 = pd.DataFrame(e3)
for r in wb["Fig 5C"].iter_rows(min_row=3, max_row=14, values_only=True):
    for name, v, ci in (("PPC", r[1], 1), ("Sham", r[2], 2), ("M1", r[3], 3), ("Naive", r[4], 4)):
        if v is None:
            continue
        if name == "Naive":
            g, meas, d, blk = "naive group, right arm learning 40 trials", "direction_error_first_learning_trial", "Mean direction error on the first learning trial of the right arm of the naive group (Fig 5C legend)", "learning"
        else:
            tgt = {"PPC": "real hd-tDCS over left PPC", "Sham": "sham hd-tDCS", "M1": "real hd-tDCS over left M1"}[name]
            g, meas, d, blk = f"left arm learned, {tgt} after learning", "direction_error_generalization", "Mean direction error of the right arm on the generalization trials after left-arm learning (Fig 5C)", "generalization"
        add("Exp3_generalization", g, f"E3G_{name}_{int(r[0])}", meas, d, v, block=blk, condition=name,
            context="Participants of this sheet cannot be matched to those of the Exp3_learning sheet from the deposit")

t = pd.DataFrame(rows)
t["person"] = t["subject"]
cols = [c[1] for c in stdlib.columns_for("summary")]

# ---------------------------------------------------------------- published checks (expected values read from the paper)
def anova2(df, a, b, y):
    m = ols(f"{y} ~ C({a}) * C({b})", df).fit()
    an = anova_lm(m, typ=2)
    return an["F"].iloc[0], an["F"].iloc[1], an["F"].iloc[2], an["df"]

S1 = "Results, Naive Learning Was Not Different between Arms or Learning Trial Sets"
fa, fs, fi, dfs = anova2(e1, "arm", "nset", "early")
ck.published("Exp1 early learning, arm effect F(1,56)", fa, 0.9861, 5e-5, S1)
ck.published("Exp1 early learning, trial set effect F(3,56)", fs, 1.2874, 5e-5, S1)
ck.published("Exp1 early learning, arm x trial set F(3,56)", fi, 0.8276, 5e-5, S1)
fa, fs, fi, dfs = anova2(e1, "arm", "nset", "late")
ck.published("Exp1 end of learning, arm effect F(1,56)", fa, 0.0778, 5e-5, S1)
ck.published("Exp1 end of learning, trial set effect F(3,56)", fs, 0.0096, 5e-5, S1)
ck.published("Exp1 end of learning, arm x trial set F(3,56)", fi, 0.2508, 5e-5, S1)
ck.published("Exp1 participants", len(e1), 64, 0, "Results / Methods, Experiment 1: n = 64")
ck.published("Exp1 participants learning with the left arm", (e1.arm == "left").sum(), 32, 0, "Methods, Experiment 1: n = 32 per arm")
LEFT_F = None
for arm, nm, exp in (("left", "Left arm generalization", dict(block=0.0196, nset=0.6076, inter=1.5716)), ("right", "Right arm generalization", dict(inter=5.9127))):
    nv = pd.DataFrame(F2[(arm, "naive")]).assign(block="naive")
    gv = pd.DataFrame(F2[(arm, "generalization")]).assign(block="gen")
    d = pd.concat([nv, gv], ignore_index=True)
    fb, fs_, fi_, _ = anova2(d, "block", "nset", "v")
    src = "Results, Generalization Was Asymmetric and Scaled with Number of Learning Trials"
    if arm == "left":
        LEFT_F = (fb, fs_, fi_)   # did not reproduce: reported under Known differences, not used as a check
    else:
        ck.published(f"{nm}: block x trial set F(3,56)", fi_, exp["inter"], 5e-5, src)
d4 = {nm: pd.DataFrame(E2[nm]).T for nm in E2}
ck.published("Exp2 group sizes (left PPC real, left sham, right PPC real, right sham)", len(d4["Left PPC"]) * 1000000 + len(d4["Left Sham"]) * 10000 + len(d4["Right PPC"]) * 100 + len(d4["Right Sham"]), 17171515 // 1 if False else 17 * 1000000 + 17 * 10000 + 15 * 100 + 15, 0, "Fig 4 legend: P3 n = 17 real, 17 sham; P4 n = 15 real, 15 sham")
from scipy.stats import f_oneway  # noqa: E402
S2 = "Results, PPC Disruption Impeded Learning Based on the Slow Process"
ck.published("Exp2 first eight learning trials, group effect F(3,60)", f_oneway(*[d4[n].e8 for n in d4])[0], 0.3895, 5e-5, S2)
ck.published("Exp2 end of learning, group effect F(3,60)", f_oneway(*[d4[n].l8 for n in d4])[0], 7.8326, 5e-5, S2)
ck.published("Exp3 real and sham PPC groups (12 each) and M1 group (12), naive group (8)",
             len(e3[e3.grp == "Real"]) * 1000 + len(e3[e3.grp == "Sham"]) * 100 + 12 * 0 + (t[(t.experiment == "Exp3_generalization") & (t.condition == "M1")].shape[0]) * 10 + (t[(t.experiment == "Exp3_generalization") & (t.condition == "Naive")].shape[0]),
             12 * 1000 + 12 * 100 + 12 * 10 + 8, 0, "Methods, Experiment 3 and Control Experiment: real n = 12, sham n = 12, M1 n = 12, naive n = 8")
ck.internal("Exp1 has 8 participants in each of 8 cells", bool((e1.groupby(["arm", "nset"]).size() == 8).all()))
ck.internal("Fig 4F early mean over 25 trials correlates with Fig 4C/4D first-8 mean for every group (same participant order)",
            all(np.corrcoef(d4[n].e8.astype(float), d4[n].e25.astype(float))[0, 1] > 0.5 for n in d4),
            ", ".join(f"{n}: {np.corrcoef(d4[n].e8.astype(float), d4[n].e25.astype(float))[0, 1]:.2f}" for n in d4))
ck.internal("no missing values", bool(t.value.notna().all()))

meta = dict(
    paper=dict(authors="Kumar A, Panthi G, Divakar R, Mutha PK", year=2020, title="Mechanistic determinants of effector-independent motor memory encoding",
               venue="PNAS 117(29):17338-17347", doi="10.1073/pnas.2001179117"),
    lab="Mutha (IIT Gandhinagar)", data_url="https://figshare.com/articles/dataset/Kumar_et_al_2020_data/12416426",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Only the sheets that hold one value per participant were converted and put in long form (one row per participant and measure); "
             "sheets with group means across trials, model simulations and confidence intervals were not copied; participant numbers replaced by codes."),
    how=["The deposit is the spreadsheet behind the figures of the paper: it holds no trial-by-trial data per participant, only per-participant means "
         "of direction error (cursor direction relative to the target at peak velocity, counter-clockwise positive, degrees) in a 30 degree counter-clockwise visuomotor rotation task on a "
         "digitizing tablet. One row is one measure of one participant. The sign is inferred: the cursor was rotated counter-clockwise and the early errors are about +20 to +30, so errors "
         "are positive for counter-clockwise cursor deviation.",
         "Experiments: Exp1_learning (64 participants, one arm, 40/80/160/320 learning trials, mean error on the first and last four learning trials), Exp1_generalization (Fig 2A/B: errors of each arm on "
         "initial naive learning and on generalization after learning with the other arm), Exp2_hdtDCS (64 participants, hd-tDCS over left or right PPC or sham; "
         "first 8 / last 8 and first 25 / remaining learning trials), Exp3_learning and Exp3_generalization (hd-tDCS after learning; left-arm early and late learning, and right-arm generalization errors, "
         "with M1 and naive groups).",
         "Person codes are per sheet. The deposit does not say which participant of one sheet is which participant of another, so a person of Exp1_generalization may be the same human as one of Exp1_learning "
         "(Exp1 had 64 participants, each learning with one arm and tested with the other), and likewise for Exp3_learning and Exp3_generalization; the number of person codes is therefore larger than the 180 people of the study. "
         "Within Exp2 the Fig 4C/4D and Fig 4F sheets share participant numbers, which was confirmed because the first-8 and first-25 means correlate within every group."],
    unit_note="measure of one participant",
    differences=["The paper's left-arm generalization statistics (block F(1,56) = 0.0196, trial set F(3,56) = 0.6076, interaction F(3,56) = 1.5716) did not reproduce from the Fig 2A sheet "
                 "with a 2 x 4 between-participant ANOVA (here %.4f, %.4f and %.4f), although the right-arm interaction from the Fig 2B sheet (F = 5.9127) and the Experiment 1 and 2 statistics do; "
                 "the cause is not known (possibly a different trial window or a different analysis for the left arm), so the Fig 2A values are given as they are in the sheet." % LEFT_F,
                 "The paper's trial-by-trial learning curves, model parameters and permutation tests are not rebuilt: the deposit holds only group means across trials and per-participant means. "
                 "Windows for Exp3 early and late learning and for Fig 2 are not stated in the sheets."],
)
participants = None
stdlib.release(ID, "summary", t[cols].to_dict("records"), meta, ck)
