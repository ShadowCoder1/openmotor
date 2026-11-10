"""Vandevoorde & Orban de Xivry (2019), Neurobiol Aging 80:138-153: visuomotor rotation (40 deg) in young and older adults on a digitizing tablet.
Four of the paper's five experiments are converted per trial (hand angle at 4 cm, 'VMRfilt'): E1a and E1b (cued adaptation: pink-square cursor = rotation,
white dot = no rotation, to separate explicit and implicit adaptation), E3 (adaptation with one-minute breaks) and E4 (task-irrelevant 40 deg clamped
feedback). E2 (single-trial error-based learning) is not converted: the deposit holds only binned analysis outputs for it, not per-trial hand angles.
Deposit: OSF https://osf.io/vncce/ (SubjectsDataE1a/E1b/E3/E4.mat, MATLAB structs allSubjectsData and subjectinfo; no license stated).
Run on the PC: python3 -I tools/std/pc.py run Vandevoorde2019_NeurobiolAging"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Vandevoorde2019_NeurobiolAging"
RAW = stdlib.raw_dir(ID)
names = [c[1] for c in stdlib.reach90_columns()]
checks = stdlib.Checks()


def load(e):
    m = sio.loadmat(RAW / f"SubjectsData{e}.mat", squeeze_me=True, struct_as_record=False)
    return m["allSubjectsData"], m["subjectinfo"]


UNCUED_E1A = [20, 33, 36, 40, 46, 52, 68, 72]          # trial numbers within each 80-trial adaptation block (authors' scripts, paper 2.3)
UNCUED_E1B = [7, 16, 25, 35, 45, 53, 61, 72, 81]       # idem, 81-trial blocks (paper 2.3)
# block tables: (name, first index (0-based), last index +1, phase)
DESIGN = {
    "E1a": [("baseline", 0, 80, "baseline"), ("learning", 80, 160, "learning"), ("washout1", 160, 260, "washout"),
            ("relearning", 260, 340, "relearning"), ("washout2", 340, 400, "washout"), ("after_design_end", 400, 500, "other")],
    "E1b": [("dual_task_not_shared", 0, 260, "other"), ("baseline", 260, 280, "baseline"), ("learning", 280, 361, "learning"),
            ("washout1", 361, 460, "washout"), ("relearning", 460, 541, "relearning"), ("washout2", 541, 622, "washout")],
    "E3": [("baseline", 0, 60, "baseline"), ("learning", 60, 140, "learning"), ("washout1", 140, 340, "washout"),
           ("relearning", 340, 420, "relearning"), ("washout2", 420, 500, "washout"), ("after_design_end", 500, 600, "other")],
    "E4": [("baseline", 0, 108, "baseline"), ("clamped_feedback", 108, 468, "learning"), ("washout", 468, 540, "washout")],
}
BREAKS = {"E3": [10, 50, 70, 100, 130, 160, 200, 240, 280, 330, 350, 380, 410, 440], "E4": [100, 200, 300, 400]}   # 1-based, "before trial" (paper 2.5, 2.6)
PRIOR = {"E1a": "true", "E3": "true", "E1b": "false", "E4": "true"}   # E1a/E3 followed a session of E2 (single perturbations) a week earlier; E4 followed E1b
rows = []
persons = {}
per_summary = {}
finfo = {}
uncued_check = {}

for exp in ("E1a", "E1b", "E3", "E4"):
    S, SI = load(exp)
    finfo[exp] = (len(S), [str(np.ravel(x.age)[0]) for x in SI], [int(x.Clockwise) for x in SI])
    n_tr = len(np.asarray(S[0].VMRfilt))
    assert n_tr == DESIGN[exp][-1][2], (exp, n_tr)
    blk = np.empty(n_tr, object)
    ph = np.empty(n_tr, object)
    for nm, a, b, p in DESIGN[exp]:
        blk[a:b] = nm
        ph[a:b] = p
    for i, (s, si) in enumerate(zip(S, SI), start=1):
        cw = int(si.Clockwise)
        sgn = 1 if cw else -1
        grp = "young" if str(np.ravel(si.age)[0]) == "young" else "older"
        pers = f"P3_{i:02d}" if exp in ("E1b", "E4") else f"{exp}_{i:02d}"
        v = np.asarray(s.VMRfilt, float) * sgn
        dur = np.asarray(s.TrialTimeDuration, float)
        spd = np.asarray(s.Speed, float)
        tg = np.asarray(s.Targets, float)[2] if exp == "E4" else None
        unc = set()
        if exp == "E1a":
            unc = {80 + c - 1 for c in UNCUED_E1A} | {260 + c - 1 for c in UNCUED_E1A}
        elif exp == "E1b":
            unc = {280 + c - 1 for c in UNCUED_E1B} | {460 + c - 1 for c in UNCUED_E1B}
        rot_sign = -40 if cw else 40   # cursor = hand + rotation, counter-clockwise positive; clockwise rotation is -40
        for t in range(n_tr):
            r = dict.fromkeys(names, "")
            adapt_block = ph[t] in ("learning", "relearning")
            missing = bool(np.isnan(v[t]))
            r.update(dataset=ID, experiment=f"{ID}_{exp}", study=ID, subject=f"{exp}_{i:02d}", person=pers, id_type="lab",
                     person_link="verified" if exp in ("E1b", "E4") else "assumed_unique", group=grp, prior_exposure=PRIOR[exp],
                     subject_excluded_by_authors="False", handedness="right", device="tablet", setting="lab", population="healthy",
                     trial=str(t), trial_orig=str(t + 1), session="0", block=str(blk[t]), phase=ph[t],
                     trial_kind="missing" if blk[t] == "dual_task_not_shared" else "reach",
                     break_before="True" if (t == 0 or (t + 1) in BREAKS.get(exp, [])) else "False",
                     target_distance_norm="1", target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
                     instr_goal="hand_to_target" if exp == "E4" else "cursor_to_target", instr_strategy="forbidden" if exp == "E4" else "allowed",
                     instr_disclosure="exists" if exp in ("E1a", "E1b") else ("exact" if exp == "E4" else "none"), instr_report="none", hand_used="right",
                     hand_angle="" if missing else "%.8g" % v[t], hand_measure="fixed_distance", hand_extent="" , excluded_by_authors="True" if missing else "False",
                     fb_visible="online", fb_form="cursor", fb_endpoint_transform_only="False")
            if blk[t] == "dual_task_not_shared":
                r.update(fb_visible="unknown", fb_mode="unknown", excluded_by_authors="True")
            if exp in ("E1a", "E1b"):
                perturbed = adapt_block and t not in unc
                r.update(fb_mode="rotated" if perturbed else "veridical", online_cursor="perturbed" if perturbed else "veridical",
                         rotation=str(rot_sign) if perturbed else ("0" if blk[t] != "dual_task_not_shared" else ""),
                         cue_elements="cursor:pink_square" if perturbed else ("cursor:white_dot" if blk[t] != "dual_task_not_shared" else ""))
                if blk[t] == "dual_task_not_shared":
                    r["fb_mode"] = "unknown"; r["online_cursor"] = ""
            elif exp == "E3":
                perturbed = adapt_block
                r.update(fb_mode="rotated" if perturbed else "veridical", online_cursor="perturbed" if perturbed else "veridical",
                         rotation=str(rot_sign) if perturbed else "0")
            else:  # E4
                clamped = ph[t] == "learning"
                r.update(fb_mode="clamped" if clamped else "veridical", online_cursor="perturbed" if clamped else "veridical",
                         clamp=str(rot_sign) if clamped else "", rotation="" if clamped else "0")
                r.update(target_id=str(int(round((tg[t] - 20) / 40)) + 1), instr_disclosure="exact")
            if not np.isnan(dur[t]):
                r["mt_ms"] = "%.8g" % (dur[t] * 1000)
            if not np.isnan(spd[t]):
                r["speed_warning"] = "1" if spd[t] in (1, 2) else "0"
            rows.append(r)
        per = dict(person=pers)
        for k, nm in (("ImplicitLevelBlock1", "implicit_block1"), ("ImplicitLevelBlock2", "implicit_block2"), ("ExplicitLevelBlock1", "explicit_block1"),
                      ("ExplicitLevelBlock2", "explicit_block2"), ("Cuedsavings", "cued_savings"), ("OverallAdaptLevel", "overall_adapt_level"), ("Implicit", "implicit")):
            if hasattr(s, k):
                per[f"{exp.lower()}_{nm}"] = float(getattr(s, k))
        per_summary.setdefault(pers, {}).update(per)
        # per-subject quantities for checks
        per_summary[pers][f"_{exp}_adapt"] = (grp, cw, v)

# ------------------------------------------------------------------ published checks
M = "Methods 2.7 and Table 1"
for exp, (n, ages, cws) in finfo.items():
    pass
cnt = {e: (finfo[e][0], finfo[e][1].count("young"), finfo[e][1].count("old")) for e in finfo}
for e, (tot, ny, no) in (("E1a", (40, 20, 20)), ("E3", (41, 21, 20)), ("E1b", (62, 31, 31)), ("E4", (62, 31, 31))):
    checks.published(f"{e}: participants", cnt[e][0], tot, 0, M)
    checks.published(f"{e}: young participants", cnt[e][1], ny, 0, M)
    checks.published(f"{e}: older participants", cnt[e][2], no, 0, M)
checks.published("young participants in the analysed sample (E1a, E3, E1b)", cnt["E1a"][1] + cnt["E3"][1] + cnt["E1b"][1], 72, 0, "Methods 2.1, Results 3.1")
checks.published("older participants in the analysed sample (E1a, E3, E1b)", cnt["E1a"][2] + cnt["E3"][2] + cnt["E1b"][2], 71, 0, "Methods 2.1, Results 3.1")
checks.published("trials in E4", len({r['trial'] for r in rows if r['experiment'].endswith('E4')}), 540, 0, "Methods 2.6")
checks.published("trials in the clamped-feedback block of E4 (40 cycles of 9)", sum(1 for r in rows if r['experiment'].endswith('E4') and r['subject'] == 'E4_01' and r['phase'] == 'learning'), 360, 0, "Methods 2.6")
checks.published("trials of one E1b adaptation (learning) block", sum(1 for r in rows if r['experiment'].endswith('E1b') and r['subject'] == 'E1b_01' and r['phase'] == 'learning'), 81, 0, "Methods 2.3")
checks.published("trials of the E1a learning block", sum(1 for r in rows if r['experiment'].endswith('E1a') and r['subject'] == 'E1a_01' and r['phase'] == 'learning'), 80, 0, "Methods 2.3")
checks.published("uncued trials per adaptation block in E1b", len(UNCUED_E1B), 9, 0, "Methods 2.3")
checks.published("uncued trials per adaptation block in E1a", len(UNCUED_E1A), 8, 0, "Methods 2.3")

# final adaptation level: mean of the last 18 learning trials minus the mean of the last 18 baseline trials; 2-way ANOVA age x rotation direction
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
RAWV = {}
for exp in ("E1a", "E3", "E1b"):
    S, SI = load(exp)
    des = {nm: (a, b) for nm, a, b, _ in DESIGN[exp]}
    la, lb = des["learning"]
    ba, bb = des["baseline"]
    recs = []
    for s, si in zip(S, SI):
        v = np.asarray(s.VMRfilt, float)
        recs.append(dict(age=str(np.ravel(si.age)[0]), rot=int(si.Clockwise),
                         y=np.nanmean(v[lb - 19:lb - 1]) - np.nanmean(v[bb - 19:bb - 1])))
    d = pd.DataFrame(recs)
    # the deposit's VMRfilt is already compensation-positive for both rotation directions, so the baseline subtraction above is in that frame
    for typ in (2, 3):
        pass
    mod = smf.ols("y ~ C(age) + C(rot)", d).fit()
    aov = sm.stats.anova_lm(mod, typ=2)
    F = float(aov.loc["C(age)", "F"]); df2 = float(aov.loc["Residual", "df"]); ss = aov.loc["C(age)", "sum_sq"]; sse = aov.loc["Residual", "sum_sq"]
    eta = ss / (ss + sse)
    print("DEV", exp, "F", round(F, 3), "df2", df2, "eta", round(eta, 3))
    RAWV[exp] = (F, df2, eta)
checks.published("E1a: age effect on final adaptation level, F(1,37) (additive model of age and rotation direction)", RAWV["E1a"][0], 22.9, 0.051, "Results 3.1")
checks.published("E3: age effect on final adaptation level, F(1,38)", RAWV["E3"][0], 8.7, 0.051, "Results 3.1")
checks.published("E1a: error df of that ANOVA", RAWV["E1a"][1], 37, 0, "Results 3.1")
checks.published("E3: error df of that ANOVA", RAWV["E3"][1], 38, 0, "Results 3.1")

# ------------------------------------------------------------------ internal checks
df = pd.DataFrame(rows)
checks.internal("row count = sum of subjects x trials", len(df) == 40 * 500 + 62 * 622 + 41 * 600 + 62 * 540, str(len(df)))
checks.internal("E1b and E4 hold the same people in the same order (age group at each index)", finfo["E1b"][1] == finfo["E4"][1])
for exp in ("E1a", "E1b"):
    d = df[(df.experiment == f"{ID}_{exp}") & (df.hand_angle != "")]
    d = d.assign(h=d.hand_angle.astype(float), t=d.trial.astype(int))
    unc = d[(d.rotation == "0") & d.phase.isin(["learning", "relearning"])]
    cued = d[(d.rotation != "0") & (d.rotation != "") & d.phase.isin(["learning", "relearning"])]
    sg = d.rotation.map(lambda x: 0)
    # compare compensation (positive = compensating) of uncued versus cued trials, via the sign-normalized frame
    cmp_ = np.nanmean([float(r.hand_angle) * (-1 if float(r.rotation) > 0 else 1) for r in cued.itertuples()]) if len(cued) else np.nan
    cmp_u = np.nanmean([float(r.hand_angle) * (-1 if float(r.rotation) > 0 else 1) for r in unc.itertuples()]) if len(unc) else np.nan
    print("DEV uncued", exp, round(cmp_u, 2), "cued", round(cmp_, 2))
ha = pd.to_numeric(df.hand_angle)
print("DEV beyond60", (ha.abs() > 60).sum(), ha.abs().max())
checks.internal("fewer than 0.5 percent of hand angles lie beyond +-60 deg (the paper says such reaches were removed)", float((ha.abs() > 60).mean()) < 0.005, f"{(ha.abs() > 60).sum()} of {ha.notna().sum()}")
for exp in ("E1a", "E1b"):
    d = df[(df.experiment == f"{ID}_{exp}") & (df.hand_angle != "") & df.phase.isin(["learning"])]
    rr = d.assign(c=[float(h) * (-1 if float(r) > 0 else 1) for h, r in zip(d.hand_angle, d.rotation.replace('', '0'))], u=(d.rotation == "0"))
    checks.internal(f"{exp}: uncued trials show less compensation than cued trials (cue positions are right)", rr[rr.u].c.mean() < rr[~rr.u].c.mean() - 3,
                    f"uncued {rr[rr.u].c.mean():.1f}, cued {rr[~rr.u].c.mean():.1f}")
bad = [c for c in checks.items if not c["ok"]]
for c in bad:
    print("FAILED", c)

part = []
pcols_all = {}
for p, v in per_summary.items():
    rec = {k: x for k, x in v.items() if not k.startswith("_")}
    part.append(rec)
cols = sorted({k for r in part for k in r if k != "person"})
DESC = {"implicit_block1": "implicit adaptation in the first learning block", "implicit_block2": "implicit adaptation in the relearning block",
        "explicit_block1": "explicit adaptation in the first learning block", "explicit_block2": "explicit adaptation in the relearning block",
        "cued_savings": "cue-evoked savings", "overall_adapt_level": "overall adaptation level", "implicit": "implicit adaptation from the task-irrelevant clamp (mean of the last 10 cycles)"}
pcol = [(c, "number", "degrees", f"{c.split('_')[0].upper()}: authors' analysis output for {DESC['_'.join(c.split('_')[1:])]}, as stored in the deposit (definitions: paper, Methods 2.11); sign as in the deposit (adaptation positive).") for c in cols]
part = pd.DataFrame(part)
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Orban de Xivry (KU Leuven)", data_url="https://osf.io/vncce/", license="none stated", license_url="",
    changes=("Converted four of the five experiments from the deposit's MATLAB structs to the OpenMotor reaching format; the 40 deg rotation direction and a sign "
             "flip for the counter-clockwise group (see 'How') were applied; participant codes replaced. E2 (error-based learning) is not included."),
    unit_note="trial of one participant in one experiment (E1a, E1b, E3 or E4)",
    how=["One row is one reach of a young or older adult (right hand, digitizing tablet, 10 cm to a target) in a 40 deg visuomotor rotation experiment. hand_angle is the authors' "
         "'VMRfilt': the direction of the hand relative to the target at the first sample beyond 4 cm from the start (hand_measure fixed_distance); trials with an error above 60 deg "
         "were removed by the authors as inattentive reaches and are kept as rows with empty hand_angle and excluded_by_authors True.",
         "Sign. In the deposit VMRfilt rises toward +25 to +30 deg during learning in both rotation directions, so the authors stored it with compensation positive; "
         "for the counter-clockwise rotation group (Clockwise = 0) it was therefore multiplied by -1 here to get counter-clockwise positive, as in other OpenMotor reaching files. "
         "This is inferred from the data (and from Perturbation = +40 for Clockwise = 1 and -40 for Clockwise = 0 in the E3 and E4 files), not stated in the paper. "
         "Rotation is -40 for clockwise and +40 for counter-clockwise rotations (cursor = hand + rotation).",
         "E1a and E1b: the cursor was a pink square during adaptation blocks (rotation in force) and a white dot on a few uncued trials (no rotation) and during baseline and washout; "
         "uncued trials are positions 20, 33, 36, 40, 46, 52, 68, 72 (E1a, blocks of 80) and 7, 16, 25, 35, 45, 53, 61, 72, 81 (E1b, blocks of 81) of the two adaptation blocks (authors' scripts and Methods 2.3). "
         "E1b trials 1-260 were a dual-task baseline whose hand angles are not in this file (rows kept as missing). E3: 60 baseline, 80 learning, 200 washout, 80 relearning and 80 washout trials; "
         "break_before marks the one-minute breaks listed in the paper. E4: 108 baseline trials, 360 trials of task-irrelevant clamped feedback (clamp = 40 deg, participants told to ignore the cursor) and 72 washout trials; "
         "target_id 1 to 9 stands for the authors' target angles 20 to 340 deg in steps of 40 (their angle convention is not stated, so target_angle is left empty). Targets of E1a, E1b and E3 are not in the files.",
         "The E1a and E3 files hold 100 more trials (E1a 401-500, E3 501-600) than the paper's design describes (400 and 500 trials); they are labelled after_design_end and their purpose is not documented. "
         "mt_ms is TrialTimeDuration (movement duration) and speed_warning is 1 when the authors' Speed code is 1 (too fast) or 2 (too slow). Ages are not in the files, only young or older (group). "
         "E1b and E4 are the same 62 people in the same order (the authors' script reads E4's rotation from E1b's subject list); E1a and E3 people had done E2 a week earlier. participants.csv holds the authors' per-participant analysis outputs."],
    differences=["The 62 people of E1b are also E1 participants of Vandevoorde2020_JNeurophysiol (identical hand angles in trials 261-622); that release also has their dual-task baseline reaches. Drop one of the two when pooling datasets. ",
                 "The E1b ANOVA (F(1,58) = 2.49 in the paper) is not used as a check: it has 58 error df, so the authors analysed 61 of the 62 people, and the exclusion is not documented. "
                 "The paper gives mean ages (22.75 and 66.88 y) but the deposit only has young or older, so these cannot be checked.",
                 "The paper says reaches with angular errors above 60 deg were removed before processing, but 285 hand angles (0.2 percent) in the deposit lie beyond 60 deg (up to 270 deg); they are kept as shared."])
if bad:
    print("not releasing: failed checks above")
else:
    stdlib.release(ID, "reach90", rows, meta, checks, participants=part, participant_columns=pcol)
