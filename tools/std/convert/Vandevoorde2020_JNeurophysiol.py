"""Vandevoorde & Orban de Xivry (2020), J Neurophysiol 124(1):152-167: why the explicit component of adaptation is limited in older adults.
Two cued visuomotor-rotation experiments (40 deg) on a digitizing tablet with dual-task baselines: E1 (81 adults; flanker dual task) and E2 (62 adults;
visuospatial working-memory dual task). Per-trial hand angle at 4 cm ('VMR_filt') from the deposit's Processed folders (E1_ANA/TRA_Rot_<code>.mat,
DualCuedadaptation_ANA/TRA_Rot_<code>.mat). Deposit: OSF https://osf.io/ks2j8/ (no license stated). Cognitive scores (FlankerDualdata.mat, WMC*.mat,
CognitiveResults xlsx) are not converted: they are not motor behaviour.
62 of the 81 E1 participants are also the 62 people of E1b in Vandevoorde2019_NeurobiolAging (identical hand angles, trials 261-622).
Run on the PC: python3 -I tools/std/pc.py run Vandevoorde2020_JNeurophysiol"""
import json
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Vandevoorde2020_JNeurophysiol"
RAW = stdlib.raw_dir(ID)
names = [c[1] for c in stdlib.reach90_columns()]
checks = stdlib.Checks()
# E1 codes also present as E1b in Vandevoorde2019_NeurobiolAging (found by comparing the hand-angle series of trials 261-622)
OVERLAP = set(range(1, 32)) | set(range(42, 73))
UNCUED_E1 = [7, 16, 25, 35, 45, 53, 61, 72, 81]   # positions within each 81-trial adaptation block (authors' Analysis3.m, paper)
DESIGN = {
    "E1": [("dual_task_baseline", 0, 260, "other"), ("baseline", 260, 280, "baseline"), ("learning", 280, 361, "learning"), ("washout1", 361, 460, "washout"),
           ("relearning", 460, 541, "relearning"), ("washout2", 541, 622, "washout")],
    "E2": [("baseline", 0, 45, "baseline"), ("learning", 45, 207, "learning"), ("washout", 207, 234, "washout")],
}


def sheet(f, name):
    wb = openpyxl.load_workbook(RAW / f, data_only=True)
    return [r[0] for r in wb[name].iter_rows(values_only=True)]


rows = []
fin = {}
part = []
for exp, pref in (("E1", "E1"), ("E2", "E2_DualCuedadaptation")):
    codes = [int(c) for c in sheet(f"{exp}_all_subjects.xlsx", "Code")]
    old = [int(o) for o in sheet(f"{exp}_all_subjects.xlsx", "Old")]
    assert len(codes) == len(old)
    n_tr = DESIGN[exp][-1][2]
    blk = np.empty(n_tr, object)
    ph = np.empty(n_tr, object)
    for nm, a, b, p in DESIGN[exp]:
        blk[a:b] = nm
        ph[a:b] = p
    recs = []
    for code, o in zip(codes, old):
        ana = sio.loadmat(RAW / f"{exp}_{pref.split('_', 1)[1] + '_' if '_' in pref else ''}ANA_Rot_{code}.mat")
        tra = sio.loadmat(RAW / f"{exp}_{pref.split('_', 1)[1] + '_' if '_' in pref else ''}TRA_Rot_{code}.mat")
        v = ana["VMR_filt"].ravel().astype(float)
        assert len(v) == n_tr, (exp, code, len(v))
        cw = int(tra["Clockwise"].ravel()[0])
        sgn = 1 if cw else -1
        if exp == "E1":
            unc = {280 + c - 1 for c in UNCUED_E1} | {460 + c - 1 for c in UNCUED_E1}
        else:
            pert = tra["Perturbation"].ravel()
            unc = {45 + int(i) for i in np.where(pert == 0)[0] if 45 + int(i) < 207}   # zero entries of the authors' Perturbation vector (0-based i is position i+1)
        recs.append(dict(code=code, old=o, cw=cw, v=v))
        grp = "older" if o else "young"
        vv = v * sgn
        rot = -40 if cw else 40
        for t in range(n_tr):
            adapt = ph[t] in ("learning", "relearning")
            pert_on = adapt and t not in unc
            missing = bool(np.isnan(vv[t]))
            r = dict.fromkeys(names, "")
            r.update(dataset=ID, experiment=f"{ID}_{exp}", study=ID, subject=f"{exp}_{code:02d}", person=f"{exp}_{code:02d}", id_type="lab",
                     person_link="unknown" if exp == "E2" else "assumed_unique", group=grp, prior_exposure="unknown", subject_excluded_by_authors="False",
                     handedness="right", device="tablet", setting="lab", population="healthy", trial=str(t), trial_orig=str(t + 1), session="0",
                     block=str(blk[t]), phase=ph[t], trial_kind="reach", break_before="True" if t == 0 else "False", target_distance_norm="1",
                     target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none", instr_goal="cursor_to_target",
                     instr_strategy="allowed", instr_disclosure="exists", instr_report="none", hand_used="right",
                     hand_angle="" if missing else "%.8g" % vv[t], hand_measure="fixed_distance", fb_visible="online", fb_form="cursor",
                     fb_endpoint_transform_only="False", fb_mode="rotated" if pert_on else "veridical", online_cursor="perturbed" if pert_on else "veridical",
                     rotation=str(rot) if pert_on else "0", cue_elements="cursor:pink_square" if pert_on else "cursor:white_dot",
                     excluded_by_authors="True" if missing else "False")
            rows.append(r)
        pr = dict(person=f"{exp}_{code:02d}")
        if exp == "E1":
            pr["also_in_Vandevoorde2019_E1b"] = code in OVERLAP
        part.append(pr)
    fin[exp] = recs

# ------------------------------------------------------------------ published checks
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
M = "Methods, Participants"
for exp, (ny, no) in (("E1", (41, 40)), ("E2", (30, 32))):
    r = fin[exp]
    checks.published(f"{exp}: young participants", sum(1 for x in r if not x["old"]), ny, 0, M)
    checks.published(f"{exp}: older participants", sum(1 for x in r if x["old"]), no, 0, M)
checks.published("participants over E1 and E2", len(fin["E1"]) + len(fin["E2"]), 143, 0, "Methods, Participants (143 recruited)")
checks.published("trials of the E1 adaptation baseline + learning + washout 1 + relearning + washout 2 (260 dual-task + 20 + 81 + 99 + 81 + 81)", len(DESIGN["E1"]) and DESIGN["E1"][-1][2], 622, 0, "Methods, Experiment 1")
checks.published("E2 learning trials (18 cycles of 9)", 207 - 45, 162, 0, "Methods, Experiment 2")
checks.published("E2 baseline trials (5 cycles of 9)", 45, 45, 0, "Methods, Experiment 2")
checks.published("E1 uncued trials per adaptation block", len(UNCUED_E1), 9, 0, "Methods, Experiment 1 (nine uncued trials per adaptation block)")


def final_adapt(recs, la, lb, ba, bb):
    out = []
    for x in recs:
        v = x["v"]
        out.append(dict(age=x["old"], rot=x["cw"], y=np.nanmean(v[lb - 19:lb - 1]) - np.nanmean(v[ba:bb])))
    return pd.DataFrame(out)


def age_F(d):
    # the authors use MATLAB anovan(..., 'model', 'full') (Analysis1.m), whose default is type III sums of squares
    mod = smf.ols("y ~ C(age, Sum) * C(rot, Sum)", d).fit()
    a = sm.stats.anova_lm(mod, typ=3)
    F = float(a.loc["C(age, Sum)", "F"])
    return F, float(a.loc["Residual", "df"]), float(a.loc["C(age, Sum)", "sum_sq"] / (a.loc["C(age, Sum)", "sum_sq"] + a.loc["Residual", "sum_sq"]))


# E1: learning block (baseline = last 18 of the 20 trials before it, authors' offsets), relearning block (baseline = last 18 trials of washout 1)
FL = age_F(final_adapt(fin["E1"], 280, 361, 261, 279))
FR = age_F(final_adapt(fin["E1"], 460, 541, 441, 459))
F2 = age_F(final_adapt(fin["E2"], 45, 207, 26, 44))
print("DEV E1 learning", FL, "relearning", FR, "E2", F2)
checks.published("E1 learning block: age effect on final adaptation, F(1,77)", FL[0], 8.5, 0.051, "Results, Experiment 1, adaptation")
checks.published("E1 relearning block: age effect on final adaptation, F(1,77)", FR[0], 11.1, 0.051, "Results, Experiment 1, adaptation")
checks.published("E2: age effect on final adaptation, F(1,58)", F2[0], 9.7, 0.051, "Results, Experiment 2, adaptation")
checks.published("E1 error df of that ANOVA", FL[1], 77, 0, "Results, Experiment 1, adaptation")
checks.published("E2 error df of that ANOVA", F2[1], 58, 0, "Results, Experiment 2, adaptation")
df = pd.DataFrame(rows)
checks.internal("row count", len(df) == 81 * 622 + 62 * 234, str(len(df)))
checks.internal("E2: 18 uncued trials per participant", int(((df.experiment.str.endswith("E2")) & (df.rotation == "0") & (df.phase == "learning")).sum()) == 18 * 62)
for exp in ("E1", "E2"):
    d = df[(df.experiment == f"{ID}_{exp}") & (df.hand_angle != "") & (df.phase == "learning")]
    c = np.array([float(h) * (-1 if float(r) > 0 else 1) for h, r in zip(d.hand_angle, d.rotation)])
    u = (d.rotation == "0").values
    checks.internal(f"{exp}: uncued trials show less compensation than cued trials", c[u].mean() < c[~u].mean() - 3, f"{c[u].mean():.1f} vs {c[~u].mean():.1f}")
bad = [c for c in checks.items if not c["ok"]]
for c in bad:
    print("FAILED", c)
paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Orban de Xivry (KU Leuven)", data_url="https://osf.io/ks2j8/", license="none stated", license_url="",
    changes=("Converted the per-participant MATLAB files of E1 and E2 to the OpenMotor reaching format; rotation direction and a sign flip for the counter-clockwise group "
             "(see 'How') applied; participant codes replaced. Cognitive test scores, dates in the TRA files and experimenter names are not included."),
    unit_note="trial of one participant in E1 or E2",
    how=["One row is one reach (right hand, digitizing tablet, 10 cm to a target) of a young or older adult in a cued 40 deg visuomotor rotation experiment. hand_angle is the authors' "
         "'VMR_filt': hand direction relative to the target at the first sample beyond 4 cm (hand_measure fixed_distance). Missing values (about 0.1 percent: reaches the authors removed "
         "as outliers) are kept as rows with empty hand_angle and excluded_by_authors True.",
         "Sign: as in the authors' other files, VMR_filt rises toward +20 to +30 deg during learning in both rotation directions, so it is stored with compensation positive; the "
         "counter-clockwise group (Clockwise = 0) was multiplied by -1 here to get counter-clockwise positive (inferred from the data; the E2 Perturbation vector is +40 for Clockwise = 1 and -40 for Clockwise = 0). "
         "Rotation is -40 for clockwise and +40 for counter-clockwise rotations (cursor = hand + rotation).",
         "E1: trials 1-260 are the dual-task baseline phase (reaches with a white-dot cursor, alone or together with a flanker task; the file does not say which trial is which, so they are one block), "
         "then 20 baseline trials, 81 learning trials, 99 washout, 81 relearning and 81 washout trials. E2: 45 baseline, 162 learning and 27 washout trials. During adaptation blocks the cursor was a pink square "
         "with the rotation; uncued trials (white dot, no rotation) are positions 7, 16, 25, 35, 45, 53, 61, 72 and 81 of each E1 adaptation block, and the 18 positions at which the authors' E2 Perturbation "
         "vector is 0 (the vector has 170 entries, 8 more than the 162 learning trials; the extra entries are not used). Targets (9 on a circle in E1, learning targets at 60, 180 and 300 deg) are not in the files, so target_angle is empty.",
         "Older (60-75 y) and young (20-35 y) adults are in group; ages and sex per person are not in the files (the Gender sheet has fewer rows than participants and no code, so it is not used). "
         "participants.csv marks the E1 participants who are also the E1b participants of Vandevoorde2019_NeurobiolAging (identical hand angles in trials 261-622): "
         "62 of the 81. Drop them when pooling the two datasets."],
    differences=["62 of the 81 E1 participants are the same people and the same reaches as E1b of Vandevoorde2019_NeurobiolAging (the paper says 62 overlapped with a prior publication); "
                 "this file adds their dual-task baseline reaches (trials 1-260) and 19 new participants.",
                 "The paper's implicit and explicit adaptation statistics need the cued/uncued trial pairs and the authors' baseline handling; only the final-adaptation ANOVAs are reproduced."])
if bad:
    print("not releasing: failed checks above")
else:
    pcol = [("also_in_Vandevoorde2019_E1b", "true/false", "", "E1 only: this person's hand angles in trials 261-622 are identical to a participant of E1b in the Vandevoorde2019_NeurobiolAging release.")]
    part = pd.DataFrame(part)
    stdlib.release(ID, "reach90", rows, meta, checks, participants=part, participant_columns=pcol)
