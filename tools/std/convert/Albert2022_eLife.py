"""Albert, Jang, Modchalingam, 't Hart, Henriques, Lerner, Della-Maggiore, Haith, Krakauer & Shadmehr (2022), eLife 11:e65361:
competition between parallel sensorimotor learning systems. The OSF deposit (https://osf.io/mzs6a/) is one ZIP per figure with
MATLAB .mat matrices (participants as columns, trials or epochs as rows). This converter releases the authors' own five
experiments' reach-angle series (and reaction time where they gave it) as the 90-column reaching format:
  Experiment 1  figure1.zip dataHIJKL.mat anglesA (264 x 36 abrupt) and anglesS (264 x 37 stepwise): one row per TRIAL
  Experiment 2  figure3.zip data-experiment-2-no-PT-limit.mat (71 x 9), data-experiment-2-limit-PT.mat (85 x 13): per EPOCH
  Experiment 3  figure3.zip data-experiment-3-no-PT-limit.mat (90 x 35), data-experiment-3-limit-PT.mat (90 x 21) and
                figure8-supplement1.zip data-experiment-3-decay-only.mat (90 x 12): per EPOCH
  Experiment 4  figure8.zip data-experiment-4.mat (200 x 10): per EPOCH
  Experiment 5  figure9.zip data-experiment-5.mat A5A/A5B (9 participants, 5 min gap), A24A/A24B (11, 24 h gap): per EPOCH
Everything else in the deposit is left out (see the README text below). Run: python3 -I tools/std/convert/Albert2022_eLife.py
The .mat files are read as data with scipy (nothing from the deposit is executed); the authors' .m scripts and readmes were
read as text to learn what each matrix is.
"""
import io
import json
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Albert2022_eLife"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
names = [c[1] for c in stdlib.reach90_columns()]
PAPER = "eLife 11:e65361, Materials and methods"


def mat(zname, fname):
    return sio.loadmat(io.BytesIO(zipfile.ZipFile(RAW / zname).read(fname)))


def num(x):
    if x is None or x != x:
        return ""
    x = float(x)
    return "%.8g" % (0.0 if x == 0 else x)


def tf(x):
    return "True" if x else "False"


rows = []
counts = OrderedDict()


def add_series(experiment, group, j, angles, rts, plan, *, setting, device, hand_measure, rt_scale, unit_label, session_of=None,
               break_at=()):
    """angles: 1-D array (one participant), plan: list of (n_rows, phase, block label, fb_visible, fb_mode, instr_strategy)."""
    subj = f"{ID}_{experiment}_{group}_{j}"
    t = 0
    for n_rows, phase, blk, fbv, fbm, strat in plan:
        for k in range(n_rows):
            if t >= len(angles):
                break
            a = angles[t]
            r = dict.fromkeys(names, "")
            miss = bool(np.isnan(a))
            r.update(dataset=ID, study=ID, experiment=experiment, group=group, subject=subj, person=subj, id_type="unknown",
                     person_link="unknown", prior_exposure="unknown", subject_excluded_by_authors="False", population="healthy",
                     setting=setting, device=device, trial=str(t), trial_orig=str(t + 1),
                     session=str(session_of(t) if session_of else 0), block=blk, phase=phase,
                     trial_kind="missing" if miss else ("reach" if experiment == "Exp1_trials" else "other"), break_before=tf(t == 0 or t in break_at),
                     target_visible_during_move="True", target_event="none", target_jump="0", probe_kind="none",
                     instr_goal="cursor_to_target" if fbm != "none" else "hand_to_target", instr_strategy=strat,
                     hand_measure=hand_measure,
                     excluded_by_authors=tf(miss), fb_visible=fbv, fb_mode=fbm, fb_form="cursor" if fbm != "none" else "none")
            if not miss:
                r["hand_angle"] = num(a)
                if rts is not None and not np.isnan(rts[t]):
                    r["rt_ms"] = num(rts[t] * rt_scale)
            rows.append(r)
            t += 1
    assert t == len(angles), (experiment, group, t, len(angles))
    counts[(experiment, group)] = counts.get((experiment, group), 0) + 1


def columns(m, key):
    a = np.asarray(m[key], dtype=float)
    return [a[:, j] for j in range(a.shape[1])]


# ---------------------------------------------------------------- Experiment 1: trials
m = mat("figure1.zip", "dataHIJKL.mat")
for grp, key, rots in (("abrupt", "anglesA", [60, 60, 60, 60]), ("stepwise", "anglesS", [15, 30, 45, 60])):
    cols = columns(m, key)
    for j, c in enumerate(cols):
        plan = [(66, "learning", f"rotation{r}_period{b + 1}", "unknown", "rotated", "unspecified") for b, r in enumerate(rots)]
        add_series("Exp1_trials", grp, j, c, None, plan, setting="lab", device="robot", hand_measure="peak_velocity",
                   rt_scale=1.0, unit_label="trial", break_at=(66, 132, 198))

# ---------------------------------------------------------------- Experiments 2-5: epochs
def rt_cols(mm, key, shape):
    return columns(mm, key) if key in mm else [None] * shape


def baseline_plan(nb, rot_epochs, nf_epochs, nf_strategy, rot):
    plan = [(nb, "baseline", "null", "unknown", "veridical", "unspecified"),
            (rot_epochs, "learning", f"rotation{rot}", "unknown", "rotated", "unspecified")]
    if nf_epochs:
        plan.append((nf_epochs, "aftereffect", "no_feedback", "none", "none", nf_strategy))
    return plan


m = mat("figure3.zip", "data-experiment-2-no-PT-limit.mat")
for j, (a, rt) in enumerate(zip(columns(m, "A"), columns(m, "RT"))):
    plan = baseline_plan(10, 60, 0, "", 30) + [(1, "probe", "implicit_probe_no_feedback", "none", "none", "remove")]
    add_series("Exp2_epochs", "no_PT_limit", j, a, rt, plan, setting="lab", device="robot", hand_measure="fixed_distance",
               rt_scale=1000.0, unit_label="epoch")
m = mat("figure3.zip", "data-experiment-2-limit-PT.mat")
for j, (a, rt) in enumerate(zip(columns(m, "A"), columns(m, "RT"))):
    add_series("Exp2_epochs", "limit_PT", j, a, rt, baseline_plan(10, 60, 15, "unspecified", 30), setting="lab", device="robot",
               hand_measure="fixed_distance", rt_scale=1000.0, unit_label="epoch")
m = mat("figure3.zip", "data-experiment-3-no-PT-limit.mat")
for j, (a, rt) in enumerate(zip(columns(m, "A"), columns(m, "RT"))):
    add_series("Exp3_epochs", "no_PT_limit", j, a, rt, baseline_plan(10, 60, 20, "remove", 30), setting="online", device="trackpad",
               hand_measure="other", rt_scale=1000.0, unit_label="epoch")
m3 = mat("figure3.zip", "data-experiment-3-limit-PT.mat")
for j, (a, rt) in enumerate(zip(columns(m3, "A"), columns(m3, "RT"))):
    add_series("Exp3_epochs", "limit_PT", j, a, rt, baseline_plan(10, 60, 20, "unspecified", 30), setting="online",
               device="trackpad", hand_measure="other", rt_scale=1000.0, unit_label="epoch")
m = mat("figure8-supplement1.zip", "data-experiment-3-decay-only.mat")
for j, (a, rt) in enumerate(zip(columns(m, "A"), columns(m, "RT"))):
    add_series("Exp3_epochs", "decay_only", j, a, rt, baseline_plan(10, 60, 20, "maintain", 30), setting="online",
               device="trackpad", hand_measure="other", rt_scale=1000.0, unit_label="epoch")
mdup = mat("figure8-supplement1.zip", "data-experiment-3-limit-PT.mat")
checks.internal("Experiment 3 limit-PT matrix in figure8-supplement1 equals the one in figure3 (same participants)",
                np.array_equal(np.nan_to_num(mdup["A"], nan=-999), np.nan_to_num(m3["A"], nan=-999)))

m = mat("figure8.zip", "data-experiment-4.mat")
for j, (a, rt) in enumerate(zip(columns(m, "A"), columns(m, "RT"))):  # RT already in ms in this file
    plan = [(10, "baseline", "null", "unknown", "veridical", "unspecified"),
            (60, "learning", "rotation30_exposure1", "unknown", "rotated", "unspecified"),
            (70, "washout", "washout", "unknown", "veridical", "unspecified"),
            (60, "relearning", "rotation30_exposure2", "unknown", "rotated", "unspecified")]
    add_series("Exp4_epochs", "all", j, a, rt, plan, setting="", device="unknown", hand_measure="other", rt_scale=1.0,
               unit_label="epoch")

m = mat("figure9.zip", "data-experiment-5.mat")
for grp, ka, kb, ra, rb, gap in (("gap_5min", "A5A", "A5B", "RT5A", "RT5B", None), ("gap_24h", "A24A", "A24B", "RT24A", "RT24B", 1)):
    A, B, RA, RB = columns(m, ka), columns(m, kb), columns(m, ra), columns(m, rb)
    for j in range(len(A)):
        a = np.concatenate([A[j], B[j]])
        rt = np.concatenate([RA[j], RB[j]])
        plan = [(5, "baseline", "null", "unknown", "veridical", "unspecified"),
                (80, "learning", "rotationA", "unknown", "rotated", "unspecified"),
                (80, "relearning", "rotationB_opposite", "unknown", "rotated", "unspecified")]
        add_series("Exp5_epochs", grp, j, a, rt, plan, setting="", device="unknown", hand_measure="other", rt_scale=1000.0,
                   unit_label="epoch", session_of=(lambda t: 1 if t >= 85 else 0) if gap else None, break_at=(85,))

# ---------------------------------------------------------------- checks
by_exp = OrderedDict()
for r in rows:
    by_exp.setdefault((r["experiment"], r["group"]), set()).add(r["subject"])
n = {k: len(v) for k, v in by_exp.items()}
for (e, g), want, src in ((("Exp1_trials", "abrupt"), 36, "Experiment 1, Participants"),
                          (("Exp1_trials", "stepwise"), 37, "Experiment 1, Participants"),
                          (("Exp2_epochs", "no_PT_limit"), 9, "Experiment 2, Participants"),
                          (("Exp2_epochs", "limit_PT"), 13, "Experiment 2, Participants"),
                          (("Exp3_epochs", "no_PT_limit"), 35, "Results, Experiment 3"),
                          (("Exp3_epochs", "limit_PT"), 21, "Results, Experiment 3"),
                          (("Exp3_epochs", "decay_only"), 12, "Figure 8 supplement 1"),
                          (("Exp4_epochs", "all"), 10, "Experiment 4, Participants"),
                          (("Exp5_epochs", "gap_5min"), 9, "Experiment 5, Participants"),
                          (("Exp5_epochs", "gap_24h"), 11, "Experiment 5, Participants")):
    checks.published(f"{e} {g}: participants", n[(e, g)], want, 0, f"{PAPER} ({src})")
checks.published("Experiment 1: trials per participant (4 learning periods of 66 trials)",
                 sum(1 for r in rows if r["subject"] == f"{ID}_Exp1_trials_abrupt_0"), 264, 0, PAPER + ", Experiment 1")
checks.published("Experiment 4: epochs per participant (10 + 60 + 70 + 60)",
                 sum(1 for r in rows if r["subject"] == f"{ID}_Exp4_epochs_all_0"), 200, 0, PAPER + ", Experiment 4")
checks.published("Experiment 5: 5 min group has 9 and 24 h group 11 participants (total)",
                 n[("Exp5_epochs", "gap_5min")] + n[("Exp5_epochs", "gap_24h")], 20, 0, PAPER + ", Experiment 5")
checks.internal("hand_angle finite and within +-90 degrees on every filled row",
                all(abs(float(r["hand_angle"])) <= 90 for r in rows if r["hand_angle"]))
checks.internal("every participant column of the used matrices is kept", sum(n.values()) == 36 + 37 + 9 + 13 + 35 + 21 + 12 + 10 + 9 + 11)

for c in checks.items:
    print("CHK", "ok " if c["ok"] else "BAD", c["what"], c.get("computed", ""), c.get("reported", ""))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf8")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID,
    paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Shadmehr (Johns Hopkins)",
    data_url="https://osf.io/mzs6a/",
    license="none stated on the OSF deposit (the eLife article is CC BY 4.0)",
    license_url="",
    changes="converted to the OpenMotor format, participant columns replaced by codes, and the authors' per-figure matrices "
            "stacked into one table; reaction times converted to ms; nothing was recomputed.",
    unit_note="Experiment 1 rows are single trials (trial_kind reach); every other experiment's rows are EPOCH means (4 trials per epoch in "
              "Experiments 2-4, 8 in Experiment 5), numbered from 0 within the participant",
    how=[
        "Experiment 1 (36 abrupt, 37 stepwise participants, robot, 3 targets, epochs of 3 trials) has one row per trial: 4 "
        "learning periods of 66 trials with a 60 deg rotation (stepwise: 15, 30, 45 and 60 deg). The paper's exclusion "
        "probes are not identifiable in these matrices, so the rows are the 264 feedback trials (a dip near trial 46 of "
        "each period suggests the probes sit between trials 45 and 46).",
        "Experiments 2-5 are released as one row per EPOCH, because the deposit holds epoch means only (each epoch visits every "
        "target once: 4 targets in Experiments 2-4, 8 in Experiment 5). hand_angle on those rows is the epoch mean, trial is "
        "the epoch index, and target_angle is left empty. Experiment 2: 10 null + 60 rotation epochs (+ one no-feedback, "
        "no-aiming probe row in the no-limit group; 15 no-feedback epochs in the limit-PT group). Experiment 3 (laptop, "
        "remote): 10 + 60 + 20 no-feedback epochs; the decay-only group was told to keep moving the imagined cursor through "
        "the target. Experiment 4: 10 null, 60 rotation, 70 washout and 60 rotation epochs with a reaction-time limit on "
        "every trial. Experiment 5: 5 null + 80 epochs of rotation A, then 80 epochs of the opposite rotation B after a "
        "5 minute or 24 hour gap (session 1 for B in the 24 h group).",
        "Epoch rows are marked trial_kind = other (not reach), so that an analyst who keeps only trial_kind = reach gets the "
        "single-trial Experiment 1 and never mixes epoch means into trial-level analyses; they stay in this reaching format "
        "(rather than the summary module, which is long-form per-participant values) because they are a learning curve with a "
        "fixed epoch order, one row per participant and epoch.",
        "Sign (audited 2026-11-10 and left as the authors stored it): the eLife Methods and the authors' figure scripts do not state "
        "whether the rotations of Experiments 1-5 were clockwise or counter-clockwise (the only directions named are for other "
        "labs' re-analysed data), so the stored sign cannot be tied to the counter-clockwise convention. In the paper's figures "
        "compensation is plotted upward for Experiments 1-3 and rotation A of Experiment 5. Neither the paper nor the deposit states the direction of the rotations, and the matrices are in different "
        "signs across experiments (compensation positive in Experiments 1, 2, 3 and rotation A of 5; negative in Experiment "
        "4 and rotation B of 5, as the authors' scripts flip them for plotting). hand_angle is the authors' stored value and "
        "is NOT verified as counter-clockwise positive; rotation and cursor columns are left empty and the rotation size is "
        "in the block label. rt_ms is the authors' reaction (preparation) time, converted from seconds (Experiments 2, 3, 5) "
        "or already ms (Experiment 4). Trials or epochs the authors stored as NaN are trial_kind missing and excluded.",
        "Left out: the figure ZIPs that hold only code, simulations or model fits; the re-analysed data of other labs "
        "(Tsay et al., Neville and Cressman, Maresch et al., Haith et al., Salomonczyk et al., McDougle et al., Mazzoni and "
        "Krakauer, Taylor and Ivry; figures 1, 2, 4, 5, 6, 10 and their supplements, mostly means and standard errors, and "
        "per-participant means in figure1-supplement2 and figure5-supplement1) because they are other groups' data "
        "without a licence for a converted copy; and the per-participant explicit and implicit angles (E, I, T vectors) "
        "because they are summaries, not series.",
    ],
    differences=[
        "Experiment 2 and 3 group sizes and the Experiment 4 and 5 counts agree with the Methods; the per-epoch values were "
        "not compared with the paper's reported means because the paper reports them only in figures.",
    ],
)
stdlib.release(ID, "reach90", rows, meta, checks)
