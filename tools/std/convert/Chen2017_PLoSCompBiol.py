"""Chen, Mohr & Galea (2017), PLoS Comput Biol: explorative motor learning. Participants slid the index finger from a start square
across a table towards a 15 cm line to find 24 hidden trajectories (direction alpha, curvature beta); each valid attempt got a score.
Raw (untrusted, read only): raw_extra/Chen2017_PLoSCompBiol/data.zip (OSF bafms): Experiment1Data/DataRaw/Data_Motor/Main_p<s>_<s>_<t>_trial.mat
(Trial matrix, one row per attempt) and Experiment2/data_Reaching/Main_p_<s>_<t>_trial.mat. Columns (the authors' README and load_motor_data.m):
1 attempt, 2 condition (1 reward, 2 punishment), 6-7 target alpha, beta, 8 score, 9-10 attempt alpha, beta, 11 invalid (too fast or slow).
Columns 3-5 are not documented and are not released. Finger trajectories (Main_p<s>_<t>.mat), decision-making and motor-noise tasks are not released."""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Chen2017_PLoSCompBiol"
z = zipfile.ZipFile(stdlib.raw_dir(ID) / "data.zip")
P = "ExplMotorLearn_DecisionMaking/"
checks = stdlib.Checks()
files = {}
for i in z.infolist():
    m = re.fullmatch(P + r"Experiment1Data/DataRaw/Data_Motor/Main_p(\d+)_\1_(\d+)_trial\.mat", i.filename)
    if m:
        files[("Experiment1", int(m.group(1)), int(m.group(2)))] = i.filename
    m = re.fullmatch(P + r"Experiment2/data_Reaching/Main_p_(\d+)_(\d+)_trial\.mat", i.filename)
    if m:
        files[("Experiment2", int(m.group(1)), int(m.group(2)))] = i.filename
VALID1 = [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]  # the authors' README
EXCL_IDX = [10, 16, 18, 22]  # Fig 2G outliers, numbered 1..24 within the valid participants (ID 16 is not in the data)
EXCL = {VALID1[k - 1] for k in EXCL_IDX}
allsub = sorted({(e, s) for e, s, t in files})
SUB_RAW1 = [s for e, s in allsub if e == "Experiment1"]
subjects = [(e, s) for e, s in allsub if e == "Experiment2" or s in VALID1]  # IDs 1 and 5 are machine failures, 16 did not finish (README)
checks.internal("raw Experiment 1 folder has the 24 valid IDs plus the machine-failure IDs 1 and 5", sorted(set(SUB_RAW1) - set(VALID1)) == [1, 5], str(SUB_RAW1))

rows = []
meanbeta = {}
inv = []
for e, s in subjects:
    seq, n = [], 0
    tars = sorted(t for ee, ss, t in files if (ee, ss) == (e, s))
    ncond = {1: 0, 2: 0}
    for t in tars:
        tr = sio.loadmat(io.BytesIO(z.read(files[(e, s, t)])), squeeze_me=True)["Trial"]
        tr = np.atleast_2d(tr)
        assert tr.shape[1] == 11 and len(set(tr[:, 1])) == 1 and len(set(tr[:, 5])) == 1 and len(set(tr[:, 6])) == 1
        cond = int(tr[0, 1])
        ncond[cond] += 1
        assert list(tr[:, 0]) == list(range(1, len(tr) + 1)), (e, s, t)
        for r in tr:
            n += 1
            excl = bool(r[10] == 1)
            rows.append(dict(
                dataset=ID, study=ID, experiment=e, group="", subject=f"{e}_{s}", person=f"{e}_{s}",
                subject_excluded_by_authors=(e == "Experiment1" and s in EXCL), age="", sex="", handedness="",
                population="healthy", setting="lab", device="finger tracking (Polhemus Fastrak), mirror display", session=1,
                block=f"target_{t}", trial=n, phase="learning", time_s="", task="reaching", unit="trial",
                condition="positive_feedback" if cond == 1 else "negative_feedback", pert_type="none", pert_value=0, pert_unit="",
                feedback="normal", excluded_by_authors=excl, search_target=t, attempt=int(r[0]), target_alpha=float(r[5]),
                target_beta=float(r[6]), attempt_alpha=float(r[8]), attempt_beta=float(r[9]), score=float(r[7])))
            inv.append(excl)
    if e == "Experiment1":
        checks.internal(f"{e} {s}: 12 targets per feedback condition", ncond == {1: 12, 2: 12}, str(ncond))
    else:
        checks.internal(f"{e} {s}: 12 targets, all in the reward condition", ncond == {1: 12, 2: 0}, str(ncond))
    meanbeta[(e, s)] = np.mean([abs(r["attempt_beta"]) for r in rows if r["subject"] == f"{e}_{s}" and not r["excluded_by_authors"]])

# the paper's outlier rule on mean |curvature| per participant (Fig 2G)
mb = np.array([meanbeta[("Experiment1", s)] for s in VALID1])
q1, q3 = np.percentile(mb, [25, 75], method="hazen")  # MATLAB prctile
out = [VALID1[i] for i in range(24) if mb[i] > q3 + 1.5 * (q3 - q1) or mb[i] < q1 - 1.5 * (q3 - q1)]
S1 = "Methods, Ethics statement and subjects; Experiment 1 procedure"
e1 = [r for r in rows if r["experiment"] == "Experiment1"]
e2 = [r for r in rows if r["experiment"] == "Experiment2"]
checks.published("Experiment 1 participants (26 took part, 1 withdrew, 1 equipment fault)", len({r["subject"] for r in e1}), 24, 0, S1)
checks.published("Experiment 2 participants (six new individuals)", len({r["subject"] for r in e2}), 6, 0, "Methods, Ethics statement and subjects")
checks.published("hidden target trajectories per Experiment 1 participant", len({(r["subject"], r["search_target"]) for r in e1}) / 24, 24, 0, S1)
checks.published("valid attempts allowed per target: most valid attempts on any target", max(
    sum(1 for r in e1 if r["subject"] == sj and r["search_target"] == t and not r["excluded_by_authors"]) for sj in {r["subject"] for r in e1} for t in range(1, 25)), 25, 0, S1)
for cond, expect in (("positive_feedback", 12), ("negative_feedback", 12)):
    k = {(r["subject"], r["search_target"]) for r in e1 if r["condition"] == cond}
    checks.published(f"targets per participant in the {cond} condition", len(k) / 24, expect, 0, S1)
checks.published("participants flagged as curvature outliers by the paper's rule (Fig 2G boxplot rule: beyond 1.5 IQR, valid attempts)", len(out), 4, 0, "Results, Fig 2G (4 of 24 participants)")
checks.published("score range: highest score in the positive condition", max(r["score"] for r in rows if r["condition"] == "positive_feedback"), 50, 0, "Methods, Eq 2")
checks.published("score range: lowest score in the negative condition", min(r["score"] for r in rows if r["condition"] == "negative_feedback"), -50, 0, "Methods, Eq 2")
checks.internal("rule outliers are the participants the authors name (numbers 10, 16, 18, 22 of the 24)", sorted(out) == sorted(EXCL), f"{out} vs {sorted(EXCL)}")

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Galea (University of Birmingham)", data_url="https://osf.io/bafms/", license="none stated", license_url="",
    changes="converted to the OpenMotor format; participant codes replaced; the finger trajectories, the decision-making task, the motor-noise "
            "task and three undocumented columns of the authors' trial matrix are not included.",
    unit_note="attempt (one drawn trajectory) of one participant",
    how=["One row is one attempt in an explorative motor learning task: the participant slid the index finger from a start square across a table to a "
         "target line 15 cm away, trying to find a hidden target trajectory x = alpha y + beta sin(pi y) (alpha direction, beta curvature); the "
         "attempt was fitted with the same two parameters and scored 0 to 50 (positive_feedback) or -50 to 0 (negative_feedback). Each participant "
         "searched for 24 hidden trajectories (12 per feedback condition), up to 25 valid attempts each. There is no hand direction relative to a "
         "visible target, so the 90-column reaching format is not used; the columns are in the trajectory_search module (target and attempt alpha and beta, score).",
         "Rows keep the authors' order within each target (attempt number) and the targets in the order of their file numbers, which is probably the "
         "order of testing (the target parameters differ between participants at the same number, but the files hold no clock times). Invalid attempts "
         "(movement under 700 ms or over 1500 ms, no feedback shown, the trial repeated) are kept and flagged excluded_by_authors; the authors' own "
         "preparation script removes them. Scores are as stored in the raw files (negative in the negative condition); the authors' prepared file adds 50 to them.",
         "Experiment1 is the 24 participants the authors list as valid. Four of them (numbers 10, 16, 18, 22 among the 24) are flagged as excluded by the "
         "authors because the paper removed them as outliers in curvature exploration; the paper's rule applied to the converted rows picks the same four, "
         "so the numbers refer to the position among the 24, not to the original IDs (inferred). Experiment2 is six new participants who did the "
         "same reaching task (12 targets each in the deposit, all in the positive feedback condition); no age or sex is shared per participant. "
         "The deposit's raw Experiment 1 folder also holds participants 1 and 5 (machine failure, per the authors' README); they are left out, as in the authors' prepared data."],
    differences=["The paper says the curvature outliers (Fig 2G) were found with a limit of q3 + 0.15 (q3 - q1); the authors' figure is a MATLAB boxplot, whose default limit is 1.5 (q3 - q1). "
                 "With 1.5 and the mean absolute curvature of valid attempts, four participants are flagged, the same four the paper names; 0.15 would flag ten.",
                 "The deposit's Experiment 2 reaching folder holds 12 targets per participant, while the paper describes the same reaching task as in Experiment 1; "
                 "all rows present are kept.",
                 "Mean age 26.46 (SD 5.96), 17 women and 25 right-handed refer to all 32 participants in the paper and are not available per participant."],
)
stdlib.release(ID, "trajectory_search", rows, meta, checks)
