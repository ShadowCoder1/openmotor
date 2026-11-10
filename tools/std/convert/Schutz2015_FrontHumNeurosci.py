"""Schutz & Souto (2015) Front Hum Neurosci 9:566: a perceptual task induces saccadic adaptation by target selection.

Raw file: raw_extra/Schutz2015_FrontHumNeurosci/data.zip (Zenodo 579835, SchutzSouto_FrontHumNeurosci_2015.zip): one CSV per
participant and condition without header (subj_<n>_exp_<k>_<condition>.csv, 700 trials each, 54 columns, described in
Readme_exp.txt) and subj_<n>_baseline.csv (fixation control, 32 columns, Readme_baseline.txt). Nothing in the deposit is run.

Run from the site folder: python3 -I tools/std/convert/Schutz2015_FrontHumNeurosci.py
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Schutz2015_FrontHumNeurosci"
MODULE = "saccade"

EXP_NAME = {1: "experiment 1 (easy vs difficult gap)", 2: "experiment 2 (valid vs random feedback)",
            3: "experiment 3 (target selection)", 4: "experiment 4 (compound without task)"}
TYPE_BLOCK = {1: "pre-adaptation", 2: "adaptation", 3: "post-adaptation"}
TYPE_PHASE = {1: "baseline", 2: "learning", 3: "aftereffect"}
OPENING = {1: "right", 2: "up", 3: "left", 4: "down"}


def fnum(x):
    return "" if x != x else f"{x:g}"


def main():
    zpath = stdlib.raw_dir(DS) / "data.zip"
    frames = []
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            m = re.fullmatch(r"subj_(\d+)_exp_(\d)_(\w+)\.csv", n)
            b = re.fullmatch(r"subj_(\d+)_baseline\.csv", n)
            if not (m or b):
                continue
            d = pd.read_csv(io.BytesIO(z.read(n)), header=None, na_values=["NaN"])
            d["subj"] = int((m or b)[1])
            d["expno"] = int(m[2]) if m else 0
            d["cond"] = m[3] if m else "baseline"
            frames.append(d)
    ex = pd.concat([f for f in frames if f["expno"].iloc[0] > 0], ignore_index=True)
    bl = pd.concat([f for f in frames if f["expno"].iloc[0] == 0], ignore_index=True)
    ck = stdlib.Checks()

    # ---- structure ---------------------------------------------------------------------------------------------------
    ck.internal("experiment files have 54 data columns, baseline files 32",
                all(f.shape[1] == 54 + 3 for f in frames if f["expno"].iloc[0] > 0)
                and all(f.shape[1] == 32 + 3 for f in frames if f["expno"].iloc[0] == 0))
    per = ex.groupby(["expno", "cond", "subj"]).size()
    ck.internal("every experiment file has 700 trials (100 pre-adaptation, 300 adaptation, 300 post-adaptation)",
                bool((per == 700).all()) and len(per) == 58)
    ck.internal("trial numbers run 1-700 in order inside every file",
                bool(ex.groupby(["expno", "cond", "subj"])[1].apply(lambda s: (s.to_numpy() == np.arange(1, 701)).all()).all()))
    ck.internal("trial type follows position: 1-100 pre, 101-400 adaptation, 401-700 post",
                bool((ex[4].to_numpy() == np.where(ex[1] <= 100, 1, np.where(ex[1] <= 400, 2, 3))).all()))
    ck.internal("baseline files: 29 participants, every trial row kept", bl["subj"].nunique() == 29 and len(bl) == 5912,
                f"{bl['subj'].nunique()} participants, {len(bl)} rows")

    # ---- the black (task-relevant) square after the saccade ---------------------------------------------------------
    cols_after = ex[list(range(32, 39))].to_numpy()
    tidx = np.array([int(np.where(r == 2)[0][0]) + 1 if (r == 2).any() else 0 for r in cols_after])
    ty = ex[4].to_numpy()
    ck.internal("the black square is at the array centre (position 4) in every pre- and post-adaptation trial and at position 2 "
                "(above) or 6 (below) in every adaptation trial",
                bool((tidx[ty != 2] == 4).all() and np.isin(tidx[ty == 2], (2, 6)).all()))
    step_y = (4 - tidx) * 0.76          # 0.76 deg per square (baseline file: target vertical positions 0, 0.76, 1.52 deg)
    ck.internal("authors' 'vertical target position' column is 0 or the same signed 1.52 deg offset as the black square's position",
                bool(((ex[45] == 0) | np.isclose(ex[45], step_y, atol=0.01)).all()))
    ck.internal("target step: end minus start of the array position is 9 deg in every trial (direction-normalised frame)",
                bool(np.allclose(ex[6] - ex[5], 9.0, atol=1e-3)))
    ck.internal("gain equals horizontal amplitude divided by 9 deg",
                bool(np.allclose(ex[51].dropna(), (ex[48] / 9.0).loc[ex[51].dropna().index], atol=2e-3)))

    # ---- published numbers ------------------------------------------------------------------------------------------
    n_by_exp = ex.groupby("expno")["subj"].nunique().to_dict()
    for k, rep in {1: 10, 2: 10, 3: 9, 4: 9}.items():
        ck.published(f"experiment {k}: number of observers", n_by_exp[k], rep, 0, "Methods, Observers")
    ck.published("observers with data (all experiments)", ex["subj"].nunique(), 38, 0, "Methods, Observers (38 students)")
    ck.published("observers in the fixation baseline", bl["subj"].nunique(), 29, 0, "Methods, Design (fixation baseline n = 29)")
    rep_vert = {  # (experiment, condition): (upward, downward) mean vertical amplitude at the end of adaptation, deg
        (1, "easy"): (0.87, -1.03), (1, "difficult"): (1.06, -1.07), (2, "valid"): (1.02, -0.86), (2, "random"): (1.08, -1.04),
        (3, "target_selection"): (1.25, -1.16), (4, "no_instruction"): (0.50, 0.09)}
    v = ex[ex[53] == 1].copy()
    v["up"] = np.where(tidx[(ex[53] == 1).to_numpy()] == 2, 1, np.where(tidx[(ex[53] == 1).to_numpy()] == 6, -1, 0))
    ck.internal("black square above the array centre exactly when -(trial direction x adaptation direction) is +1",
                bool((v["up"][v[4] == 2] == np.sign(-v[7] * v[8])[v[4] == 2]).all()))
    for (e, c), (up_rep, dn_rep) in rep_vert.items():
        g = v[(v["expno"] == e) & (v["cond"] == c) & (v[4] == 2) & (v[1] > 350)]
        res = {}
        for u in (1, -1):
            res[u] = g[g["up"] == u].groupby("subj")[49].mean().mean()
        ck.published(f"experiment {e} {c}: mean vertical amplitude, upward adaptation, last 50 adaptation trials (deg)",
                     res[1], up_rep, 0.05, "Results, Vertical amplitude at the end of adaptation (Fig. 2)")
        ck.published(f"experiment {e} {c}: mean vertical amplitude, downward adaptation, last 50 adaptation trials (deg)",
                     res[-1], dn_rep, 0.05, "Results, Vertical amplitude at the end of adaptation (Fig. 2)")
    pct = 100.0 * (1 - ex[53].mean())
    differences = []
    if abs(pct - 9.5) <= 0.6:
        ck.published("percent of adaptation-experiment trials excluded", pct, 9.5, 0.6, "Methods, Eye-movement recording (about 9.5 %)")
    else:
        differences.append(f"The paper reports that about 9.5 % of trials were excluded; the authors' valid flag marks {pct:.1f} % of the experiment rows as invalid.")

    # ---- standardized rows ------------------------------------------------------------------------------------------
    def person(s):
        return f"P{int(s):03d}"
    # experiment files
    n = len(ex)
    dirn = ex[7].to_numpy()
    typ = ex[4].to_numpy().astype(int)
    adap = typ == 2
    resp = ex[9].to_numpy()
    info = [
        f"target_index={t};gap_proportion={fnum(g)};adaptation_direction={int(a)};fixation_s={fnum(f)};iti_s={fnum(i)};"
        f"correct_opening={int(co)};auditory_feedback={int(af)};vertical_target_position={fnum(vt)}" + (f";feedback_type={fnum(ft)}" if e == 2 else "")
        for t, g, a, f, i, co, af, ft, e, vt in zip(tidx, ex[46], ex[8], ex[2], ex[3], ex[40], ex[42], ex[41], ex["expno"], ex[45])]
    out = pd.DataFrame({
        "dataset": DS, "study": DS,
        "experiment": [EXP_NAME[e] for e in ex["expno"]], "group": "",
        "subject": [f"{e}_{s}" for e, s in zip(ex["expno"], ex["subj"])],
        "person": [person(s) for s in ex["subj"]],
        "subject_excluded_by_authors": False,
        "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
        "device": "EyeLink 1000 (right eye, 1000 Hz), VIEWPixx monitor (120 Hz) at 48.5 cm, keyboard",
        "session": "",
        "block": [TYPE_BLOCK[t] for t in typ],
        "trial": ex[1].astype(int),
        "phase": [TYPE_PHASE[t] for t in typ],
        "time_s": "",
        "task": "eye movements", "unit": "saccade",
        "condition": ex["cond"],
        "pert_type": np.where(adap, "task_relevant_offset", "none"),
        "pert_value": np.where(adap, step_y, 0.0),
        "pert_unit": "deg",
        "feedback": "normal",
        "excluded_by_authors": ex[53] == 0,
        "target_amplitude_deg": (ex[6] - ex[5]).abs(),
        "target_direction_deg": np.where(dirn == 1, 0.0, 180.0),
        "step_x_deg": 0.0,
        "step_y_deg": step_y,
        "step_delay_ms": "",
        "saccade_amplitude_x_deg": ex[48],
        "saccade_amplitude_y_deg": ex[49],
        "saccade_gain": ex[51],
        "saccade_latency_ms": ex[47],
        "saccade_duration_ms": ex[50],
        "saccade_peak_velocity_deg_s": ex[52],
        "eye": "right",
        "response": [OPENING.get(int(r), "") if r == r else "" for r in resp],
        "response_correct": [bool(b) if (r == r and r != 0) else "" for r, b in zip(resp, ex[10])],
        "trial_info": info,
    })
    # baseline files (fixation control, no saccade)
    bresp = bl[8].to_numpy()
    binfo = [f"target_position_in_array={int(p)};gap_proportion={fnum(g)};correct_opening={int(co)};auditory_feedback={int(af)};"
             f"fixation_s={fnum(f)}" for p, g, co, af, f in zip(bl[4], bl[30], bl[25], bl[27], bl[2])]
    bout = pd.DataFrame({
        "dataset": DS, "study": DS,
        "experiment": "fixation baseline", "group": "",
        "subject": [f"b_{s}" for s in bl["subj"]],
        "person": [person(s) for s in bl["subj"]],
        "subject_excluded_by_authors": False,
        "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
        "device": "EyeLink 1000 (right eye, 1000 Hz), VIEWPixx monitor (120 Hz) at 48.5 cm, keyboard",
        "session": "", "block": "fixation baseline",
        "trial": bl[1].astype(int), "phase": "other", "time_s": "",
        "task": "eye movements", "unit": "saccade", "condition": "baseline",
        "pert_type": "none", "pert_value": 0.0, "pert_unit": "deg", "feedback": "normal",
        "excluded_by_authors": bl[31] == 1,
        "target_amplitude_deg": "", "target_direction_deg": "",
        "eye": "right",
        "response": [OPENING.get(int(r), "") if r == r else "" for r in bresp],
        "response_correct": [bool(b) if (r == r and r != 0) else "" for r, b in zip(bresp, bl[9])],
        "trial_info": binfo,
    })
    allrows = pd.concat([out, bout], ignore_index=True)
    ck.internal("every row the authors shared is kept: 40600 experiment rows and 5912 baseline rows", len(out) == 40600 and len(bout) == 5912)
    ck.internal("ranges: latency -1500 to 2100 ms, |horizontal amplitude| < 40 deg, |vertical amplitude| < 25 deg",
                bool(ex[47].dropna().between(-1500, 2100).all() and ex[48].abs().max() < 40 and ex[49].abs().max() < 25))

    if not differences:
        differences = None
    meta = dict(
        study=DS,
        paper=dict(authors="Schutz AC, Souto D", year=2015, title="Perceptual task induces saccadic adaptation by target selection",
                   venue="Front Hum Neurosci 9:566", doi="10.3389/fnhum.2015.00566"),
        lab="Schutz (University of Giessen)",
        data_url="https://doi.org/10.5281/zenodo.579835",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Combined the authors' per-participant CSV files (no header) into one table with named columns, participant codes "
                 "and the OpenMotor trial, phase and exclusion fields. The authors' row order, values and valid flags are kept; the "
                 "seven opening and colour codes of the stimulus array before the saccade and the remaining stimulus columns are "
                 "reduced to the position of the black square and a few protocol values in trial_info."),
        how=["One row is one trial of one participant in one condition (700 trials per file: 100 pre-adaptation, 300 adaptation, "
             "300 post-adaptation) with the primary saccade to a vertical array of squares 9 deg to the left or right; the "
             "fixation control files add 'fixation baseline' rows with no saccade (response only).",
             "The array is shown after saccade onset and the black square is the one to judge; in the adaptation phase it sits "
             "two squares above or below the array centre (step_y_deg and pert_value, +1.52 or -1.52 deg, positive up, from the position of the black square at 0.76 deg per square), in the "
             "other phases at the centre (0).",
             "Horizontal saccade amplitude is the authors' value, which is positive in the direction of the trial (target step "
             "9 deg, gain = amplitude / 9); vertical amplitude is positive upward. Target positions in the deposit are in a "
             "direction-normalised frame (end minus start is +9 deg for every trial), so target_direction_deg is taken from the "
             "authors' trial-direction column (0 = rightward, 180 = leftward).",
             "Units follow the paper (degrees of visual angle, ms, deg/s; the authors' readme gives '?'). Experiment 3's "
             "perceptual-task condition is not in the deposit."],
        unit_note="trial (one primary saccade)",
        differences=differences,
    )
    for c in stdlib.columns_for(MODULE):
        if c[1] not in allrows.columns:
            allrows[c[1]] = ""
    stdlib.release(DS, MODULE, allrows, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
