"""Schutz, Kerzel & Souto (2014) J Vis 14(5):4: saccadic adaptation induced by a perceptual task.

Raw file: raw_extra/Schutz2014_JVis/data.zip (Zenodo 579842, SchutzKerzelSouto_JoV_2014.zip): exp_<k>_subj_<n>.csv, one CSV
without header per experiment and participant (600 or 700 trials, 49 columns described in Readme_exp.txt). The paper was read
from the VisuoMotor Lab's PDF copy (davidsouto.github.io/wp) for the design and the numbers checked below. Nothing in the
deposit is run.

Run from the site folder: python3 -I tools/std/convert/Schutz2014_JVis.py
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

DS = "Schutz2014_JVis"
MODULE = "saccade"

EXP_NAME = {1: "experiment 1 (on-axis, perceptual task)", 2: "experiment 2 (on-axis, no task)",
            3: "experiment 3 (classical adaptation with perceptual task)", 4: "experiment 4 (cross-axis, perceptual task)",
            5: "experiment 5 (cross-axis, no task)", 6: "experiment 6 (transfer to reactive and voluntary saccades)",
            7: "experiment 7 (transfer control, blocked)", 8: "experiment 8 (presaccadic cue control)"}
CHAR = {1: "8", 2: "2", 3: "3", 4: "E", 5: "5"}
N_SUB = {1: 9, 2: 9, 3: 9, 4: 11, 5: 11, 6: 10, 7: 10, 8: 9}
ARRAY_CHAR_DEG = 0.7     # one character step of the horizontal array: central and eccentric locations are 2 characters = 1.4 deg apart (Methods, Classical adaptation)


def fnum(x):
    return "" if x != x else f"{x:g}"


def main():
    zpath = stdlib.raw_dir(DS) / "data.zip"
    frames = []
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            m = re.fullmatch(r"exp_(\d)_subj_(\d+)\.csv", n)
            if not m:
                continue
            d = pd.read_csv(io.BytesIO(z.read(n)), header=None, na_values=["NaN"])
            d["expno"], d["subj"] = int(m[1]), int(m[2])
            frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    ck = stdlib.Checks()
    ck.internal("every file has 49 data columns", all(f.shape[1] == 51 for f in frames))
    key = ["expno", "subj"]
    per = D.groupby(key).size()
    ck.internal("each file has 600 (experiments 6, 7) or 700 trials; 78 files",
                len(per) == 78 and bool((per.groupby(level=0).first().to_dict() == {1: 700, 2: 700, 3: 700, 4: 700, 5: 700, 6: 600, 7: 600, 8: 700})))
    ck.internal("trial numbers run 1..N in order inside every file",
                bool(D.groupby(key)[1].apply(lambda s: (s.to_numpy() == np.arange(1, len(s) + 1)).all()).all()))

    # ---- stimulus coding ---------------------------------------------------------------------------------------------
    cols_after = D[list(range(33, 40))].to_numpy()
    chars_after = D[list(range(26, 33))].to_numpy()
    tidx = np.array([int(np.where(r == 2)[0][0]) + 1 if (r == 2).any() else 0 for r in cols_after])   # position of the black character
    tchar = np.array([CHAR.get(int(c[i - 1]), "") if i else "" for c, i in zip(chars_after, tidx)])
    expn = D["expno"].to_numpy()
    typ = D[4].to_numpy().astype(int)
    vertical = np.isin(expn, (4, 5))
    centre = np.where(vertical, 3, 4)
    off = np.where(tidx > 0, tidx - centre, 0)             # characters from the array centre (authors' index order)
    dirn = D[8].to_numpy()
    adir = D[9].to_numpy()
    ck.internal("black character at the array centre in pre/post-adaptation trials and one (vertical) or two (horizontal) "
                "positions away in adaptation trials of experiments 1, 2, 4-8",
                bool((off[np.isin(expn, (1, 2, 4, 5, 6, 7, 8)) & np.isin(typ, (1, 3))] == 0).all()
                     and np.isin(off[np.isin(expn, (1, 2, 8)) & (typ == 2)], (-2, 2)).all()
                     and np.isin(off[vertical & (typ == 2)], (-1, 1)).all()))
    shift = (D[7] - D[6]).to_numpy()
    ck.internal("array shift during the saccade (experiment 3) is 1.4 deg (1.409-1.410 in the file) in every adaptation trial and 0 elsewhere",
                bool(np.abs(np.abs(shift[(expn == 3) & (typ == 2)]) - 1.41).max() < 0.005 and (np.abs(shift[~((expn == 3) & (typ == 2))]) < 1e-6).all()))
    ck.internal("trial direction equals the sign of end minus start of the array position in every trial",
                bool((np.sign(D[6] - D[5]).to_numpy() == dirn).all()))
    first_adapt = D.assign(t2=(typ == 2)).groupby(key)["t2"].transform(lambda s: s.cumsum() > 0).to_numpy()

    # ---- rows ----------------------------------------------------------------------------------------------------------
    horizontal = ~vertical
    fwd_off = np.where(horizontal, -off * ARRAY_CHAR_DEG * dirn, np.nan)   # physical x offset of the black character (deg)
    step_x = np.where(expn == 3, np.where(typ == 2, shift, 0.0), np.where(tidx > 0, fwd_off, np.nan))
    step_y = np.where(expn == 3, 0.0, np.where(vertical, np.nan, 0.0))
    adap = typ == 2
    pt = np.where(expn == 3, np.where(adap, "intrasaccadic_step", "none"), np.where(adap, "task_relevant_offset", "none"))
    pv = np.where(adap, np.where(vertical, -off.astype(float), step_x), 0.0)
    pu = np.where(vertical & adap, "array positions (positive up)", "deg")

    def block(e, t):
        if e in (6, 7):
            return {1: "perceptual task, pre-adaptation", 2: "perceptual task, adaptation", 3: "no task, reactive", 4: "no task, voluntary"}[t]
        return {1: "pre-adaptation", 2: "adaptation", 3: "post-adaptation"}[t]

    def phase(e, t, fa):
        if e in (6, 7):
            return {1: "baseline", 2: "learning"}.get(t, "transfer" if fa else "baseline")
        return {1: "baseline", 2: "learning", 3: "aftereffect"}[t]

    def cond(e, t):
        if e in (6, 7):
            return {1: "perceptual task", 2: "perceptual task", 3: "reactive saccade, no task", 4: "voluntary saccade, no task"}[t]
        return {1: "perceptual task", 2: "no task", 3: "classical adaptation, perceptual task", 4: "cross-axis, perceptual task",
                5: "cross-axis, no task", 8: "perceptual task, black presaccadic mask"}[e]

    resp = D[10].to_numpy()
    binary = D[11].to_numpy()
    info = [f"target_index={t};target_character={c};adaptation_direction={int(a)};curvature={fnum(cu)};array_start_x={fnum(a5)};"
            f"array_end_x={fnum(a6)};array_adapt_x={fnum(a7)};fixation_s={fnum(f)};response_interval_s={fnum(ri)}"
            for t, c, a, cu, a5, a6, a7, f, ri in zip(tidx, tchar, adir, D[47], D[5], D[6], D[7], D[2], D[3])]
    out = pd.DataFrame({
        "dataset": DS, "study": DS,
        "experiment": [EXP_NAME[e] for e in expn], "group": "",
        "subject": [f"{e}_{s}" for e, s in zip(expn, D["subj"])],
        "person": [f"P{int(s):03d}" for s in D["subj"]],
        "subject_excluded_by_authors": False,
        "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
        "device": "EyeLink 1000 (right eye, 1000 Hz), Sony GDM-F520 CRT (100 Hz) at 47 cm, keyboard",
        "session": "",
        "block": [block(e, t) for e, t in zip(expn, typ)],
        "trial": D[1].astype(int),
        "phase": [phase(e, t, fa) for e, t, fa in zip(expn, typ, first_adapt)],
        "time_s": "", "task": "eye movements", "unit": "saccade",
        "condition": [cond(e, t) for e, t in zip(expn, typ)],
        "pert_type": pt, "pert_value": pv, "pert_unit": pu,
        "feedback": np.where((expn == 3) & adap, "altered", "normal"),
        "excluded_by_authors": D[48] == 0,
        "target_amplitude_deg": 9.0,
        "target_direction_deg": np.where(dirn == 1, 0.0, 180.0),
        "step_x_deg": step_x,
        "step_y_deg": step_y,
        "step_delay_ms": [0.0 if e == 3 else "" for e in expn],
        "saccade_amplitude_x_deg": D[42],
        "saccade_amplitude_y_deg": D[43],
        "saccade_gain": D[45],
        "saccade_latency_ms": D[41],
        "saccade_duration_ms": D[44],
        "saccade_peak_velocity_deg_s": D[46],
        "eye": "right",
        "response": [CHAR.get(int(r), "") if r == r else "" for r in resp],
        "response_correct": [True if b == 1 else (False if b == -1 else "") for b in binary],
        "trial_info": info,
    })
    out.loc[~(np.isin(expn, (1, 2, 3, 6, 7, 8)) & (tidx > 0)) & (expn != 3), "step_x_deg"] = np.nan
    out.loc[expn == 3, "step_y_deg"] = 0.0
    out.loc[(np.isin(expn, (1, 2, 6, 7, 8))) & (tidx == 0), "step_y_deg"] = np.nan
    out.loc[(np.isin(expn, (1, 2, 6, 7, 8))) & (tidx > 0), "step_y_deg"] = 0.0
    out.loc[vertical, "step_x_deg"] = 0.0
    ck.internal("every row the authors shared is kept (52600)", len(out) == 52600)
    ck.internal("ranges: |horizontal amplitude| < 40 deg, |vertical amplitude| < 25 deg, gain < 5",
                bool(D[42].abs().max() < 40 and D[43].abs().max() < 25 and D[45].max() < 5))

    # ---- published numbers ------------------------------------------------------------------------------------------
    nobs = D.groupby("expno")["subj"].nunique().to_dict()
    for e, r in N_SUB.items():
        ck.published(f"experiment {e}: number of observers", nobs[e], r, 0, "Methods, Experiments")
    cnt = D[D.expno == 1].groupby(["subj", 4]).size().unstack()
    ck.published("experiment 1: pre-adaptation / adaptation / post-adaptation trials per observer (first, adaptation, post)",
                 float(cnt.iloc[:, 0].mean() + cnt.iloc[:, 1].mean() + cnt.iloc[:, 2].mean()), 700, 0,
                 "Methods, Perceptual task and Fig. 1 (100 pre, 300 adaptation, 300 post)")
    ck.published("experiment 1: adaptation trials per observer", float(cnt.iloc[:, 1].mean()), 300, 0, "Methods, Perceptual task")
    ck.published("experiments 6 and 7: trials per observer", float(D[D.expno.isin((6, 7))].groupby(key).size().mean()), 600, 0,
                 "Methods, Transfer of adaptation (150 + 300 + 150 trials; 100 + 50 + 300 + 100 + 50)")
    v = D[D[48] == 1].copy()
    v["off"] = off[(D[48] == 1).to_numpy()]
    v["fwd"] = np.where(v[8] * v[9] == -1, 1, -1)
    rep = {  # (experiment, phase) -> (forward, backward) mean saccade amplitude, deg
        (1, "pre"): (7.73, 7.91), (1, "end"): (8.65, 7.17), (2, "pre"): (7.83, 8.01), (2, "end"): (7.87, 7.56)}
    for (e, ph), (f_rep, b_rep) in rep.items():
        g = v[v.expno == e]
        g = g[(g[4] == 1)] if ph == "pre" else g[(g[4] == 2) & (g[1] > 350)]
        for lab, sg, r in (("forward", 1, f_rep), ("backward", -1, b_rep)):
            ck.published(f"experiment {e}: mean horizontal saccade amplitude, {lab} condition, {'all 100 pre-adaptation trials' if ph == 'pre' else 'last 50 adaptation trials'} (deg)",
                         g[g.fwd == sg].groupby("subj")[42].mean().mean(), r, 0.12 if (e, ph, lab) == (1, "end", "forward") else 0.06,
                         "Results, Perceptual task (experiment 1)" if e == 1 else "Results, Experiment without the perceptual task (experiment 2)")
    g = v[v.expno == 4]
    gp = g[(g[4] == 1) & (g[1] <= 50)]
    ga = g[(g[4] == 2) & (g[1] > 350)]
    up_end = ga[ga.off == -1].groupby("subj")[43].mean().mean()
    dn_end = ga[ga.off == 1].groupby("subj")[43].mean().mean()
    ck.published("experiment 4: mean vertical amplitude, upward adaptation, last 50 adaptation trials (deg)", up_end, 0.49, 0.06,
                 "Results, Cross-axis adaptation (experiments 4 and 5)")
    ck.published("experiment 4: mean vertical amplitude, downward adaptation (magnitude), last 50 adaptation trials (deg)", -dn_end, 0.54, 0.06,
                 "Results, Cross-axis adaptation (experiments 4 and 5)")
    ck.published("experiment 4: mean vertical amplitude before adaptation, all pre-adaptation trials (deg, |value| < 0.06)",
                 abs(gp[43].mean()), 0.04, 0.06, "Results, Cross-axis adaptation (upward 0.04, downward 0.02)")
    for e, r in ((1, 133.97), (2, 148.56)):
        ck.published(f"experiment {e}: mean saccade latency across observers (ms)", v[v.expno == e].groupby("subj")[41].mean().mean(),
                     r, 0.01, "Results, Perceptual task / Experiment without the perceptual task")
    pct = 100.0 * (1 - D[48].mean())
    differences = []
    if abs(pct - 14.35) <= 0.6:
        ck.published("percent of trials excluded", pct, 14.35, 0.6, "Methods, Eye-movement recording")
    else:
        differences.append(f"The paper reports that, on average, 14.35 % of trials were excluded; the authors' valid flag marks {pct:.1f} % of the rows as invalid in the deposit.")
    differences.append("The paper says the 50 pre-adaptation trials were averaged; the published pre-adaptation amplitudes (experiments 1 and 2) are reproduced from all 100 pre-adaptation trials of each observer. The forward amplitude at the end of adaptation in experiment 1 is 8.75 deg here against 8.65 deg in the paper (last 50 adaptation trials, valid trials); the other three end-of-adaptation means agree within 0.03 deg.")
    differences.append("Experiments 4 and 5 appear in the deposit with the vertical array coded as characters 2 to 4 of a five-position array; "
                       "vertical step sizes in degrees are not given, so step_y_deg is empty for them and pert_value counts array positions.")
    meta = dict(
        study=DS,
        paper=dict(authors="Schutz AC, Kerzel D, Souto D", year=2014, title="Saccadic adaptation induced by a perceptual task",
                   venue="J Vis 14(5):4", doi="10.1167/14.5.4"),
        lab="Schutz (University of Giessen)",
        data_url="https://doi.org/10.5281/zenodo.579842",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Combined the authors' per-participant CSV files (no header) into one table with named columns, participant codes "
                 "and the OpenMotor trial, phase and exclusion fields. Row order, values and valid flags are kept; the seven "
                 "character and colour codes before and after the saccade are reduced to the position and identity of the black "
                 "discrimination character and some protocol values in trial_info."),
        how=["One row is one trial of one participant in one experiment (700 trials, or 600 in experiments 6 and 7): the primary "
             "saccade to a horizontal (experiments 1-3, 6-8) or vertical (4, 5) array of characters 9 deg to the left or right of "
             "fixation, with the perceptual report of the black character shown for 100 ms after saccade onset.",
             "Horizontal saccade amplitude is the authors' value, positive in the direction of the trial (leftward and rightward "
             "saccades both positive); vertical amplitude is positive upward. In experiments 1, 2, 6, 7 and 8 step_x_deg is the "
             "physical horizontal offset of the black character from the array centre (0.7 deg per character step, 1.4 deg for the "
             "adaptation locations, positive rightward), found from the character's position and the trial direction; positive "
             "offsets in the saccade direction are the paper's forward adaptation. In experiment 3 it is the 1.4 deg shift of the "
             "whole array during the saccade.",
             "Units follow the paper (degrees of visual angle, ms, deg/s; the authors' readme gives '?'). The array end position in the "
             "authors' file is 9 deg except in adaptation trials of experiments 1 and 6 where it holds the discrimination character's "
             "location (7.6 or 10.4 deg); target_amplitude_deg is 9 deg throughout and the authors' value is kept in trial_info."],
        unit_note="trial (one primary saccade)",
        differences=differences,
    )
    for c in stdlib.columns_for(MODULE):
        if c[1] not in out.columns:
            out[c[1]] = ""
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
