"""Hadjiosif, Gibo & Smith (2024), PNAS: the cerebellum and sensorimotor memory. Re-analysis of two earlier curl-force-field
studies of reaching in cerebellar degeneration (the 2010 study with two targets, 11 N s/m; the 2013 study with one target,
13 N s/m): 28 controls, 20 severe and 8 milder patients in the paper, each tested with an abrupt and a gradual field session.
One row per trial of one session, module force_field_reaching (the deposit has an angular error per trial and no
force-compensation or MPE measure).

Run from the site folder: python3 -I tools/std/convert/Hadjiosif2024_PNAS.py
Raw files (untrusted data, read only) in raw_extra/Hadjiosif2024_PNAS/: dat_CH_Aug_2024.mat and dat_G_Aug_2024.mat from
https://github.com/harvardmotorlab/CerebellarAtaxiaPaper2024 (README: one line; no code; fields learned from the data and the paper).
Each holds a struct with fields c (controls), p (severe patients), m (milder patients), each with a (abrupt) and g (gradual)
trial x participant matrices: endpoint_dev, nback_, iti_, iti_nb, x_interp, y_interp (time x trial x participant); i_first (first
field trial, 1-based), weights (per trial), icars, n (group sizes).
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hadjiosif2024_PNAS"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
GROUP = {"c": "control", "p": "severe_CBL", "m": "milder_CBL"}
SESS = {"a": "abrupt", "g": "gradual"}
EXPS = [("dat_CH_Aug_2024", "dat_CH", "Criscimagna-Hemminger2010", 11.0, "two targets"),
        ("dat_G_Aug_2024", "dat_G", "Gibo2013", 13.0, "one target")]


def nz(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else v


rows = []
paths = {}
npers = {}
first_dev = {}
dropped = []
for fn, key, exp, bfield, _ in EXPS:
    d = sio.loadmat(RAW / f"{fn}.mat", squeeze_me=True, struct_as_record=False)[key]
    first = int(np.atleast_1d(d.i_first.c).flat[0])
    ntr = d.endpoint_dev.c.a.shape[0]
    w = np.asarray(d.weights, float)
    # error-clamp trials: rows where every participant has no value (only the 2010 study shows this pattern)
    allnan = np.ones(ntr, bool)
    for g in "cpm":
        for s in "ag":
            allnan &= np.isnan(getattr(getattr(d.endpoint_dev, g), s)).all(1)
    pathcols = {}
    for g in "cpm":
        ngrp = getattr(d.endpoint_dev, g).a.shape[1]
        icars = getattr(getattr(d, "icars", None), g, None) if g != "c" else None
        for i in range(ngrp):
            pkey = f"{exp}_{g}{i + 1}"
            npers[pkey] = (exp, g)
            for s in "ag":
                ep = getattr(getattr(d.endpoint_dev, g), s)[:, i].astype(float)
                if np.isnan(ep).all():
                    dropped.append((pkey, s))
                    continue
                nb = getattr(getattr(d.nback_, g), s)[:, i].astype(float)
                it = getattr(getattr(d.iti_, g), s)[:, i].astype(float)
                itnb = getattr(getattr(d.iti_nb, g), s)[:, i].astype(float)
                xs = getattr(getattr(d.x_interp, g), s)[:, :, i]
                ys = getattr(getattr(d.y_interp, g), s)[:, :, i]
                sub = f"{pkey}_{s}"
                base = np.nanmean(ep[:first - 1])
                for t in range(ntr):
                    learning = t + 1 >= first
                    clamp = bool(allnan[t])
                    strength = 0.0
                    if learning:
                        strength = float(w[t]) if s == "g" else 1.0
                    ttype = "channel" if clamp else ("field" if learning else "null")
                    nbv = nb[t]
                    rows.append(dict(
                        dataset=ID, study=ID, experiment=exp, group=GROUP[g], subject=sub, person=pkey,
                        subject_excluded_by_authors=False, age="", sex="", handedness="", setting="lab",
                        population="healthy" if g == "c" else "cerebellar_degeneration",
                        device="robotic manipulandum", session="", block=("force_field" if learning else "baseline") + "_" + SESS[s],
                        trial=t + 1, phase="learning" if learning else "baseline", time_s="", task="reaching", unit="trial",
                        condition=SESS[s],
                        pert_type="channel" if clamp else ("force_field" if learning else "none"),
                        pert_value=0 if clamp or not learning else round(bfield * strength, 6),
                        pert_unit="" if clamp or not learning else "N s/m", feedback="normal",
                        excluded_by_authors="" if (np.isnan(ep[t]) and not clamp) else False,
                        trial_type=ttype, trial_in_condition=t + 1,
                        target_direction_deg=0 if exp == "Gibo2013" else "",
                        direction_error_deg=nz(ep[t]),
                        ff_strength_fraction="" if clamp else strength,
                        iti_same_direction_s=nz(it[t]), iti_same_direction_nonbreak_s=nz(itnb[t]),
                        trials_back_same_direction="" if np.isnan(nbv) or nbv < 0 else int(nbv)))
                pathcols[(sub)] = (xs, ys)
                if s == "a" and g in "cp" and not np.isnan(ep[first - 1]):
                    first_dev.setdefault((exp, g), []).append(ep[first - 1] - base)
    # paths: one CSV per experiment, one row per trial, x then y samples
    buf = io.StringIO()
    nt = next(iter(pathcols.values()))[0].shape[0]
    buf.write("subject_key,condition,trial," + ",".join(f"x{k}" for k in range(nt)) + "," + ",".join(f"y{k}" for k in range(nt)) + "\n")
    paths[exp] = (buf, pathcols, nt, ntr)
    # (written after release assigns codes; keep raw keys and map by order of appearance below)

# map person/subject keys to the codes release() will assign (order of first appearance)
sub_order, seen = [], set()
for r in rows:
    if r["subject"] not in seen:
        seen.add(r["subject"])
        sub_order.append(r["subject"])
code = {s: f"{ID}_S{i + 1:04d}" for i, s in enumerate(sub_order)}
extra = []
for exp, (buf, pathcols, nt, ntr) in paths.items():
    for sub, (xs, ys) in pathcols.items():
        cond = "abrupt" if sub.endswith("_a") else "gradual"
        for t in range(ntr):
            x, y = xs[:, t], ys[:, t]
            if np.isnan(x).all() and np.isnan(y).all():
                continue
            fmt = lambda a: ",".join("" if np.isnan(v) else format(float(v), ".6g") for v in a)
            buf.write(f"{code[sub]},{cond},{t + 1},{fmt(x)},{fmt(y)}\n")
    extra.append((f"samples/hand_paths_{exp}.csv", buf.getvalue().encode("utf-8")))

# ---- checks
import collections
cnt = collections.Counter(v for v in npers.values())
ctrl = sum(n for (e, g), n in cnt.items() if g == "c")
mild = sum(n for (e, g), n in cnt.items() if g == "m")
checks.published("control participants (13 in the 2010 study + 15 in the 2013 study)", ctrl, 28, 0, "Results, first paragraph (n = 28)")
checks.published("milder-CBL patients", mild, 8, 0, "Methods, Participants (8 patients with milder ataxia)")
checks.published("severe-CBL patients, 2010 study", cnt[("Criscimagna-Hemminger2010", "p")], 8, 0, "Results, first paragraph (8 in the 2010 study)")
for e, g, rep, src in (("Criscimagna-Hemminger2010", "c", 14.9, "Results, Fig. 1 text (14.9 +- 1.3 deg, 2010 controls)"),
                             ("Criscimagna-Hemminger2010", "p", 18.0, "Results, Fig. 1 text (18.0 +- 3.1 deg, 2010 severe-CBL)"),
                             ("Gibo2013", "c", 9.1, "Results, Fig. 1 text (9.1 +- 0.9 deg, 2013 controls)"),
                             ("Gibo2013", "p", 8.7, "Results, Fig. 1 text (8.7 +- 1.7 deg, 2013 severe-CBL)")):
    checks.published(f"{e} {GROUP[g]}: initial deflection on the first abrupt field trial, baseline subtracted, deg",
                     np.nanmean(first_dev[(e, g)]), rep, 1.0, src)
checks.published("training trials per session, 2010 study", 330 - 90, 240, 0, "Results (training lasted 240 trials)")
checks.internal("rows per session are 330 (2010) or 211 (2013)", all(
    n == (330 if e == "Criscimagna-Hemminger2010" else 211)
    for e in ("Criscimagna-Hemminger2010", "Gibo2013")
    for n in collections.Counter(r["subject"] for r in rows if r["experiment"] == e).values()))
checks.internal("learning rows start at the deposit's first-field-trial index (91 and 42)",
                min(r["trial"] for r in rows if r["experiment"] == "Criscimagna-Hemminger2010" and r["phase"] == "learning") == 91
                and min(r["trial"] for r in rows if r["experiment"] == "Gibo2013" and r["phase"] == "learning") == 42)
checks.internal("direction errors within +-60 deg", all(abs(float(r["direction_error_deg"])) < 60 for r in rows if r["direction_error_deg"] != ""))

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
n_g_p = cnt[("Gibo2013", "p")]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")}, lab="Smith (Harvard)",
    data_url="https://github.com/harvardmotorlab/CerebellarAtaxiaPaper2024", license="none stated", license_url="",
    changes="converted to the OpenMotor format, participant codes replaced, force-field strength rebuilt from the deposit's weights, "
            "and one empty session removed.",
    unit_note="trial of one participant's abrupt or gradual force-field session",
    differences=[
        f"The paper reports 12 severe-CBL patients in the 2013 study (20 in total); the 2013 file has {n_g_p} patient columns but only 12 ICARS scores, so one patient column has no ICARS score and the paper's patient count does not reproduce exactly. All columns are kept and grouped as the authors grouped them (group severe_CBL = field p).",
        "One milder-CBL participant of the 2010 study has no abrupt-session data at all (every value empty); that empty sequence is not written.",
        "The initial deflections of the first abrupt field trial match the paper within about 0.5 deg (the paper's exact baseline correction is not given), so a tolerance of 1 deg is used."],
    how=[
        "One row is one reaching trial of one session; every participant has an abrupt-onset and a gradual-onset curl force-field session (counterbalanced in the original studies; the order is not in the deposit, so session is empty). The 2010 study has 330 trials (90 baseline, 240 training) toward two targets; the 2013 study has 211 (41 baseline, 170 training) toward one target.",
        "direction_error_deg is the deposit's 'endpoint_dev': the reach direction error at the target distance, in degrees. The deposit does not state the unit or sign; degrees follow from its size (the paper's initial deflections are 9 to 18 deg) and positive is the direction in which the field pushes, as shown by the first abrupt field trial reproducing the paper's positive deflections for every group.",
        "ff_strength_fraction is 0 in the baseline, 1 from the first field trial in abrupt sessions, and follows the deposit's weights in gradual sessions (a linear ramp over 232 trials in the 2010 study and 160 in the 2013 study, then full strength). pert_value is the field coefficient (11 N s/m in the 2010 study, 13 N s/m in the 2013 study, from the paper) times that fraction.",
        "In the 2010 study 66 trial positions have no value for any participant; these are taken to be error-clamp trials (trial_type channel; the paper says error-clamp trials were part of the design). In the 2013 study gaps differ between participants and cannot be told apart from trials the authors removed, so excluded_by_authors is empty for any trial without a direction error.",
        "The 2010 study's two target directions (31.5 and 211.5 deg) alternate in an order the deposit does not give, so target_direction_deg is empty there; in the 2013 study the target was at 0 deg (mirrored across the mid-sagittal line for participants who used the left arm; arm is not in the deposit).",
        "samples/hand_paths_<experiment>.csv holds the deposit's interpolated hand paths (x_interp, y_interp: 63 samples per trial in the 2010 study, 139 in the 2013 study; metres, inferred from their size) for each trial that has one, keyed by the subject code. How the samples are spaced in time is not documented.",
        "group is control, severe_CBL (ICARS 40 or more) or milder_CBL as in the deposit (fields c, p, m)."],
)
stdlib.release(ID, "force_field_reaching", rows, meta, checks, extra_files=extra)
