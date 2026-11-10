"""Genewein & Braun (2014), Proc R Soc B 281:20132952: Occam's razor in sensorimotor learning. Model-choice task with noisy
observations of a hidden curve, a drawing and a clicking session per participant (12 participants in two order groups) plus a
spatial-frequency control experiment. Data: Dryad 10.5061/dryad.94jt2 (CC0), read from the Zenodo copy 5003178 (Data.zip).
Run from the site folder on the PC: python3 -I tools/std/convert/Genewein2014_ProcRSocB.py"""
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

ID = "Genewein2014_ProcRSocB"
P = "Genewein & Braun 2014, "
R = stdlib.raw_dir(ID)


def fl(a):
    return ";".join(f"{v:.6g}" for v in np.ravel(a))


def build():
    z = zipfile.ZipFile(R / "Data.zip")
    items = []
    for n in z.namelist():
        m = re.match(r"(ClickingFirst|DrawingFirst) - Session(\d)/Subject(\d+)\.mat$", n)
        if m:
            items.append((0, m.group(1), int(m.group(2)), int(m.group(3)), n))
            continue
        m = re.match(r"SpatialFrequency_ControlExperiment/Subject(\d+)\.mat$", n)
        if m:
            items.append((1, "", 1, int(m.group(1)), n))
    items.sort()
    rows, st = [], {"main": set(), "ctrl": set(), "groups": set(), "probe_m1": [], "n_trials": []}
    for exp, grp, ses, sub, n in items:
        d = sio.loadmat(io.BytesIO(z.read(n)), simplify_cells=True)
        tt, cm, tg = (np.atleast_1d(d[k]).astype(int) for k in ("TrialType", "ChosenModel", "TrueGeneratingModel"))
        N = len(tt)
        assert d["NoisySamples"].shape == (N, 7, 3) and d["YSubjectTrajectory"].shape == (N, 29)
        if exp == 0:
            person = f"main{sub:02d}"
            first_mode = "drawing" if grp == "DrawingFirst" else "clicking"
            mode = first_mode if ses == 1 else ("clicking" if first_mode == "drawing" else "drawing")
            st["main"].add(person)
            st["groups"].add(grp)
            experiment = "Main experiment"
        else:
            person = f"control{sub:02d}"
            mode = "clicking"  # inferred, see README
            st["ctrl"].add(person)
            experiment = "Spatial-frequency control experiment"
        st["n_trials"].append(N)
        st["probe_m1"].append(float(np.mean(cm[tt == 2] == 1)))
        for i in range(N):
            rows.append({
                "dataset": ID, "study": ID, "experiment": experiment, "group": grp, "subject": person, "person": person,
                "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
                "device": "head-mounted display and manipulandum", "session": ses, "block": f"session {ses} ({mode})", "trial": 0,
                "phase": "learning" if tt[i] == 1 else "test", "time_s": "", "task": "reaching", "unit": "trial",
                "condition": "standard" if tt[i] == 1 else "probe", "pert_type": "none", "pert_value": 0.0, "pert_unit": "",
                "feedback": "normal", "excluded_by_authors": False,
                "trial_type": "standard" if tt[i] == 1 else "probe", "true_model": int(tg[i]), "chosen_model": int(cm[i]),
                "response_mode": mode, "first_sample": fl(d["FirstSample"][i]), "noisy_samples": fl(d["NoisySamples"][i]),
                "last_sample": fl(d["LastSample"][i]), "underlying_y": fl(d["YUnderlying"][i]),
                "drawn_y": fl(d["YSubjectTrajectory"][i]) if mode == "drawing" else ""})
    cnt = {}
    for r in rows:
        cnt[r["subject"]] = r["trial"] = cnt.get(r["subject"], 0) + 1
    return rows, st


def main():
    rows, st = build()
    c = stdlib.Checks()
    c.published("the two order groups (drawing first, clicking first)", len(st["groups"]), 2, 0, P + "Results 3(b) (subjects randomly assigned into two groups, session order reversed)")
    c.published("simpler model preferred in probe trials (pooled median of P(choose M1) above 0.5, 1 = yes)", float(np.median(st["probe_m1"]) > 0.5), 1, 0,
                P + "Results 3(b), Figure 2 (ambiguous probe trials: subjects significantly preferred the simpler model M1)")
    c.published("simpler model preferred in probe trials of the control experiment (1 = yes)",
                float(np.median(st["probe_m1"][-len(st["ctrl"]):]) > 0.5), 1, 0, P + "Results 3(d), Figure 3 (subjects significantly preferred M1)")
    c.internal("main experiment: 12 participants x 2 sessions", len(st["main"]) == 12, f"{len(st['main'])} participants")
    c.internal("every session file has 600 trials", set(st["n_trials"]) == {600}, str(sorted(set(st["n_trials"]))))
    c.internal("row count", len(rows) == 600 * len(st["n_trials"]), f"{len(rows)} rows")
    for x in c.items:
        if not x["ok"]:
            print("FAILED:", x)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8")) if x["id"] == ID)
    p0 = s["papers"][0]
    meta = dict(
        study=ID, paper={k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="CC0 (Dryad; read from the Zenodo copy 5003178)", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes="One row per trial from the authors' per-subject, per-session MATLAB files; the 3-vector and 29-point arrays are written as semicolon-separated "
                "text; participant codes replaced; drawn trajectories left empty in clicking sessions (the authors' file holds simulated values there).",
        how=["One row is one trial of the model-choice task: seven noisy observations of an unobserved curve (generated by a simple model, long length scale, or a complex "
             "model, short length scale) are shown, and the participant either draws the presumed noise-free curve with a manipulandum or clicks the model class. "
             "Standard trials cue the generating model; probe trials are ambiguous and uncued. Coordinates are (Y left-right, Z, X forward-backward) in the authors' workspace "
             "units (not stated; the paper says the observation noise is of the order of centimetres).",
             "Main experiment: 12 participants, each with a drawing and a clicking session of 600 trials; groups give the session order. Which session was drawing is taken from the "
             "group name; it matches the stored trajectories (real drawn curves deviate from the underlying curve far more than the simulated ones).",
             "The 29 x positions at which the curves are evaluated are the same on every trial and are listed in the README for the drawn_y and underlying_y columns (xunderlying in the deposit, not repeated per row).",
             "The control experiment (spatial-frequency control) has 9 participants in one session of 600 trials; its response mode is inferred as clicking because its stored trajectories look simulated."],
        unit_note="trial",
        differences=["The paper (main text, Results 3(d)) refers to 10 subjects in the control experiment; the deposit has 9 control files. No participant count or trial count for the main experiment "
                     "is given in the main text (they are in the electronic supplement, which the publisher site did not serve and which is not on PMC), so those counts are internal checks only.",
                     "The paper's regression statistics (intercepts and slopes with 95% CIs) and sign tests were not recomputed; the supplement defines the error levels and fits."])
    return stdlib.release(ID, "model_choice_curve", rows, meta, c)


if __name__ == "__main__":
    main()
