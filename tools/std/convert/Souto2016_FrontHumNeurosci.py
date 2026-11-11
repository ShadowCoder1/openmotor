"""Souto, Gegenfurtner & Schutz (2016) Front Hum Neurosci: saccade adaptation and visual uncertainty.

Raw file: raw_extra/Souto2016_FrontHumNeurosci/data.zip (Zenodo 579839): folders exp1 and exp2, per observer a headerless
subj_<n>_adaptation_exp_1.csv and subj_<n>_perception_exp_1.csv (columns in each folder's README.txt). Run from the site
folder: python3 -I tools/std/convert/Souto2016_FrontHumNeurosci.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Souto2016_FrontHumNeurosci"
MODULE = "saccade"
PXDEG = 64.81 / 1.5     # the 1.5 deg backward step is -64.81 px in the exp2 files
DEVICE = "EyeLink 1000 (1000 Hz), 120 Hz LCD monitor at 90 cm, keyboard"
Z = zipfile.ZipFile(stdlib.raw_dir(DS) / "data.zip")
PHASE = {-1: "practice", 1: "baseline", 2: "learning", 3: "aftereffect"}
PHASE_LABEL = {-1: "practice", 1: "pre-adaptation", 2: "adaptation", 3: "post-adaptation"}


def rd(name):
    return pd.read_csv(io.StringIO(Z.read(name).decode()), header=None)


def files(folder):
    out = []
    for n in Z.namelist():
        if n.startswith(f"{folder}/subj_") and n.endswith("_adaptation_exp_1.csv"):
            out.append(int(n.split("subj_")[1].split("_")[0]))
    return sorted(out)


def main():
    ck = stdlib.Checks()
    rows = []
    obs = {}
    same_as_exp1 = 0
    phase_ok = True
    max_step_px = 0.0
    steps_adapt = set()
    for folder in ("exp1", "exp2"):
        obs[folder] = set()
        for f in files(folder):
            a = rd(f"{folder}/subj_{f}_adaptation_exp_1.csv")
            p = rd(f"{folder}/subj_{f}_perception_exp_1.csv")
            sid = int(a[1].iloc[0])
            assert (a[1] == sid).all() and (p[1] == sid).all()
            obs[folder].add(sid)
            pers = f"{folder}|{sid}"
            if folder == "exp2":
                a1 = rd(f"exp1/subj_{f}_adaptation_exp_1.csv") if f in files("exp1") else None
                if a1 is not None and a1.shape == a.shape and a1.shape[1] > 9 and a1[[8, 9]].fillna(0).equals(a[[8, 9]].fillna(0)):
                    same_as_exp1 += 1
            # ---- adaptation rows
            n = len(a)
            ph = a[7].astype(int).to_numpy()
            if folder == "exp2":
                step_px = a[6].to_numpy(dtype=float)
                step = step_px / PXDEG
                cont = [""] * n
                steps_adapt |= set(np.round(step_px[ph == 2], 2))
            else:
                step = np.where(ph == 2, -1.5, 0.0)           # 1.5 deg backward step (Methods); contrast is in column 7 of this folder
                cont = a[6].to_numpy(dtype=float)
            endx = a[8].to_numpy(dtype=float) if a.shape[1] > 8 else np.full(n, np.nan)
            endy = a[9].to_numpy(dtype=float) if a.shape[1] > 9 else np.full(n, np.nan)
            err = a[10].to_numpy(dtype=float) if a.shape[1] > 10 else np.zeros(n)
            df = pd.DataFrame({
                "dataset": DS, "study": DS, "experiment": folder, "group": "", "subject": f"{pers}|adaptation", "person": pers,
                "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
                "setting": "lab", "device": DEVICE, "session": "", "block": [f"adaptation block {int(b)}, {PHASE_LABEL[q]}" for b, q in zip(a[2], ph)],
                "trial": np.arange(1, n + 1), "phase": [PHASE[q] for q in ph], "time_s": "", "task": "eye movements", "unit": "trial",
                "condition": [("contrast %g" % c) if folder == "exp1" else "step size experiment" for c in cont] if folder == "exp1" else "step size experiment",
                "pert_type": np.where(ph == 2, "intrasaccadic_step", "none"), "pert_value": step, "pert_unit": "deg",
                "feedback": "normal", "excluded_by_authors": (np.isnan(endx) | (err != 0)),
                "target_amplitude_deg": 12.0, "target_direction_deg": 0.0, "step_x_deg": step, "step_y_deg": 0.0,
                "saccade_end_x_deg": endx, "saccade_end_y_deg": endy, "eye": "",
                "trial_info": [f"trial_in_block={int(t)};stimulus_side={int(s)};fixation_time_s={ft:g}" + (f";contrast={c:g}" if folder == "exp1" else "")
                               for t, s, ft, c in zip(a[3], a[4], a[5], cont if folder == "exp1" else [0] * n)],
            })
            rows.append(df)
            phase_ok &= bool(set(np.unique(ph)) <= {-1, 1, 2, 3})
            # ---- perception rows (direction judgement of the intrasaccadic step; no saccade values in these files)
            m = len(p)
            resp = p[7].to_numpy(dtype=float)
            e = p[8].to_numpy(dtype=float) if p.shape[1] > 8 else np.full(m, np.nan)
            if folder == "exp2":
                stp = p[6].to_numpy(dtype=float)
                max_step_px = max(max_step_px, np.abs(stp).max())
                sdeg = stp / PXDEG
                cinfo = [""] * m
            else:
                sdeg = np.full(m, np.nan)
                cinfo = p[6].to_numpy(dtype=float)
            known = np.where(np.isnan(resp), False, True)
            correct = [("" if (np.isnan(r) or (folder == "exp2" and s == 0)) else bool(ee == 0)) for r, s, ee in zip(resp, sdeg if folder == "exp2" else np.ones(m), e)]
            dp = pd.DataFrame({
                "dataset": DS, "study": DS, "experiment": folder, "group": "", "subject": f"{pers}|perception", "person": pers,
                "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
                "setting": "lab", "device": DEVICE, "session": "", "block": [f"perception block {int(b)}" for b in p[2]],
                "trial": np.arange(1, m + 1), "phase": "test", "time_s": "", "task": "eye movements", "unit": "trial",
                "condition": [("contrast %g" % c) for c in cinfo] if folder == "exp1" else "step size experiment",
                "pert_type": "intrasaccadic_step", "pert_value": sdeg, "pert_unit": "deg", "feedback": "none",
                "excluded_by_authors": False, "target_amplitude_deg": 12.0, "target_direction_deg": 0.0, "step_x_deg": sdeg, "step_y_deg": 0.0,
                "response": [("" if np.isnan(r) else f"{int(r)}") for r in resp], "response_correct": correct,
                "trial_info": [f"trial_in_block={int(t)};stimulus_side={int(s)};fixation_time_s={ft:g}" + (f";contrast={c:g}" if folder == "exp1" else "")
                               for t, s, ft, c in zip(p[3], p[4], p[5], cinfo if folder == "exp1" else [0] * m)],
            })
            rows.append(dp)
    out = pd.concat(rows, ignore_index=True)
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns:
            out[c] = ""
    out = out[[c[1] for c in stdlib.columns_for(MODULE)]]

    ck.published("Experiment 1: observers (Table 1 lists 19)", len(obs["exp1"]), 19, 0, "Results, Table 1")
    ck.published("Experiment 2: observers", len(obs["exp2"]), 14, 0, "Methods, Participants (Experiment 2)")
    ad = out[(out["task"] == "eye movements") & out["subject"].str.endswith("|adaptation")]
    cnt = ad.groupby(["subject", "block"]).size()
    full = cnt[cnt.index.get_level_values(1).str.contains("adaptation block")]
    per_phase = ad.groupby(["subject", "phase"]).size().unstack(fill_value=0)
    # a standard block has 50 pre, 150 adaptation and 50 post trials; observers have two (one has three) blocks
    ck.published("Experiment 1: adaptation trials per block (150)", ad[(ad["experiment"] == "exp1") & (ad["phase"] == "learning")].groupby(["subject", "block"]).size().mean(), 150, 0.01, "Methods, Experiment 1 (50 pre, 150 adaptation, 50 post)")
    ck.published("Experiment 1: pre-adaptation trials per block (50)", ad[(ad["experiment"] == "exp1") & (ad["phase"] == "baseline")].groupby(["subject", "block"]).size().mean(), 50, 0.01, "Methods, Experiment 1")
    ck.published("Experiment 1: post-adaptation trials per block (50)", ad[(ad["experiment"] == "exp1") & (ad["phase"] == "aftereffect")].groupby(["subject", "block"]).size().mean(), 50, 0.01, "Methods, Experiment 1")
    pe = out[out["subject"].str.endswith("|perception") & (out["experiment"] == "exp2")]
    ck.published("largest perceptual step (deg)", float(pd.to_numeric(pe["pert_value"]).abs().max()), 4.0, 0.01, "Methods (steps limited to +-4 deg)")
    ck.published("adaptation step (deg, backward)", float(-pd.to_numeric(ad[(ad["experiment"] == "exp2") & (ad["phase"] == "learning")]["pert_value"]).mean()), 1.5, 0.001, "Methods (target stepped backward 1.5 deg)")
    ck.internal("all adaptation rows have phase -1, 1, 2 or 3", phase_ok)
    ck.internal("response 1 goes with positive steps: perceptual 'errors' column equals a response that disagrees with the step direction on rows with a step",
                bool((lambda q: ((pd.to_numeric(q["response"]).astype(float) == (pd.to_numeric(q["pert_value"]) > 0)) == (q["response_correct"] == True)).mean() > 0.99)(pe[(pe["response"] != "") & (pd.to_numeric(pe["pert_value"]) != 0)])))
    ck.internal("exp2 folder: number of observers whose adaptation saccade values equal those of the same-numbered exp1 file", same_as_exp1 >= 8, str(same_as_exp1))
    meta = dict(
        study=DS,
        paper=dict(authors="Souto D, Gegenfurtner KR, Schutz AC", year=2016, title="Saccade Adaptation and Visual Uncertainty",
                   venue="Front Hum Neurosci 10:227", doi="10.3389/fnhum.2016.00227"),
        lab="Schutz / Gegenfurtner (Giessen / Marburg)",
        data_url="https://doi.org/10.5281/zenodo.579839", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Joined each observer's adaptation and perception files into one table (kept as two sequences per observer, because the "
                 "deposit does not say how perception and adaptation blocks alternated), gave observers codes, labelled the phases "
                 "(-1 practice, 1 pre-adaptation, 2 adaptation, 3 post-adaptation, as the 5/50/150/50 trial counts indicate) and converted "
                 "the pixel step sizes of the exp2 folder to degrees at 43.2 px/deg (the 1.5 deg step is 64.81 px). The two folders are kept as the authors "
                 "named them; see Known differences."),
        how=["Adaptation rows: one trial of the saccade to a target 12 deg to the right of fixation (fixation varied from 0 to -5 deg). In "
             "adaptation-phase trials the target stepped back 1.5 deg during the saccade; the horizontal and vertical saccade end positions "
             "are in the authors' units, degrees of visual angle relative to the fixation point. Rows with no saccade end values or a non-zero "
             "error flag are marked excluded.",
             "Perception rows: one trial of the two-alternative judgement of the direction of an intrasaccadic target step (response 1 goes with "
             "rightward steps, 0 with leftward ones; response_correct is the authors' error flag inverted, left empty for zero steps). These "
             "files hold no saccade values. In the exp1 folder column 7 of both files is the contrast of the target (0.1, 0.5 or 1.0 as in the "
             "deposit); in the exp2 folder it is the step size in pixels, in degrees here.",
             "Participant numbers are the numbers inside the files, coded per folder; the paper says 8 of the 14 exp2 observers also took part "
             "in exp1, which the deposit does not show."],
        unit_note="trial (an adaptation trial with one saccade end position, or a perceptual judgement trial)",
        differences=["The paper describes Experiment 2 as five backward step sizes (0.25-1.25 deg) in separate sessions. The deposit's exp2 folder holds only the 1.5 deg adaptation session: its adaptation rows are the same trials and saccade values as the exp1 folder's for most observers (a copy with the step in pixels instead of the contrast), and its perception rows are the same trials as exp1's with step sizes instead of contrasts. So the exp2 folder is not an independent experiment, and the five-step-size data are not in the deposit. The paper has two target contrasts, 10% and 100% (Weber), for all 19 Experiment 1 observers. The perception files hold 0.1 and 1.0 for every observer, and the exp1 adaptation files hold 0.1 and 1.0 for 16 observers; for three observers (about 765 of the 9,946 adaptation trials) the adaptation block that the other observers ran at 1.0 is coded 0.5, although their own perception files hold 1.0. The label 'contrast 0.5' is the deposit's value, kept as shared; it is probably the paper's 100% condition for these three observers, but nothing in the deposit or the paper states that, so it was not relabelled.",
                     "A few exp1/exp2 files have fewer columns (no saccade end values in some exp2 adaptation files) or an extra block, and the paper's analysis excludes observers whose psychometric function was flat; the deposit does not say which ones, so none is flagged."],
    )
    stdlib.release(DS, MODULE, out, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
