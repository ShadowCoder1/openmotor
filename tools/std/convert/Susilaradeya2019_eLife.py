"""Susilaradeya et al. (2019), eLife 8:e40145: bimanual isometric finger-force tracking of a circular target with delayed or
perturbed cursor feedback (human Experiments 1 and 2 only; Experiments 3 and 4 of the deposit are monkeys). One row per trial.

Run from the site folder: python3 -I tools/std/convert/Susilaradeya2019_eLife.py
Raw files (untrusted data, read only) in raw_extra/Susilaradeya2019_eLife/ from the Zenodo copy of the Dryad deposit
(https://doi.org/10.5061/dryad.53sq7kn, Zenodo record 4948727): Experiment1.mat, Experiment2.mat.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Susilaradeya2019_eLife"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
STUDY = "Susilaradeya et al. 2019, eLife"
DEVICE = "two Honeywell FSG15N1A isometric force sensors (index fingers of both hands) mapped to a cursor on a monitor; force sampled at 50 Hz"

EXPS = [
    ("Experiment1.mat", "Experiment 1 (feedback delay)",
     ["c_pos_x", "c_pos_y", "t_pos_x", "t_pos_y", "c_ang", "t_ang", "c_ang_vel"],
     ["% of target radius"] * 4 + ["rad", "rad", "rad/s"]),
    ("Experiment2.mat", "Experiment 2 (feedback delay and spatial perturbation)",
     ["c_pos_x", "c_pos_y", "t_pos_x", "t_pos_y", "p_pos_x", "p_pos_y", "c_ang", "t_ang", "p_ang", "c_ang_vel", "f_ang_vel", "p_ang_vel"],
     ["% of target radius"] * 6 + ["rad"] * 3 + ["rad/s"] * 3),
]
FS = 50.0


def arr(trial, name):
    v = np.asarray(getattr(trial, name), dtype=float)
    return v


rows, extra = [], []
subj_code = {}
tcount = {}
stats = {}
t_speed, t_rad = [], []
for fn, expname, chan, units in EXPS:
    m = sio.loadmat(RAW / fn, squeeze_me=True, struct_as_record=False)
    assert float(m["s_rate"]) == FS
    for si, sj in enumerate(m["subject"], start=1):
        key = (fn, si)
        subj_code.setdefault(key, len(subj_code) + 1)
        sc = subj_code[key]
        for ci, cond in enumerate(sj.cond, start=1):
            delay_ms = int(round(float(cond.delay) * 1000)) if float(cond.delay) < 5 else int(round(float(cond.delay)))
            pf = int(cond.pert_freq) if "pert_freq" in cond._fieldnames else None
            for ti, tr in enumerate(cond.trial, start=1):
                tcount[key] = tcount.get(key, 0) + 1
                n = len(arr(tr, "c_ang"))
                cols = {}
                for c in chan:
                    base, _, ax = c.rpartition("_")
                    if c.endswith(("_x", "_y")) and base.endswith("_pos"):
                        cols[c] = arr(tr, base)[:, 0 if ax == "x" else 1]
                    else:
                        cols[c] = arr(tr, c)
                df = pd.DataFrame(cols)
                df.insert(0, "time_s", np.arange(n) / FS)
                path = f"samples/{ID}_S{sc:04d}_trial{tcount[key]:03d}.csv"
                extra.append((path, df.to_csv(index=False, float_format="%.8g", lineterminator="\r\n")))
                tang = np.unwrap(cols["t_ang"])
                t_speed.append(np.abs(np.diff(tang)) * FS)
                t_rad.append(np.hypot(cols["t_pos_x"], cols["t_pos_y"]))
                st = stats.setdefault(key, dict(n=0, conds=set(), per={}))
                st["n"] += 1
                st["conds"].add((delay_ms, pf))
                st["per"][(delay_ms, pf)] = st["per"].get((delay_ms, pf), 0) + 1
                cond_label = f"feedback delay {delay_ms} ms" + (f"; perturbation {pf} Hz" if pf is not None else "")
                ti_txt = f"delay_ms={delay_ms};trial_in_condition={ti}" + (f";perturbation_hz={pf}" if pf is not None else "")
                rows.append(dict(
                    dataset=ID, study=STUDY, experiment=expname, group="", subject=f"{fn[:11]}_{si}", person=f"P{si}",
                    subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                    device=DEVICE, session=1, block=f"condition {ci}", trial=tcount[key], phase="test", time_s="",
                    task="tracking", unit="recording", condition=cond_label,
                    pert_type="delay" if (pf in (None, 0)) else "delay and sinusoidal cursor perturbation",
                    pert_value=delay_ms, pert_unit="ms", feedback="altered" if (delay_ms > 0 or pf) else "normal",
                    excluded_by_authors=False,
                    recording=tcount[key], samples_file=path,
                    signal_kind="cursor, target" + (" and perturbation" if pf is not None else "") + " position and angle on the screen, cursor angular velocity",
                    channels=";".join(chan), channel_units=";".join(units), sample_rate_hz=FS, n_samples=n, duration_s=n / FS,
                    events="",
                    processing="as shared by the authors: a 512-sample excerpt of each 20 s trial (the authors analyse a 10 s window starting 5 s into the trial); angular velocities as computed by the authors",
                    trial_info=ti_txt))
D = pd.DataFrame(rows)

P = checks.published
e1 = [k for k in stats if k[0] == "Experiment1.mat"]
e2 = [k for k in stats if k[0] == "Experiment2.mat"]
P("participants in Experiment 1", len(e1), 8, 0, "Materials and methods, Subjects (eight subjects in each experiment)")
P("participants in Experiment 2", len(e2), 8, 0, "Materials and methods, Subjects")
P("Experiment 1 trials per subject", np.mean([stats[k]["n"] for k in e1]), 70, 0, "Materials and methods, Human tracking task (70 trials)")
P("Experiment 1 delay conditions", len({c for k in e1 for c in stats[k]["conds"]}), 5, 0, "Materials and methods (0, 100, 200, 300, 400 ms)")
P("Experiment 1 trials per condition", np.mean([v for k in e1 for v in stats[k]["per"].values()]), 14, 0, "Materials and methods (14 of each condition)")
P("Experiment 2 trials per subject", np.mean([stats[k]["n"] for k in e2]), 144, 0, "Materials and methods (144 trials)")
P("Experiment 2 conditions", len({c for k in e2 for c in stats[k]["conds"]}), 12, 0, "Materials and methods (6 perturbation frequencies x 2 delays)")
P("Experiment 2 trials per condition", np.mean([v for k in e2 for v in stats[k]["per"].values()]), 12, 0, "Materials and methods (12 trials per condition)")
P("target rotation (rotations/s)", np.concatenate(t_speed).mean() / (2 * np.pi), 0.2, 0.01, "Materials and methods (f_target = 0.2 rotations per second)")
P("target radius (% of target radius)", np.concatenate(t_rad).mean(), 100, 1, "Materials and methods (r_target = 100%)")
P("sampling rate (Hz)", D.sample_rate_hz.iloc[0], 50, 0, "Materials and methods (50 samples/s)")
P("length of analysis window available per trial (s, at least 10)", D.duration_s.min() >= 10, 1, 0, "Figure 1 legend (10 s window beginning 5 s after trial start)")
checks.internal("every trial has a samples file", len(extra) == len(D) == 560 + 1152)
checks.internal("delays are 0-400 ms", D.pert_value.between(0, 400).all())

part = pd.DataFrame([dict(person=f"P{i}", in_experiments="1, 2") for i in range(1, 9)])
pcols = [("in_experiments", "text", "", "Experiments of the deposit (1, 2) whose rows belong to this person. The deposit numbers subjects 1-8 in both files; the paper says the same eight people did Experiments 1 and 2, so equal numbers are treated as one person (assumed, not stated).")]

meta = dict(
    paper=dict(authors="Susilaradeya D, Xu W, Hall TM, Galan F, Alter K, Jackson A", year=2019,
               title="Extrinsic and intrinsic dynamics in movement intermittency", venue="eLife 8:e40145", doi="10.7554/elife.40145"),
    lab="Jackson (Newcastle University)",
    data_url="https://doi.org/10.5061/dryad.53sq7kn", license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Each trial of the two human experiment files became one samples CSV (time_s = sample index / 50 Hz, then the authors' channels; the two-column position "
             "arrays were split into _x and _y). Experiments 3 and 4 of the deposit (monkeys, with local field potentials) and the authors' MATLAB scripts "
             "are not included. Age, sex and handedness are not shared per person (the paper gives group figures only: three females, ages 23-33, one left-handed)."),
    how=["One row is one 10.24 s excerpt (512 samples at 50 Hz) of one tracking trial. The participant squeezed two force sensors with the index fingers; the forces moved a "
         "cursor, which had to follow a target circling the screen centre at 0.2 rotations/s, with the cursor delayed by 0 to 400 ms (Experiment 1) or delayed and "
         "also perturbed by a sinusoidal angular-velocity modulation of 0 to 5 Hz (Experiment 2).",
         "Samples: positions are in percent of the target's radius with the screen centre at (0, 0); angles are in radians; angular velocities in rad/s, all as shared. "
         "c = cursor, t = target, p = perturbed target path (Experiment 2: the path the cursor would follow with the perturbation), f_ang_vel = angular velocity of the "
         "unperturbed force-driven cursor, p_ang_vel = perturbation angular velocity. Rows are in the deposit's order (condition by condition), not the order the trials were presented in.",
         "pert_value is the feedback delay in ms; for Experiment 2 the perturbation frequency is in condition and trial_info (perturbation_hz, 0 = none)."],
    differences=["The paper describes 20 s trials; the deposit holds 512 samples (10.24 s) per trial, so full trials are not available. The deposit's comments say 12 trials per "
                 "Experiment 1 condition and 14 are present, matching the paper's 14.",
                 "Person codes link the same-numbered subject of the two files; this is an assumption."],
    unit_note="recording (one 10.24 s trial excerpt)",
)
stdlib.release(ID, "timeseries", D, meta, checks, participants=part, participant_columns=pcols, extra_files=extra)
