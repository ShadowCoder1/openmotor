"""Maurer, Joch, Hegele, Maurer & Muller (2022), Neuroscience 486:77-90: per-participant, per-session performance in a virtual
throwing task practised over ten sessions (16 participants). EEG files of the deposit are not part of this release.

Run from the site folder: python3 -I tools/std/convert/Maurer2022_Neuroscience.py
Raw files (untrusted data, read only): raw_extra/Maurer2022_Neuroscience/{performance.mat, Mean_Ang_Velo.txt, SD_Ang_Velo.txt}
from https://zenodo.org/records/5006535 (descriptions in the Description_*.txt files of the deposit).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Maurer2022_Neuroscience"
R = stdlib.raw_dir(ID)
checks = stdlib.Checks()

perf = sio.loadmat(R / "performance.mat")
hit, dmin = perf["hit_rates"], perf["d_values"]
mean = pd.read_csv(R / "Mean_Ang_Velo.txt", sep="\t")
sd = pd.read_csv(R / "SD_Ang_Velo.txt", sep="\t")
mean = mean.loc[:, ~mean.columns.str.startswith("Unnamed")]
sd = sd.loc[:, ~sd.columns.str.startswith("Unnamed")]
NP, NS = 16, 10
checks.internal("all tables are 16 participants x 10 sessions",
                hit.shape == dmin.shape == (NP, NS) and mean.shape == sd.shape == (NP, 2 * NS))
checks.internal("hit rates between 0 and 100 (percent), Dmin positive", bool((hit >= 0).all() and (hit <= 100).all() and (dmin > 0).all()))
checks.internal("no missing values", not (np.isnan(hit).any() or np.isnan(dmin).any() or mean.isna().any().any() or sd.isna().any().any()))

MEASURES = [
    ("hit_rate", "mean", "percent", "Percentage of the session's throws that hit the target (deposit: 'hitrate of each session').", hit),
    ("dmin", "mean", "unit not stated",
     "Average over the session's throws of Dmin, the minimum distance between the centre of the ball's flight trajectory and the "
     "centre of the target (deposit: 'average Dmin value of each session'); the deposit does not state its unit.", dmin),
    ("release_angle", "mean", "deg", "Mean over the session's throws of the angular position of the throwing lever at the moment of release.",
     mean[[f"A{s}" for s in range(1, NS + 1)]].to_numpy()),
    ("release_angle_sd", "sd", "deg", "Standard deviation over the session's throws of the lever's release angle.",
     sd[[f"SA{s}" for s in range(1, NS + 1)]].to_numpy()),
    ("release_velocity", "mean", "m/s", "Mean over the session's throws of the velocity of the end point of the throwing lever at release.",
     mean[[f"V{s}" for s in range(1, NS + 1)]].to_numpy()),
    ("release_velocity_sd", "sd", "m/s", "Standard deviation over the session's throws of the lever's release velocity.",
     sd[[f"SV{s}" for s in range(1, NS + 1)]].to_numpy()),
]
rows = []
for p in range(NP):
    for s in range(NS):
        for name, stat, unit, desc, arr in MEASURES:
            rows.append(dict(
                dataset=ID, study="Maurer et al. 2022 Neuroscience", experiment="Exp1", group="", subject=f"P{p + 1}",
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab",
                device="throwing lever (manipulandum) with a virtual ball-and-target display", session=s + 1,
                block=f"session {s + 1}", trial=len(rows) + 1, phase="learning", time_s="", task="sports and games",
                unit="participant summary", condition="", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                excluded_by_authors=False, measure=name, measure_description=desc, statistic=stat, value=float(arr[p, s]),
                value_unit=unit, n_trials=400, session_context=""))
# trial counts the rows of each participant from 1
t = pd.DataFrame(rows)
t["trial"] = t.groupby("subject").cumcount() + 1
checks.internal("960 rows = 16 participants x 10 sessions x 6 measures", len(t) == 960)

checks.published("number of participants", t.subject.nunique(), 16, 0, "Abstract ('16 participants')")
checks.published("number of sessions", t.session.nunique(), 10, 0, "Abstract ('ten sessions')")
checks.published("total number of practice trials per participant (10 sessions x 400 trials per session)",
                 t[t.measure == "hit_rate"].groupby("subject").n_trials.sum().iloc[0], 4000, 0,
                 "Abstract ('4000 trials over ten sessions'); 400 trials per session = 4 blocks of 100 from the deposit's EEG description")
h = t[t.measure == "hit_rate"].pivot(index="subject", columns="session", values="value")
checks.internal("hit rate rose from session 1 to session 10 in the group mean (the paper reports significant improvement)",
                h[10].mean() > h[1].mean(), f"{h[1].mean():.1f} -> {h[10].mean():.1f}")

meta = dict(
    study="Maurer et al. 2022",
    paper=dict(authors="Maurer LK, Joch M, Hegele M, Maurer H, Muller H", year=2022,
               title="Relevance of Predictive and Postdictive Error Information in the Course of Motor Learning",
               venue="Neuroscience 486:77-90", doi="10.1016/j.neuroscience.2021.05.007"),
    lab="Muller (Justus Liebig University Giessen)", data_url="https://zenodo.org/records/5006535",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor summary format in long form (one row per participant, session and measure); participants "
             "coded by their row in the authors' tables. The deposit's EEG data (EEG_data.zip) are not included."),
    how=["One row is one measure of one participant in one session of a virtual throwing task practised over ten sessions "
         "(400 throws per session, four blocks of 100, as in the paper's 4000 trials). The six measures are the hit rate, the "
         "average minimum distance of the ball trajectory from the target (dmin), and the mean and SD of the throwing lever's "
         "release angle (deg) and release velocity (m/s), exactly as the authors summarized them per session.",
         "Hit rate is a percentage. The deposit does not state the unit of dmin, so it is given as 'unit not stated'.",
         "The deposit's tables have no participant labels; participants are numbered by row and the same row order is used in all "
         "three files (the authors describe each as 'all 16 participants (rows)')."],
    unit_note="measure of one participant in one session",
    differences=["The deposit holds only per-session summaries; the trial-level throws are inside the EEG files and not released here.",
                 "The paper's full text could not be read (Elsevier, not open access), so the published checks are limited to the "
                 "counts in its abstract."],
)
stdlib.release(ID, "summary", t.to_dict("records"), meta, checks)
