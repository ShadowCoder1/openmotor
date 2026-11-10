"""Forte et al. (2025), PLoS ONE 20(1):e0316545: 3D shape measures of the pots made by the training group (13 beginners) before and
after four weeks of pottery training. The deposit's TMS (MEP) data are not part of this release (not movement).

Run from the site folder: python3 -I tools/std/convert/Forte2025_PLoSOne.py
Raw file (untrusted data, read only): raw_extra/Forte2025_PLoSOne/measures_experimental_pots.csv from https://zenodo.org/records/14379018.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Forte2025_PLoSOne"
checks = stdlib.Checks()
raw = pd.read_csv(stdlib.raw_dir(ID) / "measures_experimental_pots.csv", sep=";")
checks.internal("26 pots, 13 before and 13 after", len(raw) == 26 and (raw.stage.value_counts() == 13).all())
raw["pid"] = raw["sample"].str.extract(r"exp_(\d+)\.")[0].astype(int)
raw["pot"] = raw["sample"].str.extract(r"\.(\d+)$")[0].astype(int)
checks.internal("one pot before and one after for each of 13 participants",
                raw.groupby("pid").stage.apply(lambda s: sorted(s) == ["post", "pre"]).all() and raw.pid.nunique() == 13)
MEAS = [("heigth", "pot_height", "Height of the pot (authors' heigth)."),
        ("thickness.base", "base_thickness", "Wall thickness at the centre of the base (authors' thickness.base)."),
        ("thickness.wall", "lower_wall_thickness", "Thickness of the lower wall (authors' thickness.wall)."),
        ("thickness.medium.wall", "medium_wall_thickness", "Thickness of the medium wall (authors' thickness.medium.wall)."),
        ("thickness.rim", "rim_thickness", "Thickness of the rim, the higher wall (authors' thickness.rim).")]
rows = []
for pid in sorted(raw.pid.unique()):
    for stage, label in (("pre", "before training"), ("post", "after training")):
        r = raw[(raw.pid == pid) & (raw.stage == stage)].iloc[0]
        for col, name, desc in MEAS:
            rows.append(dict(
                dataset=ID, study="Forte et al. 2025 PLoS ONE", experiment="Training group", group="", subject=f"P{pid}",
                subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy beginners in pottery",
                setting="lab", device="clay coil pots, 3D photogrammetry", session=1 if stage == "pre" else 2,
                block=stage, trial=0, phase="baseline" if stage == "pre" else "test", time_s="", task="sports and games",
                unit="participant summary", condition=label, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                excluded_by_authors=False, measure=name, measure_description=desc + " One pot per participant and stage.",
                statistic="value", value=float(r[col]), value_unit="mm", n_trials=1,
                session_context=f"pot number {int(r.pot)} of the participant's series (read from the sample label)"))
t = pd.DataFrame(rows)
t["trial"] = t.groupby("subject").cumcount() + 1
checks.internal("130 rows = 13 participants x 2 stages x 5 measures", len(t) == 130)

SRC = "Results, 3D morphometry (Table of pot measures before and after training)"
for (col, name, _), (m0, m1, p) in zip(MEAS, [(91.2, 121, 0.008842), (10.8, 13.0, 0.09866), (9.65, 12.8, 0.01797), (8.41, 10.7, 0.007504),
                                           (7.23, 8.96, 0.08572)]):
    a = raw[raw.stage == "pre"].set_index("pid")[col].sort_index()
    b = raw[raw.stage == "post"].set_index("pid")[col].sort_index()
    tol = lambda x: 0.5 * 10 ** -(len(str(x).split(".")[1]) if "." in str(x) else 0)
    checks.published(f"{name}: mean before training (mm)", a.mean(), m0, tol(m0), SRC)
    checks.published(f"{name}: mean after training (mm)", b.mean(), m1, tol(m1), SRC)
checks.published("training group analysed", raw.pid.nunique(), 13, 0, "Methods, Participants (13 analysed after one exclusion)")

meta = dict(
    study="Forte et al. 2025",
    paper=dict(authors="Forte V, Sartori L, Visalli A, Yildirim M, Galati G, Vidale M, Faresin E, Vallesi A", year=2025,
               title="Neural plasticity in early potters: Shape analysis and TMS-EEG co-registration trace the rise of a new motor skill",
               venue="PLoS ONE 20(1):e0316545", doi="10.1371/journal.pone.0316545"),
    lab="Sartori (University of Padua)", data_url="https://zenodo.org/records/14379018",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor summary format in long form (one row per participant, stage and measure); participant numbers "
             "taken from the pot labels and replaced by codes. The deposit's TMS motor-evoked-potential table (data_input.csv) is not included."),
    how=["One row is one shape measure (mm) of the one pot a participant of the training group made before training and the one pot "
         "made after four weeks of pottery training, as measured on 3D models of the pots; the authors shared 13 pots per stage.",
         "Pairs are matched by the participant number in the authors' sample labels (exp_N.k); the suffix k is read as the number of the pot in the "
         "participant's series (inferred from the labels, as the paper does not explain them) and is given in session_context."],
    unit_note="measure of one pot of one participant at one stage",
    differences=["The paper's paired t-test p values (for example 0.008842 for height) do not reproduce from these rows with a paired t-test "
                 "(0.0225 for height); the deposit's own R script (3D_Measures pots.R) runs unpaired Student's t-tests instead. "
                 "They were therefore not used as checks; the pre and post means all reproduce.",
                 "The paper does not give the number of pots measured; the deposit has 13 per stage, matching its 13 analysed participants.",
                 "The TMS-EEG data and the control group's pots are not part of this release."],
)
stdlib.release(ID, "summary", t.to_dict("records"), meta, checks)
