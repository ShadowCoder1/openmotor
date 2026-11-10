"""Martel & Madelain (2025) J Neurophysiol: ineffective cues for contextual saccade adaptation.

Raw file: raw_extra/Martel2025_JNeurophysiol/table.xlsx (Recherche Data Gouv doi:10.57745/SLOUNH, the .tab original),
one row per participant (106), 18 columns described in the deposit's ReadMe.txt. Nothing from the deposit is executed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Martel2025_JNeurophysiol"
MODULE = "summary"
EXP = {"FSA": 1, "C&S": 2, "C&S_PR": 3, "Sbl": 4, "Sbl_PR": 5, "Duration": 6, "Sound": 7, "Stat1": 8, "Stat4": 9,
       "Stat10": 10, "Pos.": 11}
DIRS = {1: "north-west intrasaccadic step", 2: "south-east intrasaccadic step"}
PER_DIR = [  # (column stem, measure, description, statistic, unit)
    ("MedBL", "baseline_median_saccade_angle", "Baseline median saccade angle", "median", "deg"),
    ("MedLateLearn", "late_learning_median_saccade_angle", "Late-learning median saccade angle", "median", "deg"),
    ("NbLateLearn", "late_learning_n_saccades", "Number of correct saccades used for the late-learning analysis", "count", "count"),
    ("FirstPctlAPP", "late_learning_first_quartile", "First quartile of the late-learning saccade angles", "other", "deg"),
    ("LastPctlAPP", "late_learning_last_quartile", "Last quartile of the late-learning saccade angles", "other", "deg")]
# The ReadMe pairs FirstPctlAPP1/LastPctlAPP1 with north-west and 2 with south-east. The header order in the file is
# FirstPctlAPP1, FirstPctlAPP2, LastPctlAPP2, LastPctlAPP1; columns are read by name.
PER_PART = [("DiffMedian", "late_learning_median_difference", "Difference between the late-learning medians of the two step directions", "value", "deg"),
            ("P", "bootstrap_confidence_value", "97.5% confidence value of the late-learning median from the bootstrap (one-tailed), as labelled by the authors", "value", "deg"),
            ("Signif", "bootstrap_difference_significant", "Significance of the difference between the late-learning medians (0 = not significant, 1 = significant)", "value", "0/1"),
            ("KS_H", "ks_test_significant", "Significance of the Kolmogorov-Smirnov test on the late-learning distributions (1 = significant)", "value", "0/1"),
            ("KSscore", "ks_statistic", "Kolmogorov-Smirnov test statistic", "value", "statistic"),
            ("KS_P", "ks_p_value", "Kolmogorov-Smirnov p-value", "value", "p")]


def main():
    d = pd.read_excel(stdlib.raw_dir(DS) / "table.xlsx")
    ck = stdlib.Checks()
    rows = []
    for _, r in d.iterrows():
        e = EXP[r["Groupe"]]
        sub = f"E{e}_{int(r['Num'])}"
        base = dict(dataset=DS, study="Martel 2025 J Neurophysiol", experiment=f"Experiment {e} ({r['Groupe']})", group=None,
                    subject=sub, person=sub, subject_excluded_by_authors=False, age=None, sex=None, handedness=None,
                    population="healthy", setting="lab", device="EyeLink 1000 Plus Tower Mount, 2000 Hz", session=None,
                    block="late learning", trial=None, phase="other", time_s=None, task="eye movements",
                    unit="participant summary", pert_type="intrasaccadic_step", pert_unit=None, pert_value=None,
                    feedback="normal", excluded_by_authors=False, n_trials=None)
        for k, name in DIRS.items():
            for stem, meas, desc, st, unit in PER_DIR:
                rows.append({**base, "condition": name, "measure": meas, "measure_description": desc, "statistic": st,
                             "value": r[f"{stem}{k}"], "value_unit": unit,
                             "n_trials": r[f"NbLateLearn{k}"] if stem != "NbLateLearn" else None})
        for col, meas, desc, st, unit in PER_PART:
            rows.append({**base, "condition": "both directions", "measure": meas, "measure_description": desc,
                         "statistic": st, "value": float(r[col]), "value_unit": unit})
    ck.internal("every participant row kept: 106 x (2 x 5 + 6) rows", len(rows) == 106 * 16, str(len(rows)))
    ck.internal("participant numbers are unique within each experiment", not d.duplicated(["Groupe", "Num"]).any())
    ck.internal("the 11 experiments are all present", d["Groupe"].nunique() == 11)
    ck.published("participants", len(d), 106, 0, "Dataset ReadMe (106 participants included; the abstract's design has the same experiments)")
    ck.published("experiments", d["Groupe"].nunique(), 11, 0, "Dataset ReadMe (11 experimental conditions); abstract: nine cue types plus two report experiments")
    meta = dict(
        study=DS,
        paper=dict(authors="Martel M, Madelain L", year=2025, title="Ineffective cues for contextual saccade adaptation",
                   venue="J Neurophysiol 134(1):237-249", doi="10.1152/jn.00148.2025"),
        lab="Madelain (Universite de Lille, SCALab)", data_url="https://doi.org/10.57745/SLOUNH",
        license="Etalab Open Licence 2.0", license_url="https://www.etalab.gouv.fr/licence-ouverte-open-licence/",
        changes=("Turned the participant table (one row, 18 columns) into long form with one row per measure, with participant "
                 "codes and experiment names; values are unchanged."),
        how=["One row is one measure of one participant in one of 11 double-step experiments with contextual cues: for each of "
             "the two intrasaccadic step directions (north-west, south-east) the baseline median, the late-learning median, the "
             "number of saccades analysed and the first and last quartile of the late-learning saccade angles, then six values "
             "per participant that compare the two directions (median difference, bootstrap value, significance, Kolmogorov-Smirnov "
             "statistic and p). The deposit has no individual trials.",
             "The authors label the angles dva; the baseline values (about 42 on average) and the baseline-subtracted late-learning "
             "values are kept as given and listed as deg here, which is our reading. Quartile statistic is 'other'.",
             "Participant numbers repeat across experiments in the file (82 distinct numbers in 106 rows). The deposit does not say "
             "whether a repeated number is the same person, so each experiment's participant is coded separately."],
        unit_note="measure of one participant in one experiment",
        differences=["The paper's full text was not accessible (publisher and HAL block automated access), so the checks use the counts "
                     "given in the dataset ReadMe: 106 participants and 11 experiments."],
    )
    stdlib.release(DS, MODULE, rows, meta, ck)


if __name__ == "__main__":
    main()
