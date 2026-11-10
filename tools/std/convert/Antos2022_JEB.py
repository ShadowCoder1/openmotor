"""Antos, Kording & Gordon (2022) J Exp Biol: energy cost and preference of equally preferred step length-width patterns, 15 adults.

Raw files: raw_extra/Antos2022_JEB/data.csv and README.txt (Zenodo 6354902, the Dryad deposit's copy, CC0). data.csv has one line per
participant and stepping pattern (15 x 5) with 10 value columns (README: the order of the lines is the order experienced).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Antos2022_JEB"
MODULE = "summary"
M = {
    "energy": ("Net metabolic power of the 6-minute walking trial, normalized by body mass", "value", "W/kg"),
    "RPE": ("Borg rating of perceived exertion after the trial (6 no exertion to 20 maximum exertion)", "value", "Borg 6-20"),
    "preferenceRank": ("Participant's ranking of the Short, Wide and Long patterns from most preferred (1) to least preferred (3)", "value", "rank"),
    "energyRank": ("Participant's ranking of which pattern they believed was most (1) to least (3) energy efficient", "value", "rank"),
    "SL": ("Step length of the pattern: baseline step length for Baseline and BaselineTargets; 75% of baseline for Short; the equally preferred step length estimated in the two-alternative forced-choice task for Wide and Long (the length walked in the 6-minute trial)", "value", "m"),
    "SL_LB": ("'Lower bound of the 95% confidence interval' of the step length, as named by the authors (values are kept as given; for Wide and Long they are far below SL, so they look like distances from the estimate rather than absolute bounds)", "value", "m"),
    "SL_UB": ("'Upper bound of the 95% confidence interval' of the step length, as named by the authors (kept as given; see SL_LB)", "value", "m"),
    "SW": ("Step width of the pattern: baseline step width for Baseline and BaselineTargets and for Short and Long; the equally preferred step width from the two-alternative forced-choice task for Wide", "value", "m"),
    "SW_LB": ("'Lower bound of the 95% confidence interval' of the step width, as named by the authors (kept as given; see SL_LB)", "value", "m"),
    "SW_UB": ("'Upper bound of the 95% confidence interval' of the step width, as named by the authors (kept as given; see SL_LB)", "value", "m"),
}


def main():
    d = pd.read_csv(stdlib.raw_dir(DS) / "data.csv")
    ck = stdlib.Checks()
    d["order"] = d.groupby("subject").cumcount() + 1
    rows = []
    for r in d.itertuples(index=False):
        rec = r._asdict()
        base = dict(dataset=DS, study="Antos 2022 J Exp Biol", experiment="step length-width preference and energy", group=None,
                    subject=f"S{int(rec['subject']):02d}", person=f"S{int(rec['subject']):02d}", subject_excluded_by_authors=False,
                    age=None, sex=None, handedness=None, population="healthy", setting="lab",
                    device="treadmill with stepping-target visual feedback, indirect calorimetry", session=1, block=None,
                    trial=int(rec["order"]), phase="other", time_s=None, task="walking", unit="participant summary",
                    condition=rec["steppattern"], pert_type="none" if rec["steppattern"] == "Baseline" else "step_target",
                    pert_value=None, pert_unit=None, feedback="normal" if rec["steppattern"] == "Baseline" else "altered",
                    excluded_by_authors=False)
        for k, (desc, stat, unit) in M.items():
            v = rec[k]
            if k in ("preferenceRank", "energyRank") and rec["steppattern"] not in ("Short", "Wide", "Long"):
                continue  # README: ranks apply only to Short, Wide and Long (0 elsewhere means not applicable)
            if pd.isna(v):
                continue
            rows.append({**base, "measure": k, "measure_description": desc, "statistic": stat, "value": v, "value_unit": unit})
    ck.internal("75 lines (15 participants x 5 patterns)", len(d) == 75 and d.groupby("subject").size().eq(5).all())
    ck.internal("no missing values", not d.isna().any().any())
    ck.internal("rows = values minus 60 baseline rank cells", len(rows) == 75 * 10 - 60, str(len(rows)))
    ck.internal("ranks are 1-3 permutations per participant", all(sorted(g.preferenceRank[g.steppattern.isin(["Short", "Wide", "Long"])]) == [1, 2, 3] for _, g in d.groupby("subject")))
    w = d.pivot(index="subject", columns="steppattern")
    ck.published("participants", d.subject.nunique(), 15, 0, "Methods, Participants (15 healthy young adults)")
    ck.published("Short step length / baseline step length (mean over participants)", (w["SL"]["Short"] / w["SL"]["Baseline"]).mean(), 0.75, 0.002,
                 "Methods, Experimental protocol (Short = 75% of baseline step length)")
    ck.published("Short step width / baseline step width (mean over participants)", (w["SW"]["Short"] / w["SW"]["Baseline"]).mean(), 1.0, 0.005,
                 "Methods, Experimental protocol (Short keeps 100% of baseline step width)")
    meta = dict(
        study=DS,
        paper=dict(authors="Antos SA, Kording KP, Gordon KE", year=2022,
                   title="Energy expenditure does not solely explain step length-width choices during walking",
                   venue="Journal of Experimental Biology 225(6):jeb243104", doi="10.1242/jeb.243104"),
        lab="Gordon (Northwestern University)", data_url="https://doi.org/10.5061/dryad.b2rbnzsgk (copy: https://zenodo.org/records/6354902)",
        license="CC0 1.0 Universal (CC0 1.0) Public Domain Dedication", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Turned the one-line-per-participant-and-pattern table into a long table with one row per participant, stepping "
                 "pattern and measure; the rank cells that are 0 for Baseline and BaselineTargets (not applicable) were not written; "
                 "participant numbers were re-coded; trial is the order in which the participant experienced the pattern."),
        how=["One row is one measure of one participant in one of five 6-minute walking patterns on a treadmill (Baseline, BaselineTargets, "
             "Short, Wide, Long). The deposit shares per-pattern values only: net metabolic power (W/kg), Borg RPE, preference and "
             "perceived-energy ranks, and the step length and width of the pattern with the authors' interval values.",
             "Step lengths and widths are in metres. The interval columns SL_LB/SL_UB/SW_LB/SW_UB are described as 95% CI bounds but "
             "for Wide and Long they are smaller than the estimate by an order of magnitude, so they are kept as given and named as the authors named them.",
             "Age, sex and mass are not in the deposit (the authors left demographics out), so the participant columns are empty."],
        unit_note="measure of one participant in one stepping pattern",
        differences=["The paper (text) gives no numbers for energy, RPE or step length-width, only figures, so those could not be "
                     "checked; the published checks are the participant count and the protocol definitions of the Short pattern."])
    stdlib.release(DS, MODULE, rows, meta, ck)


if __name__ == "__main__":
    main()
