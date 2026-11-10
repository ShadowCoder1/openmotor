"""Lantagne et al. (2026) J Motor Behav: concussion and adaptation to unpredictable spring-like loads (pilot study).

Raw file: raw_extra/Lantagne2026_JMotBehav/SupplementalTable2.csv (Figshare 32157550): one row per participant and session
(107 lines, 11 of them completely empty), 70 columns described in SupplementalTable2Description.pdf. Nothing is executed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Lantagne2026_JMotBehav"
MODULE = "summary"

KIN = {
    "k_ReachN": ("Number of reach trials of the session that did not meet the authors' exclusion criteria (of 200)", "count", "trials"),
    "k_MeanReachError": ("Mean reach error over the session's trials", "mean", "m"),
    "k_StDevReachError": ("SD of reach error across the session's trials", "sd", "m"),
    "k_MeanTargCaptureTime": ("Mean time from movement onset to the turnaround point of the reach", "mean", "ms"),
    "k_MeanTargCaptureTime_TX": ("log10 of the mean target capture time", "other", "log10 ms"),
    "k_StDevTargCaptureTime": ("SD of target capture time across the session's trials", "sd", "ms"),
    "k_StDevTargCaptureTime_TX": ("log10 of the SD of target capture time", "other", "log10 ms"),
    "k_MeanReactionTime": ("Mean reaction time (GO cue to movement onset) over the session's trials", "mean", "ms"),
    "k_MeanReactionTime_TX": ("log10 of the mean reaction time", "other", "log10 ms"),
    "k_StDevReactionTime": ("SD of reaction time across the session's trials", "sd", "ms"),
    "k_StDevReactionTime_TX": ("log10 of the SD of reaction time", "other", "log10 ms"),
    "m_a1_Estimate": ("Memory-model coefficient a1: contribution of the prior reach error to the current reach error "
                      "(multilinear regression over the session's trials)", "value", "arbitrary units"),
    "m_b0_Estimate": ("Memory-model coefficient b0: contribution of the current spring strength to the current reach error",
                      "value", "m2/N"),
    "m_b1_Estimate": ("Memory-model coefficient b1: contribution of the prior spring strength to the current reach error",
                      "value", "m2/N"),
    "c_Average_Spd_Z": ("Average of the four CogState speed z-scores (z relative to the healthy group within session); the "
                        "authors' composite used to split the concussed group", "value", "z"),
}
TESTS = {"DET": "Detection", "IDN": "Identification", "ONB": "One-Back", "OCL": "One-Card Learning"}
OUTC = {"Spd_ms": ("speed (reaction time)", "ms"), "Spd_lmn": ("speed, log10 transform", "log10 ms"),
        "Spd_Z": ("speed z-score relative to the healthy group within session", "z"),
        "Acc_per": ("accuracy, proportion correct", "proportion"), "Acc_asr": ("accuracy, arcsine of the square root of the proportion", "rad"),
        "Acc_Z": ("accuracy z-score relative to the healthy group within session", "z")}
for t, tn in TESTS.items():
    for o, (od, ou) in OUTC.items():
        KIN[f"c_{t}_{o}"] = (f"CogState {tn} test, {od}", "value", ou)
SCAT = [c for c in pd.read_csv(stdlib.raw_dir(DS) / "SupplementalTable2.csv", nrows=0).columns if c.startswith("scat_")] \
    if (stdlib.raw_dir(DS) / "SupplementalTable2.csv").exists() else []


def main():
    d = pd.read_csv(stdlib.raw_dir(DS) / "SupplementalTable2.csv")
    n_lines = len(d)
    d = d.dropna(how="all").reset_index(drop=True)
    ck = stdlib.Checks()
    ck.internal("completely empty lines dropped", n_lines - len(d) == 11, f"{n_lines} lines, {len(d)} kept")
    ck.internal("one row per participant and session", not d.duplicated(["SubID", "Session"]).any() and len(d) == 96)
    for c in SCAT:
        KIN[c] = ("Sport Concussion Assessment Tool symptom severity (0 none to 6 severe)" if c not in ("scat_Sum", "scat_Count") else
                  ("SCAT sum of the 22 symptom severities (of 132)" if c == "scat_Sum" else "SCAT number of symptoms experienced (of 22)"),
                  "value", "score" if c != "scat_Count" else "symptoms")
        if c not in ("scat_Sum", "scat_Count"):
            KIN[c] = (f"SCAT symptom severity, {c[5:]} (0 none to 6 severe)", "value", "score")
    missing = [c for c in d.columns if c not in KIN and c not in
               ("SubID", "Group", "GroupCode", "GroupSplit_AvRT_Median", "Session", "Sex", "DaysPostInjury")]
    assert not missing, missing
    KIN["DaysPostInjury"] = ("Days between the concussion injury and the session", "value", "days")

    split = {0: "healthy", 1: "C- (concussed, composite reaction-time z-score below the median)",
             2: "C+ (concussed, composite reaction-time z-score above the median)"}
    rows = []
    for r in d.itertuples(index=False):
        rec = r._asdict()
        conc = rec["Group"] == "Concussed"
        base = dict(dataset=DS, study="Lantagne 2026 J Motor Behav", experiment="spring-like load adaptation",
                    group=rec["Group"], subject=rec["SubID"], person=rec["SubID"], subject_excluded_by_authors=False,
                    age=None, sex=rec["Sex"], handedness=None, population="concussion" if conc else "healthy", setting="lab",
                    device="planar robot handle (2D), CogState computerized tests", session=int(rec["Session"]), block=None,
                    trial=None, phase="other", time_s=None, task="reaching", unit="participant summary",
                    condition=split[int(rec["GroupSplit_AvRT_Median"])], pert_type="force_field", pert_value=None,
                    pert_unit="N/m", feedback="normal", excluded_by_authors=False)
        for k, (desc, stat, unit) in KIN.items():
            v = rec[k]
            if pd.isna(v):
                continue
            rows.append({**base, "measure": k, "measure_description": desc, "statistic": stat, "value": v, "value_unit": unit})
    ck.internal("rows = non-empty cells", len(rows) == int(d[list(KIN)].notna().sum().sum()), str(len(rows)))
    ck.internal("32 participants kept", d.SubID.nunique() == 32)

    ses1 = d[d.Session == 1]
    c1 = ses1[ses1.Group == "Concussed"]
    # published numbers (bioRxiv preprint of the paper, Methods and Results; the published version could not be read)
    ck.published("healthy participants analysed", d[d.Group == "Healthy"].SubID.nunique(), 18, 0, "Methods, Participants (preprint)")
    ck.published("concussed participants analysed", d[d.Group == "Concussed"].SubID.nunique(), 14, 0, "Methods, Participants (preprint)")
    ck.published("mean days post injury at session 1", c1.DaysPostInjury.mean(), 5.9, 0.05, "Methods, Participants (preprint)")
    ck.published("SD of days post injury at session 1", c1.DaysPostInjury.std(), 4.1, 0.05, "Methods, Participants (preprint)")
    ck.published("minimum days post injury at session 1", c1.DaysPostInjury.min(), 0, 0, "Methods, Participants (preprint, range 0-17)")
    ck.published("maximum days post injury at session 1", c1.DaysPostInjury.max(), 17, 0, "Methods, Participants (preprint, range 0-17)")
    ck.published("mean SCAT symptom count at session 1", c1.scat_Count.mean(), 17.8, 0.05, "Results, symptoms (preprint)")
    ck.published("SD of SCAT symptom count at session 1", c1.scat_Count.std(), 4.2, 0.05, "Results, symptoms (preprint)")
    ck.published("mean SCAT symptom severity sum at session 1", c1.scat_Sum.mean(), 59.0, 0.05, "Results, symptoms (preprint)")
    ck.published("SD of SCAT symptom severity sum at session 1", c1.scat_Sum.std(), 26.2, 0.05, "Results, symptoms (preprint)")
    h4 = d[(d.SubID == "H4") & (d.Session == 1)].iloc[0]  # the preprint's single-subject control example (reach 3.2 +- 1.8 cm, capture 201 +- 8 ms)
    ck.published("example control: SD of target capture time (ms)", h4.k_StDevTargCaptureTime, 8, 0.5, "Results, example subject figure (preprint)")
    ck.published("example control: mean reach error (cm)", h4.k_MeanReachError * 100, 3.2, 0.05, "Results, example subject figure (preprint)")
    ck.published("example control: SD of reach error (cm)", h4.k_StDevReachError * 100, 1.8, 0.05, "Results, example subject figure (preprint)")

    meta = dict(
        study=DS,
        paper=dict(authors="Lantagne DD, Mrotek LA, Ahamed SI, Beardsley SA, Grove AJ, Leigh DH, Smith CS, Thomas DG, Scheidt RA",
                   year=2026, title="Effects of Concussion on Sensorimotor Adaptation to Unpredictable Spring-Like Loads: A Pilot Study",
                   venue="Journal of Motor Behavior 58(3):361-376", doi="10.1080/00222895.2026.2659046"),
        lab="Scheidt (Marquette University)", data_url="https://doi.org/10.6084/m9.figshare.32157550",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Dropped 11 completely empty lines, turned the one-row-per-session table into a long table with one row per "
                 "participant, session and measure (empty cells are not written), replaced participant codes, and moved "
                 "sex, group and the concussed-group split into the core columns."),
        how=["One row is one measure of one participant in one session (three sessions: as soon as possible after injury, on "
             "return to activity, and about three months later; controls at similar intervals). The deposit shares session-level "
             "values only, not trials: reach error, target capture time and reaction time of 200 out-and-back reaches against "
             "unpredictable spring-like robot loads, the three memory-model coefficients (a1, b0, b1), CogState test scores, "
             "SCAT symptom scores and days since injury. Units are those in the authors' description file (reach error in m, "
             "times in ms).",
             "statistic is mean, sd or count for the measures the description defines that way, value for coefficients, scores and "
             "z-scores, and other for log10 transforms. condition holds the authors' split of the concussed group (C-, C+) by "
             "their composite reaction-time z-score, or healthy. Sex is as given (F/M). pert_value is empty because the spring "
             "strength changed unpredictably from trial to trial.",
             "SCAT scores exist only for session 1 of the concussed group (one concussed participant has only the sum and count)."],
        unit_note="measure of one participant in one session",
        differences=["The 14 concussed participants split 7 and 7 at the median composite reaction-time score in the deposit, while "
                     "the preprint describes a split of 9 (tau-) and 5 (tau+) around the cohort mean; the deposit's split is kept as given.",
                     "The preprint's example control subject reports a mean target capture time of 201 ms; the matching deposit row has 201.8 ms (the paper seems to truncate), so that number is not used as a check. The example subject was identified by its other values (SD 8 ms, reach error 3.2 +- 1.8 cm).",
                     "The published article could not be read (publisher page blocked); numbers were checked against the bioRxiv "
                     "preprint of the same study."],
    )
    stdlib.release(DS, MODULE, rows, meta, ck)


if __name__ == "__main__":
    main()
