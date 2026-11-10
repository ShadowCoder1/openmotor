"""Guerin, Boitout & Delevoye-Turrell (2021), Front Psychol 11:574396: right index finger tapping on a touchscreen to a
metronome (one central target or six targets on a circle) at five inter-stimulus intervals; Study 1 (25 adults): tapping accuracy and
lag-1 autocorrelation; Study 2 (25 other adults): the same tapping while detecting tones with the left hand (dual task).

Raw files, raw_extra/Guerin2021_FrontPsychol/ (OSF 9vg53): study1.xlsx, study2.xlsx (guerin_data_file_study1/2.xlsx), one sheet per
measure, one row per participant, one column per ISI x target number (T<ISI ms>.N<targets>).
Paper read at frontiersin.org (open access).
Run from the site folder: python3 -I tools/std/convert/Guerin2021_FrontPsychol.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Guerin2021_FrontPsychol"
P = "Guerin et al. 2021, "
# sheet -> (measure, description, statistic, unit)
MEAS = {
    "Asynchrony": ("asynchrony", "Signed asynchrony between tap and metronome beep (tap minus beep; negative: tap before the beep), averaged over the taps of the trial, as in the authors' sheet 'Asynchrony'", "mean", "s"),
    "Absolute asynchrony": ("absolute_asynchrony", "Absolute tap-to-beep time difference, averaged over the taps of the trial (the paper divides it by the ISI to give its relative asynchrony), as in the authors' sheet 'Absolute asynchrony'", "mean", "s"),
    "Spatial error": ("spatial_error", "Distance between the target centre and the fingertip location, averaged over the taps of the trial", "mean", "pixels"),
    "Detrended Lag-1 ACs": ("lag1_autocorrelation", "Mean windowed lag-1 autocorrelation of the detrended inter-response intervals (30-interval windows shifted one interval at a time); negative values indicate predictive timing", "mean", "correlation"),
    "Pressure": ("tap_pressure", "Touchscreen pressure of the taps, normalised to 0 (none) to 1, averaged over the trial", "mean", "normalised 0-1"),
    "Mean Delta TR": ("dual_task_reaction_time", "Reaction time to the secondary tone detection while tapping, as a percentage of the participant's reference (shortest single-task) reaction time (the authors' Delta TR); 100 is no cost", "mean", "percent of reference"),
}
SHEETS = {1: ["Asynchrony", "Absolute asynchrony", "Spatial error", "Detrended Lag-1 ACs"],
          2: ["Asynchrony", "Absolute asynchrony", "Pressure", "Mean Delta TR"]}


def build():
    rows, tabs = [], {}
    for st in (1, 2):
        x = pd.read_excel(stdlib.raw_dir(ID) / f"study{st}.xlsx", sheet_name=None, header=0, index_col=0)
        assert set(x) == set(SHEETS[st]), (st, list(x))
        for sheet in SHEETS[st]:
            tabs[(st, sheet)] = x[sheet]
        ids = list(x["Asynchrony"].index)
        for sid in ids:
            for sheet in SHEETS[st]:
                m, desc, stat, unit = MEAS[sheet]
                for col, v in x[sheet].loc[sid].items():
                    isi, nt = col[1:].split(".N")
                    rows.append({
                        "dataset": ID, "study": ID, "experiment": f"Study {st} ({'metronome tapping' if st == 1 else 'tapping with a tone detection task'})",
                        "group": "", "subject": f"S{st}_{sid}", "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "right",
                        "population": "healthy", "setting": "lab", "device": "touchscreen, right index finger", "session": 1, "block": "",
                        "trial": 0, "phase": "test", "time_s": "", "task": "finger tapping and timing", "unit": "participant summary",
                        "condition": f"ISI {isi} ms, {nt} target{'s' if nt != '1' else ''}", "pert_type": "none", "pert_value": 0.0,
                        "pert_unit": "", "feedback": "normal", "excluded_by_authors": False, "measure": m, "measure_description": desc,
                        "statistic": stat, "value": v, "value_unit": unit, "n_trials": "",
                        "session_context": "metronome beeps 80 ms at 294 Hz, 180 beeps per trial" if st == 1 else "metronome beeps plus a left-hand tone detection task, 60 beeps per trial"})
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, tabs


def main():
    checks = stdlib.Checks()
    a, tabs = build()
    checks.published("Study 1 participants", a[a.experiment.str.startswith("Study 1")].subject.nunique(), 25, 0, P + "Study 1 Participants")
    checks.published("Study 2 participants", a[a.experiment.str.startswith("Study 2")].subject.nunique(), 25, 0, P + "Study 2 Participants")
    se = tabs[(1, "Spatial error")]
    for isi, v1, v6 in ((300, 7.8, 16.3), (450, 7.7, 10.5), (600, 8.4, 9.1), (800, 8.4, 7.8), (1000, 8.8, 7.8)):
        checks.published(f"Study 1 spatial error, 1 target, ISI {isi} (px)", se[f"T{isi}.N1"].mean(), v1, 0.05, P + "Table 1")
        checks.published(f"Study 1 spatial error, 6 targets, ISI {isi} (px)", se[f"T{isi}.N6"].mean(), v6, 0.05, P + "Table 1")
    ac = tabs[(1, "Detrended Lag-1 ACs")]
    checks.published("Study 1 lag-1 autocorrelation at ISI 1000, mean of both tasks", ac[["T1000.N1", "T1000.N6"]].mean().mean(), -0.23, 0.005, P + "Study 1 Results (M = -0.23)")
    c8 = [f"T{i}.N{n}" for i in (300, 450, 600, 800) for n in (1, 6)]
    checks.published("Study 1 lag-1 autocorrelation at ISI 300-800, mean", ac[c8].mean().mean(), -0.15, 0.005, P + "Study 1 Results (M = -0.15)")
    pr = tabs[(2, "Pressure")]
    checks.published("Study 2 pressure, ISI 300, 6 targets", pr["T300.N6"].mean(), 0.43, 0.005, P + "Study 2 Results (M = 0.43)")
    checks.published("Study 2 pressure, ISI 300, 1 target", pr["T300.N1"].mean(), 0.49, 0.005, P + "Study 2 Results (M = 0.49)")
    checks.published("Study 2 pressure, ISI 1100, 6 targets", pr["T1100.N6"].mean(), 0.54, 0.005, P + "Study 2 Results (M = 0.54)")
    checks.published("Study 2 pressure, ISI 1100, 1 target", pr["T1100.N1"].mean(), 0.49, 0.005, P + "Study 2 Results (M = 0.49)")
    rt = tabs[(2, "Mean Delta TR")]
    checks.published("Study 2 dual-task RT, 1 target, mean over ISIs (%)", rt[[c for c in rt.columns if c.endswith("N1")]].mean().mean(), 164.22, 0.005, P + "Study 2 Results (M = 164.22)")
    checks.published("Study 2 dual-task RT, 6 targets, mean over ISIs (%)", rt[[c for c in rt.columns if c.endswith("N6")]].mean().mean(), 177.30, 0.005, P + "Study 2 Results (M = 177.30)")
    aa = tabs[(1, "Absolute asynchrony")]
    checks.published("Study 1 relative asynchrony, 1 target, ISI 300 (absolute asynchrony / ISI x 1000)", aa["T300.N1"].mean() / 0.3 * 1000, 110, 0.5, P + "Table 1")
    checks.published("Study 1 relative asynchrony, 6 targets, ISI 1000 (absolute asynchrony / ISI x 1000)", aa["T1000.N6"].mean() / 1.0 * 1000, 120, 0.5, P + "Table 1")
    checks.internal("40 values per participant", (a.groupby("subject").size() == 40).all(), "4 sheets x 10 conditions")
    checks.internal("no missing values", a.value.notna().all())
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {q: p0[q] for q in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="GNU General Public License (GPL) 3.0 (as set on the OSF project)",
        license_url="https://www.gnu.org/licenses/gpl-3.0.html",
        changes=("Reshaped the authors' two workbooks (one sheet per measure) into one long table, split the column names into the metronome "
                 "interval and the number of targets, and replaced participant codes. Values are as the authors shared them."),
        how=[
            "One row is one measure of one participant in one condition. Each condition is a metronome inter-stimulus interval (ISI, 300 "
            "to 1,100 ms) and a tapping task with one central target (N1) or six targets on a circle (N6); the paper describes 10 trials "
            "per session, one for each ISI x task cell, and each value is the average over that trial's taps (the deposit does not give a trial "
            "count, so n_trials is empty).",
            "Study 1 and Study 2 were different groups of 25 adults (the paper tests the difference between them), so the S01 to S25 codes of "
            "the two workbooks are different people. Age and sex are not in the deposit; handedness (right) and population (healthy) are from "
            "the paper. Asynchrony and absolute asynchrony are in seconds in the workbooks (inferred: the paper's relative asynchrony, "
            "absolute asynchrony divided by the ISI, reproduces its Table 1 when multiplied by 1,000); spatial error is in pixels, pressure "
            "is normalised 0 to 1, and Delta TR is a percentage of the participant's reference reaction time.",
            "No participant was excluded in the shared files (25 per study, matching the paper's samples).",
        ],
        unit_note="participant, measure and condition",
        differences=["The paper's Table 1 relative asynchrony is not in the workbooks; it was rebuilt from absolute asynchrony / ISI x 1000, and only two cells are checked for that reason.",
                     "Not used as checks: the paper's pooled ISI means for dual-task reaction time (for example 165.14 at 300/500 ms) did not reproduce from a simple pooling of the workbook columns (we get 165.24), and the paper's Study 2 spatial error cells have no sheet in the Study 2 workbook."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
