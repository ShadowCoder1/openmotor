"""Tumber, Scheerer & Jones (2014), PLoS ONE 9(10):e109968: vocal compensation to a brief pitch shift in auditory feedback
while attending to a rapid visual stream (dual task), 54 participants with usable pitch tracks.

Raw files, raw_extra/Tumber2014_PLoSOne/ (Borealis doi:10.5683/SP3/XD70OL):
  Vocal_Behavioural.xlsx   one row per participant: vocal response magnitude and latency, and two target-detection accuracies, per attentional load
  ERP.xlsx                 EEG amplitudes and latencies (not movement data; not converted)
Paper read in PubMed Central (PMC4193850).
Run from the site folder: python3 -I tools/std/convert/Tumber2014_PLoSOne.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tumber2014_PLoSOne"
P = "Tumber et al. 2014, "
LOAD = {"100ms": ("High attentional load (100 ms letter rate)", "100"), "300ms": ("Intermediate attentional load (300 ms letter rate)", "300"),
        "500ms": ("Low attentional load (500 ms letter rate)", "500"), "Control": ("Single task (baseline attentional load)", "")}
DUAL = {"100", "300", "500"}


def build():
    d = pd.read_excel(stdlib.raw_dir(ID) / "Vocal_Behavioural.xlsx", sheet_name="Data")
    d = d.replace(".", float("nan"))
    for c in d.columns[2:]:
        d[c] = pd.to_numeric(d[c])
    rows = []
    for r in d.to_dict("records"):
        grp = "Single-task block first" if r["BlockOrder"] == 1 else "Dual-task block first"

        def add(measure, desc, stat, val, unit, cond, dual):
            if pd.isna(val):
                return
            rows.append({
                "dataset": ID, "study": ID, "experiment": "Vocal compensation under attentional load", "group": grp,
                "subject": str(r["Participant"]), "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
                "population": "healthy", "setting": "lab", "device": "microphone and insert earphones with real-time pitch shifting",
                "session": 1, "block": "dual task" if dual else "single task", "trial": 0, "phase": "test", "time_s": "",
                "task": "speech", "unit": "participant summary", "condition": cond, "pert_type": "pitch_shift", "pert_value": -50.0,
                "pert_unit": "cents", "feedback": "altered", "excluded_by_authors": False, "measure": measure,
                "measure_description": desc, "statistic": stat, "value": val, "value_unit": unit, "n_trials": "", "session_context": ""})

        for k, (lab, rate) in LOAD.items():
            dual = k != "Control"
            add("response_magnitude", "Vocal response magnitude to the 50-cent downward pitch shift of 200 ms (the authors' RM), from the average f0 trace of the retained trials of the condition; positive means f0 was raised against the shift",
                "mean", r[f"RM_{k}"], "cents", lab, dual)
            add("response_latency", "Latency of the vocal response to the pitch shift (the authors' RL) in the average f0 trace; values of about 1000 mean no response was found within the 1000 ms window",
                "mean", r[f"RL_{k}"], "ms", lab, dual)
            if dual:
                add("white_letter_accuracy", "Proportion of trials on which the white target letter of the visual stream was correctly identified (the authors' WL)",
                    "proportion", r[f"WL_{rate}"], "proportion", lab, True)
                add("x_placement_accuracy", "Proportion of trials on which the position of the X target in the visual stream was correctly reported (the authors' X)",
                    "proportion", r[f"X_{rate}"], "proportion", lab, True)
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, d


def main():
    checks = stdlib.Checks()
    a, d = build()
    checks.published("participants with behavioural data", a.subject.nunique(), 54, 0, P + "Methods, Data analysis (54 remaining participants)")
    checks.published("accuracy of X placement, highest load (%)", d.X_100.mean() * 100, 78.2, 0.05, P + "Discussion (78.2% of the time)")
    checks.published("accuracy of white letter identification, highest load (%)", d.WL_100.mean() * 100, 87.5, 0.05, P + "Discussion (87.5% of the time)")
    checks.internal("every participant has 4 magnitudes", (a[a.measure == "response_magnitude"].groupby("subject").size() == 4).all())
    checks.internal("one participant has no white-letter value", a[a.measure == "white_letter_accuracy"].shape[0] == 54 * 3 - int(d[["WL_100", "WL_300", "WL_500"]].isna().sum().sum()) and d[["WL_100", "WL_300", "WL_500"]].isna().any(axis=1).sum() == 1)
    checks.internal("block order groups", set(a.group) == {"Single-task block first", "Dual-task block first"})
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY-NC-SA 4.0",
        license_url="http://creativecommons.org/licenses/by-nc-sa/4.0",
        changes="Reshaped the authors' wide participant table into one long table, labelled the conditions from the authors' data dictionary, and replaced participant codes. The EEG workbook was not converted.",
        how=[
            "The authors shared one value per participant and condition, so each row is one participant's value of one measure in one condition. "
            "Participants said /a/ while their pitch feedback was lowered by 50 cents for 200 ms, in a single-task block and in a dual-task block "
            "with a rapid visual letter stream at 100, 300 or 500 ms per letter (high, intermediate, low attentional load).",
            "response_magnitude is in cents (the paper converts f0 to cents against a 200 ms baseline); response_latency is in ms (the unit is inferred from the "
            "1000 ms analysis window, the data dictionary gives none). pert_value is -50 cents on every row, as the paper describes the shift.",
            "group is the block order (single-task block first or dual-task block first). White-letter and X accuracies exist only in the dual-task conditions.",
        ],
        unit_note="participant, measure and attentional load",
        differences=["The paper says the white-letter analysis left out one participant who did not answer as instructed; the shared file marks that participant's white-letter values with a dot, which are left out here; that participant keeps every other measure."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
