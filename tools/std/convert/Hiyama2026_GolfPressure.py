"""Hiyama, Nakata, Inoue, Manai, Hirata & Sato (2026), bioRxiv: golf putting under psychological pressure, 20 participants,
five 1.5 m putts in a practice session and five in a competition session, plus the experimenter's putts against each participant.
The deposit's fNIRS recordings are not part of this release (brain recordings, 17 x 23 MB).

Run from the site folder: python3 -I tools/std/convert/Hiyama2026_GolfPressure.py
Raw file (untrusted data, read only): Golf_Performance_Data/Golf_Performance_Data.xlsx inside raw_extra/Hiyama2026_GolfPressure/g.zip
from https://zenodo.org/records/17499032. Only the workbook's constants are read (its formulas are not evaluated).
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hiyama2026_GolfPressure"
checks = stdlib.Checks()
z = zipfile.ZipFile(stdlib.raw_dir(ID) / "g.zip")
ws = openpyxl.load_workbook(io.BytesIO(z.read("Golf_Performance_Data/Golf_Performance_Data.xlsx")))["Sheet1"]
# fNIRS files present in the deposit: participants 1-6, 8, 10, 12-20 (17), the others were excluded in the paper (preprocessing errors)
in_fnirs = {int(n.split("data_ID")[1][:2]) for n in z.namelist() if n.endswith(".mat") and "/data_ID" in n}
checks.internal("17 participants have preprocessed fNIRS data", len(in_fnirs) == 17, str(sorted(in_fnirs)))

def block(first_row):
    """5 practice rows then 5 competition rows, columns D..W = participants 1-20 (values in cm)."""
    a = np.array([[ws.cell(row=first_row + i, column=4 + j).value for j in range(20)] for i in range(10)], dtype=float)
    assert not np.isnan(a).any()
    return a

part, exper = block(4), block(20)
checks.internal("labels in the workbook are as expected",
                ws["B4"].value.startswith("Practice") and ws["B9"].value.startswith("Competition") and ws["C4"].value == "T1"
                and ws["B20"].value.startswith("Practice") and ws["B25"].value.startswith("Competition")
                and ws["D3"].value == "ID1" and ws["W3"].value == "ID20" and ws["D19"].value == "vs ID1")

rows = []
def add(who, j, errors):
    for s in range(2):
        for k in range(5):
            rows.append(dict(
                dataset=ID, study="Hiyama et al. 2026 bioRxiv", experiment="Putting under pressure", group="participant" if who == "P" else "experimenter",
                subject=f"{who}{j + 1}", person=f"{who}{j + 1}" if who == "P" else "experimenter",
                subject_excluded_by_authors=(who == "P" and (j + 1) not in in_fnirs),
                age="", sex="M" if who == "P" else "", handedness="right" if who == "P" else "",
                population="healthy, inexperienced in golf" if who == "P" else "experimenter", setting="lab",
                device="golf putter and ball, 1.5 m putt to a target cross", session=s + 1,
                block="practice session (PS)" if s == 0 else "competition session (CS)", trial=s * 5 + k + 1,
                phase="practice" if s == 0 else "test", time_s="", task="sports and games", unit="trial",
                condition="practice session" if s == 0 else "competition session with a monetary reward (psychological pressure)",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                game="golf putting (1.5 m)", attempt=k + 1, outcome_error=float(errors[s * 5 + k]), outcome_error_unit="cm",
                environment="against the experimenter" if who == "P" else "against participant sequence %d" % (j + 1)))
for j in range(20):
    add("P", j, part[:, j])
for j in range(20):
    add("E", j, exper[:, j])
t = pd.DataFrame(rows)
checks.internal("400 rows: 20 participants + 20 experimenter sequences, 10 putts each", len(t) == 400)
checks.internal("errors are non-negative distances", bool((t.outcome_error >= 0).all()))

P = t[t.group == "participant"]
checks.published("participants recruited", P.subject.nunique(), 20, 0, "Methods, Participants (20 male students)")
checks.published("putts per session per participant", P.groupby(["subject", "session"]).size().max(), 5, 0, "Methods (five trials per session)")
checks.published("participants kept after exclusions (preprocessing errors)", len(in_fnirs), 17, 0, "Results, Participants (3 excluded, 17 analysed)")
avg = P.pivot_table(index="subject", columns="session", values="outcome_error", aggfunc="mean")
avg.index = [int(s[1:]) for s in avg.index]
d = (avg[2] - avg[1]).loc[sorted(in_fnirs)]
checks.published("choking group (error larger in competition than practice), of the 17 analysed", int((d > 0).sum()), 7, 0,
                 "Results (positive delta error: 7 participants)")
checks.published("non-choking group (error smaller in competition than practice), of the 17 analysed", int((d < 0).sum()), 10, 0,
                 "Results (negative delta error: 10 participants)")

meta = dict(
    study="Hiyama et al. 2026",
    paper=dict(authors="Hiyama R, Nakata N, Inoue R, Manai T, Hirata T, Sato H",
               title="An Exploratory Study of Prefrontal Cortex Activation Related to Golf Putting Performance Under Psychological Pressure: "
                     "A Functional Near-infrared Spectroscopy Approach", venue="bioRxiv (preprint)", year=2026, doi="10.64898/2026.07.13.737385"),
    lab="Sato (Shibaura Institute of Technology)", data_url="https://doi.org/10.5281/zenodo.17499032",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor game format from the authors' putting workbook: one row per putt, participant numbers replaced by "
             "codes. The deposit's preprocessed fNIRS recordings and analysis scripts are not included. The workbook's averaging formulas "
             "were not carried over."),
    how=["One row is one putt from 1.5 m at a target cross. Each participant putted five times in a practice session (session 1) and five "
         "times in a competition session (session 2) against the experimenter, with a monetary reward introduced to induce pressure. "
         "outcome_error is the authors' error value: the shortest distance in cm between the ball's final position and the centre of "
         "the target cross (always positive, so it has no sign).",
         "The workbook also lists the experimenter's own five practice and five competition putts against each participant. Those are rows "
         "with group 'experimenter'; each of the 20 experimenter sequences (subject codes S0021 to S0040) is the experimenter's putting against "
         "the participant sequence with the same position (S0001 to S0020), and all belong to one person. The participant order is the "
         "authors' order (their participants 1 to 20).",
         "subject_excluded_by_authors marks the three participants (their numbers 7, 9, 11) that the paper left out because of preprocessing "
         "errors in the fNIRS data; their putts are in the workbook and were kept. Age is not given per participant (paper: 21.3 +/- 1.3 years, "
         "all male, right-handed); sex and handedness are the group description from the paper.",
         "Choking and non-choking groups were defined by the authors from the change in mean error between sessions (competition minus practice); "
         "the groups are not a column here, and can be recomputed."],
    unit_note="putt",
    differences=["The paper is a bioRxiv preprint (CC BY-NC-ND on bioRxiv) while the Zenodo deposit of the data states CC BY 4.0; the deposit's license is used.",
                 "The paper reports group sizes and fNIRS statistics only in text; the putting errors themselves appear only in its Figure 3, which "
                 "could not be read, so no mean or SD of putting error is checked."],
)
rows = t.to_dict("records")
stdlib.release(ID, "game", rows, meta, checks)
