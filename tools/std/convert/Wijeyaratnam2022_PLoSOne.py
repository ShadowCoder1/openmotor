"""Wijeyaratnam, Edwards, Pilutti & Cressman (2022), PLoS ONE: visually guided reaching in people with multiple sclerosis (MS)
with (MS-R) and without (MS-NR) self-reported upper limb impairment, and controls (CTL), 12 per group, each hand.
Format: summary (one row = one measure of one participant and hand). Per-participant mean and SD over 64 trials per hand.

Run from the site folder: python3 -I tools/std/convert/Wijeyaratnam2022_PLoSOne.py
Raw file (read only) in raw_extra/Wijeyaratnam2022_PLoSOne/ from https://osf.io/ntfxm/: Opensource_data.xlsx (sheets Clinical, Performance).
"""
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wijeyaratnam2022_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
X = pd.ExcelFile(RAW / "Opensource_data.xlsx")

cl = pd.read_excel(X, "Clinical")
cl["Group"] = cl["Group"].ffill()
cl = cl.rename(columns=lambda c: str(c).strip())
pf = pd.read_excel(X, "Performance", header=None)

GROUP = {"CTL": "CTL (control)", "MS-NR": "MS-NR (MS, no self-reported upper limb impairment)",
         "MS-R": "MS-R (MS, self-reported upper limb impairment)"}
POP = {"CTL": "healthy", "MS-NR": "multiple sclerosis, no self-reported upper limb impairment",
       "MS-R": "multiple sclerosis, self-reported upper limb impairment"}
# measure -> (column of mean, column of SD, description, unit); columns are positions in the Performance sheet
M = {
    "RT": (3, 4, "Reaction time: target onset to movement start (offline control).", "ms"),
    "PV": (5, 6, "Peak velocity (offline control).", "m/s"),
    "pTTPV": (7, 8, "Time to peak velocity as a percentage of movement time (offline control).", "%"),
    "abs_IDE": (9, 10, "Absolute initial direction error (offline control).", "deg"),
    "MT": (12, 13, "Movement time (online control).", "ms"),
    "RE": (14, 15, "Resultant error: distance from the target at movement end (online control).", "cm"),
    "abs_EPAE": (16, 17, "Absolute endpoint angular error (online control).", "deg"),
    "abs_dAE": (18, 19, "Absolute change in angular error from early to late in the movement (file column |AE|; online control).", "deg"),
    "PL": (20, 21, "Path length (online control).", "cm"),
}
hdr = pf.iloc[1].tolist()
for k, (a, b, _, _) in M.items():
    assert str(hdr[a]).replace("|", "").replace(" ", "").lower().startswith(k.replace("abs_", "").replace("dAE", "ae").lower()[:2]), (k, hdr[a])
    assert "SD" in str(hdr[b]), (k, hdr[b])

# group of each performance row (labels run down the sheet)
perf = []
hand = None
grp = None
for i in range(2, len(pf)):
    r = pf.iloc[i]
    if isinstance(r[0], str):
        hand = "non-dominant" if r[0].startswith("Non") else "dominant"
    if isinstance(r[1], str):
        grp = r[1].split("\n")[0]
    if pd.isna(r[2]):
        continue
    perf.append((hand, grp, int(r[2]), r))
assert len(perf) == 72, len(perf)

rows = []
persons = {}
order = {}


def common(pid, grp_code, hnd, label):
    persons.setdefault(pid, f"P{len(persons) + 1}")
    order[pid] = order.get(pid, 0) + 1
    c = cl[cl.Participants == pid].iloc[0]
    return dict(
        dataset=ID, study=ID, experiment="Reaching task, both hands", group=GROUP[grp_code], subject=persons[pid], person=persons[pid],
        subject_excluded_by_authors="False", age=c["Age"], sex="M" if int(c["Sex"]) == 1 else "F", handedness="", population=POP[grp_code],
        setting="lab", device="computer-based reaching task (movement on a device with cursor)", session="", block="", trial=order[pid],
        phase="other", time_s="", task="reaching", unit="participant summary", condition=f"{hnd} hand",
        pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors="False", session_context="")


for hnd, g, pid, r in perf:
    for k, (a, b, desc, unit) in M.items():
        for stat, col, d2 in (("mean", a, f"Mean over 64 trials. {desc}"), ("sd", b, f"Standard deviation over trials. {desc}")):
            row = common(pid, g, hnd, k)
            row.update(measure=k if stat == "mean" else k + "_sd", measure_description=d2, statistic=stat, value=float(r[col]),
                       value_unit=unit, n_trials=64)
            rows.append(row)
for hnd, col in (("non-dominant", "9_HPT NDH"), ("dominant", "9_HPT DH")):
    for c in cl.itertuples(index=False):
        c = c._asdict() if hasattr(c, "_asdict") else c
for _, c in cl.iterrows():
    for hnd, col in (("non-dominant", "9_HPT NDH"), ("dominant", "9_HPT DH")):
        row = common(int(c["Participants"]), c["Group"], hnd, "9HPT")
        row.update(measure="nine_hole_peg_test", measure_description="Nine-Hole Peg Test time, mean of two trials with this hand (a 0 in the file is kept as given).",
                   statistic="mean", value=float(c[col]), value_unit="s", n_trials=2)
        rows.append(row)

df = pd.DataFrame(rows)
# ---- participants table
pcols = [("group", "text", "", "CTL, MS-NR or MS-R."), ("height_cm", "number", "cm", "Height."), ("weight_kg", "number", "kg", "Weight."),
         ("EDSS", "number", "", "Expanded Disability Status Scale score (people with MS)."),
         ("disease_duration_years", "number", "years", "Years since MS diagnosis (as in the file column Disease Duration)."),
         ("ms_type", "text", "", "RRMS, PPMS or SPMS.")]
parts = []
for _, c in cl.iterrows():
    parts.append(dict(person=persons[int(c["Participants"])], group=GROUP[c["Group"]], height_cm=c["Height"], weight_kg=c["Weight"],
                      EDSS=c["EDSS"], disease_duration_years=c["Disease Duration"], ms_type=c["MS Type"]))

# ---- checks (numbers printed in the paper)
def grp_vals(g, meas, hand=None, per_person_mean_over_hands=True):
    d = df[(df.group == GROUP[g]) & (df.measure == meas)]
    if hand:
        d = d[d.condition == f"{hand} hand"]
    return list(d.groupby("person").value.mean())

for g in GROUP:
    checks.published(f"participants, {g}", cl[cl.Group == g].shape[0], 12, 0, "Methods, Participants (12 per group)")
for g in GROUP:
    checks.published(f"women among the participants, {g}", sum(1 for _, c in cl[cl.Group == g].iterrows() if int(c["Sex"]) == 0), 8, 0,
                     "Methods, Participants (66.7% female in each group; sex coded 0 = female inferred from this)")
for g, v in {"MS-R": 9.7, "MS-NR": 10.0, "CTL": 9.4}.items():
    checks.published(f"path length, {g}, mean of the two hands", st.mean(grp_vals(g, "PL")), v, 0.05, "Results, path length")
for g, v in {"MS-R": 6.4, "MS-NR": 5.6, "CTL": 4.5}.items():
    checks.published(f"|IDE|, {g}, mean of the two hands", st.mean(grp_vals(g, "abs_IDE")), v, 0.05, "Results, initial direction error")
for g, v in {"MS-R": 4.8, "CTL": 3.4}.items():
    checks.published(f"|change in angular error|, {g}, mean of the two hands", st.mean(grp_vals(g, "abs_dAE")), v, 0.05, "Results, online control")
for g, (mu, s_) in {"MS-NR": (12.3, 7.9), "MS-R": (16.4, 9.2)}.items():
    x = list(cl[cl.Group == g]["Disease Duration"].astype(float))
    checks.published(f"disease duration mean, {g}", st.mean(x), mu, 0.05 + 1e-9, "Results, Participants")
    checks.published(f"disease duration SD, {g}", st.stdev(x), s_, 0.05, "Results, Participants")
for g, v in {"MS-NR": 3.3, "MS-R": 5.8}.items():
    checks.published(f"median EDSS, {g}", st.median(cl[cl.Group == g]["EDSS"].astype(float)), v, 0.05, "Results, Participants")
checks.internal("every Performance cell kept: 72 hand x participant rows x 9 measures x (mean, SD) + 72 9-HPT values",
                len(df) == 72 * 18 + 72, f"{len(df)} rows")
checks.internal("no missing values", df.value.notna().all())
checks.internal("36 participants, 12 per group", df.person.nunique() == 36)

paper = next(e for e in json.load(open(stdlib.HERE / "source/extra_studies.json")) if e["id"] == ID)["papers"][0]
meta = dict(
    study=ID, paper={k: paper[k] for k in ("authors", "year", "title", "venue", "doi")},
    lab="Cressman (University of Ottawa)", data_url="https://osf.io/ntfxm/", license="none stated", license_url="",
    changes="converted from the two spreadsheet sheets to the long OpenMotor summary format, participant codes replaced; no value changed.",
    unit_note="measure of one participant and hand (long form)",
    how=["One row is one measure of one participant and hand: for 9 reaching measures the participant's mean over 64 trials (statistic mean) and the standard deviation over those trials (measure name ending _sd), plus the Nine-Hole Peg Test time per hand (mean of 2 trials). Trial-level data were not shared.",
         "Measures (offline control) RT ms, PV m/s, pTTPV %, |IDE| deg; (online control) MT ms, RE cm, |EPAE| deg, |change in angular error| deg, PL cm; units are those of the paper's Methods. The deposit's column |AE| is named abs_dAE here, after the paper's |change in angular error|.",
         "Age, sex and group are from the Clinical sheet. Sex is coded 0 and 1 in the file; 1 = male was inferred because 4 of 12 participants per group have a 1, matching the paper's 66.7% female in every group. Hand is labelled non-dominant or dominant as in the file; handedness itself is not given.",
         "participants.csv holds height, weight, EDSS, disease duration and MS type. The article itself is under CC BY 4.0, but the OSF deposit states no license, so none is claimed here."],
    differences=[])
stdlib.release(ID, "summary", df.drop(columns=[]), meta, checks, participants=parts, participant_columns=pcols)
