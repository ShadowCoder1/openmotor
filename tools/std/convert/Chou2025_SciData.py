"""Chou et al. (2025), Scientific Data 12:339 (TTSwing): 97,350 table-tennis strokes by 93 elite players, one row per stroke with the
authors' 34 racket-sensor features, in the 'game' module.

Run from the site folder: python3 -I tools/std/convert/Chou2025_SciData.py
Raw file (untrusted data, read only) in raw_extra/Chou2025_SciData/TTSWING.zip, from https://zenodo.org/records/14818398
(the Zenodo copy of Dryad doi:10.5061/dryad.0zpc8677f; Zenodo states Apache-2.0, Dryad states CC0). Only TTSWING/dataset/TTSWING.csv
and TTSWING/README.md (the field list) are read; the repository's Python files are not read or run.
"""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Chou2025_SciData"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
z = zipfile.ZipFile(RAW / "TTSWING.zip")
d = pd.read_csv(io.BytesIO(z.read("TTSWING/dataset/TTSWING.csv")))
FEATS = list(d.columns[6:40])
checks.internal("47 columns, 34 sensor features after the 6 index columns", d.shape[1] == 47 and len(FEATS) == 34 and FEATS[0] == "ax_mean" and FEATS[-1] == "g_entropy")
MODE = {0: "air swing", 1: "full power stroke", 2: "stable hitting"}
checks.internal("no missing values", int(d.isna().sum().sum()) == 0)
for c in ("gender", "age", "play years", "height", "weight", "handedness", "hold racket handed"):
    checks.internal(f"{c} constant within player", bool((d.groupby("id")[c].nunique() == 1).all()))
sizes = d.groupby(["id", "testmode", "teststage", "fileindex"]).size()
checks.internal("every (player, mode, stage, round) holds a multiple of 50 strokes", bool((sizes % 50 == 0).all()))
dates = pd.to_datetime(d.date, format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
dup = int(d.duplicated().sum())

rows = []
seq = {}
cols = {c: d[c].tolist() for c in d.columns}
for i in range(len(d)):
    pid = cols["id"][i]
    seq[pid] = seq.get(pid, 0) + 1
    r = dict(
        dataset=ID, study=ID, experiment="TTSwing", group="", subject=f"id{pid}", subject_excluded_by_authors=False, age=None,
        sex="M" if cols["gender"][i] == 1 else "F", handedness="right" if cols["handedness"][i] == 1 else "left",
        population="healthy, elite table tennis players", setting="lab", device="custom racket grip with 9-axis IMU (ICM-20948, 80 Hz)",
        session=1, block=f"round {cols['fileindex'][i]}", trial=seq[pid], phase="other", time_s=None, task="sports and games",
        unit="trial", condition=MODE[cols["testmode"][i]], pert_type="none", pert_value=0, pert_unit="", feedback="normal",
        excluded_by_authors=False, game="Table tennis stroke", attempt=cols["count"][i], date=dates.iat[i], environment="",
        ball_speed_stage=cols["teststage"][i])
    for f in FEATS:
        r[f] = cols[f][i]
    rows.append(r)

P = d.drop_duplicates("id")
checks.published("players", len(P), 93, 0, "Abstract / Data Records")
checks.published("male players", (P.gender == 1).sum(), 53, 0, "Methods / Data Records")
checks.published("female players", (P.gender == 0).sum(), 40, 0, "Methods / Data Records")
checks.published("records (strokes) in total", len(d), 97350, 0, "Data Records (97,350 records)")
for m, n in ((0, 7500), (1, 73850), (2, 16000)):
    checks.published(f"records in mode {m} ({MODE[m]})", (d.testmode == m).sum(), n, 0, "Data Records / Data Statistics")
checks.published("right-handed players", (P.handedness == 1).sum(), 78, 0, "Data Records (78 right-handed)")
checks.published("left-handed players", (P.handedness == 0).sum(), 15, 0, "Data Records (15 left-handed)")

parts = [dict(person=f"id{r.id}", racket_hand="right" if r["hold racket handed"] == 1 else "left", age_tier=r.age,
              experience_tier=r["play years"], height_tier=r.height, weight_tier=r.weight) for _, r in P.iterrows()]
pcols = [("racket_hand", "category", "right, left", "Hand that holds the racket (authors' hold racket handed)."),
         ("age_tier", "category", "low, medium, high", "Age tier (the authors binned age into tiers by the distribution of the data; no cut-offs given)."),
         ("experience_tier", "category", "low, medium, high", "Years of playing ball games, in tiers (authors' play years)."),
         ("height_tier", "category", "low, medium, high", "Height tier (authors' binning)."),
         ("weight_tier", "category", "low, medium, high", "Weight tier (authors' binning).")]
differences = [
    "The paper gives the total as 97,350 records and also says 'over 90,000 strokes'; the deposit has 97,350 rows, which match the first number.",
    "The paper's text calls mode 1 both 'aerobic swing' and 'full power stroke' and ties the three serving-machine ball speeds to different modes in "
    "different places; the deposit README (mode 0 air swing, 1 full power stroke, 2 stable hitting) and the data (stages 1-3 occur only in mode 1) agree, and are used here.",
    f"{dup} rows of the deposit are exact duplicates of another row; all rows are kept as shared. The 'fileindex' (here 'round') is not unique within a "
    "player, mode and stage: some combinations hold 100 or 150 strokes (two or three rounds on different dates), so use date as well.",
    "Age, experience, height and weight are only given as low/medium/high tiers (privacy), so age is empty and the tiers are in participants.csv.",
    "The Dryad record states CC0 and the Zenodo copy used here states Apache-2.0; both allow sharing a converted copy.",
]
meta = dict(
    paper=dict(authors="Chou CY, Chen ZH, Sheu YH, Chen HH, Sun MT, Wu SK", year=2025,
               title="TTSwing: a Dataset for Table Tennis Swing and Racket Kinematics Analysis", venue="Scientific Data 12(1):339",
               doi="10.1038/s41597-025-04680-y"),
    lab="Chen HH / Sun MT (National Central University, Taiwan)", data_url="https://doi.org/10.5061/dryad.0zpc8677f (files read from the Zenodo copy https://zenodo.org/records/14818398)",
    license="Apache-2.0 (Zenodo copy used); the Dryad record states CC0", license_url="https://www.apache.org/licenses/LICENSE-2.0",
    changes="One row per stroke kept in the authors' order; the date written as YYYY-MM-DD; player codes replaced; demographics tiers moved to participants.csv; "
            "feature values are unchanged (raw sensor units).",
    how=["One row is one table-tennis stroke (the authors' 'count', 1-50 within a round of 50 continuous swings), with 34 features the authors computed from "
         "the racket's 9-axis sensor waveform (means, variances, root mean squares, extremes, spectral and shape statistics). No raw waveforms were shared.",
         "condition is the test mode: air swing, full power stroke (hitting balls from a serving machine at three speeds, ball_speed_stage 1-3) or stable hitting. "
         "block is the round number (authors' fileindex).",
         "Acceleration is in raw sensor units (LSB/G); multiply by 2/32768 for G. Angular velocity is in LSB per deg/s; multiply by 250/32768 for deg/s (authors' README). "
         "sex is the authors' gender code (1 = male, 0 = female); handedness is the dominant hand."],
    unit_note="stroke", differences=differences)
stdlib.release(ID, "game", rows, meta, checks, participants=parts, participant_columns=pcols)
