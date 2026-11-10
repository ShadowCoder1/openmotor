"""Valk et al. (2019), Sci Rep 9:17378: rhythmic Fitts pointing recorded with two Optotrak 3020 units (100 Hz).

Raw file, raw_extra/Valk2019_SciRep/ (Figshare 8280893, one RAR archive, extracted on the PC with 7-Zip into all/):
  BAP_P<n>/BAP_P<n>_<k>.n3d  Optotrak 3D data file: 256-byte header (type 32, 37 items, 3 coordinates, number of frames,
                             frequency as float32), then frames x items x (x, y, z) float32 in mm. Missing values are
                             about -3.7e28 (Optotrak's marker-lost flag).
  BAP_P<n>/*.nco, *.nrw      collection settings (read as text only: 100 Hz, 6 rigid bodies of 3 LEDs) and raw camera data (not used)
  ConditionsValketal2019.xls one row per trial: participant, test day, ID, target distance (cm), trial number, file number
  calibratiedoelen/          calibration recordings of the targets (not converted)
What the paper says: 20 right-handed participants recruited (2 excluded for failed acquisition, 18 analysed, 8 men, age 20.8 +- 1.9),
24 conditions (ID 3.5-6 in steps of 0.5 x distance 5, 10, 20, 30 cm) in random order, 40 cycles each, six rigid bodies with
three LEDs, Optotrak 3020 at 100 Hz, 18 bony landmarks plus the stylus tip digitised.
Run from the site folder: python3 -I tools/std/convert/Valk2019_SciRep.py
"""
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Valk2019_SciRep"
SRC = "Valk et al. 2019, Sci Rep"
RAW = stdlib.raw_dir(ID)
BASE = RAW / "all"
TMP = RAW / "tmp_samples"
TMP.mkdir(exist_ok=True)

DEVICE = "Two synchronised Optotrak 3020 units (100 Hz), six rigid bodies of three LEDs on the right arm, trunk and stylus"
KIND = "3D marker positions (18 LEDs, then 19 points derived by the authors' rigid-body files)"
CHANNELS = [f"led{i:02d}_{a}" for i in range(1, 19) for a in "xyz"] + [f"pt{i:02d}_{a}" for i in range(1, 20) for a in "xyz"]
UNITS = ";".join(["mm"] * len(CHANNELS))
PROC = ("None: positions as stored in the .n3d files, in the aligned coordinate system of the authors' calibration; "
        "marker-lost flags are empty cells. Written with two decimals (0.01 mm).")

xl = pd.read_excel(RAW / "x" / "ConditionsValketal2019.xls", engine="xlrd")
assert xl.columns.tolist() == ["SubjectNr", "TestDay", "ID", "TargetDistance", "TrialNr", "TrialNrOptotrak"]
ck = stdlib.Checks()

# one file number must not serve two sheet rows; the later duplicate's file does not exist for it (see differences)
dup = xl.duplicated(["SubjectNr", "TrialNrOptotrak"], keep=False)
dropped = []
keep_rows = []
for r in xl.itertuples():
    if dup[r.Index]:
        # decide by the stylus excursion in the file against the sheet's distance (checked below)
        pass
    keep_rows.append(r)


def load(s, f):
    d = (BASE / f"BAP_P{s}" / f"BAP_P{s}_{f:03d}.n3d").read_bytes()
    t, it, sub, nf, fr = struct.unpack("<BHHIf", d[:13])
    assert (t, it, sub) == (32, 37, 3), (t, it, sub)
    a = np.frombuffer(d[256:256 + nf * it * sub * 4], "<f4").reshape(nf, it * sub).astype("f8")
    assert len(d) == 256 + nf * it * sub * 4
    return a, fr


def excursion(a):
    x = a[:, 18 * 3]  # item 19 x
    x = x[np.abs(x) < 1e10]
    return np.percentile(x, 99) - np.percentile(x, 1)


# resolve the duplicate: of the two sheet rows pointing at one file, keep the one whose distance matches the stylus excursion
drop_idx = set()
for (s, f), g in xl[dup].groupby(["SubjectNr", "TrialNrOptotrak"]):
    a, _ = load(s, f)
    ex = excursion(a)
    best = min(g.itertuples(), key=lambda r: abs(ex - r.TargetDistance * 10))
    for r in g.itertuples():
        if r.Index != best.Index:
            drop_idx.add(r.Index)
            dropped.append((s, r.TrialNr))
print("duplicate file rows dropped", dropped)

rows, subj_order = [], {}
n_missing_total = n_cells = 0
freqs, nfr, exc = set(), [], []
for r in xl.itertuples():
    if r.Index in drop_idx:
        continue
    s = int(r.SubjectNr)
    subj_order.setdefault(s, len(subj_order) + 1)
    code = f"{ID}_S{subj_order[s]:04d}"
    a, fr = load(s, int(r.TrialNrOptotrak))
    freqs.add(round(fr, 3))
    bad = np.abs(a) > 1e10
    n_missing_total += int(bad.sum()); n_cells += a.size
    a = np.where(bad, np.nan, a)
    n = len(a)
    nfr.append(n)
    exc.append((r.TargetDistance, excursion(np.where(bad, 1e30, a))))
    df = pd.DataFrame(a, columns=CHANNELS)
    df.insert(0, "time_s", np.arange(n) / fr)
    path = TMP / f"{code}_trial{int(r.TrialNr):03d}.csv"
    df.to_csv(path, index=False, float_format="%.2f", na_rep="", lineterminator="\r\n")
    cond = f"ID {r.ID:g}, {int(r.TargetDistance)} cm"
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group="", subject=f"S{s}", person=f"S{s}", subject_excluded_by_authors=False,
        age=None, sex=None, handedness="right", population="healthy", setting="lab", device=DEVICE, session=1,
        block=f"test day {int(r.TestDay)}", trial=int(r.TrialNr), phase="other", time_s=None, task="reaching (Fitts pointing)",
        unit="recording", condition=cond, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
        recording=int(r.TrialNr), samples_file=f"samples/{path.name}", signal_kind=KIND, channels=";".join(CHANNELS),
        channel_units=UNITS, sample_rate_hz=fr, n_samples=n, duration_s=n / fr, events="", processing=PROC,
        trial_info=f"index_of_difficulty={r.ID:g};target_distance_cm={int(r.TargetDistance)};optotrak_file={int(r.TrialNrOptotrak)}"))
df = pd.DataFrame(rows)
extra = [(f"samples/{Path(p).name}", p) for p in (TMP / x["samples_file"].split("/")[-1] for x in rows)]

# ---- checks ---------------------------------------------------------------------------------------------------------------------
ck.published("participants recruited", df["subject"].nunique(), 20, 0, "Methods, Participants (twenty participants recruited)")
ck.published("unique conditions", df["condition"].nunique(), 24, 0, "Methods, Experimental design (24 unique conditions)")
ck.internal("each participant has 23-24 of the conditions (one trial missing for one participant)", df.groupby("subject")["condition"].nunique().isin([23, 24]).all())
ck.published("distinct target distances", len(set(xl.TargetDistance)), 4, 0, "Methods, Experimental set-up (5, 10, 20, 30 cm)")
ck.published("distinct IDs", len(set(xl.ID)), 6, 0, "Methods, Experimental set-up (ID 3.5-6 in steps of 0.5)")
ck.published("sampling rate Hz", float(np.mean(list(freqs))), 100, 0, "Methods, Experimental set-up (100 Hz)")
ck.internal("one sampling rate in all files", len(freqs) == 1, str(freqs))
ck.internal("rows = 480 sheet rows minus duplicates", len(df) == 480 - len(drop_idx), f"{len(df)} rows")
ex = pd.DataFrame(exc, columns=["d", "e"]).groupby("d").e.median()
ck.internal("median stylus excursion within 12 mm of the nominal target distance (targets have width, so a little more than the distance is expected)", all(abs(ex[d] - d * 10) <= 12 for d in ex.index),
            str(ex.round(0).to_dict()))
ck.internal("marker-lost share below 5%", n_missing_total / n_cells < 0.05, f"{n_missing_total / n_cells:.4%}")
print(df.groupby("subject").size().to_dict(), "missing share", n_missing_total / n_cells, ex.round(1).to_dict(), min(nfr), max(nfr))
for c in ck.items:
    print(c["kind"], c["ok"], c.get("what"), c.get("computed"), c.get("detail", ""))

meta = dict(
    paper=dict(authors="Valk TA, Mouton LJ, Otten E, Bongers RM", year=2019,
               title="Synergies reciprocally relate end-effector and joint-angles in rhythmic pointing movements",
               venue="Sci Rep 9(1):17378", doi="10.1038/s41598-019-53913-9"),
    lab="Bongers (University Medical Center Groningen)",
    data_url="https://doi.org/10.6084/m9.figshare.8280893", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Each Optotrak recording (.n3d) became one row, with its samples as one CSV in samples/. Positions are copied in mm; the "
             "marker-lost flag becomes an empty cell and values are written with two decimals. Raw camera files (.nrw), the target "
             "calibration recordings and the collection settings were not converted. Participant labels are replaced by codes."),
    how=["One row is one trial of rhythmic pointing: the participant moved a stylus (on the index finger) forward and back between two "
         "targets (about 40 cycles, 17-79 s), at one combination of Fitts index of difficulty (3.5-6) and target distance (5, 10, 20 or "
         "30 cm). Each participant did the 24 combinations once, in the order of the authors' sheet, all on one test day.",
         "samples_file has one line per 10 ms frame: time_s from the start of the recording, then x, y, z in mm of 18 LEDs (led01-led18: "
         "six rigid bodies of three LEDs, in the deposit's marker order) and of 19 further points (pt01-pt19). The deposit has no readme: "
         "the 19 points are inferred to be the digitised stylus tip and 18 bony landmarks, placed in the laboratory frame through the "
         "rigid bodies, because the paper describes that digitisation. pt01 is the stylus tip by its movement: its range along x equals "
         "the target distance in every condition, so x is the pointing direction. Which rigid body and landmark is which anatomical "
         "location is not stated and is not labelled here.",
         "Joint angles and end-effector trajectories analysed in the paper are not in the deposit; only these marker positions are.",
         "All participants are in the deposit, but the paper analysed 18 of the 20 (two excluded for failed acquisition) without saying "
         "which; no participant is marked excluded here. Age and sex are not given per participant (mean 20.8 years, 8 men among 18)."],
    unit_note="recording (one trial of rhythmic pointing)",
    differences=[
        "The paper analysed 18 participants (two excluded for failed data acquisition, not identified); the deposit holds 20.",
        "The sheet lists two trials for one Optotrak file of one participant (trial 5, 3.5 / 5 cm, and trial 4, 4 / 20 cm, both file 4); "
        "the stylus excursion in the file (about 20 cm) matches trial 4, so trial 5 has no recording and is left out.",
        "Two Optotrak files (one for each of two participants) are not in the sheet (the participant's trial numbers skip them); they are not released.",
        "The paper's end-effector and joint-angle measures (NL, AD ratio and others) were not recomputed.",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "timeseries", df, meta, ck, extra_files=extra)
