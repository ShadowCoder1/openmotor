"""Danion, Mathew, Gouirand & Brenner (2021), Cortex 134:30-42: eye and hand tracking of a pseudo-random 2D target moved by a
joystick, 62 right-handed adults, 10 baseline trials each (five target patterns, each shown twice, 10 s per trial).

Raw files, raw_extra/Danion2021_Cortex/ (Zenodo 4074990, CC BY 4.0):
  DATAmain.xlsx   one sheet per group of measures, subjects in rows ("subject 1".."subject 62"), trials 1-10 in columns, side by side
                  per measure (HAND-TARGET, EYE-TARGET, EYE-CURSOR: distance and lag on x and y and residual distance after the fit lags;
                  Eye SP gain; EyeHandCorrelation; Loop1 vs Loop2 (two stacked tables); NORMAL vs SWITCH). The label of the fourth block
                  of the first three sheets reads "DIST on Y axis (ms)"; it is the y lag (unit ms, same position as the x lag block).
  pattern.txt     Pattern, Subject, Trial: which of the five target trajectories each trial showed.
Not converted: the 1 kHz eye, target and hand series (.dat files are a 10-trial example only) and SACinfo10trials.dat (saccade
table without a codebook), the Perl scripts and acquisition files.
Run from the site folder: python3 -I tools/std/convert/Danion2021_Cortex.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Danion2021_Cortex"
SRC = "Danion et al. 2021, Cortex 134:30"
RAW = stdlib.raw_dir(ID)

AX = {"X": "x (horizontal)", "Y": "y (vertical)"}


def specs():
    """(sheet, title keyword rule) -> (measure, description, statistic, unit). Built from the block titles in row order."""
    out = {}
    for sheet, nm, what in [("HAND-TARGET", "hand_target", "the cursor (moved by the hand) and the target"),
                            ("EYE-TARGET", "eye_target", "gaze and the target"),
                            ("EYE-CURSOR", "eye_cursor", "gaze and the cursor")]:
        k = 0
        for kind in ("dist", "lag", "resid"):
            for ax in ("x", "y"):
                if kind == "dist":
                    out[(sheet, k)] = (f"{nm}_distance_{ax}", f"Mean absolute {AX[ax.upper()]} distance between {what} over the trial "
                                       "(from 1 s after movement start to the end, 1000 Hz samples).", "mean", "cm")
                elif kind == "lag":
                    out[(sheet, k)] = (f"{nm}_lag_{ax}", f"Lag of {what.split(' and ')[0]} behind {what.split(' and ')[1]} on the {AX[ax.upper()]} "
                                       "axis, from a best-fit time shift (the block label for the y lag reads DIST in the deposit).", "value", "ms")
                else:
                    out[(sheet, k)] = (f"{nm}_residual_distance_{ax}", f"Mean absolute {AX[ax.upper()]} distance between {what} after "
                                       "shifting the data by the fit lag of that axis.", "mean", "cm")
                k += 1
    return out


SPEC = specs()
SPEC.update({
    ("Eye SP gain", 0): ("eye_pursuit_gain_x", "Smooth pursuit gain of the eye on the x (horizontal) axis: mean ratio of eye velocity to target velocity "
                         "outside saccades.", "mean", "ratio"),
    ("Eye SP gain", 1): ("eye_pursuit_gain_y", "Smooth pursuit gain of the eye on the y (vertical) axis.", "mean", "ratio"),
    ("EyeHandCorrelation", 0): ("eye_hand_error_correlation_x", "Correlation over time between eye and hand tracking error on the x axis.",
                                "value", "r"),
    ("EyeHandCorrelation", 1): ("eye_hand_error_correlation_y", "Correlation over time between eye and hand tracking error on the y axis.",
                                "value", "r"),
    ("Loop1 vs Loop2", 0): ("cursor_target_distance_loop1", "Mean distance between cursor and target during the first 5 s loop of the trial.",
                            "mean", "cm"),
    ("Loop1 vs Loop2", 1): ("cursor_target_distance_loop2", "Mean distance between cursor and target during the second loop.", "mean", "cm"),
    ("Loop1 vs Loop2", 2): ("eye_target_distance_loop1", "Mean distance between gaze and target during the first loop.", "mean", "cm"),
    ("Loop1 vs Loop2", 3): ("eye_target_distance_loop2", "Mean distance between gaze and target during the second loop.", "mean", "cm"),
    ("NORMAL vs SWITCH", 0): ("eye_cursor_distance_normal", "Mean eye-cursor distance over the last 4 s of each loop, averaged over the loops "
                              "(NORMAL: eye and cursor from the same loop).", "mean", "cm"),
    ("NORMAL vs SWITCH", 1): ("eye_cursor_distance_switched", "The same distance after pairing the eye data of one loop with the cursor data of "
                              "the other loop (SWITCHED).", "mean", "cm"),
})
ORDER = ["HAND-TARGET", "EYE-TARGET", "EYE-CURSOR", "Eye SP gain", "EyeHandCorrelation", "Loop1 vs Loop2", "NORMAL vs SWITCH"]


def read_sheet(ws):
    """-> list of blocks (in sheet order): dict subject_number -> list of 10 values. Tables are found by their 'trial 1' header cells."""
    rows = list(ws.iter_rows(values_only=True))
    blocks = []
    for i, r in enumerate(rows):
        starts = [j for j, v in enumerate(r) if v == "trial 1"]
        if not starts:
            continue
        data = {}
        k = i + 1
        while k < len(rows) and isinstance(rows[k][0], str) and rows[k][0].startswith("subject"):
            data[int(rows[k][0].split()[1])] = rows[k]
            k += 1
        for j in starts:
            blocks.append({s: [None if v is None else float(v) for v in row[j:j + 10]] for s, row in data.items()})
    return blocks


def main():
    checks = stdlib.Checks()
    wb = openpyxl.load_workbook(RAW / "DATAmain.xlsx", read_only=True, data_only=True)
    long = []  # (subject, trial, measure key, value)
    seq = []
    for sheet in ORDER:
        blocks = read_sheet(wb[sheet])
        n_expected = sum(1 for k in SPEC if k[0] == sheet)
        assert len(blocks) == n_expected, (sheet, len(blocks), n_expected)
        for bi, blk in enumerate(blocks):
            assert len(blk) == 62, (sheet, bi, len(blk))
            seq.append((sheet, bi))
            for s, vals in blk.items():
                assert len(vals) == 10
                for t, v in enumerate(vals, start=1):
                    long.append((s, t, (sheet, bi), v))
    pat = {}
    lines = open(RAW / "pattern.txt", encoding="utf-8", errors="replace").read().split("\n")[1:]
    for ln in lines:
        p = ln.split()
        if len(p) == 3:
            pat[(int(p[1]), int(p[2]))] = int(p[0])
    L = pd.DataFrame(long, columns=["s", "t", "key", "v"])
    pos = {k: i for i, k in enumerate(seq)}
    L["o"] = L.key.map(pos)
    L = L.sort_values(["s", "t", "o"], kind="stable").reset_index(drop=True)
    rows = []
    for r in L.itertuples(index=False):
        m = SPEC[r.key]
        rows.append(dict(
            dataset=ID, study=ID, experiment="", group="", subject=f"S{r.s:02d}", subject_excluded_by_authors=False, age="", sex="",
            handedness="right", population="healthy", setting="lab",
            device="joystick controlling a cursor on a screen, eye tracker (EyeLink 1000, 1000 Hz)",
            session=1, block="baseline tracking trials", trial=r.t, phase="baseline", time_s="", task="tracking", unit="trial",
            condition=f"target pattern {pat[(r.s, r.t)]}", pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=False, measure=m[0], measure_description=m[1], statistic=m[2], value=r.v, value_unit=m[3], n_trials=1))
    P = f"{SRC}, Results"
    M = {SPEC[k][0]: g.v for k, g in L.groupby("key")}
    PM = {SPEC[k][0]: g.groupby("s").v.mean() for k, g in L.groupby("key")}
    checks.published("participants", L.s.nunique(), 62, 0, f"{SRC}, Abstract and 2.1")
    checks.published("trials (10 for each of the 62 participants)", L.groupby("s").t.nunique().sum(), 620, 0, f"{SRC}, 3.2")
    for nm, rep, tol, src in [("eye_target_distance_x", 0.84, 0.005, "3.2"), ("eye_target_distance_y", 0.99, 0.005, "3.2"),
                              ("hand_target_distance_x", 1.05, 0.005, "3.3"), ("hand_target_distance_y", 1.21, 0.005, "3.3"),
                              ("eye_target_residual_distance_x", 0.65, 0.005, "3.2"), ("eye_target_residual_distance_y", 0.76, 0.005, "3.2"),
                              ("hand_target_residual_distance_x", 0.85, 0.005, "3.3"), ("hand_target_residual_distance_y", 0.99, 0.005, "3.3"),
                              ("eye_pursuit_gain_x", 0.77, 0.005, "3.2"), ("eye_pursuit_gain_y", 0.71, 0.005, "3.2"),
                              ("eye_hand_error_correlation_x", 0.60, 0.005, "3.4"), ("eye_hand_error_correlation_y", 0.62, 0.005, "3.4"),
                              ("eye_cursor_distance_normal", 1.54, 0.005, "3.4"), ("eye_cursor_distance_switched", 1.65, 0.005, "3.4"),
                              ("cursor_target_distance_loop1", 1.82, 0.005, "3.4"), ("cursor_target_distance_loop2", 1.70, 0.005, "3.4"),
                              ("eye_target_distance_loop1", 1.41, 0.005, "3.4"), ("eye_target_distance_loop2", 1.44, 0.005, "3.4")]:
        checks.published(f"mean {nm}", M[nm].mean(), rep, tol, f"{P} {src}")
    checks.published("eye lag: vertical minus horizontal (ms)", M["eye_target_lag_y"].mean() - M["eye_target_lag_x"].mean(), 8, 0.5, f"{P} 3.2")
    checks.published("hand lag: vertical minus horizontal (ms)", M["hand_target_lag_y"].mean() - M["hand_target_lag_x"].mean(), 6, 0.5, f"{P} 3.3")
    checks.published("relative eye distance excess, vertical over horizontal (%)",
                     100 * (M["eye_target_distance_y"].mean() / M["eye_target_distance_x"].mean() - 1), 18, 0.5, f"{P} 3.2")
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), round(c.get("computed", 0), 4), c.get("reported", ""))
    checks.internal("620 trials x 28 measures, no missing values", len(L) == 620 * 28 and not L.v.isna().any(), f"{len(L)} values")
    checks.internal("every trial has a target pattern 1-5", all(1 <= pat[(s, t)] <= 5 for s, t in zip(L.s, L.t)))
    sp = pd.Series(pat).groupby(level=0).apply(lambda x: x.value_counts().to_dict())
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = s["papers"][0]
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Read the seven sheets of DATAmain.xlsx (62 participants x 10 trials x 28 measures) into long form, one row per trial and "
                 "measure in the order of the sheets, added the target pattern of each trial from pattern.txt, and replaced participant numbers "
                 "with codes. Nothing was recomputed; the continuous eye and hand records and the saccade table were not converted."),
        how=[
            "One row is one measure of one trial. Each participant tracked a pseudo-random smooth 2D target (mean speed 16 cm/s, 10 s per "
            "trial, the 5 s pattern repeated twice) with a joystick-controlled cursor while the right eye was recorded. The 10 trials are "
            "the baseline block of three experiments with the standard mapping; trial is the order within the participant (1-10) and "
            "condition gives the target pattern (1-5), which was randomised with each pattern shown twice and not consecutively.",
            "Measures are the authors' per-trial values: mean absolute distances (cm) between hand cursor, gaze and target on the "
            "horizontal (x) and vertical (y) axes, the lag (ms) of a best-fit time shift, the distances after that shift (residual), the "
            "smooth pursuit gain, the correlation of eye and hand error, and loop-wise and normal-versus-switched eye-cursor distances. "
            "Horizontal is x and vertical is y.",
            "The deposit gives no per-participant age or sex (62 volunteers, 31 female, 26.8 +- 6.2 years, per the paper) and does not "
            "say which of the three source experiments each participant came from, so those columns are empty.",
        ],
        unit_note="measure of one 10 s tracking trial",
        differences=[
            "The deposit labels the fourth block of the HAND-TARGET, EYE-TARGET and EYE-CURSOR sheets 'DIST on Y axis (ms)'; it is read here as "
            "the lag on the y axis (unit ms, third and fourth blocks being x and y lag).",
        ],
    )
    return stdlib.release(ID, "summary", rows, meta, checks)


if __name__ == "__main__":
    main()
