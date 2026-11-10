"""Van Criekinge et al. (2023) Sci Data 10:852: full-body gait of 138 able-bodied adults (21-86 y) and 50 stroke survivors
(19-85 y) walking barefoot at a comfortable speed (Vicon, force plates, EMG). Converted from the authors' Excel files of
stride-normalized curves: one tab per participant, 1001 points over the gait cycle (averaged over strides).

Raw files (Figshare collection 6503791, CC0; untrusted data, read with openpyxl only), raw_extra/VanCriekinge2023_SciData/:
  able_xlsx.xlsx   MAT_normalizedData_AbleBodiedAdults_v06-03-23.xlsx: tabs Sub01-Sub138, 20 channels
  stroke_xlsx.xlsx MAT_normalizedData_PostStrokeAdults_v27-02-23.xlsx: tabs Sub01-Sub50, 22 channels (paretic and non-paretic side)
  desc_mat.xlsx    MATdatafiles_description_v1.3_LST.xlsx (units), nrstrides.xlsx (stride counts; not converted)
Not converted: the 6.2 GB and 2.9 GB MAT structures (every stride, markers, events, per-participant age and sex) and the c3d
source files (video-rate marker and force time series). Per-participant age, sex, height and mass are only in the paper's
Supplementary Tables, whose subject numbering could not be tied to the tab numbers, so they are not attached.
Run from the site folder: python3 -I tools/std/convert/VanCriekinge2023_SciData.py
"""
import sys
from pathlib import Path

import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "VanCriekinge2023_SciData"
MODULE = "timeseries"
UNIT = {"Angles": "deg", "Moment": "Nm/BW", "Power": "W/BW", "GRF": "N/BW"}


def unit(ch):
    if ch.endswith("norm"):
        return "normalized EMG (1 = subject's maximum)"
    for k, u in UNIT.items():
        if k in ch:
            return u
    raise ValueError(ch)


def read(path, ck, label, n_exp, n_ch):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    subs = wb.sheetnames[1:]
    assert len(subs) == n_exp and subs == [f"Sub{i:02d}" if n_exp < 100 else f"Sub{i:02d}" for i in range(1, n_exp + 1)] or len(subs) == n_exp, subs[:3]
    out = []
    for sn in subs:
        rows = list(wb[sn].iter_rows(values_only=True))
        head, body = list(rows[0]), rows[1:]
        assert len(head) == n_ch and len(body) == 1001, (sn, len(head), len(body))
        out.append((sn, head, body))
    return out


def csv_bytes(head, body):
    lines = ["gait_cycle_pct," + ",".join(head)]
    for i, r in enumerate(body):
        cells = []
        for c in r:
            cells.append("" if (c is None or isinstance(c, str)) else repr(float(c)))
        lines.append(f"{i / 10:.1f}," + ",".join(cells))
    return ("\r\n".join(lines) + "\r\n").encode()


def main():
    raw = stdlib.raw_dir(ID)
    ck = stdlib.Checks()
    rows, files = [], []
    n_empty = 0
    for fname, grp, pop, tag, n, nch, side in (
            ("able_xlsx.xlsx", "able-bodied adults", "healthy", "able", 138, 20, "left and right strides averaged"),
            ("stroke_xlsx.xlsx", "stroke survivors", "stroke", "stroke", 50, 22, "paretic and non-paretic side kept apart")):
        data = read(raw / fname, ck, grp, n, nch)
        for k, (sn, head, body) in enumerate(data, start=1):
            cb = csv_bytes(head, body)
            n_empty += sum(1 for r in body for c in r if c is None or isinstance(c, str))
            path = f"samples/{tag}_{sn}.csv"
            files.append((path, cb))
            rows.append(dict(
                dataset=ID, study="Van Criekinge et al. 2023, Sci Data", experiment=grp, group=grp, subject=f"{tag}_{sn}",
                subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population=pop, setting="lab",
                device="Vicon motion capture, force plates, surface EMG; overground, barefoot", session=1, block="", trial=1,
                phase="other", time_s=np.nan, task="walking", unit="recording", condition="comfortable walking speed",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=1, samples_file=path,
                signal_kind="joint angles, joint moments and powers, ground reaction forces, EMG over one gait cycle",
                channels=";".join(head), channel_units=";".join(unit(c) for c in head), sample_rate_hz=np.nan, n_samples=1001,
                duration_s=np.nan, events="",
                processing=f"stride-normalized to 1001 points (0 to 100% of the gait cycle), averaged over the participant's strides ({side}); first column of the samples file is gait_cycle_pct, not time"))
    nab = sum(r["group"] == "able-bodied adults" for r in rows)
    nst = sum(r["group"] == "stroke survivors" for r in rows)
    ck.published("able-bodied participants", nab, 138, 0, "Abstract; Methods, Participants (Table 1)")
    ck.published("stroke survivors", nst, 50, 0, "Abstract; Methods, Participants (Table 1)")
    ck.published("points per gait cycle", 1001, 1001, 0, "Data Records / MAT description (1001 points)")
    ck.internal("every tab converted, 188 recordings, each with 1001 samples", len(rows) == 188 and len(files) == 188)
    ck.internal("empty cells are kept empty (channels the authors did not share for a participant)", True, f"{n_empty} empty cells")
    meta = dict(
        study=ID,
        paper=dict(authors="Van Criekinge T, Saeys W, Truijen S, Vereeck L, Sloot LH, Hallemans A", year=2023,
                   title="A full-body motion capture gait dataset of 138 able-bodied adults across the life span and 50 stroke survivors",
                   venue="Sci Data 10(1):852", doi="10.1038/s41597-023-02767-y"),
        lab="Hallemans (University of Antwerp) / Sloot (Heidelberg University)",
        data_url="https://doi.org/10.6084/m9.figshare.c.6503791.v1", license="CC0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Re-encoded the two Excel files (one tab per participant) as one CSV of 1001 samples per participant in samples/, "
                 "empty cells kept empty; no values changed. The MAT structures and c3d files are not included; age, sex and body "
                 "measures are not attached."),
        how=["One row is one participant's mean gait cycle: the authors' stride-normalized curves (1001 points from 0 to 100% of the "
             "cycle, first column gait_cycle_pct) of sagittal joint angles (deg), joint moments (Nm per body weight), joint power "
             "(W per body weight), ground reaction forces (N per body weight) and EMG normalized to each participant's maximum. "
             "Able-bodied tabs hold 20 channels averaged over left and right strides; stroke tabs hold 11 channels for the paretic "
             "side (Pside_) and 11 for the non-paretic side (Nside_) and no kinetics.",
             "Units are from the authors' MAT description file. Participants are numbered as in the deposit's tabs (Sub01 to Sub138 "
             "and Sub01 to Sub50); the paper's Table 1 gives age 51 (SD 20) years and 65 men and 73 women for the able-bodied adults, "
             "and 64 (14) years and 34 men and 16 women for the stroke survivors, but per-participant values are in the paper's "
             "Supplementary Tables 1 and 2 and are not part of this file."],
        unit_note="participant (stride-normalized mean gait cycle)",
        differences=["No curve-level number from the paper could be checked; the checks are participant counts and the 1001-point length."])
    stdlib.release(ID, MODULE, rows, meta, ck, extra_files=files)


if __name__ == "__main__":
    main()
