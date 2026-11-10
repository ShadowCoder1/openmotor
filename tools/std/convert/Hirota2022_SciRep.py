"""Hirota et al. (2022) Sci Rep 12:8501: smooth pursuit eye movements in nine-gaze testing, with the target and both eyes tracked.

Raw file: raw_extra/Hirota2022_SciRep/data.zip (Zenodo 6400751): S01..S23 .xlsx (sheet 'Target data': row index, TimeStamp,
Target_names, cx, cy; sheet 'Gaze data': row index, Time, LE_x, LE_y, RE_x, RE_y) and S01..S23 .mp4 (video, not carried).
Run from the site folder: python3 -I tools/std/convert/Hirota2022_SciRep.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hirota2022_SciRep"
SRC = "Hirota et al. 2022, Sci Rep 12:8501"


def sheet(wb, name):
    rows = list(wb[name].iter_rows(values_only=True))
    return rows[0], np.array([[np.nan if v is None else v for v in r[1:]] if name == "Gaze data"
                              else [r[1], np.nan, r[3], r[4]] for r in rows[1:]], dtype=float)


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(ID) / "data.zip")
    ck = stdlib.Checks()
    rows, extra, dts = [], [], []
    for i in range(1, 24):
        wb = openpyxl.load_workbook(io.BytesIO(z.read(f"MH_Analysis_SPEM_Dataset/S{i:02d}.xlsx")), read_only=True, data_only=True)
        _, t = sheet(wb, "Target data")
        _, g = sheet(wb, "Gaze data")
        ck.internal(f"participant {i}: target and gaze sheets have the same number of rows", len(t) == len(g))
        n = len(g)
        dts.append(np.median(np.diff(g[:, 0])))
        out = pd.DataFrame({"time_s": g[:, 0], "target_timestamp": t[:, 0], "target_x_deg": t[:, 2], "target_y_deg": t[:, 3],
                            "left_eye_x_deg": g[:, 1], "left_eye_y_deg": g[:, 2],
                            "right_eye_x_deg": g[:, 3], "right_eye_y_deg": g[:, 4]})
        fn = f"samples/{ID}_S{i:03d}.csv"
        extra.append((fn, out.to_csv(index=False, float_format="%.9g", lineterminator="\r\n")))
        rows.append(dict(
            dataset=ID, study=ID, experiment="Nine-direction smooth pursuit", group="", subject=f"S{i:03d}", person=f"S{i:03d}",
            subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="clinic",
            device="Video-oculography (240 Hz, corneal reflex and pupil centre) with a scene camera (29.97 Hz); target found in the scene images by an SSD object detector",
            session=1, block="nine-direction test", trial=1, phase="test", time_s="", task="eye movements", unit="recording",
            condition="examiner moves a target by hand to nine directions", pert_type="none", pert_value=0.0, pert_unit="",
            feedback="normal", excluded_by_authors=False, recording=i, samples_file=fn,
            signal_kind="target position found in the scene camera images and left and right eye position (visual angle)",
            channels="target_timestamp;target_x_deg;target_y_deg;left_eye_x_deg;left_eye_y_deg;right_eye_x_deg;right_eye_y_deg",
            channel_units="authors' frame stamp;deg;deg;deg;deg;deg;deg", sample_rate_hz=29.97, n_samples=n,
            duration_s=float(g[-1, 0]), events="",
            processing="eye positions (240 Hz) synchronised to the scene camera frames (29.97 Hz) by the authors; blinks interpolated linearly",
            trial_info="target_names=target"))
    df = pd.DataFrame(rows)
    ck.internal("23 recordings, 2 sheets each with a header row", len(df) == 23 and df.n_samples.min() > 1000, f"{df.n_samples.min()}-{df.n_samples.max()} samples")
    ck.internal("no missing gaze values", all(not out_.isna().any().any() for out_ in [pd.read_csv(io.StringIO(b)) for _, b in extra[:3]]))
    ck.published("participants", df.person.nunique(), 23, 0, f"{SRC}, Methods, Subjects (23 young adult volunteers)")
    ck.published("scene-camera sample rate (Hz), from the time column", 1 / np.median(dts), 29.97, 0.01, f"{SRC}, Methods, apparatus (29.97 Hz)")
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = s["papers"][0]
    meta = dict(
        study=ID, paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Each participant's two Excel sheets (target and gaze) were merged into one samples CSV (rows in the authors' order, values unchanged); the examination videos were not carried.",
        how=["One row is one participant's examination: an examiner moved a rabbit-like target by hand to nine directions (centre, left, right, upper left, upper right, lower left, lower right, upper, lower) and the participant followed it with a chin rest; the examination lasted about 40-70 s.",
             "The samples file has time_s (seconds, one sample per scene-camera frame at 29.97 Hz), the authors' target_timestamp, the target position (target_x_deg, target_y_deg) found in the scene images, and the left and right eye position, all in degrees of visual angle with right and up positive (paper: calibration plate, right and upper halves positive).",
             "The unit of the positions is not printed in the deposit; degrees are inferred from the paper, which analyses all positions in degrees. The values are not corrected for the relative origin (the median target location) that the paper applies in its analysis. Age and sex were not shared per participant (paper: 21.4 +- 1.4 years)."],
        unit_note="recording (one participant's examination)",
        differences=["The paper's Table 2 (184 direction classifications) and Table 3 (latencies and gains) come from the authors' own detection algorithm and were not re-derived."])
    return stdlib.release(ID, "timeseries", df, meta, ck, extra_files=extra)


if __name__ == "__main__":
    main()
