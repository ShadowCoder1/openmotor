"""Klar, Fischer, Fleig, Bachinski & Muller (2023), ACM Trans Comput-Hum Interact 30(3):1-50: mid-air ISO multidirectional pointing in
virtual reality by 6 users with 4 interaction techniques (Virtual Cursor / Virtual Pad x Ergonomic / ID conditions).

Raw file, raw_extra/Klar2023_TOCHI/ISO_VR_Pointing_IK_Raw.zip (Zenodo 7300062, 48 MB, CC BY 4.0; the 1.65 GB
ISO_VR_Pointing_Dataset.zip with the marker data and per-movement metadata was not used): 30 OpenSim .mot files, one per user
(U1-U6) and technique plus one T-pose calibration per user. Each has a text header (nRows, nColumns, inDegrees=yes, 'Units are S.I.
units, angles in degrees'), then a tab separated table of time and 26 joint coordinates (thorax rotations and translations, shoulder
girdle, shoulder, elbow, forearm, wrist) sampled at 240 Hz (time column rounded to 4 decimals).
Run from the site folder: python3 -I tools/std/convert/Klar2023_TOCHI.py
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Klar2023_TOCHI"
SRC = "Klar et al. 2023, ACM TOCHI 30(3)"
ZIP = "ISO_VR_Pointing_IK_Raw.zip"
TECH = {"Virtual_Cursor_Ergonomic": "Virtual Cursor, Ergonomic", "Virtual_Cursor_ID": "Virtual Cursor, ID",
        "Virtual_Pad_Ergonomic": "Virtual Pad, Ergonomic", "Virtual_Pad_ID": "Virtual Pad, ID"}


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def parse(txt):
    t = txt.splitlines()
    k = t.index("endheader")
    names = t[k + 1].split("\t")
    a = np.array([[float(x) for x in l.split("\t")] for l in t[k + 2:] if l.strip()])
    return names, a


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(ID) / ZIP)
    files = {}
    for i in z.infolist():
        m = re.match(r"U(\d)_(.*)\.mot$", i.filename)
        files[(int(m.group(1)), m.group(2))] = i
    rows, extra, rates, techs = [], [], [], set()
    order = sorted(files, key=lambda k: (k[0], 0 if "pose" in k[1].lower() else 1, k[1]))
    seen = {}
    for u, rest in order:
        names, a = parse(z.read(files[(u, rest)]).decode("utf-8"))
        chan = names[1:]
        assert names[0] == "time" and a.shape[1] == len(names)
        scode = f"{ID}_S{u:04d}"
        rec = seen[u] = seen.get(u, 0) + 1
        tpose = "pose" in rest.lower()
        key = rest.replace("_ISO_15_plane", "")
        cond = "T-pose calibration" if tpose else TECH[key]
        if not tpose:
            techs.add(cond)
        fn = f"samples/{scode}_rec{rec:02d}_{'tpose' if tpose else key.lower()}.csv"
        lines = [",".join(names)] + [",".join(f"{v:.8g}" for v in r) for r in a]
        extra.append((fn, ("\r\n".join(lines) + "\r\n").encode()))
        dur = a[-1, 0] - a[0, 0]
        rate = (len(a) - 1) / dur
        rates.append(rate)
        units = ";".join("m" if re.search(r"_t[xyz]$", c) else "deg" for c in chan)
        rows.append({"dataset": ID, "study": SRC, "experiment": "", "group": "", "subject": f"U{u}", "subject_excluded_by_authors": False,
                     "age": "", "sex": "", "handedness": "right", "population": "healthy", "setting": "lab",
                     "device": "Phasespace X2E13 optical motion capture (240 Hz, 14 arm and torso markers) and HTC Vive Pro headset",
                     "session": 1, "block": "", "trial": rec, "phase": "other" if tpose else "test", "time_s": "", "task": "reaching",
                     "unit": "recording", "condition": cond, "pert_type": "none", "pert_value": 0, "pert_unit": "",
                     "feedback": "normal", "excluded_by_authors": False, "recording": rec, "samples_file": fn,
                     "signal_kind": "joint angles of a musculoskeletal arm and torso model from OpenSim inverse kinematics (rotations in degrees, thorax translations in metres)",
                     "channels": ";".join(chan), "channel_units": units, "sample_rate_hz": 240, "n_samples": len(a),
                     "duration_s": round(dur, 4), "events": "",
                     "processing": "Markers cleaned (outliers removed, gaps interpolated), smoothed with a Kalman filter, then OpenSim inverse kinematics on a model scaled to the user from the T-pose (paper, Section 5.2); one file per whole interaction-technique session, not cut into movements",
                     "trial_info": "" if tpose else "ISO multidirectional pointing, 65 movements per technique (paper); 15 targets in a plane (from file name ISO_15_plane)"})
    checks = stdlib.Checks()
    checks.published("participants (U1-U6)", len({r["subject"] for r in rows}), 6, 0, f"{SRC}, Section 5.2.1 Participants")
    checks.published("interaction techniques", len(techs), 4, 0, f"{SRC}, Section 5.2 (four interaction techniques)")
    checks.published("motion capture sampling rate (Hz), mean of all recordings", float(np.mean(rates)), 240, 0.5, f"{SRC}, Section 5.2.1 (240 Hz)")
    checks.internal("30 recordings = 6 users x (4 techniques + T-pose)", len(rows) == 30 and len(extra) == 30)
    checks.internal("every file has 26 joint coordinates and no missing values", all(r["channels"].count(";") == 25 for r in rows))
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Re-wrote the 30 OpenSim .mot files as CSV (time_s column first, same values), described each in a row of the main "
                 "table, and replaced user labels with codes. The header text was not copied; values and units are unchanged."),
        how=["One row is one recording: the whole session of one user with one interaction technique (or the user's T-pose calibration), "
             "as 26 joint coordinates from OpenSim inverse kinematics at 240 Hz (rotations in degrees; the three thorax translations in metres). "
             "Users did a mid-air ISO multidirectional pointing task in virtual reality with the right index finger, 65 movements per technique.",
             "Only the 48 MB IK subset of the deposit was converted; the 1.65 GB main file (marker data, target times and movement metadata) was not, "
             "so there are no event times or per-movement cuts here. Age and sex are not given per user (paper: mean age 28.8, SD 6.6, 4 male, all right-handed)."],
        unit_note="recording (one user, one interaction technique)",
        differences=["The paper reports 1560 pointing movements after cutting at target switches; the session recordings here are not cut, so that count "
                     "could not be reproduced from these rows."],
    )
    return stdlib.release(ID, "timeseries", rows, meta, checks, extra_files=extra)


if __name__ == "__main__":
    main()
