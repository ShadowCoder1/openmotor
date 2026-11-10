"""Browne & Franz (2017) R Soc open sci 4(11):171673: young adults walking on a force-measuring treadmill, with and without
visual biofeedback of propulsive force. The deposit is one workbook with one 3D marker trajectory per participant and condition
(100 Hz, 60 s), so the format is the timeseries module, one row per participant and condition.

Raw file: raw_extra/Browne2017_RSOS/SubjectData.xlsx (Zenodo record 5001364, a copy of the Dryad deposit 10.5061/dryad.33hc6,
CC0). Sheet 'Experiment 1' (6000 rows x 360 columns): 10 condition blocks of 36 columns (row 1 = condition name), each with
12 participants (row 2 = Sub01..Sub12) x X, Y, Z (row 3), then samples. Sheet 'Experiment 2' (6001 rows used x 180 columns): 6
blocks of 30 columns, 10 participants. Values are millimetres. The paper (Methods 2.5) says the analyses used the sacrum and heel
markers recorded at 100 Hz; the workbook holds one marker per participant, unnamed, with a vertical (Z) height near 1 m, which
fits the sacrum marker the paper uses as the centre-of-mass surrogate. Run from the site folder: python3 -I tools/std/convert/Browne2017_RSOS.py"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Browne2017_RSOS"
MODULE = "timeseries"
CH = ["marker.X", "marker.Y", "marker.Z"]


def cond_exp1(label):
    kind, v = label.split()
    if kind == "Speed":
        return (f"normal walking at {v} m/s", "none", 0.0, "", "normal", "Normal walking")
    return (f"1.3 m/s with biofeedback of the propulsive force measured at {v} m/s", "force_target_biofeedback", float(v),
            "m/s (speed whose propulsive force is the target)", "altered", "Biofeedback")


def cond_exp2(label):
    if label.startswith("No Feedback"):
        return ("normal walking without biofeedback (sheet label: " + label + ")", "none", 0.0, "", "normal", "Normal walking")
    m = re.fullmatch(r"Biofeedback (-?\+?\d+%|Normal)", label)
    t = m.group(1)
    pct = 0.0 if t == "Normal" else float(t.rstrip("%"))
    return (f"biofeedback target {'equal to normal walking' if t == 'Normal' else t + ' of normal walking'} propulsive force",
            "force_target_biofeedback", pct, "% of normal-walking propulsive force", "altered", "Biofeedback")


def main():
    ck = stdlib.Checks()
    raw = stdlib.raw_dir(DS)
    wb = openpyxl.load_workbook(raw / "SubjectData.xlsx", read_only=True, data_only=True)
    tmp = raw / "_samples"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    rows, extra = [], []
    cnt = {}
    gaps = []
    for sheet, expname, cf in (("Experiment 1", "Experiment 1", cond_exp1), ("Experiment 2", "Experiment 2", cond_exp2)):
        ws = wb[sheet]
        data = [list(r) for r in ws.iter_rows(values_only=True)]
        w = max(len(r) for r in data)
        data = [r + [None] * (w - len(r)) for r in data]
        labels = [(j, v) for j, v in enumerate(data[0]) if v is not None]
        bounds = [j for j, _ in labels] + [w]
        seqs = {}
        for bi, (j0, label) in enumerate(labels):
            width = bounds[bi + 1] - j0
            assert width % 3 == 0
            subs = [data[1][j] for j in range(j0, j0 + width, 3)]
            assert all(re.fullmatch(r"Sub\d\d", s) for s in subs), subs
            for si, sname in enumerate(subs):
                j = j0 + 3 * si
                assert [data[2][j + k] for k in range(3)] == ["X", "Y", "Z"]
                cols = []
                for k in range(3):
                    col = [r[j + k] for r in data[3:]]
                    n = max(i for i, x in enumerate(col, 1) if x is not None)
                    assert all(x is None or isinstance(x, (int, float)) for x in col[:n]), "text inside a column"
                    cols.append(np.array([np.nan if x is None else x for x in col[:n]], float))
                n = len(cols[0])
                assert all(len(c) == n for c in cols), "X, Y, Z lengths differ"
                miss = np.isnan(np.column_stack(cols))
                assert (miss.all(axis=1) == miss.any(axis=1)).all(), "X, Y, Z are missing on different frames"
                if miss.any():
                    gaps.append((sheet, sname, label, int(miss.any(axis=1).sum())))
                df = pd.DataFrame(np.column_stack(cols), columns=CH)
                df.insert(0, "time_s", np.round(np.arange(n) * 0.01, 2))
                person = f"E{expname[-1]}_{sname}"
                seqs[person] = seqs.get(person, 0) + 1
                arc = f"samples/{expname.replace(' ', '')}_{sname}_cond{bi + 1:02d}.csv"
                path = tmp / Path(arc).name
                df.to_csv(path, index=False, float_format="%.8g", lineterminator="\r\n")
                extra.append((arc, str(path)))
                cond, pt, pv, pu, fb, block = cf(label)
                rows.append(dict(
                    dataset=DS, study=DS, experiment=expname, group="", subject=person, person=person,
                    subject_excluded_by_authors=False, age=np.nan, sex="", handedness="", population="healthy young adults",
                    setting="lab", device="Bertec dual-belt force-measuring treadmill with visual biofeedback screen; 14-camera "
                    "Motion Analysis motion capture, 100 Hz", session=1, block=block, trial=seqs[person], phase="test",
                    time_s=np.nan, task="walking", unit="recording", condition=cond, pert_type=pt, pert_value=pv, pert_unit=pu,
                    feedback=fb, excluded_by_authors=False, recording=seqs[person], samples_file=arc,
                    signal_kind="3D position of one marker (the sacrum, as the values and the paper suggest; unnamed in the workbook)",
                    channels=";".join(CH), channel_units="mm;mm;mm", sample_rate_hz=100, n_samples=n, duration_s=round(n * 0.01, 2),
                    events="", processing="none stated in the workbook (the paper low-pass filtered sacrum data at 12 Hz in its own analysis)"))
        cnt[sheet] = (len(labels), len(subs))
    out = pd.DataFrame(rows)
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns:
            out[c] = np.nan
    e1, e2 = out[out.experiment == "Experiment 1"], out[out.experiment == "Experiment 2"]
    ck.internal("Experiment 1: 10 conditions (5 speeds, 5 biofeedback) x 12 participants = 120 recordings",
                len(e1) == 120 and cnt["Experiment 1"] == (10, 12), str(cnt))
    ck.internal("Experiment 2: 6 conditions (-20, -10, 0, +10, +20% targets and normal walking without feedback) x 10 participants = 60 recordings",
                len(e2) == 60 and cnt["Experiment 2"] == (6, 10), str(cnt))
    ck.internal("X, Y and Z have equal length and are missing on the same frames; the only missing frames are 62 in one recording "
                "(Experiment 1, participant 9, biofeedback 1.3), left empty",
                bool(out.n_samples.min() > 5900) and len(gaps) == 1 and gaps[0][3] == 62, f"{out.n_samples.min()}-{out.n_samples.max()} samples; gaps {gaps}")
    ck.published("Experiment 1 participants", e1.person.nunique(), 12, 0, "Methods 2.1")
    ck.published("Experiment 2 participants", e2.person.nunique(), 10, 0, "Methods 2.1")
    ck.published("Experiment 1 speeds walked (0.9, 1.0, 1.1, 1.2, 1.3 m/s)", e1[e1.block == "Normal walking"].condition.nunique(), 5, 0, "Methods 2.3")
    ck.published("Experiment 1 biofeedback targets from the five speeds", e1[e1.block == "Biofeedback"].condition.nunique(), 5, 0, "Methods 2.3")
    ck.published("Experiment 2 biofeedback targets (mean, +-10%, +-20%)", e2[e2.block == "Biofeedback"].condition.nunique(), 5, 0, "Methods 2.4")
    ck.published("marker sampling rate (Hz)", out.sample_rate_hz.median(), 100, 0, "Methods 2.5")
    ck.published("Experiment 1 trial length, median (1 min each)", e1.duration_s.median(), 60, 0.1, "Methods 2.3")

    meta = dict(
        study=DS,
        paper=dict(authors="Browne MG, Franz JR", year=2017,
                   title="Does dynamic stability govern propulsive force generation in human walking?",
                   venue="R Soc open sci 4(11):171673", doi="10.1098/rsos.171673"),
        lab="Franz (UNC / NC State)",
        data_url="https://doi.org/10.5061/dryad.33hc6",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read the one workbook (Zenodo copy of the Dryad deposit), wrote each participant's X, Y, Z columns in each "
                 "condition block as a CSV with a time_s column (sample number x 0.01 s) and the values unchanged, and "
                 "replaced Sub01-Sub12 with participant codes."),
        how=["One row is one participant in one condition (one minute of walking): its samples file has time_s and the three "
             "coordinates of one marker in millimetres at 100 Hz. The workbook does not name the marker or the axes; the "
             "values (height near 1 m) fit the sacrum marker the paper analyses.",
             "Experiment 1 (12 participants): normal walking at 0.9, 1.0, 1.1, 1.2 and 1.3 m/s, and walking at 1.3 m/s while "
             "matching the propulsive force measured at each of those speeds (biofeedback; pert_value is that speed). "
             "Experiment 2 (10 participants): walking while matching targets of -20, -10, 0, +10 and +20% of the propulsive force of "
             "normal walking, and one further condition the sheet calls 'No Feedback at 1.3 m/s'. trial counts the "
             "conditions in the sheet's order, not the order walked (the paper randomized them).",
             "In one recording (Experiment 1, participant 9, biofeedback at 1.3 m/s) 62 frames are empty in the workbook (marker dropouts); they are empty in the samples file too. "
             "The workbook does not say whether Sub01-Sub10 of Experiment 2 are among the 12 of Experiment 1, so the two "
             "experiments' participants are coded separately. Age and sex per participant are not shared (the paper gives group means)."],
        unit_note="recording (one minute of one marker's position for one participant and condition)",
        differences=["The paper does not state the treadmill speed of Experiment 2 or the length of its trials; the sheet "
                     "labels the sixth condition 'No Feedback at 1.3 m/s' and every recording has 6000-6001 samples (60 s). "
                     "The paper's results (step lengths and widths, sacrum variability, divergence exponents) need the "
                     "heel markers and force data, which the workbook does not contain, so only counts and recording "
                     "properties are checked."],
    )
    stdlib.release(DS, MODULE, out, meta, ck, extra_files=extra)
    shutil.rmtree(tmp, ignore_errors=True)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  P {c['what'][:70]} {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
