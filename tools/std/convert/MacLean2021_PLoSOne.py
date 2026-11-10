"""MacLean & Ferris (2021), PLoS ONE 16(7):e0253467: overground walking at four levels of simulated reduced gravity and four speeds.

Run from the site folder: python3 -I tools/std/convert/MacLean2021_PLoSOne.py
Raw files (untrusted data, read only) in raw_extra/MacLean2021_PLoSOne/, from https://doi.org/10.6084/m9.figshare.c.5430234 (item
'EMG and GRF data - Raw', figshare 14642817): <subject>_Raw_Data.mat for the 12 participants (the file names are the authors' participant
pseudonyms, replaced by codes here). Each file: OrderingInfo (trial order table) and RawData (one struct per trial with NiDAQ: MocapFrame,
MocapTime, DeviceFrame, TimingGate, LoadCell, EMG [n x 8]; Fp1, Fp2, Fp3 force plate signals [n x 3]; PlateOrder).
The deposit's other items (Visual3D kinematics and kinetics, processed and averaged data, quasi-stiffness) are processed derivatives and are not used.
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "MacLean2021_PLoSOne"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()

# file code -> (fraction of gravity, % body weight supported) as listed in the Figshare collection description
LEVEL = {"00": (1.0, 0), "25": (0.76, 24), "50": (0.55, 45), "75": (0.31, 69)}
SPEED = {"0p4": 0.4, "0p8": 0.8, "1p2": 1.2, "1p6": 1.6}
MUSCLE = ["soleus", "tibialis_anterior", "medial_gastrocnemius", "lateral_gastrocnemius", "rectus_femoris", "vastus_medialis", "vastus_lateralis", "biceps_femoris"]
CH = ["emg_" + m for m in MUSCLE] + ["load_cell", "timing_gate"] + [f"fp{p}_{a}" for p in (1, 2, 3) for a in "xyz"]
CU = ["unstated (EMG as stored)"] * 8 + ["unstated (voltage-like)", "unstated (voltage-like)"] + ["N (inferred)"] * 9

files = sorted(RAW.glob("*_Raw_Data.mat"))
checks.internal("12 participant files", len(files) == 12, str(len(files)))
TMP = RAW / "_tmp_samples"
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir()
rows, extra = [], []
cell_counts, dev_ok, rates_ok = {}, True, True
for fi, f in enumerate(files, 1):
    pid = f"P{fi:02d}"
    m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
    od = m["OrderingInfo"]
    rd = m["RawData"]
    keys = list(rd._fieldnames)
    order = {}
    for i, r in enumerate(od[1:]):
        nm = str(r[0])
        if "_" not in nm:
            continue
        k = "T_" + nm.split("_", 1)[1]
        order[k] = (i, r)
    seq = sorted(keys, key=lambda k: (order[k][0] if k in order else 10_000 + keys.index(k)))
    for ti, k in enumerate(seq, 1):
        t = getattr(rd, k)
        mm = re.fullmatch(r"T_(\d\d)_(\dp\d)(_W)?_(\d+)", k)
        assert mm, k
        lev, spd, weighted, num = mm.group(1), mm.group(2), bool(mm.group(3)), mm.group(4)
        n = t.NiDAQ
        dev = np.asarray(n.DeviceFrame, dtype=np.int64)
        dev_ok &= bool(((np.diff(dev) % 65536) == 1).all())
        data = np.column_stack([np.asarray(n.EMG, float), np.asarray(n.LoadCell, float), np.asarray(n.TimingGate, float),
                                np.asarray(t.Fp1, float), np.asarray(t.Fp2, float), np.asarray(t.Fp3, float)])
        df = pd.DataFrame(data, columns=CH)
        df.insert(0, "time_s", np.arange(len(df)) / 1000.0)
        fn = f"samples/{ID}_{pid}_{k}.csv"
        p = TMP / f"{ID}_{pid}_{k}.csv"
        df.to_csv(p, index=False, float_format="%.8g", lineterminator="\r\n")
        extra.append((fn, p))
        cell = (pid, lev, spd, weighted)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        info = []
        if k in order:
            r = order[k][1]
            info = [f"order_info_right={r[1]}", f"order_info_left={str(r[2]).strip()}", f"pedar_index={r[3]}", f"contact_code={r[4]}"]
            if isinstance(r[5], str) and r[5].strip():
                info.append(f"error_note={r[5].strip()}")
        info += [f"file_trial_number={int(num)}", f"mocap_frame_at_start={int(np.asarray(n.MocapFrame)[0])}", f"mocap_time_at_start_s={float(np.asarray(n.MocapTime)[0]):g}",
                 "plate_order=" + "-".join(str(int(x)) for x in np.atleast_1d(t.PlateOrder))]
        g, bws = LEVEL[lev]
        rows.append(dict(
            dataset=ID, study=ID, experiment="overground walking with simulated reduced gravity", group="", subject=pid, person=pid, subject_excluded_by_authors=False,
            age="", sex="", handedness="", population="healthy young adults", setting="lab",
            device="8 m walkway with 3 AMTI force plates, Cometa wireless EMG, infrared timing gates, spring-based bodyweight support harness",
            session=1, block=f"{int(100 * g)}% gravity", trial=ti, phase="test", time_s="", task="walking", unit="recording",
            condition=f"{g:g} G ({bws}% bodyweight supported), {SPEED[spd]:g} m/s" + (", ankle weights" if weighted else ""),
            pert_type="bodyweight_support", pert_value=bws, pert_unit="% bodyweight", feedback="normal", excluded_by_authors=False,
            recording=ti, samples_file=fn, signal_kind="EMG of 8 right-leg muscles, bodyweight-support load cell, timing gate, 3 force plates (force components)",
            channels=";".join(CH), channel_units=";".join(CU), sample_rate_hz=1000, n_samples=len(df), duration_s=round(len(df) / 1000.0, 6), events="",
            processing="raw (unprocessed) as stored by the authors", trial_info=";".join(info)))
df = pd.DataFrame(rows)

per_p = df.groupby("person").size()
checks.published("participants", df.person.nunique(), 12, 0, "Methods, Data collection (12 participants)")
checks.published("gravity levels", df.pert_value.nunique(), 4, 0, "Methods, Data collection (4 simulated gravity levels)")
sp = {c[2] for c in cell_counts}
checks.published("walking speeds", len(sp), 4, 0, "Methods, Data collection (0.4, 0.8, 1.2 and 1.6 m/s)")
nw = {c: v for c, v in cell_counts.items() if not c[3]}
checks.published("good trials per condition (most common count)", pd.Series(list(nw.values())).mode()[0], 4, 0, "Methods, Data collection ('We recorded 4 good trials per condition')")
checks.published("EMG channels (muscles)", len(MUSCLE), 8, 0, "Methods, Data collection (8 muscles)")
# participants with no 0.31 G, 1.6 m/s trials
miss = sum(1 for p in df.person.unique() if (p, "75", "1p6", False) not in cell_counts)
checks.published("participants without the 0.31 G, 1.6 m/s condition", miss, 2, 0, "Results (two of twelve participants could not walk 1.6 m/s at 0.31 G)")
complete = sum(1 for p in df.person.unique() if all((p, l, s, False) in cell_counts for l in LEVEL for s in SPEED))
print("participants with all 16 conditions:", complete, "cells not at 4:", {c: v for c, v in cell_counts.items() if v != 4 and not c[3]})
checks.internal("DeviceFrame advances by 1 (1 ms) every sample in every trial", dev_ok)
checks.internal("every participant has a recording for each gravity level", all(df[df.person == p].pert_value.nunique() >= 3 for p in df.person.unique()))

meta = dict(
    paper=dict(authors="MacLean MK, Ferris DP", year=2021,
               title="Human muscle activity and lower limb biomechanics of overground walking at varying levels of simulated reduced gravity and gait speeds",
               venue="PLoS ONE 16(7):e0253467", doi="10.1371/journal.pone.0253467"),
    lab="Ferris (University of Florida)", data_url="https://doi.org/10.6084/m9.figshare.c.5430234", license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted the raw EMG, force plate, load cell and timing-gate arrays of the 12 participant MATLAB files to one CSV per trial. Participant file names (the "
             "authors' pseudonyms) were replaced by codes. The Visual3D kinematics and kinetics, processed and averaged data and quasi-stiffness items of the collection are not included."),
    how=["One row is one overground walking trial of one of 12 healthy young adults, walking over three force plates at 0.4, 0.8, 1.2 or 1.6 m/s while a spring-based bodyweight support "
         "system simulated reduced gravity (file codes 00, 25, 50, 75 = 1, 0.76, 0.55 and 0.31 G, i.e. 0, 24, 45 and 69 percent of bodyweight supported, as the Figshare description lists them; "
         "the paper's text gives 0.45 G for the third level, which is its 45 percent support). Trials at 1.2 m/s with ankle weights are marked in condition. Rows are in the authors' trial order (OrderingInfo).",
         "Samples (1000 Hz, unprocessed) hold 8 right-leg EMG channels (soleus, tibialis anterior, medial and lateral gastrocnemius, rectus femoris, vastus medialis and lateralis, biceps femoris), the bodyweight-support "
         "load cell and the timing gate (units not stated in the deposit) and three force plate signals fp1, fp2, fp3 with three columns each. The deposit does not name the columns or units; they are "
         "labelled x, y, z and N by inference: the third column peaks near body weight (about 500 to 950 N) while the first two are small, so z is vertical, and values of that size are newtons. The plates "
         "lie in a straight line in the order 3, 1, 2 along the walkway (PlateOrder). trial_info holds the authors' ordering table columns (their headers are 'Right', 'Left', 'PEDAR', "
         "a contact code F full contact, D definite contact but not full foot, U unsure, and 'Error'); the deposit does not define the Right and Left letters, so they are given as order_info_right and order_info_left.",
         "time_s is the sample index divided by 1000 (the device frame advances by 1 each sample); the motion capture frame and time at the start of each trial are in trial_info so the rows can be matched to the motion capture frames."],
    differences=["The paper and the Figshare description list the third gravity level differently (0.45 G in the paper's text, 0.55 G and 45 percent support in the Figshare description); the 45 percent support figure is used."],
    unit_note="trial (raw EMG, load cell, timing gate and force plate samples of one walking trial in a CSV inside the ZIP)",
)
try:
    stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
finally:
    shutil.rmtree(TMP, ignore_errors=True)
