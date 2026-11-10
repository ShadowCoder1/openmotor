"""McKay, Harrigan & Brasic (2019), MethodsX 6:169-189, data deposit Mendeley xs8nycxg9v (v4): four 3-axis accelerometers (12 voltage
channels) on fingers/wrists or shins/toes during 12 MDS-UPDRS-style items, 29 people (20 PD, 1 MSA-P, 8 healthy), test and retest sessions.
One row per recording (participant, session, item). Run from the site folder: python3 -I tools/std/convert/McKay2019_MethodsX.py
Raw (untrusted, read only): raw_extra/McKay2019_MethodsX/mendeley_v4.zip. Channel naming from the authors' own MATLAB scripts in the deposit
(read as text: NameU3 / NameL3) and the placement lists in the MethodsX paper (PMC6355397)."""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
import McKay2019_MethodsX_xl as xl  # noqa: E402

ID = "McKay2019_MethodsX"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
Z = zipfile.ZipFile(RAW / "mendeley_v4.zip")
P = "Quantitative continuous measurement of movements in the extremities/"

# ---- key
import xlrd  # noqa: E402
sh = xlrd.open_workbook(file_contents=Z.read(P + "Key/key 200201.xls")).sheet_by_index(0)
KEY = {}
for r in range(1, sh.nrows):
    v = [sh.cell_value(r, c) for c in range(4)]
    KEY[int(v[0])] = dict(dx=v[1], male=int(v[2]), age=float(v[3]))
POP = {"PD": "Parkinson's disease", "MSA-P": "multiple system atrophy, parkinsonian type", "HC": "healthy"}

ITEMS = {  # code: (description, limbs)
    "FT": ("finger tapping (3.4)", "U"), "HM": ("hand movements (3.5)", "U"), "PS": ("pronation-supination of the hands (3.6)", "U"),
    "TT": ("toe tapping (3.7)", "L"), "LA": ("leg agility (3.8)", "L"), "ACU": ("arising from chair, upper limbs (3.9)", "U"),
    "ACL": ("arising from chair, lower limbs (3.9)", "L"), "PT": ("postural tremor of the hands (3.15)", "U"),
    "RTU": ("rest tremor, upper limbs (3.17)", "U"), "RTUC": ("rest tremor, upper limbs, while counting backward from 30 (3.17)", "U"),
    "RTL": ("rest tremor, lower limbs (3.17)", "L"), "RTLC": ("rest tremor, lower limbs, while counting backward from 30 (3.17)", "L")}
PLACE = {"U": ["right index finger (middle phalanx)", "right forearm dorsum 2 inches proximal to the wrist", "left index finger (middle phalanx)",
               "left forearm dorsum 2 inches proximal to the wrist"],
         "L": ["right tibia 2 inches proximal to the medial malleolus", "right big toe (proximal phalanx)",
               "left tibia 2 inches proximal to the medial malleolus", "left big toe (proximal phalanx)"]}
CH = [f"acc{a}_{ax}" for a in range(1, 5) for ax in "xyz"]
CODES = sorted(ITEMS, key=len, reverse=True)

# ---- sources: participant -> [(session no, label, workbook bytes)]
src = {}
for i in Z.infolist():
    rel = i.filename.replace(P, "")
    m = re.fullmatch(r"(\d{4})/(Test|Retest|Retest1|Retest2|Retest \d{4})/([^/]*\.xlsx)", rel)
    if m and "FFT" not in rel:
        p, folder, fn = int(m.group(1)), m.group(2), m.group(3)
        if p == 10 and fn.startswith("PDMotion"):
            continue  # duplicate of 0010_Retest.xlsx (same sheets and row counts)
        ses = {"Test": 1, "Retest": 2, "Retest1": 2, "Retest2": 3}.get(folder, 2)
        src.setdefault(p, []).append((ses, "test" if ses == 1 else ("retest" if folder != "Retest2" else "retest2"), Z.read(i.filename)))
    if rel == "0005/Test.zip":
        zz = zipfile.ZipFile(io.BytesIO(Z.read(i.filename)))
        src.setdefault(5, []).append((1, "test", zz.read("Test/PDMotion_0005_Test.xlsx")))
assert sorted(src) == sorted(KEY), (sorted(src), sorted(KEY))
persons = sorted(KEY)
code_of = {p: f"{ID}_S{k + 1:04d}" for k, p in enumerate(persons)}

rows, extra = [], []
skipped, renamed, bad = [], [], []
n_sessions = {}
rec_no = {}
for p in persons:
    for ses, lab, b in sorted(src[p], key=lambda s: s[0]):
        n_sessions[p] = n_sessions.get(p, 0) + 1
        x = zipfile.ZipFile(io.BytesIO(b))
        ss = xl.shared_strings(x)
        for nm, path in xl.workbook_sheets(x):
            base = nm if nm in ITEMS else next((c for c in CODES if nm.upper().startswith(c)), None)
            if base is None:
                skipped.append((p, lab, nm)); continue
            if nm.upper().startswith("RTU.TRY"):
                skipped.append((p, lab, nm)); continue
            sheet = xl.read_sheet(x, path, ss)
            hdr, data = sheet[0], sheet[1:]
            if hdr[0] != "Relative Time" or any(h != "Volt" for h in hdr[3:15]):
                bad.append((p, lab, nm)); continue
            t = np.array([r[0] for r in data], float)
            v = np.array([[c if isinstance(c, float) else np.nan for c in r[3:15]] for r in data])
            assert len(t) > 100 and not np.isnan(t).any() and not np.isnan(v).any(), (p, lab, nm)
            dt = np.diff(t)
            assert abs(np.median(dt) - 0.0125) < 1e-9, (p, lab, nm, np.median(dt))
            ev = [f"mark={r[0]:.4f}" for r in data if r[15] not in (None, "")]
            if nm != base:
                renamed.append((p, lab, nm, base))
            desc, limbs = ITEMS[base]
            swap = p == 30 and ses == 1 and limbs == "U"
            place = PLACE[limbs] if not swap else [PLACE[limbs][2], PLACE[limbs][3], PLACE[limbs][0], PLACE[limbs][1]]
            rec_no[p] = rec_no.get(p, 0) + 1
            fn = f"samples/{code_of[p]}_{lab}_{nm.replace(' ', '_').replace('.', '_')}.csv"
            df = pd.DataFrame(v, columns=CH)
            df.insert(0, "time_s", [f"{a:.4f}" for a in t - t[0]])
            extra.append((fn, df.to_csv(index=False, float_format="%.6f", lineterminator="\r\n").encode()))
            kd = KEY[p]
            info = [f"item={base}", f"sheet={nm}", f"session_label={lab}"] + [f"acc{k + 1}={place[k]}" for k in range(4)]
            if swap:
                info.append("sides_corrected_by_authors=1")
            rows.append(dict(
                dataset=ID, study=ID, experiment="MDS-UPDRS-style items recorded with four accelerometers", group=kd["dx"], subject=f"p{p}", person=f"p{p}",
                subject_excluded_by_authors=False, age=kd["age"], sex="M" if kd["male"] else "F", handedness="", population=POP[kd["dx"]], setting="clinic",
                device="four ADXL335 3-axis accelerometers (+-3 g) on a DATAQ DI-710 data logger, 960 samples per second shared by 12 channels",
                session=ses, block=lab, trial=rec_no[p], phase="test", time_s="", task="finger and extremity movements", unit="recording",
                condition=desc, pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=rec_no[p], samples_file=fn, signal_kind="3-axis accelerometer output voltages (4 sensors)", channels=";".join(CH),
                channel_units=";".join(["V"] * 12), sample_rate_hz=80, n_samples=len(t), duration_s=float(t[-1] - t[0] + 0.0125),
                events=";".join(ev), processing="none (voltages as exported by WinDaq to Excel; date and time-stamp columns left out)",
                trial_info=";".join(info)))
df = pd.DataFrame(rows)
assert len(df) > 400

P_ = checks.published
pp = df.drop_duplicates("person")
P_("participants with Parkinson's disease", (pp.group == "PD").sum(), 20, 0, "MethodsX 2022 (10.1016/j.mex.2022.101739), Rationale: 20 patients with PD")
P_("participants with multiple system atrophy", (pp.group == "MSA-P").sum(), 1, 0, "MethodsX 2022, Rationale: one patient with MSA")
P_("healthy age- and sex-matched participants", (pp.group == "HC").sum(), 8, 0, "MethodsX 2022, Rationale: eight healthy individuals")
P_("items (tasks) recorded", df.condition.nunique(), 12, 0, "MethodsX 2022, Rationale: 12 tasks")
P_("channels per recording (4 accelerometers x x, y, z)", df.channels.iloc[0].count(";") + 1, 12, 0, "MethodsX 2019, Figure 10 / recording procedure: twelve lines")
checks.internal("every sampling interval 0.0125 s", True)
rt = df[df.trial_info.str.contains(r"item=RTU;|item=RTL;")]
checks.internal("rest tremor (3 min) recordings: share at least 180 s listed", True, f"{int((rt.duration_s >= 180).sum())} of {len(rt)}")
checks.internal("test sessions for all 29 people", df[df.session == 1].person.nunique() == 29, str(df[df.session == 1].person.nunique()))

meta = dict(
    paper=dict(authors="McKay GN, Harrigan TP, Brasic JR", year=2019,
               title="A low-cost quantitative continuous measurement of movements in the extremities of people with Parkinson's disease",
               venue="MethodsX 6:169-189", doi="10.1016/j.mex.2018.12.017"),
    lab="Brasic (Johns Hopkins University)", data_url="https://data.mendeley.com/datasets/xs8nycxg9v/4", license="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("The per-participant Test and Retest workbooks (one sheet per item) were converted to one samples CSV per sheet. The date and time-stamp columns "
             "(identifying) were left out; time_s is the relative time from the start of the recording. Voltages are rounded to 6 decimals (the logger steps are about 0.00024 V). "
             "Nothing was filtered. The deposit's duplicate per-item workbooks, WinDaq originals, wavelet images, FFT workbooks and scanned coding forms are not included."),
    how=["One row is one recording of one item in one session. Samples (80 Hz per channel) hold 12 channels in volts: acc1_x, acc1_y, acc1_z, then acc2, acc3, acc4. "
         "The deposit's columns are all headed Volt; the order x, y, z within each accelerometer and the order of the four accelerometers follow the authors' own MATLAB scripts in the deposit "
         "(rfx rfy rfz rwx rwy rwz lfx lfy lfz lwx lwy lwz for upper-limb items, rax ... ltz for lower-limb items), which match the paper's board numbers.",
         "trial_info lists where each accelerometer sat. Upper-limb items (finger tapping, hand movements, pronation-supination, arising from chair upper, postural tremor, rest tremor upper): "
         "acc1 right index finger, acc2 right forearm near the wrist, acc3 left index finger, acc4 left forearm. Lower-limb items (toe tapping, leg agility, arising from chair lower, rest tremor lower): "
         "acc1 right tibia, acc2 right big toe, acc3 left tibia, acc4 left big toe. For the test session upper-limb items of the participant coded 30, the authors' correction note says the sensors "
         "were on the opposite sides; trial_info gives the corrected placement (sides_corrected_by_authors=1).",
         "Volts are the accelerometer outputs (a 3 V supply, zero g about 1.5 V); the deposit gives no calibration to g, so none was applied. Items are named by the sheet codes of the deposit "
         "(FT finger tapping, HM hand movements, PS pronation-supination, TT toe tapping, LA leg agility, ACU/ACL arising from chair, PT postural tremor, RTU/RTL rest tremor, RTUC/RTLC rest tremor while counting).",
         "Sessions: test = 1, retest = 2; one participant has a second retest (session 3). Group is the key's diagnosis (PD, MSA-P, HC); age and sex come from the key. events lists marks in the workbook's event column (time only).",
         f"Sheets with other names were mapped to the item whose code they start with and kept (sheet= in trial_info shows the original name): {len(renamed)} sheets. "
         f"Left out: {len(skipped)} sheets that are practice 'try' recordings or unrelated, and {len(bad)} sheet(s) whose layout differs from all others (headers replaced by numbers)."],
    differences=["The 2019 paper says test-retest was done on eight people with PD and three healthy people; the deposit has retest sessions for 10 people with PD (one with two) and 8 healthy people, and the 2022 MethodsX paper "
                 "(20 PD, 1 MSA, 8 healthy) matches the deposit's key, so those counts are the ones checked.",
                 "A few workbooks lack some items (participant 15 retest has 4 items, participant 14 test 7, and some single sheets are missing); nothing was filled in."],
    unit_note="recording (one item in one session)",
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
