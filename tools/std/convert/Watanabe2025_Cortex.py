"""Watanabe et al. (2025), Cortex 186:51-60: dual-site TMS during a visuomotor force-control task, one row per concatenated recording.

Run from the site folder: python3 -I tools/std/convert/Watanabe2025_Cortex.py
Raw files (untrusted data, read only) in raw_extra/Watanabe2025_Cortex/, from https://osf.io/eac53/:
  r1-10.zip, r11-16.zip: per participant, High gain and Low gain folders, one text file per conditioning-stimulation site (M1, PMd, PMv, SMA, DLPFC),
  tab-separated, 1 kHz, columns Force(N), EMG_FDI_right(mV), EMG_FDI_left(mV), Test(V), Conditioning(V) (README.txt on OSF).
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Watanabe2025_Cortex"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
TMP = RAW / "samples_tmp"
TMP.mkdir(exist_ok=True)
HDR = ["Force(N)", "EMG_FDI_right(mV)", "EMG_FDI_left(mV)", "Test(V)", "Conditioning(V)"]
SITES = ["M1", "PMd", "PMv", "SMA", "DLPFC"]
GAIN = {"High": ("high", 3.0), "Low": ("low", 0.05)}
pat = re.compile(r"Subject(\d+)/(High|Low)[ _][Gg]ain/Subject\d+_(?:High|Low) gain_ M1_(\w+)\.txt$")

entries = []
for z in sorted(RAW.glob("*.zip")):
    zf = zipfile.ZipFile(z)
    for i in zf.infolist():
        m = pat.search(i.filename)
        if m:
            entries.append((int(m.group(1)), m.group(2), m.group(3), zf, i))
entries.sort(key=lambda e: (e[0], e[1] != "High", SITES.index(e[2])))
subs = sorted({e[0] for e in entries})
code = {s: k + 1 for k, s in enumerate(subs)}

rows, extra = [], []
nrec, hdr_ok, nbad = {}, True, 0
for sn, gain, site, zf, info in entries:
    df_s = pd.read_csv(io.BytesIO(zf.read(info.filename)), sep="\t", dtype=str, keep_default_na=False)
    hdr_ok &= list(df_s.columns) == HDR
    num = df_s.apply(pd.to_numeric, errors="coerce")
    badmask = num.isna() & (df_s != "")
    nbad += int(badmask.values.sum())
    for cc in HDR:  # cells the authors' export corrupted (for example a time-like '12:00.0') are left empty
        df_s.loc[badmask[cc], cc] = ""
    arr = num.to_numpy(dtype=float)
    n = len(arr)
    c = code[sn]
    nrec[c] = nrec.get(c, 0) + 1
    gname, gval = GAIN[gain]
    fn = f"samples/{ID}_S{c:04d}_{gname}gain_{site}.csv"
    ev = []
    for col, nm in ((3, "test_stim_onset"), (4, "conditioning_stim_onset")):
        hi = np.nan_to_num(arr[:, col]) > 2.5
        on = np.flatnonzero(hi[1:] & ~hi[:-1]) + 1
        ev += [f"{nm}={(k / 1000):.3f}" for k in on]
    p = TMP / f"{c}_{gname}_{site}.csv"
    df_s.insert(0, "time_s", [f"{k / 1000:.3f}" for k in range(n)])
    df_s.to_csv(p, index=False, lineterminator="\r\n")
    extra.append((fn, p))
    rows.append(dict(
        dataset=ID, study=ID, experiment=ID, group="", subject=str(c), person=str(c), subject_excluded_by_authors=False,
        age="", sex="", handedness="", population="healthy", setting="lab",
        device="Isometric force transducer for right index finger abduction, surface EMG of both first dorsal interosseous muscles, paired-pulse TMS",
        session=1, block=f"{gname} gain, conditioning site {site}", trial=nrec[c], phase="test", time_s="", task="tracking",
        unit="recording", condition=f"{gname} visual gain; conditioning stimulation over {site}",
        pert_type="visual_gain", pert_value=gval, pert_unit="authors' VA setting (VA3.0 high, VA0.05 low)",
        feedback="altered" if gname == "high" else "normal", excluded_by_authors=False,
        recording=nrec[c], samples_file=fn, signal_kind="isometric index finger abduction force, bilateral FDI EMG, TMS trigger voltages",
        channels=";".join(HDR), channel_units="N;mV;mV;V;V", sample_rate_hz=1000, n_samples=n, duration_s=n / 1000.0,
        events=";".join(ev), processing="none stated (raw recordings; blocks of one gain and site concatenated into one file)",
        trial_info=f"visual_gain={gname};conditioning_site={site}"))
import pandas as pd  # noqa: E402
df = pd.DataFrame(rows)

checks.published("conditioning-stimulation sites", len(set(SITES) & set(df.condition.str.extract(r"over (\w+)")[0])), 5, 0, "Abstract (M1, PMd, PMv, SMA, DLPFC)")
checks.published("visual gain levels", df.pert_value.nunique(), 2, 0, "OSF README (VA0.05 and VA3.0)")
checks.published("columns per recording file", len(HDR), 5, 0, "OSF README (Force, EMG right, EMG left, Test, Conditioning)")
checks.internal("every file header matches the README column order", hdr_ok)
checks.internal("16 participants x 10 recordings (2 gains x 5 sites)", len(df) == 160 and set(nrec.values()) == {10} and len(nrec) == 16, str((len(df), len(nrec))))
checks.internal("test stimulation pulses found in every recording", all("test_stim_onset" in e for e in df.events))

meta = dict(
    paper=dict(authors="Watanabe T, Kuwabara T, Matsumoto T, Yunoki K, Horinouchi T, Kirimoto H", year=2025,
               title="Modulation of ventral premotor and primary motor cortex interactions for accurate visuomotor force control",
               venue="Cortex 186:51-60", doi="10.1016/j.cortex.2025.03.005"),
    lab="Kirimoto (Hiroshima University)", data_url="https://osf.io/eac53/",
    license="none stated", license_url="",
    changes=("Converted from tab-separated text to one comma-separated CSV per recording, with a time_s column added (sample number / 1000). Values are the authors' "
             "own text. The LabVIEW task program in the deposit is not used. Test and conditioning stimulus onsets are listed in events as the first sample at "
             "which the trigger channel rises above 2.5 V (the README says a pulse is about 5 V against a baseline near 0.15 V)."),
    how=["One row is one recording: all blocks of one participant for one visual gain (high or low) and one conditioning-stimulation site (M1, PMd, PMv, SMA or DLPFC, "
         "paired with a test pulse over the left M1) concatenated into a single 1 kHz file, as the authors shared them. During the task the participant abducted the right index "
         "finger to match a force target while the visual feedback gain was manipulated.",
         "Channels: Force (N, right index finger abduction), EMG of the right and left first dorsal interosseous (mV), and the Test and Conditioning trigger voltages (V).",
         "Which of the two gain settings is high is inferred: the OSF README names the settings VA0.05 and VA3.0 and the folders High gain and Low gain; VA3.0 is taken as high (pert_value 3.0), "
         "VA0.05 as low. The recordings carry no marker of the boundaries between concatenated blocks. Age, sex and handedness are not shared per participant."],
    differences=[f"{nbad} cells in the text files are not numbers (a time-like string such as 12:00.0, which looks like a spreadsheet export artefact); they are left empty in the sample files. Published results (MEP amplitudes, force error) were computed in LabChart and are not reproduced here: the paper full text was not accessible, so the checks are design counts from the abstract and README."],
    unit_note="recording (all blocks of one gain and conditioning site)",
)
stdlib.release(ID, "timeseries", df, meta, checks, extra_files=extra)
