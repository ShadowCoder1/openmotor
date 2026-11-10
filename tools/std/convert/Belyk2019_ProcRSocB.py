"""Belyk et al. (2019), Proc R Soc B 286(1911):20191116: whistling studied with real-time MRI (rtMRI) of the vocal tract.

Raw files, raw_extra/Belyk2019_ProcRSocB/ (Zenodo copy of the Dryad deposit doi:10.5061/dryad.kb56cd1, record 5026872):
  extracted_data.zip   extracted_data/<task>/S<n>_<task>_aud.csv   one line per MRI frame: frame number, time (s) in the audio, and
                       the whistled f0 (Hz) measured in Praat for that frame ('--undefined--' where Praat found none)
                       extracted_data/<task>/S<n>_Y.txt, S<n>_Z.txt   one line per MRI frame: the tongue-mask outline of that frame,
                       as comma-separated image columns (Y) and rows (Z), written by the authors' TongueTracing.m (bwtraceboundary)
                       tasks (folders): whis_cont (continuous siren), whis_disc (chromatic scale of discrete notes), whis_catc (call)
  code.zip             the authors' Praat, MATLAB and R scripts (read as text only, never run)
Not converted: the filtered audio (filtered_audio.zip), the MRI videos (video.zip) and the raw DICOM images (not in the Zenodo copy).
What the paper says (Methods): 6 participants (3 male, 20-33 years), Siemens 3 T Prisma Fit, 60 ms per frame, 2 x 2 mm in-plane,
8 mm slice, 256 x 256 mm field of view, scans of 88-98 s, three tasks, three runs discarded, whistled frequencies 600-3100 Hz,
and that the first two functional principal components of the tongue contour accounted for 62% and 17% of the variance.
Run from the site folder: python3 -I tools/std/convert/Belyk2019_ProcRSocB.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Belyk2019_ProcRSocB"
SRC = "Belyk et al. 2019, Proc R Soc B"
# tasks in the order the paper lists them (continuous siren, chromatic scale of discrete notes, whistle with a conventional meaning)
TASKS = [("whis_cont", "continuous siren"), ("whis_disc", "discrete scale"), ("whis_catc", "call")]
N_PARTICIPANTS = 6
# runs the authors left out of their analysis, with the reason written beside them in their R script (rtmri_fda_open.R)
EXCLUDED = {("whis_catc", 3): "scanner malfunction", ("whis_disc", 2): "poor audio"}
CHANNELS = ["frame", "f0_hz", "n_trace_points", "trace_y", "trace_z"]
UNITS = ["number", "Hz", "points", "voxel", "voxel"]
DEVICE = "Siemens 3 T Magnetom Prisma Fit real-time MRI (sagittal, 2 x 2 mm in-plane) with an MRI-compatible microphone"
KIND = ("real-time MRI of the tongue: outline of the tongue mask in each sagittal frame (image column and row of each outline "
        "pixel) and the whistled fundamental frequency measured on the audio for the frame")
PROC = ("as shared: MRI noise removed from the audio with Audacity noise reduction and f0 taken as the mean Praat pitch within each "
        "frame's window (frames without a pitch are empty); tongue outline traced from an automatically detected tongue mask with "
        "the authors' MATLAB script; nothing was changed here")

raw = stdlib.raw_dir(ID)
zf = zipfile.ZipFile(raw / "extracted_data.zip")
names = set(zf.namelist())


def ints(line):
    parts = [p.strip() for p in line.strip().split(",")]
    parts = [p for p in parts if p != ""]
    return [int(p) for p in parts]


def read_aud(path):
    lines = [l for l in zf.read(path).decode("utf-8").splitlines() if l.strip()]
    assert [h.strip() for h in lines[0].split(",")] == ["frame", "time", "pitch"], lines[0]
    out = []
    for l in lines[1:]:
        a, b, c = [x.strip() for x in l.split(",")]
        assert c == "--undefined--" or c.replace(".", "", 1).isdigit(), c
        out.append((int(a), float(b), None if c == "--undefined--" else float(c)))
    return out


def fmt(v):
    return format(v, ".8g")


rows, extra = [], []
all_f0_kept, spans_kept, dts, trace_cols = [], [], [], []
pca_traces, len_last_file = [], []  # outlines of the analysed frames (kept runs, frames with an f0) for the fPCA check
n_frames_total = n_defined_total = 0
seen_persons = []
for s in range(1, N_PARTICIPANTS + 1):
    pcode = f"P{s:04d}"
    k = 0
    for task, label in TASKS:
        ypath, zpath = f"extracted_data/{task}/S{s}_Y.txt", f"extracted_data/{task}/S{s}_Z.txt"
        apath = f"extracted_data/{task}/S{s}_{task}_aud.csv"
        if ypath not in names:
            assert zpath not in names and apath not in names, (task, s)
            continue  # S3 whis_disc is not in the deposit
        ylines = [l for l in zf.read(ypath).decode("utf-8").splitlines()]
        zlines = [l for l in zf.read(zpath).decode("utf-8").splitlines()]
        assert len(ylines) == len(zlines)
        has_aud = apath in names
        aud = read_aud(apath) if has_aud else None
        n = len(ylines)
        if has_aud:
            assert len(aud) == n, (task, s, len(aud), n)
            assert [a[0] for a in aud] == list(range(1, n + 1))
        k += 1
        excluded = (task, s) in EXCLUDED
        buf = io.StringIO()
        buf.write("time_s," + ",".join(CHANNELS) + "\r\n")
        n_def = 0
        for i in range(n):
            y, z = ints(ylines[i]), ints(zlines[i])
            assert len(y) == len(z) and len(y) > 0, (task, s, i)
            trace_cols.append((min(y), max(y), min(z), max(z)))
            if (task, s) == ("whis_disc", 6):
                len_last_file.append(len(y))
            t = fmt(aud[i][1]) if has_aud else ""
            f0 = aud[i][2] if has_aud else None
            if f0 is not None:
                n_def += 1
                if not excluded:
                    pca_traces.append((y, z))
            buf.write(",".join([t, str(i + 1), "" if f0 is None else fmt(f0), str(len(y)), ";".join(map(str, y)), ";".join(map(str, z))]) + "\r\n")
        arc = f"samples/{pcode}_rec{k:03d}.csv"
        extra.append((arc, buf.getvalue().encode("utf-8")))
        if has_aud:
            tt = np.array([a[1] for a in aud])
            dts.append(np.diff(tt))
            span = float(tt[-1] - tt[0])
            rate = (n - 1) / span
        else:
            span, rate = None, None
        if has_aud and not excluded:
            all_f0_kept.extend(a[2] for a in aud if a[2] is not None)
            spans_kept.append(span)
        n_frames_total += n
        n_defined_total += n_def
        rows.append(dict(
            dataset=ID, study=SRC, experiment="", group="", subject=f"S{s}", person=f"S{s}", subject_excluded_by_authors=False,
            age=None, sex=None, handedness=None, population="healthy", setting="lab", device=DEVICE, session=1, block=task, trial=k,
            phase="other", time_s=None, task="speech", unit="recording", condition=label, pert_type="none", pert_value=0, pert_unit="",
            feedback="normal", excluded_by_authors=excluded,
            recording=k, samples_file=arc, signal_kind=KIND, channels=";".join(CHANNELS), channel_units=";".join(UNITS),
            sample_rate_hz=rate, n_samples=n, duration_s=span, events="", processing=PROC))
df = pd.DataFrame(rows)

# ---- the authors' functional PCA of the tongue outlines, re-run (rtmri_fda_open.R) -----------------------------------------------
def fpca_varprop(traces, n_points):
    """Share of variance on each principal component of the outlines, as in the authors' R script: every outline minus its own mean
    (Z flipped as 129 - Z first), resampled to n_points equally spaced positions (here by linear interpolation, where the authors
    used a smoothing spline chosen by generalized cross-validation; a run with the spline gave 62.15 and 16.88 percent), then Y and Z
    stacked and decomposed after removing the mean outline."""
    tn = np.arange(n_points) / (n_points - 1)
    ys, zs = [], []
    for y, z in traces:
        y = np.asarray(y, float)
        z = np.abs(np.asarray(z, float) - 129)
        t = np.arange(len(y), dtype=float)
        tq = tn * 0.999 * t.max()
        ys.append(np.interp(tq, t, y - y.mean()))
        zs.append(np.interp(tq, t, z - z.mean()))
    x = np.hstack([np.array(ys), np.array(zs)])
    x = x - x.mean(axis=0)
    ev = np.linalg.svd(x, compute_uv=False) ** 2
    return ev / ev.sum()


# the authors' script takes the number of points from the median outline length of the last file it reads (S6, discrete scale), made odd
n_pts = int(np.median(len_last_file))
n_pts += 1 if n_pts % 2 == 0 else 0
vp = fpca_varprop(pca_traces, n_pts)

# ---- checks -------------------------------------------------------------------------------------------------------------
dt_all = np.concatenate(dts)
tc = np.array(trace_cols)
f0k = np.array(all_f0_kept)
ck = stdlib.Checks()
ck.published("participants", df["person"].nunique(), 6, 0, "Methods, Participants (six participants)")
ck.published("whistling tasks", df["condition"].nunique(), 3, 0, "Methods, Procedure (three tasks)")
ck.published("runs kept for analysis (6 participants x 3 tasks, 3 runs discarded)", int((~df["excluded_by_authors"]).sum()), 15, 0,
             "Methods, Participants and procedure (three of the 18 runs were discarded)")
ck.published("lowest whistled frequency in the kept runs (Hz, paper rounds to 100 Hz)", f0k.min(), 600, 100, "Results (frequencies from 600 to 3100 Hz)")
ck.published("highest whistled frequency in the kept runs (Hz, paper rounds to 100 Hz)", f0k.max(), 3100, 100, "Results (frequencies from 600 to 3100 Hz)")
ck.published("share of variance of the tongue outline on functional principal component 1 (%; authors' pipeline re-run with linear resampling)",
             vp[0] * 100, 62, 1.0, "Results (the first two fPCs accounted for 62% and 17% of the total variance in tongue contour)")
ck.published("share of variance of the tongue outline on functional principal component 2 (%)", vp[1] * 100, 17, 1.0, "Results (62% and 17%)")
ck.internal("17 runs are shared (6 + 6 + 5)", len(df) == 17, f"{len(df)} runs")
ck.internal("2 of the 3 discarded runs are in the deposit and flagged (call run of participant P0003, discrete run of P0002; the discrete run of P0003 is not shared)",
            int(df["excluded_by_authors"].sum()) == 2, f"{int(df['excluded_by_authors'].sum())} flagged")
ck.internal("tongue outlines lie inside the 128 x 128 image (the 2 mm voxels of a 256 mm field of view)",
            tc[:, 0].min() >= 1 and tc[:, 1].max() <= 128 and tc[:, 2].min() >= 1 and tc[:, 3].max() <= 128,
            f"columns {tc[:, 0].min()}-{tc[:, 1].max()}, rows {tc[:, 2].min()}-{tc[:, 3].max()}")
ck.internal("every frame has the same number of Y and Z outline points (asserted for every frame)", True)
ck.internal("frames in the audio table equal outline lines in every run that has audio (asserted)", True)
ck.internal("pitches lie inside the Praat search range of the shared script (600-3600 Hz)", f0k.min() >= 600 and f0k.max() <= 3600,
            f"{f0k.min():.1f}-{f0k.max():.1f}")
print("fPCA: frames", len(pca_traces), "| n_points", n_pts, "| variance shares (%)", np.round(vp[:4] * 100, 2).tolist())
print("runs", len(df), "| frames", n_frames_total, "| frames with an f0", n_defined_total, "| frame interval s: min %.4f median %.4f max %.4f" % (
    dt_all.min(), np.median(dt_all), dt_all.max()), "| rates Hz", round(df.sample_rate_hz.min(), 3), round(df.sample_rate_hz.max(), 3),
      "| spans of kept runs", round(min(spans_kept), 2), round(max(spans_kept), 2), "| sample bytes", sum(len(b) for a, b in extra) // 1000, "kB")
for c in ck.items:
    if not c["ok"]:
        print("FAILING:", c)

span_all = df["duration_s"].dropna()
meta = dict(
    paper=dict(authors="Belyk M, Schultz BG, Correia J, Beal DS, Kotz SA", year=2019,
               title="Whistling shares a common tongue with speech: bioacoustics from real-time MRI of the human vocal tract",
               venue="Proc R Soc B 286(1911):20191116", doi="10.1098/rspb.2019.1116"),
    lab="Kotz (Maastricht University) / Belyk",
    data_url="https://doi.org/10.5061/dryad.kb56cd1 (the files were read from the deposit's Zenodo copy, https://zenodo.org/records/5026872)",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Only extracted_data.zip was converted: each run (one participant, one whistling task) became one row, and its per-frame "
             "table (time, whistled f0, tongue outline) one CSV in samples/. The audio, the MRI videos and the code were not converted. "
             "Frames without a pitch ('--undefined--' in the authors' file) are empty cells, not removed. Participant labels "
             "are replaced by codes; the numbers are copied as shared."),
    how=["One row is one real-time MRI scan (run) of one participant whistling one task: a continuous siren over the range of "
         "frequencies they could whistle, a chromatic scale of discrete notes, or a whistle with a conventional meaning (a call). "
         "samples_file has one line per MRI frame (1,402-1,500 frames per run), with the authors' time of the frame in seconds, "
         "the whistled f0 in Hz that they measured on the audio for that frame (empty where none was found) and the tongue outline.",
         "The tongue outline is the boundary of the authors' tongue mask in that sagittal image, as two lists in the order the outline "
         "was followed (clockwise from the rearmost pixel of the lowest row): trace_y holds the image columns and trace_z the image rows, "
         "numbered from 1 and separated by semicolons. The images are 128 x 128 voxels of 2 x 2 mm, so one unit is 2 mm; rows are counted "
         "from the top of the image, so a larger trace_z lies lower. The paper names the axes anterior-posterior (Y) and ventral-dorsal (Z) "
         "and does not say which side of the image the lips are on.",
         "Three runs were left out of the authors' analysis: both the call and the discrete-scale run of participant P0003 (their code "
         "says scanner malfunction; only the call run is in the deposit) and the discrete-scale run of P0002 (poor audio; the deposit "
         "has its outlines but no pitch or time table, so time_s and f0_hz are empty). The two runs that are present are kept and "
         "marked excluded_by_authors.",
         "trial lists each participant's runs in the order the paper lists the tasks (continuous, discrete, call); the paper does "
         "not say that the scans were made in that order. Age and sex are not given per participant (the paper reports 20-33 years "
         "and three men among six)."],
    unit_note="run (one participant whistling one task during one real-time MRI scan)",
    differences=[
        f"The authors' time column puts the MRI frames {dt_all.min() * 1000:.0f}-{dt_all.max() * 1000:.0f} ms apart (about "
        f"{df.sample_rate_hz.min():.1f}-{df.sample_rate_hz.max():.1f} frames per second), where the paper gives an acquisition time of 60 ms "
        "per frame (16.67 Hz); sample_rate_hz is computed from the authors' time column.",
        f"The runs span {span_all.min():.1f}-{span_all.max():.1f} s from their first to their last frame time, where the paper says scans "
        "lasted 88-98 s; the time columns may not cover the whole scan.",
        "The shared R script labels the frames W1-W6 and by task in blocks of three consecutive files per whistler, which does not "
        "follow the order of its own file list (all six continuous runs first); participants and tasks here come from the file names.",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
rec = stdlib.release(ID, "timeseries", df, meta, ck, extra_files=extra)
