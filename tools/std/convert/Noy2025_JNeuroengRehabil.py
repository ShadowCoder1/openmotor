"""Noy, Hassin-Baer, Fay-Karmon, Kattouf, Israeli-Korn, van der Wel & Friedman (2025), J NeuroEngineering Rehabil 22:51: manual
tracking of a sinusoidal target (one-person mirror game) with a stylus on a graphics tablet, in 31 people with Parkinson's disease,
29 age-matched older controls and 30 younger controls; 11 one-minute trials each.

Raw files, raw_extra/Noy2025_JNeuroengRehabil/ (figshare 22786244, CC BY 4.0):
  results.zip   results/<subject>/MirrorGame/tb_trial<N>.csv (raw tablet stream, 7 columns: 1 = x, 5 = device time in ms,
                6 = time in s per the readme; the authors' allsubjects.m converts x to screen pixels as x / xmax * 2560 with
                xmax = 48767 for subject numbers below 100 and 60073 for the rest) and trialinfo_trial<N>.mat (which stimulus
                file the trial showed: thistrial.thisstimulus.dotsFilename). Columns 2, 3, 4 and 7 are not described in the deposit
                and are not carried.
  dot1..dot11.csv   the stimuli, one row per 60 Hz frame, x and y as fractions of the screen (x * 2560 = pixels)
  combined.csv      one row per participant: the authors' measures per stimulus frequency, Age, Sex, LEDD
Run from the site folder: python3 -I tools/std/convert/Noy2025_JNeuroengRehabil.py
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Noy2025_JNeuroengRehabil"
SRC = "Noy et al. 2025, J NeuroEngineering Rehabil 22:51"
RAW = stdlib.raw_dir(ID)
GROUP = {1: "Parkinson's disease", 2: "older controls", 3: "younger controls"}
FREQ4 = {"0_2500": "0.25", "0_3215": "0.3215", "0_3750": "0.375", "0_4375": "0.4375", "0_5000": "0.5", "0_5625": "0.5625"}
FREQ = {"0_25": "0.25", "0_3215": "0.3215", "0_375": "0.375", "0_4375": "0.4375", "0_5": "0.5", "0_5625": "0.5625"}


def fmt(a):
    out = []
    for v in a:
        out.append("" if v != v else format(float(v), ".8g"))
    return out


def pcols():
    cols = []
    for pre, nm, unit, d in [
        ("dX", "relative_position_error", "ratio", "Relative position error dX (tracking error relative to the movement amplitude, after registration)"),
        ("dT", "mean_timing_error", "s", "Mean timing error dT (lag from the registration to the stimulus)"),
        ("submovementRateType2", "type2_submovement_rate", "per s", "Rate of type 2 submovements (acceleration zero crossings)"),
        ("submovementRateType3", "type3_submovement_rate", "per s", "Rate of type 3 submovements (jerk zero crossings)"),
        ("submovementRate", "total_submovement_rate", "per s", "Total submovement rate (type 2 plus type 3)"),
        ("submovementsPerStrokeType2", "type2_submovements_per_stroke", "per stroke", "Type 2 submovements per stroke"),
        ("submovementsPerStrokeType3", "type3_submovements_per_stroke", "per stroke", "Type 3 submovements per stroke"),
        ("submovementsPerStroke", "submovements_per_stroke", "per stroke", "Total submovements per stroke"),
    ]:
        for k, hz in (FREQ4 if pre in ("dX", "dT") else FREQ).items():
            cols.append((f"{pre}_{k}", f"{nm}_{hz}Hz", "number", unit, f"{d}, averaged over the participant's trials at stimulus frequency {hz} Hz (deposit column {pre}_{k})."))
    for c, nm, d in [("meandX", "mean_relative_position_error", "Mean of dX over the six stimulus frequencies"),
                     ("meandT", "mean_timing_error_all", "Mean of dT over the six stimulus frequencies"),
                     ("meanType2", "mean_type2_rate", "Mean type 2 submovement rate over the six frequencies"),
                     ("meanType3", "mean_type3_rate", "Mean type 3 submovement rate over the six frequencies"),
                     ("meanCombined", "mean_total_rate", "Mean total submovement rate over the six frequencies")]:
        cols.append((c, nm, "number", "", d + f" (deposit column {c}; name read from the column, the paper's group means of dX and dT are over frequencies)."))
    cols.append(("LEDD", "levodopa_equivalent_daily_dose_mg", "number", "mg", "Levodopa equivalent daily dose, people with Parkinson's disease (column LEDD); empty where not shared."))
    return cols


def main():
    checks = stdlib.Checks()
    comb = pd.read_csv(RAW / "combined.csv")
    gmap = comb.set_index("Subject")
    z = zipfile.ZipFile(RAW / "results.zip")
    names = z.namelist()
    subs = sorted({int(m.group(1)) for n in names for m in [re.match(r"results/(\d+)/MirrorGame/tb_trial\d+\.csv$", n)] if m})
    checks.internal("subjects in the raw folder equal those of combined.csv", subs == sorted(comb.Subject.tolist()))
    stim = {k: (RAW / f"dot{k}.csv").read_bytes() for k in range(1, 12)}
    rows, files = [], []
    for s in subs:
        r0 = gmap.loc[s]
        trials = sorted(int(re.match(rf"results/{s}/MirrorGame/tb_trial(\d+)\.csv$", n).group(1))
                        for n in names if re.match(rf"results/{s}/MirrorGame/tb_trial\d+\.csv$", n))
        xmax = 48767 if s < 100 else 60073
        code = f"S{subs.index(s) + 1:02d}"
        for tr in trials:
            a = pd.read_csv(io.BytesIO(z.read(f"results/{s}/MirrorGame/tb_trial{tr}.csv")), header=None)
            ti = sio.loadmat(io.BytesIO(z.read(f"results/{s}/MirrorGame/trialinfo_trial{tr}.mat")))["thistrial"]
            dots = str(ti["thisstimulus"][0, 0]["dotsFilename"][0, 0][0])
            m = re.fullmatch(r"dot(\d+)\.csv", dots)
            assert m, dots
            t = a[5].to_numpy(float)
            t0 = t[0]
            x = a[0].to_numpy(float)
            fn = f"samples/{code}_trial{tr:02d}.csv"
            body = "time_s,x_tablet_units,x_screen_px\r\n" + "".join(
                f"{ts},{xv},{xp}\r\n" for ts, xv, xp in zip(fmt(t - t0), fmt(x), fmt(x / xmax * 2560)))
            files.append((fn, body.encode()))
            rows.append(dict(
                dataset=ID, study=ID, experiment="", group=GROUP[int(r0.Group)], subject=code, subject_excluded_by_authors=False,
                age=float(r0.Age), sex="F" if int(r0.Sex) == 2 else "M", handedness="", population="healthy" if int(r0.Group) > 1 else "Parkinson's disease",
                setting="lab", device="stylus on a Wacom Cintiq 27QHD graphics tablet and display",
                session=1, block="", trial=tr, phase="test", time_s="", task="tracking", unit="recording", condition=f"stimulus {dots[:-4]}",
                pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=tr, samples_file=fn, signal_kind="pen x position (left-right) on a graphics tablet while tracking a moving ellipse",
                channels="x_tablet_units;x_screen_px", channel_units="tablet units;screen pixels (2560 px = screen width)", sample_rate_hz="",
                n_samples=len(a), duration_s=float(t[-1] - t0), events="", processing="none (raw tablet stream; empty x where the pen was not tracked)",
                trial_info=f"stimulus={dots[:-4]};stimulus_file=stimuli/{dots};tablet_xmax={xmax}"))
    # participants
    pc = pcols()
    parts = []
    for s in subs:
        r0 = gmap.loc[s]
        d = {"person": f"S{subs.index(s) + 1:02d}"}
        for c, nm, *_ in pc:
            v = r0[c]
            d[nm] = None if pd.isna(v) else float(v)
        parts.append(d)
    pcol_spec = [(nm, ty, un, de) for _, nm, ty, un, de in pc]
    n_tr = len(rows)
    # published checks
    C = comb
    P = f"{SRC}, Methods (Participants)"
    for g, n in [(1, 31), (2, 29), (3, 30)]:
        checks.published(f"participants, {GROUP[g]}", (C.Group == g).sum(), n, 0, P)
    checks.published("participants", len(C), 90, 0, P)
    for g, n in [(1, 14), (2, 16), (3, 13)]:
        checks.published(f"females, {GROUP[g]}", ((C.Group == g) & (C.Sex == 2)).sum(), n, 0, P)
    for g, mn, sd in [(1, 67, 11), (2, 62, 11), (3, 25, 6)]:
        checks.published(f"mean age, {GROUP[g]}", C[C.Group == g].Age.mean(), mn, 0.5, P)
        checks.published(f"SD of age, {GROUP[g]}", C[C.Group == g].Age.std(ddof=1), sd, 0.5, P)
    # model-free group means of dT over frequencies, as the paper prints them (marginal means may differ slightly from raw means)
    dT = [c for c in C.columns if c.startswith("dT_")]
    dX = [c for c in C.columns if c.startswith("dX_")]
    R = f"{SRC}, Results (error measures)"
    checks.published("PwP mean timing error dT", C[C.Group == 1][dT].mean(axis=1).mean(), 0.183, 0.0005, R)
    diffs = {}
    for g, v in [(2, 0.147), (3, 0.102)]:
        diffs[f"dT {GROUP[g]}"] = (C[C.Group == g][dT].mean(axis=1).mean(), v)
    for g, v in [(1, 0.389), (2, 0.265), (3, 0.181)]:
        diffs[f"dX {GROUP[g]}"] = (C[C.Group == g][dX].mean(axis=1).mean(), v)
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    print(diffs)
    freqs = np.array([0.25, 0.3215, 0.375, 0.4375, 0.5, 0.5625])
    for g, mn, sd in [(1, 1.16, 0.73), (2, 0.75, 0.49), (3, 0.53, 0.43)]:
        sl = [np.polyfit(freqs[np.isfinite(r)], r[np.isfinite(r)], 1)[0] for r in C[C.Group == g][dX].to_numpy(float)]
        print("slope", g, np.mean(sl), np.std(sl, ddof=1) / np.sqrt(len(sl)), np.std(sl, ddof=1), mn, sd)
    checks.internal("recordings: 989 raw trial files (90 participants x 11 trials minus one missing trial)", n_tr == 989, str(n_tr))
    short = sum(r["n_samples"] < 5000 for r in rows)
    print("short", short, "n_samples min/max", min(r["n_samples"] for r in rows), max(r["n_samples"] for r in rows), [r["samples_file"] for r in rows if r["n_samples"] < 5000][:5])
    checks.internal("every sample file has samples and stimuli are dot1-dot11",
                    all(r["n_samples"] >= 2 for r in rows) and all(re.fullmatch(r"stimulus dot(\d|1[01])", r["condition"]) for r in rows))
    for k in range(1, 12):
        files.append((f"stimuli/dot{k}.csv", stim[k]))
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = s["papers"][0]
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("One row per raw trial file; each file's x position and time were copied to samples/<subject>_trial<N>.csv with time set to "
                 "zero at the first sample, x kept in tablet units and also given in screen pixels with the authors' own formula "
                 "(x / xmax * 2560), and the stimulus of the trial read from the trial's .mat file. The authors' per-frequency measures, age, sex "
                 "and levodopa dose went into participants.csv. Columns 2, 3, 4 and 7 of the raw files (not described in the deposit) and "
                 "the dV measures (not defined in the paper) are not carried."),
        how=[
            "One row is one one-minute trial of the one-person mirror game: the participant moved a stylus left and right on a graphics tablet "
            "to keep a blue ellipse level with a red ellipse that moved sinusoidally (the stimulus changed frequency or peak velocity once or twice "
            "within the trial; the red ellipse stayed still for the first 5 s). Trial is the order in which the participant did the 11 trials, "
            "and the stimulus (dot1 to dot11, shown at 60 Hz, files in stimuli/) varied across trials and participants. The condition column "
            "names the stimulus.",
            "The sample files hold the raw tablet stream, not a resampled signal: time_s counts seconds from the first sample, x_tablet_units is "
            "column 1 of the raw file, and x_screen_px is x / xmax * 2560 with xmax = 48767 (participant numbers below 100) or 60073 (others), "
            "as in the authors' loading script. The stream is irregular (more than 100 samples a second, with empty x values while the pen was not "
            "tracked), so sample_rate_hz is left empty; the authors resampled to 100 Hz for analysis. The stimulus x position in pixels is x_fraction * 2560.",
            "A few raw trial files are very short (fewer than 5000 samples, down to 2: trials that were stopped early); they are kept as shared, "
            "and n_samples says how long each is. "
            "Group is people with Parkinson's disease, older controls or younger controls; age and sex are from combined.csv (sex coded 2 = female, "
            "inferred from the paper's counts of 14, 16 and 13 women per group).",
        ],
        unit_note="one-minute tracking trial (samples in samples/)",
        differences=[
            "The group means of dX and dT in the paper (for example 0.389 for dX in people with Parkinson's disease) are marginal means of the authors' "
            "mixed ANOVA and differ slightly from the means of the six frequency columns of combined.csv, so only the dT mean that matches is a check.",
        ],
    )
    return stdlib.release(ID, "timeseries", rows, meta, checks, participants=parts, participant_columns=pcol_spec, extra_files=files)


if __name__ == "__main__":
    main()
