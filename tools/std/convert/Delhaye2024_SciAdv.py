"""Delhaye, Schiltz, Crevecoeur, Thonnard & Lefevre (2024) Sci Adv 10(3):eadh9344: 18 adults lifted an instrumented manipulandum in a
precision grip (index finger and thumb on glass plates) in 10 blocks of 5 trials, the plates being swapped between blocks for a high or a
low friction one without the participant noticing. One row per lift trial (about 2000 samples at 200 Hz) with the two six-axis force/torque
sensors, accelerometer and position; the authors' own derived grip force, load force and phase onsets are added.
Raw file: raw_extra/Delhaye2024_SciAdv/afd.zip (Zenodo 5213702 = Dryad 10.5061/dryad.q573n5tjh, CC0). Read as text/MAT only.
Not carried: the friction-measurement rubbing recordings (XX_CF_*.csv), the fingerpad strain arrays (strain_array_mvt1.mat), the file
header lines with the recording date.
Run from the site folder: python3 -I tools/std/convert/Delhaye2024_SciAdv.py"""
import io
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Delhaye2024_SciAdv"
MODULE = "timeseries"
R = "Active_friction_dataset/"
FS = 200.0
RAW = ["FTcam_Fx", "FTcam_Fy", "FTcam_Fz", "FTcam_Tx", "FTcam_Ty", "FTcam_Tz", "FTled_Fx", "FTled_Fy", "FTled_Fz", "FTled_Tx", "FTled_Ty",
       "FTled_Tz", "AccX", "AccY", "AccZ", "Position", "Moisture"]
UNITS = ["N"] * 3 + ["Nm"] * 3 + ["N"] * 3 + ["Nm"] * 3 + ["m/s^2"] * 3 + ["cm", "unspecified (not used by the authors)"]
DER = [("GF", "grip_force", "N"), ("LF", "load_force", "N"), ("LFv", "load_force_vertical", "N"), ("pos", "height_rel", "cm")]
EVENTS = [("PLPs", "pre_loading_start"), ("LPs", "loading_start"), ("TPs", "transport_start"), ("SPs", "static_start"),
          ("LP2s", "loading2_start"), ("SP2s", "static2_start"), ("LP3s", "loading3_start"), ("SP3s", "static3_start"),
          ("LP4s", "loading4_start"), ("UPs", "unloading_start"), ("RPs", "replacing_start")]
DEVICE = ("manipulandum (540 g, partly counterweighted, 220 g lifted mass) with two ATI Mini27 Ti six-axis force/torque sensors, a "
          "SICK optical distance sensor and an accelerometer, sampled at 200 Hz; high- or low-friction glass plates")


N_COMPLEX = []


def real(v):
    v = complex(v)
    if abs(v.imag) > 1e-12:
        N_COMPLEX.append(v)
        return float("nan")
    return v.real


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(ID) / "afd.zip")
    tmp = Path("tmp_" + ID)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    rows, extra, parts = [], [], []
    static_lf, started_high, n_trials = [], 0, 0
    corr_min = 1.0
    short = []
    for s in range(1, 19):
        sub = f"S{s:02d}"
        m = sio.loadmat(io.BytesIO(z.read(f"{R}{sub}/signals_not_sync.mat")), squeeze_me=True)
        cond = str(m["cond"])
        assert len(cond) == 50 and set(cond) <= {"A", "D"}, cond
        if cond[0] == "A":
            started_high += 1
        ic = sio.loadmat(io.BytesIO(z.read(f"{R}{sub}/index_CF.mat")), squeeze_me=True)
        tc = sio.loadmat(io.BytesIO(z.read(f"{R}{sub}/thumb_CF.mat")), squeeze_me=True)
        parts.append(dict(person=sub, start_friction="high" if cond[0] == "A" else "low",
                          **{f"{fg}_{k}_{h}": real(d[f"{k}_{h}"]) for fg, d in (("index", ic), ("thumb", tc)) for k in ("k", "n") for h in ("high", "low")}))
        lfs = []
        for t in range(1, 51):
            txt = z.read(f"{R}{sub}/{sub}_GL_{t}.csv").decode("ascii").splitlines()
            assert txt[3].startswith("FTcam_Fx"), txt[3][:30]
            hdr = [h.strip() for h in txt[3].split(",") if h.strip()]
            assert hdr == RAW, hdr
            arr = np.array([[float(v) for v in l.split(",") if v.strip()] for l in txt[4:] if l.strip()])
            assert arr.shape[1] == 17 and 1990 <= arr.shape[0] <= 2000, arr.shape
            n = arr.shape[0]
            short.append(n < 2000)
            i = t - 1
            der = np.column_stack([m[k][i][:n] for k, _, _ in DER])
            corr_min = min(corr_min, abs(np.corrcoef(arr[:, 2], m["NF_cam"][i][:n])[0, 1]))
            fname = f"{sub}_t{t:02d}.csv"
            names = RAW + [d[1] for d in DER]
            with open(tmp / fname, "w", newline="") as fh:
                fh.write("time_s," + ",".join(names) + "\r\n")
                for k in range(n):
                    fh.write(f"{k / FS:.3f}," + ",".join(f"{v:.8g}" for v in arr[k]) + "," + ",".join(f"{v:.8g}" for v in der[k]) + "\r\n")
            extra.append((f"samples/{fname}", str(tmp / fname)))
            ev = []
            for key, nm in EVENTS:
                idx = int(m[key][i])
                if idx > 0:
                    ev.append(f"{nm}={(idx - 1) / FS:g}")
            blk = (t - 1) // 5 + 1
            first = (t - 1) % 5 == 0
            fr = "high" if cond[i] == "A" else "low"
            change = first and t > 1
            prev = cond[i - 1] if i else ""
            lfs.append(float(np.mean(m["LF"][i][int(m["SPs"][i]):int(m["LP2s"][i])])) if t > 10 else None)
            rows.append(dict(
                dataset=ID, study="Delhaye et al. 2024 Sci Adv", experiment="", group=("high friction first" if cond[0] == "A" else "low friction first"),
                subject=sub, person=sub, subject_excluded_by_authors=False, population="healthy", setting="lab", device=DEVICE,
                session=1, block=f"block {blk}", trial=t, phase="practice" if blk <= 2 else "test", time_s=np.nan, task="grasping and lifting",
                unit="recording", condition=f"{fr} friction plates, " + ("post-change trial" if first else "normal trial"),
                pert_type="friction_change" if change else "none", pert_value=(1 if prev == "D" else -1) if change else 0,
                pert_unit="direction (+1 low to high friction, -1 high to low)" if change else "", feedback="normal",
                excluded_by_authors=blk <= 2, recording=t, samples_file=f"samples/{fname}",
                signal_kind="precision-grip lift: index and thumb force/torque, acceleration, manipulandum position",
                channels=";".join(names), channel_units=";".join(UNITS + [d[2] for d in DER]), sample_rate_hz=FS, n_samples=n,
                duration_s=n / FS, events=";".join(ev),
                processing=("raw channels as logged; grip_force, load_force, load_force_vertical and height_rel are the authors' derived signals "
                            "(4th-order low-pass Butterworth at 40 Hz, from signals_not_sync.mat)"),
                trial_info=f"block={blk};first_trial_of_block={first};friction_plate={fr}"))
            n_trials += 1
        static_lf.append(np.mean([x for x in lfs if x is not None]))
    df = pd.DataFrame(rows)
    for c in ("age", "sex", "handedness"):
        df[c] = ""
    df = df.reindex(columns=[c[1] for c in stdlib.columns_for(MODULE) if c[1] != "person"])
    pp = pd.DataFrame(parts)
    ck = stdlib.Checks()
    ck.published("participants", df.subject.nunique(), 18, 0, "Methods (Participants)")
    ck.published("participants starting with high friction (Fig. 1D, n = 9 and n = 9)", started_high, 9, 0, "Fig. 1D")
    ck.published("blocks of five trials", df.block.nunique(), 10, 0, "Fig. 1C")
    ck.published("trials analysed after dropping the two training blocks", (~df.excluded_by_authors).sum(), 720, 0, "Methods (Inspection and sorting of trials)")
    ck.published("weight of the object in the static phase (N), mean across participants", np.mean(static_lf), 2.22, 0.01, "Results (first paragraph)")
    ck.published("SD of that weight across participants (N)", np.std(static_lf, ddof=1), 0.07, 0.01, "Results (first paragraph)")
    print("complex friction parameters set empty:", len(N_COMPLEX))
    ck.internal("every trial has 17 raw plus 4 derived channels and 1990-2000 samples (6 files are shorter than 2000)", (df.n_samples >= 1990).all() and len(extra) == 900 and sum(short) == 6, f"{sum(short)} short")
    ck.internal("authors' derived signals line up with the raw files (index normal force vs FTcam_Fz, |r| > 0.9 in every trial)", corr_min > 0.9, f"min |r| {corr_min:.3f}")
    ck.internal("friction alternates every five trials", all(len(set(g.trial_info.str.extract(r"friction_plate=(\w+)")[0])) == 2 for _, g in df.groupby("subject")))
    meta = dict(
        paper=dict(authors="Delhaye BP, Schiltz F, Crevecoeur F, Thonnard JL, Lefevre P", year=2024,
                   title="Fast grip force adaptation to friction relies on localized fingerpad strains", venue="Sci Adv 10(3):eadh9344",
                   doi="10.1126/sciadv.adh9344"),
        lab="Lefevre (UCLouvain)", data_url="https://doi.org/10.5061/dryad.q573n5tjh (copy used: https://zenodo.org/records/5213702)",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Converted each XX_GL_Y.csv lift recording into one CSV of 2000 samples (values to 8 significant digits, time_s added as sample/200) "
                 "and appended the authors' derived grip force, load force, vertical load force and height from signals_not_sync.mat; phase onsets "
                 "from the same file are listed in events. Friction condition per trial comes from the file's cond string. Left out: the friction-"
                 "measurement rubbing recordings, the fingerpad strain arrays, the recording date in each file header and the empty-labelled extra "
                 "comma at the end of each row. Participant ages and sexes are not in the deposit."),
        how=["One row is one lift trial: the participant gripped and lifted the manipulandum (about 20 cm in 0.8 s), held it, then made three "
             "fast up-and-down movements, after an auditory cue; the 10 s recording holds both fingers' force (N) and torque (Nm) in the sensor "
             "frames, acceleration, the manipulandum position and a moisture channel the authors did not use. Plates were swapped between blocks of "
             "five trials; the first trial of a block is the authors' post-change trial. The first two blocks (trials 1-10) were training blocks "
             "that the authors left out of their analysis (excluded_by_authors).",
             "grip_force is the mean of the normal forces of the two fingers, load_force the norm of the summed tangential forces (paper Methods). "
             "Event times are the authors' phase onsets (sample index converted to seconds assuming MATLAB's 1-based indices).",
             "The paper additionally removes 99 of 720 analysed trials at least partly for unusable fingerpad images and three participants for the "
             "image and force analyses; the deposit does not say which, so those exclusions are not marked."],
        unit_note="recording (one grip-and-lift trial of 10 s)",
        differences=["Two friction-fit parameters in index_CF.mat/thumb_CF.mat are complex numbers in the deposit and are left empty in participants.csv.",
                     "Six recordings have 1998 or 1999 samples instead of 2000 (S06 trial 10, S07 trial 13, S08 trials 44 and 50, S11 trial 10, S15 trial 42; the derived signals are cut to the same length, assuming they start together). The deposit's Position column cannot be used in pandas-style readers because each row ends with an extra comma; the "
                     "converter reads it by hand. Strain-related results (most of the paper) could not be reproduced because the strain arrays "
                     "were not converted."])
    part_cols = [("start_friction", "text", "high, low", "Friction of the plates in block 1.")] + [
        (f"{fg}_{k}_{h}", "number", "", f"{k} of CF = k*GF^(n-1) (Barrea et al. 2016) for the {fg} finger on the {h}-friction plate, from the authors' friction measurement.")
        for fg in ("index", "thumb") for k in ("k", "n") for h in ("high", "low")]
    try:
        stdlib.release(ID, MODULE, df, meta, ck, participants=pp, participant_columns=part_cols, extra_files=extra)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))


main()
