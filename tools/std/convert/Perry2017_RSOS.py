"""Perry & Srinivasan (2017) R Soc open sci 4(9):160627: 8 adults walking on a treadmill at 1.25 m/s with self-selected steps or
prescribed step widths of 5-33 cm, torso and foot marker positions at 100 Hz (Vicon). Only position traces are shared, so the
format is the timeseries module, one row per walking trial.

Raw files: raw_extra/Perry2017_RSOS/DataAndCode_PerryAndSrinivasanAug2017.zip (Zenodo record 4960676, a copy of the Dryad deposit
10.5061/dryad.1fh71, CC0) and its README. TorsoFeet_Subject<1-8>.mat each hold a cell array TorsoFeet with one cell per
trial: a struct Torso/Lfoot/Rfoot, each with X, Y, Z arrays (samples x 4 markers); quiet-standing trials and trials that were
dropped are empty cells (the deposit does not contain the standing data). Trial types and prescribed widths come from the authors'
rootProcessingCode.m (read as text, never run): StandingIndices, SelfSelectedIndices, WidthIndices and ActualWidths (cm, indexed by
trial number); ages, sexes, heights and masses from rootSubjectHeightsMasses.m.
Samples are raw: the authors' code negates X and Y to get forward = +Y and right = +X, and picks the hindmost foot marker as the
heel; the converter keeps all 12 markers in the deposit's signs.
Run from the site folder: python3 -I tools/std/convert/Perry2017_RSOS.py"""
import re
import shutil
import sys
import zipfile
import io
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Perry2017_RSOS"
MODULE = "timeseries"
P = "DataAndCode_PerryAndSrinivasanAug2017/"
PARTS = ("Torso", "Lfoot", "Rfoot")
CH = [f"{p}{k}.{a}" for p in PARTS for k in range(1, 5) for a in "XYZ"]
HEIGHT_FT = [6 + 0 / 12, 5 + 6 / 12, 5 + 3 / 12, 6 + 2 / 12, 5 + 11 / 12, 5 + 8 / 12, 5 + 1 / 12, 5 + 8 / 12]
MASS_LB = [190, 120, 145, 170, 160, 155, 135, 165]
SEX = ["M", "F", "F", "M", "M", "M", "F", "M"]
AGE = [29, 23, 27, 28, 26, 22, 25, 32]


def parse_code(txt):
    """Trial lists per subject from rootProcessingCode.m (MATLAB index expressions of the simple forms used there)."""
    def lst(s):
        s = s.strip()
        if s == "":
            return []
        out = []
        for part in re.split(r"[;,]", s):
            part = part.strip()
            if not part:
                continue
            m = re.fullmatch(r"\((\d+):(\d+)\)'?", part)
            out += list(range(int(m.group(1)), int(m.group(2)) + 1)) if m else [int(part)]
        return out
    info = {}
    for m in re.finditer(r"subjectIndex = (\d+);(.*?)(?=subjectIndex = \d+;|%% Treadmill speed)", txt, re.S):
        k, body = int(m.group(1)), m.group(2)
        d = {}
        for name in ("StandingIndices", "SelfSelectedIndices", "WidthIndices"):
            mm = re.search(name + r"\s*=\s*\[(.*?)\];", body, re.S)
            d[name] = lst(mm.group(1))
        mm = re.search(r"ActualWidths\s*=\s*\[(.*?)\];", body, re.S)
        d["ActualWidths"] = [int(x) for x in mm.group(1).split()]
        info[k] = d
    return info


def main():
    ck = stdlib.Checks()
    raw = stdlib.raw_dir(DS)
    zf = zipfile.ZipFile(raw / "DataAndCode_PerryAndSrinivasanAug2017.zip")
    code = zf.read(P + "rootProcessingCode.m").decode("utf-8", "replace")
    info = parse_code(code)
    assert sorted(info) == list(range(1, 9))
    hm = zf.read(P + "rootSubjectHeightsMasses.m").decode("utf-8", "replace")
    assert "genderList = {'M','F','F','M','M','M','F','M'}" in hm and "[29; 23; 27; 28; 26; 22; 25; 32]" in hm
    assert "massList = [190; 120; 145; 170; 160; 155; 135; 165]" in hm
    tmp = raw / "_samples"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    rows, extra, steps, kinds = [], [], [], []
    for s in range(1, 9):
        m = sio.loadmat(io.BytesIO(zf.read(P + f"TorsoFeet_Subject{s}.mat")), struct_as_record=False, squeeze_me=True)
        cells = list(np.atleast_1d(m["TorsoFeet"]))
        d = info[s]
        walk = sorted(d["SelfSelectedIndices"] + d["WidthIndices"])
        data_idx = [i for i, c in enumerate(cells, 1) if hasattr(c, "Torso")]
        ck.internal(f"subject {s}: the cells that hold data are exactly the authors' walking trials", data_idx == walk, f"{data_idx} vs {walk}")
        ck.internal(f"subject {s}: standing trials are empty cells", all(not hasattr(cells[i - 1], "Torso") for i in d["StandingIndices"]))
        seq = 0
        for i in walk:
            c = cells[i - 1]
            blocks = []
            for p in PARTS:
                o = getattr(c, p)
                for a in "XYZ":
                    v = np.asarray(getattr(o, a), float)
                    assert v.ndim == 2 and v.shape[1] == 4, v.shape
                    blocks.append(v)
            n = blocks[0].shape[0]
            assert all(b.shape[0] == n for b in blocks)
            # blocks order: Torso X,Y,Z, Lfoot X,Y,Z, Rfoot X,Y,Z -> channels Part<k>.<axis>
            A = np.empty((n, 36))
            for pi in range(3):
                for k in range(4):
                    for ai in range(3):
                        A[:, pi * 12 + k * 3 + ai] = blocks[pi * 3 + ai][:, k]
            assert np.isfinite(A).all() and np.abs(A).max() < 5, np.abs(A).max()
            df = pd.DataFrame(A, columns=CH)
            df.insert(0, "time_s", np.round(np.arange(n) * 0.01, 2))
            arc = f"samples/Subject{s:02d}Trial{i:02d}.csv"
            path = tmp / f"S{s}T{i}.csv"
            df.to_csv(path, index=False, float_format="%.8g", lineterminator="\r\n")
            extra.append((arc, str(path)))
            # steps = forward-most points of the two heel markers (hindmost marker of each foot; forward = -Y as stored)
            cnt = 0
            for p in ("Lfoot", "Rfoot"):
                Y = np.asarray(getattr(getattr(c, p), "Y"), float)
                heel = np.argmin(Y.mean(axis=0)) if False else np.argmax(Y.mean(axis=0))
                # the authors take min(mean(Y)) of the raw Y as the heel after-sign flip? raw Y is negated later: hindmost
                # after the flip (smallest -Y) is the marker with the largest raw mean Y
                pk, _ = find_peaks(-Y[:, heel], distance=30)
                cnt += len(pk)
            steps.append((s, i, cnt))
            seq += 1
            w = d["ActualWidths"][i - 1]
            selfsel = i in d["SelfSelectedIndices"]
            kinds.append((s, selfsel, w))
            rows.append(dict(
                dataset=DS, study=DS, experiment="", group="", subject=f"S{s:02d}", person=f"S{s:02d}",
                subject_excluded_by_authors=False, age=float(AGE[s - 1]), sex=SEX[s - 1], handedness="",
                population="healthy adults", setting="lab",
                device="Motorized treadmill, 1.25 m/s; Vicon T20 motion capture, 100 Hz", session=1,
                block="self-selected step width" if selfsel else "prescribed step width", trial=seq, phase="test",
                time_s=np.nan, task="walking", unit="recording",
                condition="self-selected step width" if selfsel else f"prescribed step width {w} cm",
                pert_type="none", pert_value=0.0, pert_unit="", feedback="normal", excluded_by_authors=False,
                recording=seq, samples_file=arc, signal_kind="3D marker positions (4 torso, 4 left-foot, 4 right-foot markers)",
                channels=";".join(CH), channel_units=";".join(["m"] * 36), sample_rate_hz=100, n_samples=n,
                duration_s=round(n * 0.01, 2), events="",
                processing="none; raw marker positions as shared (the authors' code negates X and Y so that +Y is forward and +X is right)"))
    out = pd.DataFrame(rows)
    for c in [c[1] for c in stdlib.columns_for(MODULE)]:
        if c not in out.columns:
            out[c] = np.nan
    st = pd.DataFrame(steps, columns=["s", "i", "n"])
    kd = pd.DataFrame(kinds, columns=["s", "selfsel", "w"])

    pp = pd.DataFrame(dict(person=[f"S{s:02d}" for s in range(1, 9)], height_m=np.round(np.array(HEIGHT_FT) * 0.3048, 4),
                           mass_kg=np.round(np.array(MASS_LB) * 0.453592, 3)))
    ck.internal("8 participants; 88 walking recordings kept (every walking trial that has data)", out.subject.nunique() == 8 and len(out) == 88, f"{len(out)}")
    ck.internal("every recording has 36 finite channels of metres below 5 m in size and about 40-150 s of data",
                bool(((out.duration_s > 40) & (out.duration_s < 150)).all()), f"{out.duration_s.min()}-{out.duration_s.max()} s")
    ages = np.array(AGE, float)
    ck.published("participants", out.subject.nunique(), 8, 0, "Methods, Participants")
    ck.published("men (five male, three female)", sum(x == "M" for x in SEX), 5, 0, "Methods, Participants")
    ck.published("mean age (26.5 years)", ages.mean(), 26.5, 0.05, "Methods, Participants")
    ck.published("SD of age (3.25 years)", ages.std(ddof=1), 3.25, 0.005, "Methods, Participants")
    ck.published("mean height (1.72 m)", pp.height_m.mean(), 1.72, 0.005, "Methods, Participants")
    ck.published("SD of height (0.11 m)", pp.height_m.std(ddof=1), 0.11, 0.005, "Methods, Participants")
    ck.published("mean body mass (70.3 kg)", pp.mass_kg.mean(), 70.3, 0.05, "Methods, Participants")
    ck.published("SD of body mass (9.85 kg)", pp.mass_kg.std(ddof=1), 9.85, 0.05, "Methods, Participants")
    ck.published("prescribed-width trials used (75 of 80)", int((~kd.selfsel).sum()), 75, 0, "Methods, Experimental protocol")
    ck.published("self-selected trials used (13 of 16)", int(kd.selfsel.sum()), 13, 0, "Methods, Experimental protocol")
    steps_mean, steps_sd = float(st.n.mean()), float(st.n.std(ddof=1))
    widths = sorted(set(kd[~kd.selfsel].w))
    ck.internal("prescribed widths are 5, 12, 19, 26 and 33 cm", widths == [5, 12, 19, 26, 33], str(widths))

    pc = [("height_m", "number", "m", "Body height, converted from the feet and inches in the authors' rootSubjectHeightsMasses.m."),
          ("mass_kg", "number", "kg", "Body mass, converted from the pounds in rootSubjectHeightsMasses.m.")]
    meta = dict(
        study=DS,
        paper=dict(authors="Perry JA, Srinivasan M", year=2017,
                   title="Walking with wider steps changes foot placement control, increases kinematic variability and does not improve linear stability",
                   venue="R Soc open sci 4(9):160627", doi="10.1098/rsos.160627"),
        lab="Srinivasan (Ohio State University)",
        data_url="https://doi.org/10.5061/dryad.1fh71",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Read the eight MATLAB files from the Zenodo copy of the Dryad deposit, wrote each walking trial's 12 marker "
                 "trajectories as a CSV with a time_s column (sample number x 0.01 s) and the values unchanged, took trial types "
                 "and prescribed widths from the authors' code (read as text), and ages, sexes, heights and masses from their "
                 "subject-properties file; code, bootstrap helpers and the stored regression results are left out, and the "
                 "deposit holds no data for the quiet-standing trials."),
        how=["One row is one walking trial of one participant (about 2 minutes on a treadmill at 1.25 m/s); its samples file "
             "has time_s and 36 channels at 100 Hz: three X, Y, Z coordinates in metres for each of 4 torso markers, 4 "
             "left-foot markers and 4 right-foot markers (names Torso1.X ... Rfoot4.Z).",
             "Each participant walked trials with a self-selected step width and with prescribed step widths of 5, 12, 19, "
             "26 and 33 cm (condition gives the width in cm), in the order of trial (the authors' trial number, with the "
             "quiet-standing trials in between left out). The widths come from the authors' code, not from the data files.",
             "Positions are as stored: the authors' code negates X and Y to get +Y forward and +X to the right, and takes "
             "the hindmost marker on each foot as the heel and a weighted mean of the torso markers as the pelvis point."],
        unit_note="recording (one walking trial of about 2 minutes)",
        differences=["The paper says 12 trials per participant of 1-2 minutes and 75 + 13 trials used; the deposit holds 88 "
                     "walking trials, 11 per participant on average (8 to 12), matching those counts. The paper reports "
                     "210 +- 50 steps per trial; counting the forward-most points of the two heel markers in the 88 trials "
                     f"gives {steps_mean:.0f} +- {steps_sd:.0f}, so that number did not reproduce (the paper's step counts probably come from the "
                     "authors' own analysis of mid-stances, after removing spurious ones, which this deposit's code does but we did not rerun); "
                     "it is not used as a check."],
    )
    stdlib.release(DS, MODULE, out, meta, ck, participants=pp, participant_columns=pc, extra_files=extra)
    shutil.rmtree(tmp, ignore_errors=True)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  P {c['what'][:70]} {c['computed']:.4f} vs {c['reported']}")


if __name__ == "__main__":
    main()
