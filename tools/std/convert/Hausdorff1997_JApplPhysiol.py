"""PhysioNet 'Gait in Aging and Disease Database' 1.0.0 (Hausdorff and colleagues): stride-interval series of 5 healthy young adults,
5 healthy older adults and 5 older adults with Parkinson's disease, walking on level ground with force-sensitive foot switches in
the shoe. Each file <group><n>[-<age>]-si.txt has two columns: time of the heel strike (s) and stride interval (s). One row per line.

The site lists this deposit under Hausdorff et al. 1997 (J Appl Physiol 82:262-269, young vs. elderly vs. Huntington's disease);
the deposit's page cites that paper and Hausdorff et al. 1998 (Mov Disord 13:428-437) but holds PD, not HD, recordings.

Raw file (untrusted data, read with zipfile only): raw_extra/Hausdorff1997_JApplPhysiol/gaitdb.zip
Run from the site folder: python3 -I tools/std/convert/Hausdorff1997_JApplPhysiol.py
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

ID = "Hausdorff1997_JApplPhysiol"
MODULE = "walking"
GRP = {"y": ("young adults", "healthy young adults", "path around a roughly circular course, 15 min"),
       "o": ("older adults", "healthy older adults", "path around a roughly circular course, 15 min"),
       "pd": ("Parkinson's disease", "Parkinson's disease", "up and down a long hallway, 6 min")}


def dfa(x, nmin=8, nmax=None):
    y = np.cumsum(x - x.mean())
    nmax = nmax or len(y) // 4
    ns = np.unique(np.round(np.logspace(np.log10(nmin), np.log10(nmax), 14)).astype(int))
    F = []
    for n in ns:
        k = len(y) // n
        seg = y[:k * n].reshape(k, n)
        t = np.arange(n)
        res = [np.mean((s - np.polyval(np.polyfit(t, s, 1), t)) ** 2) for s in seg]
        F.append(np.sqrt(np.mean(res)))
    return np.polyfit(np.log(ns), np.log(F), 1)[0]


def main():
    raw = stdlib.raw_dir(ID)
    z = zipfile.ZipFile(raw / "gaitdb.zip")
    ck = stdlib.Checks()
    files = []
    for n in z.namelist():
        m = re.fullmatch(r".*/(y|o|pd)(\d)(?:-(\d+))?-si\.txt", n)
        if m:
            files.append((m.group(1), int(m.group(2)), int(m.group(3)) if m.group(3) else None, n))
    order = {"y": 0, "o": 1, "pd": 2}
    files.sort(key=lambda f: (order[f[0]], f[1]))
    frames, people, alpha = [], [], {"y": [], "o": [], "pd": []}
    for g, k, age, name in files:
        a = np.loadtxt(io.StringIO(z.read(name).decode()))
        assert a.ndim == 2 and a.shape[1] == 2
        n = len(a)
        grp, pop, cond = GRP[g]
        code = f"{g}{k}"
        frames.append(pd.DataFrame(dict(
            dataset=ID, study="Hausdorff et al., Gait in Aging and Disease Database (PhysioNet)", experiment="", group=grp,
            subject=code, subject_excluded_by_authors=False, age=float(age) if age else np.nan, sex="", handedness="",
            population=pop, setting="lab", device="ultra-thin force-sensitive resistors in the shoe, ankle-worn recorder (300 Hz)",
            session=1, block="", trial=np.arange(1, n + 1), phase="other", time_s=a[:, 0], task="walking", unit="stride",
            condition="self-selected pace, " + cond, pert_type="none", pert_value=0, pert_unit="", feedback="normal",
            excluded_by_authors=False, fast_leg="unknown", walking_context="overground", surface_incline_deg=0.0,
            walking_trial=1, stride_start_s=a[:, 0], stride_time_s=a[:, 1])))
        alpha[g].append(dfa(a[:, 1]))
        people.append(dict(person=code, group=grp, n_strides=n, duration_s=a[-1, 0]))
    df = pd.concat(frames, ignore_index=True)
    pp = pd.DataFrame(people)
    print("DFA alpha (informational, not a check):", {k: round(float(np.mean(v)), 2) for k, v in alpha.items()})

    ages_y = [f[2] for f in files if f[0] == "y"]
    ages_o = [f[2] for f in files if f[0] == "o"]
    src = "PhysioNet database page, Data Description"
    ck.published("subjects", len(pp), 15, 0, src)
    for g, nm in (("y", "young"), ("o", "older"), ("pd", "Parkinson's disease")):
        ck.published(f"{nm} subjects", sum(f[0] == g for f in files), 5, 0, src)
    ck.published("oldest older adult (years)", max(ages_o), 77, 0, src + " (older adults 71-77 years)")
    ck.published("youngest older adult (years)", min(ages_o), 71, 0, src + " (older adults 71-77 years)")
    ck.published("oldest young adult (years)", max(ages_y), 29, 0, src + " (young adults 23-29 years)")
    d = df.groupby("subject").time_s.max()
    ck.internal("healthy series end near 14.5 min", bool(d[[s for s in d.index if s[0] in "yo"]].between(865, 872).all()))
    ck.internal("PD series end near 5 min", bool(d[[s for s in d.index if s.startswith("pd")]].between(298, 300).all()))
    ck.internal("every line of every file kept", len(df) == sum(len(f) for f in frames), f"{len(df)} rows")
    ck.internal("stride times positive", bool((df.stride_time_s > 0).all()))
    ck.internal("heel-strike times increase within each subject", bool(df.groupby("subject").time_s.apply(lambda s: (s.diff().dropna() > 0).all()).all()))

    meta = dict(
        study=ID,
        paper=dict(authors="Hausdorff JM, Mitchell SL, Firtion R, Peng CK, Cudkowicz ME, Wei JY, Goldberger AL", year=1997,
                   title="Altered fractal dynamics of gait: reduced stride-interval correlations with aging and Huntington's disease",
                   venue="J Appl Physiol 82(1):262-269", doi="10.1152/jappl.1997.82.1.262"),
        lab="Hausdorff / Goldberger (Beth Israel Deaconess Medical Center, Harvard)",
        data_url="https://physionet.org/content/gaitdb/1.0.0/", license="ODC-By 1.0",
        license_url="https://opendatacommons.org/licenses/by/1-0/",
        changes="Kept every line of every .txt file in order; added the group (from the file name) and, for young and older adults, the age in the file name. Nothing filtered.",
        how=["One row is one stride of one foot: stride_start_s (also time_s) is the time of the heel strike in seconds from the start of the recording "
             "and stride_time_s the stride interval (heel strike to the next heel strike of the same foot), as in the deposit.",
             "Stride intervals were measured with force-sensitive resistors in the shoe sampled at 300 Hz. Healthy young and older adults walked 15 min "
             "around a roughly circular obstacle-free path; adults with Parkinson's disease walked 6 min up and down a long hallway (turns are in the series). "
             "The series start 22-31 s into the recording and end at about 14.5 min (healthy) or 5 min (PD). There is no perturbation; fast_leg is unknown.",
             "Sex and the age of the Parkinson's disease group are not in the deposit. participants.csv gives the group, number of strides and duration."],
        unit_note="stride",
        differences=["The deposit's page cites Hausdorff et al. 1997 (young, elderly and Huntington's disease subjects) and Hausdorff et al. 1998, "
                     "Mov Disord 13:428-437 (Parkinson's and Huntington's disease), but contains 5 young, 5 older and 5 Parkinson's disease recordings and no "
                     "Huntington's disease; the 1997 paper's DFA exponents (young 0.87, elderly 0.68) are therefore not checked here, only the dataset description.",
                     "The page gives the young adults as 23-29 years; one young file name carries age 21. The page says PD participants walked 6 min and healthy participants 15 min; the series end at about 5 min and 14.5 min (heel-strike times), and begin 22-31 s in, so the deposit holds a trimmed part of each walk."])
    names_ = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df.assign(**{c: np.nan for c in names_ if c not in df.columns})[names_]
    pc = [("group", "text", "", "Young adults, older adults or Parkinson's disease."), ("n_strides", "integer", "", "Number of strides in the file."),
          ("duration_s", "number", "seconds", "Time of the last heel strike.")]
    stdlib.release(ID, MODULE, df, meta, ck, participants=pp, participant_columns=pc)


if __name__ == "__main__":
    main()
