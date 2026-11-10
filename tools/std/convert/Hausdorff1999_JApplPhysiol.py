"""Hausdorff et al. (1999) J Appl Physiol 86(3):1040-1047: stride-to-stride gait timing of 50 healthy children aged 3-14 years
(PhysioNet 'Gait Maturation Database' 1.0.0). Each child walked 8 min at a self-chosen pace around a 400 m track with
force-sensitive foot switches in the right shoe; the deposit's id-age.txt files hold two columns: time of the heel strike (s) and
the stride interval (s). One row per line of a .txt file.

Raw file (untrusted data, read with zipfile only): raw_extra/Hausdorff1999_JApplPhysiol/gaitmat.zip, the PhysioNet zip of version
1.0.0 (data/<id>-<age>.txt, data/table.csv; the .str/.hea files are the same series in WFDB binary and are not converted).
Run from the site folder: python3 -I tools/std/convert/Hausdorff1999_JApplPhysiol.py
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

ID = "Hausdorff1999_JApplPhysiol"
MODULE = "walking"
GROUPS = {"Young": "3 and 4 year olds", "Middle": "6 and 7 year olds", "Old": "11 to 14 year olds"}
P = "gait-maturation-database-1.0.0/data/"


def clean(t, s):
    """The paper's preprocessing (Methods, Stride Time Dynamics): drop first 60 s and last 5 s, pauses (>2 s) with 5 s either side,
    then values more than 4 SD from the mean of the central 90%."""
    keep = (t >= 60) & (t <= t.max() - 5)
    for p in t[s > 2]:
        keep &= ~((t >= p - 5) & (t <= p + 5))
    t, s = t[keep], s[keep]
    srt = np.sort(s)
    k = int(round(0.05 * len(srt)))
    core = srt[k:len(srt) - k]
    m, sd = core.mean(), core.std(ddof=1)
    return s[np.abs(s - m) <= 4 * sd]


def main():
    raw = stdlib.raw_dir(ID)
    z = zipfile.ZipFile(raw / "gaitmat.zip")
    ck = stdlib.Checks()
    tab = pd.read_csv(io.StringIO(z.read(P + "table.csv").decode("latin1")))
    tab.columns = ["sid", "age_months", "sex", "height_in", "weight_lb", "leg_in", "speed", "grp"]
    assert len(tab) == 50
    files = {}
    for n in z.namelist():
        m = re.fullmatch(re.escape(P) + r"(\d+)-(\d+)\.txt", n)
        if m:
            files[int(m.group(1))] = (int(m.group(2)), n)
    assert sorted(files) == list(range(1, 51))
    frames, people, cleaned = [], [], {}
    for r in tab.itertuples():
        age_f, name = files[r.sid]
        assert age_f == r.age_months, (r.sid, age_f, r.age_months)
        a = np.loadtxt(io.StringIO(z.read(name).decode()))
        assert a.ndim == 2 and a.shape[1] == 2
        n = len(a)
        g = GROUPS.get(r.grp, "other ages (5, 8 and 10 years)") if isinstance(r.grp, str) else "other ages (5, 8 and 10 years)"
        code = f"c{r.sid:02d}"
        df = pd.DataFrame(dict(dataset=ID, study="Hausdorff et al. 1999, J Appl Physiol", experiment="", group=g, subject=code,
                               subject_excluded_by_authors=False, age=r.age_months / 12.0, sex=r.sex, handedness="",
                               population="healthy children", setting="lab",
                               device="force-sensitive foot switches in the right shoe (ankle-worn recorder)", session=1, block="",
                               trial=np.arange(1, n + 1), phase="other", time_s=a[:, 0], task="walking", unit="stride",
                               condition="self-selected pace, 8 min, 400 m track", pert_type="none", pert_value=0, pert_unit="",
                               feedback="normal", excluded_by_authors=False, fast_leg="unknown", walking_context="overground",
                               surface_incline_deg=0.0, walking_trial=1, stride_start_s=a[:, 0], stride_time_s=a[:, 1]))
        frames.append(df)
        cleaned[r.sid] = clean(a[:, 0], a[:, 1])
        people.append(dict(person=code, age_months=r.age_months, height_m=r.height_in * 0.0254, weight_kg=r.weight_lb * 0.45359237,
                           leg_length_m=r.leg_in * 0.0254, walking_speed_mps=r.speed, age_group=g))
    df = pd.concat(frames, ignore_index=True)
    pp = pd.DataFrame(people)

    # published numbers
    ck.published("children", len(pp), 50, 0, "Abstract and Methods, Subjects")
    ck.published("girls", (tab.sex == "F").sum(), 25, 0, "Abstract (25 girls)")
    for k, nn, src in (("Young", 11, "3 and 4 year olds"), ("Middle", 20, "6 and 7 year olds"), ("Old", 12, "11 to 14 year olds")):
        ck.published(f"children aged {src}", (tab.grp == k).sum(), nn, 0, "Methods, Subjects")
    ck.published("children aged 5, 8 or 10 years", tab.grp.isna().sum(), 7, 0, "Methods, Subjects (3 + 1 + 3)")
    for k, v, src in (("Young", 1.00, "3 and 4 year olds"), ("Middle", 1.20, "6 and 7 year olds"), ("Old", 1.28, "11 to 14 year olds")):
        ck.published(f"mean walking speed, {src} (m/s)", tab.loc[tab.grp == k, "speed"].mean(), v, 0.01,
                     "Results, Relationship of Stride Dynamics to Height")
    for k, v, src in (("Young", 0.900, "3 and 4 year olds"), ("Middle", 0.955, "6 and 7 year olds"), ("Old", 1.072, "11 to 14 year olds")):
        ids = tab.loc[tab.grp == k, "sid"]
        m = np.mean([cleaned[i].mean() for i in ids])
        ck.published(f"mean stride time after the paper's preprocessing, {src} (s)", m, v, 0.015,
                     "Results, Relationship of Stride Dynamics to Height (900, 955, 1072 ms)")
    cv_old = np.mean([100 * cleaned[i].std(ddof=1) / cleaned[i].mean() for i in tab.loc[tab.grp == "Old", "sid"]])
    ck.published("mean CV of stride time, 11 to 14 year olds (%)", cv_old, 2.1, 0.4, "Results, Stride Time Variability Measures")
    ck.internal("every line of every file kept", len(df) == sum(len(f) for f in frames), f"{len(df)} rows")
    ck.internal("walks last about 8 min: last heel strike between 7 and 9 min for every child",
                bool(df.groupby("subject").time_s.max().between(420, 540).all()),
                f"{df.groupby('subject').time_s.max().min():.0f}-{df.groupby('subject').time_s.max().max():.0f} s")
    ck.internal("stride times positive", bool((df.stride_time_s > 0).all()))

    meta = dict(
        study=ID,
        paper=dict(authors="Hausdorff JM, Zemany L, Peng CK, Goldberger AL", year=1999,
                   title="Maturation of gait dynamics: stride-to-stride variability and its temporal organization in children",
                   venue="J Appl Physiol 86(3):1040-1047", doi="10.1152/jappl.1999.86.3.1040"),
        lab="Hausdorff / Goldberger (Beth Israel Deaconess Medical Center, Harvard)",
        data_url="https://physionet.org/content/gait-maturation-db/1.0.0/", license="ODC-By 1.0",
        license_url="https://opendatacommons.org/licenses/by/1-0/",
        changes=("Kept every line of every child's .txt file in order; added age (from the age in months), sex, and the age group "
                 "from table.csv; height and weight were converted from inches and pounds to metres and kilograms and the leg "
                 "length from inches to metres (participants.csv). No filtering of the stride series."),
        how=["One row is one stride of the right foot: stride_start_s (also time_s) is the time of the heel strike in seconds from the "
             "start of the recording and stride_time_s is the stride interval (heel strike to the next heel strike of the same foot) "
             "in seconds, exactly as in the deposit. The series are the raw ones; the paper removed the first 60 s, the last 5 s, "
             "pauses and outliers before analysing them, and this file keeps them in.",
             "Each child walked at a self-determined pace for 8 min around a 400 m running track wearing their own shoes, with "
             "two force-sensitive switches (heel and ball of the foot) in the right shoe sampled at 300 Hz. There is no belt or "
             "perturbation (the _fast/_slow columns are empty, fast_leg is unknown).",
             "group is the authors' age group (3 and 4 year olds, 6 and 7 year olds, 11 to 14 year olds); the seven children of 5, 8 and 10 "
             "years, whom the authors studied but left out of the groups, are labelled 'other ages'. participants.csv holds "
             "height, weight, leg length and mean walking speed (from lap time)."],
        unit_note="stride",
        differences=["The paper's group means of stride time are computed after its preprocessing (first 60 s and last 5 s removed, "
                     "pauses and 4 SD outliers removed); the check applies the same steps to the rows, and the file itself keeps every "
                     "stride the authors shared."])
    names_ = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df.assign(**{c: np.nan for c in names_ if c not in df.columns})[names_]
    pc = [("age_months", "number", "months", "Age in months, as in the file name."),
          ("height_m", "number", "m", "Height (converted from inches)."), ("weight_kg", "number", "kg", "Body mass (converted from pounds)."),
          ("leg_length_m", "number", "m", "Leg length (converted from inches)."),
          ("walking_speed_mps", "number", "m/s", "Average walking speed over the walk, from lap time."),
          ("age_group", "text", "", "The authors' age group.")]
    stdlib.release(ID, MODULE, df, meta, ck, participants=pp, participant_columns=pc)


if __name__ == "__main__":
    main()
