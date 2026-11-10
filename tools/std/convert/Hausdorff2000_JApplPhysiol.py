"""Hausdorff et al. (2000) J Appl Physiol 88(6):2045-2053: stride-to-stride gait timing of people with ALS, Parkinson's
disease, Huntington's disease and healthy controls (PhysioNet 'Gait in Neurodegenerative Disease Database' 1.0.0).
Each person walked about 5 min at their usual pace wearing force-sensitive foot switches; the deposit's .ts file holds the
authors' derived per-stride times (13 columns, not filtered). One row per line of a .ts file (one stride of each foot).

Raw file (untrusted data, read with zipfile only): raw_extra/Hausdorff2000_JApplPhysiol/gaitndd.zip, the PhysioNet zip of
version 1.0.0 (records <group><n>.ts / .hea / .let / .rit, RECORDS, subject-description.txt). The raw foot-switch signals
(.let, .rit) are not converted; the .ts stride series derived from them are.
Run from the site folder: python3 -I tools/std/convert/Hausdorff2000_JApplPhysiol.py
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

ID = "Hausdorff2000_JApplPhysiol"
MODULE = "walking"
GROUPS = {"control": ("healthy controls", "healthy"), "hunt": ("Huntington's disease", "Huntington's disease"),
          "park": ("Parkinson's disease", "Parkinson's disease"), "als": ("amyotrophic lateral sclerosis", "amyotrophic lateral sclerosis")}
SEVERITY = {"control": "none (placeholder 0)", "hunt": "total functional capacity (lower = more advanced)",
            "park": "Hoehn and Yahr score (higher = more advanced)", "als": "months since diagnosis"}
COLS = ["time_s", "stride_time_left_s", "stride_time_right_s", "swing_time_left_s", "swing_time_right_s", "swing_pct_left",
        "swing_pct_right", "stance_time_left_s", "stance_time_right_s", "stance_pct_left", "stance_pct_right",
        "double_support_s", "double_support_pct"]


def main():
    raw = stdlib.raw_dir(ID)
    z = zipfile.ZipFile(raw / "gaitndd.zip")
    P = "gait-in-neurodegenerative-disease-database-1.0.0/"
    names = set(z.namelist())
    ck = stdlib.Checks()
    records = z.read(P + "RECORDS").decode().split()
    # subject description: tab separated; the file breaks one line (hunt20: speed MISSING, severity on the next line)
    txt = z.read(P + "subject-description.txt").decode("latin1").replace("MISSING\n\t", "MISSING\t")
    desc = {}
    for line in txt.splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 7 and f[0].strip():
            desc[f[0].strip()] = f[1:8]
    assert set(desc) == set(records), set(desc) ^ set(records)
    order = sorted(records, key=lambda r: (list(GROUPS).index(re.match(r"[a-z]+", r).group()), int(re.search(r"\d+", r).group())))
    frames, people, nstr, last_t = [], [], {}, 0
    for r in order:
        g = re.match(r"[a-z]+", r).group()
        a = np.loadtxt(io.StringIO(z.read(P + r + ".ts").decode()))
        assert a.ndim == 2 and a.shape[1] == 13, (r, a.shape)
        hea = z.read(P + r + ".hea").decode().split()
        assert hea[:4] == [r, "2", "300", "90000"], hea[:4]
        last_t = max(last_t, a[:, 0].max())
        d = desc[r]
        age = float(d[1]); sex = {"f": "F", "m": "M"}[d[4]]
        n = len(a)
        nstr.setdefault(g, []).append(n)
        f = dict(dataset=ID, study="Hausdorff et al. 2000, J Appl Physiol", experiment="", group=GROUPS[g][0], subject=r,
                 subject_excluded_by_authors=False, age=age, sex=sex, handedness="", population=GROUPS[g][1], setting="lab",
                 device="force-sensitive foot switches (ankle-worn recorder)", session=1, block="",
                 trial=np.arange(1, n + 1), phase="other", task="walking", unit="stride", condition="usual pace",
                 pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=False,
                 fast_leg="unknown", walking_context="overground", surface_incline_deg=0.0)
        df = pd.DataFrame(f)
        for i, c in enumerate(COLS):
            df[c] = a[:, i]
        frames.append(df)
        spd = d[5].strip()
        people.append(dict(person=r, height_m=float(d[2]), weight_kg=np.nan if d[3].strip() == "MISSING" else float(d[3]),
                           gait_speed_mps=np.nan if spd == "MISSING" else float(spd),
                           severity_or_duration=float(d[6]), severity_scale=SEVERITY[g]))
    df = pd.concat(frames, ignore_index=True)

    cnt = {g: len(v) for g, v in nstr.items()}
    ck.published("Parkinson's disease records", cnt["park"], 15, 0, "PhysioNet database page, Data Description")
    ck.published("Huntington's disease records", cnt["hunt"], 20, 0, "PhysioNet database page, Data Description")
    ck.published("amyotrophic lateral sclerosis records", cnt["als"], 13, 0, "PhysioNet database page, Data Description")
    ck.published("healthy control records", cnt["control"], 16, 0, "PhysioNet database page, Data Description")
    ck.published("records in the database", len(records), 64, 0, "PhysioNet database page, Data Description")
    ck.published("recording length (90000 samples at 300 Hz, s); the paper says subjects walked for 5 min", 90000 / 300, 300, 0,
                 "abstract of the paper (5 min); .hea headers")
    ck.internal("every stride line the deposit shares is kept, in order", len(df) == sum(sum(v) for v in nstr.values()), f"{len(df)} rows")
    ck.internal("last stride time stamp lies within the 300 s recording", last_t <= 300, f"max {last_t:.1f} s")
    ck.internal("stride + stance = swing + stance consistency: swing_time + stance_time within 0.01 s of stride time on 99% of rows",
                (((df.swing_time_left_s + df.stance_time_left_s - df.stride_time_left_s).abs() < 0.011).mean() > 0.99),
                f"{((df.swing_time_left_s + df.stance_time_left_s - df.stride_time_left_s).abs() < 0.011).mean():.4f}")
    ck.internal("every record has a .let, .rit, .ts and .hea file", all(P + r + e in names for r in records for e in (".let", ".rit", ".ts", ".hea")))
    ages = pd.DataFrame(people).merge(df[["subject", "age"]].drop_duplicates(), left_on="person", right_on="subject")
    ck.internal("64 participants, ages 20-80", len(ages) == 64 and ages.age.between(20, 80).all())

    meta = dict(
        study=ID,
        paper=dict(authors="Hausdorff JM, Lertratanakul A, Cudkowicz ME, Peterson AL, Kaliton D, Goldberger AL", year=2000,
                   title="Dynamic markers of altered gait rhythm in amyotrophic lateral sclerosis",
                   venue="J Appl Physiol 88(6):2045-2053", doi="10.1152/jappl.2000.88.6.2045"),
        lab="Hausdorff / Goldberger (Beth Israel Deaconess Medical Center, Harvard)",
        data_url="https://physionet.org/content/gaitndd/1.0.0/", license="ODC-By 1.0",
        license_url="https://physionet.org/content/gaitndd/1.0.0/",
        changes=("Kept every line of every record's .ts file (the authors' derived stride timing) in order; added the participant's "
                 "age and sex from subject-description.txt (height, weight, gait speed and severity are in participants.csv); "
                 "the raw left and right foot-switch signals (.let, .rit) are not included. No filtering."),
        how=["One row is one line of a record's .ts file: one stride of the left foot and one of the right foot at the same time stamp "
             "(time_s is the authors' elapsed time column). Times are in seconds and percentages of the stride, exactly as the "
             "deposit gives them; the authors say the series are not filtered, so artefacts (very long or short strides) are kept.",
             "Left and right are the body sides; walking_context is overground because the subjects walked at their usual pace (the deposit does not describe the walkway); there is no belt or perturbation (fast_leg is unknown, the _fast/_slow columns are empty). "
             "double_support_s and double_support_pct are one value per stride pair.",
             "Each person walked for 5 min at their usual pace wearing force-sensitive foot switches (the paper's "
             "abstract); recordings were 300 Hz, 90000 samples. The disease measure in participants.csv is the Hoehn and Yahr score "
             "for Parkinson's disease, total functional capacity for Huntington's disease and months since diagnosis for ALS.",
             "The paper text could not be read from this computer (publisher site blocked it), so the checks are against the PhysioNet "
             "database page and the paper's abstract."],
        unit_note="stride (one stride of each foot)",
        differences=["The full text of the paper was not available, so none of its group means or variability statistics were "
                     "checked; the checks are the record counts per group and the recording length, from the PhysioNet page."])
    names_ = [c[1] for c in stdlib.columns_for(MODULE)]
    df = df.assign(**{c: np.nan for c in names_ if c not in df.columns})[names_]
    pc = [("height_m", "number", "m", "Height."), ("weight_kg", "number", "kg", "Body mass; empty where the deposit says MISSING."),
          ("gait_speed_mps", "number", "m/s", "Usual walking speed from the deposit; empty where MISSING."),
          ("severity_or_duration", "number", "", "Disease measure in the deposit; the scale depends on the group (severity_scale)."),
          ("severity_scale", "text", "", "What severity_or_duration measures for this participant's group.")]
    stdlib.release(ID, MODULE, df, meta, ck, participants=pd.DataFrame(people), participant_columns=pc)


if __name__ == "__main__":
    main()
