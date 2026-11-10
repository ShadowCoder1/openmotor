"""Pogson et al. (2019) J Neurophysiol: vHIT in 80 normal subjects, VOR and saccades (Figshare 8247104).

Raw files: raw_extra/Pogson2019_JNeurophysiol/DataSheet.csv (tidy table, one row per impulse-saccade) and DatasheetLegend.txt.
The participant identifier column is a hash and is replaced by stdlib's codes. Nothing from the deposit is executed.

Run from the site folder: python3 -I tools/std/convert/Pogson2019_JNeurophysiol.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Pogson2019_JNeurophysiol"
MODULE = "head_impulse"


def main():
    d = pd.read_csv(stdlib.raw_dir(DS) / "DataSheet.csv")
    ck = stdlib.Checks()
    key = ["PID", "Canal", "ImpSeqCanal"]
    ck.internal("13860 rows, 80 participants, 6 canals, 12408 head impulses",
                len(d) == 13860 and d.PID.nunique() == 80 and d.Canal.nunique() == 6 and d.groupby(key).ngroups == 12408)
    ck.internal("head and VOR values are identical on all rows of one impulse",
                int(d.groupby(key)[["HeadPV", "HeadPA", "Gain_MacDougall2013", "Gain_Strupp2018"]].nunique().max().max()) == 1)
    ck.internal("saccade number 0 rows have no saccade values; rows with a saccade number >= 1 start 50 ms or later, number -1 earlier",
                bool(d[d.SacNum == 0].SacOnset.isna().all() and (d[d.SacNum >= 1].SacOnset >= 50).all()
                     and (d[d.SacNum == -1].SacOnset < 50).all()))
    ck.internal("participant sex, decade and age group are constant within a participant",
                int(d.groupby("PID")[["Sex", "Decade", "DecadeGroup"]].nunique().max().max()) == 1)
    ck.internal("velocity category agrees with the head peak velocity",
                bool((np.floor(d.HeadPV / 100).clip(upper=3) * 100 == d.HeadPVChunk).all()))

    imp = d.groupby(key).first().reset_index()
    ck.published("normal subjects in the study", d.PID.nunique(), 80, 0, "Abstract")
    ref = {"RL": 0.98, "RA": 0.91, "LP": 0.79, "LL": 0.95, "LA": 0.86, "RP": 0.76}
    g = imp.groupby("Canal").Gain_MacDougall2013.mean()
    for c, v in ref.items():
        ck.published(f"mean VOR gain, {c} canal impulses (MacDougall 2013 algorithm)", g[c], v, 0.025, "Abstract")
    old = imp[imp.DecadeGroup == "Elderly"].groupby("Canal").Gain_MacDougall2013.mean()
    yg = imp[imp.DecadeGroup != "Elderly"].groupby("Canal").Gain_MacDougall2013.mean()

    ck.published("elderly (>60 y) minus younger mean gain, left posterior canal", old["LP"] - yg["LP"], -0.09, 0.02, "Abstract")
    ck.published("elderly (>60 y) minus younger mean gain, left anterior canal", old["LA"] - yg["LA"], 0.09, 0.02, "Abstract")

    rows = pd.DataFrame({
        "dataset": DS, "study": DS, "experiment": "vHIT, normal subjects", "group": d.DecadeGroup.str.lower(),
        "subject": d.PID + "_" + d.Canal, "person": d.PID, "subject_excluded_by_authors": False,
        "age": "", "sex": d.Sex.map({"Female": "F", "Male": "M"}), "handedness": "", "population": "healthy",
        "setting": "clinic", "device": "monocular video head impulse test (right eye), manual head impulses",
        "session": "", "block": "", "trial": d.groupby(["PID", "Canal"]).cumcount() + 1, "phase": "test", "time_s": "",
        "task": "eye movements", "unit": "saccade", "condition": d.Canal,
        "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
        "canal": d.Canal, "impulse_number": d.ImpSeqCanal, "head_peak_acceleration_deg_s2": d.HeadPA,
        "head_peak_acceleration_time_ms": d.HeadPATime, "head_peak_velocity_deg_s": d.HeadPV,
        "head_peak_velocity_time_ms": d.HeadPVTime, "head_zero_crossing_ms": d.HeadZero, "head_bounce_percent": d.HeadBounce,
        "head_mean_frequency": d.HeadMeanFreq, "head_amplitude_deg": d.HeadMag,
        "vor_gain_macdougall2013": d.Gain_MacDougall2013, "vor_gain_weber2008": d.Gain_Weber2008,
        "vor_gain_mahfuz2018": d.Gain_Mahfuz2018, "vor_gain_blodow2013": d.Gain_Blodow2013,
        "vor_gain_strupp2018": d.Gain_Strupp2018, "head_velocity_category": d.HeadPVChunk,
        "saccade_number": d.SacNum, "saccade_onset_ms": d.SacOnset, "saccade_amplitude_deg": d.SacMag,
        "saccade_peak_velocity_deg_s": d.SacPV, "saccade_cv": d.SacCV, "age_decade": d.Decade,
    })
    ck.internal("every row the authors shared is kept, in order", len(rows) == len(d))

    s = d[d.SacNum >= 1]
    s_imp = imp.merge(s[key].drop_duplicates().assign(h=1), how="left", on=key).h.fillna(0)
    meta = dict(
        study=DS,
        paper=dict(authors="Pogson JM, Taylor RL, Bradshaw AP, McGarvie L, D'Souza M, Halmagyi GM, Welgampola MS", year=2019,
                   title="The human vestibulo-ocular reflex and saccades: normal subjects and the effect of age",
                   venue="J Neurophysiol 122(1):336-349", doi="10.1152/jn.00847.2018"),
        lab="Welgampola / Halmagyi (Royal Prince Alfred Hospital, Sydney)",
        data_url="https://doi.org/10.6084/m9.figshare.8247104",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Converted DataSheet.csv to the OpenMotor head-impulse format: hashed participant identifiers became codes, "
                 "sex became F/M, and each participant-by-canal sequence became one subject; all values are the authors' own."),
        how=["One row is one refixation saccade after one head impulse, or the impulse itself (saccade_number 0) when no saccade "
             "followed, exactly as the authors' tidy table lists them; the head-impulse and VOR-gain values repeat on every row "
             "of the same impulse. Counting impulses (canal and impulse_number per person) gives 12,408 impulses.",
             "Five VOR gain values are given per impulse, one per published algorithm; the paper's headline gains use the "
             "MacDougall et al. (2013) algorithm. The deposit gives age only as a decade (age_decade; group is the authors' "
             "young < 40, middle 40-59, elderly 60+ years).",
             "Head impulses are given in degrees; time values are ms from impulse onset. The deposit's legend does not state the "
             "unit of head mean frequency and saccade coefficient of variation."],
        unit_note="saccade after a head impulse (or head impulse without a saccade)",
        differences=["The deposit was re-edited in June 2026 (all five VOR gain values added), and its numbers do not reproduce "
                     "all of the paper: mean gains for the six canals agree within 0.025 of the abstract, but refixation "
                     f"saccades here average {s.SacMag.mean():.2f} deg amplitude, {s.SacPV.mean():.1f} deg/s peak velocity and "
                     f"{s.SacOnset.mean():.0f} ms latency against 1.11 deg, 63.9 deg/s and 262 ms in the abstract, and "
                     f"{100 * s_imp.mean():.0f}% of impulses have a saccade here against 40.2% (per-participant mean) in the "
                     "abstract. The paper's exact saccade selection is not stated in the deposit and was not tested further."]
    )
    stdlib.release(DS, MODULE, rows, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.4f} vs {c['reported']}  ok={c['ok']}")
    print("older-minus-younger LP, LA", old["LP"] - yg["LP"], old["LA"] - yg["LA"])


if __name__ == "__main__":
    main()
