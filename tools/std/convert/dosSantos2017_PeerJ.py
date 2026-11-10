"""dos Santos, Fukuchi, Fukuchi & Duarte (2017) PeerJ 5:e3626: 49 people (27 young, 22 older) standing still for 60 s on two force
platforms (one under each foot) in four conditions (eyes open / closed x rigid surface / foam), three trials each, with whole-body
motion capture.

Converted: the ground reaction force files (PDS<ss><cond><rep>grf.txt, 100 Hz, 6000 samples, 20 channels) and PDSinfo.txt.
Not converted: the marker files (mkr, 130 columns) and joint angle files (ang, 74 columns, units not documented per column in the
paper) and the 3 s static calibration files.

Raw files: raw_extra/dosSantos2017_PeerJ/ (Figshare 10.6084/m9.figshare.4525082, CC0). Samples are copied unchanged (text of the numbers
kept, tabs turned into commas).
Run from the site folder: python3 -I tools/std/convert/dosSantos2017_PeerJ.py"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "dosSantos2017_PeerJ"
HEAD = ("Time\tRGRF_X\tRGRF_Y\tRGRF_Z\tLGRF_X\tLGRF_Y\tLGRF_Z\tGRFNET_X\tGRFNET_Y\tGRFNET_Z\tRCOP_X\tRCOP_Y\tRCOP_Z\tLCOP_X\tLCOP_Y\tLCOP_Z"
        "\tCOPNET_X\tCOPNET_Y\tCOPNET_Z\tRFREEMOMENT_Y\tLFREEMOMENT_Y").split("\t")
CH = HEAD[1:]
UN = ["N"] * 9 + ["m"] * 9 + ["N*m"] * 2
COND = {"O": "eyes open", "C": "eyes closed"}
SURF = {"R": "rigid surface", "F": "foam pad"}

PCOLS = [("age_group", "text", "Young, Old", "The authors' age group (deposit AgeGroup): Young under 60 years, Old 60 years or more."),
         ("height_m", "number", "m", "Body height."),
         ("mass_kg", "number", "kg", "Body mass."),
         ("bmi", "number", "kg/m^2", "Body mass index, as in the deposit."),
         ("foot_length_cm", "number", "cm", "Foot length."),
         ("dominant_leg", "text", "Left, Right", "Dominant leg."),
         ("years_of_study", "integer", "years", "Years of schooling."),
         ("falls_last_12_months", "integer", "", "Number of falls in the previous 12 months (self-report)."),
         ("disability", "text", "Yes, No", "Whether the participant reported a disability (deposit Disability); the free-text detail is not included."),
         ("condition_order", "text", "", "Order in which the four conditions were done (deposit Sequence; OR = eyes open rigid, OF = eyes open foam, CR = eyes closed rigid, CF = eyes closed foam); three repetitions of each in a row.")]
PMAP = {"age_group": "AgeGroup", "height_m": "Height", "mass_kg": "Mass", "bmi": "BMI", "foot_length_cm": "FootLen",
        "dominant_leg": "DominantLeg", "years_of_study": "Ystudy", "falls_last_12_months": "Falls12m", "disability": "Disability",
        "condition_order": "Sequence"}


def main():
    raw = stdlib.raw_dir(DS)
    info = pd.read_csv(raw / "PDSinfo.txt", sep="\t")
    assert len(info) == 588
    for c in ("Age", "Sequence", "AgeGroup", "Gender"):
        assert (info.groupby("Subject")[c].nunique() == 1).all(), c
    ck = stdlib.Checks()
    recs = info.to_dict("records")
    # authors' order of conditions per person (Sequence), three repetitions each
    for r in recs:
        seq = [s.strip() for s in r["Sequence"].split(",")]
        r["_ord"] = seq.index(r["Trial"][5:7]) * 3 + int(r["Rep"]) - 1
        assert r["Trial"][5:7] in seq and int(r["Trial"][7]) == int(r["Rep"])
    recs.sort(key=lambda r: (int(r["Subject"]), r["_ord"]))
    rows, extra, ns = [], [], set()
    for r in recs:
        sub = int(r["Subject"])
        fn = raw / f"{r['Trial']}grf.txt"
        lines = fn.read_text(encoding="ascii").splitlines()
        assert lines[0].split("\t") == HEAD, lines[0]
        body = [l.split("\t") for l in lines[1:] if l.strip()]
        assert all(len(b) == 21 for b in body)
        ns.add(len(body))
        csvtxt = "time_s," + ",".join(CH) + "\r\n" + "".join(",".join(b) + "\r\n" for b in body)
        sf = f"samples/{r['Trial']}.csv"
        extra.append((sf, csvtxt.encode("ascii")))
        v, s = r["Trial"][5], r["Trial"][6]
        rows.append(dict(
            dataset=DS, study=DS, experiment="", group=r["AgeGroup"], subject=f"S{sub:02d}", person=f"S{sub:02d}",
            subject_excluded_by_authors=False, handedness="", time_s="", age=r["Age"], sex=r["Gender"], population=f"{r['AgeGroup'].lower()} adults", setting="lab",
            device="Two force platforms (one under each foot), 100 Hz; 12-camera motion capture", session=1,
            block=f"{r['Vision']} eyes, {r['Surface']} surface", trial=r["_ord"] + 1, phase="test", task="balance and posture",
            unit="recording", condition=f"{COND[v]}, {SURF[s]}", pert_type="foam_surface" if s == "F" else "none", pert_value=0,
            pert_unit="", feedback="none" if v == "C" else "normal", excluded_by_authors=False,
            recording=r["_ord"] + 1, samples_file=sf, signal_kind="ground reaction forces, centre of pressure and free moments of each foot and net, standing still",
            channels=";".join(CH), channel_units=";".join(UN), sample_rate_hz=100, n_samples=len(body),
            duration_s=round(float(body[-1][0]) - float(body[0][0]) + 0.01, 3), events="",
            processing="Low-pass filtered by the authors (4th-order zero-lag Butterworth, 10 Hz)"))
    df = pd.DataFrame(rows)
    first = info.groupby("Subject").first().reset_index()
    prow = []
    for r in first.to_dict("records"):
        d = {"person": f"S{int(r['Subject']):02d}"}
        for c in PCOLS:
            v = r[PMAP[c[0]]]
            d[c[0]] = None if pd.isna(v) else v
        prow.append(d)
    per = df.drop_duplicates("person").merge(first[["Subject", "Height", "Mass"]].assign(person=lambda t: t.Subject.map(lambda x: f"S{int(x):02d}")),
                                              on="person")
    yo, ol = per[per.group == "Young"], per[per.group == "Old"]
    ck.published("participants", len(per), 49, 0, "Results / Table 1 (49 subjects)")
    ck.published("young participants", len(yo), 27, 0, "Table 1 (Young N = 27)")
    ck.published("older participants", len(ol), 22, 0, "Table 1 (Old N = 22)")
    ck.published("young males", (yo.sex == "M").sum(), 15, 0, "Participants (15 male, 12 female young)")
    ck.published("older males", (ol.sex == "M").sum(), 11, 0, "Participants (11 male, 11 female old)")
    ck.published("older mean age (y)", ol.age.mean(), 67.8, 0.05, "Table 1")
    ck.published("young mean age (y)", yo.age.mean(), 28.1, 0.05, "Table 1")
    ck.published("older age min (y)", ol.age.min(), 61.1, 0.05, "Table 1 (range 61.1-84.7)")
    ck.published("older age max (y)", ol.age.max(), 84.7, 0.05, "Table 1 (range 61.1-84.7)")
    ck.published("young age min (y)", yo.age.min(), 21.8, 0.05, "Table 1 (range 21.8-37.9)")
    ck.published("young age max (y)", yo.age.max(), 37.9, 0.05, "Table 1 (range 21.8-37.9)")
    ck.published("older mean height (m)", ol.Height.mean(), 1.61, 0.005, "Table 1")
    ck.published("young mean height (m)", yo.Height.mean(), 1.71, 0.005, "Table 1")
    ck.published("older mean mass (kg)", ol.Mass.mean(), 68.6, 0.1, "Table 1 (68.66 here; the paper gives 68.6)")
    ck.published("young mean mass (kg)", yo.Mass.mean(), 70.3, 0.05, "Table 1")
    ck.published("recordings (49 x 12)", len(df), 588, 0, "Data description (588 trials)")
    ck.published("samples per recording", min(ns), 6000, 0, "Data description (60 s at 100 Hz)")
    ck.internal("all recordings 6000 samples", ns == {6000}, str(ns))
    ck.internal("12 recordings per person", (df.groupby("person").size() == 12).all())
    ck.internal("trial 1..12 per person", all(sorted(g.trial) == list(range(1, 13)) for _, g in df.groupby("person")))
    meta = dict(
        paper=dict(authors="dos Santos DA, Fukuchi CA, Fukuchi RK, Duarte M", year=2017,
                   title="A data set with kinematic and ground reaction forces of human balance",
                   venue="PeerJ 5:e3626", doi="10.7717/peerj.3626"),
        lab="Duarte (Federal University of ABC, Brazil)", data_url="https://doi.org/10.6084/m9.figshare.4525082",
        license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Each ground reaction force file (PDS<subject><condition><rep>grf.txt) became samples/<trial>.csv with the numbers unchanged "
                 "(tab to comma, time column named time_s); PDSinfo.txt was split into the recording table and participants.csv. Rows are "
                 "ordered by participant and by the order the conditions were actually done (the deposit lists conditions in a fixed order). "
                 "The marker files (mkr), joint angle files (ang) and the 3 s static calibration files, and the nationality, skin colour, "
                 "footwear, illness, medication, prosthesis, physical activity code and date columns, are not included."),
        how=["One row is one 60 s recording of a person standing still on two force platforms, one under each foot (three trials of each of four "
             "conditions: eyes open or closed, rigid surface or a 6 cm foam pad on each platform). The samples file has time_s (s), then for the right "
             "foot (R), left foot (L) and both together (NET): ground reaction force X, Y, Z (N), centre of pressure X, Y, Z (m) and the free "
             "moment about the vertical axis of each foot (N*m), at 100 Hz, filtered by the authors (4th-order zero-lag 10 Hz Butterworth).",
             "Axes (paper): X anterior-posterior (forward positive), Y vertical (up positive), Z mediolateral (positive to the participant's right). "
             "The COP Y column is zero because the platform surface is the origin of height.",
             "trial and recording count the person's recordings in the order done, from the authors' Sequence column; the participant's whole "
             "condition order is in participants.csv. Participant ages are as in the deposit.",
             "Participants gave written consent (paper); the deposit records no use limits."],
        unit_note="60-second standing recording (samples in samples/*.csv)")
    stdlib.release(DS, "timeseries", df, meta, ck, participants=prow, participant_columns=PCOLS, extra_files=extra)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))


main()
