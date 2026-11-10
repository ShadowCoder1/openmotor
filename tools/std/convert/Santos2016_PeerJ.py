"""Santos & Duarte (2016) PeerJ 4:e2648: 163 people standing still for 60 s on a force platform in four conditions
(eyes open / closed x rigid surface / foam pad), three trials each, plus clinical balance and questionnaire scores.

Raw file: raw_extra/Santos2016_PeerJ/BDS.zip (Figshare 10.6084/m9.figshare.3394432 v2, CC BY 4.0): BDSinfo.txt (one row per
recording, 64 columns) and BDS<nnnnn>.txt (tab-separated, 100 Hz, 6000 samples: Time, Fx, Fy, Fz, Mx, My, Mz, COPx, COPy).
Samples are copied unchanged (text of the numbers kept, tabs turned into commas). The BDSinfo date column is not used.
Run from the site folder: python3 -I tools/std/convert/Santos2016_PeerJ.py"""
import io
import math
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Santos2016_PeerJ"
CH = ["Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm", "COPx_cm", "COPy_cm"]
UN = ["N", "N", "N", "N*m", "N*m", "N*m", "cm", "cm"]
HEAD = "Time[s]\tFx[N]\tFy[N]\tFz[N]\tMx[Nm]\tMy[Nm]\tMz[Nm]\tCOPx[cm]\tCOPy[cm]"

PCOLS = [("age_group", "text", "Young, Old", "The authors' age group: Young under 60 years, Old 60 years or more."),
         ("height_cm", "number", "cm", "Body height."),
         ("mass_kg", "number", "kg", "Body mass."),
         ("bmi", "number", "kg/m^2", "Body mass index, as in the deposit."),
         ("foot_length_cm", "number", "cm", "Foot length."),
         ("years_of_study", "integer", "years", "Years of schooling."),
         ("falls_last_12_months", "integer", "", "Number of falls in the previous 12 months (self-report)."),
         ("severe_disability", "text", "Yes, No", "Whether the participant reported a disability (deposit column Disability); the free-text detail is not included."),
         ("fes_i_total", "integer", "7-28", "Short Falls Efficacy Scale-International total (deposit FES_T): concern about falling."),
         ("fes_i_class", "text", "", "Concern category assigned by the authors from the FES-I total (deposit FES_S)."),
         ("ipaq_class", "text", "Low, Moderate, High", "Physical activity class from the short IPAQ (deposit IPAQ_S)."),
         ("tmt_a_time_s", "number", "seconds", "Trail Making Test part A time."),
         ("tmt_a_errors", "integer", "", "Trail Making Test part A errors."),
         ("tmt_b_time_s", "number", "seconds", "Trail Making Test part B time."),
         ("tmt_b_errors", "integer", "", "Trail Making Test part B errors."),
         ("minibestest_total", "integer", "0-28", "Mini-BESTest total score (deposit Best_T); higher is better balance.")]
PMAP = {"age_group": "AgeGroup", "height_cm": "Height", "mass_kg": "Weight", "bmi": "BMI", "foot_length_cm": "FootLen",
        "years_of_study": "Ystudy", "falls_last_12_months": "Falls12m", "severe_disability": "Disability", "fes_i_total": "FES_T",
        "fes_i_class": "FES_S", "ipaq_class": "IPAQ_S", "tmt_a_time_s": "TMT_timeA", "tmt_a_errors": "TMT_errorsA",
        "tmt_b_time_s": "TMT_timeB", "tmt_b_errors": "TMT_errorsB", "minibestest_total": "Best_T"}
for _k in ["1", "2", "3l", "3r", "4", "5", "6l", "6r", "7", "8", "9", "10", "11", "12", "13", "14"]:
    PCOLS.append((f"minibestest_item_{_k}", "integer", "0-2", f"Mini-BESTest item {_k}, scored 0 to 2 (deposit Best_{_k})."))
    PMAP[f"minibestest_item_{_k}"] = f"Best_{_k}"


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / "BDS.zip")
    info = pd.read_csv(io.BytesIO(z.read("BDSinfo.txt")), sep="\t")
    ck = stdlib.Checks()
    names = set(z.namelist())
    rows, extra = [], []
    n_samples_all = set()
    for k, r in enumerate(info.itertuples(index=False), 1):
        r = r._asdict() if hasattr(r, "_asdict") else r
    recs = info.to_dict("records")
    seq = {}
    for r in recs:
        sub = int(r["Subject"])
        seq[sub] = seq.get(sub, 0) + 1
        fn = f"{r['Trial']}.txt"
        assert fn in names, fn
        lines = z.read(fn).decode("ascii").splitlines()
        assert lines[0].strip() == HEAD, lines[0]
        body = [l.split("\t") for l in lines[1:] if l.strip()]
        assert all(len(b) == 9 for b in body)
        n = len(body)
        n_samples_all.add(n)
        t0, t1 = float(body[0][0]), float(body[-1][0])
        csvtxt = "time_s," + ",".join(CH) + "\r\n" + "".join(",".join(b) + "\r\n" for b in body)
        sf = f"samples/{r['Trial']}.csv"
        extra.append((sf, csvtxt.encode("ascii")))
        eyes = r["Vision"].lower()
        surf = {"Firm": "rigid surface", "Foam": "foam pad"}[r["Surface"]]
        rows.append(dict(
            dataset=DS, study=DS, experiment="", group=r["AgeGroup"], subject=f"S{sub:03d}", person=f"S{sub:03d}",
            subject_excluded_by_authors=False, handedness="", time_s="", age=r["Age"], sex={"F": "F", "M": "M"}[r["Gender"]], population=f"{r['AgeGroup'].lower()} adults",
            setting="lab", device="Force platform, 100 Hz", session=1, block=f"{r['Vision']} eyes, {r['Surface']} surface", trial=seq[sub],
            phase="test", task="balance and posture", unit="recording", condition=f"eyes {eyes}, {surf}",
            pert_type="foam_surface" if r["Surface"] == "Foam" else "none", pert_value=0, pert_unit="",
            feedback="none" if eyes == "closed" else "normal", excluded_by_authors=False,
            recording=seq[sub], samples_file=sf, signal_kind="force platform forces, moments and centre of pressure while standing still",
            channels=";".join(CH), channel_units=";".join(UN), sample_rate_hz=100, n_samples=n, duration_s=round(t1 - t0 + 0.01, 3),
            events="", processing="Smoothed by the authors with a 4th-order zero-lag 10 Hz low-pass Butterworth filter"))
    df = pd.DataFrame(rows)
    # participants
    first = info.groupby("Subject").first().reset_index()
    prow = []
    for r in first.to_dict("records"):
        d = {"person": f"S{int(r['Subject']):03d}"}
        for c in PCOLS:
            v = r[PMAP[c[0]]]
            d[c[0]] = None if pd.isna(v) else v
        prow.append(d)
    assert (info.groupby("Subject").Age.nunique() == 1).all()
    # checks from the paper
    per = df.drop_duplicates("person")
    ck.published("participants", per.person.nunique(), 163, 0, "Abstract / Data description (163 subjects)")
    ck.published("female participants", (per.sex == "F").sum(), 116, 0, "Methods, Participants (116 female)")
    ck.published("male participants", (per.sex == "M").sum(), 47, 0, "Methods, Participants (47 male)")
    ck.published("young participants (<60 y)", (per.group == "Young").sum(), 87, 0, "Methods, Participants (87 young)")
    ck.published("older participants (>=60 y)", (per.group == "Old").sum(), 76, 0, "Methods, Participants (76 older)")
    ck.published("recordings (1956 files minus 26 missing)", len(df), 1930, 0, "Data description (1956 - 26 missing trials)")
    ck.published("youngest age (years, floor)", math.floor(per.age.min()), 18, 0, "Methods, Participants (ages 18 to 85)")
    ck.published("oldest age (years, floor)", math.floor(per.age.max()), 85, 0, "Methods, Participants (ages 18 to 85)")
    ck.published("samples per recording", sorted(n_samples_all)[0], 6000, 0, "Data description (6000 rows, 60 s at 100 Hz)")
    ck.internal("all recordings have 6000 samples", n_samples_all == {6000}, str(n_samples_all))
    ck.internal("at most 12 recordings, 4 conditions and 3 repetitions per condition for every participant", (df.groupby("person").size() <= 12).all() and (df.groupby("person").condition.nunique() <= 4).all() and (df.groupby(["person", "condition"]).size() <= 3).all())
    ck.internal("participants with all 12 recordings (the rest are the authors' 26 missing trials)", (df.groupby("person").size() == 12).sum() >= 150, str((df.groupby("person").size() == 12).sum()))
    ck.internal("sample files in ZIP equal rows", len(extra) == len(df))
    meta = dict(
        paper=dict(authors="Santos DA, Duarte M", year=2016, title="A public data set of human balance evaluations",
                   venue="PeerJ 4:e2648", doi="10.7717/peerj.2648"),
        lab="Duarte (Federal University of ABC, Brazil)", data_url="https://doi.org/10.6084/m9.figshare.3394432",
        license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Each BDS trial file became samples/<trial>.csv with the numbers unchanged (tab to comma, time column named time_s, "
                 "channel names with units); BDSinfo.txt was split into the recording table and participants.csv (one row per person). "
                 "Nationality, skin colour, footwear, medication, illness and prosthesis free text, IPAQ item answers, FES-I item answers "
                 "and the date column were left out."),
        how=["One row is one 60 s recording of a person standing still on a force platform (three trials of each of four conditions: eyes open "
             "or closed, rigid surface or a 6 cm foam pad). The samples file has time_s (s), three forces (N), three moments (N*m) and the "
             "centre of pressure in the platform frame (cm), at 100 Hz. Signals were smoothed by the authors (4th-order zero-lag 10 Hz "
             "Butterworth) before sharing.",
             "trial counts a person's recordings in file order (the authors randomised the order of conditions; the deposit does not "
             "give it). 26 trials are missing in the original data for five people who could not do the hardest conditions. Mini-BESTest, FES-I, "
             "IPAQ and Trail Making scores are in participants.csv.",
             "The deposit gives no participant consent restrictions; ethics approval is mentioned in the paper only."],
        unit_note="60-second standing recording (samples in samples/*.csv)")
    stdlib.release(DS, "timeseries", df, meta, ck, participants=prow, participant_columns=PCOLS, extra_files=extra)
    for c in ck.items:
        print(("  ok " if c["ok"] else "  FAIL ") + c["what"], c.get("computed", ""), c.get("reported", ""))


main()
