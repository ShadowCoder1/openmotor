"""Wang & Grigos (2024), Lang Speech 68(1):141-161: speech motor variability at habitual, fast and slow speaking rates in 40 adults
of three age groups (lower lip kinematics and acoustics of a connected speech task).

Raw files, raw_extra/Wang2024_LangSpeech/ (OSF 3jw7p):
  Acoustic_Kinematic_Durations_OSF.xlsx  one row per participant x condition x repetition: acoustic and kinematic duration, speaking rate
  Acoustic_Kinematic_STI_OSF.xlsx        one row per participant x condition: lip aperture STI, acoustic STI, mean durations
Run: python3 -I tools/std/convert/Wang2024_LangSpeech.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Wang2024_LangSpeech"
P = "Wang & Grigos 2024, "
AGE = {"young": "18-30 years", "middle": "31-40 years", "old": "41-50 years", "older": "41-50 years"}


def main():
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    d = pd.read_excel(raw / "Acoustic_Kinematic_Durations_OSF.xlsx")
    s = pd.read_excel(raw / "Acoustic_Kinematic_STI_OSF.xlsx")
    for c in ("kindur_sec", "kindur_msec"):
        d[c] = pd.to_numeric(d[c], errors="coerce")  # the authors wrote na where the lip record was missing
    for c in ("STI_LipAp", "aSTI", "avg_kindur_msec"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    print(d.agelevels.unique(), s.condition_c.unique(), d.groupby("participant").size().value_counts().to_dict())
    s = s.set_index(["subjcode", "condition_c"])
    rows = []
    for r in d.itertuples(index=False):
        st = s.loc[(r.participant, r.condition)]
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Connected speech at three speaking rates", "group": AGE[r.agelevels],
            "subject": str(r.participant), "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
            "population": "healthy", "setting": "lab", "device": "motion capture of lower lip (kinematics) with microphone",
            "session": 1, "block": r.condition, "trial": int(r.newtrial), "phase": "test", "time_s": "", "task": "speech",
            "unit": "utterance", "condition": r.condition, "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal",
            "excluded_by_authors": False, "duration_ms": float(r.acousticdur), "kinematic_duration_ms": ("" if pd.isna(r.kindur_msec) else float(r.kindur_msec)),
            "ms_per_syllable": float(r.speakingrate), "sti_lip_aperture": ("" if pd.isna(st.STI_LipAp) else float(st.STI_LipAp)), "sti_acoustic": ("" if pd.isna(st.aSTI) else float(st.aSTI)),
        })
    a = pd.DataFrame(rows)
    per = d.drop_duplicates("participant").agelevels.value_counts()
    checks.published("participants", d.participant.nunique(), 40, 0, P + "Abstract (n = 40)")
    checks.published("young adults (18-30 years)", per.get("young", 0), 15, 0, P + "Abstract")
    checks.published("adults 31-40 years", per.get("middle", 0), 13, 0, P + "Abstract")
    checks.published("adults 41-50 years", sum(v for k, v in per.items() if k not in ("young", "middle")), 12, 0, P + "Abstract")
    checks.internal("1200 rows = 40 participants x 3 rates x 10 repetitions", len(d) == 1200 and (d.groupby(["participant", "condition"]).size() == 10).all())
    checks.internal("speaking rate column = acoustic duration / 9", bool(((d.acousticdur / 9 - d.speakingrate).abs() < 1e-6).all()))
    checks.internal("kinematic durations in seconds and ms columns agree", bool(((d.kindur_sec * 1000 - d.kindur_msec).abs().dropna() < 0.2).all()))
    m = d.groupby("condition").acousticdur.mean()
    checks.internal("slow rate longer than habitual longer than fast", bool(m["slow"] > m["habitual"] > m["fast"]), str(m.round(0).to_dict()))
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    sx = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(sx["papers"][0])
    meta = dict(
        study=ID, paper={k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=sx["lab"], data_url=sx["data_url"],
        license="none stated", license_url="",
        changes="Joined the authors' per-repetition duration sheet with their per-condition STI sheet and replaced participant codes.",
        how=[
            "Each row is one repetition of the connected speech task by one participant at one self-selected rate (habitual, fast or slow), "
            "ten per rate, in the authors' repetition order (newtrial). duration_ms is the acoustic duration, kinematic_duration_ms the duration "
            "from the lower-lip movement record; ms_per_syllable is the authors' speaking rate column, which equals acoustic duration over 9 in every row.",
            "sti_lip_aperture and sti_acoustic are the authors' spatiotemporal indices for the participant at that rate; they summarize the ten "
            "repetitions, so the same value is repeated on all ten rows of a cell. The group column holds the age group.",
        ],
        unit_note="repetition of the sentence by one participant",
        differences=["Where the authors wrote na for a missing lip-movement record, kinematic_duration_ms and the lip STI are empty.", "The paper's duration and STI means were not recomputed (the full text was not available); checks are the participant counts."],
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
