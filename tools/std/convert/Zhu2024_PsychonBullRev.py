"""Zhu, Chen, Shi & Zhang (2024), Psychon Bull Rev 31(4):1661-1669: speech-to-speech synchronization (SSS) test in 60 Mandarin-speaking adults,
phase locking value (PLV) between the heard syllable stream and whispered syllables for three syllable-onset conditions.

Raw files, raw_extra/Zhu2024_PsychonBullRev/ (OSF z43ny):
  speechplv60.csv        one PLV per participant and onset condition (180 rows) with musical experience
  kcluster60new1.csv     the same 180 PLVs with the authors' high/low synchronizer cluster
  plvnew60sum.csv, correlation_musical.csv   per-participant mean PLV (used for internal checks)
  PLVscript.R            the authors' analysis (read as text; its printed results in comments are the documented values used for checks)
Paper abstract read in Europe PMC (full text is paywalled).
Run from the site folder: python3 -I tools/std/convert/Zhu2024_PsychonBullRev.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zhu2024_PsychonBullRev"
P = "Zhu et al. 2024, "
ONSET = {"as": "aspirated", "un": "unaspirated", "mix": "mixed"}
CL = {"high": "high synchronizers", "low": "low synchronizers"}


def main():
    raw = stdlib.raw_dir(ID)
    a = pd.read_csv(raw / "speechplv60.csv")
    k = pd.read_csv(raw / "kcluster60new1.csv")
    k["sid"] = k.subject.str.lstrip("s").astype(int)
    # kcluster has one row per participant and onset, in the same order as speechplv60 (subject 1..60 for each onset)
    k = k.set_index(["sid", "onset"])
    rows = []
    for r in a.itertuples(index=False):
        kk = k.loc[(r.subject, r.onset)]
        assert abs(kk.plv - r.value) < 1e-9
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Speech-to-speech synchronization test", "group": CL[kk.cluster],
            "subject": f"p{int(r.subject)}", "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "",
            "population": "healthy Mandarin-speaking adults", "setting": "lab", "device": "headphones and microphone, whispered syllables",
            "session": 1, "block": "", "trial": 0, "phase": "test", "time_s": "", "task": "speech", "unit": "participant summary",
            "condition": ONSET[r.onset] + " syllable onsets", "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal",
            "excluded_by_authors": False, "measure": "plv",
            "measure_description": "Phase locking value between the amplitude envelope of the heard syllable stream and the envelope of the whispered output in the SSS test (0 to 1; higher is stronger auditory-motor synchronization)",
            "statistic": "value", "value": r.value, "value_unit": "PLV (0 to 1)", "n_trials": "", "session_context": ""})
    df = pd.DataFrame(rows)
    df["trial"] = df.groupby("subject").cumcount() + 1
    part = [{"person": f"p{int(s)}", "musical_experience": m} for s, m in a.drop_duplicates("subject")[["subject", "musical_experience"]].itertuples(index=False)]
    pcols = [("musical_experience", "number", "as in the deposit (unit not stated; years of musical training is likely)",
              "Musical experience of the participant, used by the authors as a covariate. The deposit does not state its unit.")]

    ch = stdlib.Checks()
    ch.published("participants", df.subject.nunique(), 60, 0, P + "Abstract (sixty Mandarin-speaking adults)")
    ch.published("PLV values (60 participants x 3 conditions)", len(df), 180, 0, P + "Abstract (three onset conditions, 60 participants)")
    hi, lo = df[df.group == "high synchronizers"], df[df.group == "low synchronizers"]
    ch.published("high synchronizers", hi.subject.nunique(), 25, 0, "PLVscript.R printed summary (n = 25 per condition)")
    ch.published("low synchronizers", lo.subject.nunique(), 35, 0, "PLVscript.R printed summary (n = 35 per condition)")
    for g, d, vals in (("high", hi, {"aspirated": 0.519, "mixed": 0.446, "unaspirated": 0.528}),
                       ("low", lo, {"aspirated": 0.321, "mixed": 0.302, "unaspirated": 0.321})):
        for o, v in vals.items():
            ch.published(f"mean PLV, {g} synchronizers, {o}", d[d.condition == o + " syllable onsets"].value.mean(), v, 0.0005,
                         "PLVscript.R printed summary of mean PLV by onset and cluster")
    ch.internal("each participant has 3 rows", (df.groupby("subject").size() == 3).all())
    ch.internal("PLV in 0..1", df.value.between(0, 1).all())
    ch.internal("participant mean PLV matches the authors' per-participant table",
                bool(((df.assign(n=df.subject.str.lstrip("p").astype(int)).groupby("n").value.mean() - pd.read_csv(raw / "plvnew60sum.csv").set_index("number").mean_plv).abs() < 0.01).all()))
    for c in ch.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {x: p0[x] for x in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes="Reshaped the authors' PLV table (one value per participant and onset condition) into long form, added the authors' high/low synchronizer cluster as group, moved musical experience to participants.csv, and replaced participant codes.",
        how=["The authors shared one phase locking value per participant and syllable-onset condition from the speech-to-speech synchronization test, so each row is one participant's PLV in one condition (aspirated, unaspirated or mixed syllable onsets).",
             "In the test the participant whispers syllables in time with a stream of syllables whose rate increases; the PLV (0 to 1) measures how consistently the phase of the whispered envelope follows that of the heard envelope. Raw envelopes and the whisper recordings were not shared.",
             "group is the authors' k-means split of the 60 participants into high and low synchronizers on their mean PLV. The unit of musical_experience is not stated in the deposit; no age or sex was shared."],
        unit_note="participant and onset condition",
        differences=["The full text was behind a paywall, so the published checks use the abstract and the values the authors printed in the comments of their analysis script (PLVscript.R)."])
    return stdlib.release(ID, "summary", df.to_dict("records"), meta, ch, participants=part, participant_columns=pcols)


if __name__ == "__main__":
    main()
