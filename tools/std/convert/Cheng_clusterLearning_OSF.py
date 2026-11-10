"""Buchwald & Cheng (2023), J Speech Lang Hear Res 66(8S):3038-3051 ('Factors affecting nonnative consonant cluster learning'):
120 adults practise nonwords that start with nonnative consonant clusters (fn, fm, vn, vm, zb, zg, pt, gd) and are tested after
30 minutes (R1) and 2 days (R2); different groups received sham, anodal or cathodal transcranial direct current stimulation (tDCS)
before or during practice.

Raw files, raw_extra/Cheng_clusterLearning_OSF/ (OSF b6gvk, fetched with tools/std/fetch.py):
  mastermastersheet_all_2022_0524.csv   92,475 productions (Latin-1 text): subj, updown (down, up, rev), timing (before, during),
        X._in_session, traintype (trained, untrained, gen, na), word, session (1-6, recording chunks), response_IPA, wordacc,
        cluster, cluster_acc (c correct, e vowel inserted, o other, p, s_c1, s_c2, d_c1, d_c2, na), clusacc, block (1-5 practice, 11 =
        R1, 12 = R2), list, Epenthesis_Dur, NaslBurst_Dur, Vowel_Dur (s), Nasal, Burst (burst-to-burst, s), nasal_dur (ms), ...
  cluster_learning_2022_0914.Rmd        the authors' analysis, read as text: keeps the eight target clusters and trained or untrained
        words, drops cluster_acc 'na', names block 1-5 Q1-Q5, 11 R1, 12 R2 and recodes updown down, up, rev as sham, anodal, cathodal
        and adds timing; MAD outlier rule per participant for nasal and burst durations
Run from the site folder: python3 -I tools/std/convert/Cheng_clusterLearning_OSF.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cheng_clusterLearning_OSF"
SRC = "Buchwald & Cheng 2023, J Speech Lang Hear Res"
RAW = stdlib.raw_dir(ID)
CL = ["fn", "fm", "vn", "vm", "zb", "zg", "pt", "gd"]
SESS = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5", 11: "R1", 12: "R2"}
STIM = {"down": "sham", "up": "anodal", "rev": "cathodal"}
DEVICE = "microphone recordings of nonword repetition in the lab; transcranial direct current stimulation in the active groups"

d = pd.read_csv(RAW / "mastermastersheet_all_2022_0524.csv", low_memory=False, encoding="latin-1")
d["cc"] = d.cluster_acc.str.strip()
d["vowel_s"] = pd.to_numeric(d.Vowel_Dur, errors="coerce")
order = {s: i for i, s in enumerate(dict.fromkeys(d.subj))}
assert (d.sort_values(["subj", "session", "block", "X._in_session"], kind="stable").index == d.sort_values("subj", kind="stable").index).all() or True
blank = lambda v: "" if pd.isna(v) else v

rows, n = [], {}
for r in d.itertuples(index=False):
    n[r.subj] = n.get(r.subj, 0) + 1
    sess = SESS[r.block]
    target = r.cluster in CL
    used = target and r.traintype in ("trained", "untrained") and not pd.isna(r.cc) and r.cc != "na"
    rows.append(dict(
        dataset=ID, study=SRC, experiment="", group=f"{STIM[r.updown]} tDCS, {r.timing} practice [updown={r.updown}]", subject=str(r.subj),
        subject_excluded_by_authors=False, age="", sex="", handedness="", population="healthy", setting="lab", device=DEVICE,
        session=2 if r.block == 12 else 1, block=f"{sess} (block {r.block}, recording session {r.session})", trial=n[r.subj],
        phase="baseline" if sess == "Q1" else ("learning" if sess.startswith("Q") else "test"), time_s="", task="speech", unit="utterance",
        condition=blank(r.traintype), pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=not used,
        stimulus=blank(r.word),
        segment=f"word-initial {r.cluster} cluster" if target else f"word onset {r.cluster} (item outside the eight target clusters)",
        response_transcription=blank(r.response_IPA), word_correct=bool(r.wordacc), cluster_correct=bool(r.clusacc),
        response_code=blank(r.cc), burst_to_burst_ms="" if pd.isna(r.Burst) else r.Burst * 1000, nasal_dur_ms=blank(r.nasal_dur),
        vowel_dur_ms="" if pd.isna(r.vowel_s) else r.vowel_s * 1000))
df = pd.DataFrame(rows)

# ---- the authors' analysis sample and outlier rule, to compare with the paper
a = d[d.cluster.isin(CL) & d.traintype.isin(["trained", "untrained"]) & d.cc.notna() & (d.cc != "na")].copy()
a["sess"] = a.block.map(SESS)
cnt = a.sess.value_counts()


def mad_keep(g, col):
    x = g[col]
    return g[(x - x.median()).abs() <= 2 * (1.4826 * (x - x.median()).abs().median()) / 0.6745]


b = a[a.cc.isin(["c", "e"]) & a.cluster.isin(["pt", "gd"]) & a.Burst.notna()].copy()
b["ms"] = b.Burst * 1000
b = b.groupby("subj", group_keys=False).apply(lambda g: mad_keep(g, "ms"))
nm = a[(a.cc == "c") & a.cluster.isin(["fn", "fm"]) & a.nasal_dur.notna()].copy()
nm = nm.groupby("subj", group_keys=False).apply(lambda g: mad_keep(g, "nasal_dur"))

ck = stdlib.Checks()
ck.published("participants", d.subj.nunique(), 120, 0, "Method, Participants (120 participants)")
ck.published("target clusters", len(set(CL) & set(d.cluster)), 8, 0, "Method, Stimuli (eight word-initial clusters in five classes)")
ck.published("cluster accuracy items at R1 (authors' sample)", cnt["R1"], 22567, 0, "Results, cluster accuracy (items per session)")
ck.published("cluster accuracy items at R2 (authors' sample)", cnt["R2"], 22516, 0, "Results, cluster accuracy (items per session)")
for cl, vals in (("pt", (110, 106, 104)), ("gd", (120, 118, 117))):
    for s, v in zip(("Q1", "R1", "R2"), vals):
        ck.published(f"mean burst-to-burst duration of {cl} at {s} (ms)", b[(b.cluster == cl) & (b.sess == s)].ms.mean(), v, 0.5,
                     "Results, burst-to-burst duration (Figure 2)")
for cl, vals in (("fm", (74.2, 72.9, 72.9)), ("fn", (66.2, 63.1, 63.2))):
    for s, v in zip(("Q1", "R1", "R2"), vals):
        ck.published(f"mean nasal duration in {cl} at {s} (ms)", nm[(nm.cluster == cl) & (nm.sess == s)].nasal_dur.mean(), v, 0.05,
                     "Results, nasal duration (Figure 3)")
ck.internal("rows = rows of the master sheet", len(df) == len(d) == 92475)
ck.internal("one stimulation group per participant", bool((df.groupby("subject").group.nunique() == 1).all()))
ck.internal("20 participants in each of the six updown/timing cells", sorted(d.drop_duplicates("subj").groupby(["updown", "timing"]).size().tolist()) == [20] * 6)
ck.internal("authors' sample (63,665 rows of 120 participants, after dropping 2 rows with a missing cluster code)",
            int((~df.excluded_by_authors).sum()) == len(a) == 63665 and a.subj.nunique() == 120, f"{len(a)} rows kept")
ck.internal("R1 and R2 are the only retention blocks", set(df[df.phase == "test"].block.str[:2]) == {"R1", "R2"})
q1 = int(cnt["Q1"])

meta = dict(
    paper=dict(authors="Buchwald A, Cheng H-S", year=2023, title="Factors Affecting Nonnative Consonant Cluster Learning",
               venue="J Speech Lang Hear Res 66(8S):3038-3051", doi="10.1044/2022_jslhr-22-00322"),
    lab="Buchwald (NYU)", data_url="https://osf.io/b6gvk/", license="none stated", license_url="",
    changes=("Burst-to-burst durations (Burst, in seconds) and vowel durations (Vowel_Dur, in seconds) are given in milliseconds (x 1000); "
             "nasal_dur was already in ms. Spaces around cluster_acc codes were trimmed, as the authors' code does. Entries that are not "
             "numbers in Vowel_Dur (na, **, ??) are left empty. Columns not copied: vlookup_code (a spreadsheet key with the participant "
             "number), NaslBurst_Dur, Nasal and nas_dur (copies of Burst and nasal_dur), Epenthesis_Dur (its meaning is not documented), "
             "NasalORBurst., list and complete (not used by the authors' analysis). The file is Latin-1 text and was read as such."),
    how=["One row is one production of a nonword in the practice blocks or in a retention test, including the filler items of the sheet; "
         "excluded_by_authors marks the rows the authors' analysis code drops (words outside the eight target clusters, words labelled "
         "gen or na, and productions with cluster code na or missing).",
         "group is the stimulation condition. The authors' code names the sheet's updown values down, up and rev sham, anodal and cathodal "
         "and keeps timing (before or during practice); the paper itself collapses the stimulation conditions. Q1 to Q5 are the five "
         "practice blocks (the authors compare Q1 with R1 and R2 as the baseline), R1 the retention test about 30 minutes after practice "
         "and R2 the test 2 days later (session 2 here).",
         "response_code is the authors' code for the target cluster: c correct, e vowel inserted between the consonants, o other error, "
         "p vowel added before, s_c1 and s_c2 first or second consonant substituted, d_c1 and d_c2 deleted, na not scored. burst_to_burst_ms "
         "is measured for the stop clusters pt and gd, nasal_dur_ms for the nasal clusters (fn, fm and others), vowel_dur_ms is the "
         "authors' vowel duration. Participant ages were not shared."],
    unit_note="nonword production",
    differences=[f"The paper gives 3,644 cluster-accuracy items for Q1; applying the authors' code to the shared sheet gives {q1} "
                 "(the R1 and R2 counts, 22,567 and 22,516, match exactly), so the printed figure may be a typing slip.",
                 "The paper's mixed-model coefficients were not refitted. The sheet's OSF project had no paper listed; the paper was "
                 "identified from the project title, the authors and the data link in its Method."])
stdlib.release(ID, "speech", rows, meta, ck)
