"""Cheng & Buchwald (2021), J Speech Lang Hear Res 64(6S):2103-2120: 34 adults learn nonnative stop-stop consonant clusters
(trained on voiced clusters gb, db, gd or voiceless clusters kp, tp, kt) and are tested on trained words, untrained words
with the trained clusters, and transfer clusters of the other voicing, before and after practice.

Raw files, raw_extra/Cheng2021_JSLHR/ (OSF 27ntw, fetched with tools/std/fetch.py):
  data_master_sheet_final.csv              9,787 stop-stop productions (revised by the authors for the 2021 erratum): sub, condition
        (vd = voiced-cluster training, vless = voiceless), training_order, code_order, session 1-6, item_num, target_ipa,
        response_ipa, type (trained, untrained, gen = transfer), word_acc, cluster_acc, cluster_label, cluster_code (c correct,
        e vowel inserted, o other), schwa_dur and burst_dur and vowel_dur in seconds
  master_sheet_stop_fricative_final.csv    3,671 stop-fricative productions (post hoc transfer test): same participants and sessions,
        transcription, cluster_code, vowel_label, vowel_dur (s), c1_on and c2_off (s, time in the recording)
  voicing_transfer_results.Rmd             the authors' analysis, read as text: sessions 1-2 = Baseline, 3-4 = R1, 5-6 = R2 (two
        passes of the same 96 stop-stop items per test session), burst_dur converted from seconds to milliseconds
Run from the site folder: python3 -I tools/std/convert/Cheng2021_JSLHR.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cheng2021_JSLHR"
SRC = "Cheng & Buchwald 2021, J Speech Lang Hear Res"
RAW = stdlib.raw_dir(ID)
DEVICE = "microphone recordings of nonword repetition in the lab"
SESSION = {1: ("Baseline", 1), 2: ("Baseline", 2), 3: ("R1", 1), 4: ("R1", 2), 5: ("R2", 1), 6: ("R2", 2)}
SESNO = {"Baseline": 1, "R1": 2, "R2": 3}
GROUP = {"vd": "trained on voiced clusters", "vless": "trained on voiceless clusters"}
TYPE = {"trained": "trained items", "untrained": "generalization items (untrained words, trained clusters)",
        "gen": "transfer items (clusters of the other voicing)"}
VLESS_SF = ("kf", "tf", "ks")  # stop-fricative targets in the file: kf tf ks (voiceless) and gz gv dv (voiced)

d = pd.read_csv(RAW / "data_master_sheet_final.csv")
f = pd.read_csv(RAW / "master_sheet_stop_fricative_final.csv")
d["cc"] = d.cluster_code.str.strip()
f["cc"] = f.cluster_code.str.strip()
order = {s: i for i, s in enumerate(dict.fromkeys(d["sub"]))}
assert set(f["subject"]) == set(order)


def base():
    return dict(dataset=ID, study=SRC, experiment="", group="", subject="", subject_excluded_by_authors=False, age="", sex="",
                handedness="", population="healthy", setting="lab", device=DEVICE, session=1, block="", trial="", phase="",
                time_s="", task="speech", unit="utterance", condition="", pert_type="none", pert_value=0, pert_unit="",
                feedback="normal", excluded_by_authors=False)


recs = []
for r in d.itertuples(index=False):
    lab, k = SESSION[r.session]
    x = base()
    x.update(experiment="stop-stop clusters", group=GROUP[r.condition], subject=str(r.sub), session=SESNO[lab],
             block=f"{lab} pass {k} (authors' session {r.session})", condition=TYPE[r.type],
             phase="baseline" if lab == "Baseline" else ("transfer" if r.type == "gen" else "test"),
             stimulus=r.target_ipa, segment=f"word-initial {r.cluster} cluster", response_transcription=r.response_ipa,
             word_correct=bool(r.word_acc), cluster_correct=bool(r.cluster_acc), response_code=r.cc,
             epenthetic_vowel_ms="" if r.cc == "o" else r.schwa_dur * 1000,
             burst_to_burst_ms="" if (r.cc == "o" or r.burst_dur == 0) else r.burst_dur * 1000,
             vowel_dur_ms="" if pd.isna(r.vowel_dur) else r.vowel_dur * 1000)
    recs.append((order[r.sub], r.session, r.item_num, 0, x))
for r in f.itertuples(index=False):
    lab, k = SESSION[r.session]
    x = base()
    vless = r.target_ipa[:2] in VLESS_SF
    x.update(experiment="stop-fricative clusters (post hoc transfer test)", group=GROUP[r.condition], subject=str(r.subject),
             session=SESNO[lab], block=f"{lab} pass {k} (authors' session {r.session})",
             condition=f"stop-fricative transfer items ({'voiceless' if vless else 'voiced'} cluster)",
             phase="baseline" if lab == "Baseline" else "transfer", time_s=r.c1_on, cluster_offset_s=r.c2_off,
             stimulus=r.target_ipa, segment=f"word-initial {r.target_ipa[:2]} cluster (following vowel /{r.vowel_label}/)",
             response_transcription=r.transcription, response_code=r.cc, vowel_dur_ms=r.vowel_dur * 1000)
    recs.append((order[r.subject], r.session, r.item, 1, x))
recs.sort(key=lambda t: t[:4])
rows, n = [], {}
for key in recs:
    x = key[4]
    n[x["subject"]] = n.get(x["subject"], 0) + 1
    x["trial"] = n[x["subject"]]
    rows.append(x)
df = pd.DataFrame(rows)

# ---- checks
P = json.load(open(stdlib.HERE / "source/extra_studies.json", encoding="utf-8"))
S = next(e for e in P if e["id"] == ID)
ck = stdlib.Checks()
M = "Method, Participants and Procedure"
ck.published("participants analysed", df.subject.nunique(), 34, 0, "Method, Participants (34 neurotypical adults)")
bl = df[(df.experiment == "stop-stop clusters") & (df.block.str.startswith("Baseline"))]
ck.published("stop-stop target productions per participant at baseline (48 nonwords x 2)", bl.groupby("subject").size().median(), 96, 0,
             "Method, Procedure (baseline: all 48 target nonwords, each produced twice)")
ck.published("stop-stop target nonwords (8 per cluster x 6 clusters)", d.target_ipa.nunique(), 48, 0, "Method, Stimuli (8 nonwords per cluster, 6 clusters)")
ck.published("nonwords per stop-stop cluster", d.groupby("cluster").target_ipa.nunique().median(), 8, 0, "Method, Stimuli")
sf = df[(df.experiment != "stop-stop clusters") & (df.block.str.startswith("Baseline"))]
ck.published("stop-fricative productions per participant per test session", sf.groupby("subject").size().median(), 36, 0,
             "Method, Procedure (36 stop-fricative items per participant per session)")
for s in (1, 2, 3):
    ck.published(f"test session {s}: stop-stop productions per participant (median)",
                 df[(df.experiment == "stop-stop clusters") & (df.session == s)].groupby("subject").size().median(), 96, 0,
                 "Method, Procedure (R1 and R2 have the same structure as baseline)")
ck.internal("rows = both shared tables", len(df) == len(d) + len(f), f"{len(d)} + {len(f)}")
ck.internal("every participant has 6 passes in both tables", bool((d.groupby("sub").session.nunique() == 6).all() and (f.groupby("subject").session.nunique() == 6).all()))
ck.internal("one training group per participant", bool((df.groupby("subject").group.nunique() == 1).all()))
ck.internal("17 participants per training group", sorted(df.drop_duplicates("subject").group.value_counts().tolist()) == [17, 17],
            str(df.drop_duplicates("subject").group.value_counts().to_dict()))
dis = int(((d.cc == "c") != (d.cluster_acc == 1)).sum())
ck.internal("cluster_acc agrees with cluster_code c in nearly all rows", dis < 0.01 * len(d), f"{dis} of {len(d)} rows disagree (both kept)")
zero_burst = d[(d.cc != "o") & (d.burst_dur == 0)]
ck.internal("burst_dur of 0 only for other errors or rare unmeasured tokens", len(zero_burst) < 0.02 * len(d), f"{len(zero_burst)} c/e rows blanked")

meta = dict(
    paper=dict(authors=S["papers"][0]["authors"], year=S["papers"][0]["year"], title=S["papers"][0]["title"],
               venue=S["papers"][0]["venue"], doi=S["papers"][0]["doi"]),
    lab="Buchwald (NYU)", data_url=S["data_url"], license="none stated", license_url="",
    changes=("The authors' durations in seconds (schwa_dur, burst_dur, vowel_dur) are given in milliseconds (x 1000, as the authors' own "
             "code does for burst_dur); the times in the recording (c1_on, c2_off) stay in seconds. Burst "
             "and schwa durations of 0 that stand for productions that were not measured (other errors) are left empty. Cluster codes "
             "had stray spaces that the authors' code also trims. The two tables are merged in the order of the test passes. "
             "The authors' columns code_order, training_order and cluster_label were not copied (a stimulus-order code, the order "
             "of the two training conditions and the label of the produced cluster, which the transcription already shows)."),
    how=["One row is one nonword production (utterance) by one participant, from the authors' two shared tables: the 9,787 stop-stop "
         "productions and the 3,671 stop-fricative productions of a post hoc transfer test.",
         "Participants were trained on voiced (gb, db, gd) or voiceless (kp, tp, kt) clusters (group). Within the test passes, items are "
         "trained words, untrained words with the trained clusters (generalization), and clusters of the other voicing (transfer).",
         "The authors' sessions 1-2 are the two baseline passes, 3-4 retention test R1 (20 minutes after practice) and 5-6 R2 (2 days "
         "later); session here is baseline 1, R1 2, R2 3, and block keeps the authors' pass.",
         "response_code is the authors' code: c correct cluster, e vowel inserted between the consonants, o other error; "
         "epenthetic_vowel_ms and burst_to_burst_ms (first burst to second burst) are measured on correct and vowel-inserted productions. "
         "time_s and cluster_offset_s (stop-fricative rows) are the cluster's start and end in the recording."],
    unit_note="nonword production",
    differences=["The paper's logistic and linear mixed-model coefficients were not refitted; the authors revised the deposited data for "
                 "an erratum (September 2021), so the deposit can differ slightly from the originally printed estimates.",
                 f"In {dis} of {len(d)} stop-stop rows cluster_acc disagrees with cluster_code (for example code c with accuracy 0); both are kept as shared."])
stdlib.release(ID, "speech", rows, meta, ck)
