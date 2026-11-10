"""Cheng, Masapollo, Hagedorn & Buchwald (2024), J Speech Lang Hear Res 67(10):3549-3565: temporal coordination of articulatory
gestures in nonnative (/fn/) and native (/fl/, /sm/) onset clusters, electromagnetic articulography (Carstens AG501, 250 Hz)
in 12 American English speakers (9 analysed).

Raw files, raw_extra/Cheng2024_JSLHR/ (OSF dxeka, fetched with tools/std/fetch.py):
  all_acoustic_data_2022_0301.csv   one row per token (3,577): subj, trialnum, type (singleton, boundarySingleton, nativeCC,
      nonnativeCC, lexical), sound (the consonant or cluster the item tests), word, item, transcription (the authors' ASCII
      transcription of the production), cluster_code (c, e, o), nasal, nasal_dur and vowel_dur (seconds, 999 = none)
  all_kin_data_2022_0629.csv        one row per tracked gesture (7,511; 1-3 per token): subj, trialnum, type, sound, word,
      gesture (LA, LL_Z, TT_Z, TR_Z), landmark times GONS, PVEL, NONS, MAXC, NOFFS, PVEL2, GOFFS (ms), listnum (order within
      the token), segment, syllable (same, different)
  statistical_analysis_for_ms_20231027.Rmd   the authors' analysis, read as text: joins the two tables on subj and trialnum
      (left join from the acoustic table), drops subj 9, 16 and 18, keeps production codes c and e, and computes kin_dur = NOFFS - NONS
Run from the site folder: python3 -I tools/std/convert/Cheng2024_JSLHR.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Cheng2024_JSLHR"
SRC = "Cheng, Masapollo, Hagedorn & Buchwald 2024, J Speech Lang Hear Res"
DROP = {9, 16, 18}  # the authors' drop_subj_list (tongue body sensor fell off, paper Method)
DEVICE = "Carstens AG501 electromagnetic articulograph (250 Hz), t.bone EM9600 shotgun microphone (48 kHz)"
LAND = ["GONS", "PVEL", "NONS", "MAXC", "NOFFS", "PVEL2", "GOFFS"]
LCOL = {"GONS": "gons_ms", "PVEL": "pvel_ms", "NONS": "nons_ms", "MAXC": "maxc_ms", "NOFFS": "noffs_ms", "PVEL2": "pvel2_ms",
        "GOFFS": "goffs_ms"}
SM = ["SMAGDEEP", "SMACKPOOT", "MAGDEEP", "MACKPOOT", "SAGDEEP", "SACKPOOT"]  # the authors' word lists (Rmd line 418-420)
FN = ["FNAGBEET", "FNACKTOOP", "NAGBEET", "NACKTOOP", "FAGBEET", "FACKTOOP"]

raw = stdlib.raw_dir(ID)
ac = pd.read_csv(raw / "all_acoustic_data_2022_0301.csv")
kn = pd.read_csv(raw / "all_kin_data_2022_0629.csv")
n_ac_rows, n_kn_rows = len(ac), len(kn)
ac["cluster_code"] = ac["cluster_code"].str.strip()
ac["nasal"] = ac["nasal"].str.strip()

# token order: the acoustic table's order; a token only in the kinematic table goes after the kinematic token before it
ac_keys = list(zip(ac["subj"], ac["trialnum"]))
pos = {k: i for i, k in enumerate(ac_keys)}
assert len(pos) == len(ac_keys)
kn_keys = list(dict.fromkeys(zip(kn["subj"], kn["trialnum"])))
order = [(float(i), 0, k) for i, k in enumerate(ac_keys)]
last, extra = -1.0, 0
kin_only = []
for k in kn_keys:
    if k in pos:
        last = float(pos[k])
    else:
        extra += 1
        order.append((last + 0.5, extra, k))
        kin_only.append(k)
order.sort(key=lambda t: (t[0], t[1]))
tokens = [k for _, _, k in order]
shared_pos = [pos[k] for k in kn_keys if k in pos]
order_same = shared_pos == sorted(shared_pos)

acd = {k: r for k, r in zip(ac_keys, ac.to_dict("records"))}
kgroups = {k: g.to_dict("records") for k, g in kn.groupby(["subj", "trialnum"], sort=False)}
label_conflicts = [k for k in kgroups if k in acd and any(acd[k][c] != kgroups[k][0][c] for c in ("type", "sound", "word"))]


def num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v


rank = {}
for s_ in sorted({int(k[0]) for k in tokens}):
    for r_, tn_ in enumerate(sorted({int(k[1]) for k in tokens if int(k[0]) == s_}), start=1):
        rank[(s_, tn_)] = r_
rows = []
for key in tokens:
    s, tn = int(key[0]), int(key[1])
    a, g = acd.get(key), kgroups.get(key)
    lab = a if a is not None else g[0]
    code = a["cluster_code"] if a is not None else None
    base = dict(dataset=ID, study=SRC, experiment="", group="", subject=str(s), person=str(s),
                subject_excluded_by_authors=s in DROP, age=None, sex=None, handedness=None, population="healthy", setting="lab",
                device=DEVICE, session=1, block="", trial=rank[(s, tn)], phase="other", time_s=None, task="speech", unit="utterance",
                condition=lab["type"], pert_type="none", pert_value=0, pert_unit="", feedback="normal",
                # the authors' join drops tokens missing from either table, and their analysis keeps production codes c and e
                excluded_by_authors=bool(a is None or g is None or code == "o"),
                stimulus=lab["word"], transcription=a["transcription"] if a is not None else None, target_sound=lab["sound"],
                production_code=code,
                nasal_in_token=num(a["nasal"]) if a is not None else None,
                nasal_duration_ms=(None if a is None or a["nasal_dur"] == 999 else a["nasal_dur"] * 1000),
                vowel_duration_ms=(None if a is None or a["vowel_dur"] == 999 else a["vowel_dur"] * 1000),
                item_number=int(a["item"]) if a is not None else None, trial_number_authors=tn)
    if g is None:
        rows.append({**base, "gesture_signal": None, "gesture_order": None, "gesture_segment": None, "gesture_syllable": None,
                     **{c: None for c in LCOL.values()}})
    else:
        for r in g:
            rows.append({**base, "gesture_signal": r["gesture"], "gesture_order": int(r["listnum"]), "gesture_segment": r["segment"],
                         "gesture_syllable": r["syllable"], **{LCOL[c]: r[c] for c in LAND}})
df = pd.DataFrame(rows)

# ---- checks -------------------------------------------------------------------------------------------------------------
ck = stdlib.Checks()
SECT = "Method, Participants"
ck.published("participants recruited", df["subject"].nunique(), 12, 0, SECT)
ck.published("participants analysed (3 excluded for a sensor that fell off)",
             df.loc[~df["subject_excluded_by_authors"], "subject"].nunique(), 9, 0, f"Abstract; {SECT}")
an = df[~df["subject_excluded_by_authors"] & df["gesture_order"].notna()].copy()  # the authors' analysed table: both tables, 9 people
first = an[an["gesture_order"] == 1]
nz = first[first["target_sound"] != "zm"]
n_o = int((nz["production_code"] == "o").sum())
ck.published("utterances removed: other-error tokens plus the 3 iambic-stress tokens the paper adds", n_o + 3, 46, 0,
             "Method, data exclusions")
ck.published("removed utterances as a percentage of all utterances", 100 * (n_o + 3) / (len(nz) + 3), 2, 0.5,
             "Method, data exclusions ('approximately 2%')")
fn_all = first[first["target_sound"] == "fn"]
fn_cc = first[(first["target_sound"] == "fn") & (first["condition"] == "nonnativeCC")]
codes_all = fn_all["production_code"].value_counts().to_dict()
codes_cc = fn_cc["production_code"].value_counts().to_dict()
ck.published("/fn/ tokens (nonnative cluster + f#n boundary tokens)", len(fn_all), 304, 0, "Results, accuracy of /fn/")
ck.published("/fn/ tokens with an epenthetic vocoid", codes_all.get("e", 0), 29, 0, "Results, accuracy of /fn/")
ck.published("/fn/ tokens with a phonemic error", codes_all.get("o", 0), 6, 0, "Results, accuracy of /fn/")
ck.published("/fn/ accuracy (%)", 100 * (len(fn_all) - codes_all.get("e", 0) - codes_all.get("o", 0)) / len(fn_all), 88, 0.5,
             "Results, accuracy of /fn/")

# kinematic durations as in the authors' Rmd (NOFFS - NONS, production codes c and e, nasal or fricative gesture, their word lists)
dc = an[an["production_code"].isin(["c", "e"])].copy()
dc["kin_dur"] = dc["noffs_ms"] - dc["nons_ms"]
dc["snd"] = np.where(dc["condition"] == "boundarySingleton", dc["target_sound"].str.replace(r"^(.)(.*)$", r"\1#\2", regex=True),
                     dc["target_sound"])
MP = {"m": "#m", "sm": "#sm", "n": "#n", "fn": "#fn", "s": "#s", "f": "#f"}
dc["snd"] = dc["snd"].map(lambda x: MP.get(x, x))
nas = dc[dc["nasal_in_token"].isin(["n", "m"]) & dc["gesture_segment"].isin(["n", "m"])]
fri = dc[dc["gesture_segment"].isin(["f", "s"]) & dc["target_sound"].isin(["fn", "sm", "f", "s"])]
# (table, words, sound labels, paper means and SDs printed in Results with their printed precision)
PUB = [("nasal", SM, "/m/", [("#m", 59.0, 0.5, 13.3, 0.05), ("s#m", 49.2, 0.05, 14.4, 0.05), ("#sm", 41.1, 0.05, 9.34, 0.005)]),
       ("nasal", FN, "/n/", [("#n", 75.8, 0.05, 23, 0.5), ("f#n", 71.4, 0.05, 27, 0.5), ("#fn", 60.3, 0.05, 35, 0.5)]),
       ("fric", SM, "/s/", [("#s", 94.3, 0.05, 44.1, 0.05), ("s#m", 107, 0.5, 51.8, 0.05), ("#sm", 126, 0.5, 52.7, 0.05)]),
       ("fric", FN, "/f/", [("#f", 57.6, 0.25, 29, 0.5), ("f#n", 64.1, 0.25, 41.3, 0.05), ("#fn", 73.1, 0.05, 43, 0.5)])]
dev = []
for tab, words, seg, lst in PUB:
    d = (nas if tab == "nasal" else fri)
    d = d[d["stimulus"].isin(words)]
    for lab, mean_p, mtol, sd_p, sdtol in lst:
        x = d.loc[d["snd"] == lab, "kin_dur"]
        ck.published(f"{seg} kinematic duration {lab}: mean (ms)", x.mean(), mean_p, mtol, "Results, nasal and fricative kinematic durations")
        ck.published(f"{seg} kinematic duration {lab}: SD (ms)", x.std(), sd_p, sdtol, "Results, nasal and fricative kinematic durations")
        if abs(x.mean() - mean_p) > 0.05 + (0.45 if mtol == 0.5 else 0):
            dev.append(f"{seg} {lab}: {x.mean():.2f} here, {mean_p} in the paper")

# ---- internal checks -------------------------------------------------------------------------------------------------------
n_gest = int(df["gesture_order"].notna().sum())
ck.internal("every kinematic row kept", n_gest == n_kn_rows, f"{n_gest} gesture rows of {n_kn_rows}")
ck.internal("every acoustic token kept, plus the kinematic-only tokens",
            df[["subject", "trial_number_authors"]].drop_duplicates().shape[0] == n_ac_rows + len(kin_only) == len(tokens),
            f"{len(tokens)} tokens = {n_ac_rows} acoustic + {len(kin_only)} kinematic-only")
ck.internal("row count = kinematic rows + tokens without gesture values", len(df) == n_kn_rows + int(df["gesture_order"].isna().sum()),
            f"{len(df)} rows")
ck.internal("each participant's trial ranks run 1..n without gaps", all(
    sorted(df.loc[df["subject"] == str(s_), "trial"].unique()) == list(range(1, df.loc[df["subject"] == str(s_), "trial"].nunique() + 1))
    for s_ in range(7, 19)))
ck.internal("12 participants, none lost", df["subject"].nunique() == 12 and set(map(int, df["subject"].unique())) == set(range(7, 19)))
ck.internal("landmarks in the order GONS<=PVEL<=NONS<=MAXC<=NOFFS<=PVEL2<=GOFFS on most rows",
            float((kn[LAND].diff(axis=1).iloc[:, 1:] >= 0).all(axis=1).mean()) > 0.98,
            f"{int((kn[LAND].diff(axis=1).iloc[:, 1:] >= 0).all(axis=1).sum())} of {n_kn_rows} rows")
print("tokens", len(tokens), "kin-only", len(kin_only), "label conflicts", len(label_conflicts), "rows", len(df),
      "| kinematic file in the acoustic file's token order:", order_same)
print("fn all tokens", len(fn_all), codes_all, "| nonnativeCC fn tokens", len(fn_cc), codes_cc, "| removed", n_o, "of", len(nz))
print("deviations:", dev)
for c in ck.items:
    if not c["ok"]:
        print("FAILING:", c)

meta = dict(
    paper=dict(authors="Cheng HS, Masapollo M, Hagedorn C, Buchwald A", year=2024,
               title="Temporal Coordination of Articulatory Gestures in Nonnative Onset Clusters: Evidence From American English "
                     "Speakers Using Electromagnetic Articulography",
               venue="J Speech Lang Hear Res 67(10):3549-3565", doi="10.1044/2024_jslhr-24-00025"),
    lab="Buchwald (NYU)", data_url="https://osf.io/dxeka/", license="none stated", license_url="",
    changes=("Joined the authors' two tables (acoustic and kinematic) on participant and trial number; times and durations in ms "
             "(the acoustic durations were in seconds); the authors' 999 (no nasal in the item) left empty; white space removed from "
             "the production and nasal codes; participant numbers replaced by codes."),
    how=["One row is one articulatory gesture of one spoken token, from the authors' kinematic table (7,511 rows, one to three per "
         "token); each token's item, transcription, production code and acoustic durations from the authors' acoustic table "
         "(3,577 tokens) are repeated on its rows, and the 172 tokens with no gesture values keep one row with the gesture columns "
         "empty.",
         "Landmark times (gesture onset, peak velocity, target onset, maximum constriction, target offset, release peak velocity, "
         "gesture offset) are in milliseconds from the start of each token's recording, as the authors give them (EMA sampled at "
         "250 Hz, so most are multiples of 4 ms); nasal and vowel durations are acoustic values in ms.",
         "Three of the 12 participants are marked as excluded by the authors (a tongue body sensor fell off). excluded_by_authors "
         "marks the tokens their analysis drops: other-error tokens (production_code o) and tokens that are in only one of the two "
         "tables.",
         "The trial column is the rank of the authors' trial number within the participant, not the position of the row: the rows "
         "follow the authors' acoustic table, which is not in trial order. The authors' own trial number is in trial_number_authors."],
    unit_note="articulatory gesture of a spoken token (1 to 3 per token; a token with no gesture values has one row)",
    differences=[
        "Kinematic durations of /f/ (target offset minus target onset, as in the authors' code): the mean is 57.40 ms for #f and "
        "64.27 ms for f#n here, where the paper prints 57.6 and 64.1; the other ten means and all twelve SDs agree to the digits the "
        "paper prints.",
        "The paper's 304 /fn/ tokens pool the 154 nonnative-cluster tokens with 150 f#n boundary tokens: all 29 vocoid tokens are "
        "cluster tokens, and the 6 phonemic errors are 3 cluster and 3 boundary tokens. The 88% accuracy is therefore pooled; for the "
        "154 cluster tokens alone it is 79% (122 correct).",
        "The paper says 288 tokens were collected per participant (22 nonwords x 8 = 176); the tables hold 279 to 392 tokens per "
        "participant because they include other items (/zm/ clusters, real words).",
        "The two tables disagree about the item of 2 tokens (one is FAGBEET in the acoustic table and FNAGBEET in the kinematic "
        "table; the other is a nativeCC token in one and a singleton in the other); the acoustic table's labels are kept, as in the "
        "authors' code. The two files are sorted differently (neither is in the order of the authors' trial numbers); rows follow "
        "the acoustic table, and the trial column gives the order of the authors' trial numbers.",
    ],
)
if not ck.passed():
    sys.exit("checks failed")
stdlib.release(ID, "articulography", df, meta, ck)
