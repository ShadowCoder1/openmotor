"""Zhou et al. (2026), Acta Psychologica: joint agency in musical joint action with electronic music boxes, per participant and trial.

Run from the site folder: python3 -I tools/std/convert/Zhou2026_ActaPsychol.py
Raw files (untrusted data, read only) in raw_extra/Zhou2026_ActaPsychol/, from https://openneuro.org/datasets/ds007471 (CC0):
  behavioural_all.tsv                   one row per participant and trial (space-separated): pair, participant, block, trial,
                                        condition, part performed, tone sequence, joint agency rating, mean and SD of
                                        synchronization performance (columns described in the README)
  sub-NN_task-jointaction_beh.tsv       the same table split by pair (used as an internal check)
  participants.tsv                      age of the left (age1) and right (age2) participant of each pair
  README.md                             column and event-code descriptions
The note onsets in the EEG events (S128 = left box, S4 = right box) are not converted: the marker stream is incomplete and
irregular (see the how text), so the authors' own per-trial table is the behavioural record.
"""
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zhou2026_ActaPsychol"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
SRC = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
SEQ = {1: "Twinkle Twinkle Little Star", 2: "Hush Little Baby", 3: "B.I.N.G.O.", 4: "Yankee Doodle",
       5: "constant pitch, A4 as the higher pitch", 6: "constant pitch, C5 as the higher pitch",
       7: "constant pitch, E-flat5 as the higher pitch", 8: "constant pitch, F-sharp5 as the higher pitch"}
COND = {1: "musical duet", 0: "constant pitch sequence"}
PART = {(1, 0): "melody part", (1, 1): "accompaniment part", (0, 0): "higher-pitch part", (0, 1): "lower-pitch part"}
FIELDS = ["PairNumber", "ParticipantNumber", "BlockNumber", "TrialNumber", "ExperimentalCondition", "PartPerformed", "ToneSequence",
          "JointAgencyRatings", "MeanSynchronizationPerformance", "SDSynchronizationPerformance"]


def table(name):
    lines = [l.split() for l in open(RAW / name, encoding="utf-8").read().splitlines() if l.strip()]
    assert lines[0] == FIELDS, lines[0]
    return [dict(zip(FIELDS, l)) for l in lines[1:]]


def fnum(x):
    return None if x in ("NaN", "nan", "NA", "n/a", "") else float(x)


allrows = table("behavioural_all.tsv")
checks.internal("behavioural_all.tsv matches the downloaded hash", stdlib.sha256(RAW / "behavioural_all.tsv") == SRC["behavioural_all.tsv"]["sha256"])
ages = {}
with open(RAW / "participants.tsv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        ages[int(r["participant_id"].split("-")[1])] = (float(r["age1"]), float(r["age2"]))

# per-pair files must hold exactly the rows of the compiled table
per_pair = []
for pair in range(1, 33):
    name = f"sub-{pair:02d}_task-jointaction_beh.tsv"
    checks.internal(f"{name} matches the downloaded hash", stdlib.sha256(RAW / name) == SRC[name]["sha256"])
    per_pair += table(name)
_key = lambda r: tuple(r[k] for k in FIELDS)  # noqa: E731
_cmp = {(r["PairNumber"], r["ParticipantNumber"], r["TrialNumber"]): r for r in allrows}
_dif = [(a, _cmp[(a["PairNumber"], a["ParticipantNumber"], a["TrialNumber"])]) for a in per_pair
        if _key(a) != _key(_cmp[(a["PairNumber"], a["ParticipantNumber"], a["TrialNumber"])])]
_only_cond = all(a["ExperimentalCondition"] != b["ExperimentalCondition"] and all(a[k] == b[k] for k in FIELDS if k != "ExperimentalCondition") for a, b in _dif)
_pairs_dif = sorted({int(a["PairNumber"]) for a, _ in _dif})
_pf_wrong = all((int(a["ToneSequence"]) <= 4) != (int(a["ExperimentalCondition"]) == 1) for a, _ in _dif)
checks.internal("the 32 per-pair files hold the rows of behavioural_all.tsv; the only differences are the ExperimentalCondition values of one pair "
                "(pair-file value contradicts that pair's tone sequences, the compiled table does not), and the compiled table is used",
                len(per_pair) == len(allrows) == 2560 and _only_cond and _pf_wrong and _pairs_dif == [2] and len(_dif) == 80,
                f"{len(_dif)} rows differ, pairs {_pairs_dif}")

pairs = sorted({int(r["PairNumber"]) for r in allrows})
people = sorted({r["ParticipantNumber"] for r in allrows}, key=lambda s: (int(s[:-1]), s[-1]))
n_per = Counter(r["ParticipantNumber"] for r in allrows)
checks.internal("every participant has 40 trials numbered 1-40, in 8 blocks of 5 (README: 8 tone sequences x 5 joint trials)",
                set(n_per.values()) == {40} and all(sorted(int(r["TrialNumber"]) for r in allrows if r["ParticipantNumber"] == p) == list(range(1, 41)) for p in people[:64])
                and all(int(r["BlockNumber"]) == (int(r["TrialNumber"]) - 1) // 5 + 1 for r in allrows))
checks.internal("participant numbers: pair number and seat (1 left, 2 right) as the README says", all(
    r["ParticipantNumber"] in (f"{int(r['PairNumber'])}1", f"{int(r['PairNumber'])}2") for r in allrows))
by_pt = defaultdict(dict)
for r in allrows:
    by_pt[(int(r["PairNumber"]), int(r["TrialNumber"]))][r["ParticipantNumber"][-1]] = r
same = all(v["1"]["MeanSynchronizationPerformance"] == v["2"]["MeanSynchronizationPerformance"]
           and v["1"]["SDSynchronizationPerformance"] == v["2"]["SDSynchronizationPerformance"]
           and v["1"]["ToneSequence"] == v["2"]["ToneSequence"] and v["1"]["ExperimentalCondition"] == v["2"]["ExperimentalCondition"]
           and v["1"]["PartPerformed"] != v["2"]["PartPerformed"] for v in by_pt.values())
checks.internal("the two partners of a pair share sequence, condition and both synchronization values and play different parts", same)
checks.internal("each tone sequence belongs to one condition (1-4 duets, 5-8 constant pitch)", all(
    (int(r["ToneSequence"]) <= 4) == (int(r["ExperimentalCondition"]) == 1) for r in allrows))
ratings = [int(r["JointAgencyRatings"]) for r in allrows]
checks.internal("joint agency ratings are whole numbers from 1 to 7", min(ratings) >= 1 and max(ratings) <= 7)
duet = st.mean(int(r["JointAgencyRatings"]) for r in allrows if r["ExperimentalCondition"] == "1")
const = st.mean(int(r["JointAgencyRatings"]) for r in allrows if r["ExperimentalCondition"] == "0")
checks.internal("duets get higher joint agency ratings than constant pitch sequences (abstract: duets elicited stronger joint agency)",
                duet > const, f"{duet:.2f} against {const:.2f}")
miss = sum(fnum(r[k]) is None for r in allrows for k in FIELDS[7:])
checks.internal("no missing ratings or synchronization values", miss == 0, f"{miss} missing")

SRC_R = "OpenNeuro README of the deposit (Behavioural Data Description); the paper text was not readable"
checks.published("number of pairs", len(pairs), 32, 0, SRC_R + ": Pair Number values 1-32")
checks.published("number of participants", len(people), 64, 0, "OpenNeuro dataset page: 64 participants (32 pairs, two seats each)")
checks.published("test trials per pair", len({int(r["TrialNumber"]) for r in allrows}), 40, 0, SRC_R + ": 8 tone sequences x 5 joint trials")
checks.published("joint trials per tone sequence and pair", Counter(r["ToneSequence"] for r in per_pair if r["ParticipantNumber"].endswith("1") and r["PairNumber"] == "1")["1"], 5, 0, SRC_R)
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    else:
        print(f"  [{'ok' if ch['ok'] else 'FAILED'}] {ch['what']} {ch.get('detail', '')}")

# ---------------------------------------------------------------- the OpenMotor table (summary module, long form)
NAMES = [c[1] for c in stdlib.columns_for("summary") if c[1] != "person"]
MEASURES = [
    ("JointAgencyRatings", "joint_agency_rating", "value", "rating, 1-7",
     "The participant's rating of their sense of joint agency after the trial, on the authors' 1-7 scale."),
    ("MeanSynchronizationPerformance", "mean_synchronization_performance", "mean", "proportion of the inter-onset interval",
     "Absolute asynchrony between the two partners' note onsets at each beat, as a proportion of the inter-onset interval from the "
     "preceding note onset, averaged over the two partners and over all beats of the trial (the authors' definition; "
     "the same value for both partners of a pair; smaller means closer synchrony)."),
    ("SDSynchronizationPerformance", "sd_synchronization_performance", "sd", "proportion of the inter-onset interval",
     "Standard deviation of the asynchronies across all beats of the trial (the authors' definition; the same value for both "
     "partners of a pair)."),
]
rows = []
for r in allrows:
    pair, seat = int(r["PairNumber"]), int(r["ParticipantNumber"][-1])
    cond, part = int(r["ExperimentalCondition"]), int(r["PartPerformed"])
    for col, name, stat, unit, desc in MEASURES:
        row = {n: None for n in NAMES}
        row.update(
            dataset=ID, study="Zhou et al. 2026 Acta Psychol", experiment="joint agency, EEG hyperscanning, electronic music boxes",
            subject=r["ParticipantNumber"], subject_excluded_by_authors=None, age=ages[pair][seat - 1],
            setting="lab", device="electronic music boxes (one key press sounds the next note), 64-channel EEG hyperscanning",
            session=1, block=f"block {int(r['BlockNumber'])}", trial=int(r["TrialNumber"]), phase="test",
            task="finger tapping and timing", unit="trial",
            condition=f"{COND[cond]}: {SEQ[int(r['ToneSequence'])]}, {PART[(cond, part)]}",
            pert_type="none", pert_value=0, feedback=None, excluded_by_authors=None,
            measure=name, measure_description=desc, statistic=stat, value=fnum(r[col]), value_unit=unit,
        )
        rows.append(row)
checks.internal("7680 rows: 64 participants x 40 trials x 3 measures", len(rows) == 7680, f"{len(rows)}")

participants = []
for p in people:
    pr = next(r for r in allrows if r["ParticipantNumber"] == p)
    participants.append(dict(person=p, pair=int(pr["PairNumber"]), seat="left" if p[-1] == "1" else "right"))
participant_columns = [
    ("pair", "integer", "1-32", "Pair the participant belonged to, in the authors' pair numbering; partners performed together in every trial."),
    ("seat", "category", "left, right", "Seat of the participant in the pair (left participant: EEG channels 1-32; right participant: channels 33-64)."),
]
meta = dict(
    study="Zhou et al. 2026",
    paper=dict(authors="Zhou Z, Zamm A, Christensen J, Rao V, Loehr JD", year=2026,
               title="Attentional engagement strengthens joint agency: Evidence from intra-brain, inter-brain, and behavioural signals via EEG hyperscanning",
               venue="Acta Psychologica 266:106849", doi="10.1016/j.actpsy.2026.106849"),
    lab="Loehr (University of Saskatchewan)",
    data_url="https://openneuro.org/datasets/ds007471",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Converted to the OpenMotor format with participant codes replaced: the authors' table of one row per participant and trial "
             "(rating and two synchronization values) was re-encoded in long form, one row per measure, with the tone sequence "
             "and the part performed written out as words; the EEG recordings and their event markers were not converted."),
    how=[
        "Pairs of partners each played one part of a tone sequence on their own electronic music box (each key press sounds "
        "the next note of the part), in step with a short metronome count-in, for 8 sequences (four familiar duets, melody "
        "part and accompaniment part; four constant-pitch sequences, a higher and a lower pitch) with 5 joint trials each, "
        "grouped in 8 blocks (40 trials per pair), and each partner rated their sense of joint agency after every trial. "
        "One row is one measure of one participant in one trial: joint_agency_rating (1-7) and the two synchronization "
        "values the authors computed for the pair, mean_synchronization_performance and sd_synchronization_performance "
        "(asynchronies between the partners' note onsets as a proportion of the inter-onset interval, as defined in the "
        "deposit's README); both partners' rows hold the same two synchronization values.",
        "condition gives the authors' condition (musical duet or constant pitch sequence), the tone sequence and the part "
        "performed; the values 0 and 1 of the deposit's condition and part columns are written out as the README defines them. "
        "pair and seat are in participants.csv; age is the left or right participant's age from the deposit's participants file; "
        "the deposit gives no sex, handedness or clinical information.",
        "The note onsets of both partners are in the EEG event markers (S128 after the first five metronome onsets, S4), but "
        "they were not converted: the marker stream does not follow the README's description (for example 57 trial-start "
        "markers in the test section of the first pair against 40 trials, and 14-16 right-partner onsets against 16 notes "
        "per trial), so the authors' per-trial table is the behavioural record here.",
    ],
    unit_note="measure of one participant in one trial (three measures per trial)",
    differences=[
        "In the deposit the file of pair 2 (sub-02_task-jointaction_beh.tsv) codes ExperimentalCondition the other way round "
        "from its own tone sequences (0 for the duets, 1 for the constant pitch sequences) while behavioural_all.tsv codes it "
        "consistently with the README; the compiled table is used.",
        "The paper's text could not be read (the publisher blocks automated access), so the published checks are counts the deposit's "
        "README and OpenNeuro page give (32 pairs, 64 participants, 40 trials per pair); no statistic of the paper was reproduced.",
    ],
)


class Rows:
    def __iter__(self):
        return iter(rows)

    def __len__(self):
        return len(rows)


stdlib.release(ID, "summary", Rows(), meta, checks, participants=participants, participant_columns=participant_columns)
