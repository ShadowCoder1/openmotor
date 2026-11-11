"""Van Roy et al. (2025), J Neurosci: finger-tapping sequence learning in 42 drug-naive people with unilateral Parkinson's disease
(24 'PD-U' and 18 'PD-A' rows) and 30 controls; per-participant block-level source data of the paper's figures.
Raw file (untrusted data, read only): raw_extra/VanRoy2025_JNeurosci/VanRoy_MicroPD_FigureData.xlsx (Zenodo 15864553, CC BY 4.0).
Run: python3 -I tools/std/convert/VanRoy2025_JNeurosci.py"""
import sys
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "VanRoy2025_JNeurosci"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
wb = openpyxl.load_workbook(RAW / "VanRoy_MicroPD_FigureData.xlsx", read_only=True, data_only=True)
sheet = lambda n: [r for r in wb[n].iter_rows(values_only=True)]  # noqa: E731

t1 = sheet("Table1")
head1 = list(t1[0])
people = {}
for r in t1[1:]:
    if r[0] is None:
        continue
    people[int(r[0])] = dict(zip(head1, r))
grp = {k: v["Group"].strip() for k, v in people.items()}
POP = {"Ctrl": "healthy control", "PD-U": "Parkinson's disease (drug-naive, unilateral), group PD-U",
       "PD-A": "Parkinson's disease (drug-naive, unilateral), group PD-A"}

CORE_DEFAULT = dict(dataset=ID, study="Van Roy et al. 2025 J Neurosci", experiment="finger tapping sequence learning", group="",
                    subject="", person="", subject_excluded_by_authors=False, age=None, sex="", handedness="", population="",
                    setting="lab", device="", session=1, block="", trial=None, phase="learning", time_s=None,
                    task="sequence learning", unit="participant summary", condition="", pert_type="none", pert_value=0,
                    pert_unit="", feedback="normal", excluded_by_authors=False)
rows = []
counter = {}


def add(pid, measure, desc, value, unit, block="", session=1, phase="learning", stat="mean", n_trials=None):
    if value is None or isinstance(value, str):
        return
    p = people[pid]
    counter[pid] = counter.get(pid, 0) + 1
    r = dict(CORE_DEFAULT)
    r.update(group=grp[pid], subject=f"P{pid}", age=p["age at exam (years)"],
             sex="M" if str(p["gender"]).strip() == "male" else "F", population=POP[grp[pid]],
             session=session, block=block, trial=counter[pid],
             phase=phase)
    r.update(measure=measure, measure_description=desc, statistic=stat, value=value, value_unit=unit, n_trials=n_trials)
    rows.append(r)


def blocks(sheetname, measure, desc, unit, session=1, phase="learning"):
    s = sheet(sheetname)
    hi = [i for i, r in enumerate(s) if r[0] == "ID"][0]
    names = [str(h).strip() for h in s[hi]]
    for r in s[hi + 1:]:
        if r[0] is None:
            continue
        for j in range(2, len(r)):
            if names[j] in ("None", ""):
                continue
            add(int(r[0]), measure, desc, r[j], unit, block=names[j], session=session, phase=phase)


blocks("Fig 1A", "mean_sequence_duration", "Mean duration of the correctly performed sequences in the practice block (initial training session), as the authors' Fig 1A", "s")
blocks("Fig1B", "cumulative_total_gain", "Cumulative total performance gain up to this block, as the authors' Fig 1B (positive = faster)", "s")
blocks("Fig1C", "cumulative_micro_online_gain", "Cumulative micro-online gain up to this block (change within practice blocks), as the authors' Fig 1C", "s")
blocks("Fig1D", "cumulative_micro_offline_gain", "Cumulative micro-offline gain up to this block (change across rest intervals), as the authors' Fig 1D", "s")
# first / last correct sequence durations: columns 3-16 first sequence, 17-30 last sequence
s = sheet("SeqDur_FirstLastCorrectSeq")
hi = [i for i, r in enumerate(s) if r[0] == "ID"][0]
names = [str(h).strip() for h in s[hi]]
for r in s[hi + 1:]:
    if r[0] is None:
        continue
    for j in range(2, 30):
        which = "first" if j < 16 else "last"
        add(int(r[0]), f"duration_{which}_correct_sequence",
            f"Duration of the {which} correctly performed sequence of the practice block (initial training session)", r[j], "s",
            block=names[j], stat="value")
for r in sheet("IndicesFirstLastCorrectSeq")[2:]:
    if r[0] is None or r[1] is None:
        continue
    add(int(r[0]), "mean_index_first_key_of_first_correct_sequence", "Mean over practice blocks of the index of the first key press of the first correctly performed sequence", r[2], "key press index")
    add(int(r[0]), "mean_index_first_key_of_last_correct_sequence", "Mean over practice blocks of the index of the first key press of the last correctly performed sequence", r[3], "key press index")
for r in sheet("Fig2")[2:]:
    if r[0] is None or r[1] is None:
        continue
    for nm, v in zip(["total", "micro_online", "micro_offline"], r[2:5]):
        add(int(r[0]), f"final_cumulative_{nm}_gain", f"Final cumulative {nm.replace('_', '-')} performance gain over the initial training session (Fig 2)", v, "s", block="final", stat="value")
for r in sheet("Fig3")[2:]:
    if r[0] is None or r[1] is None:
        continue
    add(int(r[0]), "final_cumulative_micro_offline_gain_reactive_inhibition_corrected", "Final cumulative micro-offline gain after correcting for reactive inhibition (Fig 3)", r[2], "s", block="final", stat="value")
for r in sheet("FigS1")[2:]:
    if r[0] is None or r[1] is None:
        continue
    for nm, v in zip(["total", "micro_online", "micro_offline"], r[2:5]):
        add(int(r[0]), f"final_cumulative_{nm}_gain", f"Final cumulative {nm.replace('_', '-')} performance gain in the 24 hour retest session (Fig S1)", v, "s", block="final", session=2, phase="test", stat="value")

n_pd = sum(1 for v in grp.values() if v.startswith("PD"))
n_c = sum(1 for v in grp.values() if v == "Ctrl")
persons_in_rows = {r["subject"] for r in rows}
checks.published("people with Parkinson's disease", n_pd, 42, 0, "Abstract: 'Forty-two drug-naive individuals with ... unilateral PD'")
checks.published("healthy control participants", n_c, 30, 0, "Abstract: '30 healthy control subjects'")
checks.published("participants with data rows", len(persons_in_rows), 72, 0, "Abstract: 42 + 30 participants")
checks.internal("every participant has block 1 mean sequence duration", len({r["subject"] for r in rows if r["measure"] == "mean_sequence_duration" and r["block"] == "Block 1"}) == 72)
checks.internal("14 blocks in Fig 1A", len({r["block"] for r in rows if r["measure"] == "mean_sequence_duration"}) == 14)

part_cols = [("disease_duration_years", "number", "years", "Disease duration (Table1 sheet)."),
             ("education_years", "number", "years", "Education."), ("mmse", "number", "", "Mini-Mental State Examination."),
             ("gds", "number", "", "Geriatric Depression Scale."), ("hoehn_yahr", "number", "", "Hoehn and Yahr stage."),
             ("updrs1", "number", "", "MDS-UPDRS part 1."), ("updrs2", "number", "", "MDS-UPDRS part 2."),
             ("updrs3", "number", "", "MDS-UPDRS part 3."), ("updrs_left", "number", "", "UPDRS motor score, left side."),
             ("updrs_right", "number", "", "UPDRS motor score, right side.")]
keys = ["disease duration (years)", "Education (years)", "MMSE", "GDS", "H&Y ", "UPDRS1", "UPDRS2", "UPDRS3", "UPDRS - L", "UPDRS - R"]
parts = [dict(person=f"P{pid}", **{c[0]: p.get(k) for c, k in zip(part_cols, keys)}) for pid, p in people.items()]
# one cell of the UPDRS3 column holds the word "fracture" instead of a score: the score is left empty (no number is guessed)
n_text = 0
for _p in parts:
    for _c, _k in zip(part_cols, keys):
        _v = _p[_c[0]]
        if isinstance(_v, str):
            assert _c[0] == "updrs3" and _v.strip().lower() == "fracture", (_c[0], _v)
            _p[_c[0]] = None
            n_text += 1
assert n_text == 1, n_text
paper = dict(authors="Van Roy A, Dan X, Micca L, Gilat M, Chan P, Doyon J, Albouy G, King BR", year=2025,
             title="Impaired Online and Enhanced Offline Motor Sequence Learning in Individuals with Parkinson's Disease",
             venue="J Neurosci 45(35):e2193242025", doi="10.1523/jneurosci.2193-24.2025")
meta = dict(paper=paper, lab="King & Albouy (University of Utah)", data_url="https://doi.org/10.5281/zenodo.15864553",
            license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
            changes="Wide figure-data sheets were turned into long form (one row per participant, measure and block); nothing was recomputed.",
            how=["One row is one value for one participant: a block-level measure (mean sequence duration, cumulative total, micro-online and micro-offline gains, "
                 "duration of the first and last correct sequence of each of 14 practice blocks), or a whole-session value (final cumulative gains, 24 hour retest gains, "
                 "mean index of the first key of the first and last correct sequence). All durations and gains are in seconds; gains are positive when performance got faster.",
                 "Only these per-participant summaries are in the deposit; key presses are not. Group labels Ctrl, PD-U and PD-A are the deposit's; PD-U and PD-A are the "
                 "two Parkinson's subgroups (24 and 18 people, together the 42 reported). The deposit does not define the letters, so no meaning is assumed here.",
                 "Sex was coded F or M from the deposit's gender column; the table1 clinical scores are in participants.csv. Sequence and device details are not in the deposit "
                 "and are not stated in the rows."],
            unit_note="participant, measure and block (long form)",
            differences=["In the deposit's Table1 sheet, one participant's UPDRS3 cell holds the word \"fracture\" instead of a score. "
                         "That UPDRS3 value is left empty in participants.csv; no number was guessed."])
stdlib.release(ID, "summary", rows, meta, checks, participants=parts, participant_columns=part_cols)
