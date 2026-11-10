"""Tirou, Abdoun, Vekony, Tosatto, Brovelli, Vernet, Nemeth & Quentin (2026), Nat Commun 17:8372: cued alternating serial reaction time
(cASRT) task during MEG, 15 healthy young adults. Only the behavioural tables of the OpenNeuro deposit are used, not the MEG or MRI.

Raw files in raw_extra/Tirou2026_NatCommun/ (OpenNeuro ds008017, CC0; the files were saved as <path with __ for />):
  participants.tsv / participants.json            counting span and Edinburgh laterality quotient per participant
  sub-XX__beh__sub-XX_task-practice_beh.tsv        3 practice blocks of 85 trials
  sub-XX__beh__sub-XX_task-asrt_run-0R_beh.tsv     4 runs of 5 learning blocks of 85 trials (the 20 learning blocks of the paper)
  sub-XX__beh__sub-XX_task-explicit_run-0R_beh.tsv sequence reports after each learning block (an anchor row, then 12 key presses)
  sub-15__meg__*_meg.json                          recording metadata (read for the sampling frequency only)
Columns of the asrt files: ID, scenario, block (1-5 within the run), proba (trial in the block), position (cue 1-4), trialtype
(0 warm-up, 1 pattern, 2 random), triplet (authors' code, undocumented), isi_if_correct, isi_if_incorrect, hiba (error flag),
talalat (hit flag), resp (key pressed), RT (ms; 0 without a response), stim_pres_time (MEG sample units). The practice files have
the same columns minus the two isi columns.
Run from the site folder: python3 -I tools/std/convert/Tirou2026_NatCommun.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Tirou2026_NatCommun"
SRC = "Tirou et al. 2026, Nature Communications"
RAW = stdlib.raw_dir(ID)
DIR = {1: "left", 2: "up", 3: "down", 4: "right"}
COLS_A = ["ID", "scenario", "block", "proba", "position", "trialtype", "triplet", "isi_c", "isi_i", "hiba", "talalat", "resp", "RT", "tpres"]
COLS_P = ["ID", "scenario", "block", "proba", "position", "trialtype", "triplet", "hiba", "talalat", "resp", "RT", "tpres", "x1", "x2"]
N_WARM, PER_BLOCK, CYCLE = 5, 85, 8


def read(sub, name, practice=False):
    f = RAW / f"sub-{sub:02d}__beh__sub-{sub:02d}_task-{name}_beh.tsv"
    return pd.read_csv(f, sep="\t", header=0, names=COLS_P if practice else COLS_A)


def main():
    checks = stdlib.Checks()
    part = pd.read_csv(RAW / "participants.tsv", sep="\t")
    subs = [int(x[4:]) for x in part.participant_id]
    rows, cell = [], []
    for sub in subs:
        order = 0
        p = read(sub, "practice", practice=True)
        a = {r: read(sub, f"asrt_run-0{r}") for r in range(1, 5)}
        e = {r: pd.read_csv(RAW / f"sub-{sub:02d}__beh__sub-{sub:02d}_task-explicit_run-0{r}_beh.tsv", sep="\t") for r in range(1, 5)}
        # the participant's pattern: positions of the pattern trials at the 4 pattern slots of every learning block
        pat = {}
        for r in a:
            for b, g in a[r].groupby("block"):
                g = g.reset_index(drop=True)
                for k in range(N_WARM, len(g)):
                    sp = (k - N_WARM) % CYCLE + 1
                    if sp % 2 == 0:
                        pat.setdefault(sp, set()).add(int(g.position[k]))
                    assert (g.trialtype[k] == (2 if sp % 2 == 1 else 1)) and g.trialtype[:N_WARM].eq(0).all()
        assert all(len(v) == 1 for v in pat.values()) and sorted(pat) == [2, 4, 6, 8], (sub, pat)
        cyc = [next(iter(pat[s])) for s in (2, 4, 6, 8)]
        succ = {cyc[i]: cyc[(i + 1) % 4] for i in range(4)}
        seq_text = "-".join(f"r-{x}" for x in cyc)

        def block_rows(g, label, block_no, block_type, epoch, pattern_defined, sess):
            nonlocal order
            g = g.reset_index(drop=True)
            pos = g.position.to_numpy()
            for k in range(len(g)):
                order += 1
                tt = int(g.trialtype[k])
                resp_ok = g.RT[k] > 0
                correct = bool(g.talalat[k] == 1)
                if k >= N_WARM + 2:
                    aa, bb, cc = pos[k - 2], pos[k - 1], pos[k]
                    if aa == bb == cc:
                        trip = "repetition"
                    elif aa == cc and aa != bb:
                        trip = "trill"
                    elif pattern_defined:
                        trip = "high" if succ[aa] == cc else "low"
                    else:
                        trip = "none"
                else:
                    trip = "none"
                excl = (k < N_WARM + 2) or trip in ("repetition", "trill") or not correct
                sp = (k - N_WARM) % CYCLE + 1 if k >= N_WARM else ""
                rep = (k - N_WARM) // CYCLE + 1 if k >= N_WARM else ""
                rows.append(dict(
                    dataset=ID, study=SRC, experiment=ID + "_Exp1", group="", subject=f"S{sub:02d}", person=f"S{sub:02d}",
                    subject_excluded_by_authors=False, age="", sex="", handedness="right" if part.set_index("participant_id").laterality_quotient[f"sub-{sub:02d}"] > 0 else "",
                    population="healthy", setting="lab",
                    device="two optical response keypads in the MEG room (left and up keys on the left pad, down and right on the right pad)",
                    session=1, block=f"{label} block {g.block[0]}" + (f" ({g.scenario[0]})" if label != "practice" else ""), trial=order,
                    phase="practice" if block_type == "practice" else "learning", time_s="", task="sequence learning", unit="keypress",
                    condition="", pert_type="none", pert_value=0, pert_unit="", feedback="normal", excluded_by_authors=bool(excl),
                    block_number=block_no, block_type=block_type, epoch=epoch, stimulus=int(pos[k]), correct_key=DIR[int(pos[k])],
                    key_pressed=DIR[int(g.resp[k])] if resp_ok else "", correct=correct, rt_ms=float(g.RT[k]) if resp_ok else "",
                    element_type={0: "warm-up", 1: "sequence", 2: "random"}[tt], sequence=seq_text if pattern_defined else "none (the pattern trials follow no sequence in practice)",
                    sequence_position=sp, repetition=rep, triplet_type=trip))
                cell.append((sub, block_no, block_type, tt, excl, correct, float(g.RT[k]) if resp_ok else np.nan, trip))

        for b, g in p.groupby("block"):
            block_rows(g, "practice", int(b), "practice", "", False, 1)
        for r in range(1, 5):
            for b, g in a[r].groupby("block"):
                block_rows(g, "learning", (r - 1) * 5 + int(b), "training", r, True, 1)
                rep = e[r][e[r].block_within_epoch == b]
                for i, x in enumerate(rep.itertuples(index=False)):
                    order += 1
                    rows.append(dict(
                        dataset=ID, study=SRC, experiment=ID + "_Exp1", group="", subject=f"S{sub:02d}", person=f"S{sub:02d}",
                        subject_excluded_by_authors=False, age="", sex="", handedness="right", population="healthy", setting="lab",
                        device="two optical response keypads in the MEG room (left and up keys on the left pad, down and right on the right pad)",
                        session=1, block=f"explicit report after learning block {g.block.iloc[0]} (eASRT_Epoch_{r})", trial=order, phase="test", time_s="",
                        task="sequence learning", unit="keypress",
                        condition="explicit sequence report" + (" (anchor response carried over from the task)" if x.response_idx == 0 else ""),
                        pert_type="none", pert_value=0, pert_unit="", feedback="none", excluded_by_authors=bool(x.response_idx == 0),
                        block_number=(r - 1) * 5 + int(b), block_type="test", epoch=r, key_pressed=DIR[int(x.key)]))
    t = pd.DataFrame(rows)
    c = pd.DataFrame(cell, columns=["sub", "blk", "btype", "tt", "excl", "correct", "rt", "trip"])

    # ---------------------------------------------------------------- published checks
    ps = part.set_index("participant_id")
    checks.published("participants (Methods, Participants: fifteen)", len(subs), 15, 0, f"{SRC}, Methods, Participants")
    checks.published("mean Edinburgh laterality quotient", ps.laterality_quotient.mean(), 76.0, 0.05, f"{SRC}, Methods, Participants (mLQ = 76.0)")
    checks.published("SD of the laterality quotient", ps.laterality_quotient.std(ddof=1), 22.8, 0.05, f"{SRC}, Methods, Participants (22.8)")
    checks.published("lowest laterality quotient", ps.laterality_quotient.min(), 33.3, 0.05, f"{SRC}, Methods, Participants (between +33.3 and +100)")
    checks.published("highest laterality quotient", ps.laterality_quotient.max(), 100, 0, f"{SRC}, Methods, Participants (between +33.3 and +100)")
    cs_note = (f"Methods, Participants: the mean counting span is 3.78 (SD 0.95) in the paper and {ps.counting_span.mean():.2f} (SD "
               f"{ps.counting_span.std(ddof=1):.2f}) from the deposit's participants table, which gives one decimal.")
    lb = c[c.btype == "training"]
    checks.published("learning blocks per participant", lb.groupby("sub").blk.nunique().median(), 20, 0, f"{SRC}, Methods, Behavioral task")
    checks.published("practice blocks per participant", c[c.btype == "practice"].groupby("sub").blk.nunique().median(), 3, 0, f"{SRC}, Methods, Behavioral task")
    checks.published("trials per block (85)", c.groupby(["sub", "btype", "blk"]).size().median(), 85, 0, f"{SRC}, Methods, Behavioral task")
    checks.published("trials in the learning session per participant (1700)", lb.groupby("sub").size().median(), 1700, 0, f"{SRC}, Methods, Behavioral task")
    ex = t[t.condition.str.startswith("explicit")]
    checks.published("sequence-report key presses after each block, besides the anchor (12)",
                     ex[~ex.condition.str.contains("anchor")].groupby(["person", "block"]).size().median(), 12, 0,
                     f"{SRC}, Methods, Behavioral task (12 consecutive responses)")
    checks.published("warm-up trials per block (five)", (c[c.btype == "training"].groupby(["sub", "blk"]).tt.apply(lambda s: (s == 0).sum())).median(), 5, 0,
                     f"{SRC}, Methods, Behavioral task")

    # main effect of trial condition (pattern versus random), 2 x 20 repeated-measures ANOVA on mean RT of correct responses
    d = lb[(~lb.excl) & lb.tt.isin([1, 2])]
    m = d.groupby(["sub", "blk", "tt"]).rt.mean().unstack("tt")
    diff = (m[2] - m[1]).groupby("sub").mean()
    tstat = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    F_cond = tstat ** 2
    print(f"condition main effect F(1,14) = {F_cond:.3f} (paper 19.77); cells per subject: {m.groupby('sub').size().unique()}")
    diffs = [cs_note]
    if abs(F_cond - 19.77) <= 0.005:
        checks.published("main effect of trial condition (pattern vs random), F(1,14)", F_cond, 19.77, 0.005, f"{SRC}, Results (F1,14 = 19.77)")
    else:
        diffs.append(f"Results, main effect of trial condition F(1,14) = 19.77: {F_cond:.2f} here from the mean RT of correct, non-trill, non-repetition "
                     "pattern and random trials of the 20 learning blocks; the exact pre-processing of the paper's ANOVA is not given in the deposit.")
    checks.internal("every participant has the same number of rows (practice, learning and sequence-report rows)",
                    bool((t.groupby("person").size() == t.groupby("person").size().iloc[0]).all()), f"{t.groupby('person').size().iloc[0]} rows each")
    checks.internal("hits have a response time and misses none (but for a few rows)",
                    bool(((t.correct == True) & (t.rt_ms == "")).sum() == 0), "")  # noqa: E712
    for ch in checks.items:
        print(("  ok " if ch["ok"] else "FAILED ") + ch.get("what", ""), ch.get("computed", ""), ch.get("reported", ""))

    s0 = json.load(open(stdlib.HERE / "source/extra_studies.json"))
    s = next(x for x in s0 if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        changes=("Used only the behavioural tables (practice, ASRT blocks and sequence reports) of the BIDS dataset and joined them in time order into one "
                 "table; converted the cue and key codes to direction names; added the cycle position, repetition and triplet type of each trial from the "
                 "positions; recoded participant numbers as codes. The MEG recordings, MRI and the authors' trigger codes are not included."),
        how=["One row is one key press (or one missed response) in the cued alternating serial reaction time (cASRT) task. Each of the 15 participants "
             "did 3 practice blocks (no sequence), then 20 learning blocks in 4 epochs (runs) of 5 blocks, each block 85 trials: 5 random warm-up trials, then "
             "ten repetitions of an 8-trial cycle in which pattern trials (yellow arrows, element_type sequence) alternate with random trials (red arrows). "
             "The cue is an arrow (stimulus 1-4 = left, up, down, right) and the answer is the matching key with the index or middle finger of either hand. "
             "After every learning block the participant typed the sequence they had found (12 key presses after an anchor row carried over from the task): "
             "these rows have block_type test and condition explicit sequence report; their first row of each report (the anchor) is marked excluded.",
             "correct is the authors' hit flag (talalat); rt_ms is the authors' RT in ms, empty on trials without a response (the authors' RT is 0 there) "
             "and key_pressed is empty too. sequence is the participant's pattern read from the pattern trials (the same in all learning blocks), "
             "sequence_position and repetition number the trials of each block after the warm-up, and triplet_type is derived from the positions in the "
             "standard way (high: the third cue follows the first in the pattern, low: not; trill and repetition as defined in the paper); the deposit's own "
             "triplet code is not documented. Practice blocks have no sequence, so they only have trill and repetition triplets.",
             "excluded_by_authors is true for trials the paper leaves out of its reaction time analysis (it keeps correct responses and drops stimulus repetitions "
             "and trills), and also for the warm-up trials and the first two trials of each block, which have no triplet.",
             "Not carried: the MEG-sample timestamps (the unit is given only as MEG sample units), the authors' two ISI columns and triplet codes, and the "
             "MEG and MRI data. The deposit has no age or sex."],
        unit_note="key press (one cued response or one sequence-report key press)", differences=diffs + [
            "The paper reports 10 female and 5 male participants with a mean age of 25.8 years; the deposit contains neither age nor sex, so they are empty here.",
            "The paper itself is licensed CC BY-NC-ND; the data deposit is CC0 and is shared under CC0 here."])
    stdlib.release(ID, "keypress", rows, meta, checks, participants=[
        dict(person=f"S{int(i[4:]):02d}", counting_span=r.counting_span, laterality_quotient=r.laterality_quotient) for i, r in ps.iterrows()],
        participant_columns=[("counting_span", "number", "score", "Counting span working-memory score (short form), as in the authors' participants table."),
                             ("laterality_quotient", "number", "percent", "Edinburgh Handedness Inventory laterality quotient (-100 fully left-handed, +100 fully right-handed).")])


main()
