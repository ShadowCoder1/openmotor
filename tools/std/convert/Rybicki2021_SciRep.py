"""Rybicki, Galea, Schuster, Hiles, Fabian & Cook (2021), Sci Rep 11:20693: probabilistic serial reaction time task in autistic and
non-autistic adults.

Raw files in raw_extra/Rybicki2021_SciRep/ (OSF s3gfd, folder 'data'; no license stated, the paper is CC BY 4.0):
  s<N>_1_r1.mat  training run of participant N: 5 blocks of 60 trials in which the four stimulus-key associations were learned
  s<N>_1_r2.mat  main run: 7 blocks of 100 trials (blocks 1, 4, 7 random; 2-3 easy and 5-6 difficult probabilistic sequences)
  prob_seq_autism.m  the authors' analysis script (read as text, never run): its ASD_group and pnums lists give the group of the 63
                     analysed participants.
Each .mat holds a struct 'mark' with Data(block).Results = [RT ms, outcome (1 correct, 2 error), stimulus 1-4, block, trial],
TM(block).TMs = transition matrix [to, from] of the block, triallist(block).trials = the generated stimulus sequence and Resp_keys.
Run from the site folder: python3 -I tools/std/convert/Rybicki2021_SciRep.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Rybicki2021_SciRep"
SRC = "Rybicki et al. 2021, Scientific Reports"
RAW = stdlib.raw_dir(ID)
KEYS = {1: "V", 2: "B", 3: "N", 4: "M"}
EASY_SEQ, HARD_SEQ = "1-2-3-4", "1-4-2-3"


def load(sid, run):
    f = RAW / f"s{sid}_1_{run}.mat"
    return sio.loadmat(f, squeeze_me=True, struct_as_record=False)["mark"] if f.exists() else None


def script_lists():
    txt = (RAW / "prob_seq_autism.m").read_text(errors="replace")

    def arr(name):
        m = re.search(name + r"\s*=\s*\[(.*?)\]", txt, re.S)
        return [int(x) for x in re.findall(r"-?\d+", m.group(1).split("%")[0])]
    return arr("pnums"), arr("ASD_group")


def block_kind(tm):
    if np.allclose(tm, 0.25):
        return "unpredictable"
    return "predictable-easy" if np.isclose(tm[1, 0], 0.75) else "predictable-difficult"


def main():
    checks = stdlib.Checks()
    src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
    pn, asd = script_lists()
    assert len(pn) == len(asd) == 63
    group = {p: ("autism" if g else "control") for p, g in zip(pn, asd)}
    files = sorted(RAW.glob("s*_1_r*.mat"))
    sids = sorted({int(re.match(r"s(\d+)_", f.name).group(1)) for f in files})
    rows = []
    keycodes = {}
    for sid in sids:
        order = 0
        grp = group.get(sid, "")
        for run in ("r1", "r2"):
            mk = load(sid, run)
            if mk is None:
                continue
            keycodes[tuple(int(x) for x in mk.Resp_keys)] = keycodes.get(tuple(int(x) for x in mk.Resp_keys), 0) + 1
            for b in range(len(mk.Data)):
                R = np.asarray(mk.Data[b].Results, dtype=float)
                tm = np.asarray(mk.TM[b].TMs, dtype=float)
                tr = np.asarray(mk.triallist[b].trials)
                assert (R[:, 2] == tr[:len(R)]).all() and (R[:, 3] == b + 1).all() and (R[:, 4] == np.arange(1, len(R) + 1)).all()
                kind = "practice" if run == "r1" else block_kind(tm)
                if run == "r2":
                    assert kind == ["unpredictable", "predictable-easy", "predictable-easy", "unpredictable", "predictable-difficult",
                                    "predictable-difficult", "unpredictable"][b], (sid, b, kind)
                seq = {"practice": "random order (key learning)", "unpredictable": "random order", "predictable-easy": EASY_SEQ + " likely",
                       "predictable-difficult": HARD_SEQ + " likely"}[kind]
                btype = {"practice": "practice", "unpredictable": "random"}.get(kind, "training")
                phase = "practice" if run == "r1" else ("baseline" if b == 0 else ("test" if kind == "unpredictable" else "learning"))
                for k in range(len(R)):
                    order += 1
                    prev = int(R[k - 1, 2]) if k > 0 else None
                    stim = int(R[k, 2])
                    err = R[k, 1] == 2
                    rows.append(dict(
                        dataset=ID, study=SRC, experiment=ID + "_Exp1", group=grp, subject=f"S{sid:03d}", person=f"S{sid:03d}",
                        subject_excluded_by_authors=sid not in group, age="", sex="", handedness="",
                        population="autism spectrum disorder" if grp == "autism" else ("healthy" if grp == "control" else ""),
                        setting="lab", device="keyboard (V, B, N, M keys; index, middle, ring and little finger)", session=1,
                        block=("practice " if run == "r1" else "") + str(b + 1), trial=order, phase=phase, time_s="",
                        task="sequence learning", unit="keypress", condition=kind, pert_type="none", pert_value=0, pert_unit="",
                        feedback="normal", excluded_by_authors=bool(run == "r1" or err),
                        block_number=b + 1 if run == "r2" else "", block_type=btype, stimulus=stim, correct_key=KEYS[stim],
                        correct=not err, rt_ms=R[k, 0], sequence=seq, sequence_position="",
                        transition_probability=tm[stim - 1, prev - 1] if prev else "", repetition="", run=run, tr=k + 1))
    for r in rows:
        r.pop("run"); r.pop("tr")
    t = pd.DataFrame(rows)

    # ---------------------------------------------------------------- the paper's numbers, from the converted rows
    t["ana"] = t.subject_excluded_by_authors == False  # noqa: E712
    m = t[(t.phase != "practice") & t.ana]
    cond_map = {"unpredictable": "unpredictable", "predictable-easy": "easy", "predictable-difficult": "difficult"}
    per = {}
    for c in cond_map:
        g = m[m.condition == c]
        per[c] = g.groupby("subject").apply(lambda h: pd.Series({
            "rt": h.rt_ms[h.correct].mean(), "ies": h.rt_ms[h.correct].mean() / h.correct.mean()}), include_groups=False)
    n63 = m.subject.nunique()
    checks.published("participants analysed (28 autistic + 35 non-autistic)", n63, 63, 0, f"{SRC}, Abstract and Methods, Participants")
    grp = m.groupby("subject").group.first().value_counts()
    checks.published("autistic participants", grp["autism"], 28, 0, f"{SRC}, Methods, Participants")
    checks.published("non-autistic participants", grp["control"], 35, 0, f"{SRC}, Methods, Participants")
    checks.published("blocks of 100 trials per participant (seven blocks, 700 trials)", m.groupby("subject").size().median(), 700, 0,
                     f"{SRC}, Methods, Serial reaction time task")
    T = {"easy": (660.350, 11.438, 685.502), "difficult": (697.318, 12.062, 734.790), "unpredictable": (687.624, 11.316, 719.985)}
    for c, nm in cond_map.items():
        d = per[c]
        mean, se, ies = d.rt.mean(), d.rt.std(ddof=1) / np.sqrt(len(d)), d.ies.mean()
        checks.published(f"mean RT, {nm} condition, ms", mean, T[nm][0], 0.0005, f"{SRC}, Results, sequence learning (RM-ANOVA post-hoc means)")
        checks.published(f"standard error of mean RT, {nm} condition, ms", se, T[nm][1], 0.0005, f"{SRC}, Results, sequence learning")
        checks.published(f"mean inverse efficiency score (RT / proportion correct), {nm} condition, ms", ies, T[nm][2], 0.0005,
                         f"{SRC}, Results, Inverse Efficiency Scores")
    checks.internal("69 participants with a main-run file (63 analysed, 6 not in the authors' script)", t[t.phase != "practice"].subject.nunique() == 69,
                    f"{t[t.phase != 'practice'].subject.nunique()}")
    checks.internal("every main run has 7 blocks of 100 trials", bool(t[t.phase != "practice"].groupby("subject").size().eq(700).all()))
    checks.internal("outcome codes are only 1 and 2 and stimuli 1-4", bool(t.stimulus.isin([1, 2, 3, 4]).all()))
    print(f"error trials with a missing RT: {int(t.rt_ms.isna().sum())} of {int((~t.correct).sum())}; negative RTs: {int((t.rt_ms < 0).sum())}")
    print("response-key codes in the files:", keycodes)
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    t = t.drop(columns="ana")

    s0 = json.load(open(stdlib.HERE / "source/extra_studies.json"))
    s = next(x for x in s0 if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the per-participant MATLAB files (training run, then main run) into one table, one row per key press, added the "
                 "participant's group from the authors' analysis script, the transition probability of each trial from the block's transition "
                 "matrix, and recoded participant numbers as codes."),
        how=["One row is one key press in a probabilistic serial reaction time task: a cue (one of four stimuli) appeared and the participant "
             "pressed the matching key (V, B, N, M with index, middle, ring and little finger; stimulus 1-4 in that order, taken from the "
             "order of the paper's description). The files hold two different response-key code lists (100 files one, 36 another, probably two keyboard "
             "or software settings); correct_key follows the paper for all. Each participant did a training run of 5 blocks of 60 trials (phase practice, "
             "stimulus-key association learning, not analysed) and then 7 blocks of 100 trials (block_number 1-7) in the same order for all: "
             "random, easy sequence, easy, random, difficult, difficult, random.",
             "condition is unpredictable (all four stimuli equally likely, transition probability 0.25), predictable-easy (1-2-3-4 is the likely "
             "order, each likely successor with probability 0.75 to 0.8) or predictable-difficult (1-4-2-3 likely, 0.76 to 0.85). "
             "transition_probability is the probability of the cued stimulus after the previous one in the authors' transition matrix; "
             "surprising trials are the low-probability transitions. It is empty on the first trial of each block.",
             "rt_ms is the authors' reaction time from stimulus onset (ms), kept as shared: on error trials it is missing in some files and "
             "present in others, and a few values are negative. correct is the authors' outcome code (1 correct, 2 error). "
             "excluded_by_authors is true for error trials (the paper's RT analyses use correct responses only) and for the training run; "
             "the inverse efficiency scores use all trials.",
             "group is autism or control, read from the authors' analysis script. Eight further participants have files but are not in the script "
             "(and so not in the paper's 63): their group is empty and subject_excluded_by_authors is true. Age, sex and other demographics are "
             "not in the deposit."],
        unit_note="key press (one cued response)", differences=[
            "The deposit's file names give participant numbers, not the paper's Table 1; demographics (age, sex, IQ, ADOS) cannot be linked.",
            "The key assigned to each stimulus (V, B, N, M) follows the paper's finger order; the deposit stores numeric key codes in two different "
            "schemes and does not name the keys in words, so the letters are not verifiable from the files."])
    stdlib.release(ID, "keypress", rows, meta, checks)


main()
