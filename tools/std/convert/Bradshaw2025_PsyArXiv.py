"""Bradshaw, Black, Gaultier & Davis (2025), PsyArXiv jr4bp (v1 10.31234/osf.io/jr4bp_v1, v2 of 2025-12-16): speech motor adaptation
to a two-formant shift while hearing one's own voice noise-vocoded in real time (cochlear-implant simulation). 32 hearing adults,
7 blocks x 50 sentences; block 1 clear feedback, block 2 vocoded baseline, block 3 ramp (first 25 trials) of an F1/F2 shift of
49.5 mel each, blocks 4-6 hold, block 7 shift removed. Group A = F1 up / F2 down, group B = F1 down / F2 up.

Raw files, raw_extra/Bradshaw2025_PsyArXiv/ (OSF ed9zq, fetched with tools/std/pc.py fetch; zips are read in memory):
  Participants_list.csv      Participant, Group (A, B), Training (1, 2: the phase 1 training task, perception or production)
  Baseline_data.zip          Convergence_<id>.csv: Block (1, 2), F1, F2 (mel, per sentence), Group, Phase1_Group, Participant, Sentence
  Vector_of_compensation.zip Compensation_<id>.csv: Sentence, Compensation, Block (3-7), Participant, Group
  Adaptation_data.zip / After-effect_data.zip  block 6 / block 7: Sentence, Group, Formant, change_from_baseline, participant
  F1-F2plus.zip, F1plusF2-.zip  phase 1 perceptual report tests (typed sentence reports and scores): not movement data, not carried.
Sentence is the authors' sentence column; the files give no within-block presentation order.
Run from the site folder: python3 -I tools/std/convert/Bradshaw2025_PsyArXiv.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Bradshaw2025_PsyArXiv"
SHIFT = 49.5
PAPER_SRC = "Bradshaw et al. 2025, PsyArXiv jr4bp v2 (OSF preprint)"
GROUP = {"A": "F1+F2- (F1 up, F2 down)", "B": "F1-F2+ (F1 down, F2 up)"}
DEVICE = "Shure WH20 head-mounted microphone, Beyerdynamic DT 770 PRO headphones, RME Fireface UCX, Audapter (noise-vocoding build)"
MASKING = "pink noise, 65 dB"


def zip_csvs(raw, name):
    z = zipfile.ZipFile(raw / name)
    out = []
    for n in sorted(z.namelist()):
        if n.endswith(".csv") and not n.startswith("__MACOSX"):
            out.append(pd.read_csv(io.BytesIO(z.read(n)), encoding="utf-8-sig"))
    return pd.concat(out, ignore_index=True)


def load(raw):
    P = pd.read_csv(raw / "Participants_list.csv", encoding="utf-8-sig")
    C = zip_csvs(raw, "Baseline_data.zip").rename(columns={"Participant": "pid", "F1": "f1", "F2": "f2"})
    V = zip_csvs(raw, "Vector_of_compensation.zip").rename(columns={"Participant": "pid"})
    A = zip_csvs(raw, "Adaptation_data.zip").rename(columns={"participant": "pid"})
    E = zip_csvs(raw, "After-effect_data.zip").rename(columns={"participant": "pid"})
    A["Block"], E["Block"] = 6, 7
    return P, C, V, A, E


def build(checks):
    raw = stdlib.raw_dir(ID)
    P, C, V, A, E = load(raw)
    chg = pd.concat([A, E]).pivot_table(index=["pid", "Block", "Sentence"], columns="Formant", values="change_from_baseline",
                                         aggfunc="first").reset_index().rename(columns={"F1": "f1c", "F2": "f2c"})
    base = pd.concat([C[["pid", "Block", "Sentence", "f1", "f2"]], V[["pid", "Block", "Sentence", "Compensation"]]], ignore_index=True)
    base["Block"] = base.Block.astype(int)
    d = base.merge(chg, on=["pid", "Block", "Sentence"], how="outer", validate="1:1", indicator=True)
    assert (d._merge != "right_only").all()
    pl = P.set_index("Participant")
    d["grp"] = d.pid.map(pl.Group)
    d["training"] = d.pid.map(pl.Training)
    # group letters in every file agree with the participant list
    for df in (C, V):
        assert (df.Group == df.pid.map(pl.Group)).all()
    assert (C.Phase1_Group == C.pid.map(pl.Training)).all()
    d["pn"] = d.pid.str[2:].astype(int)
    d["gtrial"] = (d.Block - 1) * 50 + d.Sentence
    d = d.sort_values(["pn", "gtrial"], kind="stable").reset_index(drop=True)
    nz = lambda x: "" if pd.isna(x) else float(x)
    rows = []
    for r in d.itertuples(index=False):
        b = int(r.Block)
        shifted = 3 <= b <= 6
        hold = b in (4, 5, 6)
        sign = 1 if r.grp == "A" else -1
        val = sign * SHIFT if hold else (0.0 if not shifted else "")  # block 3 is the ramp: step sizes not in the deposit
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Vocoded feedback adaptation", "group": GROUP[r.grp] + f"; training {int(r.training)}",
            "subject": r.pid, "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEVICE, "session": 1, "block": f"Block {b}", "trial": int(r.gtrial),
            "phase": {1: "baseline", 2: "baseline", 3: "learning", 4: "learning", 5: "learning", 6: "learning", 7: "aftereffect"}[b],
            "time_s": "", "task": "speech", "unit": "utterance",
            "condition": "clear feedback" if b == 1 else "noise-vocoded feedback",
            "pert_type": "formant_shift" if shifted else "none", "pert_value": val, "pert_unit": "mel",
            "feedback": "normal" if b == 1 else "altered", "excluded_by_authors": False,
            "stimulus": f"sentence {int(r.Sentence)}", "segment": "whole sentence",
            "measure_window": "mean over the voiced portions of the sentence (clear-speech recording)",
            "f1_mel": nz(r.f1), "shift_f1": val, "shift_f2": (-val if val != "" else ""), "masking_noise": MASKING,
            "feedback_latency_ms": 20, "f2_mel": nz(r.f2), "f1_change_mel": nz(r.f1c), "f2_change_mel": nz(r.f2c),
            "compensation_mel": nz(r.Compensation),
            "_sentence": int(r.Sentence), "_training": int(r.training), "_grp": r.grp,
        })
    return rows


def welch(x, y):
    t = stats.ttest_ind(x, y, equal_var=False)
    return t.statistic, t.df


def checks_all(a, checks):
    n = a.subject.nunique()
    n7 = a[a.bn == 7].groupby("subject").compensation_mel.count()
    checks.internal("block 7 has no values for exactly two participants, as the paper says (technical issue)",
                    int((n7 == 0).sum()) == 2 and int((n7 == 50).sum()) == 30 and bool(a.groupby("subject").size().eq(350).all()))
    checks.internal("each block holds each sentence once per participant (where the block exists)",
                    bool(a.groupby(["subject", "bn"])._sentence.nunique().eq(50).all()))
    checks.internal("blocks 1-2 hold formants, blocks 3-7 hold compensation, blocks 6-7 hold change from baseline",
                    bool(a[a.bn <= 2].f1_mel.notna().all() and a[a.bn >= 3].f1_mel.isna().all() and a[a.bn.between(3, 6)].compensation_mel.notna().all()
                         and a[a.bn == 6].f1_change_mel.notna().all() and a[a.bn == 6].f2_change_mel.notna().all()
                         and a[a.bn == 7].f1_change_mel.notna().sum() == 1500 and a[a.bn <= 5].f1_change_mel.isna().all()))
    one = a.drop_duplicates("subject")
    checks.published("participants analysed", n, 32, 0, f"{PAPER_SRC}, Methods, Participants")
    checks.published("participants per training group, training 1", int((one._training == 1).sum()), 16, 0, f"{PAPER_SRC}, Methods, Procedure and Apparatus")
    checks.published("participants per training group, training 2", int((one._training == 2).sum()), 16, 0, f"{PAPER_SRC}, Methods, Procedure and Apparatus")
    checks.published("participants in the F1+F2- group (sum of Table I column)", int((one._grp == "A").sum()), 16, 0, f"{PAPER_SRC}, Results, Table I")
    checks.published("participants in the F1-F2+ group (sum of Table I column)", int((one._grp == "B").sum()), 16, 0, f"{PAPER_SRC}, Results, Table I")
    checks.published("trials per participant with all seven blocks (7 blocks of 50 sentences)", int(a.groupby("subject").size().max()), 350, 0,
                     f"{PAPER_SRC}, Methods, Procedure and Apparatus")
    # block 1 -> block 2 paired t-tests over participant x sentence pairs (t(1599))
    c = a[a.bn <= 2].pivot_table(index=["subject", "_sentence"], columns="bn", values=["f1_mel", "f2_mel"])
    for f, rep, tol in (("f1_mel", 13.59, 0.0051), ("f2_mel", 1.9982, 0.00006)):
        t = stats.ttest_rel(c[(f, 2)], c[(f, 1)])
        checks.published(f"paired t, block 2 vs block 1, {f[:2].upper()} (df {int(t.df)})", t.statistic, rep, tol,
                         f"{PAPER_SRC}, Results, Baseline: Effect of noise vocoding on speech formants")
        if f == "f1_mel":
            checks.published("degrees of freedom of that t-test (32 participants x 50 sentences - 1)", t.df, 1599, 0, f"{PAPER_SRC}, same section")
    # Table I: per participant, two-sided one-sample t-test of block 6 compensation against 0
    b6 = a[a.bn == 6]
    cls = {}
    for s, g in b6.groupby("subject"):
        t = stats.ttest_1samp(g.compensation_mel, 0)
        cls[s] = "adapt" if t.pvalue < .05 and t.statistic > 0 else ("follow" if t.pvalue < .05 else "none")
    tab = pd.Series(cls).to_frame("c").join(one.set_index("subject")._grp)
    for grp, nm in (("A", "F1+F2-"), ("B", "F1-F2+")):
        for k, lab, rep in (("adapt", "significant adaptation", {"A": 2, "B": 4}), ("follow", "significant following", {"A": 1, "B": 4}),
                            ("none", "no significant change", {"A": 13, "B": 8})):
            checks.published(f"Table I, {nm} group: participants with {lab} (block 6)", int(((tab.c == k) & (tab._grp == grp)).sum()),
                             rep[grp], 0, f"{PAPER_SRC}, Results, Table I")
    # training groups: block 6 vector of adaptation (participant means), Welch t
    pm = b6.groupby(["subject", "_training"]).compensation_mel.mean().reset_index()
    t, df = welch(pm[pm._training == 1].compensation_mel, pm[pm._training == 2].compensation_mel)
    checks.published("training groups, block 6 vector of adaptation, Welch t (code 1 minus code 2)", t, -1.74, 0.0051,
                     f"{PAPER_SRC}, Results, Phase 2 (exploratory comparison of training groups)")
    checks.published("degrees of freedom of that Welch t-test", df, 26.71, 0.0051, f"{PAPER_SRC}, same section")
    return t, df, pm


def main():
    checks = stdlib.Checks()
    rows = build(checks)
    a = pd.DataFrame(rows)
    a["bn"] = a.block.str[-1].astype(int)
    for c in ("f1_mel", "f2_mel", "f1_change_mel", "f2_change_mel", "compensation_mel"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    t, df, pm = checks_all(a, checks)
    print("training 1 vs 2 Welch t on block 6 participant means:", t, df)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    paper["venue"] = "PsyArXiv preprint"
    paper["doi"] = "10.31234/osf.io/jr4bp_v2"
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Merged the authors' four per-participant file types (baseline formants of blocks 1-2, vector of compensation of "
                 "blocks 3-7, block 6 and block 7 change from baseline) into one table ordered by block and sentence, added the "
                 "feedback shift from the paper's Methods, left out the phase 1 perceptual report tests, and replaced participant codes."),
        how=[
            "One row is one sentence read aloud by one participant while hearing their own voice (clear in block 1, noise-vocoded in "
            "blocks 2-7, with the formant shift in blocks 3-6); trial counts 1-350 as (block - 1) x 50 + the authors' sentence number, "
            "because the files do not give the within-block presentation order. stimulus is the authors' sentence number.",
            "Blocks 1 and 2 carry the produced F1 and F2 averaged over the sentence (f1_mel, f2_mel, mel). Blocks 3-7 carry compensation_mel, "
            "the authors' vector of adaptation (change from baseline projected on the direction that opposes the shift, mel, positive = "
            "opposing), and blocks 6 and 7 also carry f1_change_mel and f2_change_mel, the change from baseline of the same sentence.",
            "The shift is F1 and F2 each 49.5 mel (70 mel combined) in opposite directions: group A F1 up and F2 down, group B F1 down and "
            "F2 up (pert_value is the F1 part). It ramps over the first 25 trials of block 3 (pert_value left empty, step sizes not given), "
            "is held in blocks 4-6, and is removed in block 7. group also gives the phase 1 training task as the authors' code 1 or 2.",
            "The deposit does not say which training code is perception training and which is production training; the paper's perception-versus-production comparison of block 6 adaptation (t = -1.74) reproduces with code 1 first, which suggests code 1 = perception. Two participants have no block 7 values (left empty).",
        ],
        unit_note="sentence read by one participant",
        differences=[
            "Baseline for the block 6 and 7 change values: the paper's Methods say the production change is normed to block 1 in one place and "
            "to block 2 in another; the values are kept as the authors shared them.",
            "The raw formants of blocks 3-7 are not in the deposit (only the compensation and change values). "
            "The deposit's phase 1 perceptual tests (typed sentence reports and token-sort scores) are not carried because they are not movement data.",
            "The preprint exists in two versions (v1 2025-10-15, v2 2025-12-16); numbers were checked against v2. It had not been published in a journal when checked (Crossref, 2026-11-09).",
        ],
    )
    for r in rows:
        r.pop("_sentence"), r.pop("_training"), r.pop("_grp")
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
