"""Bradshaw, Wheeler, McGettigan & Lametti (2024/2025), Psychon Bull Rev 32(1):306-316: speech motor adaptation to a formant
perturbation while speaking in synchrony with an accompanist voice whose formants either agree (Congruent) or disagree
(Incongruent) with the direction of adaptation. 50 sentences x 7 blocks = 350 trials per participant.

Raw files, raw_extra/Bradshaw2024_PsychonBullRev/ (OSF shv5u, data component of h26ur; fetched with tools/std/pc.py fetch):
  Convergence_data.zip   one csv per participant: Block (1 solo, 2 synchronous), F1, F2 (mel, per sentence), Group, Trial_number, Sentence
  Vector_of_compensation.zip  Sentence, Compensation, Block (3-7), Trial_number, Participant, Group
  Adaptation_data.zip    block 6: change_from_baseline (mel) of F1 and F2 versus block 2 (per sentence)
  After-effect_data.zip  block 7: the same
  Participants_list.csv  Participant, Group (1 Incongruent, 2 Congruent)
  Data Dictionary Adaptation and Vocal Convergence.pdf (OSF h26ur) defines the variables; Trial_number is the trial within the
  block on which that sentence was presented, Sentence the sentence identity (rows are sorted by Sentence in the files).
Feedback shift (paper, Methods, Apparatus): F1 up and F2 down by 49.5 mel each (joint 70 mel), ramped over the first 25 trials of
block 3, held to the end of block 6, removed in block 7.
Run from the site folder: python3 -I tools/std/convert/Bradshaw2024_PsychonBullRev.py
"""
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Bradshaw2024_PsychonBullRev"
SHIFT = 49.5
PAPER_SRC = "Bradshaw et al., Psychon Bull Rev 32(1):306-316"
GROUP = {1: "Incongruent", 2: "Congruent"}
DEVICE = "Shure WH20 head-mounted microphone, Sennheiser HD 280 Pro headphones, RME Babyface Pro, Audapter"
MASKING = "speech-shaped noise, 60 dB SPL"


def zip_csvs(raw, name):
    z = zipfile.ZipFile(raw / name)
    out = []
    for n in z.namelist():
        if n.endswith(".csv") and not n.startswith("__MACOSX"):
            out.append(pd.read_csv(io.BytesIO(z.read(n)), encoding="utf-8-sig"))
    return pd.concat(out, ignore_index=True)


def build(checks):
    raw = stdlib.raw_dir(ID)
    P = pd.read_csv(raw / "Participants_list.csv", encoding="utf-8-sig")
    C = zip_csvs(raw, "Convergence_data.zip").rename(columns={"Participant": "pid"})
    V = zip_csvs(raw, "Vector_of_compensation.zip").rename(columns={"Participant": "pid"})
    A = zip_csvs(raw, "Adaptation_data.zip").rename(columns={"participant": "pid"})
    E = zip_csvs(raw, "After-effect_data.zip").rename(columns={"participant": "pid"})
    for d in (C, V, A, E):
        d["Block"] = d["Block"] if "Block" in d else np.nan
    A["Block"], E["Block"] = 6, 7
    ch = pd.concat([A, E])
    chg = ch.pivot_table(index=["pid", "Block", "Sentence"], columns="Formant", values="change_from_baseline", dropna=False,
                         aggfunc="first").reset_index().rename(columns={"F1": "f1c", "F2": "f2c"})
    base = pd.concat([C.rename(columns={"F1": "f1", "F2": "f2"})[["pid", "Block", "Sentence", "Trial_number", "f1", "f2"]],
                      V[["pid", "Block", "Sentence", "Trial_number", "Compensation"]]], ignore_index=True)
    base["Block"] = base.Block.astype(int)
    d = base.merge(chg, on=["pid", "Block", "Sentence"], how="outer")
    assert len(d) == len(base) == 31 * 350
    d["group_no"] = d.pid.map(P.set_index("Participant").Group)
    d["gtrial"] = (d.Block - 1) * 50 + d.Trial_number
    d = d.sort_values(["pid", "gtrial"], kind="stable").reset_index(drop=True)
    rows = []
    for r in d.itertuples(index=False):
        b = int(r.Block)
        ramp = b == 3 and r.Trial_number <= 25
        shifted = 3 <= b <= 6
        val = SHIFT if (b in (4, 5, 6) or (b == 3 and not ramp)) else (0.0 if not shifted else "")
        nz = lambda x: "" if pd.isna(x) else float(x)
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Synchronous speech", "group": GROUP[r.group_no], "subject": r.pid,
            "subject_excluded_by_authors": False, "age": "", "sex": "F", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEVICE, "session": 1, "block": f"Block {b}", "trial": int(r.gtrial),
            "phase": {1: "baseline", 2: "baseline", 3: "learning", 4: "learning", 5: "learning", 6: "learning", 7: "aftereffect"}[b],
            "time_s": "", "task": "speech", "unit": "utterance", "condition": "solo" if b == 1 else "synchronous",
            "pert_type": "formant_shift" if shifted else "none", "pert_value": val, "pert_unit": "mel",
            "feedback": "altered" if shifted else "normal", "excluded_by_authors": False,
            "stimulus": f"sentence {int(r.Sentence)}", "segment": "whole sentence",
            "measure_window": "mean over the voiced portions of the sentence", "duration_ms": "", "f0_hz": "", "f1_hz": "",
            "f2_hz": "", "f1_mel": nz(r.f1), "intensity_au": "",
            "shift_f1": val, "shift_f2": (-val if val != "" else ""), "masking_noise": MASKING, "feedback_latency_ms": 12,
            "f2_mel": nz(r.f2), "f1_change_mel": nz(r.f1c), "f2_change_mel": nz(r.f2c), "compensation_mel": nz(r.Compensation),
            "_trial_in_block": int(r.Trial_number), "_sentence": int(r.Sentence),
        })
    return rows


def checks_all(a, checks):
    checks.internal("350 trials for each of 31 participants", a.groupby("subject").size().eq(350).all() and a.subject.nunique() == 31)
    checks.internal("blocks 1-2 hold formants, blocks 3-7 hold compensation, blocks 6-7 hold change from baseline",
                    bool(a[a.bn <= 2].f1_mel.notna().all() and a[a.bn >= 3].f1_mel.isna().all()
                         and a[a.bn >= 3].compensation_mel.notna().sum() == 31 * 250 - 1
                         and a[a.bn.isin([6, 7])].f1_change_mel.notna().sum() == 31 * 100 - 1))
    checks.internal("each block holds every sentence once per participant",
                    bool(a.groupby(["subject", "bn"])._sentence.nunique().eq(50).all()))
    checks.internal("Trial_number runs 1-50 within each block", bool(a._trial_in_block.between(1, 50).all()))
    g = a.drop_duplicates("subject").group.value_counts().to_dict()
    checks.published("participants analysed, Congruent group", g["Congruent"], 16, 0, f"{PAPER_SRC}, Methods, Participants")
    checks.published("participants analysed, Incongruent group", g["Incongruent"], 15, 0, f"{PAPER_SRC}, Methods, Participants")
    checks.published("participants analysed (total)", a.subject.nunique(), 31, 0, f"{PAPER_SRC}, Methods, Participants")
    checks.published("trials per participant (50 sentences x 7 blocks)", 350 * 0 + a.groupby("subject").size().iloc[0], 350, 0,
                     f"{PAPER_SRC}, Methods, Procedure")
    b1 = a[a.bn == 1]
    for grp, f1, f2 in (("Congruent", 623.89, 1368.31), ("Incongruent", 593.63, 1339.06)):
        x = b1[b1.group == grp].groupby("subject")[["f1_mel", "f2_mel"]].mean().mean()
        checks.published(f"{grp} group: mean baseline F1 (block 1, mel)", x.f1_mel, f1, 0.0051, f"{PAPER_SRC}, Methods, Stimuli")
        checks.published(f"{grp} group: mean baseline F2 (block 1, mel)", x.f2_mel, f2, 0.0051, f"{PAPER_SRC}, Methods, Stimuli")
    b6 = a[a.bn == 6].groupby(["subject", "group"])[["f1_change_mel", "f2_change_mel"]].mean().reset_index()
    for grp, col, t, n in (("Congruent", "f1_change_mel", -4.73, 16), ("Congruent", "f2_change_mel", 5.76, 16),
                           ("Incongruent", "f2_change_mel", 2.53, 15)):
        x = b6[b6.group == grp][col]
        checks.published(f"{grp}: one-sample t of block 2 to block 6 change in {col[:2].upper()} (n={n})",
                         stats.ttest_1samp(x, 0).statistic, t, 0.0051, f"{PAPER_SRC}, Results, Adaptation responses")
    c = a[a.bn <= 2].pivot_table(index=["subject", "group", "_sentence"], columns="bn", values=["f1_mel", "f2_mel"]).reset_index()
    for f, grp, beta in (("f1_mel", "Congruent", -7.55), ("f2_mel", "Congruent", 10.51)):
        x = c[c.group == grp]
        checks.published(f"{grp}: mean block 1 to block 2 change in {f[:2].upper()} (paper: mixed-model beta, mel)",
                         (x[(f, 2)] - x[(f, 1)]).mean(), beta, 0.0051, f"{PAPER_SRC}, Results, Convergence responses")


def main():
    checks = stdlib.Checks()
    rows = build(checks)
    a = pd.DataFrame(rows)
    a["bn"] = a.block.str[-1].astype(int)
    for c in ("f1_mel", "f2_mel", "f1_change_mel", "f2_change_mel", "compensation_mel"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    checks_all(a, checks)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Merged the authors' four per-participant file types (convergence, vector of compensation, block 6 and block 7 "
                 "change from baseline) into one table of 350 trials per participant, ordered by the trial on which each sentence "
                 "was presented, added the feedback shift from the paper's Methods, and replaced participant codes."),
        how=[
            "One row is one sentence read in synchrony with (or, in block 1, without) the accompanist voice by one participant; "
            "trial counts 1-350 over the seven blocks of 50 sentences, and stimulus is the authors' sentence number.",
            "Blocks 1 and 2 carry the produced F1 and F2 averaged over the sentence (f1_mel, f2_mel, mel). Blocks 3-7 carry compensation_mel, "
            "the authors' adaptation measure (component of the formant change from block 2 that opposes the shift, mel, positive = "
            "adaptive), and blocks 6 and 7 also carry f1_change_mel and f2_change_mel (change from the block 2 value of the same sentence).",
            "The feedback shift is F1 +49.5 mel and F2 -49.5 mel (pert_value is the F1 part); it is ramped over trials 1-25 of block 3 "
            "(values left empty because the paper does not give the step sizes), held in blocks 3 (trials 26-50) to 6, and removed in block 7.",
            "group is the accompanist voice: Congruent (formants moved the way adaptation requires) or Incongruent; the 10 people the "
            "authors excluded (equipment fault or baseline formants not related to the accompanist as intended) are not in the deposit.",
        ],
        unit_note="sentence read by one participant",
        differences=[
            "The shared files hold block-wise formants only for blocks 1 and 2; raw formants of blocks 3-7 are not in the deposit. "
            "One sentence of one participant has no value in block 7 (NA in the authors' file) and is left empty.",
        ],
    )
    for r in rows:
        r.pop("_trial_in_block"), r.pop("_sentence")
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
