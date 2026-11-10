"""Ali, Chatburn & Immink (2024), Psychol Res 89:1: serial reaction time task with and without error feedback, 30 participants,
one row per key press (16 blocks x 180 trials).

Run from the site folder: python3 -I tools/std/convert/Ali2025_PsycholRes.py
Raw file (untrusted data, read only): raw_extra/Ali2025_PsycholRes/srtt.csv ('srtt accuracy rt.csv' from https://osf.io/bdcrk/).
Expected values for the published checks come from the paper's abstract and from the rendered output of the authors' analysis
document in the deposit ('Analysis and visualisation code.pdf', read as text, never run).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Ali2025_PsycholRes"
RAWFILE = stdlib.raw_dir(ID) / "srtt.csv"
checks = stdlib.Checks()

src = {r["file"]: r for r in json.load(open(stdlib.raw_dir(ID) / "SOURCES.json"))}
checks.internal("raw file matches the downloaded hash", stdlib.sha256(RAWFILE) == src["srtt.csv"]["sha256"])
raw = pd.read_csv(RAWFILE, na_values=[""], keep_default_na=False)
n = len(raw)
subs = raw["sub"].unique()
codes = {s: f"{ID}_S{i + 1:03d}" for i, s in enumerate(subs)}

# ---------------------------------------------------------------- internal checks
per = raw.groupby("sub").size()
checks.internal("30 participants x 16 blocks x 180 trials = 86400 rows", len(subs) == 30 and bool((per == 2880).all()) and n == 86400)
checks.internal("participants are contiguous and rows are in trial order within a participant",
                bool((raw["sub"].ne(raw["sub"].shift()).cumsum().groupby(raw["sub"]).nunique() == 1).all()))
pos = raw.groupby("sub").cumcount().to_numpy()
checks.internal("block = 1 + position // 180 on every row", bool((raw.block.to_numpy() == pos // 180 + 1).all()))
checks.internal("trial counts from 1 to 180 within each block", bool((raw.trial.to_numpy() == pos % 180 + 1).all()))
stimnum = raw.stim.str[0].astype(int)
keyfor = raw.groupby(stimnum).correct_key.agg(lambda x: sorted(set(x))).to_dict()
checks.internal("stimulus 1-4 map one to one to keys v, b, n, m", keyfor == {1: ["v"], 2: ["b"], 3: ["n"], 4: ["m"]}, str(keyfor))
acc = raw.response_accuracy.to_numpy() == 1
checks.internal("response_accuracy == (response_key == correct_key) on every row",
                bool((acc == (raw.response_key == raw.correct_key).to_numpy()).all()))
checks.internal("blocks 1-15 are the target sequence, block 16 the interference sequence",
                bool((raw.phase.eq("TargetSequence") == (raw.block < 16)).all()))
seqs = raw.assign(s=stimnum.astype(str)).groupby(["sub", "block"]).s.agg("".join)
tgt = {x[:12] for x in seqs[seqs.index.get_level_values(1) < 16]}
inter = {x[:12] for x in seqs[seqs.index.get_level_values(1) == 16]}
checks.internal("one target sequence (121342314324) and one interference sequence (124313214234) for everyone",
                tgt == {"121342314324"} and inter == {"124313214234"}, f"{tgt} {inter}")
cyc_ok = all(x == (x[:12] * 15) for x in seqs)
checks.internal("every block is 15 repetitions of its 12-element sequence", cyc_ok)
seqpos = raw.sequence_pos.to_numpy()
tb = raw.block.to_numpy() < 16
checks.internal("authors' sequence_pos = trial position mod 12 in blocks 1-15, empty in block 16",
                bool((seqpos[tb] == (raw.trial.to_numpy()[tb] - 1) % 12 + 1).all() and np.isnan(seqpos[~tb]).all()))
checks.internal("authors' blk_seq_cycle = repetition within block",
                bool((raw.blk_seq_cycle.to_numpy() == (raw.trial.to_numpy() - 1) // 12 + 1).all()))
checks.internal("rt range", True, f"{raw.response_rt.min()}-{raw.response_rt.max()} ms; {(raw.response_rt <= 0).sum()} rows <= 0")
checks.internal("errors: {:.2f}% of rows".format(100 * (~acc).mean()), True)

# ---------------------------------------------------------------- authors' RT outlier rule (analysis document)
rt = raw.response_rt.to_numpy()
cr = pd.Series(rt[acc])
q1, q3 = cr.quantile([0.25, 0.75])
lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
keep_rt = acc & (rt > lower) & (rt < upper)
excluded = acc & ~keep_rt          # correct trials removed from the RT analysis; errors stay (they enter the error analysis)

t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1", "group": raw.group.to_numpy(),
    "subject": raw["sub"].map(codes).to_numpy(), "subject_excluded_by_authors": False,
    "age": raw.age.to_numpy(), "sex": raw.sex.map({"female": "F", "male": "M"}).to_numpy(),
    "handedness": raw.hand.map({"right": "right", "left": "left", "mixed": "ambi"}).to_numpy(),
    "population": "healthy", "setting": "lab", "device": "keyboard (V, B, N, M keys)", "session": 1,
    "block": raw.block.astype(str).to_numpy(), "trial": pos + 1,
    "phase": np.where(tb, "learning", "test"), "time_s": "", "task": "sequence learning", "unit": "keypress",
    "condition": np.where(tb, "target sequence", "interference sequence"),
    "pert_type": "none", "pert_value": 0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": excluded,
    "block_number": raw.block.to_numpy(), "block_type": np.where(tb, "training", "test"), "epoch": "",
    "stimulus": stimnum.to_numpy(), "correct_key": raw.correct_key.to_numpy(), "key_pressed": raw.response_key.to_numpy(),
    "correct": acc, "rt_ms": rt, "rt_correct_ms": "", "interval_ms": "", "element_type": "sequence",
    "sequence": np.where(tb, "121342314324", "124313214234"),
    "sequence_position": pd.Series(seqpos, dtype=object).where(tb, None).map(lambda v: None if v is None else int(v)).to_numpy(),
    "repetition": raw.blk_seq_cycle.to_numpy(), "triplet_type": "none",
})
checks.internal("every row kept, in the authors' order", len(t) == n)
checks.internal("correct trials excluded by the IQR rule", True, f"{excluded.sum()} of {acc.sum()}; bounds {lower:.1f}, {upper:.1f} ms")

# ---------------------------------------------------------------- published checks
P = raw.drop_duplicates("sub")
SRC_A = "Abstract (Thirty participants, 14 females, Mage = 21.9 +/- 1.8 years)"
checks.published("number of participants", len(subs), 30, 0, SRC_A)
checks.published("number of female participants", int((P.sex == "female").sum()), 14, 0, SRC_A)
checks.published("mean age (years)", P.age.mean(), 21.9, 0.05, SRC_A)
checks.published("SD of age (years)", P.age.std(ddof=1), 1.8, 0.05, SRC_A)
DOC = "authors' analysis document (deposit, 'Analysis and visualisation code.pdf'), output of the "
pe = (1 - raw.assign(a=acc.astype(float)).groupby(["sub", "block"]).a.mean()) * 100
checks.published("mean percent error per participant and block (%)", pe.mean(), 3.98, 0.005, DOC + "percent error chunk (3.98)")
checks.published("SD of percent error per participant and block (%)", pe.std(ddof=1), 3.16, 0.005, DOC + "percent error chunk (3.16)")
checks.published("correct trials left after the 1.5 IQR RT outlier rule", int(keep_rt.sum()), 79995, 0, DOC + "outlier chunk (nrow(d.rt2) = 79995)")
checks.published("upper RT outlier bound (ms)", upper, 599.5, 0.05, DOC + "outlier chunk (Upper = 599.5)")
checks.published("mean RT of retained correct trials (ms)", rt[keep_rt].mean(), 293, 0.5, DOC + "RT table chunk (293, printed to 3 digits)")
checks.published("SD of RT of retained correct trials (ms)", rt[keep_rt].std(ddof=1), 111, 0.5, DOC + "RT table chunk (111, printed to 3 digits)")
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))

meta = dict(
    study="Ali et al. 2024",
    paper=dict(authors="Ali H, Chatburn A, Immink MA", year=2024,
               title="Post-error slowing during motor sequence learning under extrinsic and intrinsic error feedback conditions",
               venue="Psychol Res 89(1):1", doi="10.1007/s00426-024-02037-5"),
    lab="Immink (Flinders University)", data_url="https://osf.io/bdcrk/",
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes=("Converted to the OpenMotor format with participant numbers replaced by codes. The authors' derived tables "
             "(error-lag post-error slowing, improvement slopes, sequence-specificity, recognition and sequence-generation "
             "tests, participant characteristics) were not converted; they are computed from, or are questionnaire/awareness "
             "data beside, these trials. The authors' running_trial and running_seq_cycle columns (running counters over "
             "blocks 1-15) were dropped as redundant."),
    how=[
        "One row is one key press in a serial reaction time task: four stimuli (cued positions 1-4, answered with the "
        "keys V, B, N, M) in 16 blocks of 180 trials. Blocks 1-15 repeat a 12-element sequence (121342314324) fifteen "
        "times; block 16 repeats a different 12-element interference sequence (124313214234). Group EFB saw extrinsic "
        "error feedback, NoEFB did not (the authors' labels); feedback is 'normal' for both because the column is for "
        "sensory feedback.",
        "rt_ms is the authors' response_rt in ms and correct their response_accuracy (the first press). Rows keep the authors' "
        "order. sequence_position and repetition are the authors' sequence_pos and blk_seq_cycle (position empty in block 16, "
        "where the authors give none; repetition counts the 15 cycles).",
        "excluded_by_authors is true for correct trials that the authors removed from the reaction time analysis with a 1.5 x "
        "interquartile range rule over all correct trials (here between -4.5 and 599.5 ms); error trials are not flagged "
        "because the authors keep them in the error analysis. Their separate post-error-trial cleaning is not applied. "
        "Handedness 'mixed' is coded ambi.",
    ],
    unit_note="key press (one cued response in the SRT task)",
)
for c in [c[1] for c in stdlib.columns_for("keypress")]:
    if c not in t.columns and c != "person":
        t[c] = ""
stdlib.release(ID, "keypress", t, meta, checks)
