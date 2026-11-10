"""Laventure et al. (2016), PLoS Biol 14(3):e1002429: sequential finger tapping before and after a night of sleep with odor cueing.

Run from the site folder: python3 -I tools/std/convert/Laventure2016_PLoSBiol.py
Raw file (untrusted data, read only): raw_extra/Laventure2016_PLoSBiol/Data_S1.xlsx (Dryad doi:10.5061/dryad.b4t60, CC0, same
file on Zenodo record 5013749). Sheets used: 'Tab1 + Fig2A&B' (global performance index per block), 'FigS2A' (mean inter-key
time per block, s), 'FigS2B' (key-press errors per block), 'Tab1 + Fig2A&B'.'filter PSG' (polysomnography subset).
Expected values of published checks come from the paper text (Europe PMC PMC4816304: Methods, Results, Table 1, S2 Fig).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Laventure2016_PLoSBiol"
F = stdlib.raw_dir(ID) / "Data_S1.xlsx"
checks = stdlib.Checks()
T = [f"T{i}" for i in range(1, 25)]
R = [f"R{i}" for i in range(1, 9)]
BL = T + R
GROUP = {1: "Cond-NREM2", 2: "Cond-REM", 3: "NoCond"}

t1 = pd.read_excel(F, sheet_name="Tab1 + Fig2A&B")
s2a = pd.read_excel(F, sheet_name="FigS2A")
s2b = pd.read_excel(F, sheet_name="FigS2B")
checks.internal("76 rows in each of the three sheets, no missing values in the block columns",
                len(t1) == len(s2a) == len(s2b) == 76 and not (t1[BL].isna().any().any() or s2a[BL].isna().any().any() or s2b[BL].isna().any().any()))

# The speed sheet is sorted by group; its order within a group is the order of the GPI sheet. The error sheet has the GPI sheet's order.
checks.internal("error sheet lists the groups in the GPI sheet's order; speed sheet is sorted by group with the same group sizes",
                bool((t1.Group.values == s2b.Group.values).all() and (s2a.Group.diff().dropna() >= 0).all()
                     and (t1.Group.value_counts().sort_index() == s2a.Group.value_counts().sort_index()).all()))
order = pd.concat([t1[t1.Group == g].index.to_series() for g in sorted(t1.Group.unique())]).to_numpy()   # speed row i -> GPI row order[i]
speed = pd.DataFrame(index=t1.index, columns=BL, dtype=float)
speed.loc[order] = s2a[BL].to_numpy()
# the paper's GPI = exp(-speed) * exp(-accuracy); with speed = mean inter-key time (s) and accuracy = errors / 80 key presses
pred = np.exp(-speed[BL].to_numpy()) * np.exp(-s2b[BL].to_numpy() / 80.0)
err = np.abs(pred - t1[BL].to_numpy())
tol = np.where(np.arange(32) < 24, 0.0051, 0.0011)
over = err > tol
checks.internal("pairing of the speed sheet with the GPI sheet: GPI = exp(-inter-key time) x exp(-errors/80) (Methods, GPI formula) "
                "reproduces the GPI of at least 90% of the 76 x 32 cells within the rounding of the GPI columns, and none is off by more than 0.01 (the rest differ by up to 0.0055, at the rounding edge)",
                bool(over.mean() < 0.10 and err.max() < 0.01), f"{int(over.sum())} of {over.size} cells beyond rounding, largest difference {err.max():.4f}")
mis = np.array([np.mean(np.abs(np.exp(-speed[BL].to_numpy()[np.roll(np.arange(76), 1)]) * np.exp(-s2b[BL].to_numpy() / 80.0) - t1[BL].to_numpy()) <= tol)])
checks.internal("a shifted pairing (each speed row given to the neighbouring participant) fails that formula for most cells", bool(mis[0] < 0.5), f"{mis[0]:.2f} of cells within rounding")

pers = np.array([f"{ID}_P{i + 1:02d}" for i in range(len(t1))])
rows = []
for i in range(len(t1)):
    for b in BL:
        rows.append((i, b))
idx = np.array([r[0] for r in rows])
bl = [r[1] for r in rows]
isT = np.array([b.startswith("T") for b in bl])
bnum = np.array([int(b[1:]) for b in bl])
vals = lambda df: np.array([df.iloc[i][b] for i, b in rows], dtype=float)
t = pd.DataFrame({
    "dataset": ID, "study": ID, "experiment": f"{ID}_Exp1",
    "group": [GROUP[g] for g in t1.Group.to_numpy()[idx]], "subject": pers[idx], "person": pers[idx],
    "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
    "device": "four-button response box, left (non-dominant) hand", "session": np.where(isT, 1, 2),
    "block": bl, "trial": np.tile(np.arange(1, 33), len(t1)), "phase": np.where(isT, "learning", "test"), "time_s": "",
    "task": "sequence learning", "unit": "trial", "condition": "", "pert_type": "none", "pert_value": 0, "pert_unit": "",
    "feedback": "normal", "excluded_by_authors": False,
    "block_number": bnum, "block_type": np.where(isT, "training", "test"), "epoch": "", "stimulus": "", "correct_key": "",
    "key_pressed": "", "correct": "", "rt_ms": "", "rt_correct_ms": "",
    "interval_ms": np.rint(vals(speed.reset_index(drop=True)) * 1000, ), "element_type": "sequence", "sequence": "2-4-1-3-4-2-3-1",
    "sequence_position": "", "repetition": "", "triplet_type": "none",
    "gpi": vals(t1), "n_errors": vals(s2b).astype(int),
})
# interval_ms: keep the authors' value to the precision they gave (seconds with many decimals) rather than rounding to ms
t["interval_ms"] = vals(speed) * 1000.0
for col in [c[1] for c in stdlib.columns_for("keypress")]:
    if col not in t.columns:
        t[col] = ""
checks.internal("every cell kept: 76 participants x 32 blocks", len(t) == 76 * 32)

# ---------------------------------------------------------------- published checks
SRC = "Methods, Participants (Behavioral N 25, 23, 28)"
checks.published("participants in the behavioural analyses", len(t1), 76, 0, "Results, MSL Consolidation (76 participants)")
for g, nn in [(1, 25), (2, 23), (3, 28)]:
    checks.published(f"{GROUP[g]} participants", (t1.Group == g).sum(), nn, 0, "Table 1, 'All subjects' n")
psg = t1[t1["filter PSG"] == 1]
checks.published("participants in the PSG subset", len(psg), 64, 0, "Table 1, 'PSG subset (n = 64)'")
for g, nn in [(1, 21), (2, 21), (3, 22)]:
    checks.published(f"{GROUP[g]} participants in the PSG subset", (psg.Group == g).sum(), nn, 0, "Table 1, PSG subset n")
checks.published("training blocks per participant", len(T), 24, 0, "Methods / Results (24 training blocks)")
checks.published("retest blocks per participant", len(R), 8, 0, "Methods / Results (8 retest blocks)")
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    elif not ch["ok"]:
        print("  FAILED:", ch["what"], ch.get("detail", ""))
if not all(c["ok"] for c in checks.items):
    sys.exit(1)

gain = t1[R[:4]].mean(axis=1) - t1[T[-4:]].mean(axis=1)
raw_gain = gain.groupby(t1.Group).mean()
print("raw offline gains", raw_gain.round(4).to_dict())

meta = dict(
    study="Laventure et al. 2016",
    paper=dict(authors="Laventure S, Fogel S, Lungu O, Albouy G, Sevigny-Dupont P, Vien C, Sayour C, Carrier J, Benali H, Doyon J",
               year=2016, title="NREM2 and Sleep Spindles Are Instrumental to the Consolidation of Motor Sequence Memories",
               venue="PLoS Biol 14(3):e1002429", doi="10.1371/journal.pbio.1002429"),
    lab="Doyon (Universite de Montreal)",
    data_url="https://doi.org/10.5061/dryad.b4t60",
    license="CC0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    changes=("Converted to the OpenMotor format from the Dryad workbook Data S1 (copy on Zenodo record 5013749): the block-level "
             "GPI, inter-key time and error sheets were joined into one row per participant and block; inter-key time converted "
             "from seconds to milliseconds. The sleep (polysomnography, spindle, bootstrap) sheets and the two covariate columns "
             "cov1 and cov2 were not included."),
    how=[
        "One row is one block of 80 key presses of the sequential finger-tapping task (explicitly known 8-item sequence 2-4-1-3-4-2-3-1, "
        "left hand, four-button box) for one participant: blocks 1-24 are the evening training session (session 1), blocks 1-8 the "
        "retest the next morning (session 2); block names T1-T24 and R1-R8 are the authors'.",
        "interval_ms is the participant's average time between key presses in the block (the authors' speed measure, from seconds); "
        "n_errors is the number of key-press errors in the block; gpi is the authors' global performance index for the block. "
        "Per-key-press data are not in the deposit.",
        "group is the odor condition (Cond-NREM2, Cond-REM, NoCond), named from the group sizes in Table 1 of the paper (1 = 25, 2 = 23, "
        "3 = 28 participants); participants.csv says who was in the polysomnography subset. The deposit has no ages or sexes.",
        "The inter-key time sheet is sorted by group and carries no participant id; it was paired with the GPI sheet in the order "
        "within each group, which reproduces the GPI values from the paper's formula to within rounding (checked on all 2432 cells; 5% sit at the rounding edge, up to 0.0055 off).",
    ],
    unit_note="block of 80 key presses",
    differences=[
        "Table 1's offline gains (mean GPI of the first four retest blocks minus the last four training blocks: .042, .021, .012) "
        "were not used as checks: the plain group means from these blocks are .037, .022 and .015, and means adjusted for the two "
        "covariate columns are .041, .021 and .012, one unit of the third decimal off for the first group. The covariates' "
        "meaning (olfactory threshold, cueing duration) is not documented in the workbook, so they were left out of the release.",
    ],
)
participants = pd.DataFrame({"person": pers, "group": [GROUP[g] for g in t1.Group], "psg_subset": (t1["filter PSG"] == 1).to_numpy()})
stdlib.release(ID, "keypress", t, meta, checks, participants=participants,
               participant_columns=[("group", "text", "", "Odor condition."),
                                    ("psg_subset", "true/false", "", "In the polysomnography subset (n = 64) of the paper.")])
