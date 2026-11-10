"""Vishne, Jacoby, Malinovitch, Epstein, Frenkel & Ahissar (2021), Nat Commun: paced finger tapping to an auditory metronome by
neurotypical adults and adults with dyslexia or autism. Experiment 1: isochronous 500-ms metronome. Experiment 2: the metronome
alternates between two tempos (5 step sizes, 10-90 ms). One row per metronome beat of a block (the tap that answered it).

Run from the site folder: python3 -I tools/std/convert/Vishne2021_NatCommun.py
Raw files (untrusted data, read only) in raw_extra/Vishne2021_NatCommun/ (https://osf.io/83wnu/):
  raw_data.mat      tapping_data.e / .r / .s: cell arrays [109 participants x 6 conditions x 2 repetitions], each cell a vector of
                    asynchronies e (ms, tap minus metronome beat; NaN = omitted or excluded tap), inter-tap intervals r and
                    inter-stimulus intervals s (ms); subject_info and aq_data are MATLAB tables (read with the helper
                    Vishne2021_NatCommun_labtools.py, which decodes the MAT-file's object store as plain data)
  analyses_data.mat grp (1 = neurotypical, 2 = dyslexia, 3 = autism, in the order of raw_data.mat; g_size = 47, 32, 30) and
                    tempos (the two metronome periods of each of the 6 conditions)
  *.m               the authors' scripts, read as text only (analysis_data.m: grp = subject_info.Group; exclusion of taps beyond
                    +-200 ms already applied as NaN in e)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Vishne2021_NatCommun_labtools import MatObjects  # noqa: E402

ID = "Vishne2021_NatCommun"
RAW = stdlib.raw_dir(ID)
GROUP = {1: "neurotypical", 2: "dyslexia", 3: "autism"}
checks = stdlib.Checks()

# ------------------------------------------------------------------ read
raw = sio.loadmat(RAW / "raw_data.mat", squeeze_me=False, struct_as_record=True)
an = sio.loadmat(RAW / "analyses_data.mat", squeeze_me=False, struct_as_record=True)
td = raw["tapping_data"][0, 0]
E, R, S = td["e"], td["r"], td["s"]
mo = MatObjects(RAW / "raw_data.mat")
tab_info = mo.variable(mo.top["subject_info"])
names = [str(np.asarray(v).ravel()[0]) for v in tab_info["varnames"]]
info = {n: np.asarray(c, float).ravel() for n, c in zip(names, tab_info["data"])}
tempos = an["tempos"].astype(int)
grp = an["grp"].ravel().astype(int)
n_sub, n_cond, n_rep = E.shape
checks.internal("109 participants x 6 conditions x 2 repetitions in the tapping arrays; subject_info has one row each",
                (n_sub, n_cond, n_rep) == (109, 6, 2) and all(len(v) == 109 for v in info.values()))
checks.internal("subject_info.Group equals the group vector of analyses_data.mat (1 neurotypical, 2 dyslexia, 3 autism)",
                bool((info["Group"].astype(int) == grp).all()))
checks.internal("group sizes 47, 32, 30 as in analyses_data.mat g_size",
                [int((grp == g).sum()) for g in (1, 2, 3)] == an["g_size"].ravel().tolist() == [47, 32, 30])
checks.internal("tempo pairs of the six conditions: 500/500 (isochronous) and five alternating pairs 10-90 ms apart",
                tempos.tolist() == [[500, 500], [495, 505], [485, 515], [475, 525], [465, 535], [455, 545]])

# ------------------------------------------------------------------ rows
cols = {k: [] for k in ("subj", "cond", "rep", "e", "r", "s", "j", "tt")}
for i in range(n_sub):
    t = 0
    for c in range(n_cond):
        for k in range(n_rep):
            e = np.asarray(E[i, c, k], float).ravel()
            r = np.asarray(R[i, c, k], float).ravel()
            s = np.asarray(S[i, c, k], float).ravel()
            assert len(e) == len(r) == len(s) > 0
            t += 1
            n = len(e)
            cols["subj"].append(np.full(n, i))
            cols["cond"].append(np.full(n, c))
            cols["rep"].append(np.full(n, k))
            cols["e"].append(e)
            cols["r"].append(r)
            cols["s"].append(s)
            cols["j"].append(np.arange(1, n + 1))
            cols["tt"].append(np.full(n, t))
d = {k: np.concatenate(v) for k, v in cols.items()}
n = len(d["e"])
subj, cond, rep, e, r, s = d["subj"], d["cond"], d["rep"], d["e"], d["r"], d["s"]
miss = np.isnan(e)
rel = ~np.isnan(r) & ~np.isnan(e)
rel[d["j"] == 1] = False
prev_e = np.r_[np.nan, e[:-1]]
ok = rel & ~np.isnan(prev_e)
checks.internal("inter-tap interval r = inter-stimulus interval s + asynchrony e - previous asynchrony, wherever both taps exist",
                bool((np.abs(r[ok] - (s[ok] + e[ok] - prev_e[ok])) < 1e-6).all()), f"{int(ok.sum())} rows")
checks.internal("no asynchrony beyond +-200 ms (the authors' window); omitted or excluded taps are NaN",
                bool((np.abs(e[~miss]) <= 200).all()), f"{int(miss.sum())} of {n} rows without a valid tap")

step = np.abs(tempos[:, 1] - tempos[:, 0])
sex_code = info["M_1_F_2"].astype(int)
age = info["age"]
step_row = step[cond]
tempo_txt = np.array(["metronome 500 ms" if a == b else f"alternating {a}/{b} ms" for a, b in tempos])
tab = pd.DataFrame({
    "dataset": ID, "study": ID,
    "experiment": np.where(cond == 0, "Experiment 1", "Experiment 2"),
    "group": pd.Series(grp[subj]).map(GROUP).to_numpy(),
    "subject": (info["Subject_Index"][subj]).astype(int).astype(str),
    "subject_excluded_by_authors": False,
    "age": age[subj], "sex": pd.Series(sex_code[subj]).map({1: "M", 2: "F"}).to_numpy(), "handedness": "",
    "population": pd.Series(grp[subj]).map({1: "healthy", 2: "dyslexia", 3: "autism"}).to_numpy(),
    "setting": "lab", "device": "custom wooden tapping box with a microphone; metronome clicks through headphones",
    "session": 1, "block": [f"condition {c + 1}, repetition {k + 1}" for c, k in zip(cond, rep)],
    "trial": pd.Series(np.arange(n)).groupby(subj).rank().astype(int).to_numpy(),
    "phase": "test", "time_s": "", "task": "finger tapping and timing", "unit": "tap",
    "condition": tempo_txt[cond],
    "pert_type": np.where(cond == 0, "none", "tempo_switch"), "pert_value": step_row, "pert_unit": "ms",
    "feedback": "normal", "excluded_by_authors": miss,
    "tapping_trial": d["tt"], "event_in_trial": d["j"], "stimulus": tempo_txt[cond],
    "tap_time_ms": "", "stimulus_time_ms": "", "asynchrony_ms": e, "iti_ms": r, "ioi_ms": s, "stimulus_played": True,
})
tab.loc[cond == 0, "pert_value"] = 0
checks.internal("every beat of every block kept in the authors' order (109 x 12 blocks of 96-103 beats)",
                len(tab) == n == sum(len(np.ravel(E[i, c, k])) for i in range(n_sub) for c in range(n_cond) for k in range(n_rep)),
                f"{n} rows")
checks.internal("12 blocks per participant, 96-103 beats each",
                bool((tab.groupby("subject", sort=False).tapping_trial.nunique() == 12).all())
                and 96 <= tab.groupby(["subject", "tapping_trial"]).size().min() and tab.groupby(["subject", "tapping_trial"]).size().max() <= 103)

# ------------------------------------------------------------------ published checks (Results, Experiment 1)
e1 = tab[tab.experiment == "Experiment 1"].copy()
SRC = "Results, 'Experiment 1 - isochronous tapping reveals reduced online error correction in ASD, but not in dyslexia'"
for g, nn in (("neurotypical", 47), ("dyslexia", 32), ("autism", 30)):
    checks.published(f"{g} participants", tab[tab.group == g].subject.nunique(), nn, 0, SRC)
checks.published("participants", tab.subject.nunique(), 109, 0, SRC)
per = e1.groupby("subject").agg(mean_e=("asynchrony_ms", "mean"), miss=("excluded_by_authors", "mean"), grp=("group", "first"))
per["pct_miss"] = 100 * per.miss
sd_block = e1.groupby(["subject", "tapping_trial"]).asynchrony_ms.std(ddof=1).groupby("subject").mean()
per["sd"] = sd_block
checks.published("participants with a negative mean asynchrony", int((per.mean_e < 0).sum()), 105, 0, SRC)
for g, med in (("neurotypical", 0.5), ("dyslexia", 0.5), ("autism", 1.2)):
    checks.published(f"median % omitted or excluded taps, {g}, Experiment 1", per[per.grp == g].pct_miss.median(), med, 0.05,
                     "Methods, Finger tapping analyses")
for g, med in (("neurotypical", 30.6), ("dyslexia", 30.2), ("autism", 41.4)):
    checks.published(f"median SD of asynchrony (ms, average of the two blocks), {g}, Experiment 1", per[per.grp == g].sd.median(), med, 0.05,
                     SRC)
for ch in checks.items:
    if ch["kind"] == "published":
        print(f"  {ch['what']}: computed {ch['computed']:.4f}, reported {ch['reported']}, ok={ch['ok']}")
    else:
        print(f"  [{'ok' if ch['ok'] else 'FAILED'}] {ch['what']} {ch.get('detail', '')}")

meta = dict(
    study="Vishne et al. 2021",
    paper=dict(authors="Vishne G, Jacoby N, Malinovitch T, Epstein T, Frenkel O, Ahissar M", year=2021,
               title="Slow update of internal representations impedes synchronization in autism", venue="Nat Commun 12(1):5439",
               doi="10.1038/s41467-021-25740-y"),
    lab="Ahissar (Hebrew University of Jerusalem)",
    data_url="https://osf.io/83wnu/",
    license="none stated", license_url="",
    changes=("Converted to the OpenMotor format with participant codes replaced by new codes. Values are the authors' own "
             "(asynchronies, inter-tap and inter-stimulus intervals); the authors' omitted or excluded taps (NaN) are kept as "
             "rows marked excluded_by_authors. Not converted: the reading and cognitive test scores in subject_info, the AQ "
             "questionnaire answers, the model fits and simulations in analyses_data.mat. Raw tap and click times were not "
             "shared, so tap_time_ms and stimulus_time_ms are empty."),
    how=[
        "One row is one metronome beat of a tapping block and the participant's tap that answered it: asynchrony_ms is the "
        "authors' e (tap minus beat, negative when the tap came first), iti_ms their r (interval from the previous tap) and "
        "ioi_ms their s (interval from the previous beat), all in ms. A row with no valid tap (omitted, or beyond +-200 ms of "
        "the beat) has empty values and excluded_by_authors True.",
        "Each participant has 12 blocks of about 100 beats: 6 conditions x 2 repetitions, numbered tapping_trial 1-12 in the "
        "order of the authors' arrays (condition-major); the order in which blocks were run is not in the deposit. Condition 1 is "
        "Experiment 1 (isochronous 500-ms metronome). Conditions 2-6 are Experiment 2: the metronome alternates between two "
        "periods around 500 ms at unexpected times; pert_value is the difference between the two periods (10, 30, 50, 70 or "
        "90 ms), constant within the block.",
        "group is neurotypical (CON), dyslexia (DYS) or autism (ASD); sex is the authors' code M_1_F_2 (1 male, 2 female). "
        "The 24 participants the authors excluded (matching criteria) are not in the deposit.",
    ],
    unit_note="metronome beat and the tap that answered it",
    differences=[],
)
stdlib.release(ID, "tapping", tab, meta, checks)
