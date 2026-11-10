"""Gonzalez-Rubio, Torres-Oviedo & Iturralde (2025) eNeuro: two-alternative forced-choice judgements of a belt-speed
difference ("which leg is slower?") made while walking on a split-belt treadmill at a mean speed of 1.05 m/s, 39 young
adults, one row per perceptual trial.

Raw file: raw_extra/Gonzalez-Rubio2025_eNeuro/repo_61fe3dc.zip, the authors' GitHub repository
(PittSMLlab/Gonzalez-Rubioetal2025_perceptualCharacterization, commit 61fe3dc; the OSF project 9zh5b links to it), read
with zipfile. data/<participant>/Slow<k>.csv holds every trial of one treadmill run (pertSize in mm/s, reactionTime in s,
initialResponse -1 = "left is slower", 1 = "right is slower"); data/Dataset.csv is the authors' analysis table. The
authors' src/datlogManipulation/loadAllDataIntoTable.m and src/plots/accrtPlot.m were read as text (never run) to
rebuild Dataset.csv from the per-run files and to reproduce the paper's numbers.

Run from the site folder: python3 -I tools/std/convert/Gonzalez-Rubio2025_eNeuro.py
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Gonzalez-Rubio2025_eNeuro"
MODULE = "perception"
ARCHIVE = "repo_61fe3dc.zip"
# participant folders in the order of the authors' table (loadAllDataIntoTable.m: subID 1..39)
ORDER = ([f"WP{n:02d}" for n in [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17]]
         + [f"WPS{n:02d}" for n in [1, 3, 5, 6, 9, 11, 12, 13, 15, 16, 17, 18, 19]]
         + [f"WP{n:02d}" for n in [18, 19, 21, 22, 24, 26, 27, 28, 29, 31, 32, 33, 36]])
FLIPPED_SUB = 32  # authors' subID whose responses their code reverses ("this participants responses will be flipped")
NO_FIRST_DROP_SUB = 1  # authors keep the first trial of each run for subID 1 (its first stimulus was not repeated)
MEAN_SPEED_MM_S = 1050.0
DEVICE = ("split-belt treadmill (mean belt speed 1.05 m/s); a hand-held controller in each hand for the key press; "
          "noise-cancelling headphones for the cues; a drape hiding the feet")
STIMULUS = "belt-speed difference, right belt minus left belt (positive: left belt slower)"
LABEL = {-1: "left slower", 1: "right slower"}


def read_runs(z, root):
    runs = []
    for sid, fol in enumerate(ORDER, start=1):
        files = sorted(n for n in z.namelist() if n.startswith(f"{root}data/{fol}/") and n.endswith(".csv"))
        k = 1
        while f"{root}data/{fol}/Slow{k}.csv" in files:
            t = pd.read_csv(io.BytesIO(z.read(f"{root}data/{fol}/Slow{k}.csv")))
            assert list(t.columns) == ["pertSize", "reactionTime", "initialResponse"], t.columns
            t["subID"], t["bout"], t["row_in_run"] = sid, k, np.arange(len(t))
            runs.append(t)
            k += 1
        assert len(files) == k - 1, (fol, len(files), k - 1)
    return pd.concat(runs, ignore_index=True)


def logit_fit(x, y):
    """Binomial GLM with logit link by iteratively reweighted least squares (as MATLAB fitglm): coefficients and SEs."""
    X = np.c_[np.ones_like(x), x]
    b = np.zeros(2)
    for _ in range(200):
        p = 1 / (1 + np.exp(-X @ b))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), X.T @ (y - p))
        b = b + step
        if np.abs(step).max() < 1e-12:
            break
    p = 1 / (1 + np.exp(-X @ b))
    return b, np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))))


def main():
    z = zipfile.ZipFile(stdlib.raw_dir(DS) / ARCHIVE)
    root = z.namelist()[0]
    ck = stdlib.Checks()
    D = read_runs(z, root)
    n_all = len(D)

    # ---- the authors' table, rebuilt from the per-run files with their rules (loadAllDataIntoTable.m)
    has_stim = D.pertSize.notna().to_numpy()
    A = D[has_stim].copy()  # the authors drop rows without a stimulus
    first = A.groupby(["subID", "bout"]).row_in_run.transform("min").eq(A.row_in_run).to_numpy()
    last = A.groupby(["subID", "bout"]).row_in_run.transform("max").eq(A.row_in_run).to_numpy()
    ps = A.pertSize.to_numpy()
    prev_a = np.where(first, 0, np.r_[0, ps[:-1]])  # authors' prevSize, set before their deletions
    li = np.where(last)[0][:-1]
    repeated_last = np.zeros(len(A), bool)
    repeated_last[li[ps[li] == ps[li + 1]]] = True  # last trial of a run with the same stimulus as the next run's first
    first_drop = first & (A.subID.to_numpy() != NO_FIRST_DROP_SUB)
    keep = ~repeated_last & ~first_drop
    reason = np.where(repeated_last, "repeated last trial of a run", np.where(first_drop, "first trial after the treadmill started", ""))
    D["analysed"] = False
    D.loc[A.index[keep], "analysed"] = True
    D["why"] = ""
    D.loc[A.index, "why"] = reason
    D.loc[~has_stim, "why"] = "no stimulus recorded"

    ds = pd.read_csv(io.BytesIO(z.read(f"{root}data/Dataset.csv")))
    # The authors reverse one participant's responses. Their left/right choice columns (used in every fit) are reversed
    # only on trials with a speed difference that got a response; their initialResponse column is reversed on all rows.
    # response follows the choice columns, so the paper's fits reproduce.
    sub32 = D.subID == FLIPPED_SUB
    flip_choice = sub32 & D.pertSize.notna() & (D.pertSize != 0) & D.initialResponse.notna()
    D["response"] = np.where(flip_choice, -D.initialResponse, D.initialResponse)
    K = D[D.analysed].reset_index()
    same = len(K) == len(ds) and all(
        np.allclose(K[a].to_numpy(float), ds[b].to_numpy(float), rtol=0, atol=0, equal_nan=True)
        for a, b in [("pertSize", "pertSize"), ("reactionTime", "reactionTime"), ("subID", "subID")])
    ck.internal("the analysed rows (excluded_by_authors False) are exactly the authors' Dataset.csv, in its order: "
                "stimulus, reaction time and participant equal on all 6,549 rows", same, f"{len(K)} vs {len(ds)} rows")
    init_ok = np.allclose(np.where(K.subID == FLIPPED_SUB, -1, 1) * K.initialResponse.to_numpy(float),
                          ds.initialResponse.to_numpy(float), rtol=0, atol=0, equal_nan=True)
    ck.internal("Dataset.csv's initialResponse = the run files' response, reversed on every row of one participant", init_ok)
    differs = ~np.isclose(K.response.to_numpy(float), ds.initialResponse.to_numpy(float), equal_nan=True)
    ck.internal("response differs from Dataset.csv's initialResponse only on that participant's answered null trials, where the "
                "authors' choice columns keep the recorded side",
                (differs == (K.subID.eq(FLIPPED_SUB) & K.pertSize.eq(0) & K.initialResponse.notna())).all(),
                f"{int(differs.sum())} rows")
    # block numbers: each run belongs to one of the authors' three blocks
    K["blockNo"] = ds.blockNo.to_numpy()
    bmap = K.groupby(["subID", "bout"]).blockNo.agg(["min", "max"])
    ck.internal("each treadmill run (deposit file) falls in exactly one of the authors' three blocks", (bmap["min"] == bmap["max"]).all()
                and len(bmap) == D.groupby(["subID", "bout"]).ngroups, f"{len(bmap)} runs")
    D = D.merge(bmap["min"].rename("blockNo"), left_on=["subID", "bout"], right_index=True, how="left")
    derived = (
        np.array_equal((K.response == -1).to_numpy(), ds.leftResponse.astype(bool).to_numpy())
        and np.array_equal((K.response == 1).to_numpy(), ds.rightResponse.astype(bool).to_numpy())
        and np.array_equal(K.response.isna().to_numpy(), ds.noResponse.astype(bool).to_numpy())
        and np.array_equal(((K.response == -np.sign(K.pertSize)) & (K.pertSize != 0)).to_numpy(), ds.correctResponses.astype(bool).to_numpy())
        and np.array_equal(((K.response == np.sign(K.pertSize)) & (K.pertSize != 0)).to_numpy(), ds.incorrectResponses.astype(bool).to_numpy()))
    ck.internal("Dataset.csv's left/right/no-response and correct/incorrect columns follow from the response and the stimulus "
                "(correct = response opposite in sign to the stimulus, none on null trials)", derived)
    ck.internal("Dataset.csv's previous-stimulus column equals the stimulus of the previous row of the same run file (0 at a "
                "run's first row), set before the authors' deletions: the rows line up one to one",
                len(K) == len(ds) and np.array_equal(prev_a[keep], ds.prevSize.to_numpy()))

    # ---- rows
    resp = D.response
    stim = D.pertSize
    correct = np.where(resp.isna() | stim.isna() | (stim == 0), None, (resp == -np.sign(stim)))
    df = pd.DataFrame(dict(
        dataset=DS, study=DS, experiment="", group="", subject=[f"s{s}" for s in D.subID], subject_excluded_by_authors=False,
        age=np.nan, sex="", handedness="", population="healthy", setting="lab", device=DEVICE, session=np.nan,
        block=[f"block {int(b)}" for b in D.blockNo], trial=D.groupby("subID").cumcount().to_numpy() + 1, phase="test",
        time_s=np.nan, task="walking", unit="trial", condition="",
        pert_type=np.where(stim.isna(), "", np.where(stim == 0, "none", "split_belt")),
        pert_value=stim.to_numpy(float), pert_unit=np.where(stim.isna() | (stim == 0), "", "mm/s"),
        feedback="masked", excluded_by_authors=~D.analysed.to_numpy(),
        stimulus=np.where(stim.isna(), "", STIMULUS), stimulus_value=stim.to_numpy(float),
        stimulus_unit=np.where(stim.isna(), "", "mm/s"), reference_value=np.where(stim.isna(), np.nan, MEAN_SPEED_MM_S),
        response=[("" if np.isnan(r) else str(int(r))) for r in resp], response_label=[LABEL.get(r, "") if not np.isnan(r) else "" for r in resp],
        response_value=np.nan, response_unit="", response_correct=correct, reaction_time_s=D.reactionTime.to_numpy(float),
        bout=D.bout.to_numpy(),
    ))
    for c in ["stimulus_value", "pert_value", "reaction_time_s"]:
        df[c] = df[c].map(lambda v: float(format(v, ".8g")) if np.isfinite(v) else np.nan)  # what the CSV holds

    # ---- internal checks
    ck.internal("all 39 participant folders read, in the authors' order; every Slow<k>.csv of each folder read",
                D.subID.nunique() == 39, f"{D.groupby(['subID', 'bout']).ngroups} runs")
    ck.internal("every row of the per-run files is kept, in order: 6,796 rows = 6,549 analysed + 236 first trials after a "
                "treadmill start + 6 repeated last trials + 5 rows without a stimulus (3 of them empty)",
                n_all == 6796 and int(D.analysed.sum()) == 6549 and (D.why == "first trial after the treadmill started").sum() == 236
                and (D.why == "repeated last trial of a run").sum() == 6 and (D.why == "no stimulus recorded").sum() == 5,
                str(D.why.replace("", "analysed").value_counts().to_dict()))
    nb = K.groupby(["subID", "blockNo"]).size()
    ck.internal("each participant has three analysed blocks of 55 or 56 trials (56 planned)", len(nb) == 117 and nb.between(55, 56).all(),
                str(nb.value_counts().to_dict()))
    ck.internal("stimuli are 0, +-25, +-50, +-100, +-150, +-200 and +-300 mm/s",
                set(stim.dropna().astype(int)) == {0, 25, -25, 50, -50, 100, -100, 150, -150, 200, -200, 300, -300})
    ck.internal("reaction time is missing exactly when the response is missing; recorded times lie within 0-10 s "
                "(response window 8 strides)", (D.reactionTime.isna() == D.initialResponse.isna()).all()
                and D.reactionTime.dropna().between(0, 10).all(), f"{D.reactionTime.min():.3f}-{D.reactionTime.max():.3f} s")
    ck.internal("null trials: 8 per analysed block of 56", K[K.pertSize == 0].groupby(["subID", "blockNo"]).size().between(7, 8).all())

    # ---- published numbers, from the analysed rows of the released table
    a = df[~df.excluded_by_authors].copy()
    a["s"] = a.subject
    ck.published("participants", a.s.nunique(), 39, 0, "Methods, Participants")
    ck.published("blocks analysed (39 x 3)", a.groupby(["s", "block"]).ngroups, 117, 0, "Methods, Testing protocol")
    nr = a[a.response == ""]
    ck.published("non-response trials", len(nr), 16, 0, "Methods, Testing protocol")
    for mag, n in [(0, 5), (25, 5), (50, 3), (150, 2), (300, 1)]:
        ck.published(f"non-response trials at |dV| = {mag} mm/s", int((nr.stimulus_value.abs() == mag).sum()), n, 0, "Methods, Testing protocol")
    r = a[a.response != ""].copy()
    r["left"] = (r.response == "-1").astype(float)
    fits = []
    for s, g in r.groupby("s", sort=False):
        b, se = logit_fit(g.stimulus_value.to_numpy(float), g.left.to_numpy())
        acc = (g.response_correct.map({True: 1.0, False: 0.0}).fillna(0.5)).mean()  # accrtPlot.m: null trials count 0.5
        fits.append(dict(s=s, b0=b[0], b1=b[1], se0=se[0], pse=-b[0] / b[1], jnd=np.log(3) / b[1], acc=acc,
                         rt=g.reaction_time_s.mean()))
    F = pd.DataFrame(fits)
    src2 = "Results, Individual differences in PSE and JND (Fig. 2)"
    ck.published("mean individual PSE, mm/s (logistic fit per participant, PSE = -b0/b1)", F.pse.mean(), 5.3, 0.05, src2)
    ck.published("SD of individual PSEs, mm/s", F.pse.std(), 19, 0.05, src2)
    ck.published("smallest individual PSE, mm/s", F.pse.min(), -32.5, 0.05, src2)
    ck.published("largest individual PSE, mm/s", F.pse.max(), 49.1, 0.05, src2)
    ck.published("participants with a PSE significantly different from 0 (95% CI with b1 fixed)",
                 int((np.abs(F.b0) > 1.959964 * F.se0).sum()), 10, 0, src2)
    ck.published("mean individual JND, mm/s (JND = ln 3 / b1)", F.jnd.mean(), 49.7, 0.05, src2)
    ck.published("SD of individual JNDs, mm/s", F.jnd.std(), 22.6, 0.05, src2)
    ck.published("smallest individual JND, mm/s", F.jnd.min(), 14.3, 0.05, src2)
    ck.published("largest individual JND, mm/s", F.jnd.max(), 112.5, 0.05, src2)
    ck.published("mean Weber fraction, % (mean JND / 1050 mm/s)", 100 * F.jnd.mean() / MEAN_SPEED_MM_S, 4.7, 0.05, src2)
    ck.published("lowest individual accuracy, % (null trials scored 0.5, as in accrtPlot.m; the paper gives the whole "
                 "percent)", 100 * F.acc.min(), 71, 1, src2)
    ck.published("highest individual accuracy, % ('nearly 90%')", 100 * F.acc.max(), 90, 0.5, src2)
    ck.published("shortest individual mean RT, s", F.rt.min(), 1.8, 0.05, src2)
    ck.published("longest individual mean RT, s", F.rt.max(), 4.9, 0.05, src2)
    rr, pp = stats.pearsonr(F.acc, F.rt)
    ck.published("Pearson r, accuracy vs mean RT across participants", rr, -0.1347, 0.00005, src2)
    ck.published("p, accuracy vs mean RT", pp, 0.414, 0.001, src2)
    ck.published("Pearson r, JND vs accuracy", stats.pearsonr(F.jnd, F.acc)[0], -0.942, 0.0005, src2)
    rr, pp = stats.pearsonr(F.jnd, F.rt)
    ck.published("Pearson r, JND vs mean RT", rr, 0.175, 0.0005, src2)
    ck.published("p, JND vs mean RT", pp, 0.286, 0.001, src2)
    grt = r.groupby(["stimulus_value", "s"]).reaction_time_s.mean().groupby("stimulus_value").mean()
    src3 = "Results, Group-level inference (Fig. 3C)"
    ck.published("largest group mean RT across stimuli, s", grt.max(), 4.7, 0.05, src3)
    ck.published("stimulus with the largest group mean RT, mm/s (null trials)", grt.idxmax(), 0, 0, src3)
    ck.published("smallest group mean RT across stimuli, s", grt.min(), 1.8, 0.05, src3)
    acc25 = r[r.stimulus_value.abs() == 25].response_correct.astype(float).mean()
    ck.published("accuracy at |dV| = 25 mm/s pooled across participants, %", 100 * acc25, 65, 0.5, "Discussion, para 2")

    meta = dict(
        study=DS,
        paper=dict(authors="Gonzalez-Rubio M, Torres-Oviedo G, Iturralde PA", year=2025,
                   title="Characterizing Human Perception of Speed Differences in Walking: Insights From a Drift Diffusion Model",
                   venue="eNeuro 12(5):ENEURO.0343-23.2025", doi="10.1523/eneuro.0343-23.2025"),
        lab="Torres-Oviedo (University of Pittsburgh)",
        data_url="https://github.com/PittSMLlab/Gonzalez-Rubioetal2025_perceptualCharacterization",
        license="none stated",
        license_url="",
        changes=("Joined the per-run files of the 39 participants into one table in the authors' participant order, marked the "
                 "trials the authors' analysis table leaves out, added their block numbers, and wrote each response with "
                 "its meaning and correctness; participant folder names are replaced by codes."),
        how=["One row is one perceptual trial: after a belt-speed difference was set (one belt sped up and the other slowed by "
             "half of it, so the mean stayed 1.05 m/s), an audio cue opened a window of up to 8 strides in which the "
             "participant pressed a key on the left or right hand-held controller to say which leg was slower.",
             "stimulus_value and pert_value are that difference in mm/s, right belt minus left belt, so positive values mean "
             "the left belt was slower (0 on null trials); reference_value is the mean belt speed; response is -1 for 'left "
             "slower' and 1 for 'right slower', empty when the participant did not answer; reaction_time_s runs from the "
             "start cue to the key press; response_correct is empty on null trials.",
             "bout is the treadmill run (the deposit's Slow<k>.csv; the treadmill stopped between blocks and once within "
             "each) and block the authors' block (1-3) holding it; excluded_by_authors marks the 247 rows that the authors' "
             "Dataset.csv leaves out: the first trial after each treadmill start (its stimulus was repeated at the end of the "
             "block), 6 last trials whose stimulus repeats the next run's first one, and 5 rows without a stimulus (3 of them "
             "empty placeholders).",
             "For one participant the authors' table-building code reverses the responses: their left/right choice columns, "
             "which all their fits use, are reversed on the trials with a speed difference but not on the null trials "
             "(their initialResponse column is reversed on every row); response follows the choice columns, so the "
             "paper's numbers reproduce. Vision of the feet (drape) and sound "
             "(noise-cancelling headphones) were masked; a spoken message confirmed each choice without saying whether it "
             "was right. The 8 familiarization trials are not in the deposit, and which blocks fell on which of the two "
             "days is not recorded, so session is empty."],
        unit_note="perceptual trial (one judgement of which leg was slower)",
    )
    stdlib.release(DS, MODULE, df, meta, ck)
    for c in ck.items:
        if c["kind"] == "published":
            print(f"  {c['what']}: {c['computed']:.5g} vs {c['reported']}  ({c['source']})")
        elif c.get("detail"):
            print(f"  [internal] {c['what'][:70]}: {c['detail']}")


if __name__ == "__main__":
    main()
