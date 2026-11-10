"""Zeng, Niziolek & Parrell (2025), J Exp Psychol Gen: adaptation to opposing F1 shifts on the same vowel in
different words (Experiment 1: bed/head/ted; 2: seven/sever + level; 3: pedigree/pedicure + pedestal, carbonate).

Raw files, raw_extra/Zeng2025_JExpPsycholGen/ (OSF ytwqp, fetched with tools/std/fetch.py):
  F1_Exp<e>.csv   Speaker, meanF1_mel, Trial, Duration, Word, Phase, Direction   one row per trial the authors kept
  F0_Exp<e>.csv   Speaker, Trial, Word, Mean_F0                                  the same rows, in the same order
  Int_Exp<e>.csv  Speaker, Trial, Word, Mean_Int                                 the same rows, in the same order
What the values are (paper, Methods "Data processing"; authors' code at github.com/blab-lab/simonMulti, read as text):
  meanF1_mel  F1 of the first vowel averaged over 25-75% of its duration (40-75% of the vocalic chunk for 'level'), mels
  Mean_F0     f0 in Hz over the same window; Mean_Int the wave_viewer amplitude track (Hann-smoothed squared envelope,
              linear and uncalibrated) over the same window; Duration the segment duration in ms
  Trial       trial number in the experiment; trials removed for tracking errors or misproductions are absent
  Phase       Baseline, Ramp, Hold, Washout; Direction the shift condition assigned to the word for that speaker
Shift schedule, run_coAdapt_expt.m / run_simonMultisyllable_v2_expt.m / run_simonMultisyllable_expt.m:
  shiftMags = [zeros(nBaseline) sort(repmat(linspace(0,125,30),1,nwords)) 125*ones(nHold) zeros(nWashout)] .* (+1,-1,0)
  applied by Audapter as an F1-only shift in mels (bMelShift = 1, pertPhi = 0; run_*_audapter.m).
Run from the site folder: python3 -I tools/std/convert/Zeng2025_JExpPsycholGen.py
"""
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zeng2025_JExpPsycholGen"
# blocks per phase (one block = one repetition of each word), from the paper (Experiment design) and the run scripts
EXPS = {
    1: dict(words=("bed", "head", "ted"), base=30, ramp=30, hold=90, wash=30, n=15, win=(13, 16)),
    2: dict(words=("seven", "sever", "level"), base=20, ramp=30, hold=70, wash=10, n=20, win=(12, 13)),
    3: dict(words=("pedigree", "pedicure", "pedestal", "carbonate"), base=20, ramp=30, hold=70, wash=10, n=20, win=(12, 13)),
}
HOLD_MEL = 125.0
RAMP_STEPS = 30  # linspace(0, 125, 30): the first Ramp block has 0, the last 125
SIGN = {"Upshift": 1, "Downshift": -1, "Unshifted": 0}
PHASE = {"Baseline": "baseline", "Ramp": "learning", "Hold": "learning", "Washout": "washout"}
SEGMENT = {
    "bed": "vowel /ɛ/", "head": "vowel /ɛ/", "ted": "vowel /ɛ/",
    "seven": "first vowel /ɛ/ (sev-)", "sever": "first vowel /ɛ/ (sev-)",
    "level": "first vocalic portion /lɛ/ (lev-), including the /l/",
    "pedigree": "first vowel /ɛ/ (ped-)", "pedicure": "first vowel /ɛ/ (ped-)", "pedestal": "first vowel /ɛ/ (ped-)",
    "carbonate": "first vowel (car-)",
}
WINDOW = {"level": "40-75% of segment duration"}
WINDOW_DEFAULT = "25-75% of segment duration"
DEVICE = "Sennheiser MKE 600 microphone, Audapter, Beyerdynamic DT 770 PRO headphones"
MASKING = "speech-shaped noise, about 60 dB SPL"
LATENCY_MS = 18
PAPER_SRC = "Zeng, Niziolek & Parrell 2025 (accepted manuscript, PsyArXiv ceqan v2)"


def n_trials(e):
    x = EXPS[e]
    return (x["base"] + x["ramp"] + x["hold"] + x["wash"]) * len(x["words"])


def design_phase(e, trial):
    x, w = EXPS[e], len(EXPS[e]["words"])
    edges = np.cumsum([x["base"], x["ramp"], x["hold"], x["wash"]]) * w
    return ["Baseline", "Ramp", "Hold", "Washout"][int(np.searchsorted(edges, trial))]


def shift_mag(e, trial):
    """Unsigned F1 shift in mels on this trial number (expt.shiftMags before the sign of the word's condition)."""
    x, w = EXPS[e], len(EXPS[e]["words"])
    nb, nr, nh = x["base"] * w, x["ramp"] * w, x["hold"] * w
    if trial <= nb or trial > nb + nr + nh:
        return 0.0
    if trial <= nb + nr:
        return HOLD_MEL * ((trial - nb - 1) // w) / (RAMP_STEPS - 1)
    return HOLD_MEL


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def build(checks):
    raw = stdlib.raw_dir(ID)
    codes, rows = {}, []
    for e in (1, 2, 3):
        f1 = pd.read_csv(raw / f"F1_Exp{e}.csv")
        f0 = pd.read_csv(raw / f"F0_Exp{e}.csv")
        it = pd.read_csv(raw / f"Int_Exp{e}.csv")
        key = ["Speaker", "Trial", "Word"]
        aligned = (len(f1) == len(f0) == len(it) and (f1[key].values == f0[key].values).all()
                   and (f1[key].values == it[key].values).all())
        checks.internal(f"Experiment {e}: F0 and intensity files hold the same rows as the F1 file, in the same order",
                        aligned, f"{len(f1)} rows")
        assert aligned
        for r, f0v, iv in zip(f1.itertuples(index=False), f0["Mean_F0"], it["Mean_Int"]):
            code = codes.setdefault(r.Speaker, f"{ID}_S{len(codes) + 1:03d}")
            trial = int(r.Trial)
            val = SIGN[r.Direction] * shift_mag(e, trial)
            val = 0.0 if val == 0 else float(val)  # no -0.0
            rows.append({
                "dataset": ID, "study": ID, "experiment": f"Experiment {e}", "group": "", "subject": code,
                "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
                "setting": "lab", "device": DEVICE, "session": 1, "block": r.Phase, "trial": trial,
                "phase": PHASE[r.Phase], "time_s": "", "task": "speech", "unit": "utterance", "condition": r.Direction,
                "pert_type": "formant_shift" if val != 0 else "none", "pert_value": val, "pert_unit": "mel",
                "feedback": "altered" if val != 0 else "normal", "excluded_by_authors": bool(r.Word == "carbonate"),
                "stimulus": r.Word, "segment": SEGMENT[r.Word], "measure_window": WINDOW.get(r.Word, WINDOW_DEFAULT),
                "duration_ms": float(r.Duration), "f0_hz": float(f0v), "f1_hz": "", "f2_hz": "",
                "f1_mel": float(r.meanF1_mel), "intensity_au": float(iv), "shift_f1": val, "shift_f2": 0.0,
                "masking_noise": MASKING, "feedback_latency_ms": LATENCY_MS,
            })
    return rows


def internal_checks(a, checks):
    sizes = a.groupby("exp").size().to_dict()
    checks.internal("rows per experiment as in the authors' files", sizes == {1: 8001, 2: 7735, 3: 10350}, str(sizes))
    per = a.groupby("exp").subject.nunique().to_dict()
    checks.internal("participants per experiment = 15, 20, 20 (paper, Methods, Participants)",
                    per == {e: EXPS[e]["n"] for e in EXPS}, str(per))
    sets = {e: set(a.subject[a.exp == e]) for e in EXPS}
    checks.internal("53 people: two participant numbers appear in both Experiments 1 and 3; Experiment 2 shares none",
                    a.subject.nunique() == 53 and len(sets[1] & sets[3]) == 2 and not (sets[2] & (sets[1] | sets[3])),
                    f"{a.subject.nunique()} codes")
    checks.internal("subject x experiment x trial is unique and trials increase within each sequence",
                    not a.duplicated(["subject", "exp", "trial"]).any()
                    and all((g.trial.diff().dropna() > 0).all() for _, g in a.groupby(["subject", "exp"], sort=False)))
    checks.internal("trial numbers within the design (540, 390, 520 trials)",
                    all(((g.trial >= 1) & (g.trial <= n_trials(e))).all() for e, g in a.groupby("exp")))
    checks.internal("the authors' phase labels match the design's trial ranges",
                    all(design_phase(e, t) == b for e, t, b in zip(a.exp, a.trial, a.block)))
    one = a.groupby(["subject", "exp", "stimulus"]).condition.nunique()
    checks.internal("each word keeps one shift condition per participant", bool((one == 1).all()))
    cond = a.drop_duplicates(["subject", "exp", "stimulus"])
    ok = True
    for (s, e), g in cond.groupby(["subject", "exp"]):
        m = dict(zip(g.stimulus, g.condition))
        ok &= sorted(m) == sorted(EXPS[e]["words"])
        if e == 1:
            ok &= sorted(m.values()) == ["Downshift", "Unshifted", "Upshift"]
        if e == 2:
            ok &= m["level"] == "Unshifted" and {m["seven"], m["sever"]} == {"Upshift", "Downshift"}
        if e == 3:
            ok &= m["pedestal"] == m["carbonate"] == "Unshifted" and {m["pedigree"], m["pedicure"]} == {"Upshift", "Downshift"}
    checks.internal("counterbalancing as described: opposite shifts on the two test words, fillers unshifted", bool(ok))
    sh = a.pert_value
    ramp = a[a.block == "Ramp"]
    checks.internal("shift 0 in Baseline and Washout, +-125 mel in Hold for shifted words, 0-125 mel in Ramp",
                    bool((sh[a.block.isin(["Baseline", "Washout"])] == 0).all()
                         and (sh[(a.block == "Hold") & (a.condition != "Unshifted")].abs() == 125).all()
                         and (sh[a.condition == "Unshifted"] == 0).all()
                         and ramp.pert_value.abs().between(0, 125).all()
                         and all((g.pert_value.abs().diff().dropna() >= 0).all()
                                 for _, g in ramp.groupby(["subject", "exp", "stimulus"]))))
    rng = dict(f1_mel=(300, 1100), f0_hz=(70, 320), duration_ms=(10, 700), intensity_au=(0, 100))
    for c, (lo, hi) in rng.items():
        v = a[c].astype(float)
        checks.internal(f"{c} present on every row and within {lo}-{hi}", bool(v.notna().all() and v.between(lo, hi).all()),
                        f"{v.min():.3f}-{v.max():.3f}")


def published_checks(a, checks):
    import statsmodels.formula.api as smf

    # trials the authors removed, as a share of the design, averaged over participants (Methods, Data processing)
    for e, rep in ((1, 1.22), (2, 0.83)):
        pct = 100 * (n_trials(e) - a[a.exp == e].groupby("subject").size()) / n_trials(e)
        checks.published(f"Experiment {e}: mean % of trials removed per participant", pct.mean(), rep, 0.01,
                         f"{PAPER_SRC}, Methods, Data processing")
    # F1 change from baseline, the authors' way (1a/2a/3a_Exp*_Plot.Rmd): chunks of 10 blocks, baseline = mean of the
    # last Baseline chunk per participant and word, carbonate dropped after normalising; analysed chunks per experiment
    a = a.copy()
    a["chunk"] = np.ceil(a.trial / (a.exp.map(lambda e: len(EXPS[e]["words"])) * 10)).astype(int)
    lastb = a[a.block == "Baseline"].groupby("exp").chunk.max()
    base = a[a.chunk == a.exp.map(lastb)].groupby(["subject", "exp", "stimulus"]).f1_mel.mean().rename("base")
    a = a.join(base, on=["subject", "exp", "stimulus"])
    a["dF1"] = a.f1_mel - a.base
    in_win = (a.chunk == a.exp.map(lambda e: EXPS[e]["win"][0])) | (a.chunk == a.exp.map(lambda e: EXPS[e]["win"][1]))
    w = a[~a.excluded_by_authors & in_win]
    se_rep = {1: dict(Upshift=2.34, Downshift=2.63, Unshifted=2.45), 2: dict(Upshift=2.29, Downshift=2.33, Unshifted=2.55),
              3: dict(Upshift=1.82, Downshift=1.57, Unshifted=1.45)}
    for e in (1, 2, 3):
        for c, rep in se_rep[e].items():
            x = w.dF1[(w.exp == e) & (w.condition == c)]
            checks.published(f"Experiment {e}: SE of F1 change, {c}, analysed Hold and Washout blocks (mel)",
                             x.std(ddof=1) / math.sqrt(len(x)), rep, 0.01, f"{PAPER_SRC}, Results, Experiment {e}")
    cell = w[w.condition != "Unshifted"].groupby(["subject", "exp", "block", "condition"]).agg(
        dF1=("dF1", "mean"), dur=("duration_ms", "mean")).unstack("condition")
    d4 = pd.DataFrame({"Diff": cell[("dF1", "Downshift")] - cell[("dF1", "Upshift")],
                       "Duration": (cell[("dur", "Downshift")] + cell[("dur", "Upshift")]) / 2}).reset_index()
    h = d4[(d4.exp == 1) & (d4.block == "Hold")]
    wo = d4[(d4.exp == 1) & (d4.block == "Washout")]
    checks.published("Experiment 1: % of participants with F1 change Downshift > Upshift, Hold", 100 * (h.Diff > 0).mean(), 100,
                     0.5, f"{PAPER_SRC}, Results, Experiment 1")
    checks.published("Experiment 1: % of participants with F1 change Downshift > Upshift, Washout", 100 * (wo.Diff > 0).mean(), 93,
                     0.5, f"{PAPER_SRC}, Results, Experiment 1")
    x = d4.Diff[d4.exp == 2]
    checks.published("Experiment 2: differential adaptation (Downshift - Upshift), mean (mel)", x.mean(), 17.39, 0.01,
                     f"{PAPER_SRC}, Results, Comparisons of differential adaptation across experiments (Figure 6)")
    for e, rep in ((1, 6.77), (2, 3.58), (3, 3.15)):
        x = d4.Diff[d4.exp == e]
        checks.published(f"Experiment {e}: differential adaptation, SE over participant x phase means (mel)",
                         x.std(ddof=1) / math.sqrt(len(x)), rep, 0.01,
                         f"{PAPER_SRC}, Results, Comparisons of differential adaptation across experiments (Figure 6)")
    # the paper's selected models (4b_Exp123_Size_Main.Rmd), REML with a random intercept per participant; the two
    # people in Experiments 1 and 3 share one intercept, which is why these means differ from the plain means
    d4["cell"] = d4.exp.astype(str) + "_" + d4.block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("Diff ~ 0 + C(cell)", d4, groups=d4["subject"]).fit(reml=True)
        md = smf.mixedlm("Duration ~ 0 + C(exp)", d4, groups=d4["subject"]).fit(reml=True)
    for e, rep in ((1, 62.17), (3, 8.44)):
        est = (m.fe_params[f"C(cell)[{e}_Hold]"] + m.fe_params[f"C(cell)[{e}_Washout]"]) / 2
        checks.published(f"Experiment {e}: differential adaptation, model mean (Diff ~ Experiment * Phase + (1 | Speaker)) (mel)",
                         est, rep, 0.01, f"{PAPER_SRC}, Results, Comparisons of differential adaptation across experiments")
    for e, rep in ((1, 247.36), (2, 121.79), (3, 61.89)):
        checks.published(f"Experiment {e}: vowel duration of the shifted words, model mean (ms)", md.fe_params[f"C(exp)[{e}]"],
                         rep, 0.01, f"{PAPER_SRC}, Results, Comparisons of differential adaptation across experiments")


def main():
    checks = stdlib.Checks()
    rows = build(checks)
    a = pd.DataFrame(rows)
    a["exp"] = a.experiment.str[-1].astype(int)
    internal_checks(a, checks)
    published_checks(a, checks)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Merged the authors' per-trial F1, f0 and intensity files into one table, added the feedback shift on each "
                 "trial from the authors' experiment code, and replaced participant identifiers with codes."),
        how=[
            "One row is one word spoken by one participant in Experiment 1, 2 or 3, in the authors' order; trial is the "
            "authors' trial number, and its gaps are trials they removed for formant-tracking errors or misproductions.",
            "f1_mel (F1 in mels), f0_hz, intensity_au (the authors' linear amplitude measure, not dB) and duration_ms are the "
            "authors' per-trial values for the first vowel of the word, averaged over the window given in measure_window.",
            "pert_value and shift_f1 are the F1 shift heard on that trial, in mels, positive when F1 was raised; they were "
            "rebuilt from the authors' experiment code: 0 in Baseline and Washout, 30 equal steps from 0 to 125 across the "
            "Ramp blocks, 125 in Hold, signed by the word's condition (Upshift +, Downshift -, Unshifted 0).",
            "subject is one participant's sequence in one experiment and person is the participant, so the two people who "
            "took part in both Experiments 1 and 3 have two subject codes and one person code each; rows of the filler word "
            "carbonate, which the authors did not analyse, are marked excluded_by_authors.",
        ],
        unit_note="spoken word (utterance) of one participant",
        differences=[
            "Trials removed per speaker: the shared files give a mean of 0.48% in Experiment 3 (the paper reports 0.44%; "
            "the SD and range match) and a maximum of 5.00% in Experiment 1 (the paper reports 3.52%).",
            "Share of speakers who adapted: the shared files give 85% in the Experiment 2 Hold phase (paper: 95%) and 75% and "
            "95% in Experiment 3 (paper: 95% and 90%).",
        ],
    )
    for r in rows:  # subject = one experiment's sequence, person = the participant (the stdlib re-codes both)
        r["person"] = r["subject"]
        r["subject"] = f"{r['subject']}|{r['experiment']}"
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
