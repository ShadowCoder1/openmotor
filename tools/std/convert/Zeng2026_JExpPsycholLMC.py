"""Zeng, Parrell & Niziolek (2026), J Exp Psychol Learn Mem Cogn: opposing 2-D formant shifts on the two syllables of
"bedhead" (Experiment 1), and transfer of that learning to other words with /ɛ/ (Experiment 2).

Raw files, raw_extra/Zeng2026_JExpPsycholLMC/ (OSF d4t6s; the same files are in github.com/blab-lab/singleWord):
  1_Exp1.RData           R workspace of 1a/1b_SingleWord_Exp1_*.Rmd; data1 = the per-trial formant table (one row per
                         syllable of a kept trial: Speaker, Trial, meanF1_mel, meanF2_mel, Word, Item, Duration, Phase,
                         List, Shift ...); summary0 = each speaker's projected F1 and F2 shift (mels) per syllable
  2_Exp2_Training.RData  the same for Experiment 2, restricted by the authors to Baseline, Ramp, Hold, Washout
  3_Exp2_Transfer.RData  data1 = Experiment 2 Post-training (Transfer2) rows of the same table, with F1_Ref = the
                         speaker's mean F1 for that word and syllable in Pre-training (Transfer1); data2 = the F2 subset with F2_Ref
  4_Exp2_Prosody.RData   data1a = median f0 and data2 = mean and maximum intensity per syllable of the disyllabic words
                         in Transfer1 and Transfer2 (5a/5b_SingleWord_Exp2_Transfer_*.Rmd)
What the values are (paper, Methods "Data processing"; Export_*.m and the Rmd files, read as text): meanF1_mel and
meanF2_mel are F1 and F2 averaged over 25-75% of the vowel /ɛ/ of that syllable, in mels; Duration the vowel duration
in ms; F0_Median the median f0 (Hz) over the vowel; Int_Mean and Int_Max the mean and maximum of the wave_viewer
intensity track over the vowel (linear, uncalibrated). Trials removed for tracking errors or misproductions are absent.
Shift schedule (run_simonSingleWord_expt.m, run_simonSingleWord_v2_expt.m): expt.shiftMags = 0 in Baseline (and
Transfer), linspace(0, 125, 30) over the Ramp trials, 125 in Hold, 0 in Washout; Audapter shifts F1 and F2 in mels
(bMelShift = 1) along the speaker's angle per syllable (fmtPertPhi), so the F1 and F2 parts are summary0 F1 and F2
scaled by the trial's magnitude / 125. Feedback mode per trial (expt.allNoise, p.fb): Experiment 1 speech + noise on
every trial; Experiment 2 noise only in Transfer1/Transfer2, speech only in Baseline and Washout, speech + noise in Ramp
and Hold.
Run from the site folder: python3 -I tools/std/convert/Zeng2026_JExpPsycholLMC.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zeng2026_JExpPsycholLMC"
SRC = "Zeng, Parrell & Niziolek, manuscript PsyArXiv 63gaj v2 (first revision)"
HOLD = 125.0
# first Ramp trial, last Hold trial; Ramp = 30 trials (linspace(0, 125, 30)), from the run scripts and the data
DESIGN = {1: dict(ramp0=31, hold_end=210, n=20, trials=240),
          2: dict(ramp0=101, hold_end=330, n=20, trials=420)}
PHASE = {"Baseline": "baseline", "Ramp": "learning", "Hold": "learning", "Washout": "washout",
         "Transfer1": "test", "Transfer2": "transfer"}
SYLL = {("bedhead", 1): "bed", ("bedhead", 2): "head", ("bedspread", 1): "bed", ("bedspread", 2): "spread",
        ("breastfed", 1): "breast", ("breastfed", 2): "fed", ("deathbed", 1): "death", ("deathbed", 2): "bed",
        ("headrest", 1): "head", ("headrest", 2): "rest", ("redhead", 1): "red", ("redhead", 2): "head",
        ("bed", 1): "bed", ("head", 1): "head"}
DEVICE = "Sennheiser MKE 600 microphone, Scarlett 2i2 sound card, Audapter, Beyerdynamic DT 770 PRO headphones"
NOISE_MIX = "speech-shaped noise, about 60 dB SPL, mixed with the speech playback"
NOISE_ONLY = "speech-shaped noise only, about 77 dB SPL, no speech playback"
NOISE_NONE = "none (speech playback only; Audapter feedback mode 1 in the experiment code)"
W_FMT = "F1 and F2: mean over 25-75% of the vowel"
W_PROS = "f0: median over the vowel; intensity: mean and maximum over the vowel"
W_REF = ("F1 and F2: mean over the participant's Transfer1 repetitions of this syllable, each the mean over 25-75% of "
         "the vowel")


def shift_mag(e, t):
    d = DESIGN[e]
    if d["ramp0"] <= t < d["ramp0"] + 30:
        return HOLD * (t - d["ramp0"]) / 29
    if d["ramp0"] + 30 <= t <= d["hold_end"]:
        return HOLD
    return 0.0


def segment(word, item):
    s = SYLL[(word, item)]
    if word in ("bed", "head"):
        return f"vowel /ɛ/ ({s})"
    return f"vowel /ɛ/ of the {'first' if item == 1 else 'second'} syllable ({s})"


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def base_row(e, spk, lst):
    return {"dataset": ID, "study": ID, "experiment": f"Experiment {e}", "group": lst, "subject": spk,
            "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEVICE, "session": 1, "time_s": "", "task": "speech",
            "excluded_by_authors": False, "f1_hz": "", "f2_hz": "", "feedback_latency_ms": 18}


def load(checks):
    raw = stdlib.raw_dir(ID)
    R = {k: pyreadr.read_r(str(raw / f)) for k, f in (("E1", "1_Exp1.RData"), ("E2", "2_Exp2_Training.RData"),
                                                       ("T", "3_Exp2_Transfer.RData"), ("P", "4_Exp2_Prosody.RData"))}
    out = {}
    for k in ("E1", "E2"):
        d = R[k]["data1"].copy()
        for c in ("Speaker", "Word", "Phase", "List", "Shift"):
            d[c] = d[c].astype(str)
        out[k] = d
        s0 = R[k]["summary0"].copy()
        for c in ("Speaker", "List", "Shift", "F1_Shift", "F2_Shift"):
            s0[c] = s0[c].astype(str)
        s0["Item"] = s0["Syllable"].astype(int)
        out[k + "_shift"] = s0
        d2 = R[k]["data2"]
        m = d2.merge(d, on=["Speaker", "Trial", "Item"], how="left", suffixes=("", "_1"))
        checks.internal(f"Experiment {k[1]}: the authors' F2 table (data2) is a subset of the F1 table with the same values",
                        bool(m.meanF1_mel_1.notna().all() and (m.meanF2_mel == m.meanF2_mel_1).all()), f"{len(d2)} rows")
        out[k + "_f2spk"] = set(d2.Speaker.astype(str))
    t = R["T"]["data1"].copy()
    for c in ("Speaker", "Word", "Phase", "List", "Type1", "Syllable", "Content"):
        t[c] = t[c].astype(str)
    out["T"] = t
    t2 = R["T"]["data2"].copy()
    for c in ("Speaker", "Content"):
        t2[c] = t2[c].astype(str)
    m = t2.merge(t, on=["Speaker", "Trial", "Item"], how="left", suffixes=("", "_1"))
    checks.internal("Experiment 2 Post-training: the F2 table is a subset of the F1 table with the same values",
                    bool(m.meanF1_mel_1.notna().all() and (m.meanF2_mel == m.meanF2_mel_1).all()), f"{len(t2)} rows")
    out["T_f2ref"] = t2.groupby(["Speaker", "Content"]).F2_Ref.agg(["first", "nunique"])
    out["T_f1ref"] = t.groupby(["Speaker", "Content"]).F1_Ref.agg(["first", "nunique"])
    checks.internal("one Pre-training reference per speaker, word and syllable (F1_Ref, F2_Ref)",
                    bool((out["T_f1ref"]["nunique"] == 1).all() and (out["T_f2ref"]["nunique"] == 1).all()),
                    f"{len(out['T_f1ref'])} F1 and {len(out['T_f2ref'])} F2 references")
    f0 = R["P"]["data1a"].copy()
    it = R["P"]["data2"].copy()
    for d in (f0, it):
        for c in ("Speaker", "Word", "Syllable", "Phase"):
            d[c] = d[c].astype(str)
    key = ["Speaker", "Trial", "Word", "Syllable"]
    pr = f0[key + ["Phase", "Duration", "F0_Median"]].merge(it[key + ["Phase", "Duration", "Int_Mean", "Int_Max"]],
                                                            on=key, how="outer", suffixes=("", "_i"), indicator=True)
    both = pr[pr._merge == "both"]
    checks.internal("f0 and intensity tables agree on phase and vowel duration where both hold a syllable",
                    bool((both.Phase == both.Phase_i).all() and (both.Duration - both.Duration_i).abs().max() < 1e-9),
                    f"{len(both)} shared rows, {int((pr._merge == 'right_only').sum())} intensity-only, "
                    f"{int((pr._merge == 'left_only').sum())} f0-only")
    pr["Phase"] = pr.Phase.where(pr.Phase.notna() & (pr.Phase != "nan"), pr.Phase_i)
    pr["Duration"] = pr.Duration.fillna(pr.Duration_i)
    pr["Item"] = pr.Syllable.map({"A": 1, "B": 2})
    out["P"] = pr.drop(columns=["Phase_i", "Duration_i", "_merge"])
    return out


def build(D, checks):
    rows = []
    # Experiment 1 and the Experiment 2 training phases: the authors' tables, in their order
    shift = {}
    for e, k in ((1, "E1"), (2, "E2")):
        s0 = D[k + "_shift"]
        for r in s0.itertuples(index=False):
            shift[(e, r.Speaker, r.Item)] = (float(r.F1), float(r.F2), r.Shift)
    lists = {}
    for e, k in ((1, "E1"), (2, "E2")):
        d = D[k]
        ok = True
        for r in d.itertuples(index=False):
            f1c, f2c, lab = shift[(e, r.Speaker, int(r.Item))]
            ok &= {"IH": "/ɪ/-shift", "AE": "/æ/-shift"}[lab] == r.Shift
            lists[(e, r.Speaker)] = r.List
        checks.internal(f"Experiment {e}: each row's shift label matches the speaker's shift table (summary0)", bool(ok))
    tr = {}
    for e, k in ((1, "E1"), (2, "E2")):
        for r in D[k].itertuples(index=False):
            t = int(r.Trial)
            mag = shift_mag(e, t)
            f1c, f2c, _ = shift[(e, r.Speaker, int(r.Item))]
            s1 = 0.0 if mag == 0 else mag / HOLD * f1c
            s2 = 0.0 if mag == 0 else mag / HOLD * f2c
            noise = NOISE_MIX if (e == 1 or r.Phase in ("Ramp", "Hold")) else NOISE_NONE
            row = base_row(e, r.Speaker, r.List)
            row.update({"block": r.Phase, "trial": t, "phase": PHASE[r.Phase], "unit": "utterance", "condition": r.Shift,
                        "pert_type": "formant_shift" if mag else "none", "pert_value": float(mag), "pert_unit": "mel",
                        "feedback": "altered" if mag else "normal",
                        "stimulus": r.Word, "segment": segment(r.Word, int(r.Item)), "measure_window": W_FMT,
                        "duration_ms": float(r.Duration), "f0_hz": "", "f1_mel": float(r.meanF1_mel),
                        "f2_mel": float(r.meanF2_mel), "intensity_au": "", "intensity_max_au": "",
                        "shift_f1": s1, "shift_f2": s2, "masking_noise": noise,
                        "_e": e, "_item": int(r.Item), "_src": "train"})
            tr.setdefault((e, r.Speaker), []).append(row)
    # Experiment 2 transfer phases: Post-training formants merged with the prosody table; Pre-training prosody only,
    # plus the Pre-training formant means the authors shared as references
    t = D["T"]
    type1 = {(w, i): v for (w, i), v in t.groupby(["Word", "Item"]).Type1.first().items()}
    P = D["P"].set_index(["Speaker", "Trial", "Word", "Item"])
    used = set()
    n_post_pros = 0
    for r in t.itertuples(index=False):
        key = (r.Speaker, int(r.Trial), r.Word, int(r.Item))
        p = P.loc[key] if key in P.index else None
        if p is not None:
            used.add(key)
            n_post_pros += 1
            ok_dur = abs(float(p.Duration) - float(r.Duration)) < 1e-9
            if not ok_dur:
                raise AssertionError(("duration mismatch", key))
        row = base_row(2, r.Speaker, r.List)
        row.update({"block": "Transfer2", "trial": int(r.Trial), "phase": PHASE["Transfer2"], "unit": "utterance",
                    "condition": r.Type1, "pert_type": "none", "pert_value": 0.0, "pert_unit": "mel", "feedback": "masked",
                    "stimulus": r.Word, "segment": segment(r.Word, int(r.Item)),
                    "measure_window": W_FMT + ("; " + W_PROS if p is not None else ""),
                    "duration_ms": float(r.Duration), "f0_hz": float(p.F0_Median) if p is not None else "",
                    "f1_mel": float(r.meanF1_mel), "f2_mel": float(r.meanF2_mel),
                    "intensity_au": float(p.Int_Mean) if p is not None else "",
                    "intensity_max_au": float(p.Int_Max) if p is not None else "",
                    "shift_f1": 0.0, "shift_f2": 0.0, "masking_noise": NOISE_ONLY,
                    "_e": 2, "_item": int(r.Item), "_src": "post"})
        tr[(2, r.Speaker)].append(row)
    pre = D["P"][D["P"].Phase == "Pre"]
    post_left = D["P"][D["P"].Phase == "Post"]
    checks.internal("every Post-training prosody row matches a Post-training formant row",
                    n_post_pros == len(post_left) and len(used) == len(post_left), f"{n_post_pros} of {len(post_left)}")
    for r in pre.itertuples(index=False):
        row = base_row(2, r.Speaker, lists[(2, r.Speaker)])
        row.update({"block": "Transfer1", "trial": int(r.Trial), "phase": PHASE["Transfer1"], "unit": "utterance",
                    "condition": type1[(r.Word, int(r.Item))], "pert_type": "none", "pert_value": 0.0, "pert_unit": "mel",
                    "feedback": "masked", "stimulus": r.Word, "segment": segment(r.Word, int(r.Item)),
                    "measure_window": W_PROS, "duration_ms": float(r.Duration),
                    "f0_hz": "" if pd.isna(r.F0_Median) else float(r.F0_Median), "f1_mel": "", "f2_mel": "",
                    "intensity_au": float(r.Int_Mean), "intensity_max_au": float(r.Int_Max),
                    "shift_f1": 0.0, "shift_f2": 0.0, "masking_noise": NOISE_ONLY,
                    "_e": 2, "_item": int(r.Item), "_src": "pre"})
        tr[(2, r.Speaker)].append(row)
    f1ref, f2ref = D["T_f1ref"]["first"], D["T_f2ref"]["first"]
    words = ["bedhead", "bed", "head", "redhead", "bedspread", "headrest", "deathbed", "breastfed"]  # expt.words order
    for (spk, content), v in f1ref.items():
        word, item = content.rsplit("_", 1)
        row = base_row(2, spk, lists[(2, spk)])
        row.update({"block": "Transfer1", "trial": "", "phase": PHASE["Transfer1"], "unit": "participant summary",
                    "condition": type1[(word, int(item))], "pert_type": "none", "pert_value": 0.0, "pert_unit": "mel",
                    "feedback": "masked", "stimulus": word, "segment": segment(word, int(item)), "measure_window": W_REF,
                    "duration_ms": "", "f0_hz": "", "f1_mel": float(v),
                    "f2_mel": float(f2ref[(spk, content)]) if (spk, content) in f2ref.index else "",
                    "intensity_au": "", "intensity_max_au": "", "shift_f1": 0.0, "shift_f2": 0.0,
                    "masking_noise": NOISE_ONLY, "_e": 2, "_item": int(item), "_src": "ref",
                    "_order": words.index(word)})
        tr[(2, spk)].append(row)
    # order: experiment, the authors' speaker order, then trial and syllable; the reference means close Transfer1
    for e, k in ((1, "E1"), (2, "E2")):
        for spk in D[k].Speaker.unique():
            seq = tr[(e, spk)]

            def sk(r):
                if r["_src"] == "ref":
                    return (80.5, r["_order"], r["_item"])
                return (r["trial"], 0, r["_item"])
            seq.sort(key=sk)
            rows.extend(seq)
    return rows


def se(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return x.std(ddof=1) / math.sqrt(len(x))


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return abs(a.mean() - b.mean()) / sp


def boot_adapters(g, col, item_hi, item_lo, rng, n=10000):
    """The authors' adapter test: ezBoot(wid = Trial, within = Item, resample_within = FALSE), then ezPlot2 95% CIs.
    ezResample draws trials with replacement but keeps their Trial ids, so ezBoot's per-Trial cell means merge the
    copies of a trial drawn twice: each bootstrap mean is the mean over the distinct trials drawn. Returns
    Lo(item_hi) - Hi(item_lo), positive for an adapter (the authors' DownUp)."""
    trials = np.sort(g.trial.astype(int).unique())
    pos = {t: k for k, t in enumerate(trials)}
    m = np.full((2, len(trials)), np.nan)
    for t, i, v in zip(g.trial.astype(int), g._item, g[col]):
        m[int(i) - 1, pos[t]] = v
    draw = rng.integers(0, len(trials), size=(n, len(trials)))
    drawn = np.zeros((n, len(trials)), bool)
    np.put_along_axis(drawn, draw, True, axis=1)
    means = {}
    for i in (1, 2):
        v = m[i - 1]
        ok = ~np.isnan(v)
        means[i] = (drawn * np.where(ok, v, 0)).sum(1) / (drawn & ok).sum(1)
    return np.quantile(means[item_hi], 0.025) - np.quantile(means[item_lo], 0.975)


def checks_all(a, D, checks):
    a = a.copy()
    a["e"] = a._e
    utt = a[a.unit == "utterance"]
    # internal
    n = {(1, "train"): 9549, (2, "train"): 10285, (2, "post"): 2710, (2, "pre"): 2340, (2, "ref"): 280}
    got = a.groupby(["e", "_src"]).size().to_dict()
    checks.internal("rows per source table as shared (Exp 1 9549; Exp 2 training 10285, Post-training 2710, "
                    "Pre-training prosody 2340, Pre-training reference means 280)", got == n, str(got))
    checks.internal("40 people, no participant number in both experiments",
                    a.subject.nunique() == 40 and not (set(a.subject[a.e == 1]) & set(a.subject[a.e == 2])))
    checks.internal("subject x trial x syllable unique among utterance rows",
                    not utt.duplicated(["subject", "trial", "_item"]).any())
    phase_tr = {1: {"Baseline": (1, 30), "Ramp": (31, 60), "Hold": (61, 210), "Washout": (211, 240)},
                2: {"Transfer1": (1, 80), "Baseline": (81, 100), "Ramp": (101, 130), "Hold": (131, 330),
                    "Transfer2": (331, 410), "Washout": (411, 420)}}
    ok = all(phase_tr[e][b][0] <= t <= phase_tr[e][b][1] for e, b, t in zip(utt.e, utt.block, utt.trial))
    checks.internal("the authors' phase labels match the design's trial ranges (Exp 2: 80 Transfer1, 20 Baseline, "
                    "30 Ramp, 200 Hold, 80 Transfer2, 10 Washout)", bool(ok))
    mag = np.hypot(a.shift_f1.astype(float), a.shift_f2.astype(float))
    checks.internal("F1 and F2 parts of the shift add up to its size; 125 mel in Hold, 0 outside Ramp and Hold",
                    bool(np.allclose(mag, a.pert_value.astype(float), atol=1e-6)
                         and (a.pert_value[a.block == "Hold"] == 125).all()
                         and (a.pert_value[~a.block.isin(["Ramp", "Hold"])] == 0).all()))
    for c, (lo, hi) in dict(f1_mel=(350, 1400), f2_mel=(900, 2000), f0_hz=(45, 320), duration_ms=(20, 800),
                            intensity_au=(0, 250), intensity_max_au=(0, 400)).items():
        v = pd.to_numeric(a[c], errors="coerce").dropna()
        checks.internal(f"{c} within {lo}-{hi}", bool(v.between(lo, hi).all()), f"{v.min():.3f}-{v.max():.3f}, n={len(v)}")
    pub = checks.published
    for e in (1, 2):
        pub(f"Experiment {e}: participants", a.subject[a.e == e].nunique(), 20, 0, f"{SRC}, Methods, Participants")
    # trials removed (Experiment 1), counted per syllable measurement
    e1 = utt[utt.e == 1]
    pct = 100 * (480 - e1.groupby("subject").size()) / 480  # 240 trials x 2 syllables
    pub("Experiment 1: % of trials removed per participant, mean", pct.mean(), 0.53, 0.005, f"{SRC}, Methods, Data processing")
    pub("Experiment 1: % of trials removed per participant, SD", pct.std(ddof=1), 1.49, 0.005, f"{SRC}, Methods, Data processing")
    pub("Experiment 1: % of trials removed per participant, maximum", pct.max(), 6.67, 0.005, f"{SRC}, Methods, Data processing")
    # perturbations projected on F1 and F2 (Hold rows carry the full shift)
    hold = a[a.block == "Hold"].drop_duplicates(["subject", "_item"])
    sp = hold.groupby(["e", "subject", "condition"])[["shift_f1", "shift_f2"]].first().reset_index()
    f2opp = {}
    for e in (1, 2):
        s = sp[sp.e == e].pivot(index="subject", columns="condition", values="shift_f2")
        f2opp[e] = set(s.index[(s["/ɪ/-shift"] > 0) & (s["/æ/-shift"] < 0)])
        pub(f"Experiment {e}: participants whose projected F2 shifts were opposing (/ɪ/-shift up, /æ/-shift down), "
            "the F2 analyses' sample", len(f2opp[e]), {1: 13, 2: 15}[e], 0, f"{SRC}, Methods, Experiment design")
        checks.internal(f"Experiment {e}: that F2 sample is the authors' F2 table's speakers",
                        f2opp[e] == D[f"E{e}_f2spk"])
    pub("mean absolute F1 part of the shift, both experiments and directions (mel)", sp.shift_f1.abs().mean(), 107.62, 0.005,
        f"{SRC}, Methods, Experiment design")
    f2s = sp[sp.subject.isin(f2opp[1] | f2opp[2])]
    pub("mean absolute F2 part of the shift, both experiments and directions, participants of the F2 analyses (mel)",
        f2s.shift_f2.abs().mean(), 56.40, 0.005, f"{SRC}, Methods, Experiment design")
    # adaptation in training: change from the last 10 Baseline trials, analysed blocks (last 10 Hold, first 10 Washout)
    tr = utt[utt.block.isin(["Baseline", "Ramp", "Hold", "Washout"])].copy()
    ref_tr = {1: (21, 30), 2: (91, 100)}
    win = {1: ((201, 210), (211, 220)), 2: ((321, 330), (411, 420))}
    for fm in ("f1_mel", "f2_mel"):
        tr[fm] = tr[fm].astype(float)
        b = tr[[ref_tr[e][0] <= t <= ref_tr[e][1] for e, t in zip(tr.e, tr.trial)]].groupby(["subject", "_item"])[fm].mean()
        tr["d" + fm] = tr[fm] - tr.join(b.rename("ref"), on=["subject", "_item"])["ref"]
    tr["win"] = [("Hold" if win[e][0][0] <= t <= win[e][0][1] else "Washout" if win[e][1][0] <= t <= win[e][1][1] else "")
                 for e, t in zip(tr.e, tr.trial)]
    w = tr[tr.win != ""]
    se_rep = {1: {"/æ/-shift": 2.59}, 2: {"/ɪ/-shift": 1.99, "/æ/-shift": 2.33}}  # Exp 1 /ɪ/: see differences
    for e in (1, 2):
        we = w[w.e == e]
        for c, rep in se_rep[e].items():
            pub(f"Experiment {e}: SE of F1 change, {c}, analysed Hold and Washout trials (mel)", se(we.df1_mel[we.condition == c]),
                rep, 0.005, f"{SRC}, Results, Experiment {e}")
        pub(f"Experiment {e}: Cohen's d, F1 change /ɪ/-shift vs /æ/-shift", cohen_d(we.df1_mel[we.condition == "/ɪ/-shift"],
            we.df1_mel[we.condition == "/æ/-shift"]), {1: 1.10, 2: 1.60}[e], 0.005, f"{SRC}, Results, Experiment {e}")
        cell = we.groupby(["subject", "win", "condition"])[["df1_mel", "df2_mel"]].mean().unstack("condition")
        rep1 = {(1, "Hold"): 90, (2, "Hold"): 100, (2, "Washout"): 95}  # Exp 1 Washout: see differences
        rep2 = {(1, "Hold"): 100, (1, "Washout"): 100, (2, "Hold"): 100}  # Exp 2 Washout: see differences
        for ph in ("Hold", "Washout"):
            c1 = cell.xs(ph, level="win")
            p1 = 100 * (c1[("df1_mel", "/ɪ/-shift")] > c1[("df1_mel", "/æ/-shift")]).mean()
            c2 = c1[c1.index.isin(f2opp[e])]
            p2 = 100 * (c2[("df2_mel", "/ɪ/-shift")] < c2[("df2_mel", "/æ/-shift")]).mean()
            print(f"Exp {e} {ph}: F1 {p1:.1f}% F2 {p2:.1f}%")
            if (e, ph) in rep1:
                pub(f"Experiment {e}: % of participants with more positive F1 change for /ɪ/-shift than /æ/-shift, {ph}", p1,
                    rep1[(e, ph)], 0.5, f"{SRC}, Results, Experiment {e}")
            if (e, ph) in rep2:
                pub(f"Experiment {e}: % of the F2 sample with more negative F2 change for /ɪ/-shift than /æ/-shift, {ph}",
                    p2, 100, 0.5, f"{SRC}, Results, Experiment {e}")
    w2 = w[w.e == 2]
    for (ph, c), rep in {("Hold", "/ɪ/-shift"): 2.73, ("Hold", "/æ/-shift"): 3.29, ("Washout", "/ɪ/-shift"): 2.75,
                         ("Washout", "/æ/-shift"): 3.27}.items():
        pub(f"Experiment 2: SE of F1 change, {c}, {ph} (mel)", se(w2.df1_mel[(w2.win == ph) & (w2.condition == c)]), rep, 0.005,
            f"{SRC}, Results, Experiment 2, Training results")
    # transfer: Post-training minus the Pre-training mean of the same word and syllable, sign-flipped for shiftAEshiftIH
    ref = a[a._src == "ref"].set_index(["subject", "stimulus", "_item"])[["f1_mel", "f2_mel"]]
    post = utt[utt._src == "post"].copy()
    post = post.join(ref.rename(columns={"f1_mel": "r1", "f2_mel": "r2"}), on=["subject", "stimulus", "_item"])
    flip = np.where(post.group == "shiftAEshiftIH", -1.0, 1.0)
    post["t1"] = (post.f1_mel.astype(float) - post.r1.astype(float)) * flip
    post["t2"] = (post.f2_mel.astype(float) - pd.to_numeric(post.r2, errors="coerce")) * flip
    post["syl"] = [SYLL[(wd, i)] for wd, i in zip(post.stimulus, post._item)]
    bh = post[post.stimulus == "bedhead"]
    for i, rep in ((1, 2.33), (2, 2.87)):
        pub(f"Experiment 2 Post-training: SE of F1 transfer in bedhead, syllable {i}, all participants (mel)",
            se(bh.t1[bh._item == i]), rep, 0.005, f"{SRC}, Results, The maintenance of adaptation")
    pub("Experiment 2 Post-training: Cohen's d, bedhead syllable 1 vs 2, F1 transfer", cohen_d(bh.t1[bh._item == 1],
        bh.t1[bh._item == 2]), 1.32, 0.005, f"{SRC}, Results, The maintenance of adaptation")
    rng = np.random.default_rng(444)
    du1 = {s: boot_adapters(g, "t1", 1, 2, rng) for s, g in bh.groupby("subject")}
    ad1 = {s for s, v in du1.items() if v > 0}
    bh2 = bh[bh.subject.isin(f2opp[2])]
    # F2: the second syllable's CI must lie above the first's (Lo[2] - Hi[1] > 0)
    du2 = {s: boot_adapters(g, "t2", 2, 1, rng) for s, g in bh2.groupby("subject")}
    ad2 = {s for s, v in du2.items() if v > 0}
    m1 = min(abs(v) for v in du1.values())
    m2 = min(abs(v) for v in du2.values())
    pub("Experiment 2: adapters in F1 (bootstrap 95% CIs of the two bedhead syllables do not overlap)", len(ad1), 17, 0,
        f"{SRC}, Methods, Statistical analysis; Results")
    pub("Experiment 2: adapters in F2, among the F2 sample", len(ad2), 11, 0, f"{SRC}, Methods, Statistical analysis")
    checks.internal("adapter counts do not hinge on the bootstrap draw (no participant within 1 mel of the cut-off)",
                    m1 > 1 and m2 > 1, f"closest margins {m1:.2f} and {m2:.2f} mel")
    shp = sp[sp.e == 2]
    pub("Table 1: mean F1 perturbation of the F1 adapters (mel)", shp.shift_f1[shp.subject.isin(ad1)].abs().mean(), 110.00,
        0.005, f"{SRC}, Table 1")
    print("Table 1 F2 adapters mean F2 shift:", shp.shift_f2[shp.subject.isin(ad2)].abs().mean())  # paper 58.90: see differences
    # trained syllables among the F1 adapters (data1b in 3c_SingleWord_Exp2_Transfer_Model_F1.Rmd)
    ta = post[post.subject.isin(ad1)]
    tb = ta[ta.syl.isin(["bed", "head"])]
    se_tr = {("bedhead", "bed"): 2.39, ("bedhead", "head"): 2.86, ("bedspread", "bed"): 2.53, ("redhead", "head"): 2.66,
             ("bed", "bed"): 2.68, ("head", "head"): 2.43, ("deathbed", "bed"): 2.80, ("headrest", "head"): 2.77}
    for (wd, sy), rep in se_tr.items():
        pub(f"Experiment 2 F1 adapters: SE of F1 transfer, '{sy}' in {wd} (mel)",
            se(tb.t1[(tb.stimulus == wd) & (tb.syl == sy)]), rep, 0.005, f"{SRC}, Results, Comparison of the trained word and the trained syllables")
    pub("Experiment 2 F1 adapters: Cohen's d, all 'bed' vs all 'head' syllables", cohen_d(tb.t1[tb.syl == "bed"],
        tb.t1[tb.syl == "head"]), 0.77, 0.005, f"{SRC}, Results, Comparison of the trained word and the trained syllables")
    for (w1, w2_), rep in {(("bedhead", "bed"), ("bedhead", "head")): 1.60, (("bedspread", "bed"), ("redhead", "head")): 0.99,
                           (("deathbed", "bed"), ("headrest", "head")): 0.26, (("bed", "bed"), ("head", "head")): 0.43,
                           (("bedhead", "bed"), ("bedspread", "bed")): 0.23, (("bedhead", "head"), ("redhead", "head")): 0.39,
                           (("bedhead", "head"), ("head", "head")): 1.02,
                           (("bedhead", "bed"), ("deathbed", "bed")): 0.62,
                           (("bedhead", "head"), ("headrest", "head")): 0.66}.items():
        x = tb.t1[(tb.stimulus == w1[0]) & (tb.syl == w1[1])]
        y = tb.t1[(tb.stimulus == w2_[0]) & (tb.syl == w2_[1])]
        pub(f"Experiment 2 F1 adapters: Cohen's d, '{w1[1]}' in {w1[0]} vs '{w2_[1]}' in {w2_[0]}", cohen_d(x, y), rep, 0.005,
            f"{SRC}, Results, Comparison of the trained word and the trained syllables")
    # novel syllables plus bedhead among the F1 adapters (data1c)
    tc = ta[(ta.condition == "Novel") | (ta.stimulus == "bedhead")]
    for i, rep in ((1, 1.29), (2, 1.49)):
        pub(f"Experiment 2 F1 adapters: SE of F1 transfer, position {i}, bedhead and novel syllables (mel)",
            se(tc.t1[tc._item == i]), rep, 0.005, f"{SRC}, Results, Comparison of the trained word and the novel syllables")
    nov = tc[tc.condition == "Novel"]
    for i, rep in ((1, 1.51), (2, 1.54)):
        pub(f"Experiment 2 F1 adapters: SE of F1 transfer, novel syllables in position {i} (mel)", se(nov.t1[nov._item == i]),
            rep, 0.005, f"{SRC}, Results, Comparison of the trained word and the novel syllables")
    pub("Experiment 2 F1 adapters: Cohen's d, position 1 vs 2, bedhead and novel syllables", cohen_d(tc.t1[tc._item == 1],
        tc.t1[tc._item == 2]), 0.54, 0.005, f"{SRC}, Results, Comparison of the trained word and the novel syllables")
    pub("Experiment 2 F1 adapters: Cohen's d, position 1 vs 2, novel syllables", cohen_d(nov.t1[nov._item == 1],
        nov.t1[nov._item == 2]), 0.24, 0.005, f"{SRC}, Results, Comparison of the trained word and the novel syllables")
    # F2, among the F2 adapters
    f2a = post[post.subject.isin(ad2)]
    se_f2 = {("deathbed", "bed"): 2.46, ("headrest", "head"): 2.44, ("bedhead", "bed"): 2.77}  # bedhead 'head': see differences
    for (wd, sy), rep in se_f2.items():
        pub(f"Experiment 2 F2 adapters: SE of F2 transfer, '{sy}' in {wd} (mel)", se(f2a.t2[(f2a.stimulus == wd) & (f2a.syl == sy)]),
            rep, 0.005, f"{SRC}, Results, Comparison of the trained word and the trained syllables (F2)")
    f2c = f2a[(f2a.condition == "Novel") | (f2a.stimulus == "bedhead")]
    for i, rep in ((1, 1.31), (2, 1.14)):
        pub(f"Experiment 2 F2 adapters: SE of F2 transfer, position {i}, bedhead and novel syllables (mel)",
            se(f2c.t2[f2c._item == i]), rep, 0.005, f"{SRC}, Results, Comparison of the trained word and the novel syllables")
    pub("Experiment 2 F2 adapters: Cohen's d, position 1 vs 2, bedhead and novel syllables", cohen_d(f2c.t2[f2c._item == 1],
        f2c.t2[f2c._item == 2]), 0.49, 0.005, f"{SRC}, Results, Comparison of the trained word and the novel syllables")
    return dict(du1=du1, du2=du2)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = stdlib.Checks()
    D = load(checks)
    rows = build(D, checks)
    a = pd.DataFrame(rows)
    diag = checks_all(a, D, checks)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    print("bootstrap margins F1:", sorted(round(v, 2) for v in diag["du1"].values()))
    print("bootstrap margins F2:", sorted(round(v, 2) for v in diag["du2"].values()))
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Combined the authors' per-trial formant, f0 and intensity tables from their R workspaces into one table, "
                 "added the formant shift on each trial from the authors' experiment code and shift table, and replaced "
                 "participant identifiers with codes."),
        how=[
            "One row is the vowel /ɛ/ of one syllable of one spoken word (two rows per trial for two-syllable words), in "
            "the authors' trial order; trial is the authors' trial number and its gaps are trials they removed for "
            "formant-tracking errors or misproductions. f1_mel and f2_mel are F1 and F2 averaged over 25-75% of the vowel "
            "(mels); f0_hz (median), intensity_au and intensity_max_au (mean and maximum of the authors' linear "
            "intensity track) cover the whole vowel and exist only for the two-syllable words of Experiment 2's "
            "Transfer1 and Transfer2 phases (the paper's Pre-training and Post-training).",
            "The authors shared Transfer1 formants only as each participant's mean per word and syllable, the reference "
            "for their transfer measure; these are rows with unit 'participant summary' (F2 only for the 15 participants "
            "of their F2 analyses). Transfer1 trial rows hold f0, intensity and duration only.",
            "pert_value is the size of the shift in the F1-F2 plane in mels (0, a 30-step ramp to 125, 125 in Hold), and "
            "shift_f1 and shift_f2 are its F1 and F2 parts, positive when the formant was raised in what the participant "
            "heard, from the authors' per-speaker shift table; condition is the syllable's shift (/ɪ/-shift or /æ/-shift) "
            "in training phases and the authors' Trained or Novel syllable label in transfer phases; group is the "
            "counterbalancing order.",
            "Masking follows the experiment code: Experiment 2 Baseline and Washout trials played the speech back without "
            "masking noise (Audapter feedback mode 1), although the paper describes ~60 dB SPL noise on all trials. The "
            "authors' per-sample f0 tracks within each syllable are not included.",
        ],
        unit_note="vowel of one syllable of one spoken word (utterance), or one participant's Transfer1 mean (participant summary)",
        differences=[
            "The paper lists Experiment 2's Baseline as 10 trials and Ramp as 20; the trial numbers in the shared data and "
            "the experiment code give 20 Baseline and 30 Ramp trials (with the last 10 Baseline trials as the reference).",
            "Trials removed in Experiment 2 (paper: mean 1.53%, SD 1.65%, range 0-4.76%) cannot be recounted, because "
            "Transfer1 formants were shared only as means.",
            "Numbers that differ from the paper here and equally in the authors' own analysis objects in the shared R "
            "workspaces, so most likely typing slips in the paper: Experiment 1 SE of F1 change for the /ɪ/-shift 2.57 "
            "(paper 2.56); Experiment 1 Washout, 95% of speakers with a more positive F1 change for the /ɪ/-shift (paper "
            "90%; one speaker's difference is +0.9 mel); Experiment 2 Washout, 14 of the 15 speakers of the F2 analysis "
            "with a more negative F2 change for the /ɪ/-shift (paper: all); Table 1 mean F2 perturbation of the F2 "
            "adapters 53.90 mel (paper 58.90); SE of F2 transfer for 'head' in bedhead 2.53 (paper 2.46) and for novel "
            "syllables 1.45 and 1.20 (the paper prints 1.51 and 1.54, the F1 values).",
        ],
    )
    for r in rows:
        for k in [k for k in r if k.startswith("_")]:
            del r[k]
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
