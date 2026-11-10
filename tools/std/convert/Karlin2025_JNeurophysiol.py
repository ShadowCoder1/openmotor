"""Karlin, Tesch, Tang, Zeng, Niziolek & Parrell (2025), J Neurophysiol: opposing F1 shifts on the same vowel, with
Mandarin lexical tone (Experiment 1, fei1/fei2/fei4), a produced pitch cue (Experiment 2, head after a high, mid or low
tone to match) or a heard pitch cue only (Experiment 3) as the context.

Raw file, raw_extra/Karlin2025_JNeurophysiol/ (OSF v7zaf, "Data/deidToneData.csv"): one row per trial of all 60
participants (540 trials each), the authors' fullTable.csv of the three experiments joined in dataPrep_simonTone.R:
  trial, word, shift (shiftUp/shiftDown/noShift), phase, language (Mandarin, English-Match, English-Listen = Experiments
  1-3), duration (vowel, s), f1in/f2in (produced F1/F2, Hz), f1out/f2out (F1/F2 of the feedback heard, Hz), baselineF1,
  distF1, distF1_abs (derived, not kept), bCanMatch (pitch-matching group), wordTrial (repetition of the word), pubSubj.
  Trials removed for tracking or production errors hold empty formants.
Values (paper, Methods): one mean F1 per trial from 25-100 ms after vowel onset; shifts in mels, 125 mel maximum.
Shift schedule (run_simonToneLex_expt.m, read as text): shiftMags = [zeros(nBaseline) sort(repmat(linspace(0, 125, nRamp/3),
1, 3)) 125*ones(nHold) zeros(nWashout)], 30/30/90/30 repetitions of each of 3 words, signed by the word's condition.
Run from the site folder: python3 -I tools/std/convert/Karlin2025_JNeurophysiol.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Karlin2025_JNeurophysiol"
SRC = "Karlin et al. 2025 (preprint on OSF v7zaf)"
EXP = {"Mandarin": 1, "English-Match": 2, "English-Listen": 3}
SIGN = {"shiftUp": 1, "shiftDown": -1, "noShift": 0}
PHASE = {"baseline": "baseline", "ramp": "learning", "hold": "learning", "washout": "washout"}
STIM = {"fei1": "fei1 (Mandarin, tone 1)", "fei2": "fei2 (Mandarin, tone 2)", "fei4": "fei4 (Mandarin, tone 4)",
        "headH": "head (after the high pitch cue)", "headM": "head (after the mid pitch cue)",
        "headL": "head (after the low pitch cue)"}
DEVICE = "Sennheiser MKE 600 microphone, Audapter, Beyerdynamic DT 770 PRO headphones"


def mel(hz):
    return 2595 * np.log10(1 + np.asarray(hz, float) / 700)


def shift_mag(trial):
    if 91 <= trial <= 180:
        return 125 * ((trial - 91) // 3) / 29
    if 181 <= trial <= 450:
        return 125.0
    return 0.0


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = stdlib.Checks()
    d = pd.read_csv(stdlib.raw_dir(ID) / "deidToneData.csv")
    d["e"] = d.language.map(EXP)
    checks.internal("60 participants x 540 trials, trials 1-540 in order",
                    len(d) == 32400 and d.pubSubj.nunique() == 60 and d.groupby("pubSubj", sort=False).trial.apply(
                        lambda s: list(s) == list(range(1, 541))).all())
    ph = {"baseline": (1, 90), "ramp": (91, 180), "hold": (181, 450), "washout": (451, 540)}
    checks.internal("phases on trials 1-90, 91-180, 181-450, 451-540 (30/30/90/30 repetitions of 3 words)",
                    all(ph[p][0] <= t <= ph[p][1] for p, t in zip(d.phase, d.trial)))
    checks.internal("one shift condition per word and participant",
                    bool((d.groupby(["pubSubj", "word"])["shift"].nunique() == 1).all()))
    d["pv"] = [SIGN[s] * shift_mag(t) for s, t in zip(d["shift"], d.trial)]
    ok = d.f1in.notna()
    b = d[ok & (d.phase == "baseline")]
    ref = d[d.wordTrial.between(21, 30)].groupby(["pubSubj", "word"]).f1in.mean().rename("ref")
    x = d.join(ref, on=["pubSubj", "word"])
    checks.internal("the authors' baselineF1 is the mean F1 of the word's last 10 Baseline repetitions, and distF1 is F1 "
                    "minus it (in Hz)", bool(np.nanmax(np.abs(x.ref - x.baselineF1)) < 1e-6
                                            and np.nanmax(np.abs(d.distF1 - (d.f1in - d.baselineF1))) < 1e-6))
    checks.internal("the file's f1out column equals f1in on every trial (it is left out)",
                    bool(((d.f1out - d.f1in).abs().fillna(0) < 1e-9).all()))
    rows = []
    for r in d.itertuples(index=False):
        pv = 0.0 if r.pv == 0 else float(r.pv)
        rows.append({
            "dataset": ID, "study": ID, "experiment": f"Experiment {r.e}", "group": "", "subject": r.pubSubj,
            "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy",
            "setting": "lab", "device": DEVICE, "session": 1, "block": r.phase, "trial": int(r.trial),
            "phase": PHASE[r.phase], "time_s": "", "task": "speech", "unit": "utterance", "condition": r.shift,
            "pert_type": "formant_shift" if pv else "none", "pert_value": pv, "pert_unit": "mel",
            "feedback": "altered" if pv else "normal", "excluded_by_authors": bool(pd.isna(r.f1in)),
            "stimulus": STIM[r.word], "segment": "vowel /ei/" if r.e == 1 else "vowel /ɛ/",
            "measure_window": "25-100 ms after vowel onset", "duration_ms": r.duration * 1000 if pd.notna(r.duration) else "",
            "f0_hz": "", "f1_hz": r.f1in, "f2_hz": r.f2in, "f1_mel": "", "f2_mel": "", "intensity_au": "",
            "intensity_max_au": "", "f1_heard_hz": "", "f2_heard_hz": "", "shift_f1": pv, "shift_f2": 0.0,
            "masking_noise": "masking noise, about 60 dB SPL, mixed with the speech playback (about 80 dB SPL)",
            "feedback_latency_ms": 19,
        })
    # published checks
    pub = checks.published
    for e, n in ((1, 20), (2, 20)):
        pub(f"Experiment {e}: participants", d.pubSubj[d.e == e].nunique(), n, 0, f"{SRC}, Methods, Participants")
    pct = 100 * d[d.e == 1].groupby("pubSubj").f1in.apply(lambda s: s.isna().mean())
    pub("Experiment 1: % of trials excluded, mean", pct.mean(), 1.4, 0.05, f"{SRC}, Methods, Experiment 1 data analysis")
    pub("Experiment 1: % of trials excluded, maximum participant", pct.max(), 3.5, 0.05, f"{SRC}, Methods, Experiment 1 data analysis")
    pub("Experiment 1: % of trials excluded, minimum participant", pct.min(), 0, 0.05, f"{SRC}, Methods, Experiment 1 data analysis")
    # differential adaptation (dataPrep_simonTone.R): per participant, mean F1 change (mel) from the last 10 Baseline
    # productions of the word, shiftDown minus shiftUp, over the last 10 Hold and the first 10 Washout repetitions
    # the paper labels these mels, but they reproduce from F1 in Hz (the authors' distF1); see differences
    d["dist"] = d.f1in - d.baselineF1
    win = {"hold": (141, 150), "washout": (151, 160)}
    tab = {}
    for w, (lo, hi) in win.items():
        x = d[d.wordTrial.between(lo, hi)].groupby(["pubSubj", "shift"]).dist.mean().unstack()
        tab[w] = x["shiftDown"] - x["shiftUp"]
    info = d.drop_duplicates("pubSubj").set_index("pubSubj")
    rep = {(1, "hold"): (69.02, 68.3), (1, "washout"): (72.01, 55.78), (2, "hold"): (46.90, 61.98), (2, "washout"): (45.41, 56.29)}
    for (e, w), (m, sd) in rep.items():
        x = tab[w][info.e[tab[w].index] == e]
        pub(f"Experiment {e}: differential adaptation (shiftDown - shiftUp), {w}, mean (Hz)", x.mean(), m, 0.005,
            f"{SRC}, Results, Comparison across experiments")
        pub(f"Experiment {e}: differential adaptation, {w}, SD (Hz)", x.std(ddof=1), sd, 0.05 if sd == 68.3 else 0.005,
            f"{SRC}, Results, Comparison across experiments")
    grp = info.bCanMatch
    pub("Experiment 2: non-matchers", int((grp[info.e == 2] == "English-non matcher").sum()), 8, 0,
        f"{SRC}, Results, Pitch-matching ability")
    for w, (mm, mn, t_r, df_r) in {"hold": (49.92, 42.37, 0.26, 14.544), "washout": (52.67, 34.53, 0.66, 12.647)}.items():
        x = tab[w]
        a = x[(info.e[x.index] == 2) & (grp[x.index] == "English-Matcher")]
        bb = x[(info.e[x.index] == 2) & (grp[x.index] == "English-non matcher")]
        pub(f"Experiment 2: differential adaptation of matchers, {w} (Hz)", a.mean(), mm, 0.005, f"{SRC}, Results, Pitch-matching ability")
        pub(f"Experiment 2: differential adaptation of non-matchers, {w} (Hz)", bb.mean(), mn, 0.005, f"{SRC}, Results, Pitch-matching ability")
        t, df = welch(a, bb)
        pub(f"Experiment 2: Welch t, matchers vs non-matchers, {w}", t, t_r, 0.005, f"{SRC}, Results, Pitch-matching ability")
        pub(f"Experiment 2: Welch df, matchers vs non-matchers, {w}", df, df_r, 0.0005, f"{SRC}, Results, Pitch-matching ability")
    for w, (t_r, df_r) in {"hold": (-1.07, 37.65), "washout": (-1.50, 37.997)}.items():
        x = tab[w]
        t, df = welch(x[info.e[x.index] == 2], x[info.e[x.index] == 1])
        pub(f"Welch t, English (Experiment 2) vs Mandarin (Experiment 1) differential adaptation, {w}", t, t_r, 0.005,
            f"{SRC}, Results, Comparison across experiments")
        pub(f"Welch df, English vs Mandarin, {w}", df, df_r, 0.005 if w == "hold" else 0.0005,
            f"{SRC}, Results, Comparison across experiments")
    for c in checks.items:
        if c["kind"] == "published":
            print(("ok  " if c["ok"] else "BAD ") + f"{c['what'][:70]:70s} {c['computed']:10.4f} {c['reported']:10.4f}")
        elif not c["ok"]:
            print("FAILED:", c)
    parts = [{"person": s, "pitch_match_group": ("matcher" if g == "English-Matcher" else "non-matcher") if e == 2 else ""}
             for s, g, e in zip(info.index, info.bCanMatch, info.e)]
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Re-encoded the authors' per-trial table in the OpenMotor format (duration in ms), added the F1 shift on "
                 "each trial from the authors' experiment code, moved the pitch-matching group to participants.csv and "
                 "left out the authors' derived baseline and distance columns."),
        how=[
            "One row is one spoken word of one participant (540 trials: 30 Baseline, 30 Ramp, 90 Hold and 30 Washout "
            "repetitions of each of three words), in the authors' order; Experiment 1 is Mandarin speakers saying fei with "
            "tones 1, 2 and 4, Experiment 2 English speakers saying head after a high, mid or low tone they matched with "
            "their voice, and Experiment 3 the same with the tone only heard.",
            "f1_hz and f2_hz are the produced formants, each a mean over 25-100 ms after vowel onset; trials the authors "
            "removed for tracking or production errors have empty formants and are marked excluded_by_authors. The file's "
            "undocumented f1out and f2out columns are left out (f1out equals f1in on every trial; f2out differs from f2in "
            "on some trials in every phase), as are its derived baseline and distance columns.",
            "pert_value and shift_f1 are the F1 shift heard in mels (positive raises F1), rebuilt from the experiment "
            "code: 0 in Baseline and Washout, a 30-step ramp to 125 over the Ramp, 125 in Hold, signed by the word's "
            "condition (shiftUp, shiftDown, noShift).",
            "participants.csv gives the authors' pitch-matching group for Experiment 2 (non-matchers: under 60% of mid "
            "tones classified correctly).",
        ],
        unit_note="spoken word (utterance) of one participant",
        differences=[
            "The paper reports differential adaptation, its SDs and the t-tests in mels; they reproduce exactly from F1 in "
            "Hz (the authors' distF1, F1 minus the mean of the word's last 10 Baseline repetitions), so those numbers are "
            "in Hz. The shift itself was applied in mels.",
            "Experiment 3: the paper analyses 19 of 21 participants; the shared file has 20, including one the authors' "
            "code drops after the fact (sp726), whom the de-identified codes do not let us mark.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks, participants=parts,
                          participant_columns=[("pitch_match_group", "category", "matcher, non-matcher",
                                                "Experiment 2 only: the authors' pitch-matching group (k-means of trial f0).")])


if __name__ == "__main__":
    main()
