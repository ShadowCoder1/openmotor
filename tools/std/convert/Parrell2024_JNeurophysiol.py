"""Parrell, Niziolek & Chen (2024), J Neurophysiol 132(5):1437-1444: adaptation to a nonuniform "centralization" formant
perturbation (all four corner vowels pushed 50% toward the center of the participant's vowel space) generalizes to five
untrained vowels. 21 analysed participants (22 took part; one was excluded for equipment failure).

Raw files, raw_extra/Parrell2024_JNeurophysiol/ (OSF 78z93, fetched with tools/std/fetch.py; no license stated). The authors
shared four per participant tables (comma separated, NaN for missing) and, in raw_data.zip (987 MB), the formant tracks of
22 people (dataVals.mat, expt.mat). The tracks are not trial values (the paper's per trial F1 and F2 are means over the middle
50% of a hand-corrected vowel), so turning them into values would be a new analysis; they are not converted. The tables are:
  datatable_AVS.txt                AVS, phase (adapt, gen), subj        normalized average vowel spacing, 2 rows per participant
  datatable_vowels.txt             mag, ang, trainDistMag, trainDistAng, vowel, vowNUM, subj    9 vowels per participant
  datatable_vowels_byphase.txt     mag, phase (Base, End), vowel, vowNUM, subj                  9 vowels x 2 phases
  datatable_clearSpeech.txt        f0Max, f0Range, int, dur, phase (Base, End), vowel, vowNUM, subj
  processedData_25-75.mat          the same numbers as MATLAB tables (not needed)
What the columns are (paper, Methods "Data analysis"; checked below against the numbers the paper reports):
  AVS     average Euclidean F1-F2 distance (mels) between each pair of the vowels of the block, divided by the same value in
          the baseline block with the same words; adapt = last 10-repetition block of exposure (the 4 trained words),
          gen = generalization phase (the 5 untrained words)
  mag     (byphase) distance (mels) of the vowel's mean F1/F2 from the center of the vowel distribution, at baseline (Base)
          and at the end of exposure (trained vowels) or in the generalization phase (untrained vowels) (End);
          (vowels) End minus Base, exactly (checked), positive = farther from the center
  ang     direction (radians, range -pi/2 to 3pi/2) of the change vector from the baseline to the End F1/F2 means; 0 = F1
          rises with no change in F2 (Methods)
  trainDistMag, trainDistAng  the regressors of the paper's generalization models, defined only for the untrained vowels:
          the sum over the trained vowels of (trained vowel's mag / its F1-F2 distance to this vowel), and the circular mean of
          the trained vowels' angles weighted by the inverse distance (both reproduce the paper's regression and circular
          correlation, below)
  f0Max, f0Range, int, dur  peak pitch (Hz), pitch range (Hz), intensity (arbitrary units), vowel duration (s)
Run from the site folder: python3 -I tools/std/convert/Parrell2024_JNeurophysiol.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Parrell2024_JNeurophysiol"
SRC = "Parrell, Niziolek & Chen 2024, J Neurophysiol 132:1437"
DEVICE = "AKG C520 head-mounted microphone, Audapter, Beyerdynamic DT 770 headphones"
TRAINED = {"iy": "/i/", "ae": "/æ/", "aa": "/ɑ/", "uw": "/u/"}
GENERAL = {"ih": "/ɪ/", "ey": "/eɪ/", "eh": "/ɛ/", "ah": "/ʌ/", "ow": "/oʊ/"}
IPA = {**TRAINED, **GENERAL}
SHIFT = "% of the distance from each production to the center of the participant's vowel space"
MASK = "speech-shaped noise, about 80 dB SPL, played instead of the participant's own voice (masking, no playback of own speech)"
NOISE_FB = "speech-shaped noise at about 60 dB SPL mixed with the feedback (about 80 dB SPL)"

# measure -> (description, statistic, unit)
MEAS = {
    "normalized_AVS": ("Average vowel spacing (AVS): mean Euclidean F1-F2 distance in mels between each pair of vowels of the block, "
                       "divided by the same value in the baseline block with the same words (1 = no change). Column AVS.",
                       "other", "ratio to baseline"),
    "adaptation_magnitude_mel": ("Change in the distance of the vowel's mean F1/F2 from the center of the vowel space, end phase "
                                 "minus baseline (positive = farther from the center, i.e. opposing the centralizing perturbation). "
                                 "Column mag of datatable_vowels; equals End minus Base of distance_from_center_mel.",
                                 "other", "mel"),
    "adaptation_angle_rad": ("Direction of the F1-F2 change vector from the baseline to the end phase, in radians; 0 = F1 rises with "
                             "no change in F2 (the direction of positive angles is not stated in the paper). Column ang.",
                             "other", "rad"),
    "train_weighted_magnitude": ("Regressor of the paper's generalization model, defined for untrained vowels only: sum over the four "
                                 "trained vowels of (trained vowel's adaptation magnitude / its F1-F2 distance to this vowel). "
                                 "Column trainDistMag.", "other", "mel per mel"),
    "train_weighted_angle_rad": ("Circular mean of the four trained vowels' adaptation angles weighted by the inverse of their F1-F2 "
                                 "distance to this vowel, defined for untrained vowels only. Column trainDistAng.", "other", "rad"),
    "distance_from_center_mel": ("Distance of the vowel's mean F1/F2 from the center of the vowel distribution (Euclidean, mels) in "
                                 "the phase of the row. Column mag of datatable_vowels_byphase.", "other", "mel"),
    "peak_pitch_hz": ("Peak (maximum) pitch of the vowel productions in the vowel and phase of the row (clear-speech control). "
                      "Column f0Max.", "other", "Hz"),
    "pitch_range_hz": ("Pitch range of the vowel productions in the vowel and phase of the row (clear-speech control). Column f0Range.",
                       "other", "Hz"),
    "intensity_au": ("Intensity of the vowel productions in the vowel and phase of the row, in the authors' arbitrary units "
                     "(clear-speech control). Column int.", "other", "arbitrary units"),
    "vowel_duration_s": ("Vowel duration in the vowel and phase of the row (clear-speech control). Column dur.", "other", "s"),
}


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def setting(table, vowel=None, ph=None):
    """phase, block, condition, perturbation and feedback of a row. Trained vowels: baseline = second baseline phase (own voice,
    no perturbation), end = end of exposure (own voice with the centralizing shift). Untrained vowels: baseline and end both
    heard masking noise only (generalization baseline, generalization phase)."""
    trained = vowel in TRAINED if vowel else ph == "adapt"
    if ph in ("Base",):
        phase = "baseline"
    else:
        phase = "learning" if trained else "transfer"
    shifted = trained and ph != "Base"
    fb = "altered" if shifted else ("normal" if trained else "masked")
    return dict(phase=phase, trained=trained, pert_type="formant_shift" if shifted else "none",
                pert_value=50 if shifted else 0, pert_unit=SHIFT if shifted else "", feedback=fb,
                masking_noise=NOISE_FB if trained else MASK)


def build(raw, checks):
    avs = pd.read_csv(raw / "datatable_AVS.txt")
    vow = pd.read_csv(raw / "datatable_vowels.txt")
    byp = pd.read_csv(raw / "datatable_vowels_byphase.txt")
    cs = pd.read_csv(raw / "datatable_clearSpeech.txt")
    rows = []

    def add(subj, tno, rno, mno, measure, value, table, vowel=None, ph=None, block=""):
        st = setting(table, vowel, ph)
        desc, stat, unit = MEAS[measure]
        if vowel:
            cond = f"{vowel} {IPA[vowel]} ({'trained' if st['trained'] else 'untrained'} vowel)"
        else:
            cond = "4 trained vowels" if st["trained"] else "5 untrained vowels"
        rows.append(dict(
            _key=(int(subj), tno, rno, mno),
            dataset=ID, study=ID, experiment="", group="", subject=f"S{int(subj)}", subject_excluded_by_authors=False,
            age="", sex="", handedness="", population="healthy", setting="lab", device=DEVICE, session=1, block=block,
            phase=st["phase"], time_s="", task="speech", unit="participant summary", condition=cond,
            pert_type=st["pert_type"], pert_value=st["pert_value"], pert_unit=st["pert_unit"], feedback=st["feedback"],
            excluded_by_authors=False, measure=measure, measure_description=desc, statistic=stat, value=float(value),
            value_unit=unit, n_trials=""))

    for i, r in enumerate(avs.itertuples(index=False)):
        add(r.subj, 0, i, 0, "normalized_AVS", r.AVS, "avs", ph=r.phase, block=r.phase)
    for i, r in enumerate(vow.itertuples(index=False)):
        for m, (col, name) in enumerate((("mag", "adaptation_magnitude_mel"), ("ang", "adaptation_angle_rad"),
                                         ("trainDistMag", "train_weighted_magnitude"),
                                         ("trainDistAng", "train_weighted_angle_rad"))):
            v = getattr(r, col)
            if not pd.isna(v):  # the two regressors are undefined for the trained vowels
                add(r.subj, 1, i, m, name, v, "vowels", vowel=r.vowel, ph="End", block="End minus Base")
    for i, r in enumerate(byp.itertuples(index=False)):
        add(r.subj, 2, i, 0, "distance_from_center_mel", r.mag, "byphase", vowel=r.vowel, ph=r.phase, block=r.phase)
    for i, r in enumerate(cs.itertuples(index=False)):
        for m, (col, name) in enumerate((("f0Max", "peak_pitch_hz"), ("f0Range", "pitch_range_hz"), ("int", "intensity_au"),
                                         ("dur", "vowel_duration_s"))):
            add(r.subj, 3, i, m, name, getattr(r, col), "clear", vowel=r.vowel, ph=r.phase, block=r.phase)
    df = pd.DataFrame(rows).sort_values("_key", kind="stable").reset_index(drop=True)
    df["trial"] = df.groupby("subject", sort=False).cumcount() + 1
    # trained/untrained phase of the AVS rows is set by 'adapt'/'gen', so the generic setting() call needs no more handling
    return df, dict(avs=avs, vow=vow, byp=byp, cs=cs)


def hedges_g(x):
    n = len(x)
    t = stats.ttest_1samp(x, 0).statistic
    df = n - 1
    j = math.exp(gammaln(df / 2) - gammaln((df - 1) / 2)) / math.sqrt(df / 2)  # exact small-sample correction
    return t, t / math.sqrt(n) * j


def circmean(a):
    return np.arctan2(np.sin(a).sum(), np.cos(a).sum())


def circ_corrcc(a, b):
    sa, sb = np.sin(a - circmean(a)), np.sin(b - circmean(b))
    return (sa * sb).sum() / np.sqrt((sa ** 2).sum() * (sb ** 2).sum())


def check_all(df, t, checks):
    avs, vow, byp, cs = t["avs"], t["vow"], t["byp"], t["cs"]
    n = df.subject.nunique()
    R = f"{SRC}, Results"
    checks.published("participants analysed", n, 21, 0, f"{SRC}, Methods (Data analysis)")
    checks.internal("every participant has 2 AVS values, 9 vowels, 18 vowel x phase distances and 18 clear-speech rows",
                    all(g.groupby("subj").size().eq(k).all() and g.subj.nunique() == 21
                        for g, k in ((avs, 2), (vow, 9), (byp, 18), (cs, 18))))
    exp_rows = len(avs) + (vow[["mag", "ang", "trainDistMag", "trainDistAng"]].notna().sum().sum()) + len(byp) + 4 * len(cs)
    checks.internal("rows = every non-missing value the authors shared (42 AVS + 189 mag + 189 ang + 105 + 105 regressors "
                    "+ 378 distances + 4 x 378 clear-speech values)", len(df) == exp_rows == 2520, f"{len(df)} rows")
    checks.internal("mag of datatable_vowels equals End minus Base of datatable_vowels_byphase for every participant and vowel",
                    bool(np.allclose(vow.set_index(["subj", "vowNUM"]).mag.sort_index().to_numpy(),
                                     (lambda w: w["End"] - w["Base"])(byp.pivot_table(index=["subj", "vowNUM"], columns="phase",
                                                                                       values="mag")).sort_index().to_numpy(),
                                     atol=1e-6)))
    checks.internal("regressors are missing exactly for the 4 trained vowels (84 rows) and present for the 5 untrained (105)",
                    bool(vow.trainDistMag.isna().eq(vow.vowel.isin(TRAINED)).all()
                         and vow.trainDistAng.isna().eq(vow.vowel.isin(TRAINED)).all()))
    checks.internal("angles lie in [-pi/2, 3pi/2] (radians)", bool(vow.ang.between(-math.pi / 2 - 1e-6, 1.5 * math.pi + 1e-6).all()))
    a = avs.pivot(index="subj", columns="phase", values="AVS")
    for ph, lab, mean_pct, tval, gval in (("adapt", "end of exposure", 6.1, 60.1, 12.62), ("gen", "generalization phase", 7.3, 63.2, 13.27)):
        x = a[ph].to_numpy()
        tt, g = hedges_g(x)
        checks.published(f"AVS increase at the {lab} (mean over participants, %)", 100 * (x.mean() - 1), mean_pct, 0.05, f"{R}, Fig 2B")
        checks.published(f"AVS at the {lab}: standard error (%)", 100 * x.std(ddof=1) / math.sqrt(len(x)), 1.7, 0.1, f"{R}, Fig 2B")  # 1.77 at the end of exposure
        checks.published(f"AVS at the {lab}: t(20) against 0", tt, tval, 0.05, f"{R}, first paragraph")
        checks.published(f"AVS at the {lab}: Hedges' g", g, gval, 0.005, f"{R}, first paragraph")
    rho = stats.spearmanr(a["adapt"], a["gen"])
    checks.published("Spearman correlation of AVS at the end of exposure and in the generalization phase", rho.statistic, 0.54, 0.005,
                     f"{R}, Fig 2D")
    checks.published("mean increase of the distance from the vowel-space center (mels)", vow.mag.mean(), 19.6, 0.05, f"{R}, paragraph 2")
    checks.published("... its standard error over the 189 participant x vowel values (mels)", vow.mag.std(ddof=1) / math.sqrt(len(vow)),
                     2.6, 0.05, f"{R}, paragraph 2")
    checks.published("intensity at baseline (arbitrary units)", cs[cs.phase == "Base"]["int"].mean(), 11.4, 0.05,
                     f"{R}, clear-speech paragraph")
    checks.published("intensity at the end of exposure / generalization (arbitrary units)", cs[cs.phase == "End"]["int"].mean(), 14.8,
                     0.05, f"{R}, clear-speech paragraph")
    g = vow[vow.trainDistMag.notna()]
    m = sm.OLS(g.mag, sm.add_constant(g.trainDistMag)).fit()
    checks.published("generalization model: slope of mag on the distance-weighted trained-vowel magnitude", m.params.iloc[1], 28.3, 0.05,
                     f"{R}, last paragraph before Fig 4")
    checks.published("... its t value", m.tvalues.iloc[1], 4.7, 0.05, f"{R}, last paragraph before Fig 4")
    checks.published("... adjusted R squared", m.rsquared_adj, 0.17, 0.005, f"{R}, last paragraph before Fig 4")
    checks.published("circular correlation of the untrained vowels' angle with the distance-weighted trained-vowel angle",
                     circ_corrcc(g.ang.to_numpy(), g.trainDistAng.to_numpy()), 0.32, 0.005, f"{R}, last paragraph before Fig 4")
    return rho


def main():
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    df, t = build(raw, checks)
    rho = check_all(df, t, checks)
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Stacked the authors' four per participant tables (AVS, per vowel change, per vowel and phase distance, "
                 "clear-speech measures) into one long table, one row per value, in the authors' row order within each table; "
                 "left out the cells the authors left empty (the two regressors for the trained vowels); gave each value the "
                 "phase, perturbation and feedback of its condition from the paper's Methods; replaced participant numbers with codes."),
        how=[
            "One row is one value of one of the 21 analysed participants (the authors' tables hold no per trial values). measure "
            "names the value and measure_description defines it; the participant's rows run through the four tables in turn "
            "(normalized average vowel spacing, change of each vowel's distance from the vowel-space center and its direction, "
            "the distance at baseline and at the end of exposure or generalization, and the clear-speech controls: pitch, intensity, "
            "duration), and trial is their running order.",
            "Four vowels were trained (/i/, /æ/, /ɑ/, /u/: bead, bad, bod, booed) under a feedback perturbation that moved F1 and F2 "
            "toward the center of the participant's vowel space by 50% of the distance (phase learning at the end of exposure, "
            "pert_value 50); five untrained vowels (/ɪ/, /eɪ/, /ɛ/, /ʌ/, /oʊ/: bid, bayed, bed, bud, bode) were produced before and after "
            "exposure under loud masking noise without playback of the participant's own voice (feedback masked, phase transfer). "
            "Baseline rows of trained vowels had normal feedback.",
            "Formant values are in mels and are means over the middle 50% of the vowel (Methods); mag is positive when the vowel "
            "moved away from the center, against the perturbation. The authors also shared the formant tracks of 22 people "
            "(raw_data.zip, 1 GB, MATLAB); those are not converted because turning tracks into per trial values would be a new "
            "analysis. The participant who was excluded for equipment failure has no rows in the tables.",
        ],
        unit_note="value (measure) of one participant in one condition",
        differences=[
            "The standard error of the AVS increase at the end of exposure is 1.77% here (paper 1.7%, perhaps truncated); "
            "the other AVS numbers (means, SE of the generalization phase, t, Hedges' g) match.",
            f"The Spearman correlation of the two AVS values reproduces (rho = {rho.statistic:.3f}, paper 0.54) but its P is "
            f"{rho.pvalue:.3f} here (t approximation) against 0.014 in the paper (probably the exact or Edgeworth test of R).",
            "The paper's standard deviations after the intensity means (2.5 and 3.2 arbitrary units) are not reproduced from these tables "
            "(0.7 and 0.9 over the 189 participant x vowel values, 1.7 and 2.1 over participants); the means (11.4 and 14.8) are. The "
            "mixed-model and repeated-measures statistics (F and P values) were not recomputed.",
        ],
    )
    out = df.drop(columns=["_key"]).to_dict("records")
    return stdlib.release(ID, "summary", out, meta, checks)


if __name__ == "__main__":
    main()
