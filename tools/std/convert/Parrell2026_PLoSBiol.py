"""Parrell, Bae, Naber, Kim, Niziolek & McDougle (2026), PLoS Biol 24(6):e3003862: single-trial speech adaptation when the
perturbed trial is spoken (movement) or only heard (no movement). Two experiments of 30 speakers each.

Raw files, raw_extra/Parrell2026_PLoSBiol/ (OSF xqztr, fetched with tools/std/fetch.py):
  EXP1_statTable_30s_50-125.txt, EXP2_statTable_30s_50-125.txt   one row per triplet of trials, 216 triplets per participant:
      fVal      single-trial adaptation: F1 (mels, mean over 50-125 ms after vowel onset) of the third trial of the triplet minus
                the first (paper, Methods "Data analysis")
      fValreten retention: F1 of the trial after the third minus the first trial of the triplet (the paper's "aftereffects")
      subj, cond (go = movement triplet, noGo = no-movement triplet), direc (Up / Down: F1 of the feedback raised or lowered by
      125 mels on the middle trial), trial (position in the 756-trial session), failed (1 = a no-movement triplet in which the
      participant produced detectable speech on the middle trial; the authors drop these)
  analyze_EXP1_EXP2.Rmd   the authors' mixed models, read as text: fVal ~ direc*cond + (1|subj) on the triplets with failed == 0
      and a value, the same for fValreten, and one model on both measures stacked ("combined adaptation index"); in Experiment 2
      the authors rename noGo to observe.
Run from the site folder: python3 -I tools/std/convert/Parrell2026_PLoSBiol.py
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

ID = "Parrell2026_PLoSBiol"
SRC = "Parrell et al. 2026, PLoS Biol"
SHIFT = 125.0
DEVICE = "AKG C520 microphone, Focusrite Scarlett 2i2 sound card, Audapter, Beyerdynamic DT 770 closed-back headphones"
NOISE = "speech feedback at about 83 dB SPL mixed with speech-shaped noise at about 60 dB SPL"
REFERENCE = ("f1_change_mel: F1 of the third trial of the triplet minus F1 of its first trial (the perturbation is on the middle "
             "trial); f1_change_later_mel: F1 of the trial after the third minus F1 of the first trial")
COND = {1: {"go": "go", "noGo": "noGo"}, 2: {"go": "go", "noGo": "observe"}}  # Experiment 2: the authors' code renames noGo


def tol_of(printed):
    s = str(printed)
    return 0.5 * 10 ** -(len(s.split(".")[1]) if "." in s else 0) + 1e-6


def tol_unit(printed):
    s = str(printed)
    return 10 ** -(len(s.split(".")[1]) if "." in s else 0) + 1e-6


def emm(d, col):
    """The authors' lmer(value ~ direc*cond + (1|subj)) as cell means with a random intercept per participant (REML):
    estimated marginal means, and the Down - Up contrast within each condition with its SE, t and d (contrast / residual SD)."""
    import statsmodels.formula.api as smf
    d = d.copy()
    d["cell"] = d.direc + "_" + d.cond
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("v ~ 0 + C(cell)", d, groups=d["subj"]).fit(reml=True)
    names = list(m.fe_params.index)
    cov = m.cov_params().iloc[:len(names), :len(names)]
    out = {}
    for c in sorted(d.cond.unique()):
        i, j = names.index(f"C(cell)[Down_{c}]"), names.index(f"C(cell)[Up_{c}]")
        diff = m.fe_params.iloc[i] - m.fe_params.iloc[j]
        se = math.sqrt(cov.iloc[i, i] + cov.iloc[j, j] - 2 * cov.iloc[i, j])
        out[c] = dict(down=m.fe_params.iloc[i], down_se=m.bse_fe.iloc[i], up=m.fe_params.iloc[j], up_se=m.bse_fe.iloc[j],
                      diff=diff, diff_se=se, t=diff / se, d=diff / math.sqrt(m.scale))
    out["n"] = len(d)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    tabs = {e: pd.read_csv(raw / f"EXP{e}_statTable_30s_50-125.txt") for e in (1, 2)}
    rows = []
    for e, t in tabs.items():
        cnt3 = t.groupby(["subj", "cond", "direc"]).size()
        checks.internal(f"Experiment {e}: 30 participants x 216 triplets (36 go and 72 noGo per direction), trial numbers unique "
                        "within a participant, 2-755, at least 3 apart",
                        bool(t.subj.nunique() == 30 and (t.groupby("subj").size() == 216).all()
                             and (cnt3[cnt3.index.get_level_values("cond") == "go"] == 36).all()
                             and (cnt3[cnt3.index.get_level_values("cond") == "noGo"] == 72).all()
                             and (t.groupby("subj").trial.nunique() == 216).all() and t.trial.min() == 2 and t.trial.max() == 755
                             and t.groupby("subj").trial.apply(lambda s: np.diff(np.sort(s.values)).min()).min() >= 3))
        checks.internal(f"Experiment {e}: no go triplet is marked failed", bool((t[t.cond == "go"].failed == 0).all()))
        t = t.assign(_o=t.subj.map({v: i for i, v in enumerate(t.subj.drop_duplicates())})).sort_values(["_o", "trial"], kind="stable")
        for r in t.itertuples(index=False):
            sh = SHIFT if r.direc == "Up" else -SHIFT
            gone = bool(pd.isna(r.fVal) and pd.isna(r.fValreten))
            rows.append({
                "dataset": ID, "study": ID, "experiment": f"Experiment {e}", "group": "", "subject": str(r.subj), "person": str(r.subj),
                "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy", "setting": "lab",
                "device": DEVICE, "session": 1, "block": "", "trial": int(r.trial), "phase": "learning", "time_s": "",
                "task": "speech", "unit": "trial", "condition": COND[e][r.cond], "pert_type": "formant_shift", "pert_value": sh,
                "pert_unit": "mel", "feedback": "altered", "excluded_by_authors": bool(r.failed == 1 or gone),
                "stimulus": "", "segment": "vowel /ɛ/", "measure_window": "50-125 ms after vowel onset",
                "f1_change_mel": "" if pd.isna(r.fVal) else float(r.fVal),
                "f1_change_later_mel": "" if pd.isna(r.fValreten) else float(r.fValreten),
                "change_reference": REFERENCE, "shift_f1": sh, "shift_f2": 0.0, "masking_noise": NOISE, "feedback_latency_ms": 18,
                "failed": int(r.failed),
            })
    a = pd.DataFrame(rows)
    a["f1c"] = pd.to_numeric(a.f1_change_mel.replace("", np.nan))
    a["f1l"] = pd.to_numeric(a.f1_change_later_mel.replace("", np.nan))
    a["direc"] = np.where(a.pert_value > 0, "Up", "Down")
    a["exp"] = a.experiment.str[-1].astype(int)
    checks.internal("12,960 triplets = 2 experiments x 30 participants x 216; fVal missing on 328 and fValreten on 393, as shared",
                    len(a) == 12960 and int(a.f1c.isna().sum()) == 67 + 39 + 133 + 89
                    and int(a.f1l.isna().sum()) == 67 + 71 + 133 + 122)
    checks.internal("F1 changes within -700 to 700 mel (a few tracking errors are left in by the authors)",
                    bool(a.f1c.dropna().between(-700, 700).all() and a.f1l.dropna().between(-700, 700).all()),
                    f"{a.f1c.min():.0f} to {a.f1c.max():.0f}, {a.f1l.min():.0f} to {a.f1l.max():.0f}")

    P = f"{SRC}, Methods"
    R = f"{SRC}, Results"
    for e in (1, 2):
        x = a[a.exp == e]
        ppl = x.drop_duplicates("subject")
        checks.published(f"Experiment {e}: participants", len(ppl), 30, 0, f"{P} (Participants)")
        checks.published(f"Experiment {e}: triplets per participant (36 blocks x 6)", x.groupby("subject").size().max(), 216, 0, f"{P} (Procedure)")
        checks.published(f"Experiment {e}: movement perturbations per direction", x[(x.condition == "go") & (x.direc == "Up")].groupby("subject").size().max(), 36, 0, f"{P} (Procedure)")
        checks.published(f"Experiment {e}: no-movement perturbations per direction", x[(x.condition != "go") & (x.direc == "Up")].groupby("subject").size().max(), 72, 0, f"{P} (Procedure)")
        nm = x[x.condition != "go"]
        fa = nm.groupby("subject").failed.sum()
        rep = {1: (32, 4, 89), 2: (3, 0, 115)}[e]
        checks.published(f"Experiment {e}: no-movement triplets excluded for speech on the middle trial, median", fa.median(), rep[0], 0, f"{P} (Data analysis)")
        checks.published(f"Experiment {e}: ... lowest across participants", fa.min(), rep[1], 0, f"{P} (Data analysis)")
        checks.published(f"Experiment {e}: ... highest across participants", fa.max(), rep[2], 0, f"{P} (Data analysis)")
        ok = x[(x.failed == 0) & x.f1c.notna()]
        cnt = ok.groupby(["subject", "condition"]).size().unstack()
        mv = cnt["go"]
        nmv = cnt[[c for c in cnt.columns if c != "go"][0]]
        want = {1: ((71, 64, 72), (111, 52, 140)), 2: ((70, 63, 72), (137, None, None))}[e]
        checks.published(f"Experiment {e}: analysed movement triplets per participant, median", mv.median(), want[0][0], 0, f"{P} (Data analysis)")
        checks.published(f"Experiment {e}: ... lowest", mv.min(), want[0][1], 0, f"{P} (Data analysis)")
        checks.published(f"Experiment {e}: ... highest", mv.max(), want[0][2], 0, f"{P} (Data analysis)")
        checks.published(f"Experiment {e}: analysed no-movement triplets per participant, median", nmv.median(), want[1][0], 0, f"{P} (Data analysis)")
        if want[1][1] is not None:
            checks.published(f"Experiment {e}: ... lowest", nmv.min(), want[1][1], 0, f"{P} (Data analysis)")
            checks.published(f"Experiment {e}: ... highest", nmv.max(), want[1][2], 0, f"{P} (Data analysis)")
    pf = 100 * a[(a.exp == 1) & (a.condition != "go")].groupby("subject").failed.mean()
    checks.published("Experiment 1: no-movement triplets with speech on the middle trial, lowest % across participants", pf.min(), 3, 0.5, f"{P} (Data analysis)")
    checks.published("Experiment 1: ... highest % across participants", pf.max(), 62, 0.5, f"{P} (Data analysis)")
    checks.published("Experiment 1: no-movement triplets with speech on the middle trial, % across participants (the paper's "
                     "'average of 22%' is the median here; the mean is 23.4%)", pf.median(), 22, 0.5, f"{P} (Data analysis)")
    print(f"no-movement triplets with speech, Experiment 1: mean {pf.mean():.2f}%, median {pf.median():.2f}% (paper: average of 22%)")

    # the authors' mixed models: combined index (both measures stacked), adaptation (+1) and aftereffects (+2)
    def table(x, col):
        ok = x[(x.failed == 0) & x[col].notna()]
        return pd.DataFrame({"v": ok[col].values, "subj": ok.subject.values, "cond": ok.condition.values, "direc": ok.direc.values})

    fits = {}
    for e in (1, 2):
        x = a[a.exp == e]
        fits[(e, "adapt")] = emm(table(x, "f1c"), "f1c")
        fits[(e, "reten")] = emm(table(x, "f1l"), "f1l")
        fits[(e, "comb")] = emm(pd.concat([table(x, "f1c"), table(x, "f1l")], ignore_index=True), "v")
    # (experiment, model, condition) -> printed values: down, down SE, up, up SE (combined) or contrast, its SE (separate), t, d
    comb = {(1, "go"): ("2.3", "0.8", "-6.2", "0.8", "7.5", "0.23"), (1, "noGo"): ("0.5", "0.7", "-2.1", "0.8", "2.9", "0.07"),
            (2, "go"): ("1.16", "0.75", "-6.93", "0.75", "7.84", "0.24"), (2, "observe"): ("-0.80", "0.56", "-1.28", "0.55", "0.63", None)}
    for (e, c), (dm, dse, um, use, t, dd) in comb.items():
        f = fits[(e, "comb")][c]
        lab = f"Experiment {e}, combined index, {'movement' if c == 'go' else 'no movement'}"
        S = f"{R}, Experiment {e}"
        checks.published(f"{lab}: F1 change with the F1 shift down (mel)", f["down"], float(dm), tol_of(dm), S)
        checks.published(f"{lab}: SE, shift down", f["down_se"], float(dse), tol_unit(dse), S)
        checks.published(f"{lab}: F1 change with the F1 shift up (mel)", f["up"], float(um), tol_of(um), S)
        checks.published(f"{lab}: SE, shift up", f["up_se"], float(use), tol_unit(use), S)
        checks.published(f"{lab}: t of down minus up", f["t"], float(t), tol_unit(t), S)
        if dd:
            checks.published(f"{lab}: d of down minus up", f["d"], float(dd), max(tol_of(dd), 0.01), S)
    sep = {(1, "adapt", "go"): ("8.6", "1.5", "5.7", "0.24"), (1, "adapt", "noGo"): ("2.5", "1.2", "2.014", "0.07"),
           (1, "reten", "go"): ("8.5", "1.7", "4.997", "0.22"), (1, "reten", "noGo"): ("2.9", "1.4", "2.110", "0.07"),
           (2, "adapt", "go"): (None, None, "5.29", "0.23"), (2, "adapt", "observe"): (None, None, "-0.38", None),
           (2, "reten", "go"): (None, None, "5.79", ".256"), (2, "reten", "observe"): (None, None, "1.24", None)}
    for (e, k, c), (df_, dse, t, dd) in sep.items():
        f = fits[(e, k)][c]
        lab = f"Experiment {e}, {'adaptation (+1)' if k == 'adapt' else 'aftereffects (+2)'}, {'movement' if c == 'go' else 'no movement'}"
        S = f"{R}, Experiment {e}"
        if df_:
            checks.published(f"{lab}: down minus up (mel)", f["diff"], float(df_), tol_unit(df_), S)
            checks.published(f"{lab}: SE of the difference", f["diff_se"], float(dse), tol_unit(dse), S)
        checks.published(f"{lab}: t", f["t"], float(t), tol_unit(t), S)
        if dd:
            checks.published(f"{lab}: d", f["d"], float(dd), max(tol_of(dd), 0.01), S)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)

    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    paper = {k: s["papers"][0][k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the authors' two triplet tables (Experiments 1 and 2), put each participant's triplets in trial order "
                 "(the authors' tables are sorted by condition and direction), gave the shift its sign (+125 mel for Up, -125 for "
                 "Down), marked the no-movement triplets with speech on the middle trial and the triplets with no value as "
                 "excluded, and replaced participant identifiers with codes."),
        how=[
            "One row is one triplet of consecutive trials around one perturbed trial (the first and last trials of a triplet have "
            "normal feedback): F1 of the feedback on the middle trial was raised (pert_value +125) or lowered (-125) by 125 mel. "
            "In go triplets the middle word is spoken and the shift is applied in real time; in noGo triplets (named observe in "
            "Experiment 2, as in the authors' code) the participant is cued not to speak and hears the previous trial's "
            "recording with the shift applied.",
            "f1_change_mel is the authors' single-trial adaptation (F1 of the third trial minus the first trial of the triplet) and "
            "f1_change_later_mel their retention (the trial after the third minus the first), both in mel, F1 averaged over "
            "50-125 ms after vowel onset; positive means F1 rose, and an empty value is one the authors' table lacks.",
            "trial is the authors' trial number in the 756-trial session; it runs from 2 to 755 with at least 3 between triplets, "
            "so it is the middle trial. Words (bed, dead, Ted; one per block of six triplets) are not given per row. The shared "
            "tables hold no age or sex.",
            "excluded_by_authors marks no-movement triplets in which the participant spoke on the middle trial (the authors' "
            "failed = 1, dropped from their models) and triplets with neither value; a triplet with only one value is used by "
            "the authors for that measure.",
        ],
        unit_note="triplet of trials around one perturbed trial",
        differences=[
            "Experiment 1: the paper's 'average of 22%' of no-movement triplets with speech on the middle trial is the median "
            "across participants here (22.2%); the mean is 23.4%.",
            "Experiment 2: the paper gives 127-140 analysed no-movement triplets per participant (median 137); the rows give a "
            "median of 137 but a range of 27 to 144, which fits the paper's own 0-115 excluded triplets (144 - 115 = 29).",
            "Standard errors of the model means, some contrasts and some d values are truncated or differ by one unit of the last "
            "printed digit in the paper (SE 0.884 is printed 0.8, d 0.248 is printed 0.24, d 0.254 is printed .256, a contrast of "
            "2.45 is printed 2.5), so they are checked to one unit of the last printed digit; every t statistic agrees to its "
            "printed decimals.",
        ],
    )
    return stdlib.release(ID, "speech", [{k: v for k, v in r.items() if k != "failed"} for r in rows], meta, checks)


if __name__ == "__main__":
    main()
