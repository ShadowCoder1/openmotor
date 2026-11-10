"""Li, McAllister, Shiller, Tian & Grigos (2026), Front Hum Neurosci 20:1744572: Mandarin /u/ and /y/ produced by 40 English-speaking
late learners of Mandarin and 18 native Mandarin controls (monosyllabic words wu4 and yu4 read aloud), with somatosensory and auditory
acuity, phonological awareness and proficiency measures per learner.

Raw files, raw_extra/Li2026_FrontHumNeurosci/ (OSF 9qt5j, fetched with tools/std/fetch.py):
  subject_combined.csv  one row per vowel token of a learner (1,597): subject (1001-1041), sound (wu = /u/, yu = /y/), gender, height
                        class (T/M/S for women, MT/MM/MS for men), F1 and F2 in Hz at the vowel midpoint
  control_combined.csv  the same for the native controls (718 tokens, subjects 2001-2022)
  mixed_model_u.csv, mixed_model_y.csv  the authors' model rows (760 and 777): subject, sound, accuracy_height (the token's Euclidean
                        distance to the mean target of the height-matched controls), and the learner's L2_exposure, Proficiency,
                        TPT_Variability, TPT_ED, Auditory (JND), PAT, PACS and Variability (repeated on every row)
  README_Data_Description.txt (UTF-16) and the OSF wiki: column definitions
What the values are (paper, Methods; README): F1 and F2 in Hz at the vowel midpoint; accuracy = Euclidean distance in F1-F2 Bark space
to the centre of the Mandarin speakers' distribution of the same vowel and height group; outliers (ED at least 2.5 MAD from the
speaker-vowel median) were removed before the models. Here the Bark scale is checked numerically: Traunmueller's z = 26.81 / (1 + 1960 / f)
- 0.53 and a target equal to the mean Bark F1 and F2 of the controls of the same sex and height class reproduce the shared accuracy
values of most learner-vowel sets to 0.002.
Run from the site folder: python3 -I tools/std/convert/Li2026_FrontHumNeurosci.py
"""
import csv
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Li2026_FrontHumNeurosci"
PAPER = "Li et al. 2026, Front Hum Neurosci"
DEVICE = "Sennheiser HMD 280 PRO headset microphone, MacBook Pro, MOTU M2 audio interface, Praat (44.1 kHz)"
SOUND = {"wu": ("wu", "high back rounded vowel /u/ of the Mandarin word wù (物, high-falling tone)"),
         "yu": ("yu", "high front rounded vowel /y/ of the Mandarin word yù (遇, high-falling tone)")}
GROUP = {"L": "English-speaking learners of Mandarin", "C": "native Mandarin controls"}
WINDOW = "midpoint of the vowel"


def bark(f):
    return 26.81 / (1 + 1960.0 / np.asarray(f, dtype=float)) - 0.53


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    p["title"] = p["title"].rstrip(".")
    return s, {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}


def main():
    raw = stdlib.raw_dir(ID)
    rd = lambda n: pd.read_csv(raw / n, encoding="utf-8-sig")
    L, C, MU, MY = rd("subject_combined.csv"), rd("control_combined.csv"), rd("mixed_model_u.csv"), rd("mixed_model_y.csv")
    ck = stdlib.Checks()

    # ---- main table: the shared F1/F2 rows of the learners, then of the controls -----------------------------------------
    rows, n_in = [], {}
    for g, tab in (("L", L), ("C", C)):
        for r in tab.itertuples(index=False):
            lab = str(r.subject)
            n_in[lab] = n_in.get(lab, 0) + 1
            stim, seg = SOUND[r.sound]
            rows.append({
                "dataset": ID, "study": ID, "experiment": "", "group": GROUP[g], "subject": lab, "subject_excluded_by_authors": False,
                "age": "", "sex": r.gender, "handedness": "", "population": "healthy", "setting": "lab", "device": DEVICE,
                "session": 1, "block": "", "trial": n_in[lab], "phase": "test", "time_s": "", "task": "speech", "unit": "utterance",
                "condition": f"{stim} ({'/u/' if r.sound == 'wu' else '/y/'})", "pert_type": "none", "pert_value": 0, "pert_unit": "",
                "feedback": "normal", "excluded_by_authors": "",
                "stimulus": stim, "segment": seg, "measure_window": WINDOW, "f1_hz": float(r.F1), "f2_hz": float(r.F2),
            })
    pcode = {}
    for r in rows:
        pcode.setdefault(r["subject"], f"{ID}_P{len(pcode) + 1:04d}")

    # ---- participants.csv: one row per person ------------------------------------------------------------------------------
    cov_cols = ["L2_exposure", "Proficiency", "TPT_Variability", "TPT_ED", "Auditory", "PAT", "PACS"]
    cy = MY.groupby("subject")[cov_cols + ["Variability"]].first()
    cu = MU.groupby("subject")[cov_cols + ["Variability"]].first()
    ck.internal("the two model files give the same learner covariates", bool(np.allclose(cy[cov_cols].values, cu.loc[cy.index, cov_cols].values)))
    ck.internal("every covariate is constant within a learner in both model files",
                bool(all((M.groupby("subject")[c].nunique() == 1).all() for M in (MU, MY) for c in cov_cols + ["Variability"])))
    hts = pd.concat([L, C]).groupby("subject").agg(height=("height", "first"), n_h=("height", "nunique"), n_g=("gender", "nunique"))
    ck.internal("height class and sex are constant within a participant", bool((hts.n_h == 1).all() and (hts.n_g == 1).all()))
    parts = []
    for lab in pcode:
        sid = int(lab)
        p = dict(person=lab, height_class=hts.loc[sid, "height"])
        if sid in cy.index:
            for src, dst in (("L2_exposure", "l2_exposure_years"), ("Proficiency", "proficiency"), ("TPT_Variability", "tpt_variability"),
                             ("TPT_ED", "tpt_ed"), ("Auditory", "auditory_jnd"), ("PAT", "pat_percent"), ("PACS", "pacs")):
                p[dst] = cy.loc[sid, src]
            p["variability_y"] = cy.loc[sid, "Variability"]
            p["variability_u"] = cu.loc[sid, "Variability"]
        parts.append(p)
    pcols = [
        ("height_class", "category", "T, M, S for women; MT, MM, MS for men",
         "Height class of the speaker as the authors give it (women: T tall over 170 cm, M medium 160-170 cm, S short under 160 cm; "
         "men: MT over 180 cm, MM 170-180 cm, MS under 170 cm). Each learner's accuracy is measured against the controls of the same sex and class."),
        ("l2_exposure_years", "number", "years", "Years of Mandarin study (the authors' L2_exposure); learners only."),
        ("proficiency", "integer", "1-6", "Self-reported Mandarin proficiency (the authors' Proficiency; the paper's scale runs 1 to 7, the shared values 1 to 6); learners only."),
        ("tpt_variability", "number", "ellipse area (as shared)",
         "Somatosensory acuity from the tongue placement task: area of the 95% ellipse of the novel vowel in F1-F2 space (the authors' TPT_Variability); lower is better; learners only."),
        ("tpt_ed", "number", "as shared",
         "Somatosensory acuity from the tongue placement task: median distance of the novel vowel tokens to the midpoint of the participant's two reference vowels (the authors' TPT_ED); lower is better; learners only."),
        ("auditory_jnd", "number", "as shared",
         "Auditory acuity: just-noticeable difference on a /u/-/y/ synthetic continuum from the adaptive AXB task (the authors' Auditory); lower is better; learners only."),
        ("pat_percent", "number", "% correct",
         "Somatosensory acuity from the Phonetic Awareness Task: percent correct over 36 trials (the authors' PAT); higher is better; learners only."),
        ("pacs", "integer", "composite score", "Phonological Awareness Composite Score from CTOPP-2 subtests (the authors' PACS); learners only."),
        ("variability_u", "number", "ellipse area (as shared)",
         "Production variability for /u/: area of the 95% confidence ellipse around the multivariate mean in F1-F2 space (the authors' Variability in the /u/ model file); learners only."),
        ("variability_y", "number", "ellipse area (as shared)", "The same for /y/ (the authors' Variability in the /y/ model file); learners only."),
    ]

    # ---- accuracy_rows.csv: the authors' model rows with person codes ------------------------------------------------------------
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    w.writerow(["person", "sound", "row_in_model_file", "accuracy_ed_bark"])
    acc = []
    for snd, M in (("wu", MU), ("yu", MY)):
        k = {}
        for r in M.itertuples(index=False):
            lab = str(r.subject)
            k[lab] = k.get(lab, 0) + 1
            w.writerow([pcode[lab], snd, k[lab], stdlib._fmt(float(r.accuracy_height), "number")])
            acc.append((lab, snd, float(r.accuracy_height)))
    extra = [("accuracy_rows.csv", buf.getvalue().encode("utf-8"))]
    acc = pd.DataFrame(acc, columns=["lab", "sound", "acc"])

    # ---- how far the shared F1/F2 reproduce the shared accuracy (information for the README) -----------------------------------
    C["cls"] = C.gender + "_" + C.height
    L["cls"] = L.gender + "_" + L.height
    tg = {k: (bark(g.F1).mean(), bark(g.F2).mean()) for k, g in C.groupby(["sound", "cls"])}
    L["ed"] = [float(np.hypot(bark(a) - tg[(s, c)][0], bark(b) - tg[(s, c)][1])) for a, b, s, c in zip(L.F1, L.F2, L.sound, L.cls)]
    n_sets = n_ok = n_absent = 0
    small = large = incomparable = 0
    big_max = 0.0
    for snd, M in (("wu", MU), ("yu", MY)):
        for sub in sorted(L.subject.unique()):
            ed = L[(L.subject == sub) & (L.sound == snd)].ed.values
            m = M[M.subject == sub].accuracy_height.values
            j, ok = 0, True
            for v in m:
                while j < len(ed) and abs(ed[j] - v) > 2.5e-3:
                    j += 1
                if j >= len(ed):
                    ok = False
                    break
                j += 1
            n_sets += 1
            n_ok += ok
            n_absent += (len(ed) - len(m)) if ok else 0
            if not ok:
                if len(ed) == len(m):
                    md = float(np.abs(ed - m).max())
                    big_max = max(big_max, md)
                    small += md < 0.05
                    large += md >= 0.05
                else:
                    incomparable += 1
    print(f"accuracy reproduced from F1/F2 in {n_ok} of {n_sets} learner-vowel sets; {n_absent} tokens absent from the model files in those sets")

    # ---- internal checks ---------------------------------------------------------------------------------------------------
    ck.internal("every row of subject_combined.csv and control_combined.csv kept, in the file order",
                len(rows) == len(L) + len(C) == 1597 + 718, f"{len(rows)} rows")
    ck.internal("58 participants (1001-1041 learners, 2001-2022 controls), none lost", len(pcode) == 58 and len(parts) == 58)
    ck.internal("learner tokens: 797 /u/ and 800 /y/; control tokens: 359 each",
                (L.sound == "wu").sum() == 797 and (L.sound == "yu").sum() == 800 and (C.sound == "wu").sum() == 359 and (C.sound == "yu").sum() == 359)
    ck.internal("model rows: 760 for /u/ and 777 for /y/, all of the learners", len(MU) == 760 and len(MY) == 777
                and set(MU.subject) == set(L.subject) == set(MY.subject))
    ck.internal("F1 and F2 plausible (Hz)", bool(L.F1.between(150, 900).all() and C.F1.between(150, 900).all()
                                              and L.F2.between(300, 3500).all() and C.F2.between(300, 3500).all()))
    ck.internal("accuracy shared for 1,537 model rows, all positive", len(acc) == 1537 and bool((acc.acc > 0).all()))

    # ---- published numbers -------------------------------------------------------------------------------------------------
    sex_l = L.groupby("subject").gender.first().value_counts()
    sex_c = C.groupby("subject").gender.first().value_counts()
    ck.published("learners", L.subject.nunique(), 40, 0, f"{PAPER}, 2.1 Participants")
    ck.published("native controls", C.subject.nunique(), 18, 0, f"{PAPER}, 2.1 Participants")
    ck.published("learners, women", int(sex_l["F"]), 31, 0, f"{PAPER}, 2.1 Participants")
    ck.published("learners, men", int(sex_l["M"]), 9, 0, f"{PAPER}, 2.1 Participants")
    ck.published("controls, men", int(sex_c["M"]), 9, 0, f"{PAPER}, 2.1 Participants")
    ck.published("controls, women", int(sex_c["F"]), 9, 0, f"{PAPER}, 2.1 Participants")
    ck.published("share of learner tokens removed as outliers (21 tokens), %", 100 * 21 / len(L), 1.3, 0.05,
                 f"{PAPER}, 2.4 Data analysis (21 data points, 1.3%)")
    P = pd.DataFrame([{k: v for k, v in p.items() if k != "person"} for p in parts if "pat_percent" in p])
    T1 = {"auditory_jnd": ("Auditory JND", (22.68, 11.03, 5, 49.75)), "pat_percent": ("PAT (%)", (72.50, 11.57, 41.18, 94.12)),
          "pacs": ("PACS", (101.38, 11.07, 77, 122))}
    for c, (nm, (mean_r, sd_r, lo_r, hi_r)) in T1.items():
        x = P[c].astype(float)
        ck.published(f"{nm}: mean", x.mean(), mean_r, 0.006, f"{PAPER}, Table 1")
        ck.published(f"{nm}: SD", x.std(), sd_r, 0.006, f"{PAPER}, Table 1")
        ck.published(f"{nm}: minimum", x.min(), lo_r, 0.006, f"{PAPER}, Table 1")
        ck.published(f"{nm}: maximum", x.max(), hi_r, 0.006, f"{PAPER}, Table 1")
    ck.published("TPT_variability: maximum", P.tpt_variability.max(), 12.43, 0.006, f"{PAPER}, Table 1")
    ck.published("TPT_ED: mean", P.tpt_ed.mean(), 1.36, 0.01, f"{PAPER}, Table 1")
    ck.published("Variability_u: minimum", P.variability_u.min(), 0.36, 0.006, f"{PAPER}, Table 1")
    ck.published("Variability_y: minimum", P.variability_y.min(), 0.22, 0.006, f"{PAPER}, Table 1")
    ck.published("Proficiency: lowest self-rating", P.proficiency.min(), 1, 0, f"{PAPER}, 2.1 Participants (range 1-6)")
    ck.published("Proficiency: highest self-rating", P.proficiency.max(), 6, 0, f"{PAPER}, 2.1 Participants (range 1-6)")
    ck.published("L2 exposure: shortest (years; 4 months)", P.l2_exposure_years.min(), 0.33, 0.006, f"{PAPER}, 2.1 Participants")
    # the /y/ model without participant 1025: PAT is the only significant predictor (beta -0.02, SE 0.01)
    import statsmodels.formula.api as smf
    covs = {p["person"]: p for p in parts if "pat_percent" in p}
    a = acc[acc.sound == "yu"].copy()
    for c in ("l2_exposure_years", "proficiency", "auditory_jnd", "tpt_variability", "tpt_ed", "pat_percent", "pacs", "variability_y"):
        a[c] = [covs[l][c] for l in a.lab]
    a = a[a.lab != "1025"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm("acc ~ l2_exposure_years + proficiency + auditory_jnd + tpt_variability + tpt_ed + pat_percent + pacs + variability_y",
                        a, groups=a["lab"]).fit(reml=True)
    ck.published("/y/ accuracy model without participant 1025: PAT coefficient", m.fe_params["pat_percent"], -0.02, 0.005,
                 f"{PAPER}, 3.2 Predictors of accuracy (random intercept per participant)")
    ck.published("/y/ accuracy model without participant 1025: PAT standard error", m.bse_fe["pat_percent"], 0.01, 0.005,
                 f"{PAPER}, 3.2 Predictors of accuracy")
    for c in ck.items:
        if not c["ok"]:
            print("FAILED:", c)

    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the learners' and controls' token tables into one table (formants as shared, in Hz), added the group, the "
                 "vowel's description and the participants' covariates from the authors' model files (participants.csv); the "
                 "authors' accuracy values are in accuracy_rows.csv; participant numbers replaced by codes."),
        how=[
            "One row is one vowel token (utterance): Mandarin /u/ (the word wu, 物) or /y/ (yu, 遇), each a monosyllable with a "
            "high-falling tone, read aloud by one of 40 English-speaking learners of Mandarin (group 'English-speaking learners of "
            "Mandarin') or 18 native Mandarin controls, in the order of the authors' files (learners first, each speaker's /u/ then /y/ "
            "tokens, about 20 each); trial is the row's order within the participant. f1_hz and f2_hz are the formants at the vowel midpoint, in Hz.",
            "participants.csv has height class (women T/M/S, men MT/MM/MS) for everyone and, for the learners, the covariates of the "
            "paper's models: years of Mandarin study, self-rated proficiency, the two tongue-placement acuity measures, auditory "
            "acuity (JND), PAT, PACS and the production variability of /u/ and /y/.",
            "accuracy_rows.csv holds the authors' model rows (mixed_model_u.csv and mixed_model_y.csv, 760 and 777 tokens): the "
            "accuracy of each token as its Euclidean distance in F1-F2 Bark space to the mean target of the controls of the same sex "
            "and height class (smaller is more accurate); row_in_model_file is the order within the participant and vowel. The model "
            "files carry no formants, so their rows are not attached to the main table (see the differences below).",
            "excluded_by_authors is left empty: the authors removed 21 outlier tokens (distance at least 2.5 MAD from the median of "
            "the speaker's vowel) but do not say which, and participant 1025 is left out of their /y/ variability models only.",
        ],
        unit_note="vowel token (utterance) of one participant",
        differences=[
            f"The shared accuracy values equal the Bark distance recomputed from the shared F1 and F2 (Traunmueller's formula, mean "
            f"target of the controls of the same sex and height class) in {n_ok} of the {n_sets} learner-vowel sets (to 0.0025); in the "
            f"other {n_sets - n_ok} sets they do not: {small} sets differ by less than 0.05 on every token, {large} sets by more (up to "
            f"{big_max:.1f} Bark), and {incomparable} have a different number of tokens in the two files. The model files therefore "
            "seem to come from another pass over the formants (or the targets) than the shared formants; that is why accuracy is "
            "shared in its own table.",
            f"The paper removed 21 outlier tokens from the 1,597 learner tokens, but the model files hold 1,537 rows (60 fewer; "
            f"{n_absent} of the missing tokens fall in the sets that match).",
            "Table 1 of the paper does not match the deposit for several covariates: TPT_variability (paper mean 3.72, minimum 0.28; "
            "deposit 4.12, 0.93), TPT_ED (SD 0.54, minimum 0.40, maximum 3.01; deposit 0.53, 0.42, 2.99), Variability_u (mean 2.02, "
            "maximum 4.27; deposit 3.73, 29.6) and Variability_y (mean 1.64, maximum 14.96; deposit 2.49, 13.6); the paper's mean "
            "Mandarin study time is 4.37 years (SD 2.79, up to 12 years), the deposit's 4.74 (SD 3.09, up to 14).",
            "The paper's other coefficients (variability in /u/ and /y/, beta 0.31 and 0.36) and the Wilcoxon test of /y/ against /u/ "
            "variability (V = 593) were not reproduced from the deposit's variability values (0.07 and 0.24; V = 252).",
        ],
    )
    if not ck.passed():
        sys.exit("checks failed")
    return stdlib.release(ID, "speech", rows, meta, ck, participants=parts, participant_columns=pcols, extra_files=extra)


if __name__ == "__main__":
    main()
