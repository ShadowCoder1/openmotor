"""Ayala, Eads, Kabakoff, Swartz, Shiller, Hill, Hitchcock, Preston & McAllister (2023), J Speech Lang Hear Res 66(4):1252-1273:
auditory perception (/r/ identification, discrimination, category goodness) and oral somatosensory acuity (stereognosis, a phonetic
awareness task, bite block adaptation with auditory masking) in 98 children aged 9 to 15.

Raw files, raw_extra/Ayala2023_JSLHR/ (OSF 2hy65, MOSAIC):
  MOSAIC_all_data.csv                       one row per child: age, sex, the auditory and somatosensory scores, review flags, bite block summary
  MOSAIC_bite_block_means_by_vowel.csv      per child and vowel: mean ED of baseline and bite block productions
  analyses.Rmd, bitebock.Rmd                the authors' analysis scripts (read as text)
Paper read in PubMed Central (PMC10187971).
Run from the site folder: python3 -I tools/std/convert/Ayala2023_JSLHR.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Ayala2023_JSLHR"
P = "Ayala et al. 2023, "
# (column, measure name, description, statistic, unit, flag column, group of exclusion)
M = [
    ("identification", "identification_boundary_width", "Identification task (/r/-/l/ continuum of 10 steps): width of the category boundary from a fitted logistic function (25th to 75th percentile), in continuum steps; 0 means an abrupt boundary", "value", "continuum steps", "Flag_IDENT"),
    ("ident_bias", "identification_boundary_location", "Identification task: location of the category boundary along the 10-step continuum", "value", "continuum steps", "Flag_IDENT"),
    ("discrimination", "discrimination_jnd", "Discrimination task (adaptive AXB on a 240-step continuum): just noticeable difference; smaller is finer discrimination", "value", "continuum steps", "Flag_DISCRIM"),
    ("catgood", "category_goodness_percent_correct", "Category goodness judgment of 100 /r/ items: percent correct against expert consensus", "proportion", "percent", "Flag_CATGOOD"),
    ("catgood_aprime", "category_goodness_aprime", "Category goodness judgment: A-prime sensitivity", "value", "A-prime (0.5 to 1)", "Flag_CATGOOD"),
    ("catgood_right", "category_goodness_correct_accepted", "Category goodness judgment: percent of correct tokens accepted", "proportion", "percent", "Flag_CATGOOD"),
    ("catgood_wrong", "category_goodness_incorrect_rejected", "Category goodness judgment: percent of incorrect tokens rejected", "proportion", "percent", "Flag_CATGOOD"),
    ("catgood_cons", "category_goodness_consonantal", "Category goodness judgment: percent correct on consonantal /r/ items", "proportion", "percent", "Flag_CATGOOD"),
    ("catgood_voc", "category_goodness_vocalic", "Category goodness judgment: percent correct on vocalic /r/ items", "proportion", "percent", "Flag_CATGOOD"),
    ("catgood_cluster", "category_goodness_cluster", "Category goodness judgment: percent correct on /r/ items in clusters", "proportion", "percent", "Flag_CATGOOD"),
    ("catgood_not_cluster", "category_goodness_noncluster", "Category goodness judgment: percent correct on /r/ items not in clusters", "proportion", "percent", "Flag_CATGOOD"),
    ("stereognosis_MLS", "stereognosis_mean_letter_size", "Oral stereognosis: mean letter size across correct responses (smaller is better acuity)", "mean", "mm", "Flag_stereognosis"),
    ("PAT", "phonetic_awareness", "Phonetic awareness task (the paper's articulatory awareness): percent correct across all 34 items", "proportion", "percent", "Flag_PAT"),
    ("PAT_higher", "phonetic_awareness_higher", "Phonetic awareness task, part 'which vowel is higher': percent correct", "proportion", "percent", "Flag_PAT"),
    ("PAT_lower", "phonetic_awareness_lower", "Phonetic awareness task, part 'which vowel is lower': percent correct", "proportion", "percent", "Flag_PAT"),
    ("PAT_front_back", "phonetic_awareness_front_back", "Phonetic awareness task, front or back consonant questions: percent correct", "proportion", "percent", "Flag_PAT"),
    ("PAT_further_back", "phonetic_awareness_further_back", "Phonetic awareness task, part 'which vowel is further back': percent correct", "proportion", "percent", "Flag_PAT"),
    ("bite", "bite_block_ed", "Bite block task with auditory masking: mean Euclidean distance (Bark, F1-F2) between productions with the bite block and the baseline mean of the vowel, over the vowels /i u ae a/; smaller means more compensation. May include the authors' imputed value for /a/ (paper: imputed for 18 children)", "mean", "Bark", None),
    ("mean_vertical", "bite_block_ed_vertical", "Bite block task, vertical bite block (high vowels /i u/): mean Euclidean distance (Bark)", "mean", "Bark", None),
    ("mean_horizontal", "bite_block_ed_horizontal", "Bite block task, horizontal bite block (low vowels /ae a/): mean Euclidean distance (Bark)", "mean", "Bark", None),
]
# review exclusions the authors name in analyses.Rmd, by measure column and order of the children in the file (study codes not carried)
EXCL_IDS = {"identification": {3028, 3048}, "discrimination": {3067}, "stereognosis_MLS": {3000, 3003}}
VOWEL = {"@": "/æ/ (had)", "a": "/ɑ/ (hod)", "i": "/i/ (heed)", "u": "/u/ (who'd)", "o": "/oʊ/ (hod produced with /oʊ/ at one site)"}
FLAGGED = "flagged for review by the authors (their Flag column was Yes)"


def build():
    raw = stdlib.raw_dir(ID)
    d = pd.read_csv(raw / "MOSAIC_all_data.csv")
    v = pd.read_csv(raw / "MOSAIC_bite_block_means_by_vowel.csv")
    rows = []
    for c in d.itertuples(index=False):
        c = c._asdict()

        def add(measure, desc, stat, val, unit, cond="", phase="other", n=None, excl=False, flagged=False, pert="none", fb="normal"):
            rows.append({
                "dataset": ID, "study": ID, "experiment": "Auditory and somatosensory measures", "group": "",
                "subject": str(c["Study_ID"]), "subject_excluded_by_authors": False, "age": c["Age"], "sex": "F" if c["Sex"] == "Female" else "M",
                "handedness": "", "population": "children", "setting": "lab", "device": "", "session": 1, "block": "", "trial": 0,
                "phase": phase, "time_s": "", "task": "speech", "unit": "participant summary", "condition": cond,
                "pert_type": "bite_block" if pert == "bite" else "none", "pert_value": "" if pert == "bite" else 0.0, "pert_unit": "",
                "feedback": fb, "excluded_by_authors": bool(excl), "measure": measure, "measure_description": desc, "statistic": stat,
                "value": val, "value_unit": unit, "n_trials": n if n is not None else "", "session_context": FLAGGED if flagged else ""})

        bite_out = c["outlier"] == 1
        for col, name, desc, stat, unit, flag in M:
            val = c[col]
            if pd.isna(val):
                continue
            excl = c["Study_ID"] in EXCL_IDS.get(col, set())
            if col in ("bite", "mean_vertical", "mean_horizontal"):
                excl = bool(bite_out)
            add(name, desc, stat, val, unit, "", "test" if col.startswith(("bite", "mean_")) else "other",
                excl=excl, flagged=bool(flag and c.get(flag) == "Yes"),
                pert="bite" if col.startswith(("bite", "mean_")) else "none", fb="masked" if col.startswith(("bite", "mean_")) else "normal")
        for r in v[v.subject == c["Study_ID"]].itertuples(index=False):
            lab = VOWEL[r.vowel]
            add("bite_block_ed_baseline", "Mean Euclidean distance (Bark) of the baseline productions of the vowel to their own baseline mean (with auditory masking, no bite block)",
                "mean", r.Baseline, "Bark", lab, "baseline", excl=bite_out, fb="masked")
            add("bite_block_ed_by_vowel", "Mean Euclidean distance (Bark, F1-F2) between each bite block production of the vowel and its baseline mean; smaller means more compensation",
                "mean", r.Bite_Block, "Bark", lab, "test", excl=bite_out, pert="bite", fb="masked")
            add("bite_block_ed_diff_by_vowel", "Bite block value minus baseline value for the vowel (the authors' ED_diff)", "value", r.ED_diff, "Bark",
                lab, "other", excl=bite_out, pert="bite", fb="masked")
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, d, v


def main():
    checks = stdlib.Checks()
    a, d, v = build()
    chi = d.drop_duplicates("Study_ID")
    checks.published("children", d.Study_ID.nunique(), 98, 0, P + "Participants (98 children included)")
    checks.published("girls", (d.Sex == "Female").sum(), 58, 0, P + "Participants (58 female, 59.2%)")
    checks.published("mean age (years)", d.Age.mean(), 12.8, 0.05, P + "Table 1")
    checks.published("SD of age", d.Age.std(), 2.0, 0.05, P + "Table 1")
    checks.published("youngest age", d.Age.min(), 9.0, 0.05, P + "Table 1")
    checks.published("oldest age", d.Age.max(), 15.8, 0.05, P + "Table 1")
    checks.published("younger group (up to 12.5 years)", (d.Age <= 12.5).sum(), 43, 0, P + "Table 2 (n = 43)")
    checks.published("older group (over 12.5 years)", (d.Age > 12.5).sum(), 55, 0, P + "Table 2 (n = 55)")
    checks.published("children with a bite block score", d.bite.notna().sum(), 95, 0, P + "Results (bite block completed by 95 of 98)")
    checks.published("children with a stereognosis score", d.stereognosis_MLS.notna().sum(), 77, 0, P + "Methods (stereognosis completed by 77 of 98)")
    checks.published("children with an identification boundary width of 0", (d.identification == 0).sum(), 39, 0, P + "Results (39 of 98)")
    ok = a[~a.excluded_by_authors]

    def med(name):
        return ok[ok.measure == name].value.median()

    checks.published("median phonetic awareness, % correct", med("phonetic_awareness"), 73.5, 0.05, P + "Table 1")
    checks.published("median stereognosis letter size (mm)", med("stereognosis_mean_letter_size"), 5.5, 0.05, P + "Table 1")
    checks.published("median category goodness, % correct", med("category_goodness_percent_correct"), 84.0, 0.05, P + "Table 1")
    checks.internal("every child in the by-vowel file is in the main file", set(v.subject) <= set(d.Study_ID))
    checks.internal("the three children without bite block data have no bite block rows", a[a.measure == "bite_block_ed"].subject.nunique() == 95)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    paper["authors"] = "Ayala SA, Eads A, Kabakoff H, Swartz MT, Shiller DM, Hill J, Hitchcock ER, Preston JL, McAllister T"
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Reshaped the authors' wide child-level table and their per-vowel bite block table into one long table, added age and "
                 "sex from the main table, translated the vowel symbols, and replaced child codes. Marked the authors' exclusions."),
        how=[
            "The authors shared one value per child for each measure (and per child and vowel for the bite block), so each row is one "
            "child's value of one measure. Auditory perception tasks (identification, discrimination, category goodness for /r/) and "
            "the two perception-based acuity tasks are carried with the bite block adaptation task, the one task here in which speech "
            "movements are perturbed (a tongue depressor between the incisors, with loud auditory masking noise).",
            "excluded_by_authors marks the values the authors' analysis scripts removed by name: two identification scores and one "
            "discrimination score judged to reflect inattention, two stereognosis scores from an earlier version of the task, and the "
            "bite block scores of the three children flagged as outliers. Values the scripts dropped by their 3-MAD outlier rule at "
            "analysis time are not marked (the rule was applied to the data in the analysis, not in the shared file). session_context "
            "says where a value carried the authors' review flag.",
            "Vowel codes in the by-vowel file follow the authors' script: @ is /ae/ (had), a is /a/ (hod), o is hod produced with /ou/ "
            "at one site, i is heed, u is who'd. The task-level bite block mean (bite_block_ed) is the authors' column 'bite' and may "
            "include their imputed /a/ value, which the by-vowel file does not.",
        ],
        unit_note="child, measure and condition",
        differences=["The paper prints a median bite block ED of 0.93 Bark; with the three outlier children left out as the authors state, the rows give 0.918, so this number is not used as a check (it is 0.928 with them in).",
                     "The paper's medians in Table 1 are given after the authors' outlier removal; those that need it (the discrimination "
                     "and identification medians) were not used as checks. Spearman correlations were not recomputed."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
