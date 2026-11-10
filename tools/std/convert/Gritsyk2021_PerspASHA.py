"""Gritsyk, Kabakoff, Li, Ayala, Shiller & McAllister (2021), Perspect ASHA SIGs: three measures of oral somatosensory acuity
(oral stereognosis, a phonetic awareness task, bite block adaptation with auditory masking) in 20 adults, related to how much 65 adults
(in the original study, Li et al. 2019) improved producing Mandarin /y/ and /u/ with visual-acoustic or ultrasound biofeedback.

Raw files, raw_extra/Gritsyk2021_PerspASHA/ (OSF 2h9jr; analysis script read as text, manuscript read as text):
  YUS_full_results_with_outliers.csv   65 speakers x 2 vowels: auditory_1 (auditory JND), somato_1 (stereognosis letter size), PA (not documented),
                                       BF_cond, vowel_order, pre/post median ED and change, areas (not documented)
  YUS_bite_block_means.csv             20 speakers: mean ED of baseline and bite block productions
  YUS_PAT_output.csv                   33 speakers: phonetic awareness percent correct, overall and by part
Run from the site folder: python3 -I tools/std/convert/Gritsyk2021_PerspASHA.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Gritsyk2021_PerspASHA"
PAPER = "Gritsyk et al. 2021, "
BF = {"VA": "visual-acoustic biofeedback", "US": "ultrasound biofeedback"}


def build():
    raw = stdlib.raw_dir(ID)
    g = pd.read_csv(raw / "YUS_full_results_with_outliers.csv")
    bb = pd.read_csv(raw / "YUS_bite_block_means.csv")
    pat = pd.read_csv(raw / "YUS_PAT_output.csv", keep_default_na=False, na_values=[""])
    rows = []

    def add(sp, group, measure, desc, stat, value, unit, condition="", phase="other", n=None, pert="none", fb="normal", block=""):
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Somatosensory acuity measures", "group": group, "subject": str(sp),
            "subject_excluded_by_authors": False, "age": "", "sex": "", "handedness": "", "population": "healthy adults",
            "setting": "lab", "device": "", "session": 1, "block": block, "trial": 0, "phase": phase, "time_s": "", "task": "speech",
            "unit": "participant summary", "condition": condition, "pert_type": "bite_block" if pert == "bite" else "none",
            "pert_value": "" if pert == "bite" else 0.0, "pert_unit": "", "feedback": fb, "excluded_by_authors": False,
            "measure": measure, "measure_description": desc, "statistic": stat, "value": value, "value_unit": unit,
            "n_trials": n if n is not None else "", "session_context": ""})

    grp = g.drop_duplicates("speaker").set_index("speaker")
    for sp in grp.index:
        gr = BF[grp.loc[sp, "BF_cond"]]
        order = grp.loc[sp, "vowel_order"]
        for r in g[g.speaker == sp].itertuples(index=False):
            add(sp, gr, "ed_baseline", "Median Euclidean distance (Bark, F1-F2 space) of the 20 baseline productions of the vowel from the center "
                "of the target talker's productions; smaller is more accurate", "median", r.pre_median_ED, "Bark", r.vowel, "baseline", 20, block="pre-training")
            add(sp, gr, "ed_post", "The same median distance for the 20 productions after biofeedback training", "median", r.post_median_ED, "Bark",
                r.vowel, "test", 20, block="post-training")
            add(sp, gr, "ed_change", "Post-training minus baseline median distance; negative means improvement", "value", r.change_in_ED, "Bark",
                r.vowel, "other")
        r0 = g[g.speaker == sp].iloc[0]
        add(sp, gr, "auditory_jnd", "Auditory acuity: just noticeable difference on the /y/-/u/ discrimination task, mean distance between "
            "the stimuli at the last four reversals (smaller is better)", "value", r0.auditory_1, "continuum steps (unit not stated in the deposit)")
        add(sp, gr, "stereognosis_mean_letter_size", "Oral stereognosis: mean letter size across correct responses (smaller is better acuity)",
            "mean", r0.somato_1, "mm")
    for sp, d in bb.groupby("subject"):
        gr = BF[grp.loc[sp, "BF_cond"]]
        r = d.iloc[0]
        add(sp, gr, "bite_block_ed_baseline", "Mean Euclidean distance (Bark) of baseline productions to the baseline mean of their vowel "
            "(production variability, speaking with auditory masking)", "mean", r.Baseline, "Bark", "", "baseline", None, "none", "masked")
        add(sp, gr, "bite_block_ed", "Bite block task: mean Euclidean distance (Bark, F1-F2) between each production with the bite block and "
            "the baseline mean for the vowel, with auditory masking noise; larger means the vowels were perturbed more (less compensation)",
            "mean", r.Bite_Block, "Bark", "", "test", None, "bite", "masked")
        add(sp, gr, "bite_block_ed_diff", "Bite_Block value minus the baseline mean (the authors' ED_diff)", "value", r.ED_diff, "Bark", "",
            "other", None, "bite", "masked")
    names = {"overall": "overall percent correct", "front_back": "front or back consonant questions", "further_back": "which vowel is further back",
             "higher": "which vowel is higher", "lower": "which vowel is lower"}
    for r in pat.itertuples(index=False):
        if r.block not in names:
            continue
        sp = r.participant
        gr = BF[grp.loc[sp, "BF_cond"]] if sp in grp.index else ""
        add(sp, gr, "phonetic_awareness" + ("" if r.block == "overall" else "_" + r.block),
            "Phonetic awareness task (articulator proprioception): percent of correct answers, " + names[r.block],
            "proportion", r.perc, "percent", "")
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, g, bb, pat


def main():
    checks = stdlib.Checks()
    a, g, bb, pat = build()
    checks.published("participants of the original study who are in the data", g.speaker.nunique(), 65, 0,
                     PAPER + "Methods, Participants: 'all 65 participants who completed the original study'")
    checks.published("participants with bite block data (completed the follow-up)", bb.subject.nunique(), 20, 0, PAPER + "Methods, Participants")
    bf = g[g.speaker.isin(bb.subject)].drop_duplicates("speaker").BF_cond.value_counts()
    checks.published("follow-up participants in the visual-acoustic condition", bf["VA"], 8, 0, PAPER + "Methods, Participants")
    checks.published("follow-up participants in the ultrasound condition", bf["US"], 12, 0, PAPER + "Methods, Participants")
    y = g[g.vowel == "y"].set_index("speaker")
    b = bb.set_index("subject")
    p = pat[pat.block == "overall"].set_index("participant").perc
    d = y.loc[b.index].copy()
    d["bb"] = b.Bite_Block
    d["pat"] = p.reindex(d.index)
    for c in ("auditory_1", "somato_1", "pat", "bb"):
        d[c + "_c"] = d[c] - d[c].mean()

    def fit(pred):
        X = sm.add_constant(d[["auditory_1_c", pred + "_c"]])
        return sm.OLS(d.change_in_ED, X).fit()

    m = fit("somato_1")
    checks.published("stereognosis model: stereognosis coefficient", m.params["somato_1_c"], -0.167, 0.0005, PAPER + "Results, Inferential Statistics")
    checks.published("stereognosis model: its SE", m.bse["somato_1_c"], 0.142, 0.0005, PAPER + "Results, Inferential Statistics")
    checks.published("stereognosis model: p", m.pvalues["somato_1_c"], 0.256, 0.0005, PAPER + "Results, Inferential Statistics")
    checks.published("stereognosis model: auditory coefficient", m.params["auditory_1_c"], 0.016, 0.0005, PAPER + "Results, Inferential Statistics")
    m = fit("pat")
    checks.published("phonetic awareness model: coefficient", m.params["pat_c"], 0.011, 0.0005, PAPER + "Results, Inferential Statistics")
    checks.published("phonetic awareness model: SE", m.bse["pat_c"], 0.013, 0.0005, PAPER + "Results, Inferential Statistics")
    m = fit("bb")
    checks.published("bite block model: coefficient", m.params["bb_c"], -1.582, 0.0005, PAPER + "Results, Inferential Statistics")
    checks.published("bite block model: SE", m.bse["bb_c"], 0.605, 0.0005, PAPER + "Results, Inferential Statistics")
    checks.published("bite block model: p", m.pvalues["bb_c"], 0.018, 0.0005, PAPER + "Results, Inferential Statistics")
    checks.internal("change in ED equals post minus pre median ED", bool(np.allclose(g.change_in_ED, g.post_median_ED - g.pre_median_ED, atol=1e-5)))
    checks.internal("bite block ED_diff equals Bite_Block minus Baseline", bool(np.allclose(bb.ED_diff, bb.Bite_Block - bb.Baseline, atol=1e-5)))
    checks.internal("PAT in 33 participants, 5 parts each", pat[pat.block.isin(["overall", "front_back", "further_back", "higher", "lower"])].shape[0] == 165)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    paper["venue"] = "Perspectives of the ASHA Special Interest Groups 6(2):500-512"
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the authors' three result files into one long table of per-participant values. Left out the columns PA, "
                 "pre_mean_area, post_mean_area and change_in_area (the deposit and the paper do not say what they measure) and the "
                 "rows of the PAT file whose block label is empty (not documented). Participant numbers were replaced."),
        how=[
            "The authors shared one value per participant for each measure, so each row is one participant's value of one measure. "
            "Accuracy of Mandarin /y/ and /u/ production is the median Euclidean distance (ED, Bark, F1-F2) of 20 imitations from the target "
            "talker's vowel, before and after 30 minutes of biofeedback training per vowel; the group is the biofeedback type "
            "(visual-acoustic or ultrasound). The bite block measures were collected only in the 20 follow-up participants, with "
            "loud auditory masking noise (feedback = masked); the deposit does not give the bite block orientation per word.",
            "auditory_jnd and stereognosis_mean_letter_size are the authors' auditory_1 and somato_1; the deposit gives no unit for the "
            "auditory JND (the paper describes it as a distance between stimuli on a continuum), so the unit is stated as not given. "
            "Stereognosis is in mm (mean letter size).",
            "The phonetic awareness (PAT) file has 33 participants, 20 of whom are in the bite block file; one of the 33 is not among "
            "the 65 speakers of the first file, so that participant has no group. Age and sex are not shared per participant "
            "(the paper reports 20 females aged 19-23 for the follow-up).",
            "vowel_order (which vowel was trained first) is in the authors' file but is not carried; it is counterbalanced in the paper.",
        ],
        unit_note="participant, measure and condition",
        differences=["Table 1 of the paper (means and SDs) was not available as text; the regression coefficients, standard errors and p values of "
                     "the three models were reproduced instead."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
