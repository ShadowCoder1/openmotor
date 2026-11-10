"""Doty, Shiller, Retamales, Franklin, Jackson, Limon, Velasco, Novak & McAllister (2026), J Speech Lang Hear Res: adaptation to
altered formant feedback that lowers F1 and F2 (sized to the typical vocal tract difference between men and women) in 20 cisgender
men and 6 transgender women, with a perceptual rating task by listeners afterwards.

Raw files, raw_extra/Doty2026_JSLHR/ (OSF 3h8at, wiki read as text):
  Gtrain_proj_cleaned_cis.csv / _trans.csv   proj (direct adaptation, mel: change in F1 and F2 directly offsetting the perturbation),
        File (projMel_<subject>_<vowel>_<block>_MID), Subj, VowelNum, Vowel (bid bed bad booed bud bod), Block (block2-block5)
        one row per trial; blocks 2-5 = Ramp, Hold 1, Hold 2, After-Effect (GTrain_proj_analysis_public.Rmd); baseline is not a block here
  prolific_processed_data_anonymized.csv     listeners' forced choices (perception; not carried)
Run from the site folder: python3 -I tools/std/convert/Doty2026_JSLHR.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Doty2026_JSLHR"
PAPER_SRC = "Doty et al. 2026, J Speech Lang Hear Res (abstract) and the deposit's analysis script"
BLOCK = {"block2": "Ramp", "block3": "Hold 1", "block4": "Hold 2", "block5": "After-Effect"}
PHASE = {"Ramp": "learning", "Hold 1": "learning", "Hold 2": "learning", "After-Effect": "aftereffect"}
VOWELS = ["bid", "bed", "bad", "booed", "bud", "bod"]
GROUPS = {"cis": ("Cis men", "M", "healthy"), "trans": ("Trans women", "F", "transgender women")}


def build():
    raw = stdlib.raw_dir(ID)
    rows = []
    for key, (gname, sex, pop) in GROUPS.items():
        d = pd.read_csv(raw / f"Gtrain_proj_cleaned_{key}.csv")
        d["sn"] = d.Subj.str[3:].astype(int)
        d["vi"] = d.Vowel.map({v: i for i, v in enumerate(VOWELS)})
        d["bn"] = d.Block.str[-1].astype(int)
        d["pos"] = range(len(d))
        d = d.sort_values(["sn", "bn", "vi", "pos"], kind="stable")  # file order within each subject, block and vowel
        for r in d.itertuples(index=False):
            b = BLOCK[r.Block]
            shifted = b != "After-Effect"
            rows.append({
                "dataset": ID, "study": ID, "experiment": "Altered formant feedback", "group": gname,
                "subject": f"{key}{r.sn}", "subject_excluded_by_authors": False, "age": "", "sex": sex, "handedness": "",
                "population": pop, "setting": "lab", "device": "", "session": 1, "block": b, "trial": 0,
                "phase": PHASE[b], "time_s": "", "task": "speech", "unit": "utterance", "condition": r.Vowel,
                "pert_type": "formant_shift" if shifted else "none", "pert_value": "" if shifted else 0.0, "pert_unit": "mel" if shifted else "",
                "feedback": "altered" if shifted else "normal", "excluded_by_authors": False,
                "stimulus": r.Vowel, "segment": "vowel", "measure_window": "mid-vowel (file label MID)",
                "compensation_mel": float(r.proj),
            })
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1  # order in this table, see README
    return a


def main():
    checks = stdlib.Checks()
    a = build()
    n = a.groupby("group").subject.nunique().to_dict()
    checks.published("cisgender men", n["Cis men"], 20, 0, f"{PAPER_SRC}, Abstract")
    checks.published("transgender women", n["Trans women"], 6, 0, f"{PAPER_SRC}, Abstract")
    checks.published("participants in total", a.subject.nunique(), 26, 0, f"{PAPER_SRC}, Abstract (20 + 6)")
    checks.internal("21392 trials: 4920 (trans) + 16472 (cis), as in the authors' files", len(a) == 21392)
    checks.internal("every participant has all four blocks and all six vowels",
                    bool(a.groupby("subject").block.nunique().eq(4).all() and a.groupby("subject").stimulus.nunique().eq(6).all()))
    checks.internal("direct adaptation within -300 to 300 mel", bool(a.compensation_mel.between(-300, 300).all()))
    m = a.groupby(["group", "block"]).compensation_mel.mean()
    checks.internal("group means rise from Ramp to Hold 2 and fall in the After-Effect (the paper's descriptive result), both groups",
                    bool(all(m[g, "Ramp"] < m[g, "Hold 1"] < m[g, "Hold 2"] and m[g, "After-Effect"] < m[g, "Hold 2"] and m[g, "After-Effect"] > 0
                             for g in ("Cis men", "Trans women"))))
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    paper["venue"] = "Journal of Speech, Language, and Hearing Research"
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Joined the authors' cisgender and transgender trial files into one table, translated their block codes to the "
                 "block names of their analysis script, left out the listeners' perceptual ratings (perception, not movement), and "
                 "replaced participant codes."),
        how=[
            "One row is one production of one of six words (bid, bed, bad, booed, bud, bod) by one participant; compensation_mel is the "
            "authors' 'proj', their direct adaptation in mels: the change of F1 and F2 from baseline projected on the direction that offsets "
            "the feedback shift (positive = opposing the shift), measured at the middle of the vowel.",
            "Blocks are the authors' Ramp, Hold 1, Hold 2 and After-Effect (their block2 to block5); the baseline block is not in the "
            "deposit because proj is already relative to it. About 40 productions per vowel per block are present (about 20 in the After-Effect); "
            "trials the authors removed are absent.",
            "The files give no session trial number: rows are in the authors' file order within each block and vowel, and trial counts them "
            "in the order block, vowel (bid, bed, bad, booed, bud, bod), file order, so it is not the order in which the words were spoken.",
            "The shift lowered F1 and F2 (paper abstract), but its size in mels, the device and the ramp are not in the deposit, so pert_value "
            "and device are left empty; the feedback is normal in the After-Effect block.",
        ],
        unit_note="word produced by one participant",
        differences=[
            "Only counts of participants could be checked: the paper's full text was not reachable (publisher site blocked, not in "
            "PubMed Central), so its reported adaptation percentages (27% cis, 35% trans) and model estimates were not reproduced.",
        ],
    )
    a = a.copy()
    return stdlib.release(ID, "speech", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
