"""Zhang (OSF fz8jx, 2026): "Effects of somatosensory feedback and auditory feedback on speech under a tongue height obstruction".
No paper was found (extra_studies.json says so; the OSF project has no description or wiki). The deposit holds:
  productionFullTable.csv   one row per produced pseudoword token: fileName (subj_<n>_<word>_<Pre|AF2|MN1|Post>_<seq>), subj, word
                            (keggif, kiggif, peffif, piffif, tessif, tissif), block (1, 2, 4), seq, Label (E or I, the vowel of the word),
                            F0Hz, F1Hz, F2Hz (Hz), F1Bark/F2Bark (rounded to whole Bark), F1erb/F2erb, gender, BB (BB or NB), AF (AF or MN)
  subject-<n>.csv           60 OpenSesame logs (timings, perception responses, file names; no acoustic values)
  full model results.pdf, trial effect results in block2.pdf, trial effect in block 4.pdf   the authors' Bayesian model summaries
  3dplot*.html, bite block FPCA component (742ec)
What the codes mean is inferred, not documented: BB/NB = with / without a bite block (a tongue-height obstruction; the authors' model
shows F2 about 560 Hz lower in block 2 for BB and no difference in block 4), AF/MN = normal auditory feedback / masking noise during the
production block (the OpenSesame logs name 'productionBlockNormalAF', 'productionBlockMaskingNoise' and 'maskingNoise_play_start' and
hold the word-recording loops 'Pre', 'AF2'/'MN1' and 'Post'), blocks 1, 2, 4 = pretest, the production block under the condition,
posttest. The per-subject OpenSesame logs are not carried (no acoustics; their perception responses are not motor behaviour).
Run from the site folder: python3 -I tools/std/convert/Zhang_tongueObstruction_OSF.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Zhang_tongueObstruction_OSF"
SRC = "the authors' model summaries in the OSF deposit (full model results.pdf)"
GROUP = {("BB", "AF"): "bite block, normal auditory feedback", ("BB", "MN"): "bite block, masking noise",
         ("NB", "AF"): "no bite block, normal auditory feedback", ("NB", "MN"): "no bite block, masking noise"}
PHASE = {1: "baseline", 2: "learning", 4: "aftereffect"}


def nz(x):
    return "" if pd.isna(x) else float(x)


def main():
    checks = stdlib.Checks()
    raw = stdlib.raw_dir(ID)
    x = pd.read_csv(raw / "productionFullTable.csv")
    x = x.sort_values(["subj", "block", "seq"], kind="stable").reset_index(drop=True)
    x["trial"] = x.groupby("subj").cumcount() + 1
    rows = []
    for r in x.itertuples(index=False):
        bb_on = r.BB == "BB" and r.block == 2
        mn_on = r.AF == "MN" and r.block == 2
        rows.append({
            "dataset": ID, "study": ID, "experiment": "Bite block with auditory feedback or masking noise",
            "group": GROUP[(r.BB, r.AF)], "subject": f"S{r.subj}", "subject_excluded_by_authors": False, "age": "",
            "sex": r.gender, "handedness": "", "population": "healthy", "setting": "lab", "device": "", "session": 1,
            "block": f"Block {r.block}", "trial": int(r.trial), "phase": PHASE[r.block], "time_s": "", "task": "speech",
            "unit": "utterance", "condition": "", "pert_type": "bite_block" if bb_on else "none", "pert_value": "" if bb_on else 0.0,
            "pert_unit": "", "feedback": "masked" if mn_on else "normal", "excluded_by_authors": False,
            "stimulus": r.word, "segment": f"first vowel /{'ɛ' if r.Label == 'E' else 'ɪ'}/",
            "measure_window": "", "duration_ms": "", "f0_hz": nz(r.F0Hz), "f1_hz": float(r.F1Hz), "f2_hz": float(r.F2Hz),
            "f1_mel": "", "intensity_au": "", "shift_f1": "", "shift_f2": "",
            "masking_noise": "masking noise (level not documented)" if mn_on else "", "feedback_latency_ms": "",
        })
    a = pd.DataFrame(rows)
    sub = a.drop_duplicates("subject")
    checks.internal("no duplicate token files", not x.fileName.duplicated().any())
    checks.internal("one bite-block and one feedback condition per participant", x.groupby("subj")[["BB", "AF"]].nunique().max().max() == 1)
    checks.internal("every participant has blocks 1, 2 and 4", bool(x.groupby("subj").block.nunique().eq(3).all()))
    checks.internal("F1 and F2 present on every row", bool(x.F1Hz.notna().all() and x.F2Hz.notna().all()))
    checks.published("participants in the authors' models", sub.shape[0], 54, 0, SRC + ", '~subj (Number of levels)'")
    checks.published("pseudowords in the authors' models", x.word.nunique(), 6, 0, SRC + ", '~word (Number of levels)'")
    for c_ in checks.items:
        if not c_["ok"]:
            print("FAILED:", c_)
    st = next(s for s in json.load(open(stdlib.HERE / "source/extra_studies.json")) if s["id"] == ID)
    p = dict(st["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=st["lab"], data_url=st["data_url"], license="none stated", license_url="",
        changes=("Renamed the authors' columns, labelled blocks, groups and the bite-block / masking-noise conditions from the project's "
                 "own file names, ordered tokens by participant, block and sequence number, left out the derived Bark and ERB columns "
                 "and the OpenSesame logs, and replaced participant numbers with codes."),
        how=[
            "One row is one produced pseudoword (keggif, kiggif, peffif, piffif, tessif, tissif) of one participant, with F0, F1 and F2 "
            "in Hz as the authors shared them (the window over which they were measured is not documented).",
            "block 1 is the pretest, block 2 the production block under the participant's condition and block 4 the posttest. Each participant "
            "is in one of four between-participant groups: bite block (BB) or no bite block (NB), crossed with normal auditory feedback (AF) "
            "or masking noise (MN) during block 2.",
            "These meanings are inferred from the project title, the OpenSesame logs and the authors' model summaries, since the deposit has "
            "no readme; pert_type 'bite_block' marks the block-2 rows of the BB groups, the only block in which the model shows the obstruction "
            "(F2 about 560 Hz lower).",
        ],
        unit_note="produced pseudoword of one participant",
        differences=[
            "The authors' models used 2897 observations (962 in block 2, 967 in block 4); the shared table has 2847 (945 and 950), so 50 "
            "tokens of their model data are not in the shared file. 54 participants and 6 words agree.",
            "No paper was found, so no result could be checked beyond the counts above.",
        ],
    )
    return stdlib.release(ID, "speech", rows, meta, checks)


if __name__ == "__main__":
    main()
