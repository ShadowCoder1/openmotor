"""Caudrelier et al. (2019), Cognition 192:103973: formant-perturbation learning in 4-5-year-old and 7-8-year-old French children
and adults; training on the syllable /pe/, transfer to the two-syllable word /epe/ ('epee'), then /pe/ again for the after-effect.
One row per measured vowel (F1, F2 in Bark, F2-F1 relative to the participant's baseline).

Raw file (raw_extra/Caudrelier2019_Cognition/Formants_data_Cognition.csv, Zenodo 2645764, CC BY 4.0), read as text:
Subj_ID, Age_group (4, 8, 20), Adapted (0/1 per participant), Phase, Block (5-17), Trial (within block), Word (pe, epee),
Syllable_position (1, 2), Phonetic_syllable, f1_barks, f2_barks, f2mf1_norm, f2mf1_change.
Run from the site folder: python3 -I tools/std/convert/Caudrelier2019_Cognition.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Caudrelier2019_Cognition"
RAW = stdlib.raw_dir(ID)
checks = stdlib.Checks()
d = pd.read_csv(RAW / "Formants_data_Cognition.csv", sep=None, engine="python")
src = {r["file"]: r for r in json.load(open(RAW / "SOURCES.json"))}
checks.internal("file has the downloaded hash", stdlib.sha256(RAW / "Formants_data_Cognition.csv") == src["Formants_data_Cognition.csv"]["sha256"])

PHASE = {"baseline": "baseline", "training": "learning", "transfer": "transfer", "after-effect": "aftereffect"}
GROUP = {4: "age 4-5 years", 8: "age 7-8 years", 20: "adults"}
rows = []
for sid, s in d.groupby("Subj_ID", sort=False):
    for i, r in enumerate(s.itertuples(index=False), 1):
        tr = r.Phase == "training"
        rows.append(dict(
            dataset=ID, study="Caudrelier 2019 Cognition", experiment="formant perturbation, pe to epee transfer", group=GROUP[r.Age_group],
            subject=f"s{sid}", person=f"s{sid}", subject_excluded_by_authors="", age="", sex="", handedness="", population="healthy",
            setting="lab", device="microphone and headphones, real-time formant shifting", session=1, block=f"block {r.Block}",
            trial=i, phase=PHASE[r.Phase], time_s="", task="speech", unit="utterance", condition=r.Word,
            pert_type="formant_shift" if tr else "none", pert_value="" if tr else 0, pert_unit="", feedback="altered" if tr else "normal",
            excluded_by_authors="", stimulus=r.Word, segment=f"vowel {r.Phonetic_syllable}", syllable_position=r.Syllable_position,
            f1_bark=r.f1_barks, f2_bark=r.f2_barks, f2mf1_norm=r.f2mf1_norm, f2mf1_change_pct=r.f2mf1_change,
            adapted_by_authors=bool(r.Adapted)))
a = pd.DataFrame(rows)
_names = [c[1] for c in stdlib.columns_for("speech")]
a = a.reindex(columns=_names)
per = d.drop_duplicates("Subj_ID").groupby("Age_group").size()
checks.published("children aged 4-5 years", per[4], 29, 0, "abstract (PubMed): 29 preschoolers")
checks.published("children aged 7-8 years", per[8], 24, 0, "abstract (PubMed): 24 schoolchildren")
checks.published("adults", per[20], 17, 0, "abstract (PubMed): 17 adults")
checks.internal("all rows kept", len(a) == len(d) == 11650)
checks.internal("f2mf1_change = 100 x (f2mf1_norm - 1)", bool(np.allclose(d.f2mf1_change.dropna(), 100 * (d.f2mf1_norm.dropna() - 1), atol=1e-4)))
checks.internal("Adapted constant within participant", bool((d.groupby("Subj_ID").Adapted.nunique() == 1).all()))
for c in checks.items:
    if not c["ok"]:
        print("FAILED", c)
s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
p = s["papers"][0]
meta = dict(
    paper={k: p[k] for k in ("authors", "year", "title", "venue", "doi")}, lab=s["lab"], data_url=s["data_url"],
    license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
    changes="Renamed columns, added the CORE columns (phase, perturbation, feedback), recoded Phase and the Adapted flag, and replaced participant numbers with codes. Values are the authors'.",
    how=["One row is one measured vowel: the vowel of the syllable /pe/ (Word pe) or the first or second vowel of the word /epe/ (Word epee, syllable_position 1 or 2). f1_bark and f2_bark are the authors' formants in Bark, f2mf1_norm is F2-F1 divided by the participant's baseline value and f2mf1_change_pct the percent change from baseline.",
         "Blocks 5-6 (pe) and 7-8 (epee) are baseline, 9-13 training (pe), 14-15 transfer (epee) and 16-17 after-effect (pe), as the authors' Phase column labels them. trial is the row's position in the file within the participant (the authors' within-block Trial is not kept, since epee blocks hold two rows per trial).",
         "The deposit does not state the size or direction of the feedback shift or when it was applied. Training rows are coded as formant_shift with altered feedback, and all other phases as unperturbed, from the phase names and the authors' related adult study (F1 raised, F2 lowered, which raises F2-F1 as compensation); pert_value is left empty.",
         "age_group (4-5 years, 7-8 years, adults) is in group; individual age and sex are not shared. adapted_by_authors is the authors' per-participant Adapted flag; its criterion is not stated in the deposit, and no participant is removed."],
    unit_note="measured vowel", differences=["The paper's full text could not be read (publisher and repository blocked automated access), so only the participant counts from the abstract are checked."])
stdlib.release(ID, "speech", a, meta, checks)
