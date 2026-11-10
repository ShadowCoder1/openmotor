"""Beach, Johnson, Parrell & Niziolek (2024; published 2026 in J Exp Psychol Learn Mem Cogn, doi 10.1037/xlm0001663; preprint
bioRxiv 10.1101/2024.08.04.606537): sensorimotor adaptation to a vowel-centralizing formant perturbation during sentence
production, 41 speakers each in an 'adapt' and a 'null' (no perturbation) session.

Raw files, raw_extra/Beach2024_bioRxiv/ (OSF 3fhbg, fetched with tools/std/fetch.py; no license stated):
  sentenceVow_41.mat   MATLAB table, 315,649 rows: one row per vowel token of the 14 vowels of the practiced sentences
  transferVow_41.mat   MATLAB table, 12,188 rows: one row per vowel token of the 10 vowels of the untrained transfer words
     columns of both: f1, f2 (mels; get_AVS_VSA.m of the authors' code, github.com/blab-lab/vsaSentence, says the trial-wise formants
       are in mels), centdist (distance from the speaker's vowel-space center, mels), centdistdiff, diff1, diff2 (this token's
       value minus the speaker's mean for the same vowel in the same session in phase baseline2, or transfer2 for the transfer
       words; checked on the rows: f1-diff1, f2-diff2 and centdist-centdistdiff are constant per speaker, session and vowel and
       equal that mean), subj (1-41), cond (adapt or null), phase, vow (ARPABET), speaker (lab code, replaced here), adaptFirst
  speakerData.mat      MATLAB table, 41 rows: speaker, male, age, daysBtwSessions (null session date minus adapt session date), adaptFirst
     (the wide table of per-speaker phase means and listener scores built from the other files: not released again)
Not released: listenerData.mat (online listeners' demographics and their ids), sentenceAcc/transferAcc (listener transcription
scores, whose item coding is not documented), aavs/avs_vsa/segmentDuration/supplementaryData (per-speaker summaries of the token
files above), and the example speaker's raw adaptation folder (one speaker's expt and dataVals, 0.5 GB).
The tables are MATLAB table objects, decoded without MATLAB by Beach2024_bioRxiv_mcos.py (the MCOS reader).
Design (run_vsaSentence_expt.m): order baseline1, transfer1, baseline2, transfer2, ramp, hold (6 parts), transfer3, washout,
retention (after a 10 min silent delay); a 2-D perturbation field pushing vowels toward the center of the speaker's vowel space,
scaled 0 -> 0.5 over the ramp and 0.5 during hold in the adapt session, always 0 in the null session.
Run from the site folder: python3 -I tools/std/convert/Beach2024_bioRxiv.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stdlib  # noqa: E402
from Beach2024_bioRxiv_mcos import MatObjects  # noqa: E402

ID = "Beach2024_bioRxiv"
SRC = "Beach, Johnson, Parrell & Niziolek 2024"
ORDER = ["baseline1", "transfer1", "baseline2", "transfer2", "ramp", "hold1", "hold2", "hold3", "hold4", "hold5", "hold6",
         "transfer3", "washout", "retention"]
PHASE = {"baseline1": "baseline", "baseline2": "baseline", "transfer1": "transfer", "transfer2": "transfer",
         "transfer3": "transfer", "ramp": "learning", "washout": "washout", "retention": "test"}
for k in range(1, 7):
    PHASE[f"hold{k}"] = "learning"
HOLD_SCALE = 0.5
DEVICE = "microphone and headphones with Audapter real-time formant processing"


def col(c):
    if isinstance(c, np.ndarray):
        return c.ravel()
    if isinstance(c, list):
        return [str(e.ravel()[0]) if isinstance(e, np.ndarray) and e.size == 1 else e for e in c]
    raise ValueError(type(c))


def read_table(path, name=None):
    mo = MatObjects(path)
    key = [k for k in mo.top if not k.startswith("__")]
    assert len(key) == 1, key
    o = mo.variable(mo.top[key[0]])
    assert o["__class__"] == "table", o["__class__"]
    names = [str(n.ravel()[0]) for n in o["varnames"]]
    n = int(o["nrows"].ravel()[0])
    df = pd.DataFrame({k: col(c) for k, c in zip(names, o["data"])})
    assert len(df) == n
    return df


def paper_meta():
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p = dict(s["papers"][0])
    return s, dict(authors=p["authors"], year=2026, title=p["title"].rstrip("."),
                   venue="J Exp Psychol Learn Mem Cogn (preprint: bioRxiv 2024)", doi="10.1037/xlm0001663")


def build():
    raw = stdlib.raw_dir(ID)
    sp = read_table(raw / "speakerData.mat")
    sen = read_table(raw / "sentenceVow_41.mat")
    tra = read_table(raw / "transferVow_41.mat")
    sen["experiment"] = "sentences"
    tra["experiment"] = "transfer words"
    d = pd.concat([sen, tra], ignore_index=True)
    d["file_order"] = np.arange(len(d))
    d["ph_order"] = d.phase.map({p: i for i, p in enumerate(ORDER)})
    assert d.ph_order.notna().all()
    d = d.sort_values(["speaker", "cond", "ph_order", "file_order"], kind="stable").reset_index(drop=True)
    d["trial"] = d.groupby(["speaker", "cond"]).cumcount() + 1
    info = sp.set_index("speaker")
    return d, sp, info


def rows_from(d, info):
    rows = []
    for r in d.itertuples(index=False):
        i = info.loc[r.speaker]
        adapt = r.cond == "adapt"
        shifted = adapt and (r.phase.startswith("hold") or r.phase == "ramp")
        session = 1 if (int(i.adaptFirst) == 1) == adapt else 2
        if adapt and r.phase.startswith("hold"):
            pv = HOLD_SCALE
        elif adapt and r.phase == "ramp":
            pv = ""  # rises 0 -> 0.5 across the ramp trials; the deposit gives no trial order within the ramp
        else:
            pv = 0
        ref = "transfer2" if r.experiment == "transfer words" else "baseline2"
        rows.append({
            "dataset": ID, "study": SRC, "experiment": r.experiment, "group": "",
            "subject": f"{r.speaker}|{r.cond}", "person": r.speaker, "subject_excluded_by_authors": False,
            "age": float(i.age), "sex": "M" if int(i.male) == 1 else "F", "handedness": "",
            "population": "adult speakers across the lifespan", "setting": "lab", "device": DEVICE, "session": session,
            "block": r.phase, "trial": int(r.trial), "phase": PHASE[r.phase], "time_s": "", "task": "speech",
            "unit": "utterance", "condition": r.cond,
            "pert_type": "formant_shift" if shifted else "none", "pert_value": pv,
            "pert_unit": "scale factor of the vowel-centralizing perturbation field (maximum 0.5)" if shifted else "",
            "feedback": "altered" if shifted else "normal", "excluded_by_authors": False,
            "stimulus": "", "segment": f"vowel {r.vow} (ARPABET)", "measure_window": "not stated in the deposit",
            "f1_mel": float(r.f1), "f2_mel": float(r.f2),
            "f1_change_mel": float(r.diff1), "f2_change_mel": float(r.diff2),
            "change_reference": f"this token minus the speaker's mean for the same vowel in {ref} of the same session",
            "distance_to_center_mel": float(r.centdist), "distance_to_center_change_mel": float(r.centdistdiff),
        })
    return rows


def main():
    d, sp, info = build()
    checks = stdlib.Checks()
    R = f"{SRC}, deposit documentation"
    checks.published("speakers", sp.speaker.nunique(), 41, 0, "github.com/blab-lab/vsaSentence README: 'N=41 speakers contributed data'")
    sen = d[d.experiment == "sentences"]
    tra = d[d.experiment == "transfer words"]
    checks.published("vowel categories in the sentences", sen.vow.nunique(), 14, 0, "vsaSentence get_AVS_VSA.m (sentenceVowels, n = 14)")
    checks.published("vowel categories in the transfer words", tra.vow.nunique(), 10, 0, "vsaSentence get_AVS_VSA.m (transferVowels, n = 10)")
    checks.published("phases of the sentence task", sen.phase.nunique(), 11, 0, "vsaSentence get_AVS_VSA.m (sentencePhases)")
    checks.internal("every speaker has an adapt and a null session", bool((d.groupby("speaker").cond.nunique() == 2).all()))
    checks.internal("token counts", len(d) == 315649 + 12188, f"{len(d)}")
    base = d[d.phase.isin(["baseline2", "transfer2"])]
    checks.internal("baseline tokens have zero mean change per speaker, session and vowel",
                    bool(base.groupby(["speaker", "cond", "vow", "experiment"]).diff1.mean().abs().max() < 1e-6))
    checks.internal("f1 and f2 in mel range", bool(d.f1.between(100, 2500).all() and d.f2.between(300, 3500).all()))
    ages = sp.age
    checks.internal("ages", bool(ages.between(18, 80).all()), f"{ages.min()}-{ages.max()}")
    P = f"{SRC} (bioRxiv preprint), Methods, Participants"
    checks.published("male speakers", int((sp.male == 1).sum()), 10, 0, P)
    checks.published("female speakers", int((sp.male == 0).sum()), 31, 0, P)
    checks.published("youngest speaker (years)", ages.min(), 18, 0, P)
    checks.published("oldest speaker (years)", ages.max(), 76, 0, P)
    checks.published("speakers aged 18-30", int(ages.between(18, 30).sum()), 14, 0, P)
    checks.published("speakers aged 31-55", int(ages.between(31, 55).sum()), 14, 0, P)
    checks.internal("speakers aged 56 and over", int((ages >= 56).sum()) == 13, "the preprint says 14 per age band, which sums to the 42 recruited; one older speaker was excluded")
    rows = rows_from(d, info)
    parts = [{"person": r.speaker, "adapt_session_first": bool(int(r.adaptFirst)), "days_between_sessions": abs(float(r.daysBtwSessions))}
             for r in sp.itertuples(index=False)]
    pcols = [("adapt_session_first", "true/false", "", "The adapt session (perturbation) came before the null session (column adaptFirst)."),
             ("days_between_sessions", "number", "days", "Days between the two sessions (absolute value of the authors' daysBtwSessions).")]
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s, paper = paper_meta()
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes=("Decoded the authors' MATLAB tables of vowel tokens, put the sentence and transfer-word tokens into one table in "
                 "the order of the experiment's phases, added the perturbation schedule from the experiment code, and replaced "
                 "speaker identifiers with codes."),
        how=[
            "One row is one vowel token produced by one speaker in one session (adapt or null) in the sentence task or the "
            "transfer-word task; segment is the vowel (ARPABET label).",
            "f1_mel and f2_mel are the token's formants in mels (the authors' code states the trial-wise formants are in mels); "
            "distance_to_center_mel is its distance from the center of that speaker's vowel space. f1_change_mel, f2_change_mel and "
            "distance_to_center_change_mel are the authors' change from the speaker's mean for the same vowel in baseline2 of the same "
            "session (transfer2 for transfer words), which was checked on the rows.",
            "The deposit has no trial numbers or times: trial counts the tokens of a speaker and session in the order of the "
            "experiment's phases (baseline1, transfer1, baseline2, transfer2, ramp, hold1-6, transfer3, washout, retention) and, "
            "inside a phase, in the order of the authors' file, which may not be chronological. block is the authors' phase label.",
            "pert_value is the scale factor (0.5 during hold in the adapt session) of the authors' perturbation field, which pushes "
            "vowels toward the center of the speaker's vowel space; it is empty during the ramp because it rises from 0 to 0.5 and the "
            "deposit has no trial order there. The null session had no perturbation. The window the formants were measured over is not "
            "stated in the deposit.",
        ],
        unit_note="vowel token of a speaker",
        differences=["The published paper (J Exp Psychol Learn Mem Cogn, 2026) could not be read; the participant checks use the bioRxiv "
                     "preprint (Methods) and the authors' code documentation. The preprint's percent changes in average vowel spacing "
                     "were not recomputed because the authors' spacing function is not in their public code. The preprint gives 14 speakers in each of three age bands (42), but 41 speakers were kept and the shared file has 13 aged 56 or over (one older participant was excluded)."],
    )
    return stdlib.release(ID, "speech", rows, meta, checks, participants=parts, participant_columns=pcols)


if __name__ == "__main__":
    main()
