"""Gussow, Weiss & MacDonald (2023), J Exp Psychol Gen 152(10):2775-2792: repetition in tongue twisters and finger fumblers.

Raw files in raw_extra/Gussow2023_JExpPsycholGen/ (OSF ckmgq, folder 'Data and Analyses', no license stated; the accepted manuscript was read at
par.nsf.gov/servlets/purl/10420374):
  FF_data.xlsx  Experiment 1 manual task (finger fumblers): one row per template (unit) of a four-template trial, 103 participants
  MT_data.xlsx  Experiment 1 speech task (monosyllabic tongue twisters): one row per non-word, the same participants
  DT_data.xlsx  Experiment 2 (disyllabic tongue twisters): one row per non-word, 35 other participants
  LTTF_script.R the authors' analysis script (read as text, never run); it names trialType non = AABB, TT = ABBA, ctrl = Control
Columns: subject, trialNum, itemID, wordID, trialType, wordIndS (position 1-4), rep (repetition 1-3), onsetErr, rhymeErr, dvErr.
Run from the site folder: python3 -I tools/std/convert/Gussow2023_JExpPsycholGen.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Gussow2023_JExpPsycholGen"
SRC = "Gussow et al. 2023, Journal of Experimental Psychology: General"
RAW = stdlib.raw_dir(ID)
LAB = {"TT": "ABBA", "non": "AABB", "ctrl": "Control"}
EXPS = [("FF_data.xlsx", "Exp 1a finger fumblers (manual task)", "keypress", "keyboard: left-hand onset keys and right-hand offset keys of each template", "finger fumbler"),
        ("MT_data.xlsx", "Exp 1b monosyllabic tongue twisters (speech task)", "utterance", "microphone recordings, speech read aloud from a scrolling display", "tongue twister"),
        ("DT_data.xlsx", "Exp 2 disyllabic tongue twisters (speech task)", "utterance", "microphone recordings, speech read aloud from a scrolling display", "tongue twister")]


def b(v):
    return "" if pd.isna(v) else bool(v)


def main():
    checks = stdlib.Checks()
    rows, dfs = [], {}
    for fn, exp, unit, dev, kind in EXPS:
        d = pd.read_excel(RAW / fn)
        dfs[fn] = d
        for i, (sub, tn, item, word, tt, pos, rep, on, rh, dv) in enumerate(zip(d.subject, d.trialNum, d.itemID, d.wordID, d.trialType, d.wordIndS, d.rep, d.onsetErr, d.rhymeErr, d.dvErr)):
            code = f"S{int(sub)}"
            rows.append(dict(
                dataset=ID, study=SRC, experiment=exp, group="", subject=f"{fn[:2]}_{code}", person=code, subject_excluded_by_authors=False,
                age="", sex="", handedness="right" if fn != "DT_data.xlsx" else "", population="healthy", setting="lab", device=dev, session=1,
                block="", trial=0, phase="test", time_s="", task="sequence learning", unit=unit, condition=LAB[tt], pert_type="none", pert_value=0,
                pert_unit="", feedback="normal", excluded_by_authors=bool(pd.isna(dv)), word_position=int(pos), trial_repetition=int(rep),
                item_id=int(item), word_id=int(word), onset_error=b(on), offset_error=b(rh), analysed_onset_error=b(dv), authors_trial_number=int(tn)))
    # trial = order of the row within the participant's sequence of rows
    seq = {}
    for r in rows:
        seq[r["subject"]] = seq.get(r["subject"], 0) + 1
        r["trial"] = seq[r["subject"]]

    # published checks
    ff, mt, dt = dfs["FF_data.xlsx"], dfs["MT_data.xlsx"], dfs["DT_data.xlsx"]
    diffs0 = [f"Experiment 1 Method: N = 103 participants in the paper, {ff.subject.nunique()} in the deposit's manual file and {mt.subject.nunique()} in its speech file; "
              "the deposit does not say which were left out of the paper."]
    checks.published("participants, Experiment 2", dt.subject.nunique(), 35, 0, f"{SRC}, Experiment 2, Method, Participants (thirty-five)")
    checks.published("test trials per participant (16 TT/FF, 16 non, 16 Control), Exp 1a", ff.groupby("subject").trialNum.nunique().median(), 48, 0, f"{SRC}, Experiment 1, Procedure (48 test trials)")
    checks.published("speech test trials per participant, Exp 1b", mt.groupby("subject").trialNum.nunique().median(), 48, 0, f"{SRC}, Experiment 1, Method (48 test trials)")
    checks.published("Control trials per participant, Exp 1a", ff[ff.trialType == "ctrl"].groupby("subject").trialNum.nunique().median(), 16, 0, f"{SRC}, Experiment 1, Procedure")

    def tab(df, col, typ, pos):
        return df[(df.trialType == typ) & (df.wordIndS == pos)].groupby("subject")[col].mean().mean()
    T2 = {  # Table 2 onset errors: (type, position) -> M
        "FF": {("TT", 1): .10, ("TT", 2): .12, ("TT", 3): .08, ("TT", 4): .10, ("non", 1): .14, ("non", 2): .07, ("non", 3): .14, ("non", 4): .09,
               ("ctrl", 1): .14, ("ctrl", 2): .12, ("ctrl", 3): .10, ("ctrl", 4): .11},
        "MT": {("TT", 1): .37, ("TT", 2): .36, ("TT", 3): .30, ("TT", 4): .30, ("non", 1): .31, ("non", 2): .25, ("non", 3): .30, ("non", 4): .23,
               ("ctrl", 1): .35, ("ctrl", 2): .26, ("ctrl", 3): .42, ("ctrl", 4): .52}}
    T5 = {("TT", 1): .19, ("TT", 2): .19, ("TT", 3): .10, ("TT", 4): .19, ("non", 1): .16, ("non", 2): .13, ("non", 3): .16, ("non", 4): .09,
          ("ctrl", 1): .15, ("ctrl", 2): .16, ("ctrl", 3): .11, ("ctrl", 4): .16}
    diffs = list(diffs0)
    for name, df, tbl, tno in (("finger fumblers", ff, T2["FF"], "Table 2"), ("monosyllabic tongue twisters", mt, T2["MT"], "Table 2"), ("disyllabic tongue twisters", dt, T5, "Table 5")):
        for (typ, pos), want in tbl.items():
            got = tab(df, "dvErr", typ, pos)
            what = f"onset error proportion, {name}, {LAB[typ]}, position {pos}"
            if abs(got - want) <= 0.005:
                checks.published(what, got, want, 0.005, f"{SRC}, {tno} (mean of by-subject means)")
            else:
                diffs.append(f"{tno}, {what}: {got:.3f} here from the analysed onset errors (dvErr) against {want} in the paper.")
    for name, df, want in (("finger fumblers", ff, .11), ("monosyllabic tongue twisters", mt, .33), ("disyllabic tongue twisters", dt, .15)):
        got = df.dvErr.mean()
        what = f"mean onset error rate, {name}"
        if abs(got - want) <= 0.005:
            checks.published(what, got, want, 0.005, f"{SRC}, Results (M = {want})")
        else:
            diffs.append(f"Results, {what}: {got:.3f} here against {want} in the paper.")
    checks.internal("dvErr is empty or equal to onsetErr", bool(((ff.dvErr.isna()) | (ff.dvErr == ff.onsetErr)).all() and ((mt.dvErr.isna()) | (mt.dvErr == mt.onsetErr)).all()))
    checks.internal("most participants of Exp 1 are in both the manual and the speech file", len(set(ff.subject) & set(mt.subject)) > 90,
                    f"{len(set(ff.subject) & set(mt.subject))} shared of {len(set(ff.subject) | set(mt.subject))}")
    for c in checks.items:
        print(("  ok " if c["ok"] else "FAILED ") + c.get("what", ""), c.get("computed", ""), c.get("reported", ""))
    print("\n".join(diffs))

    s0 = json.load(open(stdlib.HERE / "source/extra_studies.json"))
    s = next(x for x in s0 if x["id"] == ID)
    p = dict(s["papers"][0])
    paper = {k: p[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="none stated", license_url="",
        changes="Joined the three Excel files (finger fumblers, monosyllabic and disyllabic tongue twisters), turned the error codes into true/false, named the trial types and recoded participant numbers as codes.",
        how=["One row is one unit of a four-unit trial: a key-press template in the manual task (Exp 1a: left-hand onset keys then right-hand offset keys) or a non-word in the speech task "
             "(Exp 1b and Exp 2). Each trial of four units was repeated three times (trial_repetition) at a pacing set by a scrolling frame, and each participant did 48 test trials: "
             "16 finger fumbler or tongue twister trials (ABBA onset pattern, condition ABBA), 16 non-twister trials (AABB) and 16 Controls. word_position is the unit's place (1-4) in the trial.",
             "onset_error and offset_error are the authors' coding of any deviation in the onset or the offset of the unit. analysed_onset_error is their dvErr: only the first error of a repetition "
             "was coded, so units after an earlier error are empty (and excluded_by_authors is true for them). Units where nothing was produced were removed by the authors and are absent. "
             "Reaction times and key-press times are not in the deposit.",
             "The same participants did the manual and the speech task of Experiment 1 (subject is one participant's rows in one file, person is the participant). Age, sex and other "
             "demographics are not in the deposit; handedness is right for Experiment 1 because left-handed participants were removed (the paper does not say so for Experiment 2, left empty)."],
        unit_note="unit of a four-unit trial (one key-press template or one non-word)", differences=diffs + ["Most Table 2 and Table 5 cells reproduce to the table's two decimals; the cells listed here, and the three overall mean onset error rates, come out a little lower (by up to 0.03) from dvErr, so the paper's tables were probably built from a slightly different row selection that the deposit does not show."])
    stdlib.release(ID, "speech", rows, meta, checks)


main()
