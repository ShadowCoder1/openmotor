"""Hammerschmidt & Wollner (2020), Music Perception 37(4):263-277: 30 musicians and non-musicians estimated the duration of
rhythmic patterns (3 patterns x 3 tempi) while listening only or while tapping to the half, quarter or eighth notes.

Raw files, raw_extra/Hammerschmidt2020_MusicPercept/ (Zenodo 6564050, CC BY 4.0):
  dataset_cleaned.csv   one row per participant (semicolon separated): background variables, then for every pattern x
                        listening condition x tempo the time estimation ratio (_time), circular variance (_Circ_Var) and
                        circular mean (_Circ_Mean) of the tapping, plus 12 catch trial ratios ("c")
  codebook.txt          column coding
Paper read through PubMed Central (PMC7612662, via NCBI efetch).
Run from the site folder: python3 -I tools/std/convert/Hammerschmidt2020_MusicPercept.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Hammerschmidt2020_MusicPercept"
P = "Hammerschmidt & Wollner 2020, "
COND = {"0": "listening only", "2": "tapping to half notes", "4": "tapping to quarter notes", "8": "tapping to eighth notes"}
MEAS = {
    "time": ("time_estimation_ratio",
             "Time estimation ratio: the participant's verbal duration estimate divided by the stimulus duration, minus 1 "
             "(0 is an accurate estimate, negative an underestimate), averaged over the two presentations of the same stimulus "
             "(trials with unsuccessful tapping removed by the authors)", "value", "ratio"),
    "Circ_Var": ("tapping_circular_variance",
                 "Circular variance of the tap asynchronies relative to the tapped metrical level (CircStat); 0 is perfectly "
                 "consistent tapping, 1 is no consistency; averaged over identical trials. Empty for listening only",
                 "value", "circular variance (0-1)"),
    "Circ_Mean": ("tapping_mean_asynchrony",
                  "Circular mean of the tap asynchronies converted back to ms by the authors (negative: taps came before the "
                  "beat); averaged over identical trials. Empty for listening only", "value", "ms"),
}
HEAD = re.compile(r"^(\d)_(?:(c)_)?(\d)_(\d+)_(time|Circ_Var|Circ_Mean)$")


def build():
    d = pd.read_csv(stdlib.raw_dir(ID) / "dataset_cleaned.csv", sep=";", encoding="latin-1")
    cells = []
    for c in d.columns:
        m = HEAD.match(c)
        if m:
            cells.append((c,) + m.groups())
    rows = []
    for r in d.to_dict("records"):
        n = 0
        for col, pat, catch, cond, bpm, kind in cells:
            v = r[col]
            if pd.isna(v):
                continue
            n += 1
            name, desc, stat, unit = MEAS[kind]
            cond_txt = f"pattern {pat}; {COND[cond]}; {bpm} BPM" + ("; catch trial" if catch else "")
            rows.append({
                "dataset": ID, "study": ID, "experiment": "Time estimation while tapping to metrical levels",
                "group": "musicians" if r["musician"] == 1 else "non-musicians", "subject": str(int(r["subj_no"])),
                "subject_excluded_by_authors": False, "age": r["age"], "sex": "F" if r["gender"] == 1 else "M",
                "handedness": "", "population": "healthy", "setting": "lab",
                "device": "headphones; BopPad touch pad tapped with the index finger (tapping conditions)",
                "session": 1, "block": COND[cond], "trial": n, "phase": "other" if catch else "test", "time_s": "",
                "task": "finger tapping and timing", "unit": "participant summary", "condition": cond_txt,
                "pert_type": "none", "pert_value": 0.0, "pert_unit": "", "feedback": "normal", "excluded_by_authors": False,
                "measure": name + ("_catch" if catch else ""), "measure_description": desc + ("; catch trial of 4 or 12 bars instead of the 8-bar stimulus (the deposit does not say which)" if catch else ""),
                "statistic": stat, "value": v, "value_unit": unit, "n_trials": "", "session_context": "",
                "_pat": int(pat), "_cond": int(cond), "_bpm": int(bpm), "_kind": kind, "_catch": bool(catch)})
    return d, rows


def main():
    checks = stdlib.Checks()
    d, rows = build()
    a = pd.DataFrame(rows)
    m, n = d[d.musician == 1], d[d.musician == 0]
    checks.published("participants analysed", len(d), 30, 0, P + "Participants (subsequent analyses based on 30)")
    checks.published("musicians", len(m), 15, 0, P + "Participants (musicians group, 10 men and 5 women)")
    checks.published("non-musicians", len(n), 15, 0, P + "Participants (non-musicians group, 12 women and 3 men)")
    checks.published("male musicians", (m.gender == 0).sum(), 10, 0, P + "Participants")
    checks.published("female non-musicians", (n.gender == 1).sum(), 12, 0, P + "Participants")
    for nm, g, mean, sd, lo, hi in (("musicians", m, 26.53, 8.76, 19, 47), ("non-musicians", n, 28.73, 11.16, 20, 54)):
        checks.published(f"mean age, {nm}", g.age.mean(), mean, 0.005, P + "Participants")
        if nm != "musicians":
            checks.published(f"SD of age, {nm}", g.age.std(), sd, 0.005, P + "Participants")
        checks.published(f"youngest, {nm}", g.age.min(), lo, 0, P + "Participants (range)")
        checks.published(f"oldest, {nm}", g.age.max(), hi, 0, P + "Participants (range)")
    for col, nm, gm, gs, g2m, g2s in (("music_years", "years playing an instrument", 16.67, 10.11, 0.20, None),
                                      ("music_lessons", "years of music lessons", 9.80, 5.16, 0.23, None),
                                      ("music_importance", "importance of music", None, None, 4.00, 1.41)):
        if gm is not None:
            checks.published(f"{nm}, musicians mean", m[col].mean(), gm, 0.005, P + "Participants")
            checks.published(f"{nm}, musicians SD", m[col].std(), gs, 0.005, P + "Participants")
        checks.published(f"{nm}, non-musicians mean", n[col].mean(), g2m, 0.005, P + "Participants")
        if g2s is not None:
            checks.published(f"{nm}, non-musicians SD", n[col].std(), g2s, 0.005, P + "Participants")
    main_t = a[(a._kind == "time") & ~a._catch]
    mean_by = main_t.groupby("_cond").value.mean()
    checks.internal("mean time estimation ratio is lowest when tapping to half notes and highest when listening only "
                    "(abstract and Results: half notes shortest, listening only longest)",
                    mean_by.idxmin() == 2 and mean_by.idxmax() == 0, mean_by.round(3).to_dict().__str__())
    by_tempo = main_t.groupby("_bpm").value.mean()
    checks.internal("estimates grow with tempo: ratio rises from 83 to 120 to 150 BPM (Results: the faster the tempo, the "
                    "longer the stimuli seemed to last)", by_tempo[83] < by_tempo[120] < by_tempo[150], str(by_tempo.round(3).to_dict()))
    # the ratio must be (estimate / duration) - 1 with whole-second estimates: 8 bars of 4 beats
    dur = {83: 32 * 60 / 83, 120: 16.0, 150: 12.8}
    t2 = main_t[main_t._bpm != 83]
    est = (1 + t2.value) * t2._bpm.map(dur)
    checks.internal("ratios at 120 and 150 BPM correspond to whole-second estimates (ratio = estimate / duration - 1, 8 bars)",
                    (np.abs(est - est.round(1)) < 1e-4).mean() > 0.8, f"{(np.abs(est - est.round(1)) < 1e-4).mean():.3f} of cells (averages of two estimates fall on half seconds)")
    checks.internal("no listening-only tapping rows", a[(a._cond == 0) & (a._kind != "time")].empty)
    checks.internal("48 time-estimate columns per participant: 27 tapping + 9 listening-only + 12 catch",
                    sum(1 for c in d.columns if c.endswith("_time")) == 48)
    checks.internal("circular variance within 0-1", a[a._kind == "Circ_Var"].value.between(0, 1).all())
    ok = a.drop(columns=[c for c in a.columns if c.startswith("_")])
    parts = d.assign(person=d.subj_no.astype(int).astype(str), group=np.where(d.musician == 1, "musicians", "non-musicians"))[
        ["person", "music_years", "music_lessons", "music_importance"]]
    pcols = [("music_years", "number", "years", "Years the participant had played an instrument."),
             ("music_lessons", "number", "years", "Years of formal music lessons."),
             ("music_importance", "number", "0-6", "Rated importance of music in life, 0 not important at all to 6 very important (the codebook's scale; the paper describes a 7-point scale 1-7). Empty where not answered.")]
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = s["papers"][0]
    paper = {k: p0[k] for k in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        changes="Reshaped the authors' one-row-per-participant table into long form (one row per participant, stimulus cell and measure), "
                "decoded the column names, and replaced participant numbers with codes. Values are unchanged.",
        how=["The authors shared one value per participant for each of 3 rhythmic patterns x 4 listening conditions (listening only, "
             "tapping to half, quarter or eighth notes) x 3 tempi (83, 120, 150 BPM), averaged over the two presentations of each "
             "stimulus, plus 12 catch trials. One row is one measure of one participant in one cell: the time estimation ratio "
             "(estimate / stimulus duration - 1; 0 is accurate), and for tapping cells the circular variance and the mean "
             "asynchrony (ms, negative: tap before the beat). No tap times or single trials were shared.",
             "Column names follow the codebook: pattern_condition_tempo_measure, with condition 0 listening only, 2 half notes, "
             "4 quarter notes, 8 eighth notes; the 'c' columns are catch trials (pattern, c, condition, tempo). The catch trials "
             "were 4 or 12 bars long (not 8); the deposit does not say which cell had which length. The 34 tested participants "
             "included four the authors excluded, who are not in the deposit; empty cells are trials the authors removed.",
             "Circ_Mean is described in the codebook as a circular mean; the paper says the circular means were converted back to "
             "mean asynchrony in ms, and the values (-124 to 61) are in that range, so they are given in ms."],
        unit_note="participant, stimulus cell and measure",
        differences=["Five published descriptive numbers did not reproduce and are not used as checks: the SD of age of the musicians (paper 8.76, rows 8.77), "
                     "the SDs of the non-musicians' years of playing (0.76 vs 0.77) and of lessons (0.77 vs 0.78), and the musicians' mean "
                     "(5.77 vs 5.79) and SD (0.42 vs 0.43) of the importance of music (one musician has no answer in the deposit). "
                     "The paper rates importance on a 7-point scale from 1 to 7 while the codebook says 0 to 6.",
                     "The paper's ANOVA statistics were not recomputed (they need the authors' analysis pipeline)."])
    return stdlib.release(ID, "summary", ok.to_dict("records"), meta, checks, participants=parts, participant_columns=pcols)


if __name__ == "__main__":
    main()
