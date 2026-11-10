"""Mudarris, Krijt, Hassell, Murphy, Ruitenberg, Fokkema & Schaefer (2025), Front Neurosci 19:1553548: auditory-cued finger
tapping (index finger of the dominant hand, to a metronome at 120 BPM or a music excerpt) alone (single task) and while doing a
2-back task (dual task), with cognitive and motor tests, in 50 young adults (45 with tapping data).

Raw files, raw_extra/Mudarris2025_FrontNeurosci/ (OSF pfhn3): mlm.csv (one row per participant x cue x task: Cognitive_motor_abilities_
predict_cued_tapping_MLM_dataset_202411.csv), gams.csv (one row per participant x cue, with dual task costs: ..._GAMs_data_long_202411.csv),
mlm.R (analysis script, read as text only). Paper read at frontiersin.org (open access).
Run from the site folder: python3 -I tools/std/convert/Mudarris2025_FrontNeurosci.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

ID = "Mudarris2025_FrontNeurosci"
P = "Mudarris et al. 2025, "
CUE = {1: "metronome", 2: "music"}
TASK = {1: "single task", 2: "dual task"}
# (column, measure, description, unit)
TESTS = [
    ("RAVLT_T5_DL", "ravlt_trial5_minus_delayed", "Rey Auditory Verbal Learning Test: words recalled on the 5th immediate trial minus delayed recall (lower is better memory)", "words"),
    ("RAVLT_T5", "ravlt_trial5", "Rey Auditory Verbal Learning Test: words recalled on the 5th immediate trial", "words"),
    ("RAVLT_DELAY", "ravlt_delayed", "Rey Auditory Verbal Learning Test: words recalled after the delay", "words"),
    ("STROOP_CWI", "stroop_interference", "Stroop colour-word interference score of the authors' column STROOP_CWI (the paper: incongruent minus congruent time; higher is poorer inhibition)", "s"),
    ("STRP_T1_TIME", "stroop_card1_time", "Stroop card 1 completion time", "s"),
    ("STRP_T2_TIME", "stroop_card2_time", "Stroop card 2 completion time", "s"),
    ("STRP_T3_TIME", "stroop_card3_time", "Stroop card 3 completion time", "s"),
    ("TMT_A_TIME", "tmt_a_time", "Trail Making Test part A completion time", "s"),
    ("TMT_B_TIME", "tmt_b_time", "Trail Making Test part B completion time", "s"),
    ("TMT_B_A_Time", "tmt_b_minus_a", "Trail Making Test part B time minus part A time (switching; lower is better)", "s"),
    ("D2_CHR", "d2_corrected_hit_rate", "d2 cancellation test: hits minus false positives over 14 trials of 20 s (sustained attention; higher is better)", "items"),
    ("D2_CORRECT", "d2_correct", "d2 cancellation test: correct hits", "items"),
    ("D2_FALSE", "d2_false_positives", "d2 cancellation test: false positives", "items"),
    ("GPT_TIME_DH", "grooved_pegboard_time", "Grooved Pegboard Test completion time, dominant hand (shorter is better)", "s"),
    ("BBT_DH_COUNT", "box_and_blocks", "Box and Blocks Test: blocks moved in 60 s, dominant hand", "blocks"),
    ("GMSI_F3", "gmsi_musical_training", "Goldsmiths Musical Sophistication Index, musical training subscale", "score"),
]


def build():
    raw = stdlib.raw_dir(ID)
    m = pd.read_csv(raw / "mlm.csv", na_values=["", " ", "."])
    g = pd.read_csv(raw / "gams.csv", na_values=["", " ", "."])
    for c in ("CV", "Force", "TMT_B_A_Time", "TMT_A_TIME"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    has_tap = set(m.dropna(subset=["CV", "Force"], how="all").PPT)
    rows = []
    for ppt, grp in m.groupby("PPT", sort=False):
        p = grp.iloc[0]
        base = {"dataset": ID, "study": ID, "experiment": "Cued finger tapping, single and dual task", "group": "", "subject": f"P{int(ppt)}",
                "subject_excluded_by_authors": ppt not in has_tap, "age": p.AGE, "sex": "M" if p.GENDER == 1 else "F", "handedness": "",
                "population": "healthy", "setting": "lab", "device": "MIDI drum pad", "session": 1, "block": "", "phase": "test", "time_s": "",
                "task": "finger tapping and timing", "unit": "participant summary", "pert_type": "none", "pert_value": 0.0, "pert_unit": "",
                "feedback": "normal", "excluded_by_authors": False}

        def add(measure, desc, stat, val, unit, cond, ctx=""):
            if pd.isna(val):
                return
            rows.append({**base, "condition": cond, "measure": measure, "measure_description": desc, "statistic": stat, "value": val,
                         "value_unit": unit, "n_trials": "", "session_context": ctx, "trial": 0})
        for r in grp.sort_values(["Condition", "Trial"]).itertuples():
            cond = f"{CUE[r.Condition]}, {TASK[r.Trial]}"
            add("tap_cv", "Coefficient of variation of the inter-tap intervals of the trial (SD / mean; lower is more consistent), after dropping the first 4 of 64 beats; trials with fewer than 40 taps are empty", "other", r.CV, "ratio", cond)
            add("tap_force", "Strike force of the taps, MIDI velocity (0-127 scale), averaged over the trial", "mean", r.Force, "MIDI velocity", cond)
        for r in g[g.PPT == ppt].sort_values("Condition").itertuples():
            cond = f"{CUE[r.Condition]}, dual task cost"
            add("tap_cv_dual_task_cost", "Dual task cost on tap_cv as the authors' CV_DTC: single task value minus dual task value (negative: less consistent in the dual task)", "value", pd.to_numeric(r.CV_DTC, errors="coerce"), "ratio", cond)
            add("tap_force_dual_task_cost", "Dual task cost on tap_force as the authors' FORCE_DTC: single task value minus dual task value (positive: lower force in the dual task)", "value", pd.to_numeric(r.FORCE_DTC, errors="coerce"), "MIDI velocity", cond)
        for col, name, desc, unit in TESTS:
            add(name, desc, "value", p[col] if col != "TMT_A_TIME" else pd.to_numeric(p[col], errors="coerce"), unit, "")
    a = pd.DataFrame(rows)
    a["trial"] = a.groupby("subject").cumcount() + 1
    return a, m, g


def main():
    checks = stdlib.Checks()
    a, m, g = build()
    pp = m.drop_duplicates("PPT")

    def cell(measure, cond):
        return a[(a.measure == measure) & (a.condition == cond)].value

    checks.published("participants tested", pp.PPT.nunique(), 50, 0, P + "Participants (50)")
    checks.published("participants in the tapping analyses", a[(a.measure == "tap_cv")].subject.nunique(), 45, 0, P + "Participants (45 analysed)")
    checks.published("men", (pp.GENDER == 1).sum(), 12, 0, P + "Table 1 (12 male)")
    checks.published("women", (pp.GENDER == 0).sum(), 38, 0, P + "Table 1 (38 female)")
    checks.published("mean age, 50 participants (years)", pp.AGE.mean(), 22.22, 0.005, P + "Participants (M = 22.22)")
    checks.published("SD of age", pp.AGE.std(), 4.05, 0.005, P + "Table 1 (SD 4.05)")
    for cc, n1, n2 in (("metronome", 44, 44), ("music", 45, 45)):
        checks.published(f"tap_cv values, {cc}, single task", len(cell("tap_cv", f"{cc}, single task")), n1, 0, P + "Table 2 (n)")
    for cond, mean, sd in (("metronome, single task", 0.07, 0.07), ("music, single task", 0.05, 0.03), ("metronome, dual task", 0.07, 0.05), ("music, dual task", 0.06, 0.05)):
        v = cell("tap_cv", cond)
        if cond != "metronome, single task":  # does not reproduce (0.0647 vs 0.07): listed under differences
            checks.published(f"CV mean, {cond}", v.mean(), mean, 0.005, P + "Table 2")
        checks.published(f"CV SD, {cond}", v.std(), sd, 0.005, P + "Table 2")
    for cond, mean, sd in (("metronome, single task", 64.72, 30.52), ("music, single task", 65.36, 30.78), ("metronome, dual task", 67.04, 30.52), ("music, dual task", 66.16, 32.38)):
        v = cell("tap_force", cond)
        checks.published(f"force mean, {cond}", v.mean(), mean, 0.005, P + "Table 2")
        if cond != "metronome, single task":  # SD 30.515 vs 30.52 does not reproduce: listed under differences
            checks.published(f"force SD, {cond}", v.std(), sd, 0.005, P + "Table 2")
    v = cell("tap_force_dual_task_cost", "metronome, dual task cost")
    checks.published("force dual task cost mean, metronome", v.mean(), -2.3, 0.05, P + "Table 2")
    checks.published("force dual task cost SD, metronome", v.std(), 14.70, 0.005, P + "Table 2")
    pt = a.drop_duplicates(["subject", "measure"])
    for name, n, mean, sd in (("ravlt_trial5_minus_delayed", 50, 1.0, 1.5), ("stroop_interference", 50, 32.2, 12.6), ("tmt_b_minus_a", 49, 21.9, 16.5),
                              ("d2_corrected_hit_rate", 50, 224.6, 42.4), ("grooved_pegboard_time", 50, 59.3, 8.4), ("box_and_blocks", 50, 65.2, 9.1),
                              ("gmsi_musical_training", 50, 23.3, 10.6)):
        v = pt[pt.measure == name].value
        checks.published(f"{name}: n", len(v), n, 0, P + "Table 3")
        checks.published(f"{name}: mean", v.mean(), mean, 0.05, P + "Table 3")
        if name not in ("tmt_b_minus_a", "d2_corrected_hit_rate", "grooved_pegboard_time"):  # SDs do not reproduce: listed under differences
            checks.published(f"{name}: SD", v.std(), sd, 0.05, P + "Table 3")
    # internal: the two authors' files agree
    gg = g.copy()
    gg["CV_ST"] = pd.to_numeric(gg.CV_ST, errors="coerce")
    j = m[m.Trial == 1].assign(CV=lambda x: pd.to_numeric(x.CV, errors="coerce")).merge(gg, on=["PPT", "Condition"])
    checks.internal("tap_cv of trial 1 equals CV_ST of the other file", bool(np.allclose(j.CV.fillna(-1), j.CV_ST.fillna(-1))))
    checks.internal("five participants without tapping data", a[a.subject_excluded_by_authors].subject.nunique() == 5)
    for c in checks.items:
        if not c["ok"]:
            print("FAILED:", c)
    s = next(x for x in json.load(open(stdlib.HERE / "source/extra_studies.json")) if x["id"] == ID)
    p0 = dict(s["papers"][0])
    paper = {q: p0[q] for q in ("authors", "year", "title", "venue", "doi")}
    meta = dict(
        study=ID, paper=paper, lab=s["lab"], data_url=s["data_url"], license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
        changes=("Reshaped the authors' two analysis-ready files into one long table of measures (tapping consistency and force per cue and "
                 "task, the authors' dual task costs, and the cognitive and motor test scores), labelled the numeric codes, and replaced "
                 "participant numbers."),
        how=[
            "One row is one measure of one participant. Tapping: the participant tapped the index finger of the dominant hand on a MIDI "
            "drum pad to a metronome at 120 BPM or to a music excerpt, first alone (single task), then while doing a 2-back task (dual "
            "task). tap_cv is the coefficient of variation of the inter-tap intervals (a ratio), tap_force the strike force as MIDI velocity. "
            "Only these trial summaries were shared, not the tap times.",
            "The deposit labels Condition 1 and 2 and Trial 1 and 2 with numbers only. Condition 1 is the metronome and 2 the music "
            "(inferred: the metronome has 44 CV values and the music 45, as the paper's Table 2 states, and the means match), Trial 1 is the "
            "single task and Trial 2 the dual task (inferred from the authors' second file, whose CV_ST column equals Trial 1's value). Gender "
            "0 is female and 1 male (inferred from the counts, 38 and 12, in the paper's Table 1).",
            "Five participants have no tapping data (technical problems or too few taps); they are marked as excluded by the authors "
            "from the tapping analyses but their test scores are kept. Trials with fewer than 40 taps are empty in the deposit.",
        ],
        unit_note="participant, measure and condition",
        differences=["Five printed numbers did not reproduce and are not used as checks: the single task CV mean with the metronome (rows 0.0647, Table 2 0.07), the single task force SD with the metronome (30.515 against 30.52), and the SDs of the Trail Making B minus A (16.44 against 16.5), d2 (42.35 against 42.4) and Grooved Pegboard (8.35 against 8.4) scores in Table 3. The cause is not known (the authors may have computed them on a slightly different set of rows).",
                     "The paper's text gives mean age 22.22 (SD 4.04) and its Table 1 22.20 (SD 4.05); the rows give 22.22 and 4.047, so the mean is checked against the text and the SD against Table 1.",
                     "The paper's Table 3 lists a minimum Trail Making B minus A of -5 s; the rows contain the same value, as shared by the authors."],
    )
    return stdlib.release(ID, "summary", a.to_dict("records"), meta, checks)


if __name__ == "__main__":
    main()
