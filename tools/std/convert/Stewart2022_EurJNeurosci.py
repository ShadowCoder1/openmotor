"""Stewart, Baird, Lewis, Fritz & Fridriksson (2022) Eur J Neurosci: practice of motor action selection after stroke.

Raw files (OSF v46up, folder Joystick Data): JoystickData.xlsx (joy.xlsx here; 32 participants x 49 per-participant measures of the
joystick Select / Execute task) and CodeBook_JoystickData.xlsx; Clinical_Training_Data.xlsx (clin.xlsx) is read only for the
participant columns age, sex, hand dominance, months post stroke, weakness and lesion side. Nothing is executed.
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Stewart2022_EurJNeurosci"
RAW = stdlib.raw_dir(DS)
COND = {"EO": "Execute", "AS": "Select"}
DESC = {"Accuracy": ("direction accuracy", "Percent of trials with the correct joystick direction, mean across trials", "percent"),
        "RT": ("response time", "Response (reaction) time, mean across practice trials and blocks", "s"),
        "MT": ("movement time", "Movement time, mean across practice trials and blocks", "s")}


def parse(col):
    """-> (measure key, condition, session, block label, phase, setting detail, statistic) from a column name."""
    m = re.fullmatch(r"Accuracy_(EO|AS)_MRI(Day1|Day4|fu)_run([12])", col)
    if m:
        day = {"Day1": 1, "Day4": 4, "fu": 5}[m.group(2)]
        return "Accuracy", COND[m.group(1)], day, f"MRI task {m.group(2)} run {m.group(3)}", "other", "mean"
    m = re.fullmatch(r"(Accuracy|RT|MT)_(EO|AS)_TaskPractice_(Day[1-4]|FU)", col)
    if m:
        day = 5 if m.group(3) == "FU" else int(m.group(3)[3])
        return m.group(1), COND[m.group(2)], day, f"practice {m.group(3)}", "test" if day == 5 else "learning", "mean"
    m = re.fullmatch(r"RT_Cost_(Day[1-4]|FU)", col)
    if m:
        day = 5 if m.group(1) == "FU" else int(m.group(1)[3])
        return "RT_Cost", "Select minus Execute", day, f"practice {m.group(1)}", "test" if day == 5 else "learning", "mean"
    m = re.fullmatch(r"RT_Cost_Last_Trial_Day_([14])", col)
    if m:
        return "RT_Cost_last", "Select minus Execute", int(m.group(1)), f"practice Day{m.group(1)} last trial", "learning", "value"
    raise ValueError(col)


def main():
    j = pd.read_excel(RAW / "joy.xlsx")
    c = pd.read_excel(RAW / "clin.xlsx")
    c.columns = [str(k).strip() for k in c.columns]
    ck = stdlib.Checks()
    ck.internal("joystick and clinical files list the same 32 participants in the same order",
                len(j) == 32 and (j["SubjectID"].values == c["SubjectID"].values).all())
    # The paper's Table 1 lists sex, age, months post stroke and lesion side of its 16 participants (no ids). Each row matches
    # exactly one deposit participant, which identifies the paper's analysed sample.
    T1 = [("M", 54, 64, "L"), ("M", 67, 158, "L"), ("F", 63, 49, "L"), ("M", 65, 113, "L"), ("M", 59, 25, "R"), ("M", 61, 41, "L"),
          ("M", 40, 53, "R"), ("M", 56, 13, "R"), ("M", 47, 11, "R"), ("M", 61, 12, "L"), ("M", 57, 6, "L"), ("F", 53, 13, "R"),
          ("F", 60, 7, "R"), ("F", 35, 41, "L"), ("F", 64, 12, "R"), ("M", 56, 22, "L")]
    keys = list(zip(c["Sex"], c["Age"], c["MonthsPostStroke"], c["LesionSide"].map({1: "R", 2: "L"})))
    ck.internal("each of the 16 rows of the paper's Table 1 (sex, age, months post stroke, lesion side) matches exactly one deposit participant",
                all(sum(k == t for k in keys) == 1 for t in T1))
    paper = pd.Series([k in T1 for k in keys], index=j.index)
    mri = j["Accuracy_EO_MRIDay1_run1"].notna()
    ck.internal("all 16 matched participants have MRI-task data (24 do in the deposit)", int(paper.sum()) == 16 and bool(mri[paper].all()) and int(mri.sum()) == 24)
    rows, parts, nval = [], [], 0
    for (_, r), (_, q) in zip(j.iterrows(), c.iterrows()):
        sid = f"p{int(r['SubjectID'])}"
        inp = bool(paper[_])
        parts.append(dict(person=sid, months_post_stroke=q["MonthsPostStroke"],
                          weakness_side={1: "right", 2: "left"}[int(q["WeaknessSide"])],
                          lesion_side={1: "right", 2: "left"}[int(q["LesionSide"])],
                          has_mri_task_data=bool(pd.notna(r["Accuracy_EO_MRIDay1_run1"])), in_paper_sample=inp))
        order = 0
        for col in j.columns[1:]:
            v = r[col]
            if pd.isna(v):
                continue
            key, cond, day, block, ph, stat = parse(col)
            order += 1
            nval += 1
            if key.startswith("RT_Cost"):
                nm, desc, unit = "response time cost", ("Select response time minus Execute response time, " +
                                                        ("single last trial of the day" if stat == "value" else "mean across practice blocks")), "s"
            else:
                nm, desc, unit = DESC[key]
            rows.append(dict(dataset=DS, study="Stewart 2022 Eur J Neurosci", experiment="joystick-select-execute", group=None,
                             subject=sid, person=sid, subject_excluded_by_authors=False, age=q["Age"], sex=q["Sex"],
                             handedness={1: "right", 2: "left"}[int(q["HandDominance"])], population="chronic stroke",
                             setting="lab", device="joystick (more-impaired hand)", session=day, block=block, trial=order,
                             phase=ph, time_s=None, task="reaching", unit="participant summary", condition=cond,
                             pert_type="none", pert_value=0, pert_unit=None, feedback="normal", excluded_by_authors=False,
                             measure=nm, measure_description=desc, statistic=stat, value=float(v), value_unit=unit,
                             n_trials=None, session_context="in MRI scanner" if "MRI" in col else None))
    ck.internal("every non-empty cell kept", len(rows) == nval == int(j.iloc[:, 1:].notna().sum().sum()))
    have = j.iloc[:, 1:].notna().any(axis=1)
    ck.internal("every participant with at least one value is released (30 of the 32; two have no joystick values at all)",
                len({r["person"] for r in rows}) == int(have.sum()) == 30)
    parts = [pt for pt, h in zip(parts, have) if h]
    p = j[paper]
    qa = c[paper.values]
    ck.internal("matched sample: 11 men, mean age 56.1, mean 40.0 months post stroke, 9 left lesions (Table 1 totals)",
                int((qa["Sex"] == "M").sum()) == 11 and abs(qa["Age"].mean() - 56.1) < 0.05 and abs(qa["MonthsPostStroke"].mean() - 40.0) < 0.05
                and int((qa["LesionSide"] == 2).sum()) == 9)
    S = "Section 3 (Results), subset matched through Table 1"
    ck.published("Day 1 response time cost, mean (s)", p["RT_Cost_Day1"].mean(), 0.552, 0.0005, S)
    ck.published("Day 1 response time cost, SD (s)", p["RT_Cost_Day1"].std(), 0.238, 0.0005, S)
    ck.published("Day 4 response time cost, mean (s)", p["RT_Cost_Day4"].mean(), 0.353, 0.0005, S)
    ck.published("Day 4 response time cost, SD (s)", p["RT_Cost_Day4"].std(), 0.099, 0.0005, S)
    for lab, cd, d, m_, s_ in (("Execute", "EO", "Day1", 98.1, 7.5), ("Select", "AS", "Day1", 90.3, 15.1),
                               ("Execute", "EO", "Day4", 98.4, 4.0), ("Select", "AS", "Day4", 94.7, 10.6)):
        v = p[[f"Accuracy_{cd}_MRI{d}_run1", f"Accuracy_{cd}_MRI{d}_run2"]].mean(axis=1)
        ck.published(f"scanner direction accuracy {lab} {d}, mean (%)", v.mean(), m_, 0.05, "Results, behavioural results during fMRI")
        ck.published(f"scanner direction accuracy {lab} {d}, SD (%)", v.std(), s_, 0.05, "Results, behavioural results during fMRI")
    meta = dict(
        study=DS,
        paper=dict(authors="Stewart JC, Baird JF, Lewis AF, Fritz SL, Fridriksson J", year=2022,
                   title="Effect of behavioural practice targeted at the motor action selection network after stroke",
                   venue="Eur J Neurosci 56(4):4469-4485", doi="10.1111/ejn.15754"),
        lab="Stewart (University of South Carolina)", data_url="https://osf.io/v46up/",
        license="none stated", license_url="",
        changes=("Turned the wide table (one row per participant, 49 measure columns) into a long table with a row per participant and "
                 "measure, parsed condition, day and block from the column names, dropped empty cells, replaced the participant "
                 "numbers with codes, and added age, sex and hand dominance from the clinical file; months post stroke, "
                 "weakness side and lesion side are in participants.csv."),
        how=["One row is one per-participant value of the joystick task done with the more-impaired hand: direction accuracy "
             "(percent), response time and movement time (seconds), and response time cost (Select minus Execute, seconds), for "
             "the Execute (simple response) and Select (rule-based choice) conditions, on practice Days 1 to 4 and at follow-up "
             "(session 5), plus accuracy in the MRI-scanner version of the task on Day 1, Day 4 and follow-up. Values are means "
             "over trials and blocks as the codebook says; the two RT_Cost_Last_Trial columns are single trials.",
             "Session 5 for follow-up and the grouping of MRI-scanner accuracy under 'other' phase are our labels. The deposit lists 32 "
             "participants, 30 with any joystick value (24 with MRI-task columns); the paper analyses 16. The paper's Table 1 (sex, age, months post stroke, lesion "
             "side, no ids) matches exactly 16 deposit participants, and participants.csv flags them in in_paper_sample; the other 14 "
             "with values are released too (two listed participants have no joystick values, so no rows). Empty cells are not rows; trial is the row order within the participant. Sex is as coded M/F in the "
             "files (the codebook's '1=male, 2=left' line is a typo)."],
        unit_note="per-participant value of one joystick-task measure",
        differences=["The deposit lists 32 participants (30 with joystick values, 24 with MRI-task data) but the paper analyses 16; all 30 with values are released and the 16 are flagged "
                     "(in_paper_sample), identified by matching the paper's Table 1 rows to the clinical file. Over all 32 (or all 24 with MRI "
                     "data) the paper's numbers do not reproduce (for example Day 1 response time cost 0.606 s over 28 with practice data).",
                     "For the 16 flagged participants the practice-task means that the paper quotes do not reproduce: Day 1 Execute response "
                     "time 0.536 s (paper 0.542), Select response time 1.124 s (paper 1.11), Execute accuracy 98.6% (paper 98.2), Select "
                     "accuracy 85.4% (paper 87.3). The response time cost, its SD and the scanner accuracies do reproduce and are the checks. "
                     "These four values are not used as checks."],
    )
    pc = [("months_post_stroke", "number", "months", "Months since stroke at participation."),
          ("weakness_side", "category", "left, right", "Side of arm weakness."),
          ("lesion_side", "category", "left, right", "Hemisphere of the lesion."),
          ("has_mri_task_data", "true/false", "", "Has the MRI-scanner task columns (24 of the 32)."),
          ("in_paper_sample", "true/false", "", "One of the 16 participants analysed in the paper (matched through its Table 1).")]
    stdlib.release(DS, "summary", rows, meta, ck, participants=parts, participant_columns=pc)
    for ch in ck.items:
        if ch["kind"] == "published":
            print(f"  {ch['what']}: {ch['computed']:.4f} vs {ch['reported']} ok={ch['ok']}")


if __name__ == "__main__":
    main()
