"""Robles, Anderson, Dukelow & Striemer (2023) Neuropsychologia: visually guided reaching after cerebellar stroke.

Raw files (OSF 4fk5q): Supplementary Data_v1.csv (saved as supp.csv; 12 patients, z-scores of six measures for the left and
right arm at up to four time periods, in six side-by-side blocks) and Codebook for Supplementary Data_v1.pdf. Nothing is executed.
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stdlib  # noqa: E402

DS = "Robles2023_Neuropsychologia"
RAW = stdlib.raw_dir(DS)
MEAS = {"PPT": ("purdue-pegboard", "Purdue Pegboard Task score", "Purdue Pegboard Task, expressed as a z-score"),
        "IDA": ("kinarm-visually-guided-reaching", "Initial direction angle", "Initial direction angle of the reach (feed-forward control), z-score"),
        "IDR": ("kinarm-visually-guided-reaching", "Initial distance ratio", "Initial distance ratio of the reach, z-score"),
        "SMC": ("kinarm-visually-guided-reaching", "Speed maxima count", "Number of speed maxima in the reach (smoothness), z-score"),
        "MT": ("kinarm-visually-guided-reaching", "Movement time", "Movement time of the reach, z-score"),
        "TaskScore": ("kinarm-visually-guided-reaching", "Task score", "Kinarm overall task score for the visually guided reaching task, z-score")}
PERIOD = {1: "baseline assessment (about 2 weeks post-stroke)", 2: "6 weeks post-stroke", 3: "12 weeks post-stroke",
          4: "24 weeks post-stroke"}


def main():
    R = list(csv.reader(open(RAW / "supp.csv", encoding="utf-8-sig", newline="")))
    hdr = R[2]
    pats = [(i, re.fullmatch(r"(PPT|IDA|IDR|SMC|MT|TaskScore)_([LR])_([1-4])", h.strip())) for i, h in enumerate(hdr)]
    cols = [(i, m.group(1), m.group(2), int(m.group(3))) for i, m in pats if m]
    ck = stdlib.Checks()
    ck.internal("six blocks x 8 columns", len(cols) == 48)
    pat_rows = [r for r in R[3:] if r and r[0].strip()]
    ck.internal("12 patients and each block repeats the same ID", len(pat_rows) == 12 and
                all(r[i] == r[0] for r in pat_rows for i, h in enumerate(hdr) if h.strip() == "Subject ID"))
    rows, parts, nval = [], [], 0
    for r in pat_rows:
        sid = f"p{r[0].strip()}"
        sex = r[1].strip("'")
        parts.append(dict(person=sid, lesion_side={"R": "right", "L": "left", "B": "bilateral"}[r[4].strip("'")]))
        order = 0
        for i, meas, arm, per in sorted(cols, key=lambda c: (c[3], c[1], c[2])):
            v = r[i].strip()
            if v == "":
                continue
            order += 1
            nval += 1
            exp, nm, desc = MEAS[meas]
            rows.append(dict(dataset=DS, study="Robles 2023 Neuropsychologia", experiment=exp, group=None, subject=sid, person=sid,
                             subject_excluded_by_authors=False, age=int(r[2]), sex=sex, handedness={"R": "right", "L": "left"}[r[3]],
                             population="cerebellar stroke", setting="clinic", device="Kinarm exoskeleton" if meas != "PPT" else "Purdue Pegboard",
                             session=per, block=PERIOD[per], trial=order, phase="other", time_s=None, task="reaching",
                             unit="participant summary", condition=("left arm" if arm == "L" else "right arm"),
                             pert_type="none", pert_value=0, pert_unit=None, feedback="normal", excluded_by_authors=False,
                             measure=nm, measure_description=desc, statistic="value", value=float(v), value_unit="z-score",
                             n_trials=None, session_context=None))
    ck.internal("every non-empty cell kept", len(rows) == nval and nval == sum(1 for r in pat_rows for i, *_ in cols if r[i].strip()))
    ck.published("patients with cerebellar stroke", len(pat_rows), 12, 0, "Abstract")
    ck.published("assessment time periods", len({c[3] for c in cols}), 4, 0, "Abstract (baseline, 6, 12 and 24 weeks)")
    meta = dict(
        study=DS,
        paper=dict(authors="Robles CM, Anderson B, Dukelow SP, Striemer CL", year=2023,
                   title="Assessment and recovery of visually guided reaching deficits following cerebellar stroke",
                   venue="Neuropsychologia 188:108662", doi="10.1016/j.neuropsychologia.2023.108662"),
        lab="Striemer (MacEwan University) / Dukelow (University of Calgary)", data_url="https://osf.io/4fk5q/",
        license="none stated", license_url="",
        changes=("Turned the six side-by-side blocks of the supplementary table into one long table with a row per patient, "
                 "measure, arm and time period, dropped empty cells, replaced the patient numbers with codes and moved lesion side "
                 "to participants.csv."),
        how=["One row is one z-score of one patient for one measure, one arm and one assessment: the Purdue Pegboard Task and five "
             "Kinarm visually guided reaching measures (initial direction angle, initial distance ratio, speed maxima count, movement "
             "time, task score), at baseline (about 2 weeks post-stroke) and 6, 12 and 24 weeks. The deposit shares z-scores only "
             "(units are standard deviations from the normative sample), not raw trajectories; sign is as shared.",
             "Empty cells (assessments a patient did not complete) are not rows; trial is the row order within the patient. "
             "Age is at enrolment; handedness and sex as in the file; lesion side is in participants.csv.",
             "The codebook does not say which normative sample the z-scores use; the paper's abstract names a Kinarm normative sample (n = 307) for the reaching task."],
        unit_note="z-score of one measure of one patient, arm and time period",
        differences=["The paper's impaired-patient counts (for example 9 of 12 on the Purdue Pegboard) depend on a cut-off the deposit does not state, so they were not used as checks."],
    )
    pc = [("lesion_side", "category", "left, right, bilateral", "Side of the cerebellar stroke.")]
    stdlib.release(DS, "summary", rows, meta, ck, participants=parts, participant_columns=pc)


if __name__ == "__main__":
    main()
